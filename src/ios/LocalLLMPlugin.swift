import Foundation
import llama

/// Cordova bridge to llama.cpp. Mirrors the Android implementation:
/// the model stays resident between generations, a llama_context is created
/// per generation, and one generation runs at a time on a serial queue.
@objc(LocalLLMPlugin)
class LocalLLMPlugin: CDVPlugin {

    private var model: OpaquePointer?
    private var contextSize: Int32 = 4096
    private var threads: Int32 = 4

    private var workQueue: DispatchQueue!
    private let stateLock = NSLock()
    private var generating = false
    private var cancelled = false

    override func pluginInitialize() {
        super.pluginInitialize()
        workQueue = DispatchQueue(label: "com.qarma.localllm", qos: .userInitiated)
    }

    // MARK: - Cordova actions

    @objc(isSupported:)
    func isSupported(_ command: CDVInvokedUrlCommand) {
        let totalMemMB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
        var payload: [String: Any] = [
            "supported": totalMemMB >= 3500,
            "totalMemMB": totalMemMB,
        ]
        if totalMemMB < 3500 {
            payload["reason"] = "not enough memory"
        }
        send(status: CDVCommandStatus_OK, payload: payload, callbackId: command.callbackId)
    }

    @objc(loadModel:)
    func loadModel(_ command: CDVInvokedUrlCommand) {
        let path = (command.argument(at: 0) as? String ?? "")
            .replacingOccurrences(of: "file://", with: "")
        let requestedCtx = command.argument(at: 1) as? Int ?? 4096
        let requestedThreads = command.argument(at: 2) as? Int ?? 0

        workQueue.async { [weak self] in
            guard let self = self else { return }
            guard FileManager.default.fileExists(atPath: path) else {
                self.sendError("Model file not found: \(path)", callbackId: command.callbackId)
                return
            }
            self.stateLock.lock()
            let busy = self.generating
            self.stateLock.unlock()
            if busy {
                self.sendError("Cannot load a model while generating", callbackId: command.callbackId)
                return
            }

            if let existing = self.model {
                llama_model_free(existing)
                self.model = nil
            }
            llama_backend_init()

            var params = llama_model_default_params()
            params.n_gpu_layers = 0 // CPU only; flip to 99 to experiment with Metal

            guard let loaded = llama_model_load_from_file(path, params) else {
                self.sendError("Failed to load model (incompatible or corrupt GGUF?)",
                               callbackId: command.callbackId)
                return
            }
            self.model = loaded
            self.contextSize = Int32(requestedCtx > 0 ? requestedCtx : 4096)
            let cores = ProcessInfo.processInfo.activeProcessorCount
            self.threads = Int32(requestedThreads > 0 ? requestedThreads : max(2, min(4, cores - 2)))
            self.send(status: CDVCommandStatus_OK, payload: nil, callbackId: command.callbackId)
        }
    }

    @objc(unloadModel:)
    func unloadModel(_ command: CDVInvokedUrlCommand) {
        workQueue.async { [weak self] in
            guard let self = self else { return }
            if let existing = self.model {
                llama_model_free(existing)
                self.model = nil
            }
            self.send(status: CDVCommandStatus_OK, payload: nil, callbackId: command.callbackId)
        }
    }

    @objc(cancel:)
    func cancel(_ command: CDVInvokedUrlCommand) {
        stateLock.lock()
        cancelled = true
        stateLock.unlock()
        send(status: CDVCommandStatus_OK, payload: nil, callbackId: command.callbackId)
    }

    @objc(generate:)
    func generate(_ command: CDVInvokedUrlCommand) {
        let systemPrompt = command.argument(at: 0) as? String ?? ""
        let userPrompt = command.argument(at: 1) as? String ?? ""
        let maxTokens = command.argument(at: 2) as? Int ?? 320
        let temperature = Float(truncating: command.argument(at: 3) as? NSNumber ?? 0.7)
        let topP = Float(truncating: command.argument(at: 4) as? NSNumber ?? 0.9)
        let stops = command.argument(at: 5) as? [String] ?? []

        stateLock.lock()
        if generating {
            stateLock.unlock()
            sendError("A generation is already in progress", callbackId: command.callbackId)
            return
        }
        generating = true
        cancelled = false
        stateLock.unlock()

        workQueue.async { [weak self] in
            guard let self = self else { return }
            defer {
                self.stateLock.lock()
                self.generating = false
                self.stateLock.unlock()
            }
            self.runGeneration(systemPrompt: systemPrompt,
                               userPrompt: userPrompt,
                               maxTokens: maxTokens,
                               temperature: temperature,
                               topP: topP,
                               stops: stops,
                               callbackId: command.callbackId)
        }
    }

    // MARK: - Generation

    private func runGeneration(systemPrompt: String, userPrompt: String, maxTokens: Int,
                               temperature: Float, topP: Float, stops: [String],
                               callbackId: String) {
        guard let model = model else {
            sendError("No model loaded; call loadModel() first", callbackId: callbackId)
            return
        }
        let startedAt = Date()

        let prompt = applyChatTemplate(model: model, system: systemPrompt, user: userPrompt)
        guard let vocab = llama_model_get_vocab(model) else {
            sendError("Failed to get model vocab", callbackId: callbackId)
            return
        }

        // Tokenize the templated prompt (special tokens included as text).
        let promptBytes = Array(prompt.utf8)
        var tokenCount: Int32 = 0
        promptBytes.withUnsafeBufferPointer { buf in
            let ptr = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: CChar.self)
            tokenCount = -llama_tokenize(vocab, ptr, Int32(promptBytes.count), nil, 0, true, true)
        }
        guard tokenCount > 0 else {
            sendError("Failed to tokenize prompt", callbackId: callbackId)
            return
        }
        var tokens = [llama_token](repeating: 0, count: Int(tokenCount))
        promptBytes.withUnsafeBufferPointer { buf in
            let ptr = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: CChar.self)
            _ = llama_tokenize(vocab, ptr, Int32(promptBytes.count), &tokens, tokenCount, true, true)
        }
        if tokenCount + Int32(maxTokens) >= contextSize {
            sendError("Prompt too long for context (\(tokenCount) tokens, n_ctx=\(contextSize))",
                      callbackId: callbackId)
            return
        }

        let nBatch: Int32 = 512
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = UInt32(contextSize)
        ctxParams.n_batch = UInt32(nBatch)
        ctxParams.n_threads = threads
        ctxParams.n_threads_batch = threads

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            sendError("Failed to create llama context (out of memory?)", callbackId: callbackId)
            return
        }
        defer { llama_free(ctx) }

        guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
            sendError("Failed to create sampler", callbackId: callbackId)
            return
        }
        defer { llama_sampler_free(sampler) }
        if temperature <= 0 {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40))
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(LLAMA_DEFAULT_SEED)))
        }

        // Prompt evaluation in nBatch chunks.
        var offset: Int32 = 0
        while offset < tokenCount {
            if isCancelled() { break }
            let chunk = min(nBatch, tokenCount - offset)
            let rc = tokens.withUnsafeMutableBufferPointer { buf -> Int32 in
                llama_decode(ctx, llama_batch_get_one(buf.baseAddress! + Int(offset), chunk))
            }
            if rc != 0 {
                sendError("llama_decode failed during prompt evaluation", callbackId: callbackId)
                return
            }
            offset += chunk
        }

        var outputBytes = [UInt8]()
        var pendingBytes = [UInt8]()
        var generated = 0
        var pieceBuffer = [CChar](repeating: 0, count: 256)

        while generated < maxTokens && !isCancelled() {
            var token = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, token) { break }

            let pieceLen = llama_token_to_piece(vocab, token, &pieceBuffer, Int32(pieceBuffer.count), 0, false)
            if pieceLen < 0 {
                sendError("llama_token_to_piece failed", callbackId: callbackId)
                return
            }
            let piece = pieceBuffer[0..<Int(pieceLen)].map { UInt8(bitPattern: $0) }
            pendingBytes.append(contentsOf: piece)
            outputBytes.append(contentsOf: piece)
            generated += 1

            // Stop-sequence handling on the byte level.
            var stopHit = false
            for stop in stops where !stop.isEmpty {
                let stopBytes = Array(stop.utf8)
                if let range = lastRange(of: stopBytes, in: outputBytes),
                   range.upperBound >= outputBytes.count - piece.count {
                    outputBytes.removeSubrange(range.lowerBound..<outputBytes.count)
                    stopHit = true
                    break
                }
            }

            if !stopHit {
                let holdBack = incompleteUtf8Suffix(pendingBytes)
                let emitCount = pendingBytes.count - holdBack
                if emitCount > 0 {
                    let chunk = Array(pendingBytes[0..<emitCount])
                    pendingBytes.removeFirst(emitCount)
                    if let text = String(bytes: chunk, encoding: .utf8) {
                        let partial = CDVPluginResult(status: CDVCommandStatus_OK,
                                                      messageAs: ["type": "token", "text": text])
                        partial?.setKeepCallbackAs(true)
                        commandDelegate.send(partial, callbackId: callbackId)
                    }
                }
            } else {
                break
            }

            let rc = withUnsafeMutablePointer(to: &token) { ptr in
                llama_decode(ctx, llama_batch_get_one(ptr, 1))
            }
            if rc != 0 {
                sendError("llama_decode failed during generation", callbackId: callbackId)
                return
            }
        }

        let text = String(bytes: outputBytes, encoding: .utf8) ?? ""
        let payload: [String: Any] = [
            "type": "done",
            "text": text,
            "tokensGenerated": generated,
            "durationMs": Int(Date().timeIntervalSince(startedAt) * 1000),
        ]
        send(status: CDVCommandStatus_OK, payload: payload, callbackId: callbackId)
    }

    // MARK: - Helpers

    private func applyChatTemplate(model: OpaquePointer, system: String, user: String) -> String {
        let template = llama_model_chat_template(model, nil).map { String(cString: $0) } ?? "chatml"

        var cStrings: [UnsafeMutablePointer<CChar>] = []
        defer { cStrings.forEach { free($0) } }
        func keep(_ s: String) -> UnsafePointer<CChar> {
            let dup = strdup(s)!
            cStrings.append(dup)
            return UnsafePointer(dup)
        }

        var messages: [llama_chat_message] = []
        if !system.isEmpty {
            messages.append(llama_chat_message(role: keep("system"), content: keep(system)))
        }
        messages.append(llama_chat_message(role: keep("user"), content: keep(user)))

        var buffer = [CChar](repeating: 0, count: system.utf8.count + user.utf8.count + 1024)
        var written = llama_chat_apply_template(template, &messages, messages.count, true,
                                                &buffer, Int32(buffer.count))
        if written > Int32(buffer.count) {
            buffer = [CChar](repeating: 0, count: Int(written))
            written = llama_chat_apply_template(template, &messages, messages.count, true,
                                                &buffer, Int32(buffer.count))
        }
        if written < 0 {
            // Unknown template: fall back to ChatML, which Qwen models use.
            var fallback = ""
            if !system.isEmpty {
                fallback += "<|im_start|>system\n\(system)<|im_end|>\n"
            }
            fallback += "<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
            return fallback
        }
        return buffer[0..<Int(written)].withUnsafeBufferPointer { buf in
            String(decoding: buf.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
    }

    /// Number of trailing bytes forming an incomplete UTF-8 sequence (tokens can
    /// split multi-byte characters, common with Chinese text).
    private func incompleteUtf8Suffix(_ bytes: [UInt8]) -> Int {
        let len = bytes.count
        var back = 1
        while back <= 4 && back <= len {
            let byte = bytes[len - back]
            if byte & 0x80 == 0 { return 0 }
            if byte & 0xC0 == 0xC0 {
                var expected = 2
                if byte & 0xF0 == 0xE0 { expected = 3 }
                if byte & 0xF8 == 0xF0 { expected = 4 }
                return back < expected ? back : 0
            }
            back += 1
        }
        return 0
    }

    private func lastRange(of needle: [UInt8], in haystack: [UInt8]) -> Range<Int>? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var start = haystack.count - needle.count
        while start >= 0 {
            if Array(haystack[start..<start + needle.count]) == needle {
                return start..<(start + needle.count)
            }
            start -= 1
        }
        return nil
    }

    private func isCancelled() -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return cancelled
    }

    private func send(status: CDVCommandStatus, payload: [String: Any]?, callbackId: String) {
        let result = payload != nil
            ? CDVPluginResult(status: status, messageAs: payload!)
            : CDVPluginResult(status: status)
        commandDelegate.send(result, callbackId: callbackId)
    }

    private func sendError(_ message: String, callbackId: String) {
        commandDelegate.send(CDVPluginResult(status: CDVCommandStatus_ERROR, messageAs: message),
                             callbackId: callbackId)
    }
}
