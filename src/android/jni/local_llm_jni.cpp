// JNI bridge between LocalLLMPlugin.java and llama.cpp.
//
// Threading model: the Java side guarantees a single generation at a time.
// The model stays resident between generations; a llama_context is created
// per generation, which keeps memory bounded and avoids depending on the
// KV-cache-clearing API (which has changed names across llama.cpp versions).

#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "LocalLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;
llama_model *g_model = nullptr;
int g_n_ctx = 4096;
int g_n_threads = 4;
std::atomic_bool g_cancel{false};

void throw_runtime_exception(JNIEnv *env, const std::string &message) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message.c_str());
    }
}

std::string jstring_to_utf8(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char *chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars != nullptr ? chars : "");
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

// Number of trailing bytes that form an incomplete UTF-8 sequence. A single
// token can end mid-character (common with CJK text), so streamed chunks are
// held back until the sequence completes.
size_t incomplete_utf8_suffix(const std::string &text) {
    size_t len = text.size();
    for (size_t back = 1; back <= 4 && back <= len; back++) {
        unsigned char byte = static_cast<unsigned char>(text[len - back]);
        if ((byte & 0x80) == 0x00) {
            return 0; // ASCII, complete
        }
        if ((byte & 0xC0) == 0xC0) {
            // Lead byte: how many bytes should this sequence have?
            size_t expected = 2;
            if ((byte & 0xE0) == 0xC0) expected = 2;
            else if ((byte & 0xF0) == 0xE0) expected = 3;
            else if ((byte & 0xF8) == 0xF0) expected = 4;
            return back < expected ? back : 0;
        }
        // Continuation byte (10xxxxxx): keep scanning backwards.
    }
    return 0;
}

std::string apply_chat_template(const std::string &system_prompt, const std::string &user_prompt) {
    const char *tmpl = llama_model_chat_template(g_model, /* name */ nullptr);
    if (tmpl == nullptr) {
        tmpl = "chatml";
    }

    std::vector<llama_chat_message> messages;
    if (!system_prompt.empty()) {
        messages.push_back({"system", system_prompt.c_str()});
    }
    messages.push_back({"user", user_prompt.c_str()});

    std::vector<char> buffer(system_prompt.size() + user_prompt.size() + 1024);
    int32_t written = llama_chat_apply_template(
            tmpl, messages.data(), messages.size(), /* add_assistant */ true,
            buffer.data(), static_cast<int32_t>(buffer.size()));
    if (written > static_cast<int32_t>(buffer.size())) {
        buffer.resize(written);
        written = llama_chat_apply_template(
                tmpl, messages.data(), messages.size(), true,
                buffer.data(), static_cast<int32_t>(buffer.size()));
    }
    if (written < 0) {
        // Unknown template: fall back to plain ChatML, which Qwen models use.
        std::string fallback;
        if (!system_prompt.empty()) {
            fallback += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        }
        fallback += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
        return fallback;
    }
    return std::string(buffer.data(), written);
}

jbyteArray to_jbyte_array(JNIEnv *env, const std::string &text) {
    jbyteArray array = env->NewByteArray(static_cast<jsize>(text.size()));
    if (array != nullptr && !text.empty()) {
        env->SetByteArrayRegion(array, 0, static_cast<jsize>(text.size()),
                                reinterpret_cast<const jbyte *>(text.data()));
    }
    return array;
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qarma_localllm_LocalLLMPlugin_nativeLoadModel(
        JNIEnv *env, jclass, jstring jpath, jint context_size, jint threads) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    llama_backend_init();

    std::string path = jstring_to_utf8(env, jpath);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only

    g_model = llama_model_load_from_file(path.c_str(), model_params);
    if (g_model == nullptr) {
        LOGE("Failed to load model from %s", path.c_str());
        return JNI_FALSE;
    }

    g_n_ctx = context_size > 0 ? context_size : 4096;
    g_n_threads = threads > 0 ? threads : 4;
    LOGI("Model loaded: %s (n_ctx=%d, threads=%d)", path.c_str(), g_n_ctx, g_n_threads);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qarma_localllm_LocalLLMPlugin_nativeUnloadModel(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_qarma_localllm_LocalLLMPlugin_nativeCancel(JNIEnv *, jclass) {
    g_cancel.store(true);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_qarma_localllm_LocalLLMPlugin_nativeGenerate(
        JNIEnv *env, jclass,
        jstring jsystem, jstring juser,
        jint max_tokens, jfloat temperature, jfloat top_p,
        jobjectArray jstops, jobject jcallback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cancel.store(false);

    if (g_model == nullptr) {
        throw_runtime_exception(env, "No model loaded");
        return nullptr;
    }

    // Resolve the Java callback method once.
    jclass callback_class = env->GetObjectClass(jcallback);
    jmethodID on_token = env->GetMethodID(callback_class, "onToken", "([B)V");
    if (on_token == nullptr) {
        throw_runtime_exception(env, "TokenCallback.onToken not found");
        return nullptr;
    }

    std::vector<std::string> stops;
    if (jstops != nullptr) {
        jsize stop_count = env->GetArrayLength(jstops);
        for (jsize i = 0; i < stop_count; i++) {
            auto jstop = reinterpret_cast<jstring>(env->GetObjectArrayElement(jstops, i));
            stops.push_back(jstring_to_utf8(env, jstop));
            env->DeleteLocalRef(jstop);
        }
    }

    std::string prompt = apply_chat_template(jstring_to_utf8(env, jsystem), jstring_to_utf8(env, juser));

    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // Tokenize the full templated prompt. Special tokens must be parsed since
    // the template inserts them as text.
    int32_t token_count = -llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                          nullptr, 0, /* add_special */ true, /* parse_special */ true);
    if (token_count <= 0) {
        throw_runtime_exception(env, "Failed to tokenize prompt");
        return nullptr;
    }
    std::vector<llama_token> tokens(token_count);
    llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                   tokens.data(), token_count, true, true);

    if (token_count + max_tokens >= g_n_ctx) {
        throw_runtime_exception(env, "Prompt too long for context (" + std::to_string(token_count) +
                                     " tokens, n_ctx=" + std::to_string(g_n_ctx) + ")");
        return nullptr;
    }

    const int32_t n_batch = 512;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = g_n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_threads = g_n_threads;
    ctx_params.n_threads_batch = g_n_threads;

    llama_context *ctx = llama_init_from_model(g_model, ctx_params);
    if (ctx == nullptr) {
        throw_runtime_exception(env, "Failed to create llama context (out of memory?)");
        return nullptr;
    }

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string output;
    std::string pending; // bytes not yet emitted (possibly mid-UTF-8-sequence)
    bool failed = false;
    std::string failure;

    // Prompt evaluation in n_batch chunks.
    for (int32_t i = 0; i < token_count && !failed; i += n_batch) {
        if (g_cancel.load()) {
            break;
        }
        int32_t chunk = std::min(n_batch, token_count - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, chunk);
        if (llama_decode(ctx, batch) != 0) {
            failed = true;
            failure = "llama_decode failed during prompt evaluation";
        }
    }

    // Generation loop.
    for (int32_t generated = 0; generated < max_tokens && !failed && !g_cancel.load(); generated++) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];
        int32_t piece_len = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, /* special */ false);
        if (piece_len < 0) {
            failed = true;
            failure = "llama_token_to_piece failed";
            break;
        }
        pending.append(piece, piece_len);
        output.append(piece, piece_len);

        // Stop-sequence handling: truncate the output at the match and stop.
        bool stop_hit = false;
        for (const std::string &stop : stops) {
            size_t pos = output.rfind(stop);
            if (pos != std::string::npos && pos + stop.size() >= output.size() - static_cast<size_t>(piece_len)) {
                output.resize(pos);
                stop_hit = true;
                break;
            }
        }

        // Emit all complete UTF-8 bytes to Java.
        size_t hold_back = stop_hit ? 0 : incomplete_utf8_suffix(pending);
        size_t emit_len = pending.size() - hold_back;
        if (emit_len > 0 && !stop_hit) {
            std::string chunk = pending.substr(0, emit_len);
            pending.erase(0, emit_len);
            jbyteArray jchunk = to_jbyte_array(env, chunk);
            env->CallVoidMethod(jcallback, on_token, jchunk);
            env->DeleteLocalRef(jchunk);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
        }

        if (stop_hit) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            failed = true;
            failure = "llama_decode failed during generation";
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);

    if (failed) {
        throw_runtime_exception(env, failure);
        return nullptr;
    }
    return to_jbyte_array(env, output);
}
