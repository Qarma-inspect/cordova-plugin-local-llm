# cordova-plugin-local-llm

On-device, fully offline text generation for the Qarma app, built on
[llama.cpp](https://github.com/ggml-org/llama.cpp) with Qwen GGUF models.
Primary use case: summarizing inspection/audit results (defects, not-OK
checkpoints, comments) without any network connection.

- Android: llama.cpp is compiled into the app via the NDK (CMake FetchContent, CPU-only, arm64-v8a).
- iOS: links the prebuilt `llama.xcframework` published with llama.cpp releases.
- One shared JS layer builds prompts deterministically, so both platforms produce identical prompts.

## Requirements

| | |
|---|---|
| cordova-android | >= 12, with **NDK + CMake 3.22** installed via SDK Manager |
| cordova-ios | >= 6, Xcode 15+ |
| Devices | arm64 with >= 4GB RAM (`isSupported()` reports this) |
| Build network | first Android build fetches llama.cpp sources; iOS plugin install downloads the xcframework |

## Install

```bash
cordova plugin add ../plugins/cordova-plugin-local-llm
```

**Pinned llama.cpp version: `b11669`** (2026-10-15). Validated: the JNI code
syntax-checks against b11669 headers, and the Swift plugin typechecks against
the b11669 prebuilt `llama.xcframework`. The xcframework is **not committed**
(see `.gitignore`) — the install hook downloads it from the pinned release at
plugin-add time, so adding the plugin needs network access once. Note
llama.cpp's `vX.Y.Z` releases are marker tags without binaries — the `bXXXX`
build releases carry the `llama-bXXXX-xcframework.zip` artifact.

**Upgrading llama.cpp.** Pin the *same tag* in both places and re-validate:

- Android: `src/android/jni/CMakeLists.txt` → `LLAMA_CPP_GIT_TAG` (or override
  via the gradle argument in `src/android/local-llm.gradle`).
- iOS: delete `src/ios/llama.xcframework` and re-add the plugin with
  `LLAMA_CPP_TAG=<tag>` set (the install hook downloads that release's
  xcframework), or build one with llama.cpp's `build-xcframework.sh`.

## Getting a model

Recommended: **Qwen3-1.7B Q4_K_M** (~1.1GB) as the baseline,
**Qwen3-4B-Instruct Q4_K_M** (~2.4GB) for 8GB+ devices. Download GGUF files
from Hugging Face (`Qwen/Qwen3-1.7B-GGUF`). Do **not** bundle the model in the
app binary — download it on WiFi (e.g. with cordova-plugin-file-transfer) into
`cordova.file.dataDirectory` and pass the resulting native path to `loadModel`.

## API

All methods return promises. `window.cordova.plugins.localLLM`:

```ts
isSupported(): Promise<{supported, reason?, totalMemMB}>
loadModel({modelPath, contextSize = 4096, threads?}): Promise<void>
unloadModel(): Promise<void>
generate({systemPrompt?, prompt, maxTokens = 320, temperature = 0.7, topP = 0.9, stopSequences?},
         onToken?): Promise<{text, tokensGenerated, durationMs}>
cancel(): Promise<void>   // pending generate() resolves with the partial text
summarizeInspection(data, {language = 'en', maxTokens?, temperature = 0.4}?, onToken?)
```

`generate` applies the model's own chat template natively and streams token
fragments through `onToken` (UTF-8 safe — multi-byte Chinese characters are
never split). Only one generation runs at a time.

`summarizeInspection` renders structured inspection data into a deterministic
prompt (`www/LocalLLM.js` → `buildInspectionPrompt`), instructs the model not
to invent facts or restate counts, suppresses Qwen3 thinking mode
(`/no_think` + `<think>` stripping), and supports `language: 'en' | 'zh'`.

### Usage

```js
const llm = cordova.plugins.localLLM;

const {supported} = await llm.isSupported();
if (!supported) { /* fall back to no summary / server-side summary */ }

await llm.loadModel({modelPath: modelNativePath});

const {text} = await llm.summarizeInspection({
    title: inspection.name,
    productName: inspection.product_name,
    supplierName: inspection.supplier_name,
    defects: defects.map(d => ({
        description: d.description, severity: d.severity,
        quantity: d.quantity, comment: d.comment,
    })),
    failedCheckpoints: checkpoints.filter(isNotOk).map(c => ({
        title: c.description, status: c.status_text, comment: c.comment,
    })),
    comments: comments.map(c => ({author: c.author_name, text: c.text})),
}, {language: 'en'}, partial => updateDraftUI(partial));

// Show `text` as an editable draft — the inspector reviews before it is saved.
```

## Design notes / caveats

- **The model stays loaded; a context is created per generation.** Keeps
  memory bounded and avoids llama.cpp's KV-cache API churn. Load once per
  app session, `unloadModel()` under memory pressure.
- **Expect ~10–20 tok/s on mid-range arm64 CPUs** — a 200-token summary takes
  10–25s. Generate in the background and stream into the UI.
- **Numbers belong to the app, not the model.** The summarize prompt forbids
  restating counts; inject totals into the report template programmatically.
- **arm64-v8a only** on Android (32-bit devices get `supported: false`).
- The in-IDE Swift error `No such module 'llama'` disappears once
  `llama.xcframework` is present in `src/ios/`.
- Memory: 1.7B Q4 + 4K context needs roughly 1.6–2GB free; `loadModel` /
  context creation fail gracefully with an error callback when it isn't.
