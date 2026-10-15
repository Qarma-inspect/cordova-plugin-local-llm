var exec = require("cordova/exec");

var SERVICE = "LocalLLM";

function callNative(action, args) {
    return new Promise(function (resolve, reject) {
        exec(resolve, reject, SERVICE, action, args || []);
    });
}

/**
 * Remove a Qwen3 <think>...</think> block (and any stray opening tag) from the output.
 */
function stripThink(text) {
    if (!text) {
        return "";
    }
    return text
        .replace(/<think>[\s\S]*?<\/think>/g, "")
        .replace(/<\/?think>/g, "")
        .trim();
}

function clip(text, maxChars) {
    if (typeof text !== "string") {
        return "";
    }
    text = text.replace(/\s+/g, " ").trim();
    if (text.length <= maxChars) {
        return text;
    }
    return text.slice(0, maxChars - 1) + "…";
}

var DEFAULT_LIMITS = {
    maxDefects: 40,
    maxCheckpoints: 40,
    maxComments: 20,
    itemChars: 240,
    totalChars: 9000,
};

/**
 * Deterministically render structured inspection data into prompt text.
 * Keeping this in JS means Android and iOS produce byte-identical prompts.
 *
 * data: {
 *   title?, productName?, supplierName?, conclusion?,
 *   defects?:           [{description, severity, quantity, comment}],
 *   failedCheckpoints?: [{title, status, comment}],
 *   comments?:          [{author, text}]
 * }
 */
function buildInspectionPrompt(data, options) {
    data = data || {};
    options = options || {};
    var limits = Object.assign({}, DEFAULT_LIMITS, options.limits || {});
    var lines = [];

    if (data.title) {
        lines.push("Inspection: " + clip(data.title, limits.itemChars));
    }
    if (data.productName) {
        lines.push("Product: " + clip(data.productName, limits.itemChars));
    }
    if (data.supplierName) {
        lines.push("Supplier: " + clip(data.supplierName, limits.itemChars));
    }
    if (data.conclusion) {
        lines.push("Inspector conclusion: " + clip(data.conclusion, limits.itemChars));
    }

    var defects = (data.defects || []).slice(0, limits.maxDefects);
    if (defects.length > 0) {
        lines.push("");
        lines.push("Defects found:");
        defects.forEach(function (d) {
            var parts = [];
            if (d.severity) {
                parts.push("[" + d.severity + "]");
            }
            parts.push(clip(d.description || "", limits.itemChars));
            if (d.quantity != null) {
                parts.push("(qty " + d.quantity + ")");
            }
            if (d.comment) {
                parts.push("note: " + clip(d.comment, limits.itemChars));
            }
            lines.push("- " + parts.join(" "));
        });
        if ((data.defects || []).length > defects.length) {
            lines.push("- (more defects omitted)");
        }
    }

    var failed = (data.failedCheckpoints || []).slice(0, limits.maxCheckpoints);
    if (failed.length > 0) {
        lines.push("");
        lines.push("Checkpoints not OK:");
        failed.forEach(function (c) {
            var line = "- " + clip(c.title || "", limits.itemChars);
            if (c.status) {
                line += " [" + c.status + "]";
            }
            if (c.comment) {
                line += " — " + clip(c.comment, limits.itemChars);
            }
            lines.push(line);
        });
        if ((data.failedCheckpoints || []).length > failed.length) {
            lines.push("- (more checkpoints omitted)");
        }
    }

    var comments = (data.comments || []).slice(0, limits.maxComments);
    if (comments.length > 0) {
        lines.push("");
        lines.push("Inspector comments:");
        comments.forEach(function (c) {
            var who = c.author ? c.author + ": " : "";
            lines.push("- " + who + clip(c.text || "", limits.itemChars));
        });
    }

    var prompt = lines.join("\n");
    if (prompt.length > limits.totalChars) {
        prompt = prompt.slice(0, limits.totalChars) + "\n(…input truncated)";
    }
    return prompt;
}

function systemPromptFor(language) {
    var langName = language === "zh" ? "Chinese" : "English";
    return [
        "You are a quality-control inspection assistant.",
        "Write a concise summary of the inspection results below, suitable for a QC report.",
        "Use only the facts provided. Do not invent details, causes, or numbers.",
        "Do not restate exact counts or quantities; the report shows them separately.",
        "Format: one short paragraph (2-4 sentences) on the overall condition and main problem areas,",
        "then up to 5 bullet points with the most important findings.",
        "Write the summary in " + langName + ".",
    ].join(" ");
}

var LocalLLM = {
    /**
     * Resolves {supported: boolean, reason?: string, totalMemMB: number}.
     */
    isSupported: function () {
        return callNative("isSupported");
    },

    /**
     * Load a GGUF model from an absolute filesystem path (no file:// prefix).
     * options: {modelPath: string, contextSize?: number (default 4096), threads?: number}
     */
    loadModel: function (options) {
        options = options || {};
        if (!options.modelPath) {
            return Promise.reject(new Error("modelPath is required"));
        }
        var path = options.modelPath.replace(/^file:\/\//, "");
        return callNative("loadModel", [path, options.contextSize || 4096, options.threads || 0]);
    },

    unloadModel: function () {
        return callNative("unloadModel");
    },

    /**
     * Low-level generation. Applies the model's own chat template natively.
     * options: {systemPrompt?, prompt, maxTokens?, temperature?, topP?, stopSequences?: string[]}
     * onToken: optional function(textFragment) streamed as tokens arrive.
     * Resolves {text, tokensGenerated, durationMs}.
     */
    generate: function (options, onToken) {
        options = options || {};
        if (!options.prompt) {
            return Promise.reject(new Error("prompt is required"));
        }
        return new Promise(function (resolve, reject) {
            exec(
                function (msg) {
                    if (msg && msg.type === "token") {
                        if (onToken) {
                            try {
                                onToken(msg.text);
                            } catch (ignored) {}
                        }
                    } else if (msg && msg.type === "done") {
                        resolve({
                            text: msg.text,
                            tokensGenerated: msg.tokensGenerated,
                            durationMs: msg.durationMs,
                        });
                    }
                },
                reject,
                SERVICE,
                "generate",
                [
                    options.systemPrompt || "",
                    options.prompt,
                    options.maxTokens || 320,
                    options.temperature != null ? options.temperature : 0.7,
                    options.topP != null ? options.topP : 0.9,
                    options.stopSequences || [],
                ]
            );
        });
    },

    /** Cancel the generation in progress (the pending generate() still resolves with partial text). */
    cancel: function () {
        return callNative("cancel");
    },

    /**
     * High-level: summarize inspection/audit data.
     * data: see buildInspectionPrompt. options: {language?: 'en'|'zh', maxTokens?, temperature?}
     * onToken: optional streaming callback.
     * Resolves {text, tokensGenerated, durationMs} with <think> blocks stripped.
     */
    summarizeInspection: function (data, options, onToken) {
        options = options || {};
        var prompt = buildInspectionPrompt(data, options) + "\n\n/no_think";
        return LocalLLM.generate(
            {
                systemPrompt: systemPromptFor(options.language || "en"),
                prompt: prompt,
                maxTokens: options.maxTokens || 320,
                temperature: options.temperature != null ? options.temperature : 0.4,
                topP: 0.9,
                stopSequences: options.stopSequences || [],
            },
            onToken
        ).then(function (result) {
            result.text = stripThink(result.text);
            return result;
        });
    },

    // Exposed for unit tests.
    _buildInspectionPrompt: buildInspectionPrompt,
    _stripThink: stripThink,
};

module.exports = LocalLLM;
