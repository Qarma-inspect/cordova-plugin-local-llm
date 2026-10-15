package com.qarma.localllm;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.PluginResult;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.nio.charset.StandardCharsets;

public class LocalLLMPlugin extends CordovaPlugin {

    static {
        System.loadLibrary("local_llm");
    }

    /** Called from JNI with a chunk of valid UTF-8 bytes. */
    public interface TokenCallback {
        void onToken(byte[] utf8Bytes);
    }

    private static native boolean nativeLoadModel(String path, int contextSize, int threads);

    private static native void nativeUnloadModel();

    private static native byte[] nativeGenerate(
            String systemPrompt,
            String userPrompt,
            int maxTokens,
            float temperature,
            float topP,
            String[] stopSequences,
            TokenCallback callback);

    private static native void nativeCancel();

    private final Object generateLock = new Object();
    private volatile boolean generating = false;
    private volatile boolean modelLoaded = false;

    @Override
    public boolean execute(String action, JSONArray args, CallbackContext callbackContext) {
        switch (action) {
            case "isSupported":
                cordova.getThreadPool().execute(() -> isSupported(callbackContext));
                return true;
            case "loadModel":
                cordova.getThreadPool().execute(() -> loadModel(args, callbackContext));
                return true;
            case "unloadModel":
                cordova.getThreadPool().execute(() -> unloadModel(callbackContext));
                return true;
            case "generate":
                cordova.getThreadPool().execute(() -> generate(args, callbackContext));
                return true;
            case "cancel":
                nativeCancel();
                callbackContext.success();
                return true;
            default:
                return false;
        }
    }

    private void isSupported(CallbackContext callbackContext) {
        try {
            boolean arm64 = false;
            for (String abi : Build.SUPPORTED_ABIS) {
                if ("arm64-v8a".equals(abi)) {
                    arm64 = true;
                    break;
                }
            }
            ActivityManager am = (ActivityManager) cordova.getActivity()
                    .getSystemService(Context.ACTIVITY_SERVICE);
            ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
            am.getMemoryInfo(memoryInfo);
            long totalMemMB = memoryInfo.totalMem / (1024 * 1024);

            JSONObject result = new JSONObject();
            boolean supported = arm64 && totalMemMB >= 3500;
            result.put("supported", supported);
            result.put("totalMemMB", totalMemMB);
            if (!supported) {
                result.put("reason", !arm64 ? "arm64-v8a not supported" : "not enough memory");
            }
            callbackContext.success(result);
        } catch (Exception e) {
            callbackContext.error("isSupported failed: " + e.getMessage());
        }
    }

    private void loadModel(JSONArray args, CallbackContext callbackContext) {
        try {
            String path = args.getString(0);
            int contextSize = args.optInt(1, 4096);
            int threads = args.optInt(2, 0);
            if (threads <= 0) {
                // Leave headroom for the UI; little cores add latency rather than speed.
                threads = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors() - 2));
            }
            File file = new File(path);
            if (!file.isFile()) {
                callbackContext.error("Model file not found: " + path);
                return;
            }
            synchronized (generateLock) {
                if (generating) {
                    callbackContext.error("Cannot load a model while generating");
                    return;
                }
                boolean ok = nativeLoadModel(path, contextSize, threads);
                modelLoaded = ok;
                if (ok) {
                    callbackContext.success();
                } else {
                    callbackContext.error("Failed to load model (incompatible or corrupt GGUF?)");
                }
            }
        } catch (JSONException e) {
            callbackContext.error("Invalid arguments: " + e.getMessage());
        }
    }

    private void unloadModel(CallbackContext callbackContext) {
        synchronized (generateLock) {
            if (generating) {
                callbackContext.error("Cannot unload while generating; call cancel() first");
                return;
            }
            nativeUnloadModel();
            modelLoaded = false;
        }
        callbackContext.success();
    }

    private void generate(JSONArray args, CallbackContext callbackContext) {
        synchronized (generateLock) {
            if (generating) {
                callbackContext.error("A generation is already in progress");
                return;
            }
            if (!modelLoaded) {
                callbackContext.error("No model loaded; call loadModel() first");
                return;
            }
            generating = true;
        }
        long startedAt = System.currentTimeMillis();
        final int[] tokenCount = {0};
        try {
            String systemPrompt = args.getString(0);
            String userPrompt = args.getString(1);
            int maxTokens = args.optInt(2, 320);
            float temperature = (float) args.optDouble(3, 0.7);
            float topP = (float) args.optDouble(4, 0.9);
            JSONArray stopsJson = args.optJSONArray(5);
            String[] stops = new String[stopsJson != null ? stopsJson.length() : 0];
            for (int i = 0; i < stops.length; i++) {
                stops[i] = stopsJson.getString(i);
            }

            byte[] fullText = nativeGenerate(systemPrompt, userPrompt, maxTokens, temperature, topP, stops,
                    utf8Bytes -> {
                        tokenCount[0]++;
                        try {
                            JSONObject partial = new JSONObject();
                            partial.put("type", "token");
                            partial.put("text", new String(utf8Bytes, StandardCharsets.UTF_8));
                            PluginResult result = new PluginResult(PluginResult.Status.OK, partial);
                            result.setKeepCallback(true);
                            callbackContext.sendPluginResult(result);
                        } catch (JSONException ignored) {
                        }
                    });

            JSONObject done = new JSONObject();
            done.put("type", "done");
            done.put("text", new String(fullText, StandardCharsets.UTF_8));
            done.put("tokensGenerated", tokenCount[0]);
            done.put("durationMs", System.currentTimeMillis() - startedAt);
            callbackContext.sendPluginResult(new PluginResult(PluginResult.Status.OK, done));
        } catch (Exception e) {
            callbackContext.error("Generation failed: " + e.getMessage());
        } finally {
            synchronized (generateLock) {
                generating = false;
            }
        }
    }
}
