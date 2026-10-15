/**
 * before_plugin_install (ios): ensure src/ios/llama.xcframework exists.
 *
 * llama.cpp publishes a prebuilt xcframework with each release
 * (artifact: llama-<tag>-xcframework.zip). Pin LLAMA_CPP_TAG to the release
 * you validated — keep it in sync with the tag in src/android/jni/CMakeLists.txt.
 *
 * Override the download with the LLAMA_XCFRAMEWORK_URL env var, or place a
 * prebuilt llama.xcframework into src/ios/ manually (offline installs, or
 * one you built yourself with llama.cpp's build-xcframework.sh).
 */
const fs = require("fs");
const path = require("path");
const https = require("https");
const {execSync} = require("child_process");

// Pinned llama.cpp build release; keep in sync with LLAMA_CPP_GIT_TAG in
// src/android/jni/CMakeLists.txt. (Note: llama.cpp's "vX.Y.Z" releases are
// marker tags without binaries — the bXXXX build releases carry the
// llama-bXXXX-xcframework.zip artifact.)
const LLAMA_CPP_TAG = process.env.LLAMA_CPP_TAG || "b11669";

function download(url, destination, redirectsLeft) {
    return new Promise((resolve, reject) => {
        if (redirectsLeft <= 0) {
            reject(new Error("Too many redirects downloading " + url));
            return;
        }
        https
            .get(url, {headers: {"User-Agent": "cordova-plugin-local-llm"}}, (response) => {
                if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                    response.resume();
                    resolve(download(response.headers.location, destination, redirectsLeft - 1));
                    return;
                }
                if (response.statusCode !== 200) {
                    response.resume();
                    reject(new Error("HTTP " + response.statusCode + " downloading " + url));
                    return;
                }
                const file = fs.createWriteStream(destination);
                response.pipe(file);
                file.on("finish", () => file.close(resolve));
                file.on("error", reject);
            })
            .on("error", reject);
    });
}

function findXcframework(directory) {
    const entries = fs.readdirSync(directory, {withFileTypes: true});
    for (const entry of entries) {
        if (entry.isDirectory()) {
            if (entry.name === "llama.xcframework") {
                return path.join(directory, entry.name);
            }
            const nested = findXcframework(path.join(directory, entry.name));
            if (nested) {
                return nested;
            }
        }
    }
    return null;
}

module.exports = async function (context) {
    const pluginDir = context.opts.plugin.dir;
    const target = path.join(pluginDir, "src", "ios", "llama.xcframework");
    if (fs.existsSync(target)) {
        console.log("[local-llm] llama.xcframework already present, skipping download");
        return;
    }

    let url = process.env.LLAMA_XCFRAMEWORK_URL;
    if (!url && LLAMA_CPP_TAG) {
        url =
            "https://github.com/ggml-org/llama.cpp/releases/download/" +
            LLAMA_CPP_TAG +
            "/llama-" +
            LLAMA_CPP_TAG +
            "-xcframework.zip";
    }
    if (!url) {
        throw new Error(
            "[local-llm] src/ios/llama.xcframework is missing.\n" +
                "Either set LLAMA_CPP_TAG=<llama.cpp release tag> (downloads the prebuilt " +
                "xcframework from that GitHub release), set LLAMA_XCFRAMEWORK_URL to a zip, " +
                "or place a llama.xcframework into " +
                path.join(pluginDir, "src", "ios") +
                " (build one with llama.cpp's build-xcframework.sh)."
        );
    }

    const zipPath = path.join(pluginDir, "src", "ios", "llama-xcframework.zip");
    console.log("[local-llm] downloading " + url);
    await download(url, zipPath, 5);

    const extractDir = path.join(pluginDir, "src", "ios", "llama-xcframework-tmp");
    fs.mkdirSync(extractDir, {recursive: true});
    execSync('unzip -q -o "' + zipPath + '" -d "' + extractDir + '"');

    const extracted = findXcframework(extractDir);
    if (!extracted) {
        throw new Error("[local-llm] downloaded zip does not contain llama.xcframework");
    }
    fs.renameSync(extracted, target);
    fs.rmSync(extractDir, {recursive: true, force: true});
    fs.rmSync(zipPath, {force: true});
    console.log("[local-llm] llama.xcframework installed to src/ios/");
};
