cmake_minimum_required(VERSION 3.22)
project(local_llm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pin llama.cpp to the release tag validated against this plugin's native code
# (JNI syntax-checked against b11669 headers, 2026-10-15). Keep in sync with
# the iOS xcframework tag (hooks/ios-fetch-llama-xcframework.js). Override with
# -DLLAMA_CPP_GIT_TAG=<tag> in local-llm.gradle after validating a new tag.
set(LLAMA_CPP_GIT_TAG "b11669" CACHE STRING "llama.cpp git tag or commit to build")

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_GIT_TAG}
    GIT_SHALLOW TRUE
)

# CPU-only, library-only build.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

add_library(local_llm SHARED local_llm_jni.cpp)

target_link_libraries(local_llm
    PRIVATE
    llama
    android
    log)
