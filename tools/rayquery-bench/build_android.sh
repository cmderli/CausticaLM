#!/usr/bin/env bash
# Cross-compiles the benchmark for arm64 Android and stages dist-android/.
#
# Needs: the Android NDK (ANDROID_NDK_HOME or ANDROID_NDK_ROOT), plus
# glslangValidator + spirv-val on the host. The NDK sysroot already ships the
# Vulkan headers and the libvulkan stub, so no Vulkan SDK is required.
set -euo pipefail
cd "$(dirname "$0")"

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [ -z "$NDK" ]; then
    for candidate in "$HOME"/Android/Sdk/ndk/* /opt/android-ndk*; do
        [ -d "$candidate" ] && NDK="$candidate"
    done
fi
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
    echo "set ANDROID_NDK_HOME (or ANDROID_NDK_ROOT) to your NDK" >&2
    exit 1
fi

API="${ANDROID_API:-26}"
HOST_TAG=linux-x86_64
[ "$(uname -s)" = "Darwin" ] && HOST_TAG=darwin-x86_64
CXX="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android${API}-clang++"
if [ ! -x "$CXX" ]; then
    echo "compiler not found: $CXX" >&2
    exit 1
fi

for tool in glslangValidator spirv-val; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing $tool (install glslang and spirv-tools)" >&2
        exit 1
    }
done

echo "[1/2] compiling shader"
glslangValidator -V --target-env vulkan1.2 bench.comp -o bench.comp.spv
spirv-val --target-env vulkan1.2 bench.comp.spv

echo "[2/2] cross-compiling host ($API, arm64-v8a)"
mkdir -p dist-android
# -static-libstdc++ keeps the binary self-contained: /data/local/tmp has no libc++_shared.so.
"$CXX" -std=c++17 -O2 -Wall -Wextra -static-libstdc++ bench.cpp -o dist-android/rayquery-bench -lvulkan
cp bench.comp.spv dist-android/

echo
echo "staged dist-android/ ; deploy with ./run_android.sh"
