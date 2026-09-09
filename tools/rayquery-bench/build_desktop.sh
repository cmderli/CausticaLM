#!/usr/bin/env bash
# Builds the rayQuery micro-benchmark for the host machine.
#
# Needs: glslangValidator + spirv-val (glslang / spirv-tools), a C++17 compiler,
# and Vulkan headers + a Vulkan loader. Headers are fetched into .deps/ when the
# system does not provide them.
set -euo pipefail
cd "$(dirname "$0")"

for tool in glslangValidator spirv-val; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing $tool (install glslang and spirv-tools)" >&2
        exit 1
    }
done

echo "[1/3] compiling shader"
glslangValidator -V --target-env vulkan1.2 bench.comp -o bench.comp.spv
spirv-val --target-env vulkan1.2 bench.comp.spv

echo "[2/3] resolving Vulkan headers"
CXX="${CXX:-g++}"
INCLUDE=""
# pkg-config can advertise a Vulkan version while vulkan-headers is not installed,
# so probe an actual compile instead of trusting it.
probe_headers() {
    printf '#include <vulkan/vulkan.h>\n' | "$CXX" -E - $1 - >/dev/null 2>&1
}
if probe_headers ""; then
    INCLUDE=""
elif pkg-config --exists vulkan 2>/dev/null && probe_headers "$(pkg-config --cflags vulkan)"; then
    INCLUDE="$(pkg-config --cflags vulkan)"
else
    DEPS=.deps/Vulkan-Headers
    if [ ! -f "$DEPS/include/vulkan/vulkan.h" ]; then
        echo "      vulkan/vulkan.h not found; fetching Vulkan-Headers into $DEPS"
        mkdir -p .deps
        git clone --depth 1 -q https://github.com/KhronosGroup/Vulkan-Headers "$DEPS"
    fi
    INCLUDE="-I$DEPS/include"
fi

echo "[3/3] compiling host"
# shellcheck disable=SC2086
$CXX -std=c++17 -O2 -Wall -Wextra $INCLUDE bench.cpp -o rayquery-bench -lvulkan

echo
echo "built ./rayquery-bench"
echo "run:  ./rayquery-bench --list"
