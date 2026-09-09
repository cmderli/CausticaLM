#!/usr/bin/env bash
# Pushes the Android build to the device and runs it headless over adb.
#
#   ./run_android.sh --width 960 --height 540 --tlas-rebuild --incoherent
#
# Arguments are forwarded verbatim to the benchmark; run with --help to list them.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d dist-android ]; then
    echo "run ./build_android.sh first" >&2
    exit 1
fi

REMOTE=/data/local/tmp/rayquery-bench
adb shell "mkdir -p $REMOTE"
adb push dist-android/rayquery-bench dist-android/bench.comp.spv "$REMOTE/" >/dev/null
adb shell "chmod 755 $REMOTE/rayquery-bench"

echo "--- device ---"
adb shell getprop ro.product.model
adb shell getprop ro.board.platform
echo "--- running ---"
adb shell "cd $REMOTE && ./rayquery-bench $*"
