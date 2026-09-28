#!/usr/bin/env bash
# Build the browser WASM module and check the output directory is servable.
# Canonical: bash scripts/build-wasm.sh (CI wasm-build job).
#
# zig-out/web is the directory README tells a developer to serve, so it has to
# hold the page's relative-URL assets too, not just the module. A wasm step
# that emitted agave.wasm alone would 404 on shell.js and still pass a
# size check on the module. Kept in scripts/ so the blocking
# `zig build lint-shell` analyses it.
set -euo pipefail
export LC_ALL=C TZ=UTC

web_dir="zig-out/web"
required=(agave.wasm index.html style.css agave.js shell.js)

zig build wasm

for f in "${required[@]}"; do
    if [[ ! -s "$web_dir/$f" ]]; then
        echo "build-wasm: missing or empty $web_dir/$f" >&2
        exit 1
    fi
done
echo "build-wasm: ${#required[@]} file(s) present in $web_dir"
