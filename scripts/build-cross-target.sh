#!/usr/bin/env bash
# Build agave for one cross-compilation target: `scripts/build-cross-target.sh
# <target>` (or TARGET=<target> in the env).
#
# Metal is macOS-only and the dlopen backends need a system loader, so both are
# switched off for the targets that cannot have them. Kept in scripts/ rather
# than inline in the workflow so the blocking `zig build lint-shell` analyses
# this arg-building, which is where a dropped flag silently ships a broken
# release binary.
set -euo pipefail
export LC_ALL=C TZ=UTC

target="${1:-${TARGET:-}}"
if [[ -z "$target" ]]; then
    echo "build-cross-target: no target; pass one, e.g. aarch64-linux-musl" >&2
    exit 1
fi

args=(-Dtarget="$target")
# Metal is macOS-only and unavailable on Linux runners; disable for macOS cross-compile.
if [[ "$target" == *"-macos" ]]; then
    args+=(-Denable-metal=false)
fi
# Static musl builds are CPU-only: the dlopen backends load
# glibc-linked .so files (README "Static musl builds").
if [[ "$target" == *"-musl" ]]; then
    args+=(-Denable-metal=false -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false)
fi
zig build "${args[@]}"
