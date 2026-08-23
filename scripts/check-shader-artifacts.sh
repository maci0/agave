#!/usr/bin/env bash
# check-shader-artifacts.sh — verify committed GPU kernel artifacts are fresh.
#
# The CUDA PTX (src/backend/kernels/cuda/*.ptx) and Vulkan SPIR-V
# (src/backend/kernels/vulkan/*.spv) binaries are generated artifacts checked
# into git and @embedFile'd into the binaries. They are NOT rebuilt by
# `zig build`, so editing a .zig/.comp kernel source silently leaves the
# committed artifact stale unless it is regenerated. This script rebuilds both
# artifact sets into a scratch dir and byte-compares them against the tree.
#
# Canonical regeneration commands:
#   CUDA PTX   zig build ptx -Dcuda-sm=sm_120     (see docs/KERNELS.md)
#              then copy zig-out/ptx/*.ptx to src/backend/kernels/cuda/
#   SPIR-V     glslangValidator -V --target-env vulkan1.1 foo.comp -o foo.spv
#
# Exit 0 when everything matches; exit 1 on drift or missing copies.
# Usage: scripts/check-shader-artifacts.sh [--ptx-only]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DIR="$REPO_ROOT/src/backend/kernels/cuda"
VK_DIR="$REPO_ROOT/src/backend/kernels/vulkan"
PTX_ONLY=false
[[ "${1:-}" == "--ptx-only" ]] && PTX_ONLY=true

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

drift=0

echo "== CUDA PTX (zig build ptx -Dcuda-sm=sm_120)"
if ! command -v zig >/dev/null 2>&1; then
    echo "SKIP: zig not found in PATH" >&2
else
    zig build ptx -Dcuda-sm=sm_120 --prefix "$SCRATCH/ptx-out"
    for gen in "$SCRATCH"/ptx-out/ptx/*.ptx; do
        name="$(basename "$gen")"
        committed="$CUDA_DIR/$name"
        if [[ ! -f "$committed" ]]; then
            echo "MISSING committed copy: src/backend/kernels/cuda/$name"
            drift=$((drift + 1))
        elif ! cmp -s "$gen" "$committed"; then
            echo "STALE: src/backend/kernels/cuda/$name differs from fresh build"
            drift=$((drift + 1))
        fi
    done
    [[ $drift -eq 0 ]] && echo "OK: all committed PTX match a fresh build"
fi

if $PTX_ONLY; then
    echo
    echo "Result: $drift artifact(s) drifted"
    exit "$drift"
fi

echo
echo "== Vulkan SPIR-V (glslangValidator -V --target-env vulkan1.1)"
if ! command -v glslangValidator >/dev/null 2>&1; then
    echo "SKIP: glslangValidator not found in PATH (apt/vulkan-sdk, brew install glslang)" >&2
elif compgen -G "$VK_DIR/*.comp" >/dev/null; then
    for comp in "$VK_DIR"/*.comp; do
        name="$(basename "$comp" .comp)"
        committed="$VK_DIR/$name.spv"
        out="$SCRATCH/$name.spv"
        if [[ ! -f "$committed" ]]; then
            echo "MISSING compiled shader: src/backend/kernels/vulkan/$name.spv"
            drift=$((drift + 1))
            continue
        fi
        # Byte-compare only flags true staleness when the compiler version that
        # produced the commit is used; different glslang releases embed their
        # generator version, so treat any diff as "stale OR different tool".
        if ! glslangValidator -V --target-env vulkan1.1 "$comp" -o "$out" >/dev/null 2>&1 \
            || ! cmp -s "$out" "$committed"; then
            echo "DRIFT?: src/backend/kernels/vulkan/$name.spv differs from fresh compile"
            drift=$((drift + 1))
        fi
    done
    [[ $drift -eq 0 ]] && echo "OK: all committed SPIR-V match a fresh compile"
else
    echo "SKIP: no .comp shaders under $VK_DIR"
fi

echo
if [[ $drift -gt 0 ]]; then
    echo "Result: $drift artifact(s) drifted from sources."
    echo "Regenerate with:"
    echo "  zig build ptx -Dcuda-sm=sm_120 && cp zig-out/ptx/*.ptx src/backend/kernels/cuda/"
    echo "  cd src/backend/kernels/vulkan && for f in *.comp; do glslangValidator -V --target-env vulkan1.1 \"\$f\" -o \"\${f%.comp}.spv\"; done"
    exit 1
fi
echo "Result: all shader artifacts fresh"
