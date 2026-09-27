#!/usr/bin/env bash
# check-reproducible.sh, build the same source twice and require the two
# `agave` binaries to be byte-identical.
#
# The build already strips ReleaseFast, so nothing in the shipped binary
# should depend on where or when it was built. That is a claim, not a
# measurement, and a claim nobody checks stops being true the first time an
# `@src().file`, a panic message carrying an absolute path, or a build_runner
# path slips into a release artifact. The two builds here differ in every axis
# a leak would use:
#
#   source directory   a copy under a shorter, differently named path
#   install prefix     --prefix, into a different scratch dir
#   local cache dir    --cache-dir, so nothing is shared through zig-out
#   SOURCE_DATE_EPOCH  a different value
#   TZ / LC_ALL        a different timezone and locale
#
# Both builds share the global Zig cache (~/.cache/zig), which is
# content-addressed: the second build re-links from cached objects instead of
# recompiling the engine twice. So this gate measures path, time, locale and
# install layout, not codegen determinism from a cold cache.
#
# The flags are the CPU-only, one-model set CI's docker-build job uses, so the
# gate fits a normal CI budget. Artifacts that are conditionally @embedFile'd
# away by those flags are covered by their own rebuild-and-compare gates
# (scripts/check-shader-artifacts.sh, scripts/check-web-artifacts.sh).
#
# Exit 0 when the two binaries are identical, 1 on drift.
# Usage: scripts/check-reproducible.sh
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v zig >/dev/null 2>&1; then
    echo "check-reproducible: zig not on PATH. Install $(tr -d '[:space:]' < "$ROOT/.zigversion") from https://ziglang.org/download/." >&2
    exit 1
fi

# The build refuses a Zig that is not the .zigversion pin, so report the pin
# before a long build fails for that reason at the end.
ZIG_VERSION="$(tr -d '[:space:]' < "$ROOT/.zigversion")"
echo "== Zig $ZIG_VERSION, $(zig version) on PATH"

# Same as `zig build` for a host binary, minus the GPU backends and the model
# architectures, so the gate runs in minutes rather than tens of minutes.
BUILD_FLAGS=(
    -Denable-cuda=false
    -Denable-rocm=false
    -Denable-metal=false
    -Denable-vulkan=false
    -Denable-webgpu=false
    -Denable-debug=false
    -Denable-bench=false
    -Denable-qwen35=false
    -Denable-qwen4-exp=false
    -Denable-gpt-oss=false
    -Denable-nemotron-h=false
    -Denable-nemotron-nano=false
    -Denable-glm4=false
    -Denable-gemma4=false
    -Denable-diffusion-gemma=false
    -Denable-deepseek4=false
    -Denable-llama4=false
    -Denable-dflash2=false
)

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# A copy at a different path is the only way the build root, and therefore
# every relative path the compiler bakes in, can differ between the two runs.
# .git, the caches and zig-out are build state, not source; node_modules is
# TypeScript tooling that no Zig artifact reads.
SRC_B="$SCRATCH/src"
mkdir -p "$SRC_B"
tar -C "$ROOT" -cf - \
    --exclude=./.git --exclude=./.zig-cache --exclude=./zig-out \
    --exclude=./node_modules --exclude=./models \
    . | tar -C "$SRC_B" -xf -

echo "== Build A: $ROOT"
SOURCE_DATE_EPOCH=1600000000 TZ=UTC LC_ALL=C \
    zig build --build-file "$ROOT/build.zig" --cache-dir "$SCRATCH/cache-a" \
    --prefix "$SCRATCH/out-a" "${BUILD_FLAGS[@]}"

echo "== Build B: $SRC_B (different path, SOURCE_DATE_EPOCH=1700000000, TZ=Asia/Tokyo, LC_ALL=en_US.UTF-8)"
SOURCE_DATE_EPOCH=1700000000 TZ=Asia/Tokyo LC_ALL=en_US.UTF-8 \
    zig build --build-file "$SRC_B/build.zig" --cache-dir "$SCRATCH/cache-b" \
    --prefix "$SCRATCH/out-b" "${BUILD_FLAGS[@]}"

A="$SCRATCH/out-a/bin/agave"
B="$SCRATCH/out-b/bin/agave"
for bin in "$A" "$B"; do
    if [[ ! -f "$bin" ]]; then
        echo "check-reproducible: $bin was not produced" >&2
        exit 1
    fi
done

sha_a="$(sha256sum "$A" | cut -d' ' -f1)"
sha_b="$(sha256sum "$B" | cut -d' ' -f1)"
echo "A sha256 $sha_a"
echo "B sha256 $sha_b"

if [[ "$sha_a" != "$sha_b" ]]; then
    echo "error: two builds of the same source differ; the artifact carries build path, time, locale or install state." >&2
    if command -v diffoscope >/dev/null 2>&1; then
        diffoscope "$A" "$B" || true
    else
        echo "install diffoscope and rerun to see what differs" >&2
    fi
    exit 1
fi

echo "OK: byte-identical across a different source path, prefix, cache dir, epoch, timezone and locale"
