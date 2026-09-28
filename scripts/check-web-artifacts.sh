#!/usr/bin/env bash
# check-web-artifacts.sh, verify the committed browser build outputs are fresh.
#
# src/web/app.js and src/web/style.css are embedded by the server, and
# web/shell.js, web/style.css and web/agave.js ship next to agave.wasm. They are
# bun and Tailwind builds of the .tsx sources, checked into git so `zig build`
# needs no JavaScript toolchain. Editing a source without rerunning
# scripts/build-web.sh silently leaves the committed output stale, the same
# failure mode scripts/check-shader-artifacts.sh guards for the GPU kernels.
# This script regenerates all five into a scratch dir and byte-compares them
# against the tree, then holds each one to a compressed-size ceiling so weight
# cannot accrete unnoticed between two byte-identical-looking bundles.
#
# Needs bun and `bun install --frozen-lockfile`; bun, tsc and the Tailwind CLI
# come from node_modules.
#
# Exit 0 when everything matches, 1 on drift or a missing copy.
# Canonical regeneration: scripts/build-web.sh
set -euo pipefail
export LC_ALL=C TZ=UTC

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

echo "== browser bundles and stylesheets (scripts/build-web.sh)"
bash scripts/build-web.sh "$SCRATCH"

expected=(
    "src/web/app.js"
    "src/web/style.css"
    "web/shell.js"
    "web/style.css"
    "web/agave.js"
)

generated=()
for artifact in "${expected[@]}"; do
    generated+=("$SCRATCH/$artifact")
done
for gen in "${generated[@]}"; do
    if [[ ! -f "$gen" ]]; then
        echo "error: build-web.sh did not emit ${gen#"$SCRATCH"/}" >&2
        exit 1
    fi
done

drift=0
for gen in "${generated[@]}"; do
    committed="$REPO_ROOT/${gen#"$SCRATCH"/}"
    if [[ ! -f "$committed" ]]; then
        echo "MISSING committed copy: ${committed#"$REPO_ROOT"/}"
        drift=$((drift + 1))
    elif ! cmp -s "$gen" "$committed"; then
        echo "STALE: ${committed#"$REPO_ROOT"/} differs from a fresh build"
        drift=$((drift + 1))
    fi
done

# Weight ceilings, in gzip -9 bytes: that is what the server sends
# (src/server/server.zig gzips the chat page at std.compress.flate .best), so
# it is the number a visitor waits on. app.js is inlined into the one HTML
# document, so its whole compressed size sits on the first-paint path. Each
# ceiling is roughly 10% over the size at the time it was set; raise one
# deliberately, with the measurement that justified it, never as a side effect
# of a dependency bump.
#
#   app.js    116052   (whole chat UI, React + Radix, inlined into the page)
#   style.css   6717   (Tailwind, inlined into the page)
#   shell.js   84158   (React shell, deferred script next to agave.wasm)
#   style.css   6081   (Tailwind for the shell page)
#   agave.js    2795   (hand-written SDK, unminified for embedders, so loose)
declare -a size_breaches=()
for spec in \
    "src/web/app.js 128000" \
    "src/web/style.css 8000" \
    "web/shell.js 93000" \
    "web/style.css 7500" \
    "web/agave.js 4000"
do
    artifact="${spec% *}"
    ceiling="${spec##* }"
    actual="$(gzip -9cn "$REPO_ROOT/$artifact" | wc -c)"
    if ((actual > ceiling)); then
        size_breaches+=("$artifact is ${actual} B gzipped, over the ${ceiling} B ceiling")
    fi
done

if [[ $drift -eq 0 && ${#size_breaches[@]} -eq 0 ]]; then
    echo "OK: all committed browser artifacts match a fresh build"
    echo "Result: all web artifacts fresh"
    exit 0
fi

if [[ ${#size_breaches[@]} -gt 0 ]]; then
    echo "OVER BUDGET:"
    for breach in "${size_breaches[@]}"; do
        echo "  $breach"
    done
fi
if [[ $drift -gt 0 ]]; then
    echo "Result: $drift artifact(s) drifted from sources."
    echo "Regenerate with: scripts/build-web.sh"
    exit 1
fi
echo "Result: a committed browser artifact is over its compressed-size ceiling."
exit 1
