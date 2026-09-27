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
# against the tree.
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

if [[ $drift -eq 0 ]]; then
    echo "OK: all committed browser artifacts match a fresh build"
    echo "Result: all web artifacts fresh"
    exit 0
fi

echo "Result: $drift artifact(s) drifted from sources."
echo "Regenerate with: scripts/build-web.sh"
exit 1
