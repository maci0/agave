#!/usr/bin/env bash
# check-web-artifacts.sh, verify the committed classic-script build outputs are
# fresh.
#
# src/web/app.js (embedded by the server) and web/agave.js, web/shell.js (the
# browser WASM shell) are tsc outputs of app.ts, agave.ts and shell.ts, checked
# into git so `zig build` needs no TypeScript toolchain. Editing a .ts without
# rerunning scripts/build-web.sh silently leaves the committed .js stale, the
# same failure mode scripts/check-shader-artifacts.sh guards for the GPU
# kernels. This script regenerates all three into a scratch dir and byte-compares
# them against the tree.
#
# Needs bun and `bun install --frozen-lockfile`; tsc comes from node_modules.
#
# Exit 0 when everything matches, 1 on drift or a missing copy.
# Canonical regeneration: scripts/build-web.sh
set -euo pipefail
export LC_ALL=C TZ=UTC

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

echo "== TypeScript -> classic script (scripts/build-web.sh)"
bash scripts/build-web.sh "$SCRATCH"

shopt -s nullglob
generated=("$SCRATCH"/src/web/app.js "$SCRATCH"/web/agave.js "$SCRATCH"/web/shell.js)
shopt -u nullglob
if [[ ${#generated[@]} -ne 3 ]]; then
    echo "error: build-web.sh emitted ${#generated[@]} of 3 expected files" >&2
    exit 1
fi

drift=0
for gen in "${generated[@]}"; do
    committed="$REPO_ROOT/${gen#"$SCRATCH"/}"
    if [[ ! -f "$committed" ]]; then
        echo "MISSING committed copy: ${committed#"$REPO_ROOT"/}"
        drift=$((drift + 1))
    elif ! cmp -s "$gen" "$committed"; then
        echo "STALE: ${committed#"$REPO_ROOT"/} differs from a fresh tsc build"
        drift=$((drift + 1))
    fi
done

if [[ $drift -eq 0 ]]; then
    echo "OK: all committed classic scripts match a fresh build"
    echo "Result: all web artifacts fresh"
    exit 0
fi

echo "Result: $drift artifact(s) drifted from sources."
echo "Regenerate with: scripts/build-web.sh"
exit 1
