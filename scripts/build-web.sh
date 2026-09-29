#!/usr/bin/env bash
# Build the browser artifacts that Zig embeds or ships:
#
#   src/web/app.js  src/web/style.css   server chat UI, @embedFile'd by server.zig
#   web/shell.js    web/style.css       browser WASM shell, shipped as-is
#   web/agave.js                       browser inference SDK, unminified for embedders
#
# Source of truth is the .tsx/.ts sources plus the Tailwind 4 entry stylesheets.
# The bundles and stylesheets are committed so `zig build` needs no JavaScript
# toolchain: the server embeds app.js into a single HTML page, and the WASM
# shell directory is copied next to agave.wasm.
#
# Usage: scripts/build-web.sh [out-root]
#   out-root  where the generated files land: absolute, or relative to the repo
#             root. Defaults to the repo root. Output is byte-identical for any
#             out-root, so a freshness check can emit into a scratch dir and
#             diff against the committed copies.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${1:-$ROOT}"
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$ROOT/$OUT_ROOT"

TSC="$ROOT/node_modules/.bin/tsc"
TAILWIND="$ROOT/node_modules/.bin/tailwindcss"

BUN_PIN="$(sed -n 's/.*"packageManager": "bun@\([^"]*\)".*/\1/p' "$ROOT/package.json" | head -n1)"
if [[ -z "$BUN_PIN" ]]; then
    echo "build-web: could not parse packageManager bun@X.Y.Z from package.json" >&2
    exit 1
fi
for tool in "$TSC" "$TAILWIND"; do
    if [[ ! -x "$tool" ]]; then
        echo "need bun install --frozen-lockfile ($tool missing from node_modules)" >&2
        exit 1
    fi
done
if ! command -v bun >/dev/null 2>&1; then
    echo "need bun ${BUN_PIN} on PATH (package.json packageManager) to bundle the UI" >&2
    exit 1
fi
# The committed bundles are `bun build` output, and
# scripts/check-web-artifacts.sh decides staleness by byte-comparing against
# them. A different bun release can emit different bytes for the same source,
# so regenerating with an unpinned bun commits bundles the pinned bun cannot
# reproduce, and every other checkout then reports STALE. scripts/lint-web.sh
# gates its own run the same way; this is the regeneration path, so it needs
# the same check.
if [[ "$(bun --version)" != "$BUN_PIN" ]]; then
    echo "build-web: bun $(bun --version) != package.json packageManager bun@${BUN_PIN}" >&2
    echo "  install bun ${BUN_PIN}, then rerun (scripts/lint-web.sh enforces the same pin)" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/server" "$STAGE/wasm"


# Preact and Radix are bundled to one IIFE per surface. `--format=iife` keeps the
# result a classic script, which is what the server inlines into <script> and
# what the shell loads with `defer`.
#
# These are Preact bundles: the shadcn/ui components and Radix primitives under
# src/web/ui/ import from `react` and `react-dom` because that is their published
# API, and `vendor/react` / `vendor/react-dom` are file: dependencies that
# re-export preact/compat under those names. bun 1.4.2 has no module aliasing, so
# the shim packages are how the substitution happens, for the bundler, the test
# run and tsc alike.
#
# The bundles are fully minified, identifiers included. app.js is inlined into
# the one HTML document server.zig serves, so every cold visit downloads the
# whole bundle. `scripts/check-web-artifacts.sh` byte-compares the committed
# output, so the bundler is pinned to bun 1.4.2 above; if a bun release makes the
# mangler unstable, that gate is what catches it, and the artifact goes back to
# `--minify-whitespace --minify-syntax`.
#
# NODE_ENV=production selects the production build of preact/compat; without it
# the bundle carries the development build and its warnings.
export NODE_ENV=production
bun build src/web/app.tsx --outfile "$STAGE/server/app.js" \
    --format=iife --minify --target browser
bun build web/shell.tsx --outfile "$STAGE/wasm/shell.js" \
    --format=iife --minify --target browser
# The SDK stays unminified and framework-free: it is a documented module for
# embedders that load agave.js next to their own page, so it has to stay
# readable, and nothing about the UI framework changes its contract.
bun build web/agave.ts --outfile "$STAGE/wasm/agave.js" \
    --format=iife --target browser

# server.zig concatenates head.html + style.css + body.html + app.js into one
# page, so app.js is inlined into a <script> element. The HTML parser ends a
# classic script at the first `</script` even inside a string literal, and a
# bundled dependency that builds a script node from markup carries one
# ("<script></script>"). `\/` is the same character in a JS string, so the
# bundle stays byte-identical outside that one escape. Only the inlined bundle
# needs it; the shell is loaded from a file.
if grep -q '</script' "$STAGE/server/app.js"; then
    # Write-then-rename rather than `sed -i`: GNU and BSD sed disagree on
    # whether -i takes a suffix argument.
    sed 's|</script|<\\/script|g' "$STAGE/server/app.js" > "$STAGE/server/app.js.tmp"
    mv "$STAGE/server/app.js.tmp" "$STAGE/server/app.js"
fi

"$TAILWIND" -i src/web/app.css -o "$STAGE/server/style.css" --minify
"$TAILWIND" -i web/shell.css -o "$STAGE/wasm/style.css" --minify
# Tailwind's @property registrations and their @supports-guarded fallback are
# valid CSS the W3C checker does not know; css-conform.ts keeps the fallback,
# unguarded, so both stylesheets pass vnu with the same behavior.
bun scripts/css-conform.ts "$STAGE/server/style.css"
bun scripts/css-conform.ts "$STAGE/wasm/style.css"

install_artifact() {
    mkdir -p "$(dirname "$OUT_ROOT/$1")"
    cp "$STAGE/$2" "$OUT_ROOT/$1"
}

install_artifact src/web/app.js server/app.js
install_artifact src/web/style.css server/style.css
install_artifact web/shell.js wasm/shell.js
install_artifact web/style.css wasm/style.css
install_artifact web/agave.js wasm/agave.js
