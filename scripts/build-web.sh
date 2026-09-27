#!/usr/bin/env bash
# Build the browser artifacts that Zig embeds or ships:
#
#   src/web/app.js  src/web/style.css   server chat UI, @embedFile'd by server.zig
#   web/shell.js    web/style.css       browser WASM shell, shipped as-is
#   web/agave.js                       browser inference SDK, still a tsc output
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
for tool in "$TSC" "$TAILWIND"; do
    if [[ ! -x "$tool" ]]; then
        echo "need bun install --frozen-lockfile ($tool missing from node_modules)" >&2
        exit 1
    fi
done
if ! command -v bun >/dev/null 2>&1; then
    echo "need bun on PATH (package.json packageManager) to bundle the UI" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/server" "$STAGE/wasm"


# React and Radix are bundled to one IIFE per surface. `--format=iife` keeps the
# result a classic script, which is what the server inlines into <script> and
# what the shell loads with `defer`.
#
# NODE_ENV=production selects React's production build; without it the bundle
# carries the development build and its warnings.
export NODE_ENV=production
bun build src/web/app.tsx --outfile "$STAGE/server/app.js" \
    --format=iife --minify --target browser
bun build web/shell.tsx --outfile "$STAGE/wasm/shell.js" \
    --format=iife --minify --target browser
# The SDK stays unminified and framework-free: it is a documented module for
# embedders that load agave.js next to their own page, so it has to stay
# readable, and nothing about the React UI changes its contract.
bun build web/agave.ts --outfile "$STAGE/wasm/agave.js" \
    --format=iife --target browser

# server.zig concatenates head.html + style.css + body.html + app.js into one
# page, so app.js is inlined into a <script> element. The HTML parser ends a
# classic script at the first `</script` even inside a string literal, and React
# DOM's createElement carries exactly one ("<script></script>" builds a script
# node from markup). `\/` is the same character in a JS string, so the bundle
# stays byte-identical outside that one escape. Only the inlined bundle needs
# it; the shell is loaded from a file.
if grep -q '</script' "$STAGE/server/app.js"; then
    python3 - "$STAGE/server/app.js" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(path.read_text().replace('</script', '<\\/script'))
PY
fi

"$TAILWIND" -i src/web/app.css -o "$STAGE/server/style.css" --minify
"$TAILWIND" -i web/shell.css -o "$STAGE/wasm/style.css" --minify

install_artifact() {
    mkdir -p "$(dirname "$OUT_ROOT/$1")"
    cp "$STAGE/$2" "$OUT_ROOT/$1"
}

install_artifact src/web/app.js server/app.js
install_artifact src/web/style.css server/style.css
install_artifact web/shell.js wasm/shell.js
install_artifact web/style.css wasm/style.css
install_artifact web/agave.js wasm/agave.js
