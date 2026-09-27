#!/usr/bin/env bash
# Emit classic scripts from TypeScript for Zig embed (`src/web/app.js`)
# and the standalone WASM shell (`web/*.js`).
#
# Source of truth is the .ts files. Commit the generated .js so `zig build`
# does not need a TypeScript toolchain.
#
# Usage: scripts/build-web.sh [out-root]
#   out-root  where the generated .js land: absolute, or relative to the repo
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
if [[ ! -x "$TSC" ]]; then
  echo "need bun install --frozen-lockfile (tsc missing from node_modules)" >&2
  exit 1
fi

# Both tsconfigs pin rootDir, so --outDir only relocates the tree: the file
# names below are the same the committed outDir layout produced.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

"$TSC" -p src/web/tsconfig.json --outDir "$STAGE/server"
"$TSC" -p web/tsconfig.json --outDir "$STAGE/wasm"

install_js() {
  mkdir -p "$(dirname "$OUT_ROOT/$1")"
  cp "$STAGE/$2" "$OUT_ROOT/$1"
}

install_js src/web/app.js server/app.js
install_js web/agave.js wasm/agave.js
install_js web/shell.js wasm/shell.js
