#!/usr/bin/env bash
# Emit classic scripts from TypeScript for Zig embed (`src/web/app.js`)
# and the standalone WASM shell (`web/*.js`).
#
# Source of truth is the .ts files. Commit the generated .js so `zig build`
# does not need a TypeScript toolchain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v bunx >/dev/null 2>&1; then
  echo "need bunx to run tsc" >&2
  exit 1
fi

bunx tsc -p src/web/tsconfig.json
cp .web-ts-out/server/app.js src/web/app.js
bunx tsc -p web/tsconfig.json
cp .web-ts-out/wasm/agave.js web/agave.js
cp .web-ts-out/wasm/shell.js web/shell.js
