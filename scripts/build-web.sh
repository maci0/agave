#!/usr/bin/env bash
# Emit classic scripts from TypeScript for Zig embed (`src/web/app.js`)
# and the standalone WASM shell (`web/*.js`).
#
# Source of truth is the .ts files. Commit the generated .js so `zig build`
# does not need a TypeScript toolchain.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TSC="$ROOT/node_modules/.bin/tsc"
if [[ ! -x "$TSC" ]]; then
  echo "need bun install --frozen-lockfile (tsc missing from node_modules)" >&2
  exit 1
fi

"$TSC" -p src/web/tsconfig.json
cp .web-ts-out/server/app.js src/web/app.js
"$TSC" -p web/tsconfig.json
cp .web-ts-out/wasm/agave.js web/agave.js
cp .web-ts-out/wasm/shell.js web/shell.js
