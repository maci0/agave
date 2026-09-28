#!/usr/bin/env bash
# CI lint-web job: oxlint + tsc for src/web and web, plus the chat UI smoke
# test. The UI is a Preact tree that only exists once mounted, so a broken
# import or a dead render passes every static check; `bun test src/web` mounts
# it against a stubbed server and fails if a turn does not land in the log.
# Canonical: zig build lint-web  (or this script from the repo root).
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The oxlint ignorePatterns ratchet first: it needs no toolchain, and a path
# excluded from linting is a defect the oxlint run below cannot report.
bash scripts/check-web-lint-scope.sh

PIN="$(sed -n 's/.*"packageManager": "bun@\([^"]*\)".*/\1/p' package.json | head -n1)"
if [[ -z "$PIN" ]]; then
    echo "lint-web: could not parse package.json packageManager bun@X.Y.Z" >&2
    exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
    echo "lint-web: bun not found. Install bun ${PIN} (package.json packageManager), then:" >&2
    echo "  bun install --frozen-lockfile" >&2
    exit 1
fi

got="$(bun --version)"
if [[ "$got" != "$PIN" ]]; then
    echo "lint-web: bun ${got} != package.json packageManager bun@${PIN} (CI lint-web uses that pin)" >&2
    exit 1
fi

if [[ ! -x "$ROOT/node_modules/.bin/oxlint" || ! -x "$ROOT/node_modules/.bin/tsc" ]]; then
    echo "lint-web: JS deps missing. From the repo root:" >&2
    echo "  bun install --frozen-lockfile" >&2
    exit 1
fi

bun run lint
bun run typecheck
bun test src/web
