#!/usr/bin/env bash
# Static analysis for the auxiliary Python (scripts/, tests/, tools/, research/).
# Canonical: zig build lint-python  (or this script from the repo root).
# Rules live in ruff.toml at the repo root.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIN="$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"==\([^"]*\)".*/\1/p' ruff.toml | head -n1)"
if [[ -z "$PIN" ]]; then
    echo "lint-python: could not parse required-version \"==<version>\" from ruff.toml" >&2
    exit 1
fi

if ! command -v ruff >/dev/null 2>&1; then
    echo "lint-python: ruff not found. Install the pinned ruff (uv: 'uv tool install ruff==${PIN}'), then rerun this script." >&2
    exit 1
fi

# Report the version so a local pass and a CI pass that disagree on findings
# can be told apart by tool version. ruff.toml's required-version rejects a
# mismatch outright, so this is a record, not a negotiation.
ruff --version

# CI runs `uvx ruff@<pin>` with the same config; both read ruff.toml, so the
# two invocations see the same rule set and the same tool version.
ruff check --config ruff.toml .
echo "lint-python: ruff clean"
