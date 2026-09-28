#!/usr/bin/env bash
# Static analysis for the auxiliary Python (scripts/, tests/, tools/, research/):
# ruff check plus ruff format --check, so a formatting drift fails CI instead of
# accumulating. Canonical: zig build lint-python  (or this script from the repo
# root). Rules live in ruff.toml at the repo root.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIN="$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"==\([^"]*\)".*/\1/p' ruff.toml | head -n1)"
if [[ -z "$PIN" ]]; then
    echo "lint-python: could not parse required-version \"==<version>\" from ruff.toml" >&2
    exit 1
fi

# Resolve ruff the way the CI job does. A `ruff` on PATH that already reports
# the pinned version is used as is (no network, no install). Otherwise the
# pinned ruff is run ephemerally through uvx, exactly like the CI job, so a
# contributor needs no global tool install to reproduce CI.
if command -v ruff >/dev/null 2>&1 && [[ "$(ruff --version 2>/dev/null)" == "ruff ${PIN}" ]]; then
    RUFF=(ruff)
elif command -v uvx >/dev/null 2>&1; then
    RUFF=(uvx "ruff@${PIN}")
else
    echo "lint-python: no ruff ${PIN} on PATH and no uvx to fetch it." >&2
    echo "  Either install uv (https://docs.astral.sh/uv/) and rerun, or:" >&2
    echo "    uv tool install ruff==${PIN}" >&2
    exit 1
fi

# Report the version so a local pass and a CI pass that disagree on findings
# can be told apart by tool version. ruff.toml's required-version rejects a
# mismatch outright, so this is a record, not a negotiation.
"${RUFF[@]}" --version

# CI runs `uvx ruff@<pin>` with the same config; both read ruff.toml, so the
# two invocations see the same rule set and the same tool version.
"${RUFF[@]}" check --config ruff.toml .
"${RUFF[@]}" format --check --config ruff.toml .
echo "lint-python: ruff clean"
