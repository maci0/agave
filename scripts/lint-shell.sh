#!/usr/bin/env bash
# Static analysis for every shell script in scripts/.
# Canonical: zig build lint-shell  (or this script from the repo root).
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v shellcheck >/dev/null 2>&1; then
    echo "lint-shell: shellcheck not found. Install ShellCheck, then rerun this script." >&2
    exit 1
fi

# Report the version so a local pass and a CI pass that disagree on findings
# can be told apart by tool version.
shellcheck --version | sed -n '2p'

scripts=(scripts/*.sh)
if [[ ! -e "${scripts[0]}" ]]; then
    echo "lint-shell: no shell scripts matched scripts/*.sh" >&2
    exit 1
fi

shellcheck -x "${scripts[@]}"
echo "lint-shell: ${#scripts[@]} script(s) clean"
