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

# Floor, not a pin: the two optional checks named below arrived in 0.9.0, and
# the tool only warns on an unknown -o code, so an older release would run
# with them off and still report the script set as clean. Same reason
# ruff.toml pins ruff exactly; here any release at or past the floor carries
# the checks, so a newer one is fine and a newer CI runner is not a surprise.
readonly shellcheck_floor=0.9.0
shellcheck_version="$(shellcheck --version | sed -n '2s/^version: //p')"
if [[ "$(printf '%s\n%s\n' "$shellcheck_floor" "$shellcheck_version" | sort -V | head -n1)" != "$shellcheck_floor" ]]; then
    echo "lint-shell: shellcheck $shellcheck_version is older than the $shellcheck_floor floor;" >&2
    echo "  it does not know check-set-e-suppressed and would report a false pass." >&2
    exit 1
fi

scripts=(scripts/*.sh)
if [[ ! -e "${scripts[0]}" ]]; then
    echo "lint-shell: no shell scripts matched scripts/*.sh" >&2
    exit 1
fi

# Optional checks, enabled individually. check-set-e-suppressed is on: it
# flags `set -e` in a context that silently does not apply (a command on the
# left of `&&`, in a condition list, in a `!`), which is a real way a script
# keeps running past the failure it meant to stop at. The tree passes it.
# check-extra-masked-returns stays off: it fires 47 times across 7 scripts,
# nearly all `$(...)` substitutions inside `local`, `printf` and pipelines
# where a nonzero status is not a defect. It is worth enabling per script as
# each is made to propagate status; until then it is a documented skip, not a
# silent one.
# require-variable-braces and avoid-negated-conditions stay off as pure style.
shellcheck -x -o require-double-brackets,check-unassigned-uppercase,deprecate-which,useless-use-of-cat "${scripts[@]}"
echo "lint-shell: ${#scripts[@]} script(s) clean"
