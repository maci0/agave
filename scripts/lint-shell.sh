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

# True when $1 sorts before $2 as a dotted version. Field-wise, so 0.10.0 is
# above 0.9.0 the way a version reads and not the way a string compares. Done
# in bash rather than `sort -V`, which GNU coreutils has and BSD/macOS sort
# does not, and a macOS workstation runs this script.
version_lt() {
    local -a lhs rhs
    local i l r
    IFS='.' read -r -a lhs <<<"$1"
    IFS='.' read -r -a rhs <<<"$2"
    for i in 0 1 2; do
        # Trailing `-rc1` or `+dfsg` suffixes are cut; a field with no digits
        # at all reads as 0.
        l="${lhs[i]:-0}"
        r="${rhs[i]:-0}"
        l="${l%%[^0-9]*}"
        r="${r%%[^0-9]*}"
        l="${l:-0}"
        r="${r:-0}"
        ((10#$l > 10#$r)) && return 1
        ((10#$l < 10#$r)) && return 0
    done
    return 1
}

if version_lt "$shellcheck_version" "$shellcheck_floor"; then
    echo "lint-shell: shellcheck $shellcheck_version is older than the $shellcheck_floor floor;" >&2
    echo "  it would report a false pass on the optional checks named below." >&2
    exit 1
fi

# The enabled-check list below is the only thing here that decides what the gate
# can see, and ShellCheck accepts an -o name it does not know without a word and
# still exits 0 -- a typo would silently drop a check and the run would report
# a false pass. --list-optional is the authoritative list, so every name is
# checked against it before the gate reports anything as clean.
readonly -a optional_checks=(
    require-double-brackets
    check-unassigned-uppercase
    deprecate-which
    useless-use-of-cat
)
known_optional_checks="$(shellcheck --list-optional | sed -n 's/^name:[[:space:]]*//p')"
for check_name in "${optional_checks[@]}"; do
    if ! grep -qx -- "$check_name" <<<"$known_optional_checks"; then
        echo "lint-shell: shellcheck $shellcheck_version does not know the optional check '$check_name';" >&2
        echo "  the -o below would drop it silently and the gate would report a false pass." >&2
        exit 1
    fi
done

scripts=(scripts/*.sh)
if [[ ! -e "${scripts[0]}" ]]; then
    echo "lint-shell: no shell scripts matched scripts/*.sh" >&2
    exit 1
fi

# Optional checks, enabled individually. Every one named here reports zero
# findings in this tree, so the list is a floor on what the gate can see, not a
# concession: a check that only ever fires on new code is the cheapest kind of
# defect gate.
#
# check-set-e-suppressed stays off. It flags `set -e` in a context where it
# silently does not apply, which is a real way to run past a failure, but it
# also flags the deliberate status-test idiom (`cmd || die`, `if ! cmd`) on
# functions that return their own status -- 34 sites, 36 of the 39 findings in
# conv-store-backup.sh's self-test, and every one intentional. Enabling it
# would mean 34 suppressions that all say the same thing, which teaches the
# reader to ignore the check. It is worth enabling the day a script has a
# helper whose status is assumed rather than returned; that site is a defect
# this tree does not have.
#
# check-extra-masked-returns stays off: it fires 64 times across 10 scripts,
# nearly all `$(...)` substitutions inside `local`, `printf` and pipelines
# where a nonzero status is not a defect. It is worth enabling per script as
# each is made to propagate status; until then it is a documented skip, not a
# silent one.
#
# require-variable-braces and avoid-negated-conditions stay off as pure style.
shellcheck -x -o "$(IFS=,; echo "${optional_checks[*]}")" "${scripts[@]}"
echo "lint-shell: ${#scripts[@]} script(s) clean"
