#!/usr/bin/env bash
# Ratchet on the `.oxlintrc.json` ignorePatterns list.
#
# oxlint is the blocking linter for the TypeScript under src/web/ and web/, so
# every path in ignorePatterns is code that merges without being linted. The
# list is a ratchet: a path may leave it, nothing may join it, and a path that
# no longer silences any file is dead weight that hides the fact. Both
# directions are checked here so the list cannot drift upward between CI runs.
#
# Canonical: zig build lint-web (via scripts/lint-web.sh) and CI job `lint-web`.
# Exit 0 when the list is a subset of the known-debt set below, 1 otherwise.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Every pattern oxlint is allowed to skip, with the reason it exists.
#   tools/oxlint/anti-slop/**  the local rules plugin, linted by its own tests
#   src/web/app.js, web/*.js  committed bun output, compared by
#                             scripts/check-web-artifacts.sh
# The dot directories and build output trees are tool and cache roots.
# No .ts/.tsx source is skipped: every one of them is linted.
# Removing an entry here is only correct once it is gone from ignorePatterns.
allowed_ignore_patterns=(
    "node_modules/**"
    "tools/oxlint/anti-slop/**"
    ".claude/**"
    ".opencode/**"
    ".pi-subagents/**"
    "zig-out/**"
    ".zig-cache/**"
    ".web-ts-out/**"
    "src/web/app.js"
    "web/agave.js"
    "web/shell.js"
    "vendor/**"
)

if [[ ! -f .oxlintrc.json ]]; then
    echo "check-web-lint-scope: .oxlintrc.json not found at the repo root" >&2
    exit 1
fi

# The array literal only, from the key to the line that closes it, so a
# similarly named key elsewhere in the file cannot widen the read. A `while
# read` loop, not mapfile: macOS still ships bash 3.2 and this runs on a
# macOS workstation too.
ignored_patterns=()
while IFS= read -r line; do
    ignored_patterns+=("$line")
done < <(
    sed -n '/"ignorePatterns"/,/^[[:space:]]*]/p' .oxlintrc.json |
        grep -v '"ignorePatterns"' | grep -oE '"[^"]+"' | tr -d '"' | sort -u
)

if [[ ${#ignored_patterns[@]} -eq 0 ]]; then
    echo "check-web-lint-scope: no ignorePatterns read from .oxlintrc.json" >&2
    exit 1
fi

status=0
for pattern in "${ignored_patterns[@]}"; do
    known=0
    for allowed in "${allowed_ignore_patterns[@]}"; do
        if [[ "$pattern" == "$allowed" ]]; then
            known=1
            break
        fi
    done
    if [[ $known -eq 0 ]]; then
        echo "error: $pattern is excluded from oxlint but is not on the known-debt list in $0" >&2
        status=1
        continue
    fi
    # A literal path that no longer exists silences nothing, so it can only
    # mask a later edit to a file with that name. Directory globs are skipped:
    # they describe roots that come and go with the toolchain.
    if [[ "$pattern" != *'*' && ! -e "$pattern" ]]; then
        echo "error: ignorePatterns lists $pattern, which does not exist" >&2
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    echo "Lint the file, or state in docs/TODO.md #13 why it stays excluded." >&2
    exit 1
fi

echo "check-web-lint-scope: ${#ignored_patterns[@]} ignorePatterns entry(ies), all known debt"
