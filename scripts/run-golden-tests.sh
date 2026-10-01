#!/usr/bin/env bash
# Run every tests/models/test_*.zig golden file for one backend.
# Usage: bash scripts/run-golden-tests.sh BACKEND
# Canonical for the CI golden-tests job (.github/workflows/golden_tests.yml).
#
# Each file is a `zig test` root that spawns the already-built zig-out/bin/agave
# against a real model and diffs the output, so the whole loop (one process per
# file, group markers, a per-file PASS/FAIL line, one nonzero exit at the end)
# lives here rather than in a `run:` block. Failures keep going so one broken
# model does not hide the rest; the exit status is the verdict.
#
# Kept in scripts/ so the blocking `zig build lint-shell` (scripts/*.sh)
# analyses the loop, and so a maintainer can reproduce one backend's run
# locally. The backend arrives as $1, never spliced into the script.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

backend="${1:-}"
if [[ -z "$backend" ]]; then
    echo "usage: $0 BACKEND   # e.g. cpu, metal, vulkan" >&2
    exit 2
fi

# Env indirection, not interpolation, at the call site: the backend name must
# never reach a shell body as text.
shopt -s nullglob
test_files=(tests/models/test_*.zig)
shopt -u nullglob
if [[ ${#test_files[@]} -eq 0 ]]; then
    echo "::error::No golden test files matched tests/models/test_*.zig"
    exit 1
fi

rc=0
failed=""
for test_file in "${test_files[@]}"; do
    echo "::group::$(basename "$test_file")"
    if zig test "$test_file" --test-filter "$backend"; then
        echo "PASS: $test_file"
    else
        rc=1
        failed="$failed $(basename "$test_file")"
        echo "::error::FAIL: $test_file ($backend backend)"
    fi
    echo "::endgroup::"
done

if [[ -n "$failed" ]]; then
    echo "::error::Failed golden tests:$failed"
fi
exit "$rc"
