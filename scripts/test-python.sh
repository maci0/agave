#!/usr/bin/env bash
# Python unit tests: the stdlib unittest suites under scripts/ and tools/.
# Canonical: zig build test-python  (or this script from the repo root).
# Every suite here is network-free and dependency-free; a suite that needs
# weights or an API key does not belong in this list.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
    echo "test-python: python3 not found (Python 3.11+)." >&2
    exit 1
fi

# The suites the repo actually has, one path each so a new one is added here
# rather than discovered by a grep over the tree.
suites=(
    scripts/test_check_docs.py
    scripts/test_check_third_party_notices.py
    tools/mixed-quant/test_splice_mixed_experts.py
    tools/quality-testing/test_collect_continuations.py
)

for suite in "${suites[@]}"; do
    if [[ ! -f "$suite" ]]; then
        echo "test-python: missing suite $suite" >&2
        exit 1
    fi
    echo "test-python: $suite"
    python3 "$suite"
done

echo "test-python: ${#suites[@]} suite(s) passed"
