#!/usr/bin/env bash
# Require GGUF weights under ./models before the golden tests run.
# Canonical: bash scripts/check-golden-models.sh (CI golden-tests job).
#
# The golden tests spawn the built binary against a real model, so a checkout
# with no weights is not a suite that "passes with nothing to do": every file
# would SKIP. Requiring at least one .gguf turns that into a red step naming the
# one thing the operator has to provide (a runner whose checkout has ./models
# mounted), instead of a green run that tested nothing.
#
# Kept in scripts/ rather than inline in the workflow so the blocking
# `zig build lint-shell` (scripts/*.sh) analyses it.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Hosted runners have no weights; self-hosted / manual runs must provide models/.
# `find` on a missing models/ errors under set -e, hence the 2>/dev/null + || true.
if ! find models -type f \( -name '*.gguf' -o -name '*.GGUF' \) 2>/dev/null | grep -q .; then
    echo "::error::Golden tests need at least one .gguf under models/. Re-dispatch with the 'runner' input set to a self-hosted label whose checkout has ./models mounted; GitHub-hosted runners never have weights."
    exit 1
fi
echo "Found GGUF models under models/"
