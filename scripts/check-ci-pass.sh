#!/usr/bin/env bash
# Aggregate gate for the CI workflow: every job listed in the ci-pass job's
# `needs:` must have reported success. It reads that payload from the
# NEEDS_JSON env var, which the workflow fills from ${{ toJSON(needs) }};
# splicing it into a script body instead would put workflow data in source.
#
# Kept in scripts/ rather than inline in the YAML so the blocking
# `zig build lint-shell` (scripts/*.sh) analyses it. Canonical: bash
# scripts/check-ci-pass.sh with NEEDS_JSON set.
set -euo pipefail
export LC_ALL=C TZ=UTC

if ! command -v jq >/dev/null 2>&1; then
    echo "check-ci-pass: jq not found. Install jq, then rerun this script." >&2
    exit 1
fi

if [[ -z "${NEEDS_JSON:-}" ]]; then
    echo "check-ci-pass: NEEDS_JSON is unset; pass the job results as JSON" >&2
    exit 1
fi

# Reading the payload keeps the required-job list in the workflow's `needs:`
# above, so a new job cannot be added there and silently go unchecked here.
failed="$(jq -r 'to_entries[] | select(.value.result != "success") | "\(.key)=\(.value.result)"' <<<"$NEEDS_JSON")"
if [[ -n "$failed" ]]; then
    while read -r pair; do
        echo "::error::Required CI job '${pair%%=*}' result was '${pair#*=}' (expected success)"
    done <<<"$failed"
    exit 1
fi
echo "All required CI jobs succeeded: $(jq -r 'keys | join(", ")' <<<"$NEEDS_JSON")"
