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

failed="$(jq -r 'to_entries[] | select(.value.result != "success") | "\(.key)=\(.value.result)"' <<<"$NEEDS_JSON")"
if [[ -n "$failed" ]]; then
    while read -r pair; do
        echo "::error::Required CI job '${pair%%=*}' result was '${pair#*=}' (expected success)"
    done <<<"$failed"
    exit 1
fi

# NEEDS_JSON is built from the ci-pass job's own `needs:`, so the payload can
# only ever report on the jobs that list already names. Editing a job out of
# that list is the one drift nothing downstream notices: the job keeps running,
# ci-pass goes green, and a check the repo believes is blocking stops blocking.
# Compare it against the jobs the workflow actually defines, so the required set
# is the workflow minus this gate rather than whatever the list was last edited
# to. Read from the checkout the job already makes for check-ci-pass.sh itself.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow="$ROOT/.github/workflows/ci.yml"
if [[ ! -f "$workflow" ]]; then
    echo "check-ci-pass: $workflow not found; cannot verify the required-job list" >&2
    exit 1
fi
# Job ids are the only `  <id>:` keys under `jobs:`; job fields nest deeper, and
# the workflow's own top-level keys (on:, env:, permissions:) sit before `jobs:`.
shopt -s nullglob
mapfile -t required < <(
    awk '/^jobs:[[:space:]]*$/ { in_jobs = 1; next } in_jobs && /^  [a-zA-Z0-9_-]+:[[:space:]]*$/ {
        line = $0
        sub(/^  /, "", line)
        sub(/:[[:space:]]*$/, "", line)
        print line
    }' "$workflow" |
        grep -v '^ci-pass$' |
        LC_ALL=C sort
)
shopt -u nullglob
if [[ "${#required[@]}" -eq 0 ]]; then
    echo "check-ci-pass: parsed no job ids from $workflow" >&2
    exit 1
fi
missing=""
for job in "${required[@]}"; do
    if ! jq -e --arg job "$job" 'has($job)' <<<"$NEEDS_JSON" >/dev/null; then
        missing="$missing $job"
    fi
done
if [[ -n "$missing" ]]; then
    echo "::error::CI job(s) missing from the ci-pass needs: list:${missing}"
    echo "check-ci-pass: every job except ci-pass itself must be a required check" >&2
    exit 1
fi
echo "All required CI jobs succeeded: $(jq -r 'keys | join(", ")' <<<"$NEEDS_JSON")"
