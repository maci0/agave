#!/usr/bin/env bash
# profile-analyze.sh, analyze an xctrace .trace file and show hot functions
#
# Usage: ./scripts/profile-analyze.sh <file.trace> [--hot N] [--filter PATTERN]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'
}

die() {
    echo "profile-analyze: $*" >&2
    exit 2
}

# A value-taking option at the end of argv would otherwise abort with a bare
# bash "unbound variable", which names neither the option nor the fix.

TRACE=""; HOT_N=30; FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hot | --filter)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            if [[ "$1" == "--hot" ]]; then HOT_N="$2"; else FILTER="$2"; fi
            shift 2
            ;;
        *.trace)  TRACE="$1"; shift ;;
        -h | --help) usage; exit 0 ;;
        *)        die "unknown argument '$1' (see --help)" ;;
    esac
done

[[ -n "$TRACE" ]] || die "missing <file.trace> (see --help)"
[[ -e "$TRACE" ]] || die "'$TRACE' does not exist"

echo "Analyzing: $TRACE"
mkdir -p "$ROOT/.scratch"
TMP=$(mktemp "$ROOT/.scratch/agave-analyze-XXXXXXXXXX")
trap 'rm -f "$TMP"' EXIT

xctrace export \
    --input "$TRACE" \
    --xpath '//table[@schema="time-profile"]' \
    --output "$TMP" 2>/dev/null || true

[[ ! -s "$TMP" ]] && { echo "No time-profile data. Open: open '$TRACE'"; exit 0; }

uv run "$ROOT/scripts/xctrace_report.py" hot "$TMP" --top "$HOT_N" --filter "$FILTER"

echo ""
echo "Full analysis: open '$TRACE'"
