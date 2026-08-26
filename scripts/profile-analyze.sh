#!/usr/bin/env bash
# profile-analyze.sh, analyze an xctrace .trace file and show hot functions
# Usage: ./scripts/profile-analyze.sh <file.trace> [--hot N] [--filter PATTERN]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRACE=""; HOT_N=30; FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hot)    HOT_N="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        *.trace)  TRACE="$1"; shift ;;
        *)        echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$TRACE" || ! -e "$TRACE" ]] && { echo "Usage: $0 <file.trace> [--hot N] [--filter PATTERN]" >&2; exit 1; }

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
