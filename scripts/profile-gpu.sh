#!/usr/bin/env bash
# profile-gpu.sh, GPU kernel profiling via xctrace Metal System Trace
#
# Requires: agave compiled with metal encoder labels (current build has these).
# Requires: macOS with Metal backend (Apple Silicon or AMD/Intel Mac with Metal).
#
# Usage:
#   # 1. Start server, send load
#   ./zig-out/bin/agave model.gguf --serve &
#   PID=$!; sleep 8
#
#   # 2. Profile while load runs
#   ./scripts/profile-gpu.sh $PID [duration=25] [out=.scratch/agave-gpu-<ts>.trace]
#
# Then also open the .trace in Instruments for GPU timeline visualization.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/.scratch"

TARGET="${1:-agave}"
DURATION="${2:-25}"
OUTPUT="${3:-$ROOT/.scratch/agave-gpu-$(date +%Y%m%d_%H%M%S).trace}"

echo "=================================================="
echo "agave GPU profiling (Metal System Trace)"
echo "  Target   : $TARGET"
echo "  Duration : ${DURATION}s"
echo "  Output   : $OUTPUT"
echo "=================================================="
echo ""
echo "Send requests while recording:"
echo "  for i in \$(seq 1 50); do"
echo "    curl -s http://localhost:49453/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "         -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count to 50\"}],\"max_tokens\":80}' > /dev/null &"
echo "  done"
echo ""

xctrace record \
    --template "Metal System Trace" \
    --time-limit "${DURATION}s" \
    --output "$OUTPUT" \
    --attach "$TARGET" \
    2>&1 | grep -Ev "^$|Ctrl-C" || true

[[ ! -e "$OUTPUT" ]] && { echo "ERROR: trace not saved" >&2; exit 1; }
echo ""
echo "Trace: $OUTPUT"
echo ""

TMP_G=$(mktemp "$ROOT/.scratch/agave-gpu-intv-XXXXXXXXXX")
trap 'rm -f "$TMP_G"' EXIT

xctrace export --input "$OUTPUT" \
  --xpath '//table[@schema="metal-gpu-intervals"]' \
  --output "$TMP_G" 2>/dev/null || true

[[ ! -s "$TMP_G" ]] && {
    echo "No GPU data. Open .trace in Instruments for analysis."
    exit 0
}

uv run "$ROOT/scripts/xctrace_report.py" gpu "$TMP_G" "$OUTPUT"
