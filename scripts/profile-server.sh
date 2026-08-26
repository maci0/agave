#!/usr/bin/env bash
# profile-server.sh, attach xctrace to a running agave server process
#
# Usage:
#   # 1. Start server (model loads once)
#   ./zig-out/bin/agave model.gguf --serve &
#   SERVER_PID=$!
#   sleep 8   # wait for model to load
#
#   # 2. Run requests to generate load, then profile
#   ./scripts/profile-server.sh $SERVER_PID [duration=30] [out=.scratch/agave-server-<ts>.trace]
#
# Or just profile by name:
#   ./scripts/profile-server.sh agave [duration=30]
#
# Best for: capturing pure decode hotpath without model-load overhead.
# Symbols: uses agave-debug or dsymutil'd agave for name resolution.
#
# Metal GPU work shows up here as thread_start/IOGPUCommandQueue. Use
# scripts/profile-gpu.sh for GPU-side hotpaths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/.scratch"

TARGET="${1:-agave}"   # PID or process name
DURATION="${2:-30}"
OUTPUT="${3:-$ROOT/.scratch/agave-server-$(date +%Y%m%d_%H%M%S).trace}"
HOT_N="${4:-25}"

# Generate dSYM if we have ReleaseFast binary (helps symbol resolution)
if [[ -f "$ROOT/zig-out/bin/agave" && ! -d "$ROOT/zig-out/bin/agave.dSYM" ]]; then
    echo "-> Generating dSYM for symbol resolution..."
    dsymutil "$ROOT/zig-out/bin/agave" -o "$ROOT/zig-out/bin/agave.dSYM" 2>/dev/null || true
fi

echo "=================================================="
echo "agave server profiling (attach mode)"
echo "  Target   : $TARGET"
echo "  Duration : ${DURATION}s"
echo "  Output   : $OUTPUT"
echo "=================================================="
echo ""
echo "TIP: while recording, send requests:"
echo "  curl http://localhost:49453/v1/chat/completions \\"
echo "       -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Count to 100\"}],\"max_tokens\":200}'"
echo ""

xctrace record \
    --template "Time Profiler" \
    --time-limit "${DURATION}s" \
    --output "$OUTPUT" \
    --attach "$TARGET" \
    2>&1 | grep -Ev '^$' || true

[[ ! -e "$OUTPUT" ]] && { echo "ERROR: trace not saved (is agave running?)" >&2; exit 1; }

echo ""
echo "Trace: $OUTPUT"
echo ""

TMP_XML=$(mktemp "$ROOT/.scratch/agave-srv-XXXXXXXXXX")
trap 'rm -f "$TMP_XML"' EXIT

xctrace export \
    --input "$OUTPUT" \
    --xpath '//table[@schema="time-profile"]' \
    --output "$TMP_XML" \
    2>/dev/null || true

if [[ -s "$TMP_XML" ]]; then
    uv run "$ROOT/scripts/xctrace_report.py" hot "$TMP_XML" --top "$HOT_N"
else
    echo "  (no time-profile data exported, GPU-only workload?)"
    echo "  Try: open '$OUTPUT' in Instruments for Metal System Trace analysis"
fi

echo ""
echo "Open: open '$OUTPUT'"
