#!/usr/bin/env bash
# profile.sh, xctrace profiling harness for agave
#
# Usage:
#   ./scripts/profile.sh [options] -- <agave args>
#
# Options:
#   -t <template>   xctrace template (default: "Time Profiler")
#   -d <seconds>    duration cap in seconds (default: 30)
#   -o <path>       output .trace file (default: .scratch/agave-<ts>.trace)
#   --hot <n>       show top-N hot symbols (default: 20)
#   --metal         use "Metal System Trace" template
#
# Examples:
#   ./scripts/profile.sh -- ./zig-out/bin/agave model.gguf -q -n 100 "prompt"
#   ./scripts/profile.sh -d 60 --hot 30 -- ./zig-out/bin/agave model.gguf -q -n 300 "prompt"
#   ./scripts/profile.sh --metal -- ./zig-out/bin/agave model.gguf -q -n 100 "prompt"
#
# To profile an already-running server without the model-load cost, use
# scripts/profile-server.sh (CPU) or scripts/profile-gpu.sh (Metal).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/.scratch"

TEMPLATE="Time Profiler"
DURATION=30
OUTPUT=""
HOT_N=20
METAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) TEMPLATE="$2"; shift 2 ;;
        -d) DURATION="$2"; shift 2 ;;
        -o) OUTPUT="$2"; shift 2 ;;
        --hot) HOT_N="$2"; shift 2 ;;
        --metal) METAL=true; shift ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [options] -- ./zig-out/bin/agave <model> [args...]" >&2
    exit 1
fi

# Prefer agave-debug for DWARF symbols (ReleaseFast loses most names)
CMD=("$@")
for i in "${!CMD[@]}"; do
    if [[ "${CMD[$i]}" == *"zig-out/bin/agave" && "${CMD[$i]}" != *"agave-debug"* ]]; then
        echo "-> Building debug binary for symbol resolution..."
        if ! zig build; then
            echo "warning: zig build failed; profiling without agave-debug symbols" >&2
        fi
        DBG="$(dirname "${CMD[$i]}")/agave-debug"
        if [[ -x "$DBG" ]]; then
            CMD[$i]="$DBG"
            echo "  Switched to: $DBG"
        fi
        break
    fi
done

[[ "$METAL" == true ]] && TEMPLATE="Metal System Trace"

TS=$(date +%Y%m%d_%H%M%S)
TRACE_FILE="${OUTPUT:-$ROOT/.scratch/agave-${TS}.trace}"
TMP_XML=$(mktemp "$ROOT/.scratch/agave-profile-XXXXXXXXXX")
trap 'rm -f "$TMP_XML"' EXIT

echo "=================================================="
echo "agave profiling harness"
echo "  Template : $TEMPLATE"
echo "  Duration : ${DURATION}s"
echo "  Output   : $TRACE_FILE"
echo "  Command  : ${CMD[*]}"
echo "=================================================="

xctrace record \
    --template "$TEMPLATE" \
    --time-limit "${DURATION}s" \
    --output "$TRACE_FILE" \
    --launch -- "${CMD[@]}" \
    2>&1 | grep -Ev '^$|Ctrl-C' || true

[[ ! -e "$TRACE_FILE" ]] && { echo "ERROR: trace not saved" >&2; exit 1; }
echo ""
echo "Trace: $TRACE_FILE"
echo ""

if [[ "$TEMPLATE" == "Time Profiler" || "$TEMPLATE" == "CPU Profiler" ]]; then
    echo "== Hot paths =========================================="

    xctrace export \
        --input "$TRACE_FILE" \
        --xpath '//table[@schema="time-profile"]' \
        --output "$TMP_XML" \
        2>/dev/null || true

    if [[ -s "$TMP_XML" ]]; then
        uv run "$ROOT/scripts/xctrace_report.py" hot "$TMP_XML" --top "$HOT_N"
    else
        echo "  (export failed, open in Instruments for full analysis)"
    fi

elif [[ "$TEMPLATE" == "Metal System Trace" ]]; then
    echo "== Metal GPU =========================================="
    xctrace export \
        --input "$TRACE_FILE" \
        --xpath '//table[starts-with(@schema,"gpu")]' \
        --output "$TMP_XML" \
        2>/dev/null || true
    if [[ -s "$TMP_XML" ]]; then
        uv run "$ROOT/scripts/xctrace_report.py" labels "$TMP_XML"
    fi
fi

echo ""
echo "=================================================="
echo "Interactive: open '$TRACE_FILE'"
