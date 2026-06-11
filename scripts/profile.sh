#!/usr/bin/env bash
# profile.sh — xctrace profiling harness for agave
#
# Usage:
#   ./scripts/profile.sh [options] -- <agave args>
#
# Options:
#   -t <template>   xctrace template (default: "Time Profiler")
#   -d <seconds>    duration cap in seconds (default: 30)
#   -o <path>       output .trace file (default: /tmp/agave-<ts>.trace)
#   --hot <n>       show top-N hot symbols (default: 20)
#   --metal         use "Metal System Trace" template
#
# Examples:
#   ./scripts/profile.sh -- ./zig-out/bin/agave model.gguf -q -n 100 "prompt"
#   ./scripts/profile.sh -d 60 --hot 30 -- ./zig-out/bin/agave model.gguf -q -n 300 "prompt"
#   ./scripts/profile.sh --metal -- ./zig-out/bin/agave model.gguf -q -n 100 "prompt"
#   open /tmp/agave-<ts>.trace   # Instruments for interactive drill-down

set -euo pipefail

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
        echo "→ Building debug binary for symbol resolution..."
        zig build 2>/dev/null || true
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
TRACE_FILE="${OUTPUT:-/tmp/agave-${TS}.trace}"
TMP_XML=$(mktemp /tmp/agave-profile-XXXXXXXXXX)
trap "rm -f '$TMP_XML'" EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "agave profiling harness"
echo "  Template : $TEMPLATE"
echo "  Duration : ${DURATION}s"
echo "  Output   : $TRACE_FILE"
echo "  Command  : ${CMD[*]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

xctrace record \
    --template "$TEMPLATE" \
    --time-limit "${DURATION}s" \
    --output "$TRACE_FILE" \
    --launch -- "${CMD[@]}" \
    2>&1 | grep -Ev '^$|Ctrl-C' || true

[[ ! -e "$TRACE_FILE" ]] && { echo "ERROR: trace not saved" >&2; exit 1; }
echo ""
echo "✓ Trace: $TRACE_FILE"
echo ""

# ── Parse results ──────────────────────────────────────────────────

if [[ "$TEMPLATE" == "Time Profiler" || "$TEMPLATE" == "CPU Profiler" ]]; then
    echo "━━ Hot paths ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    xctrace export \
        --input "$TRACE_FILE" \
        --xpath '//table[@schema="time-profile"]' \
        --output "$TMP_XML" \
        2>/dev/null || true

    if [[ -s "$TMP_XML" ]]; then
        # Ref-aware regex parser — no XML library (avoids XXE/DoS).
        # xctrace compresses repeated elements: second occurrence is <elem ref="N"/>
        python3 - "$TMP_XML" "$HOT_N" << 'PYEOF'
import re, sys

path, hot_n = sys.argv[1], int(sys.argv[2])
try:
    data = open(path, "r", errors="replace").read()
except Exception as e:
    print(f"  (read error: {e})")
    sys.exit(0)

weight_vals = {}
for m in re.finditer(r'<weight\b[^>]*\bid="(\d+)"[^>]*>(\d+)</weight>', data):
    weight_vals[m.group(1)] = int(m.group(2))

frame_names = {}
for m in re.finditer(r'<frame\b[^>]+>', data):
    tag = m.group(0)
    id_m = re.search(r'\bid="(\d+)"', tag)
    nm_m = re.search(r'\bname="([^"]{1,256})"', tag)
    if id_m and nm_m:
        frame_names[id_m.group(1)] = nm_m.group(1)

self_ns  = {}
total_ns = {}

for row in re.split(r'</?row\b[^>]*>', data):
    wm = re.search(r'<weight\b([^>]+)>', row)
    if not wm: continue
    wfull = re.search(r'<weight\b[^>]*>(\d+)</weight>', row)
    if wfull:
        ns = int(wfull.group(1))
    else:
        ref_m = re.search(r'\bref="(\d+)"', wm.group(1))
        if ref_m and ref_m.group(1) in weight_vals:
            ns = weight_vals[ref_m.group(1)]
        else:
            continue

    frames = []
    for tag_m in re.finditer(r'<frame\b[^>]+>', row):
        tag = tag_m.group(0)
        nm_m  = re.search(r'\bname="([^"]{1,256})"', tag)
        ref_m = re.search(r'\bref="(\d+)"', tag)
        if nm_m: frames.append(nm_m.group(1))
        elif ref_m and ref_m.group(1) in frame_names:
            frames.append(frame_names[ref_m.group(1)])

    if not frames: continue
    self_ns[frames[-1]] = self_ns.get(frames[-1], 0) + ns
    for f in set(frames): total_ns[f] = total_ns.get(f, 0) + ns

if not self_ns:
    print("  (no frame data — open .trace in Instruments)")
    sys.exit(0)

grand = sum(self_ns.values())
ranked = sorted(self_ns.items(), key=lambda x: -x[1])[:hot_n]

print(f"  {'Self%':>6}  {'Self ms':>8}  {'Tot%':>5}  Symbol")
print(f"  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*70}")
for sym, ns in ranked:
    s_pct = 100 * ns / grand if grand else 0
    t_pct = 100 * total_ns.get(sym, ns) / grand if grand else 0
    # Shorten Zig mangled names: keep last 3 segments
    parts = sym.split('.')
    short = '.'.join(parts[-3:]) if len(parts) > 3 else sym
    short = short[-85:] if len(short) > 85 else short
    print(f"  {s_pct:>6.2f}%  {ns/1e6:>8.1f}  {t_pct:>5.1f}%  {short}")

print(f"\n  Total sampled: {grand/1e9:.2f}s  |  {len(self_ns)} leaf symbols")
PYEOF
    else
        echo "  (export failed — open in Instruments for full analysis)"
    fi

elif [[ "$TEMPLATE" == "Metal System Trace" ]]; then
    echo "━━ Metal GPU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    xctrace export \
        --input "$TRACE_FILE" \
        --xpath '//table[starts-with(@schema,"gpu")]' \
        --output "$TMP_XML" \
        2>/dev/null || true
    if [[ -s "$TMP_XML" ]]; then
        python3 - "$TMP_XML" << 'PYEOF'
import re, sys
data = open(sys.argv[1], "r", errors="replace").read()
seen = set()
for m in re.finditer(r'(?:name|symbol)="([^"]{1,256})"', data):
    n = m.group(1)
    if n not in seen:
        seen.add(n)
        print(f"  {n}")
if not seen:
    print("  (no GPU data — open .trace in Instruments)")
PYEOF
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Interactive: open '$TRACE_FILE'"

# ── Server-attach helper ────────────────────────────────────────────
# Usage: profile-server <agave-pid> [duration=30] [out=/tmp/agave.trace]
#
# Attaches to a RUNNING agave server and profiles it.
# Avoids the model-load startup cost — only captures decode hotpaths.
#
# Example:
#   ./zig-out/bin/agave model.gguf --serve &
#   sleep 5                          # wait for model to load
#   ./scripts/profile.sh attach $!   # profile the running server
