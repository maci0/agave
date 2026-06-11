#!/usr/bin/env bash
# profile-server.sh — attach xctrace to a running agave server process
#
# Usage:
#   # 1. Start server (model loads once)
#   ./zig-out/bin/agave model.gguf --serve &
#   SERVER_PID=$!
#   sleep 8   # wait for model to load
#
#   # 2. Run requests to generate load, then profile
#   ./scripts/profile-server.sh $SERVER_PID [duration=30] [out=/tmp/agave-server.trace]
#
# Or just profile by name:
#   ./scripts/profile-server.sh agave [duration=30]
#
# Best for: capturing pure decode hotpath without model-load overhead.
# Symbols: uses agave-debug or dsymutil'd agave for name resolution.

set -euo pipefail

TARGET="${1:-agave}"   # PID or process name
DURATION="${2:-30}"
OUTPUT="${3:-/tmp/agave-server-$(date +%Y%m%d_%H%M%S).trace}"
HOT_N="${4:-25}"

# Generate dSYM if we have ReleaseFast binary (helps symbol resolution)
if [[ -f "zig-out/bin/agave" && ! -d "zig-out/bin/agave.dSYM" ]]; then
    echo "→ Generating dSYM for symbol resolution..."
    dsymutil zig-out/bin/agave -o zig-out/bin/agave.dSYM 2>/dev/null || true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "agave server profiling (attach mode)"
echo "  Target   : $TARGET"
echo "  Duration : ${DURATION}s"
echo "  Output   : $OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
echo "✓ Trace: $OUTPUT"
echo ""

# Analyze using the same parser as profile.sh
TMP_XML=$(mktemp /tmp/agave-srv-XXXXXXXXXX)
trap "rm -f '$TMP_XML'" EXIT

xctrace export \
    --input "$OUTPUT" \
    --xpath '//table[@schema="time-profile"]' \
    --output "$TMP_XML" \
    2>/dev/null || true

if [[ -s "$TMP_XML" ]]; then
    # Ref-aware parser: xctrace reuses <weight ref="N"/> and <frame ref="N"/>
    # after the first definition. Resolve refs before accumulating.
    python3 - "$TMP_XML" "$HOT_N" << 'PYEOF'
import re, sys

path, hot_n = sys.argv[1], int(sys.argv[2])
try:
    data = open(path, "r", errors="replace").read()
except Exception as e:
    print(f"  (read error: {e})"); sys.exit(0)

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

self_ns = {}; total_ns = {}

for row in re.split(r'</?row\b[^>]*>', data):
    wm = re.search(r'<weight\b([^>]+)>', row)
    if not wm: continue
    wattrs = wm.group(1)
    wfull = re.search(r'<weight\b[^>]*>(\d+)</weight>', row)
    if wfull:
        ns = int(wfull.group(1))
    else:
        ref_m = re.search(r'\bref="(\d+)"', wattrs)
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
    print("  (no symbol data — open in Instruments for interactive analysis)"); sys.exit(0)

grand  = sum(self_ns.values())
ranked = sorted(self_ns.items(), key=lambda x: -x[1])[:hot_n]

print(f"\n━━ Hot paths ({len(self_ns)} symbols, {grand/1e9:.2f}s CPU time) ━━━━━━━━━━━━━━━━━━━━")
print(f"  NOTE: Metal GPU work shows as 'thread_start/IOGPUCommandQueue' — use")
print(f"  Metal System Trace template for GPU-side hotpaths.")
print(f"  {'Self%':>6}  {'Self ms':>8}  {'Tot%':>5}  Symbol")
print(f"  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*70}")
for sym, ns in ranked:
    s_pct = 100 * ns / grand if grand else 0
    t_pct = 100 * total_ns.get(sym, ns) / grand if grand else 0
    parts = sym.split('.')
    short = '.'.join(parts[-3:]) if len(parts) > 3 else sym
    short = short[-80:] if len(short) > 80 else short
    print(f"  {s_pct:>6.2f}%  {ns/1e6:>8.1f}  {t_pct:>5.1f}%  {short}")
PYEOF
else
    echo "  (no time-profile data exported — GPU-only workload?)"
    echo "  Try: open '$OUTPUT' in Instruments for Metal System Trace analysis"
fi

echo ""
echo "Open: open '$OUTPUT'"
