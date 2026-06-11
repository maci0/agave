#!/usr/bin/env bash
# profile-analyze.sh — analyze xctrace .trace file, show hot functions
# Usage: ./scripts/profile-analyze.sh <file.trace> [--hot N] [--filter PATTERN]
set -euo pipefail

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
TMP=$(mktemp /tmp/agave-analyze-XXXXXXXXXX)
trap "rm -f '$TMP'" EXIT

xctrace export \
    --input "$TRACE" \
    --xpath '//table[@schema="time-profile"]' \
    --output "$TMP" 2>/dev/null || true

[[ ! -s "$TMP" ]] && { echo "No time-profile data. Open: open '$TRACE'"; exit 0; }

# Regex-based parser (no XML library — avoids XXE/DoS).
# xctrace reuses elements via ref attributes:
#   <weight id="N" fmt="X ms">NANOSECONDS</weight>  — first occurrence
#   <weight ref="N"/>                                — subsequent rows reuse
#   <frame  id="N" name="SYMBOL" addr="...">...</frame> — first occurrence
#   <frame  ref="N"/>                                    — subsequent rows reuse
python3 - "$TMP" "$HOT_N" "$FILTER" << 'PYEOF'
import re, sys

path, hot_n, filter_pat = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    data = open(path, "r", errors="replace").read()
except Exception as e:
    print(f"  Read error: {e}"); sys.exit(0)

# Pass 1: build id→value maps for elements that xctrace compresses via ref=
weight_vals = {}  # id → nanoseconds
for m in re.finditer(r'<weight\b[^>]*\bid="(\d+)"[^>]*>(\d+)</weight>', data):
    weight_vals[m.group(1)] = int(m.group(2))

frame_names = {}  # id → symbol name
for m in re.finditer(r'<frame\b[^>]+>', data):
    tag = m.group(0)
    id_m = re.search(r'\bid="(\d+)"', tag)
    nm_m = re.search(r'\bname="([^"]{1,256})"', tag)
    if id_m and nm_m:
        frame_names[id_m.group(1)] = nm_m.group(1)

# Pass 2: accumulate self/total time per symbol across all rows
self_ns  = {}
total_ns = {}

for row in re.split(r'</?row\b[^>]*>', data):
    # Resolve weight: direct or via ref
    wm = re.search(r'<weight\b([^>]+)>', row)
    if not wm:
        continue
    wattrs = wm.group(1)
    # Check for direct value first
    val_m = re.search(r'>(\d+)<', row[wm.start():wm.end()+20])
    if val_m:
        ns = int(val_m.group(1))
    else:
        ref_m = re.search(r'\bref="(\d+)"', wattrs)
        if ref_m and ref_m.group(1) in weight_vals:
            ns = weight_vals[ref_m.group(1)]
        else:
            # Try to get value from weight element content
            wfull = re.search(r'<weight\b[^>]*>(\d+)</weight>', row)
            if wfull:
                ns = int(wfull.group(1))
            else:
                continue

    # Collect frames
    frames = []
    for tag_m in re.finditer(r'<frame\b[^>]+>', row):
        tag = tag_m.group(0)
        nm_m  = re.search(r'\bname="([^"]{1,256})"', tag)
        ref_m = re.search(r'\bref="(\d+)"', tag)
        if nm_m:
            frames.append(nm_m.group(1))
        elif ref_m and ref_m.group(1) in frame_names:
            frames.append(frame_names[ref_m.group(1)])

    if not frames:
        continue
    if filter_pat and not any(filter_pat in f for f in frames):
        continue

    leaf = frames[-1]
    self_ns[leaf]  = self_ns.get(leaf, 0) + ns
    for f in set(frames):
        total_ns[f] = total_ns.get(f, 0) + ns

if not self_ns:
    print("No symbol data. Open .trace in Instruments for analysis.")
    sys.exit(0)

grand  = sum(self_ns.values())
ranked = sorted(self_ns.items(), key=lambda x: -x[1])[:hot_n]

print(f"\nTop {min(hot_n, len(ranked))} symbols  ({grand/1e9:.2f}s sampled, {len(self_ns)} unique):")
print(f"  {'Rank':>4}  {'Self%':>6}  {'Self ms':>8}  {'Tot%':>5}  Symbol")
print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*75}")
for i, (sym, ns) in enumerate(ranked, 1):
    s_pct = 100 * ns / grand if grand else 0
    t_pct = 100 * total_ns.get(sym, ns) / grand if grand else 0
    parts = sym.split('.')
    short = '.'.join(parts[-3:]) if len(parts) > 3 else sym
    short = short[-75:] if len(short) > 75 else short
    print(f"  {i:>4}  {s_pct:>6.2f}%  {ns/1e6:>8.1f}  {t_pct:>5.1f}%  {short}")
if filter_pat:
    print(f"\n  (filtered: '{filter_pat}')")
PYEOF

echo ""
echo "Full analysis: open '$TRACE'"
