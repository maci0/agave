#!/usr/bin/env bash
# profile-gpu.sh — GPU kernel profiling via xctrace Metal System Trace
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
#   ./scripts/profile-gpu.sh $PID [duration=25] [out=/tmp/agave-gpu.trace]
#
# Then also: open /tmp/agave-gpu.trace in Instruments for GPU timeline visualization.

set -euo pipefail

TARGET="${1:-agave}"
DURATION="${2:-25}"
OUTPUT="${3:-/tmp/agave-gpu-$(date +%Y%m%d_%H%M%S).trace}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "agave GPU profiling (Metal System Trace)"
echo "  Target   : $TARGET"
echo "  Duration : ${DURATION}s"
echo "  Output   : $OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
echo "✓ Trace: $OUTPUT"
echo ""

TMP_G=$(mktemp /tmp/agave-gpu-intv-XXXXXXXXXX)
TMP_L=$(mktemp /tmp/agave-gpu-lbls-XXXXXXXXXX)
trap 'rm -f "$TMP_G" "$TMP_L"' EXIT

xctrace export --input "$OUTPUT" \
  --xpath '//table[@schema="metal-gpu-intervals"]' \
  --output "$TMP_G" 2>/dev/null || true

xctrace export --input "$OUTPUT" \
  --xpath '//table[@schema="metal-object-label"]' \
  --output "$TMP_L" 2>/dev/null || true

[[ ! -s "$TMP_G" ]] && {
    echo "No GPU data. Open .trace in Instruments for analysis."
    exit 0
}

# Regex-based parser — no XML library (avoids XXE/DoS).
# Labels from metal-object-label appear in metal-gpu-intervals formatted-label field.
python3 - "$TMP_G" "$OUTPUT" << 'PYEOF'
import re, sys, collections

gpu_path = sys.argv[1]
trace_path = sys.argv[2]

try:
    data = open(gpu_path, "r", errors="replace").read().replace("&amp;", "&")
except Exception as e:
    print(f"  Read error: {e}"); sys.exit(0)

duration_vals = {}
for m in re.finditer(r'<duration\s+id="(\d+)"[^>]*>(\d+)</duration>', data):
    duration_vals[m.group(1)] = int(m.group(2))

label_vals = {}
for m in re.finditer(r'<formatted-label\s+id="(\d+)"[^>]*fmt="([^"]{0,400})"', data):
    label_vals[m.group(1)] = m.group(2)

kern_ns   = collections.defaultdict(int)
kern_cnt  = collections.Counter()
total_ns  = 0
batch_ns  = collections.defaultdict(int)  # full encoder label → time
batch_cnt = collections.Counter()

for row in re.split(r'</?row\b[^>]*>', data):
    # Skip non-agave rows (no process annotation)
    if "agave" not in row and "27" not in row:  # crude filter
        pass  # still process — might be system GPU work for agave

    dur_m   = re.search(r'<duration\b[^>]*>(\d+)</duration>', row)
    dur_ref = re.search(r'<duration\s+ref="(\d+)"', row)
    if dur_m:
        ns = int(dur_m.group(1))
    elif dur_ref and dur_ref.group(1) in duration_vals:
        ns = duration_vals[dur_ref.group(1)]
    else:
        continue
    if ns < 2000:
        continue

    lm = re.search(r'<formatted-label\b[^>]*fmt="([^"]{1,400})"', row)
    if lm:
        full = lm.group(1)
    else:
        lr = re.search(r'<formatted-label\s+ref="(\d+)"', row)
        full = label_vals.get(lr.group(1), "") if lr else ""

    if not full:
        continue

    # Only process rows labeled as agave work
    if "agave" not in full and not any(
        k in full for k in ("gemv_", "rms_norm", "sdpa_", "rope", "kv_append",
                             "Compute Command", "GPU Execution")):
        continue

    clean = re.sub(r'Command Buffer \d+:', '', full)
    clean = re.sub(r'\s*\(.*?\)\s*0x[0-9a-f]+', '', clean).strip()
    clean = re.sub(r'\s+', ' ', clean)

    if not clean or clean == "GPU Execution":
        continue

    total_ns += ns
    batch_ns[clean]  = batch_ns.get(clean, 0) + ns
    batch_cnt[clean] += 1

    parts = [p.strip() for p in clean.split("&")]
    ns_each = ns // max(1, len(parts))
    for p in parts:
        if p:
            kern_ns[p]  = kern_ns.get(p, 0) + ns_each
            kern_cnt[p] += 1

if total_ns == 0:
    print("No labeled agave GPU data. Ensure requests were sent during recording.")
    print(f"Interactive: open '{trace_path}' (Metal System Trace → GPU timeline)")
    sys.exit(0)

print(f"GPU kernel timing breakdown  (total GPU time: {total_ns/1e6:.0f}ms)\n")
print(f"  {'Kernel':<26}  {'ms':>8}  {'%':>6}  {'N':>8}")
print(f"  {'-'*26}  {'-'*8}  {'-'*6}  {'-'*8}")
for k, ns in sorted(kern_ns.items(), key=lambda x: -x[1]):
    print(f"  {k:<26}  {ns/1e6:>8.1f}  {100*ns/total_ns:>6.1f}%  {kern_cnt[k]:>8}")

print(f"\nTop encoder batches (shows which ops are grouped):")
for lbl, ns in sorted(batch_ns.items(), key=lambda x: -x[1])[:8]:
    short = lbl[:80]
    print(f"  {ns/1e6:>8.1f}ms  {batch_cnt[lbl]:>4}x  {short}")

print(f"\nOpen in Instruments: open '{trace_path}'")
PYEOF
