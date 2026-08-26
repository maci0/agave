# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize xctrace XML exports: CPU hot paths, Metal GPU kernel timings, object labels.

Shared by scripts/profile.sh, profile-server.sh, profile-analyze.sh and profile-gpu.sh.
Parsing is regex-based on purpose: no XML library, so no XXE or entity-expansion
surface on a file produced by an external tool.

xctrace compresses repeated elements. The first occurrence carries `id="N"` and the
value, every later one is `<elem ref="N"/>`. Both passes below resolve those refs.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

# Rows shorter than this are Metal dispatch noise, not measurable kernel work.
MIN_GPU_INTERVAL_NS = 2000
# Longest symbol tail kept in the hot-path table; longer names are truncated left.
MAX_SYMBOL_WIDTH = 75
# Trailing dot-separated segments kept from a mangled Zig symbol.
SYMBOL_TAIL_SEGMENTS = 3
# Encoder-label substrings that mark a GPU row as agave work.
AGAVE_GPU_LABELS = (
    "gemv_",
    "rms_norm",
    "sdpa_",
    "rope",
    "kv_append",
    "Compute Command",
    "GPU Execution",
)

WEIGHT_DEF_RE = re.compile(r'<weight\b[^>]*\bid="(\d+)"[^>]*>(\d+)</weight>')
WEIGHT_VALUE_RE = re.compile(r"<weight\b[^>]*>(\d+)</weight>")
WEIGHT_OPEN_RE = re.compile(r"<weight\b([^>]+)>")
FRAME_TAG_RE = re.compile(r"<frame\b[^>]+>")
DURATION_DEF_RE = re.compile(r'<duration\s+id="(\d+)"[^>]*>(\d+)</duration>')
DURATION_VALUE_RE = re.compile(r"<duration\b[^>]*>(\d+)</duration>")
DURATION_REF_RE = re.compile(r'<duration\s+ref="(\d+)"')
LABEL_DEF_RE = re.compile(r'<formatted-label\s+id="(\d+)"[^>]*fmt="([^"]{0,400})"')
LABEL_VALUE_RE = re.compile(r'<formatted-label\b[^>]*fmt="([^"]{1,400})"')
LABEL_REF_RE = re.compile(r'<formatted-label\s+ref="(\d+)"')
ID_ATTR_RE = re.compile(r'\bid="(\d+)"')
REF_ATTR_RE = re.compile(r'\bref="(\d+)"')
NAME_ATTR_RE = re.compile(r'\bname="([^"]{1,256})"')
ANY_NAME_RE = re.compile(r'(?:name|symbol)="([^"]{1,256})"')
ROW_SPLIT_RE = re.compile(r"</?row\b[^>]*>")


def read_export(path: str) -> str | None:
    """Return the export text, or None after reporting an unreadable file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        print(f"  (read error: {exc})")
        return None


def shorten(symbol: str) -> str:
    """Trim a mangled symbol to its trailing segments, then to the column width."""
    parts = symbol.split(".")
    if len(parts) > SYMBOL_TAIL_SEGMENTS:
        symbol = ".".join(parts[-SYMBOL_TAIL_SEGMENTS:])
    return symbol[-MAX_SYMBOL_WIDTH:]


def accumulate_time_profile(data: str, keep: str) -> tuple[dict[str, int], dict[str, int]]:
    """Sum self and total nanoseconds per symbol, keeping only stacks matching `keep`."""
    weight_vals: dict[str, int] = {
        m.group(1): int(m.group(2)) for m in WEIGHT_DEF_RE.finditer(data)
    }

    frame_names: dict[str, str] = {}
    for match in FRAME_TAG_RE.finditer(data):
        tag = match.group(0)
        frame_id = ID_ATTR_RE.search(tag)
        frame_name = NAME_ATTR_RE.search(tag)
        if frame_id and frame_name:
            frame_names[frame_id.group(1)] = frame_name.group(1)

    self_ns: dict[str, int] = collections.defaultdict(int)
    total_ns: dict[str, int] = collections.defaultdict(int)

    for row in ROW_SPLIT_RE.split(data):
        weight_open = WEIGHT_OPEN_RE.search(row)
        if not weight_open:
            continue
        inline = WEIGHT_VALUE_RE.search(row)
        if inline:
            nanos = int(inline.group(1))
        else:
            ref = REF_ATTR_RE.search(weight_open.group(1))
            if not ref or ref.group(1) not in weight_vals:
                continue
            nanos = weight_vals[ref.group(1)]

        frames: list[str] = []
        for match in FRAME_TAG_RE.finditer(row):
            tag = match.group(0)
            name = NAME_ATTR_RE.search(tag)
            ref = REF_ATTR_RE.search(tag)
            if name:
                frames.append(name.group(1))
            elif ref and ref.group(1) in frame_names:
                frames.append(frame_names[ref.group(1)])

        if not frames:
            continue
        if keep and not any(keep in frame for frame in frames):
            continue

        self_ns[frames[-1]] += nanos
        for frame in set(frames):
            total_ns[frame] += nanos

    return self_ns, total_ns


def report_hot(args: argparse.Namespace) -> int:
    data = read_export(args.export)
    if data is None:
        return 0

    self_ns, total_ns = accumulate_time_profile(data, args.filter)
    if not self_ns:
        print("  (no symbol data, open the .trace in Instruments)")
        return 0

    grand = sum(self_ns.values())
    ranked = sorted(self_ns.items(), key=lambda item: -item[1])[: args.top]

    print(
        f"\nTop {len(ranked)} symbols  "
        f"({grand / 1e9:.2f}s sampled, {len(self_ns)} leaf symbols):"
    )
    print(f"  {'Rank':>4}  {'Self%':>6}  {'Self ms':>8}  {'Tot%':>5}  Symbol")
    print(f"  {'-' * 4}  {'-' * 6}  {'-' * 8}  {'-' * 5}  {'-' * MAX_SYMBOL_WIDTH}")
    for rank, (symbol, nanos) in enumerate(ranked, 1):
        self_pct = 100 * nanos / grand if grand else 0
        total_pct = 100 * total_ns.get(symbol, nanos) / grand if grand else 0
        print(
            f"  {rank:>4}  {self_pct:>6.2f}%  {nanos / 1e6:>8.1f}  "
            f"{total_pct:>5.1f}%  {shorten(symbol)}"
        )
    if args.filter:
        print(f"\n  (filtered: '{args.filter}')")
    return 0


def report_gpu(args: argparse.Namespace) -> int:
    data = read_export(args.export)
    if data is None:
        return 0
    data = data.replace("&amp;", "&")

    duration_vals: dict[str, int] = {
        m.group(1): int(m.group(2)) for m in DURATION_DEF_RE.finditer(data)
    }
    label_vals: dict[str, str] = {
        m.group(1): m.group(2) for m in LABEL_DEF_RE.finditer(data)
    }

    kernel_ns: dict[str, int] = collections.defaultdict(int)
    kernel_count: collections.Counter[str] = collections.Counter()
    batch_ns: dict[str, int] = collections.defaultdict(int)
    batch_count: collections.Counter[str] = collections.Counter()
    total_ns = 0

    for row in ROW_SPLIT_RE.split(data):
        inline = DURATION_VALUE_RE.search(row)
        ref = DURATION_REF_RE.search(row)
        if inline:
            nanos = int(inline.group(1))
        elif ref and ref.group(1) in duration_vals:
            nanos = duration_vals[ref.group(1)]
        else:
            continue
        if nanos < MIN_GPU_INTERVAL_NS:
            continue

        inline_label = LABEL_VALUE_RE.search(row)
        if inline_label:
            label = inline_label.group(1)
        else:
            label_ref = LABEL_REF_RE.search(row)
            label = label_vals.get(label_ref.group(1), "") if label_ref else ""
        if not label:
            continue
        if "agave" not in label and not any(k in label for k in AGAVE_GPU_LABELS):
            continue

        clean = re.sub(r"Command Buffer \d+:", "", label)
        clean = re.sub(r"\s*\(.*?\)\s*0x[0-9a-f]+", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean)
        if not clean or clean == "GPU Execution":
            continue

        total_ns += nanos
        batch_ns[clean] += nanos
        batch_count[clean] += 1

        # An encoder batch label joins its fused kernels with "&"; split the
        # interval evenly since xctrace reports no per-kernel breakdown.
        parts = [part.strip() for part in clean.split("&") if part.strip()]
        share = nanos // max(1, len(parts))
        for part in parts:
            kernel_ns[part] += share
            kernel_count[part] += 1

    if total_ns == 0:
        print("No labeled agave GPU data. Ensure requests were sent during recording.")
        print(f"Interactive: open '{args.trace}' (Metal System Trace, GPU timeline)")
        return 0

    print(f"GPU kernel timing breakdown  (total GPU time: {total_ns / 1e6:.0f}ms)\n")
    print(f"  {'Kernel':<26}  {'ms':>8}  {'%':>6}  {'N':>8}")
    print(f"  {'-' * 26}  {'-' * 8}  {'-' * 6}  {'-' * 8}")
    for kernel, nanos in sorted(kernel_ns.items(), key=lambda item: -item[1]):
        pct = 100 * nanos / total_ns
        print(f"  {kernel:<26}  {nanos / 1e6:>8.1f}  {pct:>6.1f}%  {kernel_count[kernel]:>8}")

    print("\nTop encoder batches (shows which ops are grouped):")
    for label, nanos in sorted(batch_ns.items(), key=lambda item: -item[1])[: args.top]:
        print(f"  {nanos / 1e6:>8.1f}ms  {batch_count[label]:>4}x  {label[:80]}")

    print(f"\nOpen in Instruments: open '{args.trace}'")
    return 0


def report_labels(args: argparse.Namespace) -> int:
    data = read_export(args.export)
    if data is None:
        return 0
    seen: list[str] = []
    for match in ANY_NAME_RE.finditer(data):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    for name in seen:
        print(f"  {name}")
    if not seen:
        print("  (no GPU data, open the .trace in Instruments)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    hot = sub.add_parser("hot", help="rank CPU self time per symbol from a time-profile export")
    hot.add_argument("export", help="xctrace XML export of //table[@schema='time-profile']")
    hot.add_argument("--top", type=int, default=25, help="rows to show (default: 25)")
    hot.add_argument("--filter", default="", help="only stacks containing this substring")
    hot.set_defaults(func=report_hot)

    gpu = sub.add_parser("gpu", help="rank Metal GPU kernel time from a metal-gpu-intervals export")
    gpu.add_argument("export", help="xctrace XML export of //table[@schema='metal-gpu-intervals']")
    gpu.add_argument("trace", help="path of the .trace bundle, printed in the footer")
    gpu.add_argument("--top", type=int, default=8, help="encoder batches to show (default: 8)")
    gpu.set_defaults(func=report_gpu)

    labels = sub.add_parser("labels", help="list distinct name/symbol attributes in an export")
    labels.add_argument("export", help="xctrace XML export of any gpu table")
    labels.set_defaults(func=report_labels)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
