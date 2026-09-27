#!/usr/bin/env python3
"""Splice routed-expert tensors from a donor GGUF into a base GGUF.

Creates a mixed-quantization GGUF where most layers use the base file's
quantization (e.g. IQ2_XXS) but selected layers use the donor's higher
quantization (e.g. Q4_K) for routed experts only. Non-expert tensors
(shared experts, projections, routing) remain from the base file.

Usage:
    python3 splice_mixed_experts.py \
        --base model-iq2.gguf \
        --donor model-q4.gguf \
        --layers 37-42 \
        --out model-mixed.gguf

    python3 splice_mixed_experts.py \
        --base model-iq2.gguf \
        --donor model-q4.gguf \
        --layers 0-2,40-42 \
        --out model-mixed.gguf \
        --dry-run

Based on the mixed-quant splicing tool from antirez/ds4.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_io import Gguf, load, type_name, write_gguf

# `blk.N.ffn_gate_exps.weight` and the `model.layers.N.mlp.experts.*` spelling
# both name routed experts. Anything mentioning a shared expert is excluded:
# shared experts see every token, so upgrading them costs size everywhere.
EXPERT_PATTERNS = ("ffn_gate_exp", "ffn_up_exp", "ffn_down_exp", ".experts.",
                   "gate_proj", "up_proj", "down_proj")
LAYER_RE = re.compile(r"(?:blk\.|layers\.)(\d+)")


def parse_layer_ranges(spec: str) -> list[int]:
    """Parse '37-42' or '0-2,40-42' into a set of layer indices."""
    layers = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def is_routed_expert_tensor(name: str) -> bool:
    """Check if a tensor name belongs to a routed expert (not shared/dense)."""
    if "blk." not in name and "layers." not in name:
        return False
    if "shared" in name.lower():
        return False
    return any(p in name for p in EXPERT_PATTERNS)


def splice(base: Gguf, donor: Gguf, layers: set[int]):
    """Return (tensors, payloads, spliced) for a base file with donor experts."""
    donor_tensors = donor.by_name()
    tensors, payloads, spliced = [], [], []
    for t in base.tensors:
        name = t["name"].decode()
        match = LAYER_RE.search(name)
        layer = int(match.group(1)) if match else None
        source, dt = base, None
        if layer in layers and is_routed_expert_tensor(name):
            dt = donor_tensors.get(t["name"])
            if dt is None:
                continue  # not an expert tensor in the donor: keep the base bytes
            if dt["dims"] != t["dims"]:
                raise ValueError(
                    f"{name}: donor dims {dt['dims']} differ from base {t['dims']}")
            source = donor
        tensors.append({"name": t["name"], "dims": dt["dims"] if dt else t["dims"],
                        "type": dt["type"] if dt else t["type"]})
        payloads.append(source.blob(dt or t))
        if dt is not None:
            spliced.append((name, type_name(t["type"]), type_name(dt["type"]),
                            len(payloads[-1])))
    return tensors, payloads, spliced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="Base GGUF (lower quant, kept for most tensors)")
    parser.add_argument("--donor", required=True, type=Path, help="Donor GGUF (higher quant, experts copied from here)")
    parser.add_argument("--layers", required=True, help="Layer range(s) to splice, e.g. '37-42' or '0-2,40-42'")
    parser.add_argument("--out", required=True, type=Path, help="Output GGUF file")
    parser.add_argument("--dry-run", action="store_true", help="List the spliced tensors without writing")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()

    for label, path in (("--base", args.base), ("--donor", args.donor)):
        if not path.is_file():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2
    if not args.dry_run and not args.force and args.out.exists():
        print(f"error: {args.out} exists (use --force to overwrite)", file=sys.stderr)
        return 1

    layers = set(parse_layer_ranges(args.layers))
    base, donor = load(args.base), load(args.donor)
    tensors, payloads, spliced = splice(base, donor, layers)

    print(f"layers: {sorted(layers)}")
    print(f"base:   {args.base} ({args.base.stat().st_size / 1e9:.1f} GB)")
    print(f"donor:  {args.donor} ({args.donor.stat().st_size / 1e9:.1f} GB)")
    for name, from_type, to_type, nbytes in spliced:
        print(f"  {name}: {from_type} -> {to_type} ({nbytes} bytes)")
    if not spliced:
        print("  (no routed expert tensors matched; check --layers and the tensor names)")
        return 1

    if args.dry_run:
        print(f"\ndry run, {len(spliced)} tensors would be spliced, no file written")
        return 0

    out = write_gguf(base.version, base.kv, tensors, payloads, base.alignment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Publish by rename: an interrupted run leaves the previous output intact
    # rather than a half-written GGUF that reads as corrupt.
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out} ({len(out) / 2**30:.2f} GiB): "
          f"{len(spliced)} of {len(tensors)} tensors spliced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
