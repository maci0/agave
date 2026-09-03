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
import json
import mmap
import os
import struct
import sys
from pathlib import Path


GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian


def parse_layer_ranges(spec):
    """Parse '37-42' or '0-2,40-42' into a set of layer indices."""
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def is_routed_expert_tensor(name):
    """Check if a tensor name belongs to a routed expert (not shared/dense)."""
    # Common patterns for routed expert tensors:
    # blk.N.ffn_gate_exps.weight, blk.N.ffn_up_exps.weight, blk.N.ffn_down_exps.weight
    # model.layers.N.mlp.experts.*.gate_proj.weight, etc.
    expert_patterns = [
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
        "ffn_gate_exp", "ffn_up_exp", "ffn_down_exp",
        ".experts.", "gate_proj", "up_proj", "down_proj",
    ]
    # Must be in a block/layer AND have an expert pattern
    has_block = any(p in name for p in ["blk.", "layers."])
    has_expert = any(p in name for p in expert_patterns)
    # Exclude shared experts
    is_shared = "shared" in name.lower() or "shared_expert" in name
    return has_block and has_expert and not is_shared


def get_tensor_layer(name):
    """Extract layer index from a tensor name like 'blk.37.ffn_gate_exps.weight'."""
    import re
    match = re.search(r'(?:blk\.|layers\.)(\d+)', name)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description="Splice routed expert tensors between GGUFs")
    parser.add_argument("--base", required=True, help="Base GGUF file (lower quant, kept for most tensors)")
    parser.add_argument("--donor", required=True, help="Donor GGUF file (higher quant, experts copied from here)")
    parser.add_argument("--layers", required=True, help="Layer range(s) to splice, e.g. '37-42' or '0-2,40-42'")
    parser.add_argument("--out", required=True, help="Output GGUF file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without writing")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()

    target_layers = parse_layer_ranges(args.layers)
    print(f"Target layers: {target_layers}")
    print(f"Base: {args.base} ({Path(args.base).stat().st_size / 1e9:.1f} GB)")
    print(f"Donor: {args.donor} ({Path(args.donor).stat().st_size / 1e9:.1f} GB)")

    if not args.force and not args.dry_run and Path(args.out).exists():
        print(f"Error: {args.out} exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    print("\nTensors that would be spliced from donor:")
    for layer in target_layers:
        for suffix in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]:
            name = f"blk.{layer}.{suffix}"
            action = "SPLICE from donor" if not args.dry_run else "would splice"
            print(f"  {name} → {action}")

    if args.dry_run:
        print("\n(dry run, no files written)")
        return

    # TODO: implement GGUF header/tensor-table parsing and byte-level copy.
    print(f"\nNote: Full GGUF binary splicing not yet implemented.")
    print(f"See gguf-tools/ for reference implementations.")


if __name__ == "__main__":
    main()
