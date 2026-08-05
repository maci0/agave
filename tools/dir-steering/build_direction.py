#!/usr/bin/env python3
"""Build a directional steering vector from paired prompt sets.

Captures activations from two sets of prompts (target vs contrast),
averages (target - contrast) per layer, normalizes to unit vectors,
and writes a flat f32 file that agave --dir-steering-file can load.

Usage:
    python3 build_direction.py \
        --agave ./zig-out/bin/agave \
        --model model.gguf \
        --good-file prompts_succinct.txt \
        --bad-file prompts_verbose.txt \
        --out direction.f32 \
        --component ffn_out \
        --n-layers 64 \
        --n-embd 2048

The component flag selects which activation to capture:
    ffn_out  - FFN output (recommended for style/behavior)
    attn_out - attention output (more fragile)

Each prompt file has one prompt per line. Prompts should be paired:
line N in good-file and line N in bad-file ask for the same information
but in the target vs contrast style.

Requires: numpy
"""

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def get_activations(agave_bin, model_path, prompt, n_layers, n_embd):
    """Run agave with --profile to extract per-layer hidden states.

    NOTE: This is a placeholder. A full implementation would need agave
    to export per-layer activations (e.g. via --export-activations).
    For now, this generates random directions as a scaffold.
    """
    # TODO: When agave supports --export-activations, capture real activations.
    # For now, return placeholder zeros to show the pipeline structure.
    return np.zeros((n_layers, n_embd), dtype=np.float32)


def build_direction(good_acts, bad_acts):
    """Compute direction = mean(good) - mean(bad), normalize per layer."""
    diff = good_acts.mean(axis=0) - bad_acts.mean(axis=0)  # [n_layers, n_embd]
    # Normalize each layer's direction to unit length
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # avoid division by zero
    return (diff / norms).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build directional steering vectors")
    parser.add_argument("--agave", required=True, help="Path to agave binary")
    parser.add_argument("--model", required=True, help="Path to model GGUF")
    parser.add_argument("--good-file", required=True, help="Target prompts (one per line)")
    parser.add_argument("--bad-file", required=True, help="Contrast prompts (one per line)")
    parser.add_argument("--out", required=True, help="Output f32 file")
    parser.add_argument("--component", default="ffn_out", choices=["ffn_out", "attn_out"])
    parser.add_argument("--n-layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--n-embd", type=int, required=True, help="Embedding dimension")
    args = parser.parse_args()

    good_prompts = Path(args.good_file).read_text().strip().split("\n")
    bad_prompts = Path(args.bad_file).read_text().strip().split("\n")

    if len(good_prompts) != len(bad_prompts):
        print(f"Error: good ({len(good_prompts)}) and bad ({len(bad_prompts)}) prompt counts differ")
        sys.exit(1)

    print(f"Building {args.component} direction from {len(good_prompts)} prompt pairs")
    print(f"Model: {args.model} ({args.n_layers} layers × {args.n_embd} embd)")

    # Collect activations
    good_acts = []
    for i, prompt in enumerate(good_prompts):
        print(f"  good [{i+1}/{len(good_prompts)}]: {prompt[:60]}...")
        acts = get_activations(args.agave, args.model, prompt, args.n_layers, args.n_embd)
        good_acts.append(acts)

    bad_acts = []
    for i, prompt in enumerate(bad_prompts):
        print(f"  bad [{i+1}/{len(bad_prompts)}]: {prompt[:60]}...")
        acts = get_activations(args.agave, args.model, prompt, args.n_layers, args.n_embd)
        bad_acts.append(acts)

    good_arr = np.stack(good_acts)  # [n_prompts, n_layers, n_embd]
    bad_arr = np.stack(bad_acts)

    direction = build_direction(good_arr, bad_arr)  # [n_layers, n_embd]
    print(f"Direction shape: {direction.shape}")
    print(f"Per-layer norms: min={np.linalg.norm(direction, axis=1).min():.4f}, "
          f"max={np.linalg.norm(direction, axis=1).max():.4f}")

    # Write flat f32 file
    out_path = Path(args.out)
    out_path.write_bytes(direction.tobytes())
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Also write metadata JSON
    meta_path = out_path.with_suffix(".json")
    meta = {
        "component": args.component,
        "n_layers": args.n_layers,
        "n_embd": args.n_embd,
        "n_good_prompts": len(good_prompts),
        "n_bad_prompts": len(bad_prompts),
        "good_file": args.good_file,
        "bad_file": args.bad_file,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
