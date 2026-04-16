#!/usr/bin/env python3
"""Compare reference vs agave vision encoder intermediate outputs.

Loads pairs of .bin files (raw f32 little-endian) and reports:
  - Max absolute error
  - Mean absolute error
  - Pearson correlation
  - First divergence point

Usage:
  python vision_compare.py [ref_dir] [agave_dir]

Default directories:
  ref_dir:   vision_ref_dumps/
  agave_dir: ../../  (project root, where agave dumps files)
"""

import sys
import os
import numpy as np

STAGES = [
    ("01_after_patch_embed", 196 * 1152),
    ("02_after_std_pos", 196 * 1152),
    ("03_after_block_00", 196 * 1152),
    ("04_after_block_26", 196 * 1152),
    ("05_after_projection", 196 * 2816),
]


def load_f32(path: str, expected_len: int) -> np.ndarray | None:
    """Load a raw f32 binary file."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.float32)
    if len(data) != expected_len:
        print(f"  WARNING: {path} has {len(data)} floats, expected {expected_len}")
    return data


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = np.sum(a_centered * b_centered)
    denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    if denom < 1e-30:
        return 0.0
    return float(num / denom)


def compare_stage(name: str, ref: np.ndarray, agave: np.ndarray) -> dict:
    """Compare two arrays and return error metrics."""
    diff = np.abs(ref - agave)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    corr = correlation(ref, agave)

    # Find first significant divergence (>1% relative error on non-tiny values)
    rel_mask = np.abs(ref) > 0.01
    if np.any(rel_mask):
        rel_err = diff[rel_mask] / (np.abs(ref[rel_mask]) + 1e-10)
        first_bad = np.argmax(rel_err > 0.01)
        first_bad_idx = np.where(rel_mask)[0][first_bad] if rel_err[first_bad] > 0.01 else -1
    else:
        first_bad_idx = -1

    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "corr": corr,
        "first_bad_idx": first_bad_idx,
    }


def main():
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else "vision_ref_dumps"
    agave_dir = sys.argv[2] if len(sys.argv) > 2 else "../.."

    print(f"Reference dir: {ref_dir}")
    print(f"Agave dir:     {agave_dir}")
    print()

    first_diverge = None

    for stage_name, expected_len in STAGES:
        ref_path = os.path.join(ref_dir, f"{stage_name}.bin")
        agave_path = os.path.join(agave_dir, f"agave_{stage_name}.bin")

        ref = load_f32(ref_path, expected_len)
        agave = load_f32(agave_path, expected_len)

        if ref is None:
            print(f"[{stage_name}] SKIP: reference file not found ({ref_path})")
            continue
        if agave is None:
            print(f"[{stage_name}] SKIP: agave file not found ({agave_path})")
            continue

        n = min(len(ref), len(agave))
        ref = ref[:n]
        agave = agave[:n]

        m = compare_stage(stage_name, ref, agave)
        status = "OK" if m["corr"] > 0.999 and m["max_err"] < 1.0 else "MISMATCH"

        print(f"[{stage_name}] {status}")
        print(f"  Max abs error:  {m['max_err']:.6f}")
        print(f"  Mean abs error: {m['mean_err']:.6f}")
        print(f"  Correlation:    {m['corr']:.8f}")

        if m["first_bad_idx"] >= 0:
            idx = m["first_bad_idx"]
            print(f"  First diverge:  index {idx} (ref={ref[idx]:.6f}, agave={agave[idx]:.6f})")

        # Show first 10 values side by side
        print(f"  First 10 ref:   {ref[:10]}")
        print(f"  First 10 agave: {agave[:10]}")
        print()

        if first_diverge is None and status == "MISMATCH":
            first_diverge = stage_name

    if first_diverge:
        print(f"==> First divergence at stage: {first_diverge}")
    else:
        print("==> All stages match within tolerance!")


if __name__ == "__main__":
    main()
