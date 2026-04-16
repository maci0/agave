#!/usr/bin/env python3
"""
Verify Agave output against golden references.

Accepts:
- Agave output JSON (model, backend, prompt, output, tokens)
- Golden reference JSON (from generate_references.py)

Returns:
- Exit 0 if output matches (within tolerance)
- Exit 1 if output differs
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_reference(model_name: str, ref_backend: str) -> Dict[str, Any]:
    """Load golden reference for model."""
    ref_dir = Path(__file__).resolve().parent / "references"
    ref_file = ref_dir / f"{model_name}_{ref_backend}.json"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference not found: {ref_file}")
    with open(ref_file) as f:
        return json.load(f)


def compare_outputs(agave_output: str, reference_output: str) -> bool:
    """
    Compare Agave output to reference.

    For text generation, exact match is unrealistic due to:
    - Different quantization (Q4_0 vs Q8_0 vs bf16)
    - Different rounding in softmax/sampling

    Instead, check:
    1. Output is non-empty and coherent (not garbage)
    2. First N tokens match (prefix matching)
    3. Overall length is similar (70-150% of reference)
    """
    if not agave_output or len(agave_output) < 40:
        print("FAIL: Agave output too short or empty", file=sys.stderr)
        return False

    if not reference_output or len(reference_output) < 10:
        print("FAIL: Reference output is empty or too short — regenerate golden references", file=sys.stderr)
        return False

    # Length ratio check: output should be 80-120% of reference length.
    # Deterministic greedy sampling (temp=0, fixed seed) limits legitimate
    # divergence to quantization and softmax rounding differences.
    len_ratio = len(agave_output) / len(reference_output)
    if len_ratio < 0.8 or len_ratio > 1.2:
        print(f"FAIL: Output length ratio {len_ratio:.2f} outside [0.8, 1.2]", file=sys.stderr)
        print(f"  Agave: {len(agave_output)} chars, Reference: {len(reference_output)} chars", file=sys.stderr)
        return False

    # Prefix matching: first 80 characters should be close (case-sensitive to
    # catch capitalization regressions; fall back to case-normalized fuzzy match).
    prefix_len = min(80, len(agave_output), len(reference_output))
    agave_prefix = agave_output[:prefix_len]
    ref_prefix = reference_output[:prefix_len]

    if agave_prefix == ref_prefix:
        print("PASS: Prefix exact match")
    else:
        # Check if 95% of prefix characters match (case-insensitive for minor variation)
        matches = sum(1 for a, r in zip(agave_prefix.lower(), ref_prefix.lower()) if a == r)
        match_ratio = matches / prefix_len if prefix_len > 0 else 0.0
        if match_ratio < 0.95:
            print(f"FAIL: Prefix mismatch (match ratio: {match_ratio*100:.1f}%)", file=sys.stderr)
            print(f"  Agave:     {agave_prefix}", file=sys.stderr)
            print(f"  Reference: {ref_prefix}", file=sys.stderr)
            return False
        print(f"PASS: Prefix {match_ratio*100:.1f}% match (case-normalized)")

    # Mid-point check: sample from the middle of the output to catch
    # "good prefix + garbage tail" failures.
    if len(agave_output) > 80 and len(reference_output) > 80:
        mid = min(len(agave_output), len(reference_output)) // 2
        window = min(40, mid)
        agave_mid = agave_output[mid - window:mid].lower()
        ref_mid = reference_output[mid - window:mid].lower()
        mid_matches = sum(1 for a, r in zip(agave_mid, ref_mid) if a == r)
        mid_ratio = mid_matches / len(agave_mid) if len(agave_mid) > 0 else 0.0
        if mid_ratio < 0.85:
            print(f"FAIL: Mid-point mismatch (match ratio: {mid_ratio*100:.1f}%)", file=sys.stderr)
            print(f"  Agave mid:     {agave_mid}", file=sys.stderr)
            print(f"  Reference mid: {ref_mid}", file=sys.stderr)
            return False

    # Suffix check: verify the tail hasn't degenerated into garbage.
    # Catches models that produce correct prefix but loop or corrupt at the end.
    suffix_len = min(40, len(agave_output), len(reference_output))
    if suffix_len >= 20:
        agave_suffix = agave_output[-suffix_len:].lower()
        ref_suffix = reference_output[-suffix_len:].lower()
        suffix_matches = sum(1 for a, r in zip(agave_suffix, ref_suffix) if a == r)
        suffix_ratio = suffix_matches / suffix_len
        if suffix_ratio < 0.75:
            print(f"FAIL: Suffix mismatch (match ratio: {suffix_ratio*100:.1f}%)", file=sys.stderr)
            print(f"  Agave suffix:     {agave_suffix}", file=sys.stderr)
            print(f"  Reference suffix: {ref_suffix}", file=sys.stderr)
            return False

    return True


def main():
    if len(sys.argv) < 4:
        print("Usage: verify_output.py <model_name> <backend> <agave_output_json>")
        sys.exit(1)

    model_name = sys.argv[1]
    backend = sys.argv[2]
    agave_output_file = sys.argv[3]

    # Load Agave output
    try:
        with open(agave_output_file) as f:
            agave_data = json.load(f)
    except FileNotFoundError:
        print(f"FAIL: Agave output file not found: {agave_output_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"FAIL: Agave output is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Load reference: try llamacpp first (GGUF models), fall back to huggingface
    # (SafeTensors models). The compute backend (cpu/metal/cuda) is independent
    # of the reference format.
    try:
        reference = load_reference(model_name, "llamacpp")
    except FileNotFoundError:
        try:
            reference = load_reference(model_name, "huggingface")
        except FileNotFoundError:
            print(f"Reference not found for {model_name}", file=sys.stderr)
            sys.exit(1)

    # Compare
    try:
        agave_output = agave_data["output"]
    except KeyError:
        print(f"FAIL: Agave output JSON missing 'output' field. Keys: {list(agave_data.keys())}", file=sys.stderr)
        sys.exit(1)
    try:
        reference_output = reference["output"]
    except KeyError:
        print(f"FAIL: Reference JSON missing 'output' field. Keys: {list(reference.keys())}", file=sys.stderr)
        sys.exit(1)
    passed = compare_outputs(agave_output, reference_output)

    if passed:
        print(f"✓ {model_name} on {backend} matches reference")
        sys.exit(0)
    else:
        print(f"✗ {model_name} on {backend} differs from reference", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
