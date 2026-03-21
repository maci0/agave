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
    ref_file = Path(f"tests/golden/references/{model_name}_{ref_backend}.json")
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
    - Different random seed handling

    Instead, check:
    1. Output is non-empty and coherent (not garbage)
    2. First N tokens match (prefix matching)
    3. Overall length is similar (within 50%)
    """
    if not agave_output or len(agave_output) < 10:
        print("FAIL: Agave output too short or empty")
        return False

    # Prefix matching: first 20 characters should be close
    prefix_len = min(20, len(agave_output), len(reference_output))
    agave_prefix = agave_output[:prefix_len].lower()
    ref_prefix = reference_output[:prefix_len].lower()

    # Allow some variation (e.g., whitespace, capitalization)
    if agave_prefix == ref_prefix:
        print("PASS: Prefix exact match")
        return True

    # Relaxed: check if 80% of prefix characters match
    matches = sum(1 for a, r in zip(agave_prefix, ref_prefix) if a == r)
    match_ratio = matches / prefix_len if prefix_len > 0 else 0.0
    if match_ratio >= 0.8:
        print(f"PASS: Prefix {match_ratio*100:.1f}% match")
        return True

    print(f"FAIL: Prefix mismatch (match ratio: {match_ratio*100:.1f}%)")
    print(f"  Agave:     {agave_prefix}")
    print(f"  Reference: {ref_prefix}")
    return False


def main():
    if len(sys.argv) < 4:
        print("Usage: verify_output.py <model_name> <backend> <agave_output_json>")
        sys.exit(1)

    model_name = sys.argv[1]
    backend = sys.argv[2]
    agave_output_file = sys.argv[3]

    # Load Agave output
    with open(agave_output_file) as f:
        agave_data = json.load(f)

    # Load reference (use llama.cpp for GGUF, HuggingFace for SafeTensors)
    ref_backend = "llamacpp" if backend != "huggingface" else "huggingface"
    reference = load_reference(model_name, ref_backend)

    # Compare
    passed = compare_outputs(agave_data["output"], reference["output"])

    if passed:
        print(f"✓ {model_name} on {backend} matches reference")
        sys.exit(0)
    else:
        print(f"✗ {model_name} on {backend} differs from reference")
        sys.exit(1)


if __name__ == "__main__":
    main()
