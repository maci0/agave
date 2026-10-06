#!/usr/bin/env python3
"""
Generate golden reference outputs for model correctness tests.

Uses:
- llama.cpp for GGUF models (Gemma4, Qwen3.5)

Requires:
- llama.cpp built at ../llama.cpp/build/bin/llama-cli
  (skipped gracefully if not installed)

Output: JSON files in tests/golden/references/ with deterministic token sequences.
"""

import subprocess
import json
from pathlib import Path
import sys

# Test prompts, must match tests/models/test_*.zig prompts exactly
PROMPTS = {
    "gemma4": "Explain the theory of relativity.",
    "qwen35": "Explain photosynthesis in simple terms.",
    "deepseek_r1_qwen3": "Write a Python function to calculate factorial.",
}

# Model paths, must match tests/models/test_*.zig paths exactly
MODEL_PATHS = {
    "gemma4": "models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "qwen35": "models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf",
    "deepseek_r1_qwen3": "models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
}


def generate_llamacpp_reference(model_name: str, model_path: str, prompt: str) -> dict:
    """Generate reference using llama.cpp main binary."""
    # Assumes llama.cpp built at ../llama.cpp/build/bin/llama-cli
    llamacpp_bin = Path("../llama.cpp/build/bin/llama-cli")
    if not llamacpp_bin.exists():
        raise FileNotFoundError(f"llama.cpp not found at {llamacpp_bin}")

    cmd = [
        str(llamacpp_bin),
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        "32",  # Generate 32 tokens
        "-s",
        "42",  # Deterministic seed
        "--temp",
        "0.0",  # Greedy sampling
        "--no-display-prompt",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output_text = result.stdout.strip()

    if not output_text:
        raise RuntimeError(f"llama.cpp produced empty output for {model_name}")

    return {
        "model": model_name,
        "backend": "llama.cpp",
        "prompt": prompt,
        "output": output_text,
        "seed": 42,
        "temp": 0.0,
    }


def main():
    output_dir = Path(__file__).resolve().parent / "references"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failed = 0

    # GGUF models: use llama.cpp
    for model_name, model_path in MODEL_PATHS.items():
        if not Path(model_path).exists():
            print(f"Skipping {model_name} (model file not found: {model_path})")
            skipped += 1
            continue

        print(f"Generating llama.cpp reference for {model_name}...")
        try:
            ref = generate_llamacpp_reference(model_name, model_path, PROMPTS[model_name])

            output_file = output_dir / f"{model_name}_llamacpp.json"
            with open(output_file, "w") as f:
                json.dump(ref, f, indent=2)
            print(f"  Wrote {output_file}")
            generated += 1
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
            print(f"  Failed: {e}")
            failed += 1

    # Summary
    total = len(MODEL_PATHS)
    print(f"\nSummary: {generated}/{total} generated, {skipped} skipped, {failed} failed")
    if generated == 0:
        print("ERROR: No references were generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
