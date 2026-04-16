#!/usr/bin/env python3
"""
Generate golden reference outputs for model correctness tests.

Uses:
- llama.cpp for GGUF models (Gemma3, Gemma4, Qwen3.5, DeepSeek-R1-Qwen3, Nemotron Nano, GLM-4, GPT-OSS)
- Nemotron-H uses SafeTensors and is not supported by llama.cpp

Output: JSON files in tests/golden/references/ with deterministic token sequences.
"""

import subprocess
import json
import os
from pathlib import Path
import sys

# Test prompts — must match tests/models/test_*.zig prompts exactly
PROMPTS = {
    "gemma3": "What is the capital of France?",
    "gemma4": "Explain the theory of relativity.",
    "qwen35": "Explain photosynthesis in simple terms.",
    "deepseek_r1_qwen3": "Write a Python function to calculate factorial.",
    "nemotron_nano": "List three benefits of exercise.",
    "nemotron_h": "Describe the water cycle.",
    "glm4": "What is quantum computing?",
    "gpt_oss": "Once upon a time in a distant galaxy,",
}

# Model paths — must match tests/models/test_*.zig paths exactly
MODEL_PATHS = {
    "gemma3": "models/lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf",
    "gemma4": "models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "qwen35": "models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf",
    "deepseek_r1_qwen3": "models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
    "nemotron_nano": "models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
    "glm4": "models/lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q8_0.gguf",
    "gpt_oss": "models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf",
}


def generate_llamacpp_reference(model_name: str, model_path: str, prompt: str) -> dict:
    """Generate reference using llama.cpp main binary."""
    # Assumes llama.cpp built at ../llama.cpp/build/bin/llama-cli
    llamacpp_bin = Path("../llama.cpp/build/bin/llama-cli")
    if not llamacpp_bin.exists():
        raise FileNotFoundError(f"llama.cpp not found at {llamacpp_bin}")

    cmd = [
        str(llamacpp_bin),
        "-m", model_path,
        "-p", prompt,
        "-n", "32",  # Generate 32 tokens
        "-s", "42",  # Deterministic seed
        "--temp", "0.0",  # Greedy sampling
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


def generate_huggingface_reference(model_name: str, model_id: str, prompt: str) -> dict:
    """Generate reference using HuggingFace transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode generated tokens (exclude prompt) to match llama.cpp
    # --no-display-prompt behavior
    prompt_len = inputs.input_ids.shape[1]
    output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return {
        "model": model_name,
        "backend": "huggingface",
        "prompt": prompt,
        "output": output_text,
        "seed": None,  # HF is deterministic with do_sample=False
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

    # SafeTensors models: use HuggingFace transformers
    # nemotron_h is not supported by llama.cpp (SafeTensors-only architecture)
    HF_MODELS = {
        "nemotron_h": "nvidia/Nemotron-3-Nano-30B-A3B",
    }
    for model_name, model_id in HF_MODELS.items():
        print(f"Generating HuggingFace reference for {model_name}...")
        try:
            ref = generate_huggingface_reference(model_name, model_id, PROMPTS[model_name])
            output_file = output_dir / f"{model_name}_huggingface.json"
            with open(output_file, "w") as f:
                json.dump(ref, f, indent=2)
            print(f"  Wrote {output_file}")
            generated += 1
        except (ImportError, RuntimeError, OSError) as e:
            print(f"  Failed: {e}")
            failed += 1

    # Summary
    total = len(MODEL_PATHS) + len(HF_MODELS)
    print(f"\nSummary: {generated}/{total} generated, {skipped} skipped, {failed} failed")
    if generated == 0:
        print("ERROR: No references were generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
