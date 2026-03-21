#!/usr/bin/env python3
"""
Generate golden reference outputs for model correctness tests.

Uses:
- llama.cpp for GGUF models (Gemma3, Qwen3.5, DeepSeek-R1-Qwen3, Nemotron Nano, GPT-OSS, Nemotron-H)
- HuggingFace transformers for SafeTensors models (Gemma3 MLX, GLM-4)

Output: JSON files in tests/golden/references/ with deterministic token sequences.
"""

import subprocess
import json
import os
from pathlib import Path
import sys

# Test prompts (short, deterministic, cover model capabilities)
PROMPTS = {
    "gemma3": "What is the capital of France?",
    "qwen35": "Explain photosynthesis in simple terms.",
    "deepseek_r1_qwen3": "Write a Python function to calculate factorial.",
    "nemotron_nano": "List three benefits of exercise.",
    "glm4": "What is quantum computing?",
    "gpt_oss": "Once upon a time in a distant galaxy,",
    "nemotron_h": "Describe the water cycle.",
}

# Model paths (update per local setup)
MODEL_PATHS = {
    "gemma3": "models/google/gemma-3-1b-it-q4_0.gguf",
    "qwen35": "models/Qwen/Qwen3.5-0.8B-Instruct-Q8_0.gguf",
    "deepseek_r1_qwen3": "models/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
    "nemotron_nano": "models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
    "glm4": "models/glm-4-9b.gguf",
    "gpt_oss": "models/gpt-oss.gguf",
    "nemotron_h": "models/nemotron-h.gguf",
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

    # Extract token IDs (llama.cpp has --log-tokens flag, use if available)
    # For now, store text output
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

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "model": model_name,
        "backend": "huggingface",
        "prompt": prompt,
        "output": output_text,
        "seed": None,  # HF is deterministic with do_sample=False
        "temp": 0.0,
    }


def main():
    output_dir = Path("tests/golden/references")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GGUF models: use llama.cpp
    for model_name in ["gemma3", "qwen35", "deepseek_r1_qwen3", "nemotron_nano", "gpt_oss", "nemotron_h"]:
        if model_name not in MODEL_PATHS:
            print(f"Skipping {model_name} (no model path)")
            continue

        model_path = MODEL_PATHS[model_name]
        if not Path(model_path).exists():
            print(f"Skipping {model_name} (model file not found: {model_path})")
            continue

        print(f"Generating llama.cpp reference for {model_name}...")
        try:
            ref = generate_llamacpp_reference(model_name, model_path, PROMPTS[model_name])

            output_file = output_dir / f"{model_name}_llamacpp.json"
            with open(output_file, "w") as f:
                json.dump(ref, f, indent=2)
            print(f"  Wrote {output_file}")
        except Exception as e:
            print(f"  Failed: {e}")

    # GLM-4: use HuggingFace (if available)
    # Requires: pip install transformers torch
    try:
        print("Generating HuggingFace reference for glm4...")
        ref = generate_huggingface_reference("glm4", "THUDM/glm-4-9b", PROMPTS["glm4"])
        output_file = output_dir / "glm4_huggingface.json"
        with open(output_file, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"  Wrote {output_file}")
    except ImportError:
        print("  Skipping GLM-4 HuggingFace reference (transformers not installed)")
    except Exception as e:
        print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
