#!/usr/bin/env python3
"""Collect official API continuations for NLL quality testing.

Sends prompts to a hosted model API and records the greedy continuations
with token-level logprobs. The output JSONL can be scored locally with
`agave eval --continuations FILE model.gguf`.

Usage:
    export API_KEY=...
    python3 collect_continuations.py \
        --endpoint https://api.deepseek.com/chat/completions \
        --model deepseek-v4-flash \
        --prompts prompts.txt \
        --out continuations.jsonl \
        --max-tokens 128

Prompt file: one prompt per line.
Output: JSONL with {"prompt": "...", "continuation": "...", "tokens": [...]}
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: pip install requests", file=sys.stderr)
    sys.exit(1)


def collect_one(endpoint, model, prompt, api_key, max_tokens):
    """Send one prompt to the API and return the continuation."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
    }
    resp = requests.post(endpoint, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    continuation = choice["message"]["content"]
    tokens = []
    if "logprobs" in choice and choice["logprobs"] and "content" in choice["logprobs"]:
        for entry in choice["logprobs"]["content"]:
            if "token" in entry:
                tokens.append(entry["token"])

    return {
        "prompt": prompt,
        "continuation": continuation,
        "tokens_text": tokens,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect official continuations for NLL testing")
    parser.add_argument("--endpoint", required=True, help="Chat completions API endpoint URL")
    parser.add_argument("--model", required=True, help="Model name for the API")
    parser.add_argument("--prompts", required=True, help="Prompt file (one per line)")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--api-key", help="API key (or set API_KEY env var)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per continuation")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: set --api-key or API_KEY env var", file=sys.stderr)
        sys.exit(1)

    prompts = Path(args.prompts).read_text().strip().split("\n")
    print(f"Collecting {len(prompts)} continuations from {args.model}")

    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
        try:
            result = collect_one(args.endpoint, args.model, prompt, api_key, args.max_tokens)
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"prompt": prompt, "error": str(e)})
        if i < len(prompts) - 1:
            time.sleep(args.delay)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_ok = sum(1 for r in results if "continuation" in r)
    print(f"Wrote {out_path} ({n_ok}/{len(results)} successful)")


if __name__ == "__main__":
    main()
