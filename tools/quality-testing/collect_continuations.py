#!/usr/bin/env python3
"""Collect official API continuations for NLL quality testing.

Sends prompts to a hosted model API and records the greedy continuations
with token-level logprobs. The output JSONL is consumed by `src/eval.zig`
(library API; there is no `agave eval` subcommand yet).

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

The output doubles as the resume ledger: each result is appended to it
(atomically) as soon as it arrives, and a rerun skips prompts that already
have a successful continuation for the same model, so an interrupted run does
not pay twice for the same prompt. Error records are retried on the next run.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Seconds; hosted APIs can be slow for long greedy continuations.
HTTP_TIMEOUT_SEC = 120

# The API key travels in an Authorization header, so the endpoint must be
# https. A file: or plain-http endpoint would leak it.
ENDPOINT_SCHEME = "https"


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
    if urllib.parse.urlparse(endpoint).scheme != ENDPOINT_SCHEME:
        raise ValueError(f"endpoint must be {ENDPOINT_SCHEME}://, got {endpoint!r}")
    # S310 cannot see the scheme check above, so it fires on the Request
    # construction and the urlopen alike.
    req = urllib.request.Request(  # noqa: S310 - endpoint scheme checked above
        endpoint,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 - checked above
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        with e:
            detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e

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


def load_results(path):
    """Read an existing JSONL output into {prompt: record}, newest line wins.

    A rerun of an interrupted collection reads this back so prompts that
    already have a continuation are not billed a second time. Unreadable
    lines are dropped rather than failing the run: the file is a resume
    ledger, not the deliverable.
    """
    results = {}
    if not path.exists():
        return results
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "prompt" in record:
            results[record["prompt"]] = record
    return results


def is_done(results, prompt, model):
    """True when `prompt` already has a successful continuation from `model`.

    Error records are not done: the next run must retry them. A record from
    another model is not done either, since logprobs are model-specific.
    """
    record = results.get(prompt)
    return record is not None and "continuation" in record and record.get("model") == model


def write_results(path, results, order):
    """Atomically rewrite the JSONL ledger. A crash leaves the previous one."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for prompt in order:
            record = results.get(prompt)
            if record is not None:
                f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official continuations for NLL testing")
    parser.add_argument("--endpoint", required=True, help="Chat completions API endpoint URL")
    parser.add_argument("--model", required=True, help="Model name for the API")
    parser.add_argument("--prompts", required=True, help="Prompt file (one per line)")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--api-key", help="API key (or set API_KEY env var)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per continuation")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: set --api-key or API_KEY env var", file=sys.stderr)
        sys.exit(2)

    prompts = Path(args.prompts).read_text().strip().split("\n")
    out_path = Path(args.out)
    # Prompt order is the ledger's order, so a resumed file reads like one
    # uninterrupted run.
    order: list[str] = []
    for prompt in prompts:
        if prompt not in order:
            order.append(prompt)
    results = load_results(out_path)
    todo = [p for p in order if not is_done(results, p, args.model)]
    print(
        f"Collecting {len(todo)} continuations from {args.model} ({len(order) - len(todo)} already collected, skipped)"
    )

    for i, prompt in enumerate(todo):
        print(f"  [{i + 1}/{len(todo)}] {prompt[:60]}...")
        try:
            results[prompt] = collect_one(args.endpoint, args.model, prompt, api_key, args.max_tokens)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            results[prompt] = {"prompt": prompt, "model": args.model, "error": str(e)}
        # Persist after every call: an interrupt must not lose the API calls
        # already paid for, and the rerun must not repeat them.
        write_results(out_path, results, order)
        if i < len(todo) - 1:
            time.sleep(args.delay)

    write_results(out_path, results, order)

    n_ok = sum(1 for p in order if is_done(results, p, args.model))
    print(f"Wrote {out_path} ({n_ok}/{len(order)} successful)")
    # A run where every request failed still writes the error records, so exit
    # nonzero instead of reporting success to whatever drives the loop.
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
