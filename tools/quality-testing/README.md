# Quality Testing via NLL Scoring

Compare local GGUF quality against official model outputs using token-by-token
negative log-likelihood. This measures how much probability a local model
assigns to each ground-truth token from the reference model.

## Metric

**Mean NLL** (negative log-likelihood): average `-log(P(correct_token))` across
all continuation tokens. Lower = better.

**Argmax accuracy**: fraction of positions where the local model's greedy
prediction matches the reference token.

## Workflow

### 1. Collect Official Continuations

```bash
export API_KEY=your_key_here
python3 tools/quality-testing/collect_continuations.py \
    --endpoint https://api.deepseek.com/chat/completions \
    --model deepseek-v4-flash \
    --prompts prompts.txt \
    --out continuations.jsonl \
    --max-tokens 128
```

### 2. Score Local Model

There is **no `--eval` CLI flag** yet. Call the library from Zig (tests or a
thin harness) after tokenizing each JSONL line:

```zig
const result = eval.scoreCase(model, prompt_ids, continuation_ids) orelse return error.EvalFailed;
// Compare result.mean_nll / argmax accuracy across quants
```

See [`src/eval.zig`](../../src/eval.zig) and [Chapter 24](../../docs/tutorial/24-advanced-features.md).

### 3. Compare Quantizations

Score the same `continuations.jsonl` against Q4 / Q8 / F16 builds with
`scoreCase` and compare mean NLL (lower is better) and argmax accuracy.

## Prompt File Format

One prompt per line:
```
Explain why databases use indexes.
What is the capital of France?
Write a Python function to sort a list.
```

## Continuation File Format (JSONL)

```json
{"prompt": "What is 2+2?", "continuation": "The answer is 4."}
{"prompt": "Capital of France?", "continuation": "Paris is the capital of France."}
```

## Implementation

- `src/eval.zig`: `scoreCase()`, `EvalResult`, token-by-token NLL computation
- `tools/quality-testing/collect_continuations.py`: API continuation collector

Based on the quality testing approach from [antirez/ds4](https://github.com/antirez/ds4).
