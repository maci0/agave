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

```bash
agave model.gguf --eval continuations.jsonl
```

### 3. Compare Quantizations

```bash
agave model-q4.gguf --eval continuations.jsonl    # NLL: 1.234
agave model-q8.gguf --eval continuations.jsonl    # NLL: 0.987
agave model-f16.gguf --eval continuations.jsonl   # NLL: 0.954 (baseline)
```

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

- `src/eval.zig` — `scoreCase()`, `EvalResult`, token-by-token NLL computation
- `tools/quality-testing/collect_continuations.py` — API continuation collector

Based on the quality testing approach from [antirez/ds4](https://github.com/antirez/ds4).
