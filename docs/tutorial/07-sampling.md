# Chapter 7: Sampling

After the forward pass produces **logits** (raw unnormalized scores, one per vocabulary token), the model must **select** the next token. The simplest method is **greedy decoding** (pick the highest score), but this produces repetitive, **deterministic** (always the same output for the same input) output. Sampling parameters add controlled randomness for more natural text.

## Temperature

Controls randomness by scaling logits before sampling:

```
adjusted_logits[i] = logits[i] / temperature
probabilities = softmax(adjusted_logits)
next_token = sample(probabilities)
```

| Value | Effect | Use case |
|-------|--------|----------|
| `0` | **Greedy** — always pick highest (argmax) | Factual Q&A, code, math |
| `0.1-0.5` | Low randomness | Reliable but slightly varied |
| `0.7-0.9` | Balanced | General conversation, writing |
| `1.0` | Raw model probabilities | Default behavior |
| `1.5-2.0` | High randomness | Creative writing, brainstorming |

Dividing by a small temperature makes the softmax "peakier" (top token dominates). Dividing by a large temperature makes it "flatter" (more candidates get a chance). At temperature=0, Agave uses argmax — deterministic, same input always produces same output.

## Top-K

Restricts sampling to only the K highest-scoring tokens:

```
--top-k 40    Only consider the top 40 tokens
--top-k 0     Disabled (consider all tokens) — default
```

Sort tokens by score, keep the top K, **renormalize** probabilities (rescale so they sum to 1.0 again), sample. Prevents picking extremely unlikely tokens at high temperatures.

## Top-P (Nucleus Sampling)

Restricts sampling to the smallest set of tokens whose **cumulative probability** (running sum of probabilities in sorted order) exceeds P:

```
--top-p 0.9    Keep tokens until cumulative probability reaches 90%
--top-p 1.0    Disabled — default
```

More adaptive than top-k: when the model is confident (top token = 95%), top-p=0.9 keeps 1-2 candidates. When uncertain (many similar scores), it keeps dozens.

**Top-K vs Top-P**: Top-K always keeps exactly K tokens. Top-P adapts based on confidence. They can be combined.

## Repeat Penalty

Discourages repeating previously generated tokens:

```
if token was previously generated:
    logits[token] /= repeat_penalty    (if logit > 0)
    logits[token] *= repeat_penalty    (if logit < 0)
```

Prevents the common "the the the the..." failure mode. Default 1.0 (disabled).

## Min-P

Adaptive threshold that keeps tokens whose probability is at least min_p × the top token's probability:

```
max_prob = max(softmax(logits))
threshold = min_p * max_prob
keep tokens where prob >= threshold
```

```
--min-p 0.05    Keep tokens with prob >= 5% of best token's prob
--min-p 0       Disabled — default
```

More intuitive than top-p: directly controls the "quality floor" relative to the best candidate. When the model is very confident, fewer tokens pass the filter; when uncertain, more pass — similar to top-p but without needing to think about cumulative probabilities.

## Frequency and Presence Penalties

OpenAI-style penalties applied to logits before sampling:

```
logits[token] -= frequency_penalty * count(token in output)
logits[token] -= presence_penalty * (1 if token appeared, 0 otherwise)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `frequency_penalty` | `[-2, 2]` | Per-occurrence penalty — penalizes repeated tokens proportionally |
| `presence_penalty` | `[-2, 2]` | One-time penalty — discourages any reuse of generated tokens |

Positive values reduce repetition. Negative values encourage it (useful for rhyming, alliteration). Available in HTTP API; CLI uses `--repeat-penalty` (multiplicative style) instead.

## Grammar-Constrained Decoding

Forces output to match a formal grammar (GBNF format):

```bash
# Only "yes" or "no"
agave model.gguf --grammar-string 'root ::= "yes" | "no"' "Is the sky blue?"

# JSON object with specific fields
agave model.gguf --json-schema '{"type":"object","properties":{"name":{"type":"string"}}}' "User info"

# Any valid JSON
agave model.gguf --json-output "Generate a user profile"
```

The grammar state machine masks logits before sampling — tokens that would violate the grammar get set to -infinity. This guarantees syntactically valid output regardless of sampling parameters.

Supported: GBNF strings, GBNF files (`--grammar`), JSON schemas (`--json-schema`), JSON mode (`--json-output`). Full repetition (`*`/`+`/`?`) and grouped expressions.

## Combining Parameters

Applied in order: **penalties → grammar mask → temperature → min-p → top-k → softmax → top-p → sample**.

```bash
# Deterministic
agave model.gguf -t 0 "What is the capital of France?"

# Balanced
agave model.gguf -t 0.7 --top-p 0.9 "Tell me a story"

# Creative with min-p quality floor
agave model.gguf -t 1.2 --min-p 0.05 "Write a poem"

# Anti-repetition for long-form
agave model.gguf -t 0.8 --repeat-penalty 1.1 -n 1000 "Write an essay"

# Structured output
agave model.gguf --json-schema '{"type":"object","properties":{"answer":{"type":"string"}}}' "Capital of France?"
```

---

**In the code:** [src/ops/math.zig](../../src/ops/math.zig) (sampleToken, applyPenalties, applyMinP, applyRepeatPenalty), [src/grammar.zig](../../src/grammar.zig) (GBNF parser, state machine, JSON schema converter)

**Math reference:** [Argmax](appendix-math.md#argmax), [Temperature Scaling](appendix-math.md#temperature-scaling), [Top-K](appendix-math.md#top-k-selection), [Top-P](appendix-math.md#top-p-nucleus-sampling)

**Next:** [Chapter 8: Backends →](08-backends.md) | **Back:** [Chapter 6: State Space Models ←](06-state-space-models.md) | **Product docs:** [Architecture](../ARCHITECTURE.md), [HTTP API](../API.md)
