# Chapter 7: Sampling

After the forward pass produces **logits** (raw unnormalized scores, one per vocabulary token), the model must **select** the next token. The simplest method is **greedy decoding** (pick the highest score), but this produces repetitive, **deterministic** (always the same output for the same input) output. Sampling parameters add controlled randomness for more natural text.

### Code Flow

```text
logits -> filters (bias, penalties, grammar mask, temperature, XTC, min-p, top-k, top-p) -> sample
```

## Temperature

Controls randomness by scaling logits before sampling:

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Logits["Raw Logits\n[3.2, 1.1, 0.4, ...]"]:::setup
    Divide["Divide by Temperature\nlogit / T"]:::sync
    Adjusted["Adjusted Logits"]:::migration
    Softmax["Softmax\ne^x / Σe^x"]:::sync
    Probs["Probabilities\n[0.72, 0.19, 0.09, ...]"]:::migration
    Sample["Weighted Random Pick"]:::sync
    Token["Next Token"]:::success
    T_low["T=0.3 → peaky\ntop token dominates"]:::optional
    T_mid["T=1.0 → balanced\nraw model probs"]:::optional
    T_high["T=1.5 → flat\nmany tokens compete"]:::optional

    Logits --> Divide
    Divide --> Adjusted
    Adjusted --> Softmax
    Softmax --> Probs
    Probs --> Sample
    Sample --> Token

    subgraph Effect["Temperature Effect"]
        T_low
        T_mid
        T_high
    end
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

Introduced in [The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751), nucleus sampling restricts sampling to the smallest set of tokens whose **cumulative probability** (running sum of probabilities in sorted order) exceeds P:

```
--top-p 0.9    Keep tokens until cumulative probability reaches 90%
--top-p 1.0    Disabled — default
```

More adaptive than top-k: when the model is confident (top token = 95%), top-p=0.9 keeps 1-2 candidates. When uncertain (many similar scores), it keeps dozens.

**Top-K vs Top-P**: Top-K always keeps exactly K tokens. Top-P adapts based on confidence. They can be combined.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Start["Sorted Token Probabilities\n[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]"]:::setup
    TopK["Top-K Filter\nkeep only top K tokens"]:::sync
    TopP["Top-P Filter\ncumulate until sum >= P"]:::sync
    K_out["Fixed K candidates\ne.g. top-3: [0.40, 0.25, 0.15]"]:::migration
    P_out["Variable candidates\ne.g. P=0.9: [0.40, 0.25, 0.15, 0.10]\n(cumsum = 0.90)"]:::migration
    Renorm_K["Renormalize to 1.0"]:::sync
    Renorm_P["Renormalize to 1.0"]:::sync
    Combined["Both applied? Intersection wins\n(whichever is more restrictive)"]:::migration
    Sample["Sample from remaining tokens"]:::success
    Certain["Confident model\ntop-p=0.9 → 1-2 tokens"]:::optional
    Uncertain["Uncertain model\ntop-p=0.9 → 20+ tokens"]:::optional

    Start --> TopK
    Start --> TopP
    TopK --> K_out
    TopP --> P_out
    K_out --> Renorm_K
    P_out --> Renorm_P
    Renorm_K --> Combined
    Renorm_P --> Combined
    Combined --> Sample

    subgraph Confidence["Model confidence drives top-p size"]
        Certain
        Uncertain
    end
```

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
log_threshold = max(logits) + log(min_p)
keep tokens where logit >= log_threshold (set others to -inf)
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

## XTC (eXclude Top Choices)

XTC randomly excludes high-probability tokens to increase diversity. With probability `xtc_probability`, all tokens above `xtc_threshold` probability (except one) are set to -infinity, forcing the model to pick a less obvious continuation.

```json
{"xtc_probability": 0.5, "xtc_threshold": 0.1, "temperature": 0.8}
```

Combats **mode collapse** where the model repeatedly generates the same high-probability sequences. Most useful for creative writing and brainstorming. Unlike temperature which scales all probabilities, XTC specifically removes the top choices while keeping the rest of the distribution intact.

## DRY (Don't Repeat Yourself)

DRY penalizes tokens that would continue a repeated n-gram sequence. If the model has generated "the cat sat on" earlier and is about to generate it again, DRY applies increasing penalty proportional to the match length.

```json
{"dry_multiplier": 1.5, "dry_allowed_length": 3}
```

`dry_multiplier` scales the penalty (0 = disabled). `dry_allowed_length` sets the minimum n-gram length to trigger (default 2 — penalize repeated bigrams and longer). More effective than `repeat_penalty` because it detects repeated **sequences**, not just individual tokens. A token might be fine to repeat (e.g., "the") unless it's part of a repeated phrase.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    History["Generation History\n[... the cat sat on the mat ...]"]:::setup
    Window["Sliding Window Scan\nfor each candidate token C"]:::sync
    NoPenalty["No DRY penalty\nlogit unchanged"]:::success
    Extend["Extend match backward\nhow many prior tokens also match?"]:::sync
    Length["Match length L\n(tokens in common prefix)"]:::migration
    Penalty["Apply penalty\nlogit -= dry_multiplier × L"]:::danger
    Ex1["token 'sat' after 'cat'\nL=1 (just 'sat') → no penalty"]:::optional
    Ex2["token 'on' after 'cat sat'\nL=2 (bigram) → penalty x1.5^2=2.25"]:::optional
    Ex3["token 'mat' after 'cat sat on'\nL=3 → penalty x1.5^3=3.375"]:::optional
    RP["repeat_penalty: penalizes\neach token individually\n'the' always penalized"]:::optional
    DRY2["DRY: penalizes token only\nwhen it continues a phrase\n'the' fine alone, penalized in repeated phrase"]:::optional

    History --> Window
    Window --> Match{"Does token C appear\nearlier in history?"}
    Match -->|No match| NoPenalty
    Match -->|Match found| Extend
    Extend --> Length
    Length --> Allowed{"L >= dry_allowed_length?"}
    Allowed -->|No, sequence too short| NoPenalty
    Allowed -->|Yes, repeated phrase| Penalty

    subgraph Example["Example: dry_multiplier=1.5, dry_allowed_length=2"]
        Ex1
        Ex2
        Ex3
    end

    subgraph Contrast["vs repeat_penalty"]
        RP
        DRY2
    end
```

## Mirostat

Mirostat maintains consistent **perplexity** (unpredictability) during generation by dynamically adjusting the sampling threshold. Instead of fixed temperature, it targets a specific entropy level (tau) and adapts via learning rate (eta):

```json
{"mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "temperature": 0.8}
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `mirostat` | 0 | Mode: 0=disabled, 2=Mirostat 2.0 |
| `mirostat_tau` | 5.0 | Target entropy — lower = more focused, higher = more creative |
| `mirostat_eta` | 0.1 | Learning rate — how fast to adapt |

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Logits["Raw Logits\none score per vocab token"]:::setup
    Trunc["Truncate to top-k candidates\n(Mirostat controls k dynamically)"]:::sync
    Softmax["Softmax\ncompute probabilities"]:::sync
    Sample["Weighted Random Sample\npick next token"]:::sync
    Surprise["Measure Surprise\n-log2(prob of sampled token)"]:::migration
    Error["Error = surprise - tau\ntau = target entropy"]:::migration
    Update["Update mu\nmu -= eta * error"]:::migration
    NextK["Set next k\nbased on updated mu"]:::migration
    Tau["tau (target entropy)\nlower = focused\nhigher = creative"]:::optional
    Eta["eta (learning rate)\nhow fast mu adapts"]:::optional
    Mu["mu\ncurrent entropy estimate\nstarts at 2 * tau"]:::optional

    Logits --> Trunc
    Trunc --> Softmax
    Softmax --> Sample
    Sample --> Surprise
    Surprise --> Error
    Error --> Update
    Update --> NextK
    NextK -->|next token| Logits

    subgraph Params["Control Parameters"]
        Tau
        Eta
    end

    subgraph State["Running State"]
        Mu
    end

    Update -.->|adjusts| Mu
    Mu -.->|drives| NextK
    Tau -.->|anchors| Error
    Eta -.->|scales| Update
```

When Mirostat is active, top-k and top-p are bypassed — Mirostat controls its own truncation. It works by tracking a running "surprise" estimate and adjusting which tokens are eligible for sampling. Produces more consistently readable output than fixed temperature across varying prompt types.

## Logit Bias

Direct per-token adjustments to logits via the API. Specify token IDs and additive bias values:

```json
{"logit_bias": {"123": 5.0, "456": -100.0}}
```

Positive values increase the token's chance of being selected; large negative values effectively ban it. Applied before any other sampling — useful for steering output without changing the model. Max 16 entries per request.

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

**Gotcha: grammar's interaction with sampling varies by path and token position.** Both `main.zig` (CLI) and `src/server/server.zig` (HTTP) mask invalid tokens to `-infinity` first, ahead of temperature, top-k/top-p, min-p, and XTC. The CLI then runs the normal sampling pipeline over the masked logits whenever temperature is non-zero, so distinct grammar-valid completions can still be sampled. On the HTTP server, streaming (SSE) responses call `argmax` on the masked logits for every token, first and subsequent alike, so streamed grammar output is always deterministic. Non-streaming HTTP responses argmax every token after the first, but the first token still runs the full sampling pipeline when temperature is non-zero, so a non-streaming grammar-constrained response can start with a sampled token and settle into deterministic argmax from the second token on.

**Jump decoding**: When the grammar allows exactly one valid next token (e.g., a colon after a JSON key, a closing brace at the end), the forward pass is skipped entirely and that token is emitted directly. This eliminates unnecessary GPU compute for deterministic structural tokens, significantly speeding up JSON schema output where many tokens are fixed by the schema.

```mermaid
stateDiagram-v2
    [*] --> GrammarState: parse grammar / JSON schema

    GrammarState --> ValidSet: compute valid next tokens\nfrom current state

    ValidSet --> JumpCheck: how many valid tokens?

    JumpCheck --> JumpDecode: exactly one valid token\n(e.g. colon after JSON key)
    JumpCheck --> MaskLogits: multiple valid tokens

    JumpDecode --> EmitToken: emit token directly\nno GPU forward pass needed

    MaskLogits --> MaskedLogits: set invalid tokens to -infinity\nvalid tokens unchanged

    MaskedLogits --> SamplingPipeline: temperature / top-k / top-p\napplied to masked logits

    SamplingPipeline --> Sample: weighted random pick\nfrom valid-token distribution

    Sample --> EmitToken: sampled token

    EmitToken --> AdvanceState: advance grammar state machine\nwith emitted token

    AdvanceState --> GrammarState: ready for next position

    AdvanceState --> [*]: grammar accepted\n(output complete)
```

Supported: GBNF strings, GBNF files (`--grammar`), JSON schemas (`--json-schema`), JSON mode (`--json-output`). Full repetition (`*`/`+`/`?`) and grouped expressions.

## Combining Parameters

Use this decision tree to pick parameters for your use case, then the pipeline diagram below shows the order they apply at runtime.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Start["What are you generating?"]:::setup
    Greedy["temperature=0\ngreedy argmax"]:::success
    Grammar["--json-schema / --grammar\ngrammar mask handles the rest"]:::success
    LongForm["repeat_penalty=1.1\nDRY multiplier=1.5\ntemperature=0.8"]:::success
    Creative["temperature=1.2\nmin_p=0.05\nor XTC for variety"]:::success
    Mirostat["mirostat=2\ntau=5.0\n(ignores top-k/p)"]:::success
    Balanced["temperature=0.7\ntop-p=0.9"]:::success

    Start --> Q1{"Need exact,\nreproducible output?"}
    Q1 -->|Yes| Greedy
    Q1 -->|No| Q2{"Structured output\nrequired?"}
    Q2 -->|Yes - JSON/grammar| Grammar
    Q2 -->|No| Q3{"What matters most?"}
    Q3 -->|Avoid repetition in long text| LongForm
    Q3 -->|Creative + diverse| Creative
    Q3 -->|Consistent readability| Mirostat
    Q3 -->|General conversation| Balanced

    subgraph Defaults["Safe starting point"]
        Balanced
    end
```

Applied in order:

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Logits["Raw Logits\none score per vocab token"]:::setup
    Bias["Logit Bias\nper-token additive adjustments"]:::sync
    Penalties["Repetition Penalties\nrepeat / frequency / presence / DRY"]:::sync
    Grammar["Grammar Mask\nset invalid tokens to -infinity"]:::sync
    Temp["Temperature Scaling\nlogits /= temperature"]:::sync
    XTC["XTC Exclusion\nrandomly drop top tokens"]:::sync
    MinP["Min-P Filter\ndrop tokens below min_p × max_prob"]:::sync
    TopK["Top-K Filter\nkeep only K highest"]:::sync
    Softmax["Softmax\nconvert logits to probabilities"]:::migration
    TopP["Top-P Filter\nnucleus cutoff + renormalize"]:::sync
    MiroTrunc["Mirostat Truncation\nentropy-targeted cutoff"]:::optional
    FinalSample["Weighted Random Sample"]:::sync
    NextToken["Next Token ID"]:::success

    Logits --> Bias
    Bias --> Penalties
    Penalties --> Grammar
    Grammar --> Temp
    Temp --> XTC
    XTC --> MinP
    MinP --> TopK
    TopK --> Softmax
    Softmax --> TopP
    TopP --> Mirostat{"Mirostat\nactive?"}
    Mirostat -->|Yes - replaces top-k/p| MiroTrunc
    Mirostat -->|No| FinalSample
    MiroTrunc --> FinalSample
    FinalSample --> NextToken
```
logits (raw scores, one per vocab token)
  │
  ├─ logit bias (per-token additive adjust)   [API steering]
  ├─ repeat/frequency/presence penalties      [per-token logit modification]
  ├─ DRY penalty (repeated n-gram sequences)  [sequence-aware penalty]
  ├─ grammar mask (set invalid tokens to -∞)  [hard constraint]
  ├─ temperature scaling (logits /= temp)     [control sharpness]
  ├─ XTC exclusion (drop top tokens randomly) [diversity injection]
  ├─ min-p filter (drop < min_p × max)        [adaptive threshold]
  ├─ top-k filter (keep only top K tokens)    [hard cutoff]
  │
  ├─ softmax → probabilities                  [logits → probabilities]
  │
  ├─ top-p filter (keep smallest set ≥ P)     [nucleus cutoff, renormalize]
  │
  ├─ Mirostat (if active, replaces top-k/p)   [entropy-targeted truncation]
  │
  └─ sample from distribution                 [weighted random pick]
       → next token ID
```

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

## Common Configurations

A quick-reference cheat sheet for the algorithm parameters covered above (the decision tree gives the reasoning; this gives concrete starting values):

| Use case | temperature | top_k | top_p | min_p | notes |
|----------|------------:|------:|------:|------:|-------|
| Factual / code | 0–0.2 | 40 | 0.9 | 0 | near-greedy |
| General chat | 0.7–0.9 | 0 | 0.9 | 0.05 | balanced |
| Creative writing | 1.1–1.3 | 0 | 0.95 | 0.02 | wider nucleus |
| Strict structured | 0 | 0 | 1.0 | 0 | grammar/constrained |

These are starting points, not hard rules. A model with a narrower vocabulary distribution may need a lower temperature than shown here to feel equally focused.

---

**In the code:** [src/ops/math.zig](../../src/ops/math.zig) (sampleToken, applyPenalties, applyMinP, applyRepeatPenalty, applyXtc, applyDry, sampleMirostat, applyLogitBias), [src/grammar.zig](../../src/grammar.zig) (GBNF parser, state machine, JSON schema converter)

**Math reference:** [Argmax](appendix-math.md#argmax), [Temperature Scaling](appendix-math.md#temperature-scaling), [Top-K](appendix-math.md#top-k-selection), [Top-P](appendix-math.md#top-p-nucleus-sampling)

**Next:** [Chapter 8: Backends →](08-backends.md) | **Back:** [Chapter 6: State Space Models ←](06-state-space-models.md) | **Product docs:** [Architecture](../ARCHITECTURE.md), [HTTP API](../API.md)

---

## Glossary

**DRY (Don't Repeat Yourself)** — A penalty method that detects repeated n-gram sequences and penalizes tokens that would continue them.

**entropy** — A measure of uncertainty in a probability distribution; higher entropy = more uniform/unpredictable.

**frequency penalty** — An additive per-occurrence penalty proportional to how many times a token has appeared.

**GBNF (Generative BNF)** — A grammar format used to specify valid output patterns for constrained decoding.

**grammar-constrained decoding** — Masking logits so only tokens consistent with a formal grammar can be selected.

**greedy decoding** — Always selecting the highest-probability token (argmax); deterministic but often repetitive.

**jump decoding** — Skipping the forward pass when the grammar allows exactly one valid next token, emitting it directly.

**logit bias** — Direct additive adjustments to specific token logits before sampling, used for API-level steering.

**min-P** — An adaptive threshold keeping only tokens whose probability is at least min_p × the top token's probability.

**Mirostat** — An adaptive sampling method that dynamically adjusts the candidate set to maintain a target entropy level.

**mode collapse** — When sampling repeatedly produces the same high-probability sequences due to insufficient diversity.

**n-gram** — A contiguous sequence of n tokens (bigram = 2, trigram = 3, etc.).

**presence penalty** — A one-time additive penalty applied to any token that has appeared at least once.

**repeat penalty** — A multiplicative penalty applied to logits of previously generated tokens to discourage repetition.

**temperature** — A scaling factor applied to logits before softmax; lower = peakier distribution, higher = flatter.

**top-K sampling** — Restricting the candidate set to only the K highest-scoring tokens before sampling.

**top-P / nucleus sampling** — Keeping the smallest set of tokens whose cumulative probability exceeds P, then renormalizing.

**XTC (eXclude Top Choices)** — A sampling method that randomly removes high-probability tokens to increase diversity.
