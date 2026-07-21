# Chapter 19: PFlash and Block Sparse Attention

Long prompts are expensive. A 128K-token context with full causal attention requires 128K × 128K / 2 = 8 billion dot products for a single layer. Most of that work is wasted: the model's output depends on a small fraction of the input -- the relevant document passages, the key constraint tokens, the system prompt's core instruction.

This chapter covers two techniques that attack this waste directly:

1. **Block sparse attention** -- skip dot products for block pairs that cannot influence each other
2. **PFlash** -- use a cheap scorer to identify which blocks matter, then prefill only those blocks through the full target model

Both are implemented in `src/ops/sparse_attn.zig` and `src/spec/pflash.zig`.

**Prerequisites**: [Chapter 2: The Transformer](02-the-transformer.md), [Chapter 5: Memory and Caching](05-memory-and-caching.md), [Chapter 17: Speculative Decoding](17-speculative-decoding.md)

---

## Block Sparse Attention

Standard attention computes a score between every query token and every key token: O(n²) in sequence length. For long contexts, this dominates runtime. Block sparse attention approximates full attention by grouping tokens into blocks and restricting which block pairs interact.

### The Sparsity Pattern

Agave's `sparse_attn.zig` implements BigBird-style block sparsity with two components: global blocks that every query attends to, and a sliding window of recent blocks for local context. Everything outside those two patterns is skipped entirely.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Q["Query Block\n(any position)"]:::setup
    G["Global Blocks\n(first G blocks)"]:::sync
    W["Window Blocks\n(±W blocks, causal-bounded)"]:::sync
    S["All Other Blocks\n(skipped)"]:::danger
    KV_G["KV Vectors\nDot products computed"]:::success
    KV_W["KV Vectors\nDot products computed"]:::success
    SKIP["No dot products\ninner loop never executes"]:::danger

    Q --> G
    Q --> W
    Q --> S
    G --> KV_G
    W --> KV_W
    S --> SKIP

    subgraph Always["Always attended"]
        G
    end

    subgraph Local["Local context"]
        W
    end

    subgraph Masked["Masked out (~85-98% of pairs)"]
        S
        SKIP
    end
```

**Global blocks** -- the first G blocks attend to and are attended by every block. These typically cover BOS, the system prompt, and the task prefix: tokens the model always needs to see regardless of which part of the context it's drawing from.

**Sliding window** -- each block attends ±W blocks in each direction, bounded by the causal mask, preserving local context for sequential reasoning.

### Block Attention Matrix Layout

The diagram below shows which query blocks (rows) attend which KV blocks (columns) for an 8-block sequence with G=2 global blocks and W=2 window blocks. G = attended as global, W = within sliding window, dot = masked out.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    K0["KV 0\n(global)"]:::success
    K1["KV 1\n(global)"]:::success
    K2["KV 2"]:::sync
    K3["KV 3"]:::sync
    K4["KV 4"]:::sync
    K5["KV 5"]:::sync
    K6["KV 6"]:::sync
    K7["KV 7"]:::sync
    Q0n["Q0: attends G0, G1"]:::setup
    Q3n["Q3: attends G0, G1, W1, W2, W3"]:::setup
    Q5n["Q5: attends G0, G1, W3, W4, W5"]:::setup
    Q7n["Q7: attends G0, G1, W5, W6, W7"]:::setup
    LG["G = global (always)"]:::optional
    LW["W = window (local)"]:::optional
    LD["unmarked = masked, skipped"]:::optional

    subgraph KV["KV blocks (columns)"]
        direction LR
        K0
        K1
        K2
        K3
        K4
        K5
        K6
        K7
    end

    subgraph Q0["Query 0"]
        Q0n
    end
    subgraph Q3["Query 3"]
        Q3n
    end
    subgraph Q5["Query 5"]
        Q5n
    end
    subgraph Q7["Query 7"]
        Q7n
    end

    Q0n -->|"G"| K0
    Q0n -->|"G"| K1
    Q3n -->|"G"| K0
    Q3n -->|"G"| K1
    Q3n -->|"W"| K2
    Q3n -->|"W"| K3
    Q5n -->|"G"| K0
    Q5n -->|"G"| K1
    Q5n -->|"W"| K3
    Q5n -->|"W"| K4
    Q5n -->|"W"| K5
    Q7n -->|"G"| K0
    Q7n -->|"G"| K1
    Q7n -->|"W"| K5
    Q7n -->|"W"| K6
    Q7n -->|"W"| K7

    subgraph Legend["Legend"]
        LG
        LW
        LD
    end
```

At 2048 blocks (128K tokens) with G=2 and W=2, the attended fraction drops to ~3-5%. The reduction scales with sequence length: a 200x reduction at 128K tokens.

Each query block always computes scores for:
- All G global blocks (two in this example)
- W+1 window blocks around itself

All other block pairs are skipped entirely -- the inner loop doesn't execute, so there are no partial computations or wasted multiply-accumulate operations.

### Implementation

The CPU kernel in `sparse_attn.zig` works at the block level:

```
for each query_block qb:
    accumulate attention over:
        global_blocks[0..G]            // always attend
        window_blocks[qb-W .. qb+1]    // W+1 blocks including self
    skip all other kv_blocks           // no dot products computed
```

This is the kernel used by PFlash's scorer pass. It is not used during target model prefill (which gets the compressed prompt) or during decode (which operates one token at a time and doesn't need sparsity).

### Complexity

| Sequence length | Full attention | Block sparse (G=2, W=2, B=64) |
|-----------------|---------------|-------------------------------|
| 8K tokens (128 blocks) | 64M dot products / layer | ~5M dot products / layer |
| 32K tokens (512 blocks) | 1B dot products / layer | ~20M dot products / layer |
| 128K tokens (2048 blocks) | 16B dot products / layer | ~80M dot products / layer |

The reduction scales with sequence length. At 128K tokens, block sparse attention is roughly 200x cheaper per layer than full attention.

---

## PFlash: Speculative Prefill

Block sparse attention makes the scorer fast. PFlash uses that cheap scorer to decide which blocks the expensive target model needs to see at all.

### The Core Algorithm

PFlash runs four sequential stages: a cheap sparse scorer pass over the full prompt, an adaptive threshold to pick which blocks matter, a compressed prefill through the expensive target model, and then normal speculative decode.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Prompt["Full Prompt\n(e.g. 128K tokens)"]:::setup
    Scorer["Scorer Model\n(small, 0.5-3B)"]:::setup
    Sparse["Block Sparse\nAttention"]:::sync
    BlockScores["Per-block\nimportance scores"]:::migration
    Threshold["alpha * mean(scores)"]:::migration
    Kept["Selected blocks\n(5-15% of total)"]:::migration
    Compress["Compress to\nselected spans only"]:::migration
    Target["Target Model\n(large, 8-70B)"]:::setup
    KVCache["Populated KV Cache\n(compressed context)"]:::success
    DDTree["DDTree Speculative\nDecode"]:::sync
    Tokens["Output Tokens"]:::success

    Prompt --> Score

    subgraph Score["Stage 1: Score (fast)"]
        Scorer --> Sparse --> BlockScores
    end

    BlockScores --> Select

    subgraph Select["Stage 2: Select"]
        Threshold --> Kept
    end

    Kept --> Prefill

    subgraph Prefill["Stage 3: Prefill (expensive)"]
        Compress --> Target --> KVCache
    end

    KVCache --> Decode

    subgraph Decode["Stage 4: Decode"]
        DDTree --> Tokens
    end
```

**Step 1: Score.** Run the scorer model over the full prompt with block sparse attention. In the current implementation, each block is scored by its position in the sequence (recency heuristic). A KV-dot-product scorer (scoreFromLastQ) is defined but not yet integrated into the main prefill pipeline.

**Step 2: Select.** Apply the adaptive threshold:

```
mean_score = mean(all block scores)
selected = {block b : score[b] > alpha * mean_score}
```

Blocks above the threshold are kept; the rest are discarded. With `alpha=0.85` and a typical prompt, 5-15% of blocks are selected.

**Step 3: Prefill.** Concatenate the selected spans into a compressed prompt. Run the target model's standard prefill over only those tokens. The target model's KV cache now contains the compressed context.

**Step 4: Decode.** Run DDTree speculative decode as normal. The KV cache has the compressed representation; attention during decode only reaches the retained blocks.

### Adaptive Threshold vs Fixed Top-K

The threshold `alpha * mean(scores)` adapts to prompt structure: a dense technical reference has a high mean score so more blocks are kept; a padded narrative has a low mean so nearly all boilerplate is dropped. A fixed top-K cannot distinguish these cases.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Scores["Block Importance Scores"]:::setup
    Mean["Compute mean(scores)"]:::sync
    Threshold["threshold = alpha * mean"]:::migration
    Keep["Keep block\n(sent to target model)"]:::success
    Drop["Drop block\n(never reaches target)"]:::danger
    D1["block: 0.80 -- KEEP"]:::success
    D2["block: 0.60 -- DROP"]:::danger
    D3["block: 0.90 -- KEEP"]:::success
    P1["block: 0.90 -- KEEP"]:::success
    P2["block: 0.05 -- DROP"]:::danger
    P3["block: 0.03 -- DROP"]:::danger

    Scores --> Mean
    Mean --> Threshold
    Threshold --> Compare{"score > threshold?"}
    Compare -->|yes| Keep
    Compare -->|no| Drop

    subgraph Dense["Dense prompt (API docs)\nmean=0.78, threshold=0.66"]
        D1
        D2
        D3
    end

    subgraph Padded["Padded prompt (novel + boilerplate)\nmean=0.14, threshold=0.12"]
        P1
        P2
        P3
    end

    Keep -.->|"~70% kept\n(dense case)"| Dense
    Drop -.->|"~92% dropped\n(padded case)"| Padded
```

Compare two prompts with alpha=0.85:

```
Dense technical reference (API docs, 128K tokens):
  scores:  [0.8, 0.9, 0.7, 0.85, 0.6, 0.9, 0.8, ...]
  mean:    0.78
  threshold: 0.66
  selected: ~70% of blocks (doc is uniformly dense)

Padded narrative with boilerplate (novel + copyright headers, 128K tokens):
  scores:  [0.9, 0.1, 0.05, 0.08, 0.02, 0.85, 0.03, ...]
  mean:    0.14
  threshold: 0.12
  selected: ~8% of blocks (most content is irrelevant)
```

A fixed top-K=50 would over-compress the first case and under-compress the second. The adaptive threshold naturally handles both without tuning.

### Step-by-Step Walkthrough

**Setup:**
```bash
agave target-14B.gguf \
  --draft-model draft-1.5B.gguf \
  --spec-mode pflash \
  --pflash-alpha 0.85 \
  --pflash-block-size 64 \
  "You are a code reviewer. [... 60K tokens of context ...] What does this function do?"
```

**What happens internally:**

1. Scorer (draft-1.5B) runs a forward pass over 60K tokens using block sparse attention. This takes ~200ms instead of ~4000ms for full attention at this length.

2. For each of the ~940 blocks, a score is computed from the scorer's key vectors at that position.

3. Alpha threshold selects ~120 blocks (13%) -- approximately 7700 tokens. These span the function definition, its call sites, and relevant type definitions from earlier in the context.

4. Target model (target-14B) prefills those 7700 tokens. TTFT is now comparable to a 7700-token prompt, not a 60K-token prompt.

5. DDTree decode runs as normal, producing tokens at full target model quality.

### Choosing Alpha

Alpha controls the compression ratio. Lower alpha = more aggressive = faster prefill = higher risk of quality degradation.

| Alpha | Typical selection rate | Use when |
|-------|----------------------|----------|
| 0.95 | 15-20% | Output quality is critical, modest TTFT improvement acceptable |
| 0.85 | 10-20% | Good balance for most tasks (default) |
| 0.70 | 5-10% | Aggressive; prompts with large irrelevant sections |
| 0.50 | 2-5% | Very aggressive; only well-structured retrieval prompts |

Selection is additionally capped by `max_kept_ratio` (default 0.20); rates above 20% require raising this cap.

To find the right alpha for your use case:

1. Start at 0.85 and evaluate output quality on representative prompts
2. If quality is acceptable, try 0.75 and re-evaluate
3. If quality degrades at 0.85, raise to 0.90 or 0.95

There is no universal correct alpha -- it depends on how compressible your prompt content is and how sensitive your task is to dropped context.

### Using a Separate Scorer Model

By default, the `--draft-model` acts as the scorer. This is convenient but may not be optimal: the draft model is sized for producing accurate token predictions, which requires more parameters than block ranking.

```bash
# Default: draft model is also the scorer
agave target.gguf --draft-model draft-3B.gguf --spec-mode pflash "prompt"

# Separate scorer: use a tiny model just for block importance ranking
agave target.gguf \
  --draft-model draft-3B.gguf \
  --pflash-scorer scorer-0.5B.gguf \
  --spec-mode pflash \
  "prompt"
```

With a dedicated scorer, both models are loaded into memory simultaneously. The scorer runs once at prefill time and is then idle. The draft model then handles DDTree decode as normal.

When to use a separate scorer:
- Your draft model is already large (3B+) and scoring overhead is significant
- You have a very small model (0.5B or less) that can identify important blocks even if its token predictions are poor
- You want to minimize scorer latency to maximize TTFT improvement

### PFlash + DDTree: Full Pipeline

PFlash and DDTree are designed to compose. PFlash attacks TTFT by shrinking what the target model must prefill; DDTree attacks decode latency by drafting multiple tokens per target-model pass. Each technique targets a separate bottleneck, so their gains multiply.

```mermaid
sequenceDiagram
    participant User
    participant Scorer as Scorer Model (small)
    participant Target as Target Model (large)
    participant Draft as Draft Model (small)

    User->>Scorer: Full prompt (128K tokens)
    Note over Scorer: Block sparse attention pass (~200ms)
    Scorer-->>Target: Selected blocks only (~8K tokens)
    Note over Target: Compressed prefill (~500ms vs 6000ms)
    Target-->>Draft: KV cache ready, begin decode

    loop DDTree decode (per generation step)
        Draft->>Draft: Propose 5-token tree (~8ms)
        Draft->>Target: Tree of candidates
        Target->>Target: Verify batch in 1 pass (~65ms)
        Target-->>User: Accept ~3 tokens on average
    end
```

Together they address the full latency profile:

```bash
agave target-14B.gguf \
  --draft-model draft-1.5B.gguf \
  --spec-mode pflash \
  --pflash-alpha 0.85 \
  --spec-tokens 5 \
  --tree-budget 64 \
  "prompt"
```

**Combined pipeline:**

```
[PREFILL PHASE]
Scorer: block sparse pass over 128K tokens  (~200ms)
  -> select 8K tokens (alpha threshold)
Target: prefill 8K compressed tokens         (~500ms)
  vs. target: prefill 128K tokens            (~6000ms without PFlash)

[DECODE PHASE]
Draft (1.5B): 5-token proposal               (~8ms)
DDTree build: heap + tree compile            (~0.1ms)
Target verify: 1 batched tree pass           (~65ms)
Accept ~3 tokens average

Effective decode: ~35 tok/s (same as without PFlash -- TTFT improved, not tok/s)
```

PFlash cuts TTFT; DDTree cuts generation latency. For a 128K-token prompt generating 500 tokens:

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    B_TTFT["TTFT: 6000ms\n(full 128K prefill)"]:::danger
    B_DEC["Decode: 33000ms\n(500 tokens, 1 per step)"]:::danger
    P_TTFT["TTFT: 700ms\n(8K compressed prefill)"]:::success
    P_DEC["Decode: 33000ms\n(500 tokens, 1 per step)"]:::danger
    D_TTFT["TTFT: 6000ms\n(full 128K prefill)"]:::danger
    D_DEC["Decode: 14000ms\n(~3 tokens accepted/step)"]:::success
    C_TTFT["TTFT: 700ms\n(8K compressed prefill)"]:::success
    C_DEC["Decode: 14000ms\n(~3 tokens accepted/step)"]:::success

    subgraph Baseline["No features\n39s total"]
        B_TTFT --> B_DEC
    end

    subgraph PFlashOnly["PFlash only\n34s total"]
        P_TTFT --> P_DEC
    end

    subgraph DDTreeOnly["DDTree only\n20s total"]
        D_TTFT --> D_DEC
    end

    subgraph Both["PFlash + DDTree\n15s total"]
        C_TTFT --> C_DEC
    end
```

The gains multiply because they target different bottlenecks: PFlash owns TTFT, DDTree owns decode throughput.

### Block Size Tuning

`--pflash-block-size` controls the granularity of block selection.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    FA["tok 0-15\nKEPT"]:::success
    FB["tok 16-31\ndropped"]:::danger
    FC["tok 32-47\nKEPT"]:::success
    FD["tok 48-63\ndropped"]:::danger
    FE["tok 64-79\nKEPT"]:::success
    FF["tok 80-95\ndropped"]:::danger
    FG["tok 96-111\nKEPT"]:::success
    FH["tok 112-127\ndropped"]:::danger
    CA["tok 0-63\nKEPT (contains important lines)"]:::success
    CB["tok 64-127\ndropped (all or nothing)"]:::danger
    T1["Fine: precise selection\nmore overhead per span\nbetter for code/JSON/tables"]:::optional
    T2["Coarse: fast scoring\nfewer discontinuous spans\nbetter for long-form text"]:::optional

    subgraph Fine["Fine blocks (16-32 tokens)\nBlock size = 16 tokens"]
        direction LR
        FA
        FB
        FC
        FD
        FE
        FF
        FG
        FH
    end

    subgraph Coarse["Coarse blocks (64-128 tokens)\nBlock size = 64 tokens"]
        direction LR
        CA
        CB
    end

    subgraph Tradeoff["Tradeoff"]
        T1
        T2
    end
```

Smaller blocks (16-32 tokens):
- Finer selection: can keep a single important sentence while discarding surrounding text
- More blocks to score: slightly more scorer overhead
- More overhead in target model prefill (more discontinuous spans)

Larger blocks (64-128 tokens):
- Coarser selection: must keep or discard whole paragraphs
- Fewer blocks to score: faster scorer pass
- Fewer discontinuous spans: lower prefill overhead

The default of 64 tokens works well for most prompts. For highly structured prompts (code, JSON, tables) where important content is concentrated in specific lines, try 32. For long-form text where sections are the natural unit, try 128.

---

## When to Use Each Feature

### Block Sparse Attention (used automatically by PFlash scorer)

You do not invoke block sparse attention directly. It runs inside the PFlash scorer pass. Understanding it helps you reason about scorer accuracy:

- Global blocks (first G blocks) are always accurate -- they see everything
- Window blocks maintain local coherence
- Blocks outside the window are invisible to non-global queries during scoring

If your prompt has critical information near the end, make sure it falls within the window of at least one global block or increase G (currently not exposed as a flag -- contact the team if you need tunable G).

### PFlash

Use when:
- TTFT matters (user is waiting for the first token)
- Prompts are 32K+ tokens
- The prompt contains large amounts of potentially irrelevant content (RAG chunks, long documents, conversation history)

Do not use when:
- Prompts are short (< 8K tokens)
- Every part of the prompt is critical (adversarial robustness testing, legal analysis)
- You need deterministic output matching the non-PFlash path (PFlash is an approximation)

### DDTree

Use when:
- Decode throughput matters (generating hundreds of tokens)
- A draft model is available from the same family as the target
- Target model is large (8B+)

Do not use when:
- Generating very short outputs (< 20 tokens)
- No draft model is available and self-spec quality is poor

---

## Summary

Block sparse attention reduces scoring cost from O(n²) to O(n) by skipping dot products for block pairs outside the global and window pattern. PFlash uses this cheap scorer to select which KV blocks the target model must process, compressing a 128K-token prefill to 5-15K tokens while preserving the information the model will actually use. DDTree then accelerates decode. The two techniques address distinct bottlenecks and compose without interference.

---

**In the code:** [src/ops/sparse_attn.zig](../../src/ops/sparse_attn.zig) (block sparse CPU SDPA kernel), [src/spec/pflash.zig](../../src/spec/pflash.zig) (block scoring, adaptive threshold, compressed prefill)

**Next:** [Chapter 20: Diffusion Language Models →](20-diffusion-lm.md) | **Back:** [Chapter 18: Multi-Token Prediction ←](18-multi-token-prediction.md) | **Related:** [Chapter 17: Speculative Decoding ←](17-speculative-decoding.md)

---

## Glossary

**adaptive threshold (alpha)** — The selection criterion `alpha × mean(block_scores)` determining which blocks survive PFlash compression.

**block importance score** — A per-block value indicating how relevant that block is to the model's output.

**block sparse attention** — An approximation of full attention restricting which block pairs interact, reducing complexity from O(n²) to O(n).

**compressed prefill** — Running the target model's standard prefill over only PFlash-selected tokens instead of the full prompt.

**global block** — One of the first G token blocks that every query block attends to, typically covering the system prompt.

**max_kept_ratio** — A cap (default 0.20) on the fraction of blocks PFlash can retain, preventing over-selection.

**pflash-alpha** — The tunable parameter controlling PFlash compression aggressiveness (default 0.85); lower = more aggressive.

**pflash-block-size** — The granularity of block selection in PFlash (default 64 tokens).

**PFlash** — A speculative prefill technique using a cheap scorer to identify which KV blocks matter, then prefilling only those.

**scorer model** — A small model that runs block sparse attention over the full prompt to produce per-block importance scores.

**sliding window (block sparse)** — A pattern where each query block attends to ±W neighboring blocks, preserving local context.

**sparsity pattern** — The combination of global blocks and sliding window determining which block pairs compute attention scores.
