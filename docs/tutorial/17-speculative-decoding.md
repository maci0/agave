# 17. Speculative Decoding & DDTree

Standard autoregressive decoding generates one token per forward pass. For large models, each pass takes tens of milliseconds — the token generation rate is bottlenecked by model size, not memory bandwidth. Speculative decoding breaks this bottleneck by using a cheap draft model to propose multiple candidate tokens, then verifying them against the full target model.

## The Core Idea

1. **Draft**: A small, fast model generates K candidate tokens autoregressively
2. **Verify**: The target model checks whether it agrees with each draft token
3. **Accept**: Matching tokens are accepted for free (no extra target compute)
4. **Correct**: At the first disagreement, the target's prediction replaces the draft's

With a good draft model (70-80% acceptance rate), speculative decoding generates 2-3× more tokens per second with **zero quality loss** — the output distribution is mathematically identical to running the target model alone.

## Modes in Agave

### Separate Draft Model (`--draft-model`)

Load a small model alongside the target. Best speedup when the draft model is from the same family (e.g., Qwen3-1.5B drafting for Qwen3-8B):

```bash
agave Qwen3-8B.gguf --draft-model Qwen3-1.5B.gguf "What is quantum computing?"
```

The draft model shares the same GPU/CPU backend and thread pool. Memory overhead is the draft model's weight size plus a small KV cache.

### DDTree Mode (`--spec-mode ddtree`)

DDTree (Ringel & Romano, 2026) improves on standard speculative decoding by constructing a **tree** of candidate continuations instead of a single path. The tree is built using a best-first heap algorithm that selects the most probable prefixes from the draft model's per-position distributions.

```bash
agave model.gguf --draft-model draft.gguf --spec-mode ddtree --spec-tokens 5 --tree-budget 64 "prompt"
```

**How it works:**

1. Draft model runs K forward passes, saving the full logit distribution at each step
2. Top-B tokens are extracted at each depth via partial selection
3. A max-heap explores candidate continuations in order of cumulative log-probability
4. Each pop adds a node to the tree; siblings (same depth, next rank) and children (next depth, rank 0) are pushed
5. The resulting tree is compiled into flat arrays with ancestor bitmasks for tree attention
6. Verification walks the tree: at each depth, if the target model's argmax matches any child, that branch is accepted

The tree structure means the verifier can find longer accepted sequences by exploring alternative branches, yielding higher acceptance lengths than single-path speculation.

**Key parameters:**
- `--spec-tokens K` — draft depth (default: 5). More depth = deeper tree but more draft compute
- `--tree-budget B` — maximum tree nodes (default: 64). Higher budget = wider tree but more verification compute

### Self-Speculative Mode (`--spec-mode self`)

Uses the target model itself as its own draft by skipping layers during the draft phase. No extra model needed — trades quality for speed in the draft:

```bash
agave model.gguf --spec-mode self "prompt"
agave model.gguf --spec-mode self --draft-layers 9 "prompt"  # skip 9 layers
```

The `--draft-layers` flag controls how many layers to skip (default: 50% of model layers, skipping the middle). Fewer skipped layers = higher acceptance rate but less speedup per draft token.

## Architecture

```
src/spec/
├── spec_decode.zig   — orchestrator: draft, verify, generation loop
└── ddtree.zig        — DDTree: heap, tree build, compile, acceptance walk

src/backend/kernels/cpu/
└── sdpa_tree.zig     — tree-masked SDPA kernel (ancestor bitmask attention)
```

### Data Flow

```
┌─────────────┐     K tokens + logits     ┌──────────────┐
│ Draft Model │ ─────────────────────────→ │  DDTree      │
│ (small/skip)│                            │  Builder     │
└─────────────┘                            └──────┬───────┘
                                                  │ tree (B nodes)
                                                  ▼
┌─────────────┐     verify each depth     ┌──────────────┐
│ Target Model│ ←─────────────────────────│  Acceptance   │
│ (full)      │ ─────────────────────────→│  Walk         │
└─────────────┘     argmax at each node   └──────┬───────┘
                                                  │ accepted tokens
                                                  ▼
                                           ┌──────────────┐
                                           │   Output     │
                                           │   Stream     │
                                           └──────────────┘
```

### KV Cache Management

- **Separate models**: Each has independent KV cache. Draft model rolled back to accepted prefix on rejection.
- **Self-draft**: Same KV cache for both phases. Target rollback before re-verification overwrites draft entries (safe because same model produces identical KV).
- **Rollback**: `Model.setKvSeqLen(pos)` — paged blocks stay allocated, overwritten on next forward.

### DDTree Heap Algorithm

The tree construction is O(B log B) where B is the node budget:

```
Initialize: push (depth=0, rank=0) with log_prob = log q₀[best_token]

While tree_size < B:
    Pop node with highest cumulative log-probability
    Add to tree

    Push sibling: (same depth, rank + 1)
        cum_log_prob = parent_cum + log q[depth][rank+1]

    Push child: (depth + 1, rank 0)
        cum_log_prob = current_cum + log q[depth+1][best_token]
```

This produces the optimal prefix-closed tree under the draft model's factorized distribution.

### Tree Attention

Each tree node attends to:
- All prefix KV entries (shared, unconditional)
- Only its ancestor nodes within the tree (bitmask-controlled)

The ancestor bitmask is a `[8]u64` per node (512 bits), supporting trees up to 512 nodes. The CPU kernel (`sdpa_tree.zig`) iterates over attended positions; GPU kernels can mask in the inner loop.

## Correctness Guarantee

For greedy decoding (temperature=0), speculative decoding produces **byte-identical output** to non-speculative decoding. The verification step ensures every accepted token matches what the target model would have generated. This is verified in agave's test suite.

For sampling (temperature > 0), rejection sampling (Leviathan et al. 2023) preserves the target model's output distribution. Each draft token is accepted with probability min(1, p_target/p_draft); on rejection, a correction is sampled from the residual distribution max(0, p_target - p_draft).

## Performance Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `--spec-tokens` | Draft depth K | 3-8 for most models |
| `--tree-budget` | Tree width B | 32-128 (diminishing returns beyond 256) |
| `--draft-layers` | Layers skipped (self-spec) | 25-50% of total layers |
| Draft model size | Acceptance rate vs speed | 1/4 to 1/8 of target size |

### Current Limitations

Verification currently uses sequential `forward()` calls (one per tree node). This means the total compute per speculation round is `K_draft + K_verify` forward calls, vs `K` forward calls without speculation. Speculative decoding is only faster when the draft model is significantly cheaper per forward call AND the acceptance rate is high enough to offset the verification overhead.

**Planned optimization**: Batch tree verification via `forwardTree()` would process all tree nodes in a single target forward pass using tree attention (`sdpaTree`), reducing verification to O(1) target forwards per round regardless of tree size.

**When to use speculative decoding:**
- Long generations (100+ tokens) — amortizes dual-model overhead
- Large target models (8B+) — more room for speedup
- Same-family draft/target — higher acceptance rates

**When NOT to use:**
- Very short outputs (< 10 tokens)
- Small target models (< 3B) — draft overhead dominates
- No suitable draft model available (self-spec with aggressive skip may hurt quality)

## Background: DFlash and Block Diffusion

DDTree builds on **DFlash** (Block Diffusion Flash), a speculative decoding method that uses a **block diffusion model** as the drafter. Unlike autoregressive drafters that generate tokens one at a time, a block diffusion drafter produces an entire block of L draft tokens in a single forward pass by iteratively denoising a block of mask tokens.

**DFlash** (baseline):
1. Run block diffusion drafter once → L draft positions with per-position distributions
2. Sample a single sequence from those distributions
3. Verify the sequence against the target model
4. Accept matching prefix, reject at first mismatch

**DDTree** (improvement over DFlash):
1. Same drafter → same L per-position distributions
2. Instead of sampling one sequence, build an **optimal tree** of candidate continuations
3. The tree explores multiple branches at each depth, prioritized by probability
4. Verify the entire tree → accept the longest matching path (not just one sequence)

The key insight: DFlash wastes information by collapsing the draft distributions into a single path. DDTree exploits the full distribution at each position to construct a tree that maximizes expected acceptance length. The paper shows 35-62% speedup over DFlash.

**In agave's implementation**, we use autoregressive drafting (not block diffusion) since agave doesn't include a diffusion model. The DDTree tree construction algorithm works identically — it takes per-position logit distributions (however produced) and builds the optimal tree. The draft distributions come from K sequential forward passes of the draft model rather than one block diffusion pass.

### References

- [DDTree: Accelerating Speculative Decoding with Block Diffusion Draft Trees (Ringel & Romano, 2026)](https://arxiv.org/abs/2604.12989)
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- [SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference (Miao et al., 2024)](https://arxiv.org/abs/2305.09781)
