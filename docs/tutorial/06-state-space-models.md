# Chapter 6: State Space Models

SSMs, as formalized in [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752), are an alternative to attention that process tokens in **O(1)** time per step (constant time — doesn't grow with sequence length) instead of O(n²). Instead of re-reading all previous tokens, they maintain a fixed-size **state matrix** that summarizes the past:

```
state[t] = decay * state[t-1] + input[t]    (simplified)
output[t] = state[t] @ query[t]
```

The **decay** factor controls how quickly old information fades — like a leaky bucket where new information flows in and old information gradually drains out.

**Hybrid models** combine attention and SSM layers: attention every N layers for global context, SSM for the rest for speed.

## Causal Convolution

Both DeltaNet and Mamba-2 use **causal convolution** as a preprocessing step. A **convolution** is a sliding window operation that combines nearby values using learned weights. **Causal** means it only looks at past inputs (backward in time), never future ones — ensuring the model can't "cheat" by seeing ahead:

```
conv_out[t] = sum(conv_weight[k] * input[t-k] for k in 0..d_conv)
```

With `d_conv=4`, each output depends on the current input and the 3 most recent. A **ring buffer** (a fixed-size circular array where new entries overwrite the oldest, avoiding reallocation) stores the history (zero allocation in the hot path):

```
Ring buffer: [input[t-3], input[t-2], input[t-1]]
New input:   input[t]
Output:      w[0]*buf[0] + w[1]*buf[1] + w[2]*buf[2] + w[3]*input[t]
Shift left:  buffer becomes [input[t-2], input[t-1], input[t]]
```

Agave fuses the convolution with SiLU activation in a single pass.

## DeltaNet (Qwen3.5)

**The problem**: Standard attention is O(n²) — for a 100K-token context, that's 10 billion pairwise comparisons. Computationally expensive and memory-intensive.

**DeltaNet's solution**: Replace the quadratic attention computation with a **linear-complexity recurrence** (an update loop where each step depends only on the previous step's state, not all history). Instead of comparing the current token to all 100K previous tokens, maintain a fixed-size summary (the state matrix) that gets updated incrementally.

**How it works**: DeltaNet maintains a per-head state matrix `S[v_dim, k_dim]` that accumulates information via the **delta rule** — error-correcting **outer-product** updates (forming a matrix by multiplying a column vector by a row vector). The name comes from the delta rule: the update is proportional to the *error* `(v - S^T * k)`, not just the raw value. This makes the state self-correcting — if the state doesn't already contain information similar to `v`, it gets added with high weight.

**Per-timestep algorithm for each V-head `h`:**

```
1. Decay: S[h] *= exp(ssm_a[h] * softplus(alpha[h] + dt_bias[h]))
   - ssm_a is negative → decay < 1 → state exponentially forgets

2. Delta update:
   sk[vi] = sum_ki(S[h, vi, ki] * k[ki])    // project state onto current key
   delta[vi] = beta[h] * (v[vi] - sk[vi])    // error signal
   S[h, vi, ki] += k[ki] * delta[vi]         // outer product update

3. Output:
   out[vi] = sum_ki(S[h, vi, ki] * q[ki]) / sqrt(head_k_dim)
```

**GQA in DeltaNet:** Head mapping is format-dependent: GGUF uses **tiling** (`kh = h % num_k_heads`, matching `ggml_repeat` semantics), while SafeTensors uses **interleaved grouping** (`kh = h * num_k_heads / num_v_heads`). Controlled by the `kqv_order` flag set at model load time.

**Split order:** After conv1d, output splits as `[Q | K | V]` (llama.cpp convention).

**Gating:** After recurrence, output goes through per-head RMS norm, then is multiplied by `SiLU(z)` from a separate gate projection.

## Mamba-2 (Nemotron-H)

[Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) learns input-dependent **discretization** (choosing how much time passes between updates) — the `dt` (timestep, delta-time) is computed from the input, making the model selectively remember or forget.

**Per-head recurrence:**

```
dt_h = softplus(dt_raw[h] + dt_bias[h])     // input-dependent timestep
decay = exp(ssm_a[h] * dt_h)                // decay < 1

For each state element [i, j]:
  S[h][i][j] = decay * S[h][i][j] + x[i] * dt_h * B[j]   // state update
  y[i]       = sum_j(S[h][i][j] * C[j]) + D[h] * x[i]    // output + skip
```

**Key differences from DeltaNet:**

- **B/C are input-dependent projections** (**selectivity** — the model can choose what to remember based on the current input, not just a fixed decay pattern)
- **D skip connection** adds a direct path from input to output
- **Group structure**: B and C are shared within head groups
- **Group RMS norm** on output (not per-head)

## Hybrid Layer Patterns

| Model | Pattern | Rule |
| :--- | :--- | :--- |
| Qwen3.5 | DeltaNet + Attention | Attention every 4th layer |
| Nemotron-H | Mamba-2 + Attention | Detected at init via tensor presence |
| Nemotron-Nano | SSM + MoE + Attention | 52-layer pattern: M=SSM, E=MoE, *=Attention |
| GPT-OSS | Sliding + Full attention | Even = 128-token window, odd = full sequence |

Layer types are determined at init from model **metadata** (descriptive information about the model structure — layer counts, dimensions, patterns — stored in the model file header) and dispatched in each model's `forward()` loop.

---

**In the code:** [src/ops/ssm.zig](../../src/ops/ssm.zig) (causalConv1dSilu, mamba2Recurrence, groupRmsNormSiluGate), [src/backend/kernels/cpu/deltanet.zig](../../src/backend/kernels/cpu/deltanet.zig) (DeltaNet recurrence), [src/models/qwen35.zig](../../src/models/qwen35.zig) (hybrid dispatch)

**Math reference:** [Convolution (1D Causal)](appendix-math.md#convolution-1d-causal), [Outer Product](appendix-math.md#outer-product), [Softplus](appendix-math.md#softplus)

**Next:** [Chapter 7: Sampling →](07-sampling.md) | **Back:** [Chapter 5: Memory and Caching ←](05-memory-and-caching.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)
