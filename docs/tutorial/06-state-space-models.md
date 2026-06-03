# Chapter 6: State Space Models

SSMs are a family of sequence models based on state-space theory. [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) introduced **selective** state spaces — input-dependent parameters that give SSMs content-aware reasoning ability. SSMs are an alternative to attention that process tokens in **O(1) with respect to sequence length** per step (constant time — doesn't grow with the number of previous tokens) instead of O(n²). Instead of re-reading all previous tokens, they maintain a fixed-size **state matrix** that summarizes the past:

```
state[t] = decay * state[t-1] + input[t]    (simplified)
output[t] = state[t] @ query[t]
```

The **decay** factor controls how quickly old information fades — like a leaky bucket where new information flows in and old information gradually drains out.

**What does the state matrix actually store?** Think of it as a compressed lookup table mapping keys to values. After seeing the sentence "The capital of France is Paris", the state contains an approximate mapping from the key-direction for "capital of France" to the value-direction for "Paris". When a new query asks something related (e.g., "What is the capital of France?"), the output `state @ query` retrieves the stored value — approximately, because the matrix has fixed size and multiple associations overlap. Older or weaker associations decay away as new ones are written. This is fundamentally different from attention, which stores every K/V explicitly and retrieves them exactly.

**Concrete example** — a tiny 2×2 state matrix tracking two tokens:

```
State starts empty:     S = [[0, 0],
                              [0, 0]]

Token 1: k=[1, 0], v=[0.8, 0.2], decay=0.9
  After decay:          S = 0.9 * S = [[0, 0], [0, 0]]   (nothing to decay)
  Error:                delta = v - S^T @ k = [0.8, 0.2] - [0, 0] = [0.8, 0.2]
  Update:               S += outer(delta, k) = [[0.8, 0], [0.2, 0]]
  Retrieve with q=[1,0]: output = S @ q = [0.8, 0.2]  ✓ recovers v₁

Token 2: k=[0, 1], v=[0.3, 0.9], decay=0.9
  After decay:          S = 0.9 * S = [[0.72, 0], [0.18, 0]]
  Error:                delta = v - S^T @ k = [0.3, 0.9] - [0, 0] = [0.3, 0.9]
  Update:               S += outer(delta, k) = [[0.72, 0.3], [0.18, 0.9]]
  Retrieve with q=[1,0]: output = S @ q = [0.72, 0.18]  ≈ decayed v₁
  Retrieve with q=[0,1]: output = S @ q = [0.3, 0.9]    ✓ recovers v₂

Token 1's information has decayed by 10%. After 100 more tokens, it would
be multiplied by 0.9^100 ≈ 0.00003 — effectively gone. This is the
fundamental tradeoff: constant memory, but lossy recall.
```

**Hybrid models** combine attention and SSM layers: attention every N layers for global context, SSM for the rest for speed.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    subgraph Attention["Transformer Attention (O(n) per token)"]
        direction TB
        KVCache["KV Cache\n(all past tokens)"]
        QVec["Query vector\n(current token)"]
        Scores["Dot products\nwith every key"]
        AttnOut["Weighted sum\nof all values"]
        KVCache --> Scores
        QVec --> Scores
        Scores --> AttnOut
    end

    subgraph SSM["SSM Recurrence (O(1) per token)"]
        direction TB
        State["State matrix S\n(fixed size, ~1,024 KB)"]
        NewTok["New token\nx[t]"]
        Decay["Decay old state\nS *= exp(a * dt)"]
        Update["Write new info\nS += outer(v, k)"]
        Read["Read via query\nout = S @ q"]
        State --> Decay
        NewTok --> Decay
        Decay --> Update
        Update --> Read
    end

    Past100K["100K past tokens"] --> KVCache
    Past100K -. "compressed into" .-> State


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

DeltaNet builds on the delta rule for associative memory, first explored in the context of linear transformers by [Schlag et al. (2021)](https://arxiv.org/abs/2102.11174) and developed into the DeltaNet architecture by [Yang et al. (2024)](https://arxiv.org/abs/2406.06484).

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

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart TD
    Input["Input token x[t]"] --> QKV["Q / K / V projections"]
    QKV --> Q["Query q\n(what to retrieve)"]
    QKV --> K["Key k\n(what to write)"]
    QKV --> V["Value v\n(what to store)"]

    State["State matrix S\n(accumulated memory)"] --> Decay["Step 1: Decay\nS *= exp(a * softplus(alpha))"]
    Decay --> SK["sk = S^T @ k\n(what state already knows about k)"]
    V --> Delta["Step 2: Error signal\ndelta = beta * (v - sk)"]
    SK --> Delta
    Delta --> Update["Step 3: Outer product update\nS += outer(delta, k)\n(correct the state)"]
    Update --> State
    Update --> Output["Step 4: Output\nout = S @ q / sqrt(k_dim)"]
    Q --> Output
    Output --> Gate["Multiply by SiLU(z) gate"]


**GQA in DeltaNet:** GQA head mapping uses tiling (`kh = h % num_k_heads`) for both GGUF and SafeTensors formats (`kqv_order` is always false for Qwen3.5).

**Split order:** After conv1d, output splits as `[Q | K | V]` (llama.cpp convention).

**Gating:** After recurrence, output goes through per-head RMS norm, then is multiplied by `SiLU(z)` from a separate gate projection.

## Mamba-2 (Nemotron-H)

[Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) learns input-dependent **discretization** (choosing how much time passes between updates) — the `dt` (timestep, delta-time) is computed from the input, making the model selectively remember or forget.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    Input["Input x[t]"] --> DT["dt projection\n(learned, per-head)"]
    Input --> B["B projection\n(what to write)"]
    Input --> C["C projection\n(what to read)"]
    Input --> D["D skip\n(direct passthrough)"]

    DT --> Decay["decay = exp(ssm_a * dt)\n(input-dependent forget rate)"]
    State["State S[h]\n(fixed-size memory)"] --> DecayState["Decay: S *= decay"]
    Decay --> DecayState
    B --> WriteIn["Write: S += x * dt * B^T\n(add new info)"]
    DecayState --> WriteIn
    WriteIn --> State

    C --> ReadOut["Read: y = S @ C\n(query the state)"]
    WriteIn --> ReadOut
    D --> Skip["y += D * x\n(skip connection)"]
    ReadOut --> Skip
    Skip --> Output["Output y[t]"]


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

## Why SSMs are Faster

The core difference: attention re-reads all previous tokens every time, SSMs update a fixed-size summary.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart TD
    subgraph TransformerMem["Transformer memory — grows with context"]
        direction LR
        T1K["Token 1\nK, V stored"] --- T2K["Token 2\nK, V stored"] --- TdotK["..."] --- T32KK["Token 32,000\nK, V stored"]
        QNew1["Query (new token)"] --> ScanAll["Scan ALL 32,000 entries\n= 2M multiply-adds / head"]
    end

    subgraph SSMMem["SSM memory — always the same size"]
        direction LR
        SMatrix["State matrix S\n128 × 128 = 16,384 floats\n(1,024 KB for 16 heads)"]
        QNew2["New token"] --> UpdateS["One state update\n~32,768 multiply-adds / head"]
        UpdateS --> SMatrix
        SMatrix --> UpdateS
    end

    Past["100K past tokens"] --> TransformerMem
    Past -. "compressed into fixed box" .-> SSMMem


| Aspect | Transformer Attention | SSM Recurrence |
|--------|----------------------|----------------|
| Memory per token | O(n) — stores all K/V vectors | O(1) — fixed state matrix |
| Compute per token | O(n) — dot product with all keys | O(d²) — state update (constant) |
| At 100K tokens | 100K dot products per head | Same as at 100 tokens |
| Long-range memory | Exact — every past token accessible | Lossy — old information decays |

**Concrete cost comparison** — generating token 32,001 in a model with 16 heads, head_dim=64:

```
Attention layer:
  Per head: Q (64 floats) dot-product with 32,000 cached K vectors
  = 32,000 × 64 = 2,048,000 multiply-adds per head
  × 16 heads = 32.8M multiply-adds
  KV cache read: 32,000 × 128 × 16 × 2 bytes ≈ 125 MB scanned

SSM layer:
  Per head: decay state (128×128 = 16,384 muls), outer product update (16,384 muls)
  = ~32,768 multiply-adds per head
  × 16 heads = 524K multiply-adds
  State read: 128 × 128 × 16 × 4 bytes = 1,024 KB

Ratio: attention does 250× more work at 32K context.
         At 128K context, it's 1000× more.
```

The tradeoff: SSMs are faster but lose exact long-range recall. The state matrix has fixed size (128×128 = 16,384 floats per head), so it acts as a lossy compression of all past tokens — like a 1,024 KB "summary" trying to represent 125 MB of cached history. If the model saw "The capital of France is" 10,000 tokens ago, the relevant information has been multiplied by decay^10,000 and is effectively gone. Attention doesn't have this problem — it stores every K/V and can look them up exactly, at the cost of scanning all of them every token.

Hybrid models get the best of both: SSM layers for speed on most positions, attention layers every Nth layer for precise long-range access. Qwen3.5 uses attention every 4th layer — 48 of its 64 layers are cheap SSM layers, and 16 are full-attention layers that maintain exact recall. The attention layers act as "checkpoints" that periodically refresh the model's access to the full history.

## State Matrix Visualization

For DeltaNet with `head_v_dim=128` and `head_k_dim=128`:

```
State S[h]: 128×128 matrix = 16,384 floats per head
           ┌─────────────────────────┐
    v_dim  │ Accumulated K→V mapping │ 128 rows
     ↓     │ via outer product       │
           │ updates with decay      │
           └─────────────────────────┘
                   k_dim → 128 cols

Each timestep:
  1. Decay entire matrix by exp(a * softplus(alpha))
  2. Compute error: delta = beta * (v - S^T @ k)
  3. Update: S += outer(delta, k)
  4. Output: o = S @ q / sqrt(k_dim)
```

Total state per layer: `num_v_heads × v_dim × k_dim × 4 bytes`. For Qwen3.5 0.8B: 16 heads × 128 × 128 × 4 = 1,024 KB per SSM layer — negligible vs KV cache.

## Hardware Considerations

SSM recurrence is **inherently sequential** — each timestep depends on the previous state. This limits parallelism:

- **Prefill**: Cannot batch SSM layers across tokens (unlike attention with GEMM). Each token must be processed sequentially through SSM layers.
- **Decode**: SSM layers are fast (one state update per token) — the bottleneck shifts to attention layers.
- **GPU dispatch**: GPU backends (Metal, Vulkan, WebGPU, ROCm) run the full DeltaNet recurrence on the GPU. The CUDA backend falls back to the CPU SIMD kernel (V8-vectorized, not scalar). The state update loop is sequential across v-heads, not memory-bound.

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
