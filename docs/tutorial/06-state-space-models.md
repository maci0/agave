# Chapter 6: State Space Models

**Prerequisites:** [Chapter 2: The Transformer](02-the-transformer.md), [Chapter 5: Memory and Caching](05-memory-and-caching.md) (both helpful, not required)

**Time:** ~20 min

> After this chapter you can explain DeltaNet and Mamba-2 recurrences, how hybrid attention/SSM models work, and why SSMs are O(1) per step.

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

### Code Flow

```text
attention step: score(q, every cached k) -> softmax -> weighted sum over all v    # O(n) per token
ssm step:       state = decay(state) + write(k, v); out = read(state, q)         # O(1) per token
```

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Attention["Transformer Attention (O(n) per token)"]
        direction TB
        KVCache["KV Cache\n(all past tokens)"]:::setup
        QVec["Query vector\n(current token)"]:::setup
        Scores["Dot products\nwith every key"]:::sync
        AttnOut["Weighted sum\nof all values"]:::success
        KVCache --> Scores
        QVec --> Scores
        Scores --> AttnOut
    end

    subgraph SSM["SSM Recurrence (O(1) per token)"]
        direction TB
        State["State matrix S\n(fixed size, ~1,024 KB)"]:::setup
        NewTok["New token\nx[t]"]:::setup
        Decay["Decay old state\nS *= exp(a * dt)"]:::migration
        Update["Write new info\nS += outer(v, k)"]:::sync
        Read["Read via query\nout = S @ q"]:::success
        State --> Decay
        NewTok --> Decay
        Decay --> Update
        Update --> Read
    end

    Past100K["100K past tokens"]:::setup --> KVCache
    Past100K -. "compressed into" .-> State

    AttnOut -.->|"O(n) work / token"| Compare["Same past information,\ndifferent access strategy"]:::optional
    Read -.->|"O(1) work / token"| Compare
```

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

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Weights["Learned conv weights (d_conv = 4)"]
        direction LR
        W0["w[0]\noldest"]:::setup --- W1["w[1]"]:::setup --- W2["w[2]"]:::setup --- W3["w[3]\nnewest"]:::setup
    end

    subgraph RingBuf["Ring buffer — circular, fixed allocation"]
        direction LR
        B0["buf[t-3]\nslot 0"]:::setup --- B1["buf[t-2]\nslot 1"]:::setup --- B2["buf[t-1]\nslot 2"]:::setup --- B3["x[t]\nslot 3 (new)"]:::setup
    end

    subgraph Multiply["Element-wise multiply and sum"]
        direction LR
        P0["w[0] × buf[t-3]"]:::sync --> Sum["Σ = conv_out[t]"]:::migration
        P1["w[1] × buf[t-2]"]:::sync --> Sum
        P2["w[2] × buf[t-1]"]:::sync --> Sum
        P3["w[3] × x[t]"]:::sync --> Sum
    end

    subgraph Advance["Next step — oldest slot overwritten"]
        direction LR
        A0["buf[t-2]\nslot 0"]:::migration --- A1["buf[t-1]\nslot 1"]:::migration --- A2["x[t]\nslot 2"]:::migration --- A3["x[t+1]\nslot 3 (new)"]:::migration
    end

    Weights --> Multiply
    RingBuf --> Multiply
    Sum -->|"SiLU(conv_out[t])"| Out["fused output"]:::success
    RingBuf -->|"head pointer\nadvances by 1"| Advance
```

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
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Input["Input token x[t]"]:::setup --> QKV["Q / K / V projections"]:::sync
    QKV --> Q["Query q\n(what to retrieve)"]:::setup
    QKV --> K["Key k\n(what to write)"]:::setup
    QKV --> V["Value v\n(what to store)"]:::setup

    State["State matrix S\n(accumulated memory)"]:::setup --> Decay["Step 1: Decay\nS *= exp(a * softplus(alpha))"]:::migration
    Decay --> SK["sk = S^T @ k\n(what state already knows about k)"]:::sync
    V --> Delta["Step 2: Error signal\ndelta = beta * (v - sk)"]:::migration
    SK --> Delta
    Delta --> Update["Step 3: Outer product update\nS += outer(delta, k)\n(correct the state)"]:::sync
    Update --> State
    Update --> Output["Step 4: Output\nout = S @ q / sqrt(k_dim)"]:::success
    Q --> Output
    Output --> Gate["Multiply by SiLU(z) gate"]:::optional
```

**GQA in DeltaNet:** GQA head mapping uses tiling (`kh = h % num_k_heads`) for both GGUF and SafeTensors formats (`kqv_order` is always false for Qwen3.5).

**Split order:** After conv1d, output splits as `[Q | K | V]` (llama.cpp convention).

**Gating:** After recurrence, output goes through per-head RMS norm, then is multiplied by `SiLU(z)` from a separate gate projection.

## Mamba-2 (Nemotron-H)

[Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) learns input-dependent **discretization** (choosing how much time passes between updates) — the `dt` (timestep, delta-time) is computed from the input, making the model selectively remember or forget.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Input["Input x[t]"]:::setup --> DT["dt projection\n(learned, per-head)"]:::setup
    Input --> B["B projection\n(what to write)"]:::setup
    Input --> C["C projection\n(what to read)"]:::setup
    Input --> D["D skip\n(direct passthrough)"]:::optional

    DT --> Decay["decay = exp(ssm_a * dt)\n(input-dependent forget rate)"]:::migration
    State["State S[h]\n(fixed-size memory)"]:::setup --> DecayState["Decay: S *= decay"]:::migration
    Decay --> DecayState
    B --> WriteIn["Write: S += x * dt * B^T\n(add new info)"]:::sync
    DecayState --> WriteIn
    WriteIn --> State

    C --> ReadOut["Read: y = S @ C\n(query the state)"]:::sync
    WriteIn --> ReadOut
    D --> Skip["y += D * x\n(skip connection)"]:::optional
    ReadOut --> Skip
    Skip --> Output["Output y[t]"]:::success
```

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
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph TransformerMem["Transformer memory — grows with context"]
        direction LR
        T1K["Token 1\nK, V stored"]:::setup --- T2K["Token 2\nK, V stored"]:::setup --- TdotK["..."] --- T32KK["Token 32,000\nK, V stored"]:::setup
        QNew1["Query (new token)"]:::setup --> ScanAll["Scan ALL 32,000 entries\n= 2M multiply-adds / head"]:::sync
    end

    subgraph SSMMem["SSM memory — always the same size"]
        direction LR
        SMatrix["State matrix S\n128 × 128 = 16,384 floats\n(1,024 KB for 16 heads)"]:::setup
        QNew2["New token"]:::setup --> UpdateS["One state update\n~32,768 multiply-adds / head"]:::sync
        UpdateS --> SMatrix
        SMatrix --> UpdateS
    end

    Past["100K past tokens"]:::setup --> TransformerMem
    Past -. "compressed into fixed box" .-> SSMMem
```

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

Hybrid models get the best of both: SSM layers for speed on most positions, attention layers every Nth layer for precise long-range access. The published Qwen3.5 configuration uses attention every 4th layer — 48 of its 64 layers are cheap SSM layers, and 16 are full-attention layers that maintain exact recall. The attention layers act as "checkpoints" that periodically refresh the model's access to the full history. (The code detects the layer pattern dynamically from tensor presence via `layer_is_deltanet[i] = f.layerTensor(i, "attn_qkv.weight") != null`, so other Qwen3.5 variants with different layer counts or attention ratios work automatically.)

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
- **GPU dispatch**: GPU backends (Metal, Vulkan, WebGPU, ROCm) run the full DeltaNet recurrence on the GPU. The CUDA backend falls back to the CPU SIMD kernel (V8-vectorized, not scalar). The state update runs one head per thread (parallelized across v-heads) and is compute-bound, not memory-bound.

## Hybrid Layer Patterns

The published Qwen3.5 model places a full-attention layer every 4th layer across its 64-layer stack. The remaining 48 layers are DeltaNet SSM layers, making token generation cheap on most layers while preserving exact recall at regular checkpoints. Other variants may use different layer counts — the code auto-detects the pattern from the model file.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Input["Token embedding"]:::setup

    subgraph Block0["Layers 0–3"]
        direction TB
        L0["Layer 0\nDeltaNet SSM"]:::setup --> L1["Layer 1\nDeltaNet SSM"]:::setup --> L2["Layer 2\nDeltaNet SSM"]:::setup --> L3["Layer 3\nFull Attention ✦"]:::migration
    end

    subgraph Block1["Layers 4–7"]
        direction TB
        L4["Layer 4\nDeltaNet SSM"]:::setup --> L5["Layer 5\nDeltaNet SSM"]:::setup --> L6["Layer 6\nDeltaNet SSM"]:::setup --> L7["Layer 7\nFull Attention ✦"]:::migration
    end

    subgraph BlockMid["Layers 8–59  (pattern repeats × 13)"]
        direction TB
        LM["SSM → SSM → SSM → Attention ✦\nevery 4th layer is attention"]:::setup
    end

    subgraph Block60["Layers 60–63"]
        direction TB
        L60["Layer 60\nDeltaNet SSM"]:::setup --> L61["Layer 61\nDeltaNet SSM"]:::setup --> L62["Layer 62\nDeltaNet SSM"]:::setup --> L63["Layer 63\nFull Attention ✦"]:::migration
    end

    Input --> Block0 --> Block1 --> BlockMid --> Block60

    LegSSM["DeltaNet SSM — O(d²) per token\n48 layers — fast state update"]:::setup
    LegATN["Full Attention ✦ — O(n) per token\n16 layers — exact recall checkpoint"]:::migration
```

| Model | Pattern | Rule |
| :--- | :--- | :--- |
| Qwen3.5 | DeltaNet + Attention | Attention every 4th layer |
| Nemotron-H | Mamba-2 + Attention | Detected at init via tensor presence |
| Nemotron-Nano | SSM + MoE + Attention | 52-layer pattern: M=SSM, E=MoE, *=Attention |
| GPT-OSS | Sliding + Full attention | Even = 128-token window, odd = full sequence |

Layer types are determined at init from model **metadata** (descriptive information about the model structure — layer counts, dimensions, patterns — stored in the model file header) and dispatched in each model's `forward()` loop.

**Nemotron-H's Mamba-2 layers** are distinct from Qwen3.5's DeltaNet layers. Where DeltaNet uses the delta rule (error-correcting outer-product updates) for its recurrence, Mamba-2 uses selective-state-space recurrence with causal conv1d and discretized dt (timestep) gating. In the 8B variant (42 layers), Nemotron-H has 21 SSM (Mamba-2) layers on even indices, 4 attention layers at positions 1, 9, 17, 25, and 17 FFN-only layers filling the rest. Layer types are not hardcoded — they're detected at init by probing for tensor presence (`ssm_in.weight` → SSM, `attn_q.weight` → attention, else FFN-only). See [`src/models/nemotron_h.zig`](../../src/models/nemotron_h.zig).

## Gotchas

- **The state matrix's lossy recall isn't a bug you can fix by tuning decay.** Code that treats an SSM layer's state like a KV cache, expecting to retrieve an exact fact from thousands of tokens ago, will get a plausible-looking but wrong answer instead of an error: the association simply decayed below the noise floor of the other associations sharing that fixed-size matrix. If a workload needs exact long-range recall, the fix is a hybrid layer pattern with attention checkpoints (see Hybrid Layer Patterns above), not a smaller decay constant.
- **SSM recurrence can't be batched across tokens the way attention's GEMM can.** Each timestep's state update depends on the previous timestep's state, so prefill can't dispatch one wide matrix multiply across the sequence dimension for SSM layers the way it does for attention (see Hardware Considerations above). Code that tries to parallelize the recurrence loop across tokens instead of across heads will produce a state that never accumulated the intermediate steps, not just a slower kernel.

---

**In the code:** [src/ops/ssm.zig](../../src/ops/ssm.zig) (causalConv1dSilu, mamba2Recurrence, groupRmsNormSiluGate), [src/backend/kernels/cpu/deltanet.zig](../../src/backend/kernels/cpu/deltanet.zig) (DeltaNet recurrence), [src/models/qwen35.zig](../../src/models/qwen35.zig) (hybrid dispatch)

**Math reference:** [Convolution (1D Causal)](appendix-math.md#convolution-1d-causal), [Outer Product](appendix-math.md#outer-product), [Softplus](appendix-math.md#softplus)

**Next:** [Chapter 7: Sampling →](07-sampling.md) | **Back:** [Chapter 5: Memory and Caching ←](05-memory-and-caching.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)

---

## Glossary

**causal convolution** — A sliding-window operation combining nearby values using learned weights, looking only at past positions.

**d_conv** — The width of the causal convolution window (e.g., 4 = current + 3 past inputs).

**decay factor** — A multiplier < 1 applied to the state matrix each step, causing older information to exponentially fade.

**DeltaNet** — A linear-complexity recurrence using the delta rule (error-correcting outer-product updates) to maintain associative memory.

**delta rule** — An update rule where the state correction is proportional to the error between the desired value and what the state already encodes.

**discretization (dt)** — Computing a per-step timestep from the input, controlling how much the state decays and how much new information is written.

**hybrid model** — A model that interleaves attention layers (for exact recall) with SSM layers (for speed) in a single architecture.

**linear-complexity recurrence** — An update loop running in O(d²) per step (constant with respect to sequence length), unlike O(n) attention.

**Mamba-2** — An SSM architecture with input-dependent discretization (dt), allowing the model to selectively remember or forget.

**outer product** — Forming a matrix by multiplying a column vector by a row vector; used to write key-value associations into the state matrix.

**recurrence** — A computation where each step depends on the previous step's output, processing sequentially.

**ring buffer** — A fixed-size circular array where new entries overwrite the oldest, avoiding reallocation.

**SSM (State Space Model)** — A sequence model that maintains a fixed-size state matrix as a compressed summary of all past tokens, updating in O(1) per step.

**state matrix** — A fixed-size matrix (e.g., 128×128 per head) that accumulates key-value associations via outer-product updates with decay.
