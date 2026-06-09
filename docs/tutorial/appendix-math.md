# Appendix: Mathematical Operations Reference

A quick reference for the core mathematical operations used in LLM inference. Written for systems programmers — think of these as the "library functions" that get called thousands of times per token.

## Vector and Matrix Operations

### Dot Product

Multiply corresponding elements and sum:

```
dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]
          = sum_i(a[i] * b[i])
```

**Usage**: Core of attention scores (`Q · K`), computing similarity between vectors.

**Performance**: O(n), bandwidth-bound (reads 2n values, writes 1 scalar).

### Matrix-Vector Multiply (GEMV)

Each output element is a dot product of a matrix row with the input vector. The weight matrix is streamed from memory one row at a time, and each row produces one output scalar.

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
    X["Input Vector x\n[k floats]"]
    W["Weight Matrix W\n[n rows × k cols]"]

    subgraph Row0["Row 0 → output y[0]"]
        D0["dot(W[0], x)"]
    end
    subgraph Row1["Row 1 → output y[1]"]
        D1["dot(W[1], x)"]
    end
    subgraph RowN["Row n-1 → output y[n-1]"]
        DN["dot(W[n-1], x)"]
    end

    W --> D0
    W --> D1
    W --> DN
    X --> D0
    X --> D1
    X --> DN

    D0 --> Y["Output Vector y\n[n floats]"]
    D1 --> Y
    DN --> Y
```

```
y[i] = sum_j(W[i][j] * x[j])

Example (3×4 matrix × 4-element vector):
[w00 w01 w02 w03]   [x0]   [w00*x0 + w01*x1 + w02*x2 + w03*x3]
[w10 w11 w12 w13] × [x1] = [w10*x0 + w11*x1 + w12*x2 + w13*x3]
[w20 w21 w22 w23]   [x2]   [w20*x0 + w21*x1 + w22*x2 + w23*x3]
                    [x3]
```

**Usage**: Every linear projection in the model (Q/K/V projections, FFN layers, output logits).

**Performance**: O(n×k) multiply-accumulates. For decode (generating one token at a time), this is ~95% of inference time. Memory-bandwidth bound — reading the weight matrix is the bottleneck.

### Outer Product

Forms a matrix from two vectors (column × row):

```
A[i][j] = a[i] * b[j]

Example (3-element × 4-element):
[a0]              [a0*b0  a0*b1  a0*b2  a0*b3]
[a1] × [b0 b1 b2 b3] = [a1*b0  a1*b1  a1*b2  a1*b3]
[a2]              [a2*b0  a2*b1  a2*b2  a2*b3]
```

**Usage**: DeltaNet state updates (`S += k ⊗ v` where `⊗` is outer product).

**Performance**: O(n×m), produces an n×m matrix from two vectors.

## Attention-Specific Operations

### Q/K/V Projections

**What they are**: Three different linear transformations (matrix-vector multiplies) of the same input hidden state, using three different learned weight matrices.

```
Given input hidden state x (e.g., 2048-dimensional vector):

Q = W_q @ x    (Query projection)
K = W_k @ x    (Key projection)
V = W_v @ x    (Value projection)

Each weight matrix transforms x into a different representation:
- W_q: [n_heads × head_dim, hidden_dim] → produces Query
- W_k: [n_kv_heads × head_dim, hidden_dim] → produces Key
- W_v: [n_kv_heads × head_dim, hidden_dim] → produces Value
```

**Example** (simplified, single head, hidden_dim=4, head_dim=3):
```
x = [1.0, 2.0, 3.0, 4.0]

W_q = [[0.1, 0.2, 0.3, 0.4],      Q = W_q @ x = [3.0,
       [0.5, 0.6, 0.7, 0.8],  →                  7.0,
       [0.9, 1.0, 1.1, 1.2]]                     11.0]

W_k = [[0.2, 0.3, 0.4, 0.5],      K = W_k @ x = [4.0,
       [0.6, 0.7, 0.8, 0.9],  →                  8.0,
       [1.0, 1.1, 1.2, 1.3]]                     12.0]

W_v = [[0.3, 0.4, 0.5, 0.6],      V = W_v @ x = [5.0,
       [0.7, 0.8, 0.9, 1.0],  →                  9.0,
       [1.1, 1.2, 1.3, 1.4]]                     13.0]
```

**Why three different projections?**

- **Query (Q)**: Represents "what this token is looking for" in other tokens
- **Key (K)**: Represents "what this token offers" to be matched against queries
- **Value (V)**: Represents "the actual information this token carries"

The Q and K projections are used to compute **attention scores** (how much each token should attend to each other token). The V projection contains the actual information that gets mixed based on those scores.

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
    subgraph Current["Current token (e.g. 'cat')"]
        Q["Query Q\nWhat am I looking for?\ne.g. 'adjectives nearby'"]
    end

    subgraph Context["All tokens in context"]
        K1["Key K1 'fluffy'\nWhat do I offer?"]
        K2["Key K2 'sat'\nWhat do I offer?"]
        K3["Key K3 'mat'\nWhat do I offer?"]
    end

    Q -->|"dot product → score"| K1
    Q -->|"dot product → score"| K2
    Q -->|"dot product → score"| K3

    K1 -->|"high score → high weight"| Softmax["softmax(scores)\nattention weights"]
    K2 -->|"low score → low weight"| Softmax
    K3 -->|"low score → low weight"| Softmax

    subgraph Values["Values carry the content"]
        V1["Value V1 'fluffy'"]
        V2["Value V2 'sat'"]
        V3["Value V3 'mat'"]
    end

    Softmax -->|"weight × value"| V1
    Softmax -->|"weight × value"| V2
    Softmax -->|"weight × value"| V3

    V1 --> Output["Output = weighted mix\n(mostly 'fluffy' info)"]
    V2 --> Output
    V3 --> Output
```

### Attention Score Computation

Once we have Q and K, we compute similarity scores via dot products:

```
For token i attending to token j:
score[i][j] = (Q[i] · K[j]) / sqrt(head_dim)

Example (continuing from above, head_dim=3):
Q[0] = [3.0, 7.0, 11.0]
K[0] = [4.0, 8.0, 12.0]

score = (3.0×4.0 + 7.0×8.0 + 11.0×12.0) / sqrt(3)
      = (12 + 56 + 132) / 1.732
      = 200 / 1.732
      ≈ 115.5
```

The division by `sqrt(head_dim)` (called **scaled** dot-product attention) prevents scores from growing too large as head_dim increases, which would make softmax too peaked.

**Full attention mechanism**:
```
1. Compute Q, K, V for all tokens
2. Compute scores: S[i][j] = (Q[i] · K[j]) / sqrt(head_dim)
3. Apply softmax per row: weights[i][j] = softmax(S[i])
4. Weighted sum of values: output[i] = sum_j(weights[i][j] × V[j])
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
flowchart LR
    HiddenState["Hidden State x\n[hidden_dim floats]"]

    HiddenState --> Wq["W_q projection\nGEMV"]
    HiddenState --> Wk["W_k projection\nGEMV"]
    HiddenState --> Wv["W_v projection\nGEMV"]

    Wq --> Q["Query Q\n[n_heads × head_dim]"]
    Wk --> K["Key K\n(stored in KV cache)"]
    Wv --> V["Value V\n(stored in KV cache)"]

    subgraph Scores["Attention Score Computation"]
        Q --> Dot["Q · Kᵀ\n(dot products)"]
        K --> Dot
        Dot --> Scale["÷ sqrt(head_dim)\n(prevents saturation)"]
        Scale --> SM["softmax per row\n(convert to weights 0→1)"]
    end

    SM --> WeightedSum["Weighted sum\nweights × V"]
    V --> WeightedSum
    WeightedSum --> Out["Attention Output\n[hidden_dim floats]"]
```

**Multi-head attention**: Repeat this process with different W_q, W_k, W_v matrices for each head, concatenate outputs.

### Convolution (1D Causal)

Sliding window that combines nearby values using learned weights. "Causal" means it only looks backward (at past inputs):

```
y[t] = w[0]*x[t] + w[1]*x[t-1] + w[2]*x[t-2] + ... + w[k-1]*x[t-k+1]
     = sum_i(w[i] * x[t-i])   for i in 0..kernel_size
```

**Example** (kernel_size=3, weights=[0.5, 0.3, 0.2]):
```
x = [a, b, c, d, e]
y[0] = 0.5*a
y[1] = 0.5*b + 0.3*a
y[2] = 0.5*c + 0.3*b + 0.2*a
y[3] = 0.5*d + 0.3*c + 0.2*b
y[4] = 0.5*e + 0.3*d + 0.2*c
```

**Usage**: DeltaNet and Mamba-2 preprocessing — mixes information from recent time steps before the recurrence.

**Implementation**: Ring buffer stores last k-1 inputs to avoid shifting arrays.

## Normalization Operations

### Softmax

Converts raw scores into probabilities that sum to 1.0:

```
softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
```

**Example**:
```
Input:  [2.0, 1.0, 0.1]
Exp:    [7.39, 2.72, 1.11]    (sum = 11.21)
Output: [0.66, 0.24, 0.10]    (sum = 1.00)
```

**Usage**: Attention weights (convert attention scores to probabilities), MoE routing (select experts), sampling (convert logits to token probabilities).

**Numerical stability trick**: Subtract max before exp to prevent overflow. The diagram below shows why the naive path fails and what the stable path does instead.

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
    Input["Raw scores x\ne.g. [1000, 1001, 999]"]

    Input --> Naive["Naive: exp(x) directly"]
    Naive --> Overflow["exp(1000) = Inf\noverflow — unusable"]

    Input --> FindMax["Find max(x) = 1001"]
    FindMax --> Shift["Subtract max\nx_shifted = [-1, 0, -2]"]
    Shift --> ExpSafe["exp(x_shifted)\n= [0.368, 1.0, 0.135]"]
    ExpSafe --> Sum["sum = 1.503"]
    Sum --> Divide["Divide each by sum"]
    Divide --> Probs["Probabilities\n[0.245, 0.665, 0.090]\nsum = 1.0"]

    style Overflow fill:#ffcccc,color:#333
    style Probs fill:#ccffcc,color:#333
```

### RMS Normalization (RMSNorm)

Scale vector to unit RMS (Root Mean Square), then apply learned weights:

```
rms = sqrt(mean(x²) + eps)
rmsNorm(x, w) = (x / rms) * w
```

**Example** (eps=1e-6, weight=[1.0, 1.0, 1.0]):
```
x = [2.0, 4.0, 4.0]
mean(x²) = (4 + 16 + 16) / 3 = 12
rms = sqrt(12 + 1e-6) ≈ 3.464
output = [2.0/3.464, 4.0/3.464, 4.0/3.464] * w
       ≈ [0.577, 1.155, 1.155]
```

**Usage**: Applied before every attention and FFN sublayer. Stabilizes training and inference by preventing activation magnitudes from exploding.

**Why RMS not mean**: Simpler (no mean subtraction), empirically just as effective as LayerNorm.

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
    Input["Input vector x\ne.g. [2.0, 4.0, 4.0]"]

    subgraph Pass1["Pass 1: Compute RMS"]
        Sq["Square each element\nx² = [4, 16, 16]"]
        Mean["Mean of squares\nmean(x²) = 12.0"]
        Eps["Add epsilon for stability\n12.0 + 1e-6"]
        Sqrt["Square root\nrms = sqrt(12.000001) ≈ 3.464"]
        Sq --> Mean --> Eps --> Sqrt
    end

    subgraph Pass2["Pass 2: Normalize and Scale"]
        Div["Divide each element by rms\n[2/3.464, 4/3.464, 4/3.464]\n= [0.577, 1.155, 1.155]"]
        Scale["Multiply by learned weights w\n(per-element scale, trained)"]
        Out["Normalized output\nunit RMS, scaled by w"]
        Div --> Scale --> Out
    end

    Input --> Pass1
    Sqrt -->|"rms value"| Pass2
    Input -->|"original x"| Pass2
```

### L2 Normalization

Scale vector to unit length (magnitude = 1), no learnable weights:

```
magnitude = sqrt(sum(x[i]²) + eps)
l2Norm(x) = x / magnitude
```

**Usage**: DeltaNet Q/K normalization before recurrence (prevents numerical instability in the state update).

## Activation Functions

Non-linear transformations applied element-wise (independently to each value).

### SiLU (Swish)

Smooth activation with gating property:

```
silu(x) = x * sigmoid(x)
        = x / (1 + exp(-x))
```

**Shape**: Smooth S-curve that's negative for x<0, close to linear for x>3.

**Usage**: Most FFN layers (SwiGLU), causal convolution, SSM gating.

### GELU (Gaussian Error Linear Unit)

Smoother than ReLU, approximates Gaussian CDF:

```
gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
```

**Usage**: Gemma3 FFN (instead of SiLU).

### Sigmoid

Maps any value to (0, 1):

```
sigmoid(x) = 1 / (1 + exp(-x))
```

**Output range**: (0, 1) — never exactly 0 or 1.

**Usage**: Gating (how much signal to let through), MoE routing (GLM-4), attention gate (Qwen3.5).

### Softplus

Smooth approximation of ReLU, always positive:

```
softplus(x) = log(1 + exp(x))
```

**Approximation**: Linear for x > 20, `≈ exp(x)` for x < -20.

**Usage**: SSM `dt` (timestep) computation — ensures positive decay factors.

### Tanh (Hyperbolic Tangent)

Maps any value to (-1, 1):

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        = 2*sigmoid(2*x) - 1
```

**Usage**: Logit softcapping in Gemma3 (`tanh(x/cap) * cap` clamps to ±cap smoothly).

### Activation Function Comparison

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
    subgraph Bounded["Bounded outputs (0,1) or (-1,1)"]
        Sigmoid["sigmoid(x)\nRange: (0, 1)\nFormula: 1/(1+exp(-x))\nUse: gating, routing"]
        Tanh["tanh(x)\nRange: (-1, 1)\nFormula: 2·sigmoid(2x)-1\nUse: softcapping logits"]
    end

    subgraph Unbounded["Unbounded outputs — pass large values through"]
        SiLU["SiLU / Swish\nRange: (-0.28, ∞)\nFormula: x·sigmoid(x)\nUse: SwiGLU FFN layers"]
        GELU["GELU\nRange: (-0.17, ∞)\nFormula: 0.5x·(1+tanh(...))\nUse: Gemma3 FFN layers"]
    end

    subgraph AlwaysPos["Always positive"]
        Softplus["softplus(x)\nRange: (0, ∞)\nFormula: log(1+exp(x))\nUse: SSM timestep dt"]
    end

    Input["Raw activation value x\n(any real number)"] --> Bounded
    Input --> Unbounded
    Input --> AlwaysPos

    Sigmoid -->|"small x → ~0\nlarge x → ~1"| GateUse["Controls how much\nsignal passes through"]
    Tanh -->|"large x → ±1\n(saturates smoothly)"| ClampUse["Soft clamp — prevents\nlogit explosion"]
    SiLU -->|"x < 0 → small neg\nx > 0 → ~linear"| FFNUse["Gate × up projection\nin SwiGLU FFN"]
    GELU -->|"similar to SiLU\nslightly smoother"| GELUUse["Drop-in for SiLU\nin Gemma3"]
    Softplus -->|"always > 0\nno negative outputs"| DtUse["Ensures dt > 0\nfor stable SSM decay"]
```

## Sampling Operations

### Argmax

Find the index of the maximum value:

```
argmax(x) = index i where x[i] is largest

Example:
x = [0.1, 0.8, 0.3, 0.5]
argmax(x) = 1    (x[1] = 0.8 is largest)
```

**Usage**: Greedy decoding (temperature=0) — always pick the highest-scoring token.

**Implementation**: Two-pass linear scan, O(n). First pass finds the maximum value (SIMD-vectorised); second pass finds its index.

### Temperature Scaling

Scale logits to control randomness before softmax:

```
adjusted_logits = logits / temperature

temperature → 0:   peaked distribution (greedy)
temperature = 1:   unchanged
temperature → ∞:   uniform distribution
```

**Effect**: Lower temp → more deterministic (top token dominates). Higher temp → more random (flatter probabilities).

### Top-K Selection

Keep only the K highest-scoring tokens, set rest to -∞:

```
1. Scan all tokens once to find the k-th largest value (min-replacement scan, O(n))
2. In a single SIMD pass: mask tokens below that threshold to −∞ and apply exp
3. Renormalize by dividing by the accumulated sum
```

**Usage**: Prevent sampling extremely unlikely tokens at high temperatures.

### Top-P (Nucleus Sampling)

Keep smallest set of tokens whose cumulative probability ≥ P:

```
1. Sort tokens by probability descending
2. Accumulate probabilities until sum ≥ P
3. Keep those tokens, mask rest
4. Renormalize
```

**Adaptive**: When model is confident (one token = 90%), keeps 1-2 tokens. When uncertain (many similar scores), keeps dozens.

### Sampling Pipeline

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
    Logits["Raw logits\n[vocab_size floats]\ne.g. 128,000 values"]

    Logits --> TempCheck{"temperature\n= 0?"}

    TempCheck -->|"yes — greedy"| Argmax["argmax(logits)\npick highest score directly"]
    Argmax --> Token["Next token ID"]

    TempCheck -->|"no — sample"| TempScale["Divide by temperature\nlogits / T\nlower T → sharper, higher T → flatter"]

    TempScale --> TopK{"top_k\nenabled?"}
    TopK -->|"yes"| KFilter["Find k-th largest value\nmask all below threshold to -inf\n(O(n) single pass)"]
    TopK -->|"no"| TopP

    KFilter --> TopP{"top_p\nenabled?"}
    TopP -->|"yes"| Softmax2["softmax → cumulative probs\ndrop tokens past nucleus threshold P\nrenormalize remaining"]
    TopP -->|"no"| Softmax1["softmax\nconvert all logits to probs"]

    Softmax2 --> Sample["Weighted random sample\nfrom remaining distribution"]
    Softmax1 --> Sample

    Sample --> Token

    style Token fill:#ccffcc,color:#333
    style Argmax fill:#e8f0fe,color:#1a1a2e
```

## Special Operations

### Reduction Operations

**Sum**: `sum(x) = x[0] + x[1] + ... + x[n-1]`

**Mean**: `mean(x) = sum(x) / n`

**Max**: `max(x)` = largest element

**Usage**: Building blocks for softmax (sum/max), normalization (mean), reductions in GPU kernels.

**GPU implementation**: Parallel reduction — each thread reduces a chunk, then combine results in shared memory using tree reduction.

### Element-wise Operations

Applied independently to each element:

```
add(a, b)[i] = a[i] + b[i]
mul(a, b)[i] = a[i] * b[i]
```

**Usage**: Residual connections (`x = x + f(x)`), gating (`output = data * gate`).

**Performance**: Memory-bandwidth bound (2 reads + 1 write per element), trivially parallel.

---

## Common Patterns

### GEMV dominates inference

Matrix-vector multiply is ~95% of decode time. Every linear layer (`Linear(in, out)`) is a GEMV:

- Q/K/V projections: 3 GEMVs per layer
- Attention output projection: 1 GEMV per layer
- FFN: 3 GEMVs per layer (gate, up, down)
- Output logits: 1 GEMV (largest — vocab_size rows)

A 28-layer model with vocab_size=128K does ~197 GEMVs per token.

### Bandwidth vs Compute Bound

**Bandwidth-bound** (GEMV, normalization, activations): Time spent waiting for memory reads/writes dominates. Arithmetic is trivial. Quantization helps enormously (4× less data to read).

**Compute-bound** (attention for long sequences, matrix-matrix multiply during prefill): Arithmetic dominates. GPU compute power matters more than memory speed.

For single-token decode, everything is bandwidth-bound.

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
    subgraph BW["Bandwidth-bound operations\n(memory speed is the bottleneck)"]
        GEMV2["GEMV — decode\nReads weight matrix row by row\nArithmetic intensity: ~0.25 FLOP/byte\nBottleneck: DRAM bandwidth"]
        Norm["RMSNorm / LayerNorm\nReads vector, writes vector\nArithmetic intensity: ~2 FLOP/byte\nBottleneck: memory round-trips"]
        Activation["Activation functions\n(SiLU, GELU, sigmoid)\nElement-wise, trivial math\nBottleneck: reading/writing the tensor"]
        ElemWise["Element-wise ops\n(add, mul, residual)\nOne pass over data\nBottleneck: memory bandwidth"]
    end

    subgraph Compute["Compute-bound operations\n(ALU utilization is the bottleneck)"]
        Prefill["GEMM — prefill\nMatrix × matrix (all tokens at once)\nArithmetic intensity: ~O(seq_len) FLOP/byte\nBottleneck: GPU TFLOPS"]
        LongAttn["Attention — long sequences\nO(seq_len²) dot products per head\nArithmetic intensity grows with seq_len\nBottleneck: GPU TFLOPS"]
    end

    Opt1["Optimization levers\nfor bandwidth-bound:\n• Quantization (4-bit → 4× less data)\n• Kernel fusion (fewer passes)\n• Larger batch size"] --- BW
    Opt2["Optimization levers\nfor compute-bound:\n• FlashAttention (tiled SRAM)\n• Tensor parallelism\n• Higher-TFLOPS GPU"] --- Compute
```

### In-place vs Allocating

**In-place** (modifies input): `rope(x)` rotates x directly. Zero allocations.

**Allocating** (creates output): `softmax(x)` operates **in-place** over two passes (find max, then exp+normalize). No allocation — the input buffer is reused as output.

Inference hot path is allocation-free — all buffers pre-allocated, operations reuse scratch space.

---

**See also**:
- Chapter 2 (attention, RoPE, RMSNorm)
- Chapter 3 (activation functions, MoE)
- Chapter 4 (GEMV, quantization)
- Chapter 6 (convolution, outer product, SSM recurrence)
- Chapter 7 (sampling operations)

**Next:** [Appendix: Compile-Time Optimization →](appendix-compile-time.md) | **Back:** [Chapter 17: Speculative Decoding ←](17-speculative-decoding.md)
