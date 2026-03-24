# Chapter 3: Feed-Forward Networks

The **FFN (Feed-Forward Network)** is the second **sublayer** (component within a transformer layer) in each transformer layer. "Feed-forward" means data flows in one direction through the network — input → hidden layer → output, with no loops or **recurrence** (unlike **RNNs** — Recurrent Neural Networks — which cycle back on themselves, feeding outputs back as inputs).

While attention lets tokens communicate with each other, the FFN processes each position **independently** — it's a separate computation per token that doesn't look at neighboring tokens. This is where most of the model's "knowledge" lives — the learned weights in these projections encode facts, patterns, and transformations.

## SwiGLU

The standard FFN structure in modern transformers:

```
FFN(x) = down_proj(activation(gate_proj(x)) * up_proj(x))
```

Three matrix multiplies per FFN call, expanding to a larger **intermediate dimension** (the expanded size between projections, typically 4-8× the hidden size) and projecting back. The **activation function** is a **nonlinear** transformation (output is not proportional to input — e.g., sigmoid curves, not straight lines) applied element-wise (e.g., SiLU, GELU).

**SwiGLU** uses **SiLU** (Sigmoid Linear Unit, also called Swish) as the activation — hence the name (Swish-Gated Linear Unit).

## Activation Functions

| Function | Formula | Used by |
| :--- | :--- | :--- |
| **SiLU/Swish** | `x * sigmoid(x)` = `x / (1 + exp(-x))` | Most FFN layers, conv1d, SSM gating |
| **GELU** | `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x³)))` | Gemma3 FFN |
| **Softplus** | `log(1 + exp(x))`, linear for x>20 | SSM dt computation |
| **Sigmoid** | `1 / (1 + exp(-x))` | DeltaNet beta, attention gate, MoE routing |
| **ReLU²** | `max(0, x)²` | Nemotron-Nano MoE FFN |

**Clamped SwiGLU** (GPT-OSS MoE): Adds **hard clamping** (forcing values to stay within fixed bounds) `[-7.0, +7.0]` to prevent **overflow** (values becoming too large to represent, causing errors or infinity) during **mixed-precision** (using different bit widths for different operations — e.g., 16-bit for some, 32-bit for others) expert computation.

## Mixture of Experts (MoE)

Standard transformers use the same FFN weights for every token. MoE models have multiple FFN "experts" and a **router** (a learned selection mechanism that scores and picks which experts should process each token) that selects which ones to use:

```
1. Router: scores = sigmoid(hidden @ gate_weight)     # score each expert
2. Select: top_k = top-4 experts by score             # pick best K
3. Normalize: weights = softmax(top_k_scores)         # normalize selected
4. Compute: output = Σ weight[i] * expert_i(hidden)   # weighted sum (each expert's output multiplied by its weight, then added together)
5. Shared: output += shared_expert(hidden)             # always-active (if present)
```

This gives the **capacity** (total model size/knowledge) of a large model (30B total parameters) with the compute cost of a small one (3B active per token).

Expert selection uses **stack-allocated** arrays (fixed-size buffers on the call stack, automatically freed when the function returns) — zero **heap allocation** (dynamic memory from the system allocator, requires explicit free) in the hot path.

| Model | Routed Experts | Top-K | Shared Expert | Routing |
| :--- | :--- | :--- | :--- | :--- |
| GPT-OSS | 32 | 4 | No | Softmax |
| GLM-4 | varies | varies | No | Sigmoid (independent gates) |
| Nemotron-Nano | 128 | 6 | Yes (1, 2× hidden dim) | Softmax |

**Sigmoid routing** (GLM-4): Each expert gate is **independent** (evaluated separately, not competing with each other for probability mass like softmax does) — multiple experts can have high activation simultaneously without competing.

**Shared expert** (Nemotron-Nano): One expert is always active regardless of router output, providing a stable **baseline** (consistent minimum contribution that all tokens receive, ensuring basic functionality).

---

**In the code:** `src/backend/kernels/cpu/activation.zig` (SiLU, GELU), `src/ops/math.zig` (softplus, sigmoid, topKExperts), `src/models/gpt_oss.zig` (MoE implementation)

**Math reference:** [SiLU](appendix-math.md#silu-swish), [GELU](appendix-math.md#gelu-gaussian-error-linear-unit), [Sigmoid](appendix-math.md#sigmoid), [Softplus](appendix-math.md#softplus)

**Next:** [Chapter 4: Quantization →](04-quantization.md)
