# Chapter 2: The Transformer

The forward pass is the core computation: given a token, predict the next one.

```
Token ID → Embedding → N Transformer Layers → Final Norm → Logits → Argmax → Next Token
```

Each **transformer layer** has two sublayers:
1. **Attention** — lets the model look at previous tokens
2. **FFN** (Feed-Forward Network) — processes each position independently

Both use **residual connections** (`output = input + sublayer(input)`) so information flows through unchanged, preventing the **vanishing gradient problem** (where gradients get exponentially smaller in deep networks during training, making learning impossible) in deep networks.

## Attention

Attention answers: "which previous tokens should I pay attention to?"

For each token position, the model computes:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

The attention score between positions i and j is `Q_i · K_j / sqrt(d)`. After **softmax** normalization (converts raw scores into probabilities that sum to 1.0), these scores weight the V vectors:

```
output = softmax(Q @ K^T / sqrt(d)) @ V
```

This is **O(n²)** in sequence length — every token attends to every previous token.

### GQA (Grouped Query Attention)

GQA reduces memory by sharing K/V heads across multiple Q heads. With 20 Q heads and 5 KV heads, each KV head serves 4 Q heads, cutting KV cache memory by 4×.

| Model | Q heads | KV heads | Ratio |
| :--- | :--- | :--- | :--- |
| Gemma3 1B | 4 | 1 | 4:1 |
| Qwen3.5 | 16 | 4 | 4:1 |
| GPT-OSS | 64 | 8 | 8:1 |
| Nemotron-H | 40 | 8 | 5:1 |

**MLA (Multi-head Latent Attention)** goes further — it compresses K/V into a low-rank latent space before caching, reducing memory even more. Used by GLM-4.

### SDPA (Scaled Dot-Product Attention)

SDPA is the core attention computation, extracted into a shared kernel (`src/ops/attention.zig`):

```
SDPA(Q, K, V, scale) = softmax(Q @ K^T * scale) @ V
```

The implementation handles KV cache append, GQA head mapping, sliding window, attention sinks, and KV cache quantization — all dispatched to the active backend.

**FlashAttention** is an optimization that computes attention in tiles using **online softmax** (incrementally updating the softmax result as new tiles arrive, avoiding the need to store all scores at once), never materializing the full scores matrix. Metal and CUDA backends implement FlashAttention-2; the CPU backend uses a SIMD-vectorized fallback.

### Attention Variants

**Per-Head QK Normalization** (Gemma3, Qwen3.5): RMS-normalizes Q and K per head before computing scores, stabilizing attention regardless of embedding magnitude.

**Sliding Window** (GPT-OSS): Even layers attend only to the most recent 128 tokens. Odd layers attend to the full sequence. This halves KV cache cost while maintaining global context through alternation.

**Attention Sinks** (GPT-OSS): A learned per-head scalar logit prepended to attention scores. Acts as a "sink" that absorbs excess probability, preventing over-concentration on early positions.

**Sigmoid Gate** (Qwen3.5): After SDPA, output is gated element-wise by `sigmoid(gate)`, giving learned per-element control over how much attention output reaches the residual stream.

**Logit Softcapping** (Gemma3): `tanh(logits / cap) * cap` — soft-clamps final logits to `[-cap, +cap]`, preventing extreme values while preserving relative ordering.

## RoPE (Rotary Position Encoding)

Transformers are position-agnostic by default. RoPE encodes position by rotating Q and K vectors:

```
freq[i] = 1 / (theta ^ (2i / rope_dim))
angle   = pos * freq[i]

x'[i]        = x[i] * cos(angle) - x[i + half] * sin(angle)
x'[i + half] = x[i] * sin(angle) + x[i + half] * cos(angle)
```

The dot product `Q · K` then depends on *relative* distance, not absolute positions. Higher theta values produce lower-frequency rotations for better long-range discrimination:

| Model | theta | Effect |
| :--- | :--- | :--- |
| Nemotron-H | 10,000 | Standard range |
| GPT-OSS | 150,000 | Extended context |
| Gemma3 | 1,000,000 | Very long context |
| Qwen3.5 | 10,000,000 | Ultra-long context |

**Partial RoPE**: Some models (Qwen3.5, Nemotron-H) only rotate a subset of dimensions, leaving the rest for non-positional features.

## RMS Normalization

RMSNorm stabilizes the forward pass by normalizing each vector to unit RMS:

```
rmsNorm(x, weight, eps) = x / sqrt(mean(x²) + eps) * weight
```

Unlike LayerNorm, it has no mean subtraction — simpler and empirically just as effective. Every layer applies RMSNorm before attention and before FFN (pre-norm). Some models add post-norms (Gemma3) or per-head QK norms (Gemma3, Qwen3.5).

**L2 Normalization** is unit-norm without learnable weights: `x[i] /= sqrt(sum(x²) + eps)`. Used by DeltaNet to normalize Q and K before the recurrence.

---

**In the code:** `src/ops/attention.zig` (SDPA), `src/backend/kernels/cpu/rope.zig` (RoPE), `src/backend/kernels/cpu/norm.zig` (RMSNorm, L2Norm), `src/backend/kernels/cpu/sdpa.zig` (CPU FlashAttention)

**Next:** [Chapter 3: Feed-Forward Networks →](03-feed-forward-networks.md)
