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

**What are Q, K, V?** They're three different **linear projections** (matrix-vector multiplies) of the same input hidden state `x`:

```
Q = W_q @ x    (Query: "What am I looking for?")
K = W_k @ x    (Key: "What do I contain?")
V = W_v @ x    (Value: "What information do I carry?")
```

Each token produces its own Q, K, and V by multiplying `x` by three different learned weight matrices. These projections transform the hidden state into three different "views" that serve different roles in the attention mechanism.

The attention score between positions i and j is `Q_i · K_j / sqrt(d)`. After **softmax** normalization (converts raw scores into probabilities that sum to 1.0), these scores weight the V vectors:

```
output = softmax(Q @ K^T / sqrt(d)) @ V
       where K^T = transpose of K (flip rows and columns)
             sqrt(d) = scale factor = 1/sqrt(head_dim)
```

This is **O(n²)** in sequence length (computational complexity grows quadratically with the number of tokens) — every token attends to every previous token.

### GQA (Grouped Query Attention)

Attention is computed **in parallel** (all heads compute simultaneously, not one after another) across multiple **heads** (independent attention mechanisms, each focusing on different aspects of the input). GQA reduces memory by sharing K/V heads across multiple Q heads. With 20 Q heads and 5 KV heads, each KV head serves 4 Q heads, cutting KV cache memory by 4×.

| Model | Q heads | KV heads | Ratio |
| :--- | :--- | :--- | :--- |
| Gemma3 1B | 4 | 1 | 4:1 |
| Qwen3.5 | 16 | 4 | 4:1 |
| GPT-OSS | 64 | 8 | 8:1 |
| Nemotron-H | 40 | 8 | 5:1 |

**MLA (Multi-head Latent Attention)** goes further — it compresses K/V into a **low-rank latent space** (a smaller intermediate representation with fewer dimensions) before caching, reducing memory even more. Used by GLM-4.

### SDPA (Scaled Dot-Product Attention)

SDPA is the core attention computation, extracted into a shared **kernel** (a single computational function that runs on the CPU or GPU) (`src/ops/attention.zig`):

```
SDPA(Q, K, V, scale) = softmax(Q @ K^T * scale) @ V
```

The implementation handles KV cache append, GQA head mapping, sliding window, attention sinks, and KV cache quantization — all dispatched to the active backend.

**FlashAttention** is an optimization that computes attention in **tiles** (small rectangular blocks of the attention matrix processed one at a time) using **online softmax** (incrementally updating the softmax result as new tiles arrive, avoiding the need to store all scores at once), never **materializing** (allocating memory for and storing) the full scores matrix. Metal and CUDA backends implement FlashAttention-2; the CPU backend uses a **SIMD-vectorized** (using Single Instruction Multiple Data — processing multiple values at once with one CPU instruction) **fallback** (alternative implementation used when the primary method isn't available).

### Attention Variants

**Per-Head QK Normalization** (Gemma3, Qwen3.5): RMS-normalizes Q and K per head before computing scores, stabilizing attention regardless of embedding **magnitude** (the size/scale of the values — how large the numbers are).

**Sliding Window** (GPT-OSS): Even layers attend only to the most recent 128 tokens. Odd layers attend to the full sequence. This halves KV cache cost while maintaining global context through **alternation** (switching back and forth between limited and full attention across layers).

**Attention Sinks** (GPT-OSS): A learned per-head **scalar** (single number, not a vector) logit **prepended** (added to the beginning) to attention scores. Acts as a "sink" that absorbs excess probability, preventing **over-concentration** (too much attention weight) on early positions.

**Sigmoid Gate** (Qwen3.5): After SDPA, output is gated **element-wise** (applied independently to each element, not as a matrix operation) by `sigmoid(gate)`, giving learned per-element control over how much attention output reaches the **residual stream** (the main path through the model where outputs accumulate via residual connections `output = input + sublayer(input)`).

**Logit Softcapping** (Gemma3): `tanh(logits / cap) * cap` — **soft-clamps** (gently constrains via a smooth curve, unlike hard clamping which abruptly cuts off) final logits to `[-cap, +cap]`, preventing extreme values while **preserving relative ordering** (keeping the same rank order — if A > B before, then A > B after).

## RoPE (Rotary Position Encoding)

Transformers are **position-agnostic** by default (they don't know the order of tokens) — without position information, "the cat sat" and "sat the cat" look identical. Earlier models added absolute position embeddings (e.g., "this is position 5"), but RoPE encodes position through **rotation** because it has a key geometric property: **the angle difference between two rotated vectors depends only on their relative distance, not their absolute positions**.

When we rotate Q at position `i` by angle `θ_i` and K at position `j` by angle `θ_j`, their dot product includes a term `cos(θ_i - θ_j)`. Since angles are proportional to position (`θ = pos × freq`), the difference `θ_i - θ_j = (i - j) × freq` captures the *relative* distance `(i - j)` between tokens. This means attention naturally focuses on how far apart tokens are, not where they appear absolutely — which is what matters for language ("the cat" should attend the same way whether it's at the start or middle of a sentence).

**How it works:** RoPE rotates Q and K vectors in 2D planes using standard **rotation matrices** (mathematical transformations that rotate vectors by an angle without changing their length):

```
freq[i] = 1 / (theta ^ (2i / rope_dim))
angle   = pos * freq[i]

x'[i]        = x[i] * cos(angle) - x[i + half] * sin(angle)
x'[i + half] = x[i] * sin(angle) + x[i + half] * cos(angle)
```

Each pair of dimensions `[i, i+rope_dim/2]` forms a 2D plane rotated by `angle`. Different planes use different frequencies (lower dimensions rotate faster, higher dimensions rotate slower), giving the model a range of **"wavelengths"** (cycles per distance — like how light has different wavelengths for different colors) to detect patterns at different distances.

Higher theta values produce lower-frequency rotations for better long-range discrimination (allowing the model to handle longer sequences):

| Model | theta | Effect |
| :--- | :--- | :--- |
| Nemotron-H | 10,000 | Standard range |
| GPT-OSS | 150,000 | Extended **context** (context = maximum sequence length the model can process) |
| Gemma3 | 1,000,000 | Very long context |
| Qwen3.5 | 10,000,000 | Ultra-long context |

**Partial RoPE**: Some models (Qwen3.5, Nemotron-H) only rotate a subset of dimensions (e.g., first 78 out of 128), leaving the rest for non-positional features.

## RMS Normalization

RMSNorm stabilizes the forward pass by normalizing each vector to **unit RMS** (Root Mean Square — scaling so the average squared value equals 1):

```
rmsNorm(x, weight, eps) = x / sqrt(mean(x²) + eps) * weight
       where eps = epsilon, a tiny constant (e.g., 1e-6) to prevent division by zero
```

Unlike **LayerNorm** (an older normalization method that also subtracts the mean), RMSNorm has no mean subtraction — simpler and empirically just as effective. Every layer applies RMSNorm **before** attention and before FFN (**pre-norm** — normalizing the input to each sublayer). Some models add **post-norms** (normalizing the output after the sublayer, as in Gemma3) or per-head QK norms (Gemma3, Qwen3.5).

**L2 Normalization** is unit-norm without **learnable weights** (parameters that the model adjusts during training — L2 norm just scales to unit length, doesn't multiply by learned values): `x[i] /= sqrt(sum(x²) + eps)`. Used by **DeltaNet** (a linear-complexity alternative to attention covered in [Chapter 6](06-state-space-models.md#deltanet-qwen35)) to normalize Q and K before the recurrence.

---

**In the code:** `src/ops/attention.zig` (SDPA), `src/backend/kernels/cpu/rope.zig` (RoPE), `src/backend/kernels/cpu/norm.zig` (RMSNorm, L2Norm), `src/backend/kernels/cpu/sdpa.zig` (CPU FlashAttention)

**Math reference:** [Q/K/V projections](appendix-math.md#qkv-projections), [Attention scores](appendix-math.md#attention-score-computation), [Dot product](appendix-math.md#dot-product), [Softmax](appendix-math.md#softmax), [RMSNorm](appendix-math.md#rms-normalization-rmsnorm), [L2 norm](appendix-math.md#l2-normalization)

**Next:** [Chapter 3: Feed-Forward Networks →](03-feed-forward-networks.md)
