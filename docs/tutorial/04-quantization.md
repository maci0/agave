# Chapter 4: Quantization

Model weights are trained in float32 (32 bits per value) but stored compressed for inference. **Quantization** maps floating-point values to lower-**precision** (fewer bits per number, less accurate but smaller) representations, trading a small amount of accuracy for massive memory and speed gains.

## Why Quantize?

A 7B parameter model in float32 needs 28 GB of memory. In 4-bit quantization, it needs ~4 GB — small enough to fit in a laptop's GPU memory. Inference is almost always **memory-bandwidth bound** (the bottleneck is reading weights from RAM/VRAM, not arithmetic operations), so smaller weights = faster inference.

## Block Quantization

**Q4_0, Q8_0** (GGUF-style): Groups of 32 values share a single **scale factor** (a multiplier that converts small integers back to approximate float values). Each value is stored as a small integer, dequantized on-the-fly:

```
float_value = integer_value * scale
```

**Super-block formats** (Q4_K, Q5_K, Q6_K): Groups of 256 values with **hierarchical scales** (multiple levels of scale factors — a coarse scale for the whole block, then fine-grained adjustments per sub-block) — a block scale plus per-sub-block adjustments.

## MLX Affine Quantization

Used by Apple MLX models (Gemma QAT 4-bit, GLM-4 6-bit). Each group of 64 values has a scale and bias:

```
float_value = scale * uint_value + bias
```

Scales and biases are stored as bf16 in **companion tensors** (separate tensors with matching names like `weight.scales` and `weight.biases` that store per-group quantization parameters).

### MLX Memory Layout

**Quantized weights:** Packed into `u32` words (8 nibbles per word for 4-bit, 12 words per group for 6-bit):

```
4-bit: 64 elements × 4 bits = 256 bits = 8 u32 words
6-bit: 64 elements × 6 bits = 384 bits = 12 u32 words
8-bit: 64 elements × 8 bits = 512 bits = 16 u32 words
```

**Example:** 4-bit group `[0, 1, 2, ..., 63]` is packed as:

```
word[0] = elem[0..7]   (8 nibbles, low-to-high)
word[1] = elem[8..15]
...
word[7] = elem[56..63]
```

Nibble extraction:

```zig
const word_idx = elem_idx / 8;
const bit_offset = (elem_idx % 8) * 4;
const nibble = (words[word_idx] >> bit_offset) & 0xF;
```

**Scales and biases:** Separate bf16 arrays (2 bytes per value):

```
scales[group] → bf16 scale for group
biases[group] → bf16 bias for group
```

### Factored Dequantization (30-40% Speedup)

**Naive approach:** Dequantize each element before multiplying:

```zig
for (0..64) |j| {  // For each element in group
    const q = unpack(quant, j);        // Extract quantized value
    const dq = scale * q + bias;       // Dequantize
    acc += dq * x[j];                  // Multiply by input
}
```

**Cost:** 64 multiplies (scale × q) + 64 adds (+ bias) + 64 FMAs (dq × x) = **192 operations per group**.

**Optimized approach:** Factor out the scale and bias using algebra:

```
sum(x[j] * (scale * q[j] + bias)) = scale * sum(x[j] * q[j]) + bias * sum(x[j])
```

This is the **distributive property** — pull the constant scale and bias outside the sum:

```zig
var q_dot: f32 = 0;  // dot(quantized, input)
var x_sum: f32 = 0;  // sum(input)

for (0..64) |j| {
    const q = unpack(quant, j);
    q_dot += q * x[j];  // Accumulate q·x
    x_sum += x[j];      // Accumulate sum(x)
}

acc += scale * q_dot + bias * x_sum;  // Apply scale/bias ONCE
```

**Cost:** 64 multiplies (q × x) + 64 adds (accumulate) + 64 adds (sum x) + **2 final ops** = **130 operations per group**.

**Savings:** 192 → 130 ops = **32% reduction** in arithmetic. Real-world speedup: **30-40%** (measured on Apple M4 with Gemma3 27B QAT).

**Why this works:**

- Scale and bias are **constant per group** (same for all 64 elements)
- We can compute the dot product `sum(q × x)` and sum `sum(x)` separately
- Then apply scale and bias **once** at the end

**SIMD implementation** (from `src/ops/mlx.zig`):

```zig
var q_dot_acc: V8 = @splat(0.0);
var x_sum_acc: V8 = @splat(0.0);

var j: usize = 0;
while (j + 8 <= 64) : (j += 8) {
    // Unpack 8 quantized values
    const qv = unpackU4x8(quant, j);  // V8 of quantized values

    // Load 8 input values
    const xv: V8 = x[base + j ..][0..8].*;

    // FMA: q_dot += qv * xv
    q_dot_acc = @mulAdd(V8, qv, xv, q_dot_acc);

    // Accumulate x sum
    x_sum_acc += xv;
}

// Horizontal reduce
const q_dot = @reduce(.Add, q_dot_acc);
const x_sum = @reduce(.Add, x_sum_acc);

// Apply scale/bias once
acc += scale * q_dot + bias * x_sum;
```

**Additional optimization:** `@mulAdd` maps to NEON `vfma` (fused multiply-add) — 1 instruction instead of separate multiply + add.

### When to Use MLX Quantization

**Advantages:**

- **Better quality** than integer quantization at the same bit width (affine transform vs simple scaling)
- **Native Apple Silicon support** — MLX models load directly on Metal without conversion
- **Flexible bit widths** — 4-bit, 6-bit, 8-bit (GGUF typically only 4-bit or 8-bit)

**Disadvantages:**

- **Format compatibility** — only MLX and Agave support it (not llama.cpp, vLLM, etc.)
- **Larger metadata** — scales + biases = 2× overhead vs scale-only (4 bytes vs 2 bytes per group)
- **6-bit GPU support** — Metal kernel exists but not in all backends

**Recommended for:**

- Apple Silicon users with MLX-quantized models (Gemma3 QAT, GLM-4)
- Quality-sensitive workloads (affine has less quantization error than Q4_0)

**Not recommended for:**

- Cross-platform deployment (GGUF Q4_K has wider support)
- Extreme compression (Q2_K, IQ4_XS are smaller)

## Floating-Point Quantization

Unlike integer quantization (Q4_0, Q8_0), floating-point quantization keeps the exponential representation, just with fewer bits.

### FP8 E4M3 (4-bit exponent, 3-bit mantissa)

**Bit layout**: `[sign:1][exponent:4][mantissa:3]`

```
Example: 7.5 in FP8 E4M3
Binary:  0 1001 110
         │  │    └─ mantissa (0.875)
         │  └────── exponent (bias-adjusted = 2)
         └───────── sign (positive)

Value = (-1)^0 × 1.875 × 2^2 = 7.5
(mantissa 110 = 0.5 + 0.25 + 0.125 = 0.875; 1 + 0.875 = 1.875)
```

**Range**: Can represent values from ~6×10⁻⁸ to 448 (with subnormals)

**Why E4M3 for weights?**
- **High precision near zero**: 3 mantissa bits give 8 distinct values in each power-of-2 range
- **Good for small gradients**: Weight updates during training are often tiny
- **Balanced range**: 448 max is enough for most normalized weights

**Trade-off vs FP16**:

- FP16 (E5M10): ±65,504 range, 1024× more precision
- FP8 E4M3: ±448 range, but 2× smaller memory

### FP8 E5M2 (5-bit exponent, 2-bit mantissa)

**Bit layout**: `[sign:1][exponent:5][mantissa:2]`

**Range**: ~6×10⁻⁸ to 57,344 (128× wider than E4M3)

**Why E5M2 for KV cache?**

- **Wider range**: Attention activations can have large outliers
- **Less precision needed**: Small errors in K/V don't **cascade** (compound/multiply through many operations, unlike weight errors which affect every computation) (unlike weights)
- **Better for activations**: **Dynamic range** (the span from smallest to largest representable value) matters more than precision

**Practical usage in Agave**:

- **E4M3**: Weight quantization, gradient accumulation
- **E4M3**: KV cache quantization (default: q8_0-K/turbo4-V, FP8 option available via `--kv-type fp8_e4m3`)
- **int8**: Alternative to FP8 for KV cache (simpler, slightly less accurate)

### Why FP8 instead of int8?

**int8 with scale**: `float_value = int8_value × scale`

- Simple, fast dequantization (one multiply)
- Fixed precision across the range (8 bits = 256 levels)
- Works well for roughly uniform distributions

**FP8 (E4M3 or E5M2)**: `[sign][exponent][mantissa]`

- **Adaptive precision** (more bits near zero, fewer bits for large values — precision varies based on magnitude)
- Natural for values spanning many **orders of magnitude** (factors of 10 — e.g., from 0.001 to 1000)
- **Hardware-accelerated** (dedicated silicon on the chip for fast execution) on modern GPUs (H100, A100, MI300)

**When to use each**:

- int8: Uniform distributions (e.g., quantized weights after normalization)
- FP8 E4M3: Weights and gradients with small deltas
- FP8 E5M2: Activations with wide dynamic range

**NVFP4, MXFP4**: 4-bit microscaled floating-point. 16-element blocks with FP8 scales. Hardware-native on NVIDIA Blackwell and newer.

## TurboQuant — KV Cache Quantization

TurboQuant ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) is a KV cache-specific quantization method that achieves 3.6-6.4x compression vs f16 with minimal quality loss. Unlike weight quantization formats (Q4_0, Q8_0), TurboQuant is applied at runtime to the KV cache during inference.

### How It Works

Traditional KV cache quantization (q8_0, fp8) applies simple per-block scaling. TurboQuant adds a **Walsh-Hadamard Transform (WHT)** preprocessing step that gaussianizes the distribution, then uses an optimal **Lloyd-Max codebook** for scalar quantization:

```
Quantize:  normalize → WHT → codebook lookup → pack
Dequantize: unpack → codebook → inverse WHT → rescale
```

The WHT is a deterministic rotation (like a Fourier transform but with only additions and subtractions) that makes each coordinate approximately N(0,1), regardless of the original distribution. This lets us use a single fixed codebook — no per-model calibration needed.

### Format Family

| Format | Bits/elem | Compression vs f16 | PPL impact |
|--------|-----------|-------------------|-----------|
| turbo4 | 4.5 | 3.6x | +0.23% |
| turbo3 | 3.5 | 4.6x | +1.06% |
| turbo2 | 2.5 | 6.4x | +6.48% |

### Asymmetric K/V

A key insight, also explored in [KIVI (Liu et al., 2024)](https://arxiv.org/abs/2402.02750): **V compression is nearly free** — all quality degradation comes from K compression. Agave supports independent K/V cache types:

```bash
# Best quality: high-precision K + compressed V
./agave model.gguf --cache-type-k q8_0 --cache-type-v turbo4 "prompt"

# Maximum compression
./agave model.gguf --cache-type-k turbo3 --cache-type-v turbo2 "prompt"

# Symmetric (shorthand via --kv-type)
./agave model.gguf --kv-type turbo4 "prompt"
```

### Block Layout

Each 32-element block stores: `[f16 norm (2B)] [packed codebook indices]`

```
turbo4: 2B norm + 16B nibbles = 18B per 32 elements
turbo3: 2B norm + 12B packed 3-bit = 14B per 32 elements
turbo2: 2B norm + 8B packed 2-bit = 10B per 32 elements
```

### When to Use TurboQuant

**Recommended for:**
- Long-context inference (32K+ tokens) where KV cache dominates memory
- Serving multiple concurrent requests (KV cache × batch size)
- Memory-constrained devices (laptops, phones)

**Not recommended for:**
- Short prompts where KV cache is small relative to weights
- Latency-critical applications (WHT adds ~7-10% decode overhead)

**Stacking with KV eviction:** TurboQuant compresses the *bits per KV entry*, while KV cache eviction (`--kv-eviction`) reduces the *number of entries*. Combined, they can achieve ~40x KV memory reduction vs f16 baseline. See [Chapter 5: Memory and Caching](05-memory-and-caching.md#kv-cache-eviction) for eviction details.

## TurboQuant+ — The `turbo` Preset

The `--kv-type turbo` preset is a curated configuration that combines three TurboQuant optimizations:

```bash
./agave model.gguf --kv-type turbo "prompt"
# Equivalent to: --kv-type-k q8_0 --kv-type-v turbo4
# Plus: boundary V protection (first/last 2 layers at f16)
```

### Why K Precision Matters More Than V

Keys (K) control **attention routing** — they determine which positions receive weight via the softmax. Small errors in K shift the softmax distribution, causing the model to attend to the wrong tokens. Values (V) are just the payload — they're weighted-summed after softmax, so small errors average out.

Empirically, compressing K below q8_0 causes measurable perplexity degradation, while V can be compressed to turbo4 (4.5 bits) with no measurable quality loss. The `turbo` preset exploits this asymmetry: high-precision K (q8_0, 8.5 bits) + aggressive V compression (turbo4, 4.5 bits).

### Boundary V Protection

The first and last 2 transformer layers keep V at f16 even when the middle layers use turbo4. These boundary layers are disproportionately important — early layers establish token representations, and final layers directly influence the output distribution. The `turbo` preset enables this automatically:

```zig
// src/models/gemma4.zig — per-layer V type selection
inline fn layerVType(self: *const Gemma4Model, li: u32) KvQuantType {
    if (self.kv_boundary_v == 0) return self.kv_type_v;
    const b = self.kv_boundary_v;
    if (li < b or li >= self.n_layers - b) return .f16;
    return self.kv_type_v;
}
```

For a 42-layer model with `--kv-type turbo`: layers 0-1 and 40-41 use f16 V, layers 2-39 use turbo4 V. All layers use q8_0 K.

### Sparse V Dequantization

During attention, most softmax weights are near zero — only a handful of positions actually contribute to the output. Sparse V skips the V dequantization and multiply-accumulate for any position where the softmax weight is below 1e-6 (contributing less than 0.0001% to the output):

```zig
// src/ops/attention.zig
const sparse_v_threshold: f32 = 1e-6;

for (0..win_len) |wi| {
    const score = scores[score_offset + wi];
    if (score < sparse_v_threshold) continue; // Skip negligible positions
    const t = win_start + wi;
    kv_quant.kvMulAccum(attn_out + q_base, score, kv_values[v_off..].ptr, hd, kv_type_v);
}
```

At 32K context length, the majority of positions have negligible softmax weights. Skipping their V reads yields **+22.8% decode speed** with zero measured perplexity impact. This is especially effective with quantized V formats (turbo4, turbo3) because it avoids both the dequantization arithmetic and the cache-unfriendly memory reads.

## Key Principle

Dequantization happens *inside* the GEMV kernel, not before it. This avoids materializing the full-precision weight matrix:

```
// BAD: dequantize entire matrix, then multiply
f32_weights = dequantize(q4_weights)    // allocates vocab_size × n_embd × 4 bytes
y = f32_weights @ x

// GOOD: dequantize per-block inside the dot product loop
for each row i:
    for each block b:
        scale = weights.scale[b]
        for j in block:
            y[i] += (weights.quant[j] * scale) * x[j]
```

## GEMV (General Matrix-Vector Multiply)

GEMV is the dominant operation — ~95% of inference compute time. Every linear projection is a GEMV: `y[i] = sum_j(W[i][j] * x[j])`.

For a 2560×2560 matrix, that's 6.5M **multiply-accumulates** (multiply two numbers and add the result to a running sum — the core operation in matrix math) per call, and a typical model does ~210 GEMVs per token. Agave has separate kernels per **dtype** (data type — f32, bf16, q4_0, etc.) because each quantization format has completely different bit layouts.

(For the full mathematical definition with examples, see [Math Reference: GEMV](appendix-math.md#matrix-vector-multiply-gemv))

## Choosing a Format

| Use Case | Recommended | Rationale |
|----------|-------------|-----------|
| Balanced quality/speed | bf16, Q4_K | Industry standard, wide support |
| Maximum compression | Q2_K, IQ4_XS | Smallest memory footprint |
| CPU inference | IQ4_NL, Q4_0, Q5_K | Optimized SIMD kernels |
| GPU with limited VRAM | Q4_K, FP8 E4M3 | Good quality/size tradeoff |
| KV cache (default) | q8_0-K + turbo4-V | Zero quality loss, 2x compression |
| KV cache (max compress) | turbo3, turbo4 | 3.6-4.6x compression, ~1% PPL |
| KV cache (max quality) | f16, FP8 E4M3 | Fast decode, no transform overhead |
| Reference accuracy | f32 | Full precision |

**Quality hierarchy:** `f32 > bf16 > FP8 > Q6_K > Q5_K > Q4_K > Q4_0 > IQ4_NL > Q3_K > Q2_K`

---

**In the code:** `src/ops/quant.zig` (dequantization helpers), `src/ops/mlx.zig` (MLX format), `src/backend/kernels/cpu/gemv*.zig` (per-format GEMV kernels)

**Next:** [Chapter 5: Memory and Caching →](05-memory-and-caching.md)
