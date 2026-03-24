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

## Floating-Point Quantization

Unlike integer quantization (Q4_0, Q8_0), floating-point quantization keeps the exponential representation, just with fewer bits.

### FP8 E4M3 (4-bit exponent, 3-bit mantissa)

**Bit layout**: `[sign:1][exponent:4][mantissa:3]`

```
Example: 5.75 in FP8 E4M3
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
- **E5M2**: KV cache quantization (default: f16, but FP8 E5M2 option available via `--kv-type fp8_e5m2`)
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
| KV cache | f16, FP8 E5M2 | Fast decode, minimal quality loss |
| Reference accuracy | f32 | Full precision |

**Quality hierarchy:** `f32 > bf16 > FP8 > Q6_K > Q5_K > Q4_K > Q4_0 > IQ4_NL > Q3_K > Q2_K`

---

**In the code:** `src/ops/quant.zig` (dequantization helpers), `src/ops/mlx.zig` (MLX format), `src/backend/kernels/cpu/gemv*.zig` (per-format GEMV kernels)

**Next:** [Chapter 5: Memory and Caching →](05-memory-and-caching.md)
