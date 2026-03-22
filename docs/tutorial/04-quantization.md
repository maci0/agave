# Chapter 4: Quantization

Model weights are trained in float32 (32 bits per value) but stored compressed for inference. **Quantization** maps floating-point values to lower-precision representations, trading a small amount of accuracy for massive memory and speed gains.

## Why Quantize?

A 7B parameter model in float32 needs 28 GB of memory. In 4-bit quantization, it needs ~4 GB — small enough to fit in a laptop's GPU memory. Inference is almost always **memory-bandwidth bound** (the bottleneck is reading weights from RAM/VRAM, not arithmetic operations), so smaller weights = faster inference.

## Block Quantization

**Q4_0, Q8_0** (GGUF-style): Groups of 32 values share a single scale factor. Each value is stored as a small integer, dequantized on-the-fly:

```
float_value = integer_value * scale
```

**Super-block formats** (Q4_K, Q5_K, Q6_K): Groups of 256 values with hierarchical scales — a block scale plus per-sub-block adjustments.

## MLX Affine Quantization

Used by Apple MLX models (Gemma QAT 4-bit, GLM-4 6-bit). Each group of 64 values has a scale and bias:

```
float_value = scale * uint_value + bias
```

Scales and biases are stored as bf16 in companion tensors.

## Floating-Point Quantization

**FP8 E4M3/E5M2**: 8-bit IEEE-like floats. E4M3 has more mantissa precision; E5M2 has wider dynamic range. Used for KV cache quantization and weight storage.

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

For a 2560×2560 matrix, that's 6.5M multiply-accumulates per call, and a typical model does ~210 GEMVs per token. Agave has separate kernels per dtype because each quantization format has completely different bit layouts.

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
