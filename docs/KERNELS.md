# Kernel Implementation Status

**Last Updated**: 2026-03-22

This document tracks the implementation status of all compute kernels across backends. Each kernel can be:
- **Native**: Fully implemented on the target hardware (GPU shader or optimized CPU SIMD)
- **CPU perf**: Delegates to CPU because CPU is provably faster than GPU dispatch for this op
- **Missing**: No GPU kernel exists — will `@panic` at runtime if called

**Policy**: GPU backends must never silently fall back to CPU. Missing kernels panic with a clear error message. CPU delegation is only permitted where benchmarked to be faster than GPU dispatch overhead (embLookup, tiny softmax).

---

## Compute Operations

| Operation | CPU | Metal | Vulkan | CUDA | ROCm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RMS Norm | Native (SIMD) | Native (fused) | Native (fused) | Native | Native |
| RMS Norm Multi (per-head) | Native | Native | Missing | Missing | Missing |
| L2 Norm | Native (SIMD) | Native | Native (fused) | Native | Native |
| Softmax | Native (SIMD) | Native (3-pass) | Native (fused) | Native | Native |
| SiLU | Native (SIMD) | Native | Native | Native | Native |
| SiLU Mul (fused) | Native | Native | Missing | Missing | Native |
| GELU | Native (SIMD) | Native | Native | Native | Native |
| Add | Native (SIMD) | Native | Native | Native | Native |
| Mul | Native (SIMD) | Native | Native | Native | Native |
| RoPE | Native (SIMD) | Native | Native | Native | Native |
| Sigmoid Mul | Native | Native | Missing | Missing | Native |
| Deinterleave | Native | Native | Missing | Missing | Native |
| Embedding Lookup | Native | CPU perf¹ | Native (f32) | CPU perf¹ | CPU perf¹ |
| SDPA (FlashAttn-2) | Native (SIMD) | Native² | Native | Native | Native |
| Paged SDPA | Native | Missing | Missing | Missing | Missing |
| Causal Conv1d | Native | Native (DeltaNet) | Native³ | Missing | Missing |
| DeltaNet (4 kernels) | Native | Native | Missing | Missing | Missing |
| Argmax / Final Logits | Native | CPU perf | CPU perf | CPU perf | CPU perf |

¹ Single-row table read — CPU memcpy is faster than GPU dispatch + sync overhead.
² Metal FlashAttention-2 with block_size=16 (fits 32KB threadgroup memory). Online softmax, no blit encoders.
³ Vulkan conv1d does not support bias parameter — models with conv bias will panic.

## GEMV by Data Type

| Data Type | CPU | Metal | Vulkan | CUDA | ROCm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| f32 | Native (SIMD) | Native | Native | Native | Native |
| f16 | Native (SIMD) | Native | Native | Native | Native |
| bf16 | Native (SIMD) | Native | Native | Native | Native |
| q8_0 | Native (SIMD) | Native | Native | Native | Native |
| q4_0 | Native (SIMD) | Native | Native | Native | Native |
| q4_1 | Native (SIMD) | Native | Missing | Missing | Missing |
| q5_0 | Native (SIMD) | Native | Missing | Missing | Missing |
| q4_k | Native (SIMD) | Native | Native | Native | Native |
| q5_k | Native (SIMD) | Native | Native | Native | Native |
| q6_k | Native (SIMD) | Native | Native | Native | Native |
| q2_k | Native (SIMD) | Native | Missing | Missing | Missing |
| q3_k | Native (SIMD) | Native | Missing | Missing | Missing |
| iq4_nl | Native (SIMD) | Native | Missing | Missing | Missing |
| iq4_xs | Native (SIMD) | Native | Missing | Missing | Missing |
| fp8_e4m3 | Native | Native | Native | Native | Native |
| fp8_e5m2 | Native | Native | Native | Native | Native |
| nvfp4 (GGUF) | Native | Missing | Missing | Missing | Missing |
| nvfp4_st (SafeTensors) | Native | Native | Missing | Missing | Missing |
| mxfp4 | Native | Missing | Missing | Missing | Missing |
| mlx_q4 | Native | Native (4-bit only) | Missing | Missing | Native |

## Kernel File Locations

| Backend | Directory | Files |
| :--- | :--- | :--- |
| CPU | `src/backend/kernels/cpu/` | `gemv.zig` (dispatcher), `gemv_*.zig` (per-format), `norm.zig`, `activation.zig`, `elementwise.zig`, `rope.zig`, `softmax.zig`, `sdpa.zig`, `embedding.zig`, `deltanet.zig` |
| Metal | `src/backend/kernels/metal/` | `common.metal`, `elementwise.metal`, `norm.metal`, `rope.metal`, `gemv.metal`, `sdpa.metal`, `deltanet.metal` |
| Vulkan | `src/backend/kernels/vulkan/` | `silu.comp`, `gelu.comp`, `add.comp`, `mul.comp`, `rms_norm.comp`, `softmax.comp`, `l2_norm.comp`, `rope.comp`, `sdpa.comp`, `embedding.comp`, `conv1d.comp`, `gemv_{f32,q8_0,q4_0,bf16,f16,q4_k,q5_k,q6_k,fp8_e4m3,fp8_e5m2}.comp` (+compiled `.spv`) |
| CUDA | `src/backend/kernels/cuda/` | `common.zig` (shared primitives), `silu.zig`, `gelu.zig`, `add.zig`, `mul.zig`, `rms_norm.zig`, `softmax.zig`, `l2_norm.zig`, `rope.zig`, `sdpa.zig`, `gemv_{f32,bf16,f16,q8_0,q4_0,q4_0_batch,q4_k,q5_k,q6_k,fp8_e4m3,fp8_e5m2}.zig`, `all.zig` (aggregator) — compiled to PTX via `zig build ptx` |
| ROCm | `src/backend/kernels/rocm/` | `common.zig` (shared primitives), `silu.zig`, `gelu.zig`, `add.zig`, `mul.zig`, `rms_norm.zig`, `rms_norm_multi.zig`, `softmax.zig`, `l2_norm.zig`, `rope.zig`, `sdpa.zig`, `gemv_{f32,bf16,f16,q8_0,q4_0,q4_k,q5_k,q6_k,fp8_e4m3,fp8_e5m2,mlx_q4}.zig`, `sigmoid_mul.zig`, `deinterleave.zig`, `deltanet.zig`, `all.zig` (aggregator) — compiled to HSACO via `zig build amdgcn` |

## Priority Roadmap

### Missing Kernels (will @panic if model needs them)

**CUDA** — highest priority gaps:
- sigmoidMul, siluMul, deinterleave, rmsNormMulti (needed for Qwen3.5 DeltaNet layers)
- DeltaNet recurrence (4 kernels: gate_beta, conv1d, l2_norm, recurrence)
- GEMV: q4_k, q5_k, q6_k already native; NVFP4/MXFP4 still missing
- Paged SDPA

**Vulkan** — medium priority:
- sigmoidMul, siluMul, deinterleave, rmsNormMulti
- DeltaNet recurrence
- GEMV: q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs, nvfp4_st, mxfp4
- Conv1d bias support, Paged SDPA

**ROCm** — medium priority:
- rmsNormMulti (GPU kernel exists but disabled — needs validation)
- DeltaNet recurrence
- GEMV: q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs, nvfp4_st, mxfp4

**Metal** — near-complete:
- GEMV: nvfp4 (GGUF), mxfp4
- Paged SDPA
