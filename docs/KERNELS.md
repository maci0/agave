# Kernel Implementation Status

**Last Updated**: 2026-05-22

This document tracks the implementation status of all compute kernels across backends. Each kernel can be:
- **Native**: Fully implemented on the target hardware (GPU shader or optimized CPU SIMD)
- **CPU perf**: Delegates to CPU because CPU is provably faster than GPU dispatch for this op
- **Missing**: No GPU kernel exists — will `@panic` at runtime if called

**Policy**: GPU backends must never silently fall back to CPU. Missing kernels panic with a clear error message. CPU delegation is only permitted where benchmarked to be faster than GPU dispatch overhead (embLookup, tiny softmax).

---

## Compute Operations

| Operation | CPU | Metal | Vulkan | CUDA | ROCm | WebGPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RMS Norm | Native (SIMD) | Native (fused) | Native (fused) | Native | Native | Native |
| RMS Norm Multi (per-head) | Native | Native | Native | Native | Native | Native |
| L2 Norm | Native (SIMD) | Native | Native (fused) | Native | Native | Native |
| Softmax | Native (SIMD) | Native (3-pass) | Native (fused) | Native | Native | Native |
| SiLU | Native (SIMD) | Native | Native | Native | Native | Native |
| SiLU Mul (fused) | Native | Native | Native | Native | Native | Native |
| GELU | Native (SIMD) | Native | Native | Native | Native | Native |
| Add | Native (SIMD) | Native | Native | Native | Native | Native |
| Mul | Native (SIMD) | Native | Native | Native | Native | Native |
| Add+RmsNorm (fused) | Native (fused) | Native (fused) | Native (fused) | Native (fused) | Native (fused) | Native (fused) |
| Add Scaled | Native | Native | Native | Native | CPU perf | Native |
| GEMV Transposed (Q8_0) | Native | Native | Native | Native | Native | Native |
| RoPE | Native (SIMD) | Native | Native | Native | Native | Native |
| Sigmoid Mul | Native | Native | Native | Native | Native | Native |
| GELU Mul (fused) | Native | Native | Native | Native | Native | Native |
| Deinterleave | Native | Native | Native | Native | Native | Native |
| Embedding Lookup | Native | CPU perf¹ | Native (f32) | CPU perf¹ | CPU perf¹ | Native |
| SDPA (FlashAttn-2) | Native (SIMD) | Native² | Native | Native | Native | Native |
| SDPA with Stats (`sdpaWithStats`) | Native (SIMD) | Native (wraps SDPA)³ | Native (wraps SDPA)³ | Native (wraps SDPA)³ | Native (wraps SDPA)³ | Native (wraps SDPA)³ |
| SDPA Tree (DDTree verify) | Native (SIMD) | Native (f32 + turbo) | Native (f32) | Native (f32) | Native (f32) | Native (f32) |
| Paged SDPA | Native | Native | Native | Native | Native | Native |
| Causal Conv1d | Native | Native (DeltaNet) | Native | In DeltaNet | In DeltaNet | Native |
| DeltaNet (4 kernels) | Native | Native | Native | CPU delegate⁴ | Hybrid (GPU norm+recur)⁶ | Native |
| Argmax / Final Logits | Native | CPU perf | CPU perf | CPU perf | CPU perf | CPU perf |
| **Batched Prefill Ops** | | | | | | |
| GEMM (batched matmul) | Native (SIMD) | Native (f32/Q8_0/Q4_0/BF16) | Loop-of-GEMV | Native (Q8_0) | Loop-of-GEMV | Loop-of-GEMV |
| RMS Norm Batched | Native | Native (fused) | Loop-of-single | Native | Loop-of-single | Loop-of-single |
| RoPE Batched | Native | Native | Loop-of-single | Native | Loop-of-single | Loop-of-single |
| SDPA Prefill (causal FA2) | Native (SIMD) | Native (dual-source FA2) | Loop-of-SDPA | Native | Loop-of-SDPA | Loop-of-SDPA |
| **Fused FFN (Megakernel Tier 1)** | | | | | | |
| Fused Gate+Up+SiLU (Q8_0) | N/A | Native | N/A | Native | N/A | N/A |
| Fused Gate+Up+SiLU (Q4_K/Q5_K/Q6_K/Q4_0/MLX_Q4) | N/A | Native | N/A | Native (Q4_K), Source only (Q5_K/Q6_K)⁵ | N/A | N/A |
| Fused Gate+Up+GELU (Q8_0/Q4_K/Q5_K/Q6_K/Q4_0) | N/A | Native | N/A | Native (Q8_0) | N/A | N/A |

¹ Single-row table read — CPU memcpy is faster than GPU dispatch + sync overhead.
² Metal FlashAttention-2 with block_size=16 (fits 32KB threadgroup memory). Online softmax, no blit encoders. **Sparse V threshold** (1e-6) is applied in all GPU SDPA kernels (Metal, CUDA, ROCm): positions where the softmax weight falls below the threshold skip V dequantization entirely, yielding +22.8% decode speed at 32K context with zero measured PPL impact. The CPU windowed-attention fallback path (`src/ops/attention.zig`) also uses sparse V dequantization.
³ `sdpaWithStats` wraps the native GPU SDPA kernel and fills identity stats (max=0, sum=1) so the merge formula in split-attention treats GPU output as already normalized. Used by tiered KV cache split-attention (`--kv-tiers vram+ram`) for online softmax merge across tiers. CPU backend computes real per-head stats; GPU backends use identity values because their SDPA already produces final-normalized output.
⁴ CPU delegate: functional but delegates to CPU backend (no native GPU kernel yet).
⁵ CUDA fused FFN kernel source files exist for Q4_K/Q5_K/Q6_K but Q5_K/Q6_K are blocked by Zig LLVM nvptx64 aliasee bug (cross-file kernel imports create forbidden GlobalAlias). Q4_K compiles and runs.
⁶ ROCm DeltaNet: conv1d on CPU, L2 norm + recurrence kernel on GPU. Gate/beta computed on CPU. Not fully native but recurrence (the expensive part) runs on GPU.

## True Megakernels (Tier 2)

True megakernels execute an entire transformer layer (or multiple layers) in a single GPU dispatch. They use composable building blocks from `mega_common.metal` (732 lines, 18 primitives) with atomic counter grid sync for cross-threadgroup coordination.

### Building Blocks (`mega_common.metal`)

| Primitive | Description |
| :--- | :--- |
| `mega_grid_sync` | Atomic counter barrier (`memory_order_relaxed` on Metal) |
| `mega_rms_norm` / `mega_add_rms_norm` | Multi-threadgroup cooperative normalization |
| `mega_gemv_q8` / `mega_gemv_q4k` / `mega_gemv_q4_0` / `mega_gemv_q5k` / `mega_gemv_q6k` | Quantized GEMV per format |
| `mega_silu_mul` / `mega_gelu_mul` / `mega_relu_squared` / `mega_silu_mul_clamp` | Activation functions |
| `mega_rope` | Rotary position encoding |
| `mega_add` | Residual addition |
| `mega_kv_append_f32` / `mega_kv_append_tq` | KV cache append with TurboQuant encoding |
| `mega_sdpa_inline` | Full inline SDPA with TQ+ dequant, sparse V (1e-6), online softmax, GQA |

### True Megakernel Implementations

| Megakernel | Model | Quant | Metal | CUDA | ROCm |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `mega_qwen35_q8` | Qwen 3.5 | Q8_0 | Native | Native | Native |
| `mega_qwen35_q4k` | Qwen 3.5 | Q4_K | Native | Missing | Missing |
| `mega_gemma_q4k` | Gemma 3/4 | Q4_K | Native | Native | Missing |
| `mega_gemma_q8` | Gemma 3/4 | Q8_0 | Native | Native | Missing |
| `mega_nemotron_h_q8` | Nemotron-H | Q8_0 | Native | Missing | Missing |

**Total**: 5 Metal, 3 CUDA, 1 ROCm true megakernels. Each eliminates all per-layer dispatches and barriers, replacing them with a single kernel launch using atomic grid sync for cross-threadgroup coordination.

## Composed Megakernels (Tier 3)

Instead of hand-writing per-model megakernel files, the `mega_compose.zig` module generates model-specific MSL source at runtime from a `ModelDesc` struct populated from model metadata (GGUF/SafeTensors). The generated kernel references the same building blocks from `mega_common.metal` and is compiled by the Metal backend via `compileComposedMegakernel()` (runtime `newLibraryWithSource`).

The composer automatically selects the correct GEMV function (Q8_0/Q4_K/Q5_K/Q6_K/Q4_0), activation (SiLU/GELU/ReLU-squared), layer type handling (attention/DeltaNet/MoE/FFN-only), and residual pattern (fused or separate). Adding megakernel support for a new model requires only defining a `ModelDesc` -- no shader code. See [MEGAKERNEL.md](MEGAKERNEL.md) for full details.

---

## GEMV by Data Type

**NR multi-row optimization** is applied across all backends and quant formats. Each kernel computes NR output rows per thread/threadgroup, amortizing input vector loads. CPU: all formats use NR=2. Metal: Q4_K/Q5_K/Q6_K use NR=2; Q4_0/Q8_0 use NR=4; Q2_K/Q3_K/BF16/F16 use NR=2. CUDA: Q4_K/Q5_K/Q6_K use NR=2; Q4_0/Q8_0 use NR=4. ROCm: Q4_K/Q5_K/Q6_K use NR=2; Q4_0/Q8_0 use NR=4.

| Data Type | CPU | Metal | Vulkan | CUDA | ROCm | WebGPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| f32 | Native (SIMD) | Native | Native | Native | Native | Native |
| f16 | Native (SIMD) | Native | Native | Native | Native | Native |
| bf16 | Native (SIMD) | Native | Native | Native | Native | Native |
| q8_0 | Native (SIMD) | Native | Native | Native | Native | Native |
| q4_0 | Native (SIMD) | Native | Native | Native | Native | Native |
| q4_1 | Native (SIMD) | Native | Native | Native | Native | Native |
| q5_0 | Native (SIMD) | Native | Native | Native | Native | Native |
| q4_k | Native (SIMD) | Native | Native | Native | Native | Native |
| q5_k | Native (SIMD) | Native | Native | Native | Native | Native |
| q6_k | Native (SIMD) | Native | Native | Native | Native | Native |
| q2_k | Native (SIMD) | Native | Native | Native | Native | Native |
| q3_k | Native (SIMD) | Native | Native | Native | Native | Native |
| iq4_nl | Native (SIMD) | Native | Native | Native | Native | Native |
| iq4_xs | Native (SIMD) | Native | Native | Native | Native | Native |
| fp8_e4m3 | Native | Native | Native | Native | Native | Native |
| fp8_e5m2 | Native | Native | Native | Native | Native | Native |
| nvfp4 (GGUF) | Native | Missing | Missing | Missing | Missing | Missing |
| nvfp4_st (SafeTensors) | Native | Native | Native | Native | Native | Native |
| mxfp4 | Native | Native | Native | Native | Native | Native |
| mlx_q | Native | Native (4/6/8-bit) | Native (4-bit) | Native (4/6/8-bit) | Native | Native (4-bit) |

## Kernel File Locations

| Backend | Directory | Files |
| :--- | :--- | :--- |
| CPU | `src/backend/kernels/cpu/` | `gemv.zig` (dispatcher), `gemv_*.zig` (per-format), `norm.zig`, `activation.zig`, `elementwise.zig`, `rope.zig`, `softmax.zig`, `sdpa.zig`, `sdpa_tree.zig` (DDTree), `embedding.zig`, `deltanet.zig` |
| Metal | `src/backend/kernels/metal/` | `common.metal`, `elementwise.metal` (incl. `copy_f32`), `norm.metal`, `rope.metal` (incl. `rope_batched_f32`), `gemv.metal`, `gemv_tiled.metal` (Q4_K/Q8_0 tiled), `gemm.metal` (f32/Q8_0/Q4_0/BF16), `sdpa.metal` (incl. `sdpa_prefill_fa2`), `deltanet.metal`, `megakernel.metal` (11 fused FFN kernels: SiLU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, MLX_Q4} + GELU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0}), `mega_common.metal` (18 composable building blocks, 732 lines), `mega_qwen35_q8.metal`, `mega_qwen35_q4k.metal`, `mega_gemma_q4k.metal`, `mega_gemma_q8.metal`, `mega_nemotron_h_q8.metal` (true megakernels) |
| Vulkan | `src/backend/kernels/vulkan/` | `silu.comp`, `silu_mul.comp`, `gelu.comp`, `gelu_mul.comp`, `add.comp`, `add_rms_norm.comp`, `rms_norm_add.comp`, `add_scaled.comp`, `mul.comp`, `rms_norm.comp`, `rms_norm_multi.comp`, `softmax.comp`, `l2_norm.comp`, `rope.comp`, `sigmoid_mul.comp`, `deinterleave.comp`, `split_qgate.comp`, `embedding.comp`, `conv1d.comp`, `deltanet_recurrence.comp`, `sdpa.comp`, `sdpa_turbo.comp`, `sdpa_paged.comp`, `sdpa_tree.comp`, `gemv_{f32,q8_0,q4_0,q4_1,q5_0,bf16,f16,q4_k,q5_k,q6_k,q2_k,q3_k,iq4_nl,iq4_xs,fp8_e4m3,fp8_e5m2,gptq,awq,hqq,mlx_q4,nvfp4_st,mxfp4_st,t_q8_0,tq1_0,tq2_0}.comp` (+compiled `.spv`) |
| CUDA | `src/backend/kernels/cuda/` | `common.zig` (shared primitives), `silu.zig`, `silu_mul.zig`, `gelu.zig`, `gelu_mul.zig`, `add.zig`, `add_scaled.zig`, `add_rms_norm.zig`, `rms_norm_add.zig`, `mul.zig`, `rms_norm.zig`, `rms_norm_batched.zig`, `softmax.zig`, `l2_norm.zig`, `rope.zig`, `rope_batched.zig`, `sigmoid_mul.zig`, `deinterleave.zig`, `split_qgate.zig`, `sdpa.zig`, `sdpa_turbo.zig`, `sdpa_prefill.zig`, `sdpa_tree.zig`, `gemv_{f32,bf16,f16,q8_0,q4_0,q4_0_batch,q4_1,q5_0,q4_k,q5_k,q6_k,q2_k,q3_k,iq4_nl,iq4_xs,fp8_e4m3,fp8_e5m2,fp4_tc,gptq,awq,hqq,mlx_q4,mlx_q6,mlx_q8,nvfp4_st,mxfp4_st,tq1_0,tq2_0}.zig`, `gemv_t_q8_0.zig`, `gemm_q8_0.zig`, `fused_ffn_{q8_0,q4_k,q5_k,q6_k}.zig` (fused FFN megakernels)⁵, `mega_qwen35_q8.zig`, `mega_gemma_q4k.zig`, `mega_gemma_q8.zig` (true megakernels), `all.zig` (aggregator) — compiled to PTX via `zig build ptx` |
| ROCm | `src/backend/kernels/rocm/` | `common.zig` (shared primitives), `silu.zig`, `silu_mul.zig`, `gelu.zig`, `gelu_mul.zig`, `add.zig`, `add_rms_norm.zig`, `mul.zig`, `rms_norm.zig`, `rms_norm_multi.zig`, `softmax.zig`, `l2_norm.zig`, `rope.zig`, `sigmoid_mul.zig`, `deinterleave.zig`, `split_qgate.zig`, `sdpa.zig`, `sdpa_paged.zig`, `sdpa_tree.zig`, `deltanet.zig`, `deltanet_recurrence.zig`, `gemv_{f32,bf16,f16,q8_0,q4_0,q4_1,q5_0,q4_k,q5_k,q6_k,q2_k,q3_k,iq4_nl,iq4_xs,fp8_e4m3,fp8_e5m2,mlx_q4,nvfp4_st,mxfp4_st,t_q8_0,tq1_0,tq2_0}.zig`, `gemv_gptq.zig`, `gemv_awq.zig`, `gemv_hqq.zig`, `mega_qwen35_q8.zig` (true megakernel), `all.zig` (aggregator) — compiled to HSACO via `zig build amdgcn` |
| WebGPU | `src/backend/kernels/webgpu/` | `silu.wgsl`, `silu_mul.wgsl`, `gelu.wgsl`, `gelu_mul.wgsl`, `add.wgsl`, `add_rms_norm.wgsl`, `rms_norm_add.wgsl`, `add_scaled.wgsl`, `mul.wgsl`, `rms_norm.wgsl`, `rms_norm_multi.wgsl`, `softmax.wgsl`, `l2_norm.wgsl`, `rope.wgsl`, `sigmoid_mul.wgsl`, `deinterleave.wgsl`, `split_qgate.wgsl`, `embedding.wgsl`, `sdpa.wgsl`, `sdpa_paged.wgsl`, `sdpa_tree.wgsl`, `conv1d.wgsl`, `deltanet_recurrence.wgsl`, `gemv_f32.wgsl`, `gemv_bf16.wgsl`, `gemv_f16.wgsl`, `gemv_fp8_e4m3.wgsl`, `gemv_fp8_e5m2.wgsl`, `gemv_q4_1.wgsl`, `gemv_q5_0.wgsl`, `gemv_q8_0.wgsl`, `gemv_q4_0.wgsl`, `gemv_q4_k.wgsl`, `gemv_q5_k.wgsl`, `gemv_q6_k.wgsl`, `gemv_q2_k.wgsl`, `gemv_q3_k.wgsl`, `gemv_iq4_nl.wgsl`, `gemv_iq4_xs.wgsl`, `gemv_t_q8_0.wgsl`, `gemv_gptq.wgsl`, `gemv_awq.wgsl`, `gemv_hqq.wgsl`, `gemv_mlx_q4.wgsl`, `gemv_nvfp4_st.wgsl`, `gemv_mxfp4_st.wgsl`, `gemv_tq1_0.wgsl`, `gemv_tq2_0.wgsl` |

**Pipeline/kernel counts**: Metal 71 pipelines (+ 1 runtime-composed), CUDA 57 kernels, ROCm 44 kernels, Vulkan 45 shaders, WebGPU 44 shaders. Total megakernel code: ~4,640 lines across 16 files plus ~773 lines in `mega_compose.zig` (composable generator).

## Sparse GEMV (Activation Sparsity)

All CPU and GPU GEMV kernels include block-level activation sparsity checks. After SiLU/GELU activation in FFN layers, ~40% of output values are near-zero (magnitude < 0.005). Before processing each weight block, the kernel checks if all input values in that block are below the threshold. If so, the entire block (dequantization + MAC) is skipped.

Measured speedup: CPU +21%, Metal +12%, Vulkan/WebGPU (compile-verified). Inspired by [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) and [TurboSparse](https://arxiv.org/abs/2406.05955).

Implemented in all quantized GEMV kernels: Q4_K, Q8_0, Q4_0, Q4_1, Q5_0, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL, IQ4_XS, BF16, F16, FP8 (CPU only for element-level formats).

## Vision Encoder

Vision ViT (Vision Transformer) kernels run on CPU for patch embedding, positional encoding, and layer norm. GEMM/GEMV operations within the ViT encoder layers use GPU acceleration via the standard backend dispatcher for supported weight dtypes (f32, q8_0, q4_0, bf16). Vision encoding is init-time only (not in the token generation hot path). See `src/models/vision.zig`.

---

## Priority Roadmap

### Remaining Gaps

**All backends**: Every quantized GEMV format (q2_k through q8_0, q4_0/q4_1/q5_0, q4_k/q5_k/q6_k, iq4_nl/iq4_xs, f16/bf16/fp8, nvfp4_st/mxfp4, mlx_q, gptq, awq, hqq, tq1_0/tq2_0) is native on all 6 backends.

**CUDA** — functional gaps (CPU fallback):
- DeltaNet native GPU kernels (sequential recurrence delegates to CPU)
- Causal Conv1d (delegates to CPU)

**Metal** — format gap:
- GEMV: nvfp4 (GGUF) — GPU backends use SafeTensors NVFP4 (`nvfp4_st`) path instead

**Megakernel ports** (optimization, not correctness):
- ROCm: gemma_q4k, gemma_q8, qwen35_q4k, nemotron_h_q8 (requires Metal→AMDGCN porting)
- CUDA: qwen35_q4k, nemotron_h_q8 (requires Metal→PTX porting)
- Each megakernel is 200-500 lines of inline GPU assembly — not a simple format translation
