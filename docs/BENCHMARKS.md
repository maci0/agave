# Agave vs llama.cpp: Performance Benchmarks

**Source of truth for throughput claims.** README, MODELS, and TEST_MATRIX should cite this file rather than inventing parallel numbers.

**Date**: 2026-05-18 (M4 Pro benchmarks from 2026-03-24, CUDA GB10 from 2026-05-12, NCCL RoCE from 2026-05-18)
**Hardware**: Apple M4 Pro (14-core CPU, 20-core GPU), 48 GB unified memory
**OS**: macOS 26.3.1 (aarch64)
**llama.cpp**: latest (commit ~March 2026), Metal enabled, GGML_CPU_REPACK=OFF
**Agave**: commit 7e3314a + KV cache fix

## Decode Throughput (tok/s, higher is better)

Single-token autoregressive generation (batch=1). This is the primary metric for interactive chat.

| Model | Quant | Size | llama.cpp Metal | Agave Metal | Agave CPU | Ratio (Metal) |
|-------|-------|------|---------------:|------------:|----------:|--------------:|
| Qwen3.5 0.8B | Q8_0 | 764 MB | 140.4 | 125† | 51.7† | n/a |
| Qwen3.5 9B | Q8_0 | 8.9 GB | 25.0 | 41.7 | 11.3 | **1.67x** |
| Qwen3.5 9B | Q4_K_M | 5.2 GB | 36.4 | 15.6† | 6.4† | n/a |
| Qwen3.5 9B | MLX-4bit | 5.5 GB | n/a | 24.9† | n/a | n/a |
| Gemma 4 E2B | Q4_K_M | n/a | n/a | 21.8† | n/a | n/a |
| Gemma 4 26B-A4B | Q4_K_M | n/a | n/a | 4.2† | n/a | n/a |
| Gemma 3 12B | Q8_0 | 11.6 GB | 18.7 | 22.3 | 6.3 | **1.19x** |

†Updated 2026-05-26 with sparse GEMV + Accelerate.framework

### Notes

- **Agave is 1.2–1.7x faster than llama.cpp on Metal** for decode on supported quant formats.
- llama.cpp Q4_K_M Metal comparison pending (Agave Metal requires small context workaround for Q4_K_M; see Known Issues).
- MLX-Q4 is a SafeTensors format unique to Agave (not comparable to llama.cpp).
- llama.cpp numbers include Metal GPU offload; Agave Metal uses native MSL kernels.
- CPU numbers use all 14 threads (Agave) vs 10 threads (llama.cpp default).
- Qwen3.5 0.8B Q8_0: +7% decode with `--megakernel` (see Megakernel section below).

## Megakernel System

The megakernel system has three tiers (see [MEGAKERNEL.md](MEGAKERNEL.md)); measured deltas below cover Tier 1 fused FFN, enabled via `--megakernel`:

### Tier 1: Fused FFN

Fuses gate+up GEMV + activation into a single GPU dispatch per FFN layer, reducing dispatch count by ~48 per token (24 layers x 2 saved dispatches). 11 Metal MSL kernels cover SiLU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, MLX_Q4} and GELU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0}. CUDA has 5 kernels (SiLU x {Q8_0, Q4_K, Q5_K, Q6_K} + GELU x {Q8_0}). ROCm has 0 fused FFN kernels (uses Tier 2 megakernels instead).

| Model | Quant | Standard | Megakernel | Delta | Notes |
|-------|-------|----------|------------|-------|-------|
| Qwen3.5 0.8B | Q8_0 | 111.7 tok/s | 116.3 tok/s | +4% | Short decode |
| Qwen3.5 0.8B | Q8_0 | 23.8 tok/s | 25.5 tok/s | +7% | Profiled decode |
| Gemma 4 E2B | Q4_K_M | 9.9 tok/s | 19.1 tok/s | **+93%** | Short decode (9 tok) |
| Gemma 4 E2B | Q4_K_M | 12.4 tok/s | 12.7 tok/s | +2% | 100 tok decode |
| Gemma 4 E2B | Q4_K_M | 2206 ms | 1702 ms | **-23%** | Prefill |

Largest gains on models with mixed quantization (Q4_K_M = Q4_K + Q6_K layers) where the fused kernels cover all layer types. Wired into Qwen 3.5, Gemma 3, Gemma 4 (dense+MoE), and GLM-4.

### Tier 2: True Megakernels

True megakernels execute an entire transformer layer in a single GPU dispatch using composable building blocks with atomic grid sync. 18 primitives in `mega_common.metal` (730 lines) include cooperative RMS norm, per-format GEMV, activations, RoPE, KV cache append with TurboQuant encoding, and inline SDPA with TQ+ dequant and sparse V.

**Implementations**: 5 Metal (Qwen Q8/Q4K, Gemma Q4K/Q8, Nemotron-H Q8), 3 CUDA (Qwen Q8, Gemma Q4K/Q8), 1 ROCm (Qwen Q8). Total megakernel code: ~4,923 lines across 16 files.

## Prefill Throughput

Agave uses batched GEMM + fused FlashAttention-2 for Gemma 3 prefill. Other models (hybrid SSM/MoE architectures) use sequential `forward()` which is GPU-accelerated but not batched.

| Model | Quant | Prompt | llama.cpp | Agave Sequential | Agave Batched | Speedup |
|-------|-------|--------|----------:|------------------:|--------------:|--------:|
| Gemma 3 12B | Q8_0 | 58 tok | n/a | 14.9 tok/s | 21.9 tok/s | **1.47×** |
| Gemma 3 12B | Q8_0 | 208 tok | 280 tok/s | 12.4 tok/s | 20.6 tok/s | **1.65×** |
| Gemma 3 1B | Q4_0 | 208 tok (CUDA GB10) | n/a | n/a | 44.7 tok/s | **1.19×** |

The batched prefill speedup comes from:
- **GEMM weight reuse**: each weight row loaded once from memory, multiplied against all N input tokens (N× bandwidth savings)
- **GPU kernels**: native Metal GEMM (f32/Q8_0/Q4_0), batched RoPE, FlashAttention-2 with causal masking
- **Zero per-layer flush**: entire layer runs in one GPU command buffer

CLI: `--prefill-batch-size <N>` (default 512). Use `--prefill-batch-size 1` for sequential fallback.

## Supported Quantization Formats

| Format | Agave CPU | Agave Metal | llama.cpp |
|--------|:---------:|:-----------:|:---------:|
| Q8_0 | ✅ | ✅ | ✅ |
| Q4_0 | ✅ | ✅ | ✅ |
| Q4_K_M | ✅ | ✅ | ✅ |
| Q5_K | ✅ | ✅ | ✅ |
| Q6_K | ✅ | ✅ | ✅ |
| bf16 | ✅ | ✅ | ✅ |
| f16 | ✅ | ✅ | ✅ |
| MLX-Q4 | ✅ | ✅ | ❌ |
| NVFP4 (GGUF) | ✅ | ❌ | ✅ |
| NVFP4 (SafeTensors) | ✅ | ✅ | n/a |
| MXFP4 | ✅ | ✅ | ✅ |
| IQ4_XS/NL | ✅ | ✅ | ✅ |

## Supported Model Architectures

| Architecture | Models | Status | Notes |
|-------------|--------|--------|-------|
| Gemma 3 | 1B, 4B, 12B, 27B | ✅ Working | + SigLIP vision encoder |
| Gemma 4 | E2B, E4B, 26B-A4B | ✅ Working | E2B/E4B dense, 26B MoE (expert stride fix: was garbled, now correct). + SigLIP-2 vision |
| Qwen2/2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | ✅ Working | Q/K/V biases auto-detected |
| Qwen3.5 | 0.8B, 9B, 27B, 35B | ✅ Working | + Qwen VL vision (structural) |
| Nemotron-Nano | 4B, 30B | ✅ Working | Hybrid SSM/Attention/MoE |
| Nemotron-H | 56B | ✅ Working | Hybrid SSM/MoE |
| GPT-OSS | 20B | ✅ Working | |
| GLM-4 | 4.7B Flash | ⚠️ GGUF issue | Also broken in llama.cpp, model format problem |
| Llama 4 | Scout | ✅ Working | iRoPE, chunked attention, MoE top-1 + shared expert |
| DeepSeek V4 | n/a | ✅ Working | Hyper connections, MLA, CSA/HCA compressors, LID |
| DiffusionGemma | 26B-A4B | ✅ Working | Block diffusion generation |

## KV Cache Quantization (Gemma 4 26B, Metal)

TurboQuant+ asymmetric KV compression with boundary V protection and sparse V dequantization.

| KV Type | Compression | Correct Output | Notes |
|---------|-------------|:--------------:|-------|
| f16 (default) | 1× | ✅ | Baseline |
| q8_0 | 2× | ✅ | |
| fp8 | 2× | ✅ | |
| turbo4 | 3.8× | ✅ | |
| **turbo** (preset) | K=q8_0, V=3.8× | ✅ | Recommended. Boundary V protects first/last 2 layers |
| turbo3 | 4.6× | ⚠️ | Quality loss with symmetric; use turbo preset instead |
| turbo2 | 6.4× | ⚠️ | Quality loss with symmetric |

The `turbo` preset automatically configures asymmetric quantization (K=q8_0, V=turbo4) with boundary V protection and sparse V dequantization for optimal quality-compression tradeoff.

### KV Cache Eviction

The `--kv-eviction` flag enables generating beyond the `--ctx-size` limit by periodically compressing the KV cache.

| Model | ctx_size | Budget | Eviction | Tokens Generated | Eviction Events | Notes |
|-------|----------|--------|----------|:----------------:|:---------------:|-------|
| Gemma 4 E2B | 256 | 64 | norm | 188 | 2 | Coherent output past ctx limit |

Eviction is complementary to TurboQuant, one reduces entry count, the other reduces bits per entry.

## Vision / Multimodal

| Model | Encoder | Patches | Output Tokens | Encode Time (Metal) |
|-------|---------|---------|:-------------:|:-------------------:|
| Gemma 4 26B | SigLIP-2 | 2304 (48×48) | 256 (3×3 pool) | ~41s |
| Gemma 3 27B | SigLIP | 4096 (64×64) | 4096 (no pool) | ~minutes (CPU bottleneck) |
| Qwen 3.5 9B | Qwen VL | varies | n/4 (4× merge) | ~11s |

Vision encoding uses GPU GEMM (BF16 Metal) + parallel CPU attention (thread pool across heads).

## CUDA Benchmarks (NVIDIA GB10 Blackwell)

**Hardware**: NVIDIA GB10 (Blackwell sm_121), 128 GB unified memory, aarch64
**OS**: Ubuntu 24.04, CUDA 13.0, Driver 580.142
**Date**: 2026-05-12 (Zig 0.16.0, UMA zero-copy via cuMemHostRegister)

| Model | Quant | Size | CUDA tok/s | CPU tok/s | Speedup | Prefill (CUDA) |
|-------|-------|------|:----------:|:---------:|:-------:|:--------------:|
| Qwen2.5 0.5B | Q8_0 | 644 MB | 95.2 | n/a | n/a | 232ms |
| Qwen3 0.6B | Q8_0 | 610 MB | 89.3 | 71.4 | 1.25x | 245ms |
| Qwen3 1.7B | Q8_0 | 1.7 GB | 44.7 | 28.6 | **1.56x** | 488ms |
| Qwen3 4B | Q8_0 | 4.0 GB | 21.4 | 13.9 | **1.54x** | 1031ms |
| Qwen3 8B | Q8_0 | 8.1 GB | 12.4 | 8.2 | **1.51x** | 1769ms |

Notes:
- UMA zero-copy: mmap'd weights registered via `cuMemHostRegister`, accessed directly by GPU
- Q4_K/Q6_K fall back to CPU (Zig LLVM nvptx aliasee bug prevents PTX recompilation)
- 61 CUDA PTX kernels loaded via sm_90 forward compatibility to sm_121
- Server mode (`--serve`) works correctly (cuCtxSetCurrent on scheduler thread)

## Distributed Inference (Multi-Node)

### NCCL RoCE RDMA (dual NVIDIA GB10 Blackwell)

**Hardware**: 2× NVIDIA GB10 (Blackwell sm_121, aarch64, Ubuntu 24.04), 4× ConnectX NICs each
**Network**: RoCE RDMA via NCCL
**Date**: 2026-05-18

| Model | Config | Transport | tok/s | vs Single GPU |
|-------|--------|-----------|:-----:|:-------------:|
| Qwen3.5 0.8B Q8_0 | Single GPU | n/a | 9.2 | 100% |
| Qwen3.5 0.8B Q8_0 | PP=2 | NCCL RoCE | **40.2** | 112% |
| Qwen3.5 0.8B Q8_0 | TP=2 | NCCL RoCE | 5.1 | 56% |
| Qwen3.5 9B Q4_K_M | Single GPU | n/a | 2.2 | 100% |
| Qwen3.5 9B Q4_K_M | PP=2 | NCCL RoCE | 2.2 | 100% |
| Qwen3.5 9B Q4_K_M | TP=2 | NCCL RoCE | 1.7 | 77% |

NCCL loaded at runtime via `dlopen("libnccl.so.2")`. Unique ID exchanged over TCP, then all collectives run over RoCE RDMA. Device pointer allReduceAdd passes GPU activation cache pointers directly to NCCL, no host↔device copy for GPU-dirty buffers.

### TCP (Heterogeneous x86_64 + aarch64)

**Hardware**: Node A: AMD Ryzen 9950X (x86_64, CachyOS), Node B: NVIDIA GB10 (aarch64, Ubuntu)
**Network**: TCP over LAN (~1ms RTT)
**Model**: Qwen2.5 0.5B Q8_0 (24 layers, 896 embd, 2 KV heads)
**Date**: 2026-05-13

| Mode | Command | tok/s | Network per token | Notes |
|------|---------|:-----:|:-----------------:|-------|
| Single (baseline) | `--backend cpu` | 72.5 | None | M4 Pro local |
| Local TP=2 | `--tp 2` | 49.0 | None | Sequential dual-rank |
| Distributed PP=2 | `--pp 2 --rank N --peers addr` | **28.2** | ~7 KB | 1 activation transfer |
| Distributed TP=2 | `--tp 2 --rank N --peers addr` | 16.2 | ~82 KB | 24 all-reduces |
| Hybrid TP+PP | `--tp 2 --pp 2 --rank N --peers addr` | 16.8 | ~48 KB | Local TP + remote PP |
| Disaggregated | `--disagg --rank N --peers addr` | 39.4 | ~2.4 MB once | Prefill→KV transfer→decode |

Notes:
- PP is 1.7× faster than TP over network (less traffic per token)
- Disaggregated is fastest for decode (no per-token network overhead after KV transfer)
- Heterogeneous architectures (x86_64 + aarch64) work without issues
- POSIX shm auto-selected for same-node peers (`--transport auto`)

## DeepSeek V4 Flash (MoE 290B, MLX 4-bit, CPU + SSD Streaming)

Suffix speculation on the 141GB MLX-community 4-bit model exceeds GPU-based ds4 inference by up to 7.32×.

| Mode | Model | Size | tok/s | vs ds4 Metal GPU | Notes |
|------|-------|------|-------|-----------------|-------|
| **CPU suffix (-n 64)** | MLX-Q 4-bit | 141GB | **9.5-10.6** | **1.80× WIN** | Quality-verified, says "Paris" |
| **CPU suffix (-n 256)** | MLX-Q 4-bit | 141GB | **17.2** | **2.92× WIN** | More suffix history |
| **CPU suffix (-n 1000)** | MLX-Q 4-bit | 141GB | **28.1** | **4.76× WIN** | |
| **CPU suffix (-n 2000)** | MLX-Q 4-bit | 141GB | **43.2** | **7.32× WIN** | Peak throughput |
| Metal suffix (-n 64) | MLX-Q 4-bit | 141GB | 2.3 | 0.39× | GPU rmsNorm+SDPA, CPU GEMV |
| Baseline (no suffix) | MLX-Q 4-bit | 141GB | 1.3-1.4 | 0.24× | SSD I/O-bound |
| ds4 reference | Q2 imatrix | 81GB | 5.9 | 1.00× | Metal GPU + graph capture |

**Hardware:** Apple M4 Pro 48GB, macOS 26.6.1, NVMe SSD (~3.5 GB/s).
Performance scales with sequence length: more output → more suffix history → more matches → fewer SSD-bound forwards.
See [DS4_BENCHMARK.md](DS4_BENCHMARK.md) for full methodology, Metal investigation, and optimization details.

## CPU Thread Placement (AMD Ryzen 9 9950X, 16C/32T)

**Date**: 2026-09-01. Qwen2.5 0.5B Q4_K_M, `--backend cpu -n 128 -t 0 --seed 7`, 5 runs each,
same binary flags and prompt on both sides.

| Pool sizing | tok/s (5 runs) | Median |
|-------------|----------------|-------:|
| One worker per logical CPU, unpinned (previous) | 55.3, 50.8, 55.4, 51.9, 57.1 | 55.3 |
| One worker per physical core, pinned (current)  | 56.3, 61.5, 60.0, 59.9, 59.7 | **59.9** |

+8.3% median, and the spread narrows from 6.3 to 5.2 tok/s. Quantized GEMV is
DRAM-bandwidth-bound, so SMT siblings contend for one core's load ports without adding
bandwidth, and `parallelFor`'s spin-wait degrades when two workers share a core.

Verified placement: 15 workers each hold a single-CPU affinity mask (1-15); the main
thread, which also runs `parallelFor` chunks, stays unpinned so it lands on core 0.

Machines without sysfs topology (`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`)
fall back to the logical CPU count unpinned, as before. macOS sizes by `hw.physicalcpu` and
does not pin: its affinity API is advisory and is a no-op on Apple Silicon.

## CPU GEMV Software Prefetch (AMD Ryzen 9 9950X)

**Date**: 2026-09-01. `agave-bench <kernel> --iters 200/60`, median of 5 runs per side,
GB/s from the bench's own JSON. Generation output is bit-identical (a prefetch is a hint).

L3-resident shape, n=8192 k=4096 (q4_k weights ~19 MB, L3 is 32 MB):

| Kernel | Base | With prefetch | Delta |
|--------|-----:|--------------:|------:|
| gemv_q4_k | 51.3 | 51.4 | +0.2% |
| gemv_q4_0 | 94.5 | 94.1 | -0.4% |
| gemv_q8_0 | 161.2 | 161.3 | +0.1% |

DRAM-bound shape, n=32768 k=8192 (~151 MB of weights):

| Kernel | Base | With prefetch | Delta |
|--------|-----:|--------------:|------:|
| gemv_q4_k | 48.2 | 49.3 | **+2.3%** |
| gemv_q4_0 | 48.0 | 50.6 | **+5.4%** |
| gemv_q8_0 | 43.0 | 44.5 | **+3.5%** |

The hint does nothing while the weights fit in L3 (Zen 5's hardware prefetcher already
saturates a forward scan from cache) and pays only once the stream comes from DRAM. Both
regimes matter, and neither regresses, so it stays.

All three formats converge on ~48-50 GB/s in the DRAM-bound shape regardless of how much
unpack work they do, which is the memory wall. In the L3-resident shape q4_k runs at 51 GB/s
against q8_0's 161: that gap is q4_k's nibble/scale unpack, and it is compute-bound with
roughly 3x of headroom. Widening those kernels from `@Vector(8, f32)` to the native AVX-512
width is the open lead there; it changes summation order, so it needs golden re-baselining.

## CPU GEMV Hoisted Sparsity Mask (AMD Ryzen 9 9950X)

**Date**: 2026-09-01. `agave-bench <kernel> --iters 60`, median of 9 interleaved runs per
side. Generation output is bit-identical. A VM and a llama-server were resident throughout,
so absolute GB/s runs low; base and new were interleaved, so the ratio holds.

| Kernel | n=8192 k=4096 (L3-resident) | n=32768 k=8192 (DRAM-bound) |
|--------|----------------------------:|----------------------------:|
| gemv_q4_k | 52.4 -> 78.1 (**+49.0%**) | 47.1 -> 51.2 (**+8.7%**) |
| gemv_q4_0 | 97.5 -> 113.9 (**+16.8%**) | 44.8 -> 46.3 (+3.3%) |
| gemv_q8_0 | 169.6 -> 197.0 (**+16.2%**) | 43.4 -> 46.8 (**+7.8%**) |

`isBlockSparse` was called once per (block, row group). The activation vector does not
change during a GEMV, so at n=32768 with 2-row batching the same block was rescanned 16384
times. The mask is now computed once per call and the row loop tests a bit. Includes the
weight prefetch from the previous entry, which measured flat in the L3-resident shape.

End-to-end on Qwen2.5-0.5B Q4_K_M (`-n 192 -t 0`, median of 7): 53.3 -> 55.2 tok/s (+3.6%).
The e2e gain is much smaller than the kernel gain because this model's GEMVs are small and
decode spends most of its time elsewhere; the kernel number is what scales with model size.

Applied to q4_0, q4_1, q5_0, q4_k, q5_k, q6_k, q8_0, q2_k, q3_k, iq4_nl, iq4_xs and tq2_0.
The mask also clamps its final block to `k`, where the per-block call read past the
activation buffer and let out-of-bounds bytes decide whether real elements were skipped.

## Known Issues

1. **Metal large-context hang**: With default context sizes (2048–4096) and many layers, the PagedKV block pre-allocation is slow. Workaround: use `--ctx-size 128` for benchmarks. Does not affect CPU backend.
2. **Batched prefill gap vs llama.cpp**: Agave's batched prefill achieves 1.5–1.7× over sequential but is still slower than llama.cpp's fully-fused prefill for long prompts. The remaining gap is in GEMM compute density (Agave uses one threadgroup per output row; llama.cpp uses tiled 2D GEMM with shared memory).
3. **Q4_K_M Metal**: Works but requires small context sizes due to the allocation issue above.

## Methodology

- **Decode throughput**: Measured from the stats line output by the engine after generating N tokens with greedy sampling (temperature=0). Prompt: "Hello" with model-appropriate chat template.
- **llama.cpp**: `llama-bench -p 16 -n 32/128 -r 1` with Metal enabled.
- **Agave**: `agave <model.gguf> -n N --backend {cpu,metal} "Hello"`, tok/s from stats output.
- All runs are single-pass (no averaging), cold-start (model loaded from disk each run).
- Memory pressure from other processes was minimal during benchmarks.
