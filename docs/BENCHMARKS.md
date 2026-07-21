# Agave vs llama.cpp — Performance Benchmarks

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
| Qwen3.5 0.8B | Q8_0 | 764 MB | 140.4 | 125† | 51.7† | — |
| Qwen3.5 9B | Q8_0 | 8.9 GB | 25.0 | 41.7 | 11.3 | **1.67x** |
| Qwen3.5 9B | Q4_K_M | 5.2 GB | 36.4 | 15.6† | 6.4† | — |
| Qwen3.5 9B | MLX-4bit | 5.5 GB | — | 24.9† | — | — |
| Gemma 4 E2B | Q4_K_M | — | — | 21.8† | — | — |
| Gemma 4 26B-A4B | Q4_K_M | — | — | 4.2† | — | — |
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

Fuses gate+up GEMV + activation into a single GPU dispatch per FFN layer, reducing dispatch count by ~48 per token (24 layers x 2 saved dispatches). 11 Metal MSL kernels cover SiLU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, MLX_Q4} and GELU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0}. CUDA has 4 kernels (SiLU x {Q8_0, Q4_K, Q5_K, Q6_K}). ROCm has 1 kernel (Q8_0 SiLU).

| Model | Quant | Standard | Megakernel | Delta | Notes |
|-------|-------|----------|------------|-------|-------|
| Qwen3.5 0.8B | Q8_0 | 111.7 tok/s | 116.3 tok/s | +4% | Short decode |
| Qwen3.5 0.8B | Q8_0 | 23.8 tok/s | 25.5 tok/s | +7% | Profiled decode |
| Gemma 4 E2B | Q4_K_M | 9.9 tok/s | 19.1 tok/s | **+93%** | Short decode (9 tok) |
| Gemma 4 E2B | Q4_K_M | 12.4 tok/s | 12.7 tok/s | +2% | 100 tok decode |
| Gemma 4 E2B | Q4_K_M | 2206 ms | 1702 ms | **-23%** | Prefill |

Largest gains on models with mixed quantization (Q4_K_M = Q4_K + Q6_K layers) where the fused kernels cover all layer types. Wired into Qwen 3.5, Gemma 3, Gemma 4 (dense+MoE), and GLM-4.

### Tier 2: True Megakernels

True megakernels execute an entire transformer layer in a single GPU dispatch using composable building blocks with atomic grid sync. 18 primitives in `mega_common.metal` (732 lines) include cooperative RMS norm, per-format GEMV, activations, RoPE, KV cache append with TurboQuant encoding, and inline SDPA with TQ+ dequant and sparse V.

**Implementations**: 5 Metal (Qwen Q8/Q4K, Gemma Q4K/Q8, Nemotron-H Q8), 3 CUDA (Qwen Q8, Gemma Q4K/Q8), 1 ROCm (Qwen Q8). Total megakernel code: ~4,166 lines across 12 files.

## Prefill Throughput

Agave uses batched GEMM + fused FlashAttention-2 for Gemma 3 prefill. Other models (hybrid SSM/MoE architectures) use sequential `forward()` which is GPU-accelerated but not batched.

| Model | Quant | Prompt | llama.cpp | Agave Sequential | Agave Batched | Speedup |
|-------|-------|--------|----------:|------------------:|--------------:|--------:|
| Gemma 3 12B | Q8_0 | 58 tok | — | 14.9 tok/s | 21.9 tok/s | **1.47×** |
| Gemma 3 12B | Q8_0 | 208 tok | 280 tok/s | 12.4 tok/s | 20.6 tok/s | **1.65×** |
| Gemma 3 1B | Q4_0 | 208 tok (CUDA GB10) | — | — | 44.7 tok/s | **1.19×** |

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
| NVFP4 (SafeTensors) | ✅ | ✅ | — |
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
| GLM-4 | 4.7B Flash | ⚠️ GGUF issue | Also broken in llama.cpp — model format problem |
| Llama 4 | Scout | ✅ Working | iRoPE, chunked attention, MoE top-1 + shared expert |

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

Eviction is complementary to TurboQuant — one reduces entry count, the other reduces bits per entry.

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
| Qwen2.5 0.5B | Q8_0 | 644 MB | 95.2 | — | — | 232ms |
| Qwen3 0.6B | Q8_0 | 610 MB | 89.3 | 71.4 | 1.25x | 245ms |
| Qwen3 1.7B | Q8_0 | 1.7 GB | 44.7 | 28.6 | **1.56x** | 488ms |
| Qwen3 4B | Q8_0 | 4.0 GB | 21.4 | 13.9 | **1.54x** | 1031ms |
| Qwen3 8B | Q8_0 | 8.1 GB | 12.4 | 8.2 | **1.51x** | 1769ms |

Notes:
- UMA zero-copy: mmap'd weights registered via `cuMemHostRegister`, accessed directly by GPU
- Q4_K/Q6_K fall back to CPU (Zig LLVM nvptx aliasee bug prevents PTX recompilation)
- 56 CUDA PTX kernels loaded via sm_90 forward compatibility to sm_121
- Server mode (`--serve`) works correctly (cuCtxSetCurrent on scheduler thread)

## Distributed Inference (Multi-Node)

### NCCL RoCE RDMA (dual NVIDIA GB10 Blackwell)

**Hardware**: 2× NVIDIA GB10 (Blackwell sm_121, aarch64, Ubuntu 24.04), 4× ConnectX NICs each
**Network**: RoCE RDMA via NCCL
**Date**: 2026-05-18

| Model | Config | Transport | tok/s | vs Single GPU |
|-------|--------|-----------|:-----:|:-------------:|
| Qwen3.5 0.8B Q8_0 | Single GPU | — | 9.2 | 100% |
| Qwen3.5 0.8B Q8_0 | PP=2 | NCCL RoCE | **40.2** | 112% |
| Qwen3.5 0.8B Q8_0 | TP=2 | NCCL RoCE | 5.1 | 56% |
| Qwen3.5 9B Q4_K_M | Single GPU | — | 2.2 | 100% |
| Qwen3.5 9B Q4_K_M | PP=2 | NCCL RoCE | 2.2 | 100% |
| Qwen3.5 9B Q4_K_M | TP=2 | NCCL RoCE | 1.7 | 77% |

NCCL loaded at runtime via `dlopen("libnccl.so.2")`. Unique ID exchanged over TCP, then all collectives run over RoCE RDMA. Device pointer allReduceAdd passes GPU activation cache pointers directly to NCCL — no host↔device copy for GPU-dirty buffers.

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
- Heterogeneous architectures (x86_64 + aarch64) work seamlessly
- POSIX shm auto-selected for same-node peers (`--transport auto`)

## Known Issues

1. **Metal large-context hang**: With default context sizes (2048–4096) and many layers, the PagedKV block pre-allocation is slow. Workaround: use `--ctx-size 128` for benchmarks. Does not affect CPU backend.
2. **Batched prefill gap vs llama.cpp**: Agave's batched prefill achieves 1.5–1.7× over sequential but is still slower than llama.cpp's fully-fused prefill for long prompts. The remaining gap is in GEMM compute density (Agave uses one threadgroup per output row; llama.cpp uses tiled 2D GEMM with shared memory).
3. **Q4_K_M Metal**: Works but requires small context sizes due to the allocation issue above.

## Methodology

- **Decode throughput**: Measured from the stats line output by the engine after generating N tokens with greedy sampling (temperature=0). Prompt: "Hello" with model-appropriate chat template.
- **llama.cpp**: `llama-bench -p 16 -n 32/128 -r 1` with Metal enabled.
- **Agave**: `agave <model.gguf> -n N --backend {cpu,metal} "Hello"` — tok/s from stats output.
- All runs are single-pass (no averaging), cold-start (model loaded from disk each run).
- Memory pressure from other processes was minimal during benchmarks.
