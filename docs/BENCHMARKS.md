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
unpack work they do, which is the memory wall. In the L3-resident shape q4_k runs well below
q8_0: that gap is q4_k's nibble/scale unpack, and it is compute-bound.

**Closed lead, do not retry: 512-bit vectors for q4_k.** `std.simd.suggestVectorLength(f32)`
returns **8**, not 16, on this Zen 5 part, so Zig's own heuristic already picks 256-bit here
and rewriting the kernel against the suggested width changes nothing. Forcing the rewrite
anyway measured 78.8 -> 59.9 GB/s (-24%) at n=8192 k=4096, purely from LLVM making different
unrolling decisions on the restructured source. The remaining q4_k headroom is in the scalar
per-super-block work (8 `getScaleMinK4` calls and 2 f16 conversions per row), not lane width.

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

## Checkpoint Load Time (AMD Ryzen 9 9950X, NVMe)

**Date**: 2026-09-01. `agave <model> --backend cpu -n 1`, the loader's own reported time.

| Model | Size | First load | Warm (page cache) |
|-------|-----:|-----------:|------------------:|
| Qwen2.5 0.5B Q4_K_M | 468 MB | 101 ms | 20-22 ms |
| Qwen3.5 Q4_K_M | 6.6 GB | 2590 ms | 326 ms |

~2.5 GB/s on a first pass through mmap plus the page cache. This is the measurement that
gates adopting an O_DIRECT-friendly aligned checkpoint layout (FreeToken's FTW solves a
real problem at 137 GB of expert banks): at the sizes reachable here, load time is not the
bottleneck, so the aligned layout stays unbuilt. Re-measure before revisiting, on a
checkpoint large enough that the page cache cannot hold it.

## Weight Residency Budget (Vulkan, RX 7900 XTX)

**Date**: 2026-09-01. Qwen2.5 0.5B Q4_K_M, `--backend vulkan --vram-budget <GiB>`. Working
set is ~373 MB of cached weight uploads. Output is byte-identical at every budget.

| `--vram-budget` | Resident | Evictions | tok/s |
|-----------------|---------:|----------:|------:|
| 2 GiB (above working set) | 373 MB | 0 | 33.3 |
| 0.25 GiB | 255 MB | 543 | 12.3 |
| 0.1 GiB | 102 MB | 558 | 11.0 |
| 0.05 GiB | 50 MB | 604 | 7.2 |

Residency tracks the cap, and a budget above the working set costs nothing (zero evictions,
full speed). Below it, every eviction is a re-upload on the next touch, which is the price of
running a model that does not fit. Unset (the default) keeps the previous unbounded behavior.

`--vram-budget auto` sizes it from the device's free memory (75%, leaving room for the KV
cache and activations, which the weight budget does not track). On the 7900 XTX that resolves
to 13830 MB via ROCm's `hipMemGetInfo`; Vulkan reports no free-memory figure and falls back to
75% of the 24560 MB total, saying so. It returns 0 on unified memory, where evicting a weight
buys nothing back, and warns on a backend with no device-side weight cache.

## Device Buffer Reuse Under a Budget (RX 7900 XTX)

**Date**: 2026-09-01. Qwen2.5 0.5B Q4_K_M, `-n 16 -t 0`, 3 runs each. Output is identical
with the pool, without it, and with no budget at all.

| Backend | Budget | Without pool | With pool | Delta |
|---------|--------|-------------:|----------:|------:|
| ROCm | 0.25 GiB | 19.3 | 23.7 | **+22.8%** |
| ROCm | 0.1 GiB | 23.8 | 30.6 | **+28.6%** |
| Vulkan | 0.25 GiB | 12.7 | 13.6 | +7.1% |
| Vulkan | 0.1 GiB | 16.0 | 17.2 | +7.5% |

**Why this and not transfer overlap.** At a 0.25 GiB budget the model spends 51 ms per token
more than it does unbudgeted, but only ~118 MB is re-uploaded, which is ~7 ms of PCIe time at
16 GB/s. Roughly seven eighths of the penalty was the driver's allocate/free round trips, not
moving the bytes, so overlapping the transfer with compute (FreeToken's prefill
double-buffering) could recover at most the other eighth. Recycling the allocation is the
larger and simpler win, and it is why FreeToken uses fixed slots rather than a keyed cache.

Reuse is by exact size, which fits a transformer: layer N's q_proj is byte-identical in size
to layer N+1's, so a freed buffer almost always fits the next weight of the same role. ROCm
gains more than Vulkan because `hipMalloc`/`hipFree` are costlier than Vulkan's create/destroy
plus the command-buffer flush the recycle still needs.

## Re-upload Cost, and Two Things It Is Not (RX 7900 XTX)

**Date**: 2026-09-01. `agave-bench <kernel> --reupload` evicts the weight before each
iteration; the delta against a plain run is what one `--vram-budget` eviction costs.
n=8192 k=8192, median of 5.

| Backend | Kernel | Cached | Re-upload | Delta | Weight | Effective |
|---------|--------|-------:|----------:|------:|-------:|----------:|
| ROCm | q8_0 | 0.22 ms | 6.04 ms | 5.82 ms | 68 MB | 12.3 GB/s |
| ROCm | q4_k | 0.08 ms | 3.17 ms | 3.09 ms | 36 MB | 12.2 GB/s |
| Vulkan | q8_0 | 0.67 ms | 8.86 ms | 8.20 ms | 68 MB | 8.7 GB/s |
| Vulkan | q4_k | 0.64 ms | 5.01 ms | 4.37 ms | 36 MB | 8.6 GB/s |

**This is not PCIe bandwidth.** The link is PCIe 4.0 x16 (`current_link_speed` 16.0 GT/s,
x16 = 31.5 GB/s), so 12.3 is 39% of it. The gap is not the transfer: eviction also frees the
device buffer and the re-upload allocates a new one, and those driver round trips dominate.
Read the column as re-upload cost, which is what a budget actually pays.

**Closed lead, do not retry: page-locking the weight mapping under a budget.**
`hostRegister` over the whole 468 MB mapping succeeds and changes throughput not at all
(ROCm at a 0.1 GiB budget: 30.6 tok/s before, 30.6 after; at 0.25 GiB: 23.7 before, 23.8
after). Pinning speeds up a transfer that was never the bottleneck, and it wires host RAM
that the OS could otherwise reclaim. The allocation churn is the real cost, and recycling
the buffer already addresses it.

## GPU Kernel Validation (RX 7900 XTX)

**Date**: 2026-09-01. `agave-bench <kernel> --validate` re-runs the kernel on the CPU backend
with byte-identical inputs (only `ctx.be` is swapped) and reports the largest relative
difference. Exits non-zero past a 2% tolerance, so CI can gate on it.

```bash
agave-bench gemv_q6_k --n 1024 --k 896 --backend rocm --validate   # one kernel
zig build validate -Dvalidate-backend=rocm                          # the whole sweep
```

`zig build validate` runs all 41 kernels at both k values and fails the build on any
mismatch. It is deliberately not part of `zig build test`, which has to pass on a CI runner or
a cross-compile host with no GPU. Verified to catch a regression: re-introducing the Q3_K scale
transposition makes it exit 1.

41 kernels x {ROCm, Vulkan}: **82 of 82 pass**, at k=896 and at k=900 (a multiple of neither
32 nor 256). Covers all 18 GEMV dtypes the backends dispatch, the batched prefill paths
(`gemm_q8_0`, `rms_norm_batched`, `rope_batched`, `sdpa_prefill`, `rms_norm_multi`), the
elementwise and norm ops, the aliased-output forms the prefill path uses, and the
`deinterleave` / `split_q_gate` / `add_rms_norm` / `rms_norm_add` / `sigmoid_mul` / `gelu_mul`
/ `clamped_silu_mul` / `add_scaled` set that no benchmark previously reached.

Quantized fixtures do not encode any block layout. Every byte is capped at 63, which makes an
f16 read from any two adjacent bytes at most ~1.81 whatever offset a format keeps its scale at:
always finite, never NaN. Validation does not need a well-formed quantization, only that both
sides read the same bytes the same way, and arbitrary-but-finite content tests that harder than
a tidy encoding would.

**Two real bugs it found, both now fixed:**

| Bug | Symptom | Cause |
|-----|---------|-------|
| ROCm `gemv_q6_k` | rel err 6-8 at every shape | Decoded Q6_K as a sequential nibble stream. GGML interleaves each 128-element half so `ql[l]` holds elements `l` and `l+64`, `ql[l+32]` holds `l+32` and `l+96`, with `qh[l]` supplying all four high-bit pairs. Rewritten against `kernels/cpu/gemv_q6_k.zig`. |
| ROCm `gemv_q4_k` | rel err 0.1-1.1 when `k % 256 != 0` | `nblk = k / 256` truncated while the host row stride uses the rounded-up count, so the kernel both dropped the tail and read the wrong row. |
| CPU **and** Vulkan `gemv_q3_k` | rel err 27 / 3.2 vs ROCm | The 6-bit scale takes its BYTE index from `j % 4` and its SHIFT from `j / 4` (ggml's `aux[]` shuffle). Both had them swapped, which permutes the 16 group scales and is invisible whenever the scales happen to be uniform. **ROCm was the only correct one**, so the harness first flagged the right implementation as the outlier; an independent test against ggml's formula settled it. |

Fixing them makes Qwen2.5 0.5B (Q8_0), 0.5B (Q4_K_M mix) and 1.5B (Q4_K) all produce output
identical to the CPU on ROCm and Vulkan, where every one of them previously emitted garbage.

**Third bug, found by elimination and fixed on ROCm: multi-token prefill.** Every op passed
validation alone AND batched, yet output was wrong at any chunk size above 1. The fault was in
how they compose. The GPU backends cache activations by exact host address, so a batched op
implemented as a loop over `ptr + t * stride` creates a separate device allocation per token,
each sized for one slice. A later whole-range op on the same buffer (`add(hidden, hidden2,
hidden, n_tok * e)`, the residual) finds no entry big enough, evicts them, and re-uploads from
HOST, which never saw those device-only per-token writes.

`Backend.reserveActivation` fixes it on ROCm by establishing one device buffer over the whole
range before the loop, so sub-range lookups resolve into it through `findContaining`. ROCm is
now correct at every chunk size on every model, and batched prefill is worth having:

| ROCm prefill, Qwen2.5 1.5B Q4_K, 24-token prompt | Time |
|---|---:|
| `--prefill-batch-size 1` | 1154 ms |
| batched (default) | **456 ms** |

**Vulkan needed two more things and is now correct too.** Its descriptors carry a buffer
offset (`VkBuf.offset`, honoured in `dispatch`) and its activation cache gained a
containing-range lookup, so a sub-range binds a slice of the parent instead of uploading its
own copy. Sub-range offsets must be 256-aligned, which Vulkan's required limits guarantee is
always legal; anything else falls back to a private buffer.

Two cross-chunk holes closed with it, both backend-independent: the CPU writes the chunk's
embeddings into `pf_hidden` with no way for a backend to notice, so it is invalidated
explicitly; and chunk *n+1*'s attention reads the KV chunk *n* wrote, so the chunk boundary
syncs once, which is nothing against a chunk of GEMMs.

**Chunk size is now a speed choice, not a correctness one**, and it splits by backend:

| Backend | `--prefill-batch-size 1` | batched (512) | Default |
|---------|-------------------------:|--------------:|---------|
| ROCm | 1150 ms | **456 ms** | batched |
| Vulkan | **2516 ms** | 5366 ms | 1 |

Vulkan loses on establishing the parent buffers, not on looking slices up in them. Measured
with the reserves stubbed out: 2858 ms, against 5405 ms with them and 2586 ms at chunk size 1.
A one-entry memo over the sub-range lookup measured no change at all and was dropped.

The reserves cannot simply go: without a parent, each per-token slice uploads from host and
misses whatever the GPU wrote, which is the bug they exist to fix. Making them cheap means not
re-establishing a whole range per op per layer when the device copy is already current, which
is a change to how the activation cache tracks freshness. Until then one token at a time is
Vulkan's faster path. Both paths produce identical output, and an explicit flag always wins.

**A missing ROCm kernel, found by coverage.** `rmsNormAdd` is what Gemma 4 uses for its
post-norm residual, and ROCm looked up a `rms_norm_add_kernel` that was never written, so it
panicked with "kernel not loaded". Gemma 4 could not have run on ROCm at all. The kernel now
exists (`kernels/rocm/rms_norm_add.zig`) and validates exactly. It is distinct from the
existing `add_rms_norm`, which adds first and then normalizes the sum.

**`sdpa_prefill` was a bad fixture, not a bug.** It seeded KV history by writing host memory.
Vulkan uploads the KV cache and saw it; ROCm keeps it device-side and appends, so it did not,
and that difference is correct: in inference every earlier position was written by an earlier
`sdpa` call on the device. The fixture now runs a 16-token chunk from an empty cache, so the
tail tokens attend over history the same call produced, and both backends are exact.

Its values are also strictly positive now. Attention output is a convex combination of the
value vectors, so a fixture whose values straddle zero can cancel to near-zero, and then a tiny
difference in the softmax weights flips the sign of a small number. That cannot tell a wrong
kernel from float noise, which is the one thing a validation fixture has to do.

## Placement Under Budget Pressure (RX 7900 XTX)

**Date**: 2026-09-01. Qwen2.5 1.5B Q4_K, 934 MB of cached weights, `-n 32 -t 0`, median of 3.

| Config | tok/s | Resident | Evictions |
|--------|------:|---------:|----------:|
| `--backend cpu` | 38.4 | n/a | n/a |
| ROCm, budget 4 GiB | 38.6 | 934 MB | 0 |
| ROCm, budget 1.0 GiB | 39.1 | 934 MB | 0 |
| ROCm, budget 0.75 GiB | **10.4** | 762 MB | 7935 |
| ROCm, budget 0.5 GiB | **10.3** | 508 MB | 8017 |
| ROCm, budget 0.25 GiB | **10.4** | 251 MB | 8106 |

**It is a cliff, not a slope.** The first eviction costs most of the throughput, and past it the
GPU is ~4x slower than simply decoding on the CPU. Shrinking the budget further barely matters,
because the re-upload rate is already saturated.

That is the placement question FreeToken's q* policy exists to answer, in the form Agave can
act on: its unit of placement is the backend, not the individual expert. So the startup check
compares the model's weight bytes against the budget and says which device is faster, rather
than letting a user discover a 4x regression by feel. A budget above the working set costs
nothing and stays silent.

The per-expert form of q* (split a layer's routed experts between a CPU executor and a PCIe
fetch, sized by the bandwidth ratio) is **not implemented**: Agave has no CPU MoE executor, and
there is no MoE checkpoint on this machine to validate one against.

## Weight Budget Eviction Policy (RX 7900 XTX)

**Date**: 2026-09-01. Qwen2.5 1.5B Q4_K, 934 MB of cached weights. CPU 38.4 tok/s, unbudgeted
GPU 38.9 tok/s. Median of 3, eviction counts from an 8-token run.

| Budget | LRU | MRU | LRU evictions | MRU evictions |
|--------|----:|----:|--------------:|--------------:|
| 0.85 GiB | 10.3 | **32.3** | 7899 | 755 |
| 0.75 GiB | 10.3 | **25.8** | 7935 | 1854 |
| 0.50 GiB | 10.3 | **16.9** | 8017 | 4546 |
| 0.25 GiB | 10.3 | **12.5** | 8106 | 7322 |

**LRU is the worst possible policy for a dense transformer.** Its layer loop is a cyclic scan,
so by the time the loop returns to layer 0 those weights are the least-recently-used: a budget
below the working set evicts precisely what the next layer needs, and the hit rate collapses to
roughly zero however large the budget is. That is why LRU reads 10.3 tok/s at every budget,
including one holding 91% of the weights.

Evicting the most-recently-used entry keeps whatever filled the budget first, so the hit rate
becomes budget over working set and throughput degrades in proportion. `--vram-budget-policy`
defaults to `mru` for that reason; `lru` remains for skewed reuse, which is what routed MoE
experts have and what LRU is actually good at.

A unit test pins the distinction directly: over a 40-entry cyclic scan under a 30-entry budget,
LRU scores exactly zero hits and MRU over 65%.

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
