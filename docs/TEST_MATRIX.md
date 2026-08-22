# Agave Test Matrix — Model × Backend × Quant

**Date**: 2026-05-19 (last full re-run). **Status note (2026-07-21):** matrix coverage lags shipped features. Placeholder rows added for DeepSeek V4, Llama 4, DiffusionGemma, GPT-OSS, Nemotron-H (not yet tested). Still missing: full Vulkan/ROCm/WebGPU correctness, LoRA, and newer `--spec-mode` values. Re-run before release; treat PASS cells below as historical, not complete coverage.

**Hardware**:
- Metal/CPU: Apple M4 Pro (14-core CPU, 20-core GPU), 48 GB unified memory, macOS 26.4
- CUDA: NVIDIA GB10 (Blackwell sm_121), aarch64, Ubuntu 24.04, CUDA 13.0 (maci@10.10.10.212)

**Methodology**: Each test runs `agave model.gguf --backend X --max-tokens 10 "What is 2+2?"` with greedy sampling. Pass = coherent answer. tok/s from stderr stats line.

---

## Model × Backend

| # | Model | Quant | Size | Metal | CPU | CUDA (CPU fb) | Notes |
|---|-------|-------|------|:-----:|:---:|:----:|-------|
| 1 | Gemma 4 26B-A4B | Q4_K_M | 15.6 GB | PASS "4" | PASS "4" | — | MoE, 30 layers |
| 2 | Gemma 4 E2B | Q4_K_S | 2.8 GB | PASS | PASS | — | Dense, 35 layers |
| 3 | Gemma 4 E4B | Q4_K_S | 4.5 GB | PASS "4" | PASS (slow) | — | Dense, 42 layers, ~60s CPU prefill |
| 4 | Gemma 3 27B QAT | Q4_0 | 14.5 GB | PASS "Four." | PASS "Four." | — | |
| 5 | Qwen 3.5 0.8B | Q8_0 | 774 MB | PASS "Four" | PASS "Four" | PASS "4" (71 tok/s) | |
| 6 | Qwen 3.5 9B | Q4_K_M | 5.2 GB | PASS "Four" | PASS "Four" | — | |
| 7 | Qwen 3.5 9B | Q8_0 | 8.9 GB | PASS "4" | PASS "4" | — | |
| 7b | Qwen 3.8 27B | MLX-4bit ST | 15.0 GB | PASS "Hi" | PASS "Hi" | — | Dense hybrid. Greedy "Say hi in one word." EOS after Hi. WebGPU PASS "Hi" (vocab GEMV chunked at 65535; DeltaNet cache must not bump generation on SSM download). Vulkan: no ICD here (CPU fallback PASS). BF16/GGUF not on disk. Skip extra RMSNorm +1 on MLX-sanitized ST. |
| 8 | GLM-4.7 Flash | Q8_0 | 30 GB | FAIL | FAIL | — | GGUF issue — also fails in llama.cpp |
| 9 | Nemotron-Nano 4B | Q8_0 | 3.9 GB | PASS | PASS | — | Prompt-sensitive; answers correctly with clear prompts |
| 10 | DeepSeek V4 | — | Metal: **10.7-21.2 tok/s** (MLX-Q4, suffix) | Not tested | Not tested | — | MLA, CSA/HCA, LID. 14 Metal kernels. Bit-identical to CPU via dedicated CpuBackend bypass. |
| 11 | Llama 4 | — | — | Not tested | Not tested | — | iRoPE, chunked attention, MoE |
| 12 | DiffusionGemma | — | — | Not tested | Not tested | — | Block diffusion |
| 13 | GPT-OSS Harmony | — | — | Not tested | Not tested | — | |
| 14 | Nemotron-H | — | — | Not tested | Not tested | — | |

**Result: 8/14 architectures pass on Metal+CPU. 1 failure (GLM-4) also broken in llama.cpp. 5 architectures not yet tested.**

## KV Cache Quantization (Gemma 4 26B, Metal)

| KV Type | Status | Output | Compression |
|---------|:------:|--------|:-----------:|
| f16 (default) | PASS | "4" | 1× |
| q8_0 | PASS | "4" | 2× |
| fp8 | PASS | "4" | 2× |
| turbo4 | PASS | "4" | 3.8× |
| turbo3 | PASS | "4" | 4.6× |
| turbo2 | PASS | "4" | 6.4× |
| **turbo** (preset) | PASS | "4" | K=q8_0, V=3.8× |

**Result: All 7 KV quantization types pass.**

## KV Cache Eviction (Gemma 4 E2B, Metal)

| Policy | ctx_size | Budget | Tokens Generated | Eviction Events | Status |
|--------|:--------:|:------:|:----------------:|:---------------:|:------:|
| norm | 256 | 64 | 188 | 2 | PASS |
| norm | 512 | 256 | 50+ | 0 | PASS |

**Result: KV eviction works, extends context beyond --ctx-size limit.**

## KV Types on Linux aarch64 (CUDA machine, CPU fallback)

| KV Type | Status | Output |
|---------|:------:|--------|
| f16 | PASS | "What is 2+2? 4" |
| q8_0 | PASS | "What is 2+2? 4" |
| fp8 | PASS | "What is 2+2? 4" |
| turbo4 | PASS | "What is 2+2? 4" |
| turbo (preset) | PASS | "What is 2+2? 4" |

**Result: All KV types work on Linux aarch64.**

## Vision / Multimodal (Metal)

| Model | Image | Status | Output |
|-------|-------|:------:|--------|
| Gemma 4 26B + SigLIP-2 | test_scene.png | PASS | "square composed of four distinct sections of color" |

**Result: Vision working. Encode ~41s on Metal (BF16 GEMM + parallel attention).**

## CUDA Backend Status (GB10 Blackwell, 2026-05-18)

| Feature | Status | Notes |
|---------|:------:|-------|
| Q8_0 native GPU | PASS | 95.2 tok/s (0.5B), 12.4 tok/s (8B) |
| Q4_K/Q5_K/Q6_K | CPU fb | PTX register spilling on sm_121 JIT — CPU thread pool fallback |
| Q4_0 native GPU | PASS | GPU kernels unaffected by K-quant spill |
| UMA zero-copy | PASS | cuMemHostRegister for mmap'd weights |
| Cross-compiled binary | PASS | Statically linked, runs on aarch64 Linux |
| NCCL RoCE RDMA | PASS | PP=2 8.5 tok/s (93% of single GPU), TP=2 5.1 tok/s |
| Server mode | PASS | cuCtxSetCurrent on scheduler thread |

## Distributed Inference (2026-05-18)

| Config | Transport | Model | Status | Notes |
|--------|-----------|-------|:------:|-------|
| PP=2 dual GB10 | NCCL RoCE | 0.8B Q8_0 | PASS | 8.5 tok/s, 93% of single GPU |
| TP=2 dual GB10 | NCCL RoCE | 0.8B Q8_0 | PASS | 5.1 tok/s |
| PP=2 dual GB10 | NCCL RoCE | 9B Q4_K_M | PASS | 2.2 tok/s (CPU fb for K-quant) |
| TP=2 dual GB10 | NCCL RoCE | 9B Q4_K_M | PASS | 1.7 tok/s |
| PP=2 hetero | TCP | 0.5B Q8_0 | PASS | x86_64 + aarch64 |
| TP=2 hetero | TCP | 0.5B Q8_0 | PASS | x86_64 + aarch64 |
| Same-node TP=2 | POSIX shm | 0.5B Q8_0 | PASS | Zero-copy IPC |
| Disaggregated | TCP | 0.5B Q8_0 | PASS | KV transfer then decode |

## llama.cpp Comparison (Metal, M4 Pro)

| Model | Quant | llama.cpp | Agave | Ratio |
|-------|-------|:---------:|:-----:|:-----:|
| Qwen 3.5 0.8B | Q8_0 | 140.4 tok/s | 183.3 tok/s | **1.31×** |
| Qwen 3.5 9B | Q8_0 | 25.0 tok/s | 41.7 tok/s | **1.67×** |
| Gemma 3 12B | Q8_0 | 18.7 tok/s | 22.3 tok/s | **1.19×** |

*llama.cpp numbers from earlier benchmark run (2026-03-24). Agave consistently 1.2-1.7× faster on Metal decode.*

---

## Summary

| Category | Passed | Failed | Total |
|----------|:------:|:------:|:-----:|
| Model × Metal | 8 | 1 (GLM-4) | 9 |
| Model × CPU | 8 | 1 (GLM-4) | 9 |
| Model × CUDA | 5 | 0 | 5 |
| KV Quantization | 7 | 0 | 7 |
| KV Eviction | 2 | 0 | 2 |
| Vision | 1 | 0 | 1 |
| Linux KV types | 5 | 0 | 5 |
| Distributed | 8 | 0 | 8 |
| **Total** | **44** | **2** | **46** |

**Overall: 44/46 tests pass (96%). The 2 failures are GLM-4 which also fails in llama.cpp (broken GGUF conversion).**

## Additional Quant Format Tests (2026-04-16)

| Model | Quant | Metal | CPU | Notes |
|-------|-------|:-----:|:---:|-------|
| Gemma 4 E2B (bartowski) | Q4_K_M | PASS | PASS | Different converter, also works |
| Gemma 4 E4B (bartowski) | Q4_K_M | PASS | PASS | |
| Gemma 3 12B | Q8_0 | PASS | PASS | |
| Qwen 3.5 35B-A3B | Q4_K_M | PASS | PASS | MoE+SSM hybrid — fixed: addRmsNorm residual in moeLayer |

## Performance Regression Analysis (2026-04-16, Metal, M4 Pro)

> **Note:** These numbers are from a diagnostic run that identified a GEMV kernel throughput regression. They do NOT supersede the benchmark results above (which were measured at commit 7e3314a with the optimized kernel). The gap is due to a different build configuration used during profiling. See "Key Performance Improvement Opportunities" below for the optimization plan.

| Model | Quant | llama.cpp (tok/s) | agave (tok/s) | Ratio | Bottleneck |
|-------|-------|:-----------------:|:-------------:|:-----:|------------|
| Qwen 0.8B | Q8_0 | 121.9 | 62.7 | 0.51× | GEMV kernel throughput |
| Qwen 9B | Q4_K_M | 24.1 | 8.3 | 0.34× | Q4_K GEMV (51% of runtime) |
| Gemma 4 26B | Q4_K_M | 26.0 | 5.5 | 0.21× | Q4_K GEMV + MoE overhead |

### Profile Breakdown (Qwen 9B Q4_K_M, Metal)

| Operation | % Runtime | Notes |
|-----------|:---------:|-------|
| gemv_ffn | 51.2% | Primary bottleneck — Q4_K dequant + matmul |
| gemv_qkv | 19.9% | Q/K/V projections |
| rms_norm | 9.0% | |
| gemv_out | 7.9% | Output projection |
| deltanet | 4.0% | SSM layers |
| silu_mul | 3.6% | Activation |
| sdpa | 1.8% | Attention is fast |

### Key Performance Improvement Opportunities

1. **Metal GEMV kernel optimization** (est. 2-3× speedup): Current implementation uses one threadgroup per output row. llama.cpp uses tiled 2D GEMM with shared memory, achieving 2-4× higher throughput on quantized formats.

2. **Multi-row GEMV batching** (est. 1.5×): Process 2-4 output rows per threadgroup to amortize weight loading. llama.cpp's `mul_mm_q4_K_f32` tiles both output and input dimensions.

3. **Metal GEMM for single-token decode** (est. 1.3×): Even single-token decode can benefit from GEMM-style tiling when the weight matrix is large (>4K rows). Current `gemv` kernel is suboptimal for this case.

4. **Fused QKV projection** (est. 1.1×): Combine Q, K, V projections into one kernel launch with shared weight loading. Qwen already has fused QKV weights (`attn_qkv.weight`).

## New Features (2026-05-19, compile-verified)

| Feature | Type | Status |
|---------|------|:------:|
| --ctx-size auto | CLI | Compiles, needs model test |
| --spec-mode ngram | CLI | Compiles + unit tests pass |
| --spec-mode dspark | CLI | Compiles + 4/4 unit tests pass |
| --benchmark | CLI | Compiles, needs model test |
| XTC sampling (API + CLI) | Sampler | Compiles + unit tests pass |
| DRY sampling (API + CLI) | Sampler | Compiles + unit tests pass |
| Mirostat 2.0 (API + CLI) | Sampler | Compiles + unit tests pass |
| Logprobs streaming API | Server | Compiles |
| UDP peer discovery | Distributed | Compiles |
| SSM state prefix caching | Server | Compiles |
| xxHash RadixTree cache | Server | Compiles |
| Vulkan deferred dispatch | Backend | Compiles, needs GPU test |
| WebGPU deferred dispatch | Backend | Compiles, needs GPU test |
| 32 new GEMV kernels | Kernels | Compiles |
| Profile-guided spec K | Spec decode | Compiles + unit tests |
| Adaptive cooldown | Spec decode | Compiles |

## Known Issues

1. **GLM-4.7 Flash** — degenerate output on both agave and llama.cpp. Likely broken GGUF conversion. The older ChatGLM-4 9B (`chatglm` arch) is a different architecture not currently supported.
2. **Qwen 3.5 35B-A3B** — FIXED. Was missing addRmsNorm residual in moeLayer. Now producing coherent output.
3. **Gemma 4 E4B CPU** — works but extremely slow (~60s prefill for 4.5GB model with 42 layers).
