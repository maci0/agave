# Agave — Production-Ready LLM Inference Engine

## What This Is

A high-performance LLM inference engine written in Zig, optimized for extreme cross-platform portability and zero-cost abstraction. Supports multiple GPU backends (Metal, CUDA, Vulkan, ROCm, CPU), multiple model architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4), and extensive quantization formats (Q2-Q8, FP8, bf16, NVFP4, MXFP4, MLX). The engine provides both CLI and HTTP server interfaces for inference.

## Core Value

Every supported model must produce correct output on every backend at full GPU speed — no unnecessary CPU fallbacks, no broken models, no unverified paths.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- ✓ Multi-backend dispatcher (CPU, Metal, CUDA, Vulkan, ROCm) with zero-overhead tagged union dispatch — existing
- ✓ GGUF model loading with mmap zero-copy — existing
- ✓ SafeTensors loading with multi-shard support — existing
- ✓ BPE and SentencePiece tokenization — existing
- ✓ Gemma3 correct output on CPU and Metal (1B GGUF, 4B/27B SafeTensors QAT 4-bit) — existing
- ✓ Qwen3.5 correct output on CPU and Metal (0.8B, 9B GGUF) — existing
- ✓ GPU-accelerated GEMV for f32, q8_0, q4_0, q4_k, q5_k, q6_k, bf16, f16, fp8, mlx_q4 on Metal — existing
- ✓ GPU-accelerated GEMV for f32, bf16, f16, q8_0, q4_0 on CUDA — existing
- ✓ GPU-accelerated elementwise ops (silu, gelu, add, mul, rmsNorm, softmax, l2Norm, rope) on Metal — existing
- ✓ Vulkan SPIR-V compute shaders (17 shaders, GEMV + elementwise) — existing
- ✓ CUDA Zig→PTX kernel pipeline (14 kernels) — existing
- ✓ KV cache quantization (f32, f16, q8_0, int8, fp8_e4m3, nvfp4) — existing
- ✓ Flat, Paged, and Radix KV cache implementations — existing
- ✓ CLI with REPL, one-shot, and piped modes — existing
- ✓ HTTP server with OpenAI-compatible API and SSE streaming — existing
- ✓ Data-driven chat templates and recipe system — existing
- ✓ Thread pool for parallel CPU GEMV — existing
- ✓ Per-op profiling instrumentation — existing

### Active

<!-- Current scope. Building toward these. -->

**Eliminate Unnecessary CPU Fallbacks:**
- [ ] Metal GPU SDPA kernel producing correct output (currently falls back to CPU)
- [ ] CUDA GPU GEMV for Q4_K, Q5_K, Q6_K quantization formats
- [ ] CUDA GPU GEMV for FP8 (E4M3, E5M2) formats
- [ ] CUDA parallel SDPA softmax (currently serial thread-0 workaround on Blackwell)
- [ ] Vulkan GPU embedding lookup kernel
- [ ] Vulkan GPU conv1d kernel (for SSM models)

**Verify All Existing Models:**
- [ ] Nemotron Nano 30B produces correct output (currently nonsensical — MoE routing instability)
- [ ] GLM-4 produces correct output (MLA attention + MoE, unverified)
- [ ] GPT-OSS produces correct output (MoE, needs verification)
- [ ] Nemotron-H produces correct output (hybrid SSM+attention, needs verification)
- [ ] All models verified on CUDA backend (test on DGX Spark at 192.168.0.212)
- [ ] All models verified on Vulkan backend

**Production Serving:**
- [ ] Continuous batching (handle multiple concurrent inference requests)
- [ ] RadixAttention integrated into server (prefix caching across conversations)
- [ ] RadixAttention LRU eviction (memory-bounded prefix cache)
- [ ] PagedAttention integrated into model inference loop
- [ ] HTTP server rate limiting and request timeouts
- [ ] Inference cancellation timeout (prevent hung requests)

### Out of Scope

- New model architectures (Llama, Phi, DeepSeek, etc.) — stabilize existing models first
- Mobile deployment (iOS, Android) — desktop/server focus
- Training or fine-tuning — inference only
- Windows support — Linux and macOS only
- External C/C++ library dependencies — pure Zig rule

## Context

- Development primarily on Apple Silicon M4 Pro (48GB) for Metal backend
- CUDA testing on DGX Spark (NVIDIA GB10 Blackwell sm_121, UMA) at `maci@192.168.0.212`
- CUDA blockReduceMax/blockReduceAdd hang on Blackwell — serial softmax workaround in place
- Metal SDPA GPU kernel exists but produces wrong output with compute-based KV append
- Nemotron Nano MoE router scores show extreme values (-2.3e26), suggesting numerical instability
- CPU fallback is acceptable where it genuinely improves performance (e.g., small ops where GPU dispatch overhead exceeds compute)
- Project targets Level 1 maturity (Metal + Vulkan fully GPU-accelerated) before Level 2 (CUDA + ROCm optimized)

## Constraints

- **Pure Zig**: No external C/C++ inference libraries. All kernels native Zig, MSL, PTX, SPIR-V
- **Hot Path**: Zero allocations, zero syscalls, zero locks in token generation loop
- **Cross-Platform**: Must cross-compile for Linux x86_64, Linux aarch64, macOS aarch64
- **No Regressions**: Performance changes must be benchmarked. >5% regression requires justification

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CPU fallback OK when faster | GPU dispatch overhead can exceed compute for small ops (embedding, tiny norms) | — Pending |
| Metal SDPA: debug GPU kernel vs rewrite | Existing GPU SDPA kernel + compute KV append exists but produces wrong output | — Pending |
| CUDA softmax: investigate Blackwell compiler bug vs alternative algorithm | blockReduce hangs on sm_121, serial workaround functional | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-21 after initialization*
