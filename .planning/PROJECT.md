# Agave — Production-Ready LLM Inference Engine

## What This Is

A high-performance LLM inference engine written in Zig, optimized for extreme cross-platform portability and zero-cost abstraction. Supports multiple GPU backends (Metal, CUDA, Vulkan, ROCm, CPU), multiple model architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4), and extensive quantization formats (Q2-Q8, FP8, bf16, NVFP4, MXFP4, MLX). The engine provides both CLI and HTTP server interfaces for inference, with continuous batching, prefix caching, and tiered KV storage for production serving.

## Core Value

Every supported model must produce correct output on every backend at full GPU speed — no unnecessary CPU fallbacks, no broken models, no unverified paths.

## Requirements

### Validated

- ✓ Multi-backend dispatcher (CPU, Metal, CUDA, Vulkan, ROCm) with zero-overhead tagged union dispatch — existing
- ✓ GGUF model loading with mmap zero-copy — existing
- ✓ SafeTensors loading with multi-shard support — existing
- ✓ BPE and SentencePiece tokenization — existing
- ✓ Gemma3 correct output on CPU and Metal — existing
- ✓ Qwen3.5 correct output on CPU and Metal — existing
- ✓ GPU-accelerated GEMV for 10+ quantization formats on Metal — existing
- ✓ GPU-accelerated GEMV for f32, bf16, f16, q8_0, q4_0 on CUDA — existing
- ✓ GPU-accelerated elementwise ops on Metal — existing
- ✓ Vulkan SPIR-V compute shaders — existing
- ✓ CUDA Zig→PTX kernel pipeline — existing
- ✓ KV cache quantization — existing
- ✓ CLI with REPL, one-shot, and piped modes — existing
- ✓ Data-driven chat templates and recipe system — existing
- ✓ Thread pool for parallel CPU GEMV — existing
- ✓ Metal FlashAttention-2 SDPA kernel (compute-only, no blit) — v1.0
- ✓ CUDA Q4_K, Q5_K, Q6_K, FP8 GEMV kernels — v1.0
- ✓ CUDA warp-parallel SDPA softmax — v1.0
- ✓ Vulkan embedding + conv1d GPU kernels — v1.0
- ✓ ROCm Q4_K/Q5_K/Q6_K/FP8/sigmoidMul/deinterleave kernels — v1.0
- ✓ All 6+1 models verified on all 5 backends with golden tests — v1.0
- ✓ Nemotron Nano MoE routing fix (per-block NVFP4 scaling) — v1.0
- ✓ GLM-4 MLA attention + sigmoid MoE — v1.0
- ✓ Continuous batching scheduler with cache-aware priority — v1.0
- ✓ PagedAttention with block tables in all models — v1.0
- ✓ RadixAttention prefix caching with real block IDs — v1.0
- ✓ OpenAI-compatible API with SSE streaming — v1.0
- ✓ Rate limiting, authentication, metrics, health checks — v1.0
- ✓ Graceful shutdown with request draining — v1.0
- ✓ TieredKvCache (VRAM/RAM/SSD) with automatic tier migration — v1.0
- ✓ Zero-copy KV access per backend (Metal, CUDA, Vulkan) — v1.0
- ✓ Async prefetch worker for SSD→RAM promotion — v1.0
- ✓ CLI tier configuration flags — v1.0

### Active

- [ ] Multi-GPU tensor parallelism (DeviceGroup abstraction)
- [ ] Pipeline parallelism across devices
- [ ] Expert parallelism for MoE models

### Out of Scope

- New model architectures (Llama, Phi, DeepSeek, etc.) — stabilize existing models first
- Mobile deployment (iOS, Android) — desktop/server focus
- Training or fine-tuning — inference only
- Windows support — Linux and macOS only
- External C/C++ library dependencies — pure Zig rule

## Context

Shipped v1.0 with 31,378 LOC Zig across 4 phases (16 plans).
- Development on Apple Silicon M4 Pro (48GB) for Metal backend
- CUDA testing on DGX Spark (NVIDIA GB10 Blackwell sm_121, UMA)
- All 6 models produce correct output on all 5 backends
- Production server with continuous batching, prefix caching, tiered storage

## Constraints

- **Pure Zig**: No external C/C++ inference libraries. All kernels native Zig, MSL, PTX, SPIR-V
- **Hot Path**: Zero allocations, zero syscalls, zero locks in token generation loop
- **Cross-Platform**: Must cross-compile for Linux x86_64, Linux aarch64, macOS aarch64
- **No Regressions**: Performance changes must be benchmarked. >5% regression requires justification

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CPU fallback OK when faster | GPU dispatch overhead can exceed compute for small ops | ✓ Good — used for embedding, small norms |
| Metal FlashAttention-2 rewrite | Old SDPA kernel had blit encoder stalls | ✓ Good — compute-only path, no encoder switching |
| CUDA warp-only SDPA softmax | blockReduce hangs on Blackwell sm_121 | ✓ Good — warp reductions avoid shared memory deadlock |
| Dual-path model init (tiered vs flat) | Backward compat for CLI mode without tiered cache | ✓ Good — null tiered_cache = flat PagedKvCache |
| detectFreeRam() hardcoded 16GB | Platform-specific RAM detection deferred | ⚠️ Revisit — user can override with --kv-ram-budget |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-03-22 after v1.0 milestone*
