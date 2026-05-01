# Agave TODO

Comprehensive list of bugs, missing features, and improvement opportunities.

**Last updated**: 2026-05-01

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp, likely bad GGUF conversion) | Low (upstream) | Won't fix |

---

## GPU Kernel Coverage

All correctness-critical kernels are implemented as native GPU compute shaders across all 6 backends. No CPU delegation.

| Backend | Missing | Notes |
|---------|:-------:|-------|
| CUDA | 0 | Complete |
| Metal | 0 | Complete |
| WebGPU | 0 | Complete — verified correct output (Qwen 3.5 0.8B Q8_0) |
| Vulkan | 0 | Complete |
| ROCm | 1 | megakernel_gemma_q4k (performance optimization only) |

### Structural gaps (all backends)

| Kernel | Status |
|--------|--------|
| Paged SDPA (block table indirection) | Not implemented on any backend |
| NVFP4 GGUF GEMV | CPU only (GPU backends use SafeTensors NVFP4 path) |

---

## CPU Fallbacks on GPU Backends

| Operation | Metal | CUDA | ROCm | Vulkan | WebGPU |
|-----------|:-----:|:----:|:----:|:------:|:------:|
| sdpaTree (DDTree verification) | Native (f32 + turbo) | CPU delegate | CPU delegate | CPU delegate | CPU delegate |
| sdpaWithStats (split-attention) | CPU delegate | CPU delegate | CPU delegate | CPU delegate | Native (wraps SDPA) |

---

## Performance

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Primary decode bottleneck on quantized models | Optimized — group-level x register preload, needs benchmarking |
| 2 | WebGPU decode 0.7 tok/s | Synchronous per-op dispatch overhead | Open — needs batched command encoding |
| 3 | Gemma 4 E4B CPU prefill ~60s | Very slow, 42 layers with 4.5GB model | Open |
| 4 | NVFP4 model accuracy lower than MLX-4bit | May be community quantization quality, not agave bug | Open |

---

## Feature Gaps

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | Tensor/Pipeline parallelism | Design only | `docs/PARALLELISM.md` has 24-section design doc, no code |
| 2 | Structured output / grammar-constrained decoding | Working | GBNF parser, `--grammar-string`, `--json-output` (brace tracking), full-token validation, BPE encoding handled. Repetition (`*`/`+`) not yet in state machine |
| 3 | TriAttention Phase 3 | Not started | Dynamic budget, auto-tune, calibration data generator |
| 4 | Native GPU tree SDPA for CUDA/ROCm/Vulkan | Not started | Metal done (f32 + turbo), others still CPU fallback |
| 5 | Batch `forwardTree()` | Disabled | Correctness bug, disabled at `src/models/gemma3.zig` |
| 6 | Direct NVMe-to-VRAM weight loading | Not started | Tiered KV exists, weight loading still CPU-mediated |
| 7 | CUDA fused FFN megakernels (Q4_K/Q5_K/Q6_K variants) | Not started | Only Q8_0 megakernel exists for CUDA |
| 8 | WebGPU Phase 2 (WASM target) | Not started | Browser-based inference via WebAssembly + WebGPU |
| 9 | Native FP4 tensor cores on Blackwell SM121 | Research | GB10 has native FP4 via `mma.sync.m16n8k64` — 129 TFLOPS demonstrated. Needs Zig PTX inline asm + CUTLASS fragment layout port. See [forum thread](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600) |

---

## Build / CI / Infra

| # | Issue | Status |
|---|-------|--------|
| 1 | `tests/` directory has test harness but golden tests need model files | By design — manual trigger only |

---

## Documentation

No open documentation issues.
