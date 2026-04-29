# Agave TODO

Comprehensive list of bugs, missing features, and improvement opportunities.

**Last updated**: 2026-04-29

---

## Bugs

| # | Issue | File | Severity | Status |
|---|-------|------|----------|--------|
| 1 | ~~Gemma 4 Q4_K batched prefill produces wrong output~~ | `src/models/gemma4.zig` | ~~High~~ | **Fixed** — doGemm was missing Q4_K/Q5_K/Q6_K dispatch to backend GEMM |
| 2 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp, likely bad GGUF conversion) | `src/models/glm4.zig` | Low (upstream) | Won't fix |
| 3 | ~~CI/Dockerfile pin Zig 0.15.2 but codebase uses Zig 0.16.0 APIs~~ | `.github/workflows/`, `Dockerfile` | ~~Critical~~ | **Fixed** |

---

## Missing GPU Kernels

Kernels that `@panic` at runtime if called on a given backend.

### Vulkan (13 missing)

| Kernel | Notes |
|--------|-------|
| siluMul | Fused activation |
| sigmoidMul | Fused activation |
| geluMul | Fused activation |
| rmsNormMulti | Per-head norm |
| deinterleave | Tensor reorder |
| splitQGate | Q/gate split |
| GEMV transposed Q8_0 | GLM-4 MLA |
| DeltaNet (4 kernels) | SSM layers |
| conv1d bias | Bias parameter unsupported |
| NVFP4 SafeTensors GEMV | Compressed-tensors format |
| MLX GEMV | MLX quantized weights |
| MXFP4 SafeTensors GEMV | Mixed-precision format |
| embLookup non-f32 | Only f32 vocab supported |

### ROCm (8 missing)

| Kernel | Notes |
|--------|-------|
| GEMV transposed Q8_0 | GLM-4 MLA |
| NVFP4 SafeTensors GEMV | Compressed-tensors format |
| MLX GEMV | Kernel exists but panics — not integrated |
| MXFP4 SafeTensors GEMV | Mixed-precision format |
| rmsNormMulti | Kernel exists but disabled — needs validation |
| splitQGate | Q/gate split |
| DeltaNet (4 kernels) | SSM layers |
| megakernel_gemma_q4k | True megakernel variant |

### CUDA (1 missing)

| Kernel | Notes |
|--------|-------|
| splitQGate | Q/gate split |

### All Backends (structural gaps)

| Kernel | CPU | Metal | Vulkan | CUDA | ROCm |
|--------|:---:|:-----:|:------:|:----:|:----:|
| Paged SDPA (block table indirection) | Missing | Missing | Missing | Missing | Missing |
| NVFP4 GGUF GEMV | Native | Missing | Missing | Missing | Missing |

---

## CPU Fallbacks on GPU Backends

These work but delegate to CPU — should eventually be native GPU kernels.

| Operation | Metal | CUDA | ROCm | Vulkan |
|-----------|:-----:|:----:|:----:|:------:|
| sdpaTree (DDTree verification) | CPU delegate | CPU delegate | CPU delegate | CPU delegate |
| sdpaWithStats (split-attention) | CPU delegate | CPU delegate | CPU delegate | CPU delegate |
| DeltaNet | Native | CPU delegate | Missing | Missing |

---

## Performance

| # | Issue | Impact |
|---|-------|--------|
| 1 | Q4_K Metal GEMV 2-5x slower than llama.cpp | Primary decode bottleneck on quantized models |
| 2 | Q4_K Metal GEMV is 51% of runtime on Qwen 9B | Tiled GEMM approach would help |
| 3 | Gemma 4 E4B CPU prefill ~60s | Very slow, 42 layers with 4.5GB model |
| 4 | NVFP4 model accuracy lower than MLX-4bit | May be community quantization quality, not agave bug |

---

## Feature Gaps

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | Tensor/Pipeline parallelism | Design only | `docs/PARALLELISM.md` has 24-section design doc, no code |
| 2 | Structured output / grammar-constrained decoding | Not started | Design sketched in IDEAS.md |
| 3 | TriAttention Phase 3 | Not started | Dynamic budget, auto-tune, calibration data generator |
| 4 | Native GPU tree SDPA for DDTree | Not started | Currently CPU fallback on all backends |
| 5 | Batch `forwardTree()` | Disabled | Correctness bug, disabled at `src/models/gemma3.zig` |
| 6 | Direct NVMe-to-VRAM weight loading | Not started | Tiered KV exists, weight loading still CPU-mediated |
| 7 | CUDA fused FFN megakernels (Q4_K/Q5_K/Q6_K variants) | Not started | Only Q8_0 megakernel exists for CUDA |

---

## Build / CI / Infra

| # | Issue | Fix |
|---|-------|-----|
| 1 | CI workflows pin `ZIG_VERSION=0.15.2` | Update to `0.16.0` |
| 2 | Dockerfile pins `ZIG_VERSION=0.15.2` | Update to `0.16.0` |
| 3 | AGENTS.md is a copy of CLAUDE.md, not a symlink | Risk of drift — make one a symlink |
| 4 | `tests/` directory has test harness but golden tests need model files | By design — manual trigger only |

---

## Documentation

| # | Issue | Status |
|---|-------|--------|
| 1 | PARALLELISM.md is design-only, should be clearly labeled in DOCUMENTATION.md index | Fixed |
| 2 | MODELS.md GLM-4 parameter table missing expert count and dense FFN size | Open |
| 3 | README performance section has both 1.3x faster and 0.2-0.5x slower benchmarks | Confusing — different quant types, needs clarification |
