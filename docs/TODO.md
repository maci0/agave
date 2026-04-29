# Agave TODO

Comprehensive list of bugs, missing features, and improvement opportunities.

**Last updated**: 2026-04-29

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp, likely bad GGUF conversion) | Low (upstream) | Won't fix |

---

## Missing GPU Kernels

Kernels that `@panic` at runtime if called on a given backend.

### Vulkan (7 missing)

| Kernel | Notes |
|--------|-------|
| GEMV transposed Q8_0 | GLM-4 MLA |
| DeltaNet (4 kernels) | SSM layers |
| conv1d bias | Bias parameter unsupported |
| NVFP4 SafeTensors GEMV | Compressed-tensors format |
| MLX GEMV | MLX quantized weights |
| MXFP4 SafeTensors GEMV | Mixed-precision format |
| embLookup non-f32 | Only f32 vocab supported |

### ROCm (7 missing)

| Kernel | Notes |
|--------|-------|
| GEMV transposed Q8_0 | GLM-4 MLA |
| NVFP4 SafeTensors GEMV | Compressed-tensors format |
| MLX GEMV | Kernel exists but panics — not integrated |
| MXFP4 SafeTensors GEMV | Mixed-precision format |
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
| sdpaTree (DDTree verification) | Native (f32 + turbo) | CPU delegate | CPU delegate | CPU delegate |
| sdpaWithStats (split-attention) | CPU delegate | CPU delegate | CPU delegate | CPU delegate |
| DeltaNet | Native | CPU delegate | Missing | Missing |

---

## Performance

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Primary decode bottleneck on quantized models | Optimized — group-level x register preload, needs benchmarking |
| 2 | Gemma 4 E4B CPU prefill ~60s | Very slow, 42 layers with 4.5GB model | Open |
| 3 | NVFP4 model accuracy lower than MLX-4bit | May be community quantization quality, not agave bug | Open |

---

## Feature Gaps

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1 | Tensor/Pipeline parallelism | Design only | `docs/PARALLELISM.md` has 24-section design doc, no code |
| 2 | Structured output / grammar-constrained decoding | Not started | Design sketched in IDEAS.md |
| 3 | TriAttention Phase 3 | Not started | Dynamic budget, auto-tune, calibration data generator |
| 4 | Native GPU tree SDPA for CUDA/ROCm/Vulkan | Not started | Metal done (f32 + turbo), others still CPU fallback |
| 5 | Batch `forwardTree()` | Disabled | Correctness bug, disabled at `src/models/gemma3.zig` |
| 6 | Direct NVMe-to-VRAM weight loading | Not started | Tiered KV exists, weight loading still CPU-mediated |
| 7 | CUDA fused FFN megakernels (Q4_K/Q5_K/Q6_K variants) | Not started | Only Q8_0 megakernel exists for CUDA |

---

## Build / CI / Infra

| # | Issue | Status |
|---|-------|--------|
| 1 | AGENTS.md is a copy of CLAUDE.md, not a symlink | Risk of drift — make one a symlink |
| 2 | `tests/` directory has test harness but golden tests need model files | By design — manual trigger only |

---

## Documentation

| # | Issue | Status |
|---|-------|--------|
| 1 | MODELS.md GLM-4 parameter table missing expert count and dense FFN size | Open |
