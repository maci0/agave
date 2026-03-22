# Phase 1: Correctness Foundation - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Eliminate all unnecessary CPU fallbacks in Metal, CUDA, and Vulkan backends. Fix all broken/unverified models (Nemotron Nano, GLM-4, GPT-OSS, Nemotron-H). Verify all 6 models + DeepSeek-R1-Qwen3-8B on all 5 backends (CPU, Metal, CUDA, Vulkan, ROCm) with automated golden tests. CPU fallback is acceptable only where GPU dispatch overhead exceeds compute time.

</domain>

<decisions>
## Implementation Decisions

### Metal SDPA
- **D-01:** Rewrite Metal SDPA with FlashAttention-2 tiling algorithm (do NOT debug existing broken GPU kernel)
- **D-02:** Use compute-only path — NEVER use blit encoders in hot path (causes 150ms/layer stalls from encoder switching)
- **D-03:** Implement online softmax (single-pass running max+sum) for numerical stability

### CUDA Quantized GEMV
- **D-04:** Implement Q4_K, Q5_K, Q6_K, and FP8 (E4M3/E5M2) GEMV kernels all at once (they share dequant patterns)
- **D-05:** All dequantization MUST happen in-kernel (no pre-dequant to f32 scratch buffers)
- **D-06:** Use warp-only reductions for SDPA parallel softmax (avoid shared memory block reductions that hang on sm_121 Blackwell)

### Model Verification
- **D-07:** DeepSeek-R1-0528-Qwen3-8B (Q8_0 GGUF) loads via existing Qwen3.5 code path — it's a Qwen3 architecture variant
- **D-08:** Fix all 4 broken/unverified models: Nemotron Nano (MoE routing overflow), GLM-4 (MLA + MoE), GPT-OSS (sliding window MoE), Nemotron-H (hybrid SSM+attention)
- **D-09:** For Nemotron Nano specifically: investigate MoE router score overflow (-2.3e26) — likely needs per-block scaling for quantized router weights

### Testing Strategy
- **D-10:** Golden tests use BOTH reference implementations: llama.cpp for GGUF models, HuggingFace (PyTorch) for SafeTensors models
- **D-11:** Dual-delta numerical tests: compare GPU and CPU to FP64 oracle, verify GPU error <= 2x CPU error
- **D-12:** All models verified on all 5 backends: CPU, Metal, CUDA, Vulkan, ROCm

### ROCm Backend
- **D-13:** ROCm testing on `maci@192.168.0.205` (24GB VRAM — be careful with model sizes)
- **D-14:** Test all models that fit in 24GB VRAM on ROCm
- **D-15:** CUDA testing on DGX Spark at `maci@192.168.0.212` (Blackwell sm_121, UMA)

### Vulkan Fallbacks
- **D-16:** Implement Vulkan GPU embedding lookup kernel (eliminate CPU fallback)
- **D-17:** Implement Vulkan GPU conv1d kernel for SSM models (eliminate CPU fallback)

### Claude's Discretion
- Exact FlashAttention-2 block sizes and tiling strategy for Metal
- CUDA FP8 approach: native intrinsics (Ada/Hopper+) vs 256-entry LUT
- Order of model debugging (which broken model to fix first)
- Golden test tolerance thresholds (exact epsilon values)
- Whether to use deterministic seeding or multi-seed statistical comparison

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GPU Kernel Design
- `CLAUDE.md` Section 14 — Implementation Quick Reference: file locations, model parameters, implementation gotchas
- `docs/DOCUMENTATION.md` — Core concepts, algorithm details, mathematical formulations
- `.planning/research/STACK.md` — FlashAttention-2 patterns, CUDA quantized GEMV, online softmax, warp reduction patterns
- `.planning/research/PITFALLS.md` — Metal encoder switching, sm_121 blockReduce hangs, MoE overflow, pre-dequant anti-pattern

### Metal Backend
- `src/backend/metal.zig` — Metal backend implementation (SDPA CPU fallback, cpuFallback helper, getBufRef page alignment)
- `src/backend/kernels/metal/sdpa.metal` — Existing Metal SDPA GPU kernel (to be rewritten with FA-2)
- `src/backend/kernels/metal/gemv.metal` — Metal GEMV kernels (reference for kernel style)

### CUDA Backend
- `src/backend/cuda.zig` — CUDA backend (act_cache, kv_dev_cache, deferred sync pattern)
- `src/backend/kernels/cuda/common.zig` — PTX math, warp reduction, block reduction (serial softmax workaround)
- `src/backend/kernels/cuda/sdpa.zig` — CUDA SDPA (serial thread-0 softmax to be fixed)

### Model Implementations
- `src/models/nemotron_nano.zig` — Nemotron Nano (broken output, MoE routing instability)
- `src/models/glm4.zig` — GLM-4 (MLA attention, sigmoid MoE routing)
- `src/models/gpt_oss.zig` — GPT-OSS (sliding window, clamped SwiGLU MoE)
- `src/models/nemotron_h.zig` — Nemotron-H (hybrid SSM + attention, pattern-based layer dispatch)
- `src/models/qwen35.zig` — Qwen3.5 (DeepSeek-R1-Qwen3-8B loads via this path)

### Quantization
- `src/ops/quant.zig` — Quantization helpers (Q4_K, Q5_K, Q6_K block structures, FP8 LUT)
- `src/backend/kernels/cpu/gemv_q5_k.zig`, `gemv_q6_k.zig` — CPU GEMV reference for dequant logic

### Parallelism Design (future reference)
- `docs/PARALLELISM.md` — Tensor/Pipeline/Expert parallelism design document

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/backend/kernels/metal/gemv.metal`: Metal GEMV kernels with `threadgroup_reduce_sum` helper — reuse reduction pattern for SDPA
- `src/backend/kernels/cuda/common.zig`: PTX math (`ex2.approx`, `rcp.approx`, `rsqrt.approx`), warp shuffle (`shfl.sync.down.b32`), shared memory via `cvta.shared.u64` — foundation for new CUDA kernels
- `src/ops/attention.zig`: CPU SDPA with sliding window support — reference implementation for correctness
- `src/ops/quant.zig`: Q4_K/Q5_K/Q6_K block structures and CPU dequant logic — port these patterns to CUDA PTX
- CPU GEMV kernels (`gemv_q5_k.zig`, `gemv_q6_k.zig`): SIMD-optimized with V8 vectors — reference for dequant bit-unpacking

### Established Patterns
- **Backend dispatch**: Tagged union with `inline else` — new kernels follow same pattern
- **Buffer cache**: `AutoHashMap(usize, CachedBuf)` keyed by host pointer — weights uploaded once, reused forever
- **Activation cache (CUDA)**: Dirty/stale/clean state tracking — GPU ops deferred, sync only on `be.sync()`
- **Metal page alignment**: `getBufRef()` wraps enclosing page-aligned region, returns `BufRef{buf, offset}`
- **Norm weight cache**: `normAsF32()` lazily caches f32 norm weights (keyed by data pointer) — eliminates GPU sync for SafeTensors

### Integration Points
- New CUDA GEMV kernels: Add to `src/backend/kernels/cuda/` as individual files, register in PTX build
- New Metal SDPA: Replace existing `sdpa.metal` kernel, update `metal.zig` to dispatch to GPU instead of CPU fallback
- New Vulkan kernels: Add `.comp` shaders to `src/backend/kernels/vulkan/`, compile to SPIR-V
- Golden tests: Add to `tests/` directory, integrate with `zig build test`

</code_context>

<specifics>
## Specific Ideas

- DeepSeek-R1-0528-Qwen3-8B model file at: `DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf`
- ROCm machine: `maci@192.168.0.205` (24GB VRAM — need to be careful about model sizes)
- CUDA machine: `maci@192.168.0.212` (DGX Spark, Blackwell sm_121, UMA)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-correctness-foundation*
*Context gathered: 2026-03-21*
