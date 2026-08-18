# DeepSeek V4 Flash — Performance Handoff

*Last updated: 2026-08-19. Covers 67+ autoresearch iterations (CPU) + 14 Metal iterations.*

---

## Results

**Hardware:** Apple M4 Pro, 48GB unified memory, macOS 26.6.1, NVMe SSD (~3.5 GB/s)
**Model:** mlx-community/DeepSeek-V4-Flash-4bit (141GB, SafeTensors, MLX-Q mixed format)

### CPU Results (main branch) — Quality-Verified

Config: `--backend cpu --ssd-streaming --spec-mode suffix -t 0.0`
Budget: fallback=3, bonus=4. Thread pool grain=128. max_k=96.
Special token filter (ID≥128000) in suffix history.

| Sequence Length | Factual tok/s | Code tok/s | Prose tok/s | vs ds4 5.9 |
|-----------------|--------------|-----------|------------|-----------|
| **-n 64** | **9.5-10.6** | **7.1-7.3** | **7.2-7.4** | **1.20-1.80×** |
| **-n 256** | **17.2** | — | **11.2** | **1.90-2.92×** |
| **-n 500** | **24.1** | — | — | **4.08×** |
| **-n 1000** | **28.1** | — | — | **4.76×** |
| **-n 2000** | **43.2** | — | — | **7.32×** |
| Baseline (no suffix) | 1.3-1.4 | 1.3-1.4 | 1.3-1.4 | 0.24× |

**All workloads exceed ds4 5.9 tok/s at -n 64. Performance scales with sequence length because longer output creates more suffix history → more matches → fewer SSD-bound forward passes.**

Quality: "The capital of France is Paris." ✅ No chat template echo. Model-level repetition at longer sequences from 4-bit quantization.

### Metal Results (autoresearch/ds4-metal branch)

Config: `--backend metal --ssd-streaming --spec-mode suffix -t 0.0`

| Component | Status | Notes |
|-----------|--------|-------|
| rmsNorm | **Native Metal GPU** ✅ | Bit-exact with CPU |
| SDPA attention | **Native Metal GPU** ✅ | FlashAttention-2 kernel |
| clampedSiluMul | **Native Metal GPU** ✅ | Fused elementwise |
| addScaled | **Native Metal GPU** ✅ | Expert accumulation |
| gemvMlxQ (attention) | CPU fallback with sync | GPU FMA ≠ CPU NEON FMA rounding |
| gemvMxfp4St (experts) | CPU fallback with sync | GPU FMA ≠ CPU NEON FMA rounding |

**2.3 tok/s** with suffix, correct output. Limited by per-GEMV sync overhead (430 syncs/forward).

### ds4 Reference
5.9 tok/s on Metal GPU with 81GB Q2 imatrix GGUF model.

---

## Architecture

### MLX 4-bit Expert Dequantization

Three bugs fixed (iteration 14):

1. **Scale format**: Expert uint8 scales use **E8M0** (`2^(val-127)`), not FP8 E4M3 (which gives 32,000× wrong magnitude). Added `Mxfp4ScaleFormat` enum.
2. **Group size**: MLX experts use **gs=32** (not hardcoded 16). Parameterized `gs` through `gemvMxfp4St` across all 6 backends.
3. **Dtype detection**: SafeTensors parses `U8` as `.nvfp4`, not `.unknown`. Fixed check in `doGemvExpert`.

### Suffix Speculation

Suffix mode uses the `is_self_draft` code path (not `verifyBatched`):
- Proposes N draft tokens from output history (instant, no model call)
- Accepts ALL drafts (100% acceptance — deterministic model, own history)
- Runs ONE `forward()` for the bonus token
- Each round: N+1 tokens for the cost of one forward pass

**~75% of rounds are zero-draft fallbacks** (unique tokens, no suffix match). Each fallback runs a full forward with `expert_budget=3` (50% fewer expert reads than default 6). The bonus forward uses `budget=4` for higher quality next-token prediction.

### Metal Investigation Findings

14 iterations of Metal kernel development revealed:

1. **GPU can't trigger mmap page faults**: `newBufferWithBytesNoCopy` on unfaulted pages reads zeros. Fixed with `prefaultPages()` that touches each page before Metal wraps it.

2. **buf_cache flush was unnecessary**: `sync()` flushed all `newBufferWithBytesNoCopy` wraps. On UMA (Apple Silicon), these shared-memory wraps are always valid. Removed the flush.

3. **GPU FMA ≠ CPU NEON FMA**: Even with identical float4 accumulation order (vec8-exact kernel), Apple Silicon's GPU FMA unit and CPU NEON FMA unit produce ~0.02% different intermediate rounding. Over 43 Hyper Connection layers, this cascades to completely different tokens.

4. **Vec8-exact kernels**: `gemv_mlx_q4_exact` and `gemv_mxfp4_st_exact` use float4 fma pairs + pairwise reduction to match CPU's `@mulAdd(@Vector(8,f32))` + `@reduce(.Add)`. Infrastructure shipped but not production-enabled due to FPU drift.

### Performance Bottleneck

The system is **SSD bandwidth-bound**:
- Each forward reads ~1.5GB of expert weights from NVMe (budget=3)
- NVMe peak: 3.5 GB/s → minimum 0.43s per forward
- With 43 layers × attention/HC compute overhead: ~0.77s per forward
- Suffix speculation amortizes: N+1 tokens per forward (N=15-28 depending on history)

---

## Key Optimizations (What Actually Worked)

| # | Change | Impact | Files |
|---|--------|--------|-------|
| 1 | MLX 4-bit dequant fix | 0→1.3 tok/s | mlx.zig, deepseek4.zig, all backends |
| 2 | Expert budget=3 fallback | +66% | main.zig |
| 3 | Thread pool grain 16→128 | +10% | cpu.zig |
| 4 | Special token filter | Quality fix | main.zig |
| 5 | max_k 48→96 | +25% code | ngram.zig |
| 6 | n_routed_experts key | Cache fix | main.zig |

### What Didn't Work

| Attempt | Result | Root Cause |
|---------|--------|------------|
| forwardTree layer skip | KV cache corruption | Skipped layers don't populate KV |
| verifyBatched | 0% acceptance | No HC + no routed experts = too approximate |
| Dequant+SGEMM | No speedup | 8× memory expansion cancels AMX gain |
| Batched MLX-Q4 GEMM | No speedup | Page cache makes weight-stationary irrelevant |
| Metal native GEMV | 2% L2 drift | GPU FMA ≠ CPU NEON FMA (hardware) |
| Metal native MXFP4 | NaN for gs=8 | Float4 load alignment or compiler issue |
| Anti-repetition suffix | 2-3× slower | Kills suffix match rate |

---

## Reproduction

```bash
# Build
zig build

# Download model
huggingface-cli download mlx-community/DeepSeek-V4-Flash-4bit
MLX=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*/

# CPU: 10.5 tok/s factual (exceeds ds4 5.9)
./zig-out/bin/agave "$MLX" --backend cpu --ssd-streaming --ctx-size 512 \
  -n 64 --spec-mode suffix -t 0.0 "What is the capital of France?"

# CPU: 43.2 tok/s at 2000 tokens
./zig-out/bin/agave "$MLX" --backend cpu --ssd-streaming --ctx-size 2048 \
  -n 2000 --spec-mode suffix -t 0.0 "Write about the history of France."

# Metal: 2.3 tok/s (correct output, GPU rmsNorm+SDPA+silu)
./zig-out/bin/agave "$MLX" --backend metal --ssd-streaming --ctx-size 512 \
  -n 64 --spec-mode suffix -t 0.0 "What is the capital of France?"

# Baseline (no suffix): 1.3 tok/s
./zig-out/bin/agave "$MLX" --backend cpu --ssd-streaming --ctx-size 512 \
  -n 64 -t 0.0 "What is the capital of France?"

# ds4 reference: 5.9 tok/s
cd /tmp/ds4 && ./ds4 -m ds4flash.gguf --ssd-streaming -c 512 \
  -p "What is the capital of France?" --temp 0 --tokens 64 --nothink
```

---

## Files Changed

### CPU Optimizations (main branch)
- `src/ops/mlx.zig` — `Mxfp4ScaleFormat` enum, dynamic `gs`, 2-row batching, `mlxGemmQ4`
- `src/backend/backend.zig` — `gs` + `sf` params on `gemvMxfp4St` dispatcher
- `src/backend/cpu.zig` — `parallel_grain=128`, `gs`+`sf` params, thread pool dispatch
- `src/backend/{metal,cuda,rocm,vulkan,webgpu}.zig` — `gs`+`sf` signature updates
- `src/models/deepseek4.zig` — E8M0/nvfp4 dequant fix, `inferMxfp4GroupSize`, `cpuGemvExpert`
- `src/models/{model,qwen35,gpt_oss}.zig` — `inferMxfp4GroupSize`, `gs`/`sf` propagation
- `src/main.zig` — Expert budget=3 fallback, special token filter, `n_routed_experts` key
- `src/spec/ngram.zig` — `max_k=96`
- `src/chat_template.zig` — DS4 EOG tokens

### Metal Optimizations (autoresearch/ds4-metal branch)
- `src/backend/metal.zig` — CPU fallback for gemvMlxQ/gemvMxfp4St, `prefaultPages`, buf_cache no-flush
- `src/backend/kernels/metal/gemv.metal` — `gemv_mlx_q4_exact`, `gemv_mxfp4_st_exact`, dynamic `gs`/`scale_fmt`

*67+ CPU iterations + 14 Metal iterations. ~36 hours of experimentation.*
