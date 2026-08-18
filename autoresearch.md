# Autoresearch: DS4 Flash Maximum Performance

**Started:** 2026-08-15 (CPU), 2026-08-18 (Metal)
**Completed:** 2026-08-19
**Total iterations:** 67+ (CPU) + 14 (Metal) = 81+
**Hardware:** Apple M4 Pro 48GB, macOS 26.6.1, NVMe SSD (~3.5 GB/s)

## Final Results

### CPU (main branch) — Quality-Verified

| Sequence | Factual | Code | Prose | vs ds4 5.9 |
|----------|---------|------|-------|-----------|
| -n 64 | **10.5** | **7.3** | **7.4** | **1.20-1.78×** |
| -n 256 | **17.2** | — | **11.2** | **1.90-2.92×** |
| -n 500 | **24.1** | — | — | **4.08×** |
| -n 1000 | **28.1** | — | — | **4.76×** |
| -n 2000 | **43.2** | — | — | **7.32×** |

Baseline (no suffix): 1.3-1.4 tok/s

### Metal (autoresearch/ds4-metal branch)

2.3 tok/s suffix, correct output. Native GPU: rmsNorm, SDPA, silu, addScaled. CPU fallback: GEMV (GPU FMA ≠ CPU NEON FMA rounding).

## Key Optimizations

| # | Change | Impact | Iter |
|---|--------|--------|------|
| 1 | MLX 4-bit dequant fix (E8M0, gs=32, nvfp4) | 0→1.3 tok/s | 14 |
| 2 | Expert budget=3 fallback (50% less SSD) | +66% | 57 |
| 3 | Thread pool grain 16→128 | +10% | 48 |
| 4 | Special token filter (ID≥128000) | Quality fix | 67 |
| 5 | max_k 48→96 | +25% code | 20 |
| 6 | n_routed_experts metadata key | Cache fix | 64 |
| 7 | prefaultPages for Metal mmap | Metal fix | Metal-10 |
| 8 | buf_cache UMA no-flush | Metal fix | Metal-12 |
| 9 | Vec8-exact Metal kernels | Infrastructure | Metal-13 |

## What Didn't Work

| Attempt | Result | Iter |
|---------|--------|------|
| forwardTree layer skip | KV cache corruption | 23-31 |
| verifyBatched for DS4 | 0% acceptance | 56 |
| Dequant+SGEMM | No speedup (page cached) | 33-34 |
| Batched MLX-Q4 GEMM | No speedup | 29 |
| Native Metal GEMV | 2% L2 drift (hardware FPU) | Metal-1-14 |
| Anti-repetition suffix | 2-3× slower | 66 |
| budget=2 fallback | Fewer suffix matches | 58, cont |

## Bottleneck Analysis

The system is **SSD bandwidth-bound**:
- Each forward: ~1.5GB expert reads from NVMe (budget=3)
- NVMe: 3.5 GB/s → 0.43s minimum per forward
- With compute overhead: ~0.77s per forward
- Suffix amortizes: 15-28 tokens per forward depending on history length
- Longer sequences → more history → more matches → higher tok/s

## Files Changed

18 files on main branch, 4 files on Metal branch. See `notes/DS4_HANDOFF.md` for complete list.
