# Audit: docs/ — Post-Fix Re-Audit

**Date:** 2026-06-12  
**Scope:** 12 reference docs + 23 tutorial chapters (~18,400 lines total)  
**Verdict:** All 16 prior mismatches resolved. No regressions. No remaining known discrepancies.

---

## Fixes Applied & Verified (16 total)

### Reference Docs (10 fixes)

| # | File | Fix | Code Check | ✅ |
|---|------|-----|------------|---|
| 1 | `ARCHITECTURE.md` | GLM-4 template: `</think>` → `no generation prefix` | `chat_template.zig`: `generation_prefix = ""` | ✅ |
| 2 | `MEGAKERNEL.md` | `mega_compose.zig` lines: ~780 → ~1,036 | `wc -l` = 1036 | ✅ |
| 3 | `MEGAKERNEL.md` | Added `mega_sync_reset` to building block table | In `mega_common.metal` | ✅ |
| 4 | `MEGAKERNEL.md` | Kernel counts: Metal 70+→~88, CUDA 56→~59, ROCm 44→~46 | File counts match | ✅ |
| 5 | `KERNELS.md` | CPU NR: "all NR=2" → Q8_0/Q4_0/BF16/F16 NR=4 | `gemv_q8_0.zig` 4-row batch | ✅ |
| 6 | `KERNELS.md` | Added `mega_sync_reset` to building block table | — | ✅ |
| 7 | `KERNELS.md` | All counts: Metal ~88, CUDA ~59, ROCm ~46, Vulkan ~49, WebGPU ~48 | File counts match | ✅ |
| 8 | `KERNELS.md` | mega_compose + total megakernel lines updated | — | ✅ |
| 9 | `TODO.md` | All 5 backend counts with ~ prefix | — | ✅ |
| 10 | `MEGAKERNEL.md` | Key files table mega_compose line count | — | ✅ |

### Tutorial Docs (6 fixes)

| # | File | Fix | Code Check | ✅ |
|---|------|-----|------------|---|
| 11 | `tutorial/07-sampling.md` | DRY: `^ L` → `× L` | `math.zig`: `multiplier * match_len` | ✅ |
| 12 | `tutorial/04-quantization.md` | TQ1_0: 54 → 64 bytes | `backend.zig`: `tq1_0_block_bytes = 64` | ✅ |
| 13 | `tutorial/17-speculative-decoding.md` | SharedNgramPool: 8KB → ~32 KB | `ngram.zig`: `pool_capacity = 8192` × 4B | ✅ |
| 14 | `tutorial/08-backends.md` | WebGPU: 45 → ~48 | 48 `.wgsl` files | ✅ |
| 15 | `tutorial/17-speculative-decoding.md` | Removed unimplemented cooldown; documented actual logic | No cooldown in `spec_decode.zig` | ✅ |
| 16 | `tutorial/13-batched-dispatch-and-fusion.md` | Added `mega_sync_reset` | — | ✅ |

---

## Spot-Check: Unchanged Claims

| Claim | Source | Status |
|-------|--------|--------|
| 18 API endpoints | API.md vs `server.zig` | ✅ |
| Default port 49453 | API.md vs `main.zig` | ✅ |
| max_tokens default=512, cap=4096 | API.md vs `server.zig` | ✅ |
| logit_bias max 16 | API.md vs `json.zig:32` | ✅ |
| 80 file references | ARCHITECTURE.md | ✅ All exist |
| 5 recipe presets | ARCHITECTURE.md vs `recipe.zig` | ✅ |
| TransportKind: tcp/shm/nccl/rccl | PARALLELISM.md vs `transport.zig` | ✅ |
| UDP port 49460 | PARALLELISM.md vs `discovery.zig` | ✅ |
| Sparse V 1e-6 | KERNELS.md vs `attention.zig:19` | ✅ |
| N-gram range 3..10 | tutorial/17 vs `ngram.zig` | ✅ |
| Never uses .seq_cst | tutorial/appendix vs `grep src/` | ✅ |

---

## Remaining Minor Gaps (Not Regressions)

- `gemma4_unified` template (thinking channel, ≥48 layers) not in ARCHITECTURE.md table
- Qwen VL "448×448" comes from GGUF metadata, not a code constant
- Model parameter tables loaded from GGUF at runtime — unverifiable from code alone

---

## Files Modified This Session

```
docs/ARCHITECTURE.md                            |  2 +-
docs/KERNELS.md                                 |  5 +++--
docs/MEGAKERNEL.md                              |  9 +++++----
docs/TODO.md                                    | 10 +++++-----
docs/tutorial/04-quantization.md                |  2 +-
docs/tutorial/07-sampling.md                    |  2 +-
docs/tutorial/08-backends.md                    |  2 +-
docs/tutorial/13-batched-dispatch-and-fusion.md |  1 +
docs/tutorial/17-speculative-decoding.md        | 13 +++----------
                                    9 files changed, 21 insertions(+), 25 deletions(-)
```

---

## Sources

- **cs/API.md`, `docs/ARCHITECTURE.md`, `docs/KERNELS.md`, `docs/MEGAKERNEL.md`, `docs/CONTRIBUTING.md`, `docs/PARALLELISM.md`, `docs/MODELS.md`, `docs/TODO.md`, `docs/tutorial/04-quantization.md`, `docs/tutorial/07-sampling.md`, `docs/tutorial/08-backends.md`, `docs/tutorial/13-batched-dispatch-and-fusion.md`, `docs/tutorial/17-speculative-decoding.md`
- **Code:** `src/chat_template.zig`, `src/backend/mega_compose.zig`, `src/backend/backend.zig`, `src/backend/metal.zig`, `src/backend/kernels/metal/mega_common.metal`, `src/backend/kernels/cpu/gemv_q8_0.zig`, `src/ops/math.zig`, `src/ops/attention.zig`, `src/spec/spec_decode.zig`, `src/spec/ngram.zig`, `src/main.zig`, `src/server/server.zig`, `src/server/json.zig`, `src/recipe.zig`, `src/arch.zig`, `src/parallel/transport.zig`, `src/parallel/discovery.zig`
- **Repository:** `/Users/mwysocki/Experiments/agave`
