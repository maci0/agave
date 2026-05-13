# Project Documentation Review

**Date:** 2026-05-10  
**Scope:** 12 docs files (excluding `docs/tutorial/` which was reviewed separately)  
**Files:** ARCHITECTURE.md, API.md, CONTRIBUTING.md, KERNELS.md, MEGAKERNEL.md, MODELS.md, BENCHMARKS.md, DOCUMENTATION.md, TEST_MATRIX.md, IDEAS.md, TODO.md, PARALLELISM.md

---

## Executive Summary

The project documentation is **comprehensive and well-organized**. ARCHITECTURE.md is an excellent single-page reference. CONTRIBUTING.md has clear step-by-step templates for every extension point. KERNELS.md provides the definitive status matrix.

**6 issues found, 10 improvements suggested.** The most impactful: stale WHT FMA numbers in TODO.md (same bug we fixed in the tutorials), inconsistent benchmark data between BENCHMARKS.md and TEST_MATRIX.md, and some outdated status markers in TODO.md.

---

## 1. Factual Errors

### 1.1 ❌ TODO.md — Stale WHT FMA count (same bug as tutorials)

The KV cache quantization table at the bottom of TODO.md still shows `16,384` FMAs for TurboQuant and claims "PlanarQuant uses 64x fewer FMAs." These were already fixed in the tutorials (actual: ~640 FMAs per 128-dim head using 32-element block butterflies).

**Lines:** 95, 100

### 1.2 ⚠️ BENCHMARKS.md vs TEST_MATRIX.md — Contradictory Qwen 0.8B numbers

BENCHMARKS.md (top of file, "Decode Throughput"):
> Qwen3.5 0.8B Q8_0: llama.cpp 140.4, Agave 183.3 → **1.31×**

TEST_MATRIX.md ("Performance Comparison", dated 2026-04-16):
> Qwen 0.8B Q8_0: llama.cpp 121.9, Agave 62.7 → **0.51×**

These are opposite conclusions — one says 1.31× faster, the other says 0.51× slower. The BENCHMARKS.md data is dated 2026-03-24 and the TEST_MATRIX data 2026-04-16 (later). If the later numbers are correct, the headline benchmark claim is wrong.

**This needs investigation.** If both are real measurements under different conditions, they need clear labels explaining the difference (context size, megakernel on/off, etc.).

### 1.3 ⚠️ MODELS.md — Date says "2026-04-08" but content is newer

The header says `Supported Models (2026-04-08)` but references Qwen 3.6 and Gemma 4, which were added after that date based on the other docs. Minor, but stale dates erode trust.

---

## 2. Inconsistencies

### 2.1 TODO.md — "Zero CPU delegates remaining" contradicts KERNELS.md

TODO.md states:
> **Zero CPU delegates remaining.** All operations dispatch to native GPU kernels.

But KERNELS.md shows:
- `Paged SDPA`: "CPU Fallback" on Metal, CUDA, Vulkan, ROCm, WebGPU
- `sdpaWithStats`: "CPU delegate⁷" on Metal, CUDA, Vulkan, ROCm
- `DeltaNet`: "CPU delegate⁶" on CUDA

KERNELS.md was updated 2026-05-10 (today). TODO.md may be out of date, or the definitions of "CPU delegate" differ between the two docs.

### 2.2 TODO.md GPU kernel table vs KERNELS.md

TODO.md says:
> CUDA: 0 missing, WebGPU: 0 missing, Vulkan: 0 missing

KERNELS.md's priority roadmap lists missing kernels for all three:
- CUDA: DeltaNet, Conv1d
- Vulkan: DeltaNet, many GEMV formats, conv1d bias
- WebGPU: ~37 missing ops

These can't both be true. Either TODO.md was updated to reflect a different "missing" definition (maybe "correctness-critical" vs "all ops"), or it's stale.

### 2.3 BENCHMARKS.md — Megakernel section says "12 Metal MSL kernels" but count elsewhere is 11

MEGAKERNEL.md Tier 1 table lists 11 kernels (6 SiLU + 5 GELU). But the fused FFN section header says "12 kernels in `megakernel.metal`". KERNELS.md also says "11 fused FFN kernels." The count `12` in BENCHMARKS.md appears to be wrong (off by one). Check `megakernel.metal` to confirm.

---

## 3. Structural Issues

### 3.1 DOCUMENTATION.md is redundant with tutorial/README.md

DOCUMENTATION.md is essentially a table of contents that lists the same tutorials and product docs that tutorial/README.md already covers. It adds no unique information. Consider either:
- Removing it and pointing readers to `tutorial/README.md` + the individual product docs
- Or making it the true landing page with a one-sentence summary per doc (currently it has no summaries for tutorials, just titles)

### 3.2 IDEAS.md has stale "IMPLEMENTED" sections

Several sections in IDEAS.md are marked as implemented (Speculative Decoding, Structured Output, TriAttention Phase 1+2). These add clutter. Consider either:
- Moving implemented items to a separate "Completed" section at the bottom
- Or removing them entirely (the implementations are documented in their own files now)

### 3.3 PARALLELISM.md is 115 KB (2540+ lines) with zero code

This is a purely speculative design document with no implementation. At 115 KB it's by far the largest file in the docs. It's well-written design work, but its size and pre-implementation status should be clearly flagged. The `DOCUMENTATION.md` entry says "pre-implementation" which is correct, but the file itself doesn't state its status until the first line. Consider adding a prominent banner.

---

## 4. Improvements

### 4.1 API.md — Missing `Content-Type` header in curl examples

All curl examples omit `-H "Content-Type: application/json"`. While many servers infer JSON from the body, best practice is to include it. Also, no curl example uses `--json` (curl 7.82+) which is cleaner.

### 4.2 API.md — Missing error response documentation

No error responses are documented (400, 401, 404, 422, 500). Systems programmers integrating with the API need to know what errors look like.

### 4.3 CONTRIBUTING.md — Missing "How to Run Tests" section

The doc explains how to add backends, models, quants, etc. but never shows how to run the test suite. A simple section with `zig build test` and any test categories would help.

### 4.4 CONTRIBUTING.md — init() signature is stale

The template shows:
```zig
pub fn init(allocator: Allocator, fmt: Format, be: Backend, ctx_size: u32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, tiered_cache: ?*TieredKvCache) !YourModel
```
Should verify this matches the current model.zig vtable expectations.

### 4.5 MODELS.md — Missing Gemma 3 variants beyond 1B

The parameter table only shows "Gemma3 1B" but BENCHMARKS.md and TEST_MATRIX.md reference Gemma 3 4B, 12B, and 27B. The parameter table should include at least the 27B variant.

### 4.6 KERNELS.md — Footnote ⁸ "Removed" is orphaned

Footnote ⁸ says "Removed — ROCm rmsNormMulti now enabled and validated" but no cell in any table references ⁸.

### 4.7 BENCHMARKS.md — Methodology section should note version differences

The decode benchmarks are from 2026-03-24, KV/vision from 2026-04-14, megakernel from 2026-04-17. But the Performance Comparison in TEST_MATRIX.md from 2026-04-16 shows dramatically different numbers. A note explaining what changed between benchmark runs would prevent reader confusion.

### 4.8 TEST_MATRIX.md — Nemotron-H missing from model tests

The test matrix covers 9 models but Nemotron-H (which is listed in MODELS.md) doesn't appear in the Model × Backend table. It's referenced in MEGAKERNEL.md as having a true megakernel, so it presumably works.

### 4.9 API.md — /v1/messages endpoint not in ARCHITECTURE.md

ARCHITECTURE.md's server endpoint table lists `/v1/chat/completions`, `/v1/completions`, `/v1/responses` but not `/v1/messages` (Anthropic format). API.md documents it. The tables should match.

### 4.10 MEGAKERNEL.md — "11 fused FFN kernels" in kernel file table header

The Metal kernel file description says "11 fused FFN kernels" but the Tier 1 table lists 11 total (6 SiLU + 5 GELU = 11). However, there's also `fused_ffn_gate_up_silu_mlx_q4` in the table making it 11 SiLU+MLX variants + 5 GELU = possibly a discrepancy. Verify count against actual file.

---

## 5. Summary of Actions

### Must Fix
| # | Issue | File | Severity |
|---|---|---|---|
| 1.1 | WHT FMA count 16384 → ~640 in KV quant table | TODO.md | High |
| 1.2 | Contradictory Qwen 0.8B benchmarks (1.31× vs 0.51×) | BENCHMARKS.md + TEST_MATRIX.md | High |
| 2.1 | "Zero CPU delegates" contradicts KERNELS.md | TODO.md | Medium |
| 2.2 | "0 missing" kernel claims contradict KERNELS.md roadmap | TODO.md | Medium |

### Should Fix
| # | Issue | File |
|---|---|---|
| 1.3 | Stale "2026-04-08" date | MODELS.md |
| 2.3 | "12 kernels" vs "11 kernels" megakernel count | BENCHMARKS.md |
| 3.2 | Implemented items cluttering IDEAS.md | IDEAS.md |
| 4.6 | Orphaned footnote ⁸ | KERNELS.md |
| 4.9 | /v1/messages missing from ARCHITECTURE.md | ARCHITECTURE.md |

### Nice to Have
| # | Suggestion | File |
|---|---|---|
| 3.1 | Remove or enrich DOCUMENTATION.md | DOCUMENTATION.md |
| 4.1 | Add Content-Type headers to curl examples | API.md |
| 4.2 | Document error responses | API.md |
| 4.3 | Add "How to Run Tests" section | CONTRIBUTING.md |
| 4.5 | Add Gemma 3 larger variants to param table | MODELS.md |
| 4.8 | Add Nemotron-H to test matrix | TEST_MATRIX.md |

---

## 6. What's Good

- **ARCHITECTURE.md** is a standout reference doc. The module reference tables with hot-path flags, the inference pipeline diagram, and the quantization type tables are exactly what a contributor needs.
- **CONTRIBUTING.md** is genuinely useful — step-by-step templates for every extension point, including the composed megakernel path which most projects would bury in code comments.
- **KERNELS.md** is the source of truth it should be. The per-backend matrix with footnotes explaining fallback rationale is excellent.
- **MEGAKERNEL.md** clearly explains a complex three-tier system with good visual hierarchy.
- **API.md** is well-structured with curl examples for every endpoint.
