# Phase 1: Correctness Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-21
**Phase:** 1-correctness-foundation
**Areas discussed:** Metal SDPA, CUDA GEMV, Model Verification, Testing, ROCm, DeepSeek

---

## Metal SDPA Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Debug existing first | GPU kernel + compute KV append exists. Likely a race condition or barrier issue. Faster if the fix is simple. | |
| Rewrite with FA-2 | FlashAttention-2 tiling from scratch. More work but proven algorithm, better long-term. | X |
| You decide | Claude picks based on what's found during investigation | |

**User's choice:** Rewrite with FA-2
**Notes:** User prefers the proven algorithm over debugging an existing broken kernel.

---

## CUDA Quantized GEMV Priority

| Option | Description | Selected |
|--------|-------------|----------|
| Q4_K first | Most common format for 7B-13B models. Highest impact. | |
| Q6_K first | Used for critical layers in mixed-quant models. | |
| All at once | Implement Q4_K, Q5_K, Q6_K, FP8 in one batch (they share dequant patterns) | X |

**User's choice:** All at once
**Notes:** All formats share dequantization patterns, so batch implementation makes sense.

---

## Golden Test Reference

| Option | Description | Selected |
|--------|-------------|----------|
| llama.cpp | Same GGUF format, deterministic with seed. Easiest comparison. | |
| HuggingFace | PyTorch reference. More authoritative but harder to match exactly. | |
| Both | llama.cpp for GGUF models, HuggingFace for SafeTensors models | X |

**User's choice:** Both
**Notes:** Use the appropriate reference for each model format.

---

## DeepSeek-R1-Qwen3-8B

| Option | Description | Selected |
|--------|-------------|----------|
| Qwen3 variant | Should load with existing Qwen3.5 code path (same arch, different weights) | X |
| New architecture | Needs its own model implementation | |
| Not sure, test it | Try loading with Qwen3.5 path first | |

**User's choice:** Qwen3 variant
**Notes:** User confirmed it uses Qwen3 architecture. File: `DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf`

---

## ROCm Verification

| Option | Description | Selected |
|--------|-------------|----------|
| All models that fit | Run every model that fits in 24GB VRAM on ROCm | X |
| Key models only | Test Gemma3 1B + Qwen3.5 0.8B | |
| Just build/run | Make sure ROCm backend compiles and runs at all | |

**User's choice:** All models that fit
**Notes:** ROCm machine at `maci@192.168.0.205`, 24GB VRAM. Be careful with model sizes.

---

## Claude's Discretion

- FA-2 block sizes and tiling strategy for Metal
- CUDA FP8 implementation approach (native intrinsics vs LUT)
- Order of model debugging
- Golden test tolerance thresholds
- Deterministic vs statistical seed comparison

## Deferred Ideas

None
