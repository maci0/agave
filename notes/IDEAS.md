# Research Ideas for Agave Performance

## 1. Cross-Model KV Cache Transfer (arXiv 2608.03893)

**Paper:** "Cross-Model KV Cache Transfer in LLM Families" (NVIDIA, Aug 2026)
**Status:** IMPLEMENTING

### Core Idea
Transfer KV cache between models via per-head ridge regression mapper.
Skip re-prefill when switching models or sharing context with draft models.

### Application to Agave MTP
Our MTP has 6% acceptance because MTP attention has NO context history.
The main model's KV cache IS the context history but uses different weights.
Ridge-map main KV → MTP KV space to give MTP full conversation context.

### Implementation Plan
**Phase 1: Identity mapper (zero-cost baseline)**
- MTP directly reuses main model's KV cache (no transformation)
- Same kv_lora_rank=512, same n_head_kv=1 → dimensions match
- Test: does acceptance rate improve from 6%?

**Phase 2: Ridge mapper (if identity helps)**
- Offline: calibrate ridge weights from paired KV data
- Runtime: MTP_KV[l,h] = Main_KV[selected_layers, h] @ W + b
- Cost: ~0.5ms per token of context (one GEMV per head per layer)

**Phase 3: Top-k source layer selection**  
- Each MTP layer draws from k most predictive main model layers
- Greedy forward selection by R² on calibration set
- k=4-6 typically sufficient (paper: 79% K variance, 65% V variance)

### Key Design Choices (from paper)
1. Strip RoPE before mapping → position-invariant fit
2. Ridge λ=0.01 for numerical stability
3. 500 calibration sequences × 1024 tokens = ~128K tokens per head
4. Closed-form solve: W* = (X'X + λI)^-1 X'Y

### Expected Impact
- MTP acceptance: 6% → 30-50% (full context)  
- Prose throughput: 0.9 → 2-3 tok/s
- Combined with suffix: potentially 3-5 tok/s prose

## 2. Batched Verification via GEMM (mlx-dspark)
**Status:** Attempted, reverted (MXFP4 GEMM = N×GEMV on CPU)
**When viable:** With Metal graph capture or F32 weight cache

## 3. Metal Graph Capture
**Status:** Not started (~1000 lines)
**Impact:** 2× compute via batched GPU dispatch

## 4. Asymmetric Quantization (Q8 attn + Q4 experts)  
**Status:** Not started
**Impact:** ~120GB model (vs 155GB MXFP4), better quality
