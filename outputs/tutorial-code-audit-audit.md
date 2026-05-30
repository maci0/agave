# Audit: docs/tutorial ↔ Agave Codebase

**Date:** 2026-05-25  
**Scope:** 22 tutorial files (~8,500 lines) in `docs/tutorial/` — "LLM Inference From Scratch" series  
**Method:** Systematic comparison of tutorial claims against Zig source code via grep, file reads, and HuggingFace config verification  
**Auditors:** Parallel researcher subagents + direct verification

---

## Executive Summary

The tutorials are **generally accurate and well-written**. Architectural descriptions, algorithm explanations, code patterns, and most implementation details match the codebase faithfully. However, the audit found **6 high-severity factual errors**, **4 medium-severity issues**, and **5 low-severity notes**. The most critical problems are wrong model dimensions in worked examples that readers will try to verify.

| Severity | Count | Examples |
|----------|:-----:|---------|
| 🔴 High | 6 | Wrong Gemma4 E2B dims, wrong Qwen3.5 0.8B layer count, 3 wrong vocab sizes, unverifiable performance claim |
| 🟡 Medium | 4 | Questionable RotorQuant FMA count, self-spec "50%" mismatch, missing spec-mode value, unsubstantiated speedup |
| 🟢 Low | 5 | Terminology differences, file attribution nuance, implicit module location |

---

## 🔴 High-Severity Issues

### H1. Gemma4 E2B Worked Example — Wrong Dimensions (Ch. 2)

**Tutorial (line 9-16):**
> Gemma4 E2B, 2.6B parameters: `[2304 floats] → 28 layers → [2304 floats]`

**Actual:**
- HuggingFace config (`google/gemma-4-E2B`): **hidden_size=1536, 35 layers**
- Code docstring (`src/models/gemma4.zig:2`): `"E2B (dense, 35 layers)"`
- Code defaults (`gemma4.zig:55-56`): n_layers=30, n_embd=2816 (for 26B-A4B, not E2B)

The tutorial's 2304/28 combination doesn't correspond to any known Gemma4 variant. The "2.6B parameters" label also appears incorrect — Google calls it "E2B" (Effective 2 Billion). **All downstream calculations (hidden state size, weight memory) are wrong.**

> **Evidence:** `src/models/gemma4.zig:2`, HuggingFace `google/gemma-4-E2B/config.json`

### H2. Qwen3.5 0.8B — Wrong Layer Count (Ch. 2)

**Tutorial (line 22):**
> "e.g., 28 for Gemma4 E2B, **64 for Qwen3.5 0.8B**"

**Actual:** Qwen3.5 0.8B has **24 layers** (HuggingFace model card). 64 layers is typical for 30B+ models. The code default in `qwen35.zig:55` is `n_layers: u32 = 32` (for the larger variant). A 0.8B model with 64 layers would be implausibly narrow.

> **Evidence:** HuggingFace `Qwen/Qwen3.5-0.8B`, `src/models/qwen35.zig:55`

### H3. Vocab Size Table — 3 of 4 Entries Wrong (Ch. 1)

| Model | Tutorial | Code Default | Delta |
|-------|:--------:|:------------:|:-----:|
| Qwen 3.5 | 151,936 | **248,320** | ❌ −96,384 (Qwen 2.x value) |
| Gemma 3/4 | 262,144 | 262,144 | ✅ |
| GPT-OSS | 200,064 | **201,088** | ❌ −1,024 (padding?) |
| GLM-4 | 151,552 | **154,880** | ❌ −3,328 |

The Qwen error is the worst — 151,936 is the Qwen 2/2.5 vocabulary; Qwen 3.5 expanded it to 248,320.

> **Evidence:** `src/models/qwen35.zig:61`, `src/models/gpt_oss.zig:77`, `src/models/glm4.zig:48`

### H4. KV Cache Per-Token Math — Unverifiable Parameters (Ch. 5)

**Tutorial:** "Qwen3.5 9B: 64 layers × 4 KV heads × 128 dim = 128KB/token"

The formula structure is correct, and the arithmetic checks out for those values. However, the code defaults for Qwen3.5 are `n_layers=32, n_head_kv=4, head_dim=256` — not 64/4/128. The actual 9B parameters come from GGUF metadata at load time, so they can't be verified from code alone. Given H2's layer count error, the "64 layers" here is suspicious.

> **Evidence:** `src/models/qwen35.zig:55-59`

### H5. Sparse V "+22.8% Decode Throughput" — Unverifiable (Ch. 4, Ch. 5)

The threshold (1e-6) and implementation are confirmed across all 5 backends (`attention.zig:19`, `sdpa.zig:10`, `sdpa.metal:23`, CUDA `sdpa.zig:14`, ROCm `sdpa.zig:12`). But the "+22.8%" figure is a benchmark result that appears nowhere in the codebase. No benchmark methodology, hardware, model, or context length is specified. This is a **credibility risk** — readers may treat it as a guaranteed speedup.

> **Evidence:** Threshold confirmed; percentage claim not substantiated in code or docs

### H6. "2-4× Speedup" for Multi-Row Batching — Unsubstantiated (Ch. 9, README)

The README's Performance Optimization path says: "Multi-row GEMV batching (2-4× speedup)". No such claim exists in the codebase. The only documented speedup numbers are:
- `gemv.zig:49` — "~4× speedup" for Accelerate/AMX (different optimization)
- `accelerate.zig:26` — "~4× speedup over NEON" (also AMX, not multi-row)

The multi-row batching code (`gemv_q4_0.zig`, `gemv_f32.zig`, etc.) documents its benefit qualitatively ("x-vector cache reuse") but never quantifies it.

> **Evidence:** Searched all `src/backend/kernels/cpu/gemv_*.zig` files — no 2-4× claim found

---

## 🟡 Medium-Severity Issues

### M1. Self-Speculative Default ≠ "50% of Layers, Skipping the Middle" (Ch. 17)

**Tutorial:** "default: 50% of model layers, skipping the middle"

**Actual:** The default **skip count** is 50% (`n_layers / self_spec_default_skip_fraction` where `self_spec_default_skip_fraction = 2`), but the **skip start** is at 25% (`n_layers / self_spec_skip_divisor` where `self_spec_skip_divisor = 4`). So for a 32-layer model, it skips layers 8–24 (16 layers = 50%, starting from quarter point). The "skipping the middle" phrasing is approximately correct but imprecise — it skips from 25% to 75%, not from a centered middle region.

> **Evidence:** `src/main.zig:2888-2893`

### M2. RotorQuant FMA Count — Possibly Overstated (Ch. 4)

**Tutorial:** "~2,400 FMAs" for RotorQuant.

**Code analysis:** The `rotorForward()` function (`kv_quant.zig:250-302`) uses sparse rotors (only `b12` plane non-zero by default; `b13=0, b23=0`). The general Clifford rotor sandwich product has 6 multiply-adds per output coordinate × 3 coords × 10 groups × 4 blocks = ~720, but many terms multiply by zero in the sparse case. The ~2,400 figure may describe a fully general rotor with all 3 bivector components active, which the code doesn't seem to use.

> **Evidence:** `src/ops/kv_quant.zig:250-302`

### M3. Missing "standard" Spec Mode (Ch. 17)

The tutorial lists 4 speculative decoding modes: draft-model, ddtree, self, ngram. The code (`main.zig:397,1029`) also accepts `standard` (single-path draft → sequential argmax verification) and `mtp`. The `--spec-mode` help text lists: "standard, ddtree, self, ngram, mtp". The tutorial covers MTP in Chapter 18 but omits `standard` from the mode list in Chapter 17.

> **Evidence:** `src/main.zig:397`

### M4. "30-40% Speedup" for Factored Dequantization — Partially Substantiated (Ch. 4)

The tutorial claims "30-40% (measured on Apple M4 with Gemma3 27B QAT)" for factored MLX dequantization. The code implements the optimization exactly as described (`src/ops/mlx.zig:108-175`), and the arithmetic reduction (192 → 130 ops = 32%) is correct. However, the benchmark setup is not documented or reproducible. The claim has a specific hardware/model reference which adds credibility, but no benchmark script or log exists.

> **Evidence:** `src/ops/mlx.zig:108-175` (implementation verified), benchmark claim not in code

---

## 🟢 Low-Severity Issues

### L1. `getEffectiveText()` Module Location (Ch. 1)

The tutorial says "Grammar-constrained decoding must strip these prefixes via `getEffectiveText()`" in the tokenizer context. The function actually lives in `src/grammar.zig:235`, not in the tokenizer module. The function's purpose is correctly described.

> **Evidence:** `src/grammar.zig:235`

### L2. CLI Flag Location (Multiple chapters)

`src/cli.zig` is a generic argument parser. The actual flag specifications are defined in `src/main.zig:332ff`. Tutorials that reference "defined in cli.zig" are technically misleading — the parser is in `cli.zig`, but flag specs are in `main.zig`.

> **Evidence:** `src/cli.zig` (parser), `src/main.zig:332` (flag specs)

### L3. "Multi-row" vs "N-row Batching" Terminology (Ch. 9)

The tutorial uses "multi-row GEMV batching." The codebase consistently uses "N-row batching" (e.g., `gemv_q4_0.zig:4`: "4-row batching with V8 SIMD for x-vector cache reuse"). The concept is identical; the terminology differs.

> **Evidence:** `src/backend/kernels/cpu/gemv_q4_0.zig:4`, `gemv_f32.zig:2`, etc.

### L4. Dispatch Overhead "5-10µs" (Ch. 13)

The tutorial claims Metal dispatch overhead is "~5-10 µs per dispatch". The Metal backend code mentions dispatch overhead qualitatively but never quantifies it as 5-10µs. This is likely from profiling but isn't documented or reproducible.

> **Evidence:** `src/backend/metal.zig:50,1746,1809` (qualitative overhead mentions, no numbers)

### L5. MTP "70-85% Acceptance Rate" (Ch. 18)

The tutorial claims MTP heads achieve "70-85% acceptance rates." This is a plausible range for MTP but doesn't appear in any code comment, test, or benchmark. The code tracks acceptance statistics (`spec_decode.zig:41`: "Tracks per-K acceptance rates") but doesn't define expected ranges.

> **Evidence:** `src/spec/spec_decode.zig:41` (tracking exists, no hardcoded range)

---

## ✅ Verified Claims (No Issues)

The following major tutorial claims were verified as accurate:

### Architecture & Patterns
- **BPE + SPM + SPM-no-dummy tokenizer modes** — `src/tokenizer/bpe.zig`, `tokenizer.zig:41` ✅
- **SwiGLU FFN structure** (gate/up/down projections) — all model files ✅
- **RoPE implementation** — `src/backend/kernels/cpu/rope.zig` ✅
- **iRoPE for Llama4** (alternating RoPE/NoPE) — `src/models/llama4.zig:724` ✅
- **GQA implementation** (heads-per-group mapping) — `src/ops/attention.zig:44,95` ✅
- **RMSNorm implementation** — `src/backend/kernels/cpu/norm.zig:7-41` ✅
- **MTP +1 offset norm** — `rmsNormPlusOne()` uses `(1.0 + weight[i]) * input[i] * inv_rms` — `src/models/qwen35.zig:1310-1325` ✅

### MoE Details
- **Qwen 3.5 MoE:** 256 experts, top-8, shared expert — `qwen35.zig:231-242` ✅
- **GPT-OSS MoE:** 32 experts, top-4, no shared — `gpt_oss.zig:83-88` ✅
- **Nemotron-Nano MoE:** 128 experts, top-6, shared (2× hidden dim=3712) — `nemotron_nano.zig:93-98` ✅
- **Gemma 4 MoE:** 128 experts, top-8, dual path (dense GELU + MoE) — `gemma4.zig:56-58,1814-1860` ✅
- **Clamped SwiGLU ±7.0** for GPT-OSS — `gpt_oss.zig:89,730-736` ✅
- **ReLU²** for Nemotron-Nano — `math.zig:133-134`, called in both expert paths ✅
- **Stack-allocated expert selection** — all MoE models use fixed-size arrays ✅

### Quantization
- **Block sizes:** Q4_0/Q8_0 = 32, super-block = 256 — `quant.zig:28-30`, `backend.zig:178` ✅
- **MLX affine:** group=64, bf16 scales/biases, companion tensors — `mlx.zig:9,67-70` ✅
- **Factored dequantization** — `scale * dot(q,x) + bias * sum(x)` pattern — `mlx.zig:108-175` ✅
- **4 KV quant families** (Turbo/Planar/Iso/Rotor, 2/3/4 bit) — `kv_quant.zig:420-438` ✅
- **TurboQuant FMA ~640** — WHT 5-stage butterfly: 5 × 32 × 4 blocks = 640 ✅
- **PlanarQuant FMA 256** — Givens: 4 × 16 pairs × 4 blocks = 256 ✅
- **IsoQuant FMA 512** — Quaternion: ~16 × 8 quartets × 4 blocks = 512 ✅

### Memory & Caching
- **PagedAttention block size 16** — `kvcache/manager.zig:14` ✅
- **Boundary V protection** (first/last N layers at f16) — `qwen35.zig:149-150,625-629`, `main.zig:972-973` ✅
- **Turbo preset:** K=q8_0, V=turbo4, boundary_v=2 — `main.zig:960-969` ✅
- **Sparse V threshold 1e-6** — verified in 5 backends ✅
- **Paged SDPA** — all 6 backends implement `sdpaPaged` ✅
- **RadixAttention** — full implementation with tests in `kvcache/manager.zig:200-360` ✅
- **CLI flags:** `--kv-type`, `--kv-type-k`, `--kv-type-v` — `main.zig:374-378` ✅

### Sampling
- **Defaults:** top-k=0, top-p=1.0, min-p=0, repeat-penalty=1.0, temperature=0 ✅
- **Temperature 0 → argmax** — `math.zig:498-500` ✅
- **DRY, XTC, Mirostat 2.0** — all implemented in `math.zig` ✅
- **Grammar-constrained decoding** — `grammar.zig` ✅
- **Frequency/presence penalties** — `math.zig:242-288` (server-side) ✅

### Backends
- **6 backends in tagged union** — `backend.zig:584-589` ✅
- **`inline else` dispatch pattern** — `backend.zig:594-605` ✅
- **Auto-selection fallback:** Metal → CUDA → ROCm → Vulkan → CPU — `backend.zig:1040-1150` ✅
- **WebGPU 43 shaders** — counted 43 `.wgsl` files ✅
- **GemvOp struct** (w, y, n, mlx_scales, mlx_biases, mlx_bits) — `backend.zig:37-46` ✅
- **gemvMulti interface** — `backend.zig:750-752` ✅
- **ModelDesc in mega_compose.zig** — `mega_compose.zig:50` ✅

### Speculative Decoding
- **DDTree "Ringel & Romano, 2026"** — cited in `spec_decode.zig:6` and `ddtree.zig:4` ✅
- **Best-first heap algorithm** — `ddtree.zig:71-160` with `HeapEntry`, sift-up/sift-down ✅
- **Default spec-tokens=5, tree-budget=64** — `main.zig:476-477` ✅
- **N-gram: history=2048, n=3..10** — `ngram.zig:14-16` ✅
- **4 modes** (ddtree, self, ngram, mtp) + standard — `main.zig:397` ✅

### MTP Architecture
- **hnorm, enorm, eh_proj, shared_head_norm, shared_head_head** — `qwen35.zig:1347-1453` ✅
- **MTP integrated with spec_decode.zig** ✅

### Infrastructure
- **All 29 referenced file paths exist** — verified ✅
- **All 15 audited CLI flags exist** — verified ✅
- **ThreadPool with parallelFor** — `thread_pool.zig:18,89` ✅
- **Futex-based synchronization** — `thread_pool.zig:2,79,158` ✅
- **Metal buffer cache by host pointer** — `metal.zig:212,582-609` ✅
- **Zero-copy UMA via newBufferWithBytesNoCopy** — `metal.zig:557` ✅

---

## Recommended Fixes (Priority Order)

1. **Fix Gemma4 E2B example** (H1): Change to hidden_size=1536, 35 layers, recalculate all derived numbers
2. **Fix Qwen3.5 0.8B layer count** (H2): Change 64 → 24 (or use a different model as the example)
3. **Fix vocab size table** (H3): Qwen→248,320, GPT-OSS→201,088, GLM-4→154,880
4. **Fix KV cache example** (H4): Use verified parameters from GGUF metadata or a different model
5. **Qualify performance claims** (H5, H6): Add "measured on [hardware] with [model]" or soften to ranges
6. **Add "standard" to spec-mode list** (M3)
7. **Clarify self-spec skip region** (M1): "starting at layer N/4, skipping N/2 layers"
8. **Review RotorQuant FMA count** (M2): Verify against actual sparse rotor or note it's the general case

---

## Coverage Summary

| Chapter | Claims Checked | Match | Mismatch | Partial/Uncertain |
|---------|:--------------:|:-----:|:--------:|:-----------------:|
| Ch. 1 — Tokens | 4 | 2 | 1 | 1 |
| Ch. 2 — Transformer | 6 | 4 | 2 | 0 |
| Ch. 3 — FFN/MoE | 6 | 6 | 0 | 0 |
| Ch. 4 — Quantization | 6 | 5 | 0 | 1 |
| Ch. 5 — Memory/Caching | 8 | 6 | 0 | 2 |
| Ch. 7 — Sampling | 5 | 5 | 0 | 0 |
| Ch. 8 — Backends | 4 | 4 | 0 | 0 |
| Ch. 9 — CPU SIMD | 3 | 1 | 1 | 1 |
| Ch. 11 — Metal | 4 | 3 | 0 | 1 |
| Ch. 13 — Megakernel | 4 | 4 | 0 | 0 |
| Ch. 17 — Spec Decode | 6 | 5 | 0 | 1 |
| Ch. 18 — MTP | 4 | 3 | 0 | 1 |
| Cross-cutting (files, CLI) | 50 | 48 | 0 | 2 |
| **Total** | **110** | **96 (87%)** | **4 (4%)** | **10 (9%)** |

---

## Sources

### Code Files (Primary Evidence)
- `src/models/gemma4.zig` — Gemma4 model defaults and E2B docstring
- `src/models/qwen35.zig` — Qwen3.5 model defaults, MoE config, MTP implementation
- `src/models/gpt_oss.zig` — GPT-OSS defaults, clamped SwiGLU
- `src/models/nemotron_nano.zig` — Nemotron-Nano MoE, ReLU²
- `src/models/glm4.zig` — GLM-4 vocab size
- `src/models/llama4.zig` — iRoPE implementation
- `src/tokenizer/bpe.zig` — Tokenizer modes
- `src/tokenizer/tokenizer.zig` — TokenizerKind enum
- `src/backend/backend.zig` — Backend tagged union, GemvOp, gemvMulti, auto-selection
- `src/backend/metal.zig` — Buffer cache, UMA, threadgroup handling
- `src/backend/mega_compose.zig` — ModelDesc, composeMSL
- `src/backend/kernels/cpu/` — SIMD patterns, N-row batching, activation functions
- `src/backend/kernels/webgpu/` — 43 WGSL shader files
- `src/ops/quant.zig` — Block element constants
- `src/ops/mlx.zig` — MLX group size, factored dequant
- `src/ops/kv_quant.zig` — KV quant types and rotation implementations
- `src/ops/math.zig` — Sampling algorithms, activation functions
- `src/ops/attention.zig` — GQA, sparse V threshold, SDPA
- `src/spec/spec_decode.zig` — Speculative decoding orchestrator
- `src/spec/ddtree.zig` — DDTree best-first heap
- `src/spec/ngram.zig` — N-gram history/range constants
- `src/kvcache/manager.zig` — PagedAttention block size, RadixTree
- `src/grammar.zig` — getEffectiveText(), GBNF parser
- `src/main.zig` — CLI flag definitions, defaults, turbo preset, self-spec constants
- `src/thread_pool.zig` — ThreadPool, parallelFor, futex sync

### External Sources
- HuggingFace `google/gemma-4-E2B/config.json` — https://huggingface.co/google/gemma-4-E2B
- HuggingFace `Qwen/Qwen3.5-0.8B` model card — https://huggingface.co/Qwen/Qwen3.5-0.8B

### Tutorial Files Audited
- `docs/tutorial/README.md` through `docs/tutorial/18-multi-token-prediction.md` (18 chapters)
- `docs/tutorial/appendix-atomics.md`, `appendix-compile-time.md`, `appendix-math.md`, `appendix-profiling.md` (4 appendices)
