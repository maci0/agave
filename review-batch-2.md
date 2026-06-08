# Review Batch 2 — Tutorial Accuracy Review

Reviewed: `05-memory-and-caching.md`, `06-state-space-models.md`, `07-sampling.md`, `08-backends.md`
Cross-referenced against source files in `src/`.

---

## Issues Found

### [ERROR] 07-sampling.md: Sampling pipeline order — grammar mask position

**Tutorial claims (pipeline diagram and text):**
```
├─ logit bias (per-token additive adjust)
├─ repeat/frequency/presence penalties
├─ DRY penalty
├─ grammar mask (set invalid tokens to -∞)    ← shown 4th
├─ temperature scaling
```

**Source says:** In `src/server/server.zig` lines 2662–2691, grammar masking is applied **first**, before logit bias and all penalties:
```zig
// Grammar masking before sampling
if (use_grammar) { ... g.maskLogits(gs, logits, vocab_texts); }
if (sampling.logit_bias_count > 0) { math_ops.applyLogitBias(...); }
if (sampling.repetition_penalty != 1.0 ...) { math_ops.applyRepeatPenalty(...); }
if (sampling.dry_multiplier > 0 ...) { math_ops.applyDry(...); }
if (sampling.frequency_penalty != 0 ...) { math_ops.applyPenalties(...); }
```

**Fix:** Move grammar mask to the first position in both the Mermaid pipeline diagram and the text pipeline list (before logit bias).

---

### [ERROR] 07-sampling.md: Sampling pipeline order — XTC vs min-p ordering

**Tutorial claims (pipeline text):**
```
├─ temperature scaling
├─ XTC exclusion (drop top tokens randomly)   ← shown before min-p
├─ min-p filter (drop < min_p × max)
├─ top-k filter
```

**Source says:** In `src/server/server.zig` lines 2689–2691, min-p is applied **before** XTC:
```zig
if (sampling.min_p > 0) math_ops.applyMinP(logits, sampling.min_p);
if (sampling.xtc_probability > 0) math_ops.applyXtc(logits, ...);
next = math_ops.sampleToken(logits, ...);
```

**Fix:** Swap min-p and XTC in the pipeline diagram/text so min-p comes before XTC.

---

### [WARNING] 07-sampling.md: DRY `dry_allowed_length` default

**Tutorial claims:** "`dry_allowed_length` sets the minimum n-gram length to trigger (default 2 — penalize repeated bigrams and longer)."

**Source says:** In `src/server/json.zig` line 70, the default is indeed 2:
```zig
dry_allowed_length: u32 = 2,
```

**Status:** Confirmed correct. No issue.

---

### [ERROR] 07-sampling.md: Sampling pipeline Mermaid diagram — min-p and XTC order

The Mermaid `flowchart TD` diagram also shows the wrong order, with XTC node connecting to MinP node:
```
XTC["XTC Exclusion\nrandomly drop top tokens"] --> MinP["Min-P Filter\n..."]
```

**Fix:** Reverse to `MinP --> XTC` in the Mermaid diagram.

---

### [WARNING] 07-sampling.md: Temperature scaling position relative to min-p/XTC

**Tutorial claims** (pipeline text): temperature scaling happens before XTC and min-p as a separate step.

**Source says:** `sampleToken()` in `src/ops/math.zig` applies temperature internally as its first step. The server code calls `applyMinP()` and `applyXtc()` on raw logits (pre-temperature) before calling `sampleToken()` which then applies temperature.

This means min-p and XTC operate on **pre-temperature** logits, then temperature is applied inside `sampleToken`. The tutorial diagram shows temperature before XTC/min-p, but the actual flow is: min-p → XTC → sampleToken(temperature → top-k → softmax → top-p).

**Fix:** Move temperature scaling after min-p and XTC in the pipeline, or note that min-p/XTC operate on pre-temperature logits and temperature is internal to sampleToken.

---

### [ERROR] 08-backends.md: Backend tagged union claims "compile time" dispatch

**Tutorial claims:** "the `Backend` tagged union with `inline else` dispatch resolves **at compile time** (during compilation, not when the program runs — zero runtime overhead)"

**Source says:** `inline else` in a tagged union (`src/backend/backend.zig`) generates separate code paths for each variant at compile time, but the actual **dispatch** (which variant is active) is resolved at **runtime** based on the tag value. `inline else` eliminates vtable overhead and enables inlining of each variant, but it is NOT resolved at compile time — the switch still branches at runtime on the active tag. The "zero runtime overhead" claim is misleading; the overhead is minimal (a branch, not a function pointer chase) but not zero.

**Fix:** Change to: "resolved via `inline else` — the compiler generates specialized code for each backend, enabling inlining and eliminating vtable indirection. The active backend is selected at runtime, but per-variant code is optimized at compile time."

---

### [WARNING] 08-backends.md: ROCm compiled format

**Tutorial claims (table):** ROCm/HIP Compiled Format is "AMDGCN"

**Source says:** In `src/backend/rocm.zig` line 1473, ROCm uses `@import("../ops/ssm.zig")` CPU calls and GPU dispatch. The `BackendInfo` in the code would report "HSACO" (HIP Static-compiled Archive Object), which the tutorial does mention in the table as "Zig → HSACO" under Language but contradicts in the Compiled Format column which says "AMDGCN". The text block below the table also says "ROCm/HIP ──→ AMDGCN ──→ AMD only".

**Fix:** The table says "AMDGCN" for compiled format, and the Language column says "Zig → HSACO". These should be consistent. AMDGCN is the ISA, HSACO is the object format. Consider clarifying: Language="Zig → AMDGCN", Compiled Format="HSACO" (or both "AMDGCN").

---

## Verified Correct (no issues found)

- **05-memory-and-caching.md**: Default KV block size of 16 tokens matches `default_block_size: u16 = 16` in `manager.zig`. Sparse V threshold of 1e-6 matches `sparse_v_threshold: f32 = 1e-6` in `sdpa.zig`. CacheBlock fields (`keys`, `values`, `used`, `ref_count`, `access_count`) all match the struct in `manager.zig`. Tiered cache description (VRAM + RAM + SSD) matches `tiered.zig`. Shared prefix cost of 100× matches `shared_prefix_cost: f32 = 100.0` in `tiered.zig`. PagedKvView description matches source.

- **06-state-space-models.md**: `causalConv1dSilu` signature and behavior matches `ssm.zig`. Mamba-2 recurrence formula matches `mamba2Recurrence` in `ssm.zig`. `groupRmsNormSiluGate` with `norm_before_gate=False` convention (SiLU gate first, then RMS norm) matches source. DeltaNet `kqv_order` always false for Qwen3.5 matches `qwen35.zig:1025`. GPU dispatch claim (Metal/Vulkan/WebGPU/ROCm on GPU, CUDA CPU fallback) confirmed correct against all 5 backend implementations. Hybrid layer pattern table is accurate (Qwen3.5 `full_attn_interval=4`).

- **07-sampling.md**: `applyRepeatPenalty` divide/multiply logic matches `math.zig`. `applyMinP` log-threshold approach matches source. `sampleMirostat` behavior (bypasses top-k/p) matches source. `applyLogitBias` max 16 entries matches `max_logit_bias: usize = 16` in `server/json.zig`. `applyDry` and `applyXtc` semantics match source. Temperature=0 → argmax confirmed correct.

- **08-backends.md**: Tagged union with 6 variants (cpu, metal, vulkan, cuda, rocm, webgpu) confirmed in `backend.zig`. DDTree max_budget = 512 in `ddtree.zig` (matches reference constants, not mentioned in these tutorials). Backend dispatch pattern with `inline else` confirmed. UMA zero-copy description accurate.

---

## Coverage Status

- **Directly checked:** All 4 tutorial files, all 8 specified source files, plus `server/server.zig` and `server/json.zig` for sampling pipeline, `spec/ddtree.zig` for DDTree budget, and all 5 GPU backend deltaNet implementations.
- **Uncertain:** Exact kernel counts per backend (e.g., "70 pipelines" for Metal, "43 kernels" for CUDA) — not verified against actual kernel source files.
- **Not checked:** Mermaid diagram rendering correctness, link validity, APEX paper reference accuracy.
