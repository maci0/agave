# Agave Tutorial Documentation Review

**Date:** 2025-06-03  
**Scope:** All 23 tutorial files in `docs/tutorial/`  
**Method:** Parallel cross-reference of every factual claim against source code  

---

## Summary

| Severity | Count |
|----------|-------|
| **ERROR** | 18 |
| **WARNING** | 10 |
| **Clean files** | 10 of 23 |

**Files with issues:** 01 (0), 02 (2), 03 (3), 04 (0), 05 (0), 06 (0), 07 (4), 08 (2), 09 (0), 10 (2), 11 (1), 12 (3), 13 (0), 14 (0), 15 (1), 16 (1), 17 (2), 18 (1), 19 (1), appendix-atomics (0), appendix-compile-time (2), appendix-math (2), appendix-profiling (4)

---

## ERRORS (18)

### 02-the-transformer.md

**[ERROR] Gemma4 E2B hidden state size**  
Tutorial claims: `"Token 15496 → embed → [2304 floats] → 35 layers → [2304 floats]"` and `"2.6B parameters"`  
Source says: Gemma 4 E2B has `hidden_size=1536` (HuggingFace config.json; `n_embd` read from GGUF metadata). Hidden state is 1536 floats, not 2304. Model is ~2B params, not 2.6B.  
Fix: Replace all `[2304 floats]` → `[1536 floats]`, `2.6B` → `2B`, `2304 floats = 9 KB` → `1536 floats = 6 KB`.

**[ERROR] RoPE dimension pairing visualization**  
Tutorial claims: Adjacent pairs `[x0,x1], [x2,x3], ...` (interleaved layout).  
Source says: `src/backend/kernels/cpu/rope.zig:9-10` — split-complex layout: pairs `[i, i+half]`. For 8 dims: plane0=(x0,x4), plane1=(x1,x5), etc.  
Fix: Update visualization and Mermaid diagram to show split-complex pairing.

### 02-the-transformer.md (cont.)

**[ERROR] Qwen3.5 partial RoPE dimensions**  
Tutorial claims: `"first 78 out of 128"`  
Source says: `src/models/qwen35.zig:63` — `rope_dim: u32 = 64`. No variant uses 78.  
Fix: Change to `"first 64 out of 128"`.

### 03-feed-forward-networks.md

**[ERROR] Gemma4 E2B FFN dimensions**  
Tutorial claims: `"2304 → 9,216 in Gemma4 E2B"`  
Source says: E2B has hidden_size=1536, intermediate_size=6144. Expansion is 1536 → 6144.  
Fix: Change to `"1536 → 6,144 in Gemma4 E2B"`.

**[ERROR] GPT-OSS MoE routing inline comment**  
Tutorial claims: `"GPT-OSS: sigmoid+top-4"` (in code comment).  
Source says: `src/models/gpt_oss.zig:668-680` — GPT-OSS uses `topKExperts` then softmax normalization.  
Fix: Change to `"GPT-OSS: softmax+top-4"`. (The chapter's own table correctly says softmax — only the inline comment is wrong.)

**[ERROR] Mermaid FFN dimensions**  
Tutorial claims: `"2304 → 12288"` in the Mermaid diagram.  
Source says: 2304 doesn't match any model's `n_embd`. The combination 2304/12288 is fictitious.  
Fix: Use real dimensions — Gemma4 E2B: `1536 → 6144`, or Qwen3.5 9B: `4096 → 12288`.

### 07-sampling.md

**[ERROR] Grammar mask position in pipeline**  
Tutorial claims: Grammar mask is 4th (after logit bias, repeat penalty, DRY penalty).  
Source says: `src/server/server.zig:2665` — grammar masking is applied **first**, before all penalties.  
Fix: Move grammar mask to first position in pipeline diagram and text.

**[ERROR] XTC vs min-p ordering**  
Tutorial claims: XTC before min-p.  
Source says: `src/server/server.zig:2689-2690` — min-p before XTC.  
Fix: Swap ordering: min-p first, then XTC.

**[ERROR] Mermaid pipeline diagram min-p/XTC order**  
Tutorial claims: `XTC --> MinP` in flowchart.  
Source says: Same as above — min-p is applied first.  
Fix: Reverse to `MinP --> XTC`.

### 08-backends.md

**[ERROR] Backend dispatch "compile time" claim**  
Tutorial claims: `"resolves at compile time (during compilation, not when the program runs — zero runtime overhead)"`  
Source says: `inline else` generates specialized code per variant at compile time, but the dispatch (which variant is active) is resolved at **runtime** via tag branch. Not zero overhead — just very low.  
Fix: Change to: "the compiler generates specialized code for each backend, enabling inlining and eliminating vtable indirection. The active backend is selected at runtime, but per-variant code is optimized at compile time."

### 10-memory-safety.md

**[ERROR] Pitfall 1 — `defer` in a loop**  
Tutorial claims: `defer` inside a `for` loop accumulates and all run at function exit.  
Source says: In Zig, `defer` inside a `for` body is scoped to the loop iteration block. `file.close()` runs at the end of each iteration. The "BAD" example is actually correct and idiomatic.  
Fix: Remove or completely rewrite Pitfall 1. The example teaches the wrong lesson.

**[ERROR] Pitfall 2 — Conditional defer**  
Tutorial claims: `defer` inside an `if` block runs at function exit, causing use-after-free.  
Source says: `defer` inside an `if` block is scoped to that block. It runs at the closing `}` of the `if`, not at function exit. No use-after-free occurs.  
Fix: Rewrite the example. The actual pitfall is that the allocation is freed when the `if` block exits, so it can't be used outside the block (a scoping issue, not a safety bug).

### 12-cpu-parallelism.md

**[ERROR] `parallelFor` operation ordering**  
Tutorial claims: Task descriptor posted before CAS on `active`.  
Source says: `src/thread_pool.zig:109` — CAS on `active` happens **first**, before posting task fields. This ordering prevents workers from reading stale descriptors.  
Fix: Reorder to match: CAS first, then task field writes.

**[ERROR] Full implementation uses `active.store` instead of `cmpxchgWeak`**  
Tutorial claims: `self.active.store(@intCast(self.n_workers), .release);`  
Source says: `src/thread_pool.zig:109` — uses `cmpxchgWeak` with fallback to inline execution if pool is busy. The tutorial drops concurrent-use detection and introduces a race.  
Fix: Replace `active.store()` with the actual `cmpxchgWeak` pattern.

### 15-chat-templates.md

**[ERROR] `Arch.detect` parameter type**  
Tutorial claims: `Arch.detect(fmt)` — takes a `Format`.  
Source says: `Arch.detect` takes a `[]const u8` string. In `src/main.zig:1731`: `Arch.detect(arch_str)` where `arch_str = fmt.getMetaStr("general.architecture")`.  
Fix: Replace with two-step: get arch string from format, then detect.

### 16-recipe-system.md

**[ERROR] `Recipe.match` uses `displayName()`**  
Tutorial claims: `Recipe.match(arch.displayName(), backend_name, quant)`  
Source says: `src/main.zig:1811` — uses `arch_str` (raw GGUF string like `"qwen35"`), not `displayName()` (which returns `"Qwen 3.5"`). Using `displayName()` would break `startsWith` matching.  
Fix: Replace `arch.displayName()` with `arch_str`.

### 17-speculative-decoding.md

**[ERROR] Fabricated "Cooldown" subsection**  
Tutorial claims: Cooldown mechanism with `adaptive_window=8`, 25% threshold, 8-step bypass.  
Source says: No cooldown mechanism exists anywhere in `src/spec/`. `grep -rn "cooldown\|adaptive_window\|bypass" src/spec/` returns nothing. The only adaptive mechanism is `optimalK()`.  
Fix: Remove the entire "### Cooldown" subsection and related Mermaid nodes (`LowAccept`, `Cooldown`, `SingleDecode`).

### appendix-compile-time.md

**[ERROR] MetalBackend conditional missing OS guard**  
Tutorial claims: `pub const MetalBackend = if (build_options.enable_metal) @import("metal.zig").MetalBackend else NullBackend;`  
Source says: `src/backend/backend.zig:458` — `if (build_options.enable_metal and builtin.os.tag == .macos)`. The OS check is critical for cross-compilation.  
Fix: Add `and builtin.os.tag == .macos` to the condition.

---

## WARNINGS (10)

### 07-sampling.md

**[WARNING] Temperature scaling position**  
Tutorial shows temperature before XTC/min-p. Source: min-p and XTC operate on pre-temperature logits; temperature is applied inside `sampleToken()`.  
Fix: Note that min-p/XTC operate on pre-temperature logits, or move temperature after them.

### 08-backends.md

**[WARNING] ROCm compiled format inconsistency**  
Table says "AMDGCN" for compiled format, but Language column says "Zig → HSACO". AMDGCN is the ISA; HSACO is the object format.  
Fix: Make columns consistent.

### 11-metal-backend-internals.md

**[WARNING] Contradictory sync reduction numbers**  
Line ~430: "eliminated 16 syncs/token". Line ~540: "from 18 → 1" (a reduction of 17). These contradict.  
Fix: Make consistent — either 17 eliminated (18 → 1) or 16 eliminated (18 → 2).

### 12-cpu-parallelism.md

**[WARNING] Mermaid flowchart wrong operation order**  
Shows: Post descriptor → CAS → generation++. Should be: CAS → post descriptor → generation++.

### 17-speculative-decoding.md

**[WARNING] N-gram "ring buffer" terminology**  
Tutorial says "ring buffer" for the n-gram history.  
Source: Uses linear array with shift-by-half compaction (`copyForwards` from end to front), not modular ring buffer arithmetic.  
Note: The source comment in `ngram.zig:18` itself says "ring buffer", so this is really a source-code comment issue that propagated. Still technically inaccurate.  
Fix: Change "ring buffer" to "history buffer" in both the tutorial and the source comment.

### 18-multi-token-prediction.md

**[WARNING] Transformer layer counts**  
Tutorial claims: "64 layers for a 0.8B model". Source: `src/models/qwen35.zig:55` — `n_layers: u32 = 32` (default for 9B). The 0.8B Qwen3.5 has 24 layers per model card, not 64.  
Fix: Use realistic values (e.g., "24 layers for a 0.8B model").

### 19-pflash-and-block-sparse.md

**[WARNING] Complexity table uses n² instead of n²/2**  
Table shows 16B dot products for 128K tokens. The tutorial's own intro correctly states "128K × 128K / 2 = 8 billion" (causal masking halves work). The "200×" ratio should be ~100×.  
Fix: Halve all "Full attention" entries and adjust ratio.

### appendix-compile-time.md

**[WARNING] @embedFile list incomplete**  
Shows 8 MSL files, says "all MSL files". Source: `src/backend/metal.zig:24-40` has 17 files.  
Fix: Add `// ... more files ...` or remove the word "all".

### appendix-math.md

**[WARNING] Top-K sampling claims renormalization**  
Tutorial: Step 3 includes "renormalize". Source: `src/ops/math.zig` uses unnormalized sampling — scales the random threshold by the sum instead.  
Fix: Replace with "no renormalization — sampling scales the threshold instead".

**[WARNING] MXFP4 lookup table index 8**  
Tutorial shows `0.0` at index 8. Source: `src/ops/quant.zig:44` — index 8 is `-0.0` (sign bit set).  
Fix: Change to `-0.0`.

### appendix-profiling.md

**[ERROR] `PerfCounters.end()` timestamp function**  
Tutorial claims: `std.time.nanoTimestamp()`.  
Source says: `src/perf.zig:72-73` uses private `nanoTimestamp()` helper (calls `clock_gettime` directly to avoid `std.time` overhead).  
Fix: Change to `nanoTimestamp()`.

**[ERROR] Metal counter guard condition**  
Tutorial claims: `if (self.be == .metal and self.kv_seq_len == 1)`.  
Source says: `src/models/qwen35.zig:1656` — outer guard is `self.perf.enabled`, with a comptime `builtin.os.tag == .macos` switch inside. Tutorial omits the profiling-enabled check.  
Fix: Add `self.perf.enabled` check and show comptime OS guard.

**[WARNING] `gemvMlxQ` code structure**  
Tutorial shows a switch-on-bits for pipeline selection. Source: `src/backend/metal.zig:1880` uses an `if` guard for validation; the actual switch is on `wpg`.  
Fix: Mark as simplified illustration or replace with actual pattern.

**[WARNING] Instrumented operation code pattern**  
Tutorial shows inline `self.be.sync()`. Source uses `self.syncProfile()` which is conditional on `self.perf.enabled`.  
Fix: Show `self.syncProfile()` to avoid implying unconditional sync.

---

## Clean Files (no issues found)

These tutorials were cross-referenced and contain no factual errors:

1. **01-tokens-and-text.md** — tokenizer, embedding, vocab sizes all correct
2. **04-quantization.md** — all block sizes, byte counts, quant formats match `gguf.zig`
3. **05-memory-and-caching.md** — KV block size=16, sparse_v_threshold=1e-6, tiered cache all correct
4. **06-state-space-models.md** — SSM kernels, Mamba-2 recurrence, hybrid patterns all correct
5. **09-cpu-simd-optimization.md** — SIMD ops, NR values, sparse threshold, Q4_0 layout all correct
6. **13-batched-dispatch-and-fusion.md** — megakernel counts, line counts, API signatures all correct
7. **14-format-conventions.md** — GGUF layout, dim reversal, block sizes all correct
8. **appendix-atomics.md** — thread pool atomics, memory ordering, CAS patterns all correct
9. **docs/tutorial/README.md** — not reviewed (index file only)

---

## Priority Triage

### Must fix before publishing
1. **10-memory-safety.md** Pitfalls 1 & 2 — teach objectively wrong Zig semantics
2. **17-speculative-decoding.md** Cooldown subsection — entirely fabricated mechanism
3. **02-the-transformer.md** E2B dimensions — wrong hidden size propagates through multiple sections
4. **07-sampling.md** pipeline ordering — 3 errors in execution order
5. **12-cpu-parallelism.md** thread pool ordering — wrong CAS placement and missing race detection

### Should fix
6. **03-feed-forward-networks.md** — E2B dimensions and GPT-OSS routing comment
7. **15/16** — API usage in code examples (`Arch.detect`, `Recipe.match`)
8. **08-backends.md** — "compile time" dispatch claim
9. **appendix-compile-time.md** — missing OS guard on MetalBackend
10. **appendix-profiling.md** — timestamp function and guard condition

### Nice to fix
11. All remaining [WARNING] items — terminology, approximations, incomplete lists
