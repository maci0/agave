# Tutorial Cross-Cutting Claims Audit

Audited: 2026-05-25

## 1. File Path References

All 29 files verified with `ls`/`test -f`.

| # | Path | Status |
|---|------|--------|
| 1 | `src/tokenizer/bpe.zig` | ✅ MATCH |
| 2 | `src/backend/backend.zig` | ✅ MATCH |
| 3 | `src/backend/webgpu.zig` | ✅ MATCH |
| 4 | `src/backend/accelerate.zig` | ✅ MATCH |
| 5 | `src/models/model.zig` | ✅ MATCH |
| 6 | `src/models/llama4.zig` | ✅ MATCH |
| 7 | `src/ops/quant.zig` | ✅ MATCH |
| 8 | `src/ops/attention.zig` | ✅ MATCH |
| 9 | `src/ops/ssm.zig` | ✅ MATCH |
| 10 | `src/chat_template.zig` | ✅ MATCH |
| 11 | `src/recipe.zig` | ✅ MATCH |
| 12 | `src/backend/mega_compose.zig` | ✅ MATCH |
| 13 | `src/backend/megakernel.zig` | ✅ MATCH |
| 14 | `src/spec/ddtree.zig` | ✅ MATCH |
| 15 | `src/spec/spec_decode.zig` | ✅ MATCH |
| 16 | `src/spec/ngram.zig` | ✅ MATCH |
| 17 | `src/grammar.zig` | ✅ MATCH |
| 18 | `src/server/server.zig` | ✅ MATCH |
| 19 | `src/ops/math.zig` | ✅ MATCH |
| 20 | `src/ops/kv_quant.zig` | ✅ MATCH |
| 21 | `src/kvcache/manager.zig` | ✅ MATCH |
| 22 | `src/kvcache/tiered.zig` | ✅ MATCH |
| 23 | `src/parallel/transport.zig` | ✅ MATCH |
| 24 | `src/parallel/discovery.zig` | ✅ MATCH |
| 25 | `src/devices/discovery.zig` | ✅ MATCH |
| 26 | `src/pull.zig` | ✅ MATCH |
| 27 | `src/cli.zig` | ✅ MATCH |
| 28 | `src/term.zig` | ✅ MATCH |
| 29 | `src/wasm_entry.zig` | ✅ MATCH |

**Result: 29/29 MATCH**

---

## 2. CLI Flags

> **Note:** `src/cli.zig` is a **generic argument parser** (ArgSpec, ParseResult). Application-specific flag definitions live in `src/main.zig` at line ~332 (`const cli_specs = [_]cli_mod.ArgSpec{...}`). If tutorials say "defined in cli.zig", this is technically a MISMATCH — the parser lives in `cli.zig`, but the flag *specs* are in `main.zig`.

| # | Flag | Status | Location |
|---|------|--------|----------|
| 1 | `--draft-model` | ✅ MATCH | `main.zig:394` — `.long = "draft-model", .kind = .option` |
| 2 | `--spec-mode` (ddtree, self, ngram, mtp) | ✅ MATCH | `main.zig:397` — modes: standard, ddtree, self, ngram, mtp. Note: "standard" also listed, tutorials may omit it. |
| 3 | `--spec-tokens` | ✅ MATCH | `main.zig:395` — short alias `-K`, default 5 |
| 4 | `--tree-budget` | ✅ MATCH | `main.zig:396` — default 64 |
| 5 | `--draft-layers` | ✅ MATCH | `main.zig:398` — default auto (null) |
| 6 | `--kv-type-k`, `--kv-type-v`, `--kv-type` | ✅ MATCH | `main.zig:374-376` — also has `--cache-type-k`/`--cache-type-v` aliases (`main.zig:377-378`) |
| 7 | `--backend` | ✅ MATCH | `main.zig:221` (in cli_specs array, confirmed by grep) |
| 8 | `--prefill-batch-size` | ✅ MATCH | `main.zig:372` — default 512 |
| 9 | `--megakernel` | ✅ MATCH | `main.zig:404` — boolean flag (no value), help: "3→1 dispatch per layer" |
| 10 | `--ctx-size` | ✅ MATCH | `main.zig:369` — accepts integer or "auto", default 4096 or model limit |
| 11 | `--mirostat-mode` | ✅ MATCH | `main.zig:350` — values: 0 (disabled), 2 (Mirostat 2.0) |
| 12 | `--dry-multiplier` | ✅ MATCH | `main.zig:346` — default 0 (disabled) |
| 13 | `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty` | ✅ MATCH | `main.zig:342-345` |
| 14 | `--temperature` (or `-t`) | ✅ MATCH | `main.zig:341` — `.short = 't'`, default 0 (greedy) |
| 15 | `--serve` | ✅ MATCH | `main.zig:386` — `.short = 's'`, boolean flag, enables HTTP server |

**Result: 15/15 MATCH (all flags exist in `main.zig`, not `cli.zig` itself)**

### ⚠️ Nuance for tutorials

If tutorials claim flags are "defined in cli.zig", that is misleading. The file `src/cli.zig` exports the generic `ArgSpec`/`parse`/`ParseResult` types. The actual flag specifications are in `src/main.zig` at line 332ff. Tutorials should say the flags are "parsed using the CLI module from cli.zig" or "defined in main.zig".

---

## 3. CPU SIMD Claims (Chapter 9)

### 3.1 `@Vector(8, f32)` usage pattern

**Status: ✅ MATCH**

Extensively used across the CPU kernel layer. Found in 20+ files under `src/backend/kernels/cpu/`:

- `gemv_q4_0.zig:8` — `const V8 = @Vector(8, f32);`
- `gemv_f32.zig:4` — same
- `gemv_f16.zig:4` — same
- `gemv_bf16.zig:5` — same
- `gemv_fp8.zig:6` — same
- `gemv_q8_0.zig:7` — same
- `gemv_q4_k.zig:11` — same
- `gemv_q5_k.zig:8` — same
- `gemv_q6_k.zig:7` — same
- `sdpa.zig:4`, `norm.zig:3`, `elementwise.zig:3`, `activation.zig:3`, `rope.zig:28`, `deltanet.zig:8` — same
- Also used in `spec/spec_decode.zig` (lines 395, 436, 467) and `parallel/transport.zig:31`

The pattern is consistent: `const V8 = @Vector(8, f32);` used as a local type alias for 256-bit SIMD (matching AVX register width).

### 3.2 Multi-row GEMV batching

**Status: ⚠️ PARTIAL MATCH (terminology differs)**

The tutorial claim refers to "multi-row" GEMV batching. The codebase uses the term **"N-row batching"** or **"N-row batched"** instead of "multi-row":

- `gemv_q4_0.zig:4` — `//! 4-row batching with V8 SIMD for x-vector cache reuse.`
- `gemv_f32.zig:2` — `//! 4-row batching with V8 SIMD.`
- `gemv_f16.zig:2` — `//! 4-row batching with V8 SIMD.`
- `gemv_fp8.zig:3` — `//! 4-row batching with V8 SIMD`
- `gemv_q_small.zig:2` — `//! Q4_1, Q5_0, Q2_K, Q3_K — scalar implementations with 2-row batching.`
- `gemv_fp4.zig:3` — `//! 2-row batched to share x-vector cache reads.`
- `gemv_q5_k.zig:32` — `// Process 2 rows at a time for x-vector cache reuse.`

The concept matches (processing multiple weight rows per pass to reuse the x-vector in cache), but the codebase never uses the term `multiRow` or `multi_row`. It uses "N-row batching" (where N=2 or N=4 depending on the quantization format).

There is also a separate "Batched GEMV" concept (`gemvMulti` in `cpu.zig:600`) that fuses multiple independent GEMV ops into one `parallelFor` — this is different from the intra-kernel row batching.

### 3.3 "2-4× speedup" claim for multi-row batching

**Status: ❌ NOT FOUND in source**

No explicit speedup claim for multi-row batching was found anywhere in the codebase. The code describes the *benefit* as "x-vector cache reuse" and "instruction-level parallelism" but does not quantify it with a 2-4× figure. The only speedup claims found in code:

- `gemv.zig:49` — "~4× speedup" for Accelerate.framework (AMX) on macOS, **not** for multi-row batching
- `accelerate.zig:26` — "~4× speedup over NEON" for AMX, again different context
- `scheduler.zig:164` — "~2x prefill speedup" for hybrid SSM models, unrelated

The "2-4× speedup" for multi-row batching appears to be an **unsubstantiated tutorial claim** unless it refers to benchmark results documented elsewhere.

---

## 4. CPU Parallelism Claims (Chapter 12)

### 4.1 ThreadPool implementation

**Status: ✅ MATCH**

`src/thread_pool.zig` contains the `ThreadPool` struct (line 18):

```
pub const ThreadPool = struct { ... }
```

Methods: `init(n)`, `spawn(io)`, `deinit()`, `parallelFor(...)`.

Used in:
- `src/main.zig:30` — `const ThreadPool = @import("thread_pool.zig").ThreadPool;`
- `src/backend/cpu.zig:19` — imported and used at `cpu.zig:241` (`pool: ?*ThreadPool`)
- `src/backend/metal.zig:20` — imported for CPU-side parallel work
- `src/backend/backend.zig:26,939,954` — creates pool with `ThreadPool.init(n_workers)`
- `src/models/model.zig:13,702` — `setPool` method
- `src/ops/split_attention.zig:22,211` — used for parallel SDPA

### 4.2 Futex-based synchronization

**Status: ✅ MATCH**

`src/thread_pool.zig` is explicitly documented as futex-based:

- Line 2: `//! Workers sleep on Io.futex when idle. Main thread participates in work.`
- Line 17: `/// Futex-based thread pool for parallel GEMV and other data-parallel ops.`
- Line 22: `/// Io context for futex operations. Set during spawn().`
- Line 79: `self.io.futexWake(u32, &self.generation.raw, ...)`
- Line 130: `// ...spinning avoids futex syscall overhead...`
- Line 151: `/// Worker thread main loop. Sleeps on generation futex, wakes to do work.`
- Line 158: `pool.io.futexWaitUncancelable(u32, &pool.generation.raw, local_gen);`

Also used in `src/kvcache/prefetch.zig` (lines 9, 28, 38, 65, 102, 127) with the same pattern.

### 4.3 `parallelFor` function

**Status: ✅ MATCH**

Defined at `src/thread_pool.zig:89`:

```zig
pub fn parallelFor(self: *ThreadPool, ...) void { ... }
```

Extensively called in `src/backend/cpu.zig`:
- Line 269: `pool.parallelFor(n, parallel_grain, ...)` — single GEMV
- Line 624: `pool.parallelFor(total_n, parallel_grain, ...)` — batched GEMV
- Lines 732, 758, 803, 827, 902, 953: SDPA head-parallel dispatch
- Line 1042: DeltaNet head-parallel dispatch

Also in `src/models/vision.zig:990` and `src/ops/split_attention.zig:241,286`.

---

## Summary

| Category | Checked | Match | Partial | Not Found |
|----------|---------|-------|---------|-----------|
| File paths | 29 | 29 | 0 | 0 |
| CLI flags | 15 | 15 | 0 | 0 |
| CPU SIMD (Ch.9) | 3 | 1 | 1 | 1 |
| CPU Parallelism (Ch.12) | 3 | 3 | 0 | 0 |
| **Total** | **50** | **48** | **1** | **1** |

### Issues Requiring Tutorial Corrections

1. **⚠️ CLI flag location:** If tutorials say flags are "in cli.zig", they should clarify: the parser is in `cli.zig`, flag specs are in `main.zig:332`.

2. **⚠️ "Multi-row" terminology (PARTIAL):** Codebase uses "N-row batching" (2-row, 4-row), not "multi-row". The concept is identical but the grep-able identifier differs. Tutorial should use the actual terminology or note both.

3. **❌ "2-4× speedup" for multi-row batching (NOT FOUND):** No such claim exists in the source code. The codebase documents benefits qualitatively ("cache reuse", "instruction-level parallelism") but never quantifies multi-row batching speedup as 2-4×. The only `4×` claim in CPU kernels is for Accelerate/AMX vs NEON (a completely different optimization). This claim should be either substantiated with benchmarks or removed/softened.

---

## Coverage Status

- ✅ **Directly checked:** All 29 file paths, all 15 CLI flags (read full `cli.zig` + grep `main.zig`), all 6 code pattern claims (grep + context read)
- ✅ **Complete:** No remaining items to check
- ⚠️ **Caveat:** The `--spec-mode` values list in tutorials should include "standard" alongside ddtree/self/ngram/mtp — the code accepts all five (`main.zig:1029`)
