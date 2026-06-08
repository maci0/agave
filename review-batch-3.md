# Review Batch 3: Tutorials 09–12

## Issues Found

---

### [ERROR] docs/tutorial/10-memory-safety.md: Pitfall 1 — defer in a Loop

**Tutorial claims:**
> ```zig
> // BAD: defer accumulates, all run at function exit
> for (files) |path| {
>     const file = try std.fs.cwd().openFile(path, .{});
>     defer file.close();  // Wrong! All files stay open until function exits
> ```

**Source / language spec says:** In Zig, `defer` inside a `for` loop body is scoped to the loop iteration, not the enclosing function. Each iteration creates a new block scope, and `defer` executes at the end of that block — so `file.close()` runs at the end of each iteration. The code shown is actually **correct** and idiomatic. The entire Pitfall 1 section is based on a false premise.

**Fix:** Remove or rewrite Pitfall 1. If the intent is to warn about `while` loops with manual iterator state where scope behavior may be confusing, reframe the example. As-is, the `for`-loop example teaches the wrong lesson.

---

### [ERROR] docs/tutorial/10-memory-safety.md: Pitfall 2 — Conditional defer

**Tutorial claims:**
> ```zig
> if (use_cache) {
>     const cache = try allocator.alloc(u8, size);
>     defer allocator.free(cache);  // Runs when function exits, not at end of if!
> }
> // cache is out of scope, but defer still tries to free it → use-after-free
> ```

**Source / language spec says:** `defer` inside an `if` block is scoped to that block. It runs when the `if` block exits, not when the function exits. There is no use-after-free here — the `free` runs at the closing `}` of the `if` block, while `cache` is still valid. The real issue with this pattern is that `cache` goes out of scope and can't be used outside the block — but that's a scoping issue, not a memory safety issue, and the `defer` is correct.

**Fix:** Rewrite the example. The actual pitfall here is that allocating inside an `if` with `defer` means the allocation is freed immediately at the end of the block, so you can't use it outside. This is correct behavior, not a bug — but it can be surprising if someone intended the allocation to outlive the `if`. The explanation should be corrected.

---

### [ERROR] docs/tutorial/12-cpu-parallelism.md: parallelFor operation ordering

**Tutorial claims (Main Thread Participation snippet, ~line 215):**
> ```zig
> pub fn parallelFor(pool: *ThreadPool, ...) void {
>     // Post work
>     pool.task_counter.store(0, .release);
>     if (self.active.cmpxchgWeak(0, @intCast(self.n_workers), .acq_rel, .monotonic)) |still_active| {
> ```

**Source says (src/thread_pool.zig lines 109-122):** The CAS on `active` happens **first**, before posting any task fields. Only after the CAS succeeds are `task_func`, `task_ctx`, `task_total`, `task_grain`, and `task_counter` written. This ordering is critical — it prevents workers from seeing stale task descriptors before the new task is fully posted.

Additionally, the parameter is named `pool` but the CAS line uses `self.active` and `self.n_workers`, mixing naming inconsistently.

**Fix:** Reorder to match actual code — CAS first, then task descriptor fields, then generation bump and wake. Use consistent `self` naming (or `pool` throughout).

---

### [ERROR] docs/tutorial/12-cpu-parallelism.md: Full Thread Pool Implementation uses `active.store` instead of `cmpxchgWeak`

**Tutorial claims (~line 318):**
> ```zig
>     self.task_counter.store(0, .release);
>     self.active.store(@intCast(self.n_workers), .release);
> ```

**Source says (src/thread_pool.zig line 109):**
```zig
if (self.active.cmpxchgWeak(0, @intCast(self.n_workers), .acq_rel, .monotonic)) |still_active| {
    std.log.err("ThreadPool: concurrent parallelFor detected (active={d}), running inline", .{still_active});
    func(ctx, 0, total);
    return;
}
```

The actual code uses `cmpxchgWeak` (CAS) to atomically claim the pool, with fallback to inline execution if the pool is already busy. The tutorial's "Full Thread Pool Implementation" replaces this with a simple `active.store()`, which drops the concurrent-use detection and introduces a race condition.

**Fix:** Replace `self.active.store(...)` with the actual `cmpxchgWeak` pattern from the source. Also fix the ordering: CAS first, then task field writes.

---

### [WARNING] docs/tutorial/11-metal-backend-internals.md: Inconsistent sync reduction numbers

**Tutorial claims (line ~430):**
> "Qwen3.5 eliminated 16 syncs/token by moving Q/gate split from CPU (memcpy) to GPU (kernel) → 15% throughput gain."

**Tutorial also claims (line ~540):**
> "Qwen3.5 reduced sync count from 18 → 1 per token by moving Q/gate split to GPU → 15% faster."

**Issue:** 18 → 1 is a reduction of 17, not 16. These two statements about the same optimization contradict each other.

**Fix:** Make the numbers consistent. Either "eliminated 17 syncs (from 18 → 1)" or "eliminated 16 syncs (from 18 → 2)".

---

### [WARNING] docs/tutorial/12-cpu-parallelism.md: Mermaid flowchart shows wrong operation order

**Tutorial claims (flowchart at ~line 188):**
> ```
> Post task descriptor → task_counter = 0 → CAS(active, 0→n_workers) → generation++ → futexWake
> ```

**Source says:** The CAS must come before posting the task descriptor. Posting task fields before the CAS means workers could read partially-updated fields if the pool were unexpectedly active.

**Fix:** Reorder the flowchart nodes: CAS first, then "Post task descriptor / task_counter = 0", then generation++ / futexWake.

---

## Coverage Status

### Checked directly:
- `src/backend/kernels/cpu/gemv.zig` — sparse_threshold = 0.005 confirmed ✓
- `src/backend/kernels/cpu/gemv_f32.zig` — 4-row batching pattern matches tutorial ✓
- `src/backend/kernels/cpu/gemv_q4_k.zig`, `gemv_q5_k.zig`, `gemv_q6_k.zig` — NR=2 confirmed ✓
- `src/backend/kernels/cpu/gemv_q8_0.zig`, `gemv_q4_0.zig`, `gemv_bf16.zig`, `gemv_f16.zig` — NR=4 confirmed ✓
- `src/backend/kernels/cpu/sdpa.zig` — max_sdpa_seq_len = 8192, max_head_dim = 256, sparse_v_threshold = 1e-6 confirmed ✓
- `src/backend/kernels/metal/sdpa.metal` — sdpa_max_seq_len = 4096, sdpa_block_size = 16, threadgroup arrays match tutorial ✓
- `src/backend/metal.zig` — 83 pipelines, buffer cache, page alignment, profile counters, batch_mode, gemm dispatch all confirmed ✓
- `src/backend/kernels/metal/megakernel.metal` — 11 fused FFN kernels confirmed ✓
- `src/backend/kernels/metal/mega_common.metal` — 18 building blocks confirmed ✓
- `src/backend/metal.zig` — 5 true megakernels confirmed ✓
- `src/backend/mega_compose.zig` — composeMSL, ModelDesc, runtime MSL generation confirmed ✓
- `src/thread_pool.zig` — full implementation cross-referenced ✓
- `src/backend/backend.zig` — Backend union dispatch pattern confirmed ✓
- Zig `defer` scoping rules verified via language documentation ✓

### Sections verified correct (no issues):
- Tutorial 09: SIMD operations, @Vector, @splat, @reduce, @mulAdd — all correct
- Tutorial 09: F32 GEMV code — matches `gemv_f32.zig` exactly
- Tutorial 09: NR=2 for K-quants, NR=4 for Q4_0/Q8_0/BF16/F16 — correct
- Tutorial 09: Sparse threshold 0.005, ~40% sparsity — correct
- Tutorial 09: Q4_0 layout (18 bytes/block, f16 scale + 16B quants) — correct
- Tutorial 10: `defer`/`errdefer` core explanation — correct
- Tutorial 10: Struct init with errdefer pattern — correct
- Tutorial 10: ArenaAllocator pattern — correct
- Tutorial 11: UMA architecture, zero-copy wrapping — correct
- Tutorial 11: Buffer caching by aligned base address — correct
- Tutorial 11: Command buffer batching and lazy creation — correct
- Tutorial 11: beginBatch/endBatch barrier suppression — correct
- Tutorial 11: Threadgroup memory budget calculation — correct
- Tutorial 11: GEMM dtype dispatch and token tiling — correct
- Tutorial 11: Megakernel tier description (11 FFN, 18 blocks, 5 true, 83 total) — correct
- Tutorial 12: Futex-based sleep/wake, generation counter — correct
- Tutorial 12: Atomic counter work distribution — correct
- Tutorial 12: Main thread participation, spin-wait — correct
- Tutorial 12: Memory ordering explanations (.monotonic, .acquire/.release) — correct
- Tutorial 12: Worker loop with spurious wakeup handling — correct
