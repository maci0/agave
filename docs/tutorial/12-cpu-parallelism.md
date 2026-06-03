# Chapter 12: CPU Parallelism

Modern CPUs have 4-64 cores. A single-threaded GEMV can only saturate one core's memory bandwidth (~10-20 GB/s). The total system bandwidth is much higher (~100-400 GB/s). **Threading unlocks the full bandwidth.**

Agave uses a lightweight **futex-based thread pool** that wakes workers on demand, distributes work via an atomic counter, and has the main thread participate instead of just waiting.

## Why Not Just Spawn Threads?

```zig
// BAD: Spawning threads per operation
for (n_rows) |row| {
    const thread = try std.Thread.spawn(.{}, gemvRow, .{row});
    thread.join();
}
```

**Problems:**

1. **Thread creation overhead:** 10-50 µs per spawn (GEMV row takes 1-5 µs)
2. **No work sharing:** Fixed assignment, poor load balancing
3. **Main thread idle:** Wastes a core

**Better:** Maintain a **persistent pool** of worker threads that sleep when idle and wake on demand.

## Futex-Based Sleep/Wake

A **futex** (fast userspace mutex) is a kernel primitive that lets threads sleep/wake efficiently:

- **`futexWait(addr, expected)`**: Sleep until `*addr != expected`
- **`futexWake(addr, n)`**: Wake up to `n` threads waiting on `addr`

**Cost:** ~1-2 µs to wake a sleeping thread (vs 50+ µs to spawn a new thread).

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
sequenceDiagram
    participant M as Main Thread
    participant K as Kernel (futex)
    participant W1 as Worker 1
    participant W2 as Worker 2

    Note over W1,W2: Idle — sleeping on generation=0
    W1->>K: futexWait(&generation, 0)
    W2->>K: futexWait(&generation, 0)

    M->>M: generation.fetchAdd(1) → generation=1
    M->>K: futexWake(&generation, 2)
    K-->>W1: wake (generation changed)
    K-->>W2: wake (generation changed)

    W1->>W1: local_gen = generation.load() → 1
    W2->>W2: local_gen = generation.load() → 1

    par Workers process chunks
        W1->>W1: doWork()
        W2->>W2: doWork()
    end

    W1->>M: active.fetchSub(1)
    W2->>M: active.fetchSub(1)
    Note over W1,W2: Back to sleep — futexWait(&generation, 1)


In Zig 0.16, futex operations go through the `Io` context (threaded from `main(Init)` via `init.io`). The thread pool stores `io` at spawn time and uses `io.futexWaitUncancelable()` / `io.futexWake()` instead of the old `std.Thread.Futex` API.

### Generation Counter Pattern

```zig
generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
io: Io,  // stored at spawn() time

// Post work
_ = generation.fetchAdd(1, .release);  // Bump generation
io.futexWake(u32, &generation.raw, n_workers);  // Wake all workers

// Worker loop
var local_gen: u32 = 0;
while (true) {
    pool.io.futexWaitUncancelable(u32, &pool.generation.raw, local_gen);  // Sleep until gen changes
    local_gen = pool.generation.load(.acquire);          // Update local copy
    // ... do work ...
}
```

**Key insight:** Workers sleep on the `generation` variable. When new work arrives, the main thread bumps `generation` and wakes all workers. Workers see the new value and start processing.

**Why `local_gen` starts at 0?** Late-starting workers (thread creation is async) will see a non-zero `generation` immediately and proceed without missing the wake.

## Work Distribution: Atomic Counter

Instead of pre-assigning rows to threads, use an **atomic counter** that threads increment to grab the next chunk. Each thread races to claim the next available chunk by atomically advancing the counter — no coordinator needed.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart LR
    Counter["task_counter\n(atomic usize)\nstarts at 0"]

    Counter -->|fetchAdd(4)| M["Main Thread\nrows 0–3"]
    Counter -->|fetchAdd(4)| W1["Worker 1\nrows 4–7"]
    Counter -->|fetchAdd(4)| W2["Worker 2\nrows 8–11"]
    Counter -->|fetchAdd(4)| W1b["Worker 1\nrows 12–15"]
    Counter -->|fetchAdd(4)| W3["Worker 3\nrows 16–19"]
    Counter -->|fetchAdd(4)| W2b["Worker 2\nrows 20–23"]

    M --> Out["Output rows\n(y vector)"]
    W1 --> Out
    W2 --> Out
    W1b --> Out
    W3 --> Out
    W2b --> Out

    subgraph "n=24 rows, grain=4 → 6 chunks"
        Counter
    end


```zig
task_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);
task_total: usize = n_rows;
task_grain: usize = 4;  // Rows per chunk

fn doWork(pool: *ThreadPool) void {
    while (true) {
        const start = pool.task_counter.fetchAdd(pool.task_grain, .monotonic);
        if (start >= pool.task_total) break;  // No more work
        const end = @min(start + pool.task_grain, pool.task_total);

        // Process rows [start, end)
        for (start..end) |row| {
            gemvRow(row);
        }
    }
}
```

**Benefits:**

- **Dynamic load balancing:** Fast threads grab more chunks
- **No synchronization barrier:** Threads grab work independently
- **Cache-friendly:** Consecutive rows processed together (grain size)

**Grain size:** Too small = contention on `task_counter`. Too large = poor load balancing. Sweet spot: 4-16 rows for GEMV.

## Main Thread Participation

The main thread should **not** just wait — it should do work too. Instead of sitting idle while workers run, it joins the counter race and takes chunks like any other thread, then spin-waits for the stragglers.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
flowchart TD
    Start["parallelFor() called"] --> Post["Post task descriptor\n(func, ctx, total, grain)"]
    Post --> Reset["task_counter = 0\nactive = n_workers"]
    Reset --> Wake["generation++\nfutexWake(all workers)"]
    Wake --> Split["Main thread + Workers\nall racing on task_counter"]

    Split --> Main["Main Thread\ndoWork() loop\n(fetchAdd chunks)"]
    Split --> W1["Worker 1\ndoWork() loop"]
    Split --> W2["Worker 2\ndoWork() loop"]
    Split --> Wn["Worker N\ndoWork() loop"]

    Main --> Spin["Main spins:\nwhile active != 0\n spinLoopHint()"]
    W1 --> Dec1["active.fetchSub(1)"]
    W2 --> Dec2["active.fetchSub(1)"]
    Wn --> DecN["active.fetchSub(1)"]

    Dec1 --> Done["active == 0\nparallelFor returns"]
    Dec2 --> Done
    DecN --> Done
    Spin --> Done


```zig
pub fn parallelFor(pool: *ThreadPool, total: usize, grain: usize, ctx: *anyopaque, func: WorkFunc) void {
    // Post work
    pool.task_counter.store(0, .release);
    pool.active.store(@intCast(pool.n_workers), .release);
    _ = pool.generation.fetchAdd(1, .release);
    pool.io.futexWake(u32, &pool.generation.raw, @intCast(pool.n_workers));

    // Main thread participates
    pool.doWork();

    // Wait for workers to finish
    while (pool.active.load(.acquire) != 0) {
        std.atomic.spinLoopHint();  // Hint CPU to save power during spin
    }
}
```

**Why participate?** If you have 8 cores and spawn 7 worker threads, the main thread sitting idle wastes 1/8 of your compute power.

**Why spin-wait?** GEMV chunks are microsecond-scale. Futex wait/wake would add 1-2 µs overhead per operation — comparable to the work itself. Spinning is simpler and faster for short waits.

## Full Thread Pool Implementation

From `src/thread_pool.zig`:

```zig
pub const ThreadPool = struct {
    workers: [max_workers]Worker = undefined,
    n_workers: usize = 0,
    io: Io = undefined,  // Stored at spawn() for futex operations

    // Task descriptor
    task_func: ?*const fn (*anyopaque, usize, usize) void = null,
    task_ctx: ?*anyopaque = null,
    task_total: usize = 0,
    task_grain: usize = 1,
    task_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    // Cache-line pad: task_counter is the hottest field (fetchAdd per chunk).
    // Without padding, generation/active share its cache line, causing
    // cross-core invalidation when workers finish vs. pull new chunks.
    _counter_pad: [cache_line - @sizeOf(std.atomic.Value(usize))]u8 = undefined,

    // Synchronization
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn init(n: usize) ThreadPool {
        return .{ .n_workers = @min(n, max_workers) };
    }

    pub fn spawn(self: *ThreadPool, io: Io) void {
        self.io = io;  // Store Io for futex operations
        for (0..self.n_workers) |i| {
            self.workers[i] = .{
                .thread = std.Thread.spawn(.{}, workerLoop, .{self}) catch {
                    self.n_workers = i;  // Reduce count if spawn fails
                    return;
                },
            };
        }
    }

    pub fn deinit(self: *ThreadPool) void {
        self.shutdown.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, @intCast(self.n_workers));

        for (0..self.n_workers) |i| {
            self.workers[i].thread.join();
        }
    }

    pub fn parallelFor(
        self: *ThreadPool,
        total: usize,
        grain: usize,
        ctx: *anyopaque,
        func: *const fn (*anyopaque, usize, usize) void,
    ) void {
        if (total == 0) return;

        const effective_grain = @max(grain, min_grain);

        // Too small for parallelism? Run inline
        if (self.n_workers == 0 or total <= effective_grain) {
            func(ctx, 0, total);
            return;
        }

        // Post task
        self.task_func = func;
        self.task_ctx = ctx;
        self.task_total = total;
        self.task_grain = effective_grain;
        self.task_counter.store(0, .release);
        self.active.store(@intCast(self.n_workers), .release);

        // Wake workers
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, @intCast(self.n_workers));

        // Main thread participates
        self.doWork();

        // Spin-wait for completion
        while (self.active.load(.acquire) != 0) {
            std.atomic.spinLoopHint();
        }
    }

    fn doWork(self: *ThreadPool) void {
        const func = self.task_func orelse return;
        const ctx = self.task_ctx orelse return;
        const total = self.task_total;
        const grain = self.task_grain;

        while (true) {
            const start = self.task_counter.fetchAdd(grain, .monotonic);
            if (start >= total) break;
            const end = @min(start + grain, total);
            func(ctx, start, end);
        }
    }

    fn workerLoop(pool: *ThreadPool) void {
        var local_gen: u32 = 0;

        while (true) {
            // Sleep until generation changes
            pool.io.futexWaitUncancelable(u32, &pool.generation.raw, local_gen);

            if (pool.shutdown.load(.acquire)) return;

            local_gen = pool.generation.load(.acquire);

            // Do work
            pool.doWork();

            // Signal completion
            _ = pool.active.fetchSub(1, .release);
        }
    }
};
```

## Usage Example: Parallel GEMV

```zig
const GemvCtx = struct {
    x: [*]const f32,
    w: [*]const f32,
    y: [*]f32,
    k: usize,
};

fn gemvRows(ctx: *anyopaque, start: usize, end: usize) void {
    const gemv_ctx: *GemvCtx = @ptrCast(@alignCast(ctx));
    for (start..end) |row| {
        var acc: f32 = 0.0;
        const roff = row * gemv_ctx.k;
        for (0..gemv_ctx.k) |j| {
            acc += gemv_ctx.w[roff + j] * gemv_ctx.x[j];
        }
        gemv_ctx.y[row] = acc;
    }
}

pub fn gemvParallel(pool: *ThreadPool, x: [*]const f32, w: [*]const f32, y: [*]f32, n: usize, k: usize) void {
    var ctx = GemvCtx{ .x = x, .w = w, .y = y, .k = k };
    pool.parallelFor(n, 4, &ctx, gemvRows);  // 4 rows per chunk
}
```

**Performance:** On an 8-core CPU, this achieves ~6-7× speedup (not 8× due to memory bandwidth saturation and atomic contention).

## Memory Ordering

Atomic operations have different **memory ordering** guarantees. The key question is: when one thread writes data and another reads it, how do you ensure the reader sees the writer's writes?

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#e8f0fe',
  'primaryTextColor': '#1a1a2e',
  'primaryBorderColor': '#4a6cf7',
  'lineColor': '#4a6cf7',
  'secondaryColor': '#f0f4ff',
  'tertiaryColor': '#f8f9ff',
  'edgeLabelBackground': '#ffffff',
  'clusterBkg': '#f0f4ff',
  'clusterBorder': '#4a6cf7',
  'titleColor': '#1a1a2e',
  'nodeTextColor': '#1a1a2e',
  'fontFamily': 'ui-monospace, SFMono-Regular, monospace'
}}}%%
sequenceDiagram
    participant M as Main Thread
    participant W as Worker Thread

    Note over M: Write task descriptor fields
    M->>M: task_total = 1024 (plain write)
    M->>M: task_grain = 4 (plain write)
    M->>M: generation.fetchAdd(1, .release)
    Note over M: .release: all prior writes<br/>are visible before this store

    Note over M,W: futexWake / futexWait handoff

    W->>W: local_gen = generation.load(.acquire)
    Note over W: .acquire: all writes that<br/>happened before the .release<br/>are now visible here
    W->>W: read task_total → sees 1024 ✓
    W->>W: read task_grain → sees 4 ✓


### .monotonic

No synchronization — just atomicity. Use for counters:

```zig
const start = pool.task_counter.fetchAdd(grain, .monotonic);
```

**Why monotonic?** The counter value doesn't synchronize memory — it's just work assignment. Workers don't need to see other threads' writes.

### .acquire / .release

**Release** (store): All prior writes become visible before this store.
**Acquire** (load): All subsequent reads see writes that happened before the release.

Use for **handoff** between threads:

```zig
// Main thread: release
pool.task_total = total;
pool.task_grain = grain;
_ = pool.generation.fetchAdd(1, .release);  // All prior writes visible

// Worker thread: acquire
local_gen = pool.generation.load(.acquire);  // See all writes before release
// Now safe to read task_total, task_grain
```

### .seq_cst (Sequential Consistency)

Strongest guarantee — all threads see the same order of operations. **Slowest** — use only when necessary.

Agave doesn't use `.seq_cst` — acquire/release is sufficient for thread pool handoff.

## Tuning Parameters

### Number of Workers

```zig
const n_cores = std.Thread.getCpuCount() catch 1;
const n_workers = n_cores - 1;  // Leave 1 core for main thread
```

**Why n-1?** Main thread participates, so total threads = `n_workers + 1`.

### Grain Size

```zig
const min_grain: usize = 4;  // Minimum rows per chunk
```

**Heuristic:** `grain = max(min_grain, n_rows / (n_threads * 4))`

- Too small → atomic contention
- Too large → poor load balancing
- 4× oversubscription → good load balance

### Inline Threshold

```zig
if (total <= effective_grain) {
    func(ctx, 0, total);  // Run inline, skip threading overhead
    return;
}
```

**Why?** For tiny work (< 4 rows), threading overhead dominates. Faster to run inline.

## Avoiding Common Pitfalls

### Pitfall 1: Shared Mutable State

```zig
// BAD: Race condition
var sum: f32 = 0;
pool.parallelFor(n, grain, &sum, func);

fn func(ctx: *anyopaque, start: usize, end: usize) void {
    const sum_ptr: *f32 = @ptrCast(@alignCast(ctx));
    for (start..end) |i| {
        sum_ptr.* += data[i];  // WRONG: Multiple threads writing to same memory
    }
}
```

**Fix:** Use thread-local accumulators, then reduce:

```zig
// GOOD: Thread-local accumulators
const SumCtx = struct {
    data: [*]const f32,
    partial_sums: []f32,
    grain: usize,
};

fn func(ctx: *anyopaque, start: usize, end: usize) void {
    const sum_ctx: *SumCtx = @ptrCast(@alignCast(ctx));
    const thread_id = start / sum_ctx.grain;
    var local_sum: f32 = 0.0;

    for (start..end) |i| {
        local_sum += sum_ctx.data[i];
    }

    sum_ctx.partial_sums[thread_id] = local_sum;
}

// Then reduce on main thread
var total: f32 = 0.0;
for (partial_sums) |ps| total += ps;
```

### Pitfall 2: False Sharing

```zig
// BAD: Partial sums are adjacent in memory
var partial_sums: [8]f32 = undefined;  // 8 f32s = 32 bytes = half a cache line
```

**Problem:** Cache lines are 64 bytes. Multiple threads writing to the same cache line **ping-pong** it between cores → slowdown.

**Fix:** Pad to cache line size:

```zig
// GOOD: Each partial sum on its own cache line
const CacheLinePadded = struct {
    value: f32 align(64),  // Force 64-byte alignment
};

var partial_sums: [8]CacheLinePadded = undefined;
```

Agave avoids this by using per-chunk reduction in the worker function — no shared array.

### Pitfall 3: Forgetting to Call spawn()

```zig
// BAD: Workers never created
var pool = ThreadPool.init(7);  // Just sets n_workers
pool.parallelFor(...);  // No workers exist! Runs inline on main thread
```

**Fix:** Call `spawn(io)` after the pool is at its final memory location:

```zig
// GOOD
var pool = ThreadPool.init(7);
pool.spawn(io);  // Actually creates worker threads (io from main(Init))
defer pool.deinit();
```

**Why separate?** Workers capture `pool` by pointer. If you spawn before the pool is at its final location (e.g., it's a stack local that gets moved), the pointer becomes invalid. The `io` parameter is the Zig 0.16 `Io` context needed for futex operations.

## Performance Characteristics

**Speedup** (measured on Apple M4 Pro, 12 cores):

| Operation | Single-threaded | 11 workers + main | Speedup |
| --------- | --------------- | ----------------- | ------- |
| F32 GEMV (4096×4096) | 1.2 ms | 0.18 ms | 6.7× |
| Q4_0 GEMV (4096×4096) | 0.8 ms | 0.13 ms | 6.2× |
| RMSNorm (4096) | 15 µs | 3 µs | 5.0× |

**Why not 12× speedup?** Memory bandwidth saturation. At ~8 threads, all available bandwidth is used (~400 GB/s on M4 Pro).

**Overhead:**

- Thread creation: ~20 µs per worker (one-time)
- Wake latency: ~1-2 µs (per parallelFor call)
- Atomic contention: negligible with grain=4

**When not to parallelize:**

- `n_rows < 4` → inline faster
- Already on GPU → CPU threading irrelevant
- Overhead dominates (e.g., softmax with n=128)

---

**In the code:** [src/thread_pool.zig](../../src/thread_pool.zig) (full implementation), [src/backend/cpu.zig](../../src/backend/cpu.zig) (uses pool for GEMV, GEMM, SDPA)

**Related:** [std.Thread](https://ziglang.org/documentation/master/std/#std.Thread), [std.atomic](https://ziglang.org/documentation/master/std/#std.atomic), [Futex](https://man7.org/linux/man-pages/man2/futex.2.html)

**Next:** [Chapter 13: Batched Dispatch and Fusion →](13-batched-dispatch-and-fusion.md) | **Back:** [Chapter 11: Metal Backend Internals ←](11-metal-backend-internals.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)
