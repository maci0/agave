# Chapter 12: CPU Parallelism

**Prerequisites:** [Chapter 9: CPU SIMD Optimization](09-cpu-simd-optimization.md) (the single-threaded GEMV loop this chapter parallelizes)

**Time:** ~15 min

> After this chapter you can explain the futex-based thread pool, `parallelFor`, and why the main thread participates in work.

Modern CPUs have 4-64 cores. A single-threaded GEMV can only saturate one core's memory bandwidth (~10-20 GB/s). The total system bandwidth is much higher (~100-400 GB/s). **Threading unlocks the full bandwidth.**

Agave uses a lightweight **futex-based thread pool** that wakes workers on demand, distributes work via an atomic counter, and has the main thread participate instead of just waiting.

## Code Flow

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d

    Call["parallelFor(total, grain,\nctx, func)"]:::setup
    Small{"total <= grain\nor no workers?"}
    Inline["run func() inline\non caller's thread"]:::danger
    Post["post task descriptor\ncounter = 0"]:::migration
    Wake["generation++\nfutexWake(all workers)"]:::sync
    Race["main + workers\nrace on task_counter\n(fetchAdd per chunk)"]:::sync
    Spin["main spin-waits\nwhile active != 0"]:::migration
    Done["parallelFor() returns"]:::success

    Call --> Small
    Small -- "yes" --> Inline
    Small -- "no" --> Post --> Wake --> Race --> Spin --> Done
```

One `parallelFor` call: post the work descriptor, bump the generation counter to wake sleeping workers, then the main thread joins the same atomic-counter race instead of idling. No thread is ever spawned per call; workers are created once at pool startup and sleep between calls. The rest of this chapter builds up to and past that loop.

## Why Not Just Spawn Threads?

```text
BAD: spawn a thread per operation
for row in 0..n_rows:
    thread = Thread.spawn(gemvRow, row)
    thread.join()
```

**Implementation:** the pool-based alternative that avoids this cost lives in [src/thread_pool.zig](../../src/thread_pool.zig); there is no per-call spawn anywhere in that file.

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
sequenceDiagram
    participant M as Main Thread
    participant K as Kernel (futex)
    participant W1 as Worker 1
    participant W2 as Worker 2

    Note over W1,W2: Idle, sleeping on generation=0
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
    Note over W1,W2: Back to sleep, futexWait(&generation, 1)
```

In Zig 0.16, futex operations go through the `Io` context (threaded from `main(Init)` via `init.io`). The thread pool stores `io` at spawn time and uses `io.futexWaitUncancelable()` / `io.futexWake()` instead of the old `std.Thread.Futex` API.

### Generation Counter Pattern

```text
ThreadPool fields:
  generation: atomic u32 = 0
  io: Io                       # stored at spawn() time

Post work (main thread):
  generation.fetchAdd(1, release)       # bump generation
  io.futexWake(&generation, n_workers)  # wake all workers

Worker loop:
  local_gen = 0
  loop:
    io.futexWaitUncancelable(&generation, local_gen)
    new_gen = generation.load(acquire)
    if new_gen == local_gen: continue   # spurious wakeup
    local_gen = new_gen
    ... do work ...
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (`ThreadPool.generation`, `workerLoop`).

**Key insight:** Workers sleep on the `generation` variable. When new work arrives, the main thread bumps `generation` and wakes all workers. Workers see the new value and start processing.

**Why `local_gen` starts at 0?** Late-starting workers (thread creation is async) will see a non-zero `generation` immediately and proceed without missing the wake.

## Work Distribution: Atomic Counter

Instead of pre-assigning rows to threads, use an **atomic counter** that threads increment to grab the next chunk. Each thread races to claim the next available chunk by atomically advancing the counter, no coordinator needed.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Counter["task_counter\n(atomic usize)\nstarts at 0"]:::setup
    M["Main Thread\nrows 0–3"]:::sync
    W1["Worker 1\nrows 4–7"]:::sync
    W2["Worker 2\nrows 8–11"]:::sync
    W1b["Worker 1\nrows 12–15"]:::sync
    W3["Worker 3\nrows 16–19"]:::sync
    W2b["Worker 2\nrows 20–23"]:::sync
    Out["Output rows\n(y vector)"]:::success

    Counter -->|"fetchAdd(4)"| M
    Counter -->|"fetchAdd(4)"| W1
    Counter -->|"fetchAdd(4)"| W2
    Counter -->|"fetchAdd(4)"| W1b
    Counter -->|"fetchAdd(4)"| W3
    Counter -->|"fetchAdd(4)"| W2b

    M --> Out
    W1 --> Out
    W2 --> Out
    W1b --> Out
    W3 --> Out
    W2b --> Out

    subgraph "n=24 rows, grain=4 → 6 chunks"
        Counter
    end
```

```text
ThreadPool fields:
  task_counter: atomic usize = 0
  task_total: usize = n_rows
  task_grain: usize = 4          # rows per chunk

doWork(pool):
  loop:
    start = task_counter.fetchAdd(task_grain, monotonic)
    if start >= task_total: break         # no more work
    end = min(start + task_grain, task_total)
    for row in start..end:
      gemvRow(row)
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (`doWork`).

**Benefits:**

- **Dynamic load balancing:** Fast threads grab more chunks
- **No synchronization barrier:** Threads grab work independently
- **Cache-friendly:** Consecutive rows processed together (grain size)

**Grain size:** Too small = contention on `task_counter`. Too large = poor load balancing. Sweet spot: 4-16 rows for GEMV.

## Main Thread Participation

The main thread should **not** just wait, it should do work too. Instead of sitting idle while workers run, it joins the counter race and takes chunks like any other thread, then spin-waits for the stragglers.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Start["parallelFor() called"]:::setup
    Post["Post task descriptor\n(func, ctx, total, grain)"]:::migration
    Reset["task_counter = 0"]:::setup
    Wake["generation++\nfutexWake(all workers)"]:::sync
    Inline["run func() inline\n(return immediately)"]:::danger
    Split["Main thread + Workers\nall racing on task_counter"]:::sync
    Main["Main Thread\ndoWork() loop\n(fetchAdd chunks)"]:::sync
    W1["Worker 1\ndoWork() loop"]:::sync
    W2["Worker 2\ndoWork() loop"]:::sync
    Wn["Worker N\ndoWork() loop"]:::sync
    Spin["Main spins:\nwhile active != 0\n spinLoopHint()"]:::migration
    Dec1["active.fetchSub(1)"]:::migration
    Dec2["active.fetchSub(1)"]:::migration
    DecN["active.fetchSub(1)"]:::migration
    Done["active == 0\nparallelFor returns"]:::success

    Start --> Post
    Post --> Reset
    Reset --> CAS{"cmpxchgWeak\n(active, 0→n_workers)"}
    CAS -->|CAS succeeds| Wake
    CAS -->|CAS fails: pool busy| Inline
    Wake --> Split

    Split --> Main
    Split --> W1
    Split --> W2
    Split --> Wn

    Main --> Spin
    W1 --> Dec1
    W2 --> Dec2
    Wn --> DecN

    Dec1 --> Done
    Dec2 --> Done
    DecN --> Done
    Spin --> Done
```

```text
parallelFor(pool, total, grain, ctx, func):
  task_counter.store(0, release)
  if active.cmpxchgWeak(0, n_workers, acq_rel, monotonic) is Some(still_active):
    log.err("concurrent parallelFor detected, running inline")
    func(ctx, 0, total)
    return

  generation.fetchAdd(1, release)
  io.futexWake(&generation, n_workers)

  pool.doWork()                          # main thread participates

  while active.load(acquire) != 0:
    spinLoopHint()                       # hint CPU to save power during spin
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (`parallelFor`).

**Why participate?** If you have 8 cores and spawn 7 worker threads, the main thread sitting idle wastes 1/8 of your compute power.

**Why spin-wait?** GEMV chunks are microsecond-scale. Futex wait/wake would add 1-2 µs overhead per operation, comparable to the work itself. Spinning is simpler and faster for short waits.

## Thread Pool Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Uninit

    Uninit --> Ready : init(n_workers)\npool fields zeroed\nno threads exist

    state Ready {
        [*] --> Idle
        Idle --> Spawning : spawn(io)\nstores Io context
        Spawning --> Sleeping : std.Thread.spawn()\nfor each worker\nworkers enter futexWait
        Sleeping --> Working : parallelFor()\ngeneration++\nfutexWake(all)
        Working --> Sleeping : all workers complete\nactive decrements to 0\nmain returns from spin
        Sleeping --> Sleeping : spurious wakeup\nlocal_gen == generation\nworker loops back
    }

    Ready --> ShuttingDown : deinit()\nshutdown = true\ngeneration++\nfutexWake(all)
    ShuttingDown --> [*] : all workers join()\npool memory released

    note right of Uninit
        ThreadPool.init() only sets
        n_workers, no threads yet
    end note

    note right of Working
        Main thread participates
        (doWork loop) while workers
        also race on task_counter
    end note
```

## Full Thread Pool Implementation

The full struct ties every piece above together: task descriptor fields, the generation counter, the active-worker count, and the shutdown flag.

```text
ThreadPool struct (fields):
  workers: [max_workers]Worker
  n_workers: usize
  io: Io                                    # stored at spawn() for futex ops

  task_func, task_ctx, task_total, task_grain
  task_counter: atomic usize                # hottest field, cache-line padded
                                             # so generation/active don't share
                                             # its line and cause cross-core
                                             # invalidation

  generation: atomic u32
  active: atomic u32
  shutdown: atomic bool

init(n): n_workers = min(n, max_workers)

spawn(io):
  self.io = io
  for i in 0..n_workers:
    workers[i] = Thread.spawn(workerLoop, self)
    # on spawn failure: shrink n_workers to i, stop

deinit():
  shutdown.store(true, release)
  generation.fetchAdd(1, release)
  io.futexWake(&generation, n_workers)
  for each worker: thread.join()

parallelFor(total, grain, ctx, func):
  if total == 0: return
  effective_grain = max(grain, min_grain)
  if n_workers == 0 or total <= effective_grain:
    func(ctx, 0, total)                     # too small, run inline
    return
  task_counter.store(0, release)
  if active.cmpxchgWeak(0, n_workers, acq_rel, monotonic) is Some(still_active):
    log.err("concurrent parallelFor detected, running inline")
    func(ctx, 0, total)
    return
  task_func, task_ctx = func, ctx
  task_total, task_grain = total, effective_grain
  generation.fetchAdd(1, release)
  io.futexWake(&generation, n_workers)
  doWork()                                  # main thread participates
  while active.load(acquire) != 0: spinLoopHint()

doWork():
  loop:
    start = task_counter.fetchAdd(task_grain, monotonic)
    if start >= task_total: break
    end = min(start + task_grain, task_total)
    task_func(task_ctx, start, end)

workerLoop(pool):
  local_gen = 0
  loop:
    io.futexWaitUncancelable(&generation, local_gen)
    if shutdown.load(acquire): return
    new_gen = generation.load(acquire)
    if new_gen == local_gen: continue        # spurious wakeup
    local_gen = new_gen
    pool.doWork()
    active.fetchSub(1, release)
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (full `ThreadPool` struct: `init`, `spawn`, `deinit`, `parallelFor`, `doWork`, `workerLoop`).

## Usage Example: Parallel GEMV

```text
GemvCtx: { x, w, y (pointers), k }

gemvRows(ctx, start, end):
  for row in start..end:
    acc = 0.0
    roff = row * ctx.k
    for j in 0..ctx.k:
      acc += ctx.w[roff + j] * ctx.x[j]
    ctx.y[row] = acc

gemvParallel(pool, x, w, y, n, k):
  ctx = GemvCtx{x, w, y, k}
  pool.parallelFor(n, grain=4, &ctx, gemvRows)
```

**Implementation:** [src/backend/cpu.zig](../../src/backend/cpu.zig) (parallel GEMV dispatch through `ThreadPool.parallelFor`).

**Performance:** On an 8-core CPU, this achieves ~6-7× speedup (not 8× due to memory bandwidth saturation and atomic contention).

## Memory Ordering

Atomic operations have different **memory ordering** guarantees. The key question is: when one thread writes data and another reads it, how do you ensure the reader sees the writer's writes?

```mermaid
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
```

### .monotonic

No synchronization, just atomicity. Use for counters: `task_counter.fetchAdd(grain, .monotonic)`.

**Why monotonic?** The counter value doesn't synchronize memory, it's just work assignment. Workers don't need to see other threads' writes.

### .acquire / .release

**Release** (store): All prior writes become visible before this store.
**Acquire** (load): All subsequent reads see writes that happened before the release.

Use for **handoff** between threads:

```text
Main thread (release):
  task_total = total
  task_grain = grain
  generation.fetchAdd(1, release)      # all prior writes now visible

Worker thread (acquire):
  local_gen = generation.load(acquire) # sees all writes before the release
  # now safe to read task_total, task_grain
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (`parallelFor` release side, `workerLoop` acquire side).

### .seq_cst (Sequential Consistency)

Strongest guarantee, all threads see the same order of operations. **Slowest**, use only when necessary.

Agave doesn't use `.seq_cst`, acquire/release is sufficient for thread pool handoff.

## Tuning Parameters

### Number of Workers

`n_workers = Thread.getCpuCount() - 1`, leaving one core for the main thread.

**Implementation:** [src/backend/backend.zig](../../src/backend/backend.zig) (`BackendState.init`).

**Why n-1?** Main thread participates, so total threads = `n_workers + 1`.

### Grain Size

`min_grain = 4` rows per chunk (module-level constant in [src/thread_pool.zig](../../src/thread_pool.zig)).

**Heuristic:** `grain = max(min_grain, n_rows / (n_threads * 4))`

- Too small → atomic contention
- Too large → poor load balancing
- 4× oversubscription → good load balance

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Slow1["Slow"]:::danger
    Slow2["Slow"]:::danger
    Fast["Optimal:\nlow contention +\ngood balance"]:::success

    subgraph small ["grain = 1  (too small)"]
        direction TB
        S_chunks["256 chunks\nfor 256 rows"]:::migration
        S_atomic["256 fetchAdd\noperations\n(high contention)"]:::danger
        S_load["Perfect load\nbalance"]:::sync
        S_chunks --> S_atomic
        S_chunks --> S_load
    end

    subgraph sweet ["grain = 4-16  (sweet spot)"]
        direction TB
        M_chunks["16-64 chunks\nfor 256 rows"]:::setup
        M_atomic["16-64 fetchAdd\noperations\n(low contention)"]:::sync
        M_load["Good load\nbalance\n(4x oversubscription)"]:::success
        M_chunks --> M_atomic
        M_chunks --> M_load
    end

    subgraph large ["grain = 128  (too large)"]
        direction TB
        L_chunks["2 chunks\nfor 256 rows"]:::migration
        L_atomic["2 fetchAdd\noperations\n(no contention)"]:::sync
        L_load["Poor load balance:\none fast core done,\none slow core stalls all"]:::danger
        L_chunks --> L_atomic
        L_chunks --> L_load
    end

    small -->|"increase grain"| sweet
    sweet -->|"increase grain"| large

    small -. "fetchAdd cost\ndominates" .-> Slow1
    large -. "stragglers\nhurt latency" .-> Slow2
    sweet --> Fast
```

### Inline Threshold

```text
if total <= effective_grain:
    func(ctx, 0, total)   # run inline, skip threading overhead
    return
```

**Implementation:** [src/thread_pool.zig](../../src/thread_pool.zig) (`parallelFor` early-return branch).

**Why?** For tiny work (< 4 rows), threading overhead dominates. Faster to run inline.

## Gotchas

### Pitfall 1: Shared Mutable State

```text
BAD: race condition
sum: f32 = 0
pool.parallelFor(n, grain, &sum, func)

func(ctx, start, end):
  for i in start..end:
    sum.* += data[i]     # WRONG: multiple threads writing the same memory
```

**Fix:** Use thread-local accumulators, then reduce:

```text
GOOD: thread-local accumulators
SumCtx: { data, partial_sums, grain }

func(ctx, start, end):
  thread_id = start / ctx.grain
  local_sum = 0.0
  for i in start..end:
    local_sum += ctx.data[i]
  ctx.partial_sums[thread_id] = local_sum

# Then reduce on main thread
total = 0.0
for ps in partial_sums: total += ps
```

### Pitfall 2: False Sharing

BAD: `partial_sums: [8]f32` puts all 8 values (32 bytes) in half a single cache line.

**Problem:** Cache lines are 64 bytes. Multiple threads writing to the same cache line **ping-pong** it between cores → slowdown.

**Fix:** Pad to cache line size. `CacheLinePadded = struct { value: f32 align(64) }`, then `partial_sums: [8]CacheLinePadded`, so each value owns a full 64-byte line.

```mermaid
flowchart TB
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Ping["Cache line\nping-pong\n(5-30x slowdown)"]:::danger
    Solo["Each write\nindependent\nno cross-core traffic"]:::success

    subgraph bad ["BAD: [8]f32, 32 bytes, fits in one 64-byte cache line"]
        direction LR
        CL1["Cache line (64 bytes)"]:::danger
        PS0["ps[0]\n4B\nCore 0"]:::migration
        PS1["ps[1]\n4B\nCore 1"]:::migration
        PS2["ps[2]\n4B\nCore 2"]:::migration
        PS3["ps[3]\n4B\nCore 3"]:::migration
        PS4["ps[4]\n4B\nCore 4"]:::migration
        PS5["ps[5]\n4B\nCore 5"]:::migration
        PS6["ps[6]\n4B\nCore 6"]:::migration
        PS7["ps[7]\n4B\nCore 7"]:::migration
        CL1 --- PS0
        CL1 --- PS1
        CL1 --- PS2
        CL1 --- PS3
        CL1 --- PS4
        CL1 --- PS5
        CL1 --- PS6
        CL1 --- PS7
    end

    subgraph good ["GOOD: [8]CacheLinePadded, each value owns a full 64-byte cache line"]
        direction LR
        CL_A["Cache line A\n(64 bytes)"]:::setup
        CL_B["Cache line B\n(64 bytes)"]:::setup
        CL_C["..."]:::setup
        CL_H["Cache line H\n(64 bytes)"]:::setup
        PA0["ps[0].value\n4B + 60B pad\nCore 0 only"]:::success
        PA1["ps[1].value\n4B + 60B pad\nCore 1 only"]:::success
        PA7["ps[7].value\n4B + 60B pad\nCore 7 only"]:::success
        CL_A --- PA0
        CL_B --- PA1
        CL_H --- PA7
    end

    bad -->|"Core 1 writes ps[1]\ninvalidates entire line\nCore 0 must reload ps[0]"| Ping
    good -->|"Core 1 writes pa[1]\nonly its own line\nis invalidated"| Solo
```

Agave avoids this by using per-chunk reduction in the worker function, no shared array.

### Pitfall 3: Forgetting to Call spawn()

BAD: `pool = ThreadPool.init(7)` only sets `n_workers`; calling `pool.parallelFor(...)` next runs inline on the main thread because no workers exist yet.

**Fix:** Call `spawn(io)` after the pool is at its final memory location: `pool = ThreadPool.init(7)`, then `pool.spawn(io)` to actually create the worker threads (`io` comes from `main(Init)`), then `defer pool.deinit()`.

**Why separate?** Workers capture `pool` by pointer. If you spawn before the pool is at its final location (e.g., it's a stack local that gets moved), the pointer becomes invalid. The `io` parameter is the Zig 0.16 `Io` context needed for futex operations.

## Performance Characteristics

**Scaling shape** (illustrative on Apple M4 Pro, 14 cores / 13 workers + main; not a `BENCHMARKS.md` table):

| Operation | Single-threaded | Parallel (13 workers + main) | Typical speedup band |
| --------- | --------------- | ---------------------------- | -------------------- |
| F32 GEMV (large square) | baseline | much lower latency | ~5–7× |
| Q4_0 GEMV (large square) | baseline | much lower latency | ~5–7× |
| RMSNorm (hidden dim) | baseline | lower latency | ~4–6× |

**Why not 14× speedup?** Memory bandwidth saturation. With a full core count, bandwidth is exhausted well before linear scaling; marginal gains from threads beyond the saturation point are small, so actual speedup plateaus in the mid single-digit range.

**Overhead:**

- Thread creation: ~20 µs per worker (one-time)
- Wake latency: ~1-2 µs (per parallelFor call)
- Atomic contention: negligible with grain=4

**When not to parallelize:**

- `n_rows < 4` → inline faster
- Already on GPU → CPU threading irrelevant
- Overhead dominates (e.g., softmax with n=128)

The same atomic-counter work-distribution idea scales past one machine in [Chapter 22: Distributed Inference](22-distributed-inference.md): tensor parallelism shards a layer's weights across GPUs the way `parallelFor` shards rows across CPU cores, with a network `allReduceAdd` standing in for the local reduction this chapter's workers do in-process.

---

**In the code:** [src/thread_pool.zig](../../src/thread_pool.zig) (full implementation), [src/backend/cpu.zig](../../src/backend/cpu.zig) (uses pool for GEMV, GEMM, SDPA)

**Related:** [std.Thread](https://ziglang.org/documentation/master/std/#std.Thread), [std.atomic](https://ziglang.org/documentation/master/std/#std.atomic), [Futex](https://man7.org/linux/man-pages/man2/futex.2.html)

**Next:** [Chapter 13: Batched Dispatch and Fusion →](13-batched-dispatch-and-fusion.md) | **Back:** [Chapter 11: Metal Backend Internals ←](11-metal-backend-internals.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)

---

## Glossary

**cache line**, The smallest unit of data transfer between CPU cache levels; typically 64 bytes.

**cache-line padding**, Inserting unused bytes so frequently-written variables by different threads occupy separate cache lines, avoiding false sharing.

**CAS (Compare-And-Swap)**, An atomic operation that updates a value only if it currently matches an expected value; foundational for lock-free data structures.

**cmpxchgWeak**, A CAS variant that may spuriously fail; faster on ARM, best used in retry loops.

**false sharing**, Performance degradation when different threads write to variables sharing the same cache line, causing cross-core invalidation.

**fetchAdd**, An atomic operation that reads, adds, and returns the original value in one indivisible step.

**futex (fast userspace mutex)**, A kernel primitive for efficient thread sleep/wake without busy-waiting.

**generation counter**, An atomic variable workers sleep on; incrementing it signals new work.

**grain size**, The number of work units assigned per atomic fetch-add; controls contention vs. load balance trade-off.

**main thread participation**, The pattern where the main thread does useful work alongside pool workers instead of idly waiting.

**spin-wait**, Busy-looping on a condition instead of sleeping; appropriate for microsecond-scale waits.

**spinLoopHint**, A CPU hint (`pause` on x86, `yield` on ARM) reducing power during spin-wait loops.

**thread pool**, A set of persistent worker threads that sleep when idle and wake on demand, avoiding per-operation thread creation overhead.
