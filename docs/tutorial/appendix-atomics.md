# Appendix: Atomic Operations and Memory Ordering

Multi-threaded code needs **synchronization** to coordinate between threads. Zig provides **atomic operations** — CPU instructions that read-modify-write memory **atomically** (as one indivisible operation, preventing race conditions).

## The Problem: Race Conditions

Without atomics, concurrent writes corrupt data:

```zig
// WRONG: Race condition
var counter: usize = 0;

fn workerThread() void {
    counter += 1;  // Read counter, add 1, write back (3 separate operations)
}

// If 2 threads run workerThread() concurrently:
// Thread A reads counter=0
// Thread B reads counter=0  ← Both read 0 before either writes!
// Thread A writes counter=1
// Thread B writes counter=1  ← Overwrites A's update!
// Final value: 1 (expected: 2)
```

**The issue:** `counter += 1` is **not atomic** — it compiles to:

```asm
mov  r0, [counter]   ; Read
add  r0, 1           ; Modify
mov  [counter], r0   ; Write
```

Between any two instructions, another thread can run and see inconsistent state.

## Atomic Operations

**`std.atomic.Value(T)`** provides atomic read-modify-write operations:

```zig
var counter = std.atomic.Value(usize).init(0);

fn workerThread() void {
    _ = counter.fetchAdd(1, .monotonic);  // Atomic increment
}

// Guaranteed: 2 threads → counter = 2
```

**How it works:** `fetchAdd` compiles to a single CPU instruction (e.g., x86 `lock add` or ARM `ldadd`) that the hardware guarantees is atomic.

### Common Operations

```zig
var val = std.atomic.Value(u32).init(10);

// Fetch-and-add: returns old value, adds delta
const old = val.fetchAdd(5, .monotonic);  // old=10, val=15

// Fetch-and-sub: returns old value, subtracts delta
const old2 = val.fetchSub(3, .monotonic);  // old2=15, val=12

// Compare-and-swap: only update if current value matches expected
const swapped = val.cmpxchgStrong(12, 20, .monotonic, .monotonic);
if (swapped == null) {
    // Swap succeeded: val=20
} else {
    // Swap failed: val still 12, someone else changed it
}

// Load: atomic read
const current = val.load(.monotonic);

// Store: atomic write
val.store(50, .monotonic);
```

## Memory Ordering

**Memory ordering** controls **when other threads see your writes** and **when you see their writes**.

### The Four Orders (Weakest to Strongest)

#### .monotonic — No Synchronization

**Guarantees:**

- Operation is atomic (no torn reads/writes)
- **No** ordering guarantees relative to other operations

**Use for:** Simple counters where you don't care about ordering.

```zig
var counter = std.atomic.Value(usize).init(0);

// Thread A
_ = counter.fetchAdd(1, .monotonic);

// Thread B
const val = counter.load(.monotonic);
// val could be 0 or 1 — no guarantee when the write is visible
```

**Example from thread pool:**

```zig
const start = self.task_counter.fetchAdd(grain, .monotonic);
```

**Why monotonic?** The counter value doesn't carry ordering information — it's just work assignment. Each thread grabs a chunk independently.

#### .acquire (Load) / .release (Store) — Publish/Subscribe

**Release** (on store): All writes **before** this store become visible to other threads **before** the store itself.

**Acquire** (on load): All writes that happened **before** a release store are visible **after** this load.

**Use for:** Handing off data between threads.

```zig
var ready = std.atomic.Value(bool).init(false);
var data: [100]u8 = undefined;

// Producer thread
for (0..100) |i| {
    data[i] = compute(i);  // Fill data
}
ready.store(true, .release);  // Publish: data writes happen-before this store

// Consumer thread
while (!ready.load(.acquire)) {}  // Wait until ready
// Now safe to read data — all writes are visible
for (data) |d| {
    process(d);
}
```

**Guarantee:** If consumer sees `ready=true`, it's guaranteed to see the fully-filled `data` array.

**Example from thread pool:**

```zig
// Main thread: publish work
self.task_func = func;
self.task_ctx = ctx;
self.task_total = total;
_ = self.generation.fetchAdd(1, .release);  // All writes happen-before this
std.Thread.Futex.wake(&self.generation, n_workers);

// Worker thread: subscribe
local_gen = self.generation.load(.acquire);  // See all writes before release
// Safe to read task_func, task_ctx, task_total
```

#### .seq_cst — Sequential Consistency

**Guarantees:** All threads see all operations in the **same global order**.

**Use for:** When you need total ordering (rare).

**Cost:** Slowest — requires full memory fence on most architectures.

**Avoid unless necessary** — acquire/release is sufficient for most cases.

### Choosing Memory Ordering

| Use Case | Load | Store | Rationale |
| -------- | ---- | ----- | --------- |
| Simple counter | `.monotonic` | `.monotonic` | Just need atomicity, not ordering |
| Work-stealing queue | `.acquire` | `.release` | Hand off work between threads |
| Shutdown flag | `.acquire` | `.release` | Ensure all cleanup happens before shutdown visible |
| Lock-free data structure | `.acquire` | `.release` | Synchronize data structure updates |
| (Rare) Total order needed | `.seq_cst` | `.seq_cst` | All threads must agree on operation order |

**Agave uses:** Mostly `.monotonic` for counters, `.acquire`/`.release` for handoff, **never `.seq_cst`**.

## Real-World Examples from Agave

### Thread Pool Work Counter

```zig
// src/thread_pool.zig
task_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

fn doWork(self: *ThreadPool) void {
    while (true) {
        const start = self.task_counter.fetchAdd(self.task_grain, .monotonic);
        if (start >= self.task_total) break;
        const end = @min(start + self.task_grain, self.task_total);
        self.task_func.?(self.task_ctx.?, start, end);
    }
}
```

**Why `.monotonic`?**

- We only care about **which chunk** each thread gets
- No data dependency between chunks
- No synchronization needed — each thread works independently

### Generation Counter (Thread Wake-Up)

```zig
generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

// Main thread: post work
self.task_func = func;
self.task_ctx = ctx;
self.task_total = total;
self.task_counter.store(0, .release);  // Reset counter
self.active.store(@intCast(self.n_workers), .release);
_ = self.generation.fetchAdd(1, .release);  // Publish: all task fields valid
std.Thread.Futex.wake(&self.generation, @intCast(self.n_workers));
```

**Why `.release`?**

- Ensures `task_func`, `task_ctx`, `task_total` are visible to workers **before** they see the generation bump
- Without release, workers could see new generation but stale task fields → undefined behavior

```zig
// Worker thread: consume work
fn workerLoop(pool: *ThreadPool) void {
    var local_gen: u32 = 0;
    while (true) {
        std.Thread.Futex.wait(&pool.generation, local_gen);
        if (pool.shutdown.load(.acquire)) return;

        local_gen = pool.generation.load(.acquire);  // See all task fields
        pool.doWork();
        _ = pool.active.fetchSub(1, .release);  // Signal completion
    }
}
```

**Why `.acquire`?**

- Ensures worker sees all writes (task fields) that happened-before the `.release` store on main thread
- Without acquire, worker might see partial task state

### Active Thread Counter

```zig
active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

// Worker: signal completion
_ = pool.active.fetchSub(1, .release);

// Main thread: wait for completion
while (pool.active.load(.acquire) != 0) {
    std.atomic.spinLoopHint();
}
```

**Why `.release`/`.acquire`?**

- Workers' writes to output buffers must be visible to main thread when `active` reaches 0
- Release on `fetchSub` publishes all worker writes
- Acquire on `load` ensures main thread sees all worker writes

### Shutdown Flag

```zig
shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

// Main thread: signal shutdown
self.shutdown.store(true, .release);
_ = self.generation.fetchAdd(1, .release);
std.Thread.Futex.wake(&self.generation, @intCast(self.n_workers));

// Worker: check shutdown
if (pool.shutdown.load(.acquire)) return;
```

**Why `.release`/`.acquire`?**

- Ensures all cleanup (e.g., closing files, flushing buffers) happens-before shutdown is visible
- Workers see a consistent view of cleaned-up state

## Compare-and-Swap (CAS)

**Problem:** Update a value only if it hasn't changed since you last read it.

**Example:** Lock-free stack push:

```zig
pub fn push(self: *LockFreeStack, item: *Node) void {
    while (true) {
        const current_head = self.head.load(.acquire);
        item.next = current_head;

        // Try to swap: if head still equals current_head, set it to item
        const result = self.head.cmpxchgWeak(
            current_head,
            item,
            .release,  // On success: publish item.next write
            .acquire,  // On failure: see why it failed
        );

        if (result == null) {
            // Success: head was current_head, now it's item
            return;
        }
        // Failure: head changed, retry with new head value
    }
}
```

**cmpxchgWeak vs cmpxchgStrong:**

- **Weak:** May spuriously fail (return failure even if values match). Faster on some architectures (ARM).
- **Strong:** Only fails if values don't match. Slightly slower.

**Use weak in loops** (spurious failure just retries), **strong for one-shot CAS**.

## Spin Loop Hint

When spinning (busy-waiting), hint the CPU to save power:

```zig
while (pool.active.load(.acquire) != 0) {
    std.atomic.spinLoopHint();  // Maps to `pause` (x86) or `yield` (ARM)
}
```

**What it does:**

- **x86:** `pause` — reduces power consumption, lets hyperthreading switch to other logical core
- **ARM:** `yield` — hints scheduler to switch to another thread
- **Without hint:** CPU burns 100% power, spins at max frequency

**Cost:** ~5-10 cycles per hint (negligible).

## Fence

**Explicit memory barrier** — rarely needed in Zig (acquire/release is usually sufficient).

```zig
std.atomic.fence(.release);  // All writes before this are visible
// ... some non-atomic write ...
std.atomic.fence(.acquire);  // All writes after this see prior writes
```

**Use when:** Synchronizing non-atomic writes with atomic operations.

**Example (rare):**

```zig
// Non-atomic writes
self.data[0] = 42;
self.data[1] = 43;

std.atomic.fence(.release);  // Publish data writes

self.ready.store(true, .monotonic);  // Signal ready (no need for release here — fence did it)
```

**Agave doesn't use fences** — acquire/release on atomic operations is clearer and sufficient.

## Common Pitfalls

### Pitfall 1: Using Non-Atomic for Synchronization

```zig
// WRONG: Data race
var flag: bool = false;  // Not atomic!

// Thread A
data.fill();
flag = true;  // Write

// Thread B
if (flag) {  // Read
    data.process();  // May see partially-filled data!
}
```

**Fix:** Use `std.atomic.Value(bool)` with proper ordering.

### Pitfall 2: Missing Acquire/Release

```zig
// WRONG: Missing release
var counter = std.atomic.Value(usize).init(0);

// Producer
data[0] = compute();
counter.store(1, .monotonic);  // Should be .release!

// Consumer
if (counter.load(.monotonic) == 1) {  // Should be .acquire!
    process(data[0]);  // May see stale data!
}
```

**Fix:** Use `.release` on store, `.acquire` on load.

### Pitfall 3: Assuming Atomicity Without Explicit Atomic Type

```zig
// WRONG: Not atomic on all platforms
var x: u64 = 0;

// Thread A
x = 123;  // May be two 32-bit stores on 32-bit platforms!

// Thread B
const val = x;  // May read torn value (high/low half from different writes)
```

**Fix:** Use `std.atomic.Value(u64)` for guaranteed atomicity.

### Pitfall 4: Overusing .seq_cst

```zig
// WRONG: Unnecessarily slow
var counter = std.atomic.Value(usize).init(0);
_ = counter.fetchAdd(1, .seq_cst);  // Should be .monotonic!
```

**Fix:** Use weakest ordering that provides required guarantees.

## Performance Characteristics

**Atomic operation cost** (Apple M4, approximate):

| Operation | Ordering | Latency | Throughput |
| --------- | -------- | ------- | ---------- |
| `load` | `.monotonic` | ~1 cycle | ~1 per cycle |
| `load` | `.acquire` | ~1-2 cycles | ~1 per cycle |
| `store` | `.monotonic` | ~1 cycle | ~1 per cycle |
| `store` | `.release` | ~1-3 cycles | ~0.5 per cycle |
| `fetchAdd` | `.monotonic` | ~5 cycles | ~1 per 3 cycles |
| `fetchAdd` | `.acquire`/`.release` | ~10 cycles | ~1 per 5 cycles |
| `cmpxchgWeak` | `.monotonic` | ~10 cycles | ~1 per 10 cycles |
| `cmpxchgWeak` | `.release` | ~15 cycles | ~1 per 15 cycles |
| Non-atomic load/store | N/A | ~1 cycle | ~2 per cycle |

**Takeaways:**

- Atomics are 1-10× slower than non-atomic ops
- Stronger ordering = slower
- Still very fast in absolute terms (nanoseconds)

**When to use:**

- ✅ Synchronization between threads
- ✅ Counters with infrequent updates
- ❌ Hot-path per-element operations (use SIMD instead)

## Best Practices

1. **Start with `.monotonic`**, upgrade to `.acquire`/`.release` only when needed
2. **Never use `.seq_cst`** unless you can articulate why total ordering is required
3. **Pair `.release` stores with `.acquire` loads** for handoff
4. **Use non-atomic for thread-local data** (faster)
5. **Profile before optimizing** — atomics are usually not the bottleneck

## Testing for Race Conditions

**ThreadSanitizer (TSan)** detects data races at runtime:

```bash
zig build -Dsanitize-thread
./zig-out/bin/agave-test

# Output if race detected:
# WARNING: ThreadSanitizer: data race
#   Write of size 8 at 0x7b0400000000 by thread T2:
#     #0 workerThread thread_pool.zig:123
#   Previous read of size 8 at 0x7b0400000000 by thread T1:
#     #0 doWork thread_pool.zig:115
```

**Use TSan in CI** to catch races before production.

---

**In the code:** [src/thread_pool.zig](../../src/thread_pool.zig) (extensive use of atomics for synchronization), [src/backend/cpu.zig](../../src/backend/cpu.zig) (atomic cancellation flag)

**Related:** [Chapter 12: CPU Parallelism](12-cpu-parallelism.md#memory-ordering), [Zig std.atomic documentation](https://ziglang.org/documentation/master/std/#std.atomic)

**Back:** [Appendix: Profiling and Debugging ←](appendix-profiling.md)
