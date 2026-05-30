//! Lightweight thread pool for parallel-for workloads.
//! Workers sleep on Io.futex when idle. Main thread participates in work.
//! Atomic counter provides dynamic work distribution across threads.

const std = @import("std");
const Io = std.Io;

/// Maximum number of worker threads (excludes main thread which also participates).
const max_workers: usize = 31;

/// Minimum iterations per thread to avoid dispatch overhead dominating.
const min_grain: usize = 4;

/// Cache line size for padding to prevent false sharing between hot atomics.
const cache_line: usize = 64;

/// Futex-based thread pool for parallel GEMV and other data-parallel ops.
pub const ThreadPool = struct {
    workers: [max_workers]Worker = undefined,
    n_workers: usize = 0,

    /// Io context for futex operations. Set during spawn().
    io: Io = undefined,

    // ── Shared task descriptor ──────────────────────────────────
    // Written by dispatch(), read by workers. Protected by generation counter.
    task_func: ?*const fn (*anyopaque, usize, usize) void = null,
    task_ctx: ?*anyopaque = null,
    task_total: usize = 0,
    task_grain: usize = 1,
    task_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    // Pad task_counter onto its own cache line. task_counter is the hottest
    // field — every worker does fetchAdd per grain chunk. Without padding,
    // generation/active share the same cache line, causing cross-core
    // invalidation traffic when workers finish and decrement active while
    // remaining workers are still pulling from task_counter.
    _counter_pad: [cache_line - @sizeOf(std.atomic.Value(usize))]u8 = undefined,

    // ── Synchronization ─────────────────────────────────────────
    /// Incremented each time new work is posted. Workers compare against
    /// their local copy to detect new work (avoids spurious wakes).
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Number of workers still processing. Dispatcher waits until 0.
    active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    const Worker = struct {
        thread: std.Thread,
    };

    /// Create a thread pool descriptor with `n` worker threads.
    /// Does NOT spawn threads — call `spawn()` after the pool is at its
    /// final memory location (e.g. embedded in a struct, not a stack local
    /// that will be returned by value).
    pub fn init(n: usize) ThreadPool {
        return .{ .n_workers = @min(n, max_workers) };
    }

    /// Spawn worker threads. Must be called exactly once, after the pool is
    /// at its final memory location. Workers capture `self` by pointer.
    pub fn spawn(self: *ThreadPool, io: Io) void {
        self.io = io;
        for (0..self.n_workers) |i| {
            self.workers[i] = .{
                .thread = std.Thread.spawn(.{}, workerLoop, .{self}) catch |err| {
                    std.log.warn("ThreadPool: failed to spawn worker {d}: {s}", .{ i, @errorName(err) });
                    self.n_workers = i;
                    return;
                },
            };
        }
    }

    /// Shut down all worker threads.
    pub fn deinit(self: *ThreadPool) void {
        self.shutdown.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, @intCast(self.n_workers));
        for (0..self.n_workers) |i| {
            self.workers[i].thread.join();
        }
        self.n_workers = 0;
    }

    /// Execute `func(ctx, start, end)` over the range [0, total) in parallel.
    /// Splits work into chunks of `grain` items. Main thread participates.
    /// Blocks until all work is complete.
    pub fn parallelFor(
        self: *ThreadPool,
        total: usize,
        grain: usize,
        ctx: *anyopaque,
        func: *const fn (*anyopaque, usize, usize) void,
    ) void {
        if (total == 0) return;

        const effective_grain = @max(grain, min_grain);

        // If work is too small for parallelism, run inline
        if (self.n_workers == 0 or total <= effective_grain) {
            func(ctx, 0, total);
            return;
        }

        // Atomically claim the pool: active 0 → n_workers.
        // CAS eliminates the TOCTOU in a load-then-store guard: if two callers
        // race, only one succeeds; the other falls back to inline execution.
        if (self.active.cmpxchgWeak(0, @intCast(self.n_workers), .acq_rel, .monotonic)) |still_active| {
            std.log.err("ThreadPool: concurrent parallelFor detected (active={d}), running inline", .{still_active});
            func(ctx, 0, total);
            return;
        }

        // Post task (published to workers by generation.fetchAdd release below)
        self.task_func = func;
        self.task_ctx = ctx;
        self.task_total = total;
        self.task_grain = effective_grain;
        self.task_counter.store(0, .release);

        // Wake workers by bumping generation
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, @intCast(self.n_workers));

        // Main thread participates
        self.doWork();

        // Spin-wait for workers. GEMV chunks are microsecond-scale,
        // so spinning avoids futex syscall overhead and is simpler to reason about.
        while (self.active.load(.acquire) != 0) {
            std.atomic.spinLoopHint();
        }
    }

    /// Grab and execute chunks until none remain.
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

    /// Worker thread main loop. Sleeps on generation futex, wakes to do work.
    fn workerLoop(pool: *ThreadPool) void {
        // Start at 0 to ensure workers wake on the first generation bump (0→1).
        var local_gen: u32 = 0;

        while (true) {
            // Sleep until generation advances past our local copy
            pool.io.futexWaitUncancelable(u32, &pool.generation.raw, local_gen);

            if (pool.shutdown.load(.acquire)) return;

            const new_gen = pool.generation.load(.acquire);
            if (new_gen == local_gen) continue; // spurious wakeup
            local_gen = new_gen;

            // Do work
            pool.doWork();

            // Signal completion
            _ = pool.active.fetchSub(1, .release);
        }
    }
};

// ── Tests ────────────────────────────────────────────────────

/// Test context for parallelFor: atomically accumulates a sum of indices.
const SumContext = struct {
    result: std.atomic.Value(usize),

    fn callback(ctx_ptr: *anyopaque, start: usize, end: usize) void {
        const self: *SumContext = @ptrCast(@alignCast(ctx_ptr));
        var local_sum: usize = 0;
        for (start..end) |i| {
            local_sum += i;
        }
        _ = self.result.fetchAdd(local_sum, .acq_rel);
    }
};

test "parallelFor basic" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();

    var pool = ThreadPool.init(4);
    pool.spawn(threaded.io());
    defer pool.deinit();

    var ctx = SumContext{ .result = std.atomic.Value(usize).init(0) };
    const n: usize = 1000;
    pool.parallelFor(n, 16, @ptrCast(&ctx), SumContext.callback);

    // Sum of 0..999 = 999 * 1000 / 2 = 499500
    const expected: usize = (n - 1) * n / 2;
    try std.testing.expectEqual(expected, ctx.result.load(.acquire));
}

test "parallelFor single element" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();

    var pool = ThreadPool.init(4);
    pool.spawn(threaded.io());
    defer pool.deinit();

    var ctx = SumContext{ .result = std.atomic.Value(usize).init(0) };
    pool.parallelFor(1, 1, @ptrCast(&ctx), SumContext.callback);

    // Only index 0, so sum = 0
    try std.testing.expectEqual(@as(usize, 0), ctx.result.load(.acquire));
}

test "parallelFor zero elements" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();

    var pool = ThreadPool.init(4);
    pool.spawn(threaded.io());
    defer pool.deinit();

    var ctx = SumContext{ .result = std.atomic.Value(usize).init(42) };
    pool.parallelFor(0, 1, @ptrCast(&ctx), SumContext.callback);

    // No work done — result should remain at initial value
    try std.testing.expectEqual(@as(usize, 42), ctx.result.load(.acquire));
}

test "init and deinit without work" {
    var threaded = Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();

    var pool = ThreadPool.init(4);
    pool.spawn(threaded.io());

    // Verify workers were spawned
    try std.testing.expect(pool.n_workers > 0);
    try std.testing.expect(pool.n_workers <= max_workers);

    pool.deinit();

    // After deinit, n_workers should be 0
    try std.testing.expectEqual(@as(usize, 0), pool.n_workers);
}

test "fuzz: ThreadPool parallelFor" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var raw: [4]u8 = undefined;
            smith.bytesWithHash(&raw, 0);

            // Vary total and grain size with random inputs.
            const total: usize = @as(usize, raw[0]) * 4 + 1; // 1..1021
            const grain: usize = @as(usize, raw[1] % 32) + 1; // 1..32
            const n_threads: usize = @as(usize, raw[2] % 4) + 1; // 1..4

            var threaded = Io.Threaded.init(std.testing.allocator, .{});
            defer threaded.deinit();

            var pool = ThreadPool.init(n_threads);
            pool.spawn(threaded.io());
            defer pool.deinit();

            var ctx = SumContext{ .result = std.atomic.Value(usize).init(0) };
            pool.parallelFor(total, grain, @ptrCast(&ctx), SumContext.callback);

            // Invariant: sum(0..total-1) must be exact regardless of grain/thread count.
            const expected: usize = (total - 1) * total / 2;
            try std.testing.expectEqual(expected, ctx.result.load(.acquire));
        }
    }.f, .{});
}
