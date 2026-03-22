//! Lightweight thread pool for parallel-for workloads.
//! Workers sleep on futex when idle. Main thread participates in work.
//! Atomic counter provides dynamic work distribution across threads.

const std = @import("std");

/// Maximum number of worker threads (excludes main thread which also participates).
const max_workers: usize = 31;

/// Minimum rows per thread to avoid dispatch overhead dominating.
const min_grain: usize = 4;

/// Futex-based thread pool for parallel GEMV and other data-parallel ops.
pub const ThreadPool = struct {
    workers: [max_workers]Worker = undefined,
    n_workers: usize = 0,

    // ── Shared task descriptor ──────────────────────────────────
    // Written by dispatch(), read by workers. Protected by generation counter.
    task_func: ?*const fn (*anyopaque, usize, usize) void = null,
    task_ctx: ?*anyopaque = null,
    task_total: usize = 0,
    task_grain: usize = 1,
    task_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

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
    pub fn spawn(self: *ThreadPool) void {
        for (0..self.n_workers) |i| {
            self.workers[i] = .{
                .thread = std.Thread.spawn(.{}, workerLoop, .{self}) catch {
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
        std.Thread.Futex.wake(&self.generation, @intCast(self.n_workers));
        for (0..self.n_workers) |i| {
            self.workers[i].thread.join();
        }
        self.n_workers = 0;
    }

    /// Execute `func(ctx, start, end)` over the range [0, total) in parallel.
    /// Splits work into chunks of `grain` rows. Main thread participates.
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

        // Post task
        self.task_func = func;
        self.task_ctx = ctx;
        self.task_total = total;
        self.task_grain = effective_grain;
        self.task_counter.store(0, .release);
        self.active.store(@intCast(self.n_workers), .release);

        // Wake workers by bumping generation
        _ = self.generation.fetchAdd(1, .release);
        std.Thread.Futex.wake(&self.generation, @intCast(self.n_workers));

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
        // Start at 0 so late-starting workers see already-bumped generation
        // and proceed immediately instead of missing the wake.
        var local_gen: u32 = 0;

        while (true) {
            // Sleep until generation advances past our local copy
            std.Thread.Futex.wait(&pool.generation, local_gen);

            if (pool.shutdown.load(.acquire)) return;

            local_gen = pool.generation.load(.acquire);

            // Do work
            pool.doWork();

            // Signal completion
            _ = pool.active.fetchSub(1, .release);
        }
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "ThreadPool basic parallelFor" {
    var pool = ThreadPool.init(3);
    pool.spawn();
    defer pool.deinit();

    var results: [100]f32 = undefined;
    const Ctx = struct {
        out: *[100]f32,

        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |i| {
                ctx.out[i] = @floatFromInt(i * i);
            }
        }
    };
    var ctx = Ctx{ .out = &results };
    pool.parallelFor(100, 8, @ptrCast(&ctx), Ctx.work);

    for (0..100) |i| {
        try std.testing.expectApproxEqAbs(
            @as(f32, @floatFromInt(i * i)),
            results[i],
            0.001,
        );
    }
}

test "ThreadPool single item" {
    var pool = ThreadPool.init(2);
    pool.spawn();
    defer pool.deinit();

    var val: u32 = 0;
    const Ctx = struct {
        v: *u32,
        fn work(ctx_ptr: *anyopaque, _: usize, _: usize) void {
            const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
            ctx.v.* = 42;
        }
    };
    var ctx = Ctx{ .v = &val };
    pool.parallelFor(1, 1, @ptrCast(&ctx), Ctx.work);
    try std.testing.expectEqual(@as(u32, 42), val);
}

test "ThreadPool zero workers (single-threaded)" {
    var pool = ThreadPool.init(0);
    pool.spawn();
    defer pool.deinit();

    var sum: u64 = 0;
    const Ctx = struct {
        s: *u64,
        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |i| ctx.s.* += i;
        }
    };
    var ctx = Ctx{ .s = &sum };
    pool.parallelFor(10, 4, @ptrCast(&ctx), Ctx.work);
    // sum(0..9) = 45
    try std.testing.expectEqual(@as(u64, 45), sum);
}
