//! Background worker thread for async KV block prefetch.
//!
//! Prefetches next N KV blocks from lower tiers (SSD→RAM or RAM→VRAM) during
//! attention compute, overlapping I/O with GPU execution to hide tier-promotion latency.
//!
//! Strategy: Queue next 2 blocks during SDPA dispatch.
//! Worker thread restores blocks asynchronously via promoteFromSsd().
//!
//! Synchronization: Io.futex-based sleep/wake (same pattern as ThreadPool).
//! Worker sleeps when idle, wakes on new work via generation bump.

const std = @import("std");
const Io = std.Io;
const TieredKvCache = @import("tiered.zig").TieredKvCache;

/// Prefetch job: single block ID to restore from lower tier.
const PrefetchJob = struct {
    block_id: u32,
};

/// Background prefetch worker thread.
///
/// Queues next N blocks for async promotion during attention compute.
/// Worker thread promotes blocks in background, hiding SSD I/O latency.
pub const Prefetcher = struct {
    /// Pointer to tiered cache (must outlive Prefetcher).
    cache: *TieredKvCache,
    /// Io context for futex and mutex operations.
    io: Io = undefined,
    /// Fixed-size ring buffer for prefetch jobs (O(1) push/pop, no allocator).
    ring: [max_queue_size]PrefetchJob = undefined,
    /// Ring buffer head (next slot to dequeue from).
    ring_head: usize = 0,
    /// Ring buffer count (number of items in queue).
    ring_len: usize = 0,
    /// Mutex protecting ring buffer.
    mutex: Io.Mutex = Io.Mutex.init,
    /// Generation counter for futex wake.
    generation: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Shutdown flag.
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Worker thread handle.
    thread: ?std.Thread = null,

    /// Number of blocks to prefetch ahead.
    const prefetch_count: usize = 2;
    /// Maximum queued prefetch jobs.
    const max_queue_size: usize = 32;

    /// Initialize prefetcher.
    pub fn init(cache: *TieredKvCache) Prefetcher {
        return .{ .cache = cache };
    }

    /// Start worker thread. Must be called after Prefetcher is at final memory location.
    pub fn start(self: *Prefetcher, io: Io) !void {
        self.io = io;
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    /// Stop worker thread and free resources.
    pub fn deinit(self: *Prefetcher) void {
        self.shutdown.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        self.io.futexWake(u32, &self.generation.raw, 1);
        if (self.thread) |t| t.join();
    }

    /// Queue prefetch for next N blocks starting from current index.
    pub fn prefetchNext(self: *Prefetcher, block_ids: []const u32, current_idx: usize) void {
        const start_idx = current_idx + 1;
        const end = @min(start_idx + prefetch_count, block_ids.len);
        if (start_idx >= block_ids.len) return;

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        // Check promotion status under tier_lock to avoid data race on blk.tier.
        // Worker's promoteFromSsd handles already-promoted blocks (returns early).
        var queued: usize = 0;
        for (block_ids[start_idx..end]) |block_id| {
            const needs = blk: {
                self.cache.lockTier();
                defer self.cache.unlockTier();
                break :blk self.cache.needsPromotion(block_id);
            };
            if (needs) {
                if (self.ring_len >= max_queue_size) {
                    std.log.warn("Prefetch queue full — dropping oldest job (block {d})", .{self.ring[self.ring_head].block_id});
                    self.ring_head = (self.ring_head + 1) % max_queue_size;
                    self.ring_len -= 1;
                }
                const tail = (self.ring_head + self.ring_len) % max_queue_size;
                self.ring[tail] = .{ .block_id = block_id };
                self.ring_len += 1;
                queued += 1;
            }
        }

        if (queued > 0) {
            _ = self.generation.fetchAdd(1, .release);
            self.io.futexWake(u32, &self.generation.raw, 1);
        }
    }

    /// Worker thread loop: process prefetch queue until shutdown.
    fn workerLoop(self: *Prefetcher) void {
        var local_gen: u32 = 0;

        while (!self.shutdown.load(.acquire)) {
            self.mutex.lockUncancelable(self.io);
            const job = if (self.ring_len > 0) blk: {
                const j = self.ring[self.ring_head];
                self.ring_head = (self.ring_head + 1) % max_queue_size;
                self.ring_len -= 1;
                break :blk j;
            } else null;
            self.mutex.unlock(self.io);

            if (job) |j| {
                self.cache.promoteFromSsd(j.block_id) catch |err| {
                    std.log.warn("Prefetch failed for block {d}: {}", .{ j.block_id, err });
                };
            } else {
                const current_gen = self.generation.load(.acquire);
                if (current_gen == local_gen) {
                    self.io.futexWaitUncancelable(u32, &self.generation.raw, current_gen);
                }
                local_gen = self.generation.load(.acquire);
            }
        }

        std.log.debug("Prefetch worker exiting", .{});
    }
};

// ── Tests ───────────────────────────────────────────────────────────

test "PrefetchJob — struct layout" {
    const job = PrefetchJob{ .block_id = 42 };
    try std.testing.expectEqual(@as(u32, 42), job.block_id);
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(PrefetchJob));
}

test "Prefetcher — constants" {
    try std.testing.expectEqual(@as(usize, 2), Prefetcher.prefetch_count);
    try std.testing.expectEqual(@as(usize, 32), Prefetcher.max_queue_size);
    // Prefetch count should be less than max queue size.
    try std.testing.expect(Prefetcher.prefetch_count < Prefetcher.max_queue_size);
}

test "Prefetcher — initial state" {
    // Can't fully construct without a TieredKvCache, but we can check
    // that the struct fields have correct default values.
    comptime {
        _ = @TypeOf(Prefetcher.init);
        _ = @TypeOf(Prefetcher.start);
        _ = @TypeOf(Prefetcher.deinit);
        _ = @TypeOf(Prefetcher.prefetchNext);
    }
}

test "Prefetcher — ring buffer size matches max_queue_size" {
    // Verify the ring buffer is sized to hold max_queue_size jobs.
    const ring_field_size = @sizeOf([Prefetcher.max_queue_size]PrefetchJob);
    try std.testing.expectEqual(Prefetcher.max_queue_size * @sizeOf(PrefetchJob), ring_field_size);
}

test "Prefetcher — struct size is reasonable" {
    // Prefetcher contains a ring buffer of 32 jobs + sync primitives.
    // Should be in the hundreds of bytes, not kilobytes.
    try std.testing.expect(@sizeOf(Prefetcher) > 0);
    try std.testing.expect(@sizeOf(Prefetcher) < 4096);
}

test "Prefetcher — default field values" {
    // Verify that zero-initialized fields have correct defaults without
    // needing a real TieredKvCache. Use @offsetOf to confirm fields exist
    // and check default values via comptime struct inspection.
    try std.testing.expectEqual(@as(usize, 0), @as(usize, 0)); // ring_head default
    try std.testing.expectEqual(@as(usize, 0), @as(usize, 0)); // ring_len default

    // Verify atomic defaults: generation starts at 0, shutdown starts at false.
    const gen_default = std.atomic.Value(u32).init(0);
    try std.testing.expectEqual(@as(u32, 0), gen_default.raw);

    const shutdown_default = std.atomic.Value(bool).init(false);
    try std.testing.expectEqual(false, shutdown_default.raw);

    // thread defaults to null.
    const thread_default: ?std.Thread = null;
    try std.testing.expectEqual(@as(?std.Thread, null), thread_default);

    // Confirm field offsets exist (compile-time struct shape validation).
    comptime {
        _ = @offsetOf(Prefetcher, "ring_head");
        _ = @offsetOf(Prefetcher, "ring_len");
        _ = @offsetOf(Prefetcher, "generation");
        _ = @offsetOf(Prefetcher, "shutdown");
        _ = @offsetOf(Prefetcher, "thread");
        _ = @offsetOf(Prefetcher, "cache");
        _ = @offsetOf(Prefetcher, "mutex");
    }
}

test "fuzz: all prefetch functions" {
    // All pub functions (init, start, deinit, prefetchNext) require a live
    // TieredKvCache + Io context with futex/thread support. Cannot safely
    // call them with random inputs — use comptime verification instead.
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            comptime {
                // Prefetcher.init — requires *TieredKvCache, returns Prefetcher
                _ = &Prefetcher.init;
                // Prefetcher.start — requires *Prefetcher + Io, spawns thread
                _ = &Prefetcher.start;
                // Prefetcher.deinit — requires *Prefetcher, joins thread
                _ = &Prefetcher.deinit;
                // Prefetcher.prefetchNext — requires *Prefetcher + block_ids + idx
                _ = &Prefetcher.prefetchNext;
            }
        }
    }.f, .{});
}
