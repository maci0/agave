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

        var queued: usize = 0;
        for (block_ids[start_idx..end]) |block_id| {
            if (self.cache.needsPromotion(block_id)) {
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
