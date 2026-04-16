//! Background worker thread for async KV block prefetch.
//!
//! Prefetches next N KV blocks from lower tiers (SSD→RAM or RAM→VRAM) during
//! attention compute, overlapping I/O with GPU execution to hide tier-promotion latency.
//!
//! Strategy: Queue next 2 blocks during SDPA dispatch.
//! Worker thread restores blocks asynchronously via promoteFromSsd().
//!
//! Synchronization: Futex-based sleep/wake (same pattern as ThreadPool).
//! Worker sleeps when idle, wakes on new work via generation bump.

const std = @import("std");
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
    /// Fixed-size ring buffer for prefetch jobs (O(1) push/pop, no allocator).
    ring: [max_queue_size]PrefetchJob = undefined,
    /// Ring buffer head (next slot to dequeue from).
    ring_head: usize = 0,
    /// Ring buffer count (number of items in queue).
    ring_len: usize = 0,
    /// Mutex protecting ring buffer.
    mutex: std.Thread.Mutex,
    /// Generation counter for futex wake.
    generation: std.atomic.Value(u32),
    /// Shutdown flag.
    shutdown: std.atomic.Value(bool),
    /// Worker thread handle.
    thread: ?std.Thread = null,

    /// Number of blocks to prefetch ahead.
    const prefetch_count: usize = 2;
    /// Maximum queued prefetch jobs. Prevents unbounded growth if worker
    /// falls behind (e.g., slow SSD under burst). Oldest jobs are dropped.
    const max_queue_size: usize = 32;

    /// Initialize prefetcher.
    ///
    /// Parameters:
    ///   - cache: Pointer to TieredKvCache (must outlive Prefetcher).
    ///
    /// Returns: Prefetcher instance (call start() to spawn worker thread).
    pub fn init(cache: *TieredKvCache) Prefetcher {
        return .{
            .cache = cache,
            .mutex = .{},
            .generation = std.atomic.Value(u32).init(0),
            .shutdown = std.atomic.Value(bool).init(false),
        };
    }

    /// Start worker thread.
    /// Must be called after Prefetcher is at final memory location.
    pub fn start(self: *Prefetcher) !void {
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    /// Stop worker thread and free resources.
    pub fn deinit(self: *Prefetcher) void {
        // Signal shutdown
        self.shutdown.store(true, .release);
        _ = self.generation.fetchAdd(1, .release);
        std.Thread.Futex.wake(&self.generation, 1);

        // Wait for worker to exit
        if (self.thread) |t| t.join();
    }

    /// Queue prefetch for next N blocks starting from current index.
    ///
    /// Called by scheduler during attention compute.
    /// Only queues blocks that are in lower tier (SSD or RAM → VRAM).
    ///
    /// Parameters:
    ///   - block_ids: Block table for current request.
    ///   - current_idx: Current block index being processed by SDPA.
    ///
    /// Returns: void on success.
    pub fn prefetchNext(self: *Prefetcher, block_ids: []const u32, current_idx: usize) void {
        const start_idx = current_idx + 1;
        const end = @min(start_idx + prefetch_count, block_ids.len);

        if (start_idx >= block_ids.len) return; // Already at end

        self.mutex.lock();
        defer self.mutex.unlock();

        var queued: usize = 0;
        for (block_ids[start_idx..end]) |block_id| {
            // Only queue if block is in lower tier (SSD or RAM → VRAM)
            if (self.cache.needsPromotion(block_id)) {
                // Drop oldest job if ring is full to prevent unbounded growth
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

        // Wake worker if work added
        if (queued > 0) {
            _ = self.generation.fetchAdd(1, .release);
            std.Thread.Futex.wake(&self.generation, 1);
        }
    }

    /// Worker thread loop: process prefetch queue until shutdown.
    ///
    /// Promotes blocks asynchronously via TieredKvCache.promoteFromSsd().
    /// Sleeps on futex when idle, wakes on new work.
    ///
    /// Critical: local_gen MUST init to 0 (not generation.load) to avoid
    /// late-starting workers missing wake signals.
    fn workerLoop(self: *Prefetcher) void {
        var local_gen: u32 = 0; // Critical: init to 0, not generation.load

        while (!self.shutdown.load(.acquire)) {
            self.mutex.lock();
            const job = if (self.ring_len > 0) blk: {
                const j = self.ring[self.ring_head];
                self.ring_head = (self.ring_head + 1) % max_queue_size;
                self.ring_len -= 1;
                break :blk j;
            } else null;
            self.mutex.unlock();

            if (job) |j| {
                // Promote block asynchronously (SSD→RAM or RAM→VRAM)
                self.cache.promoteFromSsd(j.block_id) catch |err| {
                    std.log.warn("Prefetch failed for block {d}: {}", .{ j.block_id, err });
                };
            } else {
                // No work — sleep on futex
                const current_gen = self.generation.load(.acquire);
                if (current_gen == local_gen) {
                    std.Thread.Futex.wait(&self.generation, current_gen);
                }
                local_gen = self.generation.load(.acquire);
            }
        }

        std.log.debug("Prefetch worker exiting", .{});
    }
};
