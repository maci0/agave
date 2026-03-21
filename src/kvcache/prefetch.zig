//! Background worker thread for async KV block prefetch.
//!
//! Prefetches next N KV blocks from lower tiers (SSD→RAM) during attention
//! compute, overlapping I/O with GPU execution to hide SSD read latency.
//!
//! Strategy: Queue next 2 blocks (per decision D-07) during SDPA dispatch.
//! Worker thread restores blocks asynchronously via promoteFromSsd().
//!
//! Synchronization: Futex-based sleep/wake (same pattern as ThreadPool).
//! Worker sleeps when idle, wakes on new work via generation bump.

const std = @import("std");
const Allocator = std.mem.Allocator;
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
    /// Work queue of blocks to prefetch.
    work_queue: std.ArrayList(PrefetchJob),
    /// Mutex protecting work_queue.
    mutex: std.Thread.Mutex,
    /// Generation counter for futex wake.
    generation: std.atomic.Value(u64),
    /// Shutdown flag.
    shutdown: std.atomic.Value(bool),
    /// Worker thread handle.
    thread: ?std.Thread = null,
    /// Allocator for work queue.
    allocator: Allocator,

    /// Number of blocks to prefetch ahead (per decision D-07: next 2 blocks).
    const prefetch_count: usize = 2;

    /// Initialize prefetcher.
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for work queue.
    ///   - cache: Pointer to TieredKvCache (must outlive Prefetcher).
    ///
    /// Returns: Prefetcher instance (call start() to spawn worker thread).
    pub fn init(allocator: Allocator, cache: *TieredKvCache) !Prefetcher {
        return .{
            .cache = cache,
            .work_queue = std.ArrayList(PrefetchJob).init(allocator),
            .mutex = .{},
            .generation = std.atomic.Value(u64).init(0),
            .shutdown = std.atomic.Value(bool).init(false),
            .allocator = allocator,
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

        self.work_queue.deinit();
    }

    /// Queue prefetch for next N blocks starting from current index.
    ///
    /// Called by scheduler during attention compute (per D-07: next 2 blocks).
    /// Only queues blocks that are in lower tier (SSD or RAM → VRAM).
    ///
    /// Parameters:
    ///   - block_ids: Block table for current request.
    ///   - current_idx: Current block index being processed by SDPA.
    ///
    /// Returns: void on success.
    pub fn prefetchNext(self: *Prefetcher, block_ids: []const u32, current_idx: usize) !void {
        const start_idx = current_idx + 1;
        const end = @min(start_idx + prefetch_count, block_ids.len);

        if (start_idx >= block_ids.len) return; // Already at end

        self.mutex.lock();
        defer self.mutex.unlock();

        var queued: usize = 0;
        for (block_ids[start_idx..end]) |block_id| {
            // Only queue if block is in lower tier (SSD or RAM → VRAM)
            if (self.cache.needsPromotion(block_id)) {
                try self.work_queue.append(.{ .block_id = block_id });
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
        var local_gen: u64 = 0; // Critical: init to 0, not generation.load

        while (!self.shutdown.load(.acquire)) {
            self.mutex.lock();
            const has_work = self.work_queue.items.len > 0;
            const job = if (has_work) self.work_queue.orderedRemove(0) else null;
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
