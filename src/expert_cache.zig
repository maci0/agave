//! SSD expert streaming: demand-paged expert weight loading for MoE models.
//!
//! Large MoE models (e.g. DeepSeek V4 with 256 experts × 60 layers) may not
//! fit entirely in RAM/VRAM. This module manages a fixed-size expert cache
//! that keeps hot experts resident and streams cold experts from the mmap'd
//! model file on demand via madvise(WILLNEED).
//!
//! The cache holds `cache_experts` expert weight slabs. When the router selects
//! an expert not in the cache, the least-recently-used slab is evicted and the
//! new expert's weights are faulted in from the mmap'd file.
//!
//! Library API (no `--ssd-streaming` CLI yet; wire from MoE forward paths):
//!   var cache = try ExpertCache.init(allocator, n_layers, n_experts, cache_size);
//!   defer cache.deinit(allocator);
//!
//!   // Before MoE forward, ensure selected experts are resident:
//!   _ = cache.touch(layer, expert_id);
//!   cache.prefetch(layer, expert_id, weight_ptr, expert_bytes);
//!
//! Based on the SSD streaming approach from antirez/ds4.

const std = @import("std");
const Allocator = std.mem.Allocator;
const posix = std.posix;
const builtin = @import("builtin");

/// One cache slot tracking a resident expert.
const CacheSlot = struct {
    layer: u32 = 0,
    expert_id: u32 = 0,
    last_access: u64 = 0,
    occupied: bool = false,
};

/// A page-aligned byte range pinned via mlock, tracked for munlock on shutdown.
const PinnedRange = struct {
    ptr: [*]align(std.heap.page_size_min) const u8,
    len: usize,
};

/// Maximum number of pinned ranges (6 experts × 3 tensors × ~100 layers).
const max_pin_ranges: usize = 2048;

/// Maximum total bytes to pin (~30 GB — leave headroom for non-expert data).
const max_pinned_bytes: u64 = 30 * 1024 * 1024 * 1024;

/// LRU cache for MoE expert weights, tracking residency and eviction
/// across layers to minimize redundant GPU uploads.
pub const ExpertCache = struct {
    /// Maximum number of cached expert slots.
    pub const max_cache_slots: usize = 4096;

    slots: []CacheSlot,
    n_slots: u32,
    access_counter: u64 = 0,
    /// Per-layer, per-expert tracking for fast lookup.
    /// Maps (layer * n_experts + expert_id) → slot index or null.
    lookup: []?u32,
    n_layers: u32,
    n_experts: u32,
    /// Cache hit/miss stats.
    hits: u64 = 0,
    misses: u64 = 0,

    /// Pinned expert ranges for munlock on shutdown.
    pinned_ranges: []PinnedRange,
    n_pinned: u32 = 0,
    total_pinned_bytes: u64 = 0,
    /// When true, admit() is a no-op (no eviction during verification).
    frozen: bool = false,

    /// Allocates cache slots, the `(layer, expert) → slot` lookup table (clamped
    /// to `max_cache_slots`), and the pinned-range tracking array.
    pub fn init(allocator: Allocator, n_layers: u32, n_experts: u32, n_cache_slots: u32) !ExpertCache {
        const n_slots = @min(n_cache_slots, @as(u32, @intCast(max_cache_slots)));
        if (n_slots == 0) return error.ZeroCacheSlots;
        const slots = try allocator.alloc(CacheSlot, n_slots);
        @memset(slots, CacheSlot{});

        const lookup_size = @as(usize, n_layers) * @as(usize, n_experts);
        const lookup = try allocator.alloc(?u32, lookup_size);
        @memset(lookup, null);

        const pinned = try allocator.alloc(PinnedRange, max_pin_ranges);

        return ExpertCache{
            .slots = slots,
            .n_slots = n_slots,
            .lookup = lookup,
            .n_layers = n_layers,
            .n_experts = n_experts,
            .pinned_ranges = pinned,
        };
    }

    /// Unpins all mlocked ranges, then frees the cache slots and lookup table.
    pub fn deinit(self: *ExpertCache, allocator: Allocator) void {
        self.unpinAll();
        allocator.free(self.pinned_ranges);
        allocator.free(self.slots);
        allocator.free(self.lookup);
        self.pinned_ranges = &.{};
        self.slots = &.{};
        self.lookup = &.{};
    }

    /// Check if an expert is in the cache. If yes, update its access time.
    pub inline fn touch(self: *ExpertCache, layer: u32, expert_id: u32) bool {
        const key = @as(usize, layer) * self.n_experts + expert_id;
        if (key >= self.lookup.len) return false;
        if (self.lookup[key]) |slot_idx| {
            self.access_counter += 1;
            self.slots[slot_idx].last_access = self.access_counter;
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        return false;
    }

    /// Admit an expert to the cache, evicting LRU if full.
    /// Returns the slot index where the expert was placed.
    /// Out-of-range layer/expert IDs do not occupy a slot (would be unfindable
    /// via `touch`); returns `std.math.maxInt(u32)` as a sentinel.
    pub fn admit(self: *ExpertCache, layer: u32, expert_id: u32) u32 {
        // Frozen during verification: don't evict cached entries.
        if (self.frozen) return self.n_slots;
        // Check if already present
        const key = @as(usize, layer) * self.n_experts + expert_id;
        if (key >= self.lookup.len) return std.math.maxInt(u32);
        if (self.lookup[key]) |slot_idx| {
            self.access_counter += 1;
            self.slots[slot_idx].last_access = self.access_counter;
            return slot_idx;
        }

        // Find a free slot or evict LRU
        var target_slot: u32 = 0;
        var found_free = false;
        var min_access: u64 = std.math.maxInt(u64);

        for (self.slots, 0..) |slot, i| {
            if (!slot.occupied) {
                target_slot = @intCast(i);
                found_free = true;
                break;
            }
            if (slot.last_access < min_access) {
                min_access = slot.last_access;
                target_slot = @intCast(i);
            }
        }

        // Evict if the slot was occupied
        if (!found_free and self.slots[target_slot].occupied) {
            const old = self.slots[target_slot];
            const old_key = @as(usize, old.layer) * self.n_experts + old.expert_id;
            if (old_key < self.lookup.len) {
                self.lookup[old_key] = null;
            }
        }

        // Install the new expert
        self.access_counter += 1;
        self.slots[target_slot] = CacheSlot{
            .layer = layer,
            .expert_id = expert_id,
            .last_access = self.access_counter,
            .occupied = true,
        };
        self.lookup[key] = target_slot;

        return target_slot;
    }

    /// Prefetch an expert's weights from the mmap'd model file.
    /// Calls madvise(WILLNEED) on the expert's byte range.
    pub fn prefetch(self: *ExpertCache, layer: u32, expert_id: u32, base_ptr: [*]const u8, expert_bytes: usize) void {
        _ = self.admit(layer, expert_id);

        // madvise(WILLNEED) to trigger background page-in
        if (comptime builtin.os.tag == .linux or builtin.os.tag == .macos) {
            const page_size: usize = std.heap.page_size_min;
            const addr = @intFromPtr(base_ptr);
            const aligned_addr = addr & ~(page_size - 1);
            const offset = addr - aligned_addr;
            const total_len = offset + expert_bytes;
            _ = posix.madvise(@ptrFromInt(aligned_addr), total_len, posix.system.MADV.WILLNEED);
        }
    }

    /// Pre-pin top-K experts for a layer (startup hotlist, no madvise).
    /// Returns the number actually admitted (≤ k, bounded by cache capacity).
    pub fn admit_prepin(self: *ExpertCache, layer: u32, ids: []const u32, k: u32) u32 {
        var admitted: u32 = 0;
        for (0..@min(@as(usize, k), ids.len)) |i| {
            if (!self.touch(layer, ids[i])) {
                _ = self.admit(layer, ids[i]);
                admitted += 1;
            }
        }
        return admitted;
    }

    /// Pin an expert's weight bytes in physical RAM via mlock.
    /// The raw pointer and length are page-aligned internally.
    /// Returns true if successfully pinned, false if the wire limit would
    /// be exceeded, the pin table is full, or the mlock syscall fails.
    /// Startup-only — never call on the hot path.
    pub fn pinExpert(self: *ExpertCache, ptr: [*]const u8, len: usize) bool {
        if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) return false;
        if (self.n_pinned >= self.pinned_ranges.len) return false;
        if (self.total_pinned_bytes + len > max_pinned_bytes) return false;
        if (len == 0) return false;

        const page_size: usize = std.heap.page_size_min;
        const addr = @intFromPtr(ptr);
        const aligned = addr & ~(page_size - 1);
        const total = len + (addr - aligned);
        const aligned_ptr: [*]align(std.heap.page_size_min) const u8 = @ptrFromInt(aligned);
        const memory = aligned_ptr[0..total];

        std.process.lockMemory(memory, .{}) catch return false;

        self.pinned_ranges[self.n_pinned] = .{ .ptr = aligned_ptr, .len = total };
        self.n_pinned += 1;
        self.total_pinned_bytes += total;
        return true;
    }

    /// Unpin all previously mlocked expert ranges via munlock.
    /// Called from `deinit` before freeing allocations.
    pub fn unpinAll(self: *ExpertCache) void {
        if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) return;
        for (self.pinned_ranges[0..self.n_pinned]) |range| {
            std.process.unlockMemory(range.ptr[0..range.len]) catch {};
        }
        self.n_pinned = 0;
        self.total_pinned_bytes = 0;
    }

    /// Report cache statistics.
    pub fn reportStats(self: *const ExpertCache) void {
        const total = self.hits + self.misses;
        const hit_rate: f32 = if (total > 0) @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total)) else 0;
        std.log.info("expert cache: {d} slots, {d} hits, {d} misses, hit_rate={d:.1}%", .{
            self.n_slots, self.hits, self.misses, hit_rate * 100,
        });
    }

    /// Prefetch the K most-recently-used experts for a given layer.
    /// Called speculatively after the current layer completes to overlap SSD reads
    /// with CPU weighted accumulation. Some prefetched experts may not be selected
    /// by the next layer's router, but the cost of a wasted madvise is negligible
    /// compared to a cache miss on a needed expert.
    pub fn prefetchTopResidents(self: *ExpertCache, layer: u32, base_ptr: [*]const u8, expert_bytes: usize, k: u32) void {
        const actual_k = @min(k, max_prefetch_k);
        var best_slots: [max_prefetch_k]u32 = .{0} ** max_prefetch_k;
        var best_access: [max_prefetch_k]u64 = .{0} ** max_prefetch_k;
        var found: u32 = 0;

        for (self.slots[0..self.n_slots], 0..) |slot, idx| {
            if (!slot.occupied or slot.layer != layer) continue;
            // Insert into sorted top-k (descending by access time)
            var insert_pos: u32 = found;
            for (0..@min(found, actual_k)) |bi| {
                if (slot.last_access > best_access[bi]) {
                    insert_pos = @intCast(bi);
                    break;
                }
            }
            if (insert_pos < actual_k) {
                // Shift down to make room
                var j: u32 = @min(found, actual_k - 1);
                while (j > insert_pos) : (j -= 1) {
                    best_slots[j] = best_slots[j - 1];
                    best_access[j] = best_access[j - 1];
                }
                best_slots[insert_pos] = @intCast(idx);
                best_access[insert_pos] = slot.last_access;
                if (found < actual_k) found += 1;
            }
        }

        // Issue madvise for the top-k most recently used experts
        for (0..found) |i| {
            const eid = self.slots[best_slots[i]].expert_id;
            const offset = @as(usize, eid) * expert_bytes;
            prefetchRegion(base_ptr + offset, expert_bytes);
        }
    }

    const max_prefetch_k = 8;

    fn prefetchRegion(ptr: [*]const u8, len: usize) void {
        if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) return;
        const page_size: usize = std.heap.page_size_min;
        const addr = @intFromPtr(ptr);
        const aligned = addr & ~(page_size - 1);
        const total = len + (addr - aligned);
        posix.madvise(@ptrFromInt(aligned), total, posix.system.MADV.WILLNEED) catch {};
    }

    /// Freeze the cache: admit() becomes no-op (no evictions).
    /// Used during speculative verification to prevent cache thrashing.
    pub fn freeze(self: *ExpertCache) void {
        self.frozen = true;
    }

    /// Thaw the cache: resume normal admit/evict behavior.
    pub fn thaw(self: *ExpertCache) void {
        self.frozen = false;
    }

    /// Return the top-k most recently used expert IDs for a given layer.
    /// Does NOT issue any madvise/prefetch — just returns the IDs.
    pub fn getTopResidents(self: *ExpertCache, layer: u32, out_ids: []u32) u32 {
        const k = @min(@as(u32, @intCast(out_ids.len)), max_prefetch_k);
        var best_slots: [max_prefetch_k]u32 = .{0} ** max_prefetch_k;
        var best_access: [max_prefetch_k]u64 = .{0} ** max_prefetch_k;
        var found: u32 = 0;

        for (self.slots[0..self.n_slots], 0..) |slot, idx| {
            if (!slot.occupied or slot.layer != layer) continue;
            var insert_pos: u32 = found;
            for (0..@min(found, k)) |bi| {
                if (slot.last_access > best_access[bi]) {
                    insert_pos = @intCast(bi);
                    break;
                }
            }
            if (insert_pos < k) {
                var j: u32 = @min(found, k - 1);
                while (j > insert_pos) : (j -= 1) {
                    best_slots[j] = best_slots[j - 1];
                    best_access[j] = best_access[j - 1];
                }
                best_slots[insert_pos] = @intCast(idx);
                best_access[insert_pos] = slot.last_access;
                if (found < k) found += 1;
            }
        }

        for (0..found) |i| {
            out_ids[i] = self.slots[best_slots[i]].expert_id;
        }
        return found;
    }

};

// ── Tests ────────────────────────────────────────────────────────

test "ExpertCache init and touch" {
    const allocator = std.testing.allocator;
    var cache = try ExpertCache.init(allocator, 4, 8, 16);
    defer cache.deinit(allocator);

    // Not in cache yet
    try std.testing.expect(!cache.touch(0, 3));
    try std.testing.expectEqual(@as(u64, 1), cache.misses);

    // Admit and then touch
    const slot = cache.admit(0, 3);
    try std.testing.expectEqual(@as(u32, 0), slot);
    try std.testing.expect(cache.touch(0, 3));
    try std.testing.expectEqual(@as(u64, 1), cache.hits);

    // Re-admit is idempotent: same slot, still touchable
    try std.testing.expectEqual(slot, cache.admit(0, 3));
    try std.testing.expect(cache.touch(0, 3));
    try std.testing.expectEqual(@as(u64, 2), cache.hits);
}

test "ExpertCache LRU eviction" {
    const allocator = std.testing.allocator;
    var cache = try ExpertCache.init(allocator, 2, 4, 2); // only 2 slots
    defer cache.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), cache.admit(0, 0)); // slot 0, access=1
    try std.testing.expectEqual(@as(u32, 1), cache.admit(0, 1)); // slot 1, access=2 — cache full

    // Touch expert 1 to make expert 0 the LRU
    try std.testing.expect(cache.touch(0, 1)); // access=3
    // Now: slot 0 (expert 0) access=1, slot 1 (expert 1) access=3
    // Admitting expert 2 should evict expert 0 (lowest access time)
    const evicted_into = cache.admit(0, 2);
    try std.testing.expectEqual(@as(u32, 0), evicted_into);
    try std.testing.expect(!cache.touch(0, 0)); // evicted
    try std.testing.expect(cache.touch(0, 1)); // still present
    try std.testing.expect(cache.touch(0, 2)); // newly admitted
}

test "ExpertCache out-of-range touch is miss without hit" {
    const allocator = std.testing.allocator;
    var cache = try ExpertCache.init(allocator, 2, 4, 2);
    defer cache.deinit(allocator);

    try std.testing.expect(!cache.touch(99, 0));
    try std.testing.expectEqual(@as(u64, 0), cache.hits);
    // OOR keys short-circuit before miss accounting
    try std.testing.expectEqual(@as(u64, 0), cache.misses);
    try std.testing.expect(!cache.touch(0, 99));
    try std.testing.expectEqual(@as(u64, 0), cache.hits);
    try std.testing.expectEqual(@as(u64, 0), cache.misses);
}

test "ExpertCache zero slots rejected; OOR admit is sentinel" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.ZeroCacheSlots, ExpertCache.init(allocator, 2, 4, 0));

    var cache = try ExpertCache.init(allocator, 2, 4, 2);
    defer cache.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 0), cache.admit(0, 0));
    // OOR must not evict the resident expert
    try std.testing.expectEqual(std.math.maxInt(u32), cache.admit(99, 0));
    try std.testing.expect(cache.touch(0, 0));
}

test "fuzz: admit + touch out-of-range ids" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            const n_layers: u32 = @as(u32, smith.valueWithHash(u2, 0)) + 1; // 1..4
            const n_experts: u32 = @as(u32, smith.valueWithHash(u3, 1)) + 1; // 1..8
            const n_slots: u32 = @as(u32, smith.valueWithHash(u3, 2)) + 1; // 1..8
            var cache = try ExpertCache.init(allocator, n_layers, n_experts, n_slots);
            defer cache.deinit(allocator);

            const rounds = smith.valueWithHash(u4, 3) + 1;
            for (0..rounds) |i| {
                const layer = smith.valueWithHash(u8, @truncate(10 + i));
                const expert = smith.valueWithHash(u8, @truncate(20 + i));
                const in_range = layer < n_layers and expert < n_experts;
                _ = cache.admit(layer, expert);
                const hit = cache.touch(layer, expert);
                // In-range admit must be findable; OOR keys never count as hits.
                if (in_range) {
                    try std.testing.expect(hit);
                } else {
                    try std.testing.expect(!hit);
                }
            }
        }
    }.f, .{});
}
