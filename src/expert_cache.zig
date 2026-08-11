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

/// Maximum number of cached expert slots.
const max_cache_slots: usize = 4096;

/// One cache slot tracking a resident expert.
const CacheSlot = struct {
    layer: u32 = 0,
    expert_id: u32 = 0,
    last_access: u64 = 0,
    occupied: bool = false,
};

/// LRU cache for MoE expert weights, tracking residency and eviction
/// across layers to minimize redundant GPU uploads.
pub const ExpertCache = struct {
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

    /// Allocates cache slots and the `(layer, expert) → slot` lookup table, clamped to `max_cache_slots`.
    pub fn init(allocator: Allocator, n_layers: u32, n_experts: u32, n_cache_slots: u32) !ExpertCache {
        const n_slots = @min(n_cache_slots, @as(u32, @intCast(max_cache_slots)));
        if (n_slots == 0) return error.ZeroCacheSlots;
        const slots = try allocator.alloc(CacheSlot, n_slots);
        @memset(slots, CacheSlot{});

        const lookup_size = @as(usize, n_layers) * @as(usize, n_experts);
        const lookup = try allocator.alloc(?u32, lookup_size);
        @memset(lookup, null);

        return ExpertCache{
            .slots = slots,
            .n_slots = n_slots,
            .lookup = lookup,
            .n_layers = n_layers,
            .n_experts = n_experts,
        };
    }

    /// Frees the cache slots and the lookup table.
    pub fn deinit(self: *ExpertCache, allocator: Allocator) void {
        allocator.free(self.slots);
        allocator.free(self.lookup);
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

    /// Report cache statistics.
    pub fn reportStats(self: *const ExpertCache) void {
        const total = self.hits + self.misses;
        const hit_rate: f32 = if (total > 0) @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total)) else 0;
        std.log.info("expert cache: {d} slots, {d} hits, {d} misses, hit_rate={d:.1}%", .{
            self.n_slots, self.hits, self.misses, hit_rate * 100,
        });
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
