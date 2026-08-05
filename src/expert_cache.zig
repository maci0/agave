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
//! Usage:
//!   var cache = try ExpertCache.init(allocator, n_layers, n_experts, cache_size);
//!   defer cache.deinit(allocator);
//!
//!   // Before MoE forward, ensure selected experts are resident:
//!   cache.touch(layer, expert_id);
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

    pub fn init(allocator: Allocator, n_layers: u32, n_experts: u32, n_cache_slots: u32) !ExpertCache {
        const n_slots = @min(n_cache_slots, @as(u32, @intCast(max_cache_slots)));
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
    pub fn admit(self: *ExpertCache, layer: u32, expert_id: u32) u32 {
        // Check if already present
        const key = @as(usize, layer) * self.n_experts + expert_id;
        if (key < self.lookup.len) {
            if (self.lookup[key]) |slot_idx| {
                self.access_counter += 1;
                self.slots[slot_idx].last_access = self.access_counter;
                return slot_idx;
            }
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
        if (key < self.lookup.len) {
            self.lookup[key] = target_slot;
        }

        return target_slot;
    }

    /// Prefetch an expert's weights from the mmap'd model file.
    /// Calls madvise(WILLNEED) on the expert's byte range.
    pub fn prefetch(self: *ExpertCache, layer: u32, expert_id: u32, base_ptr: [*]const u8, expert_bytes: usize) void {
        _ = self.admit(layer, expert_id);

        // madvise(WILLNEED) to trigger background page-in
        if (comptime builtin.os.tag == .linux or builtin.os.tag == .macos) {
            const page_size: usize = 4096;
            const addr = @intFromPtr(base_ptr);
            const aligned_addr = addr & ~(page_size - 1);
            const offset = addr - aligned_addr;
            const total_len = offset + expert_bytes;
            _ = posix.madvise(@ptrFromInt(aligned_addr), total_len, posix.system.MADV.WILLNEED);
        }
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
    _ = cache.admit(0, 3);
    try std.testing.expect(cache.touch(0, 3));
    try std.testing.expectEqual(@as(u64, 1), cache.hits);
}

test "ExpertCache LRU eviction" {
    const allocator = std.testing.allocator;
    var cache = try ExpertCache.init(allocator, 2, 4, 2); // only 2 slots
    defer cache.deinit(allocator);

    _ = cache.admit(0, 0); // slot 0, access=1
    _ = cache.admit(0, 1); // slot 1, access=2 — cache full

    // Touch expert 1 to make expert 0 the LRU
    try std.testing.expect(cache.touch(0, 1)); // access=3
    // Now: slot 0 (expert 0) access=1, slot 1 (expert 1) access=3
    // Admitting expert 2 should evict expert 0 (lowest access time)
    _ = cache.admit(0, 2);
    try std.testing.expect(!cache.touch(0, 0)); // evicted
    try std.testing.expect(cache.touch(0, 1)); // still present
    try std.testing.expect(cache.touch(0, 2)); // newly admitted
}
