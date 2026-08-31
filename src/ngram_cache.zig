//! Ngram cache for Qwen3.8-Flash-Next PLE (N-gram lookup embedding).
//! Mirrors src/expert_cache.zig but for the PLE ngram table.
//! 20M ngrams × 2560 hidden ≈ 51B, 128 shards (~400MB each). Demand-paged
//! via ExpertCache's SSD streaming: same --ssd-streaming flag, same
//! madvise(WILLNEED)/mlock pattern. Keep this module standalone so it
//! doesn't pollute ExpertCache's MoE semantics — the ngram working set is
//! tiny (a few shards per sequence) vs MoE's per-layer routing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

/// LRU cache for PLE ngram shards (mirrors ExpertCache but 1-D by shard).
pub const NgramCache = struct {
    pub const max_cache_shards: usize = 128;

    n_shards: u32,
    n_slots: u32,
    slots: []Slot,
    lookup: []?u32,
    access_counter: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    first_unoccupied: u32 = 0,

    const Slot = struct {
        shard_id: u32 = 0,
        last_access: u64 = 0,
        occupied: bool = false,
    };

    pub fn init(allocator: Allocator, n_shards: u32, n_cache_slots: u32) !NgramCache {
        const n_slots = @min(n_cache_slots, @as(u32, @intCast(max_cache_shards)));
        if (n_slots == 0) return error.ZeroCacheSlots;
        const slots = try allocator.alloc(Slot, n_slots);
        @memset(slots, Slot{});
        const lookup = try allocator.alloc(?u32, n_shards);
        @memset(lookup, null);
        return .{ .n_shards = n_shards, .n_slots = n_slots, .slots = slots, .lookup = lookup };
    }

    pub fn deinit(self: *NgramCache, allocator: Allocator) void {
        allocator.free(self.slots);
        allocator.free(self.lookup);
        self.slots = &.{};
        self.lookup = &.{};
    }

    pub inline fn touch(self: *NgramCache, shard_id: u32) bool {
        if (shard_id >= self.n_shards) return false;
        if (self.lookup[shard_id]) |slot_idx| {
            self.access_counter += 1;
            self.slots[slot_idx].last_access = self.access_counter;
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        return false;
    }

    pub fn admit(self: *NgramCache, shard_id: u32) u32 {
        if (shard_id >= self.n_shards) return std.math.maxInt(u32);
        if (self.lookup[shard_id]) |slot_idx| {
            self.access_counter += 1;
            self.slots[slot_idx].last_access = self.access_counter;
            return slot_idx;
        }
        var target: u32 = 0;
        var found_free = false;
        if (self.first_unoccupied < self.n_slots) {
            target = self.first_unoccupied;
            self.first_unoccupied += 1;
            found_free = true;
        } else {
            var min_access: u64 = std.math.maxInt(u64);
            for (self.slots, 0..) |slot, i| {
                if (!slot.occupied) {
                    target = @intCast(i);
                    found_free = true;
                    break;
                }
                if (slot.last_access < min_access) {
                    min_access = slot.last_access;
                    target = @intCast(i);
                }
            }
        }
        if (!found_free and self.slots[target].occupied) {
            const old = self.slots[target];
            self.lookup[old.shard_id] = null;
        }
        self.access_counter += 1;
        self.slots[target] = .{ .shard_id = shard_id, .last_access = self.access_counter, .occupied = true };
        self.lookup[shard_id] = target;
        return target;
    }

    /// Demand-page a shard via madvise(WILLNEED) on its mapped region.
    pub fn prefetch(self: *NgramCache, shard_id: u32, base_ptr: [*]const u8, shard_bytes: usize) void {
        _ = self.admit(shard_id);
        if (comptime builtin.os.tag == .linux or builtin.os.tag == .macos) {
            const page_size: usize = std.heap.page_size_min;
            const addr = @intFromPtr(base_ptr) + @as(usize, shard_id) * shard_bytes;
            const aligned = addr & ~(page_size - 1);
            const off = addr - aligned;
            _ = std.posix.madvise(@ptrFromInt(aligned), off + shard_bytes, std.posix.MADV.WILLNEED);
        }
    }

    pub fn reportStats(self: *const NgramCache) void {
        const total = self.hits + self.misses;
        const hit_rate: f32 = if (total > 0) @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total)) else 0;
        std.log.info("ngram cache: {d} slots, {d} hits, {d} misses, hit_rate={d:.1}%", .{ self.n_slots, self.hits, self.misses, hit_rate * 100 });
    }
};

test "NgramCache basic" {
    const a = std.testing.allocator;
    var c = try NgramCache.init(a, 128, 4);
    defer c.deinit(a);
    try std.testing.expect(!c.touch(5));
    _ = c.admit(5);
    try std.testing.expect(c.touch(5));
}

test "NgramCache LRU" {
    const a = std.testing.allocator;
    var c = try NgramCache.init(a, 8, 2);
    defer c.deinit(a);
    _ = c.admit(0);
    _ = c.admit(1);
    _ = c.touch(0);
    _ = c.admit(2); // evicts 1
    try std.testing.expect(c.touch(0));
    try std.testing.expect(!c.touch(1));
    try std.testing.expect(c.touch(2));
}
