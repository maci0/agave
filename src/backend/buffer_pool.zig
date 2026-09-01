//! Free list of device buffers, reused across weight-cache evictions.
//!
//! Under `--vram-budget` every eviction frees a device allocation and the admit
//! that follows makes a new one. Measured on a 7900 XTX at a 0.25 GiB budget
//! (373 MB working set): 51 ms per token of extra time against 7 ms of actual
//! PCIe transfer, so roughly seven eighths of the cost was the driver's
//! allocate/free round trips rather than moving the bytes.
//!
//! Reuse is by EXACT size, which is the right match for a transformer: layer
//! N's q_proj is byte-identical in size to layer N+1's, so a freed buffer almost
//! always fits the next weight of the same role exactly. Rounding to size
//! classes would waste up to half the budget to buy hits this does not need.
//!
//! Generic over the handle type, so the same pool serves a CUdeviceptr, a HIP
//! device pointer, or Vulkan's (buffer, memory) pair without casting either into
//! an integer. The pool never frees anything: a handle it cannot keep is handed
//! back for the caller to release with whatever call owns it.

const std = @import("std");

/// Buffers held at once. Weight sizes repeat per layer, so the distinct-size
/// count is small (tens) and the depth per size is what matters; 256 covers a
/// deep model's worth of one-of-each without pinning meaningful memory.
pub const default_capacity: usize = 256;

/// `Handle` is whatever the backend's allocation is identified by.
pub fn BufferPool(comptime Handle: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { size: usize, handle: Handle };

        entries: []Entry,
        len: usize,
        /// Bytes held by pooled buffers. The caller charges these against its own
        /// budget: a pooled buffer is still device memory, just not in use.
        pooled_bytes: usize,
        hits: u64,
        misses: u64,

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            std.debug.assert(capacity > 0);
            return .{
                .entries = try allocator.alloc(Entry, capacity),
                .len = 0,
                .pooled_bytes = 0,
                .hits = 0,
                .misses = 0,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.entries);
            self.* = undefined;
        }

        /// Take a pooled buffer of exactly `size`, or null if none is held.
        ///
        /// Most-recently-released first: that buffer is the most likely to still be
        /// mapped and warm in the driver's own bookkeeping.
        pub fn acquire(self: *Self, size: usize) ?Handle {
            var i = self.len;
            while (i > 0) {
                i -= 1;
                if (self.entries[i].size != size) continue;
                const handle = self.entries[i].handle;
                self.entries[i] = self.entries[self.len - 1];
                self.len -= 1;
                self.pooled_bytes -= size;
                self.hits += 1;
                return handle;
            }
            self.misses += 1;
            return null;
        }

        /// Offer a buffer to the pool. Returns null when the pool took it, or the
        /// handle back when it is full and the caller must free it for real.
        pub fn release(self: *Self, handle: Handle, size: usize) ?Handle {
            if (self.len == self.entries.len) return handle;
            self.entries[self.len] = .{ .size = size, .handle = handle };
            self.len += 1;
            self.pooled_bytes += size;
            return null;
        }

        /// Hand every pooled buffer back for release, emptying the pool. Used when
        /// the budget shrinks, where holding unused device memory is the exact thing
        /// the caller is trying to stop doing.
        pub fn drain(self: *Self, out: []Handle) []const Handle {
            const n = @min(self.len, out.len);
            for (0..n) |i| {
                // Take from the end so a partial drain leaves a consistent pool.
                self.len -= 1;
                out[i] = self.entries[self.len].handle;
                self.pooled_bytes -= self.entries[self.len].size;
            }
            return out[0..n];
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────

const testing = std.testing;

test "BufferPool, exact-size round trip" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);

    try testing.expectEqual(@as(?u64, null), pool.acquire(100));
    try testing.expectEqual(@as(?u64, null), pool.release(0xAA, 100));
    try testing.expectEqual(@as(usize, 100), pool.pooled_bytes);
    try testing.expectEqual(@as(?u64, 0xAA), pool.acquire(100));
    try testing.expectEqual(@as(usize, 0), pool.pooled_bytes);
}

test "BufferPool, a different size is not reused" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);
    _ = pool.release(0xAA, 100);
    // Never hand back a buffer that is merely large enough: the caller sized its
    // allocation for the old weight and would silently over- or under-run.
    try testing.expectEqual(@as(?u64, null), pool.acquire(64));
    try testing.expectEqual(@as(?u64, null), pool.acquire(128));
    try testing.expectEqual(@as(?u64, 0xAA), pool.acquire(100));
}

test "BufferPool, most recently released is returned first" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);
    _ = pool.release(0x1, 50);
    _ = pool.release(0x2, 50);
    _ = pool.release(0x3, 50);
    try testing.expectEqual(@as(?u64, 0x3), pool.acquire(50));
    try testing.expectEqual(@as(?u64, 0x2), pool.acquire(50));
    try testing.expectEqual(@as(?u64, 0x1), pool.acquire(50));
}

test "BufferPool, a full pool hands the buffer back" {
    var pool = try BufferPool(u64).init(testing.allocator, 2);
    defer pool.deinit(testing.allocator);
    try testing.expectEqual(@as(?u64, null), pool.release(0x1, 10));
    try testing.expectEqual(@as(?u64, null), pool.release(0x2, 10));
    // Full: the caller keeps ownership and must free it.
    try testing.expectEqual(@as(?u64, 0x3), pool.release(0x3, 10));
    try testing.expectEqual(@as(usize, 20), pool.pooled_bytes);
}

test "BufferPool, removal from the middle keeps the rest intact" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);
    _ = pool.release(0x1, 10);
    _ = pool.release(0x2, 20);
    _ = pool.release(0x3, 30);
    try testing.expectEqual(@as(?u64, 0x2), pool.acquire(20)); // middle entry
    try testing.expectEqual(@as(?u64, 0x1), pool.acquire(10));
    try testing.expectEqual(@as(?u64, 0x3), pool.acquire(30));
    try testing.expectEqual(@as(usize, 0), pool.pooled_bytes);
}

test "BufferPool, drain empties and reports every handle" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);
    for ([_]u64{ 1, 2, 3 }) |h| _ = pool.release(h, 10);
    var out: [8]u64 = undefined;
    const drained = pool.drain(&out);
    try testing.expectEqual(@as(usize, 3), drained.len);
    try testing.expectEqual(@as(usize, 0), pool.len);
    try testing.expectEqual(@as(usize, 0), pool.pooled_bytes);
    try testing.expectEqual(@as(?u64, null), pool.acquire(10));
}

test "BufferPool, a partial drain leaves the pool consistent" {
    var pool = try BufferPool(u64).init(testing.allocator, 8);
    defer pool.deinit(testing.allocator);
    for ([_]u64{ 1, 2, 3, 4 }) |h| _ = pool.release(h, 10);
    var out: [2]u64 = undefined;
    try testing.expectEqual(@as(usize, 2), pool.drain(&out).len);
    try testing.expectEqual(@as(usize, 2), pool.len);
    try testing.expectEqual(@as(usize, 20), pool.pooled_bytes);
    // What remains is still acquirable, and nothing was double-counted.
    try testing.expect(pool.acquire(10) != null);
    try testing.expect(pool.acquire(10) != null);
    try testing.expectEqual(@as(?u64, null), pool.acquire(10));
}

test "BufferPool, per-layer weight sizes hit almost always" {
    // The workload the pool is for: a handful of tensor roles, each the same
    // size in every layer, cycling as the budget evicts and re-admits.
    var pool = try BufferPool(u64).init(testing.allocator, default_capacity);
    defer pool.deinit(testing.allocator);

    const role_sizes = [_]usize{ 4 << 20, 4 << 20, 1 << 20, 11 << 20, 11 << 20 };
    var next_handle: u64 = 1;
    var live: [role_sizes.len]?u64 = @splat(null);

    var layer: usize = 0;
    while (layer < 200) : (layer += 1) {
        for (role_sizes, 0..) |size, role| {
            if (live[role]) |h| {
                _ = pool.release(h, size); // evicted by the budget
                live[role] = null;
            }
            live[role] = pool.acquire(size) orelse blk: {
                const h = next_handle;
                next_handle += 1;
                break :blk h;
            };
        }
    }
    // Only the first layer should have had to allocate; every later one reuses.
    try testing.expectEqual(@as(u64, role_sizes.len + 1), next_handle);
    try testing.expect(pool.hits > pool.misses * 100);
}
