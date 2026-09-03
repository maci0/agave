//! Leaf KV-block types shared by SDPA kernels, backends, and the cache manager.
//!
//! `CacheBlock` and `PagedKvView` are layout, not allocation: kernels walk a
//! block table, the manager owns the pool. They live here so
//! `backend/kernels/cpu/sdpa.zig` does not import `manager.zig` (RadixTree,
//! `PagedKvCache`, prefix sharing). Same split as `image_tokens.zig`.

const std = @import("std");

/// A single cache block holds `block_size` positions of KV data.
pub const CacheBlock = struct {
    /// Key data: [block_size * kv_dim] f32.
    keys: []f32,
    /// Value data: [block_size * kv_dim] f32.
    values: []f32,
    /// Number of positions currently filled in this block (0..block_size).
    used: u16 = 0,
    /// Reference count for prefix sharing.
    ref_count: u16 = 1,
};

/// View into paged KV cache for one layer, passed to SDPA kernels.
/// Enables block-table indirection: kernel walks block_table to find
/// physical blocks instead of assuming contiguous memory.
pub const PagedKvView = struct {
    block_table: []const u32,
    blocks: []const CacheBlock,
    block_size: u16,
    block_shift: std.math.Log2Int(u16),
    block_mask: u16,
    kv_dim: usize,
    seq_len: usize,

    /// Construct a view into the paged KV cache for one layer. Uses bit-shift
    /// addressing when `block_size` is a power of two, division otherwise.
    pub inline fn initView(block_table: []const u32, blocks: []const CacheBlock, block_size: u16, kv_dim: usize, seq_len: usize) PagedKvView {
        std.debug.assert(block_size > 0);
        return .{
            .block_table = block_table,
            .blocks = blocks,
            .block_size = block_size,
            .block_shift = if (std.math.isPowerOfTwo(block_size)) @intCast(@ctz(block_size)) else 0,
            .block_mask = if (std.math.isPowerOfTwo(block_size)) block_size - 1 else 0,
            .kv_dim = kv_dim,
            .seq_len = seq_len,
        };
    }

    inline fn blockIdx(self: PagedKvView, position: usize) usize {
        return if (self.block_mask != 0) position >> self.block_shift else position / self.block_size;
    }

    inline fn posInBlock(self: PagedKvView, position: usize) usize {
        return if (self.block_mask != 0) position & self.block_mask else position % self.block_size;
    }

    /// Checked: bounds-validate position and block translation.
    inline fn physIdFor(self: PagedKvView, position: usize) u32 {
        // `seq_len` is the length before this step's append. `sdpaPagedHeads`
        // writes the new K/V at index `seq_len` and then attends over
        // `seq_len + 1` positions, so index `seq_len` itself is in contract.
        // Physical bounds are enforced by the two asserts below.
        std.debug.assert(position <= self.seq_len);
        const li = self.blockIdx(position);
        std.debug.assert(li < self.block_table.len);
        const phys = self.block_table[li];
        std.debug.assert(phys < self.blocks.len);
        return phys;
    }

    inline fn physOffset(self: PagedKvView, position: usize) usize {
        const off = std.math.mul(usize, self.posInBlock(position), self.kv_dim) catch @panic("phys offset overflow");
        return off;
    }

    /// Get key pointer for a specific position within the paged cache.
    pub inline fn keyPtr(self: PagedKvView, position: usize) [*]const f32 {
        const phys_id = self.physIdFor(position);
        return self.blocks[phys_id].keys.ptr + self.physOffset(position);
    }

    /// Get value pointer for a specific position within the paged cache.
    pub inline fn valuePtr(self: PagedKvView, position: usize) [*]const f32 {
        const phys_id = self.physIdFor(position);
        return self.blocks[phys_id].values.ptr + self.physOffset(position);
    }

    /// Get mutable key pointer for writing (KV append).
    pub inline fn keyPtrMut(self: PagedKvView, position: usize) [*]f32 {
        const phys_id = self.physIdFor(position);
        return self.blocks[phys_id].keys.ptr + self.physOffset(position);
    }

    /// Get mutable value pointer for writing (KV append).
    pub inline fn valuePtrMut(self: PagedKvView, position: usize) [*]f32 {
        const phys_id = self.physIdFor(position);
        return self.blocks[phys_id].values.ptr + self.physOffset(position);
    }
};

test "PagedKvView power-of-two block addressing" {
    var k0 = [_]f32{0} ** 32;
    var v0 = [_]f32{0} ** 32;
    var k1 = [_]f32{0} ** 32;
    var v1 = [_]f32{0} ** 32;
    var blocks = [_]CacheBlock{
        .{ .keys = &k0, .values = &v0 },
        .{ .keys = &k1, .values = &v1 },
    };
    var block_table = [_]u32{ 0, 1 };
    const view = PagedKvView.initView(&block_table, &blocks, 8, 4, 12);

    try std.testing.expectEqual(@as(u16, 8), view.block_size);
    try std.testing.expectEqual(@as(u16, 7), view.block_mask);
    try std.testing.expectEqual(@as(usize, 4), view.kv_dim);
    try std.testing.expectEqual(@as(usize, 12), view.seq_len);

    const key_ptr = view.keyPtrMut(3);
    key_ptr[0] = 42.0;
    key_ptr[1] = 43.0;
    const read_ptr = view.keyPtr(3);
    try std.testing.expectEqual(@as(f32, 42.0), read_ptr[0]);
    try std.testing.expectEqual(@as(f32, 43.0), read_ptr[1]);

    const val_ptr = view.valuePtrMut(9);
    val_ptr[0] = 99.0;
    try std.testing.expectEqual(@as(f32, 99.0), view.valuePtr(9)[0]);
}

test "PagedKvView non-power-of-two block addressing" {
    var k0 = [_]f32{0} ** 48;
    var v0 = [_]f32{0} ** 48;
    var k1 = [_]f32{0} ** 48;
    var v1 = [_]f32{0} ** 48;
    var blocks = [_]CacheBlock{
        .{ .keys = &k0, .values = &v0 },
        .{ .keys = &k1, .values = &v1 },
    };
    var block_table = [_]u32{ 0, 1 };
    const view = PagedKvView.initView(&block_table, &blocks, 12, 4, 20);

    try std.testing.expectEqual(@as(u16, 0), view.block_mask);

    view.keyPtrMut(11)[0] = 11.0;
    try std.testing.expectEqual(@as(f32, 11.0), view.keyPtr(11)[0]);
    view.keyPtrMut(12)[0] = 12.0;
    try std.testing.expectEqual(@as(f32, 12.0), view.keyPtr(12)[0]);
}
