//! Block table allocator for PagedAttention.
//! Manages per-request SeqBlockTable allocation and block append operations.
//! Provides both `BlockAllocator` (backed by `PagedKvCache`) and
//! `TieredBlockAllocator` (backed by `TieredKvCache`) for tiered storage.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kvcache = @import("manager.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;
const TieredKvCache = @import("tiered.zig").TieredKvCache;

/// BlockAllocator manages SeqBlockTable creation and physical block allocation.
pub const BlockAllocator = struct {
    cache: *PagedKvCache,
    allocator: Allocator,

    /// Initialize a block allocator with a reference to the paged cache.
    pub fn init(cache: *PagedKvCache, allocator: Allocator) BlockAllocator {
        return .{ .cache = cache, .allocator = allocator };
    }

    /// Update the cache pointer after the parent struct has been moved.
    /// Must be called when the struct containing PagedKvCache is moved
    /// (e.g., after returning from init by value).
    pub fn setCachePtr(self: *BlockAllocator, cache: *PagedKvCache) void {
        self.cache = cache;
    }

    /// Allocate a new SeqBlockTable with empty block tables for all layers.
    /// Caller must call freeSeqTable when done.
    pub fn allocateSeqTable(self: *BlockAllocator, n_layers: usize) !SeqBlockTable {
        const block_table = try self.allocator.alloc([]u32, n_layers);
        errdefer self.allocator.free(block_table);

        var init_count: usize = 0;
        errdefer for (0..init_count) |i| self.allocator.free(block_table[i]);

        for (0..n_layers) |i| {
            block_table[i] = try self.allocator.alloc(u32, 0); // Empty initially
            init_count = i + 1;
        }

        return .{ .block_table = block_table, .seq_len = 0 };
    }

    /// Append a new physical block to each layer of the sequence table.
    /// Each layer gets its own independent block so layers don't overwrite
    /// each other's KV cache data.
    pub fn appendBlock(self: *BlockAllocator, seq_table: *SeqBlockTable) !void {
        var allocated: usize = 0;
        errdefer {
            // Roll back: free blocks already appended to earlier layers
            for (seq_table.block_table[0..allocated]) |*layer_table| {
                const bid = layer_table.*[layer_table.len - 1];
                self.cache.freeBlock(bid);
                if (layer_table.len > 1) {
                    layer_table.* = self.allocator.realloc(layer_table.*, layer_table.len - 1) catch layer_table.*;
                }
            }
        }

        for (seq_table.block_table) |*layer_table| {
            const block_id = self.cache.allocBlock() orelse return error.OutOfBlocks;
            const new_table = try self.allocator.realloc(layer_table.*, layer_table.len + 1);
            new_table[new_table.len - 1] = block_id;
            layer_table.* = new_table;
            allocated += 1;
        }

        seq_table.seq_len += self.cache.block_size;
    }

    /// Free all blocks and memory associated with a SeqBlockTable.
    pub fn freeSeqTable(self: *BlockAllocator, seq_table: *SeqBlockTable) void {
        if (seq_table.block_table.len == 0) return;

        // Free physical blocks from ALL layers (each layer has its own blocks)
        for (seq_table.block_table) |layer_table| {
            for (layer_table) |block_id| {
                self.cache.freeBlock(block_id);
            }
        }

        // Free block_table arrays
        for (seq_table.block_table) |layer_table| {
            self.allocator.free(layer_table);
        }
        self.allocator.free(seq_table.block_table);
    }

    /// Map (layer, logical_block_idx) to physical block ID via indirection.
    pub fn getPhysicalBlock(seq_table: *const SeqBlockTable, layer: usize, logical_idx: usize) u32 {
        std.debug.assert(logical_idx < seq_table.block_table[layer].len);
        return seq_table.block_table[layer][logical_idx];
    }
};

/// TieredBlockAllocator manages SeqBlockTable creation and block allocation
/// backed by a `TieredKvCache` instead of `PagedKvCache`. Provides the same
/// interface as `BlockAllocator` so models can use either via an optional field.
pub const TieredBlockAllocator = struct {
    cache: *TieredKvCache,
    allocator: Allocator,

    /// Initialize a tiered block allocator with a reference to the tiered cache.
    pub fn init(cache: *TieredKvCache, allocator: Allocator) TieredBlockAllocator {
        return .{ .cache = cache, .allocator = allocator };
    }

    /// Allocate a new SeqBlockTable with empty block tables for all layers.
    /// Caller must call freeSeqTable when done.
    pub fn allocateSeqTable(self: *TieredBlockAllocator, n_layers: usize) !SeqBlockTable {
        const block_table = try self.allocator.alloc([]u32, n_layers);
        errdefer self.allocator.free(block_table);

        var init_count: usize = 0;
        errdefer for (0..init_count) |i| self.allocator.free(block_table[i]);

        for (0..n_layers) |i| {
            block_table[i] = try self.allocator.alloc(u32, 0); // Empty initially
            init_count = i + 1;
        }

        return .{ .block_table = block_table, .seq_len = 0 };
    }

    /// Append a new physical block to each layer of the sequence table.
    /// Each layer gets its own independent block.
    pub fn appendBlock(self: *TieredBlockAllocator, seq_table: *SeqBlockTable) !void {
        var allocated: usize = 0;
        errdefer {
            // Roll back: free blocks already appended to earlier layers
            for (seq_table.block_table[0..allocated]) |*layer_table| {
                const bid = layer_table.*[layer_table.len - 1];
                self.cache.freeBlock(bid) catch {};
                if (layer_table.len > 1) {
                    layer_table.* = self.allocator.realloc(layer_table.*, layer_table.len - 1) catch layer_table.*;
                }
            }
        }

        for (seq_table.block_table) |*layer_table| {
            const block_id = self.cache.allocBlock() catch return error.OutOfBlocks;
            const new_table = try self.allocator.realloc(layer_table.*, layer_table.len + 1);
            new_table[new_table.len - 1] = block_id;
            layer_table.* = new_table;
            allocated += 1;
        }

        seq_table.seq_len += self.cache.block_size;
    }

    /// Free all blocks and memory associated with a SeqBlockTable.
    pub fn freeSeqTable(self: *TieredBlockAllocator, seq_table: *SeqBlockTable) void {
        if (seq_table.block_table.len == 0) return;

        // Free physical blocks from ALL layers (each layer has its own blocks)
        for (seq_table.block_table) |layer_table| {
            for (layer_table) |block_id| {
                self.cache.freeBlock(block_id) catch {};
            }
        }

        // Free block_table arrays
        for (seq_table.block_table) |layer_table| {
            self.allocator.free(layer_table);
        }
        self.allocator.free(seq_table.block_table);
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "allocateSeqTable creates empty block tables" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 3, 64, 16, 16);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(3);
    defer block_alloc.freeSeqTable(&seq_table);

    // All layers should have empty block tables
    try std.testing.expectEqual(@as(usize, 3), seq_table.block_table.len);
    try std.testing.expectEqual(@as(usize, 0), seq_table.block_table[0].len);
    try std.testing.expectEqual(@as(usize, 0), seq_table.block_table[1].len);
    try std.testing.expectEqual(@as(usize, 0), seq_table.block_table[2].len);
    try std.testing.expectEqual(@as(usize, 0), seq_table.seq_len);
}

test "appendBlock allocates and appends to all layers" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 2, 64, 16, 16);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(2);
    defer block_alloc.freeSeqTable(&seq_table);

    const initial_free = paged.freeCount();

    try block_alloc.appendBlock(&seq_table);

    // Should have allocated one block PER LAYER from cache
    try std.testing.expectEqual(initial_free - 2, paged.freeCount());

    // Both layers should have one block appended (different block IDs)
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[0].len);
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[1].len);
    try std.testing.expect(seq_table.block_table[0][0] != seq_table.block_table[1][0]);

    // seq_len should be incremented by block_size
    try std.testing.expectEqual(@as(usize, 16), seq_table.seq_len);

    // Append another block
    try block_alloc.appendBlock(&seq_table);
    try std.testing.expectEqual(@as(usize, 2), seq_table.block_table[0].len);
    try std.testing.expectEqual(@as(usize, 32), seq_table.seq_len);
}

test "freeSeqTable returns blocks to free list" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 2, 64, 16, 16);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(2);

    try block_alloc.appendBlock(&seq_table);
    try block_alloc.appendBlock(&seq_table);

    const free_before = paged.freeCount();
    block_alloc.freeSeqTable(&seq_table);
    const free_after = paged.freeCount();

    // Should have freed 2 blocks per layer = 4 total (2 layers × 2 blocks)
    try std.testing.expectEqual(free_before + 4, free_after);
}

test "getPhysicalBlock returns correct block ID" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 2, 64, 16, 16);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(2);
    defer block_alloc.freeSeqTable(&seq_table);

    try block_alloc.appendBlock(&seq_table);
    try block_alloc.appendBlock(&seq_table);

    // Get block IDs via indirection
    const block0 = BlockAllocator.getPhysicalBlock(&seq_table, 0, 0);
    const block1 = BlockAllocator.getPhysicalBlock(&seq_table, 0, 1);

    // Each layer should have DIFFERENT block IDs
    try std.testing.expect(block0 != BlockAllocator.getPhysicalBlock(&seq_table, 1, 0));
    try std.testing.expect(block1 != BlockAllocator.getPhysicalBlock(&seq_table, 1, 1));
}
