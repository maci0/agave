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

/// Shared helper: allocate a new SeqBlockTable with empty block tables for all layers.
fn allocateSeqTableImpl(allocator: Allocator, n_layers: usize) !SeqBlockTable {
    const block_table = try allocator.alloc([]u32, n_layers);
    for (0..n_layers) |i| block_table[i] = &[_]u32{};
    return .{ .block_table = block_table, .seq_len = 0 };
}

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
        return allocateSeqTableImpl(self.allocator, n_layers);
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

        // Free block_table arrays (skip comptime empty slices from init)
        for (seq_table.block_table) |layer_table| {
            if (layer_table.len > 0) self.allocator.free(layer_table);
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
        return allocateSeqTableImpl(self.allocator, n_layers);
    }

    /// Append a new physical block to each layer of the sequence table.
    /// Each layer gets its own independent block.
    pub fn appendBlock(self: *TieredBlockAllocator, seq_table: *SeqBlockTable) !void {
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
                self.cache.freeBlock(block_id);
            }
        }

        // Free block_table arrays (skip comptime empty slices from init)
        for (seq_table.block_table) |layer_table| {
            if (layer_table.len > 0) self.allocator.free(layer_table);
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
    errdefer block_alloc.freeSeqTable(&seq_table);

    try block_alloc.appendBlock(&seq_table);
    try block_alloc.appendBlock(&seq_table);

    const free_before = paged.freeCount();
    block_alloc.freeSeqTable(&seq_table);
    const free_after = paged.freeCount();

    // Should have freed 2 blocks per layer = 4 total (2 layers × 2 blocks)
    try std.testing.expectEqual(free_before + 4, free_after);
}

test "appendBlock until exhaustion returns OutOfBlocks" {
    const allocator = std.testing.allocator;
    // 2 layers, 4 blocks total. Each appendBlock uses 2 blocks (1 per layer).
    // So we can append exactly 2 blocks before exhaustion.
    var paged = try PagedKvCache.init(allocator, 2, 32, 4, 8);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(2);
    defer block_alloc.freeSeqTable(&seq_table);

    // First two appends should succeed (4 blocks total: 2 layers x 2 appends)
    try block_alloc.appendBlock(&seq_table);
    try block_alloc.appendBlock(&seq_table);
    try std.testing.expectEqual(@as(usize, 0), paged.freeCount());

    // Third append should fail with OutOfBlocks
    const result = block_alloc.appendBlock(&seq_table);
    try std.testing.expectError(error.OutOfBlocks, result);

    // Verify seq_len was not incremented on failure (rollback worked)
    try std.testing.expectEqual(@as(usize, 16), seq_table.seq_len);
}

test "appendBlock errdefer rollback frees partial allocations" {
    const allocator = std.testing.allocator;
    // 3 layers, 5 blocks. First appendBlock needs 3 blocks (1 per layer) → succeeds.
    // Second appendBlock: allocates block for layer 0, layer 1, but layer 2 fails (only 2 free).
    // Should rollback the 2 blocks allocated for layers 0 and 1.
    var paged = try PagedKvCache.init(allocator, 3, 32, 5, 8);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    var seq_table = try block_alloc.allocateSeqTable(3);
    defer block_alloc.freeSeqTable(&seq_table);

    // First append: uses 3 blocks, 2 remain free
    try block_alloc.appendBlock(&seq_table);
    try std.testing.expectEqual(@as(usize, 2), paged.freeCount());
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[0].len);

    // Second append: needs 3 blocks but only 2 free → fails.
    // errdefer should roll back partial allocations.
    const result = block_alloc.appendBlock(&seq_table);
    try std.testing.expectError(error.OutOfBlocks, result);

    // After rollback, free count should be restored to 2
    try std.testing.expectEqual(@as(usize, 2), paged.freeCount());

    // Block tables should be unchanged (still 1 block per layer)
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[0].len);
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[1].len);
    try std.testing.expectEqual(@as(usize, 1), seq_table.block_table[2].len);
}

test "freeSeqTable on empty table is a no-op" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 2, 32, 4, 8);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);
    // Create an "empty" SeqBlockTable with zero layers
    var empty = SeqBlockTable{ .block_table = &.{}, .seq_len = 0 };
    // Should not crash
    block_alloc.freeSeqTable(&empty);
}

test "free and re-allocate produces valid blocks" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 1, 16, 8, 4);
    defer paged.deinit();

    var block_alloc = BlockAllocator.init(&paged, allocator);

    // Allocate sequence, append blocks, then free
    var seq1 = try block_alloc.allocateSeqTable(1);
    try block_alloc.appendBlock(&seq1);
    try block_alloc.appendBlock(&seq1);
    const b0 = BlockAllocator.getPhysicalBlock(&seq1, 0, 0);
    const b1 = BlockAllocator.getPhysicalBlock(&seq1, 0, 1);
    block_alloc.freeSeqTable(&seq1);

    // Re-allocate — should reuse freed blocks
    var seq2 = try block_alloc.allocateSeqTable(1);
    defer block_alloc.freeSeqTable(&seq2);
    try block_alloc.appendBlock(&seq2);
    try block_alloc.appendBlock(&seq2);
    const r0 = BlockAllocator.getPhysicalBlock(&seq2, 0, 0);
    const r1 = BlockAllocator.getPhysicalBlock(&seq2, 0, 1);

    // The freed blocks should be the ones we get back (LIFO from free list)
    try std.testing.expect(r0 == b0 or r0 == b1);
    try std.testing.expect(r1 == b0 or r1 == b1);
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
