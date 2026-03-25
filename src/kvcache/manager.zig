//! KV cache management: flat per-layer allocation (`allocKvCache`/`freeKvCache`),
//! block-based paged caching (`PagedKvCache`), and prefix-aware radix tree
//! sharing (`RadixTree`).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Default KV cache block size (tokens per block) used across all models.
/// Shared constant to avoid repeating the literal 16 in every model init.
pub const default_block_size: u16 = 16;

/// Result of allocating a KV cache.
/// Slices are byte arrays — the actual format (f32, f16, q8_0, etc.)
/// is tracked by the model's `kv_type` field.
pub const KvCache = struct {
    keys: [][]u8,
    values: [][]u8,
};

/// Allocate per-layer key/value cache slices with proper error cleanup.
/// Each layer gets `kv_bytes_per_layer` bytes for both keys and values.
///
/// Parameters:
///   - allocator: Memory allocator.
///   - n_layers: Number of layers to allocate caches for.
///   - kv_bytes_per_layer: Byte size per layer (use kv_quant.kvSliceBytes()).
///
/// Returns: KvCache with allocated key/value slices.
/// Caller must call freeKvCache when done.
pub fn allocKvCache(allocator: Allocator, n_layers: usize, kv_bytes_per_layer: usize) !KvCache {
    const keys = try allocator.alloc([]u8, n_layers);
    errdefer allocator.free(keys);
    const values = try allocator.alloc([]u8, n_layers);
    errdefer allocator.free(values);

    var init_count: usize = 0;
    errdefer {
        for (0..init_count) |i| {
            allocator.free(keys[i]);
            allocator.free(values[i]);
        }
    }

    for (0..n_layers) |i| {
        keys[i] = try allocator.alloc(u8, kv_bytes_per_layer);
        values[i] = try allocator.alloc(u8, kv_bytes_per_layer);
        init_count = i + 1;
    }

    return .{ .keys = keys, .values = values };
}

/// Free all per-layer KV cache slices and the outer arrays.
pub fn freeKvCache(allocator: Allocator, cache: KvCache) void {
    for (cache.keys, cache.values) |k, v| {
        allocator.free(k);
        allocator.free(v);
    }
    allocator.free(cache.keys);
    allocator.free(cache.values);
}

// ── PagedAttention ────────────────────────────────────────────────
//
// Block-based KV cache management following the vLLM PagedAttention paper.
// The cache is divided into fixed-size blocks (default 16 positions each).
// Sequences reference blocks through a block table (indirection), allowing
// fine-grained memory reclamation.

/// A single cache block holds `block_size` positions of KV data.
pub const CacheBlock = struct {
    /// Key data: [block_size * kv_dim] f32.
    keys: []f32,
    /// Value data: [block_size * kv_dim] f32.
    values: []f32,
    /// Number of positions currently filled in this block (0..block_size).
    used: u16 = 0,
    /// Reference count for copy-on-write sharing.
    ref_count: u16 = 1,
    /// Frequency tracking for eviction policy.
    access_count: u32 = 0,
    /// Last access timestamp for LRU within tier.
    last_access_ms: i64 = 0,
};

/// Per-sequence metadata: which blocks hold this sequence's KV data.
pub const SeqBlockTable = struct {
    /// Block indices per layer: block_table[layer][logical_block_idx] → physical block id.
    block_table: [][]u32,
    /// Current sequence length (total positions written).
    seq_len: usize = 0,
};

/// Block-based paged KV cache allocator.
pub const PagedKvCache = struct {
    blocks: []CacheBlock,
    free_list: std.ArrayList(u32),
    block_size: u16,
    kv_dim: usize,
    n_layers: usize,
    allocator: Allocator,

    /// Allocate a paged KV cache with `num_blocks` blocks of `block_size` positions each.
    pub fn init(allocator: Allocator, n_layers: usize, kv_dim: usize, num_blocks: usize, block_size: u16) !PagedKvCache {
        const blocks = try allocator.alloc(CacheBlock, num_blocks);
        errdefer allocator.free(blocks);

        var init_count: usize = 0;
        errdefer {
            for (0..init_count) |i| {
                allocator.free(blocks[i].keys);
                allocator.free(blocks[i].values);
            }
        }

        for (0..num_blocks) |i| {
            const slot_size = @as(usize, block_size) * kv_dim;
            blocks[i] = .{
                .keys = try allocator.alloc(f32, slot_size),
                .values = try allocator.alloc(f32, slot_size),
            };
            init_count = i + 1;
        }

        var free_list: std.ArrayList(u32) = .empty;
        try free_list.ensureTotalCapacity(allocator, num_blocks);
        // Push all blocks onto the free list (reverse order so pop gives 0,1,2,...)
        var i: u32 = @intCast(num_blocks);
        while (i > 0) {
            i -= 1;
            free_list.appendAssumeCapacity(i);
        }

        return .{
            .blocks = blocks,
            .free_list = free_list,
            .block_size = block_size,
            .kv_dim = kv_dim,
            .n_layers = n_layers,
            .allocator = allocator,
        };
    }

    /// Allocate a single physical block. Returns null if no blocks available.
    pub fn allocBlock(self: *PagedKvCache) ?u32 {
        if (self.free_list.items.len == 0) return null;
        return self.free_list.pop();
    }

    /// Release a physical block back to the free list.
    /// The free list was pre-allocated to hold all blocks, so append
    /// can only fail if the block was double-freed (exceeding capacity).
    pub fn freeBlock(self: *PagedKvCache, block_id: u32) void {
        self.blocks[block_id].used = 0;
        self.blocks[block_id].ref_count = 1;
        self.free_list.appendAssumeCapacity(block_id);
    }

    /// Number of free blocks available.
    pub fn freeCount(self: *const PagedKvCache) usize {
        return self.free_list.items.len;
    }

    /// Evict the coldest block based on frequency × cost metric.
    /// Shared prefixes (ref_count > 1) get 100× cost multiplier to preserve reuse.
    /// Returns the evicted block ID, or error if no evictable blocks found.
    pub fn evictColdestBlock(self: *PagedKvCache) !u32 {
        var min_score: f32 = std.math.floatMax(f32);
        var victim_id: u32 = 0;
        var found_victim = false;

        for (self.blocks, 0..) |*blk, id| {
            if (blk.ref_count == 0) continue; // Skip free blocks

            // Per D-05: shared prefixes (ref_count > 1) get 100× cost multiplier
            const cost: f32 = if (blk.ref_count > 1) 100.0 else 1.0;
            const score = @as(f32, @floatFromInt(blk.access_count)) * cost;

            if (score < min_score) {
                min_score = score;
                victim_id = @intCast(id);
                found_victim = true;
            }
        }

        if (!found_victim) {
            return error.NoEvictableBlocks;
        }

        // Free the victim block
        self.blocks[victim_id].ref_count = 0;
        self.blocks[victim_id].used = 0;
        self.blocks[victim_id].access_count = 0;
        try self.free_list.append(self.allocator, victim_id);

        return victim_id;
    }

    /// Free all cache blocks and the free list.
    pub fn deinit(self: *PagedKvCache) void {
        for (self.blocks) |blk| {
            self.allocator.free(blk.keys);
            self.allocator.free(blk.values);
        }
        self.allocator.free(self.blocks);
        self.free_list.deinit(self.allocator);
    }
};

// ── RadixAttention ────────────────────────────────────────────────
//
// Radix tree (prefix trie) for KV cache sharing across requests.
// Enables automatic longest-common-prefix detection, efficient prefix
// lookup, and LRU tracking (SGLang-style). Built on top of PagedKvCache blocks.
//
// Each node represents a prefix of token IDs. Matching prefixes share
// the same physical cache blocks, with copy-on-write semantics.

/// Maximum children per radix tree node. Token IDs are hashed into buckets.
const radix_fanout = 256;

/// A single node in the radix tree. Each node owns a span of token IDs
/// and references the physical blocks that hold those tokens' KV data.
const RadixNode = struct {
    /// Token IDs stored at this node (the "edge label").
    tokens: []u32,
    /// Physical block IDs holding KV data for these tokens.
    block_ids: []u32,
    /// Child nodes indexed by hash of next token ID.
    children: [radix_fanout]?*RadixNode,
    /// Reference count for sharing.
    ref_count: u16 = 1,
    /// Last access timestamp for LRU eviction.
    last_access: i64 = 0,

    fn init(allocator: Allocator, tokens: []const u32, block_ids: []const u32) !*RadixNode {
        const node = try allocator.create(RadixNode);
        errdefer allocator.destroy(node);
        const duped_tokens = try allocator.dupe(u32, tokens);
        errdefer allocator.free(duped_tokens);
        const duped_blocks = try allocator.dupe(u32, block_ids);
        node.* = .{
            .tokens = duped_tokens,
            .block_ids = duped_blocks,
            .children = .{null} ** radix_fanout,
        };
        return node;
    }

    fn deinit(self: *RadixNode, allocator: Allocator) void {
        for (&self.children) |*child| {
            if (child.*) |c| {
                c.deinit(allocator);
                child.* = null;
            }
        }
        allocator.free(self.tokens);
        allocator.free(self.block_ids);
        allocator.destroy(self);
    }
};

/// Result of a radix tree prefix match: how many tokens matched and their block IDs.
pub const PrefixMatch = struct { matched: usize, blocks: []const u32 };

/// Radix tree for prefix-aware KV cache management.
/// Supports insert (cache a token sequence) and match (find longest cached prefix).
/// LRU eviction is tracked via `last_access` timestamps but not yet implemented.
pub const RadixTree = struct {
    root: *RadixNode,
    allocator: Allocator,
    access_counter: i64 = 0,

    /// Create an empty radix tree with a root node.
    pub fn init(allocator: Allocator) !RadixTree {
        return .{
            .root = try RadixNode.init(allocator, &.{}, &.{}),
            .allocator = allocator,
        };
    }

    /// Mark a node as recently accessed and advance the global counter.
    fn touchNode(self: *RadixTree, node: *RadixNode) void {
        node.last_access = self.access_counter;
        self.access_counter +%= 1;
    }

    /// Find the longest prefix of `tokens` that exists in the tree.
    /// Returns the number of tokens matched and the block IDs for those tokens.
    pub fn matchPrefix(self: *RadixTree, tokens: []const u32) PrefixMatch {
        var node = self.root;
        var pos: usize = 0;

        while (pos < tokens.len) {
            const bucket = @as(u8, @truncate(tokens[pos] % radix_fanout));
            const child = node.children[bucket] orelse break;

            // Check if child's tokens match
            const remaining = tokens[pos..];
            if (remaining.len < child.tokens.len) break;
            if (!std.mem.eql(u32, remaining[0..child.tokens.len], child.tokens)) break;

            pos += child.tokens.len;
            self.touchNode(child);
            node = child;
        }

        return .{ .matched = pos, .blocks = node.block_ids };
    }

    /// Insert a token sequence with its associated block IDs into the tree.
    pub fn insert(self: *RadixTree, tokens: []const u32, block_ids: []const u32) !void {
        var node = self.root;
        var pos: usize = 0;

        while (pos < tokens.len) {
            const bucket = @as(u8, @truncate(tokens[pos] % radix_fanout));

            if (node.children[bucket]) |child| {
                // Check how many tokens match
                const remaining = tokens[pos..];
                const edge = child.tokens;
                var match_len: usize = 0;
                while (match_len < edge.len and match_len < remaining.len and
                    edge[match_len] == remaining[match_len]) : (match_len += 1)
                {}

                if (match_len == edge.len) {
                    // Full edge match — descend
                    pos += match_len;
                    node = child;
                } else {
                    // Partial match — split the edge.
                    // Pre-allocate all new memory before mutating the tree,
                    // so a failed allocation leaves the tree untouched.

                    const prefix_blocks = if (match_len <= child.block_ids.len)
                        child.block_ids[0..match_len]
                    else
                        child.block_ids;

                    // 1. Allocate intermediate node for the shared prefix
                    const mid = try RadixNode.init(self.allocator, edge[0..match_len], prefix_blocks);
                    errdefer mid.deinit(self.allocator);

                    // 2. Pre-allocate shortened suffix slices for existing child
                    const new_child_tokens = try self.allocator.dupe(u32, edge[match_len..]);
                    errdefer self.allocator.free(new_child_tokens);
                    const new_child_blocks = if (match_len < child.block_ids.len)
                        try self.allocator.dupe(u32, child.block_ids[match_len..])
                    else
                        try self.allocator.dupe(u32, &.{});
                    errdefer self.allocator.free(new_child_blocks);

                    // 3. Pre-allocate new leaf for remaining input tokens (if any)
                    pos += match_len;
                    const new_remaining = tokens[pos..];
                    var new_leaf: ?*RadixNode = null;
                    errdefer if (new_leaf) |nl| nl.deinit(self.allocator);
                    if (new_remaining.len > 0) {
                        const new_blocks = if (pos < block_ids.len) block_ids[pos..] else &[_]u32{};
                        const suffix_bucket = @as(u8, @truncate(new_child_tokens[0] % radix_fanout));
                        const leaf_bucket = @as(u8, @truncate(new_remaining[0] % radix_fanout));
                        if (leaf_bucket != suffix_bucket) {
                            new_leaf = try RadixNode.init(self.allocator, new_remaining, new_blocks);
                        }
                    }

                    // === All allocations succeeded — now mutate (cannot fail) ===

                    self.touchNode(mid);

                    // Shorten existing child: swap in pre-allocated slices, free old
                    const old_tokens = child.tokens;
                    const old_blocks = child.block_ids;
                    child.tokens = new_child_tokens;
                    child.block_ids = new_child_blocks;
                    self.allocator.free(old_tokens);
                    self.allocator.free(old_blocks);

                    // Re-attach shortened child under intermediate node
                    const child_bucket = @as(u8, @truncate(child.tokens[0] % radix_fanout));
                    mid.children[child_bucket] = child;

                    // Attach new leaf if created
                    if (new_leaf) |nl| {
                        self.touchNode(nl);
                        const leaf_bucket = @as(u8, @truncate(nl.tokens[0] % radix_fanout));
                        mid.children[leaf_bucket] = nl;
                    }

                    // Replace original child with intermediate node in parent
                    node.children[bucket] = mid;
                    break;
                }
            } else {
                // No child — create one with remaining tokens
                const remaining_tokens = tokens[pos..];
                const remaining_blocks = if (pos < block_ids.len) block_ids[pos..] else &[_]u32{};
                const child = try RadixNode.init(self.allocator, remaining_tokens, remaining_blocks);
                self.touchNode(child);
                node.children[bucket] = child;
                break;
            }
        }
    }

    /// Free the entire radix tree (all nodes and their token/block arrays).
    pub fn deinit(self: *RadixTree) void {
        self.root.deinit(self.allocator);
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "allocKvCache and freeKvCache" {
    const allocator = std.testing.allocator;
    const kv_bytes = 128 * 64 * 4; // 128 positions * 64 kv_dim * 4 bytes (f32)
    const cache = try allocKvCache(allocator, 4, kv_bytes);
    defer freeKvCache(allocator, cache);

    try std.testing.expectEqual(@as(usize, 4), cache.keys.len);
    try std.testing.expectEqual(@as(usize, 4), cache.values.len);
    try std.testing.expectEqual(@as(usize, kv_bytes), cache.keys[0].len);
}

test "PagedKvCache alloc and free blocks" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 2, 64, 8, 16);
    defer paged.deinit();

    try std.testing.expectEqual(@as(usize, 8), paged.freeCount());

    const b0 = paged.allocBlock().?;
    const b1 = paged.allocBlock().?;
    try std.testing.expectEqual(@as(usize, 6), paged.freeCount());
    try std.testing.expect(b0 != b1);

    // Each block has block_size * kv_dim elements
    try std.testing.expectEqual(@as(usize, 16 * 64), paged.blocks[b0].keys.len);

    paged.freeBlock(b0);
    try std.testing.expectEqual(@as(usize, 7), paged.freeCount());
}

test "PagedKvCache exhaustion" {
    const allocator = std.testing.allocator;
    var paged = try PagedKvCache.init(allocator, 1, 32, 4, 8);
    defer paged.deinit();

    // Allocate all 4 blocks
    for (0..4) |_| {
        try std.testing.expect(paged.allocBlock() != null);
    }
    // Should be exhausted
    try std.testing.expect(paged.allocBlock() == null);
    try std.testing.expectEqual(@as(usize, 0), paged.freeCount());
}

test "RadixTree insert and match" {
    const allocator = std.testing.allocator;
    var tree = try RadixTree.init(allocator);
    defer tree.deinit();

    // Insert a sequence [1, 2, 3] with block IDs [10, 11, 12]
    try tree.insert(&.{ 1, 2, 3 }, &.{ 10, 11, 12 });

    // Full prefix match
    const m1 = tree.matchPrefix(&.{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(@as(usize, 3), m1.matched);

    // Partial prefix match
    const m2 = tree.matchPrefix(&.{ 1, 2 });
    // Should match 0 since the edge is [1,2,3] and input is only [1,2]
    try std.testing.expectEqual(@as(usize, 0), m2.matched);

    // No match
    const m3 = tree.matchPrefix(&.{ 5, 6, 7 });
    try std.testing.expectEqual(@as(usize, 0), m3.matched);
}

test "RadixTree empty match" {
    const allocator = std.testing.allocator;
    var tree = try RadixTree.init(allocator);
    defer tree.deinit();

    const m = tree.matchPrefix(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 0), m.matched);
}

test "RadixTree edge splitting" {
    const allocator = std.testing.allocator;
    var tree = try RadixTree.init(allocator);
    defer tree.deinit();

    // Insert [1, 2, 3] then [1, 2, 4] — should split at position 2
    try tree.insert(&.{ 1, 2, 3 }, &.{ 10, 11, 12 });
    try tree.insert(&.{ 1, 2, 4 }, &.{ 10, 11, 14 });

    // Both should match their shared prefix [1, 2]
    const m1 = tree.matchPrefix(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 3), m1.matched);

    const m2 = tree.matchPrefix(&.{ 1, 2, 4 });
    try std.testing.expectEqual(@as(usize, 3), m2.matched);

    // Shared prefix [1, 2] should match for novel continuations
    const m3 = tree.matchPrefix(&.{ 1, 2, 5 });
    try std.testing.expectEqual(@as(usize, 2), m3.matched);

    // Unrelated sequence: no match
    const m4 = tree.matchPrefix(&.{ 9, 8, 7 });
    try std.testing.expectEqual(@as(usize, 0), m4.matched);
}
