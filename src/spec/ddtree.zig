//! DDTree: Diffusion Draft Tree construction.
//!
//! Builds an optimal draft tree from per-position logit distributions using
//! a best-first heap algorithm (Ringel & Romano, 2026). The tree maximizes
//! expected acceptance length under a fixed node budget.
//!
//! Algorithm:
//!   1. Pre-sort top-B tokens at each depth by log-probability
//!   2. Seed max-heap with (depth=0, rank=0)
//!   3. Pop highest cumulative-log-prob node, add to tree
//!   4. Push sibling (same depth, rank+1) and child (depth+1, rank=0)
//!   5. Repeat until budget exhausted
//!
//! Complexity: O(B log B) where B = node budget.

const std = @import("std");

/// Maximum tree node budget (compile-time cap for buffer sizing).
pub const max_budget: usize = 512;
/// Maximum draft depth (number of draft positions).
pub const max_depth: usize = 32;

/// A single node in the DDTree draft tree, representing one candidate token
/// at a given depth with its cumulative log-probability and parent link.
pub const TreeNode = struct {
    token_id: u32,
    /// Index of parent in nodes array. -1 for children of the root prompt token.
    parent: i32,
    depth: u16,
    rank: u16,
    cum_log_prob: f32,
};

/// Compiled tree ready for verification.
/// Maximum children per node (for child index).
const max_children_per_node: usize = 32;

/// Compiled draft tree flattened for batched verification against the target model.
pub const CompiledTree = struct {
    /// Token IDs in tree order.
    input_ids: [max_budget]u32 = undefined,
    /// Position IDs for RoPE (base_pos + depth).
    position_ids: [max_budget]u32 = undefined,
    /// Ancestor bitmask: bit j set if flat node j is an ancestor of node i.
    ancestor_masks: [max_budget][8]u64 = undefined,
    /// Parent index in flat array (-1 for root-level nodes).
    parent_in_flat: [max_budget]i32 = undefined,
    /// Per-node child token→index map for O(1) lookup.
    child_tokens: [max_budget + 1][max_children_per_node]u32 = undefined,
    child_indices: [max_budget + 1][max_children_per_node]u32 = undefined,
    child_counts: [max_budget + 1]u8 = .{0} ** (max_budget + 1),
    /// Number of nodes.
    n_nodes: u32 = 0,
    /// KV cache position where tree starts.
    base_pos: u32 = 0,

    /// O(k) child lookup where k = number of children (typically 1-3).
    pub fn findChild(self: *const CompiledTree, parent_idx: i32, token_id: u32) ?u32 {
        const idx: usize = if (parent_idx < 0) max_budget else @intCast(parent_idx);
        const count = self.child_counts[idx];
        for (0..count) |c| {
            if (self.child_tokens[idx][c] == token_id)
                return self.child_indices[idx][c];
        }
        return null;
    }

    /// Test whether bit j is set in an ancestor mask.
    pub inline fn isAncestor(mask: [8]u64, j: usize) bool {
        return mask[j / 64] & (@as(u64, 1) << @intCast(@as(u6, @truncate(j)))) != 0;
    }
};

const HeapEntry = struct {
    neg_cum_log_prob: f32,
    depth: u16,
    rank: u16,
    parent_idx: i32,
};

fn heapSiftUp(buf: []HeapEntry, start: usize) void {
    var i = start;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (buf[i].neg_cum_log_prob < buf[parent].neg_cum_log_prob) {
            std.mem.swap(HeapEntry, &buf[i], &buf[parent]);
            i = parent;
        } else break;
    }
}

fn heapSiftDown(buf: []HeapEntry, len: usize, start: usize) void {
    var i = start;
    while (true) {
        var smallest = i;
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        if (l < len and buf[l].neg_cum_log_prob < buf[smallest].neg_cum_log_prob) smallest = l;
        if (r < len and buf[r].neg_cum_log_prob < buf[smallest].neg_cum_log_prob) smallest = r;
        if (smallest == i) break;
        std.mem.swap(HeapEntry, &buf[i], &buf[smallest]);
        i = smallest;
    }
}

/// DDTree builder. All buffers are inline (stack-allocated), zero heap allocs.
pub const DDTreeBuilder = struct {
    nodes: [max_budget]TreeNode = undefined,
    n_nodes: u32 = 0,

    /// Budget (max nodes to build).
    budget: u32 = 64,
    /// Number of draft depth positions available.
    n_depths: u32 = 0,

    /// Pre-sorted top-K token IDs per depth.
    sorted_ids: [max_depth][max_budget]u32 = undefined,
    /// Corresponding log-probabilities (descending).
    sorted_lps: [max_depth][max_budget]f32 = undefined,
    /// Number of sorted entries per depth.
    n_sorted: [max_depth]u32 = undefined,

    /// Pre-sort top-B tokens at each depth from logit distributions.
    /// `logits_per_depth[d]` is a slice of vocab_size f32 log-probabilities.
    pub fn presort(self: *DDTreeBuilder, logits_per_depth: []const []const f32) void {
        self.n_depths = @intCast(@min(logits_per_depth.len, max_depth));
        for (0..self.n_depths) |d| {
            const logits = logits_per_depth[d];
            const k = @min(self.budget, @as(u32, @intCast(logits.len)));
            self.n_sorted[d] = partialTopK(
                logits,
                self.sorted_lps[d][0..k],
                self.sorted_ids[d][0..k],
            );
        }
    }

    /// Build tree via best-first expansion.
    pub fn buildTree(self: *DDTreeBuilder) void {
        self.n_nodes = 0;
        if (self.n_depths == 0) return;

        var heap_buf: [max_budget * 2]HeapEntry = undefined;
        var heap_len: usize = 0;

        // Seed: depth=0, rank=0
        if (self.n_sorted[0] > 0) {
            heap_buf[0] = .{
                .neg_cum_log_prob = -self.sorted_lps[0][0],
                .depth = 0,
                .rank = 0,
                .parent_idx = -1,
            };
            heap_len = 1;
        }

        while (heap_len > 0 and self.n_nodes < self.budget) {
            // Pop min (= max cum_log_prob because negated)
            const entry = heap_buf[0];
            heap_len -= 1;
            if (heap_len > 0) {
                heap_buf[0] = heap_buf[heap_len];
                heapSiftDown(&heap_buf, heap_len, 0);
            }

            const d = entry.depth;
            const r = entry.rank;
            const cum_lp = -entry.neg_cum_log_prob;

            const node_idx: i32 = @intCast(self.n_nodes);
            self.nodes[self.n_nodes] = .{
                .token_id = self.sorted_ids[d][r],
                .parent = entry.parent_idx,
                .depth = d,
                .rank = r,
                .cum_log_prob = cum_lp,
            };
            self.n_nodes += 1;

            // Push sibling (same depth, next rank)
            if (r + 1 < self.n_sorted[d] and heap_len < heap_buf.len) {
                const sib_lp = cum_lp - self.sorted_lps[d][r] + self.sorted_lps[d][r + 1];
                heap_buf[heap_len] = .{
                    .neg_cum_log_prob = -sib_lp,
                    .depth = d,
                    .rank = r + 1,
                    .parent_idx = entry.parent_idx,
                };
                heap_len += 1;
                heapSiftUp(&heap_buf, heap_len - 1);
            }

            // Push child (depth+1, rank=0)
            if (d + 1 < self.n_depths and self.n_sorted[d + 1] > 0 and heap_len < heap_buf.len) {
                const child_lp = cum_lp + self.sorted_lps[d + 1][0];
                heap_buf[heap_len] = .{
                    .neg_cum_log_prob = -child_lp,
                    .depth = d + 1,
                    .rank = 0,
                    .parent_idx = node_idx,
                };
                heap_len += 1;
                heapSiftUp(&heap_buf, heap_len - 1);
            }
        }
    }

    /// Compile tree into flat arrays for verification.
    pub fn compile(self: *const DDTreeBuilder, base_pos: u32) CompiledTree {
        var result = CompiledTree{};
        result.base_pos = base_pos;
        result.n_nodes = self.n_nodes;

        for (0..self.n_nodes) |i| {
            const node = self.nodes[i];
            result.input_ids[i] = node.token_id;
            result.position_ids[i] = base_pos + node.depth;
            result.parent_in_flat[i] = node.parent;

            // Build ancestor mask: walk parent chain
            @memset(&result.ancestor_masks[i], 0);
            result.ancestor_masks[i][i / 64] |= @as(u64, 1) << @intCast(@as(u6, @truncate(i)));
            var p = node.parent;
            while (p >= 0) {
                const pi: usize = @intCast(p);
                result.ancestor_masks[i][pi / 64] |= @as(u64, 1) << @intCast(@as(u6, @truncate(pi)));
                p = self.nodes[pi].parent;
            }

            // Build child index for parent
            const pidx: usize = if (node.parent < 0) max_budget else @intCast(node.parent);
            const cc = result.child_counts[pidx];
            if (cc < max_children_per_node) {
                result.child_tokens[pidx][cc] = node.token_id;
                result.child_indices[pidx][cc] = @intCast(i);
                result.child_counts[pidx] = cc + 1;
            } else {
                std.log.warn("DDTree: node {d} exceeds max_children_per_node ({d}), dropping child token {d}", .{ pidx, max_children_per_node, node.token_id });
            }
        }

        return result;
    }
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Partial top-K selection: find the K largest values from `src`.
/// Returns sorted (descending) values in `out_vals` and their indices in `out_ids`.
/// Returns actual count (may be < K if src is smaller).
fn partialTopK(src: []const f32, out_vals: []f32, out_ids: []u32) u32 {
    const k = @min(out_vals.len, src.len);
    if (k == 0) return 0;

    // Initialize with first K elements
    for (0..k) |i| {
        out_vals[i] = src[i];
        out_ids[i] = @intCast(i);
    }

    // Build min-heap on out_vals
    {
        var i: usize = k / 2;
        while (i > 0) {
            i -= 1;
            siftDown(out_vals[0..k], out_ids[0..k], i);
        }
    }

    // Scan remaining, replace min if larger
    for (k..src.len) |i| {
        if (src[i] > out_vals[0]) {
            out_vals[0] = src[i];
            out_ids[0] = @intCast(i);
            siftDown(out_vals[0..k], out_ids[0..k], 0);
        }
    }

    // Sort descending via insertion sort (K is small, typically ≤ 512)
    var i: usize = 1;
    while (i < k) : (i += 1) {
        var j = i;
        while (j > 0 and out_vals[j] > out_vals[j - 1]) {
            std.mem.swap(f32, &out_vals[j], &out_vals[j - 1]);
            std.mem.swap(u32, &out_ids[j], &out_ids[j - 1]);
            j -= 1;
        }
    }

    return @intCast(k);
}

fn siftDown(vals: []f32, ids: []u32, start: usize) void {
    var i = start;
    while (true) {
        var smallest = i;
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        if (l < vals.len and vals[l] < vals[smallest]) smallest = l;
        if (r < vals.len and vals[r] < vals[smallest]) smallest = r;
        if (smallest == i) break;
        std.mem.swap(f32, &vals[i], &vals[smallest]);
        std.mem.swap(u32, &ids[i], &ids[smallest]);
        i = smallest;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "partialTopK basic" {
    const src = [_]f32{ 0.1, 0.5, 0.2, 0.8, 0.3 };
    var vals: [3]f32 = undefined;
    var ids: [3]u32 = undefined;
    const n = partialTopK(&src, &vals, &ids);
    try std.testing.expectEqual(@as(u32, 3), n);
    // Top 3 descending: 0.8 (idx 3), 0.5 (idx 1), 0.3 (idx 4)
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vals[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), vals[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), vals[2], 0.001);
    try std.testing.expectEqual(@as(u32, 3), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 4), ids[2]);
}

test "DDTree build small tree" {
    // 2 depths, vocab=4, budget=5
    // Depth 0: log-probs [-0.1, -0.5, -1.0, -2.0] → token 0 is best
    // Depth 1: log-probs [-0.2, -0.3, -0.8, -1.5] → token 0 is best
    const depth0 = [_]f32{ -0.1, -0.5, -1.0, -2.0 };
    const depth1 = [_]f32{ -0.2, -0.3, -0.8, -1.5 };
    const logits = [_][]const f32{ &depth0, &depth1 };

    var builder = DDTreeBuilder{};
    builder.budget = 5;
    builder.presort(&logits);
    builder.buildTree();

    try std.testing.expectEqual(@as(u32, 5), builder.n_nodes);

    // First node should be (depth=0, rank=0, token=0) with cum_lp = -0.1
    try std.testing.expectEqual(@as(u32, 0), builder.nodes[0].token_id);
    try std.testing.expectEqual(@as(u16, 0), builder.nodes[0].depth);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1), builder.nodes[0].cum_log_prob, 0.01);
}

test "DDTree compile ancestor masks" {
    // Build a simple chain: node0 (root) → node1 → node2
    var builder = DDTreeBuilder{};
    builder.n_nodes = 3;
    builder.nodes[0] = .{ .token_id = 10, .parent = -1, .depth = 0, .rank = 0, .cum_log_prob = -0.1 };
    builder.nodes[1] = .{ .token_id = 20, .parent = 0, .depth = 1, .rank = 0, .cum_log_prob = -0.3 };
    builder.nodes[2] = .{ .token_id = 30, .parent = 1, .depth = 2, .rank = 0, .cum_log_prob = -0.6 };

    const tree = builder.compile(100);
    try std.testing.expectEqual(@as(u32, 3), tree.n_nodes);
    try std.testing.expectEqual(@as(u32, 100), tree.position_ids[0]);
    try std.testing.expectEqual(@as(u32, 101), tree.position_ids[1]);
    try std.testing.expectEqual(@as(u32, 102), tree.position_ids[2]);

    // Node 0: only self
    try std.testing.expectEqual(@as(u64, 0b001), tree.ancestor_masks[0][0]);
    // Node 1: self + node 0
    try std.testing.expectEqual(@as(u64, 0b011), tree.ancestor_masks[1][0]);
    // Node 2: self + node 1 + node 0
    try std.testing.expectEqual(@as(u64, 0b111), tree.ancestor_masks[2][0]);
}

test "DDTree presort selects correct top-K per depth" {
    // Verify presort directly: 6-token vocab, budget=3, should pick top 3
    const vocab = [_]f32{ -2.0, -0.1, -1.5, -0.5, -3.0, -0.3 };
    // Top 3 by descending value: -0.1 (idx 1), -0.3 (idx 5), -0.5 (idx 3)
    const logits = [_][]const f32{&vocab};

    var builder = DDTreeBuilder{};
    builder.budget = 3;
    builder.presort(&logits);

    try std.testing.expectEqual(@as(u32, 1), builder.n_depths);
    try std.testing.expectEqual(@as(u32, 3), builder.n_sorted[0]);

    // Sorted descending: -0.1 first, -0.3 second, -0.5 third
    try std.testing.expectApproxEqAbs(@as(f32, -0.1), builder.sorted_lps[0][0], 0.001);
    try std.testing.expectEqual(@as(u32, 1), builder.sorted_ids[0][0]);

    try std.testing.expectApproxEqAbs(@as(f32, -0.3), builder.sorted_lps[0][1], 0.001);
    try std.testing.expectEqual(@as(u32, 5), builder.sorted_ids[0][1]);

    try std.testing.expectApproxEqAbs(@as(f32, -0.5), builder.sorted_lps[0][2], 0.001);
    try std.testing.expectEqual(@as(u32, 3), builder.sorted_ids[0][2]);
}

test "DDTree build branching tree with siblings" {
    // 1 depth, 4 tokens, budget=4
    // All nodes should be at depth 0 with different ranks as siblings
    const depth0 = [_]f32{ -0.1, -0.2, -0.3, -0.4 };
    const logits = [_][]const f32{&depth0};

    var builder = DDTreeBuilder{};
    builder.budget = 4;
    builder.presort(&logits);
    builder.buildTree();

    try std.testing.expectEqual(@as(u32, 4), builder.n_nodes);

    // All nodes at depth 0, all with parent -1
    for (0..4) |i| {
        try std.testing.expectEqual(@as(u16, 0), builder.nodes[i].depth);
        try std.testing.expectEqual(@as(i32, -1), builder.nodes[i].parent);
    }

    // Compile and verify all are children of root
    const tree = builder.compile(0);
    try std.testing.expectEqual(@as(u32, 4), tree.n_nodes);

    // Root has 4 children
    const root_idx = max_budget;
    try std.testing.expectEqual(@as(u8, 4), tree.child_counts[root_idx]);
}

test "DDTree empty build" {
    // Zero depths should produce zero nodes
    var builder = DDTreeBuilder{};
    builder.budget = 10;
    const logits = [_][]const f32{};
    builder.presort(&logits);
    builder.buildTree();

    try std.testing.expectEqual(@as(u32, 0), builder.n_nodes);

    // Compile empty tree
    const tree = builder.compile(50);
    try std.testing.expectEqual(@as(u32, 0), tree.n_nodes);
    try std.testing.expectEqual(@as(u32, 50), tree.base_pos);
}

test "findChild and isAncestor" {
    var builder = DDTreeBuilder{};
    builder.n_nodes = 3;
    builder.nodes[0] = .{ .token_id = 10, .parent = -1, .depth = 0, .rank = 0, .cum_log_prob = 0 };
    builder.nodes[1] = .{ .token_id = 20, .parent = 0, .depth = 1, .rank = 0, .cum_log_prob = 0 };
    builder.nodes[2] = .{ .token_id = 99, .parent = -1, .depth = 0, .rank = 1, .cum_log_prob = 0 };

    const tree = builder.compile(0);

    // findChild: root (-1) has children 10 and 99
    try std.testing.expectEqual(@as(?u32, 0), tree.findChild(-1, 10));
    try std.testing.expectEqual(@as(?u32, 2), tree.findChild(-1, 99));
    try std.testing.expect(tree.findChild(-1, 50) == null);

    // findChild: node 0 has child 20
    try std.testing.expectEqual(@as(?u32, 1), tree.findChild(0, 20));
    try std.testing.expect(tree.findChild(0, 10) == null);

    // isAncestor: node 2 (depth 0) is self-only
    try std.testing.expect(CompiledTree.isAncestor(tree.ancestor_masks[2], 2));
    try std.testing.expect(!CompiledTree.isAncestor(tree.ancestor_masks[2], 0));
    try std.testing.expect(!CompiledTree.isAncestor(tree.ancestor_masks[2], 1));
}

test "fuzz: all ddtree functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // -- isAncestor: pub fn on CompiledTree --
            var mask: [8]u64 = undefined;
            for (0..8) |i| mask[i] = smith.valueWithHash(u64, @intCast(i));
            const j = smith.valueWithHash(usize, 100) % max_budget;
            _ = CompiledTree.isAncestor(mask, j);

            // -- DDTreeBuilder.presort + buildTree + compile --
            const n_depths_raw = smith.valueWithHash(u8, 200);
            const n_depths: usize = @min(n_depths_raw % 5, max_depth); // 0..4 depths
            const vocab_size: usize = 4;
            var vocab_bufs: [4][4]f32 = undefined;
            for (0..n_depths) |d| {
                for (0..vocab_size) |v| {
                    const bits = smith.valueWithHash(u32, @intCast(d * 100 + v + 300));
                    // Map to finite float: clear exponent=0xFF (inf/nan)
                    const clean_bits = bits & 0x7F7FFFFF;
                    var val = @as(f32, @bitCast(clean_bits));
                    if (!std.math.isFinite(val)) val = 0.0;
                    // Negate so values are log-probs (negative)
                    vocab_bufs[d][v] = -@abs(val);
                }
            }
            var slices: [4][]const f32 = undefined;
            for (0..n_depths) |d| slices[d] = &vocab_bufs[d];
            const logits_slice: []const []const f32 = slices[0..n_depths];

            var builder = DDTreeBuilder{};
            const budget_raw = smith.valueWithHash(u8, 400);
            builder.budget = @as(u32, (budget_raw % 16)) + 1; // 1..16

            // pub fn presort
            builder.presort(logits_slice);
            std.debug.assert(builder.n_depths == @as(u32, @intCast(n_depths)));

            // pub fn buildTree
            builder.buildTree();
            std.debug.assert(builder.n_nodes <= builder.budget);

            // pub fn compile
            const base_pos = smith.valueWithHash(u32, 500) % 4096;
            const tree = builder.compile(base_pos);
            std.debug.assert(tree.n_nodes == builder.n_nodes);
            std.debug.assert(tree.base_pos == base_pos);

            // Verify position_ids are base_pos + depth for each node
            for (0..tree.n_nodes) |i| {
                std.debug.assert(tree.position_ids[i] == base_pos + builder.nodes[i].depth);
            }

            // -- findChild: search for each node's token under its parent --
            for (0..tree.n_nodes) |i| {
                const node = builder.nodes[i];
                const found = tree.findChild(node.parent, node.token_id);
                // Should find this node (or a sibling with same token)
                std.debug.assert(found != null);
            }
            // findChild with non-existent token should return null
            _ = tree.findChild(-1, 0xDEADBEEF);

            // -- isAncestor on compiled tree: self is always ancestor --
            for (0..tree.n_nodes) |i| {
                std.debug.assert(CompiledTree.isAncestor(tree.ancestor_masks[i], i));
            }

            // -- Verify pub consts are accessible --
            std.debug.assert(max_budget == 512);
            std.debug.assert(max_depth == 32);

            // -- Verify TreeNode struct fields --
            const tn = TreeNode{
                .token_id = smith.valueWithHash(u32, 600),
                .parent = @bitCast(smith.valueWithHash(u32, 601)),
                .depth = smith.valueWithHash(u16, 602),
                .rank = smith.valueWithHash(u16, 603),
                .cum_log_prob = -1.0,
            };
            _ = tn;
        }
    }.f, .{});
}
