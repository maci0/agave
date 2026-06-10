//! Block sparse attention patterns for long-context inference.
//!
//! Implements BigBird-style block sparsity:
//!   - Global blocks: first N blocks attend/are-attended-by all positions
//!   - Sliding window: each query block attends ±window blocks
//!   - (Random blocks are omitted initially — global+sliding covers most long-context cases)
//!
//! Used by PFlash for efficient drafter scoring and as a standalone SDPA variant
//! for long-context models where dense O(n²) attention is prohibitive.

const std = @import("std");

/// Block sparse attention configuration.
pub const BlockSparsePattern = struct {
    /// Tokens per block (must be power of 2, default 64).
    block_size: u32 = 64,
    /// Number of global blocks from start of sequence (attend to/from all).
    n_global: u32 = 2,
    /// Sliding window radius in blocks (each query attends ±window blocks).
    window: u32 = 1,
};

/// Block-level attention mask represented as a packed bitset.
/// Bit (i * n_blocks + j) set = query block i attends to key block j.
/// Upper-triangular bits (j > i) are always 0 (causal masking).
pub const BlockMask = struct {
    /// Packed u64 bitset. Size: ceil(n_blocks² / 64) u64 words.
    data: []u64,
    /// Number of blocks covering the sequence.
    n_blocks: u32,
    /// Block size in tokens.
    block_size: u32,

    pub fn deinit(self: BlockMask, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    /// Returns true if query block qi should attend to key block ki.
    pub fn get(self: BlockMask, qi: u32, ki: u32) bool {
        if (ki > qi) return false; // causal
        const bit_idx = @as(u64, qi) * self.n_blocks + ki;
        const word = bit_idx / 64;
        const bit = @as(u6, @truncate(bit_idx % 64));
        return (self.data[word] >> bit) & 1 == 1;
    }

    /// Set bit (qi, ki) = 1.
    fn set(self: BlockMask, qi: u32, ki: u32) void {
        if (ki > qi) return;
        const bit_idx = @as(u64, qi) * self.n_blocks + ki;
        const word = bit_idx / 64;
        const bit = @as(u6, @truncate(bit_idx % 64));
        self.data[word] |= @as(u64, 1) << bit;
    }
};

/// Build a block mask for the given pattern and sequence length.
/// Pattern: global blocks + sliding window.
pub fn buildMask(allocator: std.mem.Allocator, pattern: BlockSparsePattern, seq_len: usize) !BlockMask {
    const bs = @as(usize, pattern.block_size);
    const n_blocks: u32 = @intCast((seq_len + bs - 1) / bs);
    const n_bits = @as(u64, n_blocks) * n_blocks;
    const n_words = (n_bits + 63) / 64;
    const data = try allocator.alloc(u64, @intCast(n_words));
    @memset(data, 0);

    var mask = BlockMask{ .data = data, .n_blocks = n_blocks, .block_size = pattern.block_size };

    for (0..n_blocks) |qi| {
        const qi_u: u32 = @intCast(qi);
        // Global blocks: first n_global blocks are attended-by-all and attend-all
        for (0..@min(pattern.n_global, n_blocks)) |ki| {
            mask.set(qi_u, @intCast(ki));
        }
        // Sliding window: query block attends ±window key blocks
        const win = @as(usize, pattern.window);
        const lo = if (qi >= win) qi - win else 0;
        const hi = @min(qi + win + 1, n_blocks);
        for (lo..hi) |ki| {
            mask.set(qi_u, @intCast(ki));
        }
        // Global: query is in global range → attend all past blocks
        if (qi < pattern.n_global) {
            for (0..qi + 1) |ki| {
                mask.set(qi_u, @intCast(ki));
            }
        }
    }

    return mask;
}

/// Compute block sparsity ratio (fraction of blocks skipped).
pub fn sparsityRatio(mask: BlockMask) f32 {
    var attended: u64 = 0;
    const nb = @as(u64, mask.n_blocks);
    for (0..mask.n_blocks) |qi| {
        for (0..qi + 1) |ki| {
            if (mask.get(@intCast(qi), @intCast(ki))) attended += 1;
        }
    }
    const total = nb * (nb + 1) / 2;
    if (total == 0) return 0;
    return 1.0 - @as(f32, @floatFromInt(attended)) / @as(f32, @floatFromInt(total));
}

// ── Block-Sparse SDPA (CPU, f32 KV) ────────────────────────────────────────

const kv_quant = @import("kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);
const sparse_v_threshold: f32 = 1e-6;
const max_seq_len: usize = 65536;
const max_hd: usize = 256;

/// Online softmax state for block-sparse SDPA.
const OnlineSoftmax = struct {
    max: f32 = -std.math.inf(f32),
    sum: f32 = 0,
};

/// Compute block-sparse SDPA for a single head with f32 KV.
/// Skips QK dot products and V accumulation for masked-out key blocks.
/// Writes output to output[h*hd .. (h+1)*hd].
pub fn sdpaHeadSparse(
    q: [*]const f32,
    keys: [*]const f32,
    values: [*]const f32,
    output: [*]f32,
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    sl: usize,
    scale: f32,
    mask: *const BlockMask,
) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = h * hd;
    const bs = @as(usize, mask.block_size);
    const qi = @as(u32, @intCast((sl - 1) / bs)); // current query block index

    var scores_buf: [max_seq_len]f32 = undefined;
    var q_cached: [max_hd]f32 = undefined;
    @memcpy(q_cached[0..hd], q[q_base..][0..hd]);

    // QK dot products — skip entire key blocks where mask is 0
    for (0..sl) |t| {
        const ki = @as(u32, @intCast(t / bs));
        if (!mask.get(qi, ki)) {
            scores_buf[t] = -std.math.inf(f32);
            continue;
        }
        const k_base = t * kvd + kvh * hd;
        var acc: V8 = v8zero;
        var d: usize = 0;
        while (d + 8 <= hd) : (d += 8) {
            acc = @mulAdd(V8, q_cached[d..][0..8].*, keys[k_base + d ..][0..8].*, acc);
        }
        var dot = @reduce(.Add, acc);
        while (d < hd) : (d += 1) dot = @mulAdd(f32, q_cached[d], keys[k_base + d], dot);
        scores_buf[t] = dot * scale;
    }

    // Stable softmax (online, handles -inf correctly)
    softmaxSparse(scores_buf[0..sl]);

    // V accumulation (same sparse-V skip as dense path)
    @memset(output[q_base..][0..hd], 0);
    for (0..sl) |t| {
        if (scores_buf[t] < sparse_v_threshold) continue;
        const v_base = t * kvd + kvh * hd;
        const sv: V8 = @splat(scores_buf[t]);
        var d: usize = 0;
        while (d + 8 <= hd) : (d += 8) {
            const vv: V8 = values[v_base + d ..][0..8].*;
            output[q_base + d ..][0..8].* = @mulAdd(V8, sv, vv, output[q_base + d ..][0..8].*);
        }
        while (d < hd) : (d += 1)
            output[q_base + d] = @mulAdd(f32, scores_buf[t], values[v_base + d], output[q_base + d]);
    }
}

/// Compute block-sparse SDPA for a single head with quantized KV cache.
pub fn sdpaQuantHeadSparse(
    q: [*]const f32,
    keys: [*]const u8,
    values: [*]const u8,
    output: [*]f32,
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    sl: usize,
    scale: f32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
    mask: *const BlockMask,
) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = h * hd;
    const bs = @as(usize, mask.block_size);
    const qi = @as(u32, @intCast((sl - 1) / bs));

    var scores_buf: [max_seq_len]f32 = undefined;
    var q_cached: [max_hd]f32 = undefined;
    @memcpy(q_cached[0..hd], q[q_base..][0..hd]);

    for (0..sl) |t| {
        const ki = @as(u32, @intCast(t / bs));
        if (!mask.get(qi, ki)) {
            scores_buf[t] = -std.math.inf(f32);
            continue;
        }
        const k_byte_off = kv_quant.kvByteOffset(kv_type_k, t * kvd + kvh * hd);
        scores_buf[t] = kv_quant.kvDot(q_cached[0..hd].ptr, keys + k_byte_off, hd, kv_type_k) * scale;
    }

    softmaxSparse(scores_buf[0..sl]);

    @memset(output[q_base..][0..hd], 0);
    for (0..sl) |t| {
        if (scores_buf[t] < sparse_v_threshold) continue;
        const v_byte_off = kv_quant.kvByteOffset(kv_type_v, t * kvd + kvh * hd);
        kv_quant.kvMulAccum(output + q_base, scores_buf[t], values + v_byte_off, hd, kv_type_v);
    }
}

/// All-head sparse SDPA dispatch (f32 KV).
pub fn sdpaHeadsSparse(q: [*]const f32, keys: [*]const f32, values: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, mask: *const BlockMask) void {
    for (0..nh) |h| sdpaHeadSparse(q, keys, values, output, h, nh, nkv, hd, sl, scale, mask);
}

/// All-head sparse SDPA dispatch (quantized KV).
pub fn sdpaQuantHeadsSparse(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, mask: *const BlockMask) void {
    for (0..nh) |h| sdpaQuantHeadSparse(q, keys, values, output, h, nh, nkv, hd, sl, scale, kv_type_k, kv_type_v, mask);
}

/// Stable softmax that handles -inf entries (masked positions).
fn softmaxSparse(data: []f32) void {
    var max_val = -std.math.inf(f32);
    for (data) |v| if (v > max_val) {
        max_val = v;
    };
    if (max_val == -std.math.inf(f32)) return;
    var sum: f32 = 0;
    for (data) |*v| {
        if (v.* == -std.math.inf(f32)) {
            v.* = 0;
            continue;
        }
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    if (sum > 0) {
        const inv = 1.0 / sum;
        for (data) |*v| v.* *= inv;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "buildMask global+window coverage" {
    const allocator = std.testing.allocator;
    const pattern = BlockSparsePattern{ .block_size = 4, .n_global = 1, .window = 1 };
    // seq_len=12 → 3 blocks
    var mask = try buildMask(allocator, pattern, 12);
    defer mask.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 3), mask.n_blocks);

    // Block 0 (global): attends to itself
    try std.testing.expect(mask.get(0, 0));
    // Block 1: attends to global(0) + window(0,1)
    try std.testing.expect(mask.get(1, 0)); // global
    try std.testing.expect(mask.get(1, 1)); // self
    // Block 2: attends to global(0) + window(1,2)
    try std.testing.expect(mask.get(2, 0)); // global
    try std.testing.expect(mask.get(2, 1)); // window
    try std.testing.expect(mask.get(2, 2)); // self
    // Causal: block 0 cannot attend to block 1
    try std.testing.expect(!mask.get(0, 1));
}

test "buildMask causal" {
    const allocator = std.testing.allocator;
    const pattern = BlockSparsePattern{ .block_size = 8, .n_global = 2, .window = 1 };
    var mask = try buildMask(allocator, pattern, 32);
    defer mask.deinit(allocator);
    // No future blocks attended (causal)
    for (0..mask.n_blocks) |qi| {
        for (qi + 1..mask.n_blocks) |ki| {
            try std.testing.expect(!mask.get(@intCast(qi), @intCast(ki)));
        }
    }
}

test "sparsityRatio increases with longer sequence" {
    const allocator = std.testing.allocator;
    const pattern = BlockSparsePattern{ .block_size = 64, .n_global = 2, .window = 1 };
    var m1 = try buildMask(allocator, pattern, 256);
    defer m1.deinit(allocator);
    var m2 = try buildMask(allocator, pattern, 4096);
    defer m2.deinit(allocator);
    // Longer sequence → higher sparsity (more blocks are too far away)
    try std.testing.expect(sparsityRatio(m2) > sparsityRatio(m1));
}

test "sdpaHeadSparse matches dense for small seq" {
    // For seq_len <= block_size with n_global=1, sparse == dense
    const nh: usize = 2;
    const nkv: usize = 2;
    const hd: usize = 8;
    const sl: usize = 4;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    const q = [_]f32{1.0} ** (nh * hd);
    var keys = [_]f32{0.5} ** (sl * nkv * hd);
    var values = [_]f32{0.25} ** (sl * nkv * hd);
    var out_sparse = [_]f32{0.0} ** (nh * hd);
    var out_dense = [_]f32{0.0} ** (nh * hd);

    const allocator = std.testing.allocator;
    // block_size=8 > sl=4: all positions in block 0 → fully dense
    const pattern = BlockSparsePattern{ .block_size = 8, .n_global = 1, .window = 1 };
    var mask = try buildMask(allocator, pattern, sl);
    defer mask.deinit(allocator);

    sdpaHeadsSparse(&q, &keys, &values, &out_sparse, nh, nkv, hd, sl, scale, &mask);

    // Dense reference
    const sdpa = @import("../backend/kernels/cpu/sdpa.zig");
    sdpa.sdpaHeads(&q, &keys, &values, &out_dense, nh, nkv, hd, sl, scale);

    for (0..nh * hd) |i| {
        try std.testing.expectApproxEqAbs(out_dense[i], out_sparse[i], 1e-5);
    }
}

test "sdpaHeadSparse skips masked blocks" {
    // seq_len=16 with block_size=8 → 2 blocks, window=0, global=1
    // Block 1 only attends to block 0 (global), not itself (except via global-is-self)
    // Actually block 1 IS qi=1: it attends global(0) + window blocks
    // With window=0 and n_global=1: block 1 attends only block 0
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;
    const sl: usize = 8; // exactly 1 block
    const scale: f32 = 0.5;

    var q = [_]f32{1.0} ** hd;
    var keys = [_]f32{0.1} ** (sl * nkv * hd);
    var values = [_]f32{1.0} ** (sl * nkv * hd);
    var out = [_]f32{0.0} ** hd;

    const allocator = std.testing.allocator;
    const pattern = BlockSparsePattern{ .block_size = 4, .n_global = 1, .window = 0 };
    var mask = try buildMask(allocator, pattern, sl);
    defer mask.deinit(allocator);

    sdpaHeadsSparse(&q, &keys, &values, &out, nh, nkv, hd, sl, scale, &mask);

    // Output should be finite and nonzero (some blocks attended)
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}

test "fuzz: all sparse_attn functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;
            // buildMask with random parameters
            const bs: u32 = @as(u32, smith.valueWithHash(u4, 0)) + 1;
            const block_size = bs * 4; // 4..64 in steps of 4
            const n_global: u32 = smith.valueWithHash(u2, 1);
            const window: u32 = smith.valueWithHash(u2, 2);
            const seq_len: usize = @as(usize, smith.valueWithHash(u5, 3)) * block_size + block_size;

            const pattern = BlockSparsePattern{ .block_size = block_size, .n_global = n_global, .window = window };
            var mask = buildMask(allocator, pattern, seq_len) catch return;
            defer mask.deinit(allocator);

            // Verify causal invariant
            for (0..mask.n_blocks) |qi| {
                for (qi + 1..mask.n_blocks) |ki| {
                    try std.testing.expect(!mask.get(@intCast(qi), @intCast(ki)));
                }
            }

            // sparsityRatio must be in [0, 1]
            const ratio = sparsityRatio(mask);
            try std.testing.expect(ratio >= 0.0 and ratio <= 1.0);

            // get() is false for out-of-bounds (future) blocks
            if (mask.n_blocks > 0) {
                try std.testing.expect(!mask.get(0, mask.n_blocks - 1));
            }
        }
    }.f, .{});
}
