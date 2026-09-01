//! Activation sparsity detection for GEMV block skipping.
//!
//! FFN outputs (after SiLU/GELU) have ~40% near-zero values. Block-level skip
//! checks avoid processing weight blocks where the input slice is negligible.
//! This leaf module is imported by every per-format GEMV kernel and by the
//! GEMV dispatcher, so it must not depend on any of them.

const std = @import("std");

/// Skip GEMV blocks where all input values are below this magnitude.
/// Measured: SiLU outputs have ~40% values below 0.01 (Qwen3.5 9B).
/// Set to 0 to disable sparsity skipping entirely.
pub const sparse_threshold: f32 = 0.005;

/// Check if all elements in a contiguous block are below the sparse threshold.
/// Uses SIMD max-abs reduction for speed (~1 cycle per 8 elements).
/// Returns true if the block can be safely skipped (all near-zero).
pub inline fn isBlockSparse(x: [*]const f32, start: usize, len: usize) bool {
    if (sparse_threshold == 0) return false;
    const V8 = @Vector(8, f32);
    const zero: V8 = @splat(0.0);
    var max_v: V8 = zero;
    var i: usize = start;
    while (i + 8 <= start + len) : (i += 8) {
        const v: V8 = x[i..][0..8].*;
        const abs_v = @select(f32, v > zero, v, zero - v);
        max_v = @max(max_v, abs_v);
    }
    // Handle tail elements
    var tail_max: f32 = 0;
    while (i < start + len) : (i += 1) {
        const a = @abs(x[i]);
        if (a > tail_max) tail_max = a;
    }
    return @max(@reduce(.Max, max_v), tail_max) < sparse_threshold;
}

/// Blocks a hoisted mask can cover. 4096 blocks is k = 131072 at the smallest
/// block size in use (32 elements) and k = 1048576 at the largest (256), so no
/// supported context reaches it; a GEMV past the bound falls back to testing
/// each block in place.
pub const max_mask_blocks: usize = 4096;

/// Per-block sparsity of one activation vector, computed once per GEMV.
///
/// The activation vector does not change while a GEMV walks its output rows, so
/// testing a block per row group repeats identical work n/rows_per_group times.
/// Hoisting it out of the row loop turns that into one pass.
pub const BlockMask = struct {
    bits: [max_mask_blocks / 64]u64,
    /// Blocks actually described. Anything at or past this reads as dense, which
    /// is the safe direction: a missed skip costs time, a wrong skip costs
    /// correctness.
    covered: usize,

    pub inline fn isSparse(self: *const BlockMask, b: usize) bool {
        // Both bounds, not just `covered`: `covered <= max_mask_blocks` holds by
        // construction but is not visible to the compiler, so a comptime-known
        // index past the array would be analyzed as an out-of-bounds read.
        if (b >= self.covered or b >= max_mask_blocks) return false;
        return (self.bits[b / 64] >> @intCast(b % 64)) & 1 != 0;
    }
};

/// Sparsity of every block of `x[0..k]`, `block_elems` elements per block.
///
/// The final block is clamped to `k`: `nb` rounds up, so scanning a full block
/// there would read past the activation buffer and let out-of-bounds bytes
/// decide whether real elements get skipped.
pub fn blockMask(x: [*]const f32, nb: usize, block_elems: usize, k: usize) BlockMask {
    var mask = BlockMask{ .bits = @splat(0), .covered = 0 };
    if (sparse_threshold == 0 or nb > max_mask_blocks) return mask;
    for (0..nb) |b| {
        const start = b * block_elems;
        if (start >= k) break;
        const len = @min(block_elems, k - start);
        if (isBlockSparse(x, start, len)) mask.bits[b / 64] |= @as(u64, 1) << @intCast(b % 64);
        mask.covered = b + 1;
    }
    return mask;
}

// ── Tests ────────────────────────────────────────────────────────

test "isBlockSparse all zeros" {
    var x = [_]f32{0.0} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
}

test "isBlockSparse below threshold" {
    var x = [_]f32{0.001} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
}

test "isBlockSparse above threshold" {
    var x = [_]f32{0.001} ** 32;
    x[16] = 0.01;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "isBlockSparse negative values" {
    var x = [_]f32{-0.004} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
    x[0] = -0.006;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "isBlockSparse partial block" {
    var x = [_]f32{0.0} ** 13;
    try std.testing.expect(isBlockSparse(&x, 0, 13));
    x[12] = 1.0;
    try std.testing.expect(!isBlockSparse(&x, 0, 13));
}

test "isBlockSparse offset" {
    var x = [_]f32{1.0} ** 64;
    // Fill second half with zeros
    for (32..64) |i| x[i] = 0.0;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
    try std.testing.expect(isBlockSparse(&x, 32, 32));
}

test "isBlockSparse disabled when threshold is zero" {
    if (sparse_threshold != 0) return error.SkipZigTest;
    var x = [_]f32{0.0} ** 32;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "blockMask matches per-block isBlockSparse" {
    var x: [256]f32 = @splat(0.0);
    for (64..128) |i| x[i] = 1.0; // block 2 and 3 of 32 dense
    const nb = 8;
    const mask = blockMask(&x, nb, 32, x.len);
    try std.testing.expectEqual(nb, mask.covered);
    for (0..nb) |b| {
        try std.testing.expectEqual(isBlockSparse(&x, b * 32, 32), mask.isSparse(b));
    }
}

test "blockMask clamps the final partial block to k" {
    // k = 40 with 32-element blocks: block 1 covers only x[32..40].
    var x: [40]f32 = @splat(0.0);
    x[39] = 1.0;
    const mask = blockMask(&x, 2, 32, x.len);
    try std.testing.expect(mask.isSparse(0));
    try std.testing.expect(!mask.isSparse(1)); // the live tail element is seen
}

test "blockMask reports dense past its coverage" {
    var x: [32]f32 = @splat(0.0);
    const mask = blockMask(&x, 1, 32, x.len);
    try std.testing.expect(mask.isSparse(0));
    // A block the mask never described must never be skipped.
    try std.testing.expect(!mask.isSparse(1));
    try std.testing.expect(!mask.isSparse(max_mask_blocks));
}

test "blockMask above the block bound skips nothing" {
    var x: [32]f32 = @splat(0.0);
    const mask = blockMask(&x, max_mask_blocks + 1, 32, x.len);
    try std.testing.expectEqual(@as(usize, 0), mask.covered);
    try std.testing.expect(!mask.isSparse(0));
}
