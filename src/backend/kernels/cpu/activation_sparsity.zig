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
