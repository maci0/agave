//! CPU Rotary Position Embedding (RoPE) kernel.

const std = @import("std");

/// Maximum half rope_dim for precomputed cos/sin buffers.
const max_rope_half_dim: usize = 512;

/// Applies Rotary Position Embedding (RoPE) in-place.
/// Uses split-complex layout: pairs `[i, i+half]` are rotated together
/// (matches CUDA convention; NOT interleaved `[2i, 2i+1]`).
pub fn rope(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
    const half = rope_dim / 2;
    const p: f32 = @floatFromInt(pos);
    const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));
    const neg_log_theta: f32 = -@log(theta);

    // Precompute cos/sin for each frequency (shared across all heads)
    std.debug.assert(half <= max_rope_half_dim);
    var cos_buf: [max_rope_half_dim]f32 = undefined;
    var sin_buf: [max_rope_half_dim]f32 = undefined;
    for (0..half) |i| {
        const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
        const angle = p * freq;
        cos_buf[i] = @cos(angle);
        sin_buf[i] = @sin(angle);
    }

    for (0..n_heads) |h| {
        const base = h * head_dim;
        for (0..half) |i| {
            const r = x[base + i];
            const im = x[base + i + half];
            x[base + i] = r * cos_buf[i] - im * sin_buf[i];
            x[base + i + half] = r * sin_buf[i] + im * cos_buf[i];
        }
    }
}

test "rope pos=0 is identity" {
    // At position 0, all angles are 0 → cos=1, sin=0 → no rotation.
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    rope(&x, 0, 1, 4, 4, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x[3], 1e-6);
}

test "rope preserves magnitude" {
    // RoPE is a rotation — magnitude should be preserved per pair.
    var x = [_]f32{ 3.0, 0.0, 4.0, 0.0 };
    rope(&x, 5, 1, 4, 4, 10000.0);
    // Pair 0: (x[0], x[2]) should have magnitude 5.0 (sqrt(3^2+4^2))
    const mag0 = @sqrt(x[0] * x[0] + x[2] * x[2]);
    const mag1 = @sqrt(x[1] * x[1] + x[3] * x[3]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), mag0, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mag1, 1e-4);
}
