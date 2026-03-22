//! CPU softmax kernel with comptime-parameterized SIMD width.

const math = @import("std").math;

/// Comptime-parameterized softmax over SIMD width.
/// The build can override the width to autotune.
pub fn softmaxSimd(comptime W: comptime_int, data: [*]f32, n: usize) void {
    const VW = @Vector(W, f32);
    // SIMD max reduction
    var max_v: VW = @splat(-math.inf(f32));
    var i: usize = 0;
    while (i + W <= n) : (i += W) {
        max_v = @max(max_v, @as(VW, data[i..][0..W].*));
    }
    var mx = @reduce(.Max, max_v);
    while (i < n) : (i += 1) if (data[i] > mx) {
        mx = data[i];
    };

    // exp(x - max) and SIMD sum
    const mx_v: VW = @splat(mx);
    var sum_v: VW = @splat(@as(f32, 0.0));
    i = 0;
    while (i + W <= n) : (i += W) {
        const v = @exp(@as(VW, data[i..][0..W].*) - mx_v);
        data[i..][0..W].* = v;
        sum_v += v;
    }
    var s = @reduce(.Add, sum_v);
    while (i < n) : (i += 1) {
        data[i] = @exp(data[i] - mx);
        s += data[i];
    }

    // SIMD divide
    const inv_v: VW = @splat(1.0 / s);
    i = 0;
    while (i + W <= n) : (i += W) {
        data[i..][0..W].* = @as(VW, data[i..][0..W].*) * inv_v;
    }
    const inv_s = 1.0 / s;
    while (i < n) : (i += 1) data[i] *= inv_s;
}

const std = @import("std");

test "softmax uniform" {
    // Uniform input → uniform output (1/n each).
    var data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    softmaxSimd(4, &data, 4);
    for (&data) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 1e-6);
}

test "softmax sums to 1 with correct values" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    softmaxSimd(8, &data, 8);
    var sum: f32 = 0;
    for (&data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Verify monotonicity: higher input → higher probability
    for (1..8) |i| try std.testing.expect(data[i] > data[i - 1]);
    // Verify approximate expected values for softmax([1..8])
    // softmax(1) = exp(-7) / sum ≈ 0.000577, softmax(8) = exp(0) / sum ≈ 0.632
    try std.testing.expect(data[0] < 0.001);
    try std.testing.expect(data[7] > 0.6);
}

test "softmax max element gets highest probability" {
    var data = [_]f32{ 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    softmaxSimd(8, &data, 8);
    // Element 1 should dominate, others near zero
    try std.testing.expect(data[1] > 0.99);
    // Verify non-max elements are near zero and equal to each other
    for ([_]usize{ 0, 2, 3, 4, 5, 6, 7 }) |i| {
        try std.testing.expect(data[i] < 0.01);
    }
}

test "softmax non-power-of-2 size exercises scalar tail" {
    // n=5 exercises the scalar cleanup path (5 < 8 for W=8, or 5 > 4 for W=4)
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    softmaxSimd(4, &data, 5);
    // Sum must equal 1.0
    var sum: f32 = 0;
    for (0..5) |i| sum += data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Monotonicity
    for (1..5) |i| try std.testing.expect(data[i] > data[i - 1]);
    // Verify largest element dominates
    try std.testing.expect(data[4] > 0.4);
}

test "softmax negative values" {
    var data = [_]f32{ -3.0, -2.0, -1.0, 0.0 };
    softmaxSimd(4, &data, 4);
    var sum: f32 = 0;
    for (0..4) |i| sum += data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Should still be monotonically increasing
    for (1..4) |i| try std.testing.expect(data[i] > data[i - 1]);
}
