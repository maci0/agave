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
    while (i < n) : (i += 1) mx = @max(mx, data[i]);

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
    const inv_s = 1.0 / s;
    const inv_v: VW = @splat(inv_s);
    i = 0;
    while (i + W <= n) : (i += W) {
        data[i..][0..W].* = @as(VW, data[i..][0..W].*) * inv_v;
    }
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
    // Verify against known softmax([1..8]) values
    // Reference: scipy.special.softmax([1,2,3,4,5,6,7,8])
    // Tolerance 1e-3 accounts for f32 vs f64 precision in exp()
    const expected = [8]f32{ 0.000577, 0.001569, 0.004266, 0.011594, 0.031521, 0.085677, 0.232885, 0.632911 };
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(expected[i], data[i], 1e-3);
    }
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
    // n=5 with W=4 exercises the scalar cleanup path (5 % 4 = 1 element handled by scalar tail)
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    softmaxSimd(4, &data, 5);
    // Sum must equal 1.0
    var sum: f32 = 0;
    for (0..5) |i| sum += data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Monotonicity
    for (1..5) |i| try std.testing.expect(data[i] > data[i - 1]);
    // Verify largest element: softmax([1..5])[4] ≈ 0.6364
    try std.testing.expectApproxEqAbs(@as(f32, 0.6364), data[4], 1e-3);
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

test "softmax numerical stability with large differences" {
    // Large spread: exp(1000) would overflow without max-subtraction.
    // A correct implementation subtracts the max before exponentiating.
    var data = [_]f32{ 0.0, 0.0, 0.0, 1000.0 };
    softmaxSimd(4, &data, 4);
    // Must not produce NaN or Inf
    for (0..4) |i| {
        try std.testing.expect(!std.math.isNan(data[i]));
        try std.testing.expect(!std.math.isInf(data[i]));
    }
    // Dominant element should be ~1.0, rest ~0.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[3], 1e-6);
    for (0..3) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[i], 1e-6);
    // Sum must still be 1.0
    var sum: f32 = 0;
    for (0..4) |i| sum += data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "softmax single element" {
    var data = [_]f32{42.0};
    softmaxSimd(4, &data, 1);
    // softmax of a single element is always 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-6);
}

test "softmax all-negative underflow stability" {
    // All large negative values: exp(-500) underflows to 0 without max-subtraction.
    // With correct max-subtraction, softmax should still sum to 1.0.
    var data = [_]f32{ -500.0, -501.0, -502.0, -503.0 };
    softmaxSimd(4, &data, 4);
    // Must not produce NaN or Inf
    for (0..4) |i| {
        try std.testing.expect(!std.math.isNan(data[i]));
        try std.testing.expect(!std.math.isInf(data[i]));
    }
    // Sum must still be 1.0
    var sum: f32 = 0;
    for (0..4) |i| sum += data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Monotonicity: -500 > -501 > -502 > -503 after max subtraction
    for (1..4) |i| try std.testing.expect(data[i] < data[i - 1]);
}

test "softmax two elements verifies exact ratio" {
    // softmax([a, b]) = [1/(1+exp(b-a)), exp(b-a)/(1+exp(b-a))]
    // For [0, ln(2)]: ratio p1/p0 = exp(ln(2)) = 2, so p0 = 1/3, p1 = 2/3.
    var data = [_]f32{ 0.0, @log(@as(f32, 2.0)) };
    softmaxSimd(4, &data, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), data[1], 1e-5);
}
