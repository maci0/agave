//! CPU RMS normalization and L2 normalization kernels.

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Applies Root Mean Square Layer Normalization: output[i] = input[i] * weight[i] / rms(input).
pub fn rmsNorm(input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
    var acc: V8 = v8zero;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const v: V8 = input[i..][0..8].*;
        acc = @mulAdd(V8, v, v, acc);
    }
    var ss = @reduce(.Add, acc);
    while (i < n) : (i += 1) ss += input[i] * input[i];
    const inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    const inv_v: V8 = @splat(inv);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const vi: V8 = input[i..][0..8].*;
        const vw: V8 = weight[i..][0..8].*;
        output[i..][0..8].* = vi * inv_v * vw;
    }
    while (i < n) : (i += 1) output[i] = input[i] * inv * weight[i];
}

/// Fused add + rms_norm: a[i] = a[i] + b[i], output = rms_norm(a+b, weight, eps).
/// Two passes instead of three (no separate add pass) — avoids intermediate write.
pub fn addRmsNorm(a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
    // Pass 1: compute a = a + b, accumulate sum of squares
    var acc: V8 = v8zero;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const va: V8 = a[i..][0..8].*;
        const vb: V8 = b[i..][0..8].*;
        const vs = va + vb;
        a[i..][0..8].* = vs;
        acc = @mulAdd(V8, vs, vs, acc);
    }
    var ss = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const v = a[i] + b[i];
        a[i] = v;
        ss += v * v;
    }
    // Pass 2: normalize
    const inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    const inv_v: V8 = @splat(inv);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const vi: V8 = a[i..][0..8].*;
        const vw: V8 = weight[i..][0..8].*;
        output[i..][0..8].* = vi * inv_v * vw;
    }
    while (i < n) : (i += 1) output[i] = a[i] * inv * weight[i];
}

/// L2 normalizes a vector in-place: x[i] /= sqrt(sum(x^2) + eps).
pub fn l2Norm(x: [*]f32, n: usize, eps: f32) void {
    var acc: V8 = v8zero;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const v: V8 = x[i..][0..8].*;
        acc = @mulAdd(V8, v, v, acc);
    }
    var ss = @reduce(.Add, acc);
    while (i < n) : (i += 1) ss += x[i] * x[i];
    const inv: f32 = 1.0 / @sqrt(ss + eps);
    const inv_v: V8 = @splat(inv);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        x[i..][0..8].* = @as(V8, x[i..][0..8].*) * inv_v;
    }
    while (i < n) : (i += 1) x[i] *= inv;
}

const std = @import("std");

test "rmsNorm unit weight" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    rmsNorm(&input, &weight, &output, 8, 1e-6);
    // RMS = sqrt((1+4+9+16+25+36+49+64)/8) = sqrt(204/8) ≈ 5.0498
    // output[i] = input[i] / RMS * weight[i]
    const rms = @sqrt(@as(f32, 204.0 / 8.0));
    for (0..8) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) / rms;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "rmsNorm with non-unit weight" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var weight = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    var output: [8]f32 = undefined;
    rmsNorm(&input, &weight, &output, 8, 1e-6);
    // With weight=2, output should be 2x the unit-weight result
    const rms = @sqrt(@as(f32, 204.0 / 8.0));
    for (0..8) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) / rms * 2.0;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "addRmsNorm fused" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    addRmsNorm(&a, &b, &weight, &output, 8, 1e-6);
    // After add: a = [2,3,4,5,6,7,8,9], RMS = sqrt((4+9+16+25+36+49+64+81)/8) = sqrt(284/8)
    const rms = @sqrt(@as(f32, 284.0 / 8.0));
    for (0..8) |i| {
        const sum_val = @as(f32, @floatFromInt(i + 2));
        try std.testing.expectApproxEqAbs(sum_val / rms, output[i], 1e-4);
    }
    // Verify a was modified in-place (a = a + b)
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i + 2)), a[i], 1e-6);
    }
}

test "rmsNorm near-zero input uses epsilon" {
    // With very small inputs, epsilon prevents division by zero / NaN
    var input = [_]f32{ 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [8]f32 = undefined;
    rmsNorm(&input, &weight, &output, 8, 1e-6);
    // With eps=1e-6, RMS ≈ sqrt(eps) ≈ 1e-3, output ≈ 1e-20 / 1e-3 = 1e-17
    for (0..8) |i| {
        try std.testing.expect(!std.math.isNan(output[i]));
        try std.testing.expect(!std.math.isInf(output[i]));
        try std.testing.expectApproxEqAbs(@as(f32, 1e-17), output[i], 1e-18);
    }
}

test "addRmsNorm matches separate add + rmsNorm" {
    // Verify the fused kernel produces the same result as separate add + rmsNorm
    var a1 = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    var a2 = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    var b = [_]f32{ 0.5, 1.5, -0.5, 2.0, -1.0, 0.0, 1.0, -0.5 };
    var weight = [_]f32{ 1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 0.5, 1.5 };
    const eps: f32 = 1e-6;

    // Reference: separate add then rmsNorm
    const elementwise_kernel = @import("elementwise.zig");
    elementwise_kernel.add(&a1, &b, &a1, 8); // a1 = a1 + b
    var ref_output: [8]f32 = undefined;
    rmsNorm(&a1, &weight, &ref_output, 8, eps);

    // Fused: addRmsNorm does both in one call
    var fused_output: [8]f32 = undefined;
    addRmsNorm(&a2, &b, &weight, &fused_output, 8, eps);

    // Results should match
    for (0..8) |i| try std.testing.expectApproxEqAbs(ref_output[i], fused_output[i], 1e-5);
    // a2 should also be modified (a2 = a2 + b)
    for (0..8) |i| try std.testing.expectApproxEqAbs(a1[i], a2[i], 1e-6);
}

test "addRmsNorm non-aligned size" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var output: [5]f32 = undefined;
    addRmsNorm(&a, &b, &weight, &output, 5, 1e-6);
    // After add: a = [2,3,4,5,6], sum_sq = 4+9+16+25+36 = 90
    // RMS = sqrt(90/5) = sqrt(18)
    const rms = @sqrt(@as(f32, 90.0 / 5.0));
    for (0..5) |i| {
        const expected = @as(f32, @floatFromInt(i + 2)) / rms;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "l2Norm produces unit vector" {
    var x = [_]f32{ 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    l2Norm(&x, 8, 1e-12);
    // L2 norm of [3,4,0...] = 5, so x[0] = 3/5, x[1] = 4/5
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), x[1], 1e-5);
    for (2..8) |i| try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[i], 1e-5);
    // Verify unit norm
    var ss: f32 = 0;
    for (&x) |v| ss += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ss, 1e-5);
}
