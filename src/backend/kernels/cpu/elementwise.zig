//! CPU element-wise kernels: arithmetic, gating, and layout transforms.

const V8 = @Vector(8, f32);

/// Element-wise addition: out[i] = a[i] + b[i].
pub fn add(a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        out[i..][0..8].* = @as(V8, a[i..][0..8].*) + @as(V8, b[i..][0..8].*);
    }
    while (i < n) : (i += 1) out[i] = a[i] + b[i];
}

/// Element-wise multiplication: out[i] = a[i] * b[i].
pub fn mul(a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        out[i..][0..8].* = @as(V8, a[i..][0..8].*) * @as(V8, b[i..][0..8].*);
    }
    while (i < n) : (i += 1) out[i] = a[i] * b[i];
}

/// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
pub fn sigmoidMul(data: [*]f32, gate: [*]const f32, n: usize) void {
    const one: V8 = @splat(@as(f32, 1.0));
    const neg: V8 = @splat(@as(f32, -1.0));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const g: V8 = gate[i..][0..8].*;
        const sig = one / (one + @exp(neg * g));
        data[i..][0..8].* = @as(V8, data[i..][0..8].*) * sig;
    }
    while (i < n) : (i += 1) {
        const g = gate[i];
        data[i] *= 1.0 / (1.0 + @exp(-g));
    }
}

/// De-interleave paired blocks on CPU.
pub fn deinterleave(input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
    for (0..n_pairs) |h| {
        @memcpy(out_a[h * stride ..][0..stride], input[h * 2 * stride ..][0..stride]);
        @memcpy(out_b[h * stride ..][0..stride], input[h * 2 * stride + stride ..][0..stride]);
    }
}

const std = @import("std");

test "add basic" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    var out: [8]f32 = undefined;
    add(&a, &b, &out, 8);
    for (0..8) |i| try std.testing.expectApproxEqAbs(@as(f32, 9.0), out[i], 1e-6);
}

test "mul basic" {
    var a = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var out: [8]f32 = undefined;
    mul(&a, &b, &out, 8);
    const expected = [_]f32{ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5 };
    for (0..8) |i| try std.testing.expectApproxEqAbs(expected[i], out[i], 1e-6);
}

test "sigmoidMul" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var gate = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    sigmoidMul(&data, &gate, 8);
    // sigmoid(0) = 0.5, so data[i] *= 0.5
    for (0..8) |i| {
        const orig: f32 = @floatFromInt(i + 1);
        try std.testing.expectApproxEqAbs(orig * 0.5, data[i], 1e-5);
    }
}

test "deinterleave" {
    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // [A0, B0] with stride=2, 1 pair
    var out_a: [2]f32 = undefined;
    var out_b: [2]f32 = undefined;
    deinterleave(&input, &out_a, &out_b, 2, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out_b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out_b[1], 1e-6);
}

test "add non-aligned exercises scalar tail" {
    // n=5: SIMD processes 0, scalar tail processes 5 elements
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var out: [5]f32 = undefined;
    add(&a, &b, &out, 5);
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0, 55.0 };
    for (0..5) |i| try std.testing.expectApproxEqAbs(expected[i], out[i], 1e-6);
}

test "mul non-aligned exercises scalar tail" {
    var a = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0 };
    var b = [_]f32{ 0.5, 2.0, -1.0, 0.1, 3.0 };
    var out: [5]f32 = undefined;
    mul(&a, &b, &out, 5);
    const expected = [_]f32{ 1.0, 6.0, -4.0, 0.5, 18.0 };
    for (0..5) |i| try std.testing.expectApproxEqAbs(expected[i], out[i], 1e-6);
}

test "sigmoidMul with non-zero gate" {
    // Test with positive/negative gate values, not just zero
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var gate = [_]f32{ 10.0, -10.0, 1.0, -1.0, 0.0, 5.0, -5.0, 2.0 };
    sigmoidMul(&data, &gate, 8);
    // sigmoid(10) ≈ 1.0, sigmoid(-10) ≈ 0.0, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-4); // 1.0 * sigmoid(10) ≈ 0.99995
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 1e-4); // 2.0 * sigmoid(-10) ≈ 0.00009
    try std.testing.expectApproxEqAbs(@as(f32, 2.1932), data[2], 1e-3); // 3.0 * sigmoid(1) ≈ 2.1932
    try std.testing.expectApproxEqAbs(@as(f32, 1.0757), data[3], 1e-3); // 4.0 * sigmoid(-1) ≈ 1.0757
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), data[4], 1e-5); // 5.0 * sigmoid(0) = 2.5
}

test "sigmoidMul non-aligned exercises scalar tail" {
    // n=5 with SIMD width 8: 0 SIMD iterations, 5 scalar iterations
    var data = [_]f32{ 2.0, 4.0, 6.0, 8.0, 10.0 };
    var gate = [_]f32{ 0.0, 10.0, -10.0, 1.0, -1.0 };
    sigmoidMul(&data, &gate, 5);
    // sigmoid(0)=0.5, sigmoid(10)≈1.0, sigmoid(-10)≈0.0, sigmoid(1)≈0.7311, sigmoid(-1)≈0.2689
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5); // 2.0 * 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), data[1], 1e-3); // 4.0 * ~1.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-3); // 6.0 * ~0.0
    try std.testing.expectApproxEqAbs(@as(f32, 5.849), data[3], 1e-2); // 8.0 * sigmoid(1)
    try std.testing.expectApproxEqAbs(@as(f32, 2.689), data[4], 1e-2); // 10.0 * sigmoid(-1)
}

test "deinterleave multi-pair" {
    // 2 pairs, stride=2: [A0a, A0b, B0a, B0b, A1a, A1b, B1a, B1b]
    var input = [_]f32{ 10, 11, 20, 21, 30, 31, 40, 41 };
    var out_a: [4]f32 = undefined;
    var out_b: [4]f32 = undefined;
    deinterleave(&input, &out_a, &out_b, 2, 2);
    // out_a = [10, 11, 30, 31], out_b = [20, 21, 40, 41]
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out_a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), out_a[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), out_a[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), out_b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), out_b[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), out_b[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 41.0), out_b[3], 1e-6);
}
