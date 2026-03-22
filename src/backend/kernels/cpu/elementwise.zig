//! CPU element-wise arithmetic kernels.

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
        try std.testing.expectApproxEqAbs(orig * 0.5, data[i], 0.001);
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
