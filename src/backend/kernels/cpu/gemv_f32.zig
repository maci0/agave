//! CPU GEMV kernel for F32 weights.
//! 4-row batching with V8 SIMD.

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// F32 GEMV: y = W @ x. 4-row batched with V8 SIMD.
pub fn gemvF32(x: [*]const f32, w: [*]const f32, y: [*]f32, n: usize, k: usize) void {
    var row: usize = 0;
    while (row + 4 <= n) : (row += 4) {
        var acc0: V8 = v8zero;
        var acc1: V8 = v8zero;
        var acc2: V8 = v8zero;
        var acc3: V8 = v8zero;
        const r0 = row * k;
        const r1 = r0 + k;
        const r2 = r1 + k;
        const r3 = r2 + k;
        var i: usize = 0;
        while (i + 8 <= k) : (i += 8) {
            const xv: V8 = x[i..][0..8].*;
            acc0 += xv * @as(V8, w[r0 + i ..][0..8].*);
            acc1 += xv * @as(V8, w[r1 + i ..][0..8].*);
            acc2 += xv * @as(V8, w[r2 + i ..][0..8].*);
            acc3 += xv * @as(V8, w[r3 + i ..][0..8].*);
        }
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        while (i < k) : (i += 1) {
            const xv = x[i];
            t0 += xv * w[r0 + i];
            t1 += xv * w[r1 + i];
            t2 += xv * w[r2 + i];
            t3 += xv * w[r3 + i];
        }
        y[row] = @reduce(.Add, acc0) + t0;
        y[row + 1] = @reduce(.Add, acc1) + t1;
        y[row + 2] = @reduce(.Add, acc2) + t2;
        y[row + 3] = @reduce(.Add, acc3) + t3;
    }
    while (row < n) : (row += 1) {
        var acc: V8 = v8zero;
        var tail: f32 = 0.0;
        const roff = row * k;
        var i: usize = 0;
        while (i + 8 <= k) : (i += 8) {
            acc += @as(V8, x[i..][0..8].*) * @as(V8, w[roff + i ..][0..8].*);
        }
        while (i < k) : (i += 1) tail += x[i] * w[roff + i];
        y[row] = @reduce(.Add, acc) + tail;
    }
}

const std = @import("std");

test "gemvF32 identity" {
    // W = I (identity), so y = x.
    var w: [16]f32 = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvF32(&x, &w, &y, 4, 4);
    for (0..4) |i| try std.testing.expectApproxEqAbs(x[i], y[i], 1e-6);
}

test "gemvF32 scaling" {
    // W = 2*I → y = 2*x.
    var w: [16]f32 = .{
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2,
    };
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvF32(&x, &w, &y, 4, 4);
    const expected = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    for (0..4) |i| try std.testing.expectApproxEqAbs(expected[i], y[i], 1e-6);
}

test "gemvF32 dense matrix" {
    // Dense W and non-uniform x to catch stride and accumulation bugs.
    // W[4×4] row-major, x[4] → y[4].
    var w: [16]f32 = .{
        1.0, 0.5, 2.0, 0.3,
        3.0, 1.0, 0.5, 0.7,
        0.5, 2.0, 1.0, 1.5,
        0.2, 0.8, 1.3, 2.1,
    };
    var x = [_]f32{ 2.0, 3.0, 1.0, 4.0 };
    var y: [4]f32 = undefined;
    gemvF32(&x, &w, &y, 4, 4);
    // y[0] = 2×1 + 3×0.5 + 1×2 + 4×0.3 = 6.7
    // y[1] = 2×3 + 3×1 + 1×0.5 + 4×0.7 = 12.3
    // y[2] = 2×0.5 + 3×2 + 1×1 + 4×1.5 = 14.0
    // y[3] = 2×0.2 + 3×0.8 + 1×1.3 + 4×2.1 = 12.5
    try std.testing.expectApproxEqAbs(@as(f32, 6.7), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.3), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), y[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), y[3], 1e-5);
}

test "gemvF32 non-aligned k exercises scalar cleanup" {
    // k=5 (not a multiple of 8) exercises the scalar tail loop.
    // n=3 (not a multiple of 4) exercises the single-row tail.
    // W is 3×5, x=[1,1,1,1,1] → each output row = sum of that row's weights.
    var w: [15]f32 = .{
        1.0, 2.0, 3.0, 4.0, 5.0,
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.0, 0.0, 0.0, 0.0, 10.0,
    };
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    var y: [3]f32 = undefined;
    gemvF32(&x, &w, &y, 3, 5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[2], 1e-5);
}
