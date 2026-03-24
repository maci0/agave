//! CPU GEMM kernel — batched matrix multiplication for prefill.
//! Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T

const std = @import("std");
const DType = @import("../../backend.zig").DType;
const gemv_kernel = @import("gemv.zig");

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// F32 GEMM: Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T
pub fn gemmF32(x: [*]const f32, w: [*]const f32, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
    for (0..n_out) |r| {
        const w_row = w + r * n_in;
        for (0..n_tok) |t| {
            const x_row = x + t * n_in;
            var acc: V8 = v8zero;
            var d: usize = 0;
            while (d + 8 <= n_in) : (d += 8) {
                const xv: V8 = x_row[d..][0..8].*;
                const wv: V8 = w_row[d..][0..8].*;
                acc = @mulAdd(V8, xv, wv, acc);
            }
            var dot = @reduce(.Add, acc);
            while (d < n_in) : (d += 1) dot += x_row[d] * w_row[d];
            y[t * n_out + r] = dot;
        }
    }
}

/// Dispatch GEMM for any quantized weight dtype.
/// Falls back to loop-of-GEMV for quantized formats.
pub fn gemmSeq(x: [*]const f32, w_data: [*]const u8, dtype: DType, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
    if (n_tok == 1) {
        gemv_kernel.gemvSeq(x, w_data, dtype, y, n_out, n_in);
        return;
    }
    switch (dtype) {
        .f32 => gemmF32(x, @ptrCast(@alignCast(w_data)), y, n_tok, n_out, n_in),
        else => {
            for (0..n_tok) |t| {
                gemv_kernel.gemvSeq(x + t * n_in, w_data, dtype, y + t * n_out, n_out, n_in);
            }
        },
    }
}

test "gemm f32 matches expected output" {
    const x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const w = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    var y: [6]f32 = undefined;
    gemmF32(&x, &w, &y, 2, 3, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), y[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), y[5], 1e-5);
}

test "gemm n_tok=1 matches gemv" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    var y_gemm: [2]f32 = undefined;
    var y_gemv: [2]f32 = undefined;
    gemmSeq(&x, @ptrCast(&w), .f32, &y_gemm, 1, 2, 4);
    gemv_kernel.gemvSeq(&x, @ptrCast(&w), .f32, &y_gemv, 2, 4);
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(y_gemv[i], y_gemm[i], 1e-5);
    }
}
