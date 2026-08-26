//! Apple Accelerate.framework bindings for AMX-accelerated BLAS operations.
//!
//! Provides thin wrappers around cblas_sgemm and cblas_sdot which internally
//! dispatch to Apple's AMX matrix coprocessor on M1+ hardware. ~4× faster
//! than NEON SIMD for F32 matrix operations.
//!
//! Only compiled on macOS (guarded by comptime os check). Other platforms
//! get no-op stubs that are never called.

const std = @import("std");
const builtin = @import("builtin");

const is_macos = builtin.os.tag == .macos;

// cblas constants (from Accelerate/vecLib/cblas.h)
const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;

// Import Accelerate BLAS functions via C ABI (resolved at link time)
extern "c" fn cblas_sgemm(order: c_int, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: f32, c_ptr: [*]f32, ldc: c_int) void;
extern "c" fn cblas_sdot(n: c_int, x: [*]const f32, incx: c_int, y: [*]const f32, incy: c_int) f32;
extern "c" fn vDSP_dotpr(a: [*]const f32, ia: c_long, b: [*]const f32, ib: c_long, c_ptr: *f32, n: c_ulong) void;

/// Matrix multiply: C[m×n] = A[m×k] × B[k×n] (row-major, B transposed).
/// Uses AMX on M1+ for ~4× speedup over NEON.
/// A is [m, k] row-major, B is [n, k] row-major (transposed: each row of B is an output column).
pub fn sgemm(m: usize, n: usize, k: usize, a: [*]const f32, b: [*]const f32, out: [*]f32) void {
    if (comptime !is_macos) unreachable;
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0,
        a,
        @intCast(k),
        b,
        @intCast(k),
        0.0,
        out,
        @intCast(n),
    );
}

/// C[m×k_out] += scale * A[m×k_inner] @ B[k_inner×k_out] (row-major, no transpose).
/// Used for LoRA delta application: merged += (alpha/rank) * lora_b @ lora_a.
pub fn sgemmAdd(m: usize, k_out: usize, k_inner: usize, scale: f32, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    if (comptime !is_macos) unreachable;
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        @intCast(m),
        @intCast(k_out),
        @intCast(k_inner),
        scale,
        a,
        @intCast(k_inner),
        b,
        @intCast(k_out),
        1.0, // beta=1: add to existing C
        c,
        @intCast(k_out),
    );
}

/// Dot product of two F32 vectors via Accelerate (AMX-accelerated).
pub fn sdot(n: usize, x: [*]const f32, y: [*]const f32) f32 {
    if (comptime !is_macos) unreachable;
    return cblas_sdot(@intCast(n), x, 1, y, 1);
}

/// Dot product via vDSP (alternative path, also AMX-accelerated).
pub fn vdspDot(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    if (comptime !is_macos) unreachable;
    var result: f32 = 0;
    vDSP_dotpr(a, 1, b, 1, &result, @intCast(n));
    return result;
}

/// GEMV: y[n] = W[n×k] × x[k] via sgemm with m=1 (single-row GEMM).
/// Weight matrix W is row-major [n, k]. For GEMV, this is equivalent to
/// sgemm(1, n, k, x, W, y) since we're computing one output vector.
pub fn sgemv(n: usize, k: usize, x: [*]const f32, w: [*]const f32, y: [*]f32) void {
    if (comptime !is_macos) unreachable;
    sgemm(1, n, k, x, w, y);
}

// ── Tests ───────────────────────────────────────────────────────────

test "accelerate, function signatures exist" {
    // Verify all public functions have correct types (compile-time check).
    // These are thin FFI wrappers so we only verify they compile.
    comptime {
        _ = @TypeOf(sgemm);
        _ = @TypeOf(sdot);
        _ = @TypeOf(vdspDot);
        _ = @TypeOf(sgemv);
    }
}

test "accelerate, cblas constants" {
    // Verify CBLAS constants match Apple Accelerate.framework header values.
    try std.testing.expectEqual(@as(c_int, 101), CblasRowMajor);
    try std.testing.expectEqual(@as(c_int, 111), CblasNoTrans);
    try std.testing.expectEqual(@as(c_int, 112), CblasTrans);
}

test "accelerate, sgemm via Accelerate.framework" {
    if (comptime !is_macos) return error.SkipZigTest;

    // 2×3 @ 3×2 = 2×2 (B transposed: B is [n=2, k=3] row-major)
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3]
    const b = [_]f32{ 1, 0, 0, 0, 1, 0 }; // [2, 3] transposed → selects cols
    var c_out: [4]f32 = undefined; // [2, 2]

    sgemm(2, 2, 3, &a, &b, &c_out);

    // Row 0: [1,2,3] dot [1,0,0] = 1, [1,2,3] dot [0,1,0] = 2
    // Row 1: [4,5,6] dot [1,0,0] = 4, [4,5,6] dot [0,1,0] = 5
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), c_out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), c_out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), c_out[3], 1e-6);
}

test "accelerate, sdot" {
    if (comptime !is_macos) return error.SkipZigTest;

    const x = [_]f32{ 1, 2, 3, 4 };
    const y = [_]f32{ 2, 3, 4, 5 };
    const result = sdot(4, &x, &y);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), result, 1e-6);
}

test "accelerate, vdspDot" {
    if (comptime !is_macos) return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 2, 3, 4, 5 };
    const result = vdspDot(&a, &b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), result, 1e-6);
}

test "accelerate, sdot and vdspDot agree" {
    if (comptime !is_macos) return error.SkipZigTest;

    const x = [_]f32{ 0.5, -1.2, 3.7, 0.0, -2.1, 1.0, 0.3, -0.8 };
    const y = [_]f32{ 1.0, 2.0, -0.5, 3.0, 0.0, -1.0, 2.5, 1.5 };
    const dot_result = sdot(8, &x, &y);
    const vdsp_result = vdspDot(&x, &y, 8);
    try std.testing.expectApproxEqAbs(dot_result, vdsp_result, 1e-5);
}

test "accelerate, sgemv via sgemm" {
    if (comptime !is_macos) return error.SkipZigTest;

    // y[3] = W[3,4] @ x[4]
    const x = [_]f32{ 1, 0, 0, 0 };
    const w = [_]f32{
        1, 2, 3, 4, // row 0
        5, 6, 7, 8, // row 1
        9, 10, 11, 12, // row 2
    };
    var y: [3]f32 = undefined;

    sgemv(3, 4, &x, &w, &y);

    // x = [1,0,0,0] selects first column of W
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), y[2], 1e-6);
}

test "fuzz: all accelerate functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            if (comptime !is_macos) return;
            // sgemm
            const a = [_]f32{ 1, 0, 0, 1 };
            const x_in = [_]f32{ 1, 2 };
            var y_out: [2]f32 = undefined;
            sgemm(1, 2, 2, &x_in, &a, &y_out);

            // sdot
            _ = sdot(2, &x_in, &x_in);

            // vdspDot
            _ = vdspDot(&x_in, &x_in, 2);

            // sgemv
            sgemv(2, 2, &x_in, &a, &y_out);
        }
    }.f, .{});
}

test "accelerate, sgemv identity matrix preserves input" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;

    // Identity matrix: y = I @ x should yield y == x.
    // W is [4,4] identity in row-major order.
    const identity = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const x = [_]f32{ 3.14, -2.71, 0.0, 42.0 };
    var y: [4]f32 = undefined;

    sgemv(4, 4, &x, &identity, &y);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x[i], y[i], 1e-6);
    }
}
