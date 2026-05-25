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
extern "c" fn vDSP_dotpr(a: [*]const f32, ia: c_int, b: [*]const f32, ib: c_int, c_ptr: *f32, n: c_int) void;

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
