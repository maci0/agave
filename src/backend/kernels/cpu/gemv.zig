//! CPU GEMV dispatcher for all quantization formats.
//! Each format's kernel lives in its own file for independent iteration and testing.
//! This file provides the unified dispatch interface used by the CPU backend.
//!
//! Activation sparsity: FFN outputs (after SiLU/GELU) have ~40% near-zero values.
//! Block-level skip checks avoid processing weight blocks where input is negligible.

const std = @import("std");
const DType = @import("../../backend.zig").DType;

// ── Activation Sparsity ────────────────────────────────────────
/// Skip GEMV blocks where all input values are below this magnitude.
/// Measured: SiLU outputs have ~40% values below 0.01 (Qwen3.5 9B).
/// Set to 0 to disable sparsity skipping entirely.
pub const sparse_threshold: f32 = 0.005;

/// Check if all elements in a contiguous block are below the sparse threshold.
/// Uses SIMD max-abs reduction for speed (~1 cycle per 8 elements).
/// Returns true if the block can be safely skipped (all near-zero).
pub inline fn isBlockSparse(x: [*]const f32, start: usize, len: usize) bool {
    if (sparse_threshold == 0) return false;
    const V8 = @Vector(8, f32);
    const zero: V8 = @splat(0.0);
    var max_v: V8 = zero;
    var i: usize = start;
    while (i + 8 <= start + len) : (i += 8) {
        const v: V8 = x[i..][0..8].*;
        const abs_v = @select(f32, v > zero, v, zero - v);
        max_v = @max(max_v, abs_v);
    }
    // Handle tail elements
    var tail_max: f32 = 0;
    while (i < start + len) : (i += 1) {
        const a = @abs(x[i]);
        if (a > tail_max) tail_max = a;
    }
    return @max(@reduce(.Max, max_v), tail_max) < sparse_threshold;
}

// ── Per-format kernel imports ────────────────────────────────────
const gemv_q4_0 = @import("gemv_q4_0.zig");
const gemv_q8_0 = @import("gemv_q8_0.zig");
const gemv_q4_k = @import("gemv_q4_k.zig");
const gemv_q5_k = @import("gemv_q5_k.zig");
const gemv_q6_k = @import("gemv_q6_k.zig");
const gemv_f32 = @import("gemv_f32.zig");
const gemv_f16 = @import("gemv_f16.zig");
const gemv_bf16 = @import("gemv_bf16.zig");
const gemv_fp8 = @import("gemv_fp8.zig");
const gemv_fp4 = @import("gemv_fp4.zig");
const gemv_iq4 = @import("gemv_iq4.zig");
const gemv_q_small = @import("gemv_q_small.zig");
const gemv_tq1_0 = @import("gemv_tq1_0.zig");
const gemv_tq2_0 = @import("gemv_tq2_0.zig");

// ── Re-exports for direct access ─────────────────────────────────
/// GEMV kernel for Q4_0 quantization (4-bit, 32 elements per block).
pub const gemvQ4_0 = gemv_q4_0.gemvQ4_0;
/// GEMV kernel for Q8_0 quantization (8-bit, 32 elements per block).
pub const gemvQ8_0 = gemv_q8_0.gemvQ8_0;
/// GEMV kernel for Q4_K quantization (4-bit K-quant, super-block structure).
pub const gemvQ4_K = gemv_q4_k.gemvQ4_K;
/// GEMV kernel for Q5_K quantization (5-bit K-quant, super-block structure).
pub const gemvQ5_K = gemv_q5_k.gemvQ5_K;
/// GEMV kernel for Q6_K quantization (6-bit K-quant, super-block structure).
pub const gemvQ6_K = gemv_q6_k.gemvQ6_K;
/// GEMV kernel for F32 (single-precision float) weights.
pub const gemvF32 = gemv_f32.gemvF32;
/// GEMV kernel for F16 (IEEE half-precision float) weights.
pub const gemvF16 = gemv_f16.gemvF16;
/// GEMV kernel for BF16 (brain floating-point 16) weights.
pub const gemvBF16 = gemv_bf16.gemvBF16;
/// GEMV kernel for FP8 E4M3 (4-bit exponent, 3-bit mantissa) weights.
pub const gemvFP8_E4M3 = gemv_fp8.gemvFP8_E4M3;
/// GEMV kernel for FP8 E5M2 (5-bit exponent, 2-bit mantissa) weights.
pub const gemvFP8_E5M2 = gemv_fp8.gemvFP8_E5M2;
/// GEMV kernel for MXFP4 (microscaling 4-bit float) weights.
pub const gemvMXFP4 = gemv_fp4.gemvMXFP4_V;
/// GEMV kernel for NVFP4 (NVIDIA 4-bit float) weights.
pub const gemvNVFP4 = gemv_fp4.gemvNVFP4;
/// GEMV kernel for IQ4_NL (importance-quantized 4-bit, non-linear) weights.
pub const gemvIQ4_NL = gemv_iq4.gemvIQ4_NL;
/// GEMV kernel for IQ4_XS (importance-quantized 4-bit, extra-small blocks) weights.
pub const gemvIQ4_XS = gemv_iq4.gemvIQ4_XS;
/// GEMV kernel for Q4_1 quantization (4-bit with min offset, 32 elements per block).
pub const gemvQ4_1 = gemv_q_small.gemvQ4_1;
/// GEMV kernel for Q5_0 quantization (5-bit, 32 elements per block).
pub const gemvQ5_0 = gemv_q_small.gemvQ5_0;
/// GEMV kernel for Q2_K quantization (2-bit K-quant, super-block structure).
pub const gemvQ2_K = gemv_q_small.gemvQ2_K;
/// GEMV kernel for Q3_K quantization (3-bit K-quant, super-block structure).
pub const gemvQ3_K = gemv_q_small.gemvQ3_K;

const builtin = @import("builtin");
const build_options = @import("build_options");
const backend_mod = @import("../../backend.zig");
// Accelerate.framework only available when Metal is enabled (they're linked together).
const accelerate = if (builtin.os.tag == .macos and build_options.enable_metal) @import("../../../backend/accelerate.zig") else struct {};

/// Computes the byte stride of one GEMV row for a given dtype and column count.
pub const gemvRowBytes = backend_mod.gemvRowBytes;

/// Sequential GEMV — dispatches to the appropriate quantized kernel.
/// F32 on macOS uses Accelerate.framework (AMX-accelerated) for ~4× speedup.
pub fn gemvSeq(x: [*]const f32, w_data: [*]const u8, dtype: DType, y: [*]f32, n: usize, k: usize) void {
    switch (dtype) {
        .q4_0 => gemvQ4_0(x, w_data, y, n, k),
        .q4_1 => gemvQ4_1(x, w_data, y, n, k),
        .q5_0 => gemvQ5_0(x, w_data, y, n, k),
        .q5_k => gemvQ5_K(x, w_data, y, n, k),
        .q6_k => gemvQ6_K(x, w_data, y, n, k),
        .q8_0 => gemvQ8_0(x, w_data, y, n, k),
        .f16 => gemvF16(x, @ptrCast(@alignCast(w_data)), y, n, k),
        .f32 => if (comptime builtin.os.tag == .macos and build_options.enable_metal)
            accelerate.sgemv(n, k, x, @ptrCast(@alignCast(w_data)), y)
        else
            gemvF32(x, @ptrCast(@alignCast(w_data)), y, n, k),
        .bf16 => gemvBF16(x, w_data, y, n, k),
        .mxfp4 => gemvMXFP4(x, w_data, y, n, k),
        .q2_k => gemvQ2_K(x, w_data, y, n, k),
        .q3_k => gemvQ3_K(x, w_data, y, n, k),
        .q4_k => gemvQ4_K(x, w_data, y, n, k),
        .iq4_nl => gemvIQ4_NL(x, w_data, y, n, k),
        .iq4_xs => gemvIQ4_XS(x, w_data, y, n, k),
        .fp8_e4m3 => gemvFP8_E4M3(x, w_data, y, n, k),
        .fp8_e5m2 => gemvFP8_E5M2(x, w_data, y, n, k),
        .nvfp4 => gemvNVFP4(x, w_data, y, n, k),
        .tq1_0 => gemv_tq1_0.gemvTQ1_0(x, w_data, y, n, k),
        .tq2_0 => gemv_tq2_0.gemvTQ2_0(x, w_data, y, n, k),
        .iq3_xxs => @import("gemv_iq_small.zig").gemvIQ3_XXS(x, w_data, y, n, k),
        .iq3_s => @import("gemv_iq_small.zig").gemvIQ3_S(x, w_data, y, n, k),
        .iq2_xxs => @import("gemv_iq_small.zig").gemvIQ2_XXS(x, w_data, y, n, k),
        .iq2_xs => @import("gemv_iq_small.zig").gemvIQ2_XS(x, w_data, y, n, k),
        .iq2_s => @import("gemv_iq_small.zig").gemvIQ2_S(x, w_data, y, n, k),
        .iq1_s => @import("gemv_iq_small.zig").gemvIQ1_S(x, w_data, y, n, k),
        .iq1_m => @import("gemv_iq_small.zig").gemvIQ1_M(x, w_data, y, n, k),
        .mlx_q, .gptq, .awq, .hqq, .unknown => {
            std.log.warn("GEMV: unsupported dtype {s}, output zeroed", .{@tagName(dtype)});
            @memset(y[0..n], 0);
        },
    }
}

// ── Tests ────────────────────────────────────────────────────────

test "isBlockSparse all zeros" {
    var x = [_]f32{0.0} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
}

test "isBlockSparse below threshold" {
    var x = [_]f32{0.001} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
}

test "isBlockSparse above threshold" {
    var x = [_]f32{0.001} ** 32;
    x[16] = 0.01;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "isBlockSparse negative values" {
    var x = [_]f32{-0.004} ** 32;
    try std.testing.expect(isBlockSparse(&x, 0, 32));
    x[0] = -0.006;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "isBlockSparse partial block" {
    var x = [_]f32{0.0} ** 13;
    try std.testing.expect(isBlockSparse(&x, 0, 13));
    x[12] = 1.0;
    try std.testing.expect(!isBlockSparse(&x, 0, 13));
}

test "isBlockSparse offset" {
    var x = [_]f32{1.0} ** 64;
    // Fill second half with zeros
    for (32..64) |i| x[i] = 0.0;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
    try std.testing.expect(isBlockSparse(&x, 32, 32));
}

test "isBlockSparse disabled when threshold is zero" {
    if (sparse_threshold != 0) return error.SkipZigTest;
    var x = [_]f32{0.0} ** 32;
    try std.testing.expect(!isBlockSparse(&x, 0, 32));
}

test "gemvSeq f32 identity" {
    // 2 rows, k=4: W = [[1,0,0,0],[0,1,0,0]], x = [3,7,0,0]
    // y[0] = 3.0, y[1] = 7.0
    const w = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const x = [_]f32{ 3.0, 7.0, 0.0, 0.0 };
    var y: [2]f32 = undefined;
    gemvSeq(&x, @ptrCast(&w), .f32, &y, 2, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), y[1], 1e-5);
}

test "gemvSeq q8_0 roundtrip" {
    // Q8_0: f16 scale + 32 i8 quants = 34 bytes. dequant: val = quant * scale.
    var block: [34]u8 align(2) = undefined;
    @as(*f16, @ptrCast(@alignCast(&block[0]))).* = @as(f16, 0.5);
    @memset(block[2..34], 2); // quant=2, dequant = 2 * 0.5 = 1.0

    const x = [_]f32{1.0} ** 32;
    var y: [1]f32 = undefined;
    gemvSeq(&x, &block, .q8_0, &y, 1, 32);
    // 32 × (2 × 0.5) × 1.0 = 32.0
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), y[0], 0.5);
}

test "fuzz: all gemv functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // k=256 satisfies all quant block alignments (32-elem and 256-elem super-blocks).
            const K = 256;
            const N = 1;

            // Random input vector, clamped to finite values.
            var x: [K]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @bitCast(smith.valueWithHash(u32, @intCast(i)));
            for (&x) |*v| {
                if (!std.math.isFinite(v.*)) v.* = 0.0;
            }

            // Weight buffer — 4096 bytes covers any single-row format at K=256.
            var w_buf: [4096]u8 align(16) = undefined;
            for (&w_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(i + K));

            var y: [N]f32 = undefined;

            _ = sparse_threshold;
            _ = isBlockSparse(&x, 0, K);

            const rb = gemvRowBytes(.f32, K);
            std.debug.assert(rb == K * 4);

            var w_f32: [K]f32 = undefined;
            for (&w_f32, 0..) |*v, i| v.* = @bitCast(smith.valueWithHash(u32, @intCast(i + 2 * K)));
            for (&w_f32) |*v| {
                if (!std.math.isFinite(v.*)) v.* = 0.0;
            }
            gemvF32(&x, &w_f32, &y, N, K);

            var w_f16: [K]f16 align(2) = undefined;
            for (&w_f16, 0..) |*v, i| {
                v.* = @bitCast(smith.valueWithHash(u16, @intCast(i + 3 * K)));
                if (!std.math.isFinite(@as(f32, v.*))) v.* = 0.0;
            }
            gemvF16(&x, &w_f16, &y, N, K);

            gemvBF16(&x, &w_buf, &y, N, K);
            gemvQ4_0(&x, &w_buf, &y, N, K);
            gemvQ8_0(&x, &w_buf, &y, N, K);
            gemvQ4_K(&x, &w_buf, &y, N, K);
            gemvQ5_K(&x, &w_buf, &y, N, K);
            gemvQ6_K(&x, &w_buf, &y, N, K);
            gemvQ4_1(&x, &w_buf, &y, N, K);
            gemvQ5_0(&x, &w_buf, &y, N, K);
            gemvQ2_K(&x, &w_buf, &y, N, K);
            gemvQ3_K(&x, &w_buf, &y, N, K);
            gemvFP8_E4M3(&x, &w_buf, &y, N, K);
            gemvFP8_E5M2(&x, &w_buf, &y, N, K);
            gemvMXFP4(&x, &w_buf, &y, N, K);
            gemvNVFP4(&x, &w_buf, &y, N, K);
            gemvIQ4_NL(&x, &w_buf, &y, N, K);
            gemvIQ4_XS(&x, &w_buf, &y, N, K);

            gemvSeq(&x, &w_buf, .q4_0, &y, N, K);
            gemvSeq(&x, &w_buf, .q4_1, &y, N, K);
            gemvSeq(&x, &w_buf, .q5_0, &y, N, K);
            gemvSeq(&x, &w_buf, .q5_k, &y, N, K);
            gemvSeq(&x, &w_buf, .q6_k, &y, N, K);
            gemvSeq(&x, &w_buf, .q8_0, &y, N, K);
            gemvSeq(&x, @ptrCast(@alignCast(&w_f16)), .f16, &y, N, K);
            gemvSeq(&x, @ptrCast(&w_f32), .f32, &y, N, K);
            gemvSeq(&x, &w_buf, .bf16, &y, N, K);
            gemvSeq(&x, &w_buf, .mxfp4, &y, N, K);
            gemvSeq(&x, &w_buf, .q2_k, &y, N, K);
            gemvSeq(&x, &w_buf, .q3_k, &y, N, K);
            gemvSeq(&x, &w_buf, .q4_k, &y, N, K);
            gemvSeq(&x, &w_buf, .iq4_nl, &y, N, K);
            gemvSeq(&x, &w_buf, .iq4_xs, &y, N, K);
            gemvSeq(&x, &w_buf, .fp8_e4m3, &y, N, K);
            gemvSeq(&x, &w_buf, .fp8_e5m2, &y, N, K);
            gemvSeq(&x, &w_buf, .nvfp4, &y, N, K);
            gemvSeq(&x, &w_buf, .tq1_0, &y, N, K);
            gemvSeq(&x, &w_buf, .tq2_0, &y, N, K);
            gemvSeq(&x, &w_buf, .mlx_q, &y, N, K);
            gemvSeq(&x, &w_buf, .gptq, &y, N, K);
            gemvSeq(&x, &w_buf, .awq, &y, N, K);
            gemvSeq(&x, &w_buf, .unknown, &y, N, K);
        }
    }.f, .{});
}
