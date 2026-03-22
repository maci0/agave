//! CPU GEMV dispatcher for all quantization formats.
//! Each format's kernel lives in its own file for independent iteration and testing.
//! This file provides the unified dispatch interface used by the CPU backend.

const std = @import("std");
const DType = @import("../../backend.zig").DType;

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

// ── Re-exports for direct access ─────────────────────────────────
pub const gemvQ4_0 = gemv_q4_0.gemvQ4_0;
pub const gemvQ8_0 = gemv_q8_0.gemvQ8_0;
pub const gemvQ4_K = gemv_q4_k.gemvQ4_K;
pub const gemvQ5_K = gemv_q5_k.gemvQ5_K;
pub const gemvQ6_K = gemv_q6_k.gemvQ6_K;
pub const gemvF32 = gemv_f32.gemvF32;
pub const gemvF16 = gemv_f16.gemvF16;
pub const gemvBF16 = gemv_bf16.gemvBF16;
pub const gemvFP8_E4M3 = gemv_fp8.gemvFP8_E4M3;
pub const gemvFP8_E5M2 = gemv_fp8.gemvFP8_E5M2;
pub const gemvMXFP4 = gemv_fp4.gemvMXFP4;
pub const gemvNVFP4 = gemv_fp4.gemvNVFP4;
pub const gemvIQ4_NL = gemv_iq4.gemvIQ4_NL;
pub const gemvIQ4_XS = gemv_iq4.gemvIQ4_XS;
pub const gemvQ4_1 = gemv_q_small.gemvQ4_1;
pub const gemvQ5_0 = gemv_q_small.gemvQ5_0;
pub const gemvQ2_K = gemv_q_small.gemvQ2_K;
pub const gemvQ3_K = gemv_q_small.gemvQ3_K;

/// Elements per small quantization block (Q4_0, Q8_0, etc.).
const quant_block_elems: usize = 32;
/// Elements per large quantization super-block (Q4_K, Q5_K, Q6_K, etc.).
const quant_super_block_elems: usize = 256;
/// Elements per NVFP4 block (8 nibble pairs + 1 scale byte).
const nvfp4_block_elems: usize = 16;

/// Returns the row stride in bytes for a given dtype and column count.
/// Used by parallel GEMV to compute per-row offsets.
pub fn gemvRowBytes(dtype: DType, k: usize) usize {
    const nb = (k + quant_block_elems - 1) / quant_block_elems;
    const nsb = (k + quant_super_block_elems - 1) / quant_super_block_elems;
    return switch (dtype) {
        .q4_0 => nb * 18,
        .q4_1 => nb * 20,
        .q5_0 => nb * 22,
        .q8_0 => nb * 34,
        .q2_k => nsb * 84,
        .q3_k => nsb * 110,
        .q4_k => nsb * 144,
        .q5_k => nsb * 176,
        .q6_k => nsb * 210,
        .iq4_nl => nb * 18,
        .iq4_xs => nsb * 138,
        .mxfp4 => nb * 17,
        .nvfp4 => ((k + nvfp4_block_elems - 1) / nvfp4_block_elems) * 9,
        .f16, .bf16 => k * 2,
        .f32 => k * 4,
        .fp8_e4m3, .fp8_e5m2 => k,
        .tq1_0, .mlx_q, .unknown => 0, // tq1_0/mlx_q: not applicable
    };
}

/// Sequential GEMV — dispatches to the appropriate quantized kernel.
pub fn gemvSeq(x: [*]const f32, w_data: [*]const u8, dtype: DType, y: [*]f32, n: usize, k: usize) void {
    switch (dtype) {
        .q4_0 => gemvQ4_0(x, w_data, y, n, k),
        .q4_1 => gemvQ4_1(x, w_data, y, n, k),
        .q5_0 => gemvQ5_0(x, w_data, y, n, k),
        .q5_k => gemvQ5_K(x, w_data, y, n, k),
        .q6_k => gemvQ6_K(x, w_data, y, n, k),
        .q8_0 => gemvQ8_0(x, w_data, y, n, k),
        .f16 => gemvF16(x, @ptrCast(@alignCast(w_data)), y, n, k),
        .f32 => gemvF32(x, @ptrCast(@alignCast(w_data)), y, n, k),
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
        .tq1_0, .mlx_q, .unknown => {
            std.log.warn("GEMV: unsupported dtype {s}, output zeroed", .{@tagName(dtype)});
            @memset(y[0..n], 0);
        },
    }
}
