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

const builtin = @import("builtin");
const backend_mod = @import("../../backend.zig");
const accelerate = if (builtin.os.tag == .macos) @import("../../../backend/accelerate.zig") else struct {};

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
        .f32 => if (comptime builtin.os.tag == .macos)
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
        .tq1_0, .mlx_q, .gptq, .unknown => {
            std.log.warn("GEMV: unsupported dtype {s}, output zeroed", .{@tagName(dtype)});
            @memset(y[0..n], 0);
        },
    }
}
