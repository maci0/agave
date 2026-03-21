//! GEMV FP8 E4M3 kernel: y[row] = dot(W_fp8[row,:], x)
//! FP8 E4M3: 1 byte per element.
//! Uses 256-entry comptime LUT for branch-free dequantization.

const cu = @import("common.zig");

/// Precomputed FP8 E4M3 → f32 lookup table (built at comptime on host).
/// Embedded into the kernel binary — no runtime computation.
const fp8e4m3_lut = blk: {
    @setEvalBranchQuota(10000);
    var table: [256]f32 = undefined;
    for (0..256) |i| {
        table[i] = fp8e4m3Compute(@intCast(i));
    }
    break :blk table;
};

/// Compute FP8 E4M3 → f32 (comptime helper for LUT generation).
fn fp8e4m3Compute(val: u8) f32 {
    const sign: u32 = @as(u32, val >> 7) << 31;
    const exp: u32 = (val >> 3) & 0x0F;
    const mant: u32 = val & 0x07;

    // NaN: e=15, m=7
    if (exp == 0x0F and mant == 0x07) {
        return @bitCast(sign | 0x7FC00000);
    }

    // Zero
    if (exp == 0 and mant == 0) {
        return @bitCast(sign);
    }

    // Denormal: 2^(-6) / 8 = 2^(-9)
    if (exp == 0) {
        const fmant: f32 = @floatFromInt(mant);
        const val_abs: f32 = fmant * (1.0 / 512.0);
        return @bitCast(sign | @as(u32, @bitCast(val_abs)));
    }

    // Normal: (-1)^s * 2^(e-7) * (1 + m/8)
    const exp_f32: u32 = (exp + 120) << 23;
    const mant_f32: u32 = mant << 20;
    return @bitCast(sign | exp_f32 | mant_f32);
}

export fn gemv_fp8_e4m3_kernel(x: [*]const f32, w: [*]const u8, y: [*]f32, n: u32, k: u32) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const row_bytes = k;
    const row_start = @intFromPtr(w) + row * row_bytes;

    var sum: f32 = 0.0;

    // Distribute k elements across threads
    var col = tid;
    while (col < k) : (col += bdim) {
        const fp8_val = @as(*const u8, @ptrFromInt(row_start + col)).*;
        const f32_val = fp8e4m3_lut[fp8_val];
        sum += f32_val * x[col];
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
