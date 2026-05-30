//! AWQ (Activation-Aware Weight Quantization) INT4 GEMV kernel.
//!
//! AWQ format: INT4 weights packed 8 per u32, COLUMN-major layout.
//! qweight: [k, n/8] — 8 output channels packed per word at same input position
//! scales:  [k/group_size, n] — FP16 per group per output channel
//! qzeros:  [k/group_size, n/8] — packed INT4 zero-points
//!
//! Dequantization: w = (nibble - zero) * scale
//!
//! Key difference from GPTQ: GPTQ packs 8 INPUT elements per word (row-major),
//! AWQ packs 8 OUTPUT elements per word (column-major).

const std = @import("std");

const nibbles_per_u32: usize = 8;

inline fn f16ToF32(v: u16) f32 {
    return @floatCast(@as(f16, @bitCast(v)));
}

/// AWQ INT4 GEMV: y[n] = dequant(W_awq[k,n]) @ x[k]
/// Column-major packed: each u32 word contains 8 output channels at same input pos.
pub fn awqGemv(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    n: usize,
    k: usize,
    group_size: u32,
) void {
    awqGemvRows(x, qweight, scales, qzeros, y, 0, n, k, group_size);
}

pub fn awqGemvRows(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    start_col: usize,
    n_cols: usize,
    k: usize,
    group_size: u32,
) void {
    const gs: usize = group_size;
    const n_groups = (k + gs - 1) / gs;
    const n_words = n_cols / nibbles_per_u32;
    _ = n_groups;

    // Zero output
    @memset(y[start_col .. start_col + n_cols], 0);

    // AWQ column-major: iterate over input positions (k), unpack 8 outputs per word
    for (0..k) |ki| {
        const xv = x[ki];
        if (@abs(xv) < 0.005) continue; // Sparse skip

        const g = ki / gs;

        // Process 8 output channels per word
        for (0..n_words) |wi| {
            const out_base = start_col + wi * nibbles_per_u32;
            const word = qweight[ki * n_words + wi];

            // Zero-point word for this group
            const z_word = qzeros[g * n_words + wi];

            inline for (0..8) |ni| {
                const nibble: u4 = @truncate(word >> @as(u5, @intCast(ni * 4)));
                const zero: u4 = @truncate(z_word >> @as(u5, @intCast(ni * 4)));
                const scale = f16ToF32(scales[g * n_cols + out_base + ni]);
                const dequant = (@as(f32, @floatFromInt(@as(i8, nibble))) - @as(f32, @floatFromInt(@as(i8, zero)))) * scale;
                y[out_base + ni] += dequant * xv;
            }
        }
    }
}
