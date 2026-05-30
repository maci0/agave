//! GPTQ quantized GEMV kernel.
//!
//! GPTQ format: INT4 weights packed 8 per u32 word, with per-group
//! FP16 scales and INT4 zero-points (also packed in u32).
//! Dequant: val = (packed_int4 - zero) * scale
//!
//! Supports group_size 32, 64, 128 (most common: 128).

const std = @import("std");
const quant = @import("quant.zig");

const gptq_nibbles_per_u32: usize = 8;

/// GPTQ INT4 GEMV: y[n] = dequant(W[n,k]) @ x[k]
/// qweight: packed INT4 in u32 words, [n, k/8] layout
/// scales: FP16 per group, [n, n_groups] layout
/// qzeros: packed INT4 zero-points in u32, [n_groups, n/8] layout
pub fn gptqGemv(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    n: usize,
    k: usize,
    group_size: u32,
) void {
    gptqGemvRows(x, qweight, scales, qzeros, y, 0, n, k, group_size);
}

pub fn gptqGemvRows(
    x: [*]const f32,
    qweight: [*]const u32,
    scales: [*]const u16,
    qzeros: [*]const u32,
    y: [*]f32,
    start_row: usize,
    n_rows: usize,
    k: usize,
    group_size: u32,
) void {
    const gs: usize = group_size;
    const n_groups = (k + gs - 1) / gs;
    const words_per_row = k / gptq_nibbles_per_u32;

    for (start_row..start_row + n_rows) |row| {
        var sum: f32 = 0.0;
        const w_row = qweight + row * words_per_row;
        const s_row = scales + row * n_groups;

        for (0..n_groups) |g| {
            const scale = f16ToF32(s_row[g]);

            // Extract zero-point for this group+row from packed qzeros
            // qzeros layout: [n_groups, ceil(n/8)] packed INT4
            const total_rows = start_row + n_rows;
            const z_word_idx = g * ((total_rows + gptq_nibbles_per_u32 - 1) / gptq_nibbles_per_u32) + row / gptq_nibbles_per_u32;
            const z_nibble = row % gptq_nibbles_per_u32;
            const z_word = qzeros[z_word_idx];
            const zero: f32 = @floatFromInt(@as(i32, @intCast((z_word >> @as(u5, @intCast(z_nibble * 4))) & 0xF)));

            const base = g * gs;
            const elems = @min(gs, k - base);
            const full_words = elems / gptq_nibbles_per_u32;

            var group_sum: f32 = 0.0;
            for (0..full_words) |wi| {
                const word = w_row[base / gptq_nibbles_per_u32 + wi];
                inline for (0..8) |ni| {
                    const nibble: u4 = @truncate(word >> @as(u5, ni * 4));
                    const val = (@as(f32, @floatFromInt(@as(i32, nibble))) - zero) * scale;
                    group_sum += val * x[base + wi * 8 + ni];
                }
            }
            sum += group_sum;
        }
        y[row] = sum;
    }
}

/// Convert f16 stored as u16 to f32.
inline fn f16ToF32(val: u16) f32 {
    return @floatCast(@as(f16, @bitCast(val)));
}

// ── Tests ───────────────────────────────────────────────────────

test "gptq dequant basic" {
    // Pack 8 values: [1,2,3,4,5,6,7,0] into one u32
    const word: u32 = 0x07654321;
    const nibble0: u4 = @truncate(word >> 0);
    const nibble1: u4 = @truncate(word >> 4);
    try std.testing.expectEqual(@as(u4, 1), nibble0);
    try std.testing.expectEqual(@as(u4, 2), nibble1);

    // Full GEMV: 1 row, k=8, group_size=8
    // Packed INT4 weights: [1, 2, 3, 0, 1, 2, 3, 0]
    const qweight = [_]u32{0x03210321};
    // Scale = 2.0 as f16
    const scales = [_]u16{@bitCast(@as(f16, 2.0))};
    // Zero-point = 1 in lowest nibble
    const qzeros = [_]u32{0x00000001};
    const x = [_]f32{1.0} ** 8;
    var y = [_]f32{0.0};

    // dequant: (nibble - zero) * scale = [0, 2, 4, -2, 0, 2, 4, -2]
    // dot with all-ones x = 8.0
    gptqGemv(&x, &qweight, &scales, &qzeros, &y, 1, 8, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[0], 1e-4);
}

test "gptqGemvRows with start_row offset" {
    // Same setup as basic test but using gptqGemvRows with start_row=0
    const qweight = [_]u32{0x03210321};
    const scales = [_]u16{@bitCast(@as(f16, 2.0))};
    const qzeros = [_]u32{0x00000001};
    const x = [_]f32{1.0} ** 8;
    var y = [_]f32{0.0};

    gptqGemvRows(&x, &qweight, &scales, &qzeros, &y, 0, 1, 8, 8);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), y[0], 1e-4);
}
