//! AWQ (Activation-Aware Weight Quantization) INT4 GEMV kernel.
//!
//! AWQ GEMM format: INT4 weights packed 8 per u32, COLUMN-major layout.
//! qweight: [k, n/8] — 8 output channels packed per word at same input position
//! scales:  [k/group_size, n] — FP16 per group per output channel (natural order)
//! qzeros:  [k/group_size, n/8] — packed INT4 zero-points (GEMM interleaved order)
//!
//! GEMM packing order: nibbles within each u32 follow [0,2,4,6,1,3,5,7] —
//! even output indices in the lower 16 bits, odd indices in the upper 16 bits.
//! Both qweight and qzeros use this interleaved order. Scales use natural order.
//!
//! Dequantization: w = (nibble - zero) * scale

const std = @import("std");

const nibbles_per_u32: usize = 8;

/// GEMM interleaved order: nibble at shift position i*4 maps to output column order_map[i].
const gemm_order: [8]u3 = .{ 0, 2, 4, 6, 1, 3, 5, 7 };

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
    const n_words = n_cols / nibbles_per_u32;

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
            const z_word = qzeros[g * n_words + wi];

            inline for (0..8) |ni| {
                const nibble: u4 = @truncate(word >> @as(u5, @intCast(ni * 4)));
                const zero: u4 = @truncate(z_word >> @as(u5, @intCast(ni * 4)));
                // GEMM order: nibble at shift ni maps to output column gemm_order[ni]
                const out_idx = out_base + gemm_order[ni];
                const scale = f16ToF32(scales[g * n_cols + out_idx]);
                const dequant = (@as(f32, @floatFromInt(@as(i8, nibble))) - @as(f32, @floatFromInt(@as(i8, zero)))) * scale;
                y[out_idx] += dequant * xv;
            }
        }
    }
}

test "awqGemv GEMM order" {
    // Verify GEMM interleaved packing: nibbles [0,2,4,6,1,3,5,7]
    // With x[0]=1.0 and known qweight/qzeros/scales, check reordered output.
    const n: usize = 8;
    const k: usize = 1;
    const gs: u32 = 128;

    // word 0x97585367: nibbles at shifts [0..7]*4 = 7,6,3,5,8,5,7,9
    // GEMM order maps these to output columns [0,2,4,6,1,3,5,7]
    // so: out[0]=nib@0=7, out[2]=nib@1=6, out[4]=nib@2=3, out[6]=nib@3=5
    //     out[1]=nib@4=8, out[3]=nib@5=5, out[5]=nib@6=7, out[7]=nib@7=9
    var qweight = [_]u32{0x97585367};
    var qzeros = [_]u32{0xb6674377};
    // zeros nibbles: 7,7,3,4,7,6,6,11
    // GEMM: z[0]=7, z[2]=7, z[4]=3, z[6]=4, z[1]=7, z[3]=6, z[5]=6, z[7]=11
    var scales = [_]u16{ 6794, 7247, 8252, 7744, 11327, 8329, 8451, 8314 };

    var x = [_]f32{1.0};
    var y = [_]f32{0} ** 8;

    awqGemvRows(&x, &qweight, &scales, &qzeros, &y, 0, n, k, gs);

    // Expected with GEMM reorder:
    // out[0]: (7-7)*sc[0] = 0
    // out[1]: (8-7)*sc[1] = 1 * 0.004208 = 0.004208
    // out[2]: (6-7)*sc[2] = -1 * 0.008270 = -0.008270
    // out[3]: (5-6)*sc[3] = -1 * 0.006104 = -0.006104
    // out[4]: (3-3)*sc[4] = 0
    // out[5]: (7-6)*sc[5] = 1 * 0.008858 = 0.008858
    // out[6]: (5-4)*sc[6] = 1 * 0.009789 = 0.009789
    // out[7]: (9-11)*sc[7] = -2 * 0.008743 = -0.017487
    const expected = [_]f32{ 0.0, 0.004208, -0.008270, -0.006104, 0.0, 0.008858, 0.009789, -0.017487 };
    const tol: f32 = 0.001;
    for (0..8) |i| {
        if (@abs(y[i] - expected[i]) > tol) {
            std.log.err("AWQ test fail: y[{d}] = {d:.6}, expected {d:.6}", .{ i, y[i], expected[i] });
            return error.TestUnexpectedResult;
        }
    }
}
