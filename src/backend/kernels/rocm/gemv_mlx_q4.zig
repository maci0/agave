//! GEMV MLX Q4 kernel: y[row] = dot(dequant(W_mlx4[row,:]), x)
//! MLX 4-bit affine quantization: groups of 64 elements.
//! Per group: 8 × u32 packed weights (8 nibbles each), 1 × bf16 scale, 1 × bf16 bias.
//! Dequant: float_val = scale * int4_val + bias
//! Launch with n workgroups of 256 threads (one row per workgroup).
//!
//! Factored computation: y[row] = sum_g(scale_g * dot_g + bias_g * xsum_g)
//!   where dot_g = sum(int_val * x) and xsum_g = sum(x) within each group.
//!   This saves one multiply per element vs direct (scale*val+bias)*x.

const cu = @import("common.zig");

/// Elements per MLX quantization group.
const mlx_group_size: u32 = 64;
/// u32 words per group (64 nibbles / 8 nibbles per word).
const words_per_group: u32 = 8;

/// Convert BF16 (stored as u16) to f32: shift left 16 bits.
inline fn bf16ToF32(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Process one u32 word (8 packed 4-bit values) against 8 x values.
/// Returns (dot, xsum) where dot = sum(nibble_i * x_i), xsum = sum(x_i).
inline fn accumWord(word: u32, x: [*]const f32, base: u32) struct { f32, f32 } {
    var dot: f32 = 0.0;
    var xsum: f32 = 0.0;
    inline for (0..8) |ni| {
        const shift: u5 = @intCast(ni * 4);
        const val: f32 = @floatFromInt(@as(u4, @truncate(word >> shift)));
        const xv = x[base + @as(u32, ni)];
        dot += val * xv;
        xsum += xv;
    }
    return .{ dot, xsum };
}

export fn gemv_mlx_q4_kernel(
    x: [*]const f32,
    w: [*]const u32,
    scales: [*]const u16,
    biases: [*]const u16,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const groups_per_row = (k + mlx_group_size - 1) / mlx_group_size;
    const wpr = groups_per_row * words_per_group;

    var sum: f32 = 0.0;
    var grp = tid;
    while (grp < groups_per_row) : (grp += bdim) {
        const scale = bf16ToF32(scales[row * groups_per_row + grp]);
        const bias = bf16ToF32(biases[row * groups_per_row + grp]);
        const base_col = grp * mlx_group_size;
        const w_base = row * wpr + grp * words_per_group;

        var dot: f32 = 0.0;
        var xsum: f32 = 0.0;

        // Process 8 u32 words = 64 nibbles = 64 elements
        inline for (0..words_per_group) |wi| {
            const r = accumWord(w[w_base + @as(u32, wi)], x, base_col + @as(u32, wi) * 8);
            dot += r[0];
            xsum += r[1];
        }

        sum += scale * dot + bias * xsum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
