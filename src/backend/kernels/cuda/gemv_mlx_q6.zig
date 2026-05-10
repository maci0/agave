//! MLX affine 6-bit GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: u32-packed 6-bit values (cross-word spanning), group_size=64.
//! Scales/biases: BF16 per group. Dequant: float = scale * uint6 + bias.
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const bf16ToF32 = cu.bf16ToF32;

/// Unpack one 6-bit value from packed u32 array at element index.
inline fn unpackU6(base: [*]const u32, idx: u32) u32 {
    const bp = idx * 6;
    const wi = bp / 32;
    const bo: u5 = @intCast(bp % 32);
    const word0 = base[wi];
    if (bo <= 26) return (word0 >> bo) & 0x3F;
    const word1 = base[wi + 1];
    return ((word0 >> bo) | (word1 << @as(u5, @intCast(32 - @as(u6, bo))))) & 0x3F;
}

export fn gemv_mlx_q6_kernel(
    x: [*]const f32,
    pw: [*]const u32,
    sc: [*]const u16,
    bi: [*]const u16,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const gs: u32 = 64;
    const wpg: u32 = 12; // 64 * 6 / 32 = 12
    const gpr = (k + gs - 1) / gs;
    const wpr = gpr * wpg;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < gpr) : (g += bdim) {
        const scale = bf16ToF32(sc[row * gpr + g]);
        const bias = bf16ToF32(bi[row * gpr + g]);
        const xo = g * gs;
        const wo = row * wpr + g * wpg;
        const elems = @min(gs, k - xo);

        var q_dot: f32 = 0.0;
        var x_sum: f32 = 0.0;

        var i: u32 = 0;
        while (i < elems) : (i += 1) {
            const q: f32 = @floatFromInt(unpackU6(pw + wo, i));
            q_dot += q * x[xo + i];
            x_sum += x[xo + i];
        }
        sum += scale * q_dot + bias * x_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
