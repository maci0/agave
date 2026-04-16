//! MLX affine 8-bit GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: u32-packed bytes (4 per word), group_size=64.
//! Scales/biases: BF16 per group. Dequant: float = scale * uint8 + bias.
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

const bf16ToF32 = cu.bf16ToF32;

export fn gemv_mlx_q8_kernel(
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
    const wpg: u32 = 16; // 64 bytes / 4 per word
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

        var wi: u32 = 0;
        while (wi < wpg and wi * 4 < elems) : (wi += 1) {
            const word = pw[wo + wi];
            const xi = xo + wi * 4;
            const rem = @min(4, elems - wi * 4);
            var i: u32 = 0;
            while (i < rem) : (i += 1) {
                const q: f32 = @floatFromInt((word >> @as(u5, @intCast(i * 8))) & 0xFF);
                q_dot += q * x[xi + i];
                x_sum += x[xi + i];
            }
        }
        sum += scale * q_dot + bias * x_sum;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
