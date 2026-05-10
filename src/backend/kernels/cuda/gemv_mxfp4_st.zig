//! MXFP4 SafeTensors GEMV: y[row] = dot(dequant(W[row,:]), x)
//! Weights: u32-packed 4-bit nibbles (8 per word), group_size=32.
//! Scales: E8M0 per group (pure power-of-2, 1 byte each). No bias.
//! Dequant: float_val = mxfp4_lut[nibble] * 2^(scale - 127).
//! Grid: n blocks of 256 threads (one row per block).

const cu = @import("common.zig");

/// E2M1 FP4 → float lookup.
const e2m1_lut = [16]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// E8M0 → f32: val = 2^(byte - 127). Pure power-of-2 (no mantissa).
inline fn e8m0ToF32(byte: u8) f32 {
    if (byte == 0) return 0.0;
    return @bitCast(@as(u32, byte) << 23);
}

export fn gemv_mxfp4_st_kernel(
    x: [*]const f32,
    w: [*]const u32,
    s: [*]const u8,
    y: [*]f32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const gs: u32 = 32;
    const wpg: u32 = 4; // 32 nibbles / 8 per word
    const gpr = (k + gs - 1) / gs;
    const wpr = gpr * wpg;

    var sum: f32 = 0.0;
    var g: u32 = tid;
    while (g < gpr) : (g += bdim) {
        const scale = e8m0ToF32(s[row * gpr + g]);
        const xo = g * gs;
        const wo = row * wpr + g * wpg;

        var gdot: f32 = 0.0;
        var wi: u32 = 0;
        while (wi < wpg and xo + wi * 8 < k) : (wi += 1) {
            const word = w[wo + wi];
            const xi = xo + wi * 8;
            const rem = @min(8, k - xi);
            var i: u32 = 0;
            while (i < rem) : (i += 1) {
                gdot += e2m1_lut[(word >> @as(u5, @intCast(i * 4))) & 0xF] * x[xi + i];
            }
        }
        sum += scale * gdot;
    }

    sum = cu.blockReduceAdd(sum);
    if (tid == 0) y[row] = sum;
}
