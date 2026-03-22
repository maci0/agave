//! CPU GEMV kernels for 4-bit floating-point formats.
//! MXFP4 (E2M1 microscaling) and NVFP4 (FP8 E4M3 block scale).

const quant = @import("../../../ops/quant.zig");

/// MXFP4: 32 values per block, 17 bytes (1 E8M0 scale + 16 nibble-packed bytes)
pub fn gemvMXFP4(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 17;
    const qk: usize = 32;
    const nb = (k + qk - 1) / qk;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const d = quant.e8m0ToF32(bp[0]);
            const bk = b * qk;
            for (0..qk / 2) |j| {
                const byte = bp[1 + j];
                const v0 = quant.mxfp4Lookup(byte & 0x0F);
                const v1 = quant.mxfp4Lookup(byte >> 4);
                const gi0 = bk + j;
                const gi1 = bk + j + qk / 2;
                if (gi0 < k) sum += x[gi0] * v0 * d;
                if (gi1 < k) sum += x[gi1] * v1 * d;
            }
        }
        y[row] = sum;
    }
}

/// NVFP4: 16-element blocks with FP8 E4M3 block scale.
/// Block layout: 1 byte FP8 scale + 8 bytes packed nibbles = 9 bytes per block.
pub fn gemvNVFP4(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb: usize = 9;
    const qk: usize = 16;
    const nb = (k + qk - 1) / qk;
    for (0..n) |row| {
        var sum: f32 = 0.0;
        const rp = w + row * nb * bpb;
        for (0..nb) |b| {
            const bp = rp + b * bpb;
            const scale = quant.fp8e4m3ToF32(bp[0]);
            const bk = b * qk;
            for (0..qk / 2) |j| {
                const byte = bp[1 + j];
                const v0 = quant.mxfp4Lookup(byte & 0x0F);
                const v1 = quant.mxfp4Lookup(byte >> 4);
                const gi0 = bk + j;
                const gi1 = bk + j + qk / 2;
                if (gi0 < k) sum += x[gi0] * v0 * scale;
                if (gi1 < k) sum += x[gi1] * v1 * scale;
            }
        }
        y[row] = sum;
    }
}
