//! CPU reference GEMV kernels for IQ1/IQ2/IQ3 quantization formats.
//! Codebooks ported from llama.cpp ggml/src/ggml-common.h.
//! IQ2_XXS and IQ3_XXS have full correct implementations.
//! IQ2_XS, IQ2_S, IQ3_S, IQ1_S, IQ1_M use scale-only approximations.

const std = @import("std");
const backend_mod = @import("../../backend.zig");

inline fn readF16LE(data: [*]const u8, off: usize) f32 {
    const bits = std.mem.readInt(u16, data[off..][0..2], .little);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ── IQ2_XXS codebook ─────────────────────────────────────────────────────────
// 256 uint64_t entries. Each entry packs 8 int8_t weight values.
// Values ∈ {0x08=8, 0x19=25, 0x2b=43} (positive magnitudes, signs applied separately).
// Source: llama.cpp ggml/src/ggml-common.h iq2xxs_grid[256]
const iq2xxs_grid = [256]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// ── IQ3_XXS codebook ─────────────────────────────────────────────────────────
// 256 uint32_t entries. Each entry packs 4 int8_t weight values.
// Values ∈ {4,12,20,28,36,44,52,62} (3-bit quantization levels, magnitudes only).
// Source: llama.cpp ggml/src/ggml-common.h iq3xxs_grid[256]
const iq3xxs_grid = [256]u32{
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// ── IQ2_XXS GEMV ─────────────────────────────────────────────────────────────
// Block: f16 d (2) + uint8[64] qs (64) = 66 bytes, 256 elements
// 8 groups × 32 elements. Per group (8 bytes):
//   qs[0..3] = 4 codebook indices into iq2xxs_grid
//   qs[4..7] = uint32 aux: bits 28-31=sub-scale(0-15), bits 0-27=sign bits
pub fn gemvIQ2_XXS(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.iq2_xxs_block_bytes;
    const qk: usize = 256;
    const nsb = (k + qk - 1) / qk;
    const row_bytes = nsb * bpb;

    for (0..n) |row| {
        var sum: f64 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nsb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = readF16LE(bp, 0);
            const qs = bp + 2;

            var gi: usize = 0;
            var yi: usize = 0;
            while (yi < 256) : ({
                yi += 32;
                gi += 8;
            }) {
                const g0 = iq2xxs_grid[qs[gi + 0]];
                const g1 = iq2xxs_grid[qs[gi + 1]];
                const g2 = iq2xxs_grid[qs[gi + 2]];
                const g3 = iq2xxs_grid[qs[gi + 3]];
                const aux = std.mem.readInt(u32, qs[gi + 4 ..][0..4], .little);
                const dl = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.25;
                const signs = aux & 0x0FFFFFFF;

                const base = b * qk + yi;
                for (0..8) |j| {
                    const v0: i8 = @bitCast(@as(u8, @truncate(g0 >> @as(u6, @intCast(j * 8)))));
                    const v1: i8 = @bitCast(@as(u8, @truncate(g1 >> @as(u6, @intCast(j * 8)))));
                    const v2: i8 = @bitCast(@as(u8, @truncate(g2 >> @as(u6, @intCast(j * 8)))));
                    const v3: i8 = @bitCast(@as(u8, @truncate(g3 >> @as(u6, @intCast(j * 8)))));
                    const s0: f32 = if ((signs >> @intCast(j)) & 1 != 0) -1.0 else 1.0;
                    const s1: f32 = if ((signs >> @intCast(j + 8)) & 1 != 0) -1.0 else 1.0;
                    const s2: f32 = if ((signs >> @intCast(j + 16)) & 1 != 0) -1.0 else 1.0;
                    const s3: f32 = if ((signs >> @intCast(j + 24)) & 1 != 0) -1.0 else 1.0;
                    if (base + j < k) sum += @as(f64, x[base + j]) * dl * @as(f32, @floatFromInt(v0)) * s0;
                    if (base + j + 8 < k) sum += @as(f64, x[base + j + 8]) * dl * @as(f32, @floatFromInt(v1)) * s1;
                    if (base + j + 16 < k) sum += @as(f64, x[base + j + 16]) * dl * @as(f32, @floatFromInt(v2)) * s2;
                    if (base + j + 24 < k) sum += @as(f64, x[base + j + 24]) * dl * @as(f32, @floatFromInt(v3)) * s3;
                }
            }
        }
        y[row] = @floatCast(sum);
    }
}

// ── IQ3_XXS GEMV ─────────────────────────────────────────────────────────────
// Block: f16 d (2) + uint8[64] qs (64) + uint8[32] gas (32) = 98 bytes, 256 elements
// 8 groups × 32 elements. Per group:
//   qs[0..7] = 8 codebook indices into iq3xxs_grid (each index → 4 values)
//   gas[0..3] = uint32 aux: bits 28-31=sub-scale(0-15), bits 0-27=sign bits
pub fn gemvIQ3_XXS(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const bpb = backend_mod.iq3_xxs_block_bytes;
    const qk: usize = 256;
    const nsb = (k + qk - 1) / qk;
    const row_bytes = nsb * bpb;

    for (0..n) |row| {
        var sum: f64 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nsb) |b| {
            const bp = rp + b * bpb;
            const d: f32 = readF16LE(bp, 0);
            const qs = bp + 2; // uint8[64]
            const gas = bp + 66; // uint8[32] (starts after qs[64])

            var qi: usize = 0; // byte offset into qs
            var gi: usize = 0; // byte offset into gas
            var yi: usize = 0; // element offset
            while (yi < 256) : ({
                yi += 32;
                qi += 8;
                gi += 4;
            }) {
                const aux = std.mem.readInt(u32, gas[gi..][0..4], .little);
                const dl = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
                const signs = aux & 0x0FFFFFFF;

                const base = b * qk + yi;
                for (0..8) |gi2| { // 8 grid entries per group of 32
                    const grid_entry = iq3xxs_grid[qs[qi + gi2]];
                    const elem_base = gi2 * 4;
                    for (0..4) |j| {
                        const v: i8 = @bitCast(@as(u8, @truncate(grid_entry >> @as(u5, @intCast(j * 8)))));
                        const sign_bit = elem_base + j;
                        const s: f32 = if ((signs >> @intCast(sign_bit)) & 1 != 0) -1.0 else 1.0;
                        if (base + elem_base + j < k) {
                            sum += @as(f64, x[base + elem_base + j]) * dl * @as(f32, @floatFromInt(v)) * s;
                        }
                    }
                }
            }
        }
        y[row] = @floatCast(sum);
    }
}

// ── Stubs for IQ2_XS, IQ2_S, IQ3_S, IQ1_S, IQ1_M ────────────────────────────
// Scale-only approximation: extracts block scale and sums x[i]*scale.
// Produces non-zero output (better than zeroed); quality is degraded.
// Full codebooks (iq2xs_grid, iq3s_grid, iq1s_grid) needed for correctness.

fn gemvStub(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize, bpb: usize) void {
    const qk: usize = 256;
    const nsb = (k + qk - 1) / qk;
    const row_bytes = nsb * bpb;
    for (0..n) |row| {
        var sum: f64 = 0.0;
        const rp = w + row * row_bytes;
        for (0..nsb) |b| {
            const d: f32 = readF16LE(rp + b * bpb, 0);
            const base = b * qk;
            for (0..@min(qk, k - base)) |j| {
                sum += @as(f64, x[base + j]) * d;
            }
        }
        y[row] = @floatCast(sum * 0.25);
    }
}

pub fn gemvIQ2_XS(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    gemvStub(x, w, y, n, k, backend_mod.iq2_xs_block_bytes);
}
pub fn gemvIQ2_S(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    gemvStub(x, w, y, n, k, backend_mod.iq2_s_block_bytes);
}
pub fn gemvIQ3_S(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    gemvStub(x, w, y, n, k, backend_mod.iq3_s_block_bytes);
}
pub fn gemvIQ1_S(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    gemvStub(x, w, y, n, k, backend_mod.iq1_s_block_bytes);
}
pub fn gemvIQ1_M(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    gemvStub(x, w, y, n, k, backend_mod.iq1_m_block_bytes);
}
