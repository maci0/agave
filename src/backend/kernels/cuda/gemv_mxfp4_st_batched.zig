//! Batched MXFP4 expert GEMV: y_slot[row] = dot(dequant(W_slot[row,:]), x_slot)
//! One launch computes ALL active experts' gate+up (or down) projections —
//! the sustained memory traffic keeps the GB10 memory clock ramped, while
//! per-expert launches (25µs bursts) leave it at the idle rate (measured:
//! 4.2MB reads at ~2GB/s = 2-5ms each when unbatched).
//!
//! Pointers are passed as DEVICE-side tables (uploaded per call):
//!   w_tab[slot]  — packed FP4 weight base for slot
//!   s_tab[slot]  — E8M0/E4M3 scale base for slot
//!   x_tab[slot]  — input base for slot (same for gate+up, per-slot for down)
//!   y_tab[slot]  — output base for slot (per-slot act_cache buffers)
//! Grid: n blocks of 256 threads (one row per block, loop over slots).

const cu = @import("common.zig");

const e2m1_lut = cu.e2m1_lut;
const fp8e4m3ToF32 = cu.fp8e4m3ToF32;
const e8m0ToF32 = cu.e8m0ToF32;

/// Scale format selectors (kernel parameters).
const scale_mode_e4m3: u32 = 0;
const scale_mode_e8m0: u32 = 1;

export fn gemv_mxfp4_st_batched_kernel(
    x_tab: [*]const [*]const f32,
    w_tab: [*]const [*]const u32,
    s_tab: [*]const [*]const u8,
    y_tab: [*]const [*]f32,
    n: u32,
    k: u32,
    gs: u32,
    scale_mode: u32,
    n_slots: u32,
) callconv(.nvptx_device) void {
    const row = cu.blockIdx();
    if (row >= n) return;
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    const wpg = gs / 8; // u32 words per scale group
    const gpr = (k + gs - 1) / gs;
    const wpr = gpr * wpg;

    var slot: u32 = 0;
    while (slot < n_slots) : (slot += 1) {
        const x = x_tab[slot];
        const w = w_tab[slot];
        const s = s_tab[slot];
        const y = y_tab[slot];

        var sum: f32 = 0.0;
        var g: u32 = tid;
        while (g < gpr) : (g += bdim) {
            const scale = if (scale_mode == scale_mode_e8m0) e8m0ToF32(s[row * gpr + g]) else fp8e4m3ToF32(s[row * gpr + g]);
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
}

const std = @import("std");

test "constants valid" {
    comptime std.debug.assert(scale_mode_e4m3 == 0);
    comptime std.debug.assert(scale_mode_e8m0 == 1);
}
