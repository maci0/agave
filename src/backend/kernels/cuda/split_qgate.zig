//! Split concatenated Q+gate per-head data into separate Q and gate arrays.
//! Input layout: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]
//! Output: q_out[h*hd..(h+1)*hd], g_out[h*hd..(h+1)*hd]
//! Grid: ceil(total / 256) blocks of 256 threads, where total = nh * hd.

const cu = @import("common.zig");

export fn split_qgate_kernel(qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: u32, nh: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    const total = nh * hd;
    if (idx >= total) return;
    const h = idx / hd;
    const d = idx % hd;
    q_out[idx] = qg[h * hd * 2 + d];
    g_out[idx] = qg[h * hd * 2 + hd + d];
}
