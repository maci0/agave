//! Rotary Position Embedding kernel, in-place.
//! Grid: ceil(n_heads * rope_dim / 2 / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn rope_kernel(x: [*]f32, pos: u32, n_heads: u32, head_dim: u32, rope_dim: u32, theta: f32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    const half_rope = rope_dim / 2;
    const total_pairs = n_heads * half_rope;
    if (idx >= total_pairs) return;

    const head = idx / half_rope;
    const i = idx % half_rope;

    // Split-complex layout: real at [base+i], imag at [base+i+half]
    const base = head * head_dim;
    const re_idx = base + i;
    const im_idx = base + i + half_rope;

    // freq = 1 / theta^(2i / rope_dim)
    const exponent = -@log(@as(f32, theta)) * @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rope_dim));
    const freq = @exp(exponent);
    const angle = @as(f32, @floatFromInt(pos)) * freq;

    const cos_a = cu.cosf(angle);
    const sin_a = cu.sinf(angle);

    const r = x[re_idx];
    const im = x[im_idx];
    x[re_idx] = r * cos_a - im * sin_a;
    x[im_idx] = r * sin_a + im * cos_a;
}
