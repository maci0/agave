//! De-interleave paired blocks into two output arrays.
//! For each pair h: out_a[h*stride..(h+1)*stride] = input[(2*h)*stride..(2*h+1)*stride]
//!                  out_b[h*stride..(h+1)*stride] = input[(2*h+1)*stride..(2*h+2)*stride]
//! Grid: ceil(total / 256) blocks of 256 threads, where total = n_pairs * stride.

const cu = @import("common.zig");

export fn deinterleave_kernel(input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: u32, n_pairs: u32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    const total = n_pairs * stride;
    if (idx >= total) return;
    const pair = idx / stride;
    const off = idx % stride;
    out_a[idx] = input[pair * 2 * stride + off];
    out_b[idx] = input[(pair * 2 + 1) * stride + off];
}
