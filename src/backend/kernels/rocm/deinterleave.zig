//! De-interleave paired blocks: [A0(stride), B0(stride), A1(stride), B1(stride), ...]
//! Extracts A into out_a (compacted) and B into out_b (compacted).
//! Total threads = n_pairs * stride. Each thread copies one element.
//! Grid: ceil(n_pairs * stride / 256) workgroups of 256 threads.

const cu = @import("common.zig");

export fn deinterleave_kernel(input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: u32, n_pairs: u32) callconv(.kernel) void {
    const total = n_pairs * stride;
    const idx = cu.globalIdx();
    if (idx >= total) return;

    const pair = idx / stride;
    const lane = idx % stride;

    out_a[pair * stride + lane] = input[pair * 2 * stride + lane];
    out_b[pair * stride + lane] = input[pair * 2 * stride + stride + lane];
}
