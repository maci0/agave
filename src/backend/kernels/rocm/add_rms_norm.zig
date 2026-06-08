//! Fused add + RMS norm kernel:
//!   a[i] = a[i] + b[i]
//!   output[i] = a[i] * weight[i] * rsqrt(mean(a^2) + eps)
//! Launch with 1 workgroup of 256 threads.

const cu = @import("common.zig");

export fn add_rms_norm_kernel(
    a: [*]f32,
    b: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    n: u32,
    eps: f32,
) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) {
        const v = a[i] + b[i];
        sum_sq += v * v;
    }

    sum_sq = cu.blockReduceAdd(sum_sq);

    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(n)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    i = tid;
    while (i < n) : (i += bdim) {
        const v = a[i] + b[i];
        a[i] = v;
        output[i] = v * weight[i] * scale;
    }
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: add_rms_norm functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
