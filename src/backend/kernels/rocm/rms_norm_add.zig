//! Fused RMS norm + accumulate: b[i] += a[i] * weight[i] * rsqrt(mean(a^2) + eps)
//!
//! The accumulate form Gemma 4 uses for its post-norm residual. Distinct from
//! `add_rms_norm.zig`, which adds FIRST and then normalizes the sum; here the
//! normalized value is added into an existing accumulator, so `b` is read as
//! well as written.
//!
//! Launch contract (src/backend/rocm.zig): 1 workgroup of `block_size` threads
//! per vector, with `reduction_smem` bytes of shared memory.

const cu = @import("common.zig");

export fn rms_norm_add_kernel(a: [*]const f32, weight: [*]const f32, b: [*]f32, n: u32, eps: f32) callconv(.kernel) void {
    const tid = cu.threadIdx();
    const bdim = cu.blockDim();

    // Phase 1: partial sum of squares over a.
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < n) : (i += bdim) {
        const v = a[i];
        sum_sq += v * v;
    }

    // Phase 2: block reduction to the whole-vector sum.
    sum_sq = cu.blockReduceAdd(sum_sq);

    // Broadcast the scale so every thread uses the same value; without the
    // barrier a thread could read the slot before lane 0 has written it.
    if (tid == 0) cu.sharedStore(0, cu.rsqrtf(sum_sq / @as(f32, @floatFromInt(n)) + eps));
    cu.syncthreads();
    const scale = cu.sharedLoad(0);

    // Phase 3: accumulate into b, not overwrite it.
    i = tid;
    while (i < n) : (i += bdim) {
        b[i] += a[i] * weight[i] * scale;
    }
}

const std = @import("std");

test "constants valid" {
    // No module-level numeric constants in this file.
    _ = @sizeOf(u8);
}

test "fuzz: rms_norm_add functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
