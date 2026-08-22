//! WebGPU MLX-Q4 GEMV: vocab-sized row counts exceed maxComputeWorkgroupsPerDimension.
const std = @import("std");
const backend = @import("backend");
const WebGpuBackend = backend.WebGpuBackend;

/// Spec minimum maxComputeWorkgroupsPerDimension. Must match webgpu.zig.
const max_workgroups_per_dim: usize = 65535;
const mlx_group_size: usize = 64;

test "WebGPU gemvMlxQ4 chunks past max_workgroups_per_dim" {
    const allocator = std.testing.allocator;
    var gpu = WebGpuBackend.init(allocator) catch |err| {
        if (err == error.WebGpuNotAvailable) return error.SkipZigTest;
        return err;
    };
    defer gpu.deinit();

    const k: usize = mlx_group_size;
    const n: usize = max_workgroups_per_dim + 1;
    const gpr = 1;
    const wpg = 8;

    const x = try allocator.alloc(f32, k);
    defer allocator.free(x);
    @memset(x, 0);
    x[0] = 1.0;
    x[1] = 1.0;

    const weight = try allocator.alloc(u32, n * wpg);
    defer allocator.free(weight);
    @memset(weight, 0);
    const sc16 = try allocator.alloc(u16, n * gpr);
    defer allocator.free(sc16);
    const bi16 = try allocator.alloc(u16, n * gpr);
    defer allocator.free(bi16);

    const bf16_one: u16 = 0x3F80;
    const bf16_half: u16 = 0x3F00;
    for (0..n) |row| {
        weight[row * wpg] = 0x53;
        sc16[row] = bf16_one;
        bi16[row] = bf16_half;
    }

    const y_gpu = try allocator.alloc(f32, n);
    defer allocator.free(y_gpu);
    @memset(y_gpu, 0);

    gpu.gemvMlxQ(x.ptr, @ptrCast(weight.ptr), @ptrCast(sc16.ptr), @ptrCast(bi16.ptr), y_gpu.ptr, n, k, 4, @intCast(mlx_group_size));
    gpu.sync();

    // y = 1.0*(1*3 + 1*5) + 0.5*(1+1) = 9
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), y_gpu[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), y_gpu[max_workgroups_per_dim], 1e-4);
    try std.testing.expect(!std.math.isNan(y_gpu[0]));
    try std.testing.expect(!std.math.isNan(y_gpu[max_workgroups_per_dim]));
}
