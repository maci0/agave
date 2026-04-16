const std = @import("std");
const backend = @import("backend");
const Backend = backend.Backend;
const CudaBackend = backend.CudaBackend;
const sdpa_harness = @import("sdpa_harness");

test "CUDA SDPA warp-parallel dual-delta correctness" {
    // Skip if CUDA not available
    var cuda_be = CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == error.CudaNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cuda_be.deinit();

    // Moderate size to stress warp-parallel softmax (8 heads, GQA 4:1, 256 positions)
    var gpu_backend: Backend = .{ .cuda = &cuda_be };
    try sdpa_harness.runDualDeltaTest(&gpu_backend, 8, 2, 128, 256, "CUDA SDPA warp-parallel");
}
