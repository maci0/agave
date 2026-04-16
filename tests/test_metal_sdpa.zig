const std = @import("std");
const builtin = @import("builtin");
const backend = @import("backend");
const Backend = backend.Backend;
const MetalBackend = backend.MetalBackend;
const sdpa_harness = @import("sdpa_harness");

test "Metal SDPA FlashAttention-2 dual-delta correctness" {
    // Skip if not on macOS
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;

    // Skip if Metal not available
    var metal_be = MetalBackend.init(std.testing.allocator) catch |err| {
        if (err == error.NoMetalDevice) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer metal_be.deinit();

    // Small for fast test, representative for correctness (4 heads, GQA 4:1, 64 positions)
    var gpu_backend: Backend = .{ .metal = &metal_be };
    try sdpa_harness.runDualDeltaTest(&gpu_backend, 4, 1, 128, 64, "Metal SDPA FlashAttention-2");
}
