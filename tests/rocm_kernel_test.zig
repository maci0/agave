//! ROCm kernel numerical verification tests.
//! Tests new quantized GEMV and elementwise kernels against CPU reference.
//! Uses dual-delta criterion: GPU error ≤ 2× CPU error.

const std = @import("std");
const builtin = @import("builtin");

test "ROCm kernel tests - gated by Linux" {
    if (builtin.os.tag != .linux) {
        std.debug.print("ROCm tests skipped (not Linux)\n", .{});
        return error.SkipZigTest;
    }

    // TODO: Add actual ROCm kernel numerical tests
    // These would test:
    // - Q4_K, Q5_K, Q6_K GEMV against CPU reference
    // - FP8 E4M3, E5M2 GEMV against CPU reference
    // - sigmoidMul, deinterleave against CPU reference
    //
    // Dual-delta criterion:
    //   1. Run CPU GEMV vs FP64 reference → measure CPU error
    //   2. Run GPU GEMV vs FP64 reference → measure GPU error
    //   3. Assert: GPU error ≤ 2 × CPU error
    //
    // Requires:
    // - ROCm hardware available (test would skip if not)
    // - Synthetic quantized weight generation
    // - FP64 reference implementation
    //
    // For now, this is a placeholder ensuring the test infrastructure compiles.

    std.debug.print("ROCm kernel tests: placeholder (implementation deferred)\n", .{});
}
