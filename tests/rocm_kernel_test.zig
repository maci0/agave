//! ROCm kernel numerical verification tests.
//! Tests new quantized GEMV and elementwise kernels against CPU reference.
//! Uses dual-delta criterion: GPU error ≤ 2× CPU error.

const std = @import("std");
const builtin = @import("builtin");

test "ROCm kernel tests - gated by Linux" {
    // TODO: Add actual ROCm kernel numerical tests (Q4_K/Q5_K/Q6_K/FP8 GEMV
    // against CPU reference with dual-delta criterion). Requires ROCm hardware.
    // Placeholder — skip until implemented to avoid false confidence.
    return error.SkipZigTest;
}
