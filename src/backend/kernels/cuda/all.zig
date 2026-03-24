//! Aggregator that re-exports all CUDA kernel entry points.
//!
//! Compiled as a single unit to produce one PTX file containing all kernels:
//!   zig build-obj src/backend/kernels/cuda/all.zig \
//!       -target nvptx64-cuda -mcpu=sm_80 -O ReleaseFast \
//!       -fno-emit-bin -femit-asm
//!
//! Individual kernel files can also be compiled standalone for iteration.

// Force Zig to analyze and emit all kernel exports
comptime {
    // Elementwise
    _ = @import("silu.zig");
    _ = @import("gelu.zig");
    _ = @import("add.zig");
    _ = @import("mul.zig");

    // Normalization
    _ = @import("rms_norm.zig");
    _ = @import("softmax.zig");
    _ = @import("l2_norm.zig");

    // Position encoding
    _ = @import("rope.zig");

    // Attention
    _ = @import("sdpa.zig");

    // GEMV
    _ = @import("gemv_f32.zig");
    _ = @import("gemv_bf16.zig");
    _ = @import("gemv_f16.zig");
    _ = @import("gemv_q8_0.zig");
    _ = @import("gemv_q4_0.zig");
    _ = @import("gemv_q4_k.zig");
    _ = @import("gemv_q5_k.zig");
    _ = @import("gemv_q6_k.zig");
    _ = @import("gemv_fp8_e4m3.zig");
    _ = @import("gemv_fp8_e5m2.zig");

    // Batched GEMV
    _ = @import("gemv_q4_0_batch.zig");

    // Batched prefill
    _ = @import("gemm_q8_0.zig");
    _ = @import("rms_norm_batched.zig");
    _ = @import("rope_batched.zig");
}
