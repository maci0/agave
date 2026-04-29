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
    _ = @import("silu_mul.zig");
    _ = @import("gelu.zig");
    _ = @import("add.zig");
    _ = @import("add_scaled.zig");
    _ = @import("mul.zig");
    _ = @import("sigmoid_mul.zig");
    _ = @import("gelu_mul.zig");
    _ = @import("deinterleave.zig");
    _ = @import("split_qgate.zig");

    // Normalization
    _ = @import("rms_norm.zig");
    _ = @import("add_rms_norm.zig");
    _ = @import("softmax.zig");
    _ = @import("l2_norm.zig");

    // Position encoding
    _ = @import("rope.zig");

    // Attention
    _ = @import("sdpa.zig");
    _ = @import("sdpa_turbo.zig");
    _ = @import("sdpa_prefill.zig");

    // GEMV
    _ = @import("gemv_f32.zig");
    _ = @import("gemv_bf16.zig");
    _ = @import("gemv_f16.zig");
    _ = @import("gemv_q8_0.zig");
    _ = @import("gemv_q4_0.zig");
    _ = @import("gemv_q4_1.zig");
    _ = @import("gemv_q4_k.zig");
    _ = @import("gemv_q5_k.zig");
    _ = @import("gemv_q6_k.zig");
    _ = @import("gemv_fp8_e4m3.zig");
    _ = @import("gemv_fp8_e5m2.zig");
    _ = @import("gemv_t_q8_0.zig");
    _ = @import("gemv_nvfp4_st.zig");
    _ = @import("gemv_mlx_q4.zig");
    _ = @import("gemv_mlx_q6.zig");
    _ = @import("gemv_mlx_q8.zig");
    _ = @import("gemv_mxfp4_st.zig");

    // Fused FFN
    _ = @import("fused_ffn_q8_0.zig");

    // Batched GEMV
    _ = @import("gemv_q4_0_batch.zig");

    // Batched prefill
    _ = @import("gemm_q8_0.zig");
    _ = @import("rms_norm_batched.zig");
    _ = @import("rope_batched.zig");

    // True megakernels
    _ = @import("mega_qwen35_q8.zig");
    _ = @import("mega_gemma_q4k.zig");
    _ = @import("mega_gemma_q8.zig");
}
