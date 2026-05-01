//! Aggregator that re-exports all ROCm kernel entry points.
//!
//! Compiled as a single unit to produce one HSACO (AMD GPU code object):
//!   zig build amdgcn [-Drocm-arch=gfx1100]
//!
//! Individual kernel files can also be compiled standalone for iteration.

// Force Zig to analyze and emit all kernel exports
comptime {
    // Elementwise
    _ = @import("silu.zig");
    _ = @import("gelu.zig");
    _ = @import("add.zig");
    _ = @import("mul.zig");
    _ = @import("sigmoid_mul.zig");
    _ = @import("deinterleave.zig");
    _ = @import("split_qgate.zig");

    // Normalization
    _ = @import("rms_norm.zig");
    _ = @import("rms_norm_multi.zig");
    _ = @import("softmax.zig");
    _ = @import("l2_norm.zig");

    // Position encoding
    _ = @import("rope.zig");

    // Attention
    _ = @import("sdpa.zig");

    // SSM
    _ = @import("deltanet.zig");

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
    _ = @import("gemv_mlx_q4.zig");
    _ = @import("gemv_t_q8_0.zig");
    _ = @import("gemv_nvfp4_st.zig");
    _ = @import("gemv_mxfp4_st.zig");

    _ = @import("deltanet_recurrence.zig");
    _ = @import("sdpa_tree.zig");

    // True megakernels
    _ = @import("mega_qwen35_q8.zig");
}
