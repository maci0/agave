//! Tensor element-type enum, shared by format loaders, quant ops, and backends.
//!
//! Leaf module: no imports, so the compute layers (ops, backend) can depend on
//! `DType` without pulling in the model-file format dispatchers in `format.zig`.
//! `format/format.zig` re-exports it as `format.DType`, which stays the
//! canonical public path for loader-side code.

/// Supported tensor data types for model weights and activations.
pub const DType = enum {
    f32,
    f16,
    bf16,
    q2_k,
    q3_k,
    q4_0,
    q4_1,
    q4_k,
    q5_0,
    q5_k,
    q6_k,
    q8_0,
    iq4_xs,
    iq4_nl,
    iq3_xxs,
    iq3_s,
    iq2_xxs,
    iq2_xs,
    iq2_s,
    iq1_s,
    iq1_m,
    fp8_e4m3,
    fp8_e5m2,
    nvfp4,
    mxfp4,
    tq1_0,
    tq2_0,
    /// MLX quantized weights (U32-packed); needs companion scales/biases tensors for dequant.
    mlx_q,
    /// GPTQ INT4 packed in INT32 (row-major); needs companion scales/qzeros tensors.
    gptq,
    /// AWQ INT4 packed in INT32 (column-major); needs companion scales/qzeros tensors.
    awq,
    /// HQQ 4-bit packed in uint8 (2 nibbles/byte); needs companion meta.scale/meta.zero tensors.
    hqq,
    unknown,
};

const std = @import("std");

test "DType enum completeness" {
    // The exact variant set, not just a count: a `>= 25` floor passes for any
    // enum with enough padding variants and hides a dropped dtype, which the
    // loaders map from GGUF/safetensors names by string.
    const expected = [_][]const u8{
        "f32",    "f16",     "bf16",    "q2_k",     "q3_k",     "q4_0",
        "q4_1",   "q4_k",    "q5_0",    "q5_k",     "q6_k",     "q8_0",
        "iq4_xs", "iq4_nl",  "iq3_xxs", "iq3_s",    "iq2_xxs",  "iq2_xs",
        "iq2_s",  "iq1_s",   "iq1_m",   "fp8_e4m3", "fp8_e5m2", "nvfp4",
        "mxfp4",  "tq1_0",   "tq2_0",   "mlx_q",    "gptq",     "awq",
        "hqq",    "unknown",
    };
    const dtype_fields = @typeInfo(DType).@"enum".field_names;
    try std.testing.expectEqual(expected.len, dtype_fields.len);
    inline for (expected, 0..) |name, i| {
        try std.testing.expectEqualStrings(name, dtype_fields[i]);
    }
}
