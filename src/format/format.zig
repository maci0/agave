//! Model format abstraction.
//! Implementations: gguf.zig, safetensors.zig

const std = @import("std");

/// Supported tensor data types for model weights and activations.
/// Shared across format loaders (tensor metadata) and backends (kernel dispatch).
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
    fp8_e4m3,
    fp8_e5m2,
    nvfp4,
    mxfp4,
    tq1_0,
    /// MLX quantized weights (U32-packed); needs companion scales/biases tensors for dequant.
    mlx_q,
    unknown,
};

const arch_key_buf_size: usize = 256;
const layer_name_buf_size: usize = 128;

/// Tensor metadata from a model file
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64 = .{ 0, 0, 0, 0 },
    dtype: DType,
    data_ptr: [*]const u8,

    /// Returns the total number of elements (product of all dimensions).
    pub fn numElements(self: *const TensorInfo) usize {
        var n: usize = 1;
        for (0..self.n_dims) |i| n = std.math.mul(usize, n, @intCast(self.dims[i])) catch return 0;
        return n;
    }
};

/// Model format interface — all model loading goes through this
pub const Format = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// True when loaded from SafeTensors (HF conventions for tensor layout).
    /// False for GGUF (llama.cpp conventions). Models use this to select the
    /// correct Q/K/V split order, GQA head mapping, and weight pre-processing.
    is_safetensors: bool = false,

    /// Virtual function table for the Format dispatcher.
    /// Each format implementation (GGUF, SafeTensors) provides these function pointers
    /// to enable polymorphic tensor/metadata lookup without runtime type checks.
    pub const VTable = struct {
        get_tensor: *const fn (self: *anyopaque, name: []const u8) ?TensorInfo,
        get_meta_str: *const fn (self: *anyopaque, key: []const u8) ?[]const u8,
        get_meta_u32: *const fn (self: *anyopaque, key: []const u8) ?u32,
        get_meta_f32: *const fn (self: *anyopaque, key: []const u8) ?f32,
        get_meta_u32_array: *const fn (self: *anyopaque, key: []const u8) ?[]const u32,
        get_vocab: *const fn (self: *anyopaque) ?[]const []const u8,
        get_merges: *const fn (self: *anyopaque) ?[]const []const u8,
    };

    /// Look up a tensor by name, returning its metadata and data pointer.
    pub fn getTensor(self: Format, name: []const u8) ?TensorInfo {
        return self.vtable.get_tensor(self.ptr, name);
    }
    /// Get a string metadata value by key.
    pub fn getMetaStr(self: Format, key: []const u8) ?[]const u8 {
        return self.vtable.get_meta_str(self.ptr, key);
    }
    /// Get a u32 metadata value by key.
    pub fn getMetaU32(self: Format, key: []const u8) ?u32 {
        return self.vtable.get_meta_u32(self.ptr, key);
    }
    /// Get an f32 metadata value by key.
    pub fn getMetaF32(self: Format, key: []const u8) ?f32 {
        return self.vtable.get_meta_f32(self.ptr, key);
    }
    /// Get a u32 array metadata value by key (e.g., EOG token IDs).
    pub fn getMetaU32Array(self: Format, key: []const u8) ?[]const u32 {
        return self.vtable.get_meta_u32_array(self.ptr, key);
    }
    /// Get the tokenizer vocabulary array.
    pub fn getVocab(self: Format) ?[]const []const u8 {
        return self.vtable.get_vocab(self.ptr);
    }
    /// Get the tokenizer merge rules array.
    pub fn getMerges(self: Format) ?[]const []const u8 {
        return self.vtable.get_merges(self.ptr);
    }

    /// Get a u32 metadata value with architecture-prefixed key (e.g., "gemma3.block_count").
    pub fn getArchU32(self: Format, arch: []const u8, suffix: []const u8) ?u32 {
        var buf: [arch_key_buf_size]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.getMetaU32(key);
    }
    /// Get an f32 metadata value with architecture-prefixed key.
    pub fn getArchF32(self: Format, arch: []const u8, suffix: []const u8) ?f32 {
        var buf: [arch_key_buf_size]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.getMetaF32(key);
    }

    /// Look up a layer-prefixed tensor by index and suffix.
    /// E.g., layerTensor(3, "attn_q.weight") looks up "blk.3.attn_q.weight".
    pub fn layerTensor(self: Format, li: u32, suffix: []const u8) ?TensorInfo {
        var buf: [layer_name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ li, suffix }) catch return null;
        return self.getTensor(name);
    }

    /// Detect the quantization scheme name by probing well-known weight tensors.
    /// Checks for SafeTensors NVFP4 scale tensors first, then probes the dtype
    /// of common layer-0/layer-1 weight tensors.
    pub fn getQuantName(self: Format) []const u8 {
        // SafeTensors NVFP4: scale tensors present alongside U32 weights
        if (self.getTensor("backbone.layers.0.mixer.in_proj.scales") != null) return "NVFP4";

        // MoE expert weights first — they dominate model size and represent
        // the primary quantization for MoE architectures (GPT-OSS, Nemotron-Nano, GLM-4).
        const test_names = [_][]const u8{ "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.attn_q.weight", "blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ssm_in.weight", "blk.1.ffn_up.weight", "output.weight" };
        for (test_names) |tname| {
            if (self.getTensor(tname)) |t| {
                return switch (t.dtype) {
                    .f32 => "F32",
                    .f16 => "F16",
                    .bf16 => "BF16",
                    .q2_k => "Q2_K",
                    .q3_k => "Q3_K",
                    .q4_0 => "Q4_0",
                    .q4_1 => "Q4_1",
                    .q4_k => "Q4_K",
                    .q5_0 => "Q5_0",
                    .q5_k => "Q5_K",
                    .q6_k => "Q6_K",
                    .q8_0 => "Q8_0",
                    .iq4_xs => "IQ4_XS",
                    .iq4_nl => "IQ4_NL",
                    .fp8_e4m3 => "FP8_E4M3",
                    .fp8_e5m2 => "FP8_E5M2",
                    .nvfp4 => "NVFP4",
                    .mxfp4 => "MXFP4",
                    .tq1_0 => "TQ1_0",
                    .mlx_q => "MLX-Q",
                    .unknown => "unknown",
                };
            }
        }
        return "unknown";
    }
};

/// GGUF file format implementation — re-exported so callers use format.zig as the single import.
pub const GGUFFile = @import("gguf.zig").GGUFFile;

/// SafeTensors directory loader — re-exported so callers use format.zig as the single import.
pub const SafeTensorsDir = @import("safetensors.zig").SafeTensorsDir;

// ── Tests ─────────────────────────────────────────────────────────

test "TensorInfo numElements" {
    var dummy: u8 = 0;
    const t = TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 3, 4, 0, 0 },
        .dtype = .f32,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    try std.testing.expectEqual(@as(usize, 12), t.numElements());
}

test "TensorInfo numElements scalar" {
    var dummy: u8 = 0;
    const t = TensorInfo{
        .name = "scalar",
        .n_dims = 0,
        .dims = .{ 0, 0, 0, 0 },
        .dtype = .f32,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    try std.testing.expectEqual(@as(usize, 1), t.numElements());
}

test "TensorInfo numElements 1D" {
    var dummy: u8 = 0;
    const t = TensorInfo{
        .name = "vec",
        .n_dims = 1,
        .dims = .{ 128, 0, 0, 0 },
        .dtype = .f32,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    try std.testing.expectEqual(@as(usize, 128), t.numElements());
}

test "TensorInfo numElements 4D" {
    var dummy: u8 = 0;
    const t = TensorInfo{
        .name = "weight",
        .n_dims = 4,
        .dims = .{ 2, 3, 4, 5 },
        .dtype = .f32,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    try std.testing.expectEqual(@as(usize, 120), t.numElements());
}
