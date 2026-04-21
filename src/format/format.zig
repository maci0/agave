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

pub const arch_key_buf_size: usize = 256;
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
        for (0..self.n_dims) |i| n = std.math.mul(usize, n, std.math.cast(usize, self.dims[i]) orelse return 0) catch return 0;
        return n;
    }

    /// Compute the raw byte size of this tensor's data on disk.
    /// Accounts for quantization block structure (e.g., Q4_K = 144 bytes per
    /// 256-element super-block). Used by prefetchLayer to size madvise hints.
    pub fn dataByteLen(self: *const TensorInfo) usize {
        const n = self.numElements();
        if (n == 0) return 0;
        return switch (self.dtype) {
            .f32 => std.math.mul(usize, n, 4) catch std.math.maxInt(usize),
            .f16, .bf16 => std.math.mul(usize, n, 2) catch std.math.maxInt(usize),
            .fp8_e4m3, .fp8_e5m2 => n,
            .q8_0 => std.math.mul(usize, n / 32, 34) catch std.math.maxInt(usize),
            .q4_0, .iq4_nl => std.math.mul(usize, n / 32, 18) catch std.math.maxInt(usize),
            .q4_1 => std.math.mul(usize, n / 32, 20) catch std.math.maxInt(usize),
            .q5_0 => std.math.mul(usize, n / 32, 22) catch std.math.maxInt(usize),
            .q4_k => std.math.mul(usize, n / 256, 144) catch std.math.maxInt(usize),
            .q5_k => std.math.mul(usize, n / 256, 176) catch std.math.maxInt(usize),
            .q6_k => std.math.mul(usize, n / 256, 210) catch std.math.maxInt(usize),
            .q2_k => std.math.mul(usize, n / 256, 84) catch std.math.maxInt(usize),
            .q3_k => std.math.mul(usize, n / 256, 110) catch std.math.maxInt(usize),
            .iq4_xs => std.math.mul(usize, n / 256, 136) catch std.math.maxInt(usize),
            .tq1_0 => std.math.mul(usize, n / 256, 64) catch std.math.maxInt(usize),
            .mxfp4 => std.math.mul(usize, n / 32, 17) catch std.math.maxInt(usize),
            .nvfp4 => std.math.mul(usize, n / 16, 9) catch std.math.maxInt(usize),
            .mlx_q, .unknown => std.math.mul(usize, n, 4) catch std.math.maxInt(usize),
        };
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
    /// Get the first element of a u32 array metadata value.
    pub fn getMetaArrayFirstU32(self: Format, key: []const u8) ?u32 {
        const arr = self.vtable.get_meta_u32_array(self.ptr, key) orelse return null;
        return if (arr.len > 0) arr[0] else null;
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
    /// Get the first element of a u32 array metadata value with architecture-prefixed key.
    /// Used for per-layer arrays like attention.head_count_kv in Gemma 4.
    pub fn getArchArrayFirstU32(self: Format, arch: []const u8, suffix: []const u8) ?u32 {
        var buf: [arch_key_buf_size]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.getMetaArrayFirstU32(key);
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

    /// Weight tensor suffixes to prefetch — covers the most bandwidth-heavy
    /// tensors (GEMV projections and expert weights). Norms are tiny and
    /// almost always cache-resident, so they're excluded.
    const prefetch_suffixes = [_][]const u8{
        "attn_q.weight",      "attn_k.weight",        "attn_v.weight",
        "attn_qkv.weight",    "attn_output.weight",
        "ffn_gate.weight",    "ffn_up.weight",        "ffn_down.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ssm_in.weight",      "ssm_out.weight",
    };

    /// Hint the OS to prefetch the next layer's weight tensors into memory.
    /// Issues madvise(WILL_NEED) on all known weight tensors for the given
    /// layer index. No-op when tensors are already resident or on non-POSIX.
    /// Call with `li + 1` at the top of each layer's forward pass to overlap
    /// I/O with the current layer's computation.
    pub fn prefetchLayer(self: Format, layer_idx: u32) void {
        for (prefetch_suffixes) |suffix| {
            if (self.layerTensor(layer_idx, suffix)) |info| {
                const byte_len = info.dataByteLen();
                if (byte_len > 0) prefetchRegion(info.data_ptr, byte_len);
            }
        }
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

/// Issue madvise(WILL_NEED) on a byte range to hint the OS to page it in.
/// Aligns the range to page boundaries as required by madvise.
fn prefetchRegion(data: [*]const u8, len: usize) void {
    if (len == 0) return;
    const page = std.heap.page_size_min;
    const addr = @intFromPtr(data);
    const start = addr & ~(@as(usize, page - 1));
    const addr_end = std.math.add(usize, addr, len) catch return;
    const end = std.mem.alignForward(usize, addr_end, page);
    const aligned_ptr: [*]u8 = @ptrFromInt(start);
    std.posix.madvise(@alignCast(aligned_ptr), end - start, std.posix.MADV.WILLNEED) catch |err| {
        std.log.debug("prefetchLayer: madvise failed: {s}", .{@errorName(err)});
    };
}

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

test "TensorInfo dataByteLen f32" {
    var dummy: u8 = 0;
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .f32, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, 256 * 4), t.dataByteLen());
}

test "TensorInfo dataByteLen f16" {
    var dummy: u8 = 0;
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .f16, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, 256 * 2), t.dataByteLen());
}

test "TensorInfo dataByteLen q4_0" {
    var dummy: u8 = 0;
    // Q4_0: 18 bytes per 32-element block
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .q4_0, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, (256 / 32) * 18), t.dataByteLen());
}

test "TensorInfo dataByteLen q4_k" {
    var dummy: u8 = 0;
    // Q4_K: 144 bytes per 256-element super-block
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .q4_k, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, 144), t.dataByteLen());
}

test "TensorInfo dataByteLen q8_0" {
    var dummy: u8 = 0;
    // Q8_0: 34 bytes per 32-element block
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .q8_0, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, (256 / 32) * 34), t.dataByteLen());
}

test "TensorInfo dataByteLen zero elements" {
    var dummy: u8 = 0;
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 0, 0, 0, 0 }, .dtype = .f32, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, 0), t.dataByteLen());
}

test "TensorInfo dataByteLen q6_k" {
    var dummy: u8 = 0;
    // Q6_K: 210 bytes per 256-element super-block
    const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 512, 0, 0, 0 }, .dtype = .q6_k, .data_ptr = @as([*]const u8, @ptrCast(&dummy)) };
    try std.testing.expectEqual(@as(usize, (512 / 256) * 210), t.dataByteLen());
}
