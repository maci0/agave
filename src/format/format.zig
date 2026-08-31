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

pub const arch_key_buf_size: usize = 256;
const layer_name_buf_size: usize = 128;

fn formatNullMetaU64Array(_: *anyopaque, _: []const u8) ?[]const u64 {
    return null;
}

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
    /// 256-element super-block). Uses ceiling division to match gguf.zig's
    /// tensorBytes, correctly handles non-block-aligned element counts.
    /// Used by prefetchLayer to size madvise hints.
    pub fn dataByteLen(self: *const TensorInfo) usize {
        const n = self.numElements();
        if (n == 0) return 0;
        const maxInt = std.math.maxInt(usize);
        return switch (self.dtype) {
            .f32 => std.math.mul(usize, n, 4) catch maxInt,
            .f16, .bf16 => std.math.mul(usize, n, 2) catch maxInt,
            .fp8_e4m3, .fp8_e5m2 => n,
            .q8_0 => std.math.mul(usize, ceilDiv(n, 32), 34) catch maxInt,
            .q4_0, .iq4_nl => std.math.mul(usize, ceilDiv(n, 32), 18) catch maxInt,
            .q4_1 => std.math.mul(usize, ceilDiv(n, 32), 20) catch maxInt,
            .q5_0 => std.math.mul(usize, ceilDiv(n, 32), 22) catch maxInt,
            .q4_k => std.math.mul(usize, ceilDiv(n, 256), 144) catch maxInt,
            .q5_k => std.math.mul(usize, ceilDiv(n, 256), 176) catch maxInt,
            .q6_k => std.math.mul(usize, ceilDiv(n, 256), 210) catch maxInt,
            .q2_k => std.math.mul(usize, ceilDiv(n, 256), 84) catch maxInt,
            .q3_k => std.math.mul(usize, ceilDiv(n, 256), 110) catch maxInt,
            .iq4_xs => std.math.mul(usize, ceilDiv(n, 256), 136) catch maxInt,
            .iq3_xxs => std.math.mul(usize, ceilDiv(n, 256), 98) catch maxInt,
            .iq3_s => std.math.mul(usize, ceilDiv(n, 256), 110) catch maxInt,
            .iq2_xxs => std.math.mul(usize, ceilDiv(n, 256), 66) catch maxInt,
            .iq2_xs => std.math.mul(usize, ceilDiv(n, 256), 74) catch maxInt,
            .iq2_s => std.math.mul(usize, ceilDiv(n, 256), 82) catch maxInt,
            .iq1_s => std.math.mul(usize, ceilDiv(n, 256), 50) catch maxInt,
            .iq1_m => std.math.mul(usize, ceilDiv(n, 256), 56) catch maxInt,
            .tq1_0 => std.math.mul(usize, ceilDiv(n, 256), 54) catch maxInt,
            .tq2_0 => std.math.mul(usize, ceilDiv(n, 256), 66) catch maxInt,
            .mxfp4 => std.math.mul(usize, ceilDiv(n, 32), 17) catch maxInt,
            .nvfp4 => std.math.mul(usize, ceilDiv(n, 16), 9) catch maxInt,
            // HQQ: tensor dims already account for nibble packing ([n_out, k_in/2] uint8),
            // so numElements() returns the packed count. Each element = 1 byte.
            .hqq => n,
            .mlx_q, .gptq, .awq, .unknown => std.math.mul(usize, n, 4) catch maxInt,
        };
    }
};

/// Ceiling division with overflow-safe addition, matching gguf.zig's tensorBytes pattern.
/// Returns `ceil(n / bs)`, saturating to maxInt(usize) on overflow.
fn ceilDiv(n: usize, comptime bs: usize) usize {
    return (std.math.add(usize, n, bs - 1) catch std.math.maxInt(usize)) / bs;
}

/// Model format interface, all model loading goes through this
pub const Format = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// True when loaded from SafeTensors (HF conventions for tensor layout).
    /// False for GGUF (llama.cpp conventions). Models use this to select the
    /// correct Q/K/V split order, GQA head mapping, and weight pre-processing.
    is_safetensors: bool = false,
    /// GGUF file descriptor for pread-based SSD streaming.
    /// Set when loading from GGUF with SSD streaming enabled.
    /// Used by expert cache to pread() weights into Metal buffers.
    file_fd: i32 = -1,
    /// Base address of the mmap'd data section (for computing file offsets).
    mmap_base: ?[*]const u8 = null,

    /// Virtual function table for the Format dispatcher.
    /// Each format implementation (GGUF, SafeTensors) provides these function pointers
    /// to enable polymorphic tensor/metadata lookup without runtime type checks.
    pub const VTable = struct {
        get_tensor: *const fn (self: *anyopaque, name: []const u8) ?TensorInfo,
        get_meta_str: *const fn (self: *anyopaque, key: []const u8) ?[]const u8,
        get_meta_u32: *const fn (self: *anyopaque, key: []const u8) ?u32,
        get_meta_f32: *const fn (self: *anyopaque, key: []const u8) ?f32,
        get_meta_u32_array: *const fn (self: *anyopaque, key: []const u8) ?[]const u32,
        /// Optional: uint64 arrays (qwen4exp PLE multipliers/offsets). Default returns null.
        get_meta_u64_array: *const fn (self: *anyopaque, key: []const u8) ?[]const u64 = formatNullMetaU64Array,
        get_vocab: *const fn (self: *anyopaque) ?[]const []const u8,
        get_merges: *const fn (self: *anyopaque) ?[]const []const u8,
        /// Free large repacked weight buffers after their device uploads
        /// (CUDA path — the host copies are dead weight). No-op for formats
        /// without repacked buffers.
        release_repacked: *const fn (self: *anyopaque) void,
        /// Free one large repacked buffer (host copy) after its device upload.
        free_repacked_tensor: *const fn (self: *anyopaque, ptr: [*]const u8) void,
    };

    /// Look up a tensor by name, returning its metadata and data pointer.
    pub fn getTensor(self: Format, name: []const u8) ?TensorInfo {
        return self.vtable.get_tensor(self.ptr, name);
    }
    /// Free large repacked weight buffers (host copies) after their device
    /// uploads — see SafeTensorsDir.releaseRepacked.
    pub fn releaseRepacked(self: Format) void {
        self.vtable.release_repacked(self.ptr);
    }
    /// Free one large repacked host buffer after its device upload.
    pub fn freeRepackedTensor(self: Format, ptr: [*]const u8) void {
        self.vtable.free_repacked_tensor(self.ptr, ptr);
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
    /// Get a u64 array metadata value by key (e.g., PLE n-gram multipliers).
    pub fn getMetaU64Array(self: Format, key: []const u8) ?[]const u64 {
        return self.vtable.get_meta_u64_array(self.ptr, key);
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
    /// Get a u32 array with architecture-prefixed key (e.g. "qwen4exp.ple.layers").
    pub fn getArchU32Array(self: Format, arch: []const u8, suffix: []const u8) ?[]const u32 {
        var buf: [arch_key_buf_size]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.getMetaU32Array(key);
    }
    /// Get a u64 array with architecture-prefixed key (e.g. "qwen4exp.ple.layer_multipliers").
    pub fn getArchU64Array(self: Format, arch: []const u8, suffix: []const u8) ?[]const u64 {
        var buf: [arch_key_buf_size]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.getMetaU64Array(key);
    }

    /// Look up a layer-prefixed tensor by index and suffix.
    /// E.g., layerTensor(3, "attn_q.weight") looks up "blk.3.attn_q.weight".
    pub fn layerTensor(self: Format, li: u32, suffix: []const u8) ?TensorInfo {
        var buf: [layer_name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ li, suffix }) catch return null;
        return self.getTensor(name);
    }

    /// True when a conv1d weight is MLX-native `[C, K, 1]` (last dim 1).
    /// HuggingFace stores `[C, 1, K]`. MLX sanitize() also bakes RMSNorm `+1`.
    pub fn conv1dIsMlxNative(t: TensorInfo) bool {
        return t.n_dims >= 2 and t.dims[t.n_dims - 1] == 1;
    }

    /// True when Qwen3.5/3.8 SafeTensors already store shifted RMSNorm gammas
    /// (`y = w * rms(x)` with w = 1+hf_w). Detected via MLX conv1d layout.
    pub fn qwen35MlxNormsShifted(self: Format) bool {
        const conv = self.layerTensor(0, "ssm_conv1d.weight") orelse return false;
        return conv1dIsMlxNative(conv);
    }

    /// Weight tensor suffixes to prefetch, covers the most bandwidth-heavy
    /// tensors (GEMV projections and expert weights). Norms are tiny and
    /// almost always cache-resident, so they're excluded.
    const prefetch_suffixes = [_][]const u8{
        "attn_q.weight",      "attn_k.weight",        "attn_v.weight",
        "attn_qkv.weight",    "attn_output.weight",   "ffn_gate.weight",
        "ffn_up.weight",      "ffn_down.weight",      "ffn_gate_exps.weight",
        "ffn_up_exps.weight", "ffn_down_exps.weight", "ssm_in.weight",
        "ssm_out.weight",
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
        // SafeTensors NVFP4: scale tensors present alongside U8/U32 weights
        if (self.getTensor("backbone.layers.0.mixer.in_proj.scales") != null or
            self.getTensor("backbone.layers.0.mixer.in_proj.weight_scale") != null) return "NVFP4";

        // MoE expert weights first, they dominate model size and represent
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
                    .iq3_xxs => "IQ3_XXS",
                    .iq3_s => "IQ3_S",
                    .iq2_xxs => "IQ2_XXS",
                    .iq2_xs => "IQ2_XS",
                    .iq2_s => "IQ2_S",
                    .iq1_s => "IQ1_S",
                    .iq1_m => "IQ1_M",
                    .fp8_e4m3 => "FP8_E4M3",
                    .fp8_e5m2 => "FP8_E5M2",
                    .nvfp4 => "NVFP4",
                    .mxfp4 => "MXFP4",
                    .tq1_0 => "TQ1_0",
                    .tq2_0 => "TQ2_0",
                    .mlx_q => "MLX-Q",
                    .gptq => "GPTQ",
                    .awq => "AWQ",
                    .hqq => "HQQ",
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
    if (comptime @import("builtin").os.tag == .freestanding) return;
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

/// GGUF file format implementation, re-exported so callers use format.zig as the single import.
pub const GGUFFile = @import("gguf.zig").GGUFFile;

/// SafeTensors directory loader, re-exported so callers use format.zig as the single import.
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

test "conv1dIsMlxNative last dim 1 vs HuggingFace last dim K" {
    var dummy: u8 = 0;
    const mlx = TensorInfo{
        .name = "blk.0.ssm_conv1d.weight",
        .n_dims = 3,
        .dims = .{ 10240, 4, 1, 0 },
        .dtype = .bf16,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    const hf = TensorInfo{
        .name = "blk.0.ssm_conv1d.weight",
        .n_dims = 3,
        .dims = .{ 10240, 1, 4, 0 },
        .dtype = .bf16,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    try std.testing.expect(Format.conv1dIsMlxNative(mlx));
    try std.testing.expect(!Format.conv1dIsMlxNative(hf));
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

test "TensorInfo dataByteLen all quantized dtypes" {
    var dummy: u8 = 0;
    const ptr = @as([*]const u8, @ptrCast(&dummy));

    // Each test: 256 elements, verify byte calculation
    const quant_tests = [_]struct { DType, usize }{
        // Scalar types
        .{ .f32, 256 * 4 },
        .{ .f16, 256 * 2 },
        .{ .bf16, 256 * 2 },
        .{ .fp8_e4m3, 256 },
        .{ .fp8_e5m2, 256 },
        // 32-element block types
        .{ .q8_0, (256 / 32) * 34 },
        .{ .q4_0, (256 / 32) * 18 },
        .{ .iq4_nl, (256 / 32) * 18 },
        .{ .q4_1, (256 / 32) * 20 },
        .{ .q5_0, (256 / 32) * 22 },
        .{ .mxfp4, (256 / 32) * 17 },
        // 256-element super-block types
        .{ .q4_k, 144 },
        .{ .q5_k, 176 },
        .{ .q6_k, 210 },
        .{ .q2_k, 84 },
        .{ .q3_k, 110 },
        .{ .iq4_xs, 136 },
        .{ .tq1_0, 54 },
        .{ .tq2_0, 66 },
        // NVFP4: 16-element group, 9 bytes per group
        .{ .nvfp4, (256 / 16) * 9 },
    };
    for (quant_tests) |qt| {
        const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = qt[0], .data_ptr = ptr };
        try std.testing.expectEqual(qt[1], t.dataByteLen());
    }
}

test "TensorInfo dataByteLen 2D tensor" {
    var dummy: u8 = 0;
    // 2D tensor: 4096 x 4096 = 16M elements in Q4_K
    const t = TensorInfo{
        .name = "w",
        .n_dims = 2,
        .dims = .{ 4096, 4096, 0, 0 },
        .dtype = .q4_k,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    const n = 4096 * 4096;
    try std.testing.expectEqual(@as(usize, n), t.numElements());
    // Q4_K: 256 elements per block, 144 bytes per block
    try std.testing.expectEqual(@as(usize, (n / 256) * 144), t.dataByteLen());
}

test "TensorInfo dataByteLen unknown and packed types" {
    var dummy: u8 = 0;
    const ptr = @as([*]const u8, @ptrCast(&dummy));

    // unknown, mlx_q, gptq, awq all use n*4 (treated as 4 bytes per element)
    const packed_types = [_]DType{ .unknown, .mlx_q, .gptq, .awq };
    // HQQ uses n*1 (uint8, 2 nibbles/byte, companion scale/zero separate)
    const t_hqq = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 256, 0, 0, 0 }, .dtype = .hqq, .data_ptr = ptr };
    try std.testing.expectEqual(@as(usize, 256), t_hqq.dataByteLen());
    for (packed_types) |dt| {
        const t = TensorInfo{ .name = "w", .n_dims = 1, .dims = .{ 100, 0, 0, 0 }, .dtype = dt, .data_ptr = ptr };
        try std.testing.expectEqual(@as(usize, 400), t.dataByteLen());
    }
}

test "TensorInfo numElements overflow protection" {
    var dummy: u8 = 0;
    // Huge dimensions that would overflow usize multiplication
    const t = TensorInfo{
        .name = "huge",
        .n_dims = 2,
        .dims = .{ std.math.maxInt(u64), 2, 0, 0 },
        .dtype = .f32,
        .data_ptr = @as([*]const u8, @ptrCast(&dummy)),
    };
    // Should return 0 on overflow, not crash
    try std.testing.expectEqual(@as(usize, 0), t.numElements());
}

test "Format getArchU32 and getArchF32 formatting" {
    // Test that arch key formatting works correctly by verifying buffer format
    var buf: [arch_key_buf_size]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ "gemma3", "block_count" }) catch unreachable;
    try std.testing.expectEqualStrings("gemma3.block_count", key);

    const key2 = std.fmt.bufPrint(&buf, "{s}.{s}", .{ "qwen3_5_moe_text", "attention.head_count" }) catch unreachable;
    try std.testing.expectEqualStrings("qwen3_5_moe_text.attention.head_count", key2);
}

test "Format layerTensor name formatting" {
    // Verify the layer tensor name format string
    const layer_name_buf_sz: usize = 128;
    var buf: [layer_name_buf_sz]u8 = undefined;
    const name0 = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ @as(u32, 0), "attn_q.weight" }) catch unreachable;
    try std.testing.expectEqualStrings("blk.0.attn_q.weight", name0);

    const name42 = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ @as(u32, 42), "ffn_gate_exps.weight" }) catch unreachable;
    try std.testing.expectEqualStrings("blk.42.ffn_gate_exps.weight", name42);
}

test "DType enum completeness" {
    // Verify all DType variants are distinct and the enum has the expected count
    const dtype_fields = @typeInfo(DType).@"enum".fields;
    // Count should match all known dtypes
    try std.testing.expect(dtype_fields.len >= 25); // At least: f32, f16, bf16, q2-q8, iq4s, fp8s, nvfp4, mxfp4, tq1_0, tq2_0, mlx_q, gptq, awq, hqq, unknown
}

test "fuzz: all format functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // -- pub const arch_key_buf_size --
            comptime {
                std.debug.assert(arch_key_buf_size == 256);
            }

            // -- pub const GGUFFile, SafeTensorsDir (re-exports) --
            comptime {
                _ = GGUFFile;
                _ = SafeTensorsDir;
            }

            // -- pub enum DType: exercise all variants via random tag --
            const dtype_int = smith.valueWithHash(u5, 0);
            const all_dtypes = comptime std.enums.values(DType);
            const dtype = all_dtypes[@as(usize, dtype_int) % all_dtypes.len];
            _ = @tagName(dtype);

            // -- pub fn TensorInfo.numElements --
            var dummy_data: [4]u8 = .{ 0, 0, 0, 0 };
            const n_dims = smith.valueWithHash(u3, 1) % 5; // 0..4
            const d0 = smith.valueWithHash(u16, 2);
            const d1 = smith.valueWithHash(u16, 3);
            const d2 = smith.valueWithHash(u8, 4);
            const d3 = smith.valueWithHash(u8, 5);
            const ti = TensorInfo{
                .name = "fuzz",
                .n_dims = @as(u32, n_dims),
                .dims = .{ @as(u64, d0), @as(u64, d1), @as(u64, d2), @as(u64, d3) },
                .dtype = dtype,
                .data_ptr = @as([*]const u8, @ptrCast(&dummy_data)),
            };
            const num_el = ti.numElements();

            // -- pub fn TensorInfo.dataByteLen --
            const byte_len = ti.dataByteLen();
            // dataByteLen returns 0 when numElements is 0
            if (num_el == 0) {
                std.debug.assert(byte_len == 0);
            }

            // -- Mock VTable for all Format methods --
            const MockVTable = struct {
                fn getTensor(_: *anyopaque, _: []const u8) ?TensorInfo {
                    return null;
                }
                fn getMetaStr(_: *anyopaque, _: []const u8) ?[]const u8 {
                    return null;
                }
                fn getMetaU32(_: *anyopaque, _: []const u8) ?u32 {
                    return null;
                }
                fn getMetaF32(_: *anyopaque, _: []const u8) ?f32 {
                    return null;
                }
                fn getMetaU32Array(_: *anyopaque, _: []const u8) ?[]const u32 {
                    return null;
                }
                fn getVocab(_: *anyopaque) ?[]const []const u8 {
                    return null;
                }
                fn getMerges(_: *anyopaque) ?[]const []const u8 {
                    return null;
                }
            };
            const vtable = Format.VTable{
                .get_tensor = MockVTable.getTensor,
                .get_meta_str = MockVTable.getMetaStr,
                .get_meta_u32 = MockVTable.getMetaU32,
                .get_meta_f32 = MockVTable.getMetaF32,
                .get_meta_u32_array = MockVTable.getMetaU32Array,
                .get_vocab = MockVTable.getVocab,
                .get_merges = MockVTable.getMerges,
                .release_repacked = struct {
                    fn f(_: *anyopaque) void {}
                }.f,
                .free_repacked_tensor = struct {
                    fn f(_: *anyopaque, _: [*]const u8) void {}
                }.f,
            };
            var mock_state: u8 = 0;
            const fmt = Format{
                .ptr = @ptrCast(&mock_state),
                .vtable = &vtable,
                .is_safetensors = smith.valueWithHash(bool, 6),
            };

            // -- pub fn Format.getTensor --
            std.debug.assert(fmt.getTensor("fuzz_tensor") == null);

            // -- pub fn Format.getMetaStr --
            std.debug.assert(fmt.getMetaStr("fuzz_key") == null);

            // -- pub fn Format.getMetaU32 --
            std.debug.assert(fmt.getMetaU32("fuzz_key") == null);

            // -- pub fn Format.getMetaF32 --
            std.debug.assert(fmt.getMetaF32("fuzz_key") == null);

            // -- pub fn Format.getMetaU32Array --
            std.debug.assert(fmt.getMetaU32Array("fuzz_key") == null);

            // -- pub fn Format.getMetaArrayFirstU32 --
            std.debug.assert(fmt.getMetaArrayFirstU32("fuzz_key") == null);

            // -- pub fn Format.getVocab --
            std.debug.assert(fmt.getVocab() == null);

            // -- pub fn Format.getMerges --
            std.debug.assert(fmt.getMerges() == null);

            // -- pub fn Format.getArchU32 --
            const arch_byte = smith.valueWithHash(u8, 7);
            var arch_buf: [3]u8 = .{ 'a', @truncate(arch_byte >> 4), @truncate(arch_byte & 0xf) };
            const arch_slice: []const u8 = &arch_buf;
            std.debug.assert(fmt.getArchU32(arch_slice, "block_count") == null);

            // -- pub fn Format.getArchArrayFirstU32 --
            std.debug.assert(fmt.getArchArrayFirstU32(arch_slice, "head_count") == null);

            // -- pub fn Format.getArchF32 --
            std.debug.assert(fmt.getArchF32(arch_slice, "rope_theta") == null);

            // -- pub fn Format.layerTensor --
            const layer_idx = smith.valueWithHash(u32, 8);
            std.debug.assert(fmt.layerTensor(layer_idx, "attn_q.weight") == null);

            // -- pub fn Format.prefetchLayer --
            // With mock returning null tensors, this is a no-op but exercises the loop
            fmt.prefetchLayer(smith.valueWithHash(u32, 9));

            // -- pub fn Format.getQuantName --
            const qname = fmt.getQuantName();
            std.debug.assert(qname.len > 0);
        }
    }.f, .{});
}
