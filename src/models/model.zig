//! Model interface for LLM inference.
//! Provides a type-erased interface via comptime vtable generation, allowing
//! the engine to work with any model architecture through a uniform API.
//!
//! Implementations: gemma3.zig, qwen35.zig, gpt_oss.zig, nemotron_h.zig,
//! nemotron_nano.zig, glm4.zig

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const Arch = @import("../arch.zig").Arch;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const kv_quant = @import("../ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

/// Buffer size for constructing companion tensor names (e.g., ".scales", ".biases").
pub const tensor_name_buf_size: usize = 256;
/// Bits per u32 word — used to compute per-tensor bit width from packed weight dimensions.
const bits_per_u32_word: u64 = 32;

/// Errors that can occur during model forward pass.
pub const ForwardError = error{
    /// A required weight tensor was not found in the model file.
    MissingTensor,
    /// The KV cache has reached its maximum capacity.
    KVCacheFull,
    /// The forward pass was cancelled by another thread.
    Cancelled,
    /// Memory allocation failed.
    OutOfMemory,
    /// No physical blocks available in PagedKvCache.
    OutOfBlocks,
};

/// Model interface — all models implement this via comptime vtable generation.
///
/// Usage: implement `forward`, `prefill`, `resetCache`, `cancel` methods and fields
/// `eos_token_id`, `vocab_size`, `n_layers`, `n_embd`, `n_head`, `n_head_kv`,
/// then call `Model.from(MyModel, &my_instance)`.
pub const Model = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// Function pointer table for polymorphic model dispatch.
    pub const VTable = struct {
        forward: *const fn (self: *anyopaque, token_id: u32) ForwardError!u32,
        prefill: *const fn (self: *anyopaque, token_ids: []const u32) ForwardError!u32,
        reset_cache: *const fn (self: *anyopaque) void,
        cancel: *const fn (self: *anyopaque) void,
        get_eos_id: *const fn (self: *anyopaque) u32,
        get_vocab_size: *const fn (self: *anyopaque) u32,
        get_n_layers: *const fn (self: *anyopaque) u32,
        get_n_embd: *const fn (self: *anyopaque) u32,
        get_n_head: *const fn (self: *anyopaque) u32,
        get_n_head_kv: *const fn (self: *anyopaque) u32,
        get_logits: *const fn (self: *anyopaque) []f32,
        get_block_table: *const fn (self: *anyopaque) []const u32,
    };

    /// Construct a Model interface from any concrete model type at comptime.
    /// The concrete type must have: forward(token_id) !u32, prefill(token_ids) !u32,
    /// resetCache(), cancel(), getBlockTable(), and fields: eos_token_id, vocab_size,
    /// n_layers, n_embd, n_head, n_head_kv, and either `logits` or `logits_buf`.
    pub fn from(comptime T: type, ptr: *T) Model {
        const vtable = comptime genVTable(T);
        return .{ .ptr = ptr, .vtable = vtable };
    }

    fn genVTable(comptime T: type) *const VTable {
        return &comptime .{
            .forward = @ptrCast(&struct {
                fn call(self: *T, token_id: u32) ForwardError!u32 {
                    return self.forward(token_id);
                }
            }.call),
            .prefill = @ptrCast(&struct {
                fn call(self: *T, token_ids: []const u32) ForwardError!u32 {
                    return self.prefill(token_ids);
                }
            }.call),
            .reset_cache = @ptrCast(&struct {
                fn call(self: *T) void {
                    self.resetCache();
                }
            }.call),
            .cancel = @ptrCast(&struct {
                fn call(self: *T) void {
                    self.cancel();
                }
            }.call),
            .get_eos_id = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.eos_token_id;
                }
            }.call),
            .get_vocab_size = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.vocab_size;
                }
            }.call),
            .get_n_layers = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_layers;
                }
            }.call),
            .get_n_embd = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_embd;
                }
            }.call),
            .get_n_head = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_head;
                }
            }.call),
            .get_n_head_kv = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_head_kv;
                }
            }.call),
            .get_logits = @ptrCast(&struct {
                fn call(self: *T) []f32 {
                    return if (@hasField(T, "logits")) self.logits else self.logits_buf;
                }
            }.call),
            .get_block_table = @ptrCast(&struct {
                fn call(self: *T) []const u32 {
                    return self.getBlockTable();
                }
            }.call),
        };
    }

    /// Run one decode step: process `token_id` through all layers,
    /// returning the predicted next-token ID (argmax of logits).
    ///
    /// Parameters:
    ///   - token_id: Input token to process.
    ///
    /// Returns: The predicted next token ID.
    /// Errors: MissingTensor, KVCacheFull, Cancelled, OutOfMemory.
    pub fn forward(self: Model, token_id: u32) ForwardError!u32 {
        return self.vtable.forward(self.ptr, token_id);
    }

    /// Run batched prefill: process all token_ids through all layers,
    /// populating the KV cache. Returns the predicted next-token ID.
    pub fn prefill(self: Model, token_ids: []const u32) ForwardError!u32 {
        return self.vtable.prefill(self.ptr, token_ids);
    }

    /// Return the raw logits buffer from the last forward() call.
    /// Used for temperature-based sampling instead of greedy argmax.
    pub fn getLogits(self: Model) []f32 {
        return self.vtable.get_logits(self.ptr);
    }

    /// Reset the KV cache position to zero, allowing a fresh conversation.
    pub fn resetCache(self: Model) void {
        self.vtable.reset_cache(self.ptr);
    }

    /// Signal the model to cancel the current forward pass.
    /// Checked between layers; the next forward() call returns error.Cancelled.
    pub fn cancel(self: Model) void {
        self.vtable.cancel(self.ptr);
    }

    /// Return the end-of-sequence token ID for this model.
    pub fn eosId(self: Model) u32 {
        return self.vtable.get_eos_id(self.ptr);
    }

    /// Return the vocabulary size.
    pub fn vocabSize(self: Model) u32 {
        return self.vtable.get_vocab_size(self.ptr);
    }

    /// Return the number of transformer layers.
    pub fn nLayers(self: Model) u32 {
        return self.vtable.get_n_layers(self.ptr);
    }

    /// Return the embedding dimension.
    pub fn nEmbd(self: Model) u32 {
        return self.vtable.get_n_embd(self.ptr);
    }

    /// Return the number of query attention heads.
    pub fn nHead(self: Model) u32 {
        return self.vtable.get_n_head(self.ptr);
    }

    /// Return the number of key/value attention heads (for GQA).
    pub fn nHeadKv(self: Model) u32 {
        return self.vtable.get_n_head_kv(self.ptr);
    }

    /// Return the physical block IDs from layer 0 of the current sequence's block table.
    /// Used by the scheduler to populate RadixTree on request completion.
    /// Returns empty slice if no blocks allocated.
    pub fn getBlockTable(self: Model) []const u32 {
        return self.vtable.get_block_table(self.ptr);
    }
};

// ── Shared helpers for model implementations ─────────────────────

/// Reset common inference state (KV cache position + cancellation flag).
/// Models with additional state (e.g. SSM conv/recurrence) should clear
/// that first, then call this.
pub inline fn resetInferenceState(kv_seq_len: *usize, cancelled: *std.atomic.Value(bool)) void {
    kv_seq_len.* = 0;
    cancelled.store(false, .release);
}

/// Compute the byte stride between consecutive experts in a packed weight tensor.
/// GGUF dims are reversed during parsing to [n_experts, rows, cols] (outermost-first).
/// Per-expert stride = weightBytes(rows * cols) = dims[1] * dims[2].
pub fn expertWeightStride(t: format_mod.TensorInfo) usize {
    std.debug.assert(t.n_dims >= 3);
    const elems: usize = @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
}

/// Dispatch GEMV for an mlx_q tensor through the backend's gemvMlxQ path.
/// Looks up companion .scales/.biases tensors and determines bit width.
/// Call this instead of be.gemv() when the tensor may be mlx_q.
/// Returns true if handled, false if the tensor is not mlx_q (caller should use be.gemv).
pub fn mlxGemv(be: backend_mod.Backend, fmt: format_mod.Format, x: [*]const f32, t: format_mod.TensorInfo, y: [*]f32, n: usize, k: usize) bool {
    if (t.dtype != .mlx_q) return false;
    const wi = std.mem.lastIndexOf(u8, t.name, ".weight") orelse return false;
    var sbuf: [tensor_name_buf_size]u8 = undefined;
    var bbuf: [tensor_name_buf_size]u8 = undefined;
    const prefix = t.name[0..wi];
    const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return false;
    const st = fmt.getTensor(s_name) orelse return false;

    if (st.dtype == .unknown) {
        // MXFP4: U8 E8M0 scales, no bias
        be.gemvMxfp4St(x, t.data_ptr, st.data_ptr, y, n, k);
    } else {
        // MLX affine: BF16 scales + biases
        const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return false;
        const bt = fmt.getTensor(b_name) orelse return false;
        // Detect bits per-tensor from weight dimensions: bits = words_per_row * 32 / k.
        // This handles mixed-quant models where default config bits differs from per-layer overrides.
        const bits: u32 = if (t.n_dims >= 2 and k > 0)
            @intCast(@as(u64, t.dims[t.n_dims - 1]) * bits_per_u32_word / @as(u64, @intCast(k)))
        else
            fmt.getMetaU32("bits") orelse 4;
        be.gemvMlxQ(x, t.data_ptr, st.data_ptr, bt.data_ptr, y, n, k, bits);
    }
    return true;
}

/// MLX companion tensor lookup result.
pub const MlxCompanion = struct { scales: [*]const u8, biases: [*]const u8, bits: u32 };

/// Find MLX companion tensors (scales + biases) for an MLX-quantized weight.
/// Returns null for non-MLX tensors, MXFP4 tensors, or when companions are missing.
pub fn findMlxCompanion(fmt: format_mod.Format, t: format_mod.TensorInfo, k: usize) ?MlxCompanion {
    if (t.dtype != .mlx_q) return null;
    const wi = std.mem.lastIndexOf(u8, t.name, ".weight") orelse return null;
    var sbuf: [tensor_name_buf_size]u8 = undefined;
    var bbuf: [tensor_name_buf_size]u8 = undefined;
    const prefix = t.name[0..wi];
    const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return null;
    const st = fmt.getTensor(s_name) orelse return null;
    if (st.dtype == .unknown) return null; // MXFP4 — not affine MLX
    const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return null;
    const bt = fmt.getTensor(b_name) orelse return null;
    const bits: u32 = if (t.n_dims >= 2 and k > 0)
        @intCast(@as(u64, t.dims[t.n_dims - 1]) * bits_per_u32_word / @as(u64, @intCast(k)))
    else
        fmt.getMetaU32("bits") orelse 4;
    return .{ .scales = st.data_ptr, .biases = bt.data_ptr, .bits = bits };
}

/// Dispatch GEMV — tries MLX path first, falls back to standard backend gemv.
/// Use this in models that support both GGUF and SafeTensors MLX weights.
pub fn dispatchGemv(be: backend_mod.Backend, fmt: format_mod.Format, x: [*]const f32, t: format_mod.TensorInfo, y: [*]f32, n: usize, k: usize) void {
    if (mlxGemv(be, fmt, x, t, y, n, k)) return;
    be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
}

/// Signal cancellation of a forward pass (thread-safe).
pub inline fn signalCancel(cancelled: *std.atomic.Value(bool)) void {
    cancelled.store(true, .release);
}

/// Ensure a KV cache block is allocated for the next token position.
/// Called at the start of each forward() pass. If the current sequence
/// length would cross into a new block boundary, allocates one more block
/// from either the tiered or paged block allocator.
///
/// The model struct must have fields: tiered_cache, paged_cache,
/// kv_seq_len, seq_table, tiered_block_allocator, block_allocator.
pub fn ensureKvBlock(self: anytype) !void {
    const bs: usize = if (self.tiered_cache) |tc| tc.block_size else self.paged_cache.block_size;
    const current_blocks = self.seq_table.block_table[0].len;
    const needed_blocks = (self.kv_seq_len + 1 + bs - 1) / bs;
    if (needed_blocks > current_blocks) {
        if (self.tiered_block_allocator) |*ta| {
            try ta.appendBlock(&self.seq_table);
        } else {
            try self.block_allocator.appendBlock(&self.seq_table);
        }
    }
}

/// Reset the paged KV cache for a new conversation: free all blocks in the
/// current sequence table, allocate a fresh table, and append the first block.
/// Also resets kv_seq_len and the cancellation flag.
///
/// The model struct must have fields: tiered_block_allocator, block_allocator,
/// seq_table, n_layers, kv_seq_len, cancelled.
pub fn resetKvCache(self: anytype) void {
    if (self.tiered_block_allocator) |*ta| {
        ta.freeSeqTable(&self.seq_table);
        self.seq_table = ta.allocateSeqTable(self.n_layers) catch return;
        ta.appendBlock(&self.seq_table) catch return;
    } else {
        self.block_allocator.freeSeqTable(&self.seq_table);
        self.seq_table = self.block_allocator.allocateSeqTable(self.n_layers) catch return;
        self.block_allocator.appendBlock(&self.seq_table) catch return;
    }
    resetInferenceState(&self.kv_seq_len, &self.cancelled);
}

// ── Model container ─────────────────────────────────────────────

/// Opaque model container — holds any concrete model type and provides
/// lifecycle and configuration methods without exposing implementation types.
/// Uses `inline else` dispatch for zero-overhead method calls.
pub const ModelStorage = union(enum) {
    gemma3: Gemma3Model,
    gemma4: Gemma4Model,
    qwen35: Qwen35Model,
    gpt_oss: GptOssModel,
    nemotron_h: NemotronHModel,
    nemotron_nano: NemotronNanoModel,
    glm4: Glm4Model,

    /// Initialize a model from its architecture type.
    /// Returns a ModelStorage union holding the initialized concrete model.
    pub fn initFromArch(arch: Arch, allocator: std.mem.Allocator, fmt: format_mod.Format, be: backend_mod.Backend, ctx_size: u32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, tiered_cache: ?*TieredKvCache) !ModelStorage {
        switch (arch) {
            inline .gemma3, .gemma4, .qwen35, .gpt_oss, .nemotron_h, .nemotron_nano, .glm4 => |a| {
                if (comptime !a.isEnabled()) unreachable;
                const M = comptime modelType(a);
                const mdl = try M.init(allocator, fmt, be, ctx_size, kv_type_k, kv_type_v, tiered_cache);
                return @unionInit(ModelStorage, @tagName(a), mdl);
            },
        }
    }

    /// Map Arch variant to concrete model type at comptime.
    fn modelType(comptime a: Arch) type {
        return switch (a) {
            .gemma3 => Gemma3Model,
            .gemma4 => Gemma4Model,
            .qwen35 => Qwen35Model,
            .gpt_oss => GptOssModel,
            .nemotron_h => NemotronHModel,
            .nemotron_nano => NemotronNanoModel,
            .glm4 => Glm4Model,
        };
    }

    /// Get the type-erased Model interface for this model.
    pub fn model(self: *ModelStorage) Model {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) == void) unreachable;
                return Model.from(@TypeOf(m.*), m);
            },
        }
    }

    /// Release all resources owned by this model.
    pub fn deinit(self: *ModelStorage) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) m.deinit();
            },
        }
    }

    /// Set the thread pool reference for CPU parallelism.
    pub fn setPool(self: *ModelStorage, pool: ?*ThreadPool) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "pool")) m.pool = pool;
                }
            },
        }
    }

    /// Fix the block allocator's cache pointer after the struct has been moved.
    pub fn fixBlockAllocator(self: *ModelStorage) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "block_allocator")) {
                        m.block_allocator.setCachePtr(&m.paged_cache);
                    }
                }
            },
        }
    }

    /// Set the prefill chunk size for batched prefill.
    pub fn setChunkSize(self: *ModelStorage, size: u32) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "chunk_size")) m.chunk_size = size;
                }
            },
        }
    }

    /// Enable per-layer performance profiling.
    pub fn enableProfiling(self: *ModelStorage) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "perf")) m.perf.enabled = true;
                }
            },
        }
    }

    /// Print accumulated performance counters.
    pub fn reportPerf(self: *ModelStorage) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "perf")) m.perf.report();
                }
            },
        }
    }
};

// ── Concrete model types (internal — access via ModelStorage) ────

const Gemma3Model = if (build_options.enable_gemma3) @import("gemma3.zig").Gemma3Model else void;
const Gemma4Model = if (build_options.enable_gemma4) @import("gemma4.zig").Gemma4Model else void;
const Qwen35Model = if (build_options.enable_qwen35) @import("qwen35.zig").Qwen35Model else void;
const GptOssModel = if (build_options.enable_gpt_oss) @import("gpt_oss.zig").GptOssModel else void;
const NemotronHModel = if (build_options.enable_nemotron_h) @import("nemotron_h.zig").NemotronHModel else void;
const Glm4Model = if (build_options.enable_glm4) @import("glm4.zig").Glm4Model else void;
const NemotronNanoModel = if (build_options.enable_nemotron_nano) @import("nemotron_nano.zig").NemotronNanoModel else void;

// ── Tests ─────────────────────────────────────────────────────────

test "expertWeightStride f32 2x2 layout" {
    // After GGUF reversal, dims = [n_experts, rows, cols].
    // Per-expert stride = rows * cols * sizeof(f32).
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 2, 4, 4, 0 },
        .dtype = .f32,
        .data_ptr = undefined,
    };
    // 4*4 = 16 elements per expert, 4 bytes each = 64 bytes.
    try std.testing.expectEqual(@as(usize, 64), expertWeightStride(t));
}
