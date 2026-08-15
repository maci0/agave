//! Model interface for LLM inference.
//! Provides a type-erased interface via comptime vtable generation, allowing
//! the engine to work with any model architecture through a uniform API.
//!
//! Implementations: gemma3.zig, gemma4.zig, deepseek4.zig, diffusion_gemma.zig,
//! qwen35.zig, gpt_oss.zig, nemotron_h.zig, nemotron_nano.zig, glm4.zig,
//! llama4.zig, vision.zig

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const Arch = @import("../arch.zig").Arch;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const mlx_ops = @import("../ops/mlx.zig");
const kv_quant = @import("../ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const Transport = @import("../parallel/transport.zig").Transport;

/// Vision encoder for multimodal models (SigLIP-2, CLIP-like image embedding).
pub const VisionEncoder = @import("vision.zig").VisionEncoder;

/// Buffer size for constructing companion tensor names (e.g., ".scales", ".biases").
pub const tensor_name_buf_size: usize = 256;
/// Bits per u32 word — used to compute per-tensor bit width from packed weight dimensions.
const bits_per_u32_word: u64 = 32;
/// Default MLX quantization bit width (4-bit).
pub const default_mlx_bits: u32 = 4;
/// Cached pointer-keyed entry for converted norm weights.
/// Used by all model implementations to cache bf16→f32 norm conversions.
pub const NormCacheEntry = struct { key: usize, data: []f32 };

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
///
/// Required for every architecture: forward, prefill, reset_cache, cancel, dims
/// getters, get_logits, KV seq accessors. Optional features (tree verify, MTP,
/// EAGLE hidden state, SSM snapshot, image embeddings, KV export) get soft
/// no-op / empty-slice stubs from `genVTable` when the concrete type omits them.
/// Prefer stubs over forcing every model to stub optional APIs by hand.
pub const Model = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// Function pointer table for polymorphic model dispatch.
    pub const VTable = struct {
        forward: *const fn (self: *anyopaque, token_id: u32) ForwardError!u32,
        prefill: *const fn (self: *anyopaque, token_ids: []const u32) ForwardError!u32,
        forward_tree: *const fn (self: *anyopaque, token_ids: []const u32, position_ids: []const u32, ancestor_masks: [*]const [8]u64, n_nodes: u32) ForwardError!void,
        tree_logits: *const fn (self: *anyopaque, node_i: u32) u32,
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
        get_kv_seq_len: *const fn (self: *anyopaque) usize,
        set_kv_seq_len: *const fn (self: *anyopaque, len: usize) void,
        prefetch_all_layers: *const fn (self: *anyopaque) void,
        freeze_expert_cache: *const fn (self: *anyopaque) void,
        thaw_expert_cache: *const fn (self: *anyopaque) void,
        set_expert_budget: *const fn (self: *anyopaque, budget: u32) void,
        set_layer_skip: *const fn (self: *anyopaque, start: u32, end: u32) void,
        set_image_embeddings: *const fn (self: *anyopaque, embeddings: ?[]const f32, n_tokens: u32, pad_token_id: u32) void,
        set_thread_context: *const fn (self: *anyopaque) void,
        save_ssm_state: *const fn (self: *anyopaque, allocator: std.mem.Allocator) ?[]u8,
        restore_ssm_state: *const fn (self: *anyopaque, snapshot: []const u8) void,
        mtp_forward: *const fn (self: *anyopaque, token_id: u32, depth: u32) ForwardError!u32,
        get_mtp_depth: *const fn (self: *anyopaque) u32,
        get_mtp_logits: *const fn (self: *anyopaque) []f32,
        reset_mtp_cache: *const fn (self: *anyopaque) void,
        /// Return the last residual hidden state (n_embd floats).
        /// Required by EAGLE speculative decoding: the draft model is conditioned on
        /// the target model's hidden state at each step.
        /// Returns an empty slice if the model doesn't expose hidden states.
        get_hidden_state: *const fn (self: *anyopaque) []const f32,
        /// Return the pre-output-norm hidden state (n_embd floats).
        /// EAGLE-3: conditions on the un-normalized residual stream after all layers
        /// but before the final output_norm. Carries magnitude info post-norm loses.
        /// Falls back to get_hidden_state() if model doesn't save pre-norm state.
        get_pre_norm_hidden_state: *const fn (self: *anyopaque) []const f32,
        /// EAGLE-conditioned forward: run one draft step using the target's hidden state
        /// as additional context. The hidden state is concatenated / added to the token
        /// embedding before the first attention layer (EAGLE-1 approach).
        /// Returns draft token. Falls back to standard forward() if not implemented.
        eagle_forward: *const fn (self: *anyopaque, token_id: u32, context_hidden: []const f32) ForwardError!u32,
        /// Export KV cache for positions [0, kv_seq_len) into a caller-allocated buffer.
        /// Returns the number of bytes written, or 0 if export is unsupported.
        /// Used for cross-instance KV cache sharing (LMCache-style prefix offload).
        /// Soft stub returns 0 when the concrete model omits `exportKvPrefix`
        /// (only Gemma4 implements today). Wire layout is unversioned f32; see
        /// docs/ARCHITECTURE.md Design Decisions and docs/API.md.
        export_kv_prefix: *const fn (self: *anyopaque, dst: []u8, n_tokens: usize) usize,
        /// Import KV cache from a buffer (previously exported by export_kv_prefix).
        /// Sets kv_seq_len to n_tokens on success. Returns false if unsupported.
        /// Soft stub returns false when the concrete model omits `importKvPrefix`.
        import_kv_prefix: *const fn (self: *anyopaque, src: []const u8, n_tokens: usize) bool,
    };

    /// Create a polymorphic Model from a concrete model type.
    /// Required methods: forward, prefill, resetCache, cancel, getBlockTable.
    /// Required fields: eos_token_id, vocab_size, n_layers, n_embd, n_head, n_head_kv, kv_seq_len, logits_buf, be.
    /// Optional methods (via @hasDecl): forwardTree, treeLogits, saveSsmState, restoreSsmState, mtpForward, resetMtpCache.
    /// Optional fields (via @hasField): layer_skip_start, image_embeddings, image_pad_token_id, visual_token_idx, n_mtp_layers, mtp_logits_buf.
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
            .forward_tree = @ptrCast(&struct {
                fn call(self: *T, token_ids: []const u32, position_ids: []const u32, ancestor_masks: [*]const [8]u64, n_nodes: u32) ForwardError!void {
                    if (comptime @hasDecl(T, "forwardTree"))
                        return self.forwardTree(token_ids, position_ids, ancestor_masks, n_nodes);
                    return error.MissingTensor;
                }
            }.call),
            .tree_logits = @ptrCast(&struct {
                fn call(self: *T, node_i: u32) u32 {
                    if (comptime @hasDecl(T, "treeLogits"))
                        return self.treeLogits(node_i);
                    return 0;
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
                    return self.logits_buf;
                }
            }.call),
            .get_block_table = @ptrCast(&struct {
                fn call(self: *T) []const u32 {
                    return self.getBlockTable();
                }
            }.call),
            .get_kv_seq_len = @ptrCast(&struct {
                fn call(self: *T) usize {
                    return self.kv_seq_len;
                }
            }.call),
            .set_kv_seq_len = @ptrCast(&struct {
                fn call(self: *T, len: usize) void {
                    self.kv_seq_len = len;
                }
            }.call),
            .prefetch_all_layers = @ptrCast(&struct {
                fn call(self: *T) void {
                    if (comptime @hasDecl(T, "prefetchAllLayers"))
                        self.prefetchAllLayers();
                }
            }.call),
            .freeze_expert_cache = @ptrCast(&struct {
                fn call(self: *T) void {
                    if (comptime @hasField(T, "expert_cache"))
                        if (self.expert_cache) |ec| ec.freeze();
                }
            }.call),
            .thaw_expert_cache = @ptrCast(&struct {
                fn call(self: *T) void {
                    if (comptime @hasField(T, "expert_cache"))
                        if (self.expert_cache) |ec| ec.thaw();
                }
            }.call),
            .set_expert_budget = @ptrCast(&struct {
                fn call(self: *T, budget: u32) void {
                    if (comptime @hasField(T, "expert_budget"))
                        self.expert_budget = budget;
                }
            }.call),
            .set_layer_skip = @ptrCast(&struct {
                fn call(self: *T, start: u32, end: u32) void {
                    if (comptime @hasField(T, "layer_skip_start")) {
                        self.layer_skip_start = start;
                        self.layer_skip_end = end;
                    }
                }
            }.call),
            .set_image_embeddings = @ptrCast(&struct {
                fn call(self: *T, embeddings: ?[]const f32, n_tokens: u32, pad_token_id: u32) void {
                    if (comptime @hasField(T, "image_embeddings")) {
                        self.image_embeddings = embeddings;
                        self.n_visual_tokens = n_tokens;
                        if (comptime @hasField(T, "image_pad_token_id")) {
                            self.image_pad_token_id = pad_token_id;
                        }
                        // Reset visual token injection counter for new image
                        if (comptime @hasField(T, "visual_token_idx")) {
                            self.visual_token_idx = 0;
                        }
                    }
                }
            }.call),
            .set_thread_context = @ptrCast(&struct {
                fn call(self: *T) void {
                    self.be.setThreadContext();
                }
            }.call),
            .save_ssm_state = @ptrCast(&struct {
                fn call(self: *T, allocator: std.mem.Allocator) ?[]u8 {
                    if (comptime @hasDecl(T, "saveSsmState"))
                        return self.saveSsmState(allocator) catch null;
                    return null;
                }
            }.call),
            .restore_ssm_state = @ptrCast(&struct {
                fn call(self: *T, snapshot: []const u8) void {
                    if (comptime @hasDecl(T, "restoreSsmState"))
                        self.restoreSsmState(snapshot);
                }
            }.call),
            .mtp_forward = @ptrCast(&struct {
                fn call(self: *T, token_id: u32, depth: u32) ForwardError!u32 {
                    if (comptime @hasDecl(T, "mtpForward"))
                        return self.mtpForward(token_id, depth);
                    return error.MissingTensor;
                }
            }.call),
            .get_mtp_depth = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    if (comptime @hasField(T, "n_mtp_layers"))
                        return self.n_mtp_layers;
                    return 0;
                }
            }.call),
            .get_mtp_logits = @ptrCast(&struct {
                fn call(self: *T) []f32 {
                    if (comptime @hasField(T, "mtp_logits_buf"))
                        return self.mtp_logits_buf;
                    return &.{};
                }
            }.call),
            .reset_mtp_cache = @ptrCast(&struct {
                fn call(self: *T) void {
                    if (comptime @hasDecl(T, "resetMtpCache"))
                        self.resetMtpCache();
                }
            }.call),
            .get_hidden_state = @ptrCast(&struct {
                fn call(self: *T) []const f32 {
                    // Models expose hidden state via .hidden field ([]f32, n_embd elements).
                    if (comptime @hasField(T, "hidden")) return self.hidden;
                    return &.{};
                }
            }.call),
            .get_pre_norm_hidden_state = @ptrCast(&struct {
                fn call(self: *T) []const f32 {
                    // Return pre-output-norm hidden if available (EAGLE-3).
                    // Falls back to post-norm .hidden for models that don't save it.
                    if (comptime @hasField(T, "hidden_pre_norm")) {
                        if (self.hidden_pre_norm.len > 0) return self.hidden_pre_norm;
                    }
                    if (comptime @hasField(T, "hidden")) return self.hidden;
                    return &.{};
                }
            }.call),
            .eagle_forward = @ptrCast(&struct {
                fn call(self: *T, token_id: u32, context_hidden: []const f32) ForwardError!u32 {
                    if (comptime @hasDecl(T, "eagleForward"))
                        return self.eagleForward(token_id, context_hidden);
                    return self.forward(token_id);
                }
            }.call),
            .export_kv_prefix = @ptrCast(&struct {
                fn call(self: *T, dst: []u8, n_tokens: usize) usize {
                    if (comptime @hasDecl(T, "exportKvPrefix"))
                        return self.exportKvPrefix(dst, n_tokens);
                    return 0;
                }
            }.call),
            .import_kv_prefix = @ptrCast(&struct {
                fn call(self: *T, src: []const u8, n_tokens: usize) bool {
                    if (comptime @hasDecl(T, "importKvPrefix"))
                        return self.importKvPrefix(src, n_tokens);
                    return false;
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

    /// Batch tree forward: process B tree nodes through all layers using batched
    /// GEMM and tree-masked SDPA. Does NOT modify the main KV cache.
    /// After this call, use `treeLogits(node_i)` to compute logits per node.
    pub fn forwardTree(self: Model, token_ids: []const u32, position_ids: []const u32, ancestor_masks: [*]const [8]u64, n_nodes: u32) ForwardError!void {
        return self.vtable.forward_tree(self.ptr, token_ids, position_ids, ancestor_masks, n_nodes);
    }

    /// Compute logits for a specific tree node (after forwardTree). Returns argmax.
    pub fn treeLogits(self: Model, node_i: u32) u32 {
        return self.vtable.tree_logits(self.ptr, node_i);
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

    /// Make GPU context current on calling thread. Required when forward()
    /// runs on a different thread than the one that initialized the backend.
    pub fn setThreadContext(self: Model) void {
        self.vtable.set_thread_context(self.ptr);
    }

    /// Save SSM state for prefix caching (returns null if model has no SSM layers).
    pub fn saveSsmState(self: Model, allocator: std.mem.Allocator) ?[]u8 {
        return self.vtable.save_ssm_state(self.ptr, allocator);
    }

    /// Restore SSM state from a cached snapshot.
    pub fn restoreSsmState(self: Model, snapshot: []const u8) void {
        self.vtable.restore_ssm_state(self.ptr, snapshot);
    }

    /// Run one MTP head forward pass at the given depth.
    pub fn mtpForward(self: Model, token_id: u32, depth: u32) ForwardError!u32 {
        return self.vtable.mtp_forward(self.ptr, token_id, depth);
    }

    /// Number of MTP prediction depths (0 = no MTP support).
    pub fn getMtpDepth(self: Model) u32 {
        return self.vtable.get_mtp_depth(self.ptr);
    }

    /// Get MTP head logits buffer (valid after mtpForward).
    pub fn getMtpLogits(self: Model) []f32 {
        return self.vtable.get_mtp_logits(self.ptr);
    }

    /// Reset MTP KV cache (on speculation rejection).
    pub fn resetMtpCache(self: Model) void {
        self.vtable.reset_mtp_cache(self.ptr);
    }

    /// Return the last residual hidden state (n_embd floats).
    /// Valid after forward() or prefill(). Empty slice if unsupported.
    pub fn getHiddenState(self: Model) []const f32 {
        return self.vtable.get_hidden_state(self.ptr);
    }

    /// Return the pre-output-norm hidden state (n_embd floats) for EAGLE-3.
    /// Valid after forward(). Falls back to post-norm hidden if not saved.
    pub fn getPreNormHiddenState(self: Model) []const f32 {
        return self.vtable.get_pre_norm_hidden_state(self.ptr);
    }

    /// EAGLE-conditioned draft forward: runs a draft step using the target model's
    /// hidden state as context (concatenated with token embedding, EAGLE-1 style).
    pub fn eagleForward(self: Model, token_id: u32, context_hidden: []const f32) ForwardError!u32 {
        return self.vtable.eagle_forward(self.ptr, token_id, context_hidden);
    }

    /// Export KV cache prefix (positions 0..n_tokens) into dst buffer.
    /// Returns bytes written; 0 if unsupported. Buffer must be large enough.
    /// For cross-instance prefix cache sharing (LMCache-style).
    pub fn exportKvPrefix(self: Model, dst: []u8, n_tokens: usize) usize {
        return self.vtable.export_kv_prefix(self.ptr, dst, n_tokens);
    }

    /// Import KV cache prefix from src buffer; sets kv_seq_len = n_tokens.
    /// Returns true on success. Enables warm-start generation from shared prefix.
    pub fn importKvPrefix(self: Model, src: []const u8, n_tokens: usize) bool {
        return self.vtable.import_kv_prefix(self.ptr, src, n_tokens);
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

    /// Return the current KV cache sequence length (number of tokens processed).
    pub fn kvSeqLen(self: Model) usize {
        return self.vtable.get_kv_seq_len(self.ptr);
    }

    /// Roll back KV cache position for speculative decoding rejection.
    /// Safe because paged blocks stay allocated and are overwritten on next forward().
    /// Set expert budget for MoE-Spec verification mode (arXiv 2602.16052).
    /// budget > 0: fewer experts per token during forward (reduces SSD reads).
    /// budget = 0: normal mode (full n_expert_used).
    pub fn setExpertBudget(self: Model, budget: u32) void {
        self.vtable.set_expert_budget(self.ptr, budget);
    }

    /// Freeze expert cache (no evictions during verification).
    pub fn freezeExpertCache(self: Model) void {
        self.vtable.freeze_expert_cache(self.ptr);
    }

    /// Thaw expert cache (resume normal eviction).
    pub fn thawExpertCache(self: Model) void {
        self.vtable.thaw_expert_cache(self.ptr);
    }

    /// Pre-madvise expert weights for ALL layers before speculative verification.
    pub fn prefetchAllLayers(self: Model) void {
        self.vtable.prefetch_all_layers(self.ptr);
    }

    pub fn setKvSeqLen(self: Model, len: usize) void {
        self.vtable.set_kv_seq_len(self.ptr, len);
    }

    /// Enable layer skipping for self-speculative draft mode.
    /// Layers in [start, end) are skipped during forward().
    /// Call with (0, 0) to disable.
    pub fn setLayerSkip(self: Model, start: u32, end: u32) void {
        self.vtable.set_layer_skip(self.ptr, start, end);
    }

    /// Set visual token embeddings for multimodal inference.
    /// The embeddings slice must contain n_tokens * n_embd f32 values
    /// (already projected to the model's hidden dimension).
    /// `pad_token_id` is the token ID used as placeholder in the input sequence
    /// (the model replaces these with visual embeddings during forward()).
    /// Pass null embeddings to clear (return to text-only mode).
    pub fn setImageEmbeddings(self: Model, embeddings: ?[]const f32, n_tokens: u32, pad_token_id: u32) void {
        self.vtable.set_image_embeddings(self.ptr, embeddings, n_tokens, pad_token_id);
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
    if (t.n_dims < 3) @panic("expertWeightStride: expected >= 3D tensor for expert weights");
    // Compressed-tensors NVFP4: dims = [rows, cols_packed, n_experts] where
    // cols_packed = in_dim/2 (raw U8 bytes). Stride = rows × cols_packed bytes.
    // Unlike GGUF NVFP4 (interleaved blocks), weight and scale are separate arrays.
    if (t.dtype == .nvfp4) {
        return @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
    }
    // GGUF 3D expert tensors: dims stored as [n_experts, rows, cols].
    // Per-expert weight size = dims[1] × dims[2] (rows × cols).
    // dims[0] is the expert count.
    const elems: usize = @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
}

/// Compute the byte stride between consecutive experts, handling both MLX (U32
/// packed) and GGUF/standard weight formats.
pub fn expertStride(t: format_mod.TensorInfo) usize {
    if (t.dtype == .mlx_q) {
        // SafeTensors MLX dims (not reversed): [n_experts, rows, words_per_row] U32
        if (t.n_dims < 3) @panic("expertStride: expected >= 3D tensor for MLX expert weights");
        return @as(usize, @intCast(t.dims[1])) * @as(usize, @intCast(t.dims[2])) * @sizeOf(u32);
    }
    return expertWeightStride(t);
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
            fmt.getMetaU32("bits") orelse default_mlx_bits;
        const group_size = inferMlxGroupSize(st, k);
        be.gemvMlxQ(x, t.data_ptr, st.data_ptr, bt.data_ptr, y, n, k, bits, group_size);
    }
    return true;
}

/// MLX companion tensor lookup result.
pub const MlxCompanion = struct { scales: [*]const u8, biases: [*]const u8, bits: u32, group_size: u32 };

/// Infer MLX group size from the scales tensor shape.
/// Returns k / scales_last_dim, or default 64 if dims are unavailable.
pub fn inferMlxGroupSize(st: format_mod.TensorInfo, k: usize) u32 {
    if (st.n_dims >= 2 and k > 0) {
        const n_groups: usize = @intCast(st.dims[st.n_dims - 1]);
        if (n_groups > 0) return @intCast(k / n_groups);
    }
    return @intCast(mlx_ops.mlx_group_size);
}

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
        fmt.getMetaU32("bits") orelse default_mlx_bits;
    const group_size = inferMlxGroupSize(st, k);
    return .{ .scales = st.data_ptr, .biases = bt.data_ptr, .bits = bits, .group_size = group_size };
}

/// Dispatch GEMV — tries MLX path first, falls back to standard backend gemv.
/// Use this in models that support both GGUF and SafeTensors MLX weights.
pub fn dispatchGemv(be: backend_mod.Backend, fmt: format_mod.Format, x: [*]const f32, t: format_mod.TensorInfo, y: [*]f32, n: usize, k: usize) void {
    if (mlxGemv(be, fmt, x, t, y, n, k)) return;
    if (t.dtype == .nvfp4) {
        // Compressed-tensors NVFP4: separate weight + scale tensors.
        // Build companion scale tensor name: replace ".weight" with ".scales"
        var name_buf: [tensor_name_buf_size]u8 = undefined;
        const dot_pos = std.mem.lastIndexOfScalar(u8, t.name, '.') orelse {
            be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
            return;
        };
        const s_name = std.fmt.bufPrint(&name_buf, "{s}.scales", .{t.name[0..dot_pos]}) catch {
            be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
            return;
        };
        if (fmt.getTensor(s_name)) |scales| {
            be.gemvNvfp4St(x, t.data_ptr, scales.data_ptr, y, n, k);
            be.sync();
            // NVFP4: GEMV output = sum(e2m1 * fp8_scale * x), divide by weight_global_scale
            const gs_name = std.fmt.bufPrint(&name_buf, "{s}.global_scale", .{t.name[0..dot_pos]}) catch "";
            if (gs_name.len > 0) {
                if (fmt.getTensor(gs_name)) |gs_t| {
                    const gs = @as(*const f32, @ptrCast(@alignCast(gs_t.data_ptr))).*;
                    if (gs != 1.0 and gs != 0.0) {
                        const inv = 1.0 / gs;
                        var yi: usize = 0;
                        while (yi < n) : (yi += 1) {
                            y[yi] *= inv;
                        }
                    }
                }
            }
        } else {
            be.gemv(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, n, k);
        }
        return;
    }
    if (t.dtype == .gptq or t.dtype == .awq) {
        // GPTQ/AWQ: INT4 packed weights + FP16 scales + INT4 packed zeros
        const base_name = blk: {
            if (std.mem.endsWith(u8, t.name, ".qweight")) {
                break :blk t.name[0 .. t.name.len - ".qweight".len];
            }
            break :blk t.name[0..(std.mem.lastIndexOfScalar(u8, t.name, '.') orelse t.name.len)];
        };
        var s_buf: [tensor_name_buf_size]u8 = undefined;
        var z_buf: [tensor_name_buf_size]u8 = undefined;
        const s_name = std.fmt.bufPrint(&s_buf, "{s}.scales", .{base_name}) catch "";
        const z_name = std.fmt.bufPrint(&z_buf, "{s}.qzeros", .{base_name}) catch "";
        const scales_t = if (s_name.len > 0) fmt.getTensor(s_name) else null;
        const zeros_t = if (z_name.len > 0) fmt.getTensor(z_name) else null;
        if (scales_t) |st| {
            const group_size = fmt.getMetaU32("group_size") orelse 128;
            const zeros_ptr: [*]const u32 = if (zeros_t) |zt|
                @ptrCast(@alignCast(zt.data_ptr))
            else
                @ptrCast(@alignCast(st.data_ptr));
            if (t.dtype == .awq) {
                be.gemvAwq(x, @ptrCast(@alignCast(t.data_ptr)), @ptrCast(@alignCast(st.data_ptr)), zeros_ptr, y, n, k, group_size);
            } else {
                // GPTQ: row-major packing (8 input elements per INT32 word)
                be.gemvGptq(x, @ptrCast(@alignCast(t.data_ptr)), @ptrCast(@alignCast(st.data_ptr)), zeros_ptr, y, n, k, group_size);
            }
            return;
        }
    }
    if (t.dtype == .hqq) {
        // HQQ: uint8 packed nibbles + bf16 companion scale/zero from meta.* tensors.
        const base_name = if (std.mem.endsWith(u8, t.name, ".W_q"))
            t.name[0 .. t.name.len - ".W_q".len]
        else
            t.name[0..(std.mem.lastIndexOfScalar(u8, t.name, '.') orelse t.name.len)];
        var s_buf: [tensor_name_buf_size]u8 = undefined;
        var z_buf: [tensor_name_buf_size]u8 = undefined;
        const s_name = std.fmt.bufPrint(&s_buf, "{s}.meta.scale", .{base_name}) catch "";
        const z_name = std.fmt.bufPrint(&z_buf, "{s}.meta.zero", .{base_name}) catch "";
        const scales_t = if (s_name.len > 0) fmt.getTensor(s_name) else null;
        const zeros_t = if (z_name.len > 0) fmt.getTensor(z_name) else null;
        if (scales_t != null and zeros_t != null) {
            // Derive group_size from scale tensor shape: scale[n_out, k/group_size]
            // dims[1] = k / group_size → group_size = k / dims[1]
            const group_size: u32 = blk: {
                const st = scales_t.?;
                if (st.n_dims >= 2 and st.dims[1] > 0 and k > 0) {
                    const gs = k / @as(usize, st.dims[1]);
                    if (gs > 0) break :blk @intCast(gs);
                }
                break :blk fmt.getMetaU32("group_size") orelse 64;
            };
            be.gemvHqq(x, t.data_ptr, scales_t.?.data_ptr, zeros_t.?.data_ptr, y, n, k, group_size);
            return;
        }
    }
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
        self.seq_table = ta.allocateSeqTable(self.n_layers) catch |err| {
            std.log.err("KV cache reset failed (tiered allocateSeqTable): {s}", .{@errorName(err)});
            resetInferenceState(&self.kv_seq_len, &self.cancelled);
            return;
        };
        ta.appendBlock(&self.seq_table) catch |err| {
            std.log.err("KV cache reset failed (tiered appendBlock): {s}", .{@errorName(err)});
            resetInferenceState(&self.kv_seq_len, &self.cancelled);
            return;
        };
    } else {
        self.block_allocator.freeSeqTable(&self.seq_table);
        self.seq_table = self.block_allocator.allocateSeqTable(self.n_layers) catch |err| {
            std.log.err("KV cache reset failed (allocateSeqTable): {s}", .{@errorName(err)});
            resetInferenceState(&self.kv_seq_len, &self.cancelled);
            return;
        };
        self.block_allocator.appendBlock(&self.seq_table) catch |err| {
            std.log.err("KV cache reset failed (appendBlock): {s}", .{@errorName(err)});
            resetInferenceState(&self.kv_seq_len, &self.cancelled);
            return;
        };
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
    diffusion_gemma: DiffusionGemmaModel,
    qwen35: Qwen35Model,
    gpt_oss: GptOssModel,
    nemotron_h: NemotronHModel,
    nemotron_nano: NemotronNanoModel,
    glm4: Glm4Model,
    deepseek4: Ds4Model,
    llama4: Llama4Model,

    /// Initialize a model from its architecture type.
    /// Returns a ModelStorage union holding the initialized concrete model.
    pub fn initFromArch(arch: Arch, allocator: std.mem.Allocator, fmt: format_mod.Format, be: backend_mod.Backend, ctx_size: u32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, kv_boundary_v: u32, kv_eviction_budget: u32, tiered_cache: ?*TieredKvCache, tp_rank: u32, tp_degree: u32) !ModelStorage {
        switch (arch) {
            inline .gemma3, .gemma4, .diffusion_gemma, .qwen35, .gpt_oss, .nemotron_h, .nemotron_nano, .glm4, .deepseek4, .llama4 => |a| {
                if (comptime !a.isEnabled()) unreachable;
                const M = comptime modelType(a);
                var mdl = try M.init(allocator, fmt, be, ctx_size, kv_type_k, kv_type_v, tiered_cache);
                // Set boundary V protection: first/last N layers use f16 for V
                if (comptime @hasField(M, "kv_boundary_v")) {
                    mdl.kv_boundary_v = kv_boundary_v;
                }
                if (comptime @hasField(M, "kv_eviction_budget")) {
                    mdl.kv_eviction_budget = kv_eviction_budget;
                }
                if (comptime @hasField(M, "tp_rank")) {
                    mdl.tp_rank = tp_rank;
                    mdl.tp_degree = tp_degree;
                    // Note: attention TP needs per-rank KV caches (not yet implemented).
                    // FFN TP works with the shared KV cache (attention runs TP=1).
                }
                return @unionInit(ModelStorage, @tagName(a), mdl);
            },
        }
    }

    /// Map Arch variant to concrete model type at comptime.
    fn modelType(comptime a: Arch) type {
        return switch (a) {
            .gemma3 => Gemma3Model,
            .gemma4 => Gemma4Model,
            .diffusion_gemma => DiffusionGemmaModel,
            .qwen35 => Qwen35Model,
            .gpt_oss => GptOssModel,
            .nemotron_h => NemotronHModel,
            .nemotron_nano => NemotronNanoModel,
            .glm4 => Glm4Model,
            .deepseek4 => Ds4Model,
            .llama4 => Llama4Model,
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

    /// Set the SSD streaming expert cache and optional activation profiler.
    /// Pre-madvise expert weights for ALL layers before speculative verification.
    /// Warms the page cache so verification forwards hit cached expert pages.
    pub fn prefetchAllLayers(self: *ModelStorage) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasDecl(@TypeOf(m.*), "prefetchAllLayers"))
                        m.prefetchAllLayers();
                }
            },
        }
    }

    pub fn setExpertCache(self: *ModelStorage, cache: *@import("../expert_cache.zig").ExpertCache, profile: ?*@import("../expert_profile.zig").ExpertProfile) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "expert_cache"))
                        m.expert_cache = cache;
                    if (comptime @hasField(@TypeOf(m.*), "expert_profile"))
                        m.expert_profile = profile;
                }
            },
        }
    }

    /// Set MTP (multi-token prediction) weights from a separate safetensors file.
    pub fn setMtpWeights(self: *ModelStorage, mtp: *@import("ds4_mtp.zig").MtpWeights) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "mtp_weights")) {
                        m.mtp_weights = mtp;
                        m.n_mtp_layers = mtp.n_depths;
                    }
                }
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

    /// Set directional steering for runtime activation editing.
    pub fn setSteering(self: *ModelStorage, steer: *const @import("../steering.zig").DirectionalSteering) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "steering"))
                        m.steering = steer
                    else
                        std.log.warn("steering not supported for this model architecture", .{});
                }
            },
        }
    }

    /// Set the TP row-shard scratch buffer for weight column extraction.
    pub fn setTpRowShardBuf(self: *ModelStorage, buf: []u8) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "tp_row_shard_buf")) m.tp_row_shard_buf = buf;
                }
            },
        }
    }

    /// Set the network transport for distributed TP all-reduce.
    pub fn setTpTransport(self: *ModelStorage, transport: *Transport) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "tp_transport")) m.tp_transport = transport;
                }
            },
        }
    }

    /// Send KV cache via transport (disaggregated prefill).
    pub fn sendKvCache(self: *ModelStorage, transport: *Transport) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasDecl(@TypeOf(m.*), "sendKvCache")) m.sendKvCache(transport);
                }
            },
        }
    }

    /// Receive KV cache via transport (disaggregated decode).
    pub fn recvKvCache(self: *ModelStorage, transport: *Transport) !void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasDecl(@TypeOf(m.*), "recvKvCache")) try m.recvKvCache(transport);
                }
            },
        }
    }

    /// Set PP config and transport.
    pub fn setPpConfig(self: *ModelStorage, rank: u32, degree: u32, transport: ?*Transport) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "pp_rank")) {
                        m.pp_rank = rank;
                        m.pp_degree = degree;
                        m.pp_transport = transport;
                    }
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

    /// Enable megakernel mode for fused single-dispatch forward pass.
    pub fn setMegakernel(self: *ModelStorage, enabled: bool) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "megakernel_enabled")) m.megakernel_enabled = enabled;
                }
            },
        }
    }

    const TriCalibration = @import("../ops/kv_evict.zig").TriCalibration;

    /// Set TriAttention calibration data for frequency-domain KV eviction.
    pub fn setTriCalibration(self: *ModelStorage, cals: []const TriCalibration) void {
        switch (self.*) {
            inline else => |*m| {
                if (@TypeOf(m.*) != void) {
                    if (comptime @hasField(@TypeOf(m.*), "tri_calibrations")) {
                        m.tri_calibrations = cals;
                    }
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
const DiffusionGemmaModel = if (build_options.enable_diffusion_gemma) @import("diffusion_gemma.zig").DiffusionGemmaModel else void;
const Qwen35Model = if (build_options.enable_qwen35) @import("qwen35.zig").Qwen35Model else void;
const GptOssModel = if (build_options.enable_gpt_oss) @import("gpt_oss.zig").GptOssModel else void;
const NemotronHModel = if (build_options.enable_nemotron_h) @import("nemotron_h.zig").NemotronHModel else void;
const Glm4Model = if (build_options.enable_glm4) @import("glm4.zig").Glm4Model else void;
const Ds4Model = if (build_options.enable_deepseek4) @import("deepseek4.zig").Ds4Model else void;
const NemotronNanoModel = if (build_options.enable_nemotron_nano) @import("nemotron_nano.zig").NemotronNanoModel else void;
const Llama4Model = if (build_options.enable_llama4) @import("llama4.zig").Llama4Model else void;

// ── Tests ─────────────────────────────────────────────────────────

test "expertWeightStride f32 2x2 layout" {
    // GGUF 3D expert tensor: dims = [n_experts, rows, cols].
    // Per-expert stride = dims[1] * dims[2] elements × sizeof(dtype).
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 2, 4, 4, 0 }, // 2 experts × 4 rows × 4 cols
        .dtype = .f32,
        .data_ptr = undefined,
    };
    // 4*4 = 16 elements per expert, 4 bytes each = 64 bytes.
    try std.testing.expectEqual(@as(usize, 64), expertWeightStride(t));
}

test "expertWeightStride q4_k quantized" {
    // Q4_K: super-block = 256 elems, 144 bytes per super-block.
    // dims = [8 experts, 256 rows, 512 cols], elems = 256×512 = 131072
    // nsb = 131072/256 = 512, stride = 512 × 144 = 73728
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_gate_exps.weight",
        .n_dims = 3,
        .dims = .{ 8, 256, 512, 0 },
        .dtype = .q4_k,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(@as(usize, 73728), expertWeightStride(t));
}

test "expertWeightStride q8_0 quantized" {
    // Q8_0: block = 32 elems, 34 bytes per block.
    // dims = [4, 128, 256], elems = 128×256 = 32768
    // nb = 32768/32 = 1024, stride = 1024 × 34 = 34816
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_up_exps.weight",
        .n_dims = 3,
        .dims = .{ 4, 128, 256, 0 },
        .dtype = .q8_0,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(@as(usize, 34816), expertWeightStride(t));
}

test "expertWeightStride f16" {
    // F16: 2 bytes per element. dims = [16, 64, 128], elems = 64×128 = 8192
    // stride = 8192 × 2 = 16384
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 16, 64, 128, 0 },
        .dtype = .f16,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(@as(usize, 16384), expertWeightStride(t));
}

test "expertWeightStride nvfp4 compressed-tensors" {
    // NVFP4: dims = [rows, cols_packed, n_experts], stride = rows × cols_packed.
    // Unlike GGUF, compressed-tensors NVFP4 uses raw byte layout.
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_gate_exps.weight",
        .n_dims = 3,
        .dims = .{ 128, 64, 4, 0 }, // 128 rows × 64 packed cols × 4 experts
        .dtype = .nvfp4,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(@as(usize, 8192), expertWeightStride(t));
}

test "expertWeightStride 4D tensor uses inner dims" {
    // 4D: dims = [batch, n_experts, rows, cols]. Only dims[1]×dims[2] used.
    // This tests the n_dims >= 3 path still uses dims[1] and dims[2].
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 4,
        .dims = .{ 2, 8, 32, 64 },
        .dtype = .f32,
        .data_ptr = undefined,
    };
    // weightBytes(.f32, 1, 8*32) = 8*32*4 = 1024
    try std.testing.expectEqual(@as(usize, 1024), expertWeightStride(t));
}

// ── expertStride tests ───────────────────────────────────────────

test "expertStride mlx_q U32 packed" {
    // MLX: dims = [n_experts, rows, words_per_row] as U32.
    // Stride = rows × words_per_row × sizeof(u32).
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_gate_exps.weight",
        .n_dims = 3,
        .dims = .{ 8, 256, 32, 0 }, // 8 experts × 256 rows × 32 u32 words
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    // 256 × 32 × 4 = 32768
    try std.testing.expectEqual(@as(usize, 32768), expertStride(t));
}

test "expertStride non-mlx delegates to expertWeightStride" {
    // For non-MLX dtypes, expertStride should return same as expertWeightStride.
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 2, 4, 4, 0 },
        .dtype = .f32,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(expertWeightStride(t), expertStride(t));
}

test "expertStride nvfp4 delegates to expertWeightStride" {
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 128, 64, 4, 0 },
        .dtype = .nvfp4,
        .data_ptr = undefined,
    };
    try std.testing.expectEqual(expertWeightStride(t), expertStride(t));
}

// ── resetInferenceState tests ────────────────────────────────────

test "resetInferenceState clears position and cancel flag" {
    var kv_seq_len: usize = 42;
    var cancelled = std.atomic.Value(bool).init(true);
    resetInferenceState(&kv_seq_len, &cancelled);
    try std.testing.expectEqual(@as(usize, 0), kv_seq_len);
    try std.testing.expectEqual(false, cancelled.load(.acquire));
}

test "resetInferenceState idempotent on already-cleared state" {
    var kv_seq_len: usize = 0;
    var cancelled = std.atomic.Value(bool).init(false);
    resetInferenceState(&kv_seq_len, &cancelled);
    try std.testing.expectEqual(@as(usize, 0), kv_seq_len);
    try std.testing.expectEqual(false, cancelled.load(.acquire));
}

// ── signalCancel tests ───────────────────────────────────────────

test "signalCancel sets flag" {
    var cancelled = std.atomic.Value(bool).init(false);
    signalCancel(&cancelled);
    try std.testing.expectEqual(true, cancelled.load(.acquire));
}

test "signalCancel is idempotent" {
    var cancelled = std.atomic.Value(bool).init(true);
    signalCancel(&cancelled);
    try std.testing.expectEqual(true, cancelled.load(.acquire));
}

// ── findMlxCompanion tests (with mock Format) ────────────────────

/// Mock format that returns configurable tensors by name.
/// Implements the Format vtable for testing companion tensor lookup.
/// Public so model implementation tests (e.g. qwen35.zig) can reuse it.
pub const MockFormat = struct {
    tensors: []const NamedTensor,
    meta_bits: ?u32 = null,

    const NamedTensor = struct { name: []const u8, info: format_mod.TensorInfo };

    fn getTensorFn(self_ptr: *anyopaque, name: []const u8) ?format_mod.TensorInfo {
        const self: *MockFormat = @ptrCast(@alignCast(self_ptr));
        for (self.tensors) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.info;
        }
        return null;
    }

    fn getMetaU32Fn(self_ptr: *anyopaque, key: []const u8) ?u32 {
        const self: *MockFormat = @ptrCast(@alignCast(self_ptr));
        if (std.mem.eql(u8, key, "bits")) return self.meta_bits;
        return null;
    }

    fn nullStrFn(_: *anyopaque, _: []const u8) ?[]const u8 {
        return null;
    }
    fn nullF32Fn(_: *anyopaque, _: []const u8) ?f32 {
        return null;
    }
    fn nullU32ArrayFn(_: *anyopaque, _: []const u8) ?[]const u32 {
        return null;
    }
    fn nullVocabFn(_: *anyopaque) ?[]const []const u8 {
        return null;
    }
    fn nullMergesFn(_: *anyopaque) ?[]const []const u8 {
        return null;
    }

    const vtable = format_mod.Format.VTable{
        .get_tensor = @ptrCast(&getTensorFn),
        .get_meta_str = @ptrCast(&nullStrFn),
        .get_meta_u32 = @ptrCast(&getMetaU32Fn),
        .get_meta_f32 = @ptrCast(&nullF32Fn),
        .get_meta_u32_array = @ptrCast(&nullU32ArrayFn),
        .get_vocab = @ptrCast(&nullVocabFn),
        .get_merges = @ptrCast(&nullMergesFn),
    };

    pub fn format(self: *MockFormat) format_mod.Format {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

// ── MockModel for vtable dispatch tests ─────────────────────────

/// Minimal mock model that satisfies the Model.from() comptime requirements.
/// Used to test vtable dispatch without requiring a real backend or format.
const MockModel = struct {
    eos_token_id: u32 = 42,
    vocab_size: u32 = 1000,
    n_layers: u32 = 8,
    n_embd: u32 = 256,
    n_head: u32 = 4,
    n_head_kv: u32 = 2,
    kv_seq_len: usize = 10,
    logits_buf: []f32 = &.{},
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    be: MockBackend = .{},
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    image_embeddings: ?[]const f32 = null,
    n_visual_tokens: u32 = 0,
    image_pad_token_id: u32 = 0,
    visual_token_idx: u32 = 0,
    cache_reset_count: u32 = 0,
    ssm_restore_count: u32 = 0,
    mtp_reset_count: u32 = 0,

    const MockBackend = struct {
        thread_context_count: u32 = 0,
        pub fn setThreadContext(self: *MockBackend) void {
            self.thread_context_count += 1;
        }
    };

    fn forward(_: *MockModel, _: u32) ForwardError!u32 {
        return 7;
    }
    fn prefill(_: *MockModel, _: []const u32) ForwardError!u32 {
        return 7;
    }
    fn resetCache(self: *MockModel) void {
        self.cache_reset_count += 1;
    }
    fn cancel(self: *MockModel) void {
        signalCancel(&self.cancelled);
    }
    fn getBlockTable(_: *MockModel) []const u32 {
        return &.{};
    }
    fn restoreSsmState(self: *MockModel, _: []const u8) void {
        self.ssm_restore_count += 1;
    }
    fn resetMtpCache(self: *MockModel) void {
        self.mtp_reset_count += 1;
    }
};

test "Model.from and vtable dispatch — eosId" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 42), m.eosId());
}

test "Model.from and vtable dispatch — vocabSize" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 1000), m.vocabSize());
}

test "Model.from and vtable dispatch — nLayers" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 8), m.nLayers());
}

test "Model.from and vtable dispatch — nEmbd" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 256), m.nEmbd());
}

test "Model.from and vtable dispatch — nHead" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 4), m.nHead());
}

test "Model.from and vtable dispatch — nHeadKv" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 2), m.nHeadKv());
}

test "Model.from and vtable dispatch — kvSeqLen and setKvSeqLen" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(usize, 10), m.kvSeqLen());
    m.setKvSeqLen(5);
    try std.testing.expectEqual(@as(usize, 5), m.kvSeqLen());
}

test "Model.from and vtable dispatch — getLogits empty" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(usize, 0), m.getLogits().len);
}

test "Model.from and vtable dispatch — getLogits with buffer" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    var mock = MockModel{ .logits_buf = &logits };
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(usize, 3), m.getLogits().len);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), m.getLogits()[1], 1e-6);
}

test "Model.from and vtable dispatch — forward" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    const result = try m.forward(0);
    try std.testing.expectEqual(@as(u32, 7), result);
}

test "Model.from and vtable dispatch — prefill" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    const ids = [_]u32{ 1, 2, 3 };
    const result = try m.prefill(&ids);
    try std.testing.expectEqual(@as(u32, 7), result);
}

test "Model.from and vtable dispatch — resetCache" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.resetCache();
    try std.testing.expectEqual(@as(u32, 1), mock.cache_reset_count);
}

test "Model.from and vtable dispatch — cancel" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.cancel();
    try std.testing.expectEqual(true, mock.cancelled.load(.acquire));
}

test "Model.from and vtable dispatch — setThreadContext" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.setThreadContext();
    try std.testing.expectEqual(@as(u32, 1), mock.be.thread_context_count);
}

test "Model.from and vtable dispatch — getBlockTable" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(usize, 0), m.getBlockTable().len);
}

test "Model.from and vtable dispatch — setLayerSkip" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.setLayerSkip(2, 6);
    try std.testing.expectEqual(@as(u32, 2), mock.layer_skip_start);
    try std.testing.expectEqual(@as(u32, 6), mock.layer_skip_end);
}

test "Model.from and vtable dispatch — setImageEmbeddings" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    var embd = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    m.setImageEmbeddings(&embd, 2, 99);
    try std.testing.expectEqual(@as(u32, 2), mock.n_visual_tokens);
    try std.testing.expectEqual(@as(u32, 99), mock.image_pad_token_id);
    try std.testing.expectEqual(@as(u32, 0), mock.visual_token_idx);
    // Clear
    m.setImageEmbeddings(null, 0, 0);
    try std.testing.expect(mock.image_embeddings == null);
}

test "Model.from and vtable dispatch — forwardTree returns MissingTensor for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    // MockModel has no forwardTree → vtable returns error.MissingTensor
    try std.testing.expectError(error.MissingTensor, m.forwardTree(&.{}, &.{}, @ptrFromInt(0x1000), 0));
}

test "Model.from and vtable dispatch — treeLogits returns 0 for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    // MockModel has no treeLogits → vtable returns 0
    try std.testing.expectEqual(@as(u32, 0), m.treeLogits(0));
}

test "Model.from and vtable dispatch — saveSsmState returns null for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expect(m.saveSsmState(std.testing.allocator) == null);
}

test "Model.from and vtable dispatch — restoreSsmState no-op for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.restoreSsmState(&[_]u8{ 1, 2, 3 });
    try std.testing.expectEqual(@as(u32, 1), mock.ssm_restore_count);
}

test "Model.from and vtable dispatch — getMtpDepth returns 0 for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(u32, 0), m.getMtpDepth());
}

test "Model.from and vtable dispatch — getMtpLogits returns empty for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectEqual(@as(usize, 0), m.getMtpLogits().len);
}

test "Model.from and vtable dispatch — resetMtpCache no-op for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    m.resetMtpCache();
    try std.testing.expectEqual(@as(u32, 1), mock.mtp_reset_count);
}

test "Model.from and vtable dispatch — mtpForward returns MissingTensor for mock" {
    var mock = MockModel{};
    const m = Model.from(MockModel, &mock);
    try std.testing.expectError(error.MissingTensor, m.mtpForward(0, 0));
}

// ── mlxGemv tests ───────────────────────────────────────────────

test "mlxGemv returns false for non-mlx tensor" {
    var cpu = @import("../backend/backend.zig").CpuBackend{};
    var mock = MockFormat{ .tensors = &.{} };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 4, 4, 0, 0 },
        .dtype = .f32,
        .data_ptr = undefined,
    };
    var y: [4]f32 = undefined;
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const handled = mlxGemv(.{ .cpu = &cpu }, mock.format(), &x, t, &y, 4, 4);
    try std.testing.expect(!handled);
}

test "mlxGemv returns false for name without .weight" {
    var cpu = @import("../backend/backend.zig").CpuBackend{};
    var mock = MockFormat{ .tensors = &.{} };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.bias",
        .n_dims = 2,
        .dims = .{ 4, 4, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    var y: [4]f32 = undefined;
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const handled = mlxGemv(.{ .cpu = &cpu }, mock.format(), &x, t, &y, 4, 4);
    try std.testing.expect(!handled);
}

test "findMlxCompanion returns null for non-mlx tensor" {
    var mock = MockFormat{ .tensors = &.{} };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 1024, 0, 0 },
        .dtype = .q4_k,
        .data_ptr = undefined,
    };
    try std.testing.expect(findMlxCompanion(mock.format(), t, 1024) == null);
}

test "findMlxCompanion returns null for name without .weight suffix" {
    var mock = MockFormat{ .tensors = &.{} };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.bias",
        .n_dims = 2,
        .dims = .{ 1024, 32, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    try std.testing.expect(findMlxCompanion(mock.format(), t, 1024) == null);
}

test "findMlxCompanion returns null when scales tensor is missing" {
    var mock = MockFormat{ .tensors = &.{} };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 128, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    try std.testing.expect(findMlxCompanion(mock.format(), t, 1024) == null);
}

test "findMlxCompanion returns null for MXFP4 (scales dtype .unknown)" {
    // MXFP4 scales use .unknown dtype (U8 E8M0) — should not be treated as MLX affine.
    var dummy: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.scales", .info = .{
            .name = "blk.0.attn_q.scales",
            .n_dims = 2,
            .dims = .{ 1024, 32, 0, 0 },
            .dtype = .unknown,
            .data_ptr = @ptrCast(&dummy),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 128, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    try std.testing.expect(findMlxCompanion(mock.format(), t, 1024) == null);
}

test "findMlxCompanion returns null when biases tensor is missing" {
    var dummy: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.scales", .info = .{
            .name = "blk.0.attn_q.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&dummy),
        } },
        // no biases tensor
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 128, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    try std.testing.expect(findMlxCompanion(mock.format(), t, 1024) == null);
}

test "findMlxCompanion 4-bit: bits computed from dims" {
    // 4-bit MLX: k=1024, words_per_row = 1024*4/32 = 128
    // bits = 128 * 32 / 1024 = 4
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.scales", .info = .{
            .name = "blk.0.attn_q.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.attn_q.biases", .info = .{
            .name = "blk.0.attn_q.biases",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 128, 0, 0 }, // 128 u32 words per row
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 1024).?;
    try std.testing.expectEqual(@as(u32, 4), companion.bits);
    try std.testing.expectEqual(@as([*]const u8, @ptrCast(&scales_data)), companion.scales);
    try std.testing.expectEqual(@as([*]const u8, @ptrCast(&biases_data)), companion.biases);
}

test "findMlxCompanion 8-bit: bits computed from dims" {
    // 8-bit MLX: k=1024, words_per_row = 1024*8/32 = 256
    // bits = 256 * 32 / 1024 = 8
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.ffn_gate.scales", .info = .{
            .name = "blk.0.ffn_gate.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.ffn_gate.biases", .info = .{
            .name = "blk.0.ffn_gate.biases",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_gate.weight",
        .n_dims = 2,
        .dims = .{ 1024, 256, 0, 0 }, // 256 u32 words per row
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 1024).?;
    try std.testing.expectEqual(@as(u32, 8), companion.bits);
}

test "findMlxCompanion 2-bit: bits computed from dims" {
    // 2-bit MLX: k=1024, words_per_row = 1024*2/32 = 64
    // bits = 64 * 32 / 1024 = 2
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_v.scales", .info = .{
            .name = "blk.0.attn_v.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.attn_v.biases", .info = .{
            .name = "blk.0.attn_v.biases",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_v.weight",
        .n_dims = 2,
        .dims = .{ 1024, 64, 0, 0 }, // 64 u32 words per row
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 1024).?;
    try std.testing.expectEqual(@as(u32, 2), companion.bits);
}

test "findMlxCompanion falls back to metadata bits when k=0" {
    // When k=0 (can't compute from dims), should use format metadata "bits" key.
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.scales", .info = .{
            .name = "blk.0.attn_q.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.attn_q.biases", .info = .{
            .name = "blk.0.attn_q.biases",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors, .meta_bits = 6 };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 1024, 128, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 0).?;
    try std.testing.expectEqual(@as(u32, 6), companion.bits);
}

test "findMlxCompanion defaults to 4-bit when no metadata" {
    // When k=0 and no "bits" metadata, should default to default_mlx_bits (4).
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.scales", .info = .{
            .name = "blk.0.attn_q.scales",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.attn_q.biases", .info = .{
            .name = "blk.0.attn_q.biases",
            .n_dims = 2,
            .dims = .{ 32, 1024, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors, .meta_bits = null };
    const t = format_mod.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 1, // n_dims < 2 triggers fallback path
        .dims = .{ 1024, 0, 0, 0 },
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 0).?;
    try std.testing.expectEqual(@as(u32, default_mlx_bits), companion.bits);
}

test "findMlxCompanion 3D expert tensor computes bits from last dim" {
    // 3D MLX expert tensor: dims = [n_experts, rows, words_per_row]
    // bits = dims[n_dims-1] * 32 / k = dims[2] * 32 / k
    // 4-bit with k=2048: words_per_row = 2048*4/32 = 256
    var scales_data: u8 = 0;
    var biases_data: u8 = 0;
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.ffn_gate_exps.scales", .info = .{
            .name = "blk.0.ffn_gate_exps.scales",
            .n_dims = 3,
            .dims = .{ 8, 32, 2048, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scales_data),
        } },
        .{ .name = "blk.0.ffn_gate_exps.biases", .info = .{
            .name = "blk.0.ffn_gate_exps.biases",
            .n_dims = 3,
            .dims = .{ 8, 32, 2048, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&biases_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    const t = format_mod.TensorInfo{
        .name = "blk.0.ffn_gate_exps.weight",
        .n_dims = 3,
        .dims = .{ 8, 2048, 256, 0 }, // 256 u32 words per row
        .dtype = .mlx_q,
        .data_ptr = undefined,
    };
    const companion = findMlxCompanion(mock.format(), t, 2048).?;
    try std.testing.expectEqual(@as(u32, 4), companion.bits);
}

test "resetKvCache compiles" {
    comptime {
        _ = &resetKvCache;
    }
}

test "ensureKvBlock compiles" {
    comptime {
        _ = &ensureKvBlock;
    }
}

test "dispatchGemv HQQ path — CPU" {
    // k=4, n=1, group_size defaults to 64 (g=0 for all elements).
    // w_q: nibbles [1,2,3,4] (bytes [0x21, 0x43])
    // scale: bf16 1.0 (0x3F80), zero: bf16 0.0 (0x0000)
    // x=[1,1,1,1] → y = (1+2+3+4)*1*1 = 10.0
    var w_q_data = [_]u8{ 0x21, 0x43 }; // nibbles: lo=1,hi=2 and lo=3,hi=4
    const one_bf16: [2]u8 = .{ 0x80, 0x3F }; // 1.0 as little-endian bf16
    const zero_bf16: [2]u8 = .{ 0x00, 0x00 }; // 0.0
    // scale and zero are [1] element tensors (one group)
    var scale_data = [_]u8{ 0x80, 0x3F }; // bf16 1.0
    var zero_data = [_]u8{ 0x00, 0x00 }; // bf16 0.0
    _ = one_bf16;
    _ = zero_bf16;

    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.W_q", .info = .{
            .name = "blk.0.attn_q.W_q",
            .n_dims = 2,
            .dims = .{ 1, 2, 0, 0 },
            .dtype = .hqq,
            .data_ptr = @ptrCast(&w_q_data),
        } },
        .{ .name = "blk.0.attn_q.meta.scale", .info = .{
            .name = "blk.0.attn_q.meta.scale",
            .n_dims = 1,
            .dims = .{ 1, 0, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&scale_data),
        } },
        .{ .name = "blk.0.attn_q.meta.zero", .info = .{
            .name = "blk.0.attn_q.meta.zero",
            .n_dims = 1,
            .dims = .{ 1, 0, 0, 0 },
            .dtype = .bf16,
            .data_ptr = @ptrCast(&zero_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    var cpu_be = backend_mod.CpuBackend{};
    const be = backend_mod.Backend{ .cpu = &cpu_be };

    const t = tensors[0].info;
    var x = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var y = [_]f32{0.0};
    dispatchGemv(be, mock.format(), &x, t, &y, 1, 4);
    // y ≈ (1+2+3+4)*1 = 10.0 (nibbles - 0) * 1.0 * x
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[0], 0.5);
}

test "dispatchGemv HQQ path — missing companion returns fallback" {
    // HQQ tensor but no companion scale/zero → falls through to be.gemv (unknown dtype → zero)
    var w_q_data = [_]u8{0x21};
    var tensors = [_]MockFormat.NamedTensor{
        .{ .name = "blk.0.attn_q.W_q", .info = .{
            .name = "blk.0.attn_q.W_q",
            .n_dims = 1,
            .dims = .{ 1, 0, 0, 0 },
            .dtype = .hqq,
            .data_ptr = @ptrCast(&w_q_data),
        } },
    };
    var mock = MockFormat{ .tensors = &tensors };
    var cpu = backend_mod.CpuBackend{};
    const be = backend_mod.Backend{ .cpu = &cpu };
    const t = tensors[0].info;
    var x = [_]f32{1.0};
    var y = [_]f32{99.0};
    // Should not panic — falls back to be.gemv with .hqq dtype → zeroed output
    dispatchGemv(be, mock.format(), &x, t, &y, 1, 1);
    // y = 0 because hqq is in the "warn + zero" branch
    try std.testing.expect(std.math.isFinite(y[0]));
}

test "fuzz: all model functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── Pub constants/types: verify they exist at comptime ───
            comptime {
                _ = VisionEncoder;
                _ = @as(usize, tensor_name_buf_size);
                _ = @as(u32, default_mlx_bits);
                _ = NormCacheEntry;
                _ = ForwardError;
                _ = Model;
                _ = Model.VTable;
                _ = MlxCompanion;
                _ = ModelStorage;
                _ = MockFormat;
            }

            // ── resetInferenceState ─────────────────────────────────
            {
                var kv_len: usize = smith.valueWithHash(usize, 0);
                var cancelled = std.atomic.Value(bool).init(smith.valueWithHash(bool, 1));
                resetInferenceState(&kv_len, &cancelled);
                std.debug.assert(kv_len == 0);
                std.debug.assert(!cancelled.load(.acquire));
            }

            // ── signalCancel ────────────────────────────────────────
            {
                var cancelled = std.atomic.Value(bool).init(smith.valueWithHash(bool, 2));
                signalCancel(&cancelled);
                std.debug.assert(cancelled.load(.acquire));
            }

            // ── expertWeightStride (f32, 3D) ────────────────────────
            {
                const d1 = @as(u64, smith.valueWithHash(u8, 3) | 1); // non-zero
                const d2 = @as(u64, smith.valueWithHash(u8, 4) | 1);
                const t = format_mod.TensorInfo{
                    .name = "test",
                    .n_dims = 3,
                    .dims = .{ 2, d1, d2, 0 },
                    .dtype = .f32,
                    .data_ptr = undefined,
                };
                const stride = expertWeightStride(t);
                std.debug.assert(stride > 0);
            }

            // ── expertWeightStride (nvfp4) ──────────────────────────
            {
                const d0 = @as(u64, smith.valueWithHash(u8, 5) | 1);
                const d1 = @as(u64, smith.valueWithHash(u8, 6) | 1);
                const t = format_mod.TensorInfo{
                    .name = "test",
                    .n_dims = 3,
                    .dims = .{ d0, d1, 4, 0 },
                    .dtype = .nvfp4,
                    .data_ptr = undefined,
                };
                const stride = expertWeightStride(t);
                std.debug.assert(stride == d0 * d1);
            }

            // ── expertStride (mlx_q) ────────────────────────────────
            {
                const d1 = @as(u64, smith.valueWithHash(u8, 7) | 1);
                const d2 = @as(u64, smith.valueWithHash(u8, 8) | 1);
                const t = format_mod.TensorInfo{
                    .name = "test",
                    .n_dims = 3,
                    .dims = .{ 8, d1, d2, 0 },
                    .dtype = .mlx_q,
                    .data_ptr = undefined,
                };
                const stride = expertStride(t);
                std.debug.assert(stride == d1 * d2 * @sizeOf(u32));
            }

            // ── expertStride (non-mlx delegates) ────────────────────
            {
                const t = format_mod.TensorInfo{
                    .name = "test",
                    .n_dims = 3,
                    .dims = .{ 2, 4, 4, 0 },
                    .dtype = .f32,
                    .data_ptr = undefined,
                };
                std.debug.assert(expertStride(t) == expertWeightStride(t));
            }

            // ── findMlxCompanion (non-mlx returns null) ─────────────
            {
                var mock = MockFormat{ .tensors = &.{} };
                const t = format_mod.TensorInfo{
                    .name = "blk.0.attn_q.weight",
                    .n_dims = 2,
                    .dims = .{ 1024, 128, 0, 0 },
                    .dtype = .f32,
                    .data_ptr = undefined,
                };
                const k: usize = smith.valueWithHash(u16, 9);
                std.debug.assert(findMlxCompanion(mock.format(), t, k) == null);
            }

            // ── findMlxCompanion (missing .weight suffix) ───────────
            {
                var mock = MockFormat{ .tensors = &.{} };
                const t = format_mod.TensorInfo{
                    .name = "blk.0.attn_q.bias",
                    .n_dims = 2,
                    .dims = .{ 1024, 128, 0, 0 },
                    .dtype = .mlx_q,
                    .data_ptr = undefined,
                };
                std.debug.assert(findMlxCompanion(mock.format(), t, 1024) == null);
            }

            // ── mlxGemv (non-mlx returns false) ─────────────────────
            {
                var cpu = @import("../backend/backend.zig").CpuBackend{};
                var mock = MockFormat{ .tensors = &.{} };
                const t = format_mod.TensorInfo{
                    .name = "blk.0.attn_q.weight",
                    .n_dims = 2,
                    .dims = .{ 4, 4, 0, 0 },
                    .dtype = .f32,
                    .data_ptr = undefined,
                };
                var y: [4]f32 = undefined;
                var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                std.debug.assert(!mlxGemv(.{ .cpu = &cpu }, mock.format(), &x, t, &y, 4, 4));
            }

            // ── mlxGemv (mlx_q without .weight returns false) ───────
            {
                var cpu = @import("../backend/backend.zig").CpuBackend{};
                var mock = MockFormat{ .tensors = &.{} };
                const t = format_mod.TensorInfo{
                    .name = "blk.0.attn_q.bias",
                    .n_dims = 2,
                    .dims = .{ 4, 4, 0, 0 },
                    .dtype = .mlx_q,
                    .data_ptr = undefined,
                };
                var y: [4]f32 = undefined;
                var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                std.debug.assert(!mlxGemv(.{ .cpu = &cpu }, mock.format(), &x, t, &y, 4, 4));
            }

            // ── dispatchGemv / ensureKvBlock / resetKvCache: comptime ref ──
            comptime {
                _ = &dispatchGemv;
                _ = &ensureKvBlock;
                _ = &resetKvCache;
            }

            // ── MockFormat.format() ─────────────────────────────────
            {
                var mock = MockFormat{ .tensors = &.{}, .meta_bits = smith.valueWithHash(u4, 10) };
                const fmt = mock.format();
                std.debug.assert(fmt.getTensor("nonexistent") == null);
            }

            // ── Model vtable dispatch via MockModel ─────────────────
            {
                const eos = smith.valueWithHash(u32, 11);
                const vocab = smith.valueWithHash(u32, 12);
                const layers = smith.valueWithHash(u32, 13);
                const embd = smith.valueWithHash(u32, 14);
                const head = smith.valueWithHash(u32, 15);
                const head_kv = smith.valueWithHash(u32, 16);
                const seq_len = smith.valueWithHash(usize, 17);

                var mock = MockModel{
                    .eos_token_id = eos,
                    .vocab_size = vocab,
                    .n_layers = layers,
                    .n_embd = embd,
                    .n_head = head,
                    .n_head_kv = head_kv,
                    .kv_seq_len = seq_len,
                };
                // Model.from
                const m = Model.from(MockModel, &mock);

                // eosId
                std.debug.assert(m.eosId() == eos);
                // vocabSize
                std.debug.assert(m.vocabSize() == vocab);
                // nLayers
                std.debug.assert(m.nLayers() == layers);
                // nEmbd
                std.debug.assert(m.nEmbd() == embd);
                // nHead
                std.debug.assert(m.nHead() == head);
                // nHeadKv
                std.debug.assert(m.nHeadKv() == head_kv);
                // kvSeqLen
                std.debug.assert(m.kvSeqLen() == seq_len);

                // setKvSeqLen
                const new_len = smith.valueWithHash(usize, 18);
                m.setKvSeqLen(new_len);
                std.debug.assert(m.kvSeqLen() == new_len);

                // getLogits (empty)
                std.debug.assert(m.getLogits().len == 0);

                // getBlockTable
                std.debug.assert(m.getBlockTable().len == 0);

                // forward
                const fwd = m.forward(smith.valueWithHash(u32, 19)) catch return;
                std.debug.assert(fwd == 7);

                // prefill
                const ids = [_]u32{smith.valueWithHash(u32, 20)};
                const pfx = m.prefill(&ids) catch return;
                std.debug.assert(pfx == 7);

                // forwardTree (MockModel has none -> MissingTensor)
                _ = m.forwardTree(&.{}, &.{}, @ptrFromInt(0x1000), 0) catch {};

                // treeLogits
                std.debug.assert(m.treeLogits(smith.valueWithHash(u32, 21)) == 0);

                // resetCache
                m.resetCache();

                // setThreadContext
                m.setThreadContext();

                // cancel
                m.cancel();
                std.debug.assert(mock.cancelled.load(.acquire));

                // saveSsmState (null for mock)
                std.debug.assert(m.saveSsmState(std.testing.allocator) == null);

                // restoreSsmState (no-op for mock)
                m.restoreSsmState(&[_]u8{ 1, 2, 3 });

                // mtpForward (MissingTensor for mock)
                _ = m.mtpForward(smith.valueWithHash(u32, 22), smith.valueWithHash(u32, 23)) catch {};

                // getMtpDepth
                std.debug.assert(m.getMtpDepth() == 0);

                // getMtpLogits
                std.debug.assert(m.getMtpLogits().len == 0);

                // resetMtpCache
                m.resetMtpCache();

                // setLayerSkip
                const skip_start = smith.valueWithHash(u32, 24);
                const skip_end = smith.valueWithHash(u32, 25);
                m.setLayerSkip(skip_start, skip_end);
                std.debug.assert(mock.layer_skip_start == skip_start);
                std.debug.assert(mock.layer_skip_end == skip_end);

                // setImageEmbeddings
                var embd_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
                const n_tokens = smith.valueWithHash(u32, 26);
                const pad_id = smith.valueWithHash(u32, 27);
                m.setImageEmbeddings(&embd_data, n_tokens, pad_id);
                std.debug.assert(mock.n_visual_tokens == n_tokens);
                std.debug.assert(mock.image_pad_token_id == pad_id);
                std.debug.assert(mock.visual_token_idx == 0);
                // Clear
                m.setImageEmbeddings(null, 0, 0);
                std.debug.assert(mock.image_embeddings == null);
            }

            // ── ModelStorage: comptime verify all pub methods ───────
            comptime {
                _ = &ModelStorage.initFromArch;
                _ = &ModelStorage.model;
                _ = &ModelStorage.deinit;
                _ = &ModelStorage.setPool;
                _ = &ModelStorage.setTpRowShardBuf;
                _ = &ModelStorage.setTpTransport;
                _ = &ModelStorage.sendKvCache;
                _ = &ModelStorage.recvKvCache;
                _ = &ModelStorage.setPpConfig;
                _ = &ModelStorage.fixBlockAllocator;
                _ = &ModelStorage.setChunkSize;
                _ = &ModelStorage.setMegakernel;
                _ = &ModelStorage.setTriCalibration;
                _ = &ModelStorage.enableProfiling;
                _ = &ModelStorage.reportPerf;
            }
        }
    }.f, .{});
}
