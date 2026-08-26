//! GLM-4 MoE Lite model with MLA (Multi-head Latent Attention) and MoE FFN.
//! Architecture: compressed KV (GLM MLA variant) + sigmoid-routed MoE.

const std = @import("std");
const Allocator = std.mem.Allocator;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const model_mod = @import("model.zig");
const math_ops = @import("../ops/math.zig");
const mlx_ops = @import("../ops/mlx.zig");
const kvcache = @import("../kvcache/manager.zig");
const block_alloc_mod = @import("../kvcache/block_allocator.zig");
const BlockAllocator = block_alloc_mod.BlockAllocator;
const TieredBlockAllocator = block_alloc_mod.TieredBlockAllocator;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;

const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Model = model_mod.Model;
const kv_quant = @import("../ops/kv_quant.zig");
const PagedKvCache = kvcache.PagedKvCache;
const SeqBlockTable = kvcache.SeqBlockTable;

const DType = format_mod.DType;
const quant_ops = @import("../ops/quant.zig");

/// Maximum top-k experts for stack-allocated selection arrays.
const max_active_experts: usize = 8;
/// Buffer size for tensor name formatting (layer prefix + suffix).
const name_buf_size: usize = model_mod.tensor_name_buf_size;
/// Default MLX quantization bit width for GLM-4 (6-bit, unlike 4-bit default for other models).
const default_glm4_mlx_bits: u32 = 6;
/// Norm weight cache capacity (final + 4 per layer).
const max_norm_entries: usize = 512;
const NormCacheEntry = model_mod.NormCacheEntry;

// ── Model struct ─────────────────────────────────────────────────

/// GLM-4 MoE Lite model state with MLA attention and sigmoid-routed MoE.
pub const Glm4Model = struct {
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // Config
    n_layers: u32 = 47,
    n_embd: u32 = 2048,
    n_head: u32 = 20,
    n_head_kv: u32 = 20, // Model vtable compatibility; MLA uses compressed KV internally
    vocab_size: u32 = 154880,
    q_lora_rank: u32 = 768,
    kv_lora_rank: u32 = 512,
    qk_nope_head_dim: u32 = 192,
    qk_rope_head_dim: u32 = 64,
    v_head_dim: u32 = 256,
    intermediate_size: u32 = 10240,
    moe_intermediate_size: u32 = 1536,
    n_routed_experts: u32 = 64,
    num_experts_per_tok: u32 = 4,
    routed_scaling_factor: f32 = 1.8,
    first_k_dense_replace: u32 = 1,
    rope_theta: f32 = 1000000.0,
    rms_eps: f32 = 1e-5,
    eos_token_id: u32 = 154820,
    max_seq_len: usize = 4096,
    mlx_bits: u32 = 6,

    // Working buffers
    hidden: []f32 = &.{},
    hidden2: []f32 = &.{},
    q_compressed: []f32 = &.{}, // [q_lora_rank]
    q_full: []f32 = &.{}, // [n_head * q_head_dim] where q_head_dim = nope + rope
    kv_proj: []f32 = &.{}, // [kv_lora_rank + qk_rope_head_dim]
    kv_latent: []f32 = &.{}, // [kv_lora_rank]
    k_buf: []f32 = &.{}, // [n_head * (qk_nope_head_dim + qk_rope_head_dim)]
    v_buf: []f32 = &.{}, // [n_head * v_head_dim]
    attn_out: []f32 = &.{}, // [n_head * v_head_dim]
    scores_buf: []f32 = &.{}, // [max_seq_len]
    ff_gate: []f32 = &.{},
    ff_up: []f32 = &.{},
    ff_down: []f32 = &.{},
    expert_buf: []f32 = &.{},
    router_logits: []f32 = &.{},
    logits_buf: []f32 = &.{},
    /// Scratch buffer for MLA dequantization in multiLinearGemv.
    /// Sized to max(nope_dim, v_head_dim) * kv_lora_rank to avoid corrupting logits_buf.
    mla_scratch: []f32 = &.{},

    // Prefill buffers (allocated via page_allocator for GPU zero-copy)
    chunk_size: usize = 256,
    pf_hidden: []f32 = &.{},
    pf_hidden2: []f32 = &.{},
    pf_q_a: []f32 = &.{},
    pf_q: []f32 = &.{},
    pf_kv_proj: []f32 = &.{},
    pf_kv_latent: []f32 = &.{},
    pf_k: []f32 = &.{},
    pf_v: []f32 = &.{},
    pf_attn_out: []f32 = &.{},
    pf_positions: []u32 = &.{},

    // KV cache (PagedAttention or TieredKvCache): store full reconstructed K and V per layer
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    layer_skip_start: u32 = 0,
    layer_skip_end: u32 = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Enable fused megakernel for single-dispatch forward pass.
    megakernel_enabled: bool = false,

    // Name buffer for tensor lookups
    name_buf: [name_buf_size]u8 = undefined,

    // Thread pool for parallel CPU work
    pool: ?*@import("../thread_pool.zig").ThreadPool = null,

    /// Permanent f32 cache for non-f32 RMS norm weights (BF16 etc.).
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    /// Initialize the model from format metadata and allocate all working buffers.
    pub fn init(allocator: Allocator, f: Format, be: Backend, ctx_size: u32, kv_type_k: kv_quant.KvQuantType, kv_type_v: kv_quant.KvQuantType, tiered_cache: ?*TieredKvCache) !Glm4Model {
        var self = Glm4Model{ .fmt = f, .be = be, .allocator = allocator };
        self.kv_type_k = kv_type_k;
        self.kv_type_v = kv_type_v;

        // Read config
        if (f.getMetaU32("num_hidden_layers")) |v| self.n_layers = v;
        if (f.getMetaU32("hidden_size")) |v| self.n_embd = v;
        if (f.getMetaU32("num_attention_heads")) |v| self.n_head = v;
        if (f.getMetaU32("vocab_size")) |v| self.vocab_size = v;
        // MLA parameters, try HF keys first, then GGUF arch-prefixed keys (deepseek2.*)
        const arch = f.getMetaStr("general.architecture") orelse "glm4";
        if (f.getMetaU32("q_lora_rank") orelse f.getArchU32(arch, "attention.q_lora_rank")) |v| self.q_lora_rank = v;
        if (f.getMetaU32("kv_lora_rank") orelse f.getArchU32(arch, "attention.kv_lora_rank")) |v| self.kv_lora_rank = v;
        if (f.getMetaU32("qk_nope_head_dim") orelse f.getArchU32(arch, "attention.key_length_mla")) |v| self.qk_nope_head_dim = v;
        if (f.getMetaU32("qk_rope_head_dim") orelse f.getArchU32(arch, "attention.rope_key_length")) |v| self.qk_rope_head_dim = v;
        if (f.getMetaU32("v_head_dim") orelse f.getArchU32(arch, "attention.value_length_mla")) |v| self.v_head_dim = v;
        if (f.getMetaU32("intermediate_size")) |v| self.intermediate_size = v;
        if (f.getMetaU32("moe_intermediate_size")) |v| self.moe_intermediate_size = v;
        if (f.getMetaU32("n_routed_experts")) |v| self.n_routed_experts = v;
        if (f.getMetaU32("num_experts_per_tok")) |v| self.num_experts_per_tok = v;
        if (f.getMetaU32("first_k_dense_replace")) |v| self.first_k_dense_replace = v;
        if (f.getMetaU32("eos_token_id")) |v| self.eos_token_id = v;
        if (f.getMetaF32("routed_scaling_factor")) |v| self.routed_scaling_factor = v;
        if (f.getMetaF32("rope_theta")) |v| self.rope_theta = v;
        if (f.getMetaF32("rms_norm_eps")) |v| self.rms_eps = v;
        self.mlx_bits = f.getMetaU32("bits") orelse default_glm4_mlx_bits;
        if (f.getMetaU32("context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

        if (self.n_routed_experts > max_active_experts * 16) {
            std.log.err("glm4: n_routed_experts ({d}) exceeds max_active_experts * 16 ({d})", .{ self.n_routed_experts, max_active_experts * 16 });
            return error.MissingTensor;
        }
        if (self.num_experts_per_tok > max_active_experts) {
            std.log.err("glm4: num_experts_per_tok ({d}) exceeds max_active_experts ({d})", .{ self.num_experts_per_tok, max_active_experts });
            return error.MissingTensor;
        }

        const nh: usize = self.n_head;
        const q_head_dim: usize = self.qk_nope_head_dim + self.qk_rope_head_dim;
        const k_head_dim: usize = q_head_dim; // K has same total dim as Q
        const kvd: usize = nh * k_head_dim;
        const vd: usize = nh * self.v_head_dim;
        const max_ff: usize = @max(self.intermediate_size, self.moe_intermediate_size);
        const nl: usize = self.n_layers;

        // Allocate working buffers
        self.hidden = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden);
        self.hidden2 = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.hidden2);
        self.q_compressed = try allocator.alloc(f32, self.q_lora_rank);
        errdefer allocator.free(self.q_compressed);
        self.q_full = try allocator.alloc(f32, nh * q_head_dim);
        errdefer allocator.free(self.q_full);
        self.kv_proj = try allocator.alloc(f32, self.kv_lora_rank + self.qk_rope_head_dim);
        errdefer allocator.free(self.kv_proj);
        self.kv_latent = try allocator.alloc(f32, self.kv_lora_rank);
        errdefer allocator.free(self.kv_latent);
        self.k_buf = try allocator.alloc(f32, kvd);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, vd);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, vd);
        errdefer allocator.free(self.attn_out);
        self.scores_buf = try allocator.alloc(f32, self.max_seq_len);
        errdefer allocator.free(self.scores_buf);
        self.ff_gate = try allocator.alloc(f32, max_ff);
        errdefer allocator.free(self.ff_gate);
        self.ff_up = try allocator.alloc(f32, max_ff);
        errdefer allocator.free(self.ff_up);
        self.ff_down = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.ff_down);
        self.expert_buf = try allocator.alloc(f32, self.n_embd);
        errdefer allocator.free(self.expert_buf);
        self.router_logits = try allocator.alloc(f32, self.n_routed_experts);
        errdefer allocator.free(self.router_logits);
        self.logits_buf = try allocator.alloc(f32, self.vocab_size);
        errdefer allocator.free(self.logits_buf);
        // MLA scratch: max(nope_dim, v_head_dim) * kv_lora_rank
        const mla_scratch_size = @max(self.qk_nope_head_dim, self.v_head_dim) * self.kv_lora_rank;
        self.mla_scratch = try allocator.alloc(f32, mla_scratch_size);
        errdefer allocator.free(self.mla_scratch);

        // Prefill buffers (page_allocator for GPU zero-copy, Metal's
        // newBufferWithBytesNoCopy requires page-aligned pointers).
        {
            const pa = std.heap.page_allocator;
            const cs = self.chunk_size;
            self.pf_hidden = try pa.alloc(f32, cs * self.n_embd);
            errdefer pa.free(self.pf_hidden);
            self.pf_hidden2 = try pa.alloc(f32, cs * self.n_embd);
            errdefer pa.free(self.pf_hidden2);
            self.pf_q_a = try pa.alloc(f32, cs * self.q_lora_rank);
            errdefer pa.free(self.pf_q_a);
            self.pf_q = try pa.alloc(f32, cs * nh * q_head_dim);
            errdefer pa.free(self.pf_q);
            self.pf_kv_proj = try pa.alloc(f32, cs * (self.kv_lora_rank + self.qk_rope_head_dim));
            errdefer pa.free(self.pf_kv_proj);
            self.pf_kv_latent = try pa.alloc(f32, cs * self.kv_lora_rank);
            errdefer pa.free(self.pf_kv_latent);
            self.pf_k = try pa.alloc(f32, cs * kvd);
            errdefer pa.free(self.pf_k);
            self.pf_v = try pa.alloc(f32, cs * vd);
            errdefer pa.free(self.pf_v);
            self.pf_attn_out = try pa.alloc(f32, cs * vd);
            errdefer pa.free(self.pf_attn_out);
            self.pf_positions = try pa.alloc(u32, cs);
            errdefer pa.free(self.pf_positions);
        }
        // Function-scoped errdefer: covers try calls below the bare {} block
        // (block-scoped errdefers above only guard within the block).
        errdefer {
            const pa = std.heap.page_allocator;
            const pf_bufs = .{
                &self.pf_hidden,   &self.pf_hidden2,   &self.pf_q_a, &self.pf_q,
                &self.pf_kv_proj,  &self.pf_kv_latent, &self.pf_k,   &self.pf_v,
                &self.pf_attn_out,
            };
            inline for (pf_bufs) |buf| if (buf.len > 0) pa.free(buf.*);
            if (self.pf_positions.len > 0) pa.free(self.pf_positions);
        }

        // KV cache: use TieredKvCache if provided, otherwise flat PagedKvCache.
        // Note: GLM4 uses different k_head_dim and v_head_dim, use larger for cache.
        if (tiered_cache) |tc| {
            var ta = TieredBlockAllocator.init(tc, allocator);
            self.seq_table = try ta.allocateSeqTable(nl);
            errdefer ta.freeSeqTable(&self.seq_table);
            try ta.appendBlock(&self.seq_table);
            self.tiered_cache = tc;
            self.tiered_block_allocator = ta;
        } else {
            const max_kv_dim = @max(kvd, vd);
            // Paged KV cache: small fixed-size blocks allocated on demand.
            // Memory scales with actual sequence length, not max context.
            const paged_block_size: u16 = 256;
            const blocks_per_layer = (self.max_seq_len + paged_block_size - 1) / paged_block_size;
            const num_blocks = nl * blocks_per_layer;
            const block_size = paged_block_size;
            self.paged_cache = try PagedKvCache.init(allocator, nl, max_kv_dim, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            // BlockAllocator stores a pointer, must point to self.paged_cache (not a local copy).
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        self.warmNormCache();
        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Glm4Model) void {
        self.be.sync();
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }

        const bufs = .{
            &self.hidden,     &self.hidden2,     &self.q_compressed,
            &self.q_full,     &self.kv_proj,     &self.kv_latent,
            &self.k_buf,      &self.v_buf,       &self.attn_out,
            &self.scores_buf, &self.ff_gate,     &self.ff_up,
            &self.ff_down,    &self.expert_buf,  &self.router_logits,
            &self.logits_buf, &self.mla_scratch,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
        // Prefill buffers (page_allocator, must match init allocation)
        {
            const pa = std.heap.page_allocator;
            const pf_bufs = .{
                &self.pf_hidden,   &self.pf_hidden2,   &self.pf_q_a, &self.pf_q,
                &self.pf_kv_proj,  &self.pf_kv_latent, &self.pf_k,   &self.pf_v,
                &self.pf_attn_out,
            };
            inline for (pf_bufs) |buf| if (buf.len > 0) pa.free(buf.*);
            if (self.pf_positions.len > 0) pa.free(self.pf_positions);
        }
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Glm4Model) Model {
        return Model.from(Glm4Model, self);
    }

    // ── Forward pass ─────────────────────────────────────────────

    /// Run one decode step, returning the argmax next-token ID.
    ///
    /// Error conditions:
    /// - `error.KVCacheFull`  , sequence length has reached `max_seq_len`.
    /// - `error.MissingTensor`, a required weight tensor was not found in the model file.
    /// - `error.Cancelled`    , the inference was cancelled via the `cancelled` flag.
    pub fn forward(self: *Glm4Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup
        try self.embLookup(token_id);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));
            try self.mlaAttention(l);
            if (li < self.first_k_dense_replace) {
                try self.denseFfn(l);
            } else {
                try self.moeFfn(l);
            }
        }

        // Final norm → logits
        const nw = self.fmt.getTensor("model.norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, self.n_embd), self.hidden.ptr, self.n_embd, self.rms_eps);

        // LM head (may be quantized)
        self.be.sync();
        try self.mlxGemv("lm_head", self.hidden, self.logits_buf, self.vocab_size, self.n_embd);

        self.kv_seq_len += 1;
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill, MLA attention uses batched GEMM for projections
    /// and sdpaPrefill for fused causal attention.  MoE FFN routing selects
    /// different experts per token, so FFN is always per-token.
    /// Falls back to per-token forward when prefill buffers are absent
    /// or the prompt is a single token.
    pub fn prefill(self: *Glm4Model, token_ids: []const u32) !u32 {
        if (token_ids.len == 0) return error.MissingTensor;
        if (token_ids.len > self.max_seq_len) return error.KVCacheFull;

        // Guard: batched path requires equal Q/V head dims for sdpaPrefill.
        const q_hd = self.qk_nope_head_dim + self.qk_rope_head_dim;
        const cs: usize = if (self.pf_hidden.len > 0 and q_hd == self.v_head_dim)
            self.chunk_size
        else
            1;
        if (cs <= 1 or token_ids.len == 1) {
            var last: u32 = 0;
            for (token_ids) |tid| last = try self.forward(tid);
            return last;
        }

        var offset: usize = 0;
        while (offset < token_ids.len) {
            const chunk_len = @min(cs, token_ids.len - offset);
            try self.prefillChunk(token_ids[offset..][0..chunk_len], @intCast(offset));
            offset += chunk_len;
        }

        // Final: rmsNorm + logits on the LAST token only.
        // In batched mode, residuals are never deferred, pf_hidden has the
        // complete residual stream, so plain rmsNorm suffices (no addRmsNorm).
        const last_in_chunk = (token_ids.len - 1) % cs;
        const e: usize = self.n_embd;
        @memcpy(self.hidden, self.pf_hidden[last_in_chunk * e ..][0..e]);

        const nw = self.fmt.getTensor("model.norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden.ptr, e, self.rms_eps);

        self.be.sync();
        try self.mlxGemv("lm_head", self.hidden, self.logits_buf, self.vocab_size, self.n_embd);

        self.be.sync();
        self.kv_seq_len = token_ids.len;
        return math_ops.argmax(self.logits_buf);
    }

    /// Reset the KV cache position for a new conversation.
    pub fn resetCache(self: *Glm4Model) void {
        model_mod.resetKvCache(self);
    }

    /// Signal an in-progress forward pass to abort. Thread-safe.
    pub fn cancel(self: *Glm4Model) void {
        model_mod.signalCancel(&self.cancelled);
    }

    /// Return physical block IDs from layer 0 of the current sequence table.
    /// All layers share the same block IDs, so layer 0 is sufficient.
    pub fn getBlockTable(self: *Glm4Model) []const u32 {
        return self.seq_table.block_table[0];
    }

    /// Helper: get flat f32 view of KV cache for a layer (assembled from paged or tiered blocks).
    fn getLayerKvView(self: *Glm4Model, layer: usize) struct { keys: []f32, values: []f32 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]f32{}, .values = &[_]f32{} };

        const block_id = self.seq_table.block_table[layer][0];
        if (self.tiered_cache) |tc| {
            return .{
                .keys = tc.blocks[block_id].base.keys,
                .values = tc.blocks[block_id].base.values,
            };
        }
        return .{
            .keys = self.paged_cache.blocks[block_id].keys,
            .values = self.paged_cache.blocks[block_id].values,
        };
    }

    const PagedKvView = kvcache.PagedKvView;

    fn getPagedKvView(self: *Glm4Model, layer: usize) PagedKvView {
        return PagedKvView.initView(
            self.seq_table.block_table[layer],
            self.paged_cache.blocks,
            self.paged_cache.block_size,
            self.paged_cache.kv_dim,
            self.kv_seq_len,
        );
    }

    fn isMultiBlock(self: *Glm4Model, layer: usize) bool {
        return self.paged_cache.block_size > 0 and self.seq_table.block_table[layer].len > 1;
    }

    // ── Embedding ────────────────────────────────────────────────

    fn embLookup(self: *Glm4Model, token_id: u32) !void {
        const w_t = self.fmt.getTensor("model.embed_tokens.weight") orelse return error.MissingTensor;
        if (w_t.dtype == .mlx_q) {
            const s_t = self.fmt.getTensor("model.embed_tokens.scales") orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor("model.embed_tokens.biases") orelse return error.MissingTensor;
            mlx_ops.mlxEmbLookup(
                self.hidden.ptr,
                @ptrCast(@alignCast(w_t.data_ptr)),
                @ptrCast(@alignCast(s_t.data_ptr)),
                @ptrCast(@alignCast(b_t.data_ptr)),
                token_id,
                self.n_embd,
                self.mlx_bits,
            );
        } else {
            self.be.embLookup(.{ .data = w_t.data_ptr, .dtype = w_t.dtype }, token_id, self.hidden.ptr, self.n_embd);
        }
    }

    // ── MLA Attention ────────────────────────────────────────────

    fn mlaAttention(self: *Glm4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nope_dim: usize = self.qk_nope_head_dim;
        const rope_dim: usize = self.qk_rope_head_dim;
        const q_head_dim: usize = nope_dim + rope_dim;
        const kv_rank: usize = self.kv_lora_rank;
        const vhd: usize = self.v_head_dim;
        // 1. Pre-norm
        const nw = self.layerTensor(li, "input_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // 2. Q path: hidden2 → q_a_proj(e→q_lora_rank) → layernorm → q_b_proj(q_lora_rank→nh*q_head_dim)
        self.be.sync();
        try self.mlxLayerGemv(li, "self_attn.q_a_proj", self.hidden2, self.q_compressed, self.q_lora_rank, e);
        const qn = self.layerTensor(li, "self_attn.q_a_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.q_compressed.ptr, self.normAsF32(qn, self.q_lora_rank), self.q_compressed.ptr, self.q_lora_rank, self.rms_eps);
        self.be.sync();
        try self.mlxLayerGemv(li, "self_attn.q_b_proj", self.q_compressed, self.q_full, nh * q_head_dim, self.q_lora_rank);
        // 3. KV path: hidden2 → kv_a_proj_with_mqa(e→kv_rank+rope_dim) → split
        try self.mlxLayerGemv(li, "self_attn.kv_a_proj_with_mqa", self.hidden2, self.kv_proj, kv_rank + rope_dim, e);
        // Split: kv_latent[0..kv_rank], k_pe[kv_rank..kv_rank+rope_dim]
        self.be.sync();
        @memcpy(self.kv_latent[0..kv_rank], self.kv_proj[0..kv_rank]);
        const k_pe = self.kv_proj[kv_rank..][0..rope_dim];

        // Layernorm on kv_latent
        const kvn = self.layerTensor(li, "self_attn.kv_a_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.kv_latent.ptr, self.normAsF32(kvn, kv_rank), self.kv_latent.ptr, kv_rank, self.rms_eps);

        // 4. Per-head K_nope and V from kv_latent via embed_q and unembed_out
        // embed_q: [nh, nope_dim, kv_rank] → K_nope per head (HF tensor name; despite the name, projects kv_latent into K_nope)
        // unembed_out: [nh, v_head_dim, kv_rank] → V per head
        self.be.sync();
        try self.multiLinearGemv(li, "self_attn.embed_q", self.kv_latent, self.k_buf.ptr, nh, nope_dim, kv_rank);
        try self.multiLinearGemv(li, "self_attn.unembed_out", self.kv_latent, self.v_buf.ptr, nh, vhd, kv_rank);

        // 5. Assemble full K per head: [k_nope(nope_dim), k_pe(rope_dim)]
        // k_buf has [nh * nope_dim] from embed_q; shift nope to make room for k_pe
        self.be.sync();
        {
            var h: usize = nh;
            while (h > 0) {
                h -= 1;
                const src_off = h * nope_dim;
                const dst_off = h * q_head_dim;
                // Move nope part (backwards to avoid overlap)
                var i: usize = nope_dim;
                while (i > 0) {
                    i -= 1;
                    self.k_buf[dst_off + i] = self.k_buf[src_off + i];
                }
                // Copy shared k_pe into rope portion
                @memcpy(self.k_buf[dst_off + nope_dim ..][0..rope_dim], k_pe);
            }
        }

        // 6. RoPE on q_pe and k_pe portions (offset nope_dim within each head)
        self.ropePartial(self.q_full.ptr, nh, q_head_dim, nope_dim, rope_dim);
        self.ropePartial(self.k_buf.ptr, nh, q_head_dim, nope_dim, rope_dim);

        // 7. Cache K and V
        const kvd = nh * q_head_dim;
        const vd = nh * vhd;
        const kv_view = self.getLayerKvView(li);
        const pos = self.kv_seq_len;
        @memcpy(kv_view.keys[pos * kvd ..][0..kvd], self.k_buf[0..kvd]);
        @memcpy(kv_view.values[pos * vd ..][0..vd], self.v_buf[0..vd]);

        // 8. Attention: Q @ K^T / sqrt(q_head_dim), softmax, @ V
        const sl = self.kv_seq_len + 1;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q_head_dim)));

        for (0..nh) |h| {
            const q_base = h * q_head_dim;
            // QK dot products
            for (0..sl) |t| {
                const k_elem_off = t * kvd + h * q_head_dim;
                const k_ptr = kv_view.keys.ptr + k_elem_off;
                var dot: f32 = 0;
                for (0..q_head_dim) |i| dot += self.q_full[q_base + i] * k_ptr[i];
                self.scores_buf[t] = dot * scale;
            }
            // Inline CPU softmax, avoids backend dispatch + sync overhead
            // since QK dot products and V accumulation are already on CPU.
            {
                var max_val: f32 = self.scores_buf[0];
                for (1..sl) |i| if (self.scores_buf[i] > max_val) {
                    max_val = self.scores_buf[i];
                };
                var sm_sum: f32 = 0;
                for (0..sl) |i| {
                    self.scores_buf[i] = @exp(self.scores_buf[i] - max_val);
                    sm_sum += self.scores_buf[i];
                }
                const inv_sum = 1.0 / sm_sum;
                for (0..sl) |i| self.scores_buf[i] *= inv_sum;
            }

            // Value accumulation (V has different dim than K)
            const v_base = h * vhd;
            @memset(self.attn_out[v_base..][0..vhd], 0);
            for (0..sl) |t| {
                const v_elem_off = t * vd + h * vhd;
                const v_ptr = kv_view.values.ptr + v_elem_off;
                const weight = self.scores_buf[t];
                for (0..vhd) |i| self.attn_out[v_base + i] += weight * v_ptr[i];
            }
        }

        // 9. Output projection
        try self.mlxLayerGemv(li, "self_attn.o_proj", self.attn_out, self.hidden2, e, nh * vhd);

        // 10. Residual add deferred: denseFfn/moeFfn fuses add(hidden, hidden2) + rmsNorm.
    }

    // ── Batched prefill ─────────────────────────────────────────────

    /// Look up a layer weight tensor by HF prefix, with GGUF fallback.
    fn layerWeight(self: *Glm4Model, li: u32, prefix: []const u8) ?TensorInfo {
        var buf: [name_buf_size]u8 = undefined;
        const w_name = std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ li, prefix }) catch return null;
        if (self.fmt.getTensor(w_name)) |t| return t;
        if (hfToGgufAttnName(prefix)) |gguf_name| {
            return self.fmt.layerTensor(li, gguf_name);
        }
        return null;
    }

    /// Return KV cache view as byte slices for sdpaPrefill compatibility.
    fn getLayerKvViewBytes(self: *Glm4Model, layer: usize) struct { keys: []u8, values: []u8 } {
        const num_blocks = self.seq_table.block_table[layer].len;
        if (num_blocks == 0) return .{ .keys = &[_]u8{}, .values = &[_]u8{} };
        const block_id = self.seq_table.block_table[layer][0];
        if (self.tiered_cache) |tc| {
            return .{
                .keys = std.mem.sliceAsBytes(tc.blocks[block_id].base.keys),
                .values = std.mem.sliceAsBytes(tc.blocks[block_id].base.values),
            };
        }
        return .{
            .keys = std.mem.sliceAsBytes(self.paged_cache.blocks[block_id].keys),
            .values = std.mem.sliceAsBytes(self.paged_cache.blocks[block_id].values),
        };
    }

    /// Batched GEMM: [n_tok, n_in] × W^T → [n_tok, n_out].
    /// Falls back to per-token GEMV for MLX quantized weights.
    fn doGemm(self: *Glm4Model, x: [*]const f32, w: TensorInfo, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        if (w.dtype == .mlx_q) {
            const wi = std.mem.lastIndexOf(u8, w.name, ".weight") orelse return;
            const prefix = w.name[0..wi];
            var sbuf: [name_buf_size]u8 = undefined;
            var bbuf: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&sbuf, "{s}.scales", .{prefix}) catch return;
            const b_name = std.fmt.bufPrint(&bbuf, "{s}.biases", .{prefix}) catch return;
            const s_t = self.fmt.getTensor(s_name) orelse return;
            const b_t = self.fmt.getTensor(b_name) orelse return;
            for (0..n_tok) |i| {
                self.be.gemvMlxQ(x + i * n_in, w.data_ptr, s_t.data_ptr, b_t.data_ptr, y + i * n_out, n_out, n_in, self.mlx_bits, model_mod.inferMlxGroupSize(s_t, n_in));
            }
            return;
        }
        self.be.gemm(x, .{ .data = w.data_ptr, .dtype = w.dtype }, y, n_tok, n_out, n_in);
    }

    /// Process one chunk of tokens through all layers using batched GEMM
    /// for MLA attention, per-token fallback for MoE/dense FFN.
    fn prefillChunk(self: *Glm4Model, token_ids: []const u32, base_pos: u32) !void {
        const n_tok = token_ids.len;
        const e: usize = self.n_embd;

        // Ensure KV blocks allocated for all new positions.
        for (0..n_tok) |t| {
            self.kv_seq_len = base_pos + t;
            try model_mod.ensureKvBlock(self);
        }

        // Embedding lookup for all tokens into batched buffer.
        for (token_ids, 0..) |tid, t| {
            try self.embLookup(tid);
            @memcpy(self.pf_hidden[t * e ..][0..e], self.hidden);
        }

        // Build position array.
        for (0..n_tok) |t| {
            self.pf_positions[t] = base_pos + @as(u32, @intCast(t));
        }

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.monotonic)) return error.Cancelled;
            const l: u32 = @intCast(li);
            if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
            self.fmt.prefetchLayer(@intCast(li + 1));

            // Batched MLA attention.
            try self.prefillMlaAttention(l, n_tok);

            // Per-token FFN (MoE routing is per-token, cannot batch).
            // Residual is deferred: pf_hidden holds pre-attention residual,
            // pf_hidden2 holds post-attention output.  denseFfn/moeFfn
            // fuses the add internally via addRmsNorm.
            self.be.sync();
            for (0..n_tok) |t| {
                @memcpy(self.hidden, self.pf_hidden[t * e ..][0..e]);
                @memcpy(self.hidden2, self.pf_hidden2[t * e ..][0..e]);
                if (li < self.first_k_dense_replace) {
                    try self.denseFfn(l);
                } else {
                    try self.moeFfn(l);
                }
                @memcpy(self.pf_hidden[t * e ..][0..e], self.hidden);
            }
        }

        self.kv_seq_len = base_pos + n_tok;
    }

    /// Batched MLA attention for one layer.
    ///
    /// Steps 1-5, 7: batched GEMM/norm for Q/KV projections.
    /// Steps 6, 8-10: per-token multiLinearGemv for per-head K/V expansion,
    ///   K assembly, and partial RoPE (rope at nope_dim offset).
    /// Step 11: sdpaPrefill for fused causal attention.
    /// Step 12-13: batched output projection, residual deferred to FFN.
    fn prefillMlaAttention(self: *Glm4Model, li: u32, n_tok: usize) !void {
        const e: usize = self.n_embd;
        const nh: usize = self.n_head;
        const nope_dim: usize = self.qk_nope_head_dim;
        const rope_dim: usize = self.qk_rope_head_dim;
        const q_head_dim: usize = nope_dim + rope_dim;
        const kv_rank: usize = self.kv_lora_rank;
        const vhd: usize = self.v_head_dim;

        // 1. Pre-norm (batched)
        const nw = self.layerTensor(li, "input_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_hidden.ptr, self.normAsF32(nw, e), self.pf_hidden2.ptr, n_tok, e, self.rms_eps);

        // 2. Q path: hidden2 → q_a_proj → q_compressed (batched GEMM)
        const qa_w = self.layerWeight(li, "self_attn.q_a_proj") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, qa_w, self.pf_q_a.ptr, n_tok, self.q_lora_rank, e);

        // 3. Q latent norm (batched)
        const qn = self.layerTensor(li, "self_attn.q_a_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_q_a.ptr, self.normAsF32(qn, self.q_lora_rank), self.pf_q_a.ptr, n_tok, self.q_lora_rank, self.rms_eps);

        // 4. Q expansion: q_compressed → q_full (batched GEMM)
        const qb_w = self.layerWeight(li, "self_attn.q_b_proj") orelse return error.MissingTensor;
        self.doGemm(self.pf_q_a.ptr, qb_w, self.pf_q.ptr, n_tok, nh * q_head_dim, self.q_lora_rank);

        // 5. KV path: hidden2 → kv_a_proj (batched GEMM)
        const kva_w = self.layerWeight(li, "self_attn.kv_a_proj_with_mqa") orelse return error.MissingTensor;
        self.doGemm(self.pf_hidden2.ptr, kva_w, self.pf_kv_proj.ptr, n_tok, kv_rank + rope_dim, e);

        // 6. Split kv_proj → kv_latent + k_pe (per-token memcpy)
        self.be.sync();
        for (0..n_tok) |t| {
            @memcpy(
                self.pf_kv_latent[t * kv_rank ..][0..kv_rank],
                self.pf_kv_proj[t * (kv_rank + rope_dim) ..][0..kv_rank],
            );
        }

        // 7. KV latent norm (batched)
        const kvn = self.layerTensor(li, "self_attn.kv_a_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNormBatched(self.pf_kv_latent.ptr, self.normAsF32(kvn, kv_rank), self.pf_kv_latent.ptr, n_tok, kv_rank, self.rms_eps);

        // 8-10. Per-token: multiLinearGemv (embed_q → K_nope, unembed_out → V),
        //        K assembly, and partial RoPE.
        self.be.sync();
        for (0..n_tok) |t| {
            const kv_lat = self.pf_kv_latent[t * kv_rank ..][0..kv_rank];
            const k_pe = self.pf_kv_proj[t * (kv_rank + rope_dim) + kv_rank ..][0..rope_dim];

            // embed_q: kv_latent → k_nope [nh * nope_dim] (per-token k_buf as scratch)
            try self.multiLinearGemv(li, "self_attn.embed_q", kv_lat, self.k_buf.ptr, nh, nope_dim, kv_rank);
            // unembed_out: kv_latent → v [nh * v_head_dim]
            try self.multiLinearGemv(li, "self_attn.unembed_out", kv_lat, self.v_buf.ptr, nh, vhd, kv_rank);
            self.be.sync();

            // Assemble K per head: [k_nope(nope_dim) | k_pe(rope_dim)]
            for (0..nh) |h| {
                const dst = self.pf_k[t * nh * q_head_dim + h * q_head_dim ..];
                @memcpy(dst[0..nope_dim], self.k_buf[h * nope_dim ..][0..nope_dim]);
                @memcpy(dst[nope_dim..][0..rope_dim], k_pe);
            }

            // Copy V into batched buffer
            @memcpy(self.pf_v[t * nh * vhd ..][0 .. nh * vhd], self.v_buf[0 .. nh * vhd]);

            // RoPE (partial, only rope portion at offset nope_dim within each head)
            const saved_pos = self.kv_seq_len;
            self.kv_seq_len = self.pf_positions[t];
            self.ropePartial(self.pf_q.ptr + t * nh * q_head_dim, nh, q_head_dim, nope_dim, rope_dim);
            self.ropePartial(self.pf_k.ptr + t * nh * q_head_dim, nh, q_head_dim, nope_dim, rope_dim);
            self.kv_seq_len = saved_pos;
        }

        // 11. Fused causal attention (sdpaPrefill writes new K/V into cache
        //     and computes attention against [cache + new] tokens)
        const kv_view = self.getLayerKvViewBytes(li);
        const prev_len: usize = self.pf_positions[0];
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q_head_dim)));
        self.be.sdpaPrefill(self.pf_q.ptr, self.pf_k.ptr, self.pf_v.ptr, kv_view.keys, kv_view.values, self.pf_attn_out.ptr, nh, nh, q_head_dim, prev_len, n_tok, scale, .f32, .f32);

        // 12. Output projection (batched GEMM)
        const ow = self.layerWeight(li, "self_attn.o_proj") orelse return error.MissingTensor;
        self.doGemm(self.pf_attn_out.ptr, ow, self.pf_hidden2.ptr, n_tok, e, nh * vhd);

        // 13. Residual deferred to denseFfn/moeFfn: pf_hidden holds the
        // pre-attention residual, pf_hidden2 holds the post-attention output.
        // The per-token FFN loop in prefillChunk copies both into the
        // single-token working buffers (hidden, hidden2) before calling
        // denseFfn/moeFfn, which fuses add(hidden, hidden2) + norm internally.
    }

    // ── Dense FFN (layers 0..first_k_dense_replace-1) ─────────────

    fn denseFfn(self: *Glm4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.intermediate_size;

        // Fused residual add + pre-FFN norm (hidden2 = deferred attention output).
        const nw = self.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        self.be.sync();

        // Fused gate+up+SiLU for MLX Q4 when megakernel enabled
        if (self.megakernel_enabled and self.mlx_bits == 4) blk: {
            var buf_gw: [name_buf_size]u8 = undefined;
            var buf_gs: [name_buf_size]u8 = undefined;
            var buf_gb: [name_buf_size]u8 = undefined;
            var buf_uw: [name_buf_size]u8 = undefined;
            var buf_us: [name_buf_size]u8 = undefined;
            var buf_ub: [name_buf_size]u8 = undefined;
            const gw = self.fmt.getTensor(std.fmt.bufPrint(&buf_gw, "model.layers.{d}.mlp.gate_proj.weight", .{li}) catch break :blk) orelse break :blk;
            const gs = self.fmt.getTensor(std.fmt.bufPrint(&buf_gs, "model.layers.{d}.mlp.gate_proj.scales", .{li}) catch break :blk) orelse break :blk;
            const gb = self.fmt.getTensor(std.fmt.bufPrint(&buf_gb, "model.layers.{d}.mlp.gate_proj.biases", .{li}) catch break :blk) orelse break :blk;
            const uw = self.fmt.getTensor(std.fmt.bufPrint(&buf_uw, "model.layers.{d}.mlp.up_proj.weight", .{li}) catch break :blk) orelse break :blk;
            const us = self.fmt.getTensor(std.fmt.bufPrint(&buf_us, "model.layers.{d}.mlp.up_proj.scales", .{li}) catch break :blk) orelse break :blk;
            const ub = self.fmt.getTensor(std.fmt.bufPrint(&buf_ub, "model.layers.{d}.mlp.up_proj.biases", .{li}) catch break :blk) orelse break :blk;
            if (gw.dtype == .mlx_q) {
                switch (self.be) {
                    inline else => |be| {
                        if (comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluMlxQ4")) {
                            be.fusedFfnGateUpSiluMlxQ4(
                                self.hidden2.ptr,
                                gw.data_ptr,
                                gs.data_ptr,
                                gb.data_ptr,
                                uw.data_ptr,
                                us.data_ptr,
                                ub.data_ptr,
                                self.ff_gate.ptr,
                                ff,
                                e,
                            );
                        } else break :blk;
                    },
                }
            } else break :blk;
        } else {
            try self.mlxLayerGemv(li, "mlp.gate_proj", self.hidden2, self.ff_gate[0..ff], ff, e);
            try self.mlxLayerGemv(li, "mlp.up_proj", self.hidden2, self.ff_up[0..ff], ff, e);
            self.applySwiGlu(ff);
        }

        try self.mlxLayerGemv(li, "mlp.down_proj", self.ff_gate[0..ff], self.hidden2, e, ff);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── MoE FFN (layers first_k_dense_replace..n_layers-1) ────────

    fn moeFfn(self: *Glm4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_intermediate_size;

        // Fused residual add + pre-FFN norm (hidden2 = deferred attention output).
        const nw = self.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.addRmsNorm(self.hidden.ptr, self.hidden2.ptr, self.normAsF32(nw, e), self.hidden2.ptr, e, self.rms_eps);

        // Router: sigmoid scoring
        const gate_t = self.layerTensor(li, "mlp.gate.weight") orelse return error.MissingTensor;
        // Router is NOT quantized (BF16), use standard GEMV
        self.be.gemv(self.hidden2.ptr, .{ .data = gate_t.data_ptr, .dtype = gate_t.dtype }, self.router_logits.ptr, self.n_routed_experts, e);

        // Sigmoid + bias correction
        self.be.sync();

        // Apply sigmoid to all router logits and save raw scores for weighting.
        if (self.n_routed_experts > max_active_experts * 16) @panic("glm4: n_routed_experts exceeds stack buffer limit");
        var raw_sigmoid: [max_active_experts * 16]f32 = undefined;
        for (0..self.n_routed_experts) |i| {
            raw_sigmoid[i] = math_ops.sigmoid(self.router_logits[i]);
        }

        // Use sigmoid + bias for top-k SELECTION only (bias shifts selection, not weights).
        // Reference: HF MoEGate.forward() applies bias only to get_topk_indices,
        // then gathers raw sigmoid scores for the selected experts.
        const bias_t = self.layerTensor(li, "mlp.gate.e_score_correction_bias");
        if (bias_t) |bt| {
            const bias_ptr: [*]const f32 = @ptrCast(@alignCast(bt.data_ptr));
            for (0..self.n_routed_experts) |i| {
                self.router_logits[i] = raw_sigmoid[i] + bias_ptr[i];
            }
        } else {
            for (0..self.n_routed_experts) |i| {
                self.router_logits[i] = raw_sigmoid[i];
            }
        }

        // Top-k selection (using bias-corrected scores)
        const top_k = self.num_experts_per_tok;
        var top_experts: [max_active_experts]usize = undefined;
        var top_scores_biased: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..self.n_routed_experts], top_k, top_experts[0..top_k], top_scores_biased[0..top_k]);

        // Gather raw sigmoid scores for the selected experts (NO bias)
        var top_scores: [max_active_experts]f32 = undefined;
        for (0..top_k) |i| {
            top_scores[i] = raw_sigmoid[top_experts[i]];
        }

        // Normalize scores
        var score_sum: f32 = 0.0;
        for (0..top_k) |ti| score_sum += top_scores[ti];
        if (score_sum > 0.0) {
            for (0..top_k) |ti| top_scores[ti] /= score_sum;
        }

        // Accumulate expert outputs, GPU addScaled avoids per-expert sync
        @memset(self.expert_buf, 0);
        for (0..top_k) |ti| {
            try self.expertFfn(li, @intCast(top_experts[ti]), self.hidden2, self.ff_down, ff, e);
            const w = top_scores[ti] * self.routed_scaling_factor;
            self.be.addScaled(self.ff_down.ptr, self.expert_buf.ptr, w, e);
        }

        // Shared expert (always active, scale = 1.0)
        try self.sharedExpertFfn(li, self.hidden2, self.ff_down, ff, e);
        self.be.addScaled(self.ff_down.ptr, self.expert_buf.ptr, 1.0, e);

        // Residual
        self.be.add(self.hidden.ptr, self.expert_buf.ptr, self.hidden.ptr, e);
    }

    fn expertFfn(self: *Glm4Model, li: u32, expert_id: u32, input: []const f32, output: []f32, ff: usize, e: usize) !void {
        // Expert weights are stacked: switch_mlp.gate_proj.weight shape [64, ff, e*6/32]
        try self.mlxExpertGemv(li, "mlp.switch_mlp.gate_proj", expert_id, input, self.ff_gate[0..ff], ff, e);
        try self.mlxExpertGemv(li, "mlp.switch_mlp.up_proj", expert_id, input, self.ff_up[0..ff], ff, e);

        // SwiGLU chains with preceding GPU gemv, no sync needed
        self.applySwiGlu(ff);

        try self.mlxExpertGemv(li, "mlp.switch_mlp.down_proj", expert_id, self.ff_gate[0..ff], output, e, ff);
    }

    fn sharedExpertFfn(self: *Glm4Model, li: u32, input: []const f32, output: []f32, ff: usize, e: usize) !void {
        try self.mlxLayerGemv(li, "mlp.shared_experts.gate_proj", input, self.ff_gate[0..ff], ff, e);
        try self.mlxLayerGemv(li, "mlp.shared_experts.up_proj", input, self.ff_up[0..ff], ff, e);

        // SwiGLU chains with preceding GPU gemv, no sync needed
        self.applySwiGlu(ff);

        try self.mlxLayerGemv(li, "mlp.shared_experts.down_proj", self.ff_gate[0..ff], output, e, ff);
    }

    // ── Helpers ──────────────────────────────────────────────────

    /// Return f32 view of a norm tensor, converting+caching non-f32 weights once.
    fn normAsF32(self: *Glm4Model, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype == .f32) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow, increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch @panic("normAsF32: out of memory converting norm weights");
        quant_ops.dequantToF32(buf, t.data_ptr, t.dtype, n);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }

    fn warmNormCache(self: *Glm4Model) void {
        const e: usize = self.n_embd;
        if (self.fmt.getTensor("model.norm.weight")) |t| _ = self.normAsF32(t, e);
        for (0..self.n_layers) |li| {
            const l: u32 = @intCast(li);
            if (self.layerTensor(l, "input_layernorm.weight")) |t| _ = self.normAsF32(t, e);
            if (self.layerTensor(l, "self_attn.q_a_layernorm.weight")) |t| _ = self.normAsF32(t, self.q_lora_rank);
            if (self.layerTensor(l, "self_attn.kv_a_layernorm.weight")) |t| _ = self.normAsF32(t, self.kv_lora_rank);
            if (self.layerTensor(l, "post_attention_layernorm.weight")) |t| _ = self.normAsF32(t, e);
        }
    }

    fn layerTensor(self: *Glm4Model, li: u32, suffix: []const u8) ?TensorInfo {
        // Try SafeTensors/HF name first (model.layers.N.suffix)
        const hf_name = std.fmt.bufPrint(&self.name_buf, "model.layers.{d}.{s}", .{ li, suffix }) catch return null;
        if (self.fmt.getTensor(hf_name)) |t| return t;
        // Try GGUF-specific translation for MLA attention tensors
        if (hfToGgufAttnName(suffix)) |gguf_name| {
            if (self.fmt.layerTensor(li, gguf_name)) |t| return t;
        }
        // Generic GGUF fallback: blk.N.suffix
        return self.fmt.layerTensor(li, suffix);
    }

    /// Translate HF attention layer prefix to GGUF tensor suffix.
    /// For GGUF: "self_attn.q_a_proj" → "attn_q_a.weight", etc.
    fn hfToGgufAttnName(prefix: []const u8) ?[]const u8 {
        const mappings = .{
            // MLA projection weights
            .{ "self_attn.q_a_proj", "attn_q_a.weight" },
            .{ "self_attn.q_b_proj", "attn_q_b.weight" },
            .{ "self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa.weight" },
            .{ "self_attn.embed_q", "attn_k_b.weight" },
            .{ "self_attn.unembed_out", "attn_v_b.weight" },
            // MLA norm weights
            .{ "self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight" },
            .{ "self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight" },
        };
        inline for (mappings) |m| {
            if (std.mem.eql(u8, prefix, m[0])) return m[1];
        }
        return null;
    }

    /// Layer GEMV: MLX quantized path or standard be.gemv for GGUF/BF16/F32.
    fn mlxLayerGemv(self: *Glm4Model, li: u32, prefix: []const u8, x: []const f32, y: []f32, n: usize, k: usize) !void {
        var buf: [name_buf_size]u8 = undefined;
        // Try SafeTensors HF name first
        const w_name = std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ li, prefix }) catch return error.MissingTensor;
        // Fallback to GGUF naming (blk.N.attn_q_a.weight, etc.)
        const w_t = self.fmt.getTensor(w_name) orelse blk: {
            if (hfToGgufAttnName(prefix)) |gguf_name| {
                break :blk self.fmt.layerTensor(li, gguf_name);
            }
            break :blk null;
        } orelse return error.MissingTensor;

        if (w_t.dtype == .mlx_q) {
            var buf2: [name_buf_size]u8 = undefined;
            var buf3: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&buf2, "model.layers.{d}.{s}.scales", .{ li, prefix }) catch return error.MissingTensor;
            const b_name = std.fmt.bufPrint(&buf3, "model.layers.{d}.{s}.biases", .{ li, prefix }) catch return error.MissingTensor;
            const s_t = self.fmt.getTensor(s_name) orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor(b_name) orelse return error.MissingTensor;
            self.be.gemvMlxQ(x.ptr, w_t.data_ptr, s_t.data_ptr, b_t.data_ptr, y.ptr, n, k, self.mlx_bits, model_mod.inferMlxGroupSize(s_t, k));
        } else {
            self.be.gemv(x.ptr, .{ .data = w_t.data_ptr, .dtype = w_t.dtype }, y.ptr, n, k);
        }
    }

    /// Top-level GEMV (e.g., "lm_head"): MLX quantized or standard be.gemv.
    fn mlxGemv(self: *Glm4Model, prefix: []const u8, x: []const f32, y: []f32, n: usize, k: usize) !void {
        var buf: [name_buf_size]u8 = undefined;
        const w_name = std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}) catch return error.MissingTensor;
        const w_t = self.fmt.getTensor(w_name) orelse return error.MissingTensor;

        if (w_t.dtype == .mlx_q) {
            var buf2: [name_buf_size]u8 = undefined;
            var buf3: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&buf2, "{s}.scales", .{prefix}) catch return error.MissingTensor;
            const b_name = std.fmt.bufPrint(&buf3, "{s}.biases", .{prefix}) catch return error.MissingTensor;
            const s_t = self.fmt.getTensor(s_name) orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor(b_name) orelse return error.MissingTensor;
            self.be.gemvMlxQ(x.ptr, w_t.data_ptr, s_t.data_ptr, b_t.data_ptr, y.ptr, n, k, self.mlx_bits, model_mod.inferMlxGroupSize(s_t, k));
        } else {
            self.be.gemv(x.ptr, .{ .data = w_t.data_ptr, .dtype = w_t.dtype }, y.ptr, n, k);
        }
    }

    /// GEMV for stacked expert weights (first dim = expert_id).
    fn mlxExpertGemv(self: *Glm4Model, li: u32, prefix: []const u8, expert_id: u32, x: []const f32, y: []f32, n: usize, k: usize) !void {
        var buf: [name_buf_size]u8 = undefined;
        const w_name = std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ li, prefix }) catch return error.MissingTensor;
        const w_t = self.fmt.getTensor(w_name) orelse return error.MissingTensor;

        if (w_t.dtype == .mlx_q) {
            var buf2: [name_buf_size]u8 = undefined;
            var buf3: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&buf2, "model.layers.{d}.{s}.scales", .{ li, prefix }) catch return error.MissingTensor;
            const b_name = std.fmt.bufPrint(&buf3, "model.layers.{d}.{s}.biases", .{ li, prefix }) catch return error.MissingTensor;
            const s_t = self.fmt.getTensor(s_name) orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor(b_name) orelse return error.MissingTensor;

            const gs: usize = model_mod.inferMlxGroupSize(s_t, k);
            const gpr = (k + gs - 1) / gs;
            const wpg = mlx_ops.wordsPerGroup(self.mlx_bits, gs);
            const wpr = gpr * wpg;

            const eid: usize = expert_id;
            // Byte offsets: weights are u32 words, scales/biases are u16 (bf16)
            const w_byte_offset = eid * n * wpr * @sizeOf(u32);
            const s_byte_offset = eid * n * gpr * @sizeOf(u16);

            self.be.gemvMlxQ(x.ptr, w_t.data_ptr + w_byte_offset, s_t.data_ptr + s_byte_offset, b_t.data_ptr + s_byte_offset, y.ptr, n, k, self.mlx_bits, @intCast(gs));
        } else {
            // Non-MLX expert: offset into expert slice
            const expert_bytes = dtypeBytes(w_t.dtype, n * k);
            const offset = @as(usize, expert_id) * expert_bytes;
            self.be.gemv(x.ptr, .{ .data = w_t.data_ptr + offset, .dtype = w_t.dtype }, y.ptr, n, k);
        }
    }

    /// Per-head linear projection. For GGUF, the weight is transposed per head
    /// ([in_dim, out_dim, nh] in GGUF convention), requiring dequant + transposed
    /// accumulation. For MLX, uses standard per-head mlxGemvRaw.
    fn multiLinearGemv(self: *Glm4Model, li: u32, prefix: []const u8, x: []const f32, y: [*]f32, nh: usize, out_dim: usize, in_dim: usize) !void {
        var buf: [name_buf_size]u8 = undefined;
        const w_name = std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ li, prefix }) catch return error.MissingTensor;
        const w_t = self.fmt.getTensor(w_name) orelse blk: {
            if (hfToGgufAttnName(prefix)) |gguf_name| {
                break :blk self.fmt.layerTensor(li, gguf_name);
            }
            break :blk null;
        } orelse return error.MissingTensor;

        if (w_t.dtype == .mlx_q) {
            var buf2: [name_buf_size]u8 = undefined;
            var buf3: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&buf2, "model.layers.{d}.{s}.scales", .{ li, prefix }) catch return error.MissingTensor;
            const b_name = std.fmt.bufPrint(&buf3, "model.layers.{d}.{s}.biases", .{ li, prefix }) catch return error.MissingTensor;
            const s_t = self.fmt.getTensor(s_name) orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor(b_name) orelse return error.MissingTensor;

            const inferred_gs: usize = model_mod.inferMlxGroupSize(s_t, in_dim);
            const groups_per_row = (in_dim + inferred_gs - 1) / inferred_gs;
            const wpg = mlx_ops.wordsPerGroup(self.mlx_bits, inferred_gs);
            const words_per_row = groups_per_row * wpg;

            for (0..nh) |h| {
                // Byte offsets: weights are u32 words, scales/biases are u16 (bf16)
                const w_byte_off = h * out_dim * words_per_row * @sizeOf(u32);
                const s_byte_off = h * out_dim * groups_per_row * @sizeOf(u16);
                self.be.gemvMlxQ(x.ptr, w_t.data_ptr + w_byte_off, s_t.data_ptr + s_byte_off, b_t.data_ptr + s_byte_off, y + h * out_dim, out_dim, in_dim, self.mlx_bits, @intCast(inferred_gs));
            }
        } else {
            // Non-MLX GGUF: dispatch via backend for in-kernel dequantization.
            // Detect layout from GGUF dims:
            // dims[0] = out_dim → transposed [in_dim × out_dim]
            // dims[0] = in_dim → standard [out_dim × in_dim]
            const head_bytes = dtypeBytes(w_t.dtype, out_dim * in_dim);
            const transposed = (w_t.n_dims >= 2 and w_t.dims[0] == out_dim);
            if (!transposed) {
                // Standard layout, backend handles dequant in-kernel.
                for (0..nh) |h| {
                    const w_ptr = w_t.data_ptr + h * head_bytes;
                    self.be.gemv(x.ptr, .{ .data = w_ptr, .dtype = w_t.dtype }, y + h * out_dim, out_dim, in_dim);
                }
            } else if (w_t.dtype == .q8_0) {
                // Transposed Q8_0, backend gemvT handles in-kernel.
                for (0..nh) |h| {
                    const w_ptr = w_t.data_ptr + h * head_bytes;
                    self.be.gemvT(x.ptr, w_ptr, y + h * out_dim, out_dim, in_dim);
                }
            } else {
                // Transposed non-Q8_0, gemvT only supports Q8_0, so CPU dequant
                // fallback until gemvT gains dtype support.
                const head_elems = out_dim * in_dim;
                if (head_elems > self.mla_scratch.len) @panic("glm4: head_elems exceeds mla_scratch buffer");
                const scratch = self.mla_scratch[0..head_elems];
                self.be.sync();
                for (0..nh) |h| {
                    const w_ptr = w_t.data_ptr + h * head_bytes;
                    quant_ops.dequantToF32(scratch, w_ptr, w_t.dtype, head_elems);
                    const y_head = y + h * out_dim;
                    @memset(y_head[0..out_dim], 0);
                    for (0..in_dim) |j| {
                        const xj = x[j];
                        const row = scratch[j * out_dim ..][0..out_dim];
                        for (0..out_dim) |i| y_head[i] += row[i] * xj;
                    }
                }
            }
        }
    }

    /// Apply SwiGLU activation in-place: gate = silu(gate) * up.
    /// Uses backend-dispatched siluMul for GPU acceleration and SIMD on CPU.
    fn applySwiGlu(self: *Glm4Model, ff: usize) void {
        self.be.siluMul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, ff);
    }

    /// Maximum rope half-dimension for precomputed frequency table.
    const max_rope_half: usize = 128;

    /// Apply RoPE only to the rope portion of each head (at offset nope_dim).
    /// Frequencies are precomputed once and reused across all heads.
    fn ropePartial(self: *Glm4Model, x: [*]f32, n_heads: usize, head_dim: usize, nope_dim: usize, rope_dim: usize) void {
        if (rope_dim % 2 != 0) @panic("glm4: rope_dim must be even");
        const half = rope_dim / 2;
        if (half > max_rope_half) @panic("glm4: rope half-dim exceeds max_rope_half");
        const p: f32 = @floatFromInt(self.kv_seq_len);
        const neg_log_theta: f32 = -@log(self.rope_theta);
        const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));

        // Precompute cos/sin table (head-independent)
        var cos_tab: [max_rope_half]f32 = undefined;
        var sin_tab: [max_rope_half]f32 = undefined;
        for (0..half) |i| {
            const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
            const angle = p * freq;
            cos_tab[i] = @cos(angle);
            sin_tab[i] = @sin(angle);
        }

        for (0..n_heads) |h| {
            const base = h * head_dim + nope_dim;
            for (0..half) |i| {
                const r = x[base + i];
                const im = x[base + i + half];
                x[base + i] = r * cos_tab[i] - im * sin_tab[i];
                x[base + i + half] = r * sin_tab[i] + im * cos_tab[i];
            }
        }
    }

    /// Byte size for `n` elements at the given dtype (for sub-tensor offset computation).
    fn dtypeBytes(dtype: DType, n: usize) usize {
        return switch (dtype) {
            .f32 => n * backend_mod.f32_elem_bytes,
            .bf16, .f16 => n * backend_mod.f16_elem_bytes,
            .q8_0 => @divExact(n, backend_mod.quant_block_elems) * backend_mod.q8_0_block_bytes,
            .q4_0 => @divExact(n, backend_mod.quant_block_elems) * backend_mod.q4_0_block_bytes,
            .q4_k => @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q4_k_block_bytes,
            .q5_k => @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q5_k_block_bytes,
            .q6_k => @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q6_k_block_bytes,
            else => n, // fallback: 1 byte per element
        };
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "GLM4 dtypeBytes f32" {
    // 256 f32 elements = 256 * 4 = 1024 bytes
    try std.testing.expectEqual(@as(usize, 1024), Glm4Model.dtypeBytes(.f32, 256));
}

test "GLM4 dtypeBytes bf16" {
    // 256 bf16 elements = 256 * 2 = 512 bytes
    try std.testing.expectEqual(@as(usize, 512), Glm4Model.dtypeBytes(.bf16, 256));
}

test "GLM4 dtypeBytes q8_0" {
    // q8_0: 32 elements per block, each block = 34 bytes (32 int8 + 2 byte scale)
    // 256 elements = 8 blocks * 34 = 272 bytes
    const n: usize = 256;
    const expected = @divExact(n, backend_mod.quant_block_elems) * backend_mod.q8_0_block_bytes;
    try std.testing.expectEqual(expected, Glm4Model.dtypeBytes(.q8_0, n));
}

test "GLM4 dtypeBytes unknown fallback" {
    // Unknown dtype: 1 byte per element
    try std.testing.expectEqual(@as(usize, 256), Glm4Model.dtypeBytes(.mlx_q, 256));
}

test "GLM4 ropePartial position 0 is identity" {
    // At position 0, angle=0 for all frequencies: cos(0)=1, sin(0)=0
    // So RoPE should be identity
    var m: Glm4Model = undefined;
    m.kv_seq_len = 0;
    m.rope_theta = 1000000.0;
    // 1 head, head_dim=8, nope_dim=4, rope_dim=4
    // rope applies to indices [4..8] within the head
    var x = [_]f32{ 10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0 };
    m.ropePartial(&x, 1, 8, 4, 4);
    // nope portion unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), x[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), x[3], 1e-5);
    // rope portion at pos=0: identity
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x[7], 1e-5);
}

test "GLM4 model vtable compiles" {
    try std.testing.expect(@hasDecl(Glm4Model, "forward"));
    try std.testing.expect(@hasDecl(Glm4Model, "prefill"));
    try std.testing.expect(@hasDecl(Glm4Model, "resetCache"));
    try std.testing.expect(@hasDecl(Glm4Model, "cancel"));
    try std.testing.expect(@hasDecl(Glm4Model, "model"));
}

test "GLM4 dtypeBytes q4_0" {
    const n: usize = 256;
    const expected = @divExact(n, backend_mod.quant_block_elems) * backend_mod.q4_0_block_bytes;
    try std.testing.expectEqual(expected, Glm4Model.dtypeBytes(.q4_0, n));
}

test "GLM4 dtypeBytes q4_k" {
    const n: usize = 256;
    const expected = @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q4_k_block_bytes;
    try std.testing.expectEqual(expected, Glm4Model.dtypeBytes(.q4_k, n));
}

test "GLM4 dtypeBytes q5_k" {
    const n: usize = 256;
    const expected = @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q5_k_block_bytes;
    try std.testing.expectEqual(expected, Glm4Model.dtypeBytes(.q5_k, n));
}

test "GLM4 dtypeBytes q6_k" {
    const n: usize = 256;
    const expected = @divExact(n, backend_mod.quant_super_block_elems) * backend_mod.q6_k_block_bytes;
    try std.testing.expectEqual(expected, Glm4Model.dtypeBytes(.q6_k, n));
}

test "GLM4 dtypeBytes f16" {
    try std.testing.expectEqual(@as(usize, 512), Glm4Model.dtypeBytes(.f16, 256));
}

test "GLM4 ropePartial non-zero position rotates" {
    // At position 1 with small rope_theta, angles are non-trivial.
    // Verify the rotation actually changes values and preserves nope dims.
    var m: Glm4Model = undefined;
    m.kv_seq_len = 1;
    m.rope_theta = 10.0; // small theta for visible rotation
    var x = [_]f32{ 10.0, 20.0, 1.0, 0.0, 0.0, 1.0 };
    // 1 head, head_dim=6, nope_dim=2, rope_dim=4
    m.ropePartial(&x, 1, 6, 2, 4);
    // nope portion [0..2] must be unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), x[1], 1e-5);
    // rope portion must have changed (freq at i=0: exp(-ln(10)*0/4)=1.0, angle=1.0)
    // cos(1) ~ 0.5403, sin(1) ~ 0.8415
    // x[2] was 1.0, x[4] was 0.0 → new x[2] = 1.0*cos(1) - 0.0*sin(1) = cos(1)
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[2], 1e-4);
    // x[4] = 1.0*sin(1) + 0.0*cos(1) = sin(1)
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x[4], 1e-4);
}

test "GLM4 ropePartial multi-head" {
    // Verify RoPE is applied independently per head
    var m: Glm4Model = undefined;
    m.kv_seq_len = 0; // pos=0 → identity
    m.rope_theta = 1000000.0;
    // 2 heads, head_dim=4, nope_dim=2, rope_dim=2
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    m.ropePartial(&x, 2, 4, 2, 2);
    // At pos=0, all values should be unchanged (identity rotation)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), x[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), x[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), x[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), x[7], 1e-5);
}

test "GLM4 default config constants" {
    // Verify default struct field values match the GLM-4 MoE Lite spec.
    // fmt/be/allocator have no defaults; use undefined for them, all other fields initialized.
    const m = Glm4Model{ .fmt = undefined, .be = undefined, .allocator = undefined };
    try std.testing.expectEqual(@as(u32, 47), m.n_layers);
    try std.testing.expectEqual(@as(u32, 2048), m.n_embd);
    try std.testing.expectEqual(@as(u32, 20), m.n_head);
    try std.testing.expectEqual(@as(u32, 768), m.q_lora_rank);
    try std.testing.expectEqual(@as(u32, 512), m.kv_lora_rank);
    try std.testing.expectEqual(@as(u32, 192), m.qk_nope_head_dim);
    try std.testing.expectEqual(@as(u32, 64), m.qk_rope_head_dim);
    try std.testing.expectEqual(@as(u32, 256), m.v_head_dim);
    try std.testing.expectEqual(@as(u32, 64), m.n_routed_experts);
    try std.testing.expectEqual(@as(u32, 4), m.num_experts_per_tok);
    try std.testing.expectEqual(@as(u32, 1), m.first_k_dense_replace);
    try std.testing.expectEqual(@as(u32, 6), m.mlx_bits);
    try std.testing.expectApproxEqAbs(@as(f32, 1.8), m.routed_scaling_factor, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1000000.0), m.rope_theta, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), m.rms_eps, 1e-10);
}

test "GLM4 module-level constants" {
    try std.testing.expectEqual(@as(usize, 8), max_active_experts);
    try std.testing.expectEqual(@as(usize, 128), Glm4Model.max_rope_half);
    try std.testing.expectEqual(@as(u32, 6), default_glm4_mlx_bits);
}

test "GLM4 cancel and cancelled flag" {
    var m: Glm4Model = undefined;
    m.cancelled = std.atomic.Value(bool).init(false);
    try std.testing.expect(!m.cancelled.load(.monotonic));
    m.cancelled.store(true, .monotonic);
    try std.testing.expect(m.cancelled.load(.monotonic));
}

test "GLM4 pub fn signatures" {
    // Compile-time verification that public function signatures exist and have expected types.
    comptime {
        _ = @TypeOf(Glm4Model.init);
        _ = @TypeOf(Glm4Model.deinit);
        _ = @TypeOf(Glm4Model.forward);
        _ = @TypeOf(Glm4Model.prefill);
        _ = @TypeOf(Glm4Model.resetCache);
        _ = @TypeOf(Glm4Model.cancel);
        _ = @TypeOf(Glm4Model.model);
        _ = @TypeOf(Glm4Model.getBlockTable);
    }
}

test "fuzz: all glm4 functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // -- Comptime verification of all pub functions that need full Backend/Format --
            comptime {
                // init: Allocator, Format, Backend, u32, KvQuantType, KvQuantType, ?*TieredKvCache -> !Glm4Model
                _ = &Glm4Model.init;
                // deinit: *Glm4Model -> void
                _ = &Glm4Model.deinit;
                // model: *Glm4Model -> Model
                _ = &Glm4Model.model;
                // forward: *Glm4Model, u32 -> !u32
                _ = &Glm4Model.forward;
                // prefill: *Glm4Model, []const u32 -> !u32
                _ = &Glm4Model.prefill;
                // resetCache: *Glm4Model -> void
                _ = &Glm4Model.resetCache;
                // getBlockTable: *Glm4Model -> []const u32
                _ = &Glm4Model.getBlockTable;
            }

            // -- cancel: directly testable (only touches atomic bool) --
            {
                var m: Glm4Model = undefined;
                m.cancelled = std.atomic.Value(bool).init(false);
                const pre = m.cancelled.load(.monotonic);
                try std.testing.expect(!pre);
                m.cancel();
                try std.testing.expect(m.cancelled.load(.monotonic));
            }

            // -- ropePartial: fuzz with random inputs (private but in-file test access) --
            {
                var m: Glm4Model = undefined;
                m.kv_seq_len = smith.valueWithHash(u8, 0);
                m.rope_theta = @as(f32, @floatFromInt(smith.valueWithHash(u8, 1) | 1)) * 10.0;
                // 2 heads, head_dim=6, nope_dim=2, rope_dim=4
                const n_heads: usize = 2;
                const head_dim: usize = 6;
                const nope_dim: usize = 2;
                const rope_dim: usize = 4;
                var x: [n_heads * head_dim]f32 = undefined;
                for (&x, 0..) |*v, i| {
                    v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @as(u32, @truncate(i +% 2)))));
                }
                m.ropePartial(&x, n_heads, head_dim, nope_dim, rope_dim);
                // All outputs must be finite
                for (x) |v| try std.testing.expect(std.math.isFinite(v));
            }

            // -- dtypeBytes: fuzz with random element counts (must be block-aligned) --
            {
                // f32 path: any n works
                const n_f32: usize = @as(usize, smith.valueWithHash(u8, 3)) + 1;
                const b_f32 = Glm4Model.dtypeBytes(.f32, n_f32);
                try std.testing.expect(b_f32 >= n_f32);

                // bf16 path
                const b_bf16 = Glm4Model.dtypeBytes(.bf16, n_f32);
                try std.testing.expect(b_bf16 >= 1);

                // f16 path
                const b_f16 = Glm4Model.dtypeBytes(.f16, n_f32);
                try std.testing.expectEqual(b_bf16, b_f16);

                // q8_0 path: needs block alignment (32 elements)
                const blocks8: usize = @as(usize, smith.valueWithHash(u4, 4)) + 1;
                const n8 = blocks8 * backend_mod.quant_block_elems;
                const b_q8 = Glm4Model.dtypeBytes(.q8_0, n8);
                try std.testing.expect(b_q8 > 0);

                // q4_0 path
                const b_q4 = Glm4Model.dtypeBytes(.q4_0, n8);
                try std.testing.expect(b_q4 > 0 and b_q4 <= b_q8);

                // q4_k path: needs super-block alignment (256 elements)
                const sblocks: usize = @as(usize, smith.valueWithHash(u2, 5)) + 1;
                const ns = sblocks * backend_mod.quant_super_block_elems;
                const b_q4k = Glm4Model.dtypeBytes(.q4_k, ns);
                try std.testing.expect(b_q4k > 0);
                const b_q5k = Glm4Model.dtypeBytes(.q5_k, ns);
                try std.testing.expect(b_q5k >= b_q4k);
                const b_q6k = Glm4Model.dtypeBytes(.q6_k, ns);
                try std.testing.expect(b_q6k >= b_q5k);

                // mlx_q fallback: 1 byte per element
                const b_mlx = Glm4Model.dtypeBytes(.mlx_q, n_f32);
                try std.testing.expectEqual(n_f32, b_mlx);
            }
        }
    }.f, .{});
}
