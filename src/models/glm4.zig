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
    n_head_kv: u32 = 20, // not used for internal KV computation (MLA has its own compressed KV path); set equal to n_head for Model vtable reporting
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

    // KV cache (PagedAttention or TieredKvCache): store full reconstructed K and V per layer
    paged_cache: PagedKvCache = undefined,
    seq_table: SeqBlockTable = undefined,
    block_allocator: BlockAllocator = undefined,
    tiered_cache: ?*TieredKvCache = null,
    tiered_block_allocator: ?TieredBlockAllocator = null,
    kv_type_k: kv_quant.KvQuantType = .f32,
    kv_type_v: kv_quant.KvQuantType = .f32,
    kv_seq_len: usize = 0,
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // Name buffer for tensor lookups
    name_buf: [name_buf_size]u8 = undefined,

    // Thread pool for parallel CPU work
    pool: ?*@import("../thread_pool.zig").ThreadPool = null,

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
        if (f.getMetaU32("q_lora_rank")) |v| self.q_lora_rank = v;
        if (f.getMetaU32("kv_lora_rank")) |v| self.kv_lora_rank = v;
        if (f.getMetaU32("qk_nope_head_dim")) |v| self.qk_nope_head_dim = v;
        if (f.getMetaU32("qk_rope_head_dim")) |v| self.qk_rope_head_dim = v;
        if (f.getMetaU32("v_head_dim")) |v| self.v_head_dim = v;
        if (f.getMetaU32("intermediate_size")) |v| self.intermediate_size = v;
        if (f.getMetaU32("moe_intermediate_size")) |v| self.moe_intermediate_size = v;
        if (f.getMetaU32("n_routed_experts")) |v| self.n_routed_experts = v;
        if (f.getMetaU32("num_experts_per_tok")) |v| self.num_experts_per_tok = v;
        if (f.getMetaU32("first_k_dense_replace")) |v| self.first_k_dense_replace = v;
        if (f.getMetaU32("eos_token_id")) |v| self.eos_token_id = v;
        if (f.getMetaF32("routed_scaling_factor")) |v| self.routed_scaling_factor = v;
        if (f.getMetaF32("rope_theta")) |v| self.rope_theta = v;
        if (f.getMetaF32("rms_norm_eps")) |v| self.rms_eps = v;
        self.mlx_bits = f.getMetaU32("bits") orelse 6;
        if (f.getMetaU32("context_length")) |cl| self.max_seq_len = cl;
        if (ctx_size > 0) self.max_seq_len = ctx_size;

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
            // One block per layer spanning the full context — MLA attention
            // indexes the KV cache flat (pos * kvd), not via block tables.
            const block_size: u16 = @intCast(self.max_seq_len);
            const num_blocks = nl;
            self.paged_cache = try PagedKvCache.init(allocator, nl, max_kv_dim, num_blocks, block_size);
            errdefer self.paged_cache.deinit();
            // BlockAllocator stores a pointer — must point to self.paged_cache (not a local copy).
            self.block_allocator = BlockAllocator.init(&self.paged_cache, allocator);
            self.seq_table = try self.block_allocator.allocateSeqTable(nl);
            errdefer self.block_allocator.freeSeqTable(&self.seq_table);
            try self.block_allocator.appendBlock(&self.seq_table);
        }

        return self;
    }

    /// Release all heap allocations owned by this model.
    pub fn deinit(self: *Glm4Model) void {
        if (self.tiered_block_allocator) |*ta| {
            ta.freeSeqTable(&self.seq_table);
        } else {
            self.block_allocator.freeSeqTable(&self.seq_table);
            self.paged_cache.deinit();
        }

        const bufs = .{
            &self.hidden,     &self.hidden2,    &self.q_compressed,
            &self.q_full,     &self.kv_proj,    &self.kv_latent,
            &self.k_buf,      &self.v_buf,      &self.attn_out,
            &self.scores_buf, &self.ff_gate,    &self.ff_up,
            &self.ff_down,    &self.expert_buf, &self.router_logits,
            &self.logits_buf,
        };
        inline for (bufs) |buf| self.allocator.free(buf.*);
    }

    /// Wrap this model in the generic `Model` interface.
    pub fn model(self: *Glm4Model) Model {
        return Model.from(Glm4Model, self);
    }

    // ── Forward pass ─────────────────────────────────────────────

    /// Run one decode step, returning the argmax next-token ID.
    pub fn forward(self: *Glm4Model, token_id: u32) !u32 {
        if (self.kv_seq_len >= self.max_seq_len) return error.KVCacheFull;

        try model_mod.ensureKvBlock(self);

        // Embedding lookup
        try self.embLookup(token_id);

        for (0..self.n_layers) |li| {
            if (self.cancelled.load(.acquire)) return error.Cancelled;
            const l: u32 = @intCast(li);
            try self.mlaAttention(l);
            if (li < self.first_k_dense_replace) {
                try self.denseFfn(l);
            } else {
                try self.moeFfn(l);
            }
        }

        // Final norm → logits
        const nw = self.fmt.getTensor("model.norm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, @ptrCast(@alignCast(nw.data_ptr)), self.hidden.ptr, self.n_embd, self.rms_eps);

        // LM head (may be quantized)
        self.be.sync();
        try self.mlxGemv("lm_head", self.hidden, self.logits_buf, self.vocab_size, self.n_embd);

        self.kv_seq_len += 1;
        self.be.sync();
        return math_ops.argmax(self.logits_buf);
    }

    /// Batched prefill — sequential. MLA attention could be batched but MoE
    /// routing selects different experts per token, preventing batched FFN.
    /// Sigmoid gating (independent expert scores) adds further complexity.
    pub fn prefill(self: *Glm4Model, token_ids: []const u32) !u32 {
        var last: u32 = 0;
        for (token_ids) |tid| last = try self.forward(tid);
        return last;
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
        self.be.rmsNorm(self.hidden.ptr, @ptrCast(@alignCast(nw.data_ptr)), self.hidden2.ptr, e, self.rms_eps);

        // 2. Q path: hidden2 → q_a_proj(e→q_lora_rank) → layernorm → q_b_proj(q_lora_rank→nh*q_head_dim)
        self.be.sync();
        try self.mlxLayerGemv(li, "self_attn.q_a_proj", self.hidden2, self.q_compressed, self.q_lora_rank, e);
        const qn = self.layerTensor(li, "self_attn.q_a_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.q_compressed.ptr, @ptrCast(@alignCast(qn.data_ptr)), self.q_compressed.ptr, self.q_lora_rank, self.rms_eps);
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
        self.be.rmsNorm(self.kv_latent.ptr, @ptrCast(@alignCast(kvn.data_ptr)), self.kv_latent.ptr, kv_rank, self.rms_eps);

        // 4. Per-head K_nope and V from kv_latent via embed_q and unembed_out
        // embed_q: [nh, nope_dim, kv_rank] → K_nope per head (HF tensor name; despite the name, projects kv_latent into K_nope)
        // unembed_out: [nh, v_head_dim, kv_rank] → V per head
        self.be.sync();
        try self.multiLinearGemv(li, "self_attn.embed_q", self.kv_latent, self.k_buf.ptr, nh, nope_dim, kv_rank);
        try self.multiLinearGemv(li, "self_attn.unembed_out", self.kv_latent, self.v_buf.ptr, nh, vhd, kv_rank);

        // 5. Assemble full K per head: [k_nope(nope_dim), k_pe(rope_dim)]
        // k_buf currently has [nh * nope_dim] from embed_q
        // We need to interleave k_pe into each head's K
        // Shift k_nope data to make room for k_pe in each head
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

        // 6. RoPE on q_pe and k_pe portions
        // q_pe is at offset nope_dim within each head of q_full
        // k_pe is at offset nope_dim within each head of k_buf
        // We need to apply RoPE only to the rope portion
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
            // Inline CPU softmax — avoids backend dispatch + sync overhead
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

        // 10. Residual
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── Dense FFN (layers 0..first_k_dense_replace-1) ─────────────

    fn denseFfn(self: *Glm4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.intermediate_size;

        const nw = self.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, @ptrCast(@alignCast(nw.data_ptr)), self.hidden2.ptr, e, self.rms_eps);

        self.be.sync();
        try self.mlxLayerGemv(li, "mlp.gate_proj", self.hidden2, self.ff_gate[0..ff], ff, e);
        try self.mlxLayerGemv(li, "mlp.up_proj", self.hidden2, self.ff_up[0..ff], ff, e);

        // SwiGLU: silu(gate) * up — chains with preceding GPU gemv ops
        self.applySwiGlu(ff);

        try self.mlxLayerGemv(li, "mlp.down_proj", self.ff_gate[0..ff], self.hidden2, e, ff);
        self.be.add(self.hidden.ptr, self.hidden2.ptr, self.hidden.ptr, e);
    }

    // ── MoE FFN (layers first_k_dense_replace..n_layers-1) ────────

    fn moeFfn(self: *Glm4Model, li: u32) !void {
        const e: usize = self.n_embd;
        const ff: usize = self.moe_intermediate_size;

        const nw = self.layerTensor(li, "post_attention_layernorm.weight") orelse return error.MissingTensor;
        self.be.rmsNorm(self.hidden.ptr, @ptrCast(@alignCast(nw.data_ptr)), self.hidden2.ptr, e, self.rms_eps);

        // Router: sigmoid scoring
        const gate_t = self.layerTensor(li, "mlp.gate.weight") orelse return error.MissingTensor;
        // Router is NOT quantized (BF16), use standard GEMV
        self.be.gemv(self.hidden2.ptr, .{ .data = gate_t.data_ptr, .dtype = gate_t.dtype }, self.router_logits.ptr, self.n_routed_experts, e);

        // Sigmoid + bias correction
        self.be.sync();
        const bias_t = self.layerTensor(li, "mlp.gate.e_score_correction_bias");
        for (0..self.n_routed_experts) |i| {
            var score = math_ops.sigmoid(self.router_logits[i]);
            if (bias_t) |bt| {
                const bias_ptr: [*]const f32 = @ptrCast(@alignCast(bt.data_ptr));
                score += bias_ptr[i];
            }
            self.router_logits[i] = score;
        }

        // Top-k selection
        const top_k = self.num_experts_per_tok;
        var top_experts: [max_active_experts]usize = undefined;
        var top_scores: [max_active_experts]f32 = undefined;
        math_ops.topKExperts(self.router_logits[0..self.n_routed_experts], top_k, top_experts[0..top_k], top_scores[0..top_k]);

        // Normalize scores
        var score_sum: f32 = 0.0;
        for (0..top_k) |ti| score_sum += top_scores[ti];
        if (score_sum > 0) {
            for (0..top_k) |ti| top_scores[ti] /= score_sum;
        }

        // Accumulate expert outputs — GPU addScaled avoids per-expert sync
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
        // We need to index into expert_id's slice
        try self.mlxExpertGemv(li, "mlp.switch_mlp.gate_proj", expert_id, input, self.ff_gate[0..ff], ff, e);
        try self.mlxExpertGemv(li, "mlp.switch_mlp.up_proj", expert_id, input, self.ff_up[0..ff], ff, e);

        // SwiGLU chains with preceding GPU gemv — no sync needed
        self.applySwiGlu(ff);

        try self.mlxExpertGemv(li, "mlp.switch_mlp.down_proj", expert_id, self.ff_gate[0..ff], output, e, ff);
    }

    fn sharedExpertFfn(self: *Glm4Model, li: u32, input: []const f32, output: []f32, ff: usize, e: usize) !void {
        try self.mlxLayerGemv(li, "mlp.shared_experts.gate_proj", input, self.ff_gate[0..ff], ff, e);
        try self.mlxLayerGemv(li, "mlp.shared_experts.up_proj", input, self.ff_up[0..ff], ff, e);

        // SwiGLU chains with preceding GPU gemv — no sync needed
        self.applySwiGlu(ff);

        try self.mlxLayerGemv(li, "mlp.shared_experts.down_proj", self.ff_gate[0..ff], output, e, ff);
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn layerTensor(self: *Glm4Model, li: u32, suffix: []const u8) ?TensorInfo {
        const name = std.fmt.bufPrint(&self.name_buf, "model.layers.{d}.{s}", .{ li, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }

    /// Layer GEMV: MLX quantized path or standard be.gemv for GGUF/BF16/F32.
    fn mlxLayerGemv(self: *Glm4Model, li: u32, prefix: []const u8, x: []const f32, y: []f32, n: usize, k: usize) !void {
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
            mlx_ops.mlxGemvRaw(x.ptr, @ptrCast(@alignCast(w_t.data_ptr)), @ptrCast(@alignCast(s_t.data_ptr)), @ptrCast(@alignCast(b_t.data_ptr)), y.ptr, n, k, self.mlx_bits);
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
            mlx_ops.mlxGemvRaw(x.ptr, @ptrCast(@alignCast(w_t.data_ptr)), @ptrCast(@alignCast(s_t.data_ptr)), @ptrCast(@alignCast(b_t.data_ptr)), y.ptr, n, k, self.mlx_bits);
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

            const gs = mlx_ops.mlx_group_size;
            const gpr = (k + gs - 1) / gs;
            const wpg = mlx_ops.wordsPerGroup(self.mlx_bits);
            const wpr = gpr * wpg;

            const pw: [*]const u32 = @ptrCast(@alignCast(w_t.data_ptr));
            const sc: [*]const u16 = @ptrCast(@alignCast(s_t.data_ptr));
            const bi: [*]const u16 = @ptrCast(@alignCast(b_t.data_ptr));

            const eid: usize = expert_id;
            const w_offset = eid * n * wpr;
            const s_offset = eid * n * gpr;

            mlx_ops.mlxGemvRaw(x.ptr, pw + w_offset, sc + s_offset, bi + s_offset, y.ptr, n, k, self.mlx_bits);
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
        const w_t = self.fmt.getTensor(w_name) orelse return error.MissingTensor;

        if (w_t.dtype == .mlx_q) {
            var buf2: [name_buf_size]u8 = undefined;
            var buf3: [name_buf_size]u8 = undefined;
            const s_name = std.fmt.bufPrint(&buf2, "model.layers.{d}.{s}.scales", .{ li, prefix }) catch return error.MissingTensor;
            const b_name = std.fmt.bufPrint(&buf3, "model.layers.{d}.{s}.biases", .{ li, prefix }) catch return error.MissingTensor;
            const s_t = self.fmt.getTensor(s_name) orelse return error.MissingTensor;
            const b_t = self.fmt.getTensor(b_name) orelse return error.MissingTensor;

            const group_size = mlx_ops.mlx_group_size;
            const groups_per_row = (in_dim + group_size - 1) / group_size;
            const wpg = mlx_ops.wordsPerGroup(self.mlx_bits);
            const words_per_row = groups_per_row * wpg;

            const pw: [*]const u32 = @ptrCast(@alignCast(w_t.data_ptr));
            const sc: [*]const u16 = @ptrCast(@alignCast(s_t.data_ptr));
            const bi: [*]const u16 = @ptrCast(@alignCast(b_t.data_ptr));

            for (0..nh) |h| {
                const w_off = h * out_dim * words_per_row;
                const s_off = h * out_dim * groups_per_row;
                mlx_ops.mlxGemvRaw(x.ptr, pw + w_off, sc + s_off, bi + s_off, y + h * out_dim, out_dim, in_dim, self.mlx_bits);
            }
        } else {
            // Non-MLX GGUF: dispatch via backend for in-kernel dequantization.
            // Detect layout from GGUF dims:
            // dims[0] = out_dim → transposed [in_dim × out_dim]
            // dims[0] = in_dim → standard [out_dim × in_dim]
            const head_bytes = dtypeBytes(w_t.dtype, out_dim * in_dim);
            const transposed = (w_t.n_dims >= 2 and w_t.dims[0] == out_dim);
            if (!transposed) {
                // Standard layout — backend handles dequant in-kernel.
                for (0..nh) |h| {
                    const w_ptr = w_t.data_ptr + h * head_bytes;
                    self.be.gemv(x.ptr, .{ .data = w_ptr, .dtype = w_t.dtype }, y + h * out_dim, out_dim, in_dim);
                }
            } else if (w_t.dtype == .q8_0) {
                // Transposed Q8_0 — backend gemvT handles in-kernel.
                for (0..nh) |h| {
                    const w_ptr = w_t.data_ptr + h * head_bytes;
                    self.be.gemvT(x.ptr, w_ptr, y + h * out_dim, out_dim, in_dim);
                }
            } else {
                // Transposed non-Q8_0 — gemvT only supports Q8_0, so CPU dequant
                // fallback until gemvT gains dtype support.
                const head_elems = out_dim * in_dim;
                std.debug.assert(head_elems <= self.logits_buf.len);
                const scratch = self.logits_buf[0..head_elems];
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

    /// Apply RoPE only to the rope portion of each head (at offset nope_dim)
    fn ropePartial(self: *Glm4Model, x: [*]f32, n_heads: usize, head_dim: usize, nope_dim: usize, rope_dim: usize) void {
        std.debug.assert(rope_dim % 2 == 0);
        const half = rope_dim / 2;
        const p: f32 = @floatFromInt(self.kv_seq_len);
        for (0..n_heads) |h| {
            const base = h * head_dim + nope_dim;
            for (0..half) |i| {
                const freq = @exp(-@log(self.rope_theta) * @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rope_dim)));
                const angle = p * freq;
                const r = x[base + i];
                const im = x[base + i + half];
                x[base + i] = r * @cos(angle) - im * @sin(angle);
                x[base + i + half] = r * @sin(angle) + im * @cos(angle);
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
