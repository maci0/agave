//! Vision encoder for multimodal inference — supports multiple mmproj architectures.
//!
//! Loads weights from an mmproj GGUF file and runs the vision transformer
//! forward pass: image -> patch embedding -> ViT blocks -> LLM projection.
//!
//! Supported architectures (auto-detected from available tensors):
//!
//!   Gemma 4 SigLIP-2 (`gemma4_siglip2`):
//!     - Conv2D patch embedding (16×16 patches), no bias
//!     - Learned 2D position encoding [embd_dim, max_pos, 2]
//!     - ViT blocks with: QK RMSNorm, post-attn/FFN RMSNorm, SwiGLU FFN
//!     - Single linear projection (mm.input_projection.weight)
//!     - 26B-A4B sub-variant: 768×768 effective image, 3×3 spatial merge → 256 tokens,
//!       standardization via v.std_scale/v.std_bias (embd=1152, blocks=27, heads=16)
//!     - E2B/E4B sub-variant: 224×224 image, no merge → 196 tokens,
//!       no standardization (embd=768, blocks=16, heads=12). mmproj also contains
//!       a.blk.* audio encoder tensors which are ignored by the vision path.
//!
//!   Gemma 3 SigLIP (`gemma3_siglip`):
//!     - Conv2D patch embedding (14×14 patches from 896×896 image), with bias
//!     - Learned 1D position embedding [embd_dim, n_patches]
//!     - ViT blocks with: LayerNorm (with bias), GELU FFN (up+down, no gate), no QK norms
//!     - Post-encoder LayerNorm (v.post_ln), then mm.soft_emb_norm + mm.input_projection
//!
//!   Qwen VL (`qwen_vl`):
//!     - Conv2D patch embedding with bias, dual patch weights (only first used)
//!     - Learned 1D position embedding
//!     - ViT blocks with: fused QKV projection, LayerNorm (with bias), GELU FFN, no QK norms
//!     - Post-encoder LayerNorm (v.post_ln), then MLP projector (mm.0 + GELU + mm.2)

const std = @import("std");
const Io = std.Io;
const math = std.math;
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");
const quant = @import("../ops/quant.zig");
const math_ops = @import("../ops/math.zig");
const ThreadPool = @import("../thread_pool.zig").ThreadPool;

const Backend = backend_mod.Backend;
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const Allocator = std.mem.Allocator;

// ── Named constants ──────────────────────────────────────────────

/// Maximum number of ViT transformer blocks (compile-time array size).
const max_blocks: usize = 64;
/// Maximum norm cache entries (2 norms/block * max_blocks + misc).
const max_norm_entries: usize = 256;
/// Vision tensor name buffer size.
const name_buf_size: usize = 64;
/// Number of image channels (RGB).
const n_channels: usize = 3;
/// Pixel value normalization: map [0..255] to [0..1].
const pixel_scale: f32 = 1.0 / 255.0;
/// Query chunk size for chunked attention (number of query rows per tile).
/// When n_patches exceeds this, attention is computed in chunks to keep the
/// scores buffer at n_heads * chunk_size * n_patches instead of n_heads * n_patches^2.
/// With 4096 patches and chunk=64: 16 * 64 * 4096 * 4 = 16MB (vs 1GB for full).
const attention_chunk_size: u32 = 64;
/// SIMD vector width for attention dot products and value accumulation.
const vec_len: usize = 8;

/// Gemma 4 spatial merge factor: 3×3 average pooling after ViT.
/// Reduces n_patches by 9x (e.g. 48×48=2304 → 16×16=256).
const gemma4_merge_kernel: u32 = 3;
/// ViT RoPE theta for Gemma 4 SigLIP-2 (from llama.cpp hparams.rope_theta).
const vit_rope_theta: f32 = 100.0;
/// Maximum half-head dimension for RoPE precomputation.
const max_rope_half: usize = 128;
/// Default Gemma 4 effective image size: 768 = 48 patches/side × 16 pixels/patch.
/// Chosen to produce 2304 pre-merge patches → 256 post-merge tokens.
const gemma4_effective_image_size: u32 = 768;

// ── Default SigLIP-2 config (Gemma 4 mmproj) ────────────────────

const default_image_size: u32 = 224;
const default_patch_size: u32 = 16;
const default_embd_dim: u32 = 1152;
const default_ffn_dim: u32 = 4304;
const default_n_blocks: u32 = 27;
const default_n_heads: u32 = 16;
const default_projection_dim: u32 = 2816;
const default_norm_eps: f32 = 1e-6;

const gelu_sqrt_2_over_pi = math_ops.sqrt_2_over_pi;
const gelu_cubic_coeff = math_ops.gelu_coeff;

/// Vision encoder architecture variant, auto-detected from available tensors.
const VisionVariant = enum {
    /// Gemma 4 SigLIP-2: QK norms, SwiGLU, 2D pos, no bias, mm.input_projection.
    /// Used by 26B-A4B (with spatial merge + standardization) and E2B/E4B (without).
    gemma4_siglip2,
    /// Gemma 3 SigLIP: bias, GELU FFN, 1D pos, post_ln, mm.soft_emb_norm + mm.input_projection.
    gemma3_siglip,
    /// Qwen VL: fused QKV, bias, GELU FFN, 1D pos, MLP projector mm.0 + mm.2.
    qwen_vl,
};

/// Cached pointer-keyed entry for converted norm weights.
const NormCacheEntry = struct { key: usize, data: []f32 };

/// Vision encoder — loads from mmproj GGUF and produces visual token embeddings.
/// Supports Gemma 4 SigLIP-2, Gemma 3 SigLIP, and Qwen VL architectures.
pub const VisionEncoder = struct {
    // ── Configuration ────────────────────────────────────────────
    /// When true, dump intermediate f32 buffers to disk for debugging.
    /// Enabled by setting the environment variable AGAVE_VISION_DEBUG=1.
    debug: bool = false,

    image_size: u32,
    patch_size: u32,
    n_patches: u32,
    /// Number of output patches AFTER spatial merge (= n_patches when n_merge == 0).
    n_output_patches: u32,
    /// Spatial merge kernel size (3 for Gemma 4, 0 for others = no merge).
    n_merge: u32,
    /// Whether image should use native resolution instead of fixed image_size.
    /// When true, the caller should resize to the original image dims aligned to
    /// patch_size * spatial_merge (e.g., 32 for Qwen VL) instead of image_size.
    use_native_resolution: bool = false,
    embd_dim: u32,
    ffn_dim: u32,
    n_blocks: u32,
    n_heads: u32,
    head_dim: u32,
    projection_dim: u32,
    norm_eps: f32,

    // ── Architecture variant and feature flags ───────────────────
    variant: VisionVariant,
    /// Gemma3/Qwen: true — all projections and norms carry additive bias.
    has_bias: bool,
    /// Gemma4: true — per-head RMSNorm on Q and K after projection.
    has_qk_norm: bool,
    /// Gemma4: true — RMSNorm after attention output and after FFN.
    has_post_norms: bool,
    /// Qwen: true — single fused QKV weight matrix instead of separate Q/K/V.
    fused_qkv: bool,
    /// Gemma4: true (SwiGLU: gate+up+down). Others: false (GELU: up+down, no gate).
    ffn_has_gate: bool,
    /// Gemma3/Qwen: true — LayerNorm applied after all ViT blocks, before projection.
    has_post_ln: bool,
    /// Gemma4: true — 2D position embedding [embd_dim, max_pos, 2]. Others: 1D.
    pos_embd_is_2d: bool,
    /// Gemma4 26B: true — input standardization via v.std_scale + v.std_bias.
    has_standardization: bool,
    /// Pixel normalization scale: (pixel/255) * pixel_norm_scale + pixel_norm_bias.
    /// 26B SigLIP-2 (mean=0.5, std=0.5): scale=2.0, bias=-1.0 → [-1,1].
    /// E2B/E4B SigLIP-2 (mean=0.0, std=1.0): scale=1.0, bias=0.0 → [0,1].
    pixel_norm_scale: f32,
    pixel_norm_bias: f32,

    // ── Dependencies ─────────────────────────────────────────────
    fmt: Format,
    be: Backend,
    allocator: Allocator,
    pool: ?*ThreadPool = null,

    // ── Norm weight cache ────────────────────────────────────────
    norm_cache: [max_norm_entries]NormCacheEntry = undefined,
    norm_cache_len: usize = 0,

    // ── Working buffers (allocated once, reused per image) ────────
    /// Patch-embedded input: [n_patches, embd_dim].
    patch_buf: []f32 = &.{},
    /// Residual / hidden state: [n_patches, embd_dim].
    hidden: []f32 = &.{},
    /// Temporary buffer for norm output: [n_patches, embd_dim].
    norm_buf: []f32 = &.{},
    /// Q projection buffer: [n_patches, embd_dim].
    q_buf: []f32 = &.{},
    /// K projection buffer: [n_patches, embd_dim].
    k_buf: []f32 = &.{},
    /// V projection buffer: [n_patches, embd_dim].
    v_buf: []f32 = &.{},
    /// Attention output buffer: [n_patches, embd_dim].
    attn_out: []f32 = &.{},
    /// Attention scores buffer: [n_heads, n_patches, n_patches].
    scores: []f32 = &.{},
    /// FFN gate buffer: [n_patches, ffn_dim] (only used for SwiGLU variants).
    ffn_gate: []f32 = &.{},
    /// FFN up buffer: [n_patches, ffn_dim].
    ffn_up: []f32 = &.{},
    /// FFN down / temp buffer: [n_patches, embd_dim].
    ffn_down: []f32 = &.{},
    /// Output projection buffer: [n_patches, projection_dim].
    output: []f32 = &.{},
    /// MLP projector intermediate buffer (Qwen only): [n_patches, mlp_intermediate_dim].
    mlp_buf: []f32 = &.{},
    /// MLP projector intermediate dimension (Qwen: mm.0 output rows).
    mlp_intermediate_dim: u32 = 0,

    /// Initialize the vision encoder from mmproj format metadata.
    /// Auto-detects the architecture variant from available tensors and
    /// allocates all working buffers.
    pub fn init(allocator: Allocator, fmt: Format, be: Backend, pool: ?*ThreadPool) !VisionEncoder {
        const arch = "clip.vision";

        const patch_size = fmt.getArchU32(arch, "patch_size") orelse default_patch_size;
        const embd_dim = fmt.getArchU32(arch, "embedding_length") orelse default_embd_dim;
        const ffn_dim = fmt.getArchU32(arch, "feed_forward_length") orelse default_ffn_dim;
        const n_blocks = fmt.getArchU32(arch, "block_count") orelse default_n_blocks;
        const n_heads = fmt.getArchU32(arch, "attention.head_count") orelse default_n_heads;
        const projection_dim = fmt.getArchU32(arch, "projection_dim") orelse default_projection_dim;
        const norm_eps = fmt.getArchF32(arch, "attention.layer_norm_epsilon") orelse default_norm_eps;

        // ── Auto-detect architecture variant ─────────────────────
        const variant = detectVariant(fmt);

        // ── Spatial merge and effective resolution ───────────────
        // Gemma 4 SigLIP-2 (26B): 3×3 avg pool after ViT → 9× reduction
        //   768×768 → 48×48=2304 patches → 3×3 pool → 16×16=256 output tokens
        // Gemma 4 SigLIP-2 (E2B/E4B): NO spatial merge — 224×224 → 14×14=196 tokens.
        //   Detected by absence of v.std_scale (standardization tensors).
        // Qwen VL: 4× merge in MLP projector → n_patches/4 output tokens
        //   Process at native resolution (not upscaled). metadata image_size is the MAX.
        //   For 224×224 input: 14×14=196 patches → 4× merge → 49 output tokens.
        // Gemma 3: no merge, process at metadata image_size (896)
        const meta_image_size = fmt.getArchU32(arch, "image_size") orelse default_image_size;
        const has_std_tensors = fmt.getTensor("v.std_scale") != null;
        // Only apply 3×3 merge when standardization tensors are present (26B-A4B variant).
        // E2B/E4B SigLIP-2 uses 224×224 without merge — patches_per_side=14 is not divisible by 3.
        const n_merge: u32 = if (variant == .gemma4_siglip2 and has_std_tensors) gemma4_merge_kernel else 0;
        const image_size: u32 = if (n_merge > 0) gemma4_effective_image_size else meta_image_size;

        const patches_per_side = image_size / patch_size;
        const n_patches = patches_per_side * patches_per_side;
        const qwen_merge_factor: u32 = if (variant == .qwen_vl) 4 else 1;
        const n_output_patches: u32 = if (n_merge > 0) blk: {
            const out_side = patches_per_side / n_merge;
            break :blk out_side * out_side;
        } else n_patches / qwen_merge_factor;
        const head_dim = embd_dim / n_heads;

        const has_bias = variant != .gemma4_siglip2;
        const has_qk_norm = variant == .gemma4_siglip2;
        const has_post_norms = variant == .gemma4_siglip2;
        const fused_qkv = variant == .qwen_vl;
        const ffn_has_gate = variant == .gemma4_siglip2;
        const has_post_ln = variant != .gemma4_siglip2;
        const pos_embd_is_2d = variant == .gemma4_siglip2;
        // Standardization only when v.std_scale/v.std_bias tensors exist (26B-A4B).
        // E2B/E4B SigLIP-2 omits these — image_mean=[0,0,0], image_std=[1,1,1].
        const has_standardization = has_std_tensors;

        // Pixel normalization parameters derived from image_mean/image_std metadata.
        // 26B/Gemma3 (mean=0.5, std=0.5): (pix/255 - 0.5)/0.5 = 2*pix/255 - 1 → [-1,1]
        // E2B/E4B (mean=0, std=1): (pix/255 - 0)/1 = pix/255 → [0,1]
        // Qwen VL also uses mean=0.5, std=0.5 → [-1,1]
        const pix_scale: f32 = if (variant == .gemma4_siglip2 and !has_std_tensors)
            pixel_scale // 1/255 → [0,1]
        else
            2.0 * pixel_scale; // 2/255 → scale part of [-1,1]
        const pix_bias: f32 = if (variant == .gemma4_siglip2 and !has_std_tensors)
            0.0 // no bias → [0,1]
        else
            -1.0; // bias part of [-1,1]

        // Determine MLP projector intermediate dim for Qwen
        var mlp_intermediate_dim: u32 = 0;
        if (variant == .qwen_vl) {
            if (fmt.getTensor("mm.0.weight")) |mm0| {
                mlp_intermediate_dim = @intCast(mm0.dims[0]);
            }
        }

        const np: usize = n_patches;
        const ed: usize = embd_dim;
        const fd: usize = ffn_dim;
        const nh: usize = n_heads;
        const pd: usize = projection_dim;

        // Scores buffer is sized for chunked attention: n_heads * chunk * n_patches.
        // When n_patches <= chunk_size, this equals full n_heads * np * np.
        // When n_patches > chunk_size (e.g., Gemma3 with 4096 patches), chunked
        // attention processes queries in tiles, keeping peak memory bounded.
        const chunk: usize = @min(np, attention_chunk_size);

        // Check environment variable for debug mode
        const vision_debug = blk: {
            var env_buf = [_]u8{0} ** 20;
            @memcpy(env_buf[0..18], "AGAVE_VISION_DEBUG");
            break :blk std.c.getenv(@ptrCast(env_buf[0..18 :0])) != null;
        };

        var self = VisionEncoder{
            .debug = vision_debug,
            .image_size = image_size,
            .patch_size = patch_size,
            .n_patches = n_patches,
            .n_output_patches = n_output_patches,
            .n_merge = n_merge,
            .use_native_resolution = variant == .qwen_vl,
            .embd_dim = embd_dim,
            .ffn_dim = ffn_dim,
            .n_blocks = n_blocks,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .projection_dim = projection_dim,
            .norm_eps = norm_eps,
            .variant = variant,
            .has_bias = has_bias,
            .has_qk_norm = has_qk_norm,
            .has_post_norms = has_post_norms,
            .fused_qkv = fused_qkv,
            .ffn_has_gate = ffn_has_gate,
            .has_post_ln = has_post_ln,
            .pos_embd_is_2d = pos_embd_is_2d,
            .has_standardization = has_standardization,
            .pixel_norm_scale = pix_scale,
            .pixel_norm_bias = pix_bias,
            .mlp_intermediate_dim = mlp_intermediate_dim,
            .fmt = fmt,
            .be = be,
            .allocator = allocator,
            .pool = pool,
        };

        const variant_name: []const u8 = switch (variant) {
            .gemma4_siglip2 => "Gemma4 SigLIP-2",
            .gemma3_siglip => "Gemma3 SigLIP",
            .qwen_vl => "Qwen VL",
        };
        if (n_merge > 0) {
            std.log.info("vision: detected variant={s} image={d}×{d} patch={d} patches={d} merge={d}×{d} -> {d} tokens embd={d} heads={d}", .{
                variant_name, image_size, image_size, patch_size, n_patches, n_merge, n_merge, n_output_patches, embd_dim, n_heads,
            });
        } else {
            std.log.info("vision: detected variant={s} image={d}×{d} patch={d} patches={d} embd={d} heads={d}", .{
                variant_name, image_size, image_size, patch_size, n_patches, embd_dim, n_heads,
            });
        }

        // ── Working buffer allocation ────────────────────────────
        self.patch_buf = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.patch_buf);
        self.hidden = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.hidden);
        self.norm_buf = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.norm_buf);
        self.q_buf = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.q_buf);
        self.k_buf = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.k_buf);
        self.v_buf = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.v_buf);
        self.attn_out = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.attn_out);
        // Scores buffer sized for chunked attention: n_heads * chunk * n_patches
        self.scores = try allocator.alloc(f32, nh * chunk * np);
        errdefer allocator.free(self.scores);
        if (ffn_has_gate) {
            self.ffn_gate = try allocator.alloc(f32, np * fd);
            errdefer allocator.free(self.ffn_gate);
        }
        self.ffn_up = try allocator.alloc(f32, np * fd);
        errdefer allocator.free(self.ffn_up);
        self.ffn_down = try allocator.alloc(f32, np * ed);
        errdefer allocator.free(self.ffn_down);
        const np_out: usize = n_output_patches;
        self.output = try allocator.alloc(f32, np_out * pd);
        errdefer allocator.free(self.output);

        // Qwen MLP projector intermediate buffer (sized for merged token count)
        if (mlp_intermediate_dim > 0) {
            const mid: usize = mlp_intermediate_dim;
            self.mlp_buf = try allocator.alloc(f32, np_out * mid);
            errdefer allocator.free(self.mlp_buf);
        }

        return self;
    }

    /// Detect the vision encoder variant from available tensors in the mmproj file.
    fn detectVariant(fmt: Format) VisionVariant {
        // Gemma 4 SigLIP-2: has per-head QK RMSNorm weights.
        // Check this FIRST — E2B/E4B mmproj files contain both v.blk.* (SigLIP-2 ViT)
        // and a.blk.* (audio encoder) tensors. The vision path uses the standard ViT.
        if (fmt.getTensor("v.blk.0.attn_q_norm.weight") != null) return .gemma4_siglip2;
        // Qwen VL: has fused QKV projection
        if (fmt.getTensor("v.blk.0.attn_qkv.weight") != null) return .qwen_vl;
        // Gemma 3 SigLIP: has separate Q/K/V with bias
        if (fmt.getTensor("v.blk.0.attn_q.bias") != null) return .gemma3_siglip;
        // Fallback: assume Gemma 4 SigLIP-2
        return .gemma4_siglip2;
    }

    /// Release all heap allocations owned by this encoder.
    pub fn deinit(self: *VisionEncoder) void {
        // Free cached norm weight conversions
        for (self.norm_cache[0..self.norm_cache_len]) |entry| self.allocator.free(entry.data);
        // Free working buffers
        const bufs = .{
            &self.patch_buf, &self.hidden,   &self.norm_buf,
            &self.q_buf,     &self.k_buf,    &self.v_buf,
            &self.attn_out,  &self.scores,   &self.ffn_gate,
            &self.ffn_up,    &self.ffn_down, &self.output,
            &self.mlp_buf,
        };
        inline for (bufs) |buf| {
            if (buf.len > 0) self.allocator.free(buf.*);
        }
    }

    // ── Public API ────────────────────────────────────────────────

    /// Run the full vision encoder pipeline on a raw RGB image.
    /// Input: u8 RGB pixel data [image_size * image_size * 3], row-major, channel-last (R,G,B per pixel).
    /// Output: visual token embeddings [n_patches, projection_dim] in self.output.
    /// Returns a slice of the output buffer.
    pub fn encode(self: *VisionEncoder, pixels: []const u8) ![]const f32 {
        const np: usize = self.n_patches;
        const ed: usize = self.embd_dim;
        const expected_len = @as(usize, self.image_size) * self.image_size * n_channels;
        if (pixels.len != expected_len) return error.MissingTensor;

        if (self.debug) std.log.info("vision debug: encode start, np={d} ed={d}", .{ np, ed });

        // 1. Preprocess: u8 RGB -> f32 channel-first ([-1,1] for Gemma4, [0,1] for others)
        self.preprocessImage(pixels);

        // 2. Patch embedding: conv2d-style unfold + linear projection
        try self.patchEmbed();
        if (self.debug) self.dumpBuf("agave_01_after_patch_embed.bin", self.patch_buf[0 .. np * ed]);

        // 3. Add position embedding (2D for Gemma4, 1D for others)
        // NOTE: For Gemma 4, standardization happens AFTER pooling (step 7b).
        try self.addPositionEmbedding();
        if (self.debug) self.dumpBuf("agave_02_after_pos.bin", self.patch_buf[0 .. np * ed]);

        // 3b. Qwen VL pixel shuffle: applied AFTER position embedding so both
        // data and positions are rearranged together. Groups 2×2 spatial
        // neighbors into contiguous blocks of 4 tokens for the 4× MLP merge.
        if (self.variant == .qwen_vl) {
            const pps = self.image_size / self.patch_size;
            const hx = pps / 2;
            const hy = pps / 2;
            // Use hidden as temp, write result back to patch_buf
            @memcpy(self.hidden[0 .. np * ed], self.patch_buf[0 .. np * ed]);
            for (0..hy) |by| {
                for (0..hx) |bx| {
                    for (0..2) |dy| {
                        for (0..2) |dx| {
                            const si = ((by * 2 + dy) * pps + (bx * 2 + dx)) * ed;
                            const di = ((by * hx + bx) * 4 + dy * 2 + dx) * ed;
                            @memcpy(self.patch_buf[di..][0..ed], self.hidden[si..][0..ed]);
                        }
                    }
                }
            }
        }

        // 4. Copy patch_buf to hidden (initial residual)
        @memcpy(self.hidden[0 .. np * ed], self.patch_buf[0 .. np * ed]);

        // 5. Run all ViT transformer blocks
        for (0..self.n_blocks) |bi| {
            try self.visionBlock(@intCast(bi));
            if (self.debug and (bi == 0)) self.dumpBuf("agave_03_after_block_00.bin", self.hidden[0 .. np * ed]);
            if (self.debug and (bi == self.n_blocks - 1)) self.dumpBuf("agave_04_after_block_last.bin", self.hidden[0 .. np * ed]);
        }

        // 6. Apply post-encoder LayerNorm (Gemma3, Qwen only)
        if (self.has_post_ln) {
            try self.applyPostLn();
        }

        // 7. Spatial merge: 3×3 average pooling + sqrt(embd_dim) scaling (Gemma 4).
        const np_proj: usize = self.n_output_patches;
        if (self.n_merge > 0) {
            self.avgPool2d();
        }

        // 7b. Post-pool standardization (Gemma 4 only).
        // Formula: (x - bias) * scale (NOT x * scale + bias).
        // Matches llama.cpp: ggml_sub(cur, std_bias) then ggml_mul(cur, std_scale).
        if (self.has_standardization) {
            try self.applyStandardization();
        }

        // 8. Project to LLM hidden dimension
        try self.projectToLlm();
        const pd: usize = self.projection_dim;
        if (self.debug) self.dumpBuf("agave_05_after_projection.bin", self.output[0 .. np_proj * pd]);

        return self.output[0 .. np_proj * pd];
    }

    /// Dump a float buffer to a file for debug comparison.
    /// Only called when self.debug is true.
    fn dumpBuf(self: *const VisionEncoder, filename: []const u8, data: []const f32) void {
        _ = self;
        const fd = std.posix.openat(std.posix.AT.FDCWD, filename, .{
            .ACCMODE = .RDWR,
            .CREAT = true,
        }, 0o644) catch |err| {
            std.log.warn("vision debug: failed to create {s}: {s}", .{ filename, @errorName(err) });
            return;
        };
        defer _ = std.c.close(fd);
        const bytes: []const u8 = @as([*]const u8, @ptrCast(data.ptr))[0 .. data.len * @sizeOf(f32)];
        _ = std.c.write(fd, bytes.ptr, bytes.len);
        std.log.info("vision debug: dumped {s} ({d} floats, {d} bytes)", .{ filename, data.len, bytes.len });
    }

    // ── Preprocessing ─────────────────────────────────────────────

    /// Convert u8 RGB pixels (channel-last, row-major) to f32 channel-first [C, H, W].
    /// Normalization is per-instance: pixel_norm_scale * (pixel/255) + pixel_norm_bias.
    ///   26B/Gemma3/Qwen (mean=0.5, std=0.5): 2/255 * pixel - 1.0 → [-1,1]
    ///   E2B/E4B (mean=0, std=1): 1/255 * pixel + 0.0 → [0,1]
    fn preprocessImage(self: *VisionEncoder, pixels: []const u8) void {
        const h: usize = self.image_size;
        const w: usize = self.image_size;
        const scale: f32 = self.pixel_norm_scale;
        const bias: f32 = self.pixel_norm_bias;
        // Convert from [H, W, 3] (channel-last) to [3, H, W] (channel-first)
        for (0..h) |y| {
            for (0..w) |x| {
                const src_idx = (y * w + x) * n_channels;
                for (0..n_channels) |c| {
                    const dst_idx = c * h * w + y * w + x;
                    self.patch_buf[dst_idx] = @as(f32, @floatFromInt(pixels[src_idx + c])) * scale + bias;
                }
            }
        }
    }

    /// Apply conv2d-style patch embedding.
    /// Unfolds image into non-overlapping patches and projects each through the
    /// patch embedding weight matrix. Adds bias if present (Gemma3, Qwen).
    ///
    /// Input: self.patch_buf contains channel-first f32 pixels [3, H, W].
    /// Output: self.hidden[0..n_patches*embd_dim] = patch embeddings.
    /// Then copies result back to self.patch_buf for subsequent steps.
    fn patchEmbed(self: *VisionEncoder) !void {
        const ps: usize = self.patch_size;
        const ed: usize = self.embd_dim;
        const h: usize = self.image_size;
        const w: usize = self.image_size;
        const pps = h / ps; // patches per side

        // Get patch embedding weight: [patch_size, patch_size, 3, embd_dim] in GGUF
        // This is a conv2d kernel. For each patch, we flatten the pixels and do a dot product
        // with each output channel (embd_dim rows).
        const emb_t = self.fmt.getTensor("v.patch_embd.weight") orelse return error.MissingTensor;
        const patch_elems = ps * ps * n_channels; // elements per flattened patch

        // Optional bias: Gemma3 and Qwen have v.patch_embd.bias [embd_dim]
        const emb_bias: ?[*]const f32 = if (self.has_bias)
            if (self.fmt.getTensor("v.patch_embd.bias")) |bt| tensorAsF32(bt) else null
        else
            null;

        // Flatten all patches into norm_buf first, then batch the GEMV.
        // norm_buf is [np * ed] and patch_elems <= ed (e.g. 768 vs 1152),
        // so np * patch_elems fits in norm_buf.
        const np: usize = pps * pps;
        for (0..pps) |py| {
            for (0..pps) |px| {
                const patch_idx = py * pps + px;
                const flat_base = patch_idx * patch_elems;

                // Flatten patch pixels from channel-first layout [C, H, W]
                for (0..n_channels) |c| {
                    for (0..ps) |dy| {
                        for (0..ps) |dx| {
                            const src_y = py * ps + dy;
                            const src_x = px * ps + dx;
                            const src_idx = c * h * w + src_y * w + src_x;
                            const flat_idx = c * ps * ps + dy * ps + dx;
                            self.norm_buf[flat_base + flat_idx] = self.patch_buf[src_idx];
                        }
                    }
                }
            }
        }

        // Batched projection: hidden[p, :] = emb_weight @ flat_patches[p, :]
        self.batchGemm(self.norm_buf.ptr, emb_t, self.hidden.ptr, np, ed, patch_elems);

        // Qwen VL: add second conv2d kernel output (dual patch embedding)
        if (self.fmt.getTensor("v.patch_embd.weight.1")) |emb_t2| {
            // Use attn_out as temp buffer for second conv result
            self.batchGemm(self.norm_buf.ptr, emb_t2, self.attn_out.ptr, np, ed, patch_elems);
            for (0..np * ed) |i| self.hidden[i] += self.attn_out[i];
        }

        // Add bias if present
        if (emb_bias) |bias| for (0..np) |p| {
            const out_base = p * ed;
            for (0..ed) |d| {
                self.hidden[out_base + d] += bias[d];
            }
        };

        // Copy result to patch_buf for position embedding (pixel shuffle applied later)
        @memcpy(self.patch_buf[0 .. np * ed], self.hidden[0 .. np * ed]);
    }

    /// Apply standardization: x = (x - bias) * scale.
    /// For Gemma 4, this runs AFTER avg pooling on self.hidden (n_output_patches).
    /// Matches llama.cpp: ggml_sub(cur, std_bias) then ggml_mul(cur, std_scale).
    fn applyStandardization(self: *VisionEncoder) !void {
        const ed: usize = self.embd_dim;
        const np: usize = self.n_output_patches;

        const scale_t = self.fmt.getTensor("v.std_scale") orelse return error.MissingTensor;
        const bias_t = self.fmt.getTensor("v.std_bias") orelse return error.MissingTensor;

        const scale_ptr = tensorAsF32(scale_t);
        const bias_ptr = tensorAsF32(bias_t);

        for (0..np) |p| {
            const base = p * ed;
            for (0..ed) |d| {
                self.hidden[base + d] = (self.hidden[base + d] - bias_ptr[d]) * scale_ptr[d];
            }
        }
    }

    /// Add position embedding to patch embeddings.
    /// Gemma4: 2D embedding [embd_dim, max_positions, 2] — row + col channels.
    /// Gemma3/Qwen: 1D embedding [embd_dim, n_patches] — one vector per patch index.
    fn addPositionEmbedding(self: *VisionEncoder) !void {
        const ed: usize = self.embd_dim;
        const ps: usize = self.patch_size;
        const h: usize = self.image_size;
        const pps = h / ps; // patches per side

        const pos_t = self.fmt.getTensor("v.position_embd.weight") orelse return error.MissingTensor;
        const pos_data: [*]const f32 = @ptrCast(@alignCast(pos_t.data_ptr));

        // Debug: check position embedding values
        if (self.debug) {
            std.log.info("vision debug: pos_embd dims=[{d},{d},{d}] first 5 values: {d:.6},{d:.6},{d:.6},{d:.6},{d:.6}", .{
                pos_t.dims[0], pos_t.dims[1], pos_t.dims[2],
                pos_data[0],   pos_data[1],   pos_data[2],
                pos_data[3],   pos_data[4],
            });
        }

        if (self.pos_embd_is_2d) {
            // 2D position embedding: GGUF dims [embd_dim, max_positions, 2]
            // GGUF memory layout is column-major (dim0=fastest varying):
            //   data[rc * max_pos * ed + pos * ed + d]
            // where rc=0 for row, rc=1 for col.
            // For each patch at grid (row, col):
            //   hidden[d] += data[0 * max_pos * ed + row * ed + d]   (row component)
            //              + data[1 * max_pos * ed + col * ed + d]   (col component)
            const max_pos: usize = @intCast(pos_t.dims[1]);
            const stride_rc = max_pos * ed;

            for (0..pps) |py| {
                for (0..pps) |px| {
                    const patch_idx = py * pps + px;
                    const base = patch_idx * ed;
                    for (0..ed) |d| {
                        const row_val = pos_data[0 * stride_rc + py * ed + d];
                        const col_val = pos_data[1 * stride_rc + px * ed + d];
                        self.patch_buf[base + d] += row_val + col_val;
                    }
                }
            }
        } else {
            // 1D position embedding: [embd_dim, n_patches] (stored row-major by embd_dim)
            // For GGUF layout: pos_data[d * n_pos + patch_idx] gives dim d for patch patch_idx.
            const n_pos: usize = @intCast(pos_t.dims[1]);
            const np: usize = self.n_patches;

            for (0..np) |p| {
                if (p >= n_pos) break; // safety: don't read beyond embedding table
                const base = p * ed;
                for (0..ed) |d| {
                    self.patch_buf[base + d] += pos_data[d * n_pos + p];
                }
            }
        }
    }

    // ── Transformer block ─────────────────────────────────────────

    /// Run one ViT transformer block. The exact operations depend on the variant:
    ///   All:    LayerNorm → Attention → residual, LayerNorm → FFN → residual
    ///   Gemma4: RMSNorm (no bias), QK norms, SwiGLU FFN, post-attn/FFN RMSNorm
    ///   Gemma3: LayerNorm (with bias), no QK norms, GELU FFN (no gate), no post-norms
    ///   Qwen:   LayerNorm (with bias), fused QKV, no QK norms, GELU FFN (no gate), no post-norms
    fn visionBlock(self: *VisionEncoder, bi: u32) !void {
        const np: usize = self.n_patches;
        const ed: usize = self.embd_dim;
        const nh: usize = self.n_heads;
        const hd: usize = self.head_dim;
        const fd: usize = self.ffn_dim;

        // ── 1. Pre-attention LayerNorm ───────────────────────────
        const ln1_w = self.blockTensor(bi, "ln1.weight") orelse return error.MissingTensor;
        const ln1_ptr = self.normAsF32(ln1_w, ed);
        const ln1_bias: ?[*]const f32 = if (self.has_bias)
            if (self.blockTensor(bi, "ln1.bias")) |bt| tensorAsF32(bt) else null
        else
            null;

        for (0..np) |p| {
            const base = p * ed;
            if (ln1_bias) |bias| {
                layerNormCpu(
                    self.hidden[base..][0..ed],
                    self.norm_buf[base..][0..ed],
                    ln1_ptr,
                    bias,
                    ed,
                    self.norm_eps,
                );
            } else {
                rmsNormCpu(
                    self.hidden[base..][0..ed],
                    self.norm_buf[base..][0..ed],
                    ln1_ptr,
                    ed,
                    self.norm_eps,
                );
            }
        }

        // ── 2. Q/K/V projections ────────────────────────────────
        if (self.fused_qkv) {
            // Qwen: fused QKV — single batched GEMV producing [np, 3*embd_dim],
            // then scatter into separate Q/K/V buffers.
            // ffn_up is [np * ffn_dim] which is >= [np * 3*ed] (ffn_dim > 3*embd_dim).
            const qkv_w = self.blockTensor(bi, "attn_qkv.weight") orelse return error.MissingTensor;
            const qkv_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "attn_qkv.bias")) |bt| tensorAsF32(bt) else null
            else
                null;

            const qkv_dim = 3 * ed;
            self.batchGemm(self.norm_buf.ptr, qkv_w, self.ffn_up.ptr, np, qkv_dim, ed);

            // Add bias and scatter into Q, K, V buffers
            for (0..np) |p| {
                const qkv_base = p * qkv_dim;
                const base = p * ed;
                if (qkv_bias) |bias| {
                    for (0..qkv_dim) |d| {
                        self.ffn_up[qkv_base + d] += bias[d];
                    }
                }
                @memcpy(self.q_buf[base..][0..ed], self.ffn_up[qkv_base..][0..ed]);
                @memcpy(self.k_buf[base..][0..ed], self.ffn_up[qkv_base + ed ..][0..ed]);
                @memcpy(self.v_buf[base..][0..ed], self.ffn_up[qkv_base + 2 * ed ..][0..ed]);
            }
        } else {
            // Separate Q/K/V projections — batched across all patches.
            // Each weight matrix is loaded once (row-by-row), multiplied against
            // all patch inputs, giving np× cache reuse vs per-patch calls.
            const qw = self.blockTensor(bi, "attn_q.weight") orelse return error.MissingTensor;
            const kw = self.blockTensor(bi, "attn_k.weight") orelse return error.MissingTensor;
            const vw = self.blockTensor(bi, "attn_v.weight") orelse return error.MissingTensor;

            const q_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "attn_q.bias")) |bt| tensorAsF32(bt) else null
            else
                null;
            const k_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "attn_k.bias")) |bt| tensorAsF32(bt) else null
            else
                null;
            const v_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "attn_v.bias")) |bt| tensorAsF32(bt) else null
            else
                null;

            self.batchGemm(self.norm_buf.ptr, qw, self.q_buf.ptr, np, ed, ed);
            self.batchGemm(self.norm_buf.ptr, kw, self.k_buf.ptr, np, ed, ed);
            self.batchGemm(self.norm_buf.ptr, vw, self.v_buf.ptr, np, ed, ed);

            if (q_bias) |bias| for (0..np) |p| {
                const base = p * ed;
                for (0..ed) |d| self.q_buf[base + d] += bias[d];
            };
            if (k_bias) |bias| for (0..np) |p| {
                const base = p * ed;
                for (0..ed) |d| self.k_buf[base + d] += bias[d];
            };
            if (v_bias) |bias| for (0..np) |p| {
                const base = p * ed;
                for (0..ed) |d| self.v_buf[base + d] += bias[d];
            };
        }

        // ── 3. Per-head QK RMSNorm (Gemma4 only) ────────────────
        if (self.has_qk_norm) {
            const qn_t = self.blockTensor(bi, "attn_q_norm.weight");
            const kn_t = self.blockTensor(bi, "attn_k_norm.weight");

            if (qn_t) |qn| {
                const qn_ptr = self.normAsF32(qn, hd);
                for (0..np) |p| {
                    for (0..nh) |h| {
                        const off = p * ed + h * hd;
                        rmsNormInPlace(self.q_buf[off..][0..hd], qn_ptr, hd, self.norm_eps);
                    }
                }
            }
            if (kn_t) |kn| {
                const kn_ptr = self.normAsF32(kn, hd);
                for (0..np) |p| {
                    for (0..nh) |h| {
                        const off = p * ed + h * hd;
                        rmsNormInPlace(self.k_buf[off..][0..hd], kn_ptr, hd, self.norm_eps);
                    }
                }
            }
        }

        // ── 3b. Rotary position encoding in attention ──────────────
        {
            const pps = self.image_size / self.patch_size;
            if (self.variant == .gemma4_siglip2) {
                // Gemma 4: 2D RoPE — first half x-pos, second half y-pos (theta=100)
                applyRope2d(self.q_buf, np, nh, hd, pps, vit_rope_theta);
                applyRope2d(self.k_buf, np, nh, hd, pps, vit_rope_theta);
            } else if (self.variant == .qwen_vl) {
                // Qwen VL: M-RoPE — 4 sections [temporal, height, height, width] (theta=10000)
                const qwen_rope_theta: f32 = 10000.0;
                applyMRope(self.q_buf, np, nh, hd, pps, qwen_rope_theta);
                applyMRope(self.k_buf, np, nh, hd, pps, qwen_rope_theta);
            }
        }

        // ── 4. Full bidirectional attention ──────────────────────
        self.fullAttention();

        // ── 5. Output projection (batched across all patches) ───
        const ow = self.blockTensor(bi, "attn_out.weight") orelse return error.MissingTensor;
        const out_bias: ?[*]const f32 = if (self.has_bias)
            if (self.blockTensor(bi, "attn_out.bias")) |bt| tensorAsF32(bt) else null
        else
            null;

        self.batchGemm(self.attn_out.ptr, ow, self.ffn_down.ptr, np, ed, ed);
        if (out_bias) |bias| for (0..np) |p| {
            const base = p * ed;
            for (0..ed) |d| self.ffn_down[base + d] += bias[d];
        };

        // ── 6. Post-attention RMSNorm + residual ────────────────
        if (self.has_post_norms) {
            const post_attn_t = self.blockTensor(bi, "attn_post_norm.weight");
            if (post_attn_t) |pan| {
                const pan_ptr = self.normAsF32(pan, ed);
                for (0..np) |p| {
                    const base = p * ed;
                    rmsNormInPlace(self.ffn_down[base..][0..ed], pan_ptr, ed, self.norm_eps);
                }
            }
        }
        for (0..np * ed) |i| {
            self.hidden[i] += self.ffn_down[i];
        }

        // ── 7. Pre-FFN LayerNorm ────────────────────────────────
        const ln2_w = self.blockTensor(bi, "ln2.weight") orelse return error.MissingTensor;
        const ln2_ptr = self.normAsF32(ln2_w, ed);
        const ln2_bias: ?[*]const f32 = if (self.has_bias)
            if (self.blockTensor(bi, "ln2.bias")) |bt| tensorAsF32(bt) else null
        else
            null;

        for (0..np) |p| {
            const base = p * ed;
            if (ln2_bias) |bias| {
                layerNormCpu(
                    self.hidden[base..][0..ed],
                    self.norm_buf[base..][0..ed],
                    ln2_ptr,
                    bias,
                    ed,
                    self.norm_eps,
                );
            } else {
                rmsNormCpu(
                    self.hidden[base..][0..ed],
                    self.norm_buf[base..][0..ed],
                    ln2_ptr,
                    ed,
                    self.norm_eps,
                );
            }
        }

        // ── 8. FFN ──────────────────────────────────────────────
        if (self.ffn_has_gate) {
            // SwiGLU FFN (Gemma4): gate + up → SiLU(gate) * up → down
            // All three projections batched across patches.
            const gw = self.blockTensor(bi, "ffn_gate.weight") orelse return error.MissingTensor;
            const uw = self.blockTensor(bi, "ffn_up.weight") orelse return error.MissingTensor;
            const dw = self.blockTensor(bi, "ffn_down.weight") orelse return error.MissingTensor;

            self.batchGemm(self.norm_buf.ptr, gw, self.ffn_gate.ptr, np, fd, ed);
            self.batchGemm(self.norm_buf.ptr, uw, self.ffn_up.ptr, np, fd, ed);

            // SiLU(gate) * up — fused across all patches
            for (0..np * fd) |i| {
                self.ffn_gate[i] = silu(self.ffn_gate[i]) * self.ffn_up[i];
            }

            // Down projection (batched)
            self.batchGemm(self.ffn_gate.ptr, dw, self.ffn_down.ptr, np, ed, fd);
        } else {
            // GELU FFN (Gemma3, Qwen): up → GELU(up) → down (no gate)
            // Up and down projections batched across patches.
            const uw = self.blockTensor(bi, "ffn_up.weight") orelse return error.MissingTensor;
            const dw = self.blockTensor(bi, "ffn_down.weight") orelse return error.MissingTensor;

            const up_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "ffn_up.bias")) |bt| tensorAsF32(bt) else null
            else
                null;
            const down_bias: ?[*]const f32 = if (self.has_bias)
                if (self.blockTensor(bi, "ffn_down.bias")) |bt| tensorAsF32(bt) else null
            else
                null;

            self.batchGemm(self.norm_buf.ptr, uw, self.ffn_up.ptr, np, fd, ed);
            if (up_bias) |bias| for (0..np) |p| {
                const ff_base = p * fd;
                for (0..fd) |d| self.ffn_up[ff_base + d] += bias[d];
            };

            // GELU activation in-place (all patches contiguous)
            geluInPlace(self.ffn_up[0 .. np * fd]);

            // Down projection (batched)
            self.batchGemm(self.ffn_up.ptr, dw, self.ffn_down.ptr, np, ed, fd);
            if (down_bias) |bias| for (0..np) |p| {
                const in_base = p * ed;
                for (0..ed) |d| self.ffn_down[in_base + d] += bias[d];
            };
        }

        // ── 9. Post-FFN RMSNorm + residual ──────────────────────
        if (self.has_post_norms) {
            const post_ffn_t = self.blockTensor(bi, "ffn_post_norm.weight");
            if (post_ffn_t) |pfn| {
                const pfn_ptr = self.normAsF32(pfn, ed);
                for (0..np) |p| {
                    const base = p * ed;
                    rmsNormInPlace(self.ffn_down[base..][0..ed], pfn_ptr, ed, self.norm_eps);
                }
            }
        }
        for (0..np * ed) |i| {
            self.hidden[i] += self.ffn_down[i];
        }
    }

    /// Non-causal multi-head self-attention over all patches.
    /// Every patch attends to every other patch (bidirectional).
    /// Parallelized across heads using the thread pool (if available).
    /// SIMD-optimized: vectorized QK dot products, softmax, and V accumulation.
    ///
    /// Uses chunked query processing: queries are tiled in groups of
    /// attention_chunk_size so the scores buffer stays bounded at
    /// n_heads * chunk * n_patches (instead of n_heads * n_patches^2).
    ///
    /// Input: self.q_buf, self.k_buf, self.v_buf [n_patches, embd_dim]
    /// Output: self.attn_out [n_patches, embd_dim]
    fn fullAttention(self: *VisionEncoder) void {
        const np: usize = self.n_patches;
        const nh: usize = self.n_heads;
        const chunk: usize = @min(np, attention_chunk_size);

        // Process queries in chunks to keep scores buffer bounded.
        var q_start: usize = 0;
        while (q_start < np) {
            const q_end = @min(q_start + chunk, np);

            // Parallelize across heads within each chunk.
            // Each head reads from shared q/k/v buffers and writes to non-overlapping
            // slices of scores[h*chunk*np..] and attn_out[..+h*hd..].
            const ctx = AttnChunkCtx{
                .self = self,
                .q_start = q_start,
                .q_count = q_end - q_start,
            };
            if (self.pool) |p| {
                p.parallelFor(nh, 1, @ptrCast(@constCast(&ctx)), &attnChunkWorker);
            } else {
                attnChunkWorker(@ptrCast(@constCast(&ctx)), 0, nh);
            }

            q_start = q_end;
        }
    }

    const AttnChunkCtx = struct {
        self: *VisionEncoder,
        q_start: usize,
        q_count: usize,
    };

    /// Worker function for parallelFor: processes heads [h_start..h_end) for one chunk.
    fn attnChunkWorker(raw_ctx: *anyopaque, h_start: usize, h_end: usize) void {
        const ctx: *const AttnChunkCtx = @ptrCast(@alignCast(raw_ctx));
        const s = ctx.self;
        const np: usize = s.n_patches;
        const hd: usize = s.head_dim;
        const ed: usize = s.embd_dim;
        const scale: f32 = if (s.variant == .gemma4_siglip2) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const chunk: usize = @min(np, attention_chunk_size);
        const q_start = ctx.q_start;
        const q_count = ctx.q_count;

        const Vec = @Vector(vec_len, f32);
        const vec_zero: Vec = @splat(0.0);

        for (h_start..h_end) |h| {
            // QK dot products: scores[h, qi, j] = Q[q_start+qi, h] . K[j, h] * scale
            for (0..q_count) |qi| {
                const i = q_start + qi;
                const q_off = i * ed + h * hd;
                const q_ptr = s.q_buf[q_off..];

                for (0..np) |j| {
                    const k_off = j * ed + h * hd;
                    const k_ptr = s.k_buf[k_off..];

                    var acc: Vec = vec_zero;
                    var d: usize = 0;
                    while (d + vec_len <= hd) : (d += vec_len) {
                        const qv: Vec = q_ptr[d..][0..vec_len].*;
                        const kv: Vec = k_ptr[d..][0..vec_len].*;
                        acc = @mulAdd(Vec, qv, kv, acc);
                    }
                    var dot: f32 = @reduce(.Add, acc);
                    while (d < hd) : (d += 1) {
                        dot = @mulAdd(f32, q_ptr[d], k_ptr[d], dot);
                    }
                    s.scores[h * chunk * np + qi * np + j] = dot * scale;
                }
            }

            // Softmax over each row
            for (0..q_count) |qi| {
                const row_start = h * chunk * np + qi * np;
                softmaxSimd(s.scores[row_start..][0..np]);
            }

            // V accumulation: attn_out[i, h] = sum_j scores[h, qi, j] * V[j, h]
            for (0..q_count) |qi| {
                const i = q_start + qi;
                const out_off = i * ed + h * hd;
                {
                    var d: usize = 0;
                    while (d + vec_len <= hd) : (d += vec_len) {
                        s.attn_out[out_off + d ..][0..vec_len].* = vec_zero;
                    }
                    while (d < hd) : (d += 1) s.attn_out[out_off + d] = 0;
                }

                for (0..np) |j| {
                    const score = s.scores[h * chunk * np + qi * np + j];
                    const sv: Vec = @splat(score);
                    const v_off = j * ed + h * hd;
                    var d: usize = 0;
                    while (d + vec_len <= hd) : (d += vec_len) {
                        const vv: Vec = s.v_buf[v_off + d ..][0..vec_len].*;
                        const cur: Vec = s.attn_out[out_off + d ..][0..vec_len].*;
                        s.attn_out[out_off + d ..][0..vec_len].* = @mulAdd(Vec, sv, vv, cur);
                    }
                    while (d < hd) : (d += 1) {
                        s.attn_out[out_off + d] = @mulAdd(f32, score, s.v_buf[v_off + d], s.attn_out[out_off + d]);
                    }
                }
            }
        }
    }

    /// Project vision embeddings to LLM hidden dimension.
    ///   Gemma4: RMSNorm (no weights) → mm.input_projection.weight
    ///           HuggingFace pipeline: VisionPooler(sqrt(hidden_size)) → RMSNorm(no_weight) → linear.
    ///           The VisionPooler scaling is absorbed by the RMSNorm (normalizes to unit RMS),
    ///           so we only need the unweighted RMSNorm followed by the linear projection.
    ///   Gemma3: mm.soft_emb_norm → mm.input_projection.weight
    ///   Qwen:   MLP projector: GELU(x @ mm.0.weight + mm.0.bias) @ mm.2.weight + mm.2.bias
    fn projectToLlm(self: *VisionEncoder) !void {
        // After spatial merge, we project n_output_patches (not n_patches).
        const np: usize = self.n_output_patches;
        const ed: usize = self.embd_dim;
        const pd: usize = self.projection_dim;

        switch (self.variant) {
            .gemma4_siglip2 => {
                // Gemma 4: linear projection THEN RMSNorm (post-projection).
                const proj_t = self.fmt.getTensor("mm.input_projection.weight") orelse return error.MissingTensor;
                self.batchGemm(self.hidden.ptr, proj_t, self.output.ptr, np, pd, ed);

                // Post-projection RMSNorm without learnable weights
                for (0..np) |p| {
                    const base = p * pd;
                    rmsNormInPlaceNoWeight(self.output[base..][0..pd], pd, self.norm_eps);
                }
            },
            .gemma3_siglip => {
                // Apply soft_emb_norm (RMSNorm) before projection
                const norm_t = self.fmt.getTensor("mm.soft_emb_norm.weight") orelse return error.MissingTensor;
                const norm_ptr = self.normAsF32(norm_t, ed);
                for (0..np) |p| {
                    const base = p * ed;
                    rmsNormInPlace(self.hidden[base..][0..ed], norm_ptr, ed, self.norm_eps);
                }

                // Linear projection (batched across all patches)
                const proj_t = self.fmt.getTensor("mm.input_projection.weight") orelse return error.MissingTensor;
                self.batchGemm(self.hidden.ptr, proj_t, self.output.ptr, np, pd, ed);
            },
            .qwen_vl => {
                // Qwen VL merger: reshape to merge 4 adjacent tokens, then MLP.
                // hidden [n_patches, embd] → reshape [n_patches/4, 4*embd] → MLP → output
                // Matches llama.cpp: ggml_reshape_3d(cur, n_embd*4, n_pos/4, batch)
                const merge_factor: usize = 4;
                const np_merged = np / merge_factor;
                const merged_dim = ed * merge_factor;

                const mm0_w = self.fmt.getTensor("mm.0.weight") orelse return error.MissingTensor;
                const mm0_bias: ?[*]const f32 = if (self.fmt.getTensor("mm.0.bias")) |bt| tensorAsF32(bt) else null;
                const mm2_w = self.fmt.getTensor("mm.2.weight") orelse return error.MissingTensor;
                const mm2_bias: ?[*]const f32 = if (self.fmt.getTensor("mm.2.bias")) |bt| tensorAsF32(bt) else null;

                const mid: usize = self.mlp_intermediate_dim;

                // The hidden buffer is already contiguous [np, ed].
                // Treating it as [np/4, 4*ed] is a zero-cost reshape.
                // Layer 0: intermediate = GELU(merged @ mm.0.weight + mm.0.bias)
                self.batchGemm(self.hidden.ptr, mm0_w, self.mlp_buf.ptr, np_merged, mid, merged_dim);
                if (mm0_bias) |bias| for (0..np_merged) |p| {
                    const mlp_base = p * mid;
                    for (0..mid) |d| self.mlp_buf[mlp_base + d] += bias[d];
                };
                geluInPlace(self.mlp_buf[0 .. np_merged * mid]);

                // Layer 2: output = intermediate @ mm.2.weight + mm.2.bias
                self.batchGemm(self.mlp_buf.ptr, mm2_w, self.output.ptr, np_merged, pd, mid);
                if (mm2_bias) |bias| for (0..np_merged) |p| {
                    const out_base = p * pd;
                    for (0..pd) |d| self.output[out_base + d] += bias[d];
                };
            },
        }
    }

    /// Apply post-encoder LayerNorm (v.post_ln) for Gemma3 and Qwen variants.
    /// Called after all ViT blocks, before projection to LLM dimension.
    fn applyPostLn(self: *VisionEncoder) !void {
        const np: usize = self.n_patches;
        const ed: usize = self.embd_dim;

        const ln_w = self.fmt.getTensor("v.post_ln.weight") orelse return error.MissingTensor;
        const ln_ptr = self.normAsF32(ln_w, ed);
        const ln_bias: ?[*]const f32 = if (self.fmt.getTensor("v.post_ln.bias")) |bt| tensorAsF32(bt) else null;

        for (0..np) |p| {
            const base = p * ed;
            if (ln_bias) |bias| {
                // Full LayerNorm (mean + variance) with bias
                layerNormInPlace(self.hidden[base..][0..ed], ln_ptr, bias, ed, self.norm_eps);
            } else {
                // RMSNorm (no mean subtraction, no bias)
                rmsNormInPlace(self.hidden[base..][0..ed], ln_ptr, ed, self.norm_eps);
            }
        }
    }

    /// 2D average pooling with kernel=n_merge, stride=n_merge (no padding).
    /// Reduces the hidden state grid from [pps_in, pps_in, embd_dim] to
    /// [pps_out, pps_out, embd_dim] by averaging non-overlapping blocks.
    /// Result is written to the BEGINNING of self.hidden, followed by
    /// sqrt(embd_dim) scaling to match llama.cpp's Gemma4V projector.
    fn avgPool2d(self: *VisionEncoder) void {
        const ed: usize = self.embd_dim;
        const k: usize = self.n_merge;
        const pps_in: usize = self.image_size / self.patch_size;
        const pps_out = pps_in / k;
        const inv_k2: f32 = 1.0 / @as(f32, @floatFromInt(k * k));
        const embd_scale: f32 = @sqrt(@as(f32, @floatFromInt(ed)));

        // Use norm_buf as temp output to avoid aliasing with self.hidden
        const out = self.norm_buf;

        for (0..pps_out) |oy| {
            for (0..pps_out) |ox| {
                const out_idx = oy * pps_out + ox;
                const out_base = out_idx * ed;
                // Zero the output slot
                @memset(out[out_base..][0..ed], 0);
                // Accumulate k×k input patches
                for (0..k) |dy| {
                    for (0..k) |dx| {
                        const iy = oy * k + dy;
                        const ix = ox * k + dx;
                        const in_idx = iy * pps_in + ix;
                        const in_base = in_idx * ed;
                        for (0..ed) |d| {
                            out[out_base + d] += self.hidden[in_base + d];
                        }
                    }
                }
                // Average and scale by sqrt(embd_dim)
                for (0..ed) |d| {
                    out[out_base + d] *= inv_k2 * embd_scale;
                }
            }
        }

        // Copy pooled result back to self.hidden (compact, n_output_patches × ed)
        const n_out = pps_out * pps_out;
        @memcpy(self.hidden[0 .. n_out * ed], out[0 .. n_out * ed]);
    }

    // ── Tensor lookup helpers ─────────────────────────────────────

    /// Look up a block-scoped tensor by block index and suffix.
    fn blockTensor(self: *VisionEncoder, bi: u32, suffix: []const u8) ?TensorInfo {
        var buf: [name_buf_size]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "v.blk.{d}.{s}", .{ bi, suffix }) catch return null;
        return self.fmt.getTensor(name);
    }

    /// Batched matrix multiply: Y[np, n] = X[np, k] @ W[n, k]^T.
    /// Uses backend GEMM for GPU acceleration when the dtype is supported.
    inline fn batchGemm(self: *VisionEncoder, x: [*]const f32, t: TensorInfo, y: [*]f32, np: usize, n: usize, k: usize) void {
        switch (t.dtype) {
            .f32, .bf16, .f16, .q8_0, .q4_0 => {
                self.be.gemm(x, .{ .data = t.data_ptr, .dtype = t.dtype }, y, np, n, k);
                self.be.sync();
            },
            else => batchedGemvCpu(x, t, y, np, n, k),
        }
    }

    /// Get norm weights as f32 pointer. Caches converted weights on first access
    /// so subsequent images return a stable pointer with zero work.
    fn normAsF32(self: *VisionEncoder, t: TensorInfo, n: usize) [*]const f32 {
        if (t.dtype != .bf16) return @ptrCast(@alignCast(t.data_ptr));

        const key = @intFromPtr(t.data_ptr);
        for (self.norm_cache[0..self.norm_cache_len]) |entry| {
            if (entry.key == key) return entry.data.ptr;
        }

        // Cache miss: allocate, convert, and store permanently.
        if (self.norm_cache_len >= max_norm_entries)
            @panic("normAsF32: norm cache overflow — increase max_norm_entries");
        const buf = self.allocator.alloc(f32, n) catch |err| {
            std.log.warn("normAsF32: alloc failed ({s}), using unconverted weights", .{@errorName(err)});
            return @ptrCast(@alignCast(t.data_ptr));
        };
        const src: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
        for (0..n) |i| buf[i] = quant.bf16ToF32(src[i]);
        self.norm_cache[self.norm_cache_len] = .{ .key = key, .data = buf };
        self.norm_cache_len += 1;
        return buf.ptr;
    }
};

// ── Module-level helper functions ─────────────────────────────────

/// CPU-side GEMV for vision encoder: y[n] = W[n,k] @ x[k].
/// Handles F32 and BF16 weight dtypes with SIMD vectorization.
/// Vision weights are small enough that CPU execution is acceptable
/// (no GPU dispatch needed).
fn gemvCpu(x: [*]const f32, t: TensorInfo, y: [*]f32, n: usize, k: usize) void {
    const V8F = @Vector(8, f32);
    switch (t.dtype) {
        .f32 => {
            const w: [*]const f32 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |row| {
                const row_off = row * k;
                var j: usize = 0;
                var acc: V8F = @splat(0.0);
                while (j + 8 <= k) : (j += 8) {
                    const xv: V8F = x[j..][0..8].*;
                    const wv: V8F = w[row_off + j ..][0..8].*;
                    acc = @mulAdd(V8F, xv, wv, acc);
                }
                var sum: f32 = @reduce(.Add, acc);
                while (j < k) : (j += 1) {
                    sum = @mulAdd(f32, x[j], w[row_off + j], sum);
                }
                y[row] = sum;
            }
        },
        .bf16 => {
            const w: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
            const V8U16 = @Vector(8, u16);
            const V8U32 = @Vector(8, u32);
            for (0..n) |row| {
                const row_off = row * k;
                var j: usize = 0;
                var acc: V8F = @splat(0.0);
                while (j + 8 <= k) : (j += 8) {
                    const xv: V8F = x[j..][0..8].*;
                    // Vectorized BF16 -> F32: zero-extend u16 to u32, shift left 16, bitcast to f32
                    const raw: V8U16 = w[row_off + j ..][0..8].*;
                    const wide: V8U32 = @as(V8U32, raw) << @splat(16);
                    const wv: V8F = @bitCast(wide);
                    acc = @mulAdd(V8F, xv, wv, acc);
                }
                var sum: f32 = @reduce(.Add, acc);
                while (j < k) : (j += 1) {
                    sum = @mulAdd(f32, x[j], quant.bf16ToF32(w[row_off + j]), sum);
                }
                y[row] = sum;
            }
        },
        else => {
            // Unsupported dtype for vision encoder — zero output
            @memset(y[0..n], 0);
        },
    }
}

/// Batched CPU-side GEMV: compute y[p*n + row] = W[row,k] @ x[p*k + ..] for all
/// patches p in [0, np). Loads each weight row once and multiplies against all
/// input patches, giving np× better cache reuse vs per-patch gemvCpu calls.
///
/// Parameters:
///   - inputs: flattened input buffer [np * k], patch-major
///   - t: weight tensor info [n, k]
///   - outputs: flattened output buffer [np * n], patch-major
///   - np: number of patches (batch size)
///   - n: output dimension (rows in weight matrix)
///   - k: input dimension (columns in weight matrix)
fn batchedGemvCpu(inputs: [*]const f32, t: TensorInfo, outputs: [*]f32, np: usize, n: usize, k: usize) void {
    const V8F = @Vector(8, f32);
    switch (t.dtype) {
        .f32 => {
            const w: [*]const f32 = @ptrCast(@alignCast(t.data_ptr));
            for (0..n) |row| {
                const w_row = w[row * k ..];
                for (0..np) |p| {
                    const x = inputs[p * k ..];
                    var j: usize = 0;
                    var acc: V8F = @splat(0.0);
                    while (j + 8 <= k) : (j += 8) {
                        const xv: V8F = x[j..][0..8].*;
                        const wv: V8F = w_row[j..][0..8].*;
                        acc = @mulAdd(V8F, xv, wv, acc);
                    }
                    var sum: f32 = @reduce(.Add, acc);
                    while (j < k) : (j += 1) {
                        sum = @mulAdd(f32, x[j], w_row[j], sum);
                    }
                    outputs[p * n + row] = sum;
                }
            }
        },
        .bf16 => {
            const w: [*]const u16 = @ptrCast(@alignCast(t.data_ptr));
            const V8U16 = @Vector(8, u16);
            const V8U32 = @Vector(8, u32);
            for (0..n) |row| {
                const w_row = w[row * k ..];
                for (0..np) |p| {
                    const x = inputs[p * k ..];
                    var j: usize = 0;
                    var acc: V8F = @splat(0.0);
                    while (j + 8 <= k) : (j += 8) {
                        const xv: V8F = x[j..][0..8].*;
                        const raw: V8U16 = w_row[j..][0..8].*;
                        const wide: V8U32 = @as(V8U32, raw) << @splat(16);
                        const wv: V8F = @bitCast(wide);
                        acc = @mulAdd(V8F, xv, wv, acc);
                    }
                    var sum: f32 = @reduce(.Add, acc);
                    while (j < k) : (j += 1) {
                        sum = @mulAdd(f32, x[j], quant.bf16ToF32(w_row[j]), sum);
                    }
                    outputs[p * n + row] = sum;
                }
            }
        },
        else => {
            @memset(outputs[0 .. np * n], 0);
        },
    }
}

/// CPU-side RMS normalization with learned weights (no bias).
/// output[i] = input[i] / rms(input) * weight[i]
/// where rms = sqrt(mean(x^2) + eps).
fn rmsNormCpu(input: []const f32, output: []f32, weight: [*]const f32, n: usize, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (0..n) |i| sum_sq += input[i] * input[i];
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    for (0..n) |i| {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// CPU-side in-place RMS normalization with learned weights (no bias).
/// x[i] = x[i] / rms(x) * weight[i]
fn rmsNormInPlace(x: []f32, weight: [*]const f32, n: usize, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (0..n) |i| sum_sq += x[i] * x[i];
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    for (0..n) |i| {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// CPU-side in-place RMS normalization without learned weights.
/// x[i] = x[i] / rms(x)
/// Used for the Gemma4 embedding_pre_projection_norm (with_scale=False).
fn rmsNormInPlaceNoWeight(x: []f32, n: usize, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (0..n) |i| sum_sq += x[i] * x[i];
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    for (0..n) |i| {
        x[i] = x[i] * inv_rms;
    }
}

/// CPU-side full LayerNorm (mean subtraction + variance normalization) with weight and bias.
/// output[i] = (input[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
fn layerNormCpu(input: []const f32, output: []f32, weight: [*]const f32, bias: [*]const f32, n: usize, eps: f32) void {
    // Compute mean
    var sum: f32 = 0.0;
    for (0..n) |i| sum += input[i];
    const mean = sum / @as(f32, @floatFromInt(n));

    // Compute variance
    var sum_sq: f32 = 0.0;
    for (0..n) |i| {
        const diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    const inv_std = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);

    // Normalize, scale, and shift
    for (0..n) |i| {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// CPU-side in-place full LayerNorm with weight and bias.
/// x[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
fn layerNormInPlace(x: []f32, weight: [*]const f32, bias: [*]const f32, n: usize, eps: f32) void {
    var sum: f32 = 0.0;
    for (0..n) |i| sum += x[i];
    const mean = sum / @as(f32, @floatFromInt(n));

    var sum_sq: f32 = 0.0;
    for (0..n) |i| {
        const diff = x[i] - mean;
        sum_sq += diff * diff;
    }
    const inv_std = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);

    for (0..n) |i| {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// SiLU activation: x * sigmoid(x).
inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Scalar GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
inline fn geluScalar(x: f32) f32 {
    const inner = @mulAdd(f32, gelu_cubic_coeff, x * x * x, x);
    const t = std.math.clamp(gelu_sqrt_2_over_pi * inner, -10.0, 10.0);
    const e2t = @exp(2.0 * t);
    return 0.5 * x * (1.0 + (e2t - 1.0) / (e2t + 1.0));
}

/// In-place GELU activation over a float slice.
/// Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
fn geluInPlace(x: []f32) void {
    for (x) |*v| {
        v.* = geluScalar(v.*);
    }
}

/// In-place SIMD-accelerated softmax over a float slice.
/// Numerically stable: subtracts max before exp to prevent overflow.
/// Follows the same vectorized pattern as sdpa.zig softmax.
fn softmaxSimd(x: []f32) void {
    const n = x.len;
    if (n == 0) return;
    const Vec = @Vector(vec_len, f32);

    // Pass 1: find max (SIMD)
    var max_acc: Vec = @splat(x[0]);
    var i: usize = 0;
    while (i + vec_len <= n) : (i += vec_len) {
        const sv: Vec = x[i..][0..vec_len].*;
        max_acc = @max(max_acc, sv);
    }
    var max_val: f32 = @reduce(.Max, max_acc);
    while (i < n) : (i += 1) max_val = @max(max_val, x[i]);

    // Pass 2: exp(x - max) and sum (fused, SIMD)
    const max_v: Vec = @splat(max_val);
    var sum_acc: Vec = @as(Vec, @splat(0.0));
    i = 0;
    while (i + vec_len <= n) : (i += vec_len) {
        const sv: Vec = x[i..][0..vec_len].*;
        const ev = @exp(sv - max_v);
        x[i..][0..vec_len].* = ev;
        sum_acc += ev;
    }
    var sum_exp: f32 = @reduce(.Add, sum_acc);
    while (i < n) : (i += 1) {
        x[i] = @exp(x[i] - max_val);
        sum_exp += x[i];
    }

    // Pass 3: normalize (SIMD)
    if (sum_exp > 0.0) {
        const inv_sum = 1.0 / sum_exp;
        const inv_v: Vec = @splat(inv_sum);
        i = 0;
        while (i + vec_len <= n) : (i += vec_len) {
            x[i..][0..vec_len].* = @as(Vec, x[i..][0..vec_len].*) * inv_v;
        }
        while (i < n) : (i += 1) x[i] *= inv_sum;
    }
}

/// Get a tensor's data as an f32 pointer (only valid for f32 dtype tensors).
fn tensorAsF32(t: TensorInfo) [*]const f32 {
    return @ptrCast(@alignCast(t.data_ptr));
}

/// Apply 2D neox-style RoPE to Q or K buffer for all patches.
/// For each patch at grid position (col=px, row=py):
///   - First half of each head: RoPE with position = px (column)
///   - Second half of each head: RoPE with position = py (row)
/// This matches llama.cpp's gemma4v.cpp add_pos callback which splits
/// each head dimension in two and applies separate RoPE for x and y.
fn applyRope2d(buf: []f32, np: usize, nh: usize, hd: usize, pps: usize, theta: f32) void {
    const half = hd / 2;
    const rope_dim = half; // each half gets its own RoPE
    const half_half = rope_dim / 2;
    const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));
    const neg_log_theta: f32 = -@log(theta);
    const ed = nh * hd;

    // Precompute cos/sin tables for all needed positions (0..pps-1)
    // Each position needs half_half frequency pairs
    std.debug.assert(half_half <= max_rope_half);
    std.debug.assert(pps <= 256); // reasonable upper bound

    for (0..np) |p| {
        const py = p / pps;
        const px = p % pps;
        const p_base = p * ed;

        for (0..nh) |h| {
            const h_base = p_base + h * hd;

            // First half of head: RoPE with position = px (column)
            {
                const pos: f32 = @floatFromInt(px);
                var i: usize = 0;
                while (i < half_half) : (i += 1) {
                    const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
                    const angle = pos * freq;
                    const cos_a = @cos(angle);
                    const sin_a = @sin(angle);
                    const r = buf[h_base + i];
                    const im = buf[h_base + i + half_half];
                    buf[h_base + i] = @mulAdd(f32, r, cos_a, -(im * sin_a));
                    buf[h_base + i + half_half] = @mulAdd(f32, r, sin_a, im * cos_a);
                }
            }

            // Second half of head: RoPE with position = py (row)
            {
                const pos: f32 = @floatFromInt(py);
                const off = h_base + half;
                var i: usize = 0;
                while (i < half_half) : (i += 1) {
                    const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
                    const angle = pos * freq;
                    const cos_a = @cos(angle);
                    const sin_a = @sin(angle);
                    const r = buf[off + i];
                    const im = buf[off + i + half_half];
                    buf[off + i] = @mulAdd(f32, r, cos_a, -(im * sin_a));
                    buf[off + i + half_half] = @mulAdd(f32, r, sin_a, im * cos_a);
                }
            }
        }
    }
}

/// Apply M-RoPE (multi-resolution rotary position encoding) for Qwen VL.
/// Splits each head into 4 equal sections and applies independent RoPE:
///   Section 0: temporal position (0 for static images)
///   Section 1: height position (py)
///   Section 2: height position (py, same for single frame)
///   Section 3: width position (px)
/// Matches llama.cpp: ggml_rope_multi with mrope_sections=[d/4,d/4,d/4,d/4].
fn applyMRope(buf: []f32, np: usize, nh: usize, hd: usize, pps: usize, theta: f32) void {
    const n_sections: usize = 4;
    const sec_dim = hd / n_sections;
    const sec_half = sec_dim / 2;
    const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(sec_dim));
    const neg_log_theta: f32 = -@log(theta);
    const ed = nh * hd;

    for (0..np) |p| {
        const py = p / pps;
        const px = p % pps;
        const positions = [n_sections]f32{
            0, // temporal (always 0 for images)
            @floatFromInt(py), // height
            @floatFromInt(py), // height (same for single frame)
            @floatFromInt(px), // width
        };
        const p_base = p * ed;

        for (0..nh) |h| {
            const h_base = p_base + h * hd;

            for (0..n_sections) |s| {
                const pos = positions[s];
                const s_base = h_base + s * sec_dim;

                for (0..sec_half) |i| {
                    const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
                    const angle = pos * freq;
                    const cos_a = @cos(angle);
                    const sin_a = @sin(angle);
                    const r = buf[s_base + i];
                    const im = buf[s_base + i + sec_half];
                    buf[s_base + i] = @mulAdd(f32, r, cos_a, -(im * sin_a));
                    buf[s_base + i + sec_half] = @mulAdd(f32, r, sin_a, im * cos_a);
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "VisionEncoder config defaults" {
    // Verify default SigLIP-2 configuration values
    try std.testing.expectEqual(@as(u32, 224), default_image_size);
    try std.testing.expectEqual(@as(u32, 16), default_patch_size);
    try std.testing.expectEqual(@as(u32, 196), default_image_size / default_patch_size * (default_image_size / default_patch_size));
    try std.testing.expectEqual(@as(u32, 1152), default_embd_dim);
    try std.testing.expectEqual(@as(u32, 72), default_embd_dim / default_n_heads);
    try std.testing.expectEqual(@as(u32, 4304), default_ffn_dim);
    try std.testing.expectEqual(@as(u32, 27), default_n_blocks);
}

test "softmaxSimd basic" {
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    softmaxSimd(&x);
    // Sum should be 1.0
    var sum: f32 = 0.0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    // Largest input should have largest probability
    try std.testing.expect(x[2] > x[1]);
    try std.testing.expect(x[1] > x[0]);
}

test "softmaxSimd uniform" {
    var x = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    softmaxSimd(&x);
    for (x) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 1e-6);
    }
}

test "softmaxSimd exercises SIMD path" {
    // 16 elements ensures the vec_len=8 SIMD loop runs at least twice.
    var x: [16]f32 = undefined;
    for (0..16) |i| x[i] = @as(f32, @floatFromInt(i)) * 0.5;
    softmaxSimd(&x);
    var sum: f32 = 0.0;
    for (x) |v| {
        try std.testing.expect(v >= 0.0);
        sum += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // Monotonically increasing input should yield monotonically increasing output
    for (1..16) |i| try std.testing.expect(x[i] >= x[i - 1]);
}

test "silu activation" {
    // silu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-6);
    // silu(x) ≈ x for large positive x
    const large_val = silu(10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), large_val, 0.01);
    // silu is odd-ish: silu(-x) = -x * sigmoid(-x)
    const neg_val = silu(-2.0);
    try std.testing.expect(neg_val < 0.0);
}

test "geluScalar activation" {
    // gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), geluScalar(0.0), 1e-6);
    // gelu(x) ≈ x for large positive x
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), geluScalar(3.0), 0.01);
    // gelu is negative for small negative x
    try std.testing.expect(geluScalar(-0.5) < 0.0);
    // gelu(-x) approaches 0 for large negative x
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), geluScalar(-5.0), 0.01);
}

test "geluInPlace" {
    var x = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    geluInPlace(&x);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[0], 1e-6);
    // GELU(1.0) ≈ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), x[1], 0.01);
    // GELU(-1.0) ≈ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), x[2], 0.01);
}

test "layerNormCpu basic" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    layerNormCpu(&input, &output, &weight, &bias, 4, 1e-6);
    // mean = 2.5, var = 1.25, std = sqrt(1.25)
    const mean: f32 = 2.5;
    const inv_std = 1.0 / @sqrt(@as(f32, 1.25) + 1e-6);
    for (0..4) |i| {
        const expected = (@as(f32, @floatFromInt(i + 1)) - mean) * inv_std;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "layerNormCpu with bias" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    const weight = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const bias = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    layerNormCpu(&input, &output, &weight, &bias, 4, 1e-6);
    const mean: f32 = 2.5;
    const inv_std = 1.0 / @sqrt(@as(f32, 1.25) + 1e-6);
    for (0..4) |i| {
        const expected = (@as(f32, @floatFromInt(i + 1)) - mean) * inv_std * 2.0 + 0.5;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "layerNormInPlace basic" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    layerNormInPlace(&x, &weight, &bias, 4, 1e-6);
    const mean: f32 = 2.5;
    const inv_std = 1.0 / @sqrt(@as(f32, 1.25) + 1e-6);
    for (0..4) |i| {
        const expected = (original[i] - mean) * inv_std;
        try std.testing.expectApproxEqAbs(expected, x[i], 1e-4);
    }
}

test "rmsNormCpu basic" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    rmsNormCpu(&input, &output, &weight, 4, 1e-6);
    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5)
    const rms = @sqrt(@as(f32, 7.5));
    for (0..4) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) / rms;
        try std.testing.expectApproxEqAbs(expected, output[i], 1e-4);
    }
}

test "rmsNormInPlace basic" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    rmsNormInPlace(&x, &weight, 4, 1e-6);
    const rms = @sqrt(@as(f32, 7.5));
    for (0..4) |i| {
        const expected = @as(f32, @floatFromInt(i + 1)) / rms * 2.0;
        try std.testing.expectApproxEqAbs(expected, x[i], 1e-4);
    }
}

test "gemvCpu f32" {
    // Test 2x3 matrix-vector multiply
    // W = [[1,2,3],[4,5,6]], x = [1,1,1]
    // y = [6, 15]
    const TI = format_mod.TensorInfo;
    const w_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x_data = [_]f32{ 1, 1, 1 };
    var y_data: [2]f32 = undefined;

    const t = TI{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 2, 3, 0, 0 },
        .dtype = .f32,
        .data_ptr = @ptrCast(&w_data),
    };

    gemvCpu(&x_data, t, &y_data, 2, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), y_data[1], 1e-5);
}

test "pixel normalization" {
    // Verify pixel_scale maps 255 -> ~1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), 255.0 * pixel_scale, 1e-6);
}

test "VisionVariant detection constants" {
    // Verify variant enum values exist and are distinct
    try std.testing.expect(@intFromEnum(VisionVariant.gemma4_siglip2) != @intFromEnum(VisionVariant.gemma3_siglip));
    try std.testing.expect(@intFromEnum(VisionVariant.gemma3_siglip) != @intFromEnum(VisionVariant.qwen_vl));
    try std.testing.expect(@intFromEnum(VisionVariant.gemma4_siglip2) != @intFromEnum(VisionVariant.qwen_vl));
}

test "attention_chunk_size guard" {
    // Verify the chunk constant is reasonable: with chunk_size=64 and 4096 patches,
    // scores buffer is 16 * 64 * 4096 * 4 = 16MB (bounded, vs 1GB for full n^2).
    try std.testing.expectEqual(@as(u32, 64), attention_chunk_size);
    const scores_bytes = @as(usize, default_n_heads) * attention_chunk_size * 4096 * @sizeOf(f32);
    // Should be 16MB
    try std.testing.expectEqual(@as(usize, 16 * 64 * 4096 * 4), scores_bytes);
}
