//! Scaled dot-product attention kernels.
//! Provides a shared SDPA implementation used by all model architectures,
//! supporting both full causal attention and sliding-window variants.

const std = @import("std");
const Backend = @import("../backend/backend.zig").Backend;
const kv_quant = @import("kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

/// SIMD vector width (number of f32 lanes) used for dot-product and accumulation loops.
const simd_width: usize = 8;

/// Append current K/V vectors to the per-layer KV cache, then compute
/// scaled dot-product attention: softmax(Q @ K^T / sqrt(hd)) @ V.
///
/// Parameters:
///   - q: Query buffer [nh * hd], pre-scaled if needed.
///   - kv_keys: KV cache keys for this layer (byte slice, format determined by kv_type).
///   - kv_values: KV cache values for this layer (byte slice).
///   - k_buf: Current key vector [kvd] to append to cache.
///   - v_buf: Current value vector [kvd] to append to cache.
///   - attn_out: Output buffer [nh * hd].
///   - scores: Scratch buffer for attention scores [max_seq_len + extra].
///   - nh: Number of query heads.
///   - nkv: Number of KV heads (for GQA; nh / nkv = heads per group).
///   - hd: Head dimension.
///   - seq_len: Current sequence position (0-indexed, before appending).
///   - scale: Attention scale factor (typically 1/sqrt(hd), or pre-baked).
///   - be: Backend for SDPA dispatch (fast path) or softmax + sync (windowed fallback).
///   - window: If non-null, sliding window config: .start and .len.
///   - score_offset: Starting index in scores buffer (for prepended sink logits).
///   - kv_type_k: KV cache quantization format for keys.
///   - kv_type_v: KV cache quantization format for values.
pub fn scaledDotProductAttention(
    q: [*]const f32,
    kv_keys: []u8,
    kv_values: []u8,
    k_buf: []const f32,
    v_buf: []const f32,
    attn_out: [*]f32,
    scores: [*]f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
    be: Backend,
    window: ?struct { start: usize, len: usize },
    score_offset: usize,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
) void {
    const kvd = nkv * hd;

    // Fast path: no window, no score offset → delegate KV append + attention to backend.
    // All backends handle KV append + attention in one call (GPU: fused kernel, CPU: inline).
    if (window == null and score_offset == 0) {
        be.sdpa(q, kv_keys, kv_values, k_buf.ptr, v_buf.ptr, attn_out, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        return;
    }

    // Windowed / offset fallback — explicit KV append + CPU-side SDPA loop.
    // Sync first to flush any pending GPU ops (no-op on CPU backend).
    be.sync();

    // KV append: quantize k_buf/v_buf into cache (use respective types)
    const k_byte_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
    const v_byte_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
    kv_quant.kvStore(kv_keys.ptr + k_byte_off, k_buf.ptr, kvd, kv_type_k);
    kv_quant.kvStore(kv_values.ptr + v_byte_off, v_buf.ptr, kvd, kv_type_v);
    const sl = seq_len + 1;

    const win_start = if (window) |w| w.start else 0;
    const win_len = if (window) |w| w.len else sl;
    const hpg = nh / nkv;

    // f32 fast path for windowed attention — use existing SIMD code
    if (kv_type_k == .f32 and kv_type_v == .f32) {
        const f32_keys: [*]const f32 = @ptrCast(@alignCast(kv_keys.ptr));
        const f32_values: [*]const f32 = @ptrCast(@alignCast(kv_values.ptr));
        const SimdVec = @Vector(simd_width, f32);

        for (0..nh) |h| {
            const kvh = h / hpg;
            const q_base = h * hd;

            for (0..win_len) |wi| {
                const t = win_start + wi;
                const k_base = t * kvd + kvh * hd;
                var acc: SimdVec = @splat(0.0);
                var d: usize = 0;
                while (d + simd_width <= hd) : (d += simd_width) {
                    const qv: SimdVec = q[q_base + d ..][0..simd_width].*;
                    const kv: SimdVec = f32_keys[k_base + d ..][0..simd_width].*;
                    acc = @mulAdd(SimdVec, qv, kv, acc);
                }
                var dot = @reduce(.Add, acc);
                while (d < hd) : (d += 1) dot += q[q_base + d] * f32_keys[k_base + d];
                scores[score_offset + wi] = dot * scale;
            }

            const n_scores = score_offset + win_len;
            be.softmax(scores, n_scores);

            // V accumulation — position-outer, dimension-inner for cache locality.
            @memset(attn_out[q_base..][0..hd], 0);

            for (0..win_len) |wi| {
                const t = win_start + wi;
                const v_base = t * kvd + kvh * hd;
                const sv: SimdVec = @splat(scores[score_offset + wi]);
                var d: usize = 0;
                while (d + simd_width <= hd) : (d += simd_width) {
                    const vv: SimdVec = f32_values[v_base + d ..][0..simd_width].*;
                    const cur: SimdVec = attn_out[q_base + d ..][0..simd_width].*;
                    attn_out[q_base + d ..][0..simd_width].* = @mulAdd(SimdVec, sv, vv, cur);
                }
                while (d < hd) : (d += 1) {
                    attn_out[q_base + d] = @mulAdd(f32, scores[score_offset + wi], f32_values[v_base + d], attn_out[q_base + d]);
                }
            }
        }
        return;
    }

    // Quantized windowed fallback — use kvDot (kv_type_k) / kvMulAccum (kv_type_v)
    for (0..nh) |h| {
        const kvh = h / hpg;
        const q_base = h * hd;

        // QK dot products (key type)
        for (0..win_len) |wi| {
            const t = win_start + wi;
            const elem_off = t * kvd + kvh * hd;
            const k_off = kv_quant.kvByteOffset(kv_type_k, elem_off);
            scores[score_offset + wi] = kv_quant.kvDot(q + q_base, kv_keys.ptr + k_off, hd, kv_type_k) * scale;
        }

        const n_scores = score_offset + win_len;
        be.softmax(scores, n_scores);

        // V accumulation (value type)
        @memset(attn_out[q_base..][0..hd], 0);
        for (0..win_len) |wi| {
            const t = win_start + wi;
            const elem_off = t * kvd + kvh * hd;
            const v_off = kv_quant.kvByteOffset(kv_type_v, elem_off);
            kv_quant.kvMulAccum(attn_out + q_base, scores[score_offset + wi], kv_values.ptr + v_off, hd, kv_type_v);
        }
    }
}

// ── PagedAttention ────────────────────────────────────────────────

const CacheBlock = @import("../kvcache/manager.zig").CacheBlock;

/// Paged SDPA: block-table-based attention (f32 KV cache only, no windowing/offsets).
///
/// Parameters:
///   - q: Query buffer [nh * hd].
///   - blocks: Pool of physical cache blocks.
///   - block_table: Mapping from logical block index to physical block id [num_blocks].
///   - k_buf: Current key vector [kvd] to append to cache.
///   - v_buf: Current value vector [kvd] to append to cache.
///   - attn_out: Output buffer [nh * hd].
///   - scores: Scratch buffer [seq_len + 1].
///   - nh, nkv, hd: Head configuration.
///   - seq_len: Current sequence position (before appending).
///   - scale: Attention scale factor.
///   - be: Backend for softmax.
///   - block_size: Positions per cache block.
pub fn pagedAttention(
    q: [*]const f32,
    blocks: []CacheBlock,
    block_table: []const u32,
    k_buf: []const f32,
    v_buf: []const f32,
    attn_out: [*]f32,
    scores: [*]f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
    be: Backend,
    block_size: usize,
) void {
    const kvd = nkv * hd;

    // Append current K/V to the block for position seq_len
    const logical_block = seq_len / block_size;
    const block_offset = seq_len % block_size;
    std.debug.assert(logical_block < block_table.len);
    {
        const phys = block_table[logical_block];
        std.debug.assert(phys < blocks.len);
        const blk = &blocks[phys];
        const off = block_offset * kvd;
        @memcpy(blk.keys[off..][0..kvd], k_buf[0..kvd]);
        @memcpy(blk.values[off..][0..kvd], v_buf[0..kvd]);
        blk.used = @intCast(@min(block_offset + 1, block_size));
    }

    const sl = seq_len + 1;
    const hpg = nh / nkv;
    const SimdVec = @Vector(simd_width, f32);

    for (0..nh) |h| {
        const kvh = h / hpg;
        const q_base = h * hd;

        // QK dot products — look up K from block table
        for (0..sl) |t| {
            const lb = t / block_size;
            const bo = t % block_size;
            const phys = block_table[lb];
            const k_start = bo * kvd + kvh * hd;

            var acc: SimdVec = @splat(0.0);
            var d: usize = 0;
            while (d + simd_width <= hd) : (d += simd_width) {
                const qv: SimdVec = q[q_base + d ..][0..simd_width].*;
                const kv: SimdVec = blocks[phys].keys[k_start + d ..][0..simd_width].*;
                acc = @mulAdd(SimdVec, qv, kv, acc);
            }
            var dot = @reduce(.Add, acc);
            while (d < hd) : (d += 1) dot += q[q_base + d] * blocks[phys].keys[k_start + d];
            scores[t] = dot * scale;
        }

        be.softmax(scores, sl);

        // Value accumulation from block table — t-outer loop to compute
        // block lookups (div/mod) once per position instead of per dimension.
        {
            // Zero-init output for this head
            @memset(attn_out[q_base..][0..hd], 0);

            for (0..sl) |t| {
                const lb = t / block_size;
                const bo = t % block_size;
                const phys = block_table[lb];
                const v_start = bo * kvd + kvh * hd;
                const sv: SimdVec = @splat(scores[t]);
                const v_row = blocks[phys].values;

                var d: usize = 0;
                while (d + simd_width <= hd) : (d += simd_width) {
                    const vv: SimdVec = v_row[v_start + d ..][0..simd_width].*;
                    const cur: SimdVec = attn_out[q_base + d ..][0..simd_width].*;
                    attn_out[q_base + d ..][0..simd_width].* = @mulAdd(SimdVec, sv, vv, cur);
                }
                while (d < hd) : (d += 1) {
                    attn_out[q_base + d] = @mulAdd(f32, scores[t], v_row[v_start + d], attn_out[q_base + d]);
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "sdpa single head single token" {
    // With seq_len=0 (first token), the attention should just return V
    var kv_keys_f32 = [_]f32{0} ** 256;
    var kv_values_f32 = [_]f32{0} ** 256;
    const kv_keys = std.mem.sliceAsBytes(&kv_keys_f32);
    const kv_values = std.mem.sliceAsBytes(&kv_values_f32);
    var k_buf = [_]f32{1.0} ** 4;
    var v_buf = [_]f32{0.5} ** 4;
    var q = [_]f32{1.0} ** 4;
    var attn_out = [_]f32{0} ** 4;
    var scores = [_]f32{0} ** 64;

    const BackendState = @import("../backend/backend.zig").BackendState;
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;

    scaledDotProductAttention(
        &q,
        kv_keys,
        kv_values,
        &k_buf,
        &v_buf,
        &attn_out,
        &scores,
        1,
        1,
        4,
        0,
        1.0,
        be,
        null,
        0,
        .f32,
        .f32,
    );

    // With single token, softmax([score]) = [1.0], so output = V
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), attn_out[i], 0.01);
    }
}

test "sdpa multi-token with GQA" {
    // 2 query heads, 1 KV head (GQA), hd=4
    // Insert two tokens then verify weighted output
    var kv_keys_f32 = [_]f32{0} ** 256;
    var kv_values_f32 = [_]f32{0} ** 256;
    const kv_keys = std.mem.sliceAsBytes(&kv_keys_f32);
    const kv_values = std.mem.sliceAsBytes(&kv_values_f32);
    var attn_out = [_]f32{0} ** 8; // 2 heads × 4 dims
    var scores = [_]f32{0} ** 64;

    const BackendState = @import("../backend/backend.zig").BackendState;
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;

    // Token 0: k=[1,0,0,0], v=[1,0,0,0]
    var k0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var q0 = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
    scaledDotProductAttention(&q0, kv_keys, kv_values, &k0, &v0, &attn_out, &scores, 2, 1, 4, 0, 1.0, be, null, 0, .f32, .f32);

    // Token 1: k=[0,0,0,1], v=[0,1,0,0], q=[0,0,0,1,...] aligns with k1
    var k1 = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    var v1 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    var q1 = [_]f32{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    scaledDotProductAttention(&q1, kv_keys, kv_values, &k1, &v1, &attn_out, &scores, 2, 1, 4, 1, 1.0, be, null, 0, .f32, .f32);

    // softmax([0, 1]) = [1/(1+e), e/(1+e)]
    const e = @exp(@as(f32, 1.0));
    const w0 = 1.0 / (1.0 + e);
    const w1 = e / (1.0 + e);
    // Head 0: output = w0*v0 + w1*v1
    try std.testing.expectApproxEqAbs(w0, attn_out[0], 1e-4);
    try std.testing.expectApproxEqAbs(w1, attn_out[1], 1e-4);
    // Head 1 shares same KV (GQA) — should produce same result
    try std.testing.expectApproxEqAbs(w0, attn_out[4], 1e-4);
    try std.testing.expectApproxEqAbs(w1, attn_out[5], 1e-4);
}

test "paged attention single head single token" {
    const allocator = std.testing.allocator;
    const manager = @import("../kvcache/manager.zig");

    // Create a paged cache with 1 block of size 16
    var paged = try manager.PagedKvCache.init(allocator, 1, 4, 2, 16);
    defer paged.deinit();

    // Allocate one block (must succeed — we configured 1 layer × block capacity)
    const blk_id = paged.allocBlock() orelse return error.TestUnexpectedResult;
    var block_table = [_]u32{blk_id};

    var q = [_]f32{1.0} ** 4;
    var k_buf = [_]f32{1.0} ** 4;
    var v_buf = [_]f32{0.5} ** 4;
    var attn_out = [_]f32{0} ** 4;
    var scores = [_]f32{0} ** 64;

    const BackendState = @import("../backend/backend.zig").BackendState;
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;

    pagedAttention(
        &q,
        paged.blocks,
        &block_table,
        &k_buf,
        &v_buf,
        &attn_out,
        &scores,
        1, // nh
        1, // nkv
        4, // hd
        0, // seq_len
        1.0,
        be,
        16, // block_size
    );

    // Single token: softmax([score]) = [1.0], output = V
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), attn_out[i], 0.01);
    }
}

test "sdpa asymmetric kv types" {
    // Test asymmetric quantization: f32 keys + q8_0 values.
    // With seq_len=0 (first token), attention should return V (single position).
    const hd = 32; // Must be multiple of 32 for q8_0 block alignment
    const kvd = hd;
    const max_sl = 4;

    // Allocate KV cache buffers: keys as f32, values as q8_0
    const k_bytes = kv_quant.kvSliceBytes(.f32, max_sl * kvd);
    const v_bytes = kv_quant.kvSliceBytes(.q8_0, max_sl * kvd);
    var kv_keys_buf: [k_bytes]u8 = .{0} ** k_bytes;
    var kv_values_buf: [v_bytes]u8 = .{0} ** v_bytes;

    var k_buf = [_]f32{1.0} ** hd;
    var v_buf: [hd]f32 = undefined;
    for (0..hd) |i| v_buf[i] = @as(f32, @floatFromInt(i)) * 0.1;
    var q = [_]f32{1.0} ** hd;
    var attn_out: [hd]f32 = .{0} ** hd;
    var scores = [_]f32{0} ** 64;

    const BackendState = @import("../backend/backend.zig").BackendState;
    var bs = BackendState{};
    bs.init(std.testing.allocator, .cpu);
    defer if (bs.pool) |*p| p.deinit();
    const be = bs.be;

    scaledDotProductAttention(
        &q,
        &kv_keys_buf,
        &kv_values_buf,
        &k_buf,
        &v_buf,
        &attn_out,
        &scores,
        1, // nh
        1, // nkv
        hd,
        0, // seq_len
        1.0,
        be,
        null,
        0,
        .f32, // keys stored as f32
        .q8_0, // values stored as q8_0
    );

    // Single token: softmax([score]) = [1.0], output ≈ V (with q8_0 precision loss)
    for (0..hd) |i| {
        try std.testing.expectApproxEqAbs(v_buf[i], attn_out[i], 0.05);
    }
}
