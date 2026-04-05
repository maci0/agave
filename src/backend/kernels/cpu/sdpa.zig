//! CPU scaled dot-product attention kernel.

const std = @import("std");
const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

/// Maximum sequence length supported by the CPU SDPA scores buffer.
pub const max_sdpa_seq_len: usize = 8192;
/// Maximum per-head dimension for the cached Q vector in SDPA.
pub const max_head_dim: usize = 256;

const kv_quant = @import("../../../ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

/// CPU scaled dot-product attention with KV cache append (f32 path).
/// Appends k_new/v_new at position seq_len, then computes attention over seq_len+1 positions.
pub fn sdpa(q: [*]const f32, keys: []f32, values: []f32, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32) void {
    const kvd = nkv * hd;
    const kv_off = seq_len * kvd;
    @memcpy(keys[kv_off..][0..kvd], k_new[0..kvd]);
    @memcpy(values[kv_off..][0..kvd], v_new[0..kvd]);
    sdpaHeads(q, keys.ptr, values.ptr, output, nh, nkv, hd, seq_len + 1, scale);
}

/// Process all heads sequentially with f32 KV data.
pub fn sdpaHeads(q: [*]const f32, keys: [*]const f32, values: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32) void {
    for (0..nh) |h| {
        sdpaHead(q, keys, values, output, h, nh, nkv, hd, sl, scale);
    }
}

/// Process a single query head with f32 KV: QK dot products, softmax, V accumulation.
/// Thread-safe — each head writes to its own output region.
pub fn sdpaHead(q: [*]const f32, keys: [*]const f32, values: [*]const f32, output: [*]f32, h: usize, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = h * hd;
    std.debug.assert(sl <= max_sdpa_seq_len);
    var scores_buf: [max_sdpa_seq_len]f32 = undefined;

    // Cache Q for this head
    var q_cached: [max_head_dim]f32 = undefined;
    @memcpy(q_cached[0..hd], q[q_base..][0..hd]);

    // QK dot products
    for (0..sl) |t| {
        const k_base = t * kvd + kvh * hd;
        var acc: V8 = v8zero;
        var d: usize = 0;
        while (d + 8 <= hd) : (d += 8) {
            const qv: V8 = q_cached[d..][0..8].*;
            const kv: V8 = keys[k_base + d ..][0..8].*;
            acc = @mulAdd(V8, qv, kv, acc);
        }
        var dot = @reduce(.Add, acc);
        while (d < hd) : (d += 1) dot = @mulAdd(f32, q_cached[d], keys[k_base + d], dot);
        scores_buf[t] = dot * scale;
    }

    // Softmax
    softmax(scores_buf[0..sl]);

    // V accumulation — position-outer, dimension-inner for cache locality.
    // Each V row (hd floats) fits in L1; iterating dimensions sequentially
    // avoids the kvd-stride cache thrashing of the dimension-outer layout.
    {
        var d: usize = 0;
        while (d + 8 <= hd) : (d += 8) {
            output[q_base + d ..][0..8].* = v8zero;
        }
        while (d < hd) : (d += 1) output[q_base + d] = 0;

        for (0..sl) |t| {
            const v_base = t * kvd + kvh * hd;
            const sv: V8 = @splat(scores_buf[t]);
            d = 0;
            while (d + 8 <= hd) : (d += 8) {
                const vv: V8 = values[v_base + d ..][0..8].*;
                const cur: V8 = output[q_base + d ..][0..8].*;
                output[q_base + d ..][0..8].* = @mulAdd(V8, sv, vv, cur);
            }
            while (d < hd) : (d += 1) {
                output[q_base + d] = @mulAdd(f32, scores_buf[t], values[v_base + d], output[q_base + d]);
            }
        }
    }
}

/// Process all heads sequentially with quantized KV data.
pub fn sdpaQuantHeads(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
    for (0..nh) |h| {
        sdpaQuantHead(q, keys, values, output, h, nh, nkv, hd, sl, scale, kv_type_k, kv_type_v);
    }
}

/// Process a single query head with quantized KV cache.
/// Uses kvDot for QK dot products (kv_type_k) and kvMulAccum for V accumulation (kv_type_v).
pub fn sdpaQuantHead(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, h: usize, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = h * hd;
    std.debug.assert(sl <= max_sdpa_seq_len);
    var scores_buf: [max_sdpa_seq_len]f32 = undefined;

    // Cache Q for this head
    var q_cached: [max_head_dim]f32 = undefined;
    @memcpy(q_cached[0..hd], q[q_base..][0..hd]);

    // QK dot products using quantized KV (key type)
    for (0..sl) |t| {
        const k_byte_off = kv_quant.kvByteOffset(kv_type_k, t * kvd + kvh * hd);
        scores_buf[t] = kv_quant.kvDot(q_cached[0..hd].ptr, keys + k_byte_off, hd, kv_type_k) * scale;
    }

    // Softmax
    softmax(scores_buf[0..sl]);

    // V accumulation using quantized KV (value type)
    @memset(output[q_base..][0..hd], 0);
    for (0..sl) |t| {
        const v_byte_off = kv_quant.kvByteOffset(kv_type_v, t * kvd + kvh * hd);
        kv_quant.kvMulAccum(output + q_base, scores_buf[t], values + v_byte_off, hd, kv_type_v);
    }
}

/// Process all heads sequentially with quantized KV, returning per-head softmax stats.
/// Same as sdpaQuantHeads but additionally outputs max and sum for online softmax merge.
pub fn sdpaQuantHeadsWithStats(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, head_max: [*]f32, head_sum: [*]f32) void {
    for (0..nh) |h| {
        sdpaQuantHeadWithStats(q, keys, values, output, h, nh, nkv, hd, sl, scale, kv_type_k, kv_type_v, head_max, head_sum);
    }
}

/// Process a single query head with quantized KV cache, returning softmax stats.
/// Same as sdpaQuantHead but additionally outputs max and sum for online softmax merge.
/// head_max[h] receives the pre-softmax max score, head_sum[h] receives sum(exp(scores - max)).
pub fn sdpaQuantHeadWithStats(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, h: usize, nh: usize, nkv: usize, hd: usize, sl: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, head_max: [*]f32, head_sum: [*]f32) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = h * hd;
    std.debug.assert(sl <= max_sdpa_seq_len);
    var scores_buf: [max_sdpa_seq_len]f32 = undefined;

    // Cache Q for this head
    var q_cached: [max_head_dim]f32 = undefined;
    @memcpy(q_cached[0..hd], q[q_base..][0..hd]);

    // QK dot products using quantized KV (key type)
    for (0..sl) |t| {
        const k_byte_off = kv_quant.kvByteOffset(kv_type_k, t * kvd + kvh * hd);
        scores_buf[t] = kv_quant.kvDot(q_cached[0..hd].ptr, keys + k_byte_off, hd, kv_type_k) * scale;
    }

    // Softmax — capture max and sum before normalization
    const scores = scores_buf[0..sl];
    var max_val: f32 = scores[0];
    for (scores[1..]) |s| max_val = @max(max_val, s);
    var sum_val: f32 = 0;
    for (scores) |*s| {
        s.* = @exp(s.* - max_val);
        sum_val += s.*;
    }

    // Export stats for online softmax merge
    head_max[h] = max_val;
    head_sum[h] = sum_val;

    // Normalize
    const inv_sum: f32 = if (sum_val > 0) 1.0 / sum_val else 0;
    for (scores) |*s| s.* *= inv_sum;

    // V accumulation using quantized KV (value type)
    @memset(output[q_base..][0..hd], 0);
    for (0..sl) |t| {
        const v_byte_off = kv_quant.kvByteOffset(kv_type_v, t * kvd + kvh * hd);
        kv_quant.kvMulAccum(output + q_base, scores_buf[t], values + v_byte_off, hd, kv_type_v);
    }
}

/// In-place softmax over a score buffer. SIMD-accelerated max, exp+sum, and normalize.
fn softmax(scores: []f32) void {
    const n = scores.len;
    if (n == 0) return;

    // Pass 1: find max (SIMD)
    var max_acc: V8 = @splat(scores[0]);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const sv: V8 = scores[i..][0..8].*;
        max_acc = @max(max_acc, sv);
    }
    var max_val: f32 = @reduce(.Max, max_acc);
    while (i < n) : (i += 1) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    // Pass 2: exp and sum (fused, SIMD)
    const max_v: V8 = @splat(max_val);
    var sum_acc: V8 = v8zero;
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const sv: V8 = scores[i..][0..8].*;
        const ev = @exp(sv - max_v);
        scores[i..][0..8].* = ev;
        sum_acc += ev;
    }
    var sum: f32 = @reduce(.Add, sum_acc);
    while (i < n) : (i += 1) {
        scores[i] = @exp(scores[i] - max_val);
        sum += scores[i];
    }

    // Pass 3: normalize (SIMD)
    const inv_sum = 1.0 / sum;
    const inv_v: V8 = @splat(inv_sum);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        scores[i..][0..8].* = @as(V8, scores[i..][0..8].*) * inv_v;
    }
    while (i < n) : (i += 1) scores[i] *= inv_sum;
}

test "sdpa single head single position" {

    // 1 head, hd=4, seq_len=0 (first token) → just copies k/v and returns v.
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var keys: [8]f32 = undefined; // space for 2 positions
    var values: [8]f32 = undefined;
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.6, 0.7, 0.8 };
    var output: [4]f32 = undefined;
    sdpa(&q, &keys, &values, &k_new, &v_new, &output, 1, 1, 4, 0, 1.0);
    // With seq_len=0, only 1 position → softmax([score]) = [1.0] → output = v_new
    for (0..4) |i| try std.testing.expectApproxEqAbs(v_new[i], output[i], 1e-5);
}

test "sdpa GQA two query heads one KV head" {

    // 2 query heads, 1 KV head (GQA), hd=4
    // Both query heads share the same KV, so outputs should match for identical queries.
    var keys = [_]f32{0} ** 24; // space for 3 positions × 1 KV head × 4 dims
    var values = [_]f32{0} ** 24;
    var output: [8]f32 = undefined; // 2 heads × 4 dims

    // Insert at pos 0: k=[1,0,0,0], v=[1,0,0,0]
    var k0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    @memcpy(keys[0..4], &k0);
    @memcpy(values[0..4], &v0);

    // Insert at pos 1: k=[0,0,0,1], v=[0,1,0,0]
    var k1 = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    var v1 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    @memcpy(keys[4..8], &k1);
    @memcpy(values[4..8], &v1);

    // Both Q heads query with [0,0,0,1] → aligns with k1
    var q = [_]f32{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    sdpaHeads(&q, &keys, &values, &output, 2, 1, 4, 2, 1.0);

    const e = @exp(@as(f32, 1.0));
    const w0 = 1.0 / (1.0 + e);
    const w1 = e / (1.0 + e);
    // Head 0 output
    try std.testing.expectApproxEqAbs(w0, output[0], 1e-4);
    try std.testing.expectApproxEqAbs(w1, output[1], 1e-4);
    // Head 1 should produce identical output (shares same KV head)
    try std.testing.expectApproxEqAbs(w0, output[4], 1e-4);
    try std.testing.expectApproxEqAbs(w1, output[5], 1e-4);
}

test "sdpa single head two positions" {

    // 1 head, hd=4, first insert a key/value at seq_len=0, then query at seq_len=1.
    var keys = [_]f32{0} ** 12; // space for 3 positions
    var values = [_]f32{0} ** 12;
    var output: [4]f32 = undefined;

    // Position 0: insert k0 = [1,0,0,0], v0 = [1,0,0,0]
    var k0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var q0 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    sdpa(&q0, &keys, &values, &k0, &v0, &output, 1, 1, 4, 0, 1.0);

    // Position 1: q aligns with k1, so attention should weight v1 more
    // k1 = [0,0,0,1], v1 = [0,1,0,0], q = [0,0,0,1]
    // QK scores: q·k0 = 0, q·k1 = 1 → softmax → [exp(0), exp(1)] / Z
    var k1 = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    var v1 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    var q1 = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    sdpa(&q1, &keys, &values, &k1, &v1, &output, 1, 1, 4, 1, 1.0);

    // Expected: softmax([0, 1]) = [1/(1+e), e/(1+e)] ≈ [0.269, 0.731]
    // output = 0.269 * v0 + 0.731 * v1 = [0.269, 0.731, 0, 0]
    const e = @exp(@as(f32, 1.0));
    const w0 = 1.0 / (1.0 + e); // ≈ 0.269
    const w1 = e / (1.0 + e); // ≈ 0.731
    try std.testing.expectApproxEqAbs(w0 * 1.0, output[0], 1e-4); // w0 * v0[0]
    try std.testing.expectApproxEqAbs(w1 * 1.0, output[1], 1e-4); // w1 * v1[1]
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[3], 1e-4);
}

test "sdpa quantized f16 roundtrip" {

    // Test that quantized SDPA path produces correct results with f16 KV
    const hd = 4;
    const kvd = hd; // 1 KV head
    const max_sl = 4;
    const kv_bytes = kv_quant.kvSliceBytes(.f16, max_sl * kvd);
    var keys_buf: [kv_bytes]u8 = .{0} ** kv_bytes;
    var vals_buf: [kv_bytes]u8 = .{0} ** kv_bytes;

    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k_new = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v_new = [_]f32{ 0.5, 0.6, 0.7, 0.8 };
    var output: [4]f32 = undefined;

    // Append at pos 0
    kv_quant.kvStore(&keys_buf, &k_new, kvd, .f16);
    kv_quant.kvStore(&vals_buf, &v_new, kvd, .f16);

    // Run quantized SDPA for 1 position
    sdpaQuantHeads(&q, &keys_buf, &vals_buf, &output, 1, 1, hd, 1, 1.0, .f16);

    // Single position: output ≈ v_new (with f16 precision)
    for (0..4) |i| try std.testing.expectApproxEqAbs(v_new[i], output[i], 1e-3);
}
