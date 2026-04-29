//! CPU tree-masked scaled dot-product attention kernel.
//!
//! Processes B tree nodes in one call. Each node has its own query vector
//! and attends to: (1) all prefix KV entries (shared), (2) ancestor nodes
//! within the tree (determined by bitmask).
//!
//! Supports quantized prefix KV (f16, q8_0, turbo, etc.) via kvDot/kvMulAccum.
//! Tree K/V is always f32 (fresh from GEMM projections).

const std = @import("std");
const kv_quant = @import("../../../ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);

const max_sdpa_seq_len: usize = 8192;
const max_head_dim: usize = 256;

inline fn isAncestor(mask: [8]u64, j: usize) bool {
    return mask[j / 64] & (@as(u64, 1) << @intCast(@as(u6, @truncate(j)))) != 0;
}

/// Tree-masked SDPA for CPU.
///
/// Prefix K/V can be any quantized format (u8 byte array).
/// Tree K/V is always f32 (from GEMM projections).
pub fn sdpaTree(
    q_all: [*]const f32,
    prefix_keys: [*]const u8,
    prefix_values: [*]const u8,
    tree_keys: [*]const f32,
    tree_values: [*]const f32,
    output: [*]f32,
    ancestor_masks: [*]const [8]u64,
    nh: usize,
    nkv: usize,
    hd: usize,
    prefix_len: usize,
    n_nodes: u32,
    scale: f32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
) void {
    for (0..n_nodes) |node_i| {
        for (0..nh) |h| {
            sdpaTreeNodeHead(
                q_all,
                prefix_keys,
                prefix_values,
                tree_keys,
                tree_values,
                output,
                ancestor_masks,
                node_i,
                h,
                nh,
                nkv,
                hd,
                prefix_len,
                n_nodes,
                scale,
                kv_type_k,
                kv_type_v,
            );
        }
    }
}

fn sdpaTreeNodeHead(
    q_all: [*]const f32,
    prefix_keys: [*]const u8,
    prefix_values: [*]const u8,
    tree_keys: [*]const f32,
    tree_values: [*]const f32,
    output: [*]f32,
    ancestor_masks: [*]const [8]u64,
    node_i: usize,
    h: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    prefix_len: usize,
    n_nodes: u32,
    scale: f32,
    kv_type_k: KvQuantType,
    kv_type_v: KvQuantType,
) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    const kvh = h / hpg;
    const q_base = node_i * nh * hd + h * hd;

    var q_cached: [max_head_dim]f32 = undefined;
    @memcpy(q_cached[0..hd], q_all[q_base..][0..hd]);

    var scores: [max_sdpa_seq_len]f32 = undefined;
    var si: usize = 0;

    // Score against prefix KV (quantized, all attended)
    for (0..prefix_len) |t| {
        const k_off = kv_quant.kvByteOffset(kv_type_k, t * kvd + kvh * hd);
        scores[si] = kv_quant.kvDot(&q_cached, prefix_keys + k_off, hd, kv_type_k) * scale;
        si += 1;
    }

    // Score against tree nodes (f32, masked by ancestor bitmask)
    const mask = ancestor_masks[node_i];
    for (0..n_nodes) |j| {
        if (isAncestor(mask, j)) {
            scores[si] = dotProductF32(&q_cached, tree_keys + j * kvd + kvh * hd, hd) * scale;
            si += 1;
        }
    }

    softmax(scores[0..si]);

    // V accumulation
    const out_base = node_i * nh * hd + h * hd;
    @memset(output[out_base..][0..hd], 0);

    si = 0;
    // Prefix V (quantized)
    for (0..prefix_len) |t| {
        const v_off = kv_quant.kvByteOffset(kv_type_v, t * kvd + kvh * hd);
        kv_quant.kvMulAccum(output + out_base, scores[si], prefix_values + v_off, hd, kv_type_v);
        si += 1;
    }
    // Tree V (f32, masked)
    for (0..n_nodes) |j| {
        if (isAncestor(mask, j)) {
            mulAccumF32(output + out_base, scores[si], tree_values + j * kvd + kvh * hd, hd);
            si += 1;
        }
    }
}

inline fn dotProductF32(q: []const f32, k: [*]const f32, hd: usize) f32 {
    var acc: V8 = v8zero;
    var d: usize = 0;
    while (d + 8 <= hd) : (d += 8) {
        const qv: V8 = q[d..][0..8].*;
        const kv: V8 = k[d..][0..8].*;
        acc = @mulAdd(V8, qv, kv, acc);
    }
    var sum = @reduce(.Add, acc);
    while (d < hd) : (d += 1) sum += q[d] * k[d];
    return sum;
}

inline fn mulAccumF32(out: [*]f32, weight: f32, v: [*]const f32, hd: usize) void {
    const wv: V8 = @splat(weight);
    var d: usize = 0;
    while (d + 8 <= hd) : (d += 8) {
        var ov: V8 = out[d..][0..8].*;
        const vv: V8 = v[d..][0..8].*;
        ov = @mulAdd(V8, wv, vv, ov);
        out[d..][0..8].* = ov;
    }
    while (d < hd) : (d += 1) out[d] += weight * v[d];
}

inline fn softmax(data: []f32) void {
    if (data.len == 0) return;
    var max_val: f32 = data[0];
    for (data[1..]) |v| if (v > max_val) {
        max_val = v;
    };
    var sum: f32 = 0;
    for (data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    const inv = 1.0 / sum;
    for (data) |*v| v.* *= inv;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "sdpaTree linear chain f32" {
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;

    const q = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    const tree_k = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    const tree_v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    var masks: [3][8]u64 = undefined;
    @memset(&masks[0], 0);
    @memset(&masks[1], 0);
    @memset(&masks[2], 0);
    masks[0][0] = 0b001;
    masks[1][0] = 0b011;
    masks[2][0] = 0b111;

    var output: [3 * hd]f32 = undefined;
    const empty: [0]u8 = .{};

    sdpaTree(&q, &empty, &empty, &tree_k, &tree_v, &output, &masks, nh, nkv, hd, 0, 3, 1.0, .f32, .f32);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.01);

    const expected_2_0 = 0.2119 * 1 + 0.2119 * 5 + 0.5761 * 9;
    try std.testing.expectApproxEqAbs(expected_2_0, output[2 * hd], 0.05);
}

test "sdpaTree with f32 prefix" {
    const nh: usize = 1;
    const nkv: usize = 1;
    const hd: usize = 4;

    const prefix_k_f32 = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const prefix_v_f32 = [_]f32{ 1, 1, 1, 1, 2, 2, 2, 2 };
    const prefix_k = std.mem.sliceAsBytes(&prefix_k_f32);
    const prefix_v = std.mem.sliceAsBytes(&prefix_v_f32);
    const q = [_]f32{ 1, 0, 0, 0 };
    const tree_k = [_]f32{ 0, 0, 1, 0 };
    const tree_v = [_]f32{ 3, 3, 3, 3 };

    var masks: [1][8]u64 = undefined;
    @memset(&masks[0], 0);
    masks[0][0] = 0b1;

    var output: [hd]f32 = undefined;
    sdpaTree(&q, prefix_k.ptr, prefix_v.ptr, &tree_k, &tree_v, &output, &masks, nh, nkv, hd, 2, 1, 1.0, .f32, .f32);

    const expected = 0.5761 * 1.0 + 0.2119 * 2.0 + 0.2119 * 3.0;
    try std.testing.expectApproxEqAbs(expected, output[0], 0.05);
}

test "sdpaTree 2-node chain with GQA" {
    const nh: usize = 2;
    const nkv: usize = 1;
    const hd: usize = 4;

    const prefix_k_f32 = [_]f32{ 1, 0, 0, 0 };
    const prefix_v_f32 = [_]f32{ 10, 20, 30, 40 };
    const prefix_k = std.mem.sliceAsBytes(&prefix_k_f32);
    const prefix_v = std.mem.sliceAsBytes(&prefix_v_f32);

    const q = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
    const tree_k = [_]f32{ 0, 0, 0, 1, 0, 1, 0, 0 };
    const tree_v = [_]f32{ 1, 1, 1, 1, 5, 5, 5, 5 };

    var masks: [2][8]u64 = undefined;
    @memset(&masks[0], 0);
    @memset(&masks[1], 0);
    masks[0][0] = 0b01;
    masks[1][0] = 0b11;

    var output: [2 * nh * hd]f32 = undefined;
    sdpaTree(&q, prefix_k.ptr, prefix_v.ptr, &tree_k, &tree_v, &output, &masks, nh, nkv, hd, 1, 2, 1.0, .f32, .f32);

    const exp_n0h0_0 = 0.731 * 10 + 0.269 * 1;
    try std.testing.expectApproxEqAbs(exp_n0h0_0, output[0], 0.1);

    const exp_n1h0_0 = (10.0 + 1.0 + 5.0) / 3.0;
    try std.testing.expectApproxEqAbs(exp_n1h0_0, output[1 * nh * hd + 0 * hd], 0.1);

    try std.testing.expect(output[0] != output[1 * nh * hd]);
}
