//! CPU fused causal attention for prefill.

const std = @import("std");
const kv_quant = @import("../../../ops/kv_quant.zig");
const KvQuantType = kv_quant.KvQuantType;

const V8 = @Vector(8, f32);
const v8zero: V8 = @splat(0.0);
const max_sdpa_seq_len: usize = 8192;
const max_head_dim: usize = 256;

pub fn sdpaPrefill(q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type: KvQuantType) void {
    const kvd = nkv * hd;
    // Append all n_tok keys/values to KV cache
    for (0..n_tok) |t| {
        const src_off = t * kvd;
        const dst_elem = (prev_len + t) * kvd;
        const dst_byte = kv_quant.kvByteOffset(kv_type, dst_elem);
        kv_quant.kvStore(kv_keys.ptr + dst_byte, k + src_off, kvd, kv_type);
        kv_quant.kvStore(kv_values.ptr + dst_byte, v + src_off, kvd, kv_type);
    }
    if (kv_type == .f32) {
        const f32_keys: [*]const f32 = @ptrCast(@alignCast(kv_keys.ptr));
        const f32_values: [*]const f32 = @ptrCast(@alignCast(kv_values.ptr));
        prefillF32(q, f32_keys, f32_values, output, nh, nkv, hd, prev_len, n_tok, scale);
    } else {
        prefillQuant(q, kv_keys.ptr, kv_values.ptr, output, nh, nkv, hd, prev_len, n_tok, scale, kv_type);
    }
}

fn prefillF32(q: [*]const f32, keys: [*]const f32, values: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    for (0..n_tok) |t| {
        const sl = prev_len + t + 1;
        std.debug.assert(sl <= max_sdpa_seq_len);
        for (0..nh) |h| {
            const kvh = h / hpg;
            const q_off = t * nh * hd + h * hd;
            const out_off = t * nh * hd + h * hd;
            var q_cached: [max_head_dim]f32 = undefined;
            @memcpy(q_cached[0..hd], q[q_off..][0..hd]);
            var scores_buf: [max_sdpa_seq_len]f32 = undefined;
            for (0..sl) |s| {
                const k_base = s * kvd + kvh * hd;
                var acc: V8 = v8zero;
                var d: usize = 0;
                while (d + 8 <= hd) : (d += 8) {
                    const qv: V8 = q_cached[d..][0..8].*;
                    const kv: V8 = keys[k_base + d ..][0..8].*;
                    acc = @mulAdd(V8, qv, kv, acc);
                }
                var dot = @reduce(.Add, acc);
                while (d < hd) : (d += 1) dot += q_cached[d] * keys[k_base + d];
                scores_buf[s] = dot * scale;
            }
            softmax(scores_buf[0..sl]);
            var d: usize = 0;
            while (d + 8 <= hd) : (d += 8) {
                var sum: V8 = v8zero;
                for (0..sl) |s| {
                    const sv: V8 = @splat(scores_buf[s]);
                    const vv: V8 = values[s * kvd + kvh * hd + d ..][0..8].*;
                    sum += sv * vv;
                }
                output[out_off + d ..][0..8].* = sum;
            }
            while (d < hd) : (d += 1) {
                var sum: f32 = 0.0;
                for (0..sl) |s| sum += scores_buf[s] * values[s * kvd + kvh * hd + d];
                output[out_off + d] = sum;
            }
        }
    }
}

fn prefillQuant(q: [*]const f32, keys: [*]const u8, values: [*]const u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type: KvQuantType) void {
    const kvd = nkv * hd;
    const hpg = nh / nkv;
    for (0..n_tok) |t| {
        const sl = prev_len + t + 1;
        std.debug.assert(sl <= max_sdpa_seq_len);
        for (0..nh) |h| {
            const kvh = h / hpg;
            const q_off = t * nh * hd + h * hd;
            const out_off = t * nh * hd + h * hd;
            var q_cached: [max_head_dim]f32 = undefined;
            @memcpy(q_cached[0..hd], q[q_off..][0..hd]);
            var scores_buf: [max_sdpa_seq_len]f32 = undefined;
            for (0..sl) |s| {
                const k_byte_off = kv_quant.kvByteOffset(kv_type, s * kvd + kvh * hd);
                scores_buf[s] = kv_quant.kvDot(q_cached[0..hd].ptr, keys + k_byte_off, hd, kv_type) * scale;
            }
            softmax(scores_buf[0..sl]);
            @memset(output[out_off..][0..hd], 0);
            for (0..sl) |s| {
                const v_byte_off = kv_quant.kvByteOffset(kv_type, s * kvd + kvh * hd);
                kv_quant.kvMulAccum(output + out_off, scores_buf[s], values + v_byte_off, hd, kv_type);
            }
        }
    }
}

fn softmax(scores: []f32) void {
    const n = scores.len;
    if (n == 0) return;
    var max_val: f32 = scores[0];
    for (scores[1..]) |s| if (s > max_val) {
        max_val = s;
    };
    var sum: f32 = 0.0;
    for (scores) |*s| {
        s.* = @exp(s.* - max_val);
        sum += s.*;
    }
    const inv = 1.0 / sum;
    for (scores) |*s| s.* *= inv;
}

test "sdpaPrefill single token matches decode sdpa" {
    const sdpa_decode = @import("sdpa.zig");
    const hd = 4;
    var q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var v = [_]f32{ 0.5, 0.6, 0.7, 0.8 };
    var kv_keys_pf = [_]f32{0} ** 256;
    var kv_values_pf = [_]f32{0} ** 256;
    var kv_keys_dec = [_]f32{0} ** 256;
    var kv_values_dec = [_]f32{0} ** 256;
    var out_pf: [4]f32 = undefined;
    var out_dec: [4]f32 = undefined;
    sdpaPrefill(&q, &k, &v, std.mem.sliceAsBytes(&kv_keys_pf), std.mem.sliceAsBytes(&kv_values_pf), &out_pf, 1, 1, hd, 0, 1, 1.0, .f32);
    sdpa_decode.sdpa(&q, &kv_keys_dec, &kv_values_dec, &k, &v, &out_dec, 1, 1, hd, 0, 1.0);
    for (0..hd) |i| try std.testing.expectApproxEqAbs(out_dec[i], out_pf[i], 1e-5);
}

test "sdpaPrefill 3 tokens causal" {
    const hd = 4;
    var q = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0 };
    var k = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 };
    var v = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    var kv_keys = [_]f32{0} ** 256;
    var kv_values = [_]f32{0} ** 256;
    var output: [12]f32 = undefined;
    sdpaPrefill(&q, &k, &v, std.mem.sliceAsBytes(&kv_keys), std.mem.sliceAsBytes(&kv_values), &output, 1, 1, hd, 0, 3, 1.0, .f32);
    // Token 0: attends only to itself -> output = v0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 1e-4);
    // Token 1: q1.k0=0, q1.k1=1 -> softmax([0,1]) -> weighted v0+v1
    const e = @exp(@as(f32, 1.0));
    const w1 = e / (1.0 + e);
    try std.testing.expectApproxEqAbs(w1, output[4 + 1], 1e-4);
}
