//! CPU DeltaNet SSM recurrence kernel.

const std = @import("std");
const math_ops = @import("../../../ops/math.zig");
const ssm_ops = @import("../../../ops/ssm.zig");
const DeltaNetParams = @import("../../backend.zig").DeltaNetParams;

const V8 = @Vector(8, f32);

/// Maximum number of SSM v-heads supported by DeltaNet stack buffers.
const max_deltanet_v_heads: usize = 128;

/// Map a V-head index onto its K/Q head using HuggingFace/llama.cpp repeat_interleave.
/// Qwen3.8-27B: 48 V heads / 16 K heads → groups of 3 (h=0,1,2 → k=0).
pub fn groupedKHead(h: usize, num_k_heads: usize, num_v_heads: usize) usize {
    if (num_k_heads == 0 or num_v_heads == 0) return 0;
    return h * num_k_heads / num_v_heads;
}

/// DeltaNet SSM recurrence: gate/beta → conv1d+SiLU → L2 norm Q&K → recurrence → gated output.
/// Operates on a single token: conv_state and ssm_state are updated in-place.
/// CPU implementation, runs inline with SIMD. No GPU sync needed.
pub fn deltaNet(conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: DeltaNetParams) void {
    const num_v_heads: usize = p.num_v_heads;
    const num_k_heads: usize = p.num_k_heads;
    const head_k_dim: usize = p.head_k_dim;
    const conv_ch: usize = p.conv_ch;
    if (num_v_heads > max_deltanet_v_heads) @panic("deltanet: num_v_heads exceeds max_deltanet_v_heads");

    // 1. Gate & beta computation
    var gate_vals: [max_deltanet_v_heads]f32 = undefined;
    var beta_vals: [max_deltanet_v_heads]f32 = undefined;
    for (0..num_v_heads) |h| {
        const alpha_biased = alpha_buf[h] + dt_bias[h];
        gate_vals[h] = ssm_a[h] * math_ops.softplus(alpha_biased);
        beta_vals[h] = math_ops.sigmoid(beta_buf[h]);
    }

    // 2. Conv1d + SiLU
    ssm_ops.causalConv1dSilu(conv_out, conv_state, conv_in, conv_w, null, conv_ch, p.d_conv);

    // 3. L2 normalize Q and K per head
    // Conv output layout: [Q (n_qk) | K (n_qk) | V (d_inner)]
    // where n_qk = num_k_heads * head_k_dim
    const q_off: usize = 0;
    const k_off: usize = num_k_heads * head_k_dim;
    for (0..num_k_heads) |h| {
        inline for ([_]usize{ q_off, k_off }) |base_off| {
            const ptr = conv_out + base_off + h * head_k_dim;
            var acc: V8 = @splat(0.0);
            var li2: usize = 0;
            while (li2 + 8 <= head_k_dim) : (li2 += 8) {
                const v: V8 = ptr[li2..][0..8].*;
                acc = @mulAdd(V8, v, v, acc);
            }
            var ss = @reduce(.Add, acc);
            while (li2 < head_k_dim) : (li2 += 1) ss += ptr[li2] * ptr[li2];
            const inv = 1.0 / @sqrt(ss + p.rms_eps);
            const inv_v: V8 = @splat(inv);
            li2 = 0;
            while (li2 + 8 <= head_k_dim) : (li2 += 8) {
                ptr[li2..][0..8].* = @as(V8, ptr[li2..][0..8].*) * inv_v;
            }
            while (li2 < head_k_dim) : (li2 += 1) ptr[li2] *= inv;
        }
    }

    // 4. Recurrence + gated output, sequential across v-heads
    const q_ptr = conv_out + q_off;
    const k_ptr = conv_out + k_off;
    const v_off: usize = 2 * num_k_heads * head_k_dim;
    const v_ptr = conv_out + v_off;

    for (0..num_v_heads) |h| {
        deltaNetHead(h, &gate_vals, &beta_vals, q_ptr, k_ptr, v_ptr, output, ssm_state.ptr, z_buf, ssm_norm_w, p);
    }
}

/// Process a single DeltaNet v-head: recurrence + gated output.
/// Public to enable parallel dispatch across heads from the backend.
pub fn deltaNetHead(h: usize, gate_vals: *const [max_deltanet_v_heads]f32, beta_vals_arr: *const [max_deltanet_v_heads]f32, q_ptr: [*]const f32, k_ptr: [*]const f32, v_ptr: [*]const f32, output: [*]f32, ssm_state: [*]f32, z_buf: [*]const f32, ssm_norm_w: [*]const f32, p: DeltaNetParams) void {
    const head_v_dim: usize = p.head_v_dim;
    const head_k_dim: usize = p.head_k_dim;
    const num_k_heads: usize = p.num_k_heads;
    const num_v_heads: usize = p.num_v_heads;
    const decay = @exp(gate_vals[h]);
    const beta_h = beta_vals_arr[h];
    const kh = groupedKHead(h, num_k_heads, num_v_heads);
    const s_off = h * head_v_dim * head_k_dim;
    const k_base = kh * head_k_dim;
    const decay_v: V8 = @splat(decay);

    // Precompute dot(K, Q)
    var kq_acc: V8 = @splat(0.0);
    var ki: usize = 0;
    while (ki + 8 <= head_k_dim) : (ki += 8) {
        kq_acc += @as(V8, k_ptr[k_base + ki ..][0..8].*) *
            @as(V8, q_ptr[k_base + ki ..][0..8].*);
    }
    var kq = @reduce(.Add, kq_acc);
    while (ki < head_k_dim) : (ki += 1) kq += k_ptr[k_base + ki] * q_ptr[k_base + ki];

    for (0..head_v_dim) |vi| {
        const row_off = s_off + vi * head_k_dim;
        var acc_k: V8 = @splat(0.0);
        var acc_q: V8 = @splat(0.0);
        ki = 0;
        while (ki + 8 <= head_k_dim) : (ki += 8) {
            const s_old: V8 = ssm_state[row_off + ki ..][0..8].*;
            const s_dec = s_old * decay_v;
            ssm_state[row_off + ki ..][0..8].* = s_dec;
            const k_v: V8 = k_ptr[k_base + ki ..][0..8].*;
            acc_k += s_dec * k_v;
            acc_q += s_dec * @as(V8, q_ptr[k_base + ki ..][0..8].*);
        }
        var sk = @reduce(.Add, acc_k);
        var sq_dec = @reduce(.Add, acc_q);
        while (ki < head_k_dim) : (ki += 1) {
            ssm_state[row_off + ki] *= decay;
            sk += ssm_state[row_off + ki] * k_ptr[k_base + ki];
            sq_dec += ssm_state[row_off + ki] * q_ptr[k_base + ki];
        }
        const delta = beta_h * (v_ptr[h * head_v_dim + vi] - sk);
        output[h * head_v_dim + vi] = (sq_dec + delta * kq) * p.q_scale;
        const delta_v: V8 = @splat(delta);
        ki = 0;
        while (ki + 8 <= head_k_dim) : (ki += 8) {
            const s_dec: V8 = ssm_state[row_off + ki ..][0..8].*;
            const k_v: V8 = k_ptr[k_base + ki ..][0..8].*;
            ssm_state[row_off + ki ..][0..8].* = @mulAdd(V8, k_v, delta_v, s_dec);
        }
        while (ki < head_k_dim) : (ki += 1) {
            ssm_state[row_off + ki] += k_ptr[k_base + ki] * delta;
        }
    }

    // Gated output: RMSNorm + SiLU
    const off = h * head_v_dim;
    var acc_sq: V8 = @splat(0.0);
    var vi: usize = 0;
    while (vi + 8 <= head_v_dim) : (vi += 8) {
        const v: V8 = output[off + vi ..][0..8].*;
        acc_sq += v * v;
    }
    var ss = @reduce(.Add, acc_sq);
    while (vi < head_v_dim) : (vi += 1) ss += output[off + vi] * output[off + vi];
    const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(head_v_dim)) + p.rms_eps);
    const inv_v_r: V8 = @splat(inv_rms);
    const one_v: V8 = @splat(1.0);
    const neg_v: V8 = @splat(-1.0);
    vi = 0;
    while (vi + 8 <= head_v_dim) : (vi += 8) {
        const o: V8 = output[off + vi ..][0..8].*;
        const w: V8 = ssm_norm_w[vi..][0..8].*;
        const normed = o * inv_v_r * w;
        const z: V8 = z_buf[off + vi ..][0..8].*;
        const silu_z = z / (one_v + @exp(neg_v * z));
        output[off + vi ..][0..8].* = normed * silu_z;
    }
    while (vi < head_v_dim) : (vi += 1) {
        const normed = output[off + vi] * inv_rms * ssm_norm_w[vi];
        const z = z_buf[off + vi];
        output[off + vi] = normed * (z / (1.0 + @exp(-z)));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "groupedKHead repeat_interleave not modulo" {
    // 4 V heads, 2 K heads: groups of 2, not h%2.
    try testing.expectEqual(@as(usize, 0), groupedKHead(0, 2, 4));
    try testing.expectEqual(@as(usize, 0), groupedKHead(1, 2, 4));
    try testing.expectEqual(@as(usize, 1), groupedKHead(2, 2, 4));
    try testing.expectEqual(@as(usize, 1), groupedKHead(3, 2, 4));
    // Qwen3.8-27B: 48 V / 16 K → groups of 3.
    try testing.expectEqual(@as(usize, 0), groupedKHead(0, 16, 48));
    try testing.expectEqual(@as(usize, 0), groupedKHead(2, 16, 48));
    try testing.expectEqual(@as(usize, 1), groupedKHead(3, 16, 48));
    try testing.expectEqual(@as(usize, 15), groupedKHead(47, 16, 48));
    try testing.expectEqual(@as(usize, 0), groupedKHead(0, 0, 48));
}

/// Build a minimal DeltaNetParams for testing with the given dimensions.
fn testParams(num_heads: u32, head_k: u32, head_v: u32) DeltaNetParams {
    return .{
        .conv_ch = 2 * num_heads * head_k + num_heads * head_v,
        .d_conv = 4,
        .d_inner = num_heads * head_v,
        .num_k_heads = num_heads,
        .head_k_dim = head_k,
        .num_v_heads = num_heads,
        .head_v_dim = head_v,
        .q_scale = 1.0,
        .rms_eps = 1e-6,
    };
}

test "deltaNetHead zero state, zero input produces zero output" {
    // With zero Q, K, V, and zero SSM state, output before gating should be zero.
    // The gating applies RMSNorm (0/eps → ~0) * SiLU(z). With z=1 the SiLU is nonzero
    // but normed is ~0, so final output should be ~0.
    const num_heads: u32 = 1;
    const head_k: u32 = 8;
    const head_v: u32 = 8;
    const p = testParams(num_heads, head_k, head_v);

    var q = [_]f32{0.0} ** head_k;
    var k = [_]f32{0.0} ** head_k;
    var v = [_]f32{0.0} ** head_v;
    var output = [_]f32{999.0} ** head_v;
    var ssm_state = [_]f32{0.0} ** (head_v * head_k);
    var z_buf = [_]f32{1.0} ** head_v; // nonzero gate to avoid 0*0 ambiguity
    var norm_w = [_]f32{1.0} ** head_v;
    var gate_vals: [max_deltanet_v_heads]f32 = undefined;
    var beta_vals: [max_deltanet_v_heads]f32 = undefined;
    gate_vals[0] = 0.0; // decay = exp(0) = 1
    beta_vals[0] = 0.5;

    deltaNetHead(0, &gate_vals, &beta_vals, &q, &k, &v, &output, &ssm_state, &z_buf, &norm_w, p);

    for (0..head_v) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), output[i], 0.01);
    }
}

test "deltaNetHead identity-like recurrence stores value in state" {
    // Set up: Q=K=unit vector along dim 0 (normalized), V=[1,0,...], beta=1, decay=1.
    // With zero initial state: sk=0, delta = beta*(v-0) = v, output = (0 + delta*kq) * q_scale.
    // kq = dot(K,Q) = 1.0, so output[0] = 1.0 * 1.0 = 1.0 before gating.
    // After gating with z=large (SiLU≈z) and norm_w=1: output ≈ normed * z.
    const num_heads: u32 = 1;
    const head_k: u32 = 8;
    const head_v: u32 = 8;
    const p = testParams(num_heads, head_k, head_v);

    // Q and K are unit vectors along dim 0
    var q = [_]f32{0.0} ** head_k;
    var k = [_]f32{0.0} ** head_k;
    q[0] = 1.0;
    k[0] = 1.0;

    var v = [_]f32{0.0} ** head_v;
    v[0] = 1.0;

    var output = [_]f32{0.0} ** head_v;
    var ssm_state = [_]f32{0.0} ** (head_v * head_k);
    // Use z=10 so SiLU(z)≈z and gating doesn't squash much.
    var z_buf = [_]f32{10.0} ** head_v;
    var norm_w = [_]f32{1.0} ** head_v;
    var gate_vals: [max_deltanet_v_heads]f32 = undefined;
    var beta_vals: [max_deltanet_v_heads]f32 = undefined;
    gate_vals[0] = 0.0; // decay = exp(0) = 1
    beta_vals[0] = 1.0; // full beta

    deltaNetHead(0, &gate_vals, &beta_vals, &q, &k, &v, &output, &ssm_state, &z_buf, &norm_w, p);

    // After recurrence: output[0] before gating = (0 + 1.0*1.0) * 1.0 = 1.0.
    // RMSNorm: rms = sqrt(1.0/8 + eps) ≈ 0.3536 → inv ≈ 2.828. normed[0] = 1.0*2.828*1.0 = 2.828.
    // SiLU(10) ≈ 10.0 (saturated). Final ≈ 2.828 * 10.0 = 28.28.
    // Other dims: normed = 0, so output stays 0.
    try testing.expect(output[0] > 1.0); // gated output is positive and amplified
    for (1..head_v) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), output[i], 0.01);
    }

    // Verify state was updated: ssm_state[0*head_k + 0] should be delta*k[0] = 1.0*1.0 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssm_state[0], 0.01);
}

test "deltaNetHead decay shrinks state" {
    // Pre-load state, run with zero V to observe decay effect.
    // gate_vals[0] = -1 → decay = exp(-1) ≈ 0.368.
    const num_heads: u32 = 1;
    const head_k: u32 = 8;
    const head_v: u32 = 8;
    const p = testParams(num_heads, head_k, head_v);

    var q = [_]f32{0.0} ** head_k;
    var k = [_]f32{0.0} ** head_k;
    q[0] = 1.0;
    k[0] = 1.0;

    var v = [_]f32{0.0} ** head_v;
    var output = [_]f32{0.0} ** head_v;

    // Pre-load state: row 0 of state has 5.0 at position 0
    var ssm_state = [_]f32{0.0} ** (head_v * head_k);
    ssm_state[0] = 5.0;

    var z_buf = [_]f32{10.0} ** head_v;
    var norm_w = [_]f32{1.0} ** head_v;
    var gate_vals: [max_deltanet_v_heads]f32 = undefined;
    var beta_vals: [max_deltanet_v_heads]f32 = undefined;
    gate_vals[0] = -1.0; // decay = exp(-1) ≈ 0.368
    beta_vals[0] = 1.0;

    deltaNetHead(0, &gate_vals, &beta_vals, &q, &k, &v, &output, &ssm_state, &z_buf, &norm_w, p);

    // After decay and delta update:
    // s_dec = 5.0 * 0.368 = 1.839
    // sk = s_dec * k[0] = 1.839 (state dot k for vi=0)
    // delta = beta * (v[0] - sk) = 1.0 * (0 - 1.839) = -1.839
    // new state[0] = s_dec + k[0]*delta = 1.839 + 1.0*(-1.839) = 0.0
    // But the decay should have reduced the original 5.0 → decayed value
    const decay = @exp(@as(f32, -1.0));
    // State should be less than original
    try testing.expect(@abs(ssm_state[0]) < 5.0 * decay + 0.01);
}

test "fuzz: all deltanet functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Fixed dimensions (must be multiples of 8 for V8 SIMD).
            const num_heads: u32 = 1;
            const head_k: u32 = 8;
            const head_v: u32 = 8;
            const d_conv: u32 = 4;
            const n_qk = num_heads * head_k;
            const conv_ch = 2 * n_qk + num_heads * head_v;

            const p = DeltaNetParams{
                .conv_ch = conv_ch,
                .d_conv = d_conv,
                .d_inner = num_heads * head_v,
                .num_k_heads = num_heads,
                .head_k_dim = head_k,
                .num_v_heads = num_heads,
                .head_v_dim = head_v,
                .q_scale = 1.0,
                .rms_eps = 1e-6,
                .kqv_order = smith.valueWithHash(bool, 0),
            };

            // Random buffers clamped to [-1,1] to avoid NaN/Inf from exp().
            var conv_in: [conv_ch]f32 = undefined;
            var conv_out_buf: [conv_ch]f32 = undefined;
            var z_buf: [num_heads * head_v]f32 = undefined;
            var alpha_buf: [num_heads]f32 = undefined;
            var beta_buf: [num_heads]f32 = undefined;
            var output: [num_heads * head_v]f32 = undefined;
            var conv_state: [conv_ch * (d_conv - 1)]f32 = undefined;
            var ssm_state_arr: [num_heads * head_v * head_k]f32 = undefined;
            var ssm_a: [num_heads]f32 = undefined;
            var dt_bias: [num_heads]f32 = undefined;
            var conv_w: [conv_ch * d_conv]f32 = undefined;
            var ssm_norm_w: [head_v]f32 = undefined;

            inline for (.{
                &conv_in,  &conv_out_buf, &z_buf,      &alpha_buf,
                &beta_buf, &output,       &conv_state, &ssm_state_arr,
                &ssm_a,    &dt_bias,      &conv_w,     &ssm_norm_w,
            }, 0..) |buf, seed| {
                for (buf, 0..) |*slot, j| {
                    const hash: u32 = @intCast(seed *% 31 +% j);
                    const raw = smith.valueWithHash(i8, hash);
                    slot.* = @as(f32, @floatFromInt(raw)) / 128.0; // [-1,1]
                }
            }

            const ssm_state: []f32 = &ssm_state_arr;

            // --- Exercise deltaNet (full pipeline) ---
            deltaNet(
                &conv_in,
                &conv_out_buf,
                &z_buf,
                &alpha_buf,
                &beta_buf,
                &output,
                &conv_state,
                ssm_state,
                &ssm_a,
                &dt_bias,
                &conv_w,
                &ssm_norm_w,
                p,
            );
            for (0..num_heads * head_v) |i| {
                if (std.math.isNan(output[i])) return;
            }

            // --- Exercise deltaNetHead (single head) ---
            var gate_vals: [max_deltanet_v_heads]f32 = undefined;
            var beta_vals: [max_deltanet_v_heads]f32 = undefined;
            const g_raw = smith.valueWithHash(i8, 99);
            gate_vals[0] = @as(f32, @floatFromInt(g_raw)) / 128.0;
            const b_raw = smith.valueWithHash(i8, 100);
            beta_vals[0] = @as(f32, @floatFromInt(b_raw)) / 128.0;

            var q2: [head_k]f32 = undefined;
            var k2: [head_k]f32 = undefined;
            var v2: [head_v]f32 = undefined;
            var out2: [head_v]f32 = undefined;
            var state2: [head_v * head_k]f32 = undefined;
            var z2: [head_v]f32 = undefined;
            var nw2: [head_v]f32 = undefined;

            inline for (.{ &q2, &k2, &v2, &out2, &state2, &z2, &nw2 }, 0..) |buf, seed| {
                for (buf, 0..) |*slot, j| {
                    const hash2: u32 = @intCast(200 +% seed *% 17 +% j);
                    const raw = smith.valueWithHash(i8, hash2);
                    slot.* = @as(f32, @floatFromInt(raw)) / 128.0;
                }
            }

            deltaNetHead(
                0,
                &gate_vals,
                &beta_vals,
                &q2,
                &k2,
                &v2,
                &out2,
                &state2,
                &z2,
                &nw2,
                p,
            );
            for (0..head_v) |i| {
                if (std.math.isNan(out2[i])) return;
            }
        }
    }.f, .{});
}

test "deltaNet full pipeline runs without crash" {
    // Smoke test: run the full deltaNet function with minimal dimensions.
    const num_heads: u32 = 1;
    const head_k: u32 = 8;
    const head_v: u32 = 8;
    const d_conv: u32 = 4;
    const n_qk = num_heads * head_k;
    const conv_ch = 2 * n_qk + num_heads * head_v;

    const p = DeltaNetParams{
        .conv_ch = conv_ch,
        .d_conv = d_conv,
        .d_inner = num_heads * head_v,
        .num_k_heads = num_heads,
        .head_k_dim = head_k,
        .num_v_heads = num_heads,
        .head_v_dim = head_v,
        .q_scale = 1.0,
        .rms_eps = 1e-6,
    };

    var conv_in = [_]f32{0.1} ** conv_ch;
    var conv_out = [_]f32{0.0} ** conv_ch;
    var z_buf = [_]f32{1.0} ** (num_heads * head_v);
    var alpha_buf = [_]f32{0.0} ** num_heads;
    var beta_buf = [_]f32{0.0} ** num_heads;
    var output = [_]f32{0.0} ** (num_heads * head_v);
    var conv_state = [_]f32{0.0} ** (conv_ch * (d_conv - 1));
    var ssm_state_arr = [_]f32{0.0} ** (num_heads * head_v * head_k);
    const ssm_state: []f32 = &ssm_state_arr;
    var ssm_a = [_]f32{-1.0} ** num_heads;
    var dt_bias = [_]f32{0.0} ** num_heads;
    var conv_w = [_]f32{0.25} ** (conv_ch * d_conv);
    var ssm_norm_w = [_]f32{1.0} ** head_v;

    deltaNet(&conv_in, &conv_out, &z_buf, &alpha_buf, &beta_buf, &output, &conv_state, ssm_state, &ssm_a, &dt_bias, &conv_w, &ssm_norm_w, p);

    // Verify output is finite (not NaN or Inf)
    for (0..num_heads * head_v) |i| {
        try testing.expect(!std.math.isNan(output[i]));
        try testing.expect(!std.math.isInf(output[i]));
    }
}
