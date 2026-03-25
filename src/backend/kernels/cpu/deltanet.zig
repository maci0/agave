//! CPU DeltaNet SSM recurrence kernel.

const math_ops = @import("../../../ops/math.zig");
const ssm_ops = @import("../../../ops/ssm.zig");
const DeltaNetParams = @import("../../backend.zig").DeltaNetParams;

const V8 = @Vector(8, f32);

/// Maximum number of SSM v-heads supported by DeltaNet stack buffers.
const max_deltanet_v_heads: usize = 128;

/// DeltaNet SSM recurrence: gate/beta → conv1d+SiLU → L2 norm Q&K → recurrence → gated output.
/// Operates on a single token: conv_state and ssm_state are updated in-place.
/// CPU implementation — runs inline with SIMD. No GPU sync needed.
pub fn deltaNet(conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: DeltaNetParams) void {
    const num_v_heads: usize = p.num_v_heads;
    const num_k_heads: usize = p.num_k_heads;
    const head_k_dim: usize = p.head_k_dim;
    const conv_ch: usize = p.conv_ch;

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
                acc += v * v;
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

    // 4. Recurrence + gated output — sequential across v-heads
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
    const kh = if (num_k_heads == num_v_heads) h else if (p.kqv_order) h * num_k_heads / num_v_heads else h % num_k_heads;
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
