//! DeltaNet SSM recurrence kernel for ROCm.
//! One workgroup per v-head (single-threaded — recurrence is sequential).

const cu = @import("common.zig");
const math = @import("std").math;

export fn deltanet_recurrence_kernel(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    gate_vals: [*]const f32,
    beta_vals: [*]const f32,
    z_buf: [*]const f32,
    ssm_norm_w: [*]const f32,
    ssm_state: [*]f32,
    output: [*]f32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    rms_eps: f32,
) callconv(.kernel) void {
    const h = cu.blockIdx();
    if (h >= num_v_heads) return;

    const decay = @exp(gate_vals[h]);
    const beta_h = beta_vals[h];
    const kh = if (num_k_heads == num_v_heads) h else h * num_k_heads / num_v_heads;
    const s_off = h * head_v_dim * head_k_dim;
    const k_base = kh * head_k_dim;

    var kq: f32 = 0.0;
    for (0..head_k_dim) |ki| kq += k_ptr[k_base + ki] * q_ptr[k_base + ki];

    for (0..head_v_dim) |vi| {
        const row_off = s_off + vi * head_k_dim;
        var sk: f32 = 0.0;
        var sq_dec: f32 = 0.0;
        for (0..head_k_dim) |ki| {
            const s_dec = ssm_state[row_off + ki] * decay;
            ssm_state[row_off + ki] = s_dec;
            sk += s_dec * k_ptr[k_base + ki];
            sq_dec += s_dec * q_ptr[k_base + ki];
        }
        const delta = beta_h * (v_ptr[h * head_v_dim + vi] - sk);
        output[h * head_v_dim + vi] = (sq_dec + delta * kq) * q_scale;
        for (0..head_k_dim) |ki| {
            ssm_state[row_off + ki] += k_ptr[k_base + ki] * delta;
        }
    }

    // Gated output: RMSNorm + SiLU
    const off = h * head_v_dim;
    var ss: f32 = 0.0;
    for (0..head_v_dim) |vi| ss += output[off + vi] * output[off + vi];
    const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(head_v_dim)) + rms_eps);
    for (0..head_v_dim) |vi| {
        const normed = output[off + vi] * ssm_norm_w[vi] * inv_rms;
        const z = z_buf[off + vi];
        const silu_z = z / (1.0 + @exp(-z));
        output[off + vi] = normed * silu_z;
    }
}
