//! DeltaNet SSM kernels for ROCm.
//! Gate/beta and conv1d kernels are loaded by rocm.zig, but the top-level
//! deltaNet dispatch still @panics — full recurrence kernel not yet ported.

const cu = @import("common.zig");

/// Softplus stability threshold: for x > this value, softplus(x) ≈ x.
const softplus_threshold: f32 = 20.0;

/// Gate/beta computation kernel (loaded by rocm.zig, used by sub-ops).
/// gate[h] = ssm_a[h] * softplus(alpha[h] + dt_bias[h])
/// beta[h] = sigmoid(beta_in[h])
export fn deltanet_gate_beta_kernel(
    alpha: [*]const f32,
    beta_in: [*]const f32,
    ssm_a: [*]const f32,
    dt_bias: [*]const f32,
    gate_out: [*]f32,
    beta_out: [*]f32,
    n_heads: u32,
) callconv(.kernel) void {
    const idx = cu.globalIdx();
    if (idx >= n_heads) return;

    const x = alpha[idx] + dt_bias[idx];
    // Softplus: log(1 + exp(x)), clamped for stability
    const sp = if (x > softplus_threshold) x else @log(1.0 + cu.expf(x));
    gate_out[idx] = ssm_a[idx] * sp;
    beta_out[idx] = cu.sigmoidf(beta_in[idx]);
}

/// Conv1d + SiLU kernel.
/// Each thread handles one channel.
export fn deltanet_conv1d_kernel(
    conv_in: [*]const f32,
    conv_state: [*]f32,
    conv_w: [*]const f32,
    conv_out: [*]f32,
    conv_ch: u32,
    d_conv: u32,
) callconv(.kernel) void {
    const ch = cu.globalIdx();
    if (ch >= conv_ch) return;

    const hist = d_conv - 1;
    var sum: f32 = 0.0;

    // Convolve over history
    for (0..hist) |k| {
        sum += conv_state[k * conv_ch + ch] * conv_w[ch * d_conv + k];
    }
    sum += conv_in[ch] * conv_w[ch * d_conv + hist];

    // SiLU activation
    conv_out[ch] = sum * cu.sigmoidf(sum);

    // Shift ring buffer left, append current input
    for (0..hist - 1) |p| {
        conv_state[p * conv_ch + ch] = conv_state[(p + 1) * conv_ch + ch];
    }
    conv_state[(hist - 1) * conv_ch + ch] = conv_in[ch];
}
