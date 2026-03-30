//! SSM (State Space Model) utility kernels.
//! Provides shared causal conv1d with SiLU activation and ring-buffer state, used by
//! Qwen3.5 (DeltaNet) and Nemotron-H (Mamba-2) architectures.

const std = @import("std");
const math_ops = @import("math.zig");

const silu = math_ops.silu;

/// Causal 1D convolution with SiLU activation and ring-buffer state.
/// Dispatches to a comptime-unrolled inner loop for common d_conv values (2-8),
/// which fully unrolls the history accumulation and ring-buffer shift.
///
/// The ring buffer holds the (d_conv-1) most recent input vectors. On each call,
/// history is shifted forward, the current input is appended, and the output is
/// computed from [history, current_input] via convolution weights + SiLU.
///
/// Parameters:
///   - conv_out: Output buffer [conv_ch]. Receives SiLU(conv(state, input)).
///   - conv_state: Ring buffer [(d_conv-1) * conv_ch], row-major. Updated in-place.
///   - conv_in: Current input vector [conv_ch].
///   - conv_w: Convolution weights [conv_ch * d_conv], row-major
///             (each channel contiguous: conv_w[ch * d_conv + k]).
///   - conv_b: Optional bias vector [conv_ch]. Pass null for no bias.
///   - conv_ch: Number of channels.
///   - d_conv: Convolution kernel width (typically 4).
pub fn causalConv1dSilu(
    conv_out: [*]f32,
    conv_state: [*]f32,
    conv_in: [*]const f32,
    conv_w: [*]const f32,
    conv_b: ?[*]const f32,
    conv_ch: usize,
    d_conv: usize,
) void {
    switch (d_conv) {
        inline 2, 3, 4, 5, 6, 7, 8 => |comptime_d| conv1dImpl(comptime_d, conv_out, conv_state, conv_in, conv_w, conv_b, conv_ch, d_conv),
        else => conv1dImpl(null, conv_out, conv_state, conv_in, conv_w, conv_b, conv_ch, d_conv),
    }
}

/// Unified conv1d implementation. When `D` is non-null (comptime-known d_conv),
/// history loop and ring-buffer shift are fully unrolled. Otherwise uses runtime `d_conv`.
fn conv1dImpl(
    comptime D: ?comptime_int,
    conv_out: [*]f32,
    conv_state: [*]f32,
    conv_in: [*]const f32,
    conv_w: [*]const f32,
    conv_b: ?[*]const f32,
    conv_ch: usize,
    d_conv: usize,
) void {
    const d = D orelse d_conv;
    const hist = d - 1;
    for (0..conv_ch) |ch| {
        var sum: f32 = if (conv_b) |b| b[ch] else 0.0;
        if (D) |_| {
            inline for (0..D.? - 1) |k| {
                sum += conv_state[k * conv_ch + ch] * conv_w[ch * d + k];
            }
        } else {
            for (0..hist) |k| {
                sum += conv_state[k * conv_ch + ch] * conv_w[ch * d + k];
            }
        }
        sum += conv_in[ch] * conv_w[ch * d + hist];
        conv_out[ch] = silu(sum);
    }
    if (D) |_| {
        if (D.? - 1 > 1) {
            inline for (0..D.? - 2) |p| {
                @memcpy(conv_state[p * conv_ch ..][0..conv_ch], conv_state[(p + 1) * conv_ch ..][0..conv_ch]);
            }
        }
    } else {
        if (hist > 1) {
            for (0..hist - 1) |p| {
                @memcpy(conv_state[p * conv_ch ..][0..conv_ch], conv_state[(p + 1) * conv_ch ..][0..conv_ch]);
            }
        }
    }
    @memcpy(conv_state[(hist - 1) * conv_ch ..][0..conv_ch], conv_in[0..conv_ch]);
}

/// Mamba-2 autoregressive recurrence for one SSM layer.
///
/// Performs per-head state update and output computation:
///   state[h][i][j] = decay * state[h][i][j] + (x_h[i] * dt_h) * B_g[j]
///   y_h[i]         = sum_j(state[h][i][j] * C_g[j]) + D[h] * x_h[i]
///
/// Parameters:
///   - y:         Output buffer [num_heads * head_dim].
///   - state:     Persistent SSM state [num_heads * head_dim * d_state], updated in place.
///   - x:         Input vector [num_heads * head_dim] (post-conv1d).
///   - B:         B projection [n_groups * d_state].
///   - C:         C projection [n_groups * d_state].
///   - dt_raw:    Raw dt values [num_heads] (pre-softplus, pre-bias).
///   - dt_bias:   Per-head dt bias [num_heads].
///   - ssm_a:     Per-head A parameter [num_heads] (already negative).
///   - ssm_d:     Per-head D skip-connection scale [num_heads].
///   - num_heads:      Number of SSM heads.
///   - head_dim:       Elements per head in x/y.
///   - d_state:        SSM state dimension (per head per element).
///   - heads_per_group: Heads per B/C group.
pub fn mamba2Recurrence(
    y: [*]f32,
    state: []f32,
    x: [*]const f32,
    B: [*]const f32,
    C: [*]const f32,
    dt_raw: [*]const f32,
    dt_bias: [*]const f32,
    ssm_a: [*]const f32,
    ssm_d: [*]const f32,
    num_heads: usize,
    head_dim: usize,
    d_state: usize,
    heads_per_group: usize,
) void {
    for (0..num_heads) |h| {
        const group = h / heads_per_group;
        const s_off = h * head_dim * d_state;

        const dt_h = math_ops.softplus(dt_raw[h] + dt_bias[h]);
        const decay = @exp(ssm_a[h] * dt_h);

        const x_h = x + h * head_dim;
        const B_g = B + group * d_state;
        const C_g = C + group * d_state;
        const y_h = y + h * head_dim;

        for (0..head_dim) |i| {
            const xd = x_h[i] * dt_h;
            var yi: f32 = ssm_d[h] * x_h[i];
            const s_row = state[s_off + i * d_state ..][0..d_state];

            // SIMD-vectorized state update + output accumulation
            const V8 = @Vector(8, f32);
            const decay_v: V8 = @splat(decay);
            const xd_v: V8 = @splat(xd);
            var yi_acc: V8 = @splat(0.0);
            var j: usize = 0;
            while (j + 8 <= d_state) : (j += 8) {
                const s_v: V8 = s_row[j..][0..8].*;
                const b_v: V8 = B_g[j..][0..8].*;
                const c_v: V8 = C_g[j..][0..8].*;
                const new_s = @mulAdd(V8, xd_v, b_v, decay_v * s_v);
                s_row[j..][0..8].* = new_s;
                yi_acc = @mulAdd(V8, new_s, c_v, yi_acc);
            }
            yi += @reduce(.Add, yi_acc);
            // Scalar tail
            while (j < d_state) : (j += 1) {
                s_row[j] = @mulAdd(f32, xd, B_g[j], decay * s_row[j]);
                yi = @mulAdd(f32, s_row[j], C_g[j], yi);
            }
            y_h[i] = yi;
        }
    }
}

/// SiLU gate then group RMS norm, applied in-place to y.
///
/// Follows the Mamba-2 `norm_before_gate=False` convention:
/// 1. Gate: gated = y * SiLU(z)
/// 2. Group RMS norm: output = rms_norm(gated) * weight
///
/// Parameters:
///   - y:       Input/output buffer [d_inner], modified in place.
///   - z:       Gate values [d_inner] (SiLU applied to each element).
///   - norm_w:  Per-element norm weights [d_inner].
///   - d_inner: Total number of elements.
///   - n_groups: Number of normalization groups.
///   - eps:     RMS norm epsilon.
pub fn groupRmsNormSiluGate(
    y: [*]f32,
    z: [*]const f32,
    norm_w: [*]const f32,
    d_inner: usize,
    n_groups: usize,
    eps: f32,
) void {
    const V8 = @Vector(8, f32);
    std.debug.assert(d_inner % n_groups == 0);
    const elem_per_group: usize = d_inner / n_groups;
    for (0..n_groups) |g| {
        const off = g * elem_per_group;
        const y_g = y + off;
        const z_g = z + off;
        const w_g = norm_w + g * elem_per_group;

        // 1. Apply SiLU gate in-place: y = y * silu(z)
        for (0..elem_per_group) |j| {
            y_g[j] *= silu(z_g[j]);
        }

        // 2. SIMD sum of squares on gated values
        var ss_acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= elem_per_group) : (i += 8) {
            const v: V8 = y_g[i..][0..8].*;
            ss_acc = @mulAdd(V8, v, v, ss_acc);
        }
        var ss: f32 = @reduce(.Add, ss_acc);
        while (i < elem_per_group) : (i += 1) ss += y_g[i] * y_g[i];

        // 3. RMS normalize and apply weight
        const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(elem_per_group)) + eps);
        for (0..elem_per_group) |j| {
            y_g[j] *= inv_rms * w_g[j];
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

test "causalConv1dSilu basic" {
    // d_conv=2, conv_ch=2: simplest case
    var conv_out = [_]f32{ 0, 0 };
    var conv_state = [_]f32{ 1.0, 2.0 }; // 1 history position × 2 channels
    const conv_in = [_]f32{ 3.0, 4.0 };
    // Weights: ch0=[w0,w1], ch1=[w2,w3]
    const conv_w = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, null, 2, 2);

    // ch0: 1.0*0.5 + 3.0*0.5 = 2.0, SiLU(2.0) ≈ 1.7616
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), conv_out[0], 0.001);

    // ch1: 2.0*0.5 + 4.0*0.5 = 3.0, SiLU(3.0) ≈ 2.8577
    try std.testing.expectApproxEqAbs(@as(f32, 2.8577), conv_out[1], 0.001);

    // State should now contain the current input
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), conv_state[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), conv_state[1], 0.01);
}

test "causalConv1dSilu ring buffer evolution" {
    // d_conv=3, conv_ch=1: verify state shifts correctly across 3 calls
    var conv_out = [_]f32{0};
    var conv_state = [_]f32{ 0.0, 0.0 }; // (d_conv-1)=2 history positions × 1 channel
    const conv_w = [_]f32{ 1.0, 1.0, 1.0 }; // all weights 1.0 for easy verification

    // Step 1: input=1.0, state=[0,0] → conv = 0*1 + 0*1 + 1*1 = 1.0
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{1.0}, &conv_w, null, 1, 3);
    const expected1 = silu(1.0);
    try std.testing.expectApproxEqAbs(expected1, conv_out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), conv_state[0], 1e-6); // shifted: was state[1]=0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), conv_state[1], 1e-6); // current input

    // Step 2: input=2.0, state=[0,1] → conv = 0*1 + 1*1 + 2*1 = 3.0
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{2.0}, &conv_w, null, 1, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), conv_state[0], 1e-6); // shifted from [1]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), conv_state[1], 1e-6); // current input
    const expected2 = silu(3.0);
    try std.testing.expectApproxEqAbs(expected2, conv_out[0], 1e-5);

    // Step 3: input=3.0, state=[1,2] → conv = 1*1 + 2*1 + 3*1 = 6.0
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{3.0}, &conv_w, null, 1, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), conv_state[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), conv_state[1], 1e-6);
    const expected3 = silu(6.0);
    try std.testing.expectApproxEqAbs(expected3, conv_out[0], 1e-5);
}

test "causalConv1dSilu with bias" {
    var conv_out = [_]f32{ 0, 0 };
    var conv_state = [_]f32{ 0, 0 };
    const conv_in = [_]f32{ 1.0, 1.0 };
    const conv_w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const conv_b = [_]f32{ 0.5, -0.5 };

    causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, &conv_b, 2, 2);

    // ch0: bias(0.5) + 0*1 + 1*1 = 1.5, SiLU(1.5) ≈ 1.2262
    try std.testing.expectApproxEqAbs(@as(f32, 1.2262), conv_out[0], 0.001);

    // ch1: bias(-0.5) + 0*1 + 1*1 = 0.5, SiLU(0.5) ≈ 0.3112
    try std.testing.expectApproxEqAbs(@as(f32, 0.3112), conv_out[1], 0.001);
}
