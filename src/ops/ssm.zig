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
    std.debug.assert(d >= 2);
    std.debug.assert(conv_ch > 0);
    const hist = d - 1;
    for (0..conv_ch) |ch| {
        var sum: f32 = if (conv_b) |b| b[ch] else 0.0;
        if (D) |_| {
            inline for (0..D.? - 1) |k| {
                const s_idx = std.math.add(usize, std.math.mul(usize, k, conv_ch) catch @panic("conv_state idx overflow"), ch) catch @panic("conv_state idx overflow");
                const w_idx = std.math.add(usize, std.math.mul(usize, ch, d) catch @panic("conv_w idx overflow"), k) catch @panic("conv_w idx overflow");
                sum += conv_state[s_idx] * conv_w[w_idx];
            }
        } else {
            for (0..hist) |k| {
                const s_idx = std.math.add(usize, std.math.mul(usize, k, conv_ch) catch @panic("conv_state idx overflow"), ch) catch @panic("conv_state idx overflow");
                const w_idx = std.math.add(usize, std.math.mul(usize, ch, d) catch @panic("conv_w idx overflow"), k) catch @panic("conv_w idx overflow");
                sum += conv_state[s_idx] * conv_w[w_idx];
            }
        }
        const w_tail = std.math.add(usize, std.math.mul(usize, ch, d) catch @panic("conv_w idx overflow"), hist) catch @panic("conv_w idx overflow");
        sum += conv_in[ch] * conv_w[w_tail];
        conv_out[ch] = silu(sum);
    }
    if (D) |_| {
        if (D.? - 1 > 1) {
            inline for (0..D.? - 2) |p| {
                const dst_off = std.math.mul(usize, p, conv_ch) catch @panic("conv_state shift overflow");
                const src_off = std.math.mul(usize, p + 1, conv_ch) catch @panic("conv_state shift overflow");
                @memcpy(conv_state[dst_off..][0..conv_ch], conv_state[src_off..][0..conv_ch]);
            }
        }
    } else {
        if (hist > 1) {
            for (0..hist - 1) |p| {
                const dst_off = std.math.mul(usize, p, conv_ch) catch @panic("conv_state shift overflow");
                const src_off = std.math.mul(usize, p + 1, conv_ch) catch @panic("conv_state shift overflow");
                @memcpy(conv_state[dst_off..][0..conv_ch], conv_state[src_off..][0..conv_ch]);
            }
        }
    }
    if (hist > 0) {
        const tail_off = std.math.mul(usize, hist - 1, conv_ch) catch @panic("conv_state tail overflow");
        @memcpy(conv_state[tail_off..][0..conv_ch], conv_in[0..conv_ch]);
    }
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
    std.debug.assert(heads_per_group > 0 and num_heads > 0 and head_dim > 0 and d_state > 0);
    std.debug.assert(num_heads % heads_per_group == 0);
    const h_d_state = std.math.mul(usize, head_dim, d_state) catch @panic("mamba state stride overflow");
    const expected = std.math.mul(usize, num_heads, h_d_state) catch @panic("mamba state size overflow");
    std.debug.assert(state.len >= expected);
    for (0..num_heads) |h| {
        const group = h / heads_per_group;
        const s_off = std.math.mul(usize, h, h_d_state) catch @panic("mamba s_off overflow");

        const dt_h = math_ops.softplus(dt_raw[h] + dt_bias[h]);
        const decay = @exp(ssm_a[h] * dt_h);

        const x_off = std.math.mul(usize, h, head_dim) catch @panic("mamba x_h overflow");
        const x_h = x + x_off;
        const b_off = std.math.mul(usize, group, d_state) catch @panic("mamba B_g overflow");
        const B_g = B + b_off;
        const c_off = std.math.mul(usize, group, d_state) catch @panic("mamba C_g overflow");
        const C_g = C + c_off;
        const y_off = std.math.mul(usize, h, head_dim) catch @panic("mamba y_h overflow");
        const y_h = y + y_off;

        for (0..head_dim) |i| {
            const xd = x_h[i] * dt_h;
            var yi: f32 = ssm_d[h] * x_h[i];
            const row_off = std.math.add(usize, s_off, std.math.mul(usize, i, d_state) catch @panic("mamba row overflow")) catch @panic("mamba row overflow");
            const s_row = state[row_off..][0..d_state];

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
    std.debug.assert(n_groups > 0 and d_inner > 0);
    if (d_inner % n_groups != 0) @panic("mamba2: d_inner must be divisible by n_groups");
    const elem_per_group: usize = d_inner / n_groups;
    for (0..n_groups) |g| {
        const off = std.math.mul(usize, g, elem_per_group) catch @panic("group offset overflow");
        const w_off = std.math.mul(usize, g, elem_per_group) catch @panic("group offset overflow");
        const y_g = y + off;
        const z_g = z + off;
        const w_g = norm_w + w_off;

        // 1. Apply SiLU gate in-place: y = y * silu(z) — SIMD vectorized
        const ones: V8 = @splat(1.0);
        var j: usize = 0;
        while (j + 8 <= elem_per_group) : (j += 8) {
            const yv: V8 = y_g[j..][0..8].*;
            const zv: V8 = z_g[j..][0..8].*;
            const sig: V8 = ones / (ones + @exp(-zv));
            y_g[j..][0..8].* = yv * zv * sig;
        }
        while (j < elem_per_group) : (j += 1) {
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

        // 3. RMS normalize and apply weight — SIMD vectorized
        const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(elem_per_group)) + eps);
        const inv_rms_v: V8 = @splat(inv_rms);
        var k: usize = 0;
        while (k + 8 <= elem_per_group) : (k += 8) {
            const yv: V8 = y_g[k..][0..8].*;
            const wv: V8 = w_g[k..][0..8].*;
            y_g[k..][0..8].* = yv * inv_rms_v * wv;
        }
        while (k < elem_per_group) : (k += 1) {
            y_g[k] *= inv_rms * w_g[k];
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
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), conv_out[0], 1e-4);

    // ch1: 2.0*0.5 + 4.0*0.5 = 3.0, SiLU(3.0) ≈ 2.8577
    try std.testing.expectApproxEqAbs(@as(f32, 2.8577), conv_out[1], 1e-4);

    // State should now contain the current input (direct copy, must be exact)
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), conv_state[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), conv_state[1], 1e-6);
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
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), conv_state[0], 1e-6); // after shift: previous state[1] now at [0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), conv_state[1], 1e-6); // current input written to tail

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

test "causalConv1dSilu varying weights" {
    // Previous ring buffer test uses uniform weights [1,1,1], which doesn't
    // verify that each weight position multiplies the correct history element.
    var conv_out = [_]f32{0};
    var conv_state = [_]f32{ 0.0, 0.0 }; // d_conv=3, so 2 history slots
    const conv_w = [_]f32{ 0.5, 1.0, 2.0 }; // distinct weights per position

    // Step 1: input=1.0, state=[0,0] → conv = 0*0.5 + 0*1.0 + 1*2.0 = 2.0
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{1.0}, &conv_w, null, 1, 3);
    try std.testing.expectApproxEqAbs(silu(2.0), conv_out[0], 1e-5);

    // Step 2: input=3.0, state=[0,1] → conv = 0*0.5 + 1*1.0 + 3*2.0 = 7.0
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{3.0}, &conv_w, null, 1, 3);
    try std.testing.expectApproxEqAbs(silu(7.0), conv_out[0], 1e-5);

    // Step 3: input=2.0, state=[1,3] → conv = 1*0.5 + 3*1.0 + 2*2.0 = 7.5
    causalConv1dSilu(&conv_out, &conv_state, &[_]f32{2.0}, &conv_w, null, 1, 3);
    try std.testing.expectApproxEqAbs(silu(7.5), conv_out[0], 1e-5);
}

test "causalConv1dSilu with bias" {
    var conv_out = [_]f32{ 0, 0 };
    var conv_state = [_]f32{ 0, 0 };
    const conv_in = [_]f32{ 1.0, 1.0 };
    const conv_w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const conv_b = [_]f32{ 0.5, -0.5 };

    causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, &conv_b, 2, 2);

    // ch0: bias(0.5) + 0*1 + 1*1 = 1.5, SiLU(1.5) ≈ 1.2264
    try std.testing.expectApproxEqAbs(@as(f32, 1.2264), conv_out[0], 1e-3);

    // ch1: bias(-0.5) + 0*1 + 1*1 = 0.5, SiLU(0.5) ≈ 0.3112
    try std.testing.expectApproxEqAbs(@as(f32, 0.3112), conv_out[1], 1e-4);
}

test "mamba2Recurrence basic" {
    // 1 head, head_dim=2, d_state=2, 1 group, zero initial state
    var y = [_]f32{ 0, 0 };
    var state = [_]f32{ 0, 0, 0, 0 }; // [head_dim × d_state]
    const x = [_]f32{ 1.0, 2.0 };
    const B = [_]f32{ 0.5, 0.5 };
    const C = [_]f32{ 1.0, 1.0 };
    const dt_raw = [_]f32{0.0};
    const dt_bias = [_]f32{0.0};
    const ssm_a = [_]f32{-1.0};
    const ssm_d = [_]f32{1.0};

    mamba2Recurrence(&y, &state, &x, &B, &C, &dt_raw, &dt_bias, &ssm_a, &ssm_d, 1, 2, 2, 1);

    // dt = softplus(0+0) = ln(2), decay = exp(-ln(2)) = 0.5
    const dt = math_ops.softplus(0.0);

    // Element 0: xd = 1.0 * dt, state = [xd*0.5, xd*0.5], y = D*x + sum(state*C) = 1 + xd
    const xd0 = 1.0 * dt;
    try std.testing.expectApproxEqAbs(1.0 + xd0, y[0], 1e-5);

    // Element 1: xd = 2.0 * dt, y = D*x + sum(state*C) = 2 + 2*dt
    try std.testing.expectApproxEqAbs(2.0 + 2.0 * dt, y[1], 1e-5);

    // Verify state was updated
    try std.testing.expectApproxEqAbs(xd0 * 0.5, state[0], 1e-5);
    try std.testing.expectApproxEqAbs(xd0 * 0.5, state[1], 1e-5);
}

test "mamba2Recurrence state decay" {
    // Verify decay is applied to existing state on second call
    var y = [_]f32{ 0, 0 };
    var state = [_]f32{ 0, 0, 0, 0 };
    const x = [_]f32{ 1.0, 0.0 };
    const B = [_]f32{ 1.0, 0.0 }; // Only update state[*][0]
    const C = [_]f32{ 1.0, 0.0 };
    const dt_raw = [_]f32{0.0};
    const dt_bias = [_]f32{0.0};
    const ssm_a = [_]f32{-1.0}; // decay = exp(-ln(2)) = 0.5
    const ssm_d = [_]f32{0.0}; // No skip connection

    // Call 1: state starts at 0
    mamba2Recurrence(&y, &state, &x, &B, &C, &dt_raw, &dt_bias, &ssm_a, &ssm_d, 1, 2, 2, 1);
    const dt = math_ops.softplus(0.0);
    const s0_call1 = 1.0 * dt * 1.0; // xd * B[0]
    try std.testing.expectApproxEqAbs(s0_call1, state[0], 1e-5);

    // Call 2: state[0] should decay by 0.5 then accumulate new input
    mamba2Recurrence(&y, &state, &x, &B, &C, &dt_raw, &dt_bias, &ssm_a, &ssm_d, 1, 2, 2, 1);
    const decay: f32 = 0.5; // exp(-1.0 * ln(2))
    const s0_call2 = decay * s0_call1 + 1.0 * dt * 1.0;
    try std.testing.expectApproxEqAbs(s0_call2, state[0], 1e-5);
}

test "groupRmsNormSiluGate basic" {
    // d_inner=4, n_groups=1: verify SiLU gate + RMS normalization
    // Using large z so SiLU(z) ≈ z, making expected output analytically clean.
    var y = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    const z = [_]f32{ 100.0, 100.0, 100.0, 100.0 }; // SiLU(100) ≈ 100.0
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    groupRmsNormSiluGate(&y, &z, &w, 4, 1, 1e-6);

    // gated ≈ [300, 400, 0, 0]
    // RMS = sqrt((300²+400²)/4) = sqrt(62500) = 250
    // output = gated / 250 * w = [1.2, 1.6, 0.0, 0.0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), y[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.6), y[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[3], 1e-6);
}

test "groupRmsNormSiluGate exercises SiLU gate" {
    // Use moderate z values where SiLU(z) ≠ z, verifying the gate is
    // actually applied rather than bypassed by large z approximation.
    var y = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const z = [_]f32{ 0.5, -0.5, 1.0, -1.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    groupRmsNormSiluGate(&y, &z, &w, 4, 1, 1e-6);

    // SiLU(0.5)≈0.3112, SiLU(-0.5)≈-0.1888, SiLU(1.0)≈0.7311, SiLU(-1.0)≈-0.2689
    // gated = y * SiLU(z) = [0.3112, -0.1888, 0.7311, -0.2689]
    // RMS = sqrt(sum_sq / 4)
    const g0 = silu(0.5);
    const g1 = silu(-0.5);
    const g2 = silu(1.0);
    const g3 = silu(-1.0);
    const sum_sq = g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3;
    const rms = @sqrt(sum_sq / 4.0);
    try std.testing.expectApproxEqAbs(g0 / rms, y[0], 1e-4);
    try std.testing.expectApproxEqAbs(g1 / rms, y[1], 1e-4);
    try std.testing.expectApproxEqAbs(g2 / rms, y[2], 1e-4);
    try std.testing.expectApproxEqAbs(g3 / rms, y[3], 1e-4);
}

test "groupRmsNormSiluGate multi-group" {
    // d_inner=4, n_groups=2 (2 elements per group): groups normalize independently
    var y = [_]f32{ 3.0, 4.0, 5.0, 0.0 };
    const z = [_]f32{ 100.0, 100.0, 100.0, 100.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    groupRmsNormSiluGate(&y, &z, &w, 4, 2, 1e-6);

    // Group 0: gated ≈ [300, 400], RMS = sqrt((90000+160000)/2) = sqrt(125000) ≈ 353.55
    // output = [300/353.55, 400/353.55] ≈ [0.8485, 1.1314]
    try std.testing.expectApproxEqAbs(@as(f32, 0.8485), y[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1314), y[1], 1e-3);

    // Group 1: gated ≈ [500, 0], RMS = sqrt(250000/2) = sqrt(125000) ≈ 353.55
    // output = [500/353.55, 0] ≈ [1.4142, 0.0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142), y[2], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[3], 1e-6);
}

test "mamba2Recurrence exercises SIMD path" {
    // d_state=16 ensures the 8-wide SIMD loop executes (requires d_state >= 8).
    // Previous tests use d_state=2 which only exercises the scalar tail.
    const num_heads = 1;
    const head_dim = 1;
    const d_state = 16;

    var y = [_]f32{0};
    var state = [_]f32{0} ** (head_dim * d_state);
    const x = [_]f32{1.0};
    var B: [d_state]f32 = undefined;
    var C: [d_state]f32 = undefined;
    for (0..d_state) |i| {
        B[i] = 1.0;
        C[i] = 1.0;
    }
    const dt_raw = [_]f32{0.0};
    const dt_bias = [_]f32{0.0};
    const ssm_a = [_]f32{-1.0};
    const ssm_d = [_]f32{0.0}; // No skip, easier to verify

    mamba2Recurrence(&y, &state, &x, &B, &C, &dt_raw, &dt_bias, &ssm_a, &ssm_d, num_heads, head_dim, d_state, 1);

    // dt = softplus(0) = ln(2), xd = 1.0 * ln(2)
    // Each state[j] = xd * B[j] = ln(2) * 1.0 = ln(2)
    // y = sum(state[j] * C[j]) = d_state * ln(2) = 16 * ln(2)
    const dt = math_ops.softplus(0.0);
    try std.testing.expectApproxEqAbs(dt * @as(f32, d_state), y[0], 1e-4);

    // Verify all state elements are equal (uniform B/C)
    for (0..d_state) |j| {
        try std.testing.expectApproxEqAbs(dt, state[j], 1e-5);
    }
}

test "fuzz: all ssm functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── causalConv1dSilu ──
            // Use fixed conv_ch=4, d_conv=4 (3 history slots)
            const conv_ch = 4;
            const d_conv = 4;
            const hist = d_conv - 1;

            var conv_out: [conv_ch]f32 = undefined;
            var conv_state: [hist * conv_ch]f32 = undefined;
            var conv_in: [conv_ch]f32 = undefined;
            var conv_w: [conv_ch * d_conv]f32 = undefined;
            var conv_b: [conv_ch]f32 = undefined;

            for (0..conv_state.len) |i| conv_state[i] = @bitCast(smith.valueWithHash(u32, @intCast(i)));
            for (0..conv_in.len) |i| conv_in[i] = @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 100));
            for (0..conv_w.len) |i| conv_w[i] = @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 200));
            for (0..conv_b.len) |i| conv_b[i] = @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 300));

            // Clamp to finite range to avoid NaN propagation masking bugs
            for (&conv_state) |*v| v.* = clampFinite(v.*);
            for (&conv_in) |*v| v.* = clampFinite(v.*);
            for (&conv_w) |*v| v.* = clampFinite(v.*);
            for (&conv_b) |*v| v.* = clampFinite(v.*);

            // Call with bias
            causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, &conv_b, conv_ch, d_conv);
            for (conv_out) |v| if (!std.math.isFinite(v)) return error.TestUnexpectedResult;

            // Call without bias (null path)
            causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, null, conv_ch, d_conv);
            for (conv_out) |v| if (!std.math.isFinite(v)) return error.TestUnexpectedResult;

            // ── mamba2Recurrence ──
            const num_heads = 2;
            const head_dim = 2;
            const d_state = 16; // exercises SIMD path (>= 8)
            const n_groups = 1;
            const hpg = num_heads / n_groups;

            var y_m: [num_heads * head_dim]f32 = undefined;
            var state_m: [num_heads * head_dim * d_state]f32 = undefined;
            var x_m: [num_heads * head_dim]f32 = undefined;
            var B_m: [n_groups * d_state]f32 = undefined;
            var C_m: [n_groups * d_state]f32 = undefined;
            var dt_raw_m: [num_heads]f32 = undefined;
            var dt_bias_m: [num_heads]f32 = undefined;
            var ssm_a_m: [num_heads]f32 = undefined;
            var ssm_d_m: [num_heads]f32 = undefined;

            for (0..state_m.len) |i| state_m[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 400))));
            for (0..x_m.len) |i| x_m[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 500))));
            for (0..B_m.len) |i| B_m[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 600))));
            for (0..C_m.len) |i| C_m[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, @as(u32, @intCast(i)) +% 700))));
            for (0..num_heads) |i| {
                const h: u32 = @intCast(i);
                dt_raw_m[i] = clampSmall(@as(f32, @bitCast(smith.valueWithHash(u32, h +% 800))));
                dt_bias_m[i] = clampSmall(@as(f32, @bitCast(smith.valueWithHash(u32, h +% 900))));
                ssm_a_m[i] = -@abs(clampSmall(@as(f32, @bitCast(smith.valueWithHash(u32, h +% 1000)))));
                ssm_d_m[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, h +% 1100))));
            }

            mamba2Recurrence(&y_m, &state_m, &x_m, &B_m, &C_m, &dt_raw_m, &dt_bias_m, &ssm_a_m, &ssm_d_m, num_heads, head_dim, d_state, hpg);
            // y values may be large but must be finite
            for (y_m) |v| if (!std.math.isFinite(v)) return error.TestUnexpectedResult;

            // ── groupRmsNormSiluGate ──
            const d_inner = 16; // exercises SIMD path, divisible by 1,2,4,8,16
            const grn_groups = 4;

            var y_g: [d_inner]f32 = undefined;
            var z_g: [d_inner]f32 = undefined;
            var norm_w_g: [d_inner]f32 = undefined;

            for (0..d_inner) |i| {
                const idx: u32 = @intCast(i);
                y_g[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, idx +% 1200))));
                z_g[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, idx +% 1300))));
                norm_w_g[i] = clampFinite(@as(f32, @bitCast(smith.valueWithHash(u32, idx +% 1400))));
            }

            const eps_bits = smith.valueWithHash(u32, @as(u32, 1500));
            const eps_raw: f32 = @bitCast(eps_bits);
            const eps = if (std.math.isFinite(eps_raw) and eps_raw > 0) @min(eps_raw, 1.0) else 1e-6;

            groupRmsNormSiluGate(&y_g, &z_g, &norm_w_g, d_inner, grn_groups, eps);
            for (y_g) |v| if (!std.math.isFinite(v)) return error.TestUnexpectedResult;
        }

        /// Clamp to [-1e6, 1e6], replacing non-finite with 0.
        fn clampFinite(v: f32) f32 {
            if (!std.math.isFinite(v)) return 0.0;
            return @max(-1e6, @min(1e6, v));
        }

        /// Clamp to [-10, 10] for values fed into exp/softplus to avoid overflow.
        fn clampSmall(v: f32) f32 {
            if (!std.math.isFinite(v)) return 0.0;
            return @max(-10.0, @min(10.0, v));
        }
    }.f, .{});
}
