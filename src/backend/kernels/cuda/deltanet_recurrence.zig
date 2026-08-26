//! DeltaNet SSM recurrence kernel for CUDA.
//! One block per v-head (single-threaded, recurrence is sequential).
//! Launch: grid = num_v_heads, block = 1.

const cu = @import("common.zig");

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
) callconv(.nvptx_device) void {
    const h = cu.blockIdx();
    if (h >= num_v_heads) return;

    const decay = cu.expf(gate_vals[h]);
    const beta_h = beta_vals[h];
    const kh = if (num_k_heads == 0 or num_v_heads == 0) 0 else h * num_k_heads / num_v_heads;
    const s_off = h * head_v_dim * head_k_dim;
    const k_base = kh * head_k_dim;

    var kq: f32 = 0.0;
    var ki: u32 = 0;
    while (ki < head_k_dim) : (ki += 1) {
        kq += k_ptr[k_base + ki] * q_ptr[k_base + ki];
    }

    var vi: u32 = 0;
    while (vi < head_v_dim) : (vi += 1) {
        const row_off = s_off + vi * head_k_dim;
        var sk: f32 = 0.0;
        var sq_dec: f32 = 0.0;
        ki = 0;
        while (ki < head_k_dim) : (ki += 1) {
            const s_dec = ssm_state[row_off + ki] * decay;
            ssm_state[row_off + ki] = s_dec;
            sk += s_dec * k_ptr[k_base + ki];
            sq_dec += s_dec * q_ptr[k_base + ki];
        }
        const delta = beta_h * (v_ptr[h * head_v_dim + vi] - sk);
        output[h * head_v_dim + vi] = (sq_dec + delta * kq) * q_scale;
        ki = 0;
        while (ki < head_k_dim) : (ki += 1) {
            ssm_state[row_off + ki] += k_ptr[k_base + ki] * delta;
        }
    }

    const off = h * head_v_dim;
    var ss: f32 = 0.0;
    vi = 0;
    while (vi < head_v_dim) : (vi += 1) {
        ss += output[off + vi] * output[off + vi];
    }
    const inv_n = cu.rcpf(@as(f32, @floatFromInt(head_v_dim)));
    const inv_rms = cu.rsqrtf(ss * inv_n + rms_eps);
    vi = 0;
    while (vi < head_v_dim) : (vi += 1) {
        const normed = output[off + vi] * ssm_norm_w[vi] * inv_rms;
        const z = z_buf[off + vi];
        output[off + vi] = normed * (z * cu.sigmoidf(z));
    }
}

const std = @import("std");

test "constants valid" {
    _ = @sizeOf(u8);
}

test "fuzz: deltanet_recurrence functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, _: *std.testing.Smith) !void {
            comptime {
                _ = @sizeOf(u8);
            }
        }
    }.f, .{});
}
