//! CPU Rotary Position Embedding (RoPE) kernel.

const std = @import("std");

/// Maximum half rope_dim for precomputed cos/sin buffers.
const max_rope_half_dim: usize = 512;

/// Qwen3.5 interleaved multimodal RoPE sections (T, H, W) over rope_dim/2 freqs.
/// HF `mrope_section` default: 11 + 11 + 10 = 32 pairs = 64 rotary dims.
pub const qwen35_mrope_section = [3]u32{ 11, 11, 10 };

/// Interleaved 3D RoPE: frequency i uses T, H, or W according to `section`.
/// Layout matches `Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`.
/// When t_pos == h_pos == w_pos this is identical to `rope`.
pub fn ropeMrope(
    x: [*]f32,
    t_pos: usize,
    h_pos: usize,
    w_pos: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    section: [3]u32,
) void {
    if (t_pos == h_pos and h_pos == w_pos) {
        rope(x, t_pos, n_heads, head_dim, rope_dim, theta);
        return;
    }

    const half = rope_dim / 2;
    const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));
    const neg_log_theta: f32 = -@log(theta);
    const t_f: f32 = @floatFromInt(t_pos);
    const h_f: f32 = @floatFromInt(h_pos);
    const w_f: f32 = @floatFromInt(w_pos);
    const h_lim: usize = @as(usize, section[1]) * 3;
    const w_lim: usize = @as(usize, section[2]) * 3;

    if (half > max_rope_half_dim) @panic("ropeMrope: half-dim exceeds max_rope_half_dim");
    var cos_buf: [max_rope_half_dim]f32 = undefined;
    var sin_buf: [max_rope_half_dim]f32 = undefined;
    for (0..half) |i| {
        const pos = if (i % 3 == 1 and i < h_lim) h_f else if (i % 3 == 2 and i < w_lim) w_f else t_f;
        const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
        const angle = pos * freq;
        cos_buf[i] = @cos(angle);
        sin_buf[i] = @sin(angle);
    }

    for (0..n_heads) |h| {
        const base = h * head_dim;
        for (0..half) |i| {
            const r = x[base + i];
            const im = x[base + i + half];
            x[base + i] = @mulAdd(f32, r, cos_buf[i], -(im * sin_buf[i]));
            x[base + i + half] = @mulAdd(f32, r, sin_buf[i], im * cos_buf[i]);
        }
    }
}

/// Applies Rotary Position Embedding (RoPE) in-place.
/// Uses split-complex layout: pairs `[i, i+half]` are rotated together
/// (matches CUDA convention; NOT interleaved `[2i, 2i+1]`).
pub fn rope(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
    const half = rope_dim / 2;
    const p: f32 = @floatFromInt(pos);
    const inv_rd: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));
    const neg_log_theta: f32 = -@log(theta);

    // Precompute cos/sin for each frequency (shared across all heads)
    if (half > max_rope_half_dim) @panic("rope: half-dim exceeds max_rope_half_dim");
    var cos_buf: [max_rope_half_dim]f32 = undefined;
    var sin_buf: [max_rope_half_dim]f32 = undefined;
    for (0..half) |i| {
        const freq = @exp(neg_log_theta * @as(f32, @floatFromInt(2 * i)) * inv_rd);
        const angle = p * freq;
        cos_buf[i] = @cos(angle);
        sin_buf[i] = @sin(angle);
    }

    const V8 = @Vector(8, f32);
    for (0..n_heads) |h| {
        const base = h * head_dim;
        var i: usize = 0;
        while (i + 8 <= half) : (i += 8) {
            const r: V8 = x[base + i ..][0..8].*;
            const im: V8 = x[base + i + half ..][0..8].*;
            const c: V8 = cos_buf[i..][0..8].*;
            const s: V8 = sin_buf[i..][0..8].*;
            const ims = im * s;
            x[base + i ..][0..8].* = @mulAdd(V8, r, c, -ims);
            x[base + i + half ..][0..8].* = @mulAdd(V8, r, s, im * c);
        }
        while (i < half) : (i += 1) {
            const r = x[base + i];
            const im = x[base + i + half];
            x[base + i] = @mulAdd(f32, r, cos_buf[i], -(im * sin_buf[i]));
            x[base + i + half] = @mulAdd(f32, r, sin_buf[i], im * cos_buf[i]);
        }
    }
}

test "rope pos=0 is identity" {
    // At position 0, all angles are 0 → cos=1, sin=0 → no rotation.
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    rope(&x, 0, 1, 4, 4, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x[3], 1e-6);
}

test "rope preserves magnitude" {
    // RoPE is a rotation — magnitude should be preserved per pair.
    var x = [_]f32{ 3.0, 0.0, 4.0, 0.0 };
    rope(&x, 5, 1, 4, 4, 10000.0);
    // Pair 0: (x[0], x[2]) should have magnitude 5.0 (sqrt(3^2+4^2))
    const mag0 = @sqrt(x[0] * x[0] + x[2] * x[2]);
    const mag1 = @sqrt(x[1] * x[1] + x[3] * x[3]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), mag0, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mag1, 1e-4);
}

test "rope pos>0 rotates values" {
    // Verify pos=1, theta=10000, rope_dim=4 produces expected rotation.
    // Split-complex pairs: (x[0], x[2]) and (x[1], x[3]).
    // freq[0] = exp(-log(10000) * 0 / 4) = 1.0, angle = 1.0 * 1 = 1.0
    // freq[1] = exp(-log(10000) * 2 / 4) = 1/100, angle = 0.01
    var x = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    rope(&x, 1, 1, 4, 4, 10000.0);
    // Pair 0 rotated by angle=1.0: x[0]=cos(1), x[2]=sin(1)
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x[2], 1e-5);
    // Pair 1 should stay near zero (input was zero)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), x[3], 1e-5);
}

test "rope multi-head consistency" {
    // Two heads with same input should produce same output.
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 };
    rope(&x, 3, 2, 4, 4, 10000.0);
    // Head 0 and head 1 should match exactly
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(x[i], x[4 + i], 1e-6);
    }
}

test "rope partial rope_dim leaves remainder untouched" {
    // head_dim=8, rope_dim=4: only first 4 dims rotated, rest unchanged.
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    rope(&x, 5, 1, 8, 4, 10000.0);
    // Dims 4..8 (outside rope_dim) should be untouched
    try std.testing.expectEqual(@as(f32, 5.0), x[4]);
    try std.testing.expectEqual(@as(f32, 6.0), x[5]);
    try std.testing.expectEqual(@as(f32, 7.0), x[6]);
    try std.testing.expectEqual(@as(f32, 8.0), x[7]);
    // Dims 0..4 (inside rope_dim) should be rotated — verify with known cos/sin
    // freq[0]=1.0, angle=5.0: x[0]=1*cos(5)-3*sin(5), x[2]=1*sin(5)+3*cos(5)
    // freq[1]=0.01, angle=0.05: x[1]=2*cos(0.05)-4*sin(0.05), x[3]=2*sin(0.05)+4*cos(0.05)
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 5.0)) - 3.0 * @sin(@as(f32, 5.0)), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 5.0)) + 3.0 * @cos(@as(f32, 5.0)), x[2], 1e-4);
}

test "rope exercises SIMD path" {
    // rope_dim=32 → half=16, which exercises the 8-wide SIMD loop (2 iterations).
    const hd = 32;
    var x: [hd]f32 = undefined;
    for (0..hd) |i| x[i] = @floatFromInt(i + 1);
    const orig = x;

    rope(&x, 1, 1, hd, hd, 10000.0);

    // Verify magnitude preservation for each rotated pair (i, i+half)
    const half = hd / 2;
    for (0..half) |i| {
        const orig_mag = @sqrt(orig[i] * orig[i] + orig[i + half] * orig[i + half]);
        const new_mag = @sqrt(x[i] * x[i] + x[i + half] * x[i + half]);
        try std.testing.expectApproxEqAbs(orig_mag, new_mag, 1e-4);
    }
    // Verify first pair is actually rotated (pos=1, not identity)
    try std.testing.expect(@abs(x[0] - orig[0]) > 0.01);
}

test "fuzz: all rope functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Constrain rope_dim to even values in [2, 64].
            const rope_dim_half_raw = smith.valueWithHash(u5, 0);
            const rope_dim_half: usize = @as(usize, rope_dim_half_raw) + 1;
            const rd: usize = rope_dim_half * 2;

            // head_dim >= rope_dim, up to rope_dim + 15 extra.
            const hd_extra = smith.valueWithHash(u4, 1);
            const hd: usize = rd + @as(usize, hd_extra);

            // n_heads in [1, 4].
            const nh: usize = @as(usize, smith.valueWithHash(u2, 2)) + 1;

            const pos: usize = @as(usize, smith.valueWithHash(u10, 3));

            // theta > 0.
            const theta_raw = smith.valueWithHash(u16, 4);
            const theta: f32 = @as(f32, @floatFromInt(@as(u32, theta_raw) + 1));

            const total = nh * hd;
            if (total > 512) return;

            var buf: [512]f32 = undefined;
            for (0..total) |i| {
                const seed: u32 = @as(u32, @intCast(i)) +% 100;
                const bits = smith.valueWithHash(u32, seed);
                const val: f32 = @bitCast(bits);
                buf[i] = if (std.math.isFinite(val)) val else 1.0;
            }

            var orig: [512]f32 = undefined;
            @memcpy(orig[0..total], buf[0..total]);

            rope(&buf, pos, nh, hd, rd, theta);

            // Verify magnitude preservation per rotated pair per head.
            const half = rd / 2;
            for (0..nh) |h| {
                const base = h * hd;
                for (0..half) |i| {
                    const o_r = orig[base + i];
                    const o_im = orig[base + i + half];
                    const n_r = buf[base + i];
                    const n_im = buf[base + i + half];
                    const orig_mag = @sqrt(o_r * o_r + o_im * o_im);
                    const new_mag = @sqrt(n_r * n_r + n_im * n_im);
                    if (std.math.isFinite(orig_mag) and orig_mag > 1e-6) {
                        if (!std.math.isFinite(new_mag)) return;
                        const rel_err = @abs(orig_mag - new_mag) / orig_mag;
                        if (rel_err > 1e-3) return error.MagnitudeNotPreserved;
                    }
                }
            }
        }
    }.f, .{});
}

test "ropeMrope equals rope when T=H=W" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    rope(&a, 7, 1, 4, 4, 10000.0);
    ropeMrope(&b, 7, 7, 7, 1, 4, 4, 10000.0, qwen35_mrope_section);
    for (0..4) |i| try std.testing.expectApproxEqAbs(a[i], b[i], 1e-6);
}

test "ropeMrope 3D differs from 1D when H and W differ" {
    var a = [_]f32{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var b = a;
    // rope_dim=8 → half=4. i=1 uses H (pair x[1], x[5]).
    rope(&a, 1, 1, 8, 8, 10000.0);
    ropeMrope(&b, 1, 5, 9, 1, 8, 8, 10000.0, qwen35_mrope_section);
    var same = true;
    for (0..8) |i| {
        if (@abs(a[i] - b[i]) > 1e-5) same = false;
    }
    try std.testing.expect(!same);
}
