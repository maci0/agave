//! Batched RoPE kernel: apply rotary embeddings to n_tok vectors at different positions.
//! Grid: ceil(n_tok * n_heads * rope_dim/2 / 256) blocks of 256 threads.

const cu = @import("common.zig");

const ln2: f32 = 0.6931471805599453;

export fn rope_batched_kernel(x: [*]f32, positions: [*]const u32, n_tok: u32, n_heads: u32, head_dim: u32, rope_dim: u32, theta: f32) callconv(.kernel) void {
    const idx = cu.globalIdx();
    const half_rope = rope_dim / 2;
    const pairs_per_tok = n_heads * half_rope;
    const total = n_tok * pairs_per_tok;
    if (idx >= total) return;

    const tok = idx / pairs_per_tok;
    const pair = idx % pairs_per_tok;
    const head = pair / half_rope;
    const i = pair % half_rope;

    const stride = n_heads * head_dim;
    const base = tok * stride + head * head_dim;
    const pos = positions[tok];

    const neg_log_theta = -cu.log2f(theta) * ln2;
    const freq = cu.expf(neg_log_theta * @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(rope_dim)));
    const angle = @as(f32, @floatFromInt(pos)) * freq;

    const cos_a = cu.cosf(angle);
    const sin_a = cu.sinf(angle);

    const re_idx = base + i;
    const im_idx = base + i + half_rope;
    const r = x[re_idx];
    const im = x[im_idx];
    x[re_idx] = r * cos_a - im * sin_a;
    x[im_idx] = r * sin_a + im * cos_a;
}
