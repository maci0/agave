// In-place Rotary Position Embedding (RoPE).
// Applies rotation to adjacent pairs (d, d+1) for d < rope_dim within each head.
// One thread per pair. Total threads = n_heads * (rope_dim / 2).

struct Params {
    pos: u32,
    n_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_rope = params.rope_dim / 2u;
    let total = params.n_heads * half_rope;
    let tid = gid.x;
    if (tid >= total) {
        return;
    }

    let h = tid / half_rope;
    let i = tid % half_rope;
    let base = h * params.head_dim;

    // freq = 1 / theta^(2i / rope_dim) = exp(-log(theta) * 2i / rope_dim)
    let freq = exp(-log(params.theta) * f32(2u * i) / f32(params.rope_dim));
    let angle = f32(params.pos) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let d = 2u * i;
    let x0 = data[base + d];
    let x1 = data[base + d + 1u];
    data[base + d]      = x0 * cos_a - x1 * sin_a;
    data[base + d + 1u] = x0 * sin_a + x1 * cos_a;
}
