// In-place Rotary Position Embedding (RoPE).
// Rotates element i against element i + rope_dim/2 within each head, which is
// the half-split (NeoX) layout the CPU and Metal kernels use and the layout
// GGUF weights are stored for. This shader used to pair adjacent elements
// (d, d+1) instead, a different rotation, so every model decoded wrong on
// WebGPU.
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

    let r = data[base + i];
    let im = data[base + i + half_rope];
    data[base + i]             = r * cos_a - im * sin_a;
    data[base + i + half_rope] = r * sin_a + im * cos_a;
}
