struct Params {
    stride: u32,
    n_pairs: u32,
}

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n_pairs * params.stride;
    if (idx >= total) { return; }
    let pair = idx / params.stride;
    let off = idx % params.stride;
    out_a[idx] = inp[pair * 2u * params.stride + off];
    out_b[idx] = inp[(pair * 2u + 1u) * params.stride + off];
}
