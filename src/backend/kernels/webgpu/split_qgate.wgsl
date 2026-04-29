struct Params {
    hd: u32,
    nh: u32,
}

@group(0) @binding(0) var<storage, read> qg: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.nh * params.hd;
    if (idx >= total) { return; }
    let h = idx / params.hd;
    let d = idx % params.hd;
    q_out[idx] = qg[h * params.hd * 2u + d];
    g_out[idx] = qg[h * params.hd * 2u + params.hd + d];
}
