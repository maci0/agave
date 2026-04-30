struct Params {
    n_heads: u32,
    head_dim: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let head = wgid.x;
    if (head >= params.n_heads) { return; }
    let tid = lid.x;
    let hd = params.head_dim;
    let offset = head * hd;

    var ss: f32 = 0.0;
    for (var i = tid; i < hd; i = i + 256u) {
        let v = data[offset + i];
        ss = ss + v * v;
    }
    sdata[tid] = ss;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        workgroupBarrier();
    }

    let inv_rms = inverseSqrt(sdata[0] / f32(hd) + params.eps);
    for (var i = tid; i < hd; i = i + 256u) {
        data[offset + i] = data[offset + i] * wt[i] * inv_rms;
    }
}
