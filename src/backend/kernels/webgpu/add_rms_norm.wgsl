struct Params {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.n;

    // Phase 1: add residual in-place + compute sum of squares
    var ss: f32 = 0.0;
    for (var i = tid; i < n; i = i + 256u) {
        let v = data[i] + residual[i];
        data[i] = v;
        ss = ss + v * v;
    }
    shared[tid] = ss;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shared[tid] = shared[tid] + shared[tid + s]; }
        workgroupBarrier();
    }

    // Phase 2: normalize
    let inv_rms = inverseSqrt(shared[0] / f32(n) + params.eps);
    for (var i = tid; i < n; i = i + 256u) {
        out[i] = data[i] * weight[i] * inv_rms;
    }
}
