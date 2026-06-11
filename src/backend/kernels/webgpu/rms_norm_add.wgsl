struct Params {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;          // FFN output (read-only)
@group(0) @binding(1) var<storage, read> weight: array<f32>;     // norm weights
@group(0) @binding(2) var<storage, read_write> b: array<f32>;    // residual stream
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.n;

    // Phase 1: sum of squares of a
    var ss: f32 = 0.0;
    for (var i = tid; i < n; i = i + 256u) {
        ss = ss + a[i] * a[i];
    }
    sdata[tid] = ss;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        workgroupBarrier();
    }

    // Phase 2: b[i] += norm(a[i]) * weight[i]
    let inv_rms = inverseSqrt(sdata[0] / f32(n) + params.eps);
    for (var i = tid; i < n; i = i + 256u) {
        b[i] = b[i] + a[i] * weight[i] * inv_rms;
    }
}
