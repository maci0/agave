struct Params {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.n;

    var ss: f32 = 0.0;
    for (var i = tid; i < n; i = i + 256u) {
        ss = ss + data[i] * data[i];
    }
    shared[tid] = ss;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shared[tid] = shared[tid] + shared[tid + s]; }
        workgroupBarrier();
    }

    let inv_norm = inverseSqrt(shared[0] + params.eps);
    for (var i = tid; i < n; i = i + 256u) {
        data[i] = data[i] * inv_norm;
    }
}
