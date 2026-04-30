// In-place softmax over n elements.
// Three phases: (1) find max, (2) exp(x - max) and sum, (3) normalize.
// One workgroup, all threads cooperate via sdata memory reduction.

const WG_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823466e+38;

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;

    // Phase 1: find max
    var mx = NEG_INF;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        mx = max(mx, data[i]);
    }
    sdata[tid] = mx;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        workgroupBarrier();
    }
    let max_val = sdata[0];
    workgroupBarrier();

    // Phase 2: exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        let v = exp(data[i] - max_val);
        data[i] = v;
        local_sum += v;
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / sdata[0];
    workgroupBarrier();

    // Phase 3: normalize
    for (var i = tid; i < params.n; i += WG_SIZE) {
        data[i] *= inv_sum;
    }
}
