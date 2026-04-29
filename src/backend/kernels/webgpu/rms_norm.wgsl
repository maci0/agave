// RMS normalization: out[i] = x[i] * w[i] * rsqrt(mean(x^2) + eps)
// One workgroup processes one normalization. Threads stride over n.
// Uses shared memory tree reduction (no subgroup intrinsics).

const WG_SIZE: u32 = 256u;

struct Params {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;

    // Pass 1: each thread accumulates sum of squares over its strided elements
    var ss: f32 = 0.0;
    for (var i = tid; i < params.n; i += WG_SIZE) {
        let v = inp[i];
        ss += v * v;
    }
    shared[tid] = ss;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        workgroupBarrier();
    }

    // Pass 2: normalize with weight
    let inv_rms = inverseSqrt(shared[0] / f32(params.n) + params.eps);
    for (var i = tid; i < params.n; i += WG_SIZE) {
        out_data[i] = inp[i] * wt[i] * inv_rms;
    }
}
