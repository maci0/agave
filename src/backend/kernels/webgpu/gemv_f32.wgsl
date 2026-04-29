// Matrix-vector multiply for f32 weights: y[row] = dot(W[row, :], x)
// One workgroup per output row. Threads stride over k, then reduce via shared memory.

const WG_SIZE: u32 = 256u;

struct Params {
    n: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x;
    let tid = lid.x;
    if (row >= params.n) {
        return;
    }

    // Each thread accumulates partial dot product
    var sum: f32 = 0.0;
    let row_off = row * params.k;
    for (var j = tid; j < params.k; j += WG_SIZE) {
        sum += w[row_off + j] * x[j];
    }
    partial_sums[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final result
    if (tid == 0u) {
        y[row] = partial_sums[0];
    }
}
