// AWQ INT4 GEMV: y[col] = sum_k dequant(qweight[k, col]) * x[k]
// GEMM interleaved nibble order: [0,2,4,6,1,3,5,7]
// Column-major packing: qweight[k, n/8], scales[groups, n], qzeros[groups, n/8]

struct Params {
    n: u32,
    k: u32,
    group_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x_data: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> qzeros: array<u32>;
@group(0) @binding(4) var<storage, read_write> y_data: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

// GEMM reverse map: output column index → nibble shift position / 4
const AWQ_REV = array<u32, 8>(0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col = wgid.x;
    let tid = lid.x;
    if (col >= params.n) { return; }

    let k = params.k;
    let group_size = params.group_size;
    let n_words = params.n / 8u;
    let word_idx = col / 8u;
    let shift = AWQ_REV[col % 8u] * 4u;
    var sum: f32 = 0.0;

    for (var ki = tid; ki < k; ki = ki + 256u) {
        let xv = x_data[ki];
        if (abs(xv) < 0.005) { continue; }

        let word = qweight[ki * n_words + word_idx];
        let nibble = f32((word >> shift) & 0xFu);

        let g = ki / group_size;
        let z_word = qzeros[g * n_words + word_idx];
        let zero = f32((z_word >> shift) & 0xFu);

        // Scales in natural order — read f16 from packed u32
        let scale_idx = g * params.n + col;
        let scale_word = scales[scale_idx / 2u];
        let scale_bits = select(scale_word >> 16u, scale_word & 0xFFFFu, scale_idx % 2u == 0u);
        let scale = unpack2x16float(scale_bits).x;

        sum = sum + (nibble - zero) * scale * xv;
    }

    // Workgroup reduction
    sdata[tid] = sum;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (tid == 0u) { y_data[col] = sdata[0u]; }
}
