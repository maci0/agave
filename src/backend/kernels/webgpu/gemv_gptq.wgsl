// GPTQ INT4 GEMV: y[row] = dequant(qweight[row,:]) @ x
// 8 INT4 nibbles per u32, FP16 per-group scales, INT4 packed zero-points.

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

var<workgroup> sdata: array<f32, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let k = params.k;
    let group_size = params.group_size;
    let words_per_row = k / 8u;
    let n_groups = (k + group_size - 1u) / group_size;
    var sum: f32 = 0.0;

    for (var wi = tid; wi < words_per_row; wi = wi + 256u) {
        let word = qweight[row * words_per_row + wi];
        let elem_base = wi * 8u;
        let g = elem_base / group_size;

        // Read f16 scale from packed u16 pairs
        let scale_idx = row * n_groups + g;
        let scale_word = scales[scale_idx / 2u];
        let scale_bits = select(scale_word >> 16u, scale_word & 0xFFFFu, scale_idx % 2u == 0u);
        let scale_val = unpack2x16float(scale_bits).x;

        // Zero-point
        let z_word_idx = g * ((params.n + 7u) / 8u) + row / 8u;
        let z_nibble = row % 8u;
        let zero = f32((qzeros[z_word_idx] >> (z_nibble * 4u)) & 0xFu);

        var local_sum: f32 = 0.0;
        local_sum = local_sum + (f32((word >> 0u)  & 0xFu) - zero) * scale_val * x_data[elem_base + 0u];
        local_sum = local_sum + (f32((word >> 4u)  & 0xFu) - zero) * scale_val * x_data[elem_base + 1u];
        local_sum = local_sum + (f32((word >> 8u)  & 0xFu) - zero) * scale_val * x_data[elem_base + 2u];
        local_sum = local_sum + (f32((word >> 12u) & 0xFu) - zero) * scale_val * x_data[elem_base + 3u];
        local_sum = local_sum + (f32((word >> 16u) & 0xFu) - zero) * scale_val * x_data[elem_base + 4u];
        local_sum = local_sum + (f32((word >> 20u) & 0xFu) - zero) * scale_val * x_data[elem_base + 5u];
        local_sum = local_sum + (f32((word >> 24u) & 0xFu) - zero) * scale_val * x_data[elem_base + 6u];
        local_sum = local_sum + (f32((word >> 28u) & 0xFu) - zero) * scale_val * x_data[elem_base + 7u];
        sum = sum + local_sum;
    }

    // Workgroup reduction
    sdata[tid % 8u] = sum;
    workgroupBarrier();
    if (tid < 8u) {
        var v = sdata[tid];
        for (var i = 0u; i < 8u; i = i + 1u) { v = v + sdata[i]; }
        if (tid == 0u) { y_data[row] = v; }
    }
}
