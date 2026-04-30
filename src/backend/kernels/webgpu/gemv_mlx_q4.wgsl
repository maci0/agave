struct Params {
    n: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_packed: array<u32>;
@group(0) @binding(2) var<storage, read> sc_packed: array<u32>;
@group(0) @binding(3) var<storage, read> bi_packed: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn bf16_to_f32(val16: u32) -> f32 {
    return bitcast<f32>(val16 << 16u);
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let group_size = 64u;
    let gpr = (params.k + group_size - 1u) / group_size;
    let words_per_group = group_size / 8u;
    let wpr = gpr * words_per_group;

    var sum: f32 = 0.0;
    for (var g = tid; g < gpr; g = g + 256u) {
        let sc_word = sc_packed[row * gpr / 2u + g / 2u];
        var scale: f32;
        if (g % 2u == 0u) { scale = bf16_to_f32(sc_word & 0xFFFFu); }
        else { scale = bf16_to_f32(sc_word >> 16u); }

        let bi_word = bi_packed[row * gpr / 2u + g / 2u];
        var bias: f32;
        if (g % 2u == 0u) { bias = bf16_to_f32(bi_word & 0xFFFFu); }
        else { bias = bf16_to_f32(bi_word >> 16u); }

        let xo = g * group_size;
        let wo = row * wpr + g * words_per_group;
        let elems = min(group_size, params.k - xo);

        var q_dot: f32 = 0.0;
        var x_sum: f32 = 0.0;
        for (var wi = 0u; wi < elems / 8u; wi = wi + 1u) {
            let word = w_packed[wo + wi];
            for (var ni = 0u; ni < 8u; ni = ni + 1u) {
                let nibble = (word >> (ni * 4u)) & 0xFu;
                let xv = x[xo + wi * 8u + ni];
                q_dot = q_dot + f32(nibble) * xv;
                x_sum = x_sum + xv;
            }
        }
        sum = sum + scale * q_dot + bias * x_sum;
    }

    partial[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { y[row] = partial[0]; }
}
