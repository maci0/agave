// GEMV for MXFP4 E2M1 weights with per-group U8 scales.
// gs = 16 (NVIDIA) or 32 (MLX MoE experts). scale_fmt: 0 = FP8 E4M3, 1 = E8M0.
struct Params {
    n: u32,
    k: u32,
    row_offset: u32,
    gs: u32,
    scale_fmt: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_packed: array<u32>;
@group(0) @binding(2) var<storage, read> s_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn mxfp4_lut(nibble: u32) -> f32 {
    let t = array<f32, 16>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
    return t[nibble & 0xFu];
}

fn fp8e4m3(val: u32) -> f32 {
    let s = (val >> 7u) & 1u;
    let e = (val >> 3u) & 0xFu;
    let m = val & 0x7u;
    if (e == 0u) {
        if (m == 0u) { return 0.0; }
        let fv = f32(m) / 8.0 * exp2(-6.0);
        if (s == 1u) { return -fv; } else { return fv; }
    }
    if (e == 15u && m == 7u) { return 0.0; }
    let fv = (1.0 + f32(m) / 8.0) * exp2(f32(e) - 7.0);
    if (s == 1u) { return -fv; } else { return fv; }
}

fn e8m0_to_f32(val: u32) -> f32 {
    if (val == 0u) { return 0.0; }
    return bitcast<f32>(val << 23u);
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let k = params.k;
    let group_size = max(params.gs, 8u);
    let gpr = (k + group_size - 1u) / group_size;
    let wpg = group_size / 8u;
    let wpr = gpr * wpg;
    let row_w = row * wpr;

    var sum: f32 = 0.0;
    for (var g = tid; g < gpr; g = g + 256u) {
        let s_idx = row * gpr + g;
        let s_word = s_packed[s_idx / 4u];
        let s_byte = (s_word >> ((s_idx % 4u) * 8u)) & 0xFFu;
        var sc: f32;
        if (params.scale_fmt == 1u) {
            sc = e8m0_to_f32(s_byte);
        } else {
            sc = fp8e4m3(s_byte);
        }

        let xo = g * group_size;
        let wg = row_w + g * wpg;
        var gdot: f32 = 0.0;
        for (var w = 0u; w < wpg && xo + w * 8u < k; w = w + 1u) {
            let word = w_packed[wg + w];
            let xi = xo + w * 8u;
            let rem = min(8u, k - xi);
            for (var i = 0u; i < rem; i = i + 1u) {
                gdot = gdot + mxfp4_lut((word >> (i * 4u)) & 0xFu) * x[xi + i];
            }
        }
        sum = sum + sc * gdot;
    }

    partial[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { y[row] = partial[0]; }
}
