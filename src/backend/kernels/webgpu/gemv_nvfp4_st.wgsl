struct Params {
    n: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_packed: array<u32>;
@group(0) @binding(2) var<storage, read> s_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn mxfp4_lut(nibble: u32) -> f32 {
    let t = array<f32, 16>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
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
    if (e == 15u) { return 0.0; }
    let fv = (1.0 + f32(m) / 8.0) * exp2(f32(e) - 7.0);
    if (s == 1u) { return -fv; } else { return fv; }
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let k = params.k;
    let bytes_per_row = k / 2u;
    let scales_per_row = k / 16u;

    var sum: f32 = 0.0;
    for (var g = tid; g < scales_per_row; g = g + 256u) {
        let s_idx = row * scales_per_row + g;
        let s_word = s_packed[s_idx / 4u];
        let s_byte = (s_word >> ((s_idx % 4u) * 8u)) & 0xFFu;
        let sc = fp8e4m3(s_byte);

        let base = g * 16u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let w_byte_idx = row * bytes_per_row + g * 8u + j;
            let w_word = w_packed[w_byte_idx / 4u];
            let byte_val = (w_word >> ((w_byte_idx % 4u) * 8u)) & 0xFFu;
            let v0 = mxfp4_lut(byte_val & 0xFu) * sc;
            let v1 = mxfp4_lut(byte_val >> 4u) * sc;
            sum = sum + v0 * x[base + 2u * j] + v1 * x[base + 2u * j + 1u];
        }
    }

    partial[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
    }
    if (tid == 0u) { y[row] = partial[0]; }
}
