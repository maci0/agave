// IQ4_NL GEMV: y[row] = dot(dequant(W[row, :]), x)
// IQ4_NL: 18 bytes per block = f16 scale (2B) + 16 nibble bytes (32 values)
// Non-linear 16-entry LUT for dequant: nibble → i8 value

const WG_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;

struct Params { n: u32, k: u32, row_offset: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

const LUT = array<f32, 16>(
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0
);

fn rb(off: u32) -> u32 { return (w_raw[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }

fn rf16(off: u32) -> f32 {
    let w = w_raw[off / 4u];
    let s = (off % 4u) * 8u;
    var bits = (w >> s) & 0xFFFFu;
    if (s > 16u) { bits = ((w >> s) | (w_raw[off / 4u + 1u] << (32u - s))) & 0xFFFFu; }
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let nb = (params.k + BLOCK_ELEMS - 1u) / BLOCK_ELEMS;
    var acc: f32 = 0.0;

    var blk = tid;
    while (blk < nb) {
        let bk = blk * BLOCK_ELEMS;

        // Sparse skip: check if all 32 input values are near-zero
        var bmax: f32 = 0.0;
        for (var i = 0u; i < BLOCK_ELEMS; i += 4u) {
            let v = abs(vec4<f32>(x[bk+i], x[bk+i+1u], x[bk+i+2u], x[bk+i+3u]));
            bmax = max(bmax, max(max(v.x, v.y), max(v.z, v.w)));
        }
        if (bmax < 0.005) { blk += WG_SIZE; continue; }

        let bp = row * nb * BLOCK_BYTES + blk * BLOCK_BYTES;
        let d = rf16(bp);
        var block_sum: f32 = 0.0;

        for (var j: u32 = 0u; j < 16u; j++) {
            let byte_val = rb(bp + 2u + j);
            let col_lo = bk + j;
            let col_hi = bk + j + 16u;
            if (col_lo < params.k) { block_sum += x[col_lo] * LUT[byte_val & 0xFu]; }
            if (col_hi < params.k) { block_sum += x[col_hi] * LUT[byte_val >> 4u]; }
        }
        acc += block_sum * d;
        blk += WG_SIZE;
    }

    partial_sums[tid] = acc;
    workgroupBarrier();
    var stride: u32 = WG_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) { partial_sums[tid] += partial_sums[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    if (tid == 0u) { y[row] = partial_sums[0]; }
}
