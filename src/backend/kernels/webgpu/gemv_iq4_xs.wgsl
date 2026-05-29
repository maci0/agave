// IQ4_XS super-block GEMV: y[row] = dot(dequant(W[row, :]), x)
// 136 bytes = f16 d + u16 scales_h + u8 scales_l[4] + u8 qs[128]
// 256 values, 8 sub-blocks of 32, IQ4_NL LUT + per-sub-block 6-bit scales

const WG_SIZE: u32 = 256u;
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
const SCALE_BIAS: i32 = -32;

fn rb(off: u32) -> u32 { return (w_raw[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu; }

fn rf16(off: u32) -> f32 {
    let w = w_raw[off / 4u]; let s = (off % 4u) * 8u;
    var bits = (w >> s) & 0xFFFFu;
    if (s > 16u) { bits = ((w >> s) | (w_raw[off / 4u + 1u] << (32u - s))) & 0xFFFFu; }
    return unpack2x16float(bits).x;
}

fn ru16(off: u32) -> u32 {
    let w = w_raw[off / 4u]; let s = (off % 4u) * 8u;
    var bits = (w >> s) & 0xFFFFu;
    if (s > 16u) { bits = ((w >> s) | (w_raw[off / 4u + 1u] << (32u - s))) & 0xFFFFu; }
    return bits;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let nb = (params.k + 255u) / 256u;
    var acc: f32 = 0.0;

    var blk = tid;
    while (blk < nb) {
        let bk = blk * 256u;

        // Sparse skip: check if all 256 input values are near-zero
        var bmax: f32 = 0.0;
        let check_end = min(256u, params.k - bk);
        for (var i = 0u; i < check_end; i += 4u) {
            let v = abs(vec4<f32>(x[bk+i], x[bk+i+1u], x[bk+i+2u], x[bk+i+3u]));
            bmax = max(bmax, max(max(v.x, v.y), max(v.z, v.w)));
        }
        if (bmax < 0.005) { blk += WG_SIZE; continue; }

        let bp = row * nb * 136u + blk * 136u;
        let d = rf16(bp);
        let scales_h = ru16(bp + 2u);

        for (var sb: u32 = 0u; sb < 8u; sb++) {
            let sl_byte = rb(bp + 4u + sb / 2u);
            let lo4 = select(sl_byte & 0xFu, sl_byte >> 4u, sb % 2u != 0u);
            let hi2 = (scales_h >> (sb * 2u)) & 0x3u;
            let sub_scale = d * f32(i32(lo4 | (hi2 << 4u)) + SCALE_BIAS);
            let sub_bk = bk + sb * 32u;
            var block_sum: f32 = 0.0;

            for (var j: u32 = 0u; j < 16u; j++) {
                let bv = rb(bp + 8u + sb * 16u + j);
                let cl = sub_bk + j;
                let ch = sub_bk + j + 16u;
                if (cl < params.k) { block_sum += x[cl] * LUT[bv & 0xFu]; }
                if (ch < params.k) { block_sum += x[ch] * LUT[bv >> 4u]; }
            }
            acc += block_sum * sub_scale;
        }
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
