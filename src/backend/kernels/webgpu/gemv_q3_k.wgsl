// Q3_K super-block GEMV: y[row] = dot(dequant(W[row, :]), x)
// Q3_K: 110 bytes = 32B hmask + 64B qs + 12B scales + 2B f16 d
// 256 values per super-block, 16 groups of 16 elements
// 3-bit: 2 bits from qs + 1 bit from hmask
// Dequant: val = d * scale * ((q_lo | (q_hi << 2)) - 4)

const WG_SIZE: u32 = 256u;

struct Params { n: u32, k: u32, row_offset: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

fn rb(byte_off: u32) -> u32 { return (w_raw[byte_off / 4u] >> ((byte_off % 4u) * 8u)) & 0xFFu; }

fn rf16(byte_off: u32) -> f32 {
    let w = w_raw[byte_off / 4u];
    let s = (byte_off % 4u) * 8u;
    var bits = (w >> s) & 0xFFFFu;
    if (s > 16u) { bits = ((w >> s) | (w_raw[byte_off / 4u + 1u] << (32u - s))) & 0xFFFFu; }
    return unpack2x16float(bits).x;
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

        let bp = row * nb * 110u + blk * 110u;
        let d = rf16(bp + 108u);

        for (var g: u32 = 0u; g < 16u; g++) {
            let base = bk + g * 16u;
            if (base >= params.k) { break; }

            let scale_idx = select(g, g - 8u, g >= 8u);
            let sc_byte = rb(bp + 96u + scale_idx);
            let sn = select(sc_byte & 0xFu, sc_byte >> 4u, g >= 8u);
            let d_sc = d * f32(i32(sn) - 8);

            for (var l: u32 = 0u; l < 16u; l++) {
                if (base + l >= params.k) { break; }
                let fi = g * 16u + l;
                let qs_bi = fi / 4u;
                let qs_bs = (fi % 4u) * 2u;
                let q_lo = (rb(bp + 32u + qs_bi) >> qs_bs) & 0x3u;

                let hm_bi = fi % 32u;
                let hm_bit = fi / 32u;
                let q_hi = (rb(bp + hm_bi) >> hm_bit) & 1u;

                let q3 = i32(q_lo | (q_hi << 2u)) - 4;
                acc += x[base + l] * d_sc * f32(q3);
            }
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
