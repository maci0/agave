// Q6_K GEMV: y[row] = dot(dequant(W[row,:]), x)
// Q6_K super-block: 210 bytes, 256 values.
// Layout: ql[128] + qh[64] + scales[16](i8) + d(f16).
// 3 bits from ql + 2 bits from qh = 6-bit value, bias -32.

const WG_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

struct Params { n: u32, k: u32, row_offset: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

fn read_byte(base: u32, offset: u32) -> u32 {
    let addr = base + offset;
    return (w_raw[addr / 4u] >> ((addr % 4u) * 8u)) & 0xFFu;
}

fn read_f16(base: u32, offset: u32) -> f32 {
    let addr = base + offset;
    let word = w_raw[addr / 4u];
    let shift = (addr % 4u) * 8u;
    var bits: u32;
    if (shift <= 16u) { bits = (word >> shift) & 0xFFFFu; }
    else { bits = ((word >> 24u) & 0xFFu) | ((w_raw[addr / 4u + 1u] & 0xFFu) << 8u); }
    return unpack2x16float(bits).x;
}

fn sign_extend_i8(val: u32) -> i32 {
    let s = i32(val);
    if (s > 127) { return s - 256; }
    return s;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let nb = (params.k + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    var sum: f32 = 0.0;

    for (var b = tid; b < nb; b += WG_SIZE) {
        let bk = b * BLOCK_SIZE;

        // Sparse skip: check if all 256 input values are near-zero
        var bmax: f32 = 0.0;
        let check_end = min(BLOCK_SIZE, params.k - bk);
        for (var i = 0u; i < check_end; i += 4u) {
            let v = abs(vec4<f32>(x[bk+i], x[bk+i+1u], x[bk+i+2u], x[bk+i+3u]));
            bmax = max(bmax, max(max(v.x, v.y), max(v.z, v.w)));
        }
        if (bmax < 0.005) { continue; }

        let bp = row * nb * BLOCK_BYTES + b * BLOCK_BYTES;
        let d = read_f16(bp, 208u);

        // 2 chunks of 128 elements each
        for (var chunk = 0u; chunk < 2u; chunk++) {
            let ql_off = chunk * 64u;
            let qh_off = 128u + chunk * 32u;
            let sc_off = 192u + chunk * 8u;
            let base = bk + chunk * 128u;

            for (var l = 0u; l < 32u; l++) {
                let is = l / 16u;
                let ql_byte = read_byte(bp, ql_off + l);
                let ql32_byte = read_byte(bp, ql_off + l + 32u);
                let qh_byte = read_byte(bp, qh_off + l);

                let q1 = i32((ql_byte & 0xFu) | ((qh_byte & 3u) << 4u)) - 32;
                let q2 = i32((ql32_byte & 0xFu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((ql_byte >> 4u) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((ql32_byte >> 4u) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;

                let sc1 = f32(sign_extend_i8(read_byte(bp, sc_off + is)));
                let sc2 = f32(sign_extend_i8(read_byte(bp, sc_off + is + 2u)));
                let sc3 = f32(sign_extend_i8(read_byte(bp, sc_off + is + 4u)));
                let sc4 = f32(sign_extend_i8(read_byte(bp, sc_off + is + 6u)));

                let gi0 = base + l;
                let gi1 = base + l + 32u;
                let gi2 = base + l + 64u;
                let gi3 = base + l + 96u;

                if (gi0 < params.k) { sum += x[gi0] * d * sc1 * f32(q1); }
                if (gi1 < params.k) { sum += x[gi1] * d * sc2 * f32(q2); }
                if (gi2 < params.k) { sum += x[gi2] * d * sc3 * f32(q3); }
                if (gi3 < params.k) { sum += x[gi3] * d * sc4 * f32(q4); }
            }
        }
    }

    partial_sums[tid] = sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial_sums[tid] += partial_sums[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) { y[row] = partial_sums[0]; }
}
