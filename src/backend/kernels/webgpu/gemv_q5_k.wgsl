// Q5_K GEMV: y[row] = dot(dequant(W[row,:]), x)
// Q5_K super-block: 176 bytes, 256 values.
// Layout: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128].
// Dequant: val = d*sc*(lo|hi<<4) - dmin*m

const WG_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 256u;
const BLOCK_BYTES: u32 = 176u;

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

fn getScaleMinK4(j: u32, scales_base: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(read_byte(scales_base, j) & 63u, read_byte(scales_base, j + 4u) & 63u);
    }
    let sc = (read_byte(scales_base, j + 4u) & 0xFu) | ((read_byte(scales_base, j - 4u) >> 6u) << 4u);
    let m = (read_byte(scales_base, j + 4u) >> 4u) | ((read_byte(scales_base, j) >> 6u) << 4u);
    return vec2<u32>(sc, m);
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
        let d = read_f16(bp, 0u);
        let dmin = read_f16(bp, 2u);
        let scales_base = bp + 4u;
        let qh_base = bp + 16u;
        let qs_base = bp + 48u;

        for (var g = 0u; g < 4u; g++) {
            let gi_base = bk + g * 64u;
            if (gi_base >= params.k) { break; }
            let shift = g * 2u;
            let umask1 = 1u << shift;
            let umask2 = 2u << shift;
            let sm_a = getScaleMinK4(g * 2u, scales_base);
            let sm_b = getScaleMinK4(g * 2u + 1u, scales_base);
            let d_sc_a = d * f32(sm_a.x);
            let dm_m_a = dmin * f32(sm_a.y);
            let d_sc_b = d * f32(sm_b.x);
            let dm_m_b = dmin * f32(sm_b.y);

            for (var l = 0u; l < 32u; l++) {
                let gi = gi_base + l;
                if (gi >= params.k) { break; }
                let lo = f32(read_byte(qs_base, g * 32u + l) & 0xFu);
                let qh_byte = read_byte(qh_base, l);
                var hi: f32 = 0.0;
                if ((qh_byte & umask1) != 0u) { hi = 16.0; }
                sum += x[gi] * (d_sc_a * (lo + hi) - dm_m_a);
            }
            for (var l = 0u; l < 32u; l++) {
                let gi = gi_base + 32u + l;
                if (gi >= params.k) { break; }
                let lo = f32(read_byte(qs_base, g * 32u + l) >> 4u);
                let qh_byte = read_byte(qh_base, l);
                var hi: f32 = 0.0;
                if ((qh_byte & umask2) != 0u) { hi = 16.0; }
                sum += x[gi] * (d_sc_b * (lo + hi) - dm_m_b);
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
