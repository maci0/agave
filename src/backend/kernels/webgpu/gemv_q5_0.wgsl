// Q5_0 quantized matrix-vector multiply: y[row] = dot(dequant(W[row, :]), x)
// Q5_0 format: 22 bytes per block = f16 d (2B) + u32 qh (4B) + 16 nibble bytes (32 elems).
// Dequant: value = ((lo_nibble | (qh_bit << 4)) - 16) * d
// One workgroup per output row. Threads process blocks in stride.

const WG_SIZE: u32 = 256u;
const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 22u;

struct Params {
    n: u32,
    k: u32,
    row_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

fn read_u16_unaligned(byte_off: u32) -> u32 {
    let word = w_raw[byte_off / 4u];
    let shift = (byte_off % 4u) * 8u;
    var bits = (word >> shift) & 0xFFFFu;
    if (shift > 16u) {
        bits = bits | ((w_raw[byte_off / 4u + 1u] << (32u - shift)) & 0xFFFFu);
    }
    return bits;
}

fn read_u32_unaligned(byte_off: u32) -> u32 {
    let word = w_raw[byte_off / 4u];
    let shift = (byte_off % 4u) * 8u;
    if (shift == 0u) { return word; }
    return (word >> shift) | (w_raw[byte_off / 4u + 1u] << (32u - shift));
}

fn read_byte(byte_off: u32) -> u32 {
    return (w_raw[byte_off / 4u] >> ((byte_off % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x + params.row_offset;
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

        let block_byte_off = row * nb * BLOCK_BYTES + blk * BLOCK_BYTES;

        let d = unpack2x16float(read_u16_unaligned(block_byte_off)).x;
        let qh = read_u32_unaligned(block_byte_off + 2u);
        var block_sum: f32 = 0.0;

        for (var j: u32 = 0u; j < 16u; j++) {
            let byte_val = read_byte(block_byte_off + 6u + j);
            let lo_nibble = byte_val & 0xFu;
            let hi_nibble = byte_val >> 4u;
            let qh_lo = (qh >> j) & 1u;
            let qh_hi = (qh >> (j + 16u)) & 1u;

            let val_lo = f32(i32((lo_nibble | (qh_lo << 4u))) - 16) * d;
            let val_hi = f32(i32((hi_nibble | (qh_hi << 4u))) - 16) * d;

            let col_lo = bk + j;
            let col_hi = bk + j + 16u;
            if (col_lo < params.k) { block_sum += val_lo * x[col_lo]; }
            if (col_hi < params.k) { block_sum += val_hi * x[col_hi]; }
        }
        acc += block_sum;
        blk += WG_SIZE;
    }

    partial_sums[tid] = acc;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        y[row] = partial_sums[0];
    }
}
