// Q4_1 quantized matrix-vector multiply: y[row] = dot(dequant(W[row, :]), x)
// Q4_1 format: 32 elements per block, 20 bytes: f16 scale (2B) + f16 min (2B) + 16 nibble bytes.
// Dequant: value = nibble * d + m (affine, no subtract-8).
// Factored: sum(x * (q*d + m)) = d * sum(x*q) + m * sum(x).
// One workgroup per output row. Threads process blocks in stride.

const WG_SIZE: u32 = 256u;
const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 20u;

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

fn extract_byte(word: u32, byte_pos: u32) -> u32 {
    return (word >> (byte_pos * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) {
        return;
    }

    let nb = params.k / BLOCK_ELEMS;
    // Each block is 20 bytes = 5 u32 words
    let words_per_block: u32 = 5u;
    let row_word_offset = row * nb * words_per_block;

    var acc: f32 = 0.0;
    var blk = tid;
    while (blk < nb) {
        let base = row_word_offset + blk * words_per_block;

        // Word 0: low 16 bits = f16 scale (d), high 16 bits = f16 min (m)
        let header = w_raw[base];
        let d_f16 = unpack2x16float(header & 0xFFFFu).x;
        let m_f16 = unpack2x16float(header >> 16u).x;

        // Words 1-4: 16 nibble-packed bytes (32 elements)
        let bk = blk * BLOCK_ELEMS;
        var qx_sum: f32 = 0.0;
        var x_sum: f32 = 0.0;

        // Process 4 bytes (8 elements) per u32 word
        for (var wi: u32 = 0u; wi < 4u; wi++) {
            let word = w_raw[base + 1u + wi];
            for (var bi: u32 = 0u; bi < 4u; bi++) {
                let byte_val = extract_byte(word, bi);
                let col_lo = bk + wi * 4u + bi;
                let col_hi = col_lo + 16u;
                let lo = f32(byte_val & 0xFu);
                let hi = f32(byte_val >> 4u);
                if (col_lo < params.k) {
                    qx_sum += lo * x[col_lo];
                    x_sum += x[col_lo];
                }
                if (col_hi < params.k) {
                    qx_sum += hi * x[col_hi];
                    x_sum += x[col_hi];
                }
            }
        }

        acc += qx_sum * d_f16 + x_sum * m_f16;
        blk += WG_SIZE;
    }

    // Workgroup reduction
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
