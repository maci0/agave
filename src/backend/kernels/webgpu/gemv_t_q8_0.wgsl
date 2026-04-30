// Transposed GEMV for Q8_0 3D weights: y[out_dim] = W^T @ x[in_dim].
// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks (row-major).
// Each row has ceil(out_dim / 32) blocks of 34 bytes (f16 scale + 32 x i8).
// One workgroup per output element. Threads stride over input rows,
// dequantizing W[i, col] and accumulating x[i] * dequant(W[i, col]).
// Uses array<u32> for byte-level access via bitwise extraction.

const WG_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

struct Params {
    out_dim: u32,
    in_dim: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

// Extract a single byte from a u32 word at the given byte position (0..3)
fn extract_byte(word: u32, byte_pos: u32) -> u32 {
    return (word >> (byte_pos * 8u)) & 0xFFu;
}

// Sign-extend an 8-bit value to i32
fn sign_extend_i8(val: u32) -> i32 {
    let signed = i32(val);
    if (signed > 127) {
        return signed - 256;
    }
    return signed;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x;
    let tid = lid.x;
    if (col >= params.out_dim) {
        return;
    }

    let blocks_per_row = (params.out_dim + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let row_bytes = blocks_per_row * BLOCK_BYTES;

    // Which block and offset within the block does column 'col' land in?
    let blk_idx = col / BLOCK_SIZE;
    let elem_in_blk = col % BLOCK_SIZE;

    // Byte offsets within a row to the target block's scale and quant
    let blk_byte_off = blk_idx * BLOCK_BYTES;
    let quant_byte_off = blk_byte_off + 2u + elem_in_blk;

    var sum: f32 = 0.0;

    // Each thread processes rows in stride
    for (var row = tid; row < params.in_dim; row += WG_SIZE) {
        let row_base = row * row_bytes;

        // Read f16 scale from first 2 bytes of the block
        let s_byte = row_base + blk_byte_off;
        let s_word = s_byte / 4u;
        let s_shift = s_byte % 4u;
        let raw_word = w_raw[s_word];
        var scale_bits: u32;
        if (s_shift <= 2u) {
            scale_bits = (raw_word >> (s_shift * 8u)) & 0xFFFFu;
        } else {
            // Straddles two u32 words
            let lo = (raw_word >> 24u) & 0xFFu;
            let hi = w_raw[s_word + 1u] & 0xFFu;
            scale_bits = lo | (hi << 8u);
        }
        let d = unpack2x16float(scale_bits).x;

        // Read single i8 quant value
        let q_byte = row_base + quant_byte_off;
        let q_word = w_raw[q_byte / 4u];
        let q_shift = q_byte % 4u;
        let qval_unsigned = extract_byte(q_word, q_shift);
        let qval = sign_extend_i8(qval_unsigned);

        sum += d * f32(qval) * x[row];
    }

    partial_sums[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        y[col] = partial_sums[0];
    }
}
