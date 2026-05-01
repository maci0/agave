// Q8_0 quantized matrix-vector multiply: y[row] = dot(dequant(W[row, :]), x)
// Q8_0 format: 32 elements per block, 34 bytes per block: f16 scale (2 bytes) + i8 quants[32].
// One workgroup per output row. Threads process blocks in stride.
// Uses array<u32> for byte-level access via bitwise extraction.

const WG_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

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

// Unpack f16 from two bytes stored in the low 16 bits of a u32
fn unpack_f16_scale(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
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

    let nb = (params.k + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    var sum: f32 = 0.0;

    for (var b = tid; b < nb; b += WG_SIZE) {
        // Locate this block in the raw byte stream
        let block_byte_off = row * nb * BLOCK_BYTES + b * BLOCK_BYTES;
        let word_off = block_byte_off / 4u;
        let byte_in_word = block_byte_off % 4u;

        // Read f16 scale from first 2 bytes of the block
        let raw_word = w_raw[word_off];
        var scale_bits: u32;
        if (byte_in_word <= 2u) {
            // Scale fits within one u32 word
            scale_bits = (raw_word >> (byte_in_word * 8u)) & 0xFFFFu;
        } else {
            // Scale straddles two u32 words (byte_in_word == 3)
            let lo = (raw_word >> 24u) & 0xFFu;
            let hi = w_raw[word_off + 1u] & 0xFFu;
            scale_bits = lo | (hi << 8u);
        }
        let d = unpack2x16float(scale_bits).x;

        // Process 32 int8 quantized values
        var block_sum: f32 = 0.0;
        let bk = b * BLOCK_SIZE;
        for (var j = 0u; j < BLOCK_SIZE; j++) {
            if (bk + j >= params.k) {
                break;
            }
            let qbyte_off = block_byte_off + 2u + j;
            let qword = w_raw[qbyte_off / 4u];
            let qshift = (qbyte_off % 4u);
            let qval_unsigned = extract_byte(qword, qshift);
            let qval = sign_extend_i8(qval_unsigned);
            block_sum += f32(qval) * x[bk + j];
        }
        sum += block_sum * d;
    }

    partial_sums[tid] = sum;
    workgroupBarrier();

    // Tree reduction in sdata memory
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        y[row] = partial_sums[0];
    }
}
