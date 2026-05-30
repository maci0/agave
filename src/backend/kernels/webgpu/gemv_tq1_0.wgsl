// TQ1_0 ternary 1.58-bit GEMV: y[row] = dot(dequant(W[row, :]), x)
// TQ1_0 format: 256 elements per block, 54 bytes per block:
//   f16 scale (2 bytes) + 48 bytes (5 trits/byte, 240 elems) + 4 bytes (4 trits/byte, 16 elems).
// Values are {-1, 0, +1}: decoded as (trit - 1) * scale.
// One workgroup per output row. Threads process blocks in stride.

const WG_SIZE: u32 = 256u;
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 54u;
const SCALE_BYTES: u32 = 2u;
const PACKED_BYTES_5: u32 = 48u;
const PACKED_BYTES_4: u32 = 4u;

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

// Read a single byte from the raw u32 array at a given byte offset.
fn read_byte(byte_off: u32) -> u32 {
    let word = w_raw[byte_off / 4u];
    return extract_byte(word, byte_off % 4u);
}

// Unpack f16 from two bytes at byte_off (may straddle u32 boundary).
fn read_scale(byte_off: u32) -> f32 {
    let word_off = byte_off / 4u;
    let byte_in_word = byte_off % 4u;
    let raw_word = w_raw[word_off];
    var scale_bits: u32;
    if (byte_in_word <= 2u) {
        scale_bits = (raw_word >> (byte_in_word * 8u)) & 0xFFFFu;
    } else {
        // Straddles two u32 words (byte_in_word == 3)
        let lo = (raw_word >> 24u) & 0xFFu;
        let hi = w_raw[word_off + 1u] & 0xFFu;
        scale_bits = lo | (hi << 8u);
    }
    return unpack2x16float(scale_bits).x;
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

    let nb = (params.k + BLOCK_ELEMS - 1u) / BLOCK_ELEMS;
    var sum: f32 = 0.0;

    for (var b = tid; b < nb; b += WG_SIZE) {
        let bk = b * BLOCK_ELEMS;
        let block_byte_off = row * nb * BLOCK_BYTES + b * BLOCK_BYTES;

        let scale = read_scale(block_byte_off);
        let trit_off = block_byte_off + SCALE_BYTES;

        var elem: u32 = 0u;

        // First 240 elements: 5 trits per byte, 48 bytes
        for (var bi = 0u; bi < PACKED_BYTES_5; bi++) {
            let bv = read_byte(trit_off + bi);
            if (bv < 243u) {
                var rem = i32(bv);
                for (var ti = 0u; ti < 5u; ti++) {
                    if (bk + elem < params.k) {
                        let trit = (rem % 3) - 1;
                        sum += f32(trit) * scale * x[bk + elem];
                    }
                    rem /= 3;
                    elem++;
                }
            } else {
                elem += 5u;
            }
        }

        // Last 16 elements: 4 trits per byte, 4 bytes
        for (var bi = 0u; bi < PACKED_BYTES_4; bi++) {
            let bv = read_byte(trit_off + PACKED_BYTES_5 + bi);
            var rem = i32(bv);
            for (var ti = 0u; ti < 4u; ti++) {
                if (bk + elem < params.k) {
                    let trit = (rem % 3) - 1;
                    sum += f32(trit) * scale * x[bk + elem];
                }
                rem /= 3;
                elem++;
            }
        }
    }

    partial_sums[tid] = sum;
    workgroupBarrier();

    // Tree reduction
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
