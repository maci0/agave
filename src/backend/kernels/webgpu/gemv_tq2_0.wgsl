// TQ2_0 ternary 2-bit GEMV: y[row] = dot(dequant(W[row, :]), x)
// TQ2_0 format: 256 elements per block, 66 bytes per block:
//   f16 scale (2 bytes) + 64 bytes qs (4 values per byte, 256 elems).
// Values are {-1, 0, +1}: decoded as ((byte >> (slot*2)) & 3) - 1.
// One workgroup per output row. Threads process blocks in stride.

const WG_SIZE:    u32 = 256u;
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 66u;
const SCALE_BYTES: u32 = 2u;
const QS_BYTES:    u32 = 64u;

struct Params {
    n:          u32,
    k:          u32,
    row_offset: u32,
    _pad:       u32,
}

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       w_raw:  array<u32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

var<workgroup> partial_sums: array<f32, 256>;

// Extract a single byte from a u32 word at the given byte position (0..3).
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
    let word_off     = byte_off / 4u;
    let byte_in_word = byte_off % 4u;
    let raw_word     = w_raw[word_off];
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
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let row = wg_id.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) {
        return;
    }

    let nb  = (params.k + BLOCK_ELEMS - 1u) / BLOCK_ELEMS;
    var sum: f32 = 0.0;

    for (var b = tid; b < nb; b += WG_SIZE) {
        let bk             = b * BLOCK_ELEMS;
        let block_byte_off = row * nb * BLOCK_BYTES + b * BLOCK_BYTES;

        let scale  = read_scale(block_byte_off);
        let qs_off = block_byte_off + SCALE_BYTES;

        // 64 bytes × 4 values/byte = 256 elements per block
        for (var bi = 0u; bi < QS_BYTES; bi++) {
            let bv = read_byte(qs_off + bi);
            for (var slot = 0u; slot < 4u; slot++) {
                let elem = bi * 4u + slot;
                let xi   = bk + elem;
                if (xi < params.k) {
                    let xv = x[xi];
                    // Sparse skip: avoid multiply when x is near zero
                    if (abs(xv) >= 0.005) {
                        let q = i32((bv >> (slot * 2u)) & 3u) - 1;
                        sum += f32(q) * scale * xv;
                    }
                }
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
