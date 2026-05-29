// Q4_0 GEMV: y[row] = dot(dequant(W[row,:]), x)
// Q4_0: 32 elements per block, 18 bytes: f16 scale + 16 bytes (32 x 4-bit nibbles).
// Dequant: val = (nibble - 8) * scale

const WG_SIZE: u32 = 256u;
const BLOCK_SIZE: u32 = 32u;
const BLOCK_BYTES: u32 = 18u;

struct Params { n: u32, k: u32, row_offset: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x + params.row_offset;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let nb = (params.k + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    var sum: f32 = 0.0;

    for (var b = tid; b < nb; b += WG_SIZE) {
        let bk = b * BLOCK_SIZE;

        // Sparse skip: check if all 32 input values are near-zero
        var bmax: f32 = 0.0;
        for (var i = 0u; i < BLOCK_SIZE; i += 4u) {
            let v = abs(vec4<f32>(x[bk+i], x[bk+i+1u], x[bk+i+2u], x[bk+i+3u]));
            bmax = max(bmax, max(max(v.x, v.y), max(v.z, v.w)));
        }
        if (bmax < 0.005) { continue; }

        let block_byte_off = row * nb * BLOCK_BYTES + b * BLOCK_BYTES;
        let word_off = block_byte_off / 4u;
        let byte_in_word = block_byte_off % 4u;

        // f16 scale from first 2 bytes
        let raw_word = w_raw[word_off];
        var scale_bits: u32;
        if (byte_in_word <= 2u) {
            scale_bits = (raw_word >> (byte_in_word * 8u)) & 0xFFFFu;
        } else {
            scale_bits = ((raw_word >> 24u) & 0xFFu) | ((w_raw[word_off + 1u] & 0xFFu) << 8u);
        }
        let d = unpack2x16float(scale_bits).x;

        var block_sum: f32 = 0.0;
        for (var j = 0u; j < 16u; j++) {
            let qbyte_off = block_byte_off + 2u + j;
            let qword = w_raw[qbyte_off / 4u];
            let byte_val = (qword >> ((qbyte_off % 4u) * 8u)) & 0xFFu;
            let lo = f32(i32(byte_val & 0xFu) - 8);
            let hi = f32(i32(byte_val >> 4u) - 8);
            if (bk + j < params.k) { block_sum += lo * x[bk + j]; }
            if (bk + j + 16u < params.k) { block_sum += hi * x[bk + j + 16u]; }
        }
        sum += block_sum * d;
    }

    partial_sums[tid] = sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial_sums[tid] += partial_sums[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) { y[row] = partial_sums[0]; }
}
