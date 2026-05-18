// FP8 E4M3 matrix-vector multiply: y[row] = dot(fp8_to_f32(W[row, :]), x)
// FP8 E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa. 1 byte per element.
// Packed 4 values per u32. One workgroup per output row.

const WG_SIZE: u32 = 256u;

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

// Convert FP8 E4M3 byte to f32
// Layout: S EEEE MMM (1+4+3 bits, bias=7, no inf, NaN=0x7F/0xFF)
fn fp8e4m3_to_f32(bits: u32) -> f32 {
    let sign = bits >> 7u;
    let exp_bits = (bits >> 3u) & 0xFu;
    let mant_bits = bits & 0x7u;

    // NaN: exp=15, mant=7
    if (exp_bits == 15u && mant_bits == 7u) {
        return 0.0;
    }

    var result: f32;
    if (exp_bits == 0u) {
        // Subnormal: value = (-1)^s * 2^(-6) * (0.mant)
        result = f32(mant_bits) / 512.0;
    } else {
        // Normal: value = (-1)^s * 2^(exp-7) * (1.mant)
        let mantissa = 1.0 + f32(mant_bits) / 8.0;
        let exponent = f32(i32(exp_bits) - 7);
        result = mantissa * exp2(exponent);
    }

    if (sign != 0u) {
        return -result;
    }
    return result;
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

    // Each row has k fp8 elements = k/4 u32 words
    let words_per_row = (params.k + 3u) / 4u;
    let row_word_offset = row * words_per_row;

    var acc: f32 = 0.0;
    var word_idx = tid;
    while (word_idx < words_per_row) {
        let packed = w_raw[row_word_offset + word_idx];
        let col = word_idx * 4u;

        let v0 = fp8e4m3_to_f32(packed & 0xFFu);
        acc += v0 * x[col];

        if (col + 1u < params.k) {
            let v1 = fp8e4m3_to_f32((packed >> 8u) & 0xFFu);
            acc += v1 * x[col + 1u];
        }
        if (col + 2u < params.k) {
            let v2 = fp8e4m3_to_f32((packed >> 16u) & 0xFFu);
            acc += v2 * x[col + 2u];
        }
        if (col + 3u < params.k) {
            let v3 = fp8e4m3_to_f32((packed >> 24u) & 0xFFu);
            acc += v3 * x[col + 3u];
        }

        word_idx += WG_SIZE;
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
