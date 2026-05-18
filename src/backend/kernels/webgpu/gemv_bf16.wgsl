// BF16 matrix-vector multiply: y[row] = dot(bf16_to_f32(W[row, :]), x)
// BF16 format: 2 bytes per element, packed as u16 in u32 words.
// One workgroup per output row. Threads process elements in stride.

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

// Convert BF16 (stored in low 16 bits) to f32 by shifting left 16 bits
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
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

    // Each row has k bf16 elements = k/2 u32 words
    let words_per_row = (params.k + 1u) / 2u;
    let row_word_offset = row * words_per_row;

    var acc: f32 = 0.0;
    // Process 2 bf16 values per u32 word
    var word_idx = tid;
    while (word_idx < words_per_row) {
        let packed = w_raw[row_word_offset + word_idx];
        let col = word_idx * 2u;

        // Low 16 bits = first element
        let v0 = bf16_to_f32(packed & 0xFFFFu);
        acc += v0 * x[col];

        // High 16 bits = second element (check bounds)
        if (col + 1u < params.k) {
            let v1 = bf16_to_f32(packed >> 16u);
            acc += v1 * x[col + 1u];
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
