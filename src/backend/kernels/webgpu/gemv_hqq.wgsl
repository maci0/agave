// HQQ INT4 GEMV: y[row] = dot(dequant(W[row, :]), x)
//
// HQQ format:
//   w_q   : uint8, shape [n_out, k_in/2] — low nibble = even k, high nibble = odd k
//   scale : bf16,  shape [n_out, k_in/group_size]
//   zero  : bf16,  shape [n_out, k_in/group_size]
//
// Dequant: w = (nibble - zero) * scale
//
// One workgroup per output row. 256 threads per workgroup.
// Threads accumulate over their stripe of k, then a workgroup tree reduction
// writes the final scalar to y[row].

const WG_SIZE: u32 = 256u;

struct Params {
    n:          u32, // output dimension (rows)
    k:          u32, // input  dimension
    group_size: u32, // quantization group size (typically 64)
    _pad:       u32,
}

// binding 0: x          — f32 activations [k]
// binding 1: w_q        — packed uint8 nibbles stored as u32 words [n * k/2 / 4]
// binding 2: scale      — bf16 packed two-per-u32 [n * k/group_size]
// binding 3: zero       — bf16 packed two-per-u32 [n * k/group_size]
// binding 4: y          — f32 output [n]
// binding 5: params     — uniform Params
@group(0) @binding(0) var<storage, read>       x_data:     array<f32>;
@group(0) @binding(1) var<storage, read>       wq_data:    array<u32>;
@group(0) @binding(2) var<storage, read>       scale_data: array<u32>;
@group(0) @binding(3) var<storage, read>       zero_data:  array<u32>;
@group(0) @binding(4) var<storage, read_write> y_data:     array<f32>;
@group(0) @binding(5) var<uniform>             params:     Params;

var<workgroup> partial_sums: array<f32, 256>;

// Decode bf16 (u16 bits) to f32 by placing the 16 bits in the upper half of a u32.
fn bf16_to_f32(bits16: u32) -> f32 {
    return bitcast<f32>(bits16 << 16u);
}

// Read the bf16 value at logical index idx from a u32 buffer packed two-per-word.
// Even indices occupy the low 16 bits; odd indices occupy the high 16 bits.
fn read_bf16(buf_idx: u32, src: ptr<storage, array<u32>, read>) -> f32 {
    let word   = (*src)[buf_idx / 2u];
    let bits16 = select(word >> 16u, word & 0xFFFFu, buf_idx % 2u == 0u);
    return bf16_to_f32(bits16);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;

    if (row >= params.n) { return; }

    let k          = params.k;
    let group_size = params.group_size;
    let n_groups   = (k + group_size - 1u) / group_size;

    // Byte offset to the first byte of this row in w_q (k/2 bytes per row).
    let row_byte_off = row * (k / 2u);

    var sum: f32 = 0.0;

    var ki = tid;
    loop {
        if (ki >= k) { break; }

        let xv = x_data[ki];
        // Sparse skip: avoid multiply-adds for near-zero activations.
        if (abs(xv) >= 0.005) {
            // Locate the packed byte for column ki.
            // Even ki -> low nibble (bits 3:0); odd ki -> high nibble (bits 7:4).
            let byte_idx = row_byte_off + ki / 2u;
            let word     = wq_data[byte_idx / 4u];
            let byte_val = (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
            let nibble   = f32(select(byte_val >> 4u, byte_val & 0xFu, ki % 2u == 0u));

            // Fetch bf16 scale and zero for this group.
            let g      = ki / group_size;
            let sq_idx = row * n_groups + g;
            let scale  = read_bf16(sq_idx, &scale_data);
            let zero   = read_bf16(sq_idx, &zero_data);

            sum += (nibble - zero) * scale * xv;
        }

        ki += WG_SIZE;
    }

    // Store partial sum, then tree-reduce within the workgroup.
    partial_sums[tid] = sum;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        y_data[row] = partial_sums[0u];
    }
}
