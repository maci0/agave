// Embedding lookup with inline dequant for f32, bf16, f16, q8_0

struct Params {
    n_embd: u32,
    dtype: u32, // 0=f32, 1=bf16, 2=f16, 3=q8_0
    token_id: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> emb_table: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_embd) { return; }

    let tid = params.token_id;

    if (params.dtype == 0u) {
        // f32: 1 element per u32 word
        let offset = tid * params.n_embd + i;
        output[i] = bitcast<f32>(emb_table[offset]);
    } else if (params.dtype == 1u) {
        // bf16: 2 elements per u32 word
        let elem_idx = tid * params.n_embd + i;
        let word = emb_table[elem_idx / 2u];
        let half_val = select(word >> 16u, word & 0xFFFFu, elem_idx % 2u == 0u);
        output[i] = bitcast<f32>(half_val << 16u);
    } else if (params.dtype == 2u) {
        // f16: 2 elements per u32 word
        let elem_idx = tid * params.n_embd + i;
        let word = emb_table[elem_idx / 2u];
        let packed = select(word >> 16u, word & 0xFFFFu, elem_idx % 2u == 0u);
        output[i] = unpack2x16float(packed).x;
    } else {
        // q8_0: 32 elements per block, 34 bytes per block (f16 scale + 32 int8)
        let block_size = 32u;
        let block_bytes = 34u;
        let elem_idx = tid * params.n_embd + i;
        let block_idx = elem_idx / block_size;
        let elem_in_block = elem_idx % block_size;

        let block_byte_start = block_idx * block_bytes;
        // f16 scale at bytes 0-1
        let scale_word_idx = block_byte_start / 4u;
        let scale_byte_off = block_byte_start % 4u;
        var scale_bits: u32;
        if (scale_byte_off <= 2u) {
            scale_bits = (emb_table[scale_word_idx] >> (scale_byte_off * 8u)) & 0xFFFFu;
        } else {
            scale_bits = (emb_table[scale_word_idx] >> 24u) | ((emb_table[scale_word_idx + 1u] & 0xFFu) << 8u);
        }
        let scale = unpack2x16float(scale_bits).x;

        // int8 quant at byte offset 2 + elem_in_block
        let quant_byte_pos = block_byte_start + 2u + elem_in_block;
        let quant_word_idx = quant_byte_pos / 4u;
        let quant_byte_off = quant_byte_pos % 4u;
        let quant_word = emb_table[quant_word_idx];
        var quant_u8 = (quant_word >> (quant_byte_off * 8u)) & 0xFFu;
        // Sign extend i8 → i32
        var quant_i32: i32;
        if (quant_u8 >= 128u) {
            quant_i32 = i32(quant_u8) - 256;
        } else {
            quant_i32 = i32(quant_u8);
        }
        output[i] = scale * f32(quant_i32);
    }
}
