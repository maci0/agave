// Embedding lookup: output[i] = emb_table[token_id * n_embd + i] for i in 0..n_embd

struct Params {
    vocab_size: u32,
    n_embd: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> emb_table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_embd) {
        return;
    }
    let token_id = token_ids[0];
    let offset = token_id * params.n_embd + i;
    output[i] = emb_table[offset];
}
