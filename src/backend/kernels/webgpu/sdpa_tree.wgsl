// Tree-Masked SDPA — FlashAttention-2 with ancestor bitmask
// One workgroup per (node, head) pair.

struct Params {
    nh: u32,
    nkv: u32,
    hd: u32,
    prefix_len: u32,
    n_nodes: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> q_all: array<f32>;
@group(0) @binding(1) var<storage, read> prefix_k: array<f32>;
@group(0) @binding(2) var<storage, read> prefix_v: array<f32>;
@group(0) @binding(3) var<storage, read> tree_k: array<f32>;
@group(0) @binding(4) var<storage, read> tree_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read> ancestor_masks: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 16u;
var<workgroup> sdata: array<f32, 8>;
var<workgroup> scores: array<f32, 16>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let flat_id = wgid.x;
    let tid = lid.x;
    let node_i = flat_id / params.nh;
    let h = flat_id % params.nh;
    if (node_i >= params.n_nodes || h >= params.nh) { return; }

    let hpg = params.nh / params.nkv;
    let kvh = h / hpg;
    let kvd = params.nkv * params.hd;
    let hd = params.hd;
    let q_base = node_i * params.nh * hd + h * hd;

    var m_i: f32 = -3.402823466e+38;
    var l_i: f32 = 0.0;

    // Phase 1: Prefix blocks
    let prefix_blocks = (params.prefix_len + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    for (var block = 0u; block < prefix_blocks; block = block + 1u) {
        let block_start = block * BLOCK_SIZE;
        let block_len = min(BLOCK_SIZE, params.prefix_len - block_start);

        // Compute scores
        for (var t = tid; t < block_len; t = t + 256u) {
            let t_global = block_start + t;
            let k_base = t_global * kvd + kvh * hd;
            var dot_val: f32 = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot_val = dot_val + q_all[q_base + d] * prefix_k[k_base + d];
            }
            scores[t] = dot_val * params.scale;
        }
        workgroupBarrier();

        // Find block max
        var block_max: f32 = -3.402823466e+38;
        for (var t = tid; t < block_len; t = t + 256u) {
            block_max = max(block_max, scores[t]);
        }
        sdata[tid % 8u] = block_max;
        workgroupBarrier();
        if (tid < 8u) {
            var v = sdata[tid];
            for (var i = 0u; i < 8u; i = i + 1u) { v = max(v, sdata[i]); }
            sdata[0] = v;
        }
        workgroupBarrier();
        let m_new = sdata[0];

        let m_prev = m_i;
        m_i = max(m_i, m_new);
        let rescale = exp(m_prev - m_i);
        l_i = l_i * rescale;
        for (var d = tid; d < hd; d = d + 256u) {
            let idx = node_i * params.nh * hd + h * hd + d;
            output[idx] = output[idx] * rescale;
        }
        workgroupBarrier();

        // Exp + sum
        var block_sum: f32 = 0.0;
        for (var t = tid; t < block_len; t = t + 256u) {
            let w = exp(scores[t] - m_i);
            scores[t] = w;
            block_sum = block_sum + w;
        }
        sdata[tid % 8u] = block_sum;
        workgroupBarrier();
        if (tid < 8u) {
            var v = sdata[tid];
            for (var i = 0u; i < 8u; i = i + 1u) { v = v + sdata[i]; }
            sdata[0] = v;
        }
        workgroupBarrier();
        l_i = l_i + sdata[0];

        // V accumulate
        for (var t = 0u; t < block_len; t = t + 1u) {
            let w = scores[t];
            if (w < 1e-6) { continue; }
            let t_global = block_start + t;
            let v_base = t_global * kvd + kvh * hd;
            for (var d = tid; d < hd; d = d + 256u) {
                let idx = node_i * params.nh * hd + h * hd + d;
                output[idx] = output[idx] + w * prefix_v[v_base + d];
            }
        }
        workgroupBarrier();
    }

    // Phase 2: Tree nodes (ancestor bitmask)
    for (var j = 0u; j < params.n_nodes; j = j + 1u) {
        // ancestor_masks stored as u32 array (2 u32s per u64)
        let mask_idx = (node_i * 8u + j / 64u) * 2u;
        let sub_idx = (j % 64u) / 32u;
        let bit = j % 32u;
        let mask_word = ancestor_masks[mask_idx + sub_idx];
        if ((mask_word & (1u << bit)) == 0u) { continue; }

        // Score (single thread)
        if (tid == 0u) {
            var dot_val: f32 = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot_val = dot_val + q_all[q_base + d] * tree_k[j * kvd + kvh * hd + d];
            }
            sdata[0] = dot_val * params.scale;
        }
        workgroupBarrier();
        let tree_score = sdata[0];

        let m_prev2 = m_i;
        m_i = max(m_i, tree_score);
        let resc = exp(m_prev2 - m_i);
        l_i = l_i * resc;
        let w = exp(tree_score - m_i);
        l_i = l_i + w;

        for (var d = tid; d < hd; d = d + 256u) {
            let idx = node_i * params.nh * hd + h * hd + d;
            output[idx] = output[idx] * resc + w * tree_v[j * kvd + kvh * hd + d];
        }
        workgroupBarrier();
    }

    // Normalize
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    for (var d = tid; d < hd; d = d + 256u) {
        let idx = node_i * params.nh * hd + h * hd + d;
        output[idx] = output[idx] * inv_l;
    }
}
