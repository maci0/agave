// Scaled Dot-Product Attention — FlashAttention-2 style
// One workgroup per attention head. Threads cooperate on KV block processing.
// Online softmax with rescaling for numerical stability.

struct Params {
    nh: u32,
    nkv: u32,
    hd: u32,
    sl: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 16u;
const MAX_HD: u32 = 256u;

var<workgroup> q_local: array<f32, 256>;
var<workgroup> kv_block: array<f32, 4096>; // BLOCK_SIZE * MAX_HD
var<workgroup> scores: array<f32, 16>;
var<workgroup> sdata: array<f32, 8>;
var<workgroup> out_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wgid.x;
    let tid = lid.x;
    if (h >= params.nh) { return; }

    let hd = params.hd;
    let sl = params.sl;
    let nkv = params.nkv;
    let hpg = params.nh / nkv;
    let kvh = h / hpg;
    let kvd = nkv * hd;
    let scale = params.scale;

    // Load Q for this head
    let q_base = h * hd;
    for (var d = tid; d < hd; d = d + 256u) {
        q_local[d] = q[q_base + d];
    }
    for (var d = tid; d < hd; d = d + 256u) {
        out_acc[d] = 0.0;
    }
    workgroupBarrier();

    var m_i: f32 = -3.402823466e+38;
    var l_i: f32 = 0.0;

    let n_blocks = (sl + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    for (var block = 0u; block < n_blocks; block = block + 1u) {
        let block_start = block * BLOCK_SIZE;
        let block_len = min(BLOCK_SIZE, sl - block_start);

        // Load K block
        for (var t = tid; t < block_len; t = t + 256u) {
            let t_global = block_start + t;
            let k_base = t_global * kvd + kvh * hd;
            for (var d = 0u; d < hd; d = d + 1u) {
                kv_block[t * hd + d] = keys[k_base + d];
            }
        }
        workgroupBarrier();

        // Compute scores = Q @ K^T * scale
        for (var t = tid; t < block_len; t = t + 256u) {
            var dot_val: f32 = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot_val = dot_val + q_local[d] * kv_block[t * hd + d];
            }
            scores[t] = dot_val * scale;
        }
        workgroupBarrier();

        // Online softmax: find block max
        var block_max: f32 = -3.402823466e+38;
        for (var t = tid; t < block_len; t = t + 256u) {
            block_max = max(block_max, scores[t]);
        }
        // Workgroup reduce max
        sdata[tid % 8u] = block_max;
        workgroupBarrier();
        if (tid < 8u) {
            var v = sdata[tid];
            for (var i = 0u; i < 8u; i = i + 1u) {
                v = max(v, sdata[i]);
            }
            sdata[0] = v;
        }
        workgroupBarrier();
        let m_new = sdata[0];

        // Rescale existing accumulator
        let m_prev = m_i;
        m_i = max(m_i, m_new);
        let rescale = exp(m_prev - m_i);
        l_i = l_i * rescale;
        for (var d = tid; d < hd; d = d + 256u) {
            out_acc[d] = out_acc[d] * rescale;
        }
        workgroupBarrier();

        // Exp scores + sum
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
            for (var i = 0u; i < 8u; i = i + 1u) {
                v = v + sdata[i];
            }
            sdata[0] = v;
        }
        workgroupBarrier();
        l_i = l_i + sdata[0];

        // Load V block and accumulate weighted values
        for (var t = 0u; t < block_len; t = t + 1u) {
            let w = scores[t];
            if (w < 1e-6) { continue; }
            let t_global = block_start + t;
            let v_base = t_global * kvd + kvh * hd;
            for (var d = tid; d < hd; d = d + 256u) {
                out_acc[d] = out_acc[d] + w * values[v_base + d];
            }
        }
        workgroupBarrier();
    }

    // Normalize and write output
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    let out_base = h * hd;
    for (var d = tid; d < hd; d = d + 256u) {
        output[out_base + d] = out_acc[d] * inv_l;
    }
}
