// Tree-Masked SDPA — FlashAttention-2 with ancestor bitmask masking
//
// One threadgroup per (node, head) pair. Each tree node attends to:
//   1. All prefix KV entries (unconditional)
//   2. Ancestor tree nodes (controlled by bitmask)
//
// Used during DDTree speculative decoding batch verification.

#include <metal_stdlib>
using namespace metal;

constant uint tree_block_size = 16;
constant uint tree_max_head_dim = 256;

kernel void sdpa_tree_fa2(
    device const float* Q_all,          // [n_nodes * nh * hd]
    device const float* prefix_K,       // [prefix_len * kvd] — f32 from paged cache
    device const float* prefix_V,       // [prefix_len * kvd]
    device const float* tree_K,         // [n_nodes * kvd]
    device const float* tree_V,         // [n_nodes * kvd]
    device float* output,               // [n_nodes * nh * hd]
    device const ulong* ancestor_masks, // [n_nodes * 8]
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& prefix_len,
    constant uint& n_nodes,
    constant float& scale,
    uint flat_id [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_sz   [[threads_per_threadgroup]])
{
    uint node_i = flat_id / nh;
    uint h = flat_id % nh;
    if (node_i >= n_nodes || h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;

    // Threadgroup memory
    threadgroup float q_local[tree_max_head_dim];
    threadgroup float kv_block[tree_block_size * tree_max_head_dim];
    threadgroup float scores[tree_block_size];
    threadgroup float shared[8];
    threadgroup float out_acc[tree_max_head_dim];

    // Load Q for this node/head
    uint q_base = node_i * nh * hd + h * hd;
    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q_all[q_base + d];
    }
    for (uint d = tid; d < hd; d += tg_sz) {
        out_acc[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ── Phase 1: Prefix blocks (unconditional) ───────────────────
    uint prefix_blocks = (prefix_len + tree_block_size - 1) / tree_block_size;
    for (uint block = 0; block < prefix_blocks; block++) {
        uint block_start = block * tree_block_size;
        uint block_len = min(tree_block_size, prefix_len - block_start);

        // Load K block from prefix
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            uint k_base = t_global * kvd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                kv_block[t * hd + d] = prefix_K[k_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores
        for (uint t = tid; t < block_len; t += tg_sz) {
            float dot_val = 0.0f;
            for (uint d = 0; d < hd; d++) {
                dot_val += q_local[d] * kv_block[t * hd + d];
            }
            scores[t] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax max
        uint simd_lane = tid % 32;
        uint simd_group = tid / 32;
        uint num_sg = (tg_sz + 31) / 32;

        float block_max = -INFINITY;
        for (uint t = tid; t < block_len; t += tg_sz) {
            block_max = max(block_max, scores[t]);
        }
        block_max = simd_max(block_max);
        if (simd_lane == 0) shared[simd_group] = block_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < num_sg) block_max = shared[tid]; else block_max = -INFINITY;
        if (tid < 32) { block_max = simd_max(block_max); if (tid == 0) shared[0] = block_max; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = shared[0];

        // Rescale
        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale = exp(m_prev - m_i);
        l_i *= rescale;
        for (uint d = tid; d < hd; d += tg_sz) { out_acc[d] *= rescale; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exp + sum + V accumulate
        float block_sum = 0.0f;
        for (uint t = tid; t < block_len; t += tg_sz) {
            float w = exp(scores[t] - m_i);
            scores[t] = w;
            block_sum += w;
        }
        block_sum = simd_sum(block_sum);
        if (simd_lane == 0) shared[simd_group] = block_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < num_sg) block_sum = shared[tid]; else block_sum = 0.0f;
        if (tid < 32) { block_sum = simd_sum(block_sum); if (tid == 0) shared[0] = block_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += shared[0];

        // Load V and accumulate
        for (uint t = 0; t < block_len; t++) {
            float w = scores[t];
            if (w < 1e-6f) continue;
            uint t_global = block_start + t;
            uint v_base = t_global * kvd + kvh * hd;
            for (uint d = tid; d < hd; d += tg_sz) {
                out_acc[d] += w * prefix_V[v_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 2: Tree nodes (masked by ancestor bitmask) ─────────
    // Process tree nodes one at a time (typically < 64, not worth blocking)
    for (uint j = 0; j < n_nodes; j++) {
        // Check ancestor mask
        ulong mask_word = ancestor_masks[node_i * 8 + j / 64];
        if ((mask_word & (1UL << (j % 64))) == 0) continue;

        // Load tree K for node j
        for (uint d = tid; d < hd; d += tg_sz) {
            kv_block[d] = tree_K[j * kvd + kvh * hd + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Score
        float dot_val = 0.0f;
        if (tid == 0) {
            for (uint d = 0; d < hd; d++) {
                dot_val += q_local[d] * kv_block[d];
            }
            scores[0] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update
        float m_new = scores[0];
        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale = exp(m_prev - m_i);
        l_i *= rescale;
        for (uint d = tid; d < hd; d += tg_sz) { out_acc[d] *= rescale; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w = exp(scores[0] - m_i);
        l_i += w;

        // V accumulate
        for (uint d = tid; d < hd; d += tg_sz) {
            out_acc[d] += w * tree_V[j * kvd + kvh * hd + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Normalize and write output ───────────────────────────────
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    uint out_base = node_i * nh * hd + h * hd;
    for (uint d = tid; d < hd; d += tg_sz) {
        output[out_base + d] = out_acc[d] * inv_l;
    }
}

// TurboQuant variant — prefix K/V are quantized (turbo2/3/4), tree K/V remain f32.
kernel void sdpa_tree_fa2_turbo(
    device const float* Q_all,
    device const uchar* prefix_K,       // turbo-quantized
    device const uchar* prefix_V,       // turbo-quantized
    device const float* tree_K,
    device const float* tree_V,
    device float* output,
    device const ulong* ancestor_masks,
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& prefix_len,
    constant uint& n_nodes,
    constant float& scale,
    constant uint& bits_k,
    constant uint& bits_v,
    constant uint& block_bytes_k,
    constant uint& block_bytes_v,
    uint flat_id [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_sz   [[threads_per_threadgroup]])
{
    uint node_i = flat_id / nh;
    uint h = flat_id % nh;
    if (node_i >= n_nodes || h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;

    threadgroup float q_local[tree_max_head_dim];
    threadgroup float kv_block[tree_block_size * tree_max_head_dim];
    threadgroup float scores[tree_block_size];
    threadgroup float shared[8];
    threadgroup float out_acc[tree_max_head_dim];

    uint q_base = node_i * nh * hd + h * hd;
    for (uint d = tid; d < hd; d += tg_sz) q_local[d] = Q_all[q_base + d];
    for (uint d = tid; d < hd; d += tg_sz) out_acc[d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ── Phase 1: Prefix blocks with turbo dequant ───────────────
    uint prefix_blocks = (prefix_len + tree_block_size - 1) / tree_block_size;
    for (uint block = 0; block < prefix_blocks; block++) {
        uint block_start = block * tree_block_size;
        uint block_len = min(tree_block_size, prefix_len - block_start);

        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            uint elem_base = t_global * kvd + kvh * hd;
            uint n_turbo_blocks = hd / 32;
            for (uint blk = 0; blk < n_turbo_blocks; blk++) {
                uint turbo_block_idx = (elem_base + blk * 32) / 32;
                uint byte_off = turbo_block_idx * block_bytes_k;
                float dequant_buf[32];
                turbo_dequant_block(prefix_K + byte_off, dequant_buf, bits_k);
                for (uint d = 0; d < 32; d++) {
                    kv_block[t * hd + blk * 32 + d] = dequant_buf[d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = tid; t < block_len; t += tg_sz) {
            float dot_val = 0.0f;
            for (uint d = 0; d < hd; d++) dot_val += q_local[d] * kv_block[t * hd + d];
            scores[t] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint simd_lane = tid % 32;
        uint simd_group = tid / 32;
        uint num_sg = (tg_sz + 31) / 32;

        float block_max = -INFINITY;
        for (uint t = tid; t < block_len; t += tg_sz) block_max = max(block_max, scores[t]);
        block_max = simd_max(block_max);
        if (simd_lane == 0) shared[simd_group] = block_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < num_sg) block_max = shared[tid]; else block_max = -INFINITY;
        if (tid < 32) { block_max = simd_max(block_max); if (tid == 0) shared[0] = block_max; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = shared[0];

        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale = exp(m_prev - m_i);
        l_i *= rescale;
        for (uint d = tid; d < hd; d += tg_sz) out_acc[d] *= rescale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = 0.0f;
        for (uint t = tid; t < block_len; t += tg_sz) {
            float w = exp(scores[t] - m_i);
            scores[t] = w;
            block_sum += w;
        }
        block_sum = simd_sum(block_sum);
        if (simd_lane == 0) shared[simd_group] = block_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < num_sg) block_sum = shared[tid]; else block_sum = 0.0f;
        if (tid < 32) { block_sum = simd_sum(block_sum); if (tid == 0) shared[0] = block_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += shared[0];

        // V dequant + accumulate
        for (uint t = 0; t < block_len; t++) {
            float w = scores[t];
            if (w < 1e-6f) continue;
            uint t_global = block_start + t;
            uint elem_base = t_global * kvd + kvh * hd;
            uint n_turbo_blocks = hd / 32;
            for (uint blk = 0; blk < n_turbo_blocks; blk++) {
                uint turbo_block_idx = (elem_base + blk * 32) / 32;
                uint byte_off = turbo_block_idx * block_bytes_v;
                float dequant_buf[32];
                turbo_dequant_block(prefix_V + byte_off, dequant_buf, bits_v);
                for (uint d = tid; d < 32; d += tg_sz) {
                    out_acc[blk * 32 + d] += w * dequant_buf[d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 2: Tree nodes (f32, same as non-turbo) ────────────
    for (uint j = 0; j < n_nodes; j++) {
        ulong mask_word = ancestor_masks[node_i * 8 + j / 64];
        if ((mask_word & (1UL << (j % 64))) == 0) continue;

        for (uint d = tid; d < hd; d += tg_sz) {
            kv_block[d] = tree_K[j * kvd + kvh * hd + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float dot_val = 0.0f;
        if (tid == 0) {
            for (uint d = 0; d < hd; d++) dot_val += q_local[d] * kv_block[d];
            scores[0] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_new = scores[0];
        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale = exp(m_prev - m_i);
        l_i *= rescale;
        for (uint d = tid; d < hd; d += tg_sz) out_acc[d] *= rescale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float w = exp(scores[0] - m_i);
        l_i += w;
        for (uint d = tid; d < hd; d += tg_sz) {
            out_acc[d] += w * tree_V[j * kvd + kvh * hd + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    uint out_base = node_i * nh * hd + h * hd;
    for (uint d = tid; d < hd; d += tg_sz) {
        output[out_base + d] = out_acc[d] * inv_l;
    }
}
