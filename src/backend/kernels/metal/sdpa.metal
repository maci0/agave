// Scaled Dot-Product Attention (SDPA) — FlashAttention-2 tiling algorithm
//
// One threadgroup per query head. Uses online softmax to avoid materializing
// full attention matrix in device memory. K,V are processed in blocks to fit
// threadgroup memory constraints (~32KB).
//
// Algorithm: FlashAttention-2 (Tri Dao, 2023) with conservative block sizes.
// Block sizes tuned for Apple Silicon M-series (32KB threadgroup memory):
// - Bc = 16 (K,V block size along sequence dimension)
// - Br = 1  (single Q head per threadgroup — decode only)
// - max_d = 256 (maximum head dimension)
//
// Memory layout: K_block and V_block reuse same threadgroup memory (sequential processing).
// Total threadgroup memory: q_local[256] + kv_block[16*256] + out_acc[256] + scores[16] + shared[8]
//                         = 1KB + 16KB + 1KB + 64B + 32B ≈ 18KB (within 32KB budget)

#include <metal_stdlib>
using namespace metal;

constant uint sdpa_max_seq_len = 4096;
constant uint sdpa_max_head_dim = 256;
constant uint sdpa_block_size = 16;  // Bc: K,V block size (16 to fit 32KB threadgroup limit)

kernel void sdpa_fa2(
    device const float* Q,       // [nh * hd]
    device const float* K_cache, // [>= sl * kvd]
    device const float* V_cache, // [>= sl * kvd]
    device float* output,        // [nh * hd]
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& sl,           // actual sequence length (1..4096)
    constant float& scale,
    uint h     [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    if (h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;
    uint num_blocks = (sl + sdpa_block_size - 1) / sdpa_block_size;

    // Threadgroup memory
    threadgroup float q_local[sdpa_max_head_dim];
    threadgroup float kv_block[sdpa_block_size * sdpa_max_head_dim];  // Reused for K and V
    threadgroup float scores[sdpa_block_size];
    threadgroup float shared[8];  // Reduction scratch (one per SIMD group)

    // ── Load Q into threadgroup memory ────────────────────────────
    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Online softmax state (per-thread register) ────────────────
    float m_i = -INFINITY;      // Running max
    float l_i = 0.0f;           // Running sum
    threadgroup float out_acc[sdpa_max_head_dim];  // Output accumulator (threadgroup for all threads)
    for (uint d = tid; d < hd; d += tg_sz) {
        out_acc[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Outer loop: iterate over K,V blocks ───────────────────────
    for (uint block = 0; block < num_blocks; block++) {
        uint block_start = block * sdpa_block_size;
        uint block_len = min(sdpa_block_size, sl - block_start);

        // ── Load K_block ───────────────────────────────────────
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            uint k_base = t_global * kvd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                kv_block[t * hd + d] = K_cache[k_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute scores = Q @ K_block^T ─────────────────────
        for (uint t = tid; t < block_len; t += tg_sz) {
            float dot_val = 0.0f;
            for (uint d = 0; d < hd; d++) {
                dot_val += q_local[d] * kv_block[t * hd + d];
            }
            scores[t] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Online softmax max reduction ───────────────────────
        uint simd_lane  = tid % 32;
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
        if (tid < 32) {
            block_max = simd_max(block_max);
            if (tid == 0) shared[0] = block_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = shared[0];

        // ── Online softmax: rescale previous sum and accumulator ──
        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale_factor = exp(m_prev - m_i);
        l_i *= rescale_factor;
        for (uint d = tid; d < hd; d += tg_sz) {
            out_acc[d] *= rescale_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute exp(scores - m_new) and sum ───────────────────
        float local_sum = 0.0f;
        for (uint t = tid; t < block_len; t += tg_sz) {
            float v = exp(scores[t] - m_i);
            scores[t] = v;
            local_sum += v;
        }
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) shared[simd_group] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < num_sg) local_sum = shared[tid]; else local_sum = 0.0f;
        if (tid < 32) {
            local_sum = simd_sum(local_sum);
            if (tid == 0) shared[0] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += shared[0];

        // ── Load V_block (reuses kv_block memory) ──────────────
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            uint v_base = t_global * kvd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                kv_block[t * hd + d] = V_cache[v_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Accumulate: out_acc += exp(scores - m_i) @ V_block ────
        for (uint d = tid; d < hd; d += tg_sz) {
            float acc = 0.0f;
            for (uint t = 0; t < block_len; t++) {
                acc += scores[t] * kv_block[t * hd + d];
            }
            out_acc[d] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Final normalization: output = out_acc / l_i ────────────────
    for (uint d = tid; d < hd; d += tg_sz) {
        output[h * hd + d] = out_acc[d] / l_i;
    }
}

// ── Prefill SDPA (causal, multi-token) ───────────────────────────
// Processes n_tok query tokens in a single dispatch, each with causal masking.
// Grid: n_tok * nh threadgroups. Each threadgroup handles one query head for
// one token, attending to positions 0..prev_len+tok (causal).
// KV cache must already contain all positions (appended before dispatch).

kernel void sdpa_prefill_fa2(
    device const float* Q,       // [n_tok * nh * hd]
    device const float* K_cache, // [>= prev_len * kvd] — cached positions
    device const float* V_cache, // [>= prev_len * kvd]
    device const float* K_new,   // [n_tok * kvd] — new positions from GEMM
    device const float* V_new,   // [n_tok * kvd]
    device float* output,        // [n_tok * nh * hd]
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& prev_len,
    constant uint& n_tok,
    constant float& scale,
    uint flat_id  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_sz    [[threads_per_threadgroup]])
{
    uint tok = flat_id / nh;
    uint h = flat_id % nh;
    if (tok >= n_tok || h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;
    uint sl = prev_len + tok + 1;  // Causal: attend to 0..prev_len+tok
    uint num_blocks = (sl + sdpa_block_size - 1) / sdpa_block_size;

    uint q_base = tok * nh * hd + h * hd;
    uint o_base = tok * nh * hd + h * hd;

    threadgroup float q_local[sdpa_max_head_dim];
    threadgroup float kv_block[sdpa_block_size * sdpa_max_head_dim];
    threadgroup float scores[sdpa_block_size];
    threadgroup float shared[8];

    // Load Q for this token+head
    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q[q_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    threadgroup float out_acc[sdpa_max_head_dim];
    for (uint d = tid; d < hd; d += tg_sz) {
        out_acc[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FlashAttention-2 outer loop over KV blocks
    for (uint block = 0; block < num_blocks; block++) {
        uint block_start = block * sdpa_block_size;
        uint block_len = min(sdpa_block_size, sl - block_start);

        // Load K block — from cache (old positions) or K_new (new positions)
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            if (t_global < prev_len) {
                uint k_base = t_global * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++)
                    kv_block[t * hd + d] = K_cache[k_base + d];
            } else {
                uint k_base = (t_global - prev_len) * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++)
                    kv_block[t * hd + d] = K_new[k_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK dot products
        for (uint t = tid; t < block_len; t += tg_sz) {
            float dot_val = 0.0f;
            for (uint d = 0; d < hd; d++) {
                dot_val += q_local[d] * kv_block[t * hd + d];
            }
            scores[t] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: max reduction
        uint simd_lane  = tid % 32;
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
        if (tid < 32) {
            block_max = simd_max(block_max);
            if (tid == 0) shared[0] = block_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = shared[0];

        // Rescale previous accumulator
        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale_factor = exp(m_prev - m_i);
        l_i *= rescale_factor;
        for (uint d = tid; d < hd; d += tg_sz) {
            out_acc[d] *= rescale_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute exp(scores) and sum
        float local_sum = 0.0f;
        for (uint t = tid; t < block_len; t += tg_sz) {
            float v = exp(scores[t] - m_i);
            scores[t] = v;
            local_sum += v;
        }
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) shared[simd_group] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < num_sg) local_sum = shared[tid]; else local_sum = 0.0f;
        if (tid < 32) {
            local_sum = simd_sum(local_sum);
            if (tid == 0) shared[0] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += shared[0];

        // Load V block — from cache or V_new
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            if (t_global < prev_len) {
                uint v_base = t_global * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++)
                    kv_block[t * hd + d] = V_cache[v_base + d];
            } else {
                uint v_base = (t_global - prev_len) * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++)
                    kv_block[t * hd + d] = V_new[v_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: out_acc += scores @ V_block
        for (uint d = tid; d < hd; d += tg_sz) {
            float acc = 0.0f;
            for (uint t = 0; t < block_len; t++) {
                acc += scores[t] * kv_block[t * hd + d];
            }
            out_acc[d] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization
    for (uint d = tid; d < hd; d += tg_sz) {
        output[o_base + d] = out_acc[d] / l_i;
    }
}
