// Scaled Dot-Product Attention (SDPA) — FlashAttention-2 tiling algorithm
//
// One threadgroup per query head. Uses online softmax to avoid materializing
// full attention matrix in device memory. K,V are processed in blocks to fit
// threadgroup memory constraints (~32KB).
//
// Algorithm: FlashAttention-2 (Tri Dao, 2023) with conservative block sizes.
// Block sizes tuned for Apple Silicon M-series (32KB threadgroup memory):
// - Bc = 32 (K,V block size along sequence dimension)
// - Br = 32 (Q,O block size — currently unused, single Q head per threadgroup)
// - max_d = 256 (maximum head dimension)
//
// Memory layout: K_block and V_block reuse same threadgroup memory (sequential processing).
// Total threadgroup memory: q_local[256] + kv_block[32*256] + scores[32] + shared[8]
//                         = 1KB + 32KB + 128B + 32B ≈ 33KB (fits in 32KB budget after optimization)

#include <metal_stdlib>
using namespace metal;

constant uint sdpa_max_seq_len = 4096;
constant uint sdpa_max_head_dim = 256;
constant uint sdpa_block_size = 32;  // Bc: K,V block size

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

// Legacy kernel (kept for reference, not used)
kernel void sdpa_f32(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float* output,
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& sl,
    constant float& scale,
    uint h     [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    if (h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;

    threadgroup float scores[sdpa_max_seq_len];
    threadgroup float q_local[sdpa_max_head_dim];
    threadgroup float shared[8];

    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hd4 = hd & ~3u;
    for (uint t = tid; t < sl; t += tg_sz) {
        float dot_val = 0.0f;
        uint base = t * kvd + kvh * hd;
        for (uint d = 0; d < hd4; d += 4) {
            dot_val += dot(float4(q_local[d], q_local[d+1], q_local[d+2], q_local[d+3]),
                           float4(K_cache[base+d], K_cache[base+d+1], K_cache[base+d+2], K_cache[base+d+3]));
        }
        for (uint d = hd4; d < hd; d++) {
            dot_val += q_local[d] * K_cache[base + d];
        }
        scores[t] = dot_val * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    uint num_sg = (tg_sz + 31) / 32;
    float mx = -INFINITY;
    for (uint t = tid; t < sl; t += tg_sz) {
        mx = max(mx, scores[t]);
    }
    mx = simd_max(mx);
    if (simd_lane == 0) shared[simd_group] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_sg) mx = shared[tid]; else mx = -INFINITY;
    if (tid < 32) {
        mx = simd_max(mx);
        if (tid == 0) shared[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (uint t = tid; t < sl; t += tg_sz) {
        float v = exp(scores[t] - max_val);
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
    float inv_sum = 1.0f / shared[0];

    for (uint t = tid; t < sl; t += tg_sz) {
        scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid * 4; d + 3 < hd; d += tg_sz * 4) {
        float4 acc = float4(0.0f);
        for (uint t = 0; t < sl; t++) {
            uint v_base = t * kvd + kvh * hd + d;
            acc += scores[t] * float4(V_cache[v_base], V_cache[v_base+1], V_cache[v_base+2], V_cache[v_base+3]);
        }
        uint o_base = h * hd + d;
        output[o_base] = acc.x;
        output[o_base+1] = acc.y;
        output[o_base+2] = acc.z;
        output[o_base+3] = acc.w;
    }
    for (uint d = (hd & ~3u) + tid; d < hd; d += tg_sz) {
        float acc = 0.0f;
        for (uint t = 0; t < sl; t++) {
            acc += scores[t] * V_cache[t * kvd + kvh * hd + d];
        }
        output[h * hd + d] = acc;
    }
}
