// DeepSeek V4 Flash — GPU-only kernels for hyper-connection mixing,
// table-based RoPE, weighted expert accumulation, and turbo SDPA hd=512.
// Eliminates all CPU reads of Metal-written activation buffers between layers,
// fixing L2 cache coherency issues with newBufferWithBytesNoCopy shared-memory wraps.

#include <metal_stdlib>
using namespace metal;

// ── Constants ────────────────────────────────────────────────────
constant uint ds4_n_hc = 4;         // Number of hyper-connection streams
constant uint ds4_hc_mix_dim = 24;  // (2 + n_hc) * n_hc = 24
constant uint ds4_sinkhorn_iters = 20;
constant float ds4_hc_eps = 1e-6f;

// ── HC Pre: compute mixing weights from HC state ─────────────────
// Phase 1: Compute RMS scale of hc_state[n_hc * n_embd], then
//          GEMV hc_fn @ hc_state → mixes[24], post-scale by rms_inv.
//          Compute pre_w[4], post_w[4], comb[16] from mixes + base + scale.
//          Then weighted sum of HC streams → hidden[n_embd].
//
// This is a SMALL operation (24-output GEMV + 4096-dim weighted sum).
// Two sub-kernels: (1) compute weights, (2) apply weighted sum.

// Sub-kernel 1: Compute HC mixing weights.
// One threadgroup, tg_size threads. Outputs: pre_w[4], post_w[4], comb[16].
// hc_fn weights are f32 (tiny matrix: 24 × n_hc*n_embd).
kernel void ds4_hc_weights(
    device const float* hc_state  [[buffer(0)]],  // [n_hc * n_embd]
    device const float* hc_fn     [[buffer(1)]],  // [hc_mix_dim × flat_size] row-major
    device const float* hc_base   [[buffer(2)]],  // [hc_mix_dim]
    device const float* hc_scale  [[buffer(3)]],  // [3]
    device float* pre_w           [[buffer(4)]],  // [n_hc] output
    device float* post_w          [[buffer(5)]],  // [n_hc] output
    device float* comb            [[buffer(6)]],  // [n_hc * n_hc] output
    constant uint& n_embd         [[buffer(7)]],
    constant float& rms_eps       [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    uint flat_size = ds4_n_hc * n_embd;

    // Step 1: RMS of hc_state (parallel reduction)
    threadgroup float shared_rms[8];
    float local_ss = 0.0f;
    for (uint i = tid; i < flat_size; i += tg_sz) {
        float v = hc_state[i];
        local_ss += v * v;
    }
    // SIMD reduction
    local_ss = simd_sum(local_ss);
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (lane == 0) shared_rms[sg] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) local_ss = shared_rms[tid]; else local_ss = 0.0f;
    if (tid < 32) {
        local_ss = simd_sum(local_ss);
        if (tid == 0) shared_rms[0] = local_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = rsqrt(shared_rms[0] / float(flat_size) + rms_eps);

    // Step 2: GEMV — mixes[r] = dot(hc_fn[r,:], hc_state) * rms_inv
    // 24 output rows, each dot product over flat_size elements.
    // Each thread handles a subset of output rows.
    threadgroup float mixes[ds4_hc_mix_dim];
    for (uint r = tid; r < ds4_hc_mix_dim; r += tg_sz) {
        float acc = 0.0f;
        device const float* row = hc_fn + r * flat_size;
        for (uint i = 0; i < flat_size; i++) {
            acc += row[i] * hc_state[i];
        }
        mixes[r] = acc * rms_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute pre_w, post_w, comb from mixes + base + scale
    if (tid < ds4_n_hc) {
        pre_w[tid] = 1.0f / (1.0f + exp(-(mixes[tid] * hc_scale[0] + hc_base[tid]))) + ds4_hc_eps;
        post_w[tid] = (1.0f / (1.0f + exp(-(mixes[ds4_n_hc + tid] * hc_scale[1] + hc_base[ds4_n_hc + tid])))) * 2.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Comb: raw affine values for sinkhorn
    if (tid < ds4_n_hc * ds4_n_hc) {
        uint idx = 2 * ds4_n_hc + tid;
        comb[tid] = mixes[idx] * hc_scale[2] + hc_base[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Sinkhorn normalization on comb[4×4] (tiny, single thread).
    // Matches CPU hcSinkhorn: initial row-softmax + eps, then alternating
    // column-normalize / row-normalize for sinkhorn_iters iterations.
    if (tid == 0) {
        float m[16];
        for (int i = 0; i < 16; i++) m[i] = comb[i];

        // Initial softmax: for each row r, softmax over columns
        for (uint r = 0; r < ds4_n_hc; r++) {
            float mx = -INFINITY;
            for (uint c = 0; c < ds4_n_hc; c++) mx = max(mx, m[r * ds4_n_hc + c]);
            float sm = 0.0f;
            for (uint c = 0; c < ds4_n_hc; c++) {
                m[r * ds4_n_hc + c] = exp(m[r * ds4_n_hc + c] - mx);
                sm += m[r * ds4_n_hc + c];
            }
            float inv = 1.0f / sm;
            for (uint c = 0; c < ds4_n_hc; c++) m[r * ds4_n_hc + c] *= inv;
        }
        // Add eps
        for (int i = 0; i < 16; i++) m[i] += ds4_hc_eps;

        // Sinkhorn iterations: alternate column / row normalization
        for (uint iter = 0; iter < ds4_sinkhorn_iters; iter++) {
            // Column normalization
            for (uint c = 0; c < ds4_n_hc; c++) {
                float col_sum = 0.0f;
                for (uint r = 0; r < ds4_n_hc; r++) col_sum += m[r * ds4_n_hc + c];
                col_sum += ds4_hc_eps;
                for (uint r = 0; r < ds4_n_hc; r++) m[r * ds4_n_hc + c] /= col_sum;
            }
            // Row normalization
            for (uint r = 0; r < ds4_n_hc; r++) {
                float row_sum = 0.0f;
                for (uint c = 0; c < ds4_n_hc; c++) row_sum += m[r * ds4_n_hc + c];
                row_sum += ds4_hc_eps;
                for (uint c = 0; c < ds4_n_hc; c++) m[r * ds4_n_hc + c] /= row_sum;
            }
        }
        for (int i = 0; i < 16; i++) comb[i] = m[i];
    }
}

// Sub-kernel 2: Weighted sum of HC streams → hidden.
// hidden[i] = Σ_s pre_w[s] * hc_state[s * n_embd + i]
kernel void ds4_hc_pre_mix(
    device const float* hc_state [[buffer(0)]],  // [n_hc * n_embd]
    device const float* pre_w    [[buffer(1)]],  // [n_hc]
    device float* hidden         [[buffer(2)]],  // [n_embd]
    constant uint& n_embd        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_embd) return;
    float acc = 0.0f;
    for (uint s = 0; s < ds4_n_hc; s++) {
        acc += pre_w[s] * hc_state[s * n_embd + tid];
    }
    hidden[tid] = acc;
}

// ── HC Post: update HC state after sublayer ──────────────────────
// new_hc[dst * n_embd + i] = post_w[dst] * hidden[i]
//                           + Σ_src comb[dst + src * n_hc] * hc_state[src * n_embd + i]
kernel void ds4_hc_post(
    device const float* hidden   [[buffer(0)]],  // [n_embd] — sublayer output
    device const float* hc_state [[buffer(1)]],  // [n_hc * n_embd] — current state
    device const float* post_w   [[buffer(2)]],  // [n_hc]
    device const float* comb     [[buffer(3)]],  // [n_hc * n_hc], col-major: comb[dst + src*n_hc]
    device float* new_hc         [[buffer(4)]],  // [n_hc * n_embd] — output
    constant uint& n_embd        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = ds4_n_hc * n_embd;
    if (tid >= total) return;
    uint dst = tid / n_embd;
    uint i = tid % n_embd;
    float v = post_w[dst] * hidden[i];
    for (uint src = 0; src < ds4_n_hc; src++) {
        v += comb[dst + src * ds4_n_hc] * hc_state[src * n_embd + i];
    }
    new_hc[tid] = v;
}

// ── HC Head: merge streams for final output ──────────────────────
// Head has only n_hc=4 outputs (not 24). Uses sigmoid (no post_w/comb/sinkhorn).
// GEMV: pre_w[4] = hc_fn[4 × flat_size] @ hc_state[flat_size]
kernel void ds4_hc_head_weights(
    device const float* hc_state  [[buffer(0)]],  // [n_hc * n_embd]
    device const float* hc_fn     [[buffer(1)]],  // [n_hc × flat_size] row-major
    device const float* hc_base   [[buffer(2)]],  // [n_hc]
    device const float* hc_scale  [[buffer(3)]],  // [1]
    device float* pre_w           [[buffer(4)]],  // [n_hc] output
    constant uint& n_embd         [[buffer(5)]],
    constant float& rms_eps       [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    uint flat_size = ds4_n_hc * n_embd;

    // Step 1: RMS of hc_state
    threadgroup float shared_rms[8];
    float local_ss = 0.0f;
    for (uint i = tid; i < flat_size; i += tg_sz) {
        float v = hc_state[i];
        local_ss += v * v;
    }
    local_ss = simd_sum(local_ss);
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (lane == 0) shared_rms[sg] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) local_ss = shared_rms[tid]; else local_ss = 0.0f;
    if (tid < 32) {
        local_ss = simd_sum(local_ss);
        if (tid == 0) shared_rms[0] = local_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = rsqrt(shared_rms[0] / float(flat_size) + rms_eps);

    // Step 2: GEMV — pre_w[r] = dot(hc_fn[r,:], hc_state) * rms_inv
    // Only 4 output rows (not 24).
    threadgroup float mixes[4];
    for (uint r = tid; r < ds4_n_hc; r += tg_sz) {
        float acc = 0.0f;
        device const float* row = hc_fn + r * flat_size;
        for (uint i = 0; i < flat_size; i++) {
            acc += row[i] * hc_state[i];
        }
        mixes[r] = acc * rms_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Sigmoid activation
    if (tid < ds4_n_hc) {
        float s0 = hc_scale[0];
        pre_w[tid] = 1.0f / (1.0f + exp(-(mixes[tid] * s0 + hc_base[tid]))) + ds4_hc_eps;
    }
}

// Phase 2: weighted sum → hidden.
// Reuses ds4_hc_pre_mix kernel.

// ── Embedding broadcast to HC streams ────────────────────────────
// After embedding lookup, copy emb[n_embd] to all n_hc streams.
kernel void ds4_emb_broadcast(
    device const float* emb      [[buffer(0)]],  // [n_embd]
    device float* hc_state       [[buffer(1)]],  // [n_hc * n_embd]
    constant uint& n_embd        [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_embd) return;
    float v = emb[tid];
    for (uint s = 0; s < ds4_n_hc; s++) {
        hc_state[s * n_embd + tid] = v;
    }
}

// ── RoPE with precomputed cos/sin table ──────────────────────────
// Applies RoPE to Q (all n_heads) and KV (1 head) using precomputed cos/sin.
// Q layout: q[h * head_dim + nope + 2*j], q[h * head_dim + nope + 2*j+1]
// KV layout: kv[nope + 2*j], kv[nope + 2*j+1]
kernel void ds4_rope_table(
    device float* data           [[buffer(0)]],  // Q or KV buffer
    device const float* cos_t    [[buffer(1)]],  // [rope_dim/2]
    device const float* sin_t    [[buffer(2)]],  // [rope_dim/2]
    constant uint& n_heads       [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& nope          [[buffer(5)]],  // offset within head to rope region
    constant uint& rope_dim      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint nd = rope_dim / 2;
    uint total = n_heads * nd;
    if (tid >= total) return;
    uint h = tid / nd;
    uint j = tid % nd;
    uint base = h * head_dim + nope + 2 * j;
    float re = data[base];
    float im = data[base + 1];
    float c = cos_t[j];
    float s = sin_t[j];
    data[base]     = re * c - im * s;
    data[base + 1] = re * s + im * c;
}

// ── Inverse RoPE with precomputed cos/sin table ──────────────────
// Same as ds4_rope_table but negates sin (conjugate rotation).
kernel void ds4_inv_rope_table(
    device float* data           [[buffer(0)]],
    device const float* cos_t    [[buffer(1)]],
    device const float* sin_t    [[buffer(2)]],
    constant uint& n_heads       [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& nope          [[buffer(5)]],
    constant uint& rope_dim      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint nd = rope_dim / 2;
    uint total = n_heads * nd;
    if (tid >= total) return;
    uint h = tid / nd;
    uint j = tid % nd;
    uint base = h * head_dim + nope + 2 * j;
    float re = data[base];
    float im = data[base + 1];
    float c = cos_t[j];
    float s = sin_t[j];
    // Inverse: negate sin
    data[base]     = re * c + im * s;
    data[base + 1] = -re * s + im * c;
}

// ── Weighted expert accumulation ─────────────────────────────────
// hidden[i] = Σ_slot weights[slot] * expert_scratch[slot * n_embd + i]
// For slot 0: direct scaled write. For slot 1+: fused multiply-add.
kernel void ds4_weighted_accum(
    device const float* expert_scratch [[buffer(0)]],  // [n_slots * n_embd]
    device const float* weights        [[buffer(1)]],  // [n_slots]
    device float* hidden               [[buffer(2)]],  // [n_embd]
    constant uint& n_embd              [[buffer(3)]],
    constant uint& n_slots             [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_embd) return;
    float acc = 0.0f;
    for (uint slot = 0; slot < n_slots; slot++) {
        acc += weights[slot] * expert_scratch[slot * n_embd + tid];
    }
    hidden[tid] = acc;
}

// ── Turbo/Q8_0 SDPA for head_dim=512 ─────────────────────────────
// Same as sdpa_fa2_turbo but with reduced block size (8) to fit 32KB
// threadgroup memory with hd=512:
//   q_local[512] + kv_block[8*512] + out_acc[512] + scores[8] + shared[8]
//   = 2KB + 16KB + 2KB + 32B + 32B ≈ 20KB

constant uint ds4_sdpa_block_size = 8;
constant uint ds4_sdpa_max_hd = 512;
constant float ds4_sparse_v_threshold = 1e-6f;

// Forward-declare turbo dequant (defined in sdpa.metal via common include)
// We inline the dequant logic here to avoid cross-file linkage issues.

// Lloyd-Max centroids for TurboQuant
constant float ds4_turbo2_centroids[4] = {-1.5104176f, -0.4527800f, 0.4527800f, 1.5104176f};
constant float ds4_turbo3_centroids[8] = {-2.1519927f, -1.3439093f, -0.7560053f, -0.2451210f,
                                           0.2451210f,  0.7560053f,  1.3439093f,  2.1519927f};
constant float ds4_turbo4_centroids[16] = {
    -2.5777843f, -1.8854330f, -1.4447610f, -1.0921330f,
    -0.7832860f, -0.5005730f, -0.2334144f,  0.0000000f,
     0.2334144f,  0.5005730f,  0.7832860f,  1.0921330f,
     1.4447610f,  1.8854330f,  2.5777843f,  0.0000000f
};

// WHT-32 in-place transform
inline void ds4_wht32(thread float* x) {
    // Radix-2 butterfly: 5 stages for N=32
    // Note: 'half' is a reserved type in MSL, use 'h_size' instead.
    for (int h_size = 1; h_size < 32; h_size <<= 1) {
        for (int i = 0; i < 32; i += h_size * 2) {
            for (int j = i; j < i + h_size; j++) {
                float a = x[j];
                float b = x[j + h_size];
                x[j]          = a + b;
                x[j + h_size] = a - b;
            }
        }
    }
}

inline void ds4_turbo_dequant_block(device const uchar* src, thread float* dst, uint bits) {
    // First 4 bytes: norm as float
    float norm = *((device const float*)src);
    device const uchar* packed = src + 4;

    if (bits == 2) {
        for (int i = 0; i < 32; i++) {
            uint byte_idx = i / 4;
            uint bit_off = (i % 4) * 2;
            uint val = (uint(packed[byte_idx]) >> bit_off) & 0x3;
            dst[i] = ds4_turbo2_centroids[val];
        }
    } else if (bits == 3) {
        for (int i = 0; i < 32; i++) {
            uint bit_pos = i * 3;
            uint byte_idx = bit_pos / 8;
            uint bit_off = bit_pos % 8;
            uint val = (uint(packed[byte_idx]) >> bit_off);
            if (bit_off > 5) {
                val |= uint(packed[byte_idx + 1]) << (8 - bit_off);
            }
            dst[i] = ds4_turbo3_centroids[val & 0x7];
        }
    } else { // bits == 4
        for (int i = 0; i < 32; i++) {
            uint byte_idx = i / 2;
            uint nibble = (i % 2 == 0) ? (uint(packed[byte_idx]) & 0xF)
                                        : (uint(packed[byte_idx]) >> 4);
            dst[i] = ds4_turbo4_centroids[nibble];
        }
    }

    // Inverse WHT + rescale
    ds4_wht32(dst);
    float s = norm / 32.0;
    for (int i = 0; i < 32; i++) dst[i] *= s;
}

kernel void sdpa_fa2_turbo_hd512(
    device const float* Q,           // [nh * hd]
    device const uchar* K_cache,     // turbo-packed or f32 KV cache
    device const uchar* V_cache,     // turbo-packed or f32 KV cache
    device float* output,            // [nh * hd]
    constant uint& nh,
    constant uint& nkv,
    constant uint& hd,
    constant uint& sl,
    constant float& scale,
    constant uint& bits_k,           // 0=f32, 2/3/4=turbo, 8=q8_0
    constant uint& bits_v,
    constant uint& block_bytes_k,
    constant uint& block_bytes_v,
    uint h     [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    if (h >= nh) return;

    uint hpg = nh / nkv;
    uint kvh = h / hpg;
    uint kvd = nkv * hd;
    uint num_blocks = (sl + ds4_sdpa_block_size - 1) / ds4_sdpa_block_size;

    threadgroup float q_local[ds4_sdpa_max_hd];
    threadgroup float kv_block[ds4_sdpa_block_size * ds4_sdpa_max_hd];
    threadgroup float scores[ds4_sdpa_block_size];
    threadgroup float shared[8];

    for (uint d = tid; d < hd; d += tg_sz) {
        q_local[d] = Q[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    threadgroup float out_acc[ds4_sdpa_max_hd];
    for (uint d = tid; d < hd; d += tg_sz) {
        out_acc[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block = 0; block < num_blocks; block++) {
        uint block_start = block * ds4_sdpa_block_size;
        uint block_len = min(ds4_sdpa_block_size, sl - block_start);

        // Load K block with dequant
        for (uint t = tid; t < block_len; t += tg_sz) {
            uint t_global = block_start + t;
            if (bits_k == 0) {
                device const float* K_f32 = (device const float*)K_cache;
                uint k_base = t_global * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++) {
                    kv_block[t * hd + d] = K_f32[k_base + d];
                }
            } else if (bits_k == 8) {
                uint elem_base = t_global * kvd + kvh * hd;
                uint n_q8_blocks = hd / 32;
                for (uint blk = 0; blk < n_q8_blocks; blk++) {
                    uint elem_idx = elem_base + blk * 32;
                    uint q8_block_idx = elem_idx / 32;
                    uint byte_off = q8_block_idx * 34;
                    device const uchar* bp = K_cache + byte_off;
                    float s = float(*((device const half*)bp));
                    for (uint d = 0; d < 32; d++) {
                        kv_block[t * hd + blk * 32 + d] = s * float(((device const char*)(bp + 2))[d]);
                    }
                }
            } else {
                uint elem_base = t_global * kvd + kvh * hd;
                uint n_turbo_blocks = hd / 32;
                for (uint blk = 0; blk < n_turbo_blocks; blk++) {
                    uint elem_idx = elem_base + blk * 32;
                    uint turbo_block_idx = elem_idx / 32;
                    uint byte_off = turbo_block_idx * block_bytes_k;
                    float dequant_buf[32];
                    ds4_turbo_dequant_block(K_cache + byte_off, dequant_buf, bits_k);
                    for (uint d = 0; d < 32; d++) {
                        kv_block[t * hd + blk * 32 + d] = dequant_buf[d];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK scores
        for (uint t = tid; t < block_len; t += tg_sz) {
            float dot_val = 0.0f;
            for (uint d = 0; d < hd; d++) {
                dot_val += q_local[d] * kv_block[t * hd + d];
            }
            scores[t] = dot_val * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax max reduction
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

        float m_prev = m_i;
        m_i = max(m_i, m_new);
        float rescale_factor = exp(m_prev - m_i);
        l_i *= rescale_factor;
        for (uint d = tid; d < hd; d += tg_sz) {
            out_acc[d] *= rescale_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        // Load V block with dequant
        for (uint t = tid; t < block_len; t += tg_sz) {
            if (scores[t] < ds4_sparse_v_threshold) continue;
            uint t_global = block_start + t;
            if (bits_v == 0) {
                device const float* V_f32 = (device const float*)V_cache;
                uint v_base = t_global * kvd + kvh * hd;
                for (uint d = 0; d < hd; d++) {
                    kv_block[t * hd + d] = V_f32[v_base + d];
                }
            } else if (bits_v == 8) {
                uint elem_base = t_global * kvd + kvh * hd;
                uint n_q8_blocks = hd / 32;
                for (uint blk = 0; blk < n_q8_blocks; blk++) {
                    uint elem_idx = elem_base + blk * 32;
                    uint q8_block_idx = elem_idx / 32;
                    uint byte_off = q8_block_idx * 34;
                    device const uchar* bp = V_cache + byte_off;
                    float s = float(*((device const half*)bp));
                    for (uint d = 0; d < 32; d++) {
                        kv_block[t * hd + blk * 32 + d] = s * float(((device const char*)(bp + 2))[d]);
                    }
                }
            } else {
                uint elem_base = t_global * kvd + kvh * hd;
                uint n_turbo_blocks = hd / 32;
                for (uint blk = 0; blk < n_turbo_blocks; blk++) {
                    uint elem_idx = elem_base + blk * 32;
                    uint turbo_block_idx = elem_idx / 32;
                    uint byte_off = turbo_block_idx * block_bytes_v;
                    float dequant_buf[32];
                    ds4_turbo_dequant_block(V_cache + byte_off, dequant_buf, bits_v);
                    for (uint d = 0; d < 32; d++) {
                        kv_block[t * hd + blk * 32 + d] = dequant_buf[d];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate V
        for (uint d = tid; d < hd; d += tg_sz) {
            float acc = 0.0f;
            for (uint t = 0; t < block_len; t++) {
                if (scores[t] < ds4_sparse_v_threshold) continue;
                acc += scores[t] * kv_block[t * hd + d];
            }
            out_acc[d] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalize
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (uint d = tid; d < hd; d += tg_sz) {
        output[h * hd + d] = out_acc[d] * inv_l;
    }
}

// ── GPU Top-K for MoE routing ────────────────────────────────────
// Finds top-k expert indices from router logits. Single threadgroup.
// Writes selected expert IDs + weights to device buffers for GPU expert dispatch.
kernel void ds4_topk_routing(
    device const float* logits       [[buffer(0)]],  // [n_experts]
    device uint* top_ids             [[buffer(1)]],  // [k] output
    device float* top_weights        [[buffer(2)]],  // [k] output
    constant uint& n_experts         [[buffer(3)]],
    constant uint& k                 [[buffer(4)]],
    constant float& weight_scale     [[buffer(5)]],  // expert_weights_scale
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // sqrt_softplus: sqrt(log(1 + exp(x)))
    threadgroup float probs[256];
    threadgroup float selection[256];
    
    // Compute probs = sqrt_softplus(logits)
    for (uint i = tid; i < n_experts; i += tg_size) {
        float x = logits[i];
        float sp = log(1.0f + exp(x));
        probs[i] = sqrt(sp);
        selection[i] = probs[i]; // selection = probs (no bias in this path)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Serial top-k (single thread — k is small, n_experts=256)
    if (tid == 0) {
        for (uint j = 0; j < k; j++) {
            float best = -INFINITY;
            uint best_idx = 0;
            for (uint i = 0; i < n_experts; i++) {
                if (selection[i] > best) {
                    best = selection[i];
                    best_idx = i;
                }
            }
            top_ids[j] = best_idx;
            top_weights[j] = probs[best_idx]; // unbiased prob
            selection[best_idx] = -INFINITY; // exclude
        }
        // L2 normalize weights: scale = weight_scale / sqrt(sum(w^2))
        float wss = 0.0f;
        for (uint j = 0; j < k; j++) wss += top_weights[j] * top_weights[j];
        if (wss > 0.0f) {
            float inv = weight_scale / sqrt(wss);
            for (uint j = 0; j < k; j++) top_weights[j] *= inv;
        }
    }
}

// ── Batched MoE Expert GEMV (MXFP4) ─────────────────────────────────
// Computes gate+up GEMVs for all selected experts in ONE dispatch.
// Expert IDs read from GPU device memory (from ds4_topk_routing).
// Each threadgroup handles one output row of one expert.
// Total threadgroups = n_active_experts * ff_exp.
//
// Weight layout: experts packed contiguously, each expert at offset eid * w_stride.
// Scales layout: experts packed contiguously, each at offset eid * s_stride.
// E2M1 nibble dequant with E8M0 per-group scales (MXFP4 SafeTensors format).

kernel void ds4_moe_gate_up_mxfp4(
    device const float* x             [[buffer(0)]],   // [n_in]
    device const packed_uchar4* gate_w [[buffer(1)]],  // all experts gate weights
    device const uchar* gate_s        [[buffer(2)]],   // all experts gate scales
    device const packed_uchar4* up_w   [[buffer(3)]],  // all experts up weights
    device const uchar* up_s          [[buffer(4)]],   // all experts up scales
    device float* gate_out            [[buffer(5)]],   // [n_slots * ff]
    device float* up_out              [[buffer(6)]],   // [n_slots * ff]
    device const uint* expert_ids     [[buffer(7)]],   // [k] selected expert indices
    constant uint& k_experts          [[buffer(8)]],
    constant uint& ff                 [[buffer(9)]],
    constant uint& n_in               [[buffer(10)]],
    constant uint& w_stride_words     [[buffer(11)]],  // expert stride in u32 words for weights
    constant uint& s_stride           [[buffer(12)]],  // expert stride in bytes for scales
    constant uint& gs                 [[buffer(13)]],
    constant uint& slot_offset        [[buffer(14)]],  // output slot offset (shared expert slots)
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    uint slot = tgid / ff;
    uint row = tgid % ff;
    if (slot >= k_experts) return;
    
    uint eid = expert_ids[slot];
    uint out_slot = slot_offset + slot;
    
    // MXFP4 GEMV for gate
    uint gpr = (n_in + gs - 1) / gs;
    uint wpg = gs / 8;
    
    // Gate GEMV
    {
        uint w_base = eid * w_stride_words + row * gpr * wpg;
        uint s_base = eid * s_stride + row * gpr;
        
        float sum = 0.0f;
        for (uint g = tid; g < gpr; g += tg_size) {
            float scale = (gate_s[s_base + g] == 0) ? 0.0f
                : as_type<float>(uint(gate_s[s_base + g]) << 23); // E8M0
            uint xo = g * gs;
            uint wg = w_base + g * wpg;
            float gdot = 0.0f;
            for (uint w = 0; w < wpg && xo + w * 8 < n_in; w++) {
                uint xi = xo + w * 8;
                packed_uchar4 bytes = gate_w[wg + w];
                uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) |
                            (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
                uint rem = min(uint(8), n_in - xi);
                if (rem == 8) {
                    float4 q_lo = float4(mxfp4_lut[word & 0xF], mxfp4_lut[(word >> 4) & 0xF],
                                          mxfp4_lut[(word >> 8) & 0xF], mxfp4_lut[(word >> 12) & 0xF]);
                    float4 q_hi = float4(mxfp4_lut[(word >> 16) & 0xF], mxfp4_lut[(word >> 20) & 0xF],
                                          mxfp4_lut[(word >> 24) & 0xF], mxfp4_lut[(word >> 28) & 0xF]);
                    float4 x_lo = *(device const float4*)(x + xi);
                    float4 x_hi = *(device const float4*)(x + xi + 4);
                    gdot += dot(q_lo, x_lo) + dot(q_hi, x_hi);
                } else {
                    for (uint i = 0; i < rem; i++)
                        gdot += mxfp4_lut[(word >> (i * 4)) & 0xF] * x[xi + i];
                }
            }
            sum += scale * gdot;
        }
        threadgroup float shared_g[8];
        sum = threadgroup_reduce_sum(sum, shared_g, tid, tg_size);
        if (tid == 0) gate_out[out_slot * ff + row] = sum;
    }
    
    // Up GEMV (same structure, different weights)
    {
        uint w_base = eid * w_stride_words + row * gpr * wpg;
        uint s_base = eid * s_stride + row * gpr;
        
        float sum = 0.0f;
        for (uint g = tid; g < gpr; g += tg_size) {
            float scale = (up_s[s_base + g] == 0) ? 0.0f
                : as_type<float>(uint(up_s[s_base + g]) << 23);
            uint xo = g * gs;
            uint wg = w_base + g * wpg;
            float gdot = 0.0f;
            for (uint w = 0; w < wpg && xo + w * 8 < n_in; w++) {
                uint xi = xo + w * 8;
                packed_uchar4 bytes = up_w[wg + w];
                uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) |
                            (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
                uint rem = min(uint(8), n_in - xi);
                if (rem == 8) {
                    float4 q_lo = float4(mxfp4_lut[word & 0xF], mxfp4_lut[(word >> 4) & 0xF],
                                          mxfp4_lut[(word >> 8) & 0xF], mxfp4_lut[(word >> 12) & 0xF]);
                    float4 q_hi = float4(mxfp4_lut[(word >> 16) & 0xF], mxfp4_lut[(word >> 20) & 0xF],
                                          mxfp4_lut[(word >> 24) & 0xF], mxfp4_lut[(word >> 28) & 0xF]);
                    float4 x_lo = *(device const float4*)(x + xi);
                    float4 x_hi = *(device const float4*)(x + xi + 4);
                    gdot += dot(q_lo, x_lo) + dot(q_hi, x_hi);
                } else {
                    for (uint i = 0; i < rem; i++)
                        gdot += mxfp4_lut[(word >> (i * 4)) & 0xF] * x[xi + i];
                }
            }
            sum += scale * gdot;
        }
        threadgroup float shared_u[8];
        sum = threadgroup_reduce_sum(sum, shared_u, tid, tg_size);
        if (tid == 0) up_out[out_slot * ff + row] = sum;
    }
}

// ── Batched MoE Down GEMV (MXFP4) ──────────────────────────────────
// Down projection for all selected experts. Reads from gate_scratch (post-activation).
// Each threadgroup handles one output row of one expert.
kernel void ds4_moe_down_mxfp4(
    device const float* activated     [[buffer(0)]],   // [n_slots * ff] post-activation input
    device const packed_uchar4* down_w [[buffer(1)]],  // all experts down weights
    device const uchar* down_s        [[buffer(2)]],   // all experts down scales
    device float* expert_out          [[buffer(3)]],   // [n_slots * n_out]
    device const uint* expert_ids     [[buffer(4)]],   // [k] selected expert indices
    constant uint& k_experts          [[buffer(5)]],
    constant uint& n_out              [[buffer(6)]],   // output dim (n_embd)
    constant uint& ff                 [[buffer(7)]],   // input dim (ff_exp)
    constant uint& w_stride_words     [[buffer(8)]],
    constant uint& s_stride           [[buffer(9)]],
    constant uint& gs                 [[buffer(10)]],
    constant uint& slot_offset        [[buffer(11)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    uint slot = tgid / n_out;
    uint row = tgid % n_out;
    if (slot >= k_experts) return;
    
    uint eid = expert_ids[slot];
    uint out_slot = slot_offset + slot;
    uint in_slot = out_slot;
    
    device const float* x = activated + in_slot * ff;
    
    uint gpr = (ff + gs - 1) / gs;
    uint wpg = gs / 8;
    uint w_base = eid * w_stride_words + row * gpr * wpg;
    uint s_base = eid * s_stride + row * gpr;
    
    float sum = 0.0f;
    for (uint g = tid; g < gpr; g += tg_size) {
        float scale = (down_s[s_base + g] == 0) ? 0.0f
            : as_type<float>(uint(down_s[s_base + g]) << 23);
        uint xo = g * gs;
        uint wg = w_base + g * wpg;
        float gdot = 0.0f;
        for (uint w = 0; w < wpg && xo + w * 8 < ff; w++) {
            uint xi = xo + w * 8;
            packed_uchar4 bytes = down_w[wg + w];
            uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) |
                        (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
            uint rem = min(uint(8), ff - xi);
            if (rem == 8) {
                float4 q_lo = float4(mxfp4_lut[word & 0xF], mxfp4_lut[(word >> 4) & 0xF],
                                      mxfp4_lut[(word >> 8) & 0xF], mxfp4_lut[(word >> 12) & 0xF]);
                float4 q_hi = float4(mxfp4_lut[(word >> 16) & 0xF], mxfp4_lut[(word >> 20) & 0xF],
                                      mxfp4_lut[(word >> 24) & 0xF], mxfp4_lut[(word >> 28) & 0xF]);
                float4 x_lo = *(device const float4*)(x + xi);
                float4 x_hi = *(device const float4*)(x + xi + 4);
                gdot += dot(q_lo, x_lo) + dot(q_hi, x_hi);
            } else {
                for (uint i = 0; i < rem; i++)
                    gdot += mxfp4_lut[(word >> (i * 4)) & 0xF] * x[xi + i];
            }
        }
        sum += scale * gdot;
    }
    
    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) expert_out[out_slot * n_out + row] = sum;
}

// ── Weightless RMS Norm ──────────────────────────────────────────────
// RMS normalize in-place without learned weight: x /= sqrt(mean(x²) + eps).
// Used for DS4 per-head Q normalization (no weight parameter).
kernel void ds4_rms_norm_noweight(
    device float* data               [[buffer(0)]],  // [n] in-place
    constant uint& n                 [[buffer(1)]],
    constant float& eps              [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Phase 1: sum of squares
    threadgroup float shared[8];
    float local_ss = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = data[i];
        local_ss += v * v;
    }
    local_ss = threadgroup_reduce_sum(local_ss, shared, tid, tg_size);
    
    threadgroup float rms_inv_shared;
    if (tid == 0) {
        rms_inv_shared = rsqrt(local_ss / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: scale
    float rms_inv = rms_inv_shared;
    for (uint i = tid; i < n; i += tg_size) {
        data[i] *= rms_inv;
    }
}
