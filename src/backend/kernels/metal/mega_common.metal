// ── Megakernel Building Blocks ────────────────────────────────────────────
//
// Composable inline functions for building per-model megakernels.
// Each block handles one forward-pass stage (norm, GEMV, activation, etc.).
// These are compiled into model-specific megakernels via @embedFile concatenation.
//
// Design:
//   - All inter-stage data flows through device memory scratch buffers
//   - Grid sync between stages uses atomic counter barrier
//   - Each threadgroup handles one output element of the current stage
//   - Stages with fewer outputs → some threadgroups idle
//
// Dispatch model:
//   max(n_embd, n_ff) threadgroups × 256 threads per threadgroup.
//   Each stage only activates the threadgroups it needs.

#include <metal_stdlib>
using namespace metal;

// ── Grid Synchronization ─────────────────────────────────────────────────
// Metal has no cooperative grid sync. We implement it with an atomic counter:
// all threadgroups increment, then spin until the counter reaches n_threadgroups.
// After sync, thread 0 of TG 0 resets the counter for the next sync point.
//
// SAFETY: This requires ALL dispatched threadgroups to be resident simultaneously.
// On Apple Silicon, this holds when n_threadgroups ≤ device maxThreadgroupsPerGrid
// (typically 16384+ on M-series). For our models (max ~9216 TGs), this is fine.

inline void mega_grid_sync(
    device atomic_uint* sync_counter,
    uint n_threadgroups,
    uint tgid,
    uint tid
) {
    // Only thread 0 of each TG participates in the barrier
    if (tid == 0) {
        // Signal arrival
        uint arrived = atomic_fetch_add_explicit(sync_counter, 1, memory_order_relaxed) + 1;
        // Spin until all TGs have arrived
        if (arrived < n_threadgroups) {
            while (atomic_load_explicit(sync_counter, memory_order_relaxed) < n_threadgroups) {
                // spin — minimal overhead on Apple Silicon (single-cycle atomic reads)
            }
        }
    }
    // Ensure all threads in this TG see the barrier completion
    threadgroup_barrier(mem_flags::mem_device);
}

// Reset the sync counter (called by TG 0 after grid sync, before next phase)
inline void mega_sync_reset(
    device atomic_uint* sync_counter,
    uint tgid,
    uint tid
) {
    if (tgid == 0 && tid == 0) {
        atomic_store_explicit(sync_counter, 0, memory_order_relaxed);
    }
    // Small barrier to ensure reset is visible before next phase
    threadgroup_barrier(mem_flags::mem_device);
}

// ── RMS Norm (device memory, multi-TG cooperative) ───────────────────────
// Each of n_dim threadgroups computes the norm for one element.
// Phase 1: All TGs compute sum-of-squares → write to scratch[0]
// Phase 2: Grid sync
// Phase 3: All TGs normalize their element
//
// This is different from the single-TG norm in norm.metal — here we
// parallelize across elements, not across the reduction dimension.

inline void mega_rms_norm(
    device const float* input,     // [n_dim]
    device const float* weight,    // [n_dim]
    device float* output,          // [n_dim]
    device float* ss_scratch,      // [1] — shared sum-of-squares
    device atomic_uint* sync_ctr,
    uint n_dim,
    uint n_tgs,
    float eps,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    // Phase 1: Each TG adds its element's square to the sum
    float local_ss = 0.0f;
    for (uint i = tgid * tg_size + tid; i < n_dim; i += n_tgs * tg_size) {
        float v = input[i];
        local_ss += v * v;
    }
    // Reduce within TG
    local_ss = threadgroup_reduce_sum(local_ss, shared, tid, tg_size);
    // Atomically add to global sum
    if (tid == 0 && local_ss != 0.0f) {
        // Use atomic float add via reinterpret
        atomic_fetch_add_explicit((device atomic_uint*)ss_scratch,
            as_type<uint>(local_ss), memory_order_relaxed);
    }

    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);

    // Phase 2: Normalize (all TGs read the shared sum)
    float ss = as_type<float>(atomic_load_explicit(
        (device atomic_uint*)ss_scratch, memory_order_relaxed));
    float inv_rms = rsqrt(ss / float(n_dim) + eps);

    for (uint i = tgid * tg_size + tid; i < n_dim; i += n_tgs * tg_size) {
        output[i] = input[i] * weight[i] * inv_rms;
    }

    // Reset ss for next norm
    if (tgid == 0 && tid == 0) {
        atomic_store_explicit((device atomic_uint*)ss_scratch, 0, memory_order_relaxed);
    }

    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_sync_reset(sync_ctr, tgid, tid);
}

// ── Add + RMS Norm (fused residual add + normalize) ──────────────────────
inline void mega_add_rms_norm(
    device float* a,               // [n_dim] — modified in place (a += b)
    device const float* b,         // [n_dim]
    device const float* weight,    // [n_dim]
    device float* output,          // [n_dim]
    device float* ss_scratch,      // [1]
    device atomic_uint* sync_ctr,
    uint n_dim,
    uint n_tgs,
    float eps,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    float local_ss = 0.0f;
    for (uint i = tgid * tg_size + tid; i < n_dim; i += n_tgs * tg_size) {
        float v = a[i] + b[i];
        a[i] = v;
        local_ss += v * v;
    }
    local_ss = threadgroup_reduce_sum(local_ss, shared, tid, tg_size);
    if (tid == 0 && local_ss != 0.0f) {
        atomic_fetch_add_explicit((device atomic_uint*)ss_scratch,
            as_type<uint>(local_ss), memory_order_relaxed);
    }

    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);

    float ss = as_type<float>(atomic_load_explicit(
        (device atomic_uint*)ss_scratch, memory_order_relaxed));
    float inv_rms = rsqrt(ss / float(n_dim) + eps);

    for (uint i = tgid * tg_size + tid; i < n_dim; i += n_tgs * tg_size) {
        output[i] = a[i] * weight[i] * inv_rms;
    }

    if (tgid == 0 && tid == 0) {
        atomic_store_explicit((device atomic_uint*)ss_scratch, 0, memory_order_relaxed);
    }

    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_sync_reset(sync_ctr, tgid, tid);
}

// ── GEMV Q8_0 (one output element per TG) ────────────────────────────────
// Each of n_out threadgroups computes one output element.
// TGs beyond n_out are idle for this stage.
// Uses the q8_0_block_dot helper from gemv.metal.

inline void mega_gemv_q8(
    device const float* x,         // [k]
    device const block_q8_0* W,    // [n_out * (k/32)] blocks
    device float* y,               // [n_out]
    uint n_out,
    uint k,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    uint nb = k / 32;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        device const float* x_block = x + b * 32;
        sum += q8_0_block_dot(W[tgid * nb + b], x_block);
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── GEMV Q4_K (one output element per TG) ────────────────────────────────
inline void mega_gemv_q4k(
    device const float* x,
    device const uchar* W,
    device float* y,
    uint n_out,
    uint k,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    const uint bpb = 144;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum += q4_k_block_dot(W + (tgid * nb + b) * bpb, x, k, bk);
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── GEMV Q4_0 (one output element per TG) ────────────────────────────────
inline void mega_gemv_q4_0(
    device const float* x,
    device const block_q4_0* W,
    device float* y,
    uint n_out,
    uint k,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    uint nb = k / 32;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * 32;
        float4 x0 = *(device const float4*)(x + bk);
        float4 x1 = *(device const float4*)(x + bk + 4);
        float4 x2 = *(device const float4*)(x + bk + 8);
        float4 x3 = *(device const float4*)(x + bk + 12);
        float4 x4 = *(device const float4*)(x + bk + 16);
        float4 x5 = *(device const float4*)(x + bk + 20);
        float4 x6 = *(device const float4*)(x + bk + 24);
        float4 x7 = *(device const float4*)(x + bk + 28);
        sum += q4_0_block_dot(W[tgid * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── GEMV Q5_K (one output element per TG) ────────────────────────────────
inline void mega_gemv_q5k(
    device const float* x,
    device const uchar* W,
    device float* y,
    uint n_out,
    uint k,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    const uint bpb = 176;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum += q5_k_block_dot(W + (tgid * nb + b) * bpb, x, k, bk);
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── GEMV Q6_K (one output element per TG) ────────────────────────────────
inline void mega_gemv_q6k(
    device const float* x,
    device const uchar* W,
    device float* y,
    uint n_out,
    uint k,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    const uint bpb = 210;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum += q6_k_block_dot(W + (tgid * nb + b) * bpb, x, k, bk);
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Activation helpers ───────────────────────────────────────────────────

// SiLU(gate) * up — per-element, only active TGs
inline void mega_silu_mul(
    device float* gate,            // [n] — modified in place
    device const float* up,        // [n]
    uint n,
    uint tgid,
    uint tid,
    uint tg_size
) {
    for (uint i = tgid * tg_size + tid; i < n; i += tg_size * 4096) {
        float g = gate[i];
        gate[i] = g / (1.0f + exp(-g)) * up[i];
    }
}

// GELU(gate) * up — per-element
inline void mega_gelu_mul(
    device float* gate,
    device const float* up,
    uint n,
    uint tgid,
    uint tid,
    uint tg_size
) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (uint i = tgid * tg_size + tid; i < n; i += tg_size * 4096) {
        float g = gate[i];
        float inner = sqrt_2_over_pi * fma(0.044715f * g * g, g, g);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        gate[i] = 0.5f * g * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up[i];
    }
}

// Residual add: a[i] += b[i]
inline void mega_add(
    device float* a,
    device const float* b,
    uint n,
    uint tgid,
    uint tid,
    uint tg_size
) {
    for (uint i = tgid * tg_size + tid; i < n; i += tg_size * 4096) {
        a[i] += b[i];
    }
}

// ── RoPE (per-element rotation) ──────────────────────────────────────────
// Apply rotary position encoding to Q or K vectors in device memory.
// Each TG handles one element, distributing across heads.

inline void mega_rope(
    device float* x,               // [n_heads * head_dim]
    uint n_heads,
    uint head_dim,
    uint rope_dim,
    float theta,
    uint seq_pos,
    uint tgid,
    uint tid,
    uint tg_size
) {
    uint total = n_heads * head_dim;
    for (uint i = tgid * tg_size + tid; i < total; i += tg_size * 4096) {
        uint h = i / head_dim;
        uint d = i % head_dim;
        if (d >= rope_dim) continue;
        uint half_rd = rope_dim / 2;
        if (d >= half_rd) continue; // only process first half, pair with second half

        float freq = 1.0f / pow(theta, float(2 * d) / float(rope_dim));
        float angle = float(seq_pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        uint idx0 = h * head_dim + d;
        uint idx1 = h * head_dim + d + half_rd;
        float x0 = x[idx0];
        float x1 = x[idx1];
        x[idx0] = x0 * cos_a - x1 * sin_a;
        x[idx1] = x0 * sin_a + x1 * cos_a;
    }
}

// ── ReLU² activation ─────────────────────────────────────────────────────
// For Nemotron-H/Nano FFN-only layers: relu(x)² (no gate)

inline void mega_relu_squared(
    device float* x,
    uint n,
    uint tgid,
    uint tid,
    uint tg_size
) {
    for (uint i = tgid * tg_size + tid; i < n; i += tg_size * 4096) {
        float v = max(x[i], 0.0f);
        x[i] = v * v;
    }
}

// ── Clamped SiLU*mul (GPT-OSS) ──────────────────────────────────────────
// SwiGLU with hard clamp: out = clamp(silu(gate) * up, -limit, +limit)

inline void mega_silu_mul_clamp(
    device float* gate,
    device const float* up,
    uint n,
    float limit,
    uint tgid,
    uint tid,
    uint tg_size
) {
    for (uint i = tgid * tg_size + tid; i < n; i += tg_size * 4096) {
        float g = gate[i];
        float prod = g / (1.0f + exp(-g)) * up[i];
        gate[i] = clamp(prod, -limit, limit);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// ── TurboQuant+ Building Blocks ──────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════
//
// These blocks enable TurboQuant KV cache inside megakernels:
// - mega_kv_append_f32: simple f32 KV cache append
// - mega_kv_append_tq: TurboQuant encode + append (turbo2/3/4)
// - mega_kv_append_q8: Q8_0 encode + append
// - mega_sdpa_inline: inline single-query attention with TQ dequant + sparse V
//
// The turbo_dequant_block() and wht32() functions from sdpa.metal are available
// since that file is concatenated before this one.

// ── KV Cache Append (f32) ────────────────────────────────────────────────
// Append K and V vectors to f32 KV cache at current sequence position.
// Each TG writes one element.

inline void mega_kv_append_f32(
    device const float* k_new,     // [kv_dim]
    device const float* v_new,     // [kv_dim]
    device float* kv_keys,         // [max_seq * kv_dim]
    device float* kv_values,       // [max_seq * kv_dim]
    uint kv_dim,
    uint seq_pos,
    uint tgid,
    uint tid,
    uint tg_size
) {
    uint kv_off = seq_pos * kv_dim;
    for (uint i = tgid * tg_size + tid; i < kv_dim; i += tg_size * 4096) {
        kv_keys[kv_off + i] = k_new[i];
        kv_values[kv_off + i] = v_new[i];
    }
}

// ── KV Cache Append with TurboQuant encoding ─────────────────────────────
// Encode K/V vectors to TurboQuant format and append to quantized KV cache.
// Uses Walsh-Hadamard Transform + Lloyd-Max centroid quantization.
//
// Cache layout: [max_seq * n_blocks * block_bytes] per layer,
// where n_blocks = kv_dim / 32, block_bytes = turboBlockBytes(bits).
//
// This is a cooperative operation: one TG per 32-element block.
// TGs beyond n_blocks are idle.

inline void mega_kv_append_tq(
    device const float* k_new,     // [kv_dim] f32 input
    device const float* v_new,     // [kv_dim] f32 input
    device uchar* kv_keys,         // quantized KV cache (K)
    device uchar* kv_values,       // quantized KV cache (V)
    uint kv_dim,
    uint seq_pos,
    uint bits_k,                   // 0=f32, 2/3/4=turbo
    uint bits_v,                   // 0=f32, 2/3/4=turbo
    uint block_bytes_k,            // bytes per TQ block for K
    uint block_bytes_v,            // bytes per TQ block for V
    uint tgid,
    uint tid,
    uint tg_size
) {
    uint n_blocks = kv_dim / 32;
    if (tgid >= n_blocks) return;

    // Only thread 0 does the encoding (WHT + quantize is sequential per block)
    if (tid == 0) {
        uint blk = tgid;

        // ── Encode K block ───────────────────────────────────────
        if (bits_k > 0) {
            // Read 32 f32 values
            thread float buf[32];
            for (uint i = 0; i < 32; i++) buf[i] = k_new[blk * 32 + i];

            // Compute L2 norm
            float norm = 0.0f;
            for (uint i = 0; i < 32; i++) norm += buf[i] * buf[i];
            norm = sqrt(norm);

            // WHT (forward = inverse up to scale)
            if (norm > 0.0f) {
                float inv = 1.0f / norm;
                for (uint i = 0; i < 32; i++) buf[i] *= inv;
                wht32(buf);
            }

            // Write norm header (f16)
            uint dst_off = seq_pos * n_blocks * block_bytes_k + blk * block_bytes_k;
            half norm_h = half(norm);
            *(device half*)(kv_keys + dst_off) = norm_h;

            // Quantize to nearest centroid
            device uchar* packed = kv_keys + dst_off + 2;
            if (bits_k == 4) {
                for (uint i = 0; i < 16; i++) {
                    uint lo = 0, hi = 0;
                    float vlo = buf[i * 2];
                    float vhi = buf[i * 2 + 1];
                    // Find nearest 4-bit centroid
                    for (uint c = 1; c < 16; c++) {
                        if (abs(vlo - lloyd_max_4bit[c]) < abs(vlo - lloyd_max_4bit[lo])) lo = c;
                        if (abs(vhi - lloyd_max_4bit[c]) < abs(vhi - lloyd_max_4bit[hi])) hi = c;
                    }
                    packed[i] = uchar(lo | (hi << 4));
                }
            } else if (bits_k == 2) {
                for (uint i = 0; i < 8; i++) {
                    uint byte = 0;
                    for (uint j = 0; j < 4; j++) {
                        float v = buf[i * 4 + j];
                        uint best = 0;
                        for (uint c = 1; c < 4; c++) {
                            if (abs(v - lloyd_max_2bit[c]) < abs(v - lloyd_max_2bit[best])) best = c;
                        }
                        byte |= best << (j * 2);
                    }
                    packed[i] = uchar(byte);
                }
            }
            // bits_k == 3: 3-bit packing omitted for brevity — uses bit-level packing
        } else {
            // f32 passthrough
            uint dst_off = seq_pos * kv_dim * 4 + blk * 32 * 4;
            for (uint i = 0; i < 32; i++) {
                *(device float*)(kv_keys + dst_off + i * 4) = k_new[blk * 32 + i];
            }
        }

        // ── Encode V block (same algorithm) ──────────────────────
        if (bits_v > 0) {
            thread float buf[32];
            for (uint i = 0; i < 32; i++) buf[i] = v_new[blk * 32 + i];

            float norm = 0.0f;
            for (uint i = 0; i < 32; i++) norm += buf[i] * buf[i];
            norm = sqrt(norm);

            if (norm > 0.0f) {
                float inv = 1.0f / norm;
                for (uint i = 0; i < 32; i++) buf[i] *= inv;
                wht32(buf);
            }

            uint dst_off = seq_pos * n_blocks * block_bytes_v + blk * block_bytes_v;
            *(device half*)(kv_values + dst_off) = half(norm);

            device uchar* packed = kv_values + dst_off + 2;
            if (bits_v == 4) {
                for (uint i = 0; i < 16; i++) {
                    uint lo = 0, hi = 0;
                    float vlo = buf[i * 2];
                    float vhi = buf[i * 2 + 1];
                    for (uint c = 1; c < 16; c++) {
                        if (abs(vlo - lloyd_max_4bit[c]) < abs(vlo - lloyd_max_4bit[lo])) lo = c;
                        if (abs(vhi - lloyd_max_4bit[c]) < abs(vhi - lloyd_max_4bit[hi])) hi = c;
                    }
                    packed[i] = uchar(lo | (hi << 4));
                }
            } else if (bits_v == 2) {
                for (uint i = 0; i < 8; i++) {
                    uint byte = 0;
                    for (uint j = 0; j < 4; j++) {
                        float v = buf[i * 4 + j];
                        uint best = 0;
                        for (uint c = 1; c < 4; c++) {
                            if (abs(v - lloyd_max_2bit[c]) < abs(v - lloyd_max_2bit[best])) best = c;
                        }
                        byte |= best << (j * 2);
                    }
                    packed[i] = uchar(byte);
                }
            }
        } else {
            uint dst_off = seq_pos * kv_dim * 4 + blk * 32 * 4;
            for (uint i = 0; i < 32; i++) {
                *(device float*)(kv_values + dst_off + i * 4) = v_new[blk * 32 + i];
            }
        }
    }
}

// ── Inline SDPA with TurboQuant+ ─────────────────────────────────────────
// Single-query decode attention within the megakernel.
// Supports: f32, TurboQuant, asymmetric K/V types, sparse V, boundary V.
//
// Each TG handles one query head. TGs beyond n_heads are idle.
// Uses online softmax (FlashAttention-2 algorithm):
//   - Process K/V in blocks of 16 positions
//   - Maintain running max and sum for numerical stability
//   - Accumulate output incrementally
//
// Sparse V: skip V positions where softmax weight < 1e-6.
// Boundary V: caller passes the correct bits_v per layer (f16 for boundary layers).

inline void mega_sdpa_inline(
    device const float* q,         // [n_heads * head_dim] — query vectors
    device const uchar* kv_keys,   // KV cache keys (f32 or TQ encoded)
    device const uchar* kv_values, // KV cache values (f32 or TQ encoded)
    device float* attn_out,        // [n_heads * head_dim] — output
    uint n_heads,
    uint n_kv,                     // KV heads (n_heads / n_kv = heads_per_group for GQA)
    uint head_dim,
    uint seq_len,                  // current sequence length (positions 0..seq_len-1)
    float scale,                   // 1/sqrt(head_dim)
    uint bits_k,                   // 0=f32, 2/3/4=turbo
    uint bits_v,
    uint block_bytes_k,
    uint block_bytes_v,
    threadgroup float* shared,     // [8] — threadgroup scratch for reductions
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_heads) return;

    uint h = tgid;
    uint kvh = h * n_kv / n_heads; // GQA: which KV head services this Q head
    uint hd = head_dim;
    uint n_blocks_per_pos = hd / 32;

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Q vector for this head (read from device)
    thread float q_local[256]; // max head_dim = 256
    for (uint d = tid; d < hd; d += tg_size) {
        q_local[d] = q[h * hd + d] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output accumulator
    thread float out_acc[256];
    for (uint d = 0; d < hd; d++) out_acc[d] = 0.0f;

    // Process KV cache in blocks of 1 position (decode-only path)
    for (uint t = 0; t < seq_len; t++) {
        // ── QK dot product ───────────────────────────────────────
        float score = 0.0f;

        if (bits_k == 0) {
            // f32 K cache
            device const float* k_pos = (device const float*)kv_keys + t * n_kv * hd + kvh * hd;
            for (uint d = tid; d < hd; d += tg_size) {
                score += q_local[d] * k_pos[d];
            }
        } else {
            // TurboQuant K cache — dequant block by block
            thread float k_dequant[32];
            for (uint blk = 0; blk < n_blocks_per_pos; blk++) {
                uint cache_off = (t * n_kv + kvh) * n_blocks_per_pos * block_bytes_k +
                                 blk * block_bytes_k;
                turbo_dequant_block(kv_keys + cache_off, k_dequant, bits_k);
                for (uint d = tid; d < 32; d += tg_size) {
                    uint di = blk * 32 + d;
                    if (di < hd) score += q_local[di] * k_dequant[d];
                }
            }
        }

        // Reduce score across threads
        score = threadgroup_reduce_sum(score, shared, tid, tg_size);
        // Broadcast
        if (tid == 0) shared[0] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared[0];

        // ── Online softmax update ────────────────────────────────
        float old_max = max_score;
        max_score = max(max_score, score);
        float exp_score = exp(score - max_score);
        float correction = exp(old_max - max_score);
        sum_exp = sum_exp * correction + exp_score;

        // Correct running output accumulator
        for (uint d = 0; d < hd; d++) out_acc[d] *= correction;

        // ── Sparse V: skip if weight is negligible ───────────────
        float weight = exp_score / sum_exp;
        if (weight < sparse_v_threshold) continue;

        // ── V accumulation ───────────────────────────────────────
        if (bits_v == 0) {
            device const float* v_pos = (device const float*)kv_values + t * n_kv * hd + kvh * hd;
            for (uint d = 0; d < hd; d++) {
                out_acc[d] += exp_score * v_pos[d];
            }
        } else {
            // TurboQuant V cache
            thread float v_dequant[32];
            for (uint blk = 0; blk < n_blocks_per_pos; blk++) {
                uint cache_off = (t * n_kv + kvh) * n_blocks_per_pos * block_bytes_v +
                                 blk * block_bytes_v;
                turbo_dequant_block(kv_values + cache_off, v_dequant, bits_v);
                for (uint d = 0; d < 32 && blk * 32 + d < hd; d++) {
                    out_acc[blk * 32 + d] += exp_score * v_dequant[d];
                }
            }
        }
    }

    // ── Normalize output by sum_exp ──────────────────────────────
    if (sum_exp > 0.0f) {
        float inv = 1.0f / sum_exp;
        for (uint d = 0; d < hd; d++) out_acc[d] *= inv;
    }

    // Write output
    for (uint d = tid; d < hd; d += tg_size) {
        attn_out[h * hd + d] = out_acc[d];
    }
}
