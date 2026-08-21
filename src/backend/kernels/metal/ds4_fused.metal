// DeepSeek V4 — Fused attention projection kernel.
// Combines 6 operations into 1 GPU dispatch:
//   rmsNorm(hidden→hidden2) → q_a GEMV → rmsNorm(q_compressed) →
//   q_b GEMV → kv_a GEMV → rmsNorm(kv_proj)
// Eliminates 5 dispatch + barrier cycles per layer (215 per forward).
//
// Uses mega_common.metal grid sync primitives for cross-threadgroup coordination.
// Weight format: MLX-Q 4-bit affine (packed u32 nibbles + bf16 scales/biases).

#include <metal_stdlib>
using namespace metal;

// Forward-declare threadgroup_reduce_sum from mega_common.metal (included via msl_source concat)
// Already available: threadgroup_reduce_sum, mega_grid_sync, mega_sync_reset

// ── MLX-Q4 GEMV building block for megakernels ───────────────────────
// One threadgroup per output row. Reads packed u32 nibbles + bf16 scales/biases.
inline void mega_gemv_mlx_q4(
    device const float* x,               // [k]
    device const packed_uchar4* W,       // packed weights
    device const packed_uchar2* scales,  // bf16 per-group scales
    device const packed_uchar2* biases,  // bf16 per-group biases
    device float* y,                     // [n_out]
    uint n_out,
    uint k,
    uint gs,                             // group size (typically 64)
    threadgroup float* shared,           // [8]
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    const uint wpg = gs / 8;
    uint gpr = (k + gs - 1) / gs;
    uint w_row = tgid * gpr * wpg;
    float sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        uint sb_idx = tgid * gpr + g;
        packed_uchar2 sb = scales[sb_idx];
        float scale = as_type<float>(uint(ushort(sb[0]) | (ushort(sb[1]) << 8)) << 16);
        packed_uchar2 bb = biases[sb_idx];
        float bias  = as_type<float>(uint(ushort(bb[0]) | (ushort(bb[1]) << 8)) << 16);

        uint xo = g * gs;
        uint wg = w_row + g * wpg;
        float q_dot = 0.0f;
        float x_sum = 0.0f;

        for (uint w = 0; w < wpg; w++) {
            uint xi = xo + w * 8;
            if (xi + 8 > k) {
                packed_uchar4 bytes = W[wg + w];
                uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) | (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
                for (uint i = 0; i < k - xi; i++) {
                    float q = float((word >> (i * 4)) & 0xF);
                    q_dot += q * x[xi + i];
                    x_sum += x[xi + i];
                }
                break;
            }
            packed_uchar4 bytes = W[wg + w];
            uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) | (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
            float4 q_lo = float4(float(word & 0xF), float((word >> 4) & 0xF),
                                  float((word >> 8) & 0xF), float((word >> 12) & 0xF));
            float4 q_hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                                  float((word >> 24) & 0xF), float((word >> 28) & 0xF));
            float4 x_lo = *(device const float4*)(x + xi);
            float4 x_hi = *(device const float4*)(x + xi + 4);
            q_dot += dot(q_lo, x_lo) + dot(q_hi, x_hi);
            x_sum += (x_lo.x + x_lo.y + x_lo.z + x_lo.w) +
                     (x_hi.x + x_hi.y + x_hi.z + x_hi.w);
        }
        sum += scale * q_dot + bias * x_sum;
    }

    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── MXFP4 SafeTensors GEMV building block ─────────────────────────────
// E2M1 nibble dequant with E8M0 or E4M3 per-group scales.
inline void mega_gemv_mxfp4_st(
    device const float* x,
    device const packed_uchar4* W,
    device const uchar* scales,
    device float* y,
    uint n_out,
    uint k,
    uint gs,
    uint scale_fmt,  // 0=E4M3, 1=E8M0
    threadgroup float* shared,
    uint tgid,
    uint tid,
    uint tg_size
) {
    if (tgid >= n_out) return;
    const uint wpg = gs / 8;
    uint gpr = (k + gs - 1) / gs;
    uint w_row = tgid * gpr * wpg;
    float sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        float scale;
        uchar sv = scales[tgid * gpr + g];
        if (scale_fmt == 1) {
            scale = (sv == 0) ? 0.0f : as_type<float>(uint(sv) << 23); // E8M0
        } else {
            // FP8 E4M3 decode
            uint e = (sv >> 3) & 0xF;
            uint m = sv & 0x7;
            float mag = (e == 0) ? float(m) / 8.0f * exp2(-6.0f)
                                 : (1.0f + float(m) / 8.0f) * exp2(float(e) - 7.0f);
            scale = (sv & 0x80) ? -mag : mag;
        }

        uint xo = g * gs;
        uint wg = w_row + g * wpg;
        float gdot = 0.0f;

        for (uint w = 0; w < wpg && xo + w * 8 < k; w++) {
            uint xi = xo + w * 8;
            packed_uchar4 bytes = W[wg + w];
            uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) |
                        (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
            uint rem = min(uint(8), k - xi);
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

    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Fused DS4 Attention Projection ────────────────────────────────────
// Single dispatch: rmsNorm → q_a → rmsNorm → q_b → kv_a → rmsNorm
// All 6 stages use grid sync for cross-threadgroup coordination.
// n_tgs = max(n_embd, q_lora_rank, n_head*kv_lora_rank, kv_lora_rank) = n_head*kv_lora_rank
//
// Buffer layout:
//  0: hidden        [n_embd] input
//  1: hidden2       [n_embd] rmsNorm output / GEMV input
//  2: q_compressed  [q_lora_rank] q_a output
//  3: q_full        [n_head * kv_lora_rank] q_b output
//  4: kv_proj       [kv_lora_rank] kv_a output
//  5: attn_norm_w   [n_embd] rmsNorm weight
//  6: q_a_w         weight tensor (MLX-Q4 packed)
//  7: q_a_s         scales
//  8: q_a_b         biases
//  9: q_a_norm_w    [q_lora_rank] rmsNorm weight
// 10: q_b_w         weight tensor
// 11: q_b_s         scales
// 12: q_b_b         biases
// 13: kv_a_w        weight tensor
// 14: kv_a_s        scales
// 15: kv_a_b        biases
// 16: kv_a_norm_w   [kv_lora_rank] rmsNorm weight
// 17: scratch       [4] for rmsNorm sum-of-squares + sync counter
// 18: params        [8] packed uint params

kernel void ds4_fused_attn_proj(
    device float* hidden          [[buffer(0)]],
    device float* hidden2         [[buffer(1)]],
    device float* q_compressed    [[buffer(2)]],
    device float* q_full          [[buffer(3)]],
    device float* kv_proj         [[buffer(4)]],
    device const float* attn_norm_w [[buffer(5)]],
    device const packed_uchar4* q_a_w [[buffer(6)]],
    device const packed_uchar2* q_a_s [[buffer(7)]],
    device const packed_uchar2* q_a_b [[buffer(8)]],
    device const float* q_a_norm_w [[buffer(9)]],
    device const packed_uchar4* q_b_w [[buffer(10)]],
    device const packed_uchar2* q_b_s [[buffer(11)]],
    device const packed_uchar2* q_b_b [[buffer(12)]],
    device const packed_uchar4* kv_a_w [[buffer(13)]],
    device const packed_uchar2* kv_a_s [[buffer(14)]],
    device const packed_uchar2* kv_a_b [[buffer(15)]],
    device const float* kv_a_norm_w [[buffer(16)]],
    device float* scratch         [[buffer(17)]],   // [4]: ss[0], sync_ctr[1]
    constant uint* params         [[buffer(18)]],   // [8]: n_embd, ql, nh*kd, kd, gs_qa, gs_qb, gs_kv, n_tgs
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    const uint n_embd = params[0];
    const uint ql     = params[1];   // q_lora_rank
    const uint nhkd   = params[2];   // n_head * kv_lora_rank
    const uint kd     = params[3];   // kv_lora_rank
    const uint gs_qa  = params[4];   // group size for q_a
    const uint gs_qb  = params[5];   // group size for q_b
    const uint gs_kv  = params[6];   // group size for kv_a
    const uint n_tgs  = params[7];   // total threadgroups
    const float eps   = 1e-6f;

    threadgroup float shared[8];
    device atomic_uint* sync_ctr = (device atomic_uint*)(scratch + 2);
    device float* ss = scratch;

    // Stage 1: rmsNorm(hidden → hidden2) using attn_norm weights
    if (tgid == 0 && tid == 0) { ss[0] = 0.0f; ss[1] = 0.0f; }
    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_rms_norm(hidden, attn_norm_w, hidden2, ss, sync_ctr, n_embd, n_tgs, eps, shared, tgid, tid, tg_size);
    mega_sync_reset(sync_ctr, tgid, tid);

    // Stage 2: q_a GEMV (hidden2 → q_compressed) [ql × n_embd]
    mega_gemv_mlx_q4(hidden2, q_a_w, q_a_s, q_a_b, q_compressed, ql, n_embd, gs_qa, shared, tgid, tid, tg_size);
    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_sync_reset(sync_ctr, tgid, tid);

    // Stage 3: rmsNorm(q_compressed) in-place
    if (tgid == 0 && tid == 0) ss[0] = 0.0f;
    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_rms_norm(q_compressed, q_a_norm_w, q_compressed, ss, sync_ctr, ql, n_tgs, eps, shared, tgid, tid, tg_size);
    mega_sync_reset(sync_ctr, tgid, tid);

    // Stage 4: q_b GEMV (q_compressed → q_full) [nhkd × ql]
    mega_gemv_mlx_q4(q_compressed, q_b_w, q_b_s, q_b_b, q_full, nhkd, ql, gs_qb, shared, tgid, tid, tg_size);
    // kv_a can overlap with q_b completion — no sync needed between them
    // (different output buffers, same input hidden2)

    // Stage 5: kv_a GEMV (hidden2 → kv_proj) [kd × n_embd]
    // Only first kd threadgroups do this; rest idle.
    mega_gemv_mlx_q4(hidden2, kv_a_w, kv_a_s, kv_a_b, kv_proj, kd, n_embd, gs_kv, shared, tgid, tid, tg_size);
    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_sync_reset(sync_ctr, tgid, tid);

    // Stage 6: rmsNorm(kv_proj) in-place
    if (tgid == 0 && tid == 0) ss[0] = 0.0f;
    mega_grid_sync(sync_ctr, n_tgs, tgid, tid);
    mega_rms_norm(kv_proj, kv_a_norm_w, kv_proj, ss, sync_ctr, kd, n_tgs, eps, shared, tgid, tid, tg_size);
}
