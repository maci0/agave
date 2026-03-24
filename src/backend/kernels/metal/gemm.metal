// GEMM — Matrix-Matrix Multiply with dequantization for batched prefill
//
// Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T
//
// Dispatch: one threadgroup per output row (dispatchThreadgroups(n_out,1,1)),
// 256 threads per threadgroup. Each threadgroup computes one output row
// for ALL n_tok tokens — weight row loaded once, reused across tokens.
// This gives n_tok× bandwidth savings vs loop-of-GEMV.

// ── F32 GEMM ─────────────────────────────────────────────────

kernel void gemm_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y       [[buffer(2)]],
    constant uint& n_out  [[buffer(3)]],
    constant uint& n_in   [[buffer(4)]],
    constant uint& n_tok  [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n_out) return;

    uint row_off = tgid * n_in;

    for (uint t = 0; t < n_tok; t++) {
        float sum = 0.0f;
        uint x_off = t * n_in;
        uint k4 = n_in & ~3u;
        for (uint j = tid * 4; j < k4; j += tg_size * 4) {
            float4 wv = float4(W[row_off+j], W[row_off+j+1], W[row_off+j+2], W[row_off+j+3]);
            float4 xv = float4(X[x_off+j], X[x_off+j+1], X[x_off+j+2], X[x_off+j+3]);
            sum += dot(wv, xv);
        }
        for (uint j = k4 + tid; j < n_in; j += tg_size) {
            sum += W[row_off + j] * X[x_off + j];
        }

        threadgroup float shared[8];
        sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
        if (tid == 0) Y[t * n_out + tgid] = sum;
        if (t + 1 < n_tok) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Q8_0 GEMM ────────────────────────────────────────────────
// One threadgroup per output row. Weight blocks dequantized once,
// reused across TILE_T=8 tokens. Uses q8_0_block_dot from GEMV.

constant uint gemm_q8_tile_t = 8;

kernel void gemm_q8_0(
    device const float* X      [[buffer(0)]],
    device const block_q8_0* W [[buffer(1)]],
    device float* Y            [[buffer(2)]],
    constant uint& n_out       [[buffer(3)]],
    constant uint& n_in        [[buffer(4)]],
    constant uint& n_tok       [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n_out) return;
    uint nb = n_in / 32;

    for (uint t_base = 0; t_base < n_tok; t_base += gemm_q8_tile_t) {
        uint t_end = min(t_base + gemm_q8_tile_t, n_tok);
        uint nt = t_end - t_base;

        float sums[gemm_q8_tile_t] = {0};

        for (uint b = tid; b < nb; b += tg_size) {
            // Reuse existing q8_0_block_dot — compiler hoists x loads across calls
            for (uint ti = 0; ti < nt; ti++) {
                sums[ti] += q8_0_block_dot(W[tgid * nb + b],
                                           X + (t_base + ti) * n_in + b * 32);
            }
        }

        threadgroup float shared[8];
        for (uint ti = 0; ti < nt; ti++) {
            float s = threadgroup_reduce_sum(sums[ti], shared, tid, tg_size);
            if (tid == 0) Y[(t_base + ti) * n_out + tgid] = s;
            if (ti + 1 < nt || t_base + gemm_q8_tile_t < n_tok)
                threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


// ── Q4_0 GEMM ────────────────────────────────────────────────
// 32 values per block, 18 bytes (f16 scale + 16 bytes of packed 4-bit values).

kernel void gemm_q4_0(
    device const float* X       [[buffer(0)]],
    device const uchar* W       [[buffer(1)]],
    device float* Y             [[buffer(2)]],
    constant uint& n_out        [[buffer(3)]],
    constant uint& n_in         [[buffer(4)]],
    constant uint& n_tok        [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n_out) return;
    uint nb = n_in / 32;

    for (uint t = 0; t < n_tok; t++) {
        float sum = 0.0f;
        for (uint b = tid; b < nb; b += tg_size) {
            device const uchar* blk_ptr = W + (tgid * nb + b) * 18;
            half d = *(device const half*)blk_ptr;
            float scale = float(d);
            device const uchar* qs = blk_ptr + 2;
            device const float* x_block = X + t * n_in + b * 32;

            float block_sum = 0.0f;
            for (uint j = 0; j < 16; j++) {
                uchar byte = qs[j];
                float v0 = float(int(byte & 0xF) - 8);
                float v1 = float(int(byte >> 4) - 8);
                block_sum += x_block[j] * v0 + x_block[j + 16] * v1;
            }
            sum += block_sum * scale;
        }

        threadgroup float shared[8];
        sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
        if (tid == 0) Y[t * n_out + tgid] = sum;
        if (t + 1 < n_tok) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
