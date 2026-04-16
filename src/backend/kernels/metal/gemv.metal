// GEMV — Matrix-Vector Multiply with dequantization
//
// Dispatch model: one threadgroup per output row (dispatchThreadgroups(n,1,1)),
// 256 threads per threadgroup. Each thread strides over the k-dimension, then
// all threads cooperatively reduce using threadgroup_reduce_sum (common.metal).

// ── F32 GEMV ─────────────────────────────────────────────────

kernel void gemv_f32(
    device const float* x [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    float sum = 0.0f;
    uint row_off = tgid * k;
    uint k4 = k & ~3u;
    for (uint j = tid * 4; j < k4; j += tg_size * 4) {
        float4 wv = float4(W[row_off+j], W[row_off+j+1], W[row_off+j+2], W[row_off+j+3]);
        float4 xv = float4(x[j], x[j+1], x[j+2], x[j+3]);
        sum += dot(wv, xv);
    }
    for (uint j = k4 + tid; j < k; j += tg_size) {
        sum += W[row_off + j] * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Q8_0 GEMV ────────────────────────────────────────────────
// 32 values per block, 34 bytes (f16 scale + 32 int8 values).
// NR=4: each threadgroup processes 4 output rows, sharing x vector cache lines.
// char4 vectorized weight loads (4 bytes at once vs scalar byte reads).

struct block_q8_0 {
    half d;
    char qs[32];
};

constant uint q8_0_nr = 4;

// Dot product of one Q8_0 block with x vector at given offset.
// Processes 8 elements per iteration (4 iterations per block instead of 8).
// Inline: x loads are hoisted by the compiler when called multiple times.
inline float q8_0_block_dot(device const block_q8_0& blk,
                            device const float* x_block) {
    float block_sum = 0.0f;
    for (uint j = 0; j < 32; j += 8) {
        float4 xv0 = *(device const float4*)(x_block + j);
        float4 xv1 = *(device const float4*)(x_block + j + 4);
        float4 wv0 = float4(float(blk.qs[j]),   float(blk.qs[j+1]),
                            float(blk.qs[j+2]), float(blk.qs[j+3]));
        float4 wv1 = float4(float(blk.qs[j+4]), float(blk.qs[j+5]),
                            float(blk.qs[j+6]), float(blk.qs[j+7]));
        block_sum += dot(xv0, wv0) + dot(xv1, wv1);
    }
    return block_sum * float(blk.d);
}

kernel void gemv_q8_0(
    device const float* x      [[buffer(0)]],
    device const block_q8_0* W [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    uint row_base = tgid * q8_0_nr;
    if (row_base >= n) return;
    uint nb = k / 32;
    uint nr_active = min(q8_0_nr, n - row_base);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const float* x_block = x + b * 32;

        // Each row reuses the same x_block; compiler hoists the loads
        sum0 += q8_0_block_dot(W[row_base * nb + b], x_block);
        if (nr_active > 1)
            sum1 += q8_0_block_dot(W[(row_base + 1) * nb + b], x_block);
        if (nr_active > 2)
            sum2 += q8_0_block_dot(W[(row_base + 2) * nb + b], x_block);
        if (nr_active > 3)
            sum3 += q8_0_block_dot(W[(row_base + 3) * nb + b], x_block);
    }

    // Reduce each row sum independently
    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
    if (nr_active > 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum2 = threadgroup_reduce_sum(sum2, shared, tid, tg_size);
        if (tid == 0) y[row_base + 2] = sum2;
    }
    if (nr_active > 3) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum3 = threadgroup_reduce_sum(sum3, shared, tid, tg_size);
        if (tid == 0) y[row_base + 3] = sum3;
    }
}

// ── Q4_0 GEMV ────────────────────────────────────────────────
// 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes).
// Block layout: each byte packs 2 values — low nibble for elements 0-15, high nibble for elements 16-31.

struct block_q4_0 {
    half d;
    uchar qs[16];
};

// NR=4: each threadgroup processes 4 output rows, sharing x vector cache lines.
// 256 threads per TG, each thread works on all 4 rows.
// Dispatch: ceil(n/4) threadgroups.
constant uint q4_0_nr = 4;

// Dequantize one Q4_0 block (16 bytes → 32 floats) and dot with x vector.
inline float q4_0_block_dot(device const block_q4_0& blk,
                            float4 x0, float4 x1, float4 x2, float4 x3,
                            float4 x4, float4 x5, float4 x6, float4 x7) {
    // Load 16 bytes as 4×uchar4 (1-byte aligned)
    uchar4 q0 = *(device const uchar4*)(blk.qs);
    uchar4 q1 = *(device const uchar4*)(blk.qs + 4);
    uchar4 q2 = *(device const uchar4*)(blk.qs + 8);
    uchar4 q3 = *(device const uchar4*)(blk.qs + 12);

    // Low nibbles (elements 0-15) → dot with x0..x3
    float s = dot(float4(int4(q0 & 0xF) - 8), x0)
            + dot(float4(int4(q1 & 0xF) - 8), x1)
            + dot(float4(int4(q2 & 0xF) - 8), x2)
            + dot(float4(int4(q3 & 0xF) - 8), x3);
    // High nibbles (elements 16-31) → dot with x4..x7
    s += dot(float4(int4(q0 >> 4) - 8), x4)
       + dot(float4(int4(q1 >> 4) - 8), x5)
       + dot(float4(int4(q2 >> 4) - 8), x6)
       + dot(float4(int4(q3 >> 4) - 8), x7);

    return s * float(blk.d);
}

kernel void gemv_q4_0(
    device const float* x      [[buffer(0)]],
    device const block_q4_0* W [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    uint row_base = tgid * q4_0_nr;
    if (row_base >= n) return;
    uint nb = k / 32;
    uint nr_active = min(q4_0_nr, n - row_base);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * 32;

        // Load x once, shared across all 4 rows
        float4 x0 = *(device const float4*)(x + bk);
        float4 x1 = *(device const float4*)(x + bk + 4);
        float4 x2 = *(device const float4*)(x + bk + 8);
        float4 x3 = *(device const float4*)(x + bk + 12);
        float4 x4 = *(device const float4*)(x + bk + 16);
        float4 x5 = *(device const float4*)(x + bk + 20);
        float4 x6 = *(device const float4*)(x + bk + 24);
        float4 x7 = *(device const float4*)(x + bk + 28);

        // Fully unrolled: each row is independent, compiler can schedule freely
        sum0 += q4_0_block_dot(W[row_base * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
        if (nr_active > 1)
            sum1 += q4_0_block_dot(W[(row_base + 1) * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
        if (nr_active > 2)
            sum2 += q4_0_block_dot(W[(row_base + 2) * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
        if (nr_active > 3)
            sum3 += q4_0_block_dot(W[(row_base + 3) * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
    }

    // Reduce each row sum independently
    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
    if (nr_active > 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum2 = threadgroup_reduce_sum(sum2, shared, tid, tg_size);
        if (tid == 0) y[row_base + 2] = sum2;
    }
    if (nr_active > 3) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum3 = threadgroup_reduce_sum(sum3, shared, tid, tg_size);
        if (tid == 0) y[row_base + 3] = sum3;
    }
}

// ── Q4_1 GEMV ────────────────────────────────────────────────
// 32 values per block, 20 bytes (f16 scale + f16 min + 16 nibble-packed bytes).
// Value = nibble * d + min (no subtract-8, has zero-point).

struct block_q4_1 {
    half d;
    half m; // min
    uchar qs[16];
};

kernel void gemv_q4_1(
    device const float* x      [[buffer(0)]],
    device const block_q4_1* W [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;
    uint nb = k / 32;
    float sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        device const block_q4_1& blk = W[tgid * nb + b];
        float d = float(blk.d);
        float m = float(blk.m);
        float block_sum = 0.0f;
        float x_sum = 0.0f; // sum of x for min offset
        uint bk = b * 32;
        for (uint j = 0; j < 16; j += 4) {
            float4 lo = float4(float(blk.qs[j  ] & 0xF),
                               float(blk.qs[j+1] & 0xF),
                               float(blk.qs[j+2] & 0xF),
                               float(blk.qs[j+3] & 0xF));
            float4 hi = float4(float(blk.qs[j  ] >> 4),
                               float(blk.qs[j+1] >> 4),
                               float(blk.qs[j+2] >> 4),
                               float(blk.qs[j+3] >> 4));
            float4 xlo = float4(x[bk+j], x[bk+j+1], x[bk+j+2], x[bk+j+3]);
            float4 xhi = float4(x[bk+j+16], x[bk+j+17], x[bk+j+18], x[bk+j+19]);
            block_sum += dot(lo, xlo) + dot(hi, xhi);
            x_sum += xlo.x + xlo.y + xlo.z + xlo.w + xhi.x + xhi.y + xhi.z + xhi.w;
        }
        sum += block_sum * d + x_sum * m;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── BF16 GEMV ────────────────────────────────────────────────
// Weights stored as raw ushort (bfloat16 bit pattern).
// Read as uchar pairs to handle potentially unaligned SafeTensors data.

inline float read_bf16(device const uchar* p, uint idx) {
    uint byte_off = idx * 2;
    uint lo = p[byte_off];
    uint hi = p[byte_off + 1];
    return as_type<float>((hi << 24) | (lo << 16));
}

kernel void gemv_bf16(
    device const float* x  [[buffer(0)]],
    device const uchar* W  [[buffer(1)]],
    device float* y        [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    constant uint& k       [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    float sum = 0.0f;
    uint row_off = tgid * k;
    uint k4 = k & ~3u;
    for (uint j = tid * 4; j < k4; j += tg_size * 4) {
        float4 wv = float4(read_bf16(W, row_off+j),
                            read_bf16(W, row_off+j+1),
                            read_bf16(W, row_off+j+2),
                            read_bf16(W, row_off+j+3));
        float4 xv = float4(x[j], x[j+1], x[j+2], x[j+3]);
        sum += dot(wv, xv);
    }
    for (uint j = k4 + tid; j < k; j += tg_size) {
        sum += read_bf16(W, row_off + j) * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── F16 GEMV ─────────────────────────────────────────────────
// Read as uchar pairs for unaligned SafeTensors data.

inline float read_f16(device const uchar* p, uint idx) {
    uint byte_off = idx * 2;
    uint lo = p[byte_off];
    uint hi = p[byte_off + 1];
    return float(as_type<half>(ushort((hi << 8) | lo)));
}

kernel void gemv_f16(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    float sum = 0.0f;
    uint row_off = tgid * k;
    uint k4 = k & ~3u;
    for (uint j = tid * 4; j < k4; j += tg_size * 4) {
        float4 wv = float4(read_f16(W, row_off+j), read_f16(W, row_off+j+1),
                            read_f16(W, row_off+j+2), read_f16(W, row_off+j+3));
        float4 xv = float4(x[j], x[j+1], x[j+2], x[j+3]);
        sum += dot(wv, xv);
    }
    for (uint j = k4 + tid; j < k; j += tg_size) {
        sum += read_f16(W, row_off + j) * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── NVFP4 SafeTensors GEMV ───────────────────────────────────
// Separated weight nibbles + FP8 E4M3 scales.
// Group size = 16 elements per FP8 E4M3 scale (8 packed bytes per group).

// E2M1 FP4 → float lookup (OCP Microscaling Spec).
constant float e2m1_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP8 E4M3 → float: precomputed 256-entry LUT eliminates all runtime branches.
constant float fp8e4m3_lut[256] = {
    0.0f, 1.0f/512, 2.0f/512, 3.0f/512, 4.0f/512, 5.0f/512, 6.0f/512, 7.0f/512,
    1.0f/64, 1.125f/64, 1.25f/64, 1.375f/64, 1.5f/64, 1.625f/64, 1.75f/64, 1.875f/64,
    1.0f/32, 1.125f/32, 1.25f/32, 1.375f/32, 1.5f/32, 1.625f/32, 1.75f/32, 1.875f/32,
    1.0f/16, 1.125f/16, 1.25f/16, 1.375f/16, 1.5f/16, 1.625f/16, 1.75f/16, 1.875f/16,
    1.0f/8, 1.125f/8, 1.25f/8, 1.375f/8, 1.5f/8, 1.625f/8, 1.75f/8, 1.875f/8,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, NAN,
    -0.0f, -1.0f/512, -2.0f/512, -3.0f/512, -4.0f/512, -5.0f/512, -6.0f/512, -7.0f/512,
    -1.0f/64, -1.125f/64, -1.25f/64, -1.375f/64, -1.5f/64, -1.625f/64, -1.75f/64, -1.875f/64,
    -1.0f/32, -1.125f/32, -1.25f/32, -1.375f/32, -1.5f/32, -1.625f/32, -1.75f/32, -1.875f/32,
    -1.0f/16, -1.125f/16, -1.25f/16, -1.375f/16, -1.5f/16, -1.625f/16, -1.75f/16, -1.875f/16,
    -1.0f/8, -1.125f/8, -1.25f/8, -1.375f/8, -1.5f/8, -1.625f/8, -1.75f/8, -1.875f/8,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, NAN,
};

inline float fp8e4m3_to_f32(uchar val) {
    return fp8e4m3_lut[val];
}

kernel void gemv_nvfp4_st(
    device const float* x     [[buffer(0)]],
    device const uchar* W     [[buffer(1)]],
    device const uchar* S     [[buffer(2)]],
    device float* y            [[buffer(3)]],
    constant uint& n           [[buffer(4)]],
    constant uint& k           [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    uint bytes_per_row = k / 2;
    uint groups_per_row = k / 16;
    float sum = 0.0f;

    for (uint g = tid; g < groups_per_row; g += tg_size) {
        float scale = fp8e4m3_to_f32(S[tgid * groups_per_row + g]);
        uint w_base = tgid * bytes_per_row + g * 8;
        uint x_base = g * 16;
        for (uint j = 0; j < 8; j += 4) {
            uchar b0 = W[w_base + j];
            uchar b1 = W[w_base + j + 1];
            uchar b2 = W[w_base + j + 2];
            uchar b3 = W[w_base + j + 3];
            float4 w_lo = float4(e2m1_lut[b0 & 0xF], e2m1_lut[b1 & 0xF],
                                  e2m1_lut[b2 & 0xF], e2m1_lut[b3 & 0xF]);
            float4 w_hi = float4(e2m1_lut[b0 >> 4], e2m1_lut[b1 >> 4],
                                  e2m1_lut[b2 >> 4], e2m1_lut[b3 >> 4]);
            uint xi = x_base + 2 * j;
            float4 x_lo = float4(x[xi], x[xi + 2], x[xi + 4], x[xi + 6]);
            float4 x_hi = float4(x[xi + 1], x[xi + 3], x[xi + 5], x[xi + 7]);
            sum += (dot(w_lo, x_lo) + dot(w_hi, x_hi)) * scale;
        }
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Q4_K GEMV ────────────────────────────────────────────────
// 256 values per super-block, 144 bytes.
// Layout: { half d; half dmin; uchar scales[12]; uchar qs[128]; }

inline void getScaleMinK4(uint j, device const uchar* q, thread uint& sc, thread uint& m) {
    if (j < 4) {
        sc = q[j] & 63u;
        m  = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j] >> 6) << 4);
    }
}

// NR=2: each threadgroup processes 2 output rows, sharing x vector loads.
// For Q4_K, NR=2 is optimal because each superblock has more work (144 bytes,
// 4 sub-groups) than Q8_0 (34 bytes), so 2 rows saturate the ALU better than 4.
constant uint q4_k_nr = 2;

// Inline: compute one superblock's dot product for a single row.
// x vector is shared across rows (compiler hoists the loads).
inline float q4_k_block_dot(device const uchar* bp, device const float* x, uint k, uint bk) {
    float d    = float(as_type<half>(ushort(bp[0] | (uint(bp[1]) << 8))));
    float dmin = float(as_type<half>(ushort(bp[2] | (uint(bp[3]) << 8))));
    device const uchar* scales = bp + 4;
    device const uchar* qs = bp + 16;
    float sum = 0.0f;

    for (uint g = 0; g < 4; g++) {
        uint sc_lo, m_lo, sc_hi, m_hi;
        getScaleMinK4(g * 2, scales, sc_lo, m_lo);
        getScaleMinK4(g * 2 + 1, scales, sc_hi, m_hi);
        float d_lo = d * float(sc_lo);
        float dm_lo = dmin * float(m_lo);
        float d_hi = d * float(sc_hi);
        float dm_hi = dmin * float(m_hi);
        uint ql_off = g * 32;
        uint gi_lo = bk + g * 64;
        uint gi_hi = gi_lo + 32;

        if (gi_lo + 63 < k) {
            float q_dot_lo = 0.0f, x_sum_lo = 0.0f;
            for (uint l = 0; l < 32; l += 4) {
                float4 xv = *(device const float4*)(x + gi_lo + l);
                float4 qv = float4(qs[ql_off + l] & 0xF, qs[ql_off + l + 1] & 0xF,
                                   qs[ql_off + l + 2] & 0xF, qs[ql_off + l + 3] & 0xF);
                q_dot_lo += dot(xv, qv);
                x_sum_lo += xv.x + xv.y + xv.z + xv.w;
            }
            sum += d_lo * q_dot_lo - dm_lo * x_sum_lo;

            float q_dot_hi = 0.0f, x_sum_hi = 0.0f;
            for (uint l = 0; l < 32; l += 4) {
                float4 xv = *(device const float4*)(x + gi_hi + l);
                float4 qv = float4(qs[ql_off + l] >> 4, qs[ql_off + l + 1] >> 4,
                                   qs[ql_off + l + 2] >> 4, qs[ql_off + l + 3] >> 4);
                q_dot_hi += dot(xv, qv);
                x_sum_hi += xv.x + xv.y + xv.z + xv.w;
            }
            sum += d_hi * q_dot_hi - dm_hi * x_sum_hi;
        } else {
            for (uint l = 0; l < 32; l++) {
                uint gi = gi_lo + l;
                if (gi >= k) break;
                sum += x[gi] * (d_lo * float(qs[ql_off + l] & 0xF) - dm_lo);
            }
            for (uint l = 0; l < 32; l++) {
                uint gi = gi_hi + l;
                if (gi >= k) break;
                sum += x[gi] * (d_hi * float(qs[ql_off + l] >> 4) - dm_hi);
            }
        }
    }
    return sum;
}

kernel void gemv_q4_k(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    const uint bpb = 144;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    uint row_base = tgid * q4_k_nr;
    if (row_base >= n) return;
    uint nr_active = min(q4_k_nr, n - row_base);

    float sum0 = 0.0f, sum1 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum0 += q4_k_block_dot(W + (row_base * nb + b) * bpb, x, k, bk);
        if (nr_active > 1)
            sum1 += q4_k_block_dot(W + ((row_base + 1) * nb + b) * bpb, x, k, bk);
    }

    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}

// ── Q2_K GEMV ────────────────────────────────────────────────
// 256 values per super-block, 84 bytes.
// Layout: { uchar scales[16]; uchar qs[64]; half d; half dmin; }
// Dequant: 2-bit values packed 4 per byte, per-sub-block (16 values)
// 4-bit scale (low nibble) and min (high nibble).

kernel void gemv_q2_k(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint bpb = 84;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const uchar* bp = W + (tgid * nb + b) * bpb;
        device const uchar* scales = bp;
        device const uchar* qs = bp + 16;
        float d    = float(as_type<half>(ushort(bp[80] | (uint(bp[81]) << 8))));
        float dmin = float(as_type<half>(ushort(bp[82] | (uint(bp[83]) << 8))));
        uint bk = b * bs;

        for (uint sb = 0; sb < 16; sb++) {
            float sc = float(scales[sb] & 0x0F);
            float m  = float(scales[sb] >> 4);
            float d_sc = d * sc;
            float dm_m = dmin * m;
            uint gi_base = bk + sb * 16;

            for (uint l = 0; l < 16; l++) {
                uint gi = gi_base + l;
                if (gi >= k) break;
                uint qi = sb * 16 + l;
                uint byte_idx = qi / 4;
                uint shift = (qi % 4) * 2;
                float q = float((qs[byte_idx] >> shift) & 0x03);
                sum += x[gi] * (d_sc * q - dm_m);
            }
        }
    }

    threadgroup float tg_shared[8];
    sum = threadgroup_reduce_sum(sum, tg_shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Q5_0 GEMV ────────────────────────────────────────────────
// 32 values per block, 22 bytes (f16 scale + u32 qh + 16 nibble-packed bytes).
// 5-bit: 4 bits from qs nibbles + 1 high bit from qh bitmask, subtract 16.

kernel void gemv_q5_0(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint bpb = 22;
    const uint qk = 32;
    uint nb = k / qk;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const uchar* bp = W + (tgid * nb + b) * bpb;
        float d = float(as_type<half>(ushort(bp[0] | (uint(bp[1]) << 8))));
        uint qh = uint(bp[2]) | (uint(bp[3]) << 8) | (uint(bp[4]) << 16) | (uint(bp[5]) << 24);
        device const uchar* qs = bp + 6;
        uint bk = b * qk;
        float block_sum = 0.0f;

        for (uint j = 0; j < 16; j++) {
            uint lo_nib = qs[j] & 0xF;
            uint hi_nib = qs[j] >> 4;
            uint hb0 = (qh >> j) & 1;
            uint hb1 = (qh >> (j + 16)) & 1;
            int v0 = int(lo_nib | (hb0 << 4)) - 16;
            int v1 = int(hi_nib | (hb1 << 4)) - 16;
            block_sum += x[bk + j] * float(v0) + x[bk + j + 16] * float(v1);
        }
        sum += block_sum * d;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Q3_K GEMV ────────────────────────────────────────────────
// 256 values per super-block, 110 bytes.
// Layout: { uchar hmask[32]; uchar qs[64]; uchar scales_raw[12]; half d; }
// Dequant: q3 = ((qs[qi/4] >> (qi%4)*2) & 3) | ((hmask[qi%32] >> (qi/32)) & 1) << 2) - 4
// Per-sub-block (16 values) scale: decoded from scales_raw nibbles, biased by -8.

kernel void gemv_q3_k(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint bpb = 110;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const uchar* bp = W + (tgid * nb + b) * bpb;
        device const uchar* hmask = bp;
        device const uchar* qs = bp + 32;
        device const uchar* raw_scales = bp + 96;
        float d = float(as_type<half>(ushort(bp[108] | (uint(bp[109]) << 8))));
        uint bk = b * bs;

        // Decode 16 sub-block scales from 12 packed nibble bytes
        int scales[16];
        for (uint j = 0; j < 8; j++) {
            scales[j]     = int(raw_scales[j] & 0xF) - 8;
            scales[8 + j] = int(raw_scales[j] >> 4) - 8;
        }

        for (uint l = 0; l < 256; l++) {
            uint gi = bk + l;
            if (gi >= k) break;
            uint byte_idx = l / 4;
            uint shift = (l % 4) * 2;
            uint q_lo = (qs[byte_idx] >> shift) & 0x03;
            uint hm_byte = l % 32;
            uint hm_bit = l / 32;
            uint q_hi = (hmask[hm_byte] >> hm_bit) & 1;
            int q3 = int(q_lo | (q_hi << 2)) - 4;
            uint sb = l / 16;
            sum += x[gi] * d * float(scales[sb]) * float(q3);
        }
    }

    threadgroup float tg_shared[8];
    sum = threadgroup_reduce_sum(sum, tg_shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Q6_K GEMV ────────────────────────────────────────────────
// 256 values per super-block, 210 bytes.
// Layout: { uchar ql[128]; uchar qh[64]; char scales[16]; half d; }

constant uint q6_k_nr = 2;

inline float q6_k_block_dot(device const uchar* bp, device const float* x, uint k, uint bk) {
    float d = float(as_type<half>(ushort(bp[208] | (uint(bp[209]) << 8))));
    float sum = 0.0f;

    for (uint chunk = 0; chunk < 2; chunk++) {
        device const uchar* ql = bp + chunk * 64;
        device const uchar* qh = bp + 128 + chunk * 32;
        device const char* sc = (device const char*)(bp + 192 + chunk * 8);
        uint base = bk + chunk * 128;

        if (base + 127 < k) {
            float ds0_a = d * float(sc[0]), ds1_a = d * float(sc[2]);
            float ds2_a = d * float(sc[4]), ds3_a = d * float(sc[6]);
            float ds0_b = d * float(sc[1]), ds1_b = d * float(sc[3]);
            float ds2_b = d * float(sc[5]), ds3_b = d * float(sc[7]);

            for (uint l = 0; l < 16; l++) {
                float q1 = float(int((ql[l] & 0xF) | ((qh[l] & 3u) << 4)) - 32);
                float q2 = float(int((ql[l+32] & 0xF) | (((qh[l]>>2)&3u)<<4)) - 32);
                float q3 = float(int((ql[l] >> 4) | (((qh[l]>>4)&3u)<<4)) - 32);
                float q4 = float(int((ql[l+32] >> 4) | (((qh[l]>>6)&3u)<<4)) - 32);
                sum += x[base+l]*ds0_a*q1 + x[base+l+32]*ds1_a*q2
                     + x[base+l+64]*ds2_a*q3 + x[base+l+96]*ds3_a*q4;
                uint l2 = l + 16;
                float q1b = float(int((ql[l2] & 0xF) | ((qh[l2] & 3u) << 4)) - 32);
                float q2b = float(int((ql[l2+32] & 0xF) | (((qh[l2]>>2)&3u)<<4)) - 32);
                float q3b = float(int((ql[l2] >> 4) | (((qh[l2]>>4)&3u)<<4)) - 32);
                float q4b = float(int((ql[l2+32] >> 4) | (((qh[l2]>>6)&3u)<<4)) - 32);
                sum += x[base+l2]*ds0_b*q1b + x[base+l2+32]*ds1_b*q2b
                     + x[base+l2+64]*ds2_b*q3b + x[base+l2+96]*ds3_b*q4b;
            }
        } else {
            for (uint l = 0; l < 32; l++) {
                uint is = l / 16;
                int q1 = int((ql[l] & 0xF) | ((qh[l] & 3u) << 4)) - 32;
                int q2 = int((ql[l+32] & 0xF) | (((qh[l]>>2)&3u)<<4)) - 32;
                int q3 = int((ql[l] >> 4) | (((qh[l]>>4)&3u)<<4)) - 32;
                int q4 = int((ql[l+32] >> 4) | (((qh[l]>>6)&3u)<<4)) - 32;
                if (base+l < k) sum += x[base+l] * d * float(sc[is]) * float(q1);
                if (base+l+32 < k) sum += x[base+l+32] * d * float(sc[is+2]) * float(q2);
                if (base+l+64 < k) sum += x[base+l+64] * d * float(sc[is+4]) * float(q3);
                if (base+l+96 < k) sum += x[base+l+96] * d * float(sc[is+6]) * float(q4);
            }
        }
    }
    return sum;
}

kernel void gemv_q6_k(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    const uint bpb = 210;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    uint row_base = tgid * q6_k_nr;
    if (row_base >= n) return;
    uint nr_active = min(q6_k_nr, n - row_base);

    float sum0 = 0.0f, sum1 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum0 += q6_k_block_dot(W + (row_base * nb + b) * bpb, x, k, bk);
        if (nr_active > 1)
            sum1 += q6_k_block_dot(W + ((row_base + 1) * nb + b) * bpb, x, k, bk);
    }

    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}

// ── Q5_K GEMV ────────────────────────────────────────────────
// 256 values per super-block, 176 bytes.
// Layout: { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }

constant uint q5_k_nr = 2;

inline float q5_k_block_dot(device const uchar* bp, device const float* x, uint k, uint bk) {
    float d    = float(as_type<half>(ushort(bp[0] | (uint(bp[1]) << 8))));
    float dmin = float(as_type<half>(ushort(bp[2] | (uint(bp[3]) << 8))));
    device const uchar* scales = bp + 4;
    device const uchar* qh = bp + 16;
    device const uchar* qs = bp + 48;
    float sum = 0.0f;

    uint is = 0, ql_off = 0;
    uchar umask1 = 1, umask2 = 2;
    for (uint j = 0; j < 256; j += 64) {
        uint sc1, m1, sc2, m2;
        getScaleMinK4(is, scales, sc1, m1);
        getScaleMinK4(is + 1, scales, sc2, m2);
        float d1 = d * float(sc1), dm1 = dmin * float(m1);
        float d2 = d * float(sc2), dm2 = dmin * float(m2);

        if (bk + j + 63 < k) {
            for (uint l = 0; l < 32; l++) {
                uint qv = (qs[ql_off + l] & 0xF) + ((qh[l] & umask1) != 0 ? 16u : 0u);
                sum += x[bk + j + l] * (float(qv) * d1 - dm1);
            }
            for (uint l = 0; l < 32; l++) {
                uint qv = (qs[ql_off + l] >> 4) + ((qh[l] & umask2) != 0 ? 16u : 0u);
                sum += x[bk + j + 32 + l] * (float(qv) * d2 - dm2);
            }
        } else {
            for (uint l = 0; l < 32; l++) {
                uint gi = bk + j + l;
                if (gi >= k) break;
                uint qv = (qs[ql_off + l] & 0xF) + ((qh[l] & umask1) != 0 ? 16u : 0u);
                sum += x[gi] * (float(qv) * d1 - dm1);
            }
            for (uint l = 0; l < 32; l++) {
                uint gi = bk + j + 32 + l;
                if (gi >= k) break;
                uint qv = (qs[ql_off + l] >> 4) + ((qh[l] & umask2) != 0 ? 16u : 0u);
                sum += x[gi] * (float(qv) * d2 - dm2);
            }
        }
        ql_off += 32; is += 2; umask1 <<= 2; umask2 <<= 2;
    }
    return sum;
}

kernel void gemv_q5_k(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    const uint bpb = 176;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    uint row_base = tgid * q5_k_nr;
    if (row_base >= n) return;
    uint nr_active = min(q5_k_nr, n - row_base);

    float sum0 = 0.0f, sum1 = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        sum0 += q5_k_block_dot(W + (row_base * nb + b) * bpb, x, k, bk);
        if (nr_active > 1)
            sum1 += q5_k_block_dot(W + ((row_base + 1) * nb + b) * bpb, x, k, bk);
    }

    threadgroup float shared[8];
    sum0 = threadgroup_reduce_sum(sum0, shared, tid, tg_size);
    if (tid == 0) y[row_base] = sum0;

    if (nr_active > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum1 = threadgroup_reduce_sum(sum1, shared, tid, tg_size);
        if (tid == 0) y[row_base + 1] = sum1;
    }
}

// ── MLX Q6 GEMV ─────────────────────────────────────────────
// MLX affine 6-bit quantization: group_size=64, 12 u32 words per group.
// 6-bit values can span across u32 word boundaries (32 / 6 = 5 remainder 2),
// so extraction is scalar (cross-word reads needed for indices 5, 10, ...).
// Dequant: float_val = scale * uint6_val + bias (per-group bf16 scale and bias).

// Extract one 6-bit value from packed u32 array at element index idx.
inline uint unpack_u6(device const packed_uchar4* base, uint idx) {
    uint bp = idx * 6;
    uint wi = bp / 32;
    uint bo = bp % 32;

    // Reconstruct u32 from packed_uchar4 (alignment-safe)
    packed_uchar4 b0 = base[wi];
    uint word0 = uint(b0[0]) | (uint(b0[1]) << 8) | (uint(b0[2]) << 16) | (uint(b0[3]) << 24);

    if (bo <= 26) {
        // Value fits in single word
        return (word0 >> bo) & 0x3F;
    }
    // Value spans two words
    packed_uchar4 b1 = base[wi + 1];
    uint word1 = uint(b1[0]) | (uint(b1[1]) << 8) | (uint(b1[2]) << 16) | (uint(b1[3]) << 24);
    uint lo = word0 >> bo;
    uint hi = word1 << (32 - bo);
    return (lo | hi) & 0x3F;
}

kernel void gemv_mlx_q6(
    device const float* x              [[buffer(0)]],
    device const packed_uchar4* W      [[buffer(1)]],
    device const packed_uchar2* scales [[buffer(2)]],
    device const packed_uchar2* biases [[buffer(3)]],
    device float* y                    [[buffer(4)]],
    constant uint& n                   [[buffer(5)]],
    constant uint& k                   [[buffer(6)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint gs = 64;
    const uint wpg = 12;  // u32 words per group (64 * 6 / 32 = 12)
    uint gpr = (k + gs - 1) / gs;
    uint w_row = tgid * gpr * wpg;
    float sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        // BF16 scale and bias
        uint sb_idx = tgid * gpr + g;
        packed_uchar2 sb = scales[sb_idx];
        float scale = as_type<float>(uint(ushort(sb[0]) | (ushort(sb[1]) << 8)) << 16);
        packed_uchar2 bb = biases[sb_idx];
        float bias  = as_type<float>(uint(ushort(bb[0]) | (ushort(bb[1]) << 8)) << 16);

        uint xo = g * gs;
        device const packed_uchar4* wg = W + w_row + g * wpg;
        uint elems = min(gs, k - xo);

        float q_dot = 0.0f;
        float x_sum = 0.0f;

        for (uint i = 0; i < elems; i++) {
            float q = float(unpack_u6(wg, i));
            float xv = x[xo + i];
            q_dot += q * xv;
            x_sum += xv;
        }
        sum += scale * q_dot + bias * x_sum;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── MLX Q4 GEMV ─────────────────────────────────────────────
// MLX affine 4-bit quantization: group_size=64, 8 u32 words per group.
// Dequant: float_val = scale * uint4_val + bias (per-group bf16 scale and bias).
// Dot product: sum_i(x[i] * (s*q[i] + b)) = s * sum(x*q) + b * sum(x).
// Weights are uint* (4-byte aligned), scales/biases are ushort* (bf16, 2-byte
// aligned). The host backend copies misaligned SafeTensors data to aligned
// Metal buffers on first use (one-time cost, cached).

kernel void gemv_mlx_q4(
    device const float* x              [[buffer(0)]],
    device const packed_uchar4* W      [[buffer(1)]],
    device const packed_uchar2* scales [[buffer(2)]],
    device const packed_uchar2* biases [[buffer(3)]],
    device float* y                    [[buffer(4)]],
    constant uint& n                   [[buffer(5)]],
    constant uint& k                   [[buffer(6)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint gs = 64;
    const uint wpg = 8;    // u32 words per group (64 nibbles / 8 per word)
    uint gpr = (k + gs - 1) / gs;

    // packed_uchar4 loads: alignment-free 4-byte reads from mmap'd data
    // (SafeTensors tensors may start at odd byte offsets within pages)
    uint w_row = tgid * gpr * wpg;
    float sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        // BF16 scale and bias — packed_uchar2 load, reconstruct ushort
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
            // packed_uchar4 → reconstruct uint word (no alignment needed)
            packed_uchar4 bytes = W[wg + w];
            uint word = uint(bytes[0]) | (uint(bytes[1]) << 8) | (uint(bytes[2]) << 16) | (uint(bytes[3]) << 24);
            if (xi + 8 > k) {
                // Partial word at end of row
                for (uint i = 0; i < k - xi; i++) {
                    float q = float((word >> (i * 4)) & 0xF);
                    q_dot += q * x[xi + i];
                    x_sum += x[xi + i];
                }
                break;
            }
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

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── MLX 8-bit affine GEMV ────────────────────────────────────
// U32-packed bytes (4 per word), bf16 per-group scale + bias, group_size=64.
// Dequant: float_val = scale * u8_val + bias

kernel void gemv_mlx_q8(
    device const float* x              [[buffer(0)]],
    device const packed_uchar4* W      [[buffer(1)]],
    device const packed_uchar2* scales [[buffer(2)]],
    device const packed_uchar2* biases [[buffer(3)]],
    device float* y                    [[buffer(4)]],
    constant uint& n                   [[buffer(5)]],
    constant uint& k                   [[buffer(6)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint gs = 64;
    const uint wpg = 16;   // u32 words per group (64 bytes / 4 per word)
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

        for (uint w = 0; w < wpg && xo + w * 4 < k; w++) {
            uint xi = xo + w * 4;
            packed_uchar4 bytes = W[wg + w];
            float4 q = float4(float(bytes[0]), float(bytes[1]),
                              float(bytes[2]), float(bytes[3]));
            uint rem = min(uint(4), k - xi);
            if (rem == 4) {
                float4 xv = *(device const float4*)(x + xi);
                q_dot += dot(q, xv);
                x_sum += xv.x + xv.y + xv.z + xv.w;
            } else {
                for (uint i = 0; i < rem; i++) {
                    q_dot += q[i] * x[xi + i];
                    x_sum += x[xi + i];
                }
            }
        }
        sum += scale * q_dot + bias * x_sum;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── MXFP4 SafeTensors GEMV ──────────────────────────────────
// U32-packed 4-bit nibbles (8 per word), E8M0 per-group scale, group_size=32.
// Dequant: float_val = mxfp4_lut[nibble] * 2^(scale_byte - 127)
// E8M0 is a pure power-of-2 format (OCP Microscaling spec). No bias.

// E8M0 → float: val = 2^(byte - 127). Pure exponent, no mantissa bits.
inline float e8m0_to_f32(uchar val) {
    if (val == 0) return 0.0f;
    return as_type<float>(uint(val) << 23);
}

// MXFP4 E2M1 dequant lookup table
constant float mxfp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

kernel void gemv_mxfp4_st(
    device const float* x             [[buffer(0)]],
    device const packed_uchar4* W     [[buffer(1)]],
    device const uchar* scales        [[buffer(2)]],
    device float* y                   [[buffer(3)]],
    constant uint& n                  [[buffer(4)]],
    constant uint& k                  [[buffer(5)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint gs = 32;
    const uint wpg = 4;   // u32 words per group (32 nibbles / 8 per word)
    uint gpr = (k + gs - 1) / gs;
    uint w_row = tgid * gpr * wpg;
    float sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        float scale = e8m0_to_f32(scales[tgid * gpr + g]);
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
                for (uint i = 0; i < rem; i++) {
                    gdot += mxfp4_lut[(word >> (i * 4)) & 0xF] * x[xi + i];
                }
            }
        }
        sum += scale * gdot;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── MXFP4 GGUF GEMV ─────────────────────────────────────────
// 32 values per block, 17 bytes: 1 E8M0 scale + 16 nibble-packed bytes.
// Split-half packing: low nibble → position j, high nibble → position j+16.

kernel void gemv_mxfp4(
    device const float* x     [[buffer(0)]],
    device const uchar* W     [[buffer(1)]],
    device float* y            [[buffer(2)]],
    constant uint& n           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint qk = 32;
    const uint bpb = 17;
    uint nb = (k + qk - 1) / qk;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bp = tgid * nb * bpb + b * bpb;
        float d = e8m0_to_f32(W[bp]);
        uint bk = b * qk;

        for (uint j = 0; j < qk / 2; j++) {
            uchar byte_val = W[bp + 1 + j];
            float v0 = mxfp4_lut[byte_val & 0xF];
            float v1 = mxfp4_lut[byte_val >> 4];
            uint gi0 = bk + j;
            uint gi1 = bk + j + qk / 2;
            if (gi0 < k) sum += x[gi0] * v0 * d;
            if (gi1 < k) sum += x[gi1] * v1 * d;
        }
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── IQ4_NL GEMV ──────────────────────────────────────────────
// 32 values per block, 18 bytes (f16 scale + 16 nibble-packed bytes).
// Non-linear lookup table dequant instead of linear (Q4_0-like structure).

constant float iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

struct block_iq4_nl {
    half d;
    uchar qs[16];
};

kernel void gemv_iq4_nl(
    device const float* x        [[buffer(0)]],
    device const block_iq4_nl* W [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n             [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;
    uint nb = k / 32;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const block_iq4_nl& blk = W[tgid * nb + b];
        float d = float(blk.d);
        uint bk = b * 32;
        float block_sum = 0.0f;
        for (uint j = 0; j < 16; j += 4) {
            float4 lo = float4(iq4nl_lut[blk.qs[j]   & 0xF],
                               iq4nl_lut[blk.qs[j+1] & 0xF],
                               iq4nl_lut[blk.qs[j+2] & 0xF],
                               iq4nl_lut[blk.qs[j+3] & 0xF]);
            float4 hi = float4(iq4nl_lut[blk.qs[j]   >> 4],
                               iq4nl_lut[blk.qs[j+1] >> 4],
                               iq4nl_lut[blk.qs[j+2] >> 4],
                               iq4nl_lut[blk.qs[j+3] >> 4]);
            float4 xlo = *(device const float4*)(x + bk + j);
            float4 xhi = *(device const float4*)(x + bk + j + 16);
            block_sum += dot(lo, xlo) + dot(hi, xhi);
        }
        sum += block_sum * d;
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── IQ4_XS GEMV ──────────────────────────────────────────────
// 256 values per super-block, 138 bytes.
// Layout: { half d; ushort scales_h; uchar scales_l[8]; uchar qs[128]; }
// 8 sub-blocks of 32 values each, per-sub-block 6-bit scale (biased by -32).
// Uses IQ4_NL lookup table for nibble dequant.

kernel void gemv_iq4_xs(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    const uint bpb = 136;
    const uint bs = 256;
    uint nb = (k + bs - 1) / bs;
    float sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const uchar* bp = W + (tgid * nb + b) * bpb;
        float d = float(as_type<half>(ushort(bp[0] | (uint(bp[1]) << 8))));
        uint scales_h = uint(bp[2]) | (uint(bp[3]) << 8);
        device const uchar* scales_l = bp + 4;
        device const uchar* qs = bp + 8;
        uint bk = b * bs;

        for (uint sb = 0; sb < 8; sb++) {
            uint lo4 = (sb % 2 == 0) ? (scales_l[sb / 2] & 0xF) : (scales_l[sb / 2] >> 4);
            uint hi2 = (scales_h >> (sb * 2)) & 0x03;
            int scale_raw = int(lo4 | (hi2 << 4)) - 32;
            float sub_scale = d * float(scale_raw);

            device const uchar* sub_qs = qs + sb * 16;
            uint sub_bk = bk + sb * 32;
            float block_sum = 0.0f;

            for (uint j = 0; j < 16; j += 4) {
                float4 lo = float4(iq4nl_lut[sub_qs[j]   & 0xF],
                                   iq4nl_lut[sub_qs[j+1] & 0xF],
                                   iq4nl_lut[sub_qs[j+2] & 0xF],
                                   iq4nl_lut[sub_qs[j+3] & 0xF]);
                float4 hi = float4(iq4nl_lut[sub_qs[j]   >> 4],
                                   iq4nl_lut[sub_qs[j+1] >> 4],
                                   iq4nl_lut[sub_qs[j+2] >> 4],
                                   iq4nl_lut[sub_qs[j+3] >> 4]);
                float4 xlo = *(device const float4*)(x + sub_bk + j);
                float4 xhi = *(device const float4*)(x + sub_bk + j + 16);
                block_sum += dot(lo, xlo) + dot(hi, xhi);
            }
            sum += block_sum * sub_scale;
        }
    }

    threadgroup float tg_shared[8];
    sum = threadgroup_reduce_sum(sum, tg_shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── FP8 E4M3 GEMV ────────────────────────────────────────────
// 1 element per byte. Uses the fp8e4m3_lut already defined above.

kernel void gemv_fp8_e4m3(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    float sum = 0.0f;
    uint row_off = tgid * k;
    for (uint j = tid; j < k; j += tg_size) {
        sum += fp8e4m3_lut[W[row_off + j]] * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── FP8 E5M2 GEMV ────────────────────────────────────────────
// 1 element per byte. Bit layout: seeeeemm (5 exponent, 2 mantissa, bias=15).

inline float fp8e5m2_to_f32(uchar val) {
    uint sign = (uint(val) >> 7) << 31;
    uint exp5 = (uint(val) >> 2) & 0x1F;
    uint mant = uint(val) & 0x3;

    if (exp5 == 0) {
        // Denorm: 2^(-14) * (mant/4)
        float m = float(mant) / 4.0f;
        float result = m * 6.103515625e-05f; // 2^(-14)
        return (sign != 0) ? -result : result;
    }
    if (exp5 == 31) {
        if (mant == 0) return as_type<float>(sign | 0x7F800000u); // inf
        return as_type<float>(0x7FC00000u); // NaN
    }
    return as_type<float>(sign | ((exp5 + 112) << 23) | (mant << 21));
}

kernel void gemv_fp8_e5m2(
    device const float* x [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    constant uint& k      [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= n) return;

    float sum = 0.0f;
    uint row_off = tgid * k;
    for (uint j = tid; j < k; j += tg_size) {
        sum += fp8e5m2_to_f32(W[row_off + j]) * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}

// ── Transposed Q8_0 GEMV ─────────────────────────────────────
// Computes y[out_dim] = W^T @ x[in_dim] where W is stored as
// [in_dim rows, out_dim cols] in Q8_0 (GGUF 3D multi-head layout).
// Used for MLA K_nope/V projections (embed_q, unembed_out).
//
// Dispatch: one threadgroup per output element (dispatchThreadgroups(out_dim,1,1)).
// Each threadgroup reduces across all in_dim rows for its output column.
// Q8_0 blocks run along out_dim (each row has ceil(out_dim/32) blocks).

kernel void gemv_t_q8_0(
    device const float* x          [[buffer(0)]],
    device const block_q8_0* W     [[buffer(1)]],
    device float* y                [[buffer(2)]],
    constant uint& out_dim         [[buffer(3)]],
    constant uint& in_dim          [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]])
{
    if (tgid >= out_dim) return;

    // Which Q8_0 block and element within block for output column tgid
    uint blocks_per_row = (out_dim + 31) / 32;
    uint blk_col = tgid / 32;
    uint blk_off = tgid % 32;

    float sum = 0.0f;
    // Each thread strides over in_dim rows
    for (uint j = tid; j < in_dim; j += tg_size) {
        device const block_q8_0& blk = W[j * blocks_per_row + blk_col];
        float w = float(blk.qs[blk_off]) * float(blk.d);
        sum += w * x[j];
    }

    threadgroup float shared[8];
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) y[tgid] = sum;
}
