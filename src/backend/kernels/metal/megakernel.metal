// Fused FFN megakernels — eliminates per-layer dispatch overhead.
//
// Standard FFN path: 4 dispatches per layer (gate GEMV, up GEMV, siluMul, down GEMV)
//   = 96 dispatches for 24 layers.
// Fused path: 1 dispatch per layer for gate+up+siluMul = 24 dispatches saved.
//   Down GEMV remains a separate dispatch (different output dimension).
//   Total savings: 72 dispatches -> 24 dispatches for gate+up+siluMul across all layers.
//
// Dispatch model: n_ff threadgroups x 256 threads per threadgroup.
// Each threadgroup computes one output element of the fused gate+up+siluMul result.
// The kernel reuses the established GEMV pattern from gemv.metal: one TG per output
// row, cooperative reduction via simd_sum, thread 0 writes the final result.
//
// Prerequisites (provided by earlier .metal files in the concatenated source):
//   common.metal:  threadgroup_reduce_sum()
//   gemv.metal:    block_q8_0 struct, q8_0_block_dot() inline helper

// ── Fused Gate + Up + SiLU*Mul kernel (Q8_0) ─────────────────
//
// Replaces 3 separate dispatches per FFN layer:
//   1. gate GEMV:  gate[n_ff] = W_gate[n_ff, n_embd] @ x[n_embd]
//   2. up GEMV:    up[n_ff]   = W_up[n_ff, n_embd]   @ x[n_embd]
//   3. silu_mul:   out[i]     = silu(gate[i]) * up[i]
//
// with a single dispatch that computes all three in one pass per row.
//
// Buffer layout:
//   buffer(0): x[n_embd]                — normalized hidden state (f32)
//   buffer(1): W_gate[n_ff * nb * 34]   — gate weights, Q8_0 row-major
//   buffer(2): W_up[n_ff * nb * 34]     — up weights, Q8_0 row-major
//   buffer(3): ff_out[n_ff]             — output: silu(gate) * up (f32)
//   buffer(4): n_ff                     — FFN intermediate dimension
//   buffer(5): n_embd                   — model embedding dimension
//
// Dispatch: n_ff threadgroups x 256 threads.
// Each TG computes ff_out[tgid] = silu(dot(W_gate[tgid,:], x)) * dot(W_up[tgid,:], x).

kernel void fused_ffn_gate_up_silu_q8(
    device const float*       x       [[buffer(0)]],
    device const block_q8_0*  W_gate  [[buffer(1)]],
    device const block_q8_0*  W_up    [[buffer(2)]],
    device float*             ff_out  [[buffer(3)]],
    constant uint&            n_ff    [[buffer(4)]],
    constant uint&            n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;

    uint nb = n_embd / 32; // Q8_0 blocks per row

    // Accumulate gate and up dot products in parallel.
    // Both rows read the same x blocks — compiler hoists x loads
    // because q8_0_block_dot is inlined.
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        device const float* x_block = x + b * 32;
        gate_sum += q8_0_block_dot(W_gate[tgid * nb + b], x_block);
        up_sum   += q8_0_block_dot(W_up[tgid * nb + b],   x_block);
    }

    // Reduce gate sum across threadgroup
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);

    // Barrier before reusing shared memory for the up reduction
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce up sum across threadgroup
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);

    // SiLU(gate) * up — only thread 0 writes the final result
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}

// ── Fused Gate + Up + GELU*Mul kernel (Q8_0) ─────────────────
// GELU variant for Gemma models with Q8_0 weights.

kernel void fused_ffn_gate_up_gelu_q8(
    device const float*       x       [[buffer(0)]],
    device const block_q8_0*  W_gate  [[buffer(1)]],
    device const block_q8_0*  W_up    [[buffer(2)]],
    device float*             ff_out  [[buffer(3)]],
    constant uint&            n_ff    [[buffer(4)]],
    constant uint&            n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    uint nb = n_embd / 32;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        device const float* x_block = x + b * 32;
        gate_sum += q8_0_block_dot(W_gate[tgid * nb + b], x_block);
        up_sum   += q8_0_block_dot(W_up[tgid * nb + b],   x_block);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * fma(0.044715f * gate_sum * gate_sum, gate_sum, gate_sum);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        ff_out[tgid] = 0.5f * gate_sum * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up_sum;
    }
}

// ── Fused Gate + Up + SiLU*Mul kernel (Q4_K) ─────────────────
// Same fusion for Q4_K quantization (256 values per super-block, 144 bytes).
// Uses q4_k_block_dot from gemv.metal for both gate and up rows.
//
// Buffer layout:
//   buffer(0): x[n_embd]                — normalized hidden state (f32)
//   buffer(1): W_gate[n_ff * nb * 144]  — gate weights, Q4_K row-major (raw bytes)
//   buffer(2): W_up[n_ff * nb * 144]    — up weights, Q4_K row-major (raw bytes)
//   buffer(3): ff_out[n_ff]             — output: silu(gate) * up (f32)
//   buffer(4): n_ff                     — FFN intermediate dimension
//   buffer(5): n_embd                   — model embedding dimension
//
// Dispatch: n_ff threadgroups x 256 threads.

kernel void fused_ffn_gate_up_silu_q4_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;

    const uint bpb = 144; // bytes per Q4_K superblock
    const uint bs = 256;  // values per superblock
    uint nb = (n_embd + bs - 1) / bs;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q4_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q4_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }

    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);

    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}

// ── Fused Gate + Up + SiLU*Mul kernel (Q4_0) ─────────────────
// Same fusion for Q4_0 quantization (32 values per block).
// Uses q4_0_block_dot from gemv.metal with pre-loaded float4 x values.
//
// Buffer layout:
//   buffer(0): x[n_embd]                    — normalized hidden state (f32)
//   buffer(1): W_gate[n_ff * nb]            — gate weights, Q4_0 row-major
//   buffer(2): W_up[n_ff * nb]              — up weights, Q4_0 row-major
//   buffer(3): ff_out[n_ff]                 — output: silu(gate) * up (f32)
//   buffer(4): n_ff                         — FFN intermediate dimension
//   buffer(5): n_embd                       — model embedding dimension
//
// Dispatch: n_ff threadgroups x 256 threads.

kernel void fused_ffn_gate_up_silu_q4_0(
    device const float*       x       [[buffer(0)]],
    device const block_q4_0*  W_gate  [[buffer(1)]],
    device const block_q4_0*  W_up    [[buffer(2)]],
    device float*             ff_out  [[buffer(3)]],
    constant uint&            n_ff    [[buffer(4)]],
    constant uint&            n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    uint nb = n_embd / 32;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

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

        gate_sum += q4_0_block_dot(W_gate[tgid * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
        up_sum   += q4_0_block_dot(W_up[tgid * nb + b],   x0, x1, x2, x3, x4, x5, x6, x7);
    }

    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);

    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}

// ── Fused Gate + Up + GELU*Mul kernels (for Gemma 3/4) ───────
// Same dispatch pattern as SiLU variants, but uses GELU activation.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

kernel void fused_ffn_gate_up_gelu_q4_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint bpb = 144;
    const uint bs = 256;
    uint nb = (n_embd + bs - 1) / bs;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q4_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q4_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * fma(0.044715f * gate_sum * gate_sum, gate_sum, gate_sum);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        ff_out[tgid] = 0.5f * gate_sum * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up_sum;
    }
}

kernel void fused_ffn_gate_up_gelu_q4_0(
    device const float*       x       [[buffer(0)]],
    device const block_q4_0*  W_gate  [[buffer(1)]],
    device const block_q4_0*  W_up    [[buffer(2)]],
    device float*             ff_out  [[buffer(3)]],
    constant uint&            n_ff    [[buffer(4)]],
    constant uint&            n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    uint nb = n_embd / 32;
    float gate_sum = 0.0f, up_sum = 0.0f;
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
        gate_sum += q4_0_block_dot(W_gate[tgid * nb + b], x0, x1, x2, x3, x4, x5, x6, x7);
        up_sum   += q4_0_block_dot(W_up[tgid * nb + b],   x0, x1, x2, x3, x4, x5, x6, x7);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * fma(0.044715f * gate_sum * gate_sum, gate_sum, gate_sum);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        ff_out[tgid] = 0.5f * gate_sum * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up_sum;
    }
}

// ── Fused Gate + Up + SiLU*Mul kernel (Q6_K) ─────────────────
// Same fusion for Q6_K quantization (256 values per super-block, 210 bytes).
// Uses q6_k_block_dot from gemv.metal for both gate and up rows.
//
// Buffer layout:
//   buffer(0): x[n_embd]                — normalized hidden state (f32)
//   buffer(1): W_gate[n_ff * nb * 210]  — gate weights, Q6_K row-major (raw bytes)
//   buffer(2): W_up[n_ff * nb * 210]    — up weights, Q6_K row-major (raw bytes)
//   buffer(3): ff_out[n_ff]             — output: silu(gate) * up (f32)
//   buffer(4): n_ff                     — FFN intermediate dimension
//   buffer(5): n_embd                   — model embedding dimension
//
// Dispatch: n_ff threadgroups x 256 threads.

kernel void fused_ffn_gate_up_silu_q6_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint bpb = 210;
    const uint bs = 256;
    uint nb = (n_embd + bs - 1) / bs;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q6_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q6_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}

// ── Fused Gate + Up + GELU*Mul kernel (Q6_K) ─────────────────

kernel void fused_ffn_gate_up_gelu_q6_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint bpb = 210;
    const uint bs = 256;
    uint nb = (n_embd + bs - 1) / bs;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q6_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q6_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * fma(0.044715f * gate_sum * gate_sum, gate_sum, gate_sum);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        ff_out[tgid] = 0.5f * gate_sum * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up_sum;
    }
}

// ── Fused Gate + Up + SiLU*Mul kernel (Q5_K) ─────────────────
// Same fusion for Q5_K quantization (256 values per super-block, 176 bytes).
// Uses q5_k_block_dot from gemv.metal for both gate and up rows.
//
// Buffer layout:
//   buffer(0): x[n_embd]                — normalized hidden state (f32)
//   buffer(1): W_gate[n_ff * nb * 176]  — gate weights, Q5_K row-major (raw bytes)
//   buffer(2): W_up[n_ff * nb * 176]    — up weights, Q5_K row-major (raw bytes)
//   buffer(3): ff_out[n_ff]             — output: silu(gate) * up (f32)
//   buffer(4): n_ff                     — FFN intermediate dimension
//   buffer(5): n_embd                   — model embedding dimension
//
// Dispatch: n_ff threadgroups x 256 threads.

kernel void fused_ffn_gate_up_silu_q5_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint bpb = 176;
    const uint bs = 256;
    uint nb = (n_embd + bs - 1) / bs;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q5_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q5_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}

// ── Fused Gate + Up + GELU*Mul kernel (Q5_K) ─────────────────

kernel void fused_ffn_gate_up_gelu_q5_k(
    device const float*  x       [[buffer(0)]],
    device const uchar*  W_gate  [[buffer(1)]],
    device const uchar*  W_up    [[buffer(2)]],
    device float*        ff_out  [[buffer(3)]],
    constant uint&       n_ff    [[buffer(4)]],
    constant uint&       n_embd  [[buffer(5)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint bpb = 176;
    const uint bs = 256;
    uint nb = (n_embd + bs - 1) / bs;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < nb; b += tg_size) {
        uint bk = b * bs;
        gate_sum += q5_k_block_dot(W_gate + (tgid * nb + b) * bpb, x, n_embd, bk);
        up_sum   += q5_k_block_dot(W_up   + (tgid * nb + b) * bpb, x, n_embd, bk);
    }
    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);
    if (tid == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float inner = sqrt_2_over_pi * fma(0.044715f * gate_sum * gate_sum, gate_sum, gate_sum);
        float clamped = clamp(inner, -10.0f, 10.0f);
        float e2 = exp(2.0f * clamped);
        ff_out[tgid] = 0.5f * gate_sum * (1.0f + (e2 - 1.0f) / (e2 + 1.0f)) * up_sum;
    }
}

// ── Fused Gate + Up + SiLU*Mul kernel (MLX Q4) ───────────────
// MLX affine 4-bit: group_size=64, 8 u32 words/group, bf16 scale+bias per group.
// Fuses gate GEMV + up GEMV + SiLU*mul for GLM-4 and other MLX-quantized models.
// Buffer count: 10 (x, gate_w/s/b, up_w/s/b, out, n_ff, n_embd)

kernel void fused_ffn_gate_up_silu_mlx_q4(
    device const float*          x         [[buffer(0)]],
    device const packed_uchar4*  W_gate    [[buffer(1)]],
    device const packed_uchar2*  gs_gate   [[buffer(2)]],
    device const packed_uchar2*  gb_gate   [[buffer(3)]],
    device const packed_uchar4*  W_up      [[buffer(4)]],
    device const packed_uchar2*  gs_up     [[buffer(5)]],
    device const packed_uchar2*  gb_up     [[buffer(6)]],
    device float*                ff_out    [[buffer(7)]],
    constant uint&               n_ff      [[buffer(8)]],
    constant uint&               n_embd    [[buffer(9)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= n_ff) return;
    const uint grp_size = 64;
    const uint wpg = 8;
    uint gpr = (n_embd + grp_size - 1) / grp_size;
    uint w_row = tgid * gpr * wpg;

    float gate_sum = 0.0f, up_sum = 0.0f;

    for (uint g = tid; g < gpr; g += tg_size) {
        uint sb_idx = tgid * gpr + g;
        uint xo = g * grp_size;

        // Gate scale+bias (BF16)
        packed_uchar2 gsb = gs_gate[sb_idx];
        float g_scale = as_type<float>(uint(ushort(gsb[0]) | (ushort(gsb[1]) << 8)) << 16);
        packed_uchar2 gbb = gb_gate[sb_idx];
        float g_bias  = as_type<float>(uint(ushort(gbb[0]) | (ushort(gbb[1]) << 8)) << 16);

        // Up scale+bias (BF16)
        packed_uchar2 usb = gs_up[sb_idx];
        float u_scale = as_type<float>(uint(ushort(usb[0]) | (ushort(usb[1]) << 8)) << 16);
        packed_uchar2 ubb = gb_up[sb_idx];
        float u_bias  = as_type<float>(uint(ushort(ubb[0]) | (ushort(ubb[1]) << 8)) << 16);

        float g_qd = 0.0f, g_xs = 0.0f;
        float u_qd = 0.0f;

        for (uint w = 0; w < wpg; w++) {
            uint xi = xo + w * 8;
            packed_uchar4 gb4 = W_gate[w_row + g * wpg + w];
            uint gword = uint(gb4[0]) | (uint(gb4[1]) << 8) | (uint(gb4[2]) << 16) | (uint(gb4[3]) << 24);
            packed_uchar4 ub4 = W_up[w_row + g * wpg + w];
            uint uword = uint(ub4[0]) | (uint(ub4[1]) << 8) | (uint(ub4[2]) << 16) | (uint(ub4[3]) << 24);

            for (uint i = 0; i < 8 && xi + i < n_embd; i++) {
                float xv = x[xi + i];
                g_qd += float((gword >> (i * 4)) & 0xF) * xv;
                u_qd += float((uword >> (i * 4)) & 0xF) * xv;
                g_xs += xv;
            }
        }
        gate_sum += g_scale * g_qd + g_bias * g_xs;
        up_sum   += u_scale * u_qd + u_bias * g_xs;
    }

    threadgroup float shared[8];
    gate_sum = threadgroup_reduce_sum(gate_sum, shared, tid, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    up_sum = threadgroup_reduce_sum(up_sum, shared, tid, tg_size);

    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        ff_out[tgid] = silu_gate * up_sum;
    }
}
