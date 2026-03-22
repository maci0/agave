// Normalization kernels: RMS norm (2-pass + fused), add+RMS norm (fused), softmax (3-pass), L2 norm.

// ── RMS Norm (two-kernel approach) ────────────────────────────

// Pass 1: Compute sum of squares (reduction into a single float)
// Uses simd_sum for fast warp-level reduction. Launch with 1 threadgroup.
kernel void rms_norm_ss(
    device const float* input [[buffer(0)]],
    device float* ss_out      [[buffer(1)]],  // single float output
    constant uint& n          [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[8];
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        sum += input[i] * input[i];
    }
    sum = threadgroup_reduce_sum(sum, shared, tgid, tg_size);
    if (tgid == 0) ss_out[0] = sum;
}

// Pass 2: Normalize: output[i] = input[i] * weight[i] * rsqrt(ss/n + eps)
kernel void rms_norm_apply(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    device const float* ss_buf [[buffer(3)]],  // sum-of-squares from pass 1
    constant uint& n           [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float inv_rms = rsqrt(ss_buf[0] / float(n) + eps);
    output[tid] = input[tid] * weight[tid] * inv_rms;
}

// ── Fused RMS Norm (single-dispatch, one threadgroup per head) ──

// Single-dispatch fused RMS norm with separate input/output.
// Can handle both single vectors (1 threadgroup) and batched per-head
// normalization (n_heads threadgroups, shared weight).
// Each threadgroup: simd_sum reduction → barrier → normalize.
// Supports in-place when input == output.
kernel void rms_norm_fused_f32(
    device const float* input     [[buffer(0)]],
    device const float* weight    [[buffer(1)]],
    device float* output          [[buffer(2)]],
    constant uint& head_dim       [[buffer(3)]],
    constant float& eps           [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tgid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[8];
    uint base = gid * head_dim;

    // Sum of squares reduction using simd_sum (much faster than tree reduction)
    float sum = 0.0f;
    for (uint i = tgid; i < head_dim; i += tg_size) {
        float v = input[base + i];
        sum += v * v;
    }
    sum = threadgroup_reduce_sum(sum, shared, tgid, tg_size);
    // Broadcast sum to all threads
    if (tgid == 0) shared[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = shared[0];

    // Apply normalization
    float inv_rms = rsqrt(sum / float(head_dim) + eps);
    for (uint i = tgid; i < head_dim; i += tg_size) {
        output[base + i] = input[base + i] * weight[i] * inv_rms;
    }
}

// ── Fused Add + RMS Norm (single-dispatch) ────────────────────
// Computes: a[i] = a[i] + b[i], output[i] = rms_norm(a+b, weight, eps)
// Two-pass within one threadgroup: sum-of-squares reduction, then normalize.
// Re-reads a and b in pass 2 (L2-hot from pass 1) to avoid device memory
// write-then-read of the intermediate sum.
// Supports output == a (final logits case — raw sum is not needed).
kernel void add_rms_norm_fused_f32(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[8];

    // Pass 1: sum of squares of (a + b)
    float sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = a[i] + b[i];
        sum += v * v;
    }
    sum = threadgroup_reduce_sum(sum, shared, tid, tg_size);
    if (tid == 0) shared[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = shared[0];

    // Pass 2: write a = a+b and output = norm(a+b)
    float inv_rms = rsqrt(sum / float(n) + eps);
    for (uint i = tid; i < n; i += tg_size) {
        float v = a[i] + b[i];
        a[i] = v;
        output[i] = v * weight[i] * inv_rms;
    }
}

// ── Softmax (three-pass with threadgroup reduction) ───────────

// Pass 1: Find max (reduction) — uses simd_max for fast warp-level reduction.
kernel void softmax_max(
    device const float* data [[buffer(0)]],
    device float* max_out    [[buffer(1)]],
    constant uint& n         [[buffer(2)]],
    uint tgid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[8];
    float mx = -INFINITY;
    for (uint i = tgid; i < n; i += tg_size) {
        mx = max(mx, data[i]);
    }
    mx = threadgroup_reduce_max(mx, shared, tgid, tg_size);
    if (tgid == 0) max_out[0] = mx;
}

// Pass 2: exp(x - max) and sum (reduction) — uses simd_sum for fast reduction.
kernel void softmax_exp_sum(
    device float* data          [[buffer(0)]],
    device const float* max_buf [[buffer(1)]],
    device float* sum_out       [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared[8];
    float mx = max_buf[0];
    float s = 0.0f;
    for (uint i = tgid; i < n; i += tg_size) {
        float v = exp(data[i] - mx);
        data[i] = v;
        s += v;
    }
    s = threadgroup_reduce_sum(s, shared, tgid, tg_size);
    if (tgid == 0) sum_out[0] = s;
}

// Pass 3: Normalize by sum
kernel void softmax_div(
    device float* data          [[buffer(0)]],
    device const float* sum_buf [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    data[tid] /= sum_buf[0];
}

// ── L2 Norm (in-place) ───────────────────────────────────────

// Pass 1: sum of squares — reuse rms_norm_ss kernel above.
// Pass 2: normalize in-place.
kernel void l2_norm_apply(
    device float* x            [[buffer(0)]],
    device const float* ss_buf [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float inv = rsqrt(ss_buf[0] + eps);
    x[tid] *= inv;
}
