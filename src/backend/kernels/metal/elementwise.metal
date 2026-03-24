// Element-wise operations: SiLU, add, mul, GELU, sigmoid_mul, deinterleave, silu_mul, kv_append.

// SiLU activation: y = x * sigmoid(x)
kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float x = input[tid];
    output[tid] = x / (1.0f + exp(-x));
}

// Vector add: out = a + b
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    out[tid] = a[tid] + b[tid];
}

// Vector multiply: out = a * b
kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    out[tid] = a[tid] * b[tid];
}

// Sigmoid-gated multiply: out[i] = a[i] * sigmoid(b[i])
// Used by Qwen3.5 attention gate to avoid a CPU sync.
// When a == out, this is in-place: data[i] *= sigmoid(gate[i]).
kernel void sigmoid_mul_f32(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    out[tid] = a[tid] / (1.0f + exp(-b[tid]));
}

// De-interleave paired blocks: input has [A0(stride), B0(stride), A1(stride), B1(stride), ...]
// Extracts A into out_a (compacted) and B into out_b (compacted).
// Total threads = n_pairs * stride. Each thread copies one element.
kernel void deinterleave_f32(
    device const float* input [[buffer(0)]],
    device float* out_a       [[buffer(1)]],
    device float* out_b       [[buffer(2)]],
    constant uint& stride     [[buffer(3)]],
    constant uint& n_pairs    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = n_pairs * stride;
    if (tid >= total) return;
    uint pair = tid / stride;
    uint lane = tid % stride;
    out_a[pair * stride + lane] = input[pair * 2 * stride + lane];
    out_b[pair * stride + lane] = input[pair * 2 * stride + stride + lane];
}

// Fused SiLU + multiply: out[i] = silu(a[i]) * b[i]
// Used in SwiGLU FFN to avoid separate silu + mul dispatches.
kernel void silu_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float x = a[tid];
    out[tid] = (x / (1.0f + exp(-x))) * b[tid];
}

// GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    float x = input[tid];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * fma(coeff * x * x, x, x);
    // tanh via (exp(2x)-1)/(exp(2x)+1), clamped to prevent exp overflow
    float clamped = clamp(inner, -10.0f, 10.0f);
    float e2 = exp(2.0f * clamped);
    output[tid] = 0.5f * x * (1.0f + (e2 - 1.0f) / (e2 + 1.0f));
}

// Scaled accumulate: dst[i] += src[i] * scale
// Used for MoE expert output accumulation to avoid per-expert GPU sync.
kernel void add_scaled_f32(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant float& scale    [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;
    dst[tid] += src[tid] * scale;
}

// ── KV cache append (compute-based) ─────────────────────────
// Copies k_new and v_new into KV cache at the given offset.
// Runs on the compute encoder — avoids blit encoder switching overhead.

kernel void kv_append(
    device const float* k_new  [[buffer(0)]],
    device const float* v_new  [[buffer(1)]],
    device float* keys         [[buffer(2)]],
    device float* values       [[buffer(3)]],
    constant uint& kvd         [[buffer(4)]],
    constant uint& kv_off      [[buffer(5)]],  // seq_len * kvd
    uint tid [[thread_position_in_grid]])
{
    if (tid >= kvd) return;
    keys[kv_off + tid] = k_new[tid];
    values[kv_off + tid] = v_new[tid];
}
