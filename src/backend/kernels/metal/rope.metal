// RoPE (Rotary Position Embedding)
//
// Split-complex layout: pairs [i, i+half] are rotated together.
// Each thread handles one (real, imag) pair.
// Grid size = n_heads * rope_dim / 2

kernel void rope_f32(
    device float* x         [[buffer(0)]],
    constant uint& pos      [[buffer(1)]],
    constant uint& n_heads  [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& rope_dim [[buffer(4)]],
    constant float& theta   [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint half_rope = rope_dim / 2;
    uint total = n_heads * half_rope;
    if (tid >= total) return;

    uint h = tid / half_rope;
    uint i = tid % half_rope;
    uint base = h * head_dim;

    float freq = exp(-log(theta) * float(2 * i) / float(rope_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float r  = x[base + i];
    float im = x[base + i + half_rope];
    x[base + i]             = r * cos_a - im * sin_a;
    x[base + i + half_rope] = r * sin_a + im * cos_a;
}

// ── Batched RoPE ──────────────────────────────────────────────
// Apply RoPE to n_tok vectors at different positions.
// Grid size = n_tok * n_heads * rope_dim / 2
// Each thread handles one (real, imag) pair for one token.

kernel void rope_batched_f32(
    device float* x                  [[buffer(0)]],
    device const uint* positions     [[buffer(1)]],
    constant uint& n_tok             [[buffer(2)]],
    constant uint& n_heads           [[buffer(3)]],
    constant uint& head_dim          [[buffer(4)]],
    constant uint& rope_dim          [[buffer(5)]],
    constant float& theta            [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint half_rope = rope_dim / 2;
    uint pairs_per_tok = n_heads * half_rope;
    uint total = n_tok * pairs_per_tok;
    if (tid >= total) return;

    uint tok = tid / pairs_per_tok;
    uint pair = tid % pairs_per_tok;
    uint h = pair / half_rope;
    uint i = pair % half_rope;

    uint stride = n_heads * head_dim;
    uint base = tok * stride + h * head_dim;
    uint pos = positions[tok];

    float freq = exp(-log(theta) * float(2 * i) / float(rope_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float r  = x[base + i];
    float im = x[base + i + half_rope];
    x[base + i]             = r * cos_a - im * sin_a;
    x[base + i + half_rope] = r * sin_a + im * cos_a;
}
