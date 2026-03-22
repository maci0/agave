"""
PyTorch reference implementations of Agave's core kernels.
These produce ground-truth outputs for golden tests and serve as
correctness oracles for the Zig/MSL/GLSL implementations.

All functions operate on CPU tensors (f32) for deterministic output.
"""

import torch
import torch.nn.functional as F
import math


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight"""
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return (x / rms) * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation (tanh approximation)"""
    return F.gelu(x, approximate="tanh")


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos: int,
    head_dim: int,
    rope_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Position Embedding applied to Q and K.
    q, k: [n_heads * head_dim] flat vectors.
    Only the first rope_dim dimensions of each head are rotated.
    """
    n_heads = q.shape[0] // head_dim
    half = rope_dim // 2
    q_out = q.clone()
    k_out = k.clone()

    freqs = torch.exp(
        -math.log(theta) * torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim
    )
    angles = pos * freqs  # [half]

    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            # Q
            r, im = q[base + i].item(), q[base + i + half].item()
            q_out[base + i] = r * cos_vals[i] - im * sin_vals[i]
            q_out[base + i + half] = r * sin_vals[i] + im * cos_vals[i]
            # K
            r, im = k[base + i].item(), k[base + i + half].item()
            k_out[base + i] = r * cos_vals[i] - im * sin_vals[i]
            k_out[base + i + half] = r * sin_vals[i] + im * cos_vals[i]

    return q_out, k_out


def sdpa(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    scale: float = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention (naive, for reference).
    q: [n_heads * head_dim]
    keys: [seq_len * n_kv_heads * head_dim]
    values: [seq_len * n_kv_heads * head_dim]
    Returns: [n_heads * head_dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len = keys.shape[0] // (n_kv_heads * head_dim)
    kvd = n_kv_heads * head_dim
    hpg = n_heads // n_kv_heads
    output = torch.zeros(n_heads * head_dim)

    for h in range(n_heads):
        kvh = h // hpg
        q_head = q[h * head_dim : (h + 1) * head_dim]

        # QK dot products
        scores = torch.zeros(seq_len)
        for t in range(seq_len):
            k_head = keys[t * kvd + kvh * head_dim : t * kvd + (kvh + 1) * head_dim]
            scores[t] = torch.dot(q_head, k_head) * scale

        # Softmax
        weights = F.softmax(scores, dim=0)

        # Value accumulation
        for t in range(seq_len):
            v_head = values[t * kvd + kvh * head_dim : t * kvd + (kvh + 1) * head_dim]
            output[h * head_dim : (h + 1) * head_dim] += weights[t] * v_head

    return output


def sdpa_online(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    scale: float = None,
) -> torch.Tensor:
    """
    FlashAttention-style SDPA with online softmax.
    Produces identical output to sdpa() but uses O(1) memory per head
    (no scores buffer). This is the algorithm to port to Zig.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len = keys.shape[0] // (n_kv_heads * head_dim)
    kvd = n_kv_heads * head_dim
    hpg = n_heads // n_kv_heads
    output = torch.zeros(n_heads * head_dim)

    for h in range(n_heads):
        kvh = h // hpg
        q_head = q[h * head_dim : (h + 1) * head_dim]

        m_i = float("-inf")  # running max
        l_i = 0.0  # running sum
        acc = torch.zeros(head_dim)

        for t in range(seq_len):
            k_head = keys[t * kvd + kvh * head_dim : t * kvd + (kvh + 1) * head_dim]
            v_head = values[t * kvd + kvh * head_dim : t * kvd + (kvh + 1) * head_dim]

            score = torch.dot(q_head, k_head).item() * scale

            # Online softmax update
            m_new = max(m_i, score)
            alpha = math.exp(m_i - m_new) if m_i != float("-inf") else 0.0
            p = math.exp(score - m_new)

            l_i = l_i * alpha + p
            acc = acc * alpha + p * v_head
            m_i = m_new

        output[h * head_dim : (h + 1) * head_dim] = acc / l_i

    return output


def gemv_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Matrix-vector multiply: y = W @ x"""
    return w @ x


def swiglu_fused(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    """Fused SwiGLU: silu(W_gate @ x) * (W_up @ x)"""
    gate = silu(w_gate @ x)
    up = w_up @ x
    return gate * up


def argmax(x: torch.Tensor) -> int:
    """Index of maximum element."""
    return torch.argmax(x).item()


# ============================================================================
# Elementwise & Math Operations
# ============================================================================


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector add: out = a + b"""
    return a + b


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector multiply: out = a * b"""
    return a * b


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return torch.sigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    """Softplus: log(1 + exp(x)), numerically stable"""
    return F.softplus(x)


# ============================================================================
# Normalization Operations
# ============================================================================


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax: exp(x - max) / sum(exp(x - max))"""
    return F.softmax(x, dim=0)


def l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalization: x / sqrt(sum(x^2) + eps)"""
    return x / torch.sqrt(torch.sum(x * x) + eps)


# ============================================================================
# Embedding
# ============================================================================


def embedding_lookup(table: torch.Tensor, idx: int) -> torch.Tensor:
    """Embedding lookup: return table[idx] (copies row, then scales by sqrt(n_embd) for Gemma)."""
    return table[idx].clone()


# ============================================================================
# SSM Operations
# ============================================================================


def conv1d_causal(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Causal 1D convolution for SSM layers (single-step, ring buffer).
    x: [d_conv * n_ch] input vector
    weight: [d_conv, n_ch] convolution weights
    state: [d_conv, n_ch] ring buffer state (d_conv rows, each n_ch wide)
    Returns: (output [n_ch], updated_state [d_conv, n_ch])

    Implementation: shift state left, append x at end, dot product with weight.
    """
    d_conv = weight.shape[0]
    n_ch = weight.shape[1]

    new_state = state.clone()
    # Shift left: rows 1..d_conv-1 move to 0..d_conv-2
    new_state[:-1] = state[1:]
    # Append current input at end
    new_state[-1] = x[:n_ch]

    # Dot product: sum(weight * state) along conv dimension
    output = (new_state * weight).sum(dim=0)
    return output, new_state


def deltanet_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeltaNet linear attention recurrence (single step).
    q: [head_dim] query vector
    k: [head_dim] key vector
    v: [head_dim] value vector
    beta: scalar learning rate
    state: [head_dim, head_dim] recurrent state matrix
    Returns: (output [head_dim], updated_state [head_dim, head_dim])

    Update: S' = S + beta * (v - S @ k) outer k
    Output: o = S' @ q
    """
    # Compute error: v - S @ k
    error = v - state @ k
    # Outer product update: S += beta * error @ k^T
    new_state = state + beta * torch.outer(error, k)
    # Output
    output = new_state @ q
    return output, new_state


# ============================================================================
# MoE Routing
# ============================================================================


def moe_routing_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k expert routing: softmax over logits, select top-k experts.
    logits: [n_experts] raw routing logits
    k: number of experts to select
    Returns: (indices [k], weights [k]) — selected expert indices and normalized weights
    """
    probs = F.softmax(logits, dim=0)
    topk_vals, topk_idx = torch.topk(probs, k)
    # Renormalize weights to sum to 1
    topk_weights = topk_vals / topk_vals.sum()
    return topk_idx, topk_weights


def moe_routing_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid expert routing (GLM4-style): each expert gate is independent.
    logits: [n_experts] raw routing logits
    Returns: [n_experts] sigmoid gating weights (no normalization)
    """
    return torch.sigmoid(logits)


# ============================================================================
# Quantization (Dequantization)
# ============================================================================


def fp8_e4m3_dequant(data: torch.Tensor) -> torch.Tensor:
    """
    FP8 E4M3 dequantization.
    data: [n] tensor of uint8 values representing FP8 E4M3
    Returns: [n] f32 tensor

    Format: 1 sign, 4 exponent, 3 mantissa bits. Bias=7.
    """
    result = torch.zeros(data.shape[0])
    for i in range(data.shape[0]):
        bits = int(data[i].item())
        sign = -1.0 if (bits >> 7) else 1.0
        exp = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exp == 0:
            # Subnormal
            result[i] = sign * (mantissa / 8.0) * (2.0 ** -6)
        elif exp == 15:
            # NaN (E4M3 has no inf)
            result[i] = float('nan') if mantissa != 0 else sign * 448.0
        else:
            result[i] = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7))
    return result


def fp8_e5m2_dequant(data: torch.Tensor) -> torch.Tensor:
    """
    FP8 E5M2 dequantization.
    data: [n] tensor of uint8 values representing FP8 E5M2
    Returns: [n] f32 tensor

    Format: 1 sign, 5 exponent, 2 mantissa bits. Bias=15.
    """
    result = torch.zeros(data.shape[0])
    for i in range(data.shape[0]):
        bits = int(data[i].item())
        sign = -1.0 if (bits >> 7) else 1.0
        exp = (bits >> 2) & 0x1F
        mantissa = bits & 0x3
        if exp == 0:
            # Subnormal
            result[i] = sign * (mantissa / 4.0) * (2.0 ** -14)
        elif exp == 31:
            result[i] = float('inf') * sign if mantissa == 0 else float('nan')
        else:
            result[i] = sign * (1.0 + mantissa / 4.0) * (2.0 ** (exp - 15))
    return result


def nvfp4_dequant(data: torch.Tensor, scales: torch.Tensor, tensor_scale: float = 1.0) -> torch.Tensor:
    """
    NVFP4 dequantization with hierarchical scaling.
    data: [n] tensor of uint8 values (each byte = 2 nibbles, 2 E2M1 values)
    scales: [n//16] tensor of FP8 E4M3 block scales (1 per 16 elements)
    tensor_scale: global tensor scale (f32)
    Returns: [n*2] f32 tensor (each input byte expands to 2 values)

    E2M1 nibble format: 1 sign, 2 exponent, 1 mantissa.
    """
    # E2M1 lookup table (4 bits → f32)
    e2m1_table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    n_bytes = data.shape[0]
    result = torch.zeros(n_bytes * 2)
    fp8_scales = fp8_e4m3_dequant(scales)

    for i in range(n_bytes):
        byte_val = int(data[i].item())
        lo = byte_val & 0xF
        hi = (byte_val >> 4) & 0xF
        block_idx = i // 8  # 16 elements = 8 bytes per block
        block_scale = fp8_scales[block_idx].item() if block_idx < len(fp8_scales) else 1.0
        result[2*i] = e2m1_table[lo] * block_scale * tensor_scale
        result[2*i+1] = e2m1_table[hi] * block_scale * tensor_scale

    return result


# ============================================================================
# Paged Attention
# ============================================================================


def paged_sdpa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    block_size: int,
    scale: float = None,
) -> torch.Tensor:
    """
    Paged SDPA: attention with block-table indirection into a paged KV cache.

    q: [n_heads * head_dim] query vector
    k_cache: [n_blocks, block_size, n_kv_heads * head_dim] paged key cache
    v_cache: [n_blocks, block_size, n_kv_heads * head_dim] paged value cache
    block_table: [n_logical_blocks] maps logical block index → physical block index
    n_heads, n_kv_heads, head_dim: attention dimensions
    seq_len: actual sequence length (may not fill all blocks)
    block_size: tokens per cache block (from attention.zig, typically 64)
    scale: attention scale (default: 1/sqrt(head_dim))

    Returns: [n_heads * head_dim] attention output
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    kvd = n_kv_heads * head_dim
    hpg = n_heads // n_kv_heads
    output = torch.zeros(n_heads * head_dim)

    for h in range(n_heads):
        kvh = h // hpg
        q_head = q[h * head_dim : (h + 1) * head_dim]

        # Gather keys and values through block table
        scores = torch.zeros(seq_len)
        gathered_v = torch.zeros(seq_len, head_dim)

        for t in range(seq_len):
            logical_block = t // block_size
            block_offset = t % block_size
            physical_block = int(block_table[logical_block].item())

            k_vec = k_cache[physical_block, block_offset, kvh * head_dim : (kvh + 1) * head_dim]
            v_vec = v_cache[physical_block, block_offset, kvh * head_dim : (kvh + 1) * head_dim]

            scores[t] = torch.dot(q_head, k_vec) * scale
            gathered_v[t] = v_vec

        weights = F.softmax(scores, dim=0)
        output[h * head_dim : (h + 1) * head_dim] = (weights.unsqueeze(1) * gathered_v).sum(0)

    return output
