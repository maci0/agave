#!/usr/bin/env python3
"""Gemma 3 layer-0 reference implementation for debugging.

Reads a GGUF file, dequantizes weights, and runs the BOS token through
embedding + layer 0 (attention + FFN), printing intermediate values at
each step. Compare output against agave's debug diagnostics to find
where values diverge.

Usage:
    python3 research/gemma3_ref.py models/lmstudio-community/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q8_0.gguf
"""

import struct, sys, math, mmap
import numpy as np

# ── GGUF parser ──────────────────────────────────────────────────────

GGUF_TYPES = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32',
              6: 'f32', 7: 'bool', 8: 'str', 9: 'array', 10: 'u64', 11: 'i64', 12: 'f64'}
GGML_DTYPES = {0: 'f32', 1: 'f16', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1',
               8: 'q8_0', 9: 'q8_1', 10: 'q2_k', 11: 'q3_k', 12: 'q4_k',
               13: 'q5_k', 14: 'q6_k'}

def read_val(f, vtype):
    if vtype == 0: return struct.unpack('<B', f.read(1))[0]
    if vtype == 1: return struct.unpack('<b', f.read(1))[0]
    if vtype == 2: return struct.unpack('<H', f.read(2))[0]
    if vtype == 3: return struct.unpack('<h', f.read(2))[0]
    if vtype == 4: return struct.unpack('<I', f.read(4))[0]
    if vtype == 5: return struct.unpack('<i', f.read(4))[0]
    if vtype == 6: return struct.unpack('<f', f.read(4))[0]
    if vtype == 7: return struct.unpack('<?', f.read(1))[0]
    if vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8')
    if vtype == 9:
        atype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        return [read_val(f, atype) for _ in range(alen)]
    if vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    if vtype == 11: return struct.unpack('<q', f.read(8))[0]
    if vtype == 12: return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f"Unknown type {vtype}")

def parse_gguf(path):
    f = open(path, 'rb')
    magic = f.read(4)
    assert magic == b'GGUF', f"Not a GGUF file: {magic}"
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    # Read metadata
    meta = {}
    for _ in range(n_kv):
        klen = struct.unpack('<Q', f.read(8))[0]
        key = f.read(klen).decode('utf-8')
        vtype = struct.unpack('<I', f.read(4))[0]
        meta[key] = read_val(f, vtype)

    # Read tensor info
    tensors = {}
    for _ in range(n_tensors):
        nlen = struct.unpack('<Q', f.read(8))[0]
        name = f.read(nlen).decode('utf-8')
        ndim = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}

    # Alignment padding
    alignment = meta.get('general.alignment', 32)
    pos = f.tell()
    pad = (alignment - (pos % alignment)) % alignment
    data_start = pos + pad

    return f, meta, tensors, data_start

# ── Dequantization ───────────────────────────────────────────────────

def dequant_q8_0(raw_bytes, n_elements):
    """Dequantize Q8_0: 32 elements per block, 34 bytes (f16 scale + 32 int8)."""
    block_size = 32
    bpb = 34
    nb = (n_elements + block_size - 1) // block_size
    out = np.zeros(n_elements, dtype=np.float32)
    for b in range(nb):
        bp = b * bpb
        scale = np.frombuffer(raw_bytes[bp:bp+2], dtype=np.float16).astype(np.float32)[0]
        vals = np.frombuffer(raw_bytes[bp+2:bp+34], dtype=np.int8).astype(np.float32)
        start = b * block_size
        end = min(start + block_size, n_elements)
        out[start:end] = vals[:end-start] * scale
    return out

def dequant_q6_k(raw_bytes, n_elements):
    """Dequantize Q6_K: 256 elements per super-block, 210 bytes."""
    bs = 256
    bpb = 210
    nb = (n_elements + bs - 1) // bs
    out = np.zeros(n_elements, dtype=np.float32)
    for b in range(nb):
        bp = b * bpb
        d = np.frombuffer(raw_bytes[bp+208:bp+210], dtype=np.float16).astype(np.float32)[0]
        for chunk in range(2):
            ql = raw_bytes[bp + chunk*64 : bp + chunk*64 + 64]
            qh = raw_bytes[bp + 128 + chunk*32 : bp + 128 + chunk*32 + 32]
            sc = np.frombuffer(raw_bytes[bp + 192 + chunk*8 : bp + 192 + chunk*8 + 8], dtype=np.int8)
            base = b * bs + chunk * 128
            for l in range(32):
                isc = l // 16
                q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                gi0, gi1, gi2, gi3 = base+l, base+l+32, base+l+64, base+l+96
                if gi0 < n_elements: out[gi0] = d * float(sc[isc+0]) * q1
                if gi1 < n_elements: out[gi1] = d * float(sc[isc+2]) * q2
                if gi2 < n_elements: out[gi2] = d * float(sc[isc+4]) * q3
                if gi3 < n_elements: out[gi3] = d * float(sc[isc+6]) * q4
    return out

def read_f32(raw_bytes, n_elements):
    return np.frombuffer(raw_bytes[:n_elements*4], dtype=np.float32).copy()

# ── Tensor loading ───────────────────────────────────────────────────

def tensor_bytes(info):
    """Compute raw byte size of a tensor."""
    n = 1
    for d in info['dims']:
        n *= d
    dtype = info['dtype']
    if dtype == 0:   return n * 4        # f32
    if dtype == 1:   return n * 2        # f16
    if dtype == 8:   return (n // 32) * 34  # q8_0
    if dtype == 14:  return (n // 256) * 210  # q6_k
    raise ValueError(f"Unknown dtype {dtype} for byte size")

def load_tensor(mm, data_start, info, name=""):
    """Load and dequantize a tensor to f32."""
    n = 1
    for d in info['dims']:
        n *= d
    offset = data_start + info['offset']
    nbytes = tensor_bytes(info)
    raw = mm[offset:offset+nbytes]
    dtype = info['dtype']
    if dtype == 0:   return read_f32(raw, n)
    if dtype == 8:   return dequant_q8_0(raw, n)
    if dtype == 14:  return dequant_q6_k(raw, n)
    raise ValueError(f"Unsupported dtype {dtype} ({GGML_DTYPES.get(dtype, '?')}) for {name}")

# ── Gemma 3 ops ──────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * weight

def rope(x, pos, n_heads, head_dim, theta=1_000_000.0):
    """Split-complex RoPE: pairs [i, i+half] rotated together."""
    half = head_dim // 2
    out = x.copy()
    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            freq = 1.0 / (theta ** (2.0 * i / head_dim))
            angle = pos * freq
            c, s = math.cos(angle), math.sin(angle)
            r  = out[base + i]
            im = out[base + i + half]
            out[base + i]        = r * c - im * s
            out[base + i + half] = r * s + im * c
    return out

def gelu(x):
    """GELU tanh approximation (gelu_pytorch_tanh)."""
    c = 0.044715
    s2p = math.sqrt(2.0 / math.pi)
    inner = s2p * (x + c * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))

def dump(label, v):
    print(f"  {label:20s}: [{v.min():.4f}, {v.max():.4f}] rms={np.sqrt(np.mean(v*v)):.4f} [0]={v[0]:.6f} [1]={v[1]:.6f}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...")
    f, meta, tensors, data_start = parse_gguf(path)

    # mmap the file for tensor data access
    f.seek(0)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # Read model params
    arch = meta.get('general.architecture', 'gemma3')
    n_embd = meta.get(f'{arch}.embedding_length', 1152)
    n_head = meta.get(f'{arch}.attention.head_count', 4)
    n_head_kv = meta.get(f'{arch}.attention.head_count_kv', 1)
    head_dim = meta.get(f'{arch}.attention.key_length', 256)
    n_ff = meta.get(f'{arch}.feed_forward_length', 6912)
    n_layers = meta.get(f'{arch}.block_count', 26)
    rope_theta = meta.get(f'{arch}.rope.freq_base', 1_000_000.0)
    rms_eps = meta.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)
    vocab_size = tensors['token_embd.weight']['dims'][1]

    print(f"  arch={arch} n_embd={n_embd} nh={n_head} nkv={n_head_kv} hd={head_dim} "
          f"nff={n_ff} nl={n_layers} vocab={vocab_size} theta={rope_theta} eps={rms_eps}")

    # Attention scale: 1/sqrt(query_pre_attn_scalar or head_dim)
    scalar = head_dim  # default
    for key in ['query_pre_attn_scalar', f'{arch}.attention.query_pre_attn_scalar']:
        if key in meta:
            scalar = meta[key]
            break
    attn_scale = 1.0 / math.sqrt(scalar)
    embd_scale = math.sqrt(n_embd)
    print(f"  attn_scale={attn_scale:.6f} embd_scale={embd_scale:.4f}")

    # ── Load weights ─────────────────────────────────────────────────
    def load(name):
        return load_tensor(mm, data_start, tensors[name], name)

    print("Loading embedding...")
    emb_w = load('token_embd.weight').reshape(vocab_size, n_embd)

    print("Loading layer 0 weights...")
    attn_norm_w = load('blk.0.attn_norm.weight')
    q_w = load('blk.0.attn_q.weight').reshape(n_head * head_dim, n_embd)
    k_w = load('blk.0.attn_k.weight').reshape(n_head_kv * head_dim, n_embd)
    v_w = load('blk.0.attn_v.weight').reshape(n_head_kv * head_dim, n_embd)
    qn_w = load('blk.0.attn_q_norm.weight')
    kn_w = load('blk.0.attn_k_norm.weight')
    o_w = load('blk.0.attn_output.weight').reshape(n_embd, n_head * head_dim)
    post_attn_norm_w = load('blk.0.post_attention_norm.weight')

    ffn_norm_w = load('blk.0.ffn_norm.weight')
    gate_w = load('blk.0.ffn_gate.weight').reshape(n_ff, n_embd)
    up_w = load('blk.0.ffn_up.weight').reshape(n_ff, n_embd)
    down_w = load('blk.0.ffn_down.weight').reshape(n_embd, n_ff)
    post_ffw_norm_w = load('blk.0.post_ffw_norm.weight')

    # ── BOS token embedding ──────────────────────────────────────────
    bos_id = 2
    print(f"\n=== BOS token (id={bos_id}) ===")
    hidden = emb_w[bos_id].copy() * embd_scale
    dump("embedding", hidden)

    # ── Layer 0: Attention ───────────────────────────────────────────
    print("\n--- Layer 0 Attention ---")

    # Pre-norm
    hidden2 = rms_norm(hidden, attn_norm_w, rms_eps)
    dump("after_norm", hidden2)

    # QKV projections
    q = q_w @ hidden2
    k = k_w @ hidden2
    v = v_w @ hidden2
    dump("Q", q)
    dump("K", k)
    dump("V", v)

    # Per-head QK norms
    for h in range(n_head):
        s = h * head_dim
        q[s:s+head_dim] = rms_norm(q[s:s+head_dim], qn_w, rms_eps)
    for h in range(n_head_kv):
        s = h * head_dim
        k[s:s+head_dim] = rms_norm(k[s:s+head_dim], kn_w, rms_eps)
    dump("Q_normed", q)
    dump("K_normed", k)

    # RoPE (pos=0 → identity, but let's run it for correctness)
    q = rope(q, 0, n_head, head_dim, rope_theta)
    k = rope(k, 0, n_head_kv, head_dim, rope_theta)
    dump("Q_rope", q)

    # SDPA (seq_len=0 → single position → output = V, expanded for GQA)
    hpg = n_head // n_head_kv  # heads per group
    attn_out = np.zeros(n_head * head_dim, dtype=np.float32)
    for h in range(n_head):
        kvh = h // hpg
        attn_out[h*head_dim:(h+1)*head_dim] = v[kvh*head_dim:(kvh+1)*head_dim]
    dump("attn_out", attn_out)

    # Output projection
    out_proj = o_w @ attn_out
    dump("out_proj", out_proj)

    # Post-attention norm + residual
    out_normed = rms_norm(out_proj, post_attn_norm_w, rms_eps)
    hidden = hidden + out_normed
    dump("hidden_L0_attn", hidden)

    # ── Layer 0: FFN ─────────────────────────────────────────────────
    print("\n--- Layer 0 FFN ---")

    # Pre-FFN norm
    hidden2 = rms_norm(hidden, ffn_norm_w, rms_eps)
    dump("after_ffn_norm", hidden2)

    # Gate + Up projections
    ff_gate = gate_w @ hidden2
    ff_up = up_w @ hidden2

    # GELU * up (GeGLU)
    ff_gate = gelu(ff_gate) * ff_up
    dump("geglu_out", ff_gate)

    # Down projection
    ff_down = down_w @ ff_gate
    dump("ffn_down", ff_down)

    # Post-FFN norm + residual
    ff_normed = rms_norm(ff_down, post_ffw_norm_w, rms_eps)
    hidden = hidden + ff_normed
    dump("hidden_L0_full", hidden)

    # ── Run more layers ───────────────────────────────────────────────
    n_run = n_layers  # Run ALL layers
    for li in range(1, n_run):
        print(f"\n--- Layer {li} ---")
        # Load layer weights
        norm_w  = load(f'blk.{li}.attn_norm.weight')
        qw_l    = load(f'blk.{li}.attn_q.weight').reshape(n_head * head_dim, n_embd)
        kw_l    = load(f'blk.{li}.attn_k.weight').reshape(n_head_kv * head_dim, n_embd)
        vw_l    = load(f'blk.{li}.attn_v.weight').reshape(n_head_kv * head_dim, n_embd)
        qnw_l   = load(f'blk.{li}.attn_q_norm.weight')
        knw_l   = load(f'blk.{li}.attn_k_norm.weight')
        ow_l    = load(f'blk.{li}.attn_output.weight').reshape(n_embd, n_head * head_dim)
        panw_l  = load(f'blk.{li}.post_attention_norm.weight')
        fnw_l   = load(f'blk.{li}.ffn_norm.weight')
        gw_l    = load(f'blk.{li}.ffn_gate.weight').reshape(n_ff, n_embd)
        uw_l    = load(f'blk.{li}.ffn_up.weight').reshape(n_ff, n_embd)
        dw_l    = load(f'blk.{li}.ffn_down.weight').reshape(n_embd, n_ff)
        pfnw_l  = load(f'blk.{li}.post_ffw_norm.weight')

        # Attention
        h2 = rms_norm(hidden, norm_w, rms_eps)
        ql = qw_l @ h2
        kl = kw_l @ h2
        vl = vw_l @ h2
        for h in range(n_head):
            s = h * head_dim
            ql[s:s+head_dim] = rms_norm(ql[s:s+head_dim], qnw_l, rms_eps)
        for h in range(n_head_kv):
            s = h * head_dim
            kl[s:s+head_dim] = rms_norm(kl[s:s+head_dim], knw_l, rms_eps)
        ql = rope(ql, 0, n_head, head_dim, rope_theta)  # pos=0 for BOS (kv_seq_len=0 for all layers)
        kl = rope(kl, 0, n_head_kv, head_dim, rope_theta)

        # SDPA: at BOS forward, kv_seq_len = 0 for first call. But for subsequent layers
        # in the SAME forward pass, kv_seq_len is still 0 (incremented after all layers).
        # So each layer appends at pos=0. WAIT — that means ALL layers write to the same
        # KV cache position (0). For the BOS token, kv_seq_len=0 for ALL layers.
        # Each layer gets its OWN KV cache (per-layer blocks), so they don't conflict.
        # SDPA with seq_len=0 → output = V (expanded for GQA)
        ao = np.zeros(n_head * head_dim, dtype=np.float32)
        for h in range(n_head):
            kvh = h // hpg
            ao[h*head_dim:(h+1)*head_dim] = vl[kvh*head_dim:(kvh+1)*head_dim]

        op = ow_l @ ao
        op_n = rms_norm(op, panw_l, rms_eps)
        hidden = hidden + op_n

        # FFN
        h2 = rms_norm(hidden, fnw_l, rms_eps)
        fg = gw_l @ h2
        fu = uw_l @ h2
        fg = gelu(fg) * fu
        fd = dw_l @ fg
        fd_n = rms_norm(fd, pfnw_l, rms_eps)
        hidden = hidden + fd_n

        dump(f"hidden_L{li}", hidden)

    print(f"\n=== Final hidden after L{n_run-1}: rms={np.sqrt(np.mean(hidden*hidden)):.4f} ===")

    # Final norm + logits
    print("\nComputing final norm + logits...")
    out_norm_w = load('output_norm.weight')
    hidden = rms_norm(hidden, out_norm_w, rms_eps)
    dump("final_normed", hidden)

    # Logits = emb_w @ hidden (tied weights)
    logits = emb_w @ hidden  # [vocab_size]
    dump("logits", logits)

    # Top-5 predictions
    top5_idx = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 predictions:")
    for idx in top5_idx:
        print(f"  [{idx}] = {logits[idx]:.4f}")

    mm.close()
    f.close()

if __name__ == '__main__':
    main()
