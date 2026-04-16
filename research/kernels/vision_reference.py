#!/usr/bin/env python3
"""Generate reference outputs for the Gemma 4 SigLIP-2 vision encoder.

Loads the mmproj GGUF file, implements the forward pass step-by-step,
and dumps intermediate values as raw f32 .bin files for comparison
against the Zig implementation.
"""

import os
import sys
import struct
import numpy as np
from gguf import GGUFReader

# ── Config ────────────────────────────────────────────────────────
IMAGE_SIZE = 224
PATCH_SIZE = 16
N_PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE  # 14
N_PATCHES = N_PATCHES_SIDE * N_PATCHES_SIDE  # 196
EMBD_DIM = 1152
FFN_DIM = 4304
N_HEADS = 16
HEAD_DIM = EMBD_DIM // N_HEADS  # 72
N_BLOCKS = 27
PROJECTION_DIM = 2816
NORM_EPS = 1e-6

GGUF_PATH = os.path.join(os.path.dirname(__file__), "../../models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/mmproj-gemma-4-26B-A4B-it-BF16.gguf")
OUT_DIR = "vision_ref_dumps"


def bf16_to_f32(raw_bytes: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert BF16 raw uint8 bytes to float32 numpy array.

    BF16 is stored as 2 bytes (uint16). To convert to float32,
    we zero-extend by shifting left 16 bits.
    """
    # raw_bytes is uint8 with shape (..., n*2) where last dim is byte-packed
    flat = raw_bytes.reshape(-1)
    n_elements = len(flat) // 2
    # Interpret as uint16 (little-endian)
    u16 = np.frombuffer(flat.tobytes(), dtype=np.uint16)
    # Zero-extend to uint32 and shift left 16
    u32 = u16.astype(np.uint32) << 16
    # Reinterpret as float32
    f32 = u32.view(np.float32)
    return f32.reshape(shape)


class GGUFWeights:
    """Load and provide access to GGUF tensor weights."""

    def __init__(self, path: str):
        self.reader = GGUFReader(path)
        self._tensors = {}
        for t in self.reader.tensors:
            self._tensors[t.name] = t

    def get(self, name: str) -> np.ndarray:
        """Get a tensor as float32 numpy array.

        GGUF dimensions are [dim0, dim1, ...] where dim0 is fastest-varying.
        The gguf-py library returns data with reversed shape (numpy C-order).

        For a 2D weight W with GGUF shape [dim0, dim1]:
          - dim0 = input dim (k), dim1 = output dim (n)
          - data is returned with shape (dim1, dim0) for f32
          - or (dim1, dim0*2) for bf16 raw bytes
          - We reshape to (n, k) = (dim1, dim0) for standard matmul

        For GEMV: y = W @ x means y[i] = sum_j W[i,j] * x[j]
        So W has shape (n, k) where n = output, k = input.
        """
        t = self._tensors[name]
        gguf_shape = [int(d) for d in t.shape]  # [dim0, dim1, ...]

        if t.data.dtype == np.float32:
            # F32: data is already shaped correctly (reversed from GGUF dims)
            return t.data.copy()
        else:
            # BF16 (dtype=30): data is uint8 raw bytes
            # Reversed shape from GGUF: (dim_last, ..., dim0*2)
            reversed_shape = list(gguf_shape[::-1])
            return bf16_to_f32(t.data, reversed_shape)

    def has(self, name: str) -> bool:
        return name in self._tensors

    def info(self, name: str):
        t = self._tensors[name]
        return [int(d) for d in t.shape], t.tensor_type


def save_f32(path: str, arr: np.ndarray):
    """Save array as raw f32 little-endian binary."""
    arr = arr.astype(np.float32)
    arr.tofile(path)
    print(f"  Saved {path}: shape={arr.shape}, {arr.nbytes} bytes, "
          f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    """RMS normalization: x / rms(x) * weight.

    x: (..., dim), weight: (dim,)
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading GGUF from {GGUF_PATH}...")
    W = GGUFWeights(GGUF_PATH)

    # ── 1. Create synthetic red image ─────────────────────────────
    # All pixels [255, 0, 0] -> channel-first, normalized to [0,1]
    # Channel-first: [3, 224, 224]
    image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    image[0, :, :] = 1.0  # R channel = 255/255 = 1.0
    # G and B channels = 0.0

    print(f"Image: shape={image.shape}, R mean={image[0].mean():.2f}")

    # ── 2. Patch embedding ────────────────────────────────────────
    # v.patch_embd.weight: GGUF shape [16, 16, 3, 1152]
    # numpy data shape: (1152, 3, 16, 16) = (out_channels, in_channels, kh, kw)
    # This is a conv2d kernel.
    patch_w = W.get("v.patch_embd.weight")  # (1152, 3, 16, 16)
    print(f"Patch embedding weight: {patch_w.shape}")

    # Unfold image into patches and apply conv2d
    # patches: (n_patches, 3, patch_size, patch_size)
    patches = np.zeros((N_PATCHES, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for py in range(N_PATCHES_SIDE):
        for px in range(N_PATCHES_SIDE):
            patch_idx = py * N_PATCHES_SIDE + px
            y0, x0 = py * PATCH_SIZE, px * PATCH_SIZE
            patches[patch_idx] = image[:, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

    # Flatten patches: (n_patches, 3*16*16) = (196, 768)
    patches_flat = patches.reshape(N_PATCHES, -1)  # (196, 768)
    # Reshape weight to (1152, 768)
    patch_w_flat = patch_w.reshape(EMBD_DIM, -1)  # (1152, 768)

    # Patch embedding: x = patches @ patch_w^T
    # patches_flat: (196, 768), patch_w_flat: (1152, 768)
    # result: (196, 1152)
    x = patches_flat @ patch_w_flat.T  # (196, 1152)

    print(f"After patch embedding: shape={x.shape}")
    save_f32(os.path.join(OUT_DIR, "01_after_patch_embed.bin"), x)

    # ── 3. Input standardization ──────────────────────────────────
    # x = scale * x + bias
    std_scale = W.get("v.std_scale")  # (1152,)
    std_bias = W.get("v.std_bias")    # (1152,)
    x = std_scale * x + std_bias

    # ── 4. 2D Position embedding ──────────────────────────────────
    # v.position_embd.weight: GGUF shape [1152, 10240, 2]
    # numpy data shape: (2, 10240, 1152)
    # dim2=0 is row embedding, dim2=1 is col embedding
    pos_w = W.get("v.position_embd.weight")  # (2, 10240, 1152)
    print(f"Position embedding weight: {pos_w.shape}")

    for py in range(N_PATCHES_SIDE):
        for px in range(N_PATCHES_SIDE):
            patch_idx = py * N_PATCHES_SIDE + px
            # row embedding: pos_w[0, py, :] (dim=1152)
            # col embedding: pos_w[1, px, :] (dim=1152)
            x[patch_idx] += pos_w[0, py, :] + pos_w[1, px, :]

    print(f"After standardization + position embedding: shape={x.shape}")
    save_f32(os.path.join(OUT_DIR, "02_after_std_pos.bin"), x)

    # ── 5. ViT transformer blocks ────────────────────────────────
    hidden = x.copy()  # (196, 1152) -- residual stream

    for bi in range(N_BLOCKS):
        # ── 5a. Pre-attention RMSNorm ────────────────────────────
        ln1_w = W.get(f"v.blk.{bi}.ln1.weight")  # (1152,)
        normed = rms_norm(hidden, ln1_w)

        # ── 5b. Q/K/V projections ───────────────────────────────
        # attn_q.weight: numpy shape (1152, 1152) = (n_out, n_in)
        # GEMV: Q = normed @ W^T
        q_w = W.get(f"v.blk.{bi}.attn_q.weight")  # (1152, 1152)
        k_w = W.get(f"v.blk.{bi}.attn_k.weight")  # (1152, 1152)
        v_w = W.get(f"v.blk.{bi}.attn_v.weight")  # (1152, 1152)

        Q = normed @ q_w.T  # (196, 1152)
        K = normed @ k_w.T  # (196, 1152)
        V = normed @ v_w.T  # (196, 1152)

        # ── 5c. QK RMSNorm (per-head) ──────────────────────────
        q_norm_w = W.get(f"v.blk.{bi}.attn_q_norm.weight")  # (72,)
        k_norm_w = W.get(f"v.blk.{bi}.attn_k_norm.weight")  # (72,)

        Q_heads = Q.reshape(N_PATCHES, N_HEADS, HEAD_DIM)  # (196, 16, 72)
        K_heads = K.reshape(N_PATCHES, N_HEADS, HEAD_DIM)  # (196, 16, 72)

        Q_heads = rms_norm(Q_heads, q_norm_w)
        K_heads = rms_norm(K_heads, k_norm_w)

        # ── 5d. Full bidirectional attention ────────────────────
        # scale = 1/sqrt(head_dim)
        scale = 1.0 / np.sqrt(HEAD_DIM)

        # Q_heads: (196, 16, 72), K_heads: (196, 16, 72)
        # Transpose to (16, 196, 72) for batched matmul
        Qt = Q_heads.transpose(1, 0, 2)  # (16, 196, 72)
        Kt = K_heads.transpose(1, 0, 2)  # (16, 196, 72)
        Vt = V.reshape(N_PATCHES, N_HEADS, HEAD_DIM).transpose(1, 0, 2)  # (16, 196, 72)

        # scores: (16, 196, 196) = Q @ K^T * scale
        scores = np.matmul(Qt, Kt.transpose(0, 2, 1)) * scale
        attn_weights = softmax(scores, axis=-1)

        # attn output: (16, 196, 72) = weights @ V
        attn_out = np.matmul(attn_weights, Vt)  # (16, 196, 72)

        # Transpose back to (196, 16, 72) and reshape to (196, 1152)
        attn_out = attn_out.transpose(1, 0, 2).reshape(N_PATCHES, EMBD_DIM)

        # ── 5e. Output projection ──────────────────────────────
        out_w = W.get(f"v.blk.{bi}.attn_out.weight")  # (1152, 1152)
        attn_proj = attn_out @ out_w.T  # (196, 1152)

        # ── 5f. Post-attention RMSNorm ─────────────────────────
        post_attn_norm_w = W.get(f"v.blk.{bi}.attn_post_norm.weight")  # (1152,)
        attn_proj = rms_norm(attn_proj, post_attn_norm_w)

        # ── 5g. Residual ───────────────────────────────────────
        hidden = hidden + attn_proj

        # ── 5h. Pre-FFN RMSNorm ────────────────────────────────
        ln2_w = W.get(f"v.blk.{bi}.ln2.weight")  # (1152,)
        ffn_input = rms_norm(hidden, ln2_w)

        # ── 5i. SwiGLU FFN ─────────────────────────────────────
        # gate, up: (1152 -> 4304), down: (4304 -> 1152)
        gate_w = W.get(f"v.blk.{bi}.ffn_gate.weight")  # (4304, 1152)
        up_w = W.get(f"v.blk.{bi}.ffn_up.weight")      # (4304, 1152)
        down_w = W.get(f"v.blk.{bi}.ffn_down.weight")  # (1152, 4304)

        gate = ffn_input @ gate_w.T  # (196, 4304)
        up = ffn_input @ up_w.T      # (196, 4304)
        ffn_out = silu(gate) * up     # SwiGLU
        ffn_out = ffn_out @ down_w.T  # (196, 1152)

        # ── 5j. Post-FFN RMSNorm ──────────────────────────────
        post_ffn_norm_w = W.get(f"v.blk.{bi}.ffn_post_norm.weight")  # (1152,)
        ffn_out = rms_norm(ffn_out, post_ffn_norm_w)

        # ── 5k. Residual ──────────────────────────────────────
        hidden = hidden + ffn_out

        if bi == 0:
            save_f32(os.path.join(OUT_DIR, "03_after_block_00.bin"), hidden)
        if bi == N_BLOCKS - 1:
            save_f32(os.path.join(OUT_DIR, "04_after_block_26.bin"), hidden)

        if bi % 5 == 0:
            print(f"  Block {bi}: hidden min={hidden.min():.4f} max={hidden.max():.4f} mean={hidden.mean():.6f}")

    # ── 6. Pre-projection RMSNorm (no learnable weights) ─────────
    # HuggingFace Gemma4MultimodalEmbedder applies:
    #   embs_normed = rms_norm(hidden, with_scale=False)
    #   output = linear_projection(embs_normed)
    # The VisionPooler's sqrt(hidden_size) scaling gets absorbed by the RMSNorm,
    # so we only need the unweighted RMSNorm here.
    def rms_norm_no_weight(x, eps=NORM_EPS):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms

    hidden = rms_norm_no_weight(hidden)

    # ── 7. Final projection ───────────────────────────────────────
    # mm.input_projection.weight: GGUF shape [1152, 2816]
    # numpy data shape: (2816, 1152) = (output_dim, input_dim)
    proj_w = W.get("mm.input_projection.weight")  # (2816, 1152)
    print(f"Projection weight: {proj_w.shape}")
    output = hidden @ proj_w.T  # (196, 2816)

    print(f"Final output: shape={output.shape}")
    save_f32(os.path.join(OUT_DIR, "05_after_projection.bin"), output)
    save_f32(os.path.join(OUT_DIR, "vision_reference_output.bin"), output)

    # Also save first 10 values of each stage for quick inspection
    print("\n=== Quick reference values (first 10 per stage) ===")
    for fname in ["01_after_patch_embed.bin", "02_after_std_pos.bin",
                   "03_after_block_00.bin", "04_after_block_26.bin",
                   "05_after_projection.bin"]:
        path = os.path.join(OUT_DIR, fname)
        data = np.fromfile(path, dtype=np.float32)
        print(f"{fname}: {data[:10]}")

    print("\nDone! Reference outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
