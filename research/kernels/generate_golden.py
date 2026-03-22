#!/usr/bin/env python3
"""
Generate golden test data for Agave's Zig kernel tests.

Produces .bin files (raw f32 little-endian) that Zig tests can @embedFile
and compare against. Supports generating data for individual kernels or all.

Usage:
    uv run generate_golden.py              # Generate all
    uv run generate_golden.py sdpa         # Only SDPA
    uv run generate_golden.py gemv rope    # GEMV + RoPE
    uv run generate_golden.py --list       # List available generators
"""

import os
import sys
import numpy as np
import torch

from reference import (
    rms_norm, silu, gelu, rope, sdpa, sdpa_online, gemv_f32, swiglu_fused,
    add, mul, sigmoid, softplus, softmax, l2_norm,
    embedding_lookup, conv1d_causal, deltanet_recurrence,
    moe_routing_topk, moe_routing_sigmoid,
    fp8_e4m3_dequant, fp8_e5m2_dequant, nvfp4_dequant,
    paged_sdpa,
)

GOLDEN_DIR = "golden"


def save(name: str, tensor: torch.Tensor):
    """Save tensor as raw f32 little-endian binary."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, f"{name}.bin")
    tensor.float().contiguous().numpy().tofile(path)
    print(f"  {path}: {tensor.shape} ({os.path.getsize(path)} bytes)")


# ── Per-kernel generators ─────────────────────────────────────────
# Each generator is registered in GENERATORS and can be called individually.

GENERATORS: dict[str, callable] = {}


def generator(name: str):
    """Decorator to register a golden data generator."""
    def wrap(fn):
        GENERATORS[name] = fn
        return fn
    return wrap


@generator("sdpa")
def generate_sdpa():
    """SDPA golden tests with Gemma3 1B and GQA dimensions."""
    print("Generating SDPA golden tests...")
    torch.manual_seed(42)

    # Gemma3 1B: n_heads=4, n_kv_heads=1, head_dim=256
    nh, nkv, hd = 4, 1, 256
    for seq_len in [1, 8, 64, 256]:
        q = torch.randn(nh * hd)
        keys = torch.randn(seq_len * nkv * hd)
        values = torch.randn(seq_len * nkv * hd)

        out_naive = sdpa(q, keys, values, nh, nkv, hd)
        out_online = sdpa_online(q, keys, values, nh, nkv, hd)

        tag = f"sdpa_nh{nh}_nkv{nkv}_hd{hd}_sl{seq_len}"
        save(f"{tag}_q", q)
        save(f"{tag}_keys", keys)
        save(f"{tag}_values", values)
        save(f"{tag}_out_naive", out_naive)
        save(f"{tag}_out_online", out_online)

        diff = (out_naive - out_online).abs().max().item()
        print(
            f"  naive vs online max diff: {diff:.2e} {'OK' if diff < 1e-5 else 'MISMATCH!'}"
        )

    # GQA variant: n_heads=20, n_kv_heads=5, head_dim=128
    nh, nkv, hd = 20, 5, 128
    for seq_len in [1, 32]:
        q = torch.randn(nh * hd)
        keys = torch.randn(seq_len * nkv * hd)
        values = torch.randn(seq_len * nkv * hd)
        out = sdpa(q, keys, values, nh, nkv, hd)
        tag = f"sdpa_nh{nh}_nkv{nkv}_hd{hd}_sl{seq_len}"
        save(f"{tag}_q", q)
        save(f"{tag}_keys", keys)
        save(f"{tag}_values", values)
        save(f"{tag}_out", out)


@generator("rms_norm")
def generate_rms_norm():
    """RMSNorm golden tests."""
    print("Generating RMSNorm golden tests...")
    torch.manual_seed(42)

    for n in [128, 1152, 2560, 3136]:
        x = torch.randn(n)
        w = torch.randn(n)
        out = rms_norm(x, w, eps=1e-5)
        save(f"rms_norm_n{n}_x", x)
        save(f"rms_norm_n{n}_w", w)
        save(f"rms_norm_n{n}_out", out)


@generator("silu")
def generate_silu():
    """SiLU golden tests."""
    print("Generating SiLU golden tests...")
    torch.manual_seed(42)

    for n in [128, 1152, 6912]:
        x = torch.randn(n)
        save(f"silu_n{n}_x", x)
        save(f"silu_n{n}_out", silu(x))


@generator("gelu")
def generate_gelu():
    """GELU golden tests."""
    print("Generating GELU golden tests...")
    torch.manual_seed(42)

    for n in [128, 1152, 6912]:
        x = torch.randn(n)
        save(f"gelu_n{n}_x", x)
        save(f"gelu_n{n}_out", gelu(x))


@generator("rope")
def generate_rope():
    """RoPE golden tests."""
    print("Generating RoPE golden tests...")
    torch.manual_seed(42)

    # Gemma3: 4 heads, head_dim=256, rope_dim=256, theta=10000
    nh, hd, rd = 4, 256, 256
    q = torch.randn(nh * hd)
    k = torch.randn(nh * hd)
    for pos in [0, 1, 10, 100, 1000]:
        q_out, k_out = rope(q, k, pos, hd, rd, theta=10000.0)
        save(f"rope_nh{nh}_hd{hd}_pos{pos}_q_in", q)
        save(f"rope_nh{nh}_hd{hd}_pos{pos}_k_in", k)
        save(f"rope_nh{nh}_hd{hd}_pos{pos}_q_out", q_out)
        save(f"rope_nh{nh}_hd{hd}_pos{pos}_k_out", k_out)


@generator("gemv")
def generate_gemv():
    """GEMV golden tests."""
    print("Generating GEMV golden tests...")
    torch.manual_seed(42)

    for n, k in [(128, 128), (1152, 1152), (6912, 1152), (262144, 1152)]:
        x = torch.randn(k)
        w = torch.randn(n, k)
        y = gemv_f32(x, w)
        save(f"gemv_n{n}_k{k}_x", x)
        save(f"gemv_n{n}_k{k}_w", w)
        save(f"gemv_n{n}_k{k}_y", y)


@generator("swiglu")
def generate_swiglu():
    """Fused SwiGLU golden tests."""
    print("Generating SwiGLU golden tests...")
    torch.manual_seed(42)

    for ff, embd in [(6912, 1152), (12288, 3584)]:
        x = torch.randn(embd)
        w_gate = torch.randn(ff, embd)
        w_up = torch.randn(ff, embd)
        out = swiglu_fused(x, w_gate, w_up)
        save(f"swiglu_ff{ff}_e{embd}_x", x)
        save(f"swiglu_ff{ff}_e{embd}_gate", w_gate)
        save(f"swiglu_ff{ff}_e{embd}_up", w_up)
        save(f"swiglu_ff{ff}_e{embd}_out", out)


@generator("add")
def generate_add():
    """Elementwise add golden tests."""
    print("Generating add golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152, 4096]:
        a = torch.randn(n)
        b = torch.randn(n)
        save(f"add_n{n}_a", a)
        save(f"add_n{n}_b", b)
        save(f"add_n{n}_out", add(a, b))


@generator("mul")
def generate_mul():
    """Elementwise mul golden tests."""
    print("Generating mul golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152, 4096]:
        a = torch.randn(n)
        b = torch.randn(n)
        save(f"mul_n{n}_a", a)
        save(f"mul_n{n}_b", b)
        save(f"mul_n{n}_out", mul(a, b))


@generator("sigmoid")
def generate_sigmoid():
    """Sigmoid golden tests."""
    print("Generating sigmoid golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152]:
        x = torch.randn(n)
        save(f"sigmoid_n{n}_x", x)
        save(f"sigmoid_n{n}_out", sigmoid(x))


@generator("softplus")
def generate_softplus():
    """Softplus golden tests."""
    print("Generating softplus golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152]:
        x = torch.randn(n)
        save(f"softplus_n{n}_x", x)
        save(f"softplus_n{n}_out", softplus(x))


@generator("softmax")
def generate_softmax():
    """Softmax golden tests."""
    print("Generating softmax golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152, 4096]:
        x = torch.randn(n)
        save(f"softmax_n{n}_x", x)
        save(f"softmax_n{n}_out", softmax(x))


@generator("l2_norm")
def generate_l2_norm():
    """L2 norm golden tests."""
    print("Generating l2_norm golden tests...")
    torch.manual_seed(42)
    for n in [128, 1152, 4096]:
        x = torch.randn(n)
        save(f"l2_norm_n{n}_x", x)
        save(f"l2_norm_n{n}_out", l2_norm(x))


@generator("embedding")
def generate_embedding():
    """Embedding lookup golden tests."""
    print("Generating embedding golden tests...")
    torch.manual_seed(42)
    # Small table: 100 entries, 128 dims
    table_small = torch.randn(100, 128)
    for idx in [0, 42, 99]:
        save(f"embedding_100x128_idx{idx}_table", table_small)
        save(f"embedding_100x128_idx{idx}_out", embedding_lookup(table_small, idx))
    # Gemma3 dims: 500 entries, 1152 dims
    table_gemma = torch.randn(500, 1152)
    for idx in [0, 123, 499]:
        save(f"embedding_500x1152_idx{idx}_table", table_gemma)
        save(f"embedding_500x1152_idx{idx}_out", embedding_lookup(table_gemma, idx))


@generator("conv1d")
def generate_conv1d():
    """Causal conv1d golden tests."""
    print("Generating conv1d golden tests...")
    torch.manual_seed(42)
    # Qwen3.5 dims: d_conv=4, n_ch=128
    for d_conv, n_ch in [(4, 128), (4, 64)]:
        x = torch.randn(n_ch)
        weight = torch.randn(d_conv, n_ch)
        state = torch.randn(d_conv, n_ch)
        out, new_state = conv1d_causal(x, weight, state)
        tag = f"conv1d_dc{d_conv}_ch{n_ch}"
        save(f"{tag}_x", x)
        save(f"{tag}_weight", weight)
        save(f"{tag}_state", state)
        save(f"{tag}_out", out)
        save(f"{tag}_new_state", new_state)


@generator("deltanet")
def generate_deltanet():
    """DeltaNet recurrence golden tests."""
    print("Generating deltanet golden tests...")
    torch.manual_seed(42)
    for hd in [128, 64]:
        q = torch.randn(hd)
        k = torch.randn(hd)
        v = torch.randn(hd)
        beta = torch.tensor(0.1)
        state = torch.randn(hd, hd)
        out, new_state = deltanet_recurrence(q, k, v, beta, state)
        tag = f"deltanet_hd{hd}"
        save(f"{tag}_q", q)
        save(f"{tag}_k", k)
        save(f"{tag}_v", v)
        save(f"{tag}_beta", beta)
        save(f"{tag}_state", state)
        save(f"{tag}_out", out)
        save(f"{tag}_new_state", new_state)


@generator("moe_routing")
def generate_moe_routing():
    """MoE routing golden tests."""
    print("Generating MoE routing golden tests...")
    torch.manual_seed(42)
    # Top-k routing (GPT-OSS style: 128 experts, top-6)
    logits = torch.randn(128)
    idx, weights = moe_routing_topk(logits, 6)
    save("moe_topk_128e_k6_logits", logits)
    save("moe_topk_128e_k6_idx", idx.float())
    save("moe_topk_128e_k6_weights", weights)
    # Sigmoid routing (GLM4 style: 40 experts)
    logits_sig = torch.randn(40)
    save("moe_sigmoid_40e_logits", logits_sig)
    save("moe_sigmoid_40e_out", moe_routing_sigmoid(logits_sig))


@generator("fp8")
def generate_fp8():
    """FP8 dequant golden tests."""
    print("Generating FP8 golden tests...")
    torch.manual_seed(42)
    # E4M3: test a range of representative values
    data_e4m3 = torch.tensor([0, 1, 7, 8, 56, 120, 126, 127, 128, 255], dtype=torch.uint8)
    save("fp8_e4m3_data", data_e4m3.float())
    save("fp8_e4m3_out", fp8_e4m3_dequant(data_e4m3.float()))
    # E5M2: test a range
    data_e5m2 = torch.tensor([0, 1, 3, 4, 60, 123, 124, 128, 252], dtype=torch.uint8)
    save("fp8_e5m2_data", data_e5m2.float())
    save("fp8_e5m2_out", fp8_e5m2_dequant(data_e5m2.float()))


@generator("nvfp4")
def generate_nvfp4():
    """NVFP4 dequant golden tests."""
    print("Generating NVFP4 golden tests...")
    torch.manual_seed(42)
    # 32 bytes = 64 elements, 4 blocks of 16
    data = torch.randint(0, 256, (32,)).float()
    scales = torch.tensor([60, 62, 58, 64], dtype=torch.uint8).float()  # FP8 E4M3 scale values
    out = nvfp4_dequant(data, scales, tensor_scale=1.0)
    save("nvfp4_32b_data", data)
    save("nvfp4_32b_scales", scales)
    save("nvfp4_32b_out", out)


@generator("paged_sdpa")
def generate_paged_sdpa():
    """Paged SDPA golden tests."""
    print("Generating paged SDPA golden tests...")
    torch.manual_seed(42)
    # Small test: 2 heads, 2 KV heads, head_dim=64, seq_len=128, block_size=64
    nh, nkv, hd = 2, 2, 64
    seq_len = 128
    block_size = 64
    n_blocks = 4  # over-provisioned
    n_logical = (seq_len + block_size - 1) // block_size

    q = torch.randn(nh * hd)
    k_cache = torch.randn(n_blocks, block_size, nkv * hd)
    v_cache = torch.randn(n_blocks, block_size, nkv * hd)
    # Simple identity mapping: logical block i → physical block i
    block_table = torch.arange(n_logical).float()

    out = paged_sdpa(q, k_cache, v_cache, block_table, nh, nkv, hd, seq_len, block_size)
    save("paged_sdpa_nh2_nkv2_hd64_sl128_q", q)
    save("paged_sdpa_nh2_nkv2_hd64_sl128_k_cache", k_cache)
    save("paged_sdpa_nh2_nkv2_hd64_sl128_v_cache", v_cache)
    save("paged_sdpa_nh2_nkv2_hd64_sl128_block_table", block_table)
    save("paged_sdpa_nh2_nkv2_hd64_sl128_out", out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate golden test data")
    parser.add_argument(
        "kernels", nargs="*",
        help=f"Kernels to generate (default: all). Available: {', '.join(sorted(GENERATORS))}",
    )
    parser.add_argument("--list", action="store_true", help="List available generators")
    args = parser.parse_args()

    if args.list:
        print("Available golden data generators:")
        for name, fn in sorted(GENERATORS.items()):
            print(f"  {name:<16} {fn.__doc__.strip() if fn.__doc__ else ''}")
        return

    targets = args.kernels if args.kernels else list(GENERATORS.keys())

    # Validate
    for t in targets:
        if t not in GENERATORS:
            print(f"Unknown generator: {t!r}. Available: {', '.join(sorted(GENERATORS))}")
            sys.exit(1)

    print(f"Generating golden test data in {GOLDEN_DIR}/\n")
    for t in targets:
        GENERATORS[t]()

    print(f"\nDone. Files in {GOLDEN_DIR}/")


if __name__ == "__main__":
    main()
