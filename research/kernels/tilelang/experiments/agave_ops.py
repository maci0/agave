"""TileLang kernels mirroring Agave hot-path ops, targeting the HIP backend.

Agave is a pure-Zig inference engine with hand-written kernels per backend
(CUDA PTX, HIP amdgcn, Metal MSL, Vulkan SPIR-V, WebGPU WGSL, CPU). This
experiment asks: can TileLang generate competitive device kernels for the
same ops, and what would porting its output back into Zig-orchestrated
backends look like?

Ops chosen from src/backend/kernels/ (decode path, batch size 1):
  - rms_norm   : y = w * x / sqrt(mean(x^2) + eps)          (hidden 5120)
  - silu_mul   : y = silu(gate) * up                        (ff 17408)
  - gemv_bf16  : y[n] = sum_k W[n,k] * x[k]                 (4096x5120 class)
  - gemv_q8_0  : block-wise Q8_0 dequant-in-kernel GEMV     (agave's workhorse)

All kernels execute on AMD ROCm here (`target="hip"`); TileLang emits CUDA
for NVIDIA boxes with identical source semantics.

Run: .venv/bin/python experiments/agave_ops.py [--bench]
"""

# NOTE: intentionally NO `from __future__ import annotations` here.
# TileLang's eager-mode type-hint resolver evaluates annotation strings
# against globals + closure cells; symbols that appear ONLY inside a string
# annotation (e.g. a derived `n_blocks = k // 32` local) are not closure
# cells and raise NameError. Eager annotations capture everything naturally.
import argparse
import json

import torch
import tilelang
import tilelang.language as T


# ── Kernel definitions ────────────────────────────────────────────────────────


def make_rms_norm(n: int, eps: float = 1e-6):
    """y[i] = w[i] * x[i] * rsqrt(sum(x^2)/n + eps)."""

    @T.prim_func
    def rms_norm(X: T.Tensor((n,), "float32"), W: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=256) as _bx:
            sq = T.alloc_fragment((n,), "float32")
            tot = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(n):
                sq[i] = X[i] * X[i]
            T.reduce_sum(sq, tot, dim=0)
            scale = T.rsqrt(tot[0] / n + eps)
            for i in T.Parallel(n):
                Y[i] = W[i] * X[i] * scale

    return rms_norm


def make_silu_mul(n: int):
    """y[i] = gate[i] * sigmoid(gate[i]) * up[i]."""

    @T.prim_func
    def silu_mul(G: T.Tensor((n,), "float32"), U: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for i in T.Parallel(256):
                idx = bx * 256 + i
                g = G[idx]
                Y[idx] = g * (1.0 / (1.0 + T.exp(-g))) * U[idx]

    return silu_mul


def make_gemv_bf16(m: int, k: int, blk_m: int = 16, blk_k: int = 128):
    """y[m] = W[m,k] @ x[k]; bf16 inputs, f32 accumulate. m,k divisible by blocks."""

    @T.prim_func
    def gemv(W: T.Tensor((m, k), "bfloat16"), X: T.Tensor((k,), "bfloat16"), Y: T.Tensor((m,), "float32")):
        with T.Kernel(m // blk_m, threads=128) as bx:
            acc = T.alloc_fragment((blk_m,), "float32")
            red = T.alloc_fragment((blk_m,), "float32")
            xf = T.alloc_fragment((blk_k,), "float32")
            prod = T.alloc_fragment((blk_m, blk_k), "float32")
            T.clear(acc)
            for ko in T.serial(k // blk_k):
                for j in T.Parallel(blk_k):
                    xf[j] = T.Cast("float32", X[ko * blk_k + j])
                for i, j in T.Parallel(blk_m, blk_k):
                    prod[i, j] = T.Cast("float32", W[bx * blk_m + i, ko * blk_k + j]) * xf[j]
                T.reduce_sum(prod, red, dim=1)
                for i in T.Parallel(blk_m):
                    acc[i] += red[i]
            for i in T.Parallel(blk_m):
                Y[bx * blk_m + i] = acc[i]

    return gemv


def make_gemv_q8_0(m: int, k: int, block: int = 32, blk_m: int = 16, blk_kb: int = 4):
    """Dequant-in-kernel Q8_0 GEMV, matching gguf layout.

    Per row m and per 32-col block b: W[m, b*32+j] = Scales[m, b] * Qs[m, b*32+j].
    Dequantization happens inside the kernel exactly like agave's native
    gemv_q8_0 (no full-tensor pre-conversion pass).
    """
    n_blocks = k // block
    cols = blk_kb * block  # columns of the flattened weight row processed per iter

    @T.prim_func
    def gemv_q80(
        Scales: T.Tensor((m, n_blocks), "float16"),
        Qs: T.Tensor((m, k), "int8"),
        X: T.Tensor((k,), "bfloat16"),
        Y: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(m // blk_m, threads=128) as bx:
            acc = T.alloc_fragment((blk_m,), "float32")
            red = T.alloc_fragment((blk_m,), "float32")
            xf = T.alloc_fragment((cols,), "float32")
            prod = T.alloc_fragment((blk_m, cols), "float32")
            T.clear(acc)
            for co in T.serial(k // cols):
                for j in T.Parallel(cols):
                    xf[j] = T.Cast("float32", X[co * cols + j])
                for i, j in T.Parallel(blk_m, cols):
                    col = co * cols + j
                    d = T.Cast("float32", Scales[bx * blk_m + i, col // block])
                    wq = T.Cast("float32", Qs[bx * blk_m + i, col])
                    prod[i, j] = d * wq * xf[j]
                T.reduce_sum(prod, red, dim=1)
                for mi in T.Parallel(blk_m):
                    acc[mi] += red[mi]
            for mi in T.Parallel(blk_m):
                Y[bx * blk_m + mi] = acc[mi]

    return gemv_q80


# ── References & helpers ──────────────────────────────────────────────────────


def ref_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * x * torch.rsqrt(x.pow(2).mean() + eps)


def ref_silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(g) * u


def pack_q8_0(w: torch.Tensor, block: int = 32):
    """Quantize [m,k] float tensor to per-(row,block) fp16 scales + int8 payload."""
    m, k = w.shape
    assert k % block == 0
    wb = w.view(m, k // block, block)
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    d = (amax / 127.0).to(torch.float16)
    q = (wb / d.float()).round().clamp(-127, 127).to(torch.int8)
    return d.view(m, k // block), q.view(m, k)


DEV = "cuda"  # rocm torch exposes HIP through the cuda API


def bench(fn, iters: int = 300, warmup: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


# ── Harness ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also benchmark vs torch")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    ok = True

    # ── rms_norm (Qwen3.8 hidden size) ────────────────────────────
    n = 5120
    k_rms = tilelang.JITKernel(make_rms_norm(n), target="hip", out_idx=[-1])
    x = torch.randn(n, device=DEV, dtype=torch.float32)
    w = 1.0 + torch.randn(n, device=DEV, dtype=torch.float32)
    y = k_rms(x, w)
    ref = ref_rms_norm(x, w)
    err = (y - ref).abs().max().item()
    results["rms_norm"] = {"max_abs_err": err, "shape": [n]}
    ok &= err < 1e-3
    if args.bench:
        results["rms_norm"]["tilelang_us"] = bench(lambda: k_rms(x, w))
        results["rms_norm"]["torch_us"] = bench(lambda: ref_rms_norm(x, w))

    # ── silu_mul (Qwen3.8 ff size) ────────────────────────────────
    n_ff = 17408
    k_silu = tilelang.JITKernel(make_silu_mul(n_ff), target="hip", out_idx=[-1])
    g = torch.randn(n_ff, device=DEV, dtype=torch.float32)
    u = torch.randn(n_ff, device=DEV, dtype=torch.float32)
    y = k_silu(g, u)
    ref = ref_silu_mul(g, u)
    err = (y - ref).abs().max().item()
    results["silu_mul"] = {"max_abs_err": err, "shape": [n_ff]}
    ok &= err < 1e-3
    if args.bench:
        results["silu_mul"]["tilelang_us"] = bench(lambda: k_silu(g, u))
        results["silu_mul"]["torch_us"] = bench(lambda: ref_silu_mul(g, u))

    # ── gemv bf16 ────────────────────────────────────────────────
    m, kk = 4096, 5120
    k_gemv = tilelang.JITKernel(make_gemv_bf16(m, kk), target="hip", out_idx=[-1])
    wm = (torch.randn(m, kk, device=DEV) * 0.02).to(torch.bfloat16)
    xv = torch.randn(kk, device=DEV, dtype=torch.bfloat16)
    y = k_gemv(wm, xv).float()
    ref = wm.float() @ xv.float()
    scale = ref.abs().max().clamp_min(1e-9).item()
    err = (y - ref).abs().max().item() / scale
    results["gemv_bf16"] = {"rel_err": err, "shape": [m, kk]}
    ok &= err < 5e-3
    if args.bench:
        results["gemv_bf16"]["tilelang_us"] = bench(lambda: k_gemv(wm, xv))
        results["gemv_bf16"]["torch_us"] = bench(lambda: wm.float() @ xv.float())

    # ── gemv q8_0 (dequant in kernel) ────────────────────────────
    m, kk = 4096, 5120
    k_q80 = tilelang.JITKernel(make_gemv_q8_0(m, kk), target="hip", out_idx=[-1])
    wf = torch.randn(m, kk, device=DEV) * 0.05
    d, q = pack_q8_0(wf)
    xv = torch.randn(kk, device=DEV, dtype=torch.bfloat16)
    y = k_q80(d, q, xv).float()
    deq = d.float().repeat_interleave(32, dim=1) * q.float()
    ref = deq @ xv.float()
    denom = ref.abs().max().clamp_min(1e-9).item()
    err = (y - ref).abs().max().item() / denom
    results["gemv_q8_0"] = {"rel_err_vs_dequant_ref": err, "shape": [m, kk]}
    ok &= err < 5e-3
    if args.bench:
        results["gemv_q8_0"]["tilelang_us"] = bench(lambda: k_q80(d, q, xv))
        results["gemv_q8_0"]["torch_dequant_gemm_us"] = bench(lambda: deq @ xv.float())

    print(json.dumps(results, indent=2))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("VALIDATION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
