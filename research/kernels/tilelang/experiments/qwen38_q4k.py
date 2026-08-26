"""TileLang Q4_K dequant-in-kernel GEMV for Qwen3.8-27B (Q4_K_M).

Reads the REAL quantized tensor straight out of the downloaded GGUF, uploads
only packed bytes to the GPU, and runs fused unpack+GEMV: same contract as
agave's native gemv_q4_k (no pre-conversion pass).

Q4_K layout (ggml block_q4_K, 144 B per 256-elem superblock):
  [0:2) fp16 d   [2:4) fp16 dmin   [4:16) 12 B of 6-bit sc/min pairs
  [16:144) nibbles in 4 groups of 32 B; group g holds 64 vals:
    low nibble -> val[g*64+l]      scaled by pair (2g)
    high nibble -> val[g*64+32+l]  scaled by pair (2g+1)
  Dequant: v = d * sc * q - dmin * mn.

Thread map: one output row per workgroup; 128 threads = 16 copies x 8 lanes;
a lane owns one 32-elem sub-block, copies split the superblock range; one
block-wide all-reduce produces the dot product.

Run: .venv/bin/python experiments/qwen38_q4k.py [--bench]
"""

import argparse
import json

import numpy as np
import torch
import tilelang
import tilelang.language as T
import gguf

GGUF_PATH = "/home/maci/.cache/models-e2e/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf"
DEV = "cuda"


# -- Reference unpacker (validated bit-exact vs gguf.dequantize) --------------


def dequant_q4_k_rows(raw, n_rows, k):
    sb = k // 256
    view = np.asarray(raw)[: n_rows * sb * 144].reshape(n_rows * sb, 144)
    d = view[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = view[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scl = np.asarray(view[:, 4:16])
    qs = np.asarray(view[:, 16:144]).reshape(n_rows * sb, 4, 32)

    def sm(j):
        if j < 4:
            return np.int32(scl[:, j] & 63), np.int32(scl[:, j + 4] & 63)
        return (np.int32((scl[:, j + 4] & 0xF) | ((scl[:, j - 4] >> 6) << 4)),
                np.int32((scl[:, j + 4] >> 4) | ((scl[:, j] >> 6) << 4)))

    out = np.empty((n_rows * sb, 256), dtype=np.float32)
    for g in range(4):
        s0, m0 = sm(2 * g)
        s1, m1 = sm(2 * g + 1)
        lo = np.float32(qs[:, g, :] & 0xF)
        hi = np.float32(qs[:, g, :] >> 4)
        out[:, g*64:g*64+32] = np.float32(d*s0)[:, None]*lo - np.float32(dmin*m0)[:, None]
        out[:, g*64+32:g*64+64] = np.float32(d*s1)[:, None]*hi - np.float32(dmin*m1)[:, None]
    return out.reshape(n_rows, k)


# -- Kernel -------------------------------------------------------------------

def make_gemv_q4k(m, k, lanes=8, threads=128):
    nblk = k // 256
    copies = threads // lanes
    spc = (nblk + copies - 1) // copies

    @T.prim_func
    def gemv(
        Wq: T.Tensor((m * nblk * 144,), "uint8"),    # packed gguf payload (flat)
        Wh: T.Tensor((m * nblk * 72,), "float16"),   # SAME bytes viewed as fp16
        X: T.Tensor((k,), "bfloat16"),
        Y: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(m, threads=threads) as bx:
            part = T.alloc_fragment((threads,), "float32")
            tot = T.alloc_fragment((1,), "float32")
            for t in T.Parallel(threads):
                cp = t // lanes
                ln = t % lanes
                part[t] = 0.0
                for si in T.serial(spc):
                    sb = cp * spc + si
                    if sb < nblk:
                        hb = (bx * nblk + sb) * 72
                        dv = T.Cast("float32", Wh[hb])
                        dmv = T.Cast("float32", Wh[hb + 1])
                        # get_scale_min_k4(lane) over the 12-byte scale table.
                        # q[t] lives at absolute byte BB + 4 + t.
                        BB = bx * nblk * 144 + sb * 144
                        q1 = Wq[BB + 4 + ln]
                        q2 = Wq[BB + 8 + ln]
                        scv = T.if_then_else(
                            ln < 4,
                            q1 & 63,
                            (q2 & 15) | ((Wq[BB + ln] >> 6) << 4),
                        )
                        mnv = T.if_then_else(
                            ln < 4,
                            q2 & 63,
                            (q2 >> 4) | ((q1 >> 6) << 4),
                        )
                        g = ln // 2
                        hbit = ln % 2
                        qb = BB + 16 + g * 32
                        xb = sb * 256 + ln * 32
                        sh = 4 * hbit
                        for e in T.serial(32):
                            u = Wq[qb + e]
                            nib = (u >> sh) & 15
                            wv = dv * T.Cast("float32", scv) * T.Cast("float32", nib) \
                                - dmv * T.Cast("float32", mnv)
                            part[t] += wv * T.Cast("float32", X[xb + e])
            T.reduce_sum(part, tot, dim=0)
            Y[bx] = tot[0]

    return gemv


def make_gemv_q4k_v2(m, k, lanes=8, threads=128):
    """v2: weight nibble bytes read as uint32 words (4x fewer loads); per-byte
    work emitted with constant shifts; x as fp16 scalars (cache-resident)."""
    nblk = k // 256
    copies = threads // lanes
    spc = (nblk + copies - 1) // copies

    @T.prim_func
    def gemv(
        Wq: T.Tensor((m * nblk * 144,), "uint8"),
        Wq32: T.Tensor((m * nblk * 36,), "uint32"),   # SAME bytes as uint32
        Wh: T.Tensor((m * nblk * 72,), "float16"),
        X: T.Tensor((k,), "float16"),
        Y: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(m, threads=threads) as bx:
            part = T.alloc_fragment((threads,), "float32")
            tot = T.alloc_fragment((1,), "float32")
            for t in T.Parallel(threads):
                cp = t // lanes
                ln = t % lanes
                part[t] = 0.0
                for si in T.serial(spc):
                    sb = cp * spc + si
                    if sb < nblk:
                        rbase = bx * nblk + sb
                        dv = T.Cast("float32", Wh[rbase * 72])
                        dmv = T.Cast("float32", Wh[rbase * 72 + 1])
                        BB = rbase * 144
                        q1 = Wq[BB + 4 + ln]
                        q2 = Wq[BB + 8 + ln]
                        scf = T.Cast("float32", T.if_then_else(
                            ln < 4, q1 & 63,
                            (q2 & 15) | ((Wq[BB + ln] >> 6) << 4)))
                        mnf = T.Cast("float32", T.if_then_else(
                            ln < 4, q2 & 63,
                            (q2 >> 4) | ((q1 >> 6) << 4)))
                        g = ln // 2
                        sh = 4 * (ln % 2)
                        wbase = rbase * 36 + 4 + g * 8
                        xb = sb * 256 + ln * 32
                        for wi in T.serial(8):
                            packed = Wq32[wbase + wi]
                            eb = xb + wi * 4
                            # byte b of word: bits [8b+sh, 8b+sh+4)
                            n0 = (packed >> (0 + sh)) & 15
                            n1 = (packed >> (8 + sh)) & 15
                            n2 = (packed >> (16 + sh)) & 15
                            n3 = (packed >> (24 + sh)) & 15
                            part[t] += (dv * scf * T.Cast("float32", n0) - dmv * mnf) * T.Cast("float32", X[eb + 0])
                            part[t] += (dv * scf * T.Cast("float32", n1) - dmv * mnf) * T.Cast("float32", X[eb + 1])
                            part[t] += (dv * scf * T.Cast("float32", n2) - dmv * mnf) * T.Cast("float32", X[eb + 2])
                            part[t] += (dv * scf * T.Cast("float32", n3) - dmv * mnf) * T.Cast("float32", X[eb + 3])
            T.reduce_sum(part, tot, dim=0)
            Y[bx] = tot[0]

    return gemv


def load_tensor(name):
    r = gguf.GGUFReader(GGUF_PATH)
    ts = next(t for t in r.tensors if t.name == name)
    assert ts.tensor_type == gguf.GGMLQuantizationType.Q4_K, ts.tensor_type
    ne = list(ts.shape)              # [k, m] gguf order (fastest first)
    k, m = int(ne[0]), int(ne[1])
    raw = np.frombuffer(ts.data, dtype=np.uint8)
    assert raw.nbytes == m * k // 256 * 144
    return np.ascontiguousarray(raw), m, k


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--val-rows", type=int, default=512, help="rows for validation")
    ap.add_argument("--threads", type=int, default=128)
    args = ap.parse_args()

    results = {}
    ok = True

    # ---- validation on a real tensor slice --------------------------------
    raw_full, M, K = load_tensor("blk.1.ffn_gate.weight")
    vr = args.val_rows
    wq_u8 = torch.from_numpy(np.ascontiguousarray(raw_full[: vr * (K // 256) * 144])).to(DEV)
    wq_h = wq_u8.view(torch.float16).reshape(-1)
    wq32 = wq_u8.view(torch.uint32).reshape(-1)
    x = torch.randn(K, device=DEV, dtype=torch.bfloat16)
    kern = tilelang.JITKernel(make_gemv_q4k(vr, K), target="hip", out_idx=[-1])
    y = kern(wq_u8, wq_h, x).float()

    deq = dequant_q4_k_rows(raw_full, vr, K)
    ref = torch.from_numpy(deq).to(DEV) @ x.float()
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    results["gemv_q4_k"] = {"tensor": "blk.1.ffn_gate.weight", "val_rows": vr,
                            "shape": [M, K], "rel_err": rel}
    ok &= rel < 2e-3
    print(f"v1 validate rows={vr}: rel_err={rel:.3e}")

    # v2 validation (fp16 activations)
    x16 = x.to(torch.float16)
    kern2 = tilelang.JITKernel(make_gemv_q4k_v2(vr, K, threads=args.threads), target="hip", out_idx=[-1])
    y2 = kern2(wq_u8, wq32, wq_h, x16).float()
    ref2 = torch.from_numpy(deq).to(DEV) @ x16.float()
    rel2 = ((y2 - ref2).abs().max() / ref2.abs().max()).item()
    results["gemv_q4_k_v2"] = {"rel_err": rel2}
    ok &= rel2 < 2e-3
    print(f"v2 validate rows={vr}: rel_err={rel2:.3e}")

    if args.bench:
        for label, tname, use_v2 in [
            ("ffn_gate/up_v1", "blk.1.ffn_gate.weight", False),
            ("ffn_gate/up_v2", "blk.1.ffn_gate.weight", True),
            ("ffn_down_v2", "blk.1.ffn_down.weight", True),
        ]:
            raw_t, tm, tk = load_tensor(tname)
            nbytes_row = tk // 256 * 144
            need = tm * nbytes_row
            u8 = torch.from_numpy(np.ascontiguousarray(raw_t)).to(DEV)
            h16 = u8.view(torch.float16).reshape(-1)
            w32 = u8.view(torch.uint32).reshape(-1)
            xv16 = torch.randn(tk, device=DEV, dtype=torch.float16)
            xvbf = xv16.to(torch.bfloat16)
            maker = make_gemv_q4k_v2 if use_v2 else make_gemv_q4k
            kt = tilelang.JITKernel(maker(tm, tk, threads=args.threads), target="hip", out_idx=[-1])
            if use_v2:
                us = bench(lambda: kt(u8, w32, h16, xv16))
            else:
                us = bench(lambda: kt(u8, h16, xvbf))
            results[label] = {"shape": [tm, tk], "tilelang_us": us,
                              "weight_MB": need / 1e6, "eff_GBps": need / us / 1e3}
            del u8, h16, w32, kt
            torch.cuda.empty_cache()

        # lm_head is Q6_K in this checkpoint; token_embd is Q4_K but not decode-hot.
        # NOTE: dense-fp16 torch baseline omitted: rocBLAS Tensile segfaults on
        # GEMV-ish shapes with the rocm6.4 wheel here. Analytic roofline:
        # 50.1 MB weights @ ~900 GB/s effective HBM = ~56 us floor.

        print(json.dumps(results, indent=2))
        with open("results_q4k.json", "w") as f:
            json.dump(results, f, indent=2)

    print("VALIDATION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
