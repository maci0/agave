"""Probe TileLang backend support on this machine.

Agave backends: cpu, cuda, rocm, vulkan, webgpu, metal. TileLang compiles
through TVM; its first-class targets are CUDA (nvcc) and HIP (rocm/hipcc).
This script reports which targets can compile and run here.

Run: .venv/bin/python probes/target_probe.py
"""

from __future__ import annotations

import shutil
import sys
import traceback

import torch
import tilelang
import tilelang.language as T


def make_add_one():
    @T.prim_func
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=16) as bx:
            for i in T.Parallel(16):
                B[i] = A[i] + 1.0

    return add_one


def main() -> int:
    print(f"tilelang {tilelang.__version__}")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} hip={torch.version.hip}")
    for tool in ("nvcc", "hipcc", "clang"):
        print(f"{tool}: {shutil.which(tool)}")

    func = make_add_one()
    rc = 0
    for target in ("cuda", "hip", "cpu"):
        try:
            kern = tilelang.JITKernel(func, target=target, out_idx=-1)
            # Try to execute when device memory is available.
            try:
                if target == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("no CUDA device")
                dev = {"cuda": "cuda", "hip": "cuda"}[target]
                a = torch.randn(16, device=dev)
                b = kern(a)
                ok = b is None or bool((b == a + 1).all().item())
                print(f"target={target}: COMPILE OK, RUN {'OK' if ok else 'MISMATCH'}")
            except Exception as e:  # noqa: BLE001
                print(f"target={target}: COMPILE OK, RUN SKIPPED ({type(e).__name__}: {e})")
        except Exception as e:  # noqa: BLE001
            rc = 1
            first = str(e).splitlines()[0][:140] if str(e) else type(e).__name__
            print(f"target={target}: FAIL {type(e).__name__}: {first}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
