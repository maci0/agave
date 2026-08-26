# TileLang Kernel-Generation Experiment

Research-only probe of [TileLang](https://github.com/tile-ai/tilelang) as a
kernel generator for Agave's backends. Nothing here ships in the binary
(AGENTS.md "External Prototypes" rule): TileLang output is a *reference* for
porting decisions, not a build dependency.

## TL;DR

| Agave backend | TileLang target | Status here | Verdict |
|---|---|---|---|
| ROCm/HIP | `hip` | **Compiles + runs + validates on RX 7900 XTX** | Viable now |
| CUDA | `cuda` | Needs nvcc (no NVIDIA GPU/toolkit on this box) | Viable on any NVIDIA machine; scripts run unchanged |
| CPU | `cpu` | Unsupported by TileLang 0.1.13 | Not viable; native Zig stays |
| Metal | - | No target (macOS-only backend anyway) | Not viable |
| Vulkan / WebGPU | - | No SPIR-V/WGSL codegen in TileLang | Not viable |

Conclusion: TileLang is useful to Agave as a **CUDA/HIP kernel prototyping
and tuning harness** whose output is hand-ported into native Zig kernels,
mirroring how Triton/CUTLASS prototypes are treated today.

## Setup

```bash
cd research/kernels/tilelang
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install tilelang numpy
# ROCm execution needs a ROCm torch build (CUDA torch cannot allocate HIP tensors):
VIRTUAL_ENV=$PWD/.venv uv pip install "torch==2.8.0+rocm6.4" --index-url https://download.pytorch.org/whl/rocm6.4
```

## Probe results (`probes/target_probe.py`)

```
target=cuda: FAIL  no CUDA_HOME/nvcc on this host
target=hip : COMPILE OK, RUN OK   (gfx1100, ROCm 7.2.4 host runtime)
target=cpu : FAIL  unsupported target
```

## Kernels & validation (`experiments/agave_ops.py`)

Ops mirror agave's decode hot path at Qwen3.8-27B shapes:

| kernel | shape | max err vs reference |
|---|---|---|
| rms_norm | n=5120 | 4.8e-7 |
| silu_mul | n=17408 | 4.8e-7 |
| gemv_bf16 | 4096x5120 | rel 2e-7 |
| gemv_q8_0 (dequant-in-kernel, gguf layout) | 4096x5120 | rel 3e-7 |

Q8_0 dequantization happens inside the kernel (`Scales[m,kb] fp16`,
`Qs[m,k] int8`) exactly like agave's native `gemv_q8_0`, demonstrating that
TileLang can express agave's quantized-GEMV pattern, not just dense math.

## Benchmarks (RX 7900 XTX, decode batch=1)

| op | TileLang us | torch us | note |
|---|---|---|---|
| rms_norm 5120 | 8.4 | 30.4 | 3.6x faster than unfused torch composite |
| silu_mul 17408 | 11.2 | 10.4 | parity, both bandwidth-bound |
| gemv_bf16 4096x5120 | 129 | 231 | 1.8x faster than torch GEMM path |
| gemv_q8_0 4096x5120 | 136 | 50* | *torch baseline is pre-dequantized fp32 GEMM (84 MB reads vs our 31 MB); untuned config is latency-bound |

An untuned TileLang GEMV beating rocBLAS's bf16 GEMM by 1.8x on a skinny
decode shape is the headline result: TileLang's reduction-tree codegen suits
memory-bound GEMV far better than a GEMM-library fallback.

## Generated source

`artifacts/*.hip.c` hold the emitted HIP C++ for inspection. Output is plain
`__global__` functions using `hip_runtime.h`; it would compile to HSACO via
hipcc and load through agave's existing `hipModuleLoadData` /
`hipModuleGetFunction` launcher unchanged.

## Porting path to agave backends

1. Prototype/tune an op here against golden vectors from
   `research/kernels/golden/`.
2. Copy the generated device function into the matching Zig kernel file
   (`src/backend/kernels/cuda/*.zig` emits PTX from Zig; HIP kernels live in
   `src/backend/kernels/rocm/*.zig` compiled to `kernels.hsaco`).
3. Benchmark through the standard harness (`research/kernels/run.py bench`)
   and only merge on a win, per the >5%-regression rule.

## Case study: Qwen3.8-27B Q4_K_M (`experiments/qwen38_q4k.py`)

End-to-end on the REAL checkpoint: reads `blk.1.ffn_gate.weight` /
`blk.1.ffn_down.weight` straight out of the downloaded
`Qwen3.8-27B-Q4_K_M.gguf`, uploads **only packed bytes**, and runs a fused
Q4_K dequant-in-kernel GEMV (ggml block layout: 144 B / 256 elems, fp16
d/dmin, 12 B of 6-bit scale/min pairs, nibble payload). The Python unpacker
is validated bit-exact against `gguf.dequantize`; both kernel variants match
a full-precision reference to rel_err <= 2e-7.

| variant | shape | us | eff. GB/s |
|---|---|---|---|
| v1 byte loads, 128 thr | gate 17408x5120 | 481 | 104 |
| v2 u32-word loads, 128 thr | gate 17408x5120 | 456 | 110 |
| v2, 256 thr | gate 17408x5120 | **456** | **110** |
| v2, 256 thr | down 5120x17408 | 653 | 77 |

Roofline: 50.1 MB of weights @ ~900 GB/s effective HBM is a ~56 us floor, so
these kernels sit at ~8-13% of peak. Diagnostics (identical-work variant with
the same loads) show the epilogue reduce is NOT the bottleneck; the limit is
the decomposition: one workgroup per output row, threads owning only 1-2
superblocks each, with byte-granular gather across 144 B-strided windows and
a barrier per row (m = 17k rows).

Identified next levers (not yet exhausted):
- multi-row workgroups + per-row segmented reduction (amortize the barrier),
- split-K two-phase (partials to gmem, tiny second kernel),
- 128-bit vector loads over the packed stream + LDS staging,
- occupancy shaping; verify with rocprof.

Takeaway for agave: TileLang's HIP path can express gguf Q4_K dequant-in-kernel
GEMV correctly against real checkpoints with modest effort; reaching the
hand-tuned-kernel bandwidth class needs the structural changes above, which is
exactly the prototype-then-port workflow AGENTS.md prescribes.

## Head-to-head: agave native (Zig/HIP) vs TileLang, RX 7900 XTX

**Measurement correction:** the first "native" bench round ran while
`hipModuleGetFunction` was broken (Bug 3 below); agave-bench silently fell
back to CPU and mislabeled rows as ROCm. Those numbers are void. After fixing
the loader (metadata/symbol normalization, committed HSACO regenerated), the
TileLang designs were **ported into the Zig HIP kernels themselves**:

- `gemv_q4_k.zig`: one row per workgroup, 32 copies x 8 lanes, lane owns one
  32-elem sub-block slice read as 8 u32 words with constant-shift nibble
  extraction; block-wide reduce. Launcher grid n.
- `gemv_bf16.zig`: paired bf16 weights per u32 load, per-chunk activation
  rescans removed.
- `agave-bench gemv_q4_k` gained a host-reference validator
  (`max_rel_err` JSON line) mirroring the gguf.dequantize-checked reference.

Results (validated, machine under external load so treat +-30%):

| kernel | shape | old native* | ported Zig | TileLang gen |
|---|---|---|---|---|
| gemv_q4_k | 17408x5120 | ~1480 us (CPU fallback) | **183 us / 276 GB/s, err 1.2e-7** | 456 us |
| gemv_q4_k | 5120x17408 | - | **199 us / 252 GB/s, err 1.2e-7** | 653 us |
| gemv_bf16 | 4096x5120 | ~910 us (suspect) | **155 us / 272 GB/s** | 129 us |
| gemv_q8_0 | 4096x5120 | already good | unchanged | 136 us |

The hand-ported Zig version **beats the generated TileLang kernel ~2.5x** on
Q4_K GEMV (real LDS two-phase reduce via `blockReduceAdd` + f32 activations vs
the generic tl::AllReduce path) while carrying a numeric validator in the bench.

## Pre-existing gap this work surfaced

Small-model greedy runs produce garbage on BOTH Vulkan and ROCm while CPU is
correct (Qwen2.5-0.5B/1.5B GGUF). This predates the port (Vulkan shares none
of these kernels) and matches docs/TEST_MATRIX.md's own note that full
Vulkan/ROCm correctness coverage is missing. Follow-up: extend the
bench-validator pattern (host reference vs device output) to the remaining
GEMV dtypes (q5_0/q6_k/q2_k...) and to rms_norm/rope/sdpa, then bisect the
E2E divergence per layer.

## Gotchas found (would bite anyone adopting TileLang)## Gotchas found (would bite anyone adopting TileLang)

- Eager-mode annotations must NOT use `from __future__ import annotations`:
  dims referenced only in string annotations are not closure cells and raise
  `NameError`.
- `T.reduce_sum` accepts a single dim (no tuple); flatten fragments instead.
- The HIP path requires tensors from a **ROCm** torch build; CUDA-torch
  dlpacks as the wrong device type.
- JIT cache can serve stale host libs across torch rebuilds; clear
  `~/.tilelang` if you swap torch variants.
- rocBLAS (rocm6.4 wheel) segfaults/Tensile-throws on GEMV-ish `mv`/`matmul`
  shapes on gfx1100; avoid it as a baseline for skinny shapes.
- `Tensor.view(torch.uint32)` + dlpack works for zero-copy dual views of the
  same packed buffer (uint8 scalar path and uint32 vector path side by side).
