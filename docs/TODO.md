# Agave TODO & Roadmap

Bugs, performance issues, and future work. Detailed designs inline.

**Last updated**: 2026-10-20

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash, degenerate output (also broken in llama.cpp) | Low (upstream) | Won't fix |
| ~~6~~ | ~~ROCm HSACO: wrong ISA triple + LOCAL .kd symbols~~ | ~~Fixed~~ | ~~Two Zig 0.16 bugs: (1) amdhsa semver appended to ISA triple, (2) .kd symbols emitted LOCAL. Worked around via fix_kd_isa.py (ET_REL patch + llvm-objcopy). hipModuleLoad now returns hipSuccess on ROCm 7.2.4 / gfx1100~~ |
| 4 | Vulkan push descriptor crashes on RADV gfx1100 | Medium | Suspected RADV driver issue. Worked around: push descriptors disabled, deferred path used |
| ~~11~~ | ~~`zig build test` never exits on an all-pass run~~ | ~~Fixed~~ | ~~Three fuzz tests (`display`, `devices.discovery`, `calibrate`) called functions that write to fd 1. Under the server-mode test runner (build.zig) fd 1 is the protocol pipe to the build runner, so plain text desynchronized it and both processes waited forever. They now redirect fd 1 via `src/test_stdout.zig`. Masked until now because the suite always ended on a failing test.~~ |
| ~~14~~ | ~~Speculative decoding corrupts recurrent state on hybrid models~~ | ~~Fixed~~ | ~~Verification rewinds the target with `setKvSeqLen(pre_draft_pos)` and replays the accepted prefix. Exact for attention, a no-op for the DeltaNet conv and recurrence states, which had already absorbed every token the drafter and the previous verify attempt pushed through them, so the replay double-consumed the prefix and drifted further every round. Hit `qwen35` and `qwen4exp` on every mode that drafts through the target (`self`, `ddtree`, `mtp`) and on the `commitTargetPrefix` replay of all tree modes. `SpecState.beginRound` now snapshots the recurrent state at the committed position and `rewindTarget` restores it alongside the KV length. `qwen4exp` had no snapshot methods at all; its PLE n-gram and conv histories are in the snapshot too.~~ |
| ~~15~~ | ~~Three Metal kernels wrong: `gemm_q8_0`, `gemm_q4_0`, `gemv_t`~~ | ~~Fixed~~ | ~~`zig build validate -Dvalidate-backend=metal` failed on `gemm_q8_0` and both `gemv_t` sizes. Both GEMM kernels truncated the Q8_0/Q4_0 block count (`n_in / 32`), which is also the row stride, so any `n_in` off a 32 multiple strided every row after the first into the previous row's blocks; their tail block also read past the token's row into the next token's activations. `gemv_t` was dispatched with `endEncode1D`, which issues `out_dim` threads rather than the `out_dim` threadgroups the kernel is written for, and at the pipeline's 1024-thread maximum against an 8-entry reduction scratch. Separately the `gemv_t` bench case fed f32 bytes to a Q8_0 kernel, so the comparison was two readings of garbage and reported a flat 1.0 error; it now builds real Q8_0 weights, and `gemm_q4_0` has a bench case and a validate entry at all.~~ |
| ~~16~~ | ~~Three WebGPU kernels wrong, one crashing~~ | ~~Fixed~~ | ~~`zig build validate -Dvalidate-backend=webgpu` failed 7 checks. `rope.wgsl` rotated adjacent pairs `(d, d+1)` where the CPU and Metal kernels and the GGUF weight layout use the half-split `(i, i + rope_dim/2)`, so every model decoded wrong on WebGPU; `ropeBatched` loops it, so both failed. `gemv_q4_1` truncated its block count, which is also the row stride. `embLookup` sized its f32 upload as `dim * 4 * emb_max_vocab` because the signature carries no row count: 1.05 GB read at dim 1024, past the end of any model with a vocab under 256000, and an outright segfault on a smaller table. It now reads the single row on the host for every dtype, the way Metal does, and `embedding.wgsl` is gone with the `emb_max_vocab` guess. A successful library load was also logged at `warn`, which kept the validate step red even when every kernel passed.~~ |
| 10 | Qwen 3.5 GGUF decodes to garbage on ROCm and Vulkan (gfx1100) | High | Open. Reproduced 2026-08-26 on RX 7900 XTX with `qwen2.5-0.5b` Q8_0 and `qwen2.5-1.5b` Q4_K: `--backend cpu` answers "The answer is 4.", `--backend rocm` and `--backend vulkan` both emit `  1  (  (1)  (9`. Ruled out by `tests/test_backend_parity.zig`: add, mul, silu, rmsNorm, softmax, l2Norm, rope, GEMV f32/q8_0/q4_k, and SDPA (GQA 4:1, hd=128) all match CPU on both backends, with SDPA driven incrementally over 256 positions the way decode drives it. So every individual op is correct and the fault is above them: model-level op sequencing in `qwen35.zig`, or the device-buffer caches. Prime suspect is those caches (`act_cache`, `buf_cache`, `getOrAllocKvBuf` in `rocm.zig`/`cuda.zig`), which are keyed by **host pointer address**; any reuse of an address for a different tensor silently returns stale device memory. Next step: dump per-layer activations on CPU vs Vulkan and bisect to the first diverging layer. |
| 17 | DS4 Flash 0731 on 2-node DGX Spark fails with `CUDA_ERROR_OUT_OF_MEMORY` (700) | High | Open, blocked on the hardware. Hit on both GB10 nodes (sm_121, CUDA 13.0, NCCL 2.31.2) at the first drain's D2H after a node reboot, in every code state including the first commit of the work, and the same build had run once at 272 tok/s on a tiny standin before the reboots. A C harness on the same fresh nodes passes every operation the engine performs: the full 36-alloc `cuMemAlloc` sequence, blocking and async H2D/D2H from stack, heap, anon mmap and the mmap'd shard, 4B-1KB D2H, and the engine's own PTX module load plus `silu_kernel` and `gemv_f32_kernel` launches with correct results. Under `CUDA_LAUNCH_BLOCKING=1` the engine's launch returns 700 where the identical C launch passes, so the fault is in the process state, not any single call. Suspects: the post-reboot driver version, a UVA collision with the Zig allocator's buffers, the context state. Needs on-node debugging: run vLLM on a fresh node to confirm the driver serves other CUDA apps, run the engine with `CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0` at the console, check `journalctl -k` for NVMe/UMA driver faults. Launch procedure and NCCL environment are in `scripts/test-tp-nccl.sh`. |
| 12 | Device buffer caches are keyed by host pointer address | Medium | Open. `getOrAllocKvBuf`, `act_cache` and `buf_cache` in `rocm.zig` and `cuda.zig` map a host address to a device allocation and never invalidate. Free a host buffer, allocate another that lands on the same address, and the backend hands back the previous device buffer with the previous contents. Reproduced while writing `tests/test_backend_parity.zig`: running several SDPA shapes in one process, each with fresh allocations, produced errors of ~0.2 that vanished when each shape ran in its own process. Also: neither backend ever uploads existing host KV to the device, so a cache populated on the host (prefix-cache restore, KV import, backend switch) reads as uninitialised device memory. |
| ~~13~~ | ~~oxlint not yet enforced on the large web sources~~ | ~~Fixed~~ | The chat UI and the browser shell are component trees now (`src/web/*.tsx`, `web/shell.tsx`; React, and since then Preact through `vendor/react`), every source is out of `.oxlintrc.json` `ignorePatterns`, and `scripts/check-web-lint-scope.sh` now holds build outputs only. `src/web/app.test.tsx` mounts the real entry against a stubbed server, so a broken bundle is a red `bun test` rather than a blank page. |
| ~~8~~ | ~~Gemma 4 26B-A4B MoE garbled output~~ | ~~Fixed~~ | ~~expertWeightStride used dims[0]*dims[1] instead of dims[1]*dims[2]~~ |
| ~~9~~ | ~~GPT-OSS MXFP4 garbled output~~ | ~~Fixed~~ | ~~Two bugs: mxfp4_group_size=32 (should be 16 per NVIDIA spec), scale decoded as E8M0 (should be FP8 E4M3). Fixed across CPU + all 6 GPU backends (Metal/CUDA/Vulkan/WebGPU/ROCm) + host buffer sizing~~ |

<details><summary>Fixed bugs</summary>

| # | Issue | Fix |
|---|-------|-----|
| 2 | ROCm GEMV Q8_0 wrong results | Replaced dword-packed with byte-by-byte (same as CUDA) |
| 3 | Vulkan segfault on RADV NAVI31 | Descriptor buffer overflow [4]→[16] |
| 5 | Vulkan 2.7 tok/s dispatch bottleneck | Deferred dispatch with per-op descriptor sets |
| 7 | CUDA K-quant PTX spilling on sm_121 | CPU fallback with thread pool |
</details>

---

## GPU Kernel Coverage

All quantized GEMV formats native on all 6 backends. See [KERNELS.md](KERNELS.md).

"Complete" below means every kernel is present, not that every model decodes
correctly on that backend. Bug 10 above has Qwen 3.5 producing garbage on ROCm
and Vulkan (gfx1100) despite full kernel coverage.

| Backend | Status | Notes |
|---------|:------:|-------|
| Metal | Complete | 104 pipelines, GPTQ, paged SDPA, 10 DS4 kernels |
| CUDA | Complete | 61 kernels, fused FFN, 3 megakernels |
| Vulkan | Complete | ~49 shaders, deferred dispatch |
| WebGPU | Complete | ~48 shaders, lazy readback |
| ROCm | Complete | 49 kernels, GPTQ, 1 megakernel |

---

## Tried and rejected

### Metal 4 tensor-API FlashAttention (2026-10-19)

llama.cpp #29570 added a tensor-API flash attention kernel for F16 KV. The
capability it delivers, **F16 KV attention, has shipped** (see Done): the FA2
kernels dequant f16 in-kernel. What is rejected here is the tensor API as the
vehicle for it, because the blocker is the language version, not the kernel.

`metal::tensor` needs MSL 4.0, requested through
`MTLCompileOptions.languageVersion`, and MSL 4.0 needs macOS 26. Agave
compiles every shader into one library from one `newLibraryWithSource:` call,
so raising the version raises it for all of them: Metal would stop working on
macOS 15, which is what CI's `test-macos` job runs on
(`.github/workflows/ci.yml`, `runs-on: macos-15`), and on every user below
26. Compiling a second library at MSL 4.0 and picking at runtime avoids that
and doubles shader compile time while splitting the kernel set in two.

Not worth either cost on the evidence available: the SDPA kernels are already
FlashAttention-2 with sparse-V skipping, and `agave-bench` varies about 2x
run to run (see below), so a win could not be demonstrated on this hardware
anyway. Revisit when the macOS floor is 26, where it becomes a single-library
change.

### Output-row tiling in the Metal Q8_0 GEMM (2026-10-19)

`gemm_q8_0` runs one threadgroup per output row, so every threadgroup streams
the whole of X: at `n_out = n_in = 4096` with 4 tokens that is 268 MB of X
traffic against 18 MB of W. Holding 4 rows per threadgroup should cut the X
passes fourfold, which is why `gemv_q8_0` carries `q8_0_nr = 4`.

Measured on an M4 Pro, NR=4 is **slower at every shape**: -3% at 1024², -8%
at n_out 4096 / n_in 1024, -11% at 2048², -19% at n_out 1024 / n_in 4096, and
-44% at 4096². Tiling divides the grid by NR along with the traffic, and the
M-series GPU wants the threadgroups more than it wants the bandwidth. Reverted.

Two things to carry forward rather than relearn:

- A runtime row count is far worse than either fixed value. `sums[r][ti]`
  under a non-constant bound is dynamically indexed, which Metal spills out
  of registers; it measured below both NR=1 and NR=4 everywhere. Any future
  tiling needs one kernel per NR, the way `gemv_tiled.metal` does it.
- The rejection was re-measured after `agave-bench` switched to reporting the
  minimum instead of the median, interleaving the two binaries and taking the
  best of three runs each. NR=4 is slower at every shape and gets worse with
  size: -8.3% at 1024², -12.7% at 2048², -6.8% at n_out 4096 / n_in 1024,
  -18.2% at n_out 1024 / n_in 4096, -22.9% at 4096², -26.4% at 8192x4096.
  The original median-based numbers were not trustworthy; these are.
- A single run is still not a measurement on Apple silicon: the GPU picks one
  of two power states about 2x apart per process. See
  [benchmarks/README.md](../benchmarks/README.md).

---

## Performance

| # | Issue | Status |
|---|-------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Optimized, needs benchmarking |
| 2 | WebGPU decode 0.7 tok/s | Fixed, deferred buffer lifecycle + lazy readback (0.6 tok/s) |
| 3 | Gemma 4 E4B CPU prefill ~60s | Partial, MoE batched via gemvMulti |
| 4 | NVFP4 accuracy lower than MLX-4bit | Open, may be community quant quality |

---

## Roadmap

### Done

| Feature | Source |
|---------|--------|
| `--ctx-size auto` (memory-safe context fitting) | vLLM |
| xxHash prefix cache (RadixTree fast path) | vLLM |
| N-gram speculative decoding (`--spec-mode ngram`) | vLLM |
| SSM state prefix caching | vLLM |
| Async scheduler + PP overlap | vLLM |
| Profile-guided speculative decoding (adaptive K) | llama.cpp |
| Mirostat sampling (target-entropy adaptive) | llama.cpp |
| XTC sampling (`xtc_probability`, `xtc_threshold`) | llama.cpp |
| DRY sampling (n-gram repetition penalty) | llama.cpp |
| Tool/function calling (OpenAI-compatible, streaming + non-streaming) | llama.cpp |
| Logit bias support | llama.cpp |
| Zero-config P2P discovery (UDP) | Mesh-LLM |
| Downstream-first stage startup (RTT handshake) | Mesh-LLM |
| Peer RTT measurement | Mesh-LLM |
| Pre-sharded weight files (design) | Mesh-LLM |
| `agave pull` (HF Hub model download with resume) | n/a |
| Deferred dispatch (Vulkan/WebGPU single-submit) | n/a |
| All quantized GEMV kernel gaps closed (32 new kernel files) | n/a |
| MTP speculative decoding (`--spec-mode mtp`, nextn heads) | llama.cpp |
| Llama 4 architecture (iRoPE, chunked attention, top-1 MoE, temperature scaling) | n/a |
| Jump decoding (skip forward pass for deterministic grammar tokens) | vLLM |
| API prompt prefix caching (KV reuse for shared conversation prefix) | vLLM |
| Batched KV swap (TransferCallback + batchPromoteToVram) | vLLM |
| TurboQuant in SDPA kernel (sdpa_fa2_turbo) | n/a |
| Spec decode thinking budget (adaptive cooldown) | n/a |
| WebGPU buffer lifecycle fix (defer params + cache destruction) | n/a |
| WebGPU backend enabled by default | n/a |
| Apple Accelerate.framework (AMX BLAS for F32 CPU GEMV/GEMM) | n/a |
| Word-level BPE cache (skip applyBpe for recurring pretokens) | gigatoken |
| MLX-4bit SafeTensors rope_theta + vocab_size fix | n/a |
| GGUF MoE expert stride fix (dims[1]*dims[2]) | n/a |
| LoRA adapter loading (`--lora <path>`, load-time merge, all quant formats) | llama.cpp |
| Sparse GEMV (skip near-zero FFN activations, ~40% sparsity measured; CPU +21%, Metal +12%) | PowerInfer/TurboSparse |
| DeepSeek V4 Flash 0731 full support (HC, MLA, CSA/HCA, LID, hash routing; 10 Metal kernels in `ds4.metal` + `ds4_fused.metal`; dedicated CpuBackend bypass gives bit-identical `--backend cpu` / `--backend metal` output at 10.7-21.2 tok/s with suffix speculation, see [DS4_METAL_DIVERGENCE.md](DS4_METAL_DIVERGENCE.md)) | n/a |
| AWQ column-major INT4 GEMV kernel (+ nibble order fix, all 6 backends) | n/a |
| TQ1_0 ternary GEMV kernel (BitNet 1.58-bit, {-1,0,1}, 5 trits/byte, all 6 backends) | n/a |
| TQ2_0 ternary GEMV kernel (2-bit ternary, faster on AVX2, all 6 backends) | n/a |
| HQQ half-quadratic quantization (Metal/Vulkan/WebGPU/CUDA/ROCm native + CPU) | n/a |
| Scattered-row prefetch for the Qwen4Exp PLE table (`Format.prefetchRows`) | llama.cpp |
| Throughput + greedy-determinism release gate (`zig build gate`) | Atlas |
| Qwen4-Exp QSA Q/K/V batched into one `gemvMulti` dispatch (Qwen 3.5 already had it) | vLLM |
| Prefix-cache LRU eviction (`RadixTree.evictLru`); at the cap an insert evicts instead of refusing | SGLang |
| Windowed SDPA on the backend's fused kernel (shifted cache base), so sliding-window layers stop attending to the whole sequence and read the window instead | llama.cpp / ggml |
| Real per-head softmax stats from GPU SDPA, so mixed-tier attention merges one partial per tier instead of falling back to a full-sequence CPU pass | vLLM HiSparse |
| Per-request KV sequences on Gemma 4 plus scheduler slot admission, so `--max-batch-size` binds instead of being inert | LM Studio / vLLM |
| Sliding-window layers store their window instead of the whole context (`src/kvcache/window.zig`) | SGLang |
| F16 KV attention on Metal (in-kernel dequant, `bits_k == 16`); `--kv-type f16` used to hit the unsupported-type panic despite being the default | llama.cpp |
| `agave-bench` reports the minimum, not the median, with `--warmup-ms` for the GPU clock ramp | n/a |
| `/v1/systemone` decision endpoint (noul / choice / score by logit read) | llama.cpp / vLLM / SGLang |
| Tool-call and reasoning delimiters read from the chat template, not server literals | vLLM |
| KV prefix wire format carries a header (version, geometry, RoPE bases) | llama.cpp |

### Partial

| Feature | State |
|---------|-------|
| Topology-aware auto partitioning (device cap exchange) | Partial: device cap exchange incomplete |

### High Priority

(empty, all high priority items done)

### Medium Priority

| # | Feature | Impact | Source |
|---|---------|--------|--------|
| 14 | gRPC server (HTTP/2) | Lower overhead serving, use nginx/envoy for now | vLLM |
| 33 | SWA-aware *prefix-cache* retention | The storage half shipped: a sliding-window layer holds its window instead of the whole context (`src/kvcache/window.zig`). What is left is the cross-request half, freeing a shared prefix's sliding-layer blocks early, which needs `RadixNode.block_ids` split by layer class and block sharing to exist at all. The radix tree is metrics-only today (`src/server/scheduler.zig`, the insert site), so there is nothing shared to free. SGLang measured hit rate 43.8% → 60.8% and mean TTFT 1.57s → 1.07s for the full thing. | SGLang |
| 34 | Per-request KV on the remaining six architectures | Gemma 4 holds up to `max_kv_slots` sequences and the scheduler admits one request per sequence, so `--max-batch-size` binds there. Every other architecture reports one sequence through `Model.kvSlotCount` and still runs requests one at a time. Each needs its per-layer caches turned into per-slot sets the way `Gemma4Model.allocKvSlot` does. | LM Studio / vLLM |
| 35 | Batched attention across slots | Slots interleave requests at token granularity; they do not batch the forward pass, so throughput per token is unchanged and the win is latency fairness. Batching the attention across resident sequences is the next step and needs the paged view wired first. | vLLM |

### Low Priority

| # | Feature | Source |
|---|---------|--------|
| 15 | Conditional compilation (`-Denable-<quant>`) | vLLM |
| 16 | Flash Linear Attention kernels | vLLM |
| 17 | Fused GPU rejection sampling | vLLM |
| 18 | Cross-layer KV sharing | vLLM |
| 19 | Heterogeneous TP (mixed devices) | vLLM/Exo |
| 20 | Coordinator-only nodes (`--no-worker`) | Exo |
| 21 | Demand-aware model rebalancing | Mesh-LLM |
| 22 | QUIC transport | Mesh-LLM |
| 23 | Nostr-based discovery | Mesh-LLM |
| 24 | RDMA over Thunderbolt 5 | Exo |
| 25 | Inter-model collaboration (MoM) | Mesh-LLM |
| 31 | EXL2 mixed-precision codebook (NVIDIA only) | ExLlama |

Items 26, 27, 28, 29, 30 and 32 shipped and moved to Done above.

---

## Implemented Features

| Feature | Status |
|---------|--------|
| Tensor/Pipeline parallelism (2-rank TP or PP, TCP/shm/NCCL; RCCL declared) | Working |
| Grammar-constrained decoding (GBNF + JSON schema) | Working |
| TriAttention KV eviction (norm + frequency) | Phase 1+2 |
| Speculative decoding (15 modes: standard, ddtree, self, ngram, suffix, lookahead, mtp, medusa, eagle, eagle3, mlp, pflash, dspark, dflash2, auto) | Working |
| CUDA fused FFN megakernels | Done |
| GPTQ SafeTensors support | Working |
| Native FP4 on Blackwell SM121 | Working |
| WebGPU Phase 2 WASM | Blocked by Zig 0.16 codegen bug |

---

## Model Abstraction (Deferred)

All 12 chat architectures plus the DFlash2 drafter share near-identical skeletons. A `ModelBuilder` could save ~600 lines but adds comptime complexity. Deferred because:
1. Models rarely change once working
2. Each has unique quirks (Gemma scaling, GPT-OSS sinks, Qwen DeltaNet, GLM4 MLA)
3. Self-contained files are easier to debug

---

## Direct-to-VRAM Loading

> Partially implemented: tiered KV cache via `--kv-tiers`. Weight loading via cuFile is future work.

UMA platforms (Apple Silicon, GB10) already optimal via zero-copy mmap. Discrete GPUs would benefit from GPUDirect Storage (`cuFileRead`) bypassing CPU RAM. Low priority since most targets are UMA.

---

## Pre-Sharded Weights

Not built. The subcommands today are `pull`, `calibrate`, `update`, and `help`; there is no `agave shard`.

Planned: an `agave shard` subcommand splitting GGUF by TP degree → `model-tp0.gguf`, `model-tp1.gguf`. Zero init-time sharding, peak memory = shard size. Auto-detect via GGUF metadata.

---

## KV Cache Quantization Methods

| Method | Rotation | FMAs (d=128) | Storage | CLI |
|--------|----------|:------------:|---------|-----|
| TurboQuant | WHT-32 | ~640 | f16 norm + packed indices | `tq2/tq3/tq4` |
| PlanarQuant | Givens 2D | 256 | same | `pq2/pq3/pq4` |
| IsoQuant | Quaternion 4D | 512 | same | `iq2/iq3/iq4` |
| RotorQuant | Cl(3,0) rotor | ~2,400 | same | `rq2/rq3/rq4` |

PlanarQuant uses ~2.5x fewer FMAs than TurboQuant. All share Lloyd-Max codebook (2.5/3.5/4.5 bpe).

---

## Build / CI

| Issue | Status |
|-------|--------|
| Golden tests need model files | By design, manual trigger |
