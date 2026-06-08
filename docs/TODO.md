# Agave TODO & Roadmap

Bugs, performance issues, and future work. Detailed designs inline.

**Last updated**: 2026-06-08

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp) | Low (upstream) | Won't fix |
| 6 | ROCm HSACO target triple rejected by kernel 7.0.6+ | High | Needs Zig std library patch for amdgcn target triple |
| 4 | Vulkan push descriptor crashes on RADV gfx1100 | Medium | Suspected RADV driver issue. Infrastructure in place, disabled |
| ~~8~~ | ~~Gemma 4 26B-A4B MoE garbled output~~ | ~~Fixed~~ | ~~expertWeightStride used dims[0]*dims[1] instead of dims[1]*dims[2]~~ |
| 9 | GPT-OSS MXFP4 garbled output | Low | MXFP4 dequant or attention sink bug |

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

| Backend | Status | Notes |
|---------|:------:|-------|
| Metal | Complete | 70+ pipelines, GPTQ, paged SDPA |
| CUDA | Complete | 56 kernels, fused FFN, 3 megakernels |
| Vulkan | Complete | 44 shaders, deferred dispatch |
| WebGPU | Complete | 43 shaders, lazy readback |
| ROCm | Complete | 44 kernels, GPTQ, 1 megakernel |

---

## Performance

| # | Issue | Status |
|---|-------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Optimized — needs benchmarking |
| 2 | WebGPU decode 0.7 tok/s | Fixed — deferred buffer lifecycle + lazy readback (0.6 tok/s) |
| 3 | Gemma 4 E4B CPU prefill ~60s | Partial — MoE batched via gemvMulti |
| 4 | NVFP4 accuracy lower than MLX-4bit | Open — may be community quant quality |

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
| `agave pull` (HF Hub model download with resume) | — |
| Deferred dispatch (Vulkan/WebGPU single-submit) | — |
| All quantized GEMV kernel gaps closed (32 new kernel files) | — |
| MTP speculative decoding (`--spec-mode mtp`, nextn heads) | llama.cpp |
| Llama 4 architecture (iRoPE, chunked attention, top-1 MoE, temperature scaling) | — |
| Jump decoding (skip forward pass for deterministic grammar tokens) | vLLM |
| API prompt prefix caching (KV reuse for shared conversation prefix) | vLLM |
| Batched KV swap (TransferCallback + batchPromoteToVram) | vLLM |
| TurboQuant in SDPA kernel (sdpa_fa2_turbo) | — |
| Spec decode thinking budget (adaptive cooldown) | — |
| Topology-aware auto partitioning (device cap exchange) | Partial |
| Sparse GEMV for all GPU backends (Metal +12%, Vulkan, WebGPU) | PowerInfer/TurboSparse |
| WebGPU buffer lifecycle fix (defer params + cache destruction) | — |
| WebGPU backend enabled by default | — |
| Apple Accelerate.framework (AMX BLAS for F32 CPU GEMV/GEMM) | — |
| MLX-4bit SafeTensors rope_theta + vocab_size fix | — |
| GGUF MoE expert stride fix (dims[1]*dims[2]) | — |

### High Priority

(empty — all high priority items done)

### Medium Priority

| # | Feature | Impact | Source |
|---|---------|--------|--------|
| 14 | gRPC server (HTTP/2) | Lower overhead serving — use nginx/envoy for now | vLLM |

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
| ~~26~~ | ~~Sparse GEMV (skip near-zero FFN activations, ~40% sparsity measured)~~ | Done (CPU +21%, Metal +12%) |
| 27 | DeepSeek V4 support (mHC hyper-connections, CSA/HCA attention) | — |
| 28 | AWQ column-major INT4 GEMV kernel (currently uses GPTQ row-major — wrong packing) | AWQ |
| ~~29~~ | ~~TQ1_0 ternary GEMV kernel (BitNet 1.58-bit, {-1,0,1}, 5 trits/byte)~~ | Done (all 6 backends) |
| ~~30~~ | ~~TQ2_0 ternary GEMV kernel (2-bit ternary, faster on AVX2)~~ | Done (all 6 backends) |
| 31 | EXL2 mixed-precision codebook (NVIDIA only) | ExLlama |
| 32 | HQQ half-quadratic quantization | HQQ |

---

## Design Notes

## Implemented Features

| Feature | Status |
|---------|--------|
| Tensor/Pipeline parallelism (6 modes, TCP/shm/NCCL) | Working |
| Grammar-constrained decoding (GBNF + JSON schema) | Working |
| TriAttention KV eviction (norm + frequency) | Phase 1+2 |
| Speculative decoding (DDTree, self-spec, draft, n-gram) | Working |
| CUDA fused FFN megakernels | Done |
| GPTQ SafeTensors support | Working |
| Native FP4 on Blackwell SM121 | Working |
| WebGPU Phase 2 WASM | Blocked by Zig 0.16 codegen bug |

---

## Model Abstraction (Deferred)

All 8 models share near-identical skeletons. A `ModelBuilder` could save ~600 lines but adds comptime complexity. Deferred because:
1. Models rarely change once working
2. Each has unique quirks (Gemma scaling, GPT-OSS sinks, Qwen DeltaNet, GLM4 MLA)
3. Self-contained files are easier to debug

---

## Direct-to-VRAM Loading

> Partially implemented: tiered KV cache via `--kv-tiers`. Weight loading via cuFile is future work.

UMA platforms (Apple Silicon, GB10) already optimal via zero-copy mmap. Discrete GPUs would benefit from GPUDirect Storage (`cuFileRead`) bypassing CPU RAM. Low priority since most targets are UMA.

---

## Pre-Sharded Weights

`agave shard` subcommand: split GGUF by TP degree → `model-tp0.gguf`, `model-tp1.gguf`. Zero init-time sharding, peak memory = shard size. Auto-detect via GGUF metadata.

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
| Golden tests need model files | By design — manual trigger |
