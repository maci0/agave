# Agave TODO & Roadmap

Bugs, performance issues, and future work. Detailed designs inline.

**Last updated**: 2026-05-19

---

## Bugs

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | GLM-4.7 Flash — degenerate output (also broken in llama.cpp) | Low (upstream) | Won't fix |
| 6 | ROCm HSACO target triple rejected by kernel 7.0.6+ | High | Needs Zig std library patch for amdgcn target triple |
| 4 | Vulkan push descriptor crashes on RADV gfx1100 | Medium | Suspected RADV driver issue. Infrastructure in place, disabled |

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
| CUDA | Complete | 54 kernels, fused FFN, 3 megakernels |
| Vulkan | Complete | 42 shaders, deferred dispatch |
| WebGPU | Complete | 43 shaders, lazy readback |
| ROCm | Complete | 42 kernels, GPTQ, 1 megakernel |

---

## Performance

| # | Issue | Status |
|---|-------|--------|
| 1 | Q4_K Metal GEMV slower than llama.cpp | Optimized — needs benchmarking |
| 2 | WebGPU decode 0.7 tok/s | Optimized — deferred dispatch + lazy readback |
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
| XTC sampling (`xtc_probability`, `xtc_threshold`) | llama.cpp |
| Downstream-first stage startup (RTT handshake) | Mesh-LLM |
| Peer RTT measurement | Mesh-LLM |
| Pre-sharded weight files (design) | Mesh-LLM |

### High Priority

| # | Feature | Impact | Source |
|---|---------|--------|--------|
| 1 | ~~SSM state prefix caching~~ | ~~~2x prefill speedup for hybrid SSM models~~ | Done |
| 2 | ~~Async scheduler + PP overlap~~ | ~~Decode-first + chunked prefill~~ | Done |
| 3 | Batched KV swap (cuMemcpyBatchAsync) | Reduce tiered cache API overhead | vLLM |
| 4 | ~~Mirostat sampling~~ | ~~Target-entropy adaptive sampling~~ | Done |
| 5 | MTP (Multi-Token Prediction) heads | Native multi-token output for Qwen3.6/DeepSeek | llama.cpp |

### Medium Priority

| # | Feature | Impact | Source |
|---|---------|--------|--------|
| 6 | TurboQuant in SDPA kernel | Skip decode-time dequant | vLLM |
| 7 | ~~Spec decode thinking budget~~ | ~~Adaptive cooldown on low acceptance~~ | Done |
| 8 | Multi-stream pre-attention GEMM | Overlap QKV(N+1) with SDPA(N) | vLLM |
| 9 | ~~Profile-guided speculative decoding~~ | ~~Adaptive K per step~~ | Done |
| 10 | Router mode (multi-model server) | Switch models per request | llama.cpp |
| 11 | Hybrid memory abstraction | Unified KV+SSM cache | llama.cpp |
| 12 | ~~Topology-aware auto partitioning~~ | ~~Device cap exchange done, weighted split pending~~ | Partial |
| 13 | ~~Zero-config P2P discovery (UDP)~~ | ~~No --peers needed on LAN~~ | Done |
| 14 | gRPC server (HTTP/2) | Lower overhead serving | vLLM |

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

---

## Design Notes

### SSM State Prefix Caching (#1)

> vLLM v0.15.0: ~2x speedup by caching Mamba states directly.

DeltaNet (Qwen3.5) and Mamba-2 (Nemotron-H) maintain per-head state matrices computed sequentially. For shared prefixes, the SSM state is deterministic and cacheable.

- Extend `RadixTree` to store SSM state snapshots alongside KV block IDs
- After prefill, save per-layer state (`state_matrix[n_v_heads][v_dim][k_dim]`)
- On cache hit: restore state, skip SSM prefill for cached prefix
- Memory: Qwen3.5 0.8B = 48 layers × 16 heads × 64×64 × 4B = 12 MB/snapshot
- Complexity: model forward must accept "start from saved state" parameter

### Async Scheduler (#2)

> vLLM v0.16.0: 30.8% E2E throughput, 31.8% TPOT improvement.

Current `runSchedulerLoop` calls `step()` synchronously. Proposed:
- Prefill/decode interleaving across requests
- PP overlap: stage 0 processes request B while stage 1 finishes A
- Double-buffer activations tagged by request ID
- High complexity — touches scheduler, model forward, KV cache, transport

### MTP Heads (#5)

> llama.cpp active development. Qwen3.6, DeepSeek models.

Models with built-in multi-token prediction heads output K tokens per forward pass natively. Unlike spec decode, MTP is part of the model architecture. Requires:
- GGUF metadata detection for MTP head count
- Modified forward pass to return K logit vectors
- Acceptance logic similar to spec decode verification

### Profile-Guided Spec Decode (#9)

> llama.cpp Nov 2025.

Instead of fixed K draft tokens, measure actual batch-verify cost at each K during warmup. Choose K dynamically: `E[accepted] × token_value - cost_of_verify(K)`.

### Router Mode (#10)

> llama.cpp server: switch models per request.

Model registry with load/unload on demand. Requires per-model KV cache isolation and reference counting. The `model` field in API requests selects which model to route to.

### Topology-Aware Partitioning (#12)

> Exo: optimal split based on device topology.

At transport setup, exchange device capabilities. Weighted layer assignment: `layers[i] = total × (speed[i] / Σspeeds)`. Store as layer→rank map in PP config.

### Mirostat Sampling (#4)

> llama.cpp: target-entropy sampling.

Controls perplexity by adjusting sampling threshold to maintain target entropy (tau). Track running surprise estimate, adjust mu parameter. When active, top-k/top-p ignored. Two modes: Mirostat 1 and Mirostat 2.0.

---

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

All 7 models share near-identical skeletons. A `ModelBuilder` could save ~600 lines but adds comptime complexity. Deferred because:
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
