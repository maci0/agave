# Future Ideas

> **Recently Implemented** (2026-04/05): Vision/multimodal (SigLIP-2 + SigLIP + Qwen VL),
> TurboQuant+ (asymmetric KV, boundary V, sparse V), BF16 Metal GEMM, split-attention
> (APEX), Gemma 4 E2B/E4B support, thread-parallel vision attention, speculative decoding
> (3 modes: draft model, DDTree, self-speculative), grammar-constrained decoding (GBNF +
> JSON schema), TriAttention KV eviction (Phase 1+2), distributed inference (TP/PP with
> TCP + POSIX shm + NCCL RoCE RDMA transports, dual GB10 PP=2 at 93% of single GPU),
> `agave pull` (HF Hub download with resume), PlanarQuant/IsoQuant/RotorQuant KV cache
> quantization, native GPU sdpaTree for all backends. See BENCHMARKS.md.

## Model init/deinit/forward Abstraction

All 7 models (gemma3, gemma4, qwen35, gpt_oss, nemotron_h, nemotron_nano, glm4) share
near-identical skeletons for init(), deinit(), forward(), resetCache(), and cancel().
Extracting shared logic could save ~30% LoC per model.

### What's duplicated
- **init()**: Read metadata via getArchU32/getMetaU32, allocate working buffers with
  errdefer, allocate KV cache, return struct
- **deinit()**: Free buffers via inline-for tuple, free KV cache
- **forward()**: KV cache overflow check → embedding lookup → layer loop with
  cancellation check → final norm → logits → argmax → increment kv_seq_len
- **resetCache()/cancel()**: Delegate to model_mod helpers (identical in all models)
- **attention()**: pre-norm → QKV projections → optional per-head norms → RoPE → SDPA →
  output projection → post-norm → residual
- **feedForward()**: pre-norm → gate/up projections → activation → down projection →
  post-norm → residual

### Why it's deferred
Each model is self-contained and easy to modify independently. A generic abstraction
would add comptime complexity (parameterized by activation type, rope config, layer type
dispatch, MoE vs dense FFN, per-head norms, sliding window patterns, etc.) that could
hurt readability. The current ~200-line-per-model duplication is acceptable because:
1. Models rarely change once working
2. Each model has unique quirks (Gemma embedding scaling, GPT-OSS attention sinks,
   Qwen3.5 DeltaNet hybrid, GLM4 MLA compression, etc.)
3. Self-contained files are easier to debug

### If pursued
A reasonable approach would be a `ModelBuilder` in model.zig with:
- `allocBuffers(names_and_sizes)` → allocates + errdefer
- `forwardLoop(embed_fn, layer_fn, final_fn)` → handles KV check + cancellation
- `genericAttention(config)` → parameterized by rope_dim, has_bias, has_per_head_norms
- `genericFfn(activation)` → parameterized by comptime activation enum

Estimated savings: ~600 lines across 7 models. Estimated effort: 2-3 days.


## Direct-to-VRAM Model Loading

> **Partially implemented:** Tiered KV cache (VRAM + RAM + SSD) is available via
> `--kv-tiers`. Direct-to-VRAM *weight* loading from NVMe remains future work.

Model weights are currently loaded via `mmap` into system RAM, then uploaded to GPU
memory per-tensor on first use (buffer cache pattern). For large models this means the
full weight file transits: NVMe → CPU RAM → PCIe/fabric → VRAM. Direct storage APIs
can bypass CPU RAM entirely, reading weights from NVMe straight into GPU memory.

### Current loading path
```
NVMe SSD ──mmap──→ CPU RAM ──buffer upload──→ VRAM
           ~7 GB/s            ~32 GB/s (PCIe 4)
```
Bottleneck is the double-copy and CPU involvement. A 27B model (~15 GB quantized)
takes several seconds to fully populate GPU caches.

### Direct storage path
```
NVMe SSD ──GPUDirect/Metal IO──→ VRAM
           ~7 GB/s (per drive, stackable)
```
Single copy, zero CPU involvement. With multiple NVMe drives, bandwidth scales
linearly (4 drives = 28 GB/s). Load time for 15 GB model: ~0.5s vs ~2-4s.

### Platform APIs
- **NVIDIA GPUDirect Storage** (`cuFile`): `cuFileRead()` transfers directly from
  file descriptor to GPU device pointer.
- **Apple Metal**: UMA means "VRAM" is system RAM. Existing zero-copy mmap +
  `newBufferWithBytesNoCopy` is already optimal. No improvement needed.
- **Vulkan**: `VK_KHR_external_memory_fd` + Linux `io_uring` for async reads.
- **AMD ROCm**: Future GDS-equivalent via `hsa_amd_ipc_memory_attach`.

### Considerations
- Alignment requirements: GPUDirect Storage requires 4KB-aligned file offsets.
- UMA platforms (Apple Silicon, NVIDIA GB10): already optimal via zero-copy mmap.
- Compressed formats: decompression must happen on GPU or fall back to CPU-staged.

## Paged SDPA on GPU

> **Partially implemented:** The block allocator (`kvcache/block_allocator.zig`) and
> `PagedKvCache` (`kvcache/manager.zig`) manage paged blocks on the CPU side. Only
> the GPU SDPA kernels need updating to dereference the block table.

Current GPU SDPA kernels only support flat (contiguous) KV cache layouts. With
PagedAttention, KV data is stored in non-contiguous blocks referenced by a block
table. GPU SDPA needs to dereference the block table to find physical KV positions,
adding an indirection layer to the attention kernel.

### What changes
- SDPA kernel accepts a block table (`[]const u32`) alongside K/V cache
- Inner loop iterates over block IDs, loads K/V from physical block addresses
- Block size alignment (16 tokens) naturally maps to SIMD/warp widths

## TriAttention — Frequency-Domain KV Cache Eviction

> **Phase 1+2 implemented** (`--kv-eviction norm` / `--kv-eviction tri`). Phase 1 uses K-norm
> scoring; Phase 2 adds trigonometric frequency-domain scoring with `TriCalibration` stats.
> Periodic compression every 128 tokens. Calibration data generator is future work.

KV cache eviction based on trigonometric frequency analysis of pre-RoPE Q/K vectors.
Instead of scoring tokens by expensive attention computation, it uses statistical
properties of Q/K cluster centers to determine which KV entries are important.
Unimportant entries are pruned from the cache entirely.

### Key results ([Mao et al., 2025](https://github.com/WeianMao/triattention))

- **10.7× KV memory reduction** with accuracy parity on reasoning benchmarks
- **2.5× throughput boost** on AIME25
- Works on Qwen3, DeepSeek-R1, GPT-OSS (models we already support)
- No retraining — inference-only, uses precomputed Q/K frequency statistics

### How it stacks with our existing KV optimizations

| Layer | Technique | Reduction | Status |
|-------|-----------|-----------|--------|
| 1. Bits per entry | TurboQuant turbo4 | 3.8× | ✅ Implemented |
| 2. V-only compression | Asymmetric K=q8_0/V=turbo4 | +quality | ✅ Implemented |
| 3. Skip negligible V | Sparse V dequant (softmax < 1e-6) | +22% decode | ✅ Implemented |
| 4. Evict old entries | TriAttention (norm + frequency) | 10.7× | ✅ Phase 1+2 implemented |
| **Combined** | **1 + 2 + 3 + 4** | **~40×** | |

### Core insight

Pre-RoPE Q and K vectors in reasoning models cluster around fixed frequency centers.
Token importance can be scored cheaply by measuring distance from these centers (via
vector norm and cosine similarity), without computing full attention. This is O(n) per
token vs O(n²) for attention-based importance scoring.

### Implementation status

- **Phase 1 — Norm-based eviction**: ✅ Implemented (`--kv-eviction norm`). K-norm scoring, periodic compression every 128 tokens, attention sink preservation (first 4 positions never evicted).
- **Phase 2 — Trigonometric frequency scoring**: ✅ Implemented (`--kv-eviction tri`). Uses `.cal` calibration files with per-head Q/K frequency statistics. Generated via `agave calibrate model.gguf`.
- **Phase 3 — Dynamic budget**: Not started. Auto-tune KV budget based on available memory, adaptive eviction threshold.

### References
- [TriAttention: KV Cache Compression via Trigonometric Frequency-Domain Analysis (Mao et al., 2025)](https://github.com/WeianMao/triattention)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024)](https://arxiv.org/abs/2402.02750)
- [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression (Liu et al., 2023)](https://arxiv.org/abs/2305.17118)

## Missing GPU Kernels

See [KERNELS.md](KERNELS.md) for the full, current per-backend matrix. Remaining key gaps:
- **All GPU**: NVFP4 (GGUF) — GPU backends use SafeTensors NVFP4 path instead
- **CUDA, ROCm**: DeltaNet recurrence (delegates to CPU)
- **WebGPU**: bf16/f16/fp8 GEMV formats

## Pre-Sharded Weight Files

> Inspired by [Mesh-LLM](https://github.com/Mesh-LLM/mesh-llm)'s zero-transfer weight loading.

Current TP setup: each rank mmap's the full model file and shards at init via
`shardColumnWeight`/`shardRowWeight`. Peak memory = full model size during init.

Pre-sharded files (`model-tp0.gguf`, `model-tp1.gguf`) would let each rank load
only its shard. Benefits:
- Zero init-time sharding overhead
- Peak memory = shard size (not full model)
- Faster startup for large models over slow storage

Implementation: offline `agave shard` subcommand that reads GGUF, splits weight
tensors by TP degree respecting block alignment, writes per-rank GGUF files.
Model code auto-detects sharded format via GGUF metadata (`tp_rank`, `tp_degree`).

## QUIC Transport

> Inspired by Mesh-LLM's QUIC-based inter-node RPC.

QUIC (RFC 9000) over UDP with built-in encryption, 0-RTT connection setup, and
multiplexed streams. Benefits over TCP for distributed inference:
- Lower connection setup latency (1 RTT vs 3 RTT)
- Built-in TLS 1.3 (secure by default)
- Stream multiplexing (activation transfer + control messages on one connection)
- Better performance over lossy/high-latency links (WAN inference)

Low priority since NCCL handles high-perf LAN case. Useful for cross-datacenter
or edge-to-cloud inference. `--transport quic` placeholder already in
`TransportChoice` (currently mapped to `udp`).

## Inter-Model Collaboration (Mixture of Models)

> Inspired by Mesh-LLM's inter-model routing during inference.

Models consult each other during generation:
- **Vision fallback**: text model receives image → routes to vision peer for
  captioning → injects caption into context
- **Uncertainty routing**: model with low-confidence output → second model
  provides alternative → best response selected
- **Loop recovery**: repetition detected → different model generates continuation

Architecture: HTTP server already supports multiple models via scheduler. Add
model-to-model routing rules (e.g., "if image input and no vision encoder,
forward to model X"). Orthogonal to kernel-level perf — purely server/scheduler
layer.

## SSM State Prefix Caching

> Inspired by vLLM v0.15.0 "Mamba prefix caching" (~2x speedup).

DeltaNet (Qwen3.5) and Mamba-2 (Nemotron-H) maintain per-head state matrices
that must be computed sequentially. For shared prefixes (e.g., system prompt),
the SSM state after the prefix is deterministic and can be cached.

### Current behavior
Each request recomputes SSM state from scratch through the full prefix. For a
1000-token system prompt with 48 SSM layers, that's 48,000 sequential state
updates — the slowest part of Qwen3.5 prefill.

### Proposed design
- Extend `RadixTree` to store SSM state snapshots alongside KV cache block IDs
- After prefill, save per-layer SSM state (`state_matrix[n_v_heads][v_dim][k_dim]`)
- On cache hit: restore SSM state from snapshot, skip SSM prefill for cached prefix
- Memory: Qwen3.5 0.8B = 48 layers × 16 heads × 64×64 × 4B = 12 MB per snapshot
- Cache eviction: same LRU as KV blocks, shared eviction cost multiplier

### Complexity
- Model forward pass must accept "start from saved state" parameter
- DeltaNet `ssm_state` buffers must be serializable/restorable
- State depends on token sequence (no partial restore — all-or-nothing per prefix)

## Async Scheduler with PP Overlap

> Inspired by vLLM v0.16.0: 30.8% E2E throughput, 31.8% TPOT improvement.

### Current behavior
`runSchedulerLoop` calls `manager.step()` synchronously — one request's forward
pass blocks all others. With PP, stage 0 is idle while stage 1 computes.

### Proposed design
- **Prefill/decode interleaving**: while one request is in decode (layer-by-layer),
  start prefilling the next request's tokens
- **PP overlap**: stage 0 processes request B while stage 1 finishes request A
- Implementation: double-buffer activations, tag each buffer with request ID
- Scheduler tracks per-request pipeline stage position
- Requires careful KV cache isolation between concurrent requests

### Complexity
High — touches scheduler, model forward, KV cache manager, and transport layer.
Best approached after continuous batching is proven stable.
