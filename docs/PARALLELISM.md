# Agave — Parallelism Design Document

**Status**: Design / Pre-implementation (not yet implemented in the codebase)
**Scope**: Tensor Parallelism (TP), Pipeline Parallelism (PP), KV Cache Transfer
**Backends**: Metal (Apple), CUDA (NVIDIA), ROCm (AMD), Vulkan, CPU

---

## Table of Contents

1. [Overview](#1-overview)
2. [Current Architecture](#2-current-architecture)
3. [Tensor Parallelism (TP)](#3-tensor-parallelism-tp)
4. [Pipeline Parallelism (PP)](#4-pipeline-parallelism-pp)
5. [Backend-Specific Transport](#5-backend-specific-transport)
   - [Metal](#51-metal-apple-silicon) · [CUDA](#52-cuda-nvidia) · [ROCm](#53-rocm-amd) · [Vulkan](#54-vulkan-cross-platform) · [CPU](#55-cpu-multi-socket--numa) · [Multi-Node & RDMA](#56-multi-node-networking--rdma)
6. [KV Cache Partitioning & Transfer](#6-kv-cache-partitioning--transfer)
   - [Under TP](#61-kv-cache-under-tensor-parallelism) · [Under PP](#62-kv-cache-under-pipeline-parallelism) · [Migration](#63-kv-cache-migration-request-scheduling) · [Mixed TP+PP](#64-kv-cache-with-mixed-parallelism-tp--pp)
   - [Offloading (GPU→CPU→NVMe)](#65-kv-cache-offloading-gpu--cpu--nvme) · [Quantized Storage](#66-kv-cache-quantization-in-place) · [Disaggregated Prefill/Decode](#67-disaggregated-prefill-and-decode)
   - [Cross-Node Sharing](#68-cross-node-kv-cache-sharing-distributed-radixattention) · [IPC](#69-ipc-for-multi-process-serving) · [Data Structures](#610-kv-cache-data-structure-changes)
7. [DeviceGroup Abstraction](#7-devicegroup-abstraction)
8. [Synchronization Primitives](#8-synchronization-primitives)
9. [Weight Distribution](#9-weight-distribution)
10. [Hybrid TP + PP](#10-hybrid-tp--pp)
11. [Sequence Parallelism & Context Parallelism](#11-sequence-parallelism--context-parallelism)
12. [Expert Parallelism (EP) for MoE](#12-expert-parallelism-ep-for-moe)
13. [Continuous Batching with Parallelism](#13-continuous-batching-with-parallelism)
14. [Prefill with Tensor Parallelism](#14-prefill-with-tensor-parallelism)
15. [CLI Configuration & Auto-Detection](#15-cli-configuration--auto-detection)
16. [Testing Strategy](#16-testing-strategy)
17. [Model Compatibility Matrix](#17-model-compatibility-matrix)
18. [Startup & Weight Loading](#18-startup--weight-loading)
19. [Implementation Plan](#19-implementation-plan)
20. [KV Cache Offload Granularity](#20-kv-cache-offload-granularity)
21. [Disaggregated Scheduling](#21-disaggregated-scheduling)
22. [RDMA Transport: libibverbs vs libfabric](#22-rdma-transport-layer-libibverbs-vs-libfabric)
23. [CXL Memory Pooling](#23-cxl-memory-pooling)
24. [Open Questions](#24-open-questions)

---

## 1. Overview

Large models (70B+) exceed the memory of a single device. Parallelism strategies split the work across multiple devices to fit in memory and increase throughput.

| Strategy | What gets split | Communication pattern | Primary benefit |
| :--- | :--- | :--- | :--- |
| **Tensor Parallelism (TP)** | Individual weight matrices within a layer | All-reduce after every GEMV/GEMM | Reduces per-device memory; same latency |
| **Pipeline Parallelism (PP)** | Layers across devices | Point-to-point activation transfer between stages | Reduces per-device memory; overlaps compute |
| **TP + PP (Hybrid)** | Both tensors and layers | All-reduce within TP group, P2P between PP stages | Maximum model size; best utilization |

**Design goals:**
- Zero-copy where hardware allows (UMA, NVLink, Infinity Fabric)
- No allocations in the hot path (pre-allocate all comm buffers at init)
- Composable with existing Backend tagged union (not a rewrite)
- comptime-resolved device topology when possible

---

## 2. Current Architecture

Today each model holds a single `Backend` value and processes all layers sequentially on one device:

```
main.zig → Model.forward(token_id)
             │
             ├─ for each layer:
             │    be.gemv(...)     ← one backend, one device
             │    be.rmsNorm(...)
             │    be.sdpa(...)
             │
             └─ be.sync()
                argmax(logits)
```

Key observations:
- `Backend` is `union(enum) { cpu, metal, vulkan, cuda, rocm }` — one active variant
- Models own all weight pointers, working buffers, and KV cache
- `be.sync()` is the only synchronization point (GPU→CPU for argmax)
- KV cache is per-layer flat slices (`[][]f32`) or `PagedKvCache` blocks

**What must change:**
- Models need access to **multiple backends** (one per device)
- Weight tensors must be **shardable** across devices
- Communication primitives (all-reduce, send/recv) must exist per backend
- KV cache must be partitioned (TP) or assigned per-stage (PP)

---

## 3. Tensor Parallelism (TP)

### 3.1 Concept

In tensor parallelism, each participating device holds a **shard** (column or row slice) of every weight matrix. After each GEMV, a collective operation (all-reduce) combines partial results across devices.

For a transformer layer with TP degree `N`:

```
            Device 0              Device 1             Device N-1
            ────────              ────────             ──────────
Input x ──► W_q[:,0:d/N] @ x    W_q[:,d/N:2d/N] @ x   ...
            W_k[:,0:d/N] @ x    W_k[:,d/N:2d/N] @ x   ...
            W_v[:,0:d/N] @ x    W_v[:,d/N:2d/N] @ x   ...
                 │                    │                    │
            local SDPA           local SDPA            local SDPA
                 │                    │                    │
            W_o[0:d/N,:] @ attn  W_o[d/N:2d/N,:] @ attn  ...
                 │                    │                    │
                 └────── all-reduce (sum) ─────────────────┘
                              │
                        combined output
```

### 3.2 Column-Parallel vs Row-Parallel

Megatron-LM established the standard split pattern for transformers:

**Attention block:**
1. **Q, K, V projections** → column-parallel (each device gets `n_heads/TP` heads)
2. **Output projection (W_o)** → row-parallel (each device holds rows, outputs are summed)
3. **Single all-reduce** after W_o (not after each of Q/K/V — they're independent)

**FFN block (SwiGLU):**
1. **W_gate, W_up** → column-parallel (each device gets `ff_dim/TP` columns)
2. **W_down** → row-parallel
3. **Single all-reduce** after W_down

**MoE block:**
- **Expert parallelism** (EP): distribute entire experts across devices (expert `i` lives on device `i % TP`). Router runs on all devices; each device computes only its assigned experts. All-to-all exchange of tokens→experts and results→tokens.
- Alternative: TP within each expert (for very large experts). Rarely needed at current model sizes.

**Key insight:** A transformer layer requires exactly **2 all-reduces per layer** (one after attention, one after FFN) regardless of TP degree. This is the Megatron pattern.

### 3.3 Head-Level Partitioning

For GQA models, the natural TP split is by **attention head groups**:

| Model | n_heads | n_kv_heads | Valid TP degrees |
| :--- | :--- | :--- | :--- |
| Gemma3 1B | 4 | 1 | 1 |
| Qwen3.5 0.8B | 16 | 4 | 1, 2, 4 |
| GPT-OSS | 64 | 8 | 1, 2, 4, 8 |
| Nemotron-H | 40 | 8 | 1, 2, 4, 8 |
| Llama-3.1-70B | 64 | 8 | 1, 2, 4, 8 |

TP degree must divide both `n_heads` and `n_kv_heads`. Validate at init:

```zig
if (n_heads % tp_degree != 0 or n_kv_heads % tp_degree != 0)
    return error.InvalidTPDegree;
```

### 3.4 Embedding & Final Projection

- **Embedding table**: replicated on all TP ranks (small relative to model). Each device does its own lookup — no communication needed.
- **Final logits projection (lm_head)**: column-parallel across vocab dimension. Each device produces `vocab_size/TP` logits. Gather (not reduce) to assemble full logits before argmax. Alternatively, compute local argmax per shard and reduce the `(value, index)` pair — avoids transferring full logits.

### 3.5 Norms

RMSNorm and L2Norm operate on the **full** hidden state. Two options:

1. **All-gather before norm**: each device gets full hidden vector, computes norm locally. Adds communication but is simple.
2. **Partial norm + all-reduce**: each device computes local sum-of-squares, all-reduce to get global sum, then each device applies `1/sqrt(global_ss)` to its local shard. More efficient, no extra memory.

Option 2 is preferred — it avoids ever materializing the full hidden vector on any single device.

### 3.6 Data Layout for Sharded Weights

Weights must be loadable in sharded form without reading the full tensor. Two approaches:

**Approach A: Load-and-shard (simple, higher peak memory)**
1. Each device mmap's the full GGUF/SafeTensors file
2. At init, each device extracts its column/row shard and discards the rest
3. Peak memory = full model size briefly during init

**Approach B: Pre-sharded weights (optimal, zero overhead)**
1. An offline tool splits weight files into per-rank shards: `model-tp0.gguf`, `model-tp1.gguf`, ...
2. Each device loads only its shard
3. Peak memory = sharded size from the start

Approach A is simpler for the initial implementation. Approach B should be a follow-up tool (`agave-shard`).

For quantized weights (Q4_K, NVFP4, etc.), sharding must respect **block boundaries**. Q4_0 uses 32-element blocks; the shard boundary must fall on a block-aligned column. This is usually satisfied because `n_heads` is a multiple of the TP degree and `head_dim` is a multiple of 32/64/128.

---

## 4. Pipeline Parallelism (PP)

### 4.1 Concept

In pipeline parallelism, the model's layers are distributed across devices in contiguous stages. Each device processes its assigned layers and forwards the activation to the next stage.

```
Device 0 (layers 0-12)    Device 1 (layers 13-25)
─────────────────────     ──────────────────────
token → emb → L0..L12 ──►  L13..L25 → logits → argmax
         ▲                      │
         └──── next token ──────┘
```

### 4.2 Stage Assignment

Layers are divided into `PP` contiguous stages. Simple uniform split:

```zig
fn stageForLayer(layer: u32, n_layers: u32, pp_degree: u32) u32 {
    return layer * pp_degree / n_layers;
}
```

For hybrid models (Qwen3.5, Nemotron-H, Nemotron-Nano) with mixed layer types (attention, SSM, MoE), the split should be **compute-balanced** rather than uniform. MoE layers are heavier than attention layers; SSM layers are lighter. A cost model should guide the split:

| Layer type | Relative cost (single token decode) |
| :--- | :--- |
| Attention | 1.0× |
| SSM (DeltaNet/Mamba-2) | 0.3-0.5× |
| MoE (top-K routing) | 1.5-3.0× (depends on K and expert size) |

### 4.3 Micro-Batching (Prefill Overlap)

During single-token decode, pipeline parallelism introduces **bubbles** — Device 1 is idle while Device 0 processes, and vice versa. The pipeline utilization for decode is only `1/PP`.

Mitigation strategies:
1. **Micro-batching during prefill**: Split the prompt into micro-batches. While Device 0 processes micro-batch 2, Device 1 processes micro-batch 1. This fills the pipeline during the compute-intensive prefill phase.
2. **Continuous batching**: Process multiple independent requests simultaneously. While Device 0 works on request A's next token, Device 1 finishes request B's previous token.
3. **Accept the bubble for decode**: For decode, the per-token latency is low enough that the bubble is tolerable. The primary benefit of PP is fitting the model in memory, not improving decode throughput.

### 4.4 Activation Transfer

Between pipeline stages, only the **hidden state vector** needs to be transferred: `n_embd` × `sizeof(f32)` bytes per token.

| Model | n_embd | Transfer size (f32) | Transfer size (bf16) |
| :--- | :--- | :--- | :--- |
| Gemma3 1B | 1152 | 4.5 KB | 2.25 KB |
| Llama-3.1-70B | 8192 | 32 KB | 16 KB |
| GPT-OSS | 2880 | 11.25 KB | 5.6 KB |

These are tiny — the overhead is dominated by latency, not bandwidth. On high-bandwidth interconnects (NVLink: 900 GB/s, Infinity Fabric: 896 GB/s), this is negligible. Even on PCIe Gen4 x16 (32 GB/s), 32 KB takes ~1 μs.

### 4.5 First/Last Stage Special Cases

- **Stage 0 (first)**: owns the embedding table, performs embedding lookup + scaling
- **Stage N-1 (last)**: owns the lm_head (final projection), computes logits and argmax
- All intermediate stages: receive hidden state, process layers, send hidden state

---

## 5. Backend-Specific Transport

### 5.1 Metal (Apple Silicon)

**Multi-device scenario**: Apple Silicon Ultra chips have dual-die architecture exposing 2 GPU devices. Future Apple Silicon may expose more. Additionally, TP/PP across Mac Studio + Mac Pro via Thunderbolt is possible but higher latency.

**Same-chip (UMA) transport:**
- Both GPU dies share unified memory — zero-copy is native
- Use `MTLSharedEvent` for cross-queue synchronization between the two dies
- No data transfer needed — both dies read/write the same physical memory
- `MTLBuffer` created on one device can be shared to the other via `newSharedTextureHandle` or `makeAliasable`

```
Die 0 (GPU 0)                    Die 1 (GPU 1)
─────────────                    ─────────────
Command Queue 0                  Command Queue 1
    │                                │
    ├── encode layer 0-12           ├── (waiting on SharedEvent)
    ├── signal SharedEvent(1)       ├── encode layer 13-25
    │   (hidden state in UMA)       ├── signal SharedEvent(2)
    └── wait SharedEvent(2)         └──
```

**Implementation primitives:**
- `MTLDevice.newSharedEvent()` → create sync primitive
- `MTLCommandBuffer.encodeSignalEvent(event, value)` → signal from GPU timeline
- `MTLCommandBuffer.encodeWaitForEvent(event, value)` → wait in GPU timeline
- For all-reduce: `MPSMatrixSum` or custom MSL kernel writing to shared buffer + barrier

**Multi-Mac transport (Thunderbolt/network):**
- Treat as discrete — serialize activation to buffer, DMA via `IOSurface` sharing or fall back to TCP/RDMA
- Higher latency (~10-50 μs), best suited for PP only (not TP)

### 5.2 CUDA (NVIDIA)

**Multi-GPU transport options (ranked by performance):**

| Method | Bandwidth | Latency | Use case |
| :--- | :--- | :--- | :--- |
| NVLink (same node) | 900 GB/s (H100) | ~1 μs | TP all-reduce |
| NVSwitch (same node, 8-GPU) | 900 GB/s per GPU | ~1 μs | TP all-reduce |
| PCIe P2P (same node) | 32 GB/s (Gen4) | ~5 μs | PP activation transfer |
| GPUDirect RDMA (cross-node) | 400 Gbps InfiniBand | ~2-5 μs | Multi-node PP |
| Host staging (fallback) | 2× PCIe BW | ~10 μs | When P2P unavailable |

**Implementation primitives:**

```
// Peer-to-peer memory access
cuDeviceCanAccessPeer(&can, dev0, dev1)   // Check P2P support
cuCtxEnablePeerAccess(peer_ctx, 0)        // Enable direct access
cuMemcpyPeerAsync(dst, ctx1, src, ctx0, size, stream)  // Async P2P copy

// All-reduce via NCCL (recommended for TP)
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream)

// Low-level: custom all-reduce via shared buffers + barriers
// For small messages (< 256 KB), a custom ring or tree all-reduce
// using cuMemcpyPeerAsync can outperform NCCL by avoiding launch overhead
```

**NCCL integration:**
- NCCL is a C library — linking it statically would violate the "Pure Zig Rule," but dynamic loading at runtime is acceptable since NCCL is already installed on virtually all multi-GPU CUDA systems.
- **Option A (custom Zig ring)**: Implement a minimal all-reduce in Zig using CUDA driver API P2P copies + barriers. Sufficient for 2-8 GPU TP. Lower dependency footprint, full control over fusion and scheduling.
- **Option B (NCCL via dlopen)**: Load `libnccl.so` at runtime via `std.DynLib`. No compile-time linking, no vendored C code. NCCL's tree/ring hybrid outperforms a naive ring at 8+ GPUs and is battle-tested for multi-node InfiniBand/RoCE. Recommended for production multi-node serving.
- **Recommendation**: Support both. Use custom Zig ring as the default (zero external dependencies), auto-detect and prefer NCCL when available via `dlopen`. A `--nccl` CLI flag or `AGAVE_USE_NCCL=1` env var enables explicit selection.

**Custom Zig all-reduce for CUDA (default for ≤8 GPUs):**

Ring all-reduce for `N` GPUs with `M` elements:
1. Split buffer into `N` chunks
2. Each GPU sends chunk `i` to GPU `(rank+1) % N`, receives into chunk `(rank-1) % N`
3. Repeat `N-1` times for reduce-scatter, `N-1` times for all-gather
4. Total data moved per GPU: `2M(N-1)/N` — same as NCCL ring

```zig
// Pseudo-code for one ring step
fn ringReduceStep(
    local_buf: CuDevicePtr,
    peer_buf: CuDevicePtr,
    chunk_size: usize,
    send_chunk: usize,
    recv_chunk: usize,
    peer_ctx: CuContext,
    stream: CuStream,
) void {
    // 1. Async copy local[send_chunk] → peer[recv_chunk]
    cuMemcpyPeerAsync(
        peer_buf + recv_chunk * chunk_size * @sizeOf(f32),
        peer_ctx,
        local_buf + send_chunk * chunk_size * @sizeOf(f32),
        local_ctx,
        chunk_size * @sizeOf(f32),
        stream,
    );
    // 2. Launch reduce kernel on peer to sum recv_chunk
    // (peer does this in its own stream after event wait)
}
```

### 5.3 ROCm (AMD)

**Multi-GPU transport:**

| Method | Bandwidth | Use case |
| :--- | :--- | :--- |
| Infinity Fabric (MI300X) | 896 GB/s | TP all-reduce |
| XGMI (MI200/MI250) | 400 GB/s | TP all-reduce |
| PCIe P2P | 32 GB/s | PP activation transfer |

**Implementation primitives:**

```
// HSA inter-agent communication
hsa_agent_t agents[N];         // One per GPU
hsa_signal_t signals[N];       // Inter-GPU barriers

// HIP peer-to-peer
hipDeviceCanAccessPeer(&can, dev0, dev1)
hipDeviceEnablePeerAccess(peer_dev, 0)
hipMemcpyPeerAsync(dst, dev1, src, dev0, size, stream)

// RCCL (ROCm's NCCL equivalent)
rcclAllReduce(sendbuf, recvbuf, count, rcclFloat, rcclSum, comm, stream)
```

**Same considerations as CUDA**: support both a custom Zig ring all-reduce using HIP P2P (default, zero dependencies) and RCCL via `dlopen("librccl.so")` (auto-detected, preferred when available for 8+ GPUs or multi-node).

**MI300X special case**: The MI300X has 8 XCDs (compute dies) connected by Infinity Fabric on a single package. Each XCD has its own HBM stack. AMD exposes these either as 1 logical device or as 8 partitioned devices. When partitioned, intra-package TP is essentially free (shared memory controller, 896 GB/s fabric). This is similar to Apple Ultra's dual-die setup but at higher bandwidth.

### 5.4 Vulkan (Cross-Platform)

**Multi-GPU support:**

Vulkan has native multi-device support via **device groups** (`VK_KHR_device_group`):
- `vkEnumeratePhysicalDeviceGroups` → discover multi-GPU configurations
- `VkDeviceGroupDeviceCreateInfo` → create a logical device spanning multiple physical devices
- Memory can be allocated with `VkMemoryAllocateFlagsInfo` + `VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT` to target specific physical devices

**Transport options:**

| Method | Mechanism | Use case |
| :--- | :--- | :--- |
| Device group peer memory | `vkCmdCopyBuffer` with device masks | TP/PP same node |
| External memory (`VK_KHR_external_memory`) | Export/import `VkDeviceMemory` | Cross-process sharing |
| Host-visible staging buffers | Map→write→unmap→copy | Fallback when P2P unavailable |
| Compute shader reduction | Custom SPIR-V kernel | All-reduce for TP |

**All-reduce implementation:**

Vulkan lacks a built-in collective. Two approaches:

1. **Buffer-copy ring**: Use `vkCmdCopyBuffer` in a ring pattern between device group members, with pipeline barriers for synchronization. Simple but requires `N-1` submissions.

2. **Shared-buffer compute**: Allocate a buffer visible to all devices (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` on all device masks). Launch a compute shader that reads partial sums and writes the reduced result. Single dispatch, but requires peer memory support.

```
// Vulkan sync between devices in a device group:
VkSemaphore timeline_sem;  // VK_SEMAPHORE_TYPE_TIMELINE
vkCmdPipelineBarrier2(cmd, &dep_info);  // Memory barrier
vkQueueSubmit2(queue, &submit_info);    // With timeline semaphore signal/wait
```

**Limitations:**
- Not all Vulkan drivers support device groups (primarily NVIDIA multi-GPU on Linux)
- Peer-to-peer buffer access may not be available on all hardware
- Fallback: host-staged copies (slower but universal)

### 5.5 CPU (Multi-Socket / NUMA)

**Multi-"device" = NUMA nodes.** A dual-socket server has 2 NUMA nodes, each with its own memory controller and CPU cores.

**Transport: shared memory (zero-cost)**
- All NUMA nodes share the same virtual address space — no copies needed
- **Critical**: allocations must be NUMA-aware. Weight shards for NUMA node `i` should be allocated on node `i`'s local memory. Remote memory access across the interconnect (UPI/Infinity Fabric) has 2-3× higher latency.

**Implementation:**

```zig
// NUMA-aware allocation
const numa_node = getCurrentNumaNode();
const local_mem = numaAlloc(size, numa_node);  // mmap + mbind

// Thread pinning for NUMA locality
std.Thread.setCpuAffinity(thread, core_set_for_numa_node);

// All-reduce = direct memory read + SIMD sum
// No copies needed — just read peer's buffer and accumulate
fn cpuAllReduce(bufs: [][]f32, result: []f32) void {
    @memcpy(result, bufs[0]);
    for (bufs[1..]) |peer| {
        for (result, peer) |*r, p| r.* += p;  // SIMD auto-vectorized
    }
}
```

**NUMA topology detection:**
- Linux: parse `/sys/devices/system/node/` or use `libnuma`
- macOS: single NUMA domain (Apple Silicon is always uniform)
- Windows: `GetNumaNodeProcessorMask`

---

## 6. KV Cache Partitioning & Transfer

### 6.1 KV Cache Under Tensor Parallelism

With TP, each device handles `n_kv_heads / TP` heads. The KV cache is **naturally partitioned** — each device stores only its local KV heads:

```
TP=2, n_kv_heads=8, head_dim=128

Device 0: KV cache for heads 0-3 → [max_seq_len × 4 × 128] f32
Device 1: KV cache for heads 4-7 → [max_seq_len × 4 × 128] f32
```

**No KV cache transfer needed during normal decode** — each device writes to its own KV shard and reads only its shard during SDPA.

**KV cache transfer is needed for:**
- **Request migration** (moving a request from one TP group to another for load balancing)
- **Prefix cache sharing** (RadixAttention across TP groups)
- **Checkpointing** (saving/restoring state)

### 6.2 KV Cache Under Pipeline Parallelism

With PP, each stage owns the KV cache for its layers. No cross-device KV sharing is needed during normal inference — each device reads/writes KV cache only for its own layers.

```
PP=2, 26 layers

Device 0 (layers 0-12): KV cache layers 0-12  → local memory
Device 1 (layers 13-25): KV cache layers 13-25 → local memory
```

**No KV cache transfer during decode.** The only cross-device data is the activation vector between stages.

### 6.3 KV Cache Migration (Request Scheduling)

When a serving system needs to migrate a request from one device group to another (e.g., for load balancing or preemption), the full KV cache for that request must be transferred.

**Transfer size:**
```
KV cache per request = 2 × n_layers × seq_len × n_kv_heads × head_dim × sizeof(dtype)
```

| Model | KV per token (f32) | 2048 tokens |
| :--- | :--- | :--- |
| Gemma3 1B | 2 × 26 × 4 × 256 × 4 = 213 KB | 426 MB |
| Llama-3.1-70B | 2 × 80 × 8 × 128 × 4 = 655 KB | 1.3 GB |

**Optimization: KV cache quantization for transfer.** Quantize KV values to FP8 or even Q4 during transfer (not during compute). This halves or quarters the transfer size at minimal quality loss for cached values.

**Paged KV cache migration:** Only transfer **occupied pages**. With `PagedKvCache`, walk the `SeqBlockTable` and transfer only blocks with `used > 0`.

**RadixAttention and migration:** If migrating to a device that shares the same prefix tree (e.g., within the same node), only the **unique suffix** needs to be transferred. The shared prefix blocks are already present via the radix tree's reference counting.

### 6.4 KV Cache with Mixed Parallelism (TP + PP)

With TP=2, PP=2 on 4 devices:

```
       TP Group 0          TP Group 1
       ──────────          ──────────
PP 0:  Dev 0 (L0-12, H0-3)  Dev 1 (L0-12, H4-7)
PP 1:  Dev 2 (L13-25, H0-3) Dev 3 (L13-25, H4-7)
```

Each device owns KV cache for its PP layers and TP head shard. Migration requires transferring from all `TP` devices in a stage.

### 6.5 KV Cache Offloading (GPU → CPU → NVMe)

When GPU memory is exhausted (many concurrent requests, long contexts), KV cache pages can be **offloaded** to cheaper storage tiers:

```
Tier 0: GPU HBM/VRAM    — fastest, most expensive, limited
Tier 1: CPU DRAM         — 10-50× larger, ~10× slower access
Tier 2: NVMe SSD         — 100-1000× larger, ~100× slower access
```

**Page-level offloading with PagedKvCache:**

The `PagedKvCache` block structure is ideal for tiered offloading. Individual blocks can be evicted to CPU DRAM when GPU memory pressure is high, and fetched back when needed for attention.

```zig
pub const CacheBlock = struct {
    keys: []f32,
    values: []f32,
    used: u16 = 0,
    ref_count: u16 = 1,
    /// Storage tier for this block.
    tier: StorageTier = .gpu,
    /// Handle to offloaded data (CPU pointer or NVMe offset).
    offload_handle: ?OffloadHandle = null,
};

pub const StorageTier = enum { gpu, cpu, nvme };
```

**Offload/fetch pipeline:**
1. **Eviction**: When GPU free blocks < threshold, select LRU blocks → async `cudaMemcpyAsync(D2H)` → mark tier=cpu
2. **Prefetch**: Before SDPA, check if any required KV blocks are offloaded → async `cudaMemcpyAsync(H2D)` → wait before kernel launch
3. **NVMe tier**: CPU DRAM overflow → `io_uring` async write to memory-mapped file → mark tier=nvme

**Overlap with compute**: Prefetch KV blocks for layer `L+1` while computing layer `L`. The scheduler knows which blocks each request needs (from the block table) and can issue prefetches early.

**GPUDirect Storage (NVIDIA):** On supported systems, NVMe↔GPU transfers can bypass CPU entirely via `cuFile` API (GDS). This eliminates the CPU DRAM staging hop:

```
Without GDS: NVMe → CPU DRAM → GPU HBM  (2 copies)
With GDS:    NVMe → GPU HBM             (1 copy, CPU not involved)
```

### 6.6 KV Cache Quantization (In-Place)

Separate from transfer quantization (§6.3), KV values can be **stored** in lower precision to reduce memory footprint permanently. This is increasingly standard in production systems.

| KV dtype | Bytes/element | Memory vs f32 | Quality impact |
| :--- | :--- | :--- | :--- |
| f32 | 4 | 1.0× | Baseline |
| f16/bf16 | 2 | 0.5× | Negligible |
| FP8 (E4M3) | 1 | 0.25× | <0.5% perplexity increase |
| INT8 (per-head scale) | 1 | 0.25× | <0.5% with calibration |
| INT4 (per-group scale) | 0.5 | 0.125× | ~1-2% perplexity increase |

**Implementation**: Quantize K/V after projection, before writing to cache. Dequantize inside the SDPA kernel (same pattern as weight dequantization — never materialize full f32 in the hot path):

```zig
// Write quantized KV
const k_fp8 = quantizeToFp8(k_buf); // In register, no allocation
writeKvCache(layer, pos, k_fp8, v_fp8);

// SDPA kernel reads quantized, dequants on the fly
sdpaWithQuantizedKv(q, kv_cache_fp8, output, ...);
```

**Per-head dynamic scaling**: Each head maintains a running max absolute value. Scale factor = `max_abs / 127.0` (INT8) or `max_abs / 448.0` (FP8 E4M3). Stored alongside the cache block (1 f32 per head per block — negligible overhead).

**Memory savings example (Llama-3.1-70B, 2048 tokens, 1 request):**
- f32 KV: 1.3 GB
- FP8 KV: 327 MB (4× reduction)
- With TP=4: 82 MB per device

### 6.7 Disaggregated Prefill and Decode

A major architecture for production serving (Mooncake, DistServe, Splitwise): **prefill** and **decode** run on separate device pools, with KV cache transferred between them.

**Motivation:**
- Prefill is compute-bound (GEMM, processes many tokens at once)
- Decode is memory-bandwidth-bound (GEMV, one token at a time)
- Mixing them on the same GPU causes interference — prefill spikes block decode latency
- Different GPU types are optimal: high-FLOPS for prefill, high-bandwidth for decode

```
Prefill Pool                          Decode Pool
(high-FLOPS GPUs)                     (high-BW GPUs)
┌──────────────────┐                  ┌──────────────────┐
│  Process prompt   │                  │  Generate tokens  │
│  Build KV cache   │── KV transfer ─→│  Read KV cache    │
│  for all layers   │   (RDMA/P2P)    │  Append new K/V   │
└──────────────────┘                  └──────────────────┘
```

**KV transfer is the critical path.** The entire KV cache for all layers must reach the decode pool before the first token can be generated.

**Transfer strategies:**
1. **Bulk transfer after prefill**: Send entire KV cache at once. Simple, but TTFT includes full transfer time.
2. **Layer-pipelined transfer**: As each layer completes prefill, immediately begin transferring its KV cache to the decode pool. Decode can start processing layer 0 while layer N is still transferring. Reduces effective TTFT.
3. **Streamed with compression**: Quantize KV to FP8 before transfer (4× bandwidth savings). Decompress on decode side.

**Transfer time estimates (layer-pipelined, FP8 KV, InfiniBand NDR 400Gbps):**

| Model | Prompt len | KV size (FP8) | Transfer time | vs TTFT |
| :--- | :--- | :--- | :--- | :--- |
| Llama-70B | 2048 | 327 MB | 6.5 ms | ~15% of prefill |
| Llama-70B | 8192 | 1.3 GB | 26 ms | ~10% of prefill |
| Llama-405B | 2048 | 1.6 GB | 32 ms | ~8% of prefill |

**With GPUDirect RDMA** (§5.6.2), the KV transfer bypasses CPU entirely: prefill GPU → NIC → network → NIC → decode GPU. Without it, each side adds a CPU staging copy.

**PagedKvCache integration**: Transfer individual blocks rather than contiguous buffers. The decode pool allocates blocks from its own `PagedKvCache` and receives data directly into them. Block tables are rebuilt on the decode side.

### 6.8 Cross-Node KV Cache Sharing (Distributed RadixAttention)

For multi-node serving clusters, prefix cache sharing across nodes avoids redundant prefill of common system prompts.

**Architecture:**

```
Node 0                    Node 1                    Node 2
┌────────────────┐       ┌────────────────┐       ┌────────────────┐
│ Local RadixTree │       │ Local RadixTree │       │ Local RadixTree │
│ (prefix cache)  │       │ (prefix cache)  │       │ (prefix cache)  │
└───────┬────────┘       └───────┬────────┘       └───────┬────────┘
        │                        │                        │
        └────────── Global Prefix Registry ───────────────┘
                   (distributed hash table)
```

**Global prefix registry**: A lightweight metadata service (not the KV data itself) that tracks which nodes have which prefixes cached. When a request arrives at Node 1 with a prefix that's only cached on Node 0:

1. Query registry: "who has prefix hash `0xABCD`?"
2. Registry responds: "Node 0, blocks 42-58"
3. Node 1 requests KV blocks from Node 0 via RDMA
4. Node 0 sends blocks directly from its `PagedKvCache` → Node 1's GPU
5. Node 1 inserts into its local RadixTree (now both nodes have the prefix)

**Consistency**: The registry is eventually consistent. Eviction notifications are best-effort. If a node evicts a prefix that the registry still lists, the requesting node falls back to re-prefilling.

**When this matters**: Large-scale serving with common system prompts (customer support, coding assistants). A 2048-token system prompt cached across the cluster saves ~50ms of prefill per request.

### 6.9 IPC for Multi-Process Serving

Production serving often uses multiple processes (one per GPU) rather than threads, for fault isolation and independent scaling.

**CUDA IPC:**
```
// Process A (owner): export GPU memory handle
cuIpcGetMemHandle(&handle, dev_ptr);
// Send handle to Process B via Unix socket / shared memory

// Process B (consumer): import and map into its address space
cuIpcOpenMemHandle(&dev_ptr_b, handle, cudaIpcMemLazyEnablePeerAccess);
// dev_ptr_b now points to the same physical GPU memory — zero copy
```

**HIP IPC (ROCm):** Identical API — `hipIpcGetMemHandle` / `hipIpcOpenMemHandle`.

**Metal IPC:** Use `IOSurface` to share GPU buffers between processes on macOS. `MTLDevice.newBuffer(iosurface:)` wraps an `IOSurface` as a Metal buffer in each process.

**Vulkan IPC:** `VK_KHR_external_memory_fd` (Linux) or `VK_KHR_external_memory_win32` (Windows). Export `VkDeviceMemory` as a file descriptor, import in another process.

**Linux DMA-BUF:** Generic mechanism for sharing GPU buffers across processes and even across different GPU APIs. Both CUDA and ROCm can export/import via `dma_buf_fd`. Vulkan supports it natively via `VK_EXT_external_memory_dma_buf`.

**Use cases in Agave:**
- Multi-process model server where each GPU runs in its own process
- KV cache sharing between worker processes via IPC memory
- Weight sharing (mmap the same file, or share GPU buffers) to avoid duplicating model weights in each process

### 6.10 KV Cache Data Structure Changes

The current `KvCache` (flat slices) and `PagedKvCache` need a **device assignment** field:

```zig
pub const DistributedKvCache = struct {
    /// Per-device KV caches, indexed by device rank.
    shards: []ShardedCache,
    tp_degree: u32,
    pp_degree: u32,
    /// Global prefix registry for cross-node RadixAttention sharing.
    prefix_registry: ?*PrefixRegistry = null,

    pub const ShardedCache = struct {
        /// Which device this shard lives on.
        device_rank: u32,
        /// Layer range [start, end) assigned to this device.
        layer_start: u32,
        layer_end: u32,
        /// Head range [start, end) assigned to this device.
        head_start: u32,
        head_end: u32,
        /// The actual cache (paged or flat).
        cache: PagedKvCache,
        /// KV precision (f32, f16, fp8_e4m3, int8).
        kv_dtype: KvDType = .f32,
        /// Per-head dynamic scale factors (for quantized KV).
        head_scales: ?[]f32 = null,
        /// CPU-side shadow for offloaded blocks.
        cpu_shadow: ?[]CacheBlock = null,
    };

    pub const KvDType = enum { f32, f16, bf16, fp8_e4m3, int8 };
};
```

### 5.6 Multi-Node Networking & RDMA

Single-node parallelism covers 1-8 GPUs. Beyond that, multi-node communication becomes critical.

#### 5.6.1 Network Fabrics

| Fabric | Bandwidth | Latency | Typical use |
| :--- | :--- | :--- | :--- |
| InfiniBand HDR | 200 Gbps (25 GB/s) | ~1 μs | HPC clusters |
| InfiniBand NDR | 400 Gbps (50 GB/s) | ~1 μs | Modern GPU clusters (H100) |
| RoCE v2 (RDMA over Converged Ethernet) | 100-400 Gbps | ~2-3 μs | Ethernet-based GPU clusters |
| Thunderbolt 4/5 | 40-120 Gbps (5-15 GB/s) | ~5-10 μs | Apple multi-Mac setups |
| TCP/IP (fallback) | 10-100 Gbps | ~20-50 μs | Commodity networks |

#### 5.6.2 GPUDirect RDMA (NVIDIA)

GPUDirect RDMA allows a network adapter (InfiniBand HCA) to read/write GPU memory directly, bypassing CPU staging entirely. This is critical for large KV cache transfers across nodes.

**Setup requirements:**
- `nvidia_peermem` kernel module loaded (bridges CUDA memory and InfiniBand verbs)
- GPU memory must be registered with both CUDA and the RDMA adapter
- Requires MLNX_OFED drivers or equivalent

**Data flow comparison:**

```
Without GPUDirect RDMA (3 copies):
  GPU → cudaMemcpy → CPU buf → ibv_post_send → NIC → network
  network → NIC → ibv_post_recv → CPU buf → cudaMemcpy → GPU

With GPUDirect RDMA (0 CPU copies):
  GPU → nvidia_peermem → NIC → network
  network → NIC → nvidia_peermem → GPU
```

**Implementation via libibverbs (Zig `dlopen`):**

```zig
// Register GPU memory for RDMA
const gpu_ptr = cuMemAlloc(size);
const mr = ibv_reg_mr(pd, @ptrFromInt(gpu_ptr), size,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

// Post RDMA write directly from GPU memory
var sge = ibv_sge{ .addr = gpu_ptr, .length = size, .lkey = mr.lkey };
var wr = ibv_send_wr{ .sg_list = &sge, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE };
ibv_post_send(qp, &wr, &bad_wr);
```

**Integration approach:** Like NCCL, `libibverbs` is a C library. Use `std.DynLib` to `dlopen("libibverbs.so")` at runtime. Wrap the subset of verbs needed: `ibv_open_device`, `ibv_alloc_pd`, `ibv_reg_mr`, `ibv_create_qp`, `ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq`.

#### 5.6.3 ROCm RDMA

AMD's equivalent is **ROCm RDMA** (`amdgpu` + `amd_peer_direct` kernel module). Same concept — HIP device memory registered with InfiniBand verbs for direct NIC↔GPU transfers. API is identical `libibverbs` since the transport layer is the same.

#### 5.6.4 Topology-Aware Placement

Multi-node clusters have specific GPU↔NIC affinity. A DGX H100 has 8 GPUs and 8 NICs, each GPU paired with a specific NIC on the same PCIe switch. Sending data through the wrong NIC adds a PCIe hop.

```
Typical DGX H100 topology:
  NIC 0 ←→ GPU 0,1 (PCIe switch 0)    TP group A
  NIC 1 ←→ GPU 2,3 (PCIe switch 1)    TP group B
  NIC 2 ←→ GPU 4,5 (PCIe switch 2)    TP group C
  NIC 3 ←→ GPU 6,7 (PCIe switch 3)    TP group D
  GPU 0-7 all connected via NVSwitch (intra-node TP)
```

**Rail-optimized all-reduce** (used by Megatron, DeepSpeed): for cross-node TP, each GPU sends its shard through its **paired NIC only**. This maximizes per-NIC bandwidth instead of funneling all traffic through one NIC.

We should detect GPU↔NIC affinity at init via:
- CUDA: `cudaDeviceGetPCIBusId` + parse sysfs for NIC on same PCIe switch
- ROCm: `hipDeviceGetPCIBusId` + same sysfs approach
- Store in `DeviceGroup.Topology` for routing decisions

---

## 7. DeviceGroup Abstraction

### 7.1 DeviceGroup

A new `DeviceGroup` abstraction wraps multiple backends and tracks topology:

```zig
pub const DeviceGroup = struct {
    devices: []Device,
    topology: Topology,
    tp_degree: u32,
    pp_degree: u32,
    allocator: Allocator,

    pub const Device = struct {
        rank: u32,
        backend: Backend,
        /// Pre-allocated send/recv buffers for activation transfer.
        send_buf: []f32,
        recv_buf: []f32,
        /// Pre-allocated reduction buffer for TP all-reduce.
        reduce_buf: []f32,
    };

    pub const Topology = enum {
        /// All devices share memory (UMA, same-package)
        shared_memory,
        /// Direct peer-to-peer access (NVLink, XGMI, Infinity Fabric)
        peer_to_peer,
        /// Must go through host staging
        host_staged,
        /// Network (multi-node)
        network,
    };

    /// Returns devices in the same TP group for a given rank.
    pub fn tpPeers(self: *const DeviceGroup, rank: u32) []const Device { ... }

    /// Returns the next PP stage device for a given rank.
    pub fn ppNext(self: *const DeviceGroup, rank: u32) ?*const Device { ... }

    /// Returns the previous PP stage device for a given rank.
    pub fn ppPrev(self: *const DeviceGroup, rank: u32) ?*const Device { ... }
};
```

### 7.2 Integrating with the Existing Backend

The `Backend` union remains unchanged — each `Device` in the group holds its own `Backend`. The model's forward pass changes from:

```zig
// Current: single backend
be.gemv(x, w, y, n, k);
```

To:

```zig
// Parallel: each device does its shard, then reduce
for (group.tpPeers(my_rank)) |peer| {
    peer.backend.gemv(x, w_shard[peer.rank], y_partial[peer.rank], n/tp, k);
}
group.allReduce(y_partial, y, .sum);
```

The existing `inline else` dispatch within each `Backend` is preserved — parallelism is layered **above** the backend, not inside it.

---

## 8. Synchronization Primitives

### 8.1 Required Collective Operations

| Operation | TP | PP | Description |
| :--- | :--- | :--- | :--- |
| `allReduce(sum)` | Yes | No | Combine partial GEMV results across TP group |
| `send(dst, buf)` | No | Yes | Forward activation to next PP stage |
| `recv(src, buf)` | No | Yes | Receive activation from previous PP stage |
| `barrier()` | Yes | Yes | Synchronize all devices (init, shutdown) |
| `allGather()` | Yes | No | Gather partial logits for final argmax |
| `reduceScatter()` | Yes | No | Fused reduce + scatter (optimization) |

### 8.2 Backend-Specific Implementation

```zig
pub const CommOps = struct {
    /// All-reduce: sum partial results across TP group.
    /// `buf` is both input and output (in-place).
    allReduce: *const fn (buf: []f32, group: []const Device) void,

    /// Point-to-point send (async).
    send: *const fn (buf: []const f32, dst: *const Device) void,

    /// Point-to-point recv (blocking until data arrives).
    recv: *const fn (buf: []f32, src: *const Device) void,

    /// Barrier: wait for all devices to reach this point.
    barrier: *const fn (group: []const Device) void,
};
```

Per-backend implementations:

| Backend | allReduce | send/recv | Barrier |
| :--- | :--- | :--- | :--- |
| **Metal** | Shared buffer + compute kernel + `MTLSharedEvent` | Zero-copy (UMA) + `MTLSharedEvent` signal/wait | `MTLSharedEvent` |
| **CUDA** | Custom Zig ring or NCCL via `dlopen` | `cuMemcpyPeerAsync` or `ncclSend/Recv` | `cuStreamWaitEvent` |
| **ROCm** | Custom Zig ring or RCCL via `dlopen` | `hipMemcpyPeerAsync` or `rcclSend/Recv` | `hsa_signal_wait` |
| **Vulkan** | Shared buffer compute shader + timeline semaphores | `vkCmdCopyBuffer` with device masks + timeline semaphores | `vkQueueWaitIdle` |
| **CPU** | Direct memory sum (SIMD) | No-op (shared address space) | `std.Thread.Futex` or `@fence` |

### 8.3 Double-Buffering for Overlap

To hide communication latency, use **double-buffered** activation transfers:

```
Step T:    Device 0 computes layer L, writes to buf_A
           Device 1 reads buf_B (from step T-1), computes layer L+offset
Step T+1:  Device 0 computes layer L+1, writes to buf_B
           Device 1 reads buf_A (from step T), computes next layer
```

Pre-allocate two send/recv buffers per device at init. Alternate between them each step.

---

## 9. Weight Distribution

### 9.1 Sharding Strategy

At model init, weights are loaded and distributed:

```zig
pub fn loadShardedWeights(
    fmt: Format,
    group: *DeviceGroup,
    comptime shard_plan: ShardPlan,
) !ShardedWeights {
    for (group.devices) |*dev| {
        const rank = dev.rank;
        const tp_rank = rank % group.tp_degree;
        const pp_stage = rank / group.tp_degree;

        // Only load layers assigned to this PP stage
        const layer_start = pp_stage * layers_per_stage;
        const layer_end = (pp_stage + 1) * layers_per_stage;

        for (layer_start..layer_end) |layer| {
            // Column-parallel: Q, K, V, W_gate, W_up
            dev.weights.q[layer] = extractColumns(
                fmt.getTensor("blk.{}.attn_q.weight", layer),
                tp_rank, group.tp_degree,
            );
            // Row-parallel: W_o, W_down
            dev.weights.o[layer] = extractRows(
                fmt.getTensor("blk.{}.attn_output.weight", layer),
                tp_rank, group.tp_degree,
            );
        }
    }
}
```

### 9.2 Memory Budget Per Device

With TP=T and PP=P, each device's weight memory ≈ `total_params / (T × P)`:

| Model | Total params | TP=2, PP=1 | TP=2, PP=2 | TP=4, PP=2 |
| :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B (Q4_K_M) | ~4.5 GB | 2.25 GB | 1.12 GB | 0.56 GB |
| Llama-3.1-70B (Q4_K_M) | ~40 GB | 20 GB | 10 GB | 5 GB |
| Llama-3.1-405B (Q4_K_M) | ~230 GB | 115 GB | 57.5 GB | 28.75 GB |

---

## 10. Hybrid TP + PP

### 10.1 When to Use What

| Scenario | Recommended strategy |
| :--- | :--- |
| Model fits on 1 GPU | No parallelism |
| Model fits on 1 GPU with quantization | No parallelism (quantize instead) |
| 2 GPUs, same node, fast interconnect | TP=2 |
| 4 GPUs, same node, NVLink/XGMI | TP=4 or TP=2×PP=2 |
| 8 GPUs, same node | TP=4×PP=2 or TP=8 |
| Multi-node | TP within node, PP across nodes |
| Apple Ultra (2 dies) | TP=2 (UMA, zero-copy) |
| CPU dual-socket | TP=2 (shared memory, NUMA-aware) |

**Rule of thumb**: TP within fast-interconnect domains, PP across slower interconnects. TP requires frequent all-reduce (bandwidth-sensitive); PP only sends activations (latency-sensitive).

### 10.2 Device Mapping

For TP=2, PP=2 on 4 GPUs:

```
Device 0: TP_rank=0, PP_stage=0 → layers 0-12, heads 0-3
Device 1: TP_rank=1, PP_stage=0 → layers 0-12, heads 4-7
Device 2: TP_rank=0, PP_stage=1 → layers 13-25, heads 0-3
Device 3: TP_rank=1, PP_stage=1 → layers 13-25, heads 4-7
```

Communication pattern:
- **Within PP stage 0**: Dev 0 ↔ Dev 1 (all-reduce, high bandwidth needed)
- **Within PP stage 1**: Dev 2 ↔ Dev 3 (all-reduce, high bandwidth needed)
- **Between PP stages**: Dev 0→Dev 2, Dev 1→Dev 3 (activation send, latency-sensitive)

### 10.3 Forward Pass with Hybrid Parallelism

```zig
pub fn forwardParallel(
    self: *ParallelModel,
    token_id: u32,
) !u32 {
    const rank = self.device_rank;
    const tp_rank = rank % self.tp_degree;
    const pp_stage = rank / self.tp_degree;

    // Stage 0: embedding (replicated)
    if (pp_stage == 0) {
        self.be.embLookup(self.emb_table, token_id, self.hidden.ptr, self.n_embd);
    } else {
        // Receive activation from previous stage
        self.group.recv(self.hidden, self.group.ppPrev(rank).?);
    }

    // Process local layers
    for (self.layer_start..self.layer_end) |layer| {
        // --- Attention (column-parallel Q/K/V, row-parallel O) ---
        self.be.gemv(self.hidden.ptr, self.w_q[layer], self.q_buf.ptr, self.local_qkv_dim, self.n_embd);
        self.be.gemv(self.hidden.ptr, self.w_k[layer], self.k_buf.ptr, self.local_kv_dim, self.n_embd);
        self.be.gemv(self.hidden.ptr, self.w_v[layer], self.v_buf.ptr, self.local_kv_dim, self.n_embd);

        // SDPA on local heads (no communication needed)
        self.be.sdpa(self.q_buf.ptr, self.kv_cache.keys[layer], ...);

        // W_o (row-parallel) → produces partial hidden contribution
        self.be.gemv(self.attn_out.ptr, self.w_o[layer], self.partial.ptr, self.n_embd, self.local_qkv_dim);

        // ★ All-reduce across TP group (2 per layer: attention + FFN)
        self.group.allReduce(self.partial, self.group.tpPeers(rank));
        self.be.add(self.hidden.ptr, self.partial.ptr, self.hidden.ptr, self.n_embd);

        // --- FFN (column-parallel gate/up, row-parallel down) ---
        self.be.gemv(self.hidden.ptr, self.w_gate[layer], self.ff_gate.ptr, self.local_ff_dim, self.n_embd);
        self.be.gemv(self.hidden.ptr, self.w_up[layer], self.ff_up.ptr, self.local_ff_dim, self.n_embd);
        self.be.silu(self.ff_gate.ptr, self.ff_gate.ptr, self.local_ff_dim);
        self.be.mul(self.ff_gate.ptr, self.ff_up.ptr, self.ff_gate.ptr, self.local_ff_dim);
        self.be.gemv(self.ff_gate.ptr, self.w_down[layer], self.partial.ptr, self.n_embd, self.local_ff_dim);

        // ★ All-reduce (FFN)
        self.group.allReduce(self.partial, self.group.tpPeers(rank));
        self.be.add(self.hidden.ptr, self.partial.ptr, self.hidden.ptr, self.n_embd);
    }

    // Pipeline: send activation to next stage or compute logits
    if (pp_stage < self.pp_degree - 1) {
        self.group.send(self.hidden, self.group.ppNext(rank).?);
        return 0; // Not the last stage; token comes from last stage
    } else {
        // Last stage: final norm + logits
        self.be.rmsNorm(self.hidden.ptr, self.final_norm.ptr, self.hidden.ptr, self.n_embd, self.eps);
        // Column-parallel lm_head: each device produces vocab_size/TP logits
        self.be.gemv(self.hidden.ptr, self.lm_head_shard, self.logits_shard.ptr, self.local_vocab, self.n_embd);
        self.be.sync();
        // All-gather logits or distributed argmax
        return self.distributedArgmax();
    }
}
```

---

## 11. Sequence Parallelism & Context Parallelism

### 11.1 Motivation

TP and PP split the **model** dimension (weights, heads, layers). For very long contexts (32K-1M+ tokens), the **sequence** dimension itself becomes the bottleneck:

- KV cache grows linearly with sequence length — at 128K tokens, Llama-70B's KV cache is ~83 GB (FP8), exceeding a single GPU's HBM
- Attention computation is O(seq_len^2) during prefill — splitting across devices gives linear speedup
- Even decode-phase SDPA reads the entire KV cache each step — distributing it across devices reduces per-device bandwidth pressure

### 11.2 Sequence Parallelism (SP)

Megatron-style sequence parallelism splits operations that act on the **full hidden state** across the sequence dimension, complementing TP. With TP, operations like LayerNorm, dropout, and residual additions have the full `n_embd` dimension but can be split along the sequence (token) axis.

In standard TP, norms require an all-reduce of the sum-of-squares. With SP, each device processes a different subset of tokens through the norm — no all-reduce needed for the norm itself. The communication pattern changes:

```
Without SP (standard TP):
  GEMV (column-parallel) → all-reduce → Norm (full hidden) → GEMV (row-parallel) → all-reduce

With SP:
  GEMV (column-parallel) → reduce-scatter → Norm (local tokens) → all-gather → GEMV (row-parallel)
```

The total communication volume is the same, but `reduce-scatter` + `all-gather` can be overlapped with compute better than `all-reduce` (which is `reduce-scatter` + `all-gather` fused but blocking).

**SP is only useful during prefill** (multiple tokens). During single-token decode, there's only 1 token — nothing to split.

### 11.3 Context Parallelism (Ring Attention)

For ultra-long contexts, **context parallelism (CP)** splits the sequence across devices so each device holds a chunk of the KV cache and computes attention over its chunk.

**Ring Attention** (Liu et al., 2023) is the key algorithm:

```
4 devices, sequence length 32K (8K per device):

Device 0: tokens 0-8K       Device 1: tokens 8K-16K
Device 2: tokens 16K-24K    Device 3: tokens 24K-32K

Each device holds Q for its local chunk, K/V for its local chunk.
In a ring, each device passes its K/V to the next device:

Step 0: Device 0 attends Q[0:8K] × K/V[0:8K]     (local)
Step 1: Device 0 attends Q[0:8K] × K/V[24K:32K]   (received from Device 3)
Step 2: Device 0 attends Q[0:8K] × K/V[16K:24K]   (received from Device 2)
Step 3: Device 0 attends Q[0:8K] × K/V[8K:16K]    (received from Device 1)

Each step: compute local attention + pass K/V to next device in ring.
Final: combine partial attention outputs using log-sum-exp trick.
```

**Communication pattern**: Each device sends its K/V chunk to the next device in a ring. Total data moved per device = `(CP-1) × chunk_kv_size`. This overlaps with compute — while computing attention for chunk `i`, asynchronously send/receive chunk `i+1`.

**Log-sum-exp online correction**: Partial softmax outputs from different chunks must be combined correctly. Each chunk produces a local attention output + the log-sum-exp of its softmax. The global output is computed by rescaling each partial output:

```zig
// Combine two partial attention results:
//   out_a with logsumexp_a  (from chunk A)
//   out_b with logsumexp_b  (from chunk B)
fn combinePartialAttention(
    out_a: []f32, lse_a: f32,
    out_b: []f32, lse_b: f32,
    result: []f32,
) void {
    const max_lse = @max(lse_a, lse_b);
    const scale_a = @exp(lse_a - max_lse);
    const scale_b = @exp(lse_b - max_lse);
    const denom = scale_a + scale_b;
    for (result, out_a, out_b) |*r, a, b| {
        r.* = (a * scale_a + b * scale_b) / denom;
    }
}
```

### 11.4 CP + TP Interaction

Context parallelism composes with tensor parallelism. With CP=4 and TP=2 on 8 GPUs:

```
                  CP Group 0           CP Group 1
                  (tokens 0-16K)       (tokens 16K-32K)
TP Group 0:       Dev 0 (heads 0-31)   Dev 2 (heads 0-31)
TP Group 1:       Dev 1 (heads 32-63)  Dev 3 (heads 32-63)
                  Dev 4 (heads 0-31)   Dev 6 (heads 0-31)
                  Dev 5 (heads 32-63)  Dev 7 (heads 32-63)
```

- TP all-reduce within each TP group (fast, NVLink)
- CP ring K/V exchange within each CP group (can be cross-node)
- KV cache is split both by heads (TP) and by sequence position (CP)

### 11.5 When to Use CP

| Scenario | Strategy |
| :--- | :--- |
| seq_len < 8K | No CP needed |
| 8K-32K, fits in single GPU KV budget | No CP, use TP/PP only |
| 32K-128K, KV exceeds single GPU | CP=2-4 |
| 128K-1M | CP=8-16, combine with TP |
| >1M (research) | CP=32+, likely needs custom attention variants |

**For decode**: CP is less critical since decode only appends one K/V per step. But the existing KV cache is still distributed, so each decode step still does the ring exchange (though only for the dot product, not writing new K/V). Alternatively, replicate the new K/V to all CP ranks (broadcast, tiny).

---

## 12. Expert Parallelism (EP) for MoE

### 12.1 Why EP is Different

MoE models (GPT-OSS, Nemotron-Nano, GLM4) have a fundamentally different communication pattern than standard TP. In TP, the same tokens go to all devices but with different weight shards. In EP, **different tokens go to different devices** based on the router's expert selection.

```
Standard TP:                     Expert Parallelism:
  All devices process             Each device processes
  same tokens, different          different tokens for
  weight shards                   different experts

  Token → [shard 0] ──►          Token → Router → Expert 2 (on Dev 0)
          [shard 1] ──► reduce                  → Expert 5 (on Dev 1)
          [shard 2] ──►                         → Expert 0 (on Dev 0)
```

### 12.2 All-to-All Communication

EP requires an **all-to-all** exchange, not an all-reduce:

```
Phase 1 - Dispatch: Each device sends tokens to the device hosting the selected expert
Phase 2 - Compute:  Each device runs its local experts on received tokens
Phase 3 - Combine:  Each device sends results back to the originating device
```

```
Before all-to-all (tokens on originating devices):
  Dev 0: [tok_a→exp2, tok_b→exp0]     Dev 1: [tok_c→exp1, tok_d→exp3]

After dispatch all-to-all:
  Dev 0: [tok_a(exp0), tok_b(exp0), tok_c(exp1)]  ← experts 0,1 local
  Dev 1: [tok_a(exp2), tok_d(exp3)]                ← experts 2,3 local

After compute + combine all-to-all:
  Dev 0: [result_a, result_b]     Dev 1: [result_c, result_d]
```

### 12.3 Implementation for Agave MoE Models

**GPT-OSS**: 32 experts, top-4 routing, no shared expert.
**Nemotron-Nano**: 128 routed experts, top-6, shared expert.
**GLM4**: MoE with sigmoid routing.

With EP degree `E`, each device hosts `N/E` experts (N=32 for GPT-OSS, 128 for Nemotron-Nano):

| EP | Experts per device (GPT-OSS / Nemotron-Nano) | Valid devices |
| :--- | :--- | :--- |
| 1 | 32 / 128 | Any (no parallelism) |
| 2 | 16 / 64 | 2+ |
| 4 | 32 | 4+ |
| 8 | 16 | 8+ |
| 16 | 8 | 16+ |

### 12.4 All-to-All Implementation

**NCCL/RCCL**: `ncclGroupStart` + per-peer `ncclSend`/`ncclRecv` + `ncclGroupEnd`. This is the easiest path.

**Custom Zig**: More complex than ring all-reduce. Each device must:
1. Run the router to determine expert assignments for all tokens in the batch
2. Build per-destination scatter lists (which tokens go to which device)
3. Pack tokens into per-destination buffers
4. Exchange buffers (all-to-all)
5. Unpack, compute, repack results
6. Reverse all-to-all

```zig
pub fn moeAllToAll(
    tokens: []const []f32,        // Input tokens [batch × hidden]
    expert_ids: []const []u32,    // Router output [batch × top_k]
    local_experts: []const Expert, // This device's experts
    group: []const Device,
    result: [][]f32,              // Output [batch × hidden]
) void {
    // 1. Count tokens per destination device
    var counts: [max_devices]usize = .{0} ** max_devices;
    for (expert_ids) |ids| {
        for (ids) |eid| {
            const dst_dev = eid / experts_per_device;
            counts[dst_dev] += 1;
        }
    }

    // 2. Exchange counts (all-to-all metadata)
    allToAllCounts(counts, recv_counts, group);

    // 3. Pack tokens into per-destination send buffers
    packTokens(tokens, expert_ids, send_bufs);

    // 4. All-to-all data exchange
    allToAllData(send_bufs, recv_bufs, counts, recv_counts, group);

    // 5. Compute local experts on received tokens
    for (recv_bufs) |buf, expert_local_id| {
        local_experts[expert_local_id].forward(buf, out_buf);
    }

    // 6. Reverse all-to-all to return results
    allToAllData(out_bufs, result_bufs, recv_counts, counts, group);

    // 7. Weighted combine with router scores
    combineExpertOutputs(result_bufs, expert_scores, result);
}
```

### 12.5 EP + TP Interaction

EP and TP can be combined. Two strategies:

**Strategy A: EP across devices, no TP within experts.**
Simple. Each expert lives on one device. Works when individual experts are small enough to fit.

**Strategy B: TP within experts, EP across groups.**
For very large experts, shard each expert across a TP group. The TP group runs the expert with all-reduce, then EP handles the token routing. Requires `EP × TP` devices total.

**Agave recommendation**: GPT-OSS and Nemotron-Nano have small experts (relative to total model). Strategy A is sufficient. Use TP for the non-MoE layers (attention, shared expert), EP for the routed experts.

### 12.6 Load Balancing

Token-to-expert routing is non-uniform — popular experts receive more tokens. With EP, this creates load imbalance: the device hosting popular experts becomes the bottleneck.

**Mitigation**:
- **Capacity factor**: Cap the number of tokens per expert (drop excess tokens). GPT-OSS already uses top-4 with equal weights, which helps.
- **Expert replication**: Replicate popular experts across multiple devices. The router can break ties by sending tokens to the least-loaded replica.
- **Dynamic rebalancing**: Monitor per-device token counts per batch and adjust expert placement between batches. Expensive but optimal.

For inference (vs training), load imbalance is less severe because batch sizes are typically small (1-8 requests). With batch=1, only a few experts are active per token (top-4 or top-6), which naturally distributes across devices.

---

## 13. Continuous Batching with Parallelism

### 13.1 Overview

Continuous batching (also called iteration-level batching or in-flight batching) processes multiple requests simultaneously, adding and removing requests from the batch at each decode step rather than waiting for all requests in a batch to complete.

This interacts with parallelism in several ways:

```
Without continuous batching:
  Batch = [req_A, req_B, req_C] → all must finish before new requests enter

With continuous batching:
  Step 1: [req_A, req_B, req_C]   ← req_C finishes (EOS)
  Step 2: [req_A, req_B, req_D]   ← req_D joins immediately
  Step 3: [req_A, req_B, req_D]   ← req_A finishes
  Step 4: [req_E, req_B, req_D]   ← req_E joins
```

### 13.2 Batching with Tensor Parallelism

With TP, all devices in the TP group process the **same batch** of tokens. Adding/removing requests from the batch must be coordinated across all TP ranks:

```
Scheduler decides: "add req_D, remove req_C"
  → Broadcast decision to all TP ranks
  → All ranks update their local batch state simultaneously
  → All ranks process the new batch in lockstep
```

**KV cache implications**: When a request is added, all TP ranks allocate KV blocks for their respective head shards. When removed, all ranks free their blocks. The block allocation must be **deterministic** across ranks — either use a centralized allocator or seed the same allocation order.

```zig
pub const BatchState = struct {
    /// Active requests, indexed by slot.
    slots: [max_batch_size]?*Request,
    /// Number of active requests.
    active: u32,

    /// Add a request (called identically on all TP ranks).
    pub fn addRequest(self: *BatchState, req: *Request) !u8 {
        const slot = self.findFreeSlot() orelse return error.BatchFull;
        self.slots[slot] = req;
        self.active += 1;
        return slot;
    }

    /// Remove a request (called identically on all TP ranks).
    pub fn removeRequest(self: *BatchState, slot: u8) void {
        self.slots[slot] = null;
        self.active -= 1;
    }
};
```

### 13.3 Batching with Pipeline Parallelism

PP adds complexity: different stages may be processing different tokens from different steps. With continuous batching, a request might be at layer 15 on stage 1 while a new request starts at layer 0 on stage 0.

**Micro-batch scheduling with continuous batching:**

```
Time ──────────────────────────────────────────────────►

Stage 0:  [req_A step3] [req_B step3] [req_A step4] [req_C step1(prefill)]
                   │           │            │
Stage 1:  [req_A step2] [req_B step2] [req_A step3] [req_B step3]
                   │           │            │
Stage 2:  [req_A step1] [req_B step1] [req_A step2] [req_B step2]
```

Each stage maintains its own view of which request is currently in-flight. The PP scheduler must ensure:
1. Activations flow correctly between stages (each stage knows which request an activation belongs to)
2. New requests are injected at stage 0 only
3. Completed requests (EOS at last stage) trigger cleanup across all stages

### 13.4 Batched GEMV vs GEMM

With batch size > 1, the per-token GEMV (`[n,k] × [k,1] → [n,1]`) becomes a GEMM (`[n,k] × [k,B] → [n,B]`). This changes the computational profile:

| Batch size | Operation | Bottleneck | Arithmetic intensity |
| :--- | :--- | :--- | :--- |
| 1 | GEMV | Memory bandwidth | O(1) — 2 FLOPs per byte loaded |
| 2-8 | Small GEMM | Transitional | O(B) |
| 16+ | Large GEMM | Compute | O(B) — approaching peak FLOPS |

**Implication for TP**: With larger batches, the compute-to-communication ratio improves. All-reduce overhead (fixed per layer) is amortized over more useful compute. TP becomes more efficient at higher batch sizes.

**Backend dispatch**: The backend should detect batch size and switch between GEMV and GEMM kernels:

```zig
pub inline fn matMul(self: Backend, x: [*]const f32, w: TensorData,
                     y: [*]f32, n: usize, k: usize, batch: usize) void {
    if (batch == 1) {
        // GEMV path (bandwidth-bound, current implementation)
        self.gemv(x, w, y, n, k);
    } else {
        // GEMM path (compute-bound, tiled)
        self.gemm(x, w, y, n, k, batch);
    }
}
```

### 13.5 KV Cache with Continuous Batching

Each request in the batch has independent KV cache state. With `PagedKvCache`, each request has its own `SeqBlockTable`:

```
Request A (slot 0): blocks [0,1,2,3]     (seq_len=60)
Request B (slot 1): blocks [4,5]          (seq_len=28)
Request C (slot 2): blocks [6,7,8,9,10]  (seq_len=73)
```

**SDPA with batched KV cache**: The attention kernel must handle variable sequence lengths per request. Two approaches:

1. **Iterate per request**: Run SDPA once per request with its specific KV blocks. Simple but doesn't leverage GPU parallelism across requests.
2. **Batched paged attention**: A single kernel processes all requests, with indirect block access per request. This is vLLM's PagedAttention kernel — each query attends to blocks from its own block table.

For Agave, approach 1 is sufficient initially (batch=1-8). Approach 2 is needed for high-throughput serving (batch=32+).

---

## 14. Prefill with Tensor Parallelism

### 14.1 GEMM vs GEMV in Prefill

During prefill, the model processes `P` prompt tokens at once. Each weight matrix multiplication is a GEMM (`[n,k] × [k,P] → [n,P]`) rather than a GEMV (`[n,k] × [k,1] → [n,1]`).

This fundamentally changes the TP efficiency:

| Phase | Operation | Compute | All-reduce | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| Decode (1 token) | GEMV: `[4096,4096] × [4096,1]` | 33M FLOPs | 32 KB | 1M FLOP/byte |
| Prefill (512 tokens) | GEMM: `[4096,4096] × [4096,512]` | 17B FLOPs | 32 KB | 530M FLOP/byte |
| Prefill (2048 tokens) | GEMM: `[4096,4096] × [4096,2048]` | 69B FLOPs | 32 KB | 2.1B FLOP/byte |

**Key insight**: The all-reduce size stays the same (hidden dimension, not sequence length), but compute scales with sequence length. Prefill has a much better compute-to-communication ratio, making TP highly efficient.

### 14.2 Attention During Prefill

Prefill attention differs from decode:

- **Decode SDPA**: `Q=[1,hd]`, `K/V=[sl,hd]` — one query against all cached keys
- **Prefill SDPA**: `Q=[P,hd]`, `K=[P,hd]`, `V=[P,hd]` — all queries against all keys (causal mask)

With TP, each device has `n_heads/TP` query heads. Prefill attention is embarrassingly parallel across heads — no cross-device communication needed for SDPA itself (same as decode).

The expensive part during prefill is the QKV projection GEMM, which benefits from TP's column-parallel split and the high compute-to-communication ratio.

### 14.3 FlashAttention for Parallel Prefill

Prefill benefits enormously from fused attention kernels (FlashAttention):
- Avoids materializing the `[P,P]` attention matrix (O(P^2) memory → O(P) with tiling)
- Each TP rank runs FlashAttention independently on its local heads
- With CP (context parallelism), each CP rank runs FlashAttention on its local sequence chunk

The Agave Metal SDPA kernel (`kernels.metal`) currently handles single-query attention. For prefill, a batched/tiled variant is needed that processes multiple queries simultaneously using the online softmax trick.

### 14.4 Chunked Prefill with TP

For long prompts, prefill can be **chunked** to limit memory usage and enable interleaving with decode (§13.2):

```zig
// Prefill 2048 tokens in chunks of 512
const chunk_size = 512;
var pos: usize = 0;
while (pos < prompt_len) {
    const end = @min(pos + chunk_size, prompt_len);
    const chunk_tokens = prompt[pos..end];

    // All TP ranks process the same chunk simultaneously
    for (self.layer_start..self.layer_end) |layer| {
        // GEMM: QKV projection for chunk_size tokens
        self.be.gemm(hidden_chunk, w_qkv[layer], qkv_chunk, ...);
        // TP all-reduce after W_o
        self.group.allReduce(partial_chunk, self.group.tpPeers(rank));
        // Write chunk's K/V to cache
        writeKvRange(layer, pos, end, k_chunk, v_chunk);
    }
    pos = end;
}
```

Each chunk writes its K/V to the cache. Subsequent chunks attend to all previously written K/V (growing causal context).

---

## 15. CLI Configuration & Auto-Detection

### 15.1 User Interface

```bash
# Explicit parallelism configuration
agave model.gguf --tp 2                          # TP=2 on auto-detected devices
agave model.gguf --tp 4 --pp 2                   # TP=4, PP=2 (8 devices total)
agave model.gguf --tp 2 --devices 0,1            # TP=2 on specific GPUs
agave model.gguf --pp 4 --devices 0,1,2,3        # PP=4 on specific GPUs
agave model.gguf --ep 8 --tp 2                   # EP=8 for MoE, TP=2 for attention

# NCCL/RCCL selection
agave model.gguf --tp 8 --nccl                   # Force NCCL for all-reduce
AGAVE_USE_NCCL=1 agave model.gguf --tp 8         # Same via env var

# Auto-detection (recommended for most users)
agave model.gguf --auto-parallel                 # Detect devices, choose TP/PP
agave model.gguf --auto-parallel --max-memory 24G # Constrain per-device memory

# Serving with parallelism
agave model.gguf --serve --tp 4 --batch 32       # Multi-GPU serving
agave model.gguf --serve --disaggregate 2,6      # 2 prefill GPUs, 6 decode GPUs

# KV cache configuration
agave model.gguf --kv-dtype fp8                  # FP8 KV cache
agave model.gguf --kv-offload cpu                # Enable CPU offloading
agave model.gguf --kv-offload nvme --offload-path /tmp/agave_kv  # NVMe tier
```

### 15.2 Auto-Detection Logic

When `--auto-parallel` is specified (or when the model doesn't fit on one device):

```zig
pub fn autoDetectParallelism(
    model_size: usize,
    kv_cache_size: usize,
    devices: []const DeviceInfo,
) ParallelConfig {
    const total_vram = sumVram(devices);
    const per_device_budget = minVram(devices);

    // 1. Can the model fit on a single device?
    if (model_size + kv_cache_size < per_device_budget * 0.9)
        return .{ .tp = 1, .pp = 1 };

    // 2. Determine minimum TP to fit model weights per device
    var tp: u32 = 1;
    while (model_size / tp > per_device_budget * 0.7) : (tp *= 2) {}

    // 3. If TP alone isn't enough, add PP
    var pp: u32 = 1;
    const weight_per_dev = model_size / (tp * pp);
    const kv_per_dev = kv_cache_size / (tp * pp);
    while (weight_per_dev + kv_per_dev > per_device_budget * 0.9) : (pp *= 2) {}

    // 4. Validate: tp * pp <= num_devices
    if (tp * pp > devices.len) {
        // Fall back: reduce TP, increase PP, or error
        return error.InsufficientDevices;
    }

    // 5. Check valid TP degree for model's head count
    if (model_n_heads % tp != 0 or model_n_kv_heads % tp != 0) {
        tp = findValidTp(model_n_heads, model_n_kv_heads, tp);
    }

    return .{ .tp = tp, .pp = pp };
}
```

### 15.3 Device Discovery

```zig
pub const DeviceInfo = struct {
    id: u32,
    backend: BackendType,
    vram_bytes: usize,
    name: []const u8,
    /// Peer-to-peer capability with other devices.
    p2p_peers: []u32,
    /// NUMA node (for CPU/topology awareness).
    numa_node: ?u32,
};

pub fn discoverDevices(allocator: Allocator) ![]DeviceInfo {
    var devices = std.ArrayList(DeviceInfo).init(allocator);

    // Metal: MTLCopyAllDevices()
    if (builtin.os.tag == .macos) {
        try discoverMetalDevices(&devices);
    }

    // CUDA: cuDeviceGetCount + cuDeviceGet
    if (try probeCuda()) |cuda| {
        try discoverCudaDevices(cuda, &devices);
    }

    // ROCm: hipGetDeviceCount + hipGetDeviceProperties
    if (try probeRocm()) |rocm| {
        try discoverRocmDevices(rocm, &devices);
    }

    // Vulkan: vkEnumeratePhysicalDevices
    try discoverVulkanDevices(&devices);

    // CPU: always available, one "device" per NUMA node
    try discoverCpuDevices(&devices);

    return devices.toOwnedSlice();
}
```

### 15.4 Topology Detection

After discovery, detect the interconnect topology to determine optimal TP/PP placement:

```zig
pub fn detectTopology(devices: []const DeviceInfo) TopologyMap {
    var map: TopologyMap = undefined;

    for (devices, 0..) |dev_a, i| {
        for (devices, 0..) |dev_b, j| {
            if (i == j) continue;
            map.bandwidth[i][j] = measureP2PBandwidth(dev_a, dev_b);
            map.latency[i][j] = measureP2PLatency(dev_a, dev_b);
            map.link_type[i][j] = classifyLink(dev_a, dev_b);
        }
    }

    return map;
}

pub const LinkType = enum {
    uma,            // Shared memory (Apple Ultra, same-die)
    nvlink,         // NVLink (NVIDIA same-node)
    xgmi,           // XGMI/Infinity Fabric (AMD same-node)
    pcie_p2p,       // PCIe peer-to-peer
    host_staged,    // Must go through CPU
    network,        // Cross-node
};
```

Place TP groups on the fastest links (NVLink/XGMI/UMA) and PP stages across slower links (PCIe, network).

---

## 16. Testing Strategy

### 16.1 Correctness Testing Without Multiple GPUs

Most development happens on single-GPU machines. The parallelism code must be testable without actual multi-device hardware.

**Simulated multi-device on CPU:**

```zig
/// Create N "virtual devices" backed by the CPU backend.
/// Each virtual device has its own memory region, simulating
/// separate GPU memories. Communication goes through explicit copies.
pub fn createVirtualDeviceGroup(
    allocator: Allocator,
    n_devices: u32,
    vram_per_device: usize,
) !DeviceGroup {
    var devices = try allocator.alloc(Device, n_devices);
    for (devices, 0..) |*dev, i| {
        var cpu_be = try CpuBackend.init(allocator);
        dev.* = .{
            .rank = @intCast(i),
            .backend = .{ .cpu = &cpu_be },
            .send_buf = try allocator.alloc(f32, max_hidden),
            .recv_buf = try allocator.alloc(f32, max_hidden),
            .reduce_buf = try allocator.alloc(f32, max_hidden),
        };
    }
    return .{
        .devices = devices,
        .topology = .host_staged, // Simulated — copies through CPU
        .tp_degree = n_devices,
        .pp_degree = 1,
        .allocator = allocator,
    };
}
```

### 16.2 Golden Test: TP=1 vs TP=N

The most important correctness test: given identical inputs and weights, `TP=1` and `TP=N` must produce **bit-identical** (or within tolerance) outputs.

```zig
test "TP=2 matches TP=1" {
    const allocator = std.testing.allocator;

    // Run with TP=1
    var model_1 = try loadModel(allocator, test_weights, .{ .tp = 1 });
    defer model_1.deinit();
    const output_1 = try model_1.forward(test_token);

    // Run with TP=2 (virtual devices)
    var group = try createVirtualDeviceGroup(allocator, 2, 1 << 30);
    defer group.deinit();
    var model_2 = try loadModel(allocator, test_weights, .{ .tp = 2, .group = &group });
    defer model_2.deinit();
    const output_2 = try model_2.forward(test_token);

    // Outputs must match (f32 tolerance for reduction order)
    try std.testing.expectApproxEqAbs(output_1.logits, output_2.logits, 1e-5);
}
```

### 16.3 Test Categories

| Category | What it tests | Devices needed |
| :--- | :--- | :--- |
| **Unit: sharding** | Weight column/row extraction, block alignment | 0 (CPU only) |
| **Unit: all-reduce** | Ring reduce correctness, various sizes | 0 (virtual devices) |
| **Unit: KV partition** | Head splitting, block table construction | 0 (CPU only) |
| **Integration: TP=2** | Full forward pass, output matches TP=1 | 0 (virtual devices) |
| **Integration: PP=2** | Activation transfer, stage handoff | 0 (virtual devices) |
| **Integration: TP+PP** | Combined parallelism | 0 (virtual devices) |
| **Backend: Metal TP** | Real MTLSharedEvent sync, UMA zero-copy | 1 Ultra chip (2 dies) |
| **Backend: CUDA TP** | Real NVLink P2P, NCCL dlopen | 2+ NVIDIA GPUs |
| **Backend: ROCm TP** | Real XGMI P2P, RCCL dlopen | 2+ AMD GPUs |
| **Perf: scaling** | Throughput vs TP degree | Multi-GPU |
| **Stress: long seq** | Context parallelism correctness at 128K | Virtual or multi-GPU |

### 16.4 CI Integration

```yaml
# CI matrix — parallelism tests
jobs:
  test-parallel-cpu:
    # Runs on every PR — virtual devices, no GPU needed
    - zig build test -Dtest-filter="parallel"

  test-parallel-metal:
    # Runs on macOS Ultra runner (if available)
    runs-on: macos-ultra
    - zig build test -Dtest-filter="parallel_metal"

  test-parallel-cuda:
    # Runs on GPU runner (scheduled, not per-PR)
    runs-on: gpu-2x-a100
    - zig build test -Dtest-filter="parallel_cuda"
```

### 16.5 Debugging Parallel Issues

**Common bugs:**
1. **Deadlock**: One rank calls all-reduce, another doesn't (divergent control flow). Detect with timeout + rank-aware logging.
2. **Silent corruption**: Sharding boundary off by one → partial garbage. Catch with TP=1 vs TP=N golden test.
3. **Non-deterministic reduction**: Floating-point addition order differs across runs. Accept tolerance in tests, use Kahan summation for critical paths.
4. **Race conditions**: Multiple ranks writing to shared buffer without proper sync. Use `MTLSharedEvent` / `cuStreamWaitEvent` religiously.

**Debug mode**: With `--debug-parallel`, emit per-rank, per-layer checksums:

```
[rank 0] layer 0 attn: hidden_checksum=0x3F7A2B1C reduce_checksum=0x4E9D8F0A
[rank 1] layer 0 attn: hidden_checksum=0x3F7A2B1C reduce_checksum=0x4E9D8F0A  ← must match
```

---

## 17. Model Compatibility Matrix

### 17.1 Parallelism Support by Model

Not all models support all parallelism modes. Hybrid models (SSM + attention + MoE) have specific constraints:

| Model | TP | PP | EP | CP | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma3** | Yes (max TP depends on variant; 1B is TP=1 only) | Yes | N/A | Yes | Pure transformer, straightforward |
| **Qwen3.5** | Partial | Yes | N/A | Partial | DeltaNet layers need special handling (§17.2) |
| **GPT-OSS** | Yes (max TP=8) | Yes | Yes (32 experts) | Yes | MoE layers use EP; attention uses TP |
| **Nemotron-H** | Partial | Yes | N/A | Partial | Mamba-2 SSM layers need special handling |
| **Nemotron-Nano** | Partial | Yes | Yes (128 experts) | Partial | Mixed SSM + MoE + attention |
| **GLM4** | Yes | Yes | Yes | Yes | MoE with sigmoid routing |

### 17.2 SSM Layer Constraints (Qwen3.5, Nemotron-H, Nemotron-Nano)

SSM layers (DeltaNet, Mamba-2) use recurrent state instead of KV cache. Under TP:

**Problem**: The recurrence `h_t = A * h_{t-1} + B * x_t` is sequential — it cannot be split across heads the way attention can, because the state matrix `A` couples all dimensions.

**Options:**
1. **Replicate SSM on all TP ranks** — each rank runs the full SSM independently. Wastes compute (TP× redundant) but produces correct results. No all-reduce needed for SSM output if it feeds directly into a TP-sharded attention layer.
2. **Shard SSM state dimension** — split the state vector across TP ranks. Requires all-reduce after each SSM step because the state evolution couples dimensions. Adds one all-reduce per SSM layer per token.
3. **Assign SSM layers to one rank** — PP-style: SSM layers run on rank 0 only, attention layers are TP-sharded. Awkward hybrid but avoids SSM parallelism entirely.

**Recommendation**: Option 1 (replicate) for Agave. SSM layers are computationally cheap (~0.3-0.5× of attention), so the wasted compute is minimal. The implementation is trivial — SSM layers simply skip the TP shard logic and use the full weight.

### 17.3 Validation at Init

```zig
pub fn validateParallelConfig(
    arch: Arch,
    config: ParallelConfig,
    model_params: ModelParams,
) !void {
    // Common: TP must divide head counts
    if (model_params.n_heads % config.tp != 0)
        return error.InvalidTPDegree;
    if (model_params.n_kv_heads % config.tp != 0)
        return error.InvalidTPDegree;

    // MoE: EP must divide expert count
    if (arch.hasMoE()) {
        if (config.ep > 0 and model_params.n_experts % config.ep != 0)
            return error.InvalidEPDegree;
    }

    // Hybrid: warn about SSM replication overhead
    if (arch.hasSSM() and config.tp > 1) {
        std.log.warn("SSM layers will be replicated across {d} TP ranks " ++
            "(~{d}% compute overhead for SSM layers)", .{
            config.tp, (config.tp - 1) * 100 / config.tp,
        });
    }

    // PP: layer count must be reasonably divisible
    if (config.pp > model_params.n_layers)
        return error.TooFewLayers;
}
```

---

## 18. Startup & Weight Loading

### 18.1 The Problem

Loading and distributing weights for large models is non-trivial:

| Model | Size (Q4_K_M) | Load time (NVMe, 7 GB/s) | GPU transfer (PCIe Gen4) |
| :--- | :--- | :--- | :--- |
| Llama-8B | 4.5 GB | 0.6 s | 0.14 s |
| Llama-70B | 40 GB | 5.7 s | 1.25 s |
| Llama-405B | 230 GB | 33 s | 7.2 s |

For a 405B model across 8 GPUs with TP=8, each GPU needs ~29 GB of weights. Loading all 230 GB sequentially then distributing is slow. We need parallel loading.

### 18.2 Parallel Weight Loading

**Strategy A: mmap + parallel sharding (single file)**

```
                    ┌─── NVMe ───┐
                    │  model.gguf │
                    │  (230 GB)   │
                    └──────┬──────┘
                           │ mmap (lazy, demand-paged)
                    ┌──────┴──────┐
          ┌─────────┤ Virtual mem │──────────┐
          │         └─────────────┘          │
    Thread 0              Thread 1              Thread 7
    (GPU 0)               (GPU 1)               (GPU 7)
    Extract cols          Extract cols          Extract cols
    0..d/8                d/8..2d/8             7d/8..d
    for all layers        for all layers        for all layers
          │                    │                    │
    cudaMemcpy            cudaMemcpy            cudaMemcpy
    to GPU 0              to GPU 1              to GPU 7
```

Each thread reads its weight shard from the mmap'd file and uploads to its GPU. The OS page cache serves the reads — if all threads read the same file, the kernel only loads each page once.

**NVMe bandwidth**: Multiple threads reading different regions of the same file achieves near-peak NVMe bandwidth (sequential reads from different offsets). Modern NVMe SSDs handle 4-8 concurrent reads at full speed.

**Strategy B: pre-sharded files (multiple files)**

```
model-tp0.gguf → GPU 0 (load directly)
model-tp1.gguf → GPU 1 (load directly)
...
model-tp7.gguf → GPU 7 (load directly)
```

Each GPU loads its own file independently. Maximum I/O parallelism. Requires the `agave-shard` offline tool.

**Strategy C: network loading (multi-node)**

For multi-node setups, weights must be transferred to remote nodes:

```
Node 0 (has weights on NVMe):
  Load full model → shard → send shard to each node via RDMA

Node 0-N (parallel, if each has weights):
  Each node loads model.gguf from local NVMe → shard → upload to local GPUs
```

**Recommendation**: Use shared filesystem (NFS/Lustre/S3) so each node loads directly. Avoid bottlenecking on one node's NVMe. In cloud environments, weights are typically on shared storage already.

### 18.3 Staged Initialization

Weight loading overlaps with other init work. Pipeline the startup:

```
Time ──────────────────────────────────────────────────►

I/O thread:    [load layers 0-12 from NVMe ][load layers 13-25]
                         │                          │
GPU 0 thread:  [wait][upload L0-12 shards][wait][upload L13-25 shards]
GPU 1 thread:  [wait][upload L0-12 shards][wait][upload L13-25 shards]
                                            │
Init thread:   [tokenizer][KV cache alloc][warmup forward pass]
```

**Warmup**: After weights are loaded, run a dummy forward pass with a short prompt. This:
- Forces GPU kernel compilation (Metal shader compilation, CUDA JIT)
- Warms CPU caches for weight access patterns
- Validates the model works end-to-end before accepting requests

### 18.4 Weight Loading with PP

With PP, each stage only needs its layers' weights. Load them in stage order to minimize time-to-first-stage-ready:

```zig
pub fn loadWeightsParallel(
    fmt: Format,
    group: *DeviceGroup,
    thread_pool: *ThreadPool,
) !void {
    // Launch one load task per device
    for (group.devices) |*dev| {
        try thread_pool.spawn(struct {
            fn loadDeviceWeights(d: *Device, f: Format, g: *DeviceGroup) void {
                const pp_stage = d.rank / g.tp_degree;
                const tp_rank = d.rank % g.tp_degree;
                const layer_start = pp_stage * g.layers_per_stage;
                const layer_end = (pp_stage + 1) * g.layers_per_stage;

                for (layer_start..layer_end) |layer| {
                    // Extract and upload this device's weight shard
                    loadLayerShard(d, f, layer, tp_rank, g.tp_degree);
                }
            }
        }.loadDeviceWeights, .{ dev, fmt, group });
    }

    // Wait for all devices to finish loading
    thread_pool.waitAll();
}
```

### 18.5 Memory Budget During Loading

Peak memory during loading can be 2× the final weight size (original mmap'd file + GPU copy). Strategies to reduce peak:

1. **Layer-by-layer**: Load one layer, shard it, upload to GPUs, then `madvise(MADV_DONTNEED)` the source pages. Peak = 1 layer's weights + all GPU shards.
2. **Streaming shard**: Read the weight file with a small buffer, extract the shard on the fly, upload immediately. Peak = buffer size + GPU shard. Only works for formats where weight layout is known (GGUF tensor offsets are in the header).
3. **Pre-sharded files**: Each GPU mmap's only its shard file. Peak = shard size (no overhead).

For Approach A (mmap + parallel sharding), the OS will evict pages naturally under memory pressure, but explicit `madvise` is better for predictable behavior.

---

## 19. Implementation Plan

### Phase 0: Foundations

1. **Device discovery & topology detection** (`src/parallel/discovery.zig`)
2. **CLI flags**: `--tp`, `--pp`, `--ep`, `--auto-parallel`, `--devices`
3. **Virtual device group for testing** (`src/parallel/virtual.zig`)
4. **Model compatibility validation** at init

### Phase 1: Foundation (TP=2 on Metal)

Apple Ultra is the most accessible multi-die target for development. UMA eliminates data transfer complexity.

1. **DeviceGroup abstraction** (`src/parallel/device_group.zig`)
   - Discover Metal devices (`MTLCopyAllDevices()`)
   - Create Backend per device
   - Detect topology (UMA shared memory)

2. **Communication primitives** (`src/parallel/comm.zig`)
   - `allReduce` via shared UMA buffer + `MTLSharedEvent`
   - `send`/`recv` as pointer exchange + event signal (zero-copy)
   - `barrier` via `MTLSharedEvent`

3. **Weight sharding** (`src/parallel/shard.zig`)
   - Column/row extraction from mmap'd weights
   - Block-boundary alignment for quantized formats
   - Per-rank weight view (no copy on UMA)

4. **KV cache partitioning** (`src/kvcache/manager.zig` extension)
   - `ShardedKvCache` with per-rank head ranges
   - Integrate with existing `PagedKvCache`

5. **Model adaptation** (single model, e.g., Gemma3)
   - Refactor `forward()` to accept `DeviceGroup` + rank
   - Insert all-reduce at correct points
   - Verify correctness against single-device output

### Phase 2: Pipeline Parallelism (PP on Metal/CUDA)

6. **Stage assignment** — layer-to-device mapping with cost model
7. **Activation send/recv** — async double-buffered transfer
8. **Micro-batch scheduling** for prefill overlap
9. **Extend to CUDA backend** — P2P memory + stream events + NCCL `dlopen` wrapper

### Phase 3: Generalization

10. **ROCm transport** — HIP P2P + HSA signals + RCCL `dlopen` wrapper
11. **Vulkan device groups** — timeline semaphores + buffer copies
12. **CPU NUMA** — `mbind` + thread pinning + direct memory reduce
13. **Hybrid TP+PP** — compose phases 1+2
14. **Pre-sharded weight format** — offline `agave-shard` tool
15. **KV cache migration** — for continuous batching / load balancing

### Phase 4: Batching & Prefill

16. **Continuous batching** integration with TP/PP
17. **GEMM path for batched/prefill** — backend `gemm()` for batch>1
18. **Chunked prefill** — interleave with decode on same GPU
19. **Batched paged attention** kernel for multi-request SDPA

### Phase 5: MoE & Sequence Parallelism

20. **Expert parallelism** — all-to-all implementation for GPT-OSS / Nemotron-Nano
21. **Context parallelism** — ring attention with log-sum-exp correction
22. **SSM replication** under TP for hybrid models

### Phase 6: KV Cache Advanced Features

23. **KV cache quantization** — FP8/INT8 in-place storage with per-head scaling
24. **KV cache offloading** — GPU→CPU→NVMe tiered eviction with async prefetch
25. **IPC memory sharing** — CUDA IPC / HIP IPC / IOSurface for multi-process serving
26. **Disaggregated prefill/decode** — separate pools with layer-pipelined KV transfer

### Phase 7: Multi-Node & Optimization

27. **RDMA transport** — `libibverbs` + `libfabric` dlopen wrappers
28. **GPUDirect RDMA** — zero-copy GPU↔NIC for cross-node KV transfer
29. **GPUDirect Storage** — NVMe↔GPU via cuFile for KV offload tier 2
30. **Distributed RadixAttention** — global prefix registry for cross-node cache sharing
31. **Optimized all-reduce kernels** (custom MSL/PTX/AMDGCN/SPIR-V ring kernels; benchmark against NCCL/RCCL to identify crossover points)
32. **Fused all-reduce + bias + norm** (reduce kernel launch overhead)
33. **Quantized all-reduce** (FP8 or BF16 reduction for bandwidth savings)
34. **Overlap compute and communication** (split GEMV into chunks, interleave reduce)
35. **Topology-aware placement** — GPU↔NIC affinity detection, rail-optimized all-reduce

---

## 20. KV Cache Offload Granularity

### 20.1 The Problem

When GPU memory runs out, KV cache blocks must be evicted to a cheaper tier (CPU DRAM or NVMe). The key design question is: **at what granularity do we evict and fetch?**

Three options, each with distinct tradeoffs:

| Granularity | Unit size | Metadata cost | Flexibility | Fetch latency |
| :--- | :--- | :--- | :--- | :--- |
| **Per-block** | `block_size × kv_dim × sizeof(dtype)` (e.g., 16 × 128 × 4 = 8 KB) | High — 1 tier tag + handle per block | Highest — evict exactly the coldest positions | Low — small DMA |
| **Per-layer** | `seq_len × kv_dim × sizeof(dtype)` (e.g., 2048 × 128 × 4 = 1 MB) | Medium — 1 tier tag per layer per request | Medium — must evict entire layers | Medium |
| **Per-request** | `2 × n_layers × seq_len × kv_dim × sizeof(dtype)` (e.g., 1.3 GB for 70B) | Low — 1 tier tag per request | Lowest — all or nothing | High — massive DMA |

### 20.2 Recommendation: Per-Block with Batched I/O

Per-block gives the finest control and maps naturally to `PagedKvCache`. The metadata overhead (one `StorageTier` enum + one `OffloadHandle` per block) is small — at 8 bytes per block with 4096 blocks, that's 32 KB of metadata total.

The concern with per-block is I/O fragmentation: thousands of tiny 8 KB reads/writes are slower than fewer large transfers. Solve this with **batched scatter/gather I/O**:

```zig
pub const OffloadBatch = struct {
    /// Blocks to evict, sorted by physical address for sequential I/O.
    evict_list: []BlockId,
    /// Blocks to prefetch, sorted by layer then position for access locality.
    fetch_list: []BlockId,

    /// Execute all evictions as a single batched DMA.
    pub fn flush(self: *OffloadBatch, cache: *PagedKvCache) void {
        // Sort evict_list by GPU address → sequential cudaMemcpy
        std.sort.sort(BlockId, self.evict_list, {}, byGpuAddr);

        // Coalesce adjacent blocks into larger transfers
        var i: usize = 0;
        while (i < self.evict_list.len) {
            const run = coalesceRun(self.evict_list[i..]);
            asyncCopyD2H(run.gpu_start, run.cpu_start, run.total_bytes, stream);
            i += run.count;
        }
    }
};
```

**Coalescing**: When adjacent blocks in GPU memory are both evicted, merge them into one larger transfer. With 16-position blocks, 8 adjacent blocks = 128 KB — a much more efficient DMA unit.

### 20.3 Eviction Policy

**LRU with layer-awareness**: Pure LRU evicts the least-recently-accessed blocks regardless of layer. But layers early in the model are accessed before later layers in each forward pass. An attention-aware policy:

1. **Score**: `lru_score = last_access_time - layer_bias[layer]` where `layer_bias` penalizes early layers (they'll be needed sooner)
2. **Threshold**: Only evict blocks where `position < seq_len - sliding_window` (for models with sliding window attention, those positions will never be accessed again)
3. **Hysteresis**: Evict in bulk when free blocks < `low_watermark`, stop when free blocks > `high_watermark`. Prevents thrashing.

### 20.4 Prefetch Scheduling

The scheduler knows the request's block table and can predict which blocks will be needed:

```
Layer L executing now:
  → Prefetch KV blocks for layer L+1 (and L+2 if bandwidth allows)
  → If layer L+1 blocks are on CPU: issue cudaMemcpyAsync(H2D) now
  → If layer L+1 blocks are on NVMe: they should have been staged to CPU already
     (background thread moves NVMe→CPU for blocks predicted to be needed in ~5 layers)
```

**Pipeline**: NVMe→CPU prefetch runs 5-10 layers ahead. CPU→GPU prefetch runs 1-2 layers ahead. This hides both latencies in the forward pass.

### 20.5 NVMe Tier Details

For the NVMe tier, use a memory-mapped file as a block store:

```zig
const OffloadFile = struct {
    /// Memory-mapped file backing NVMe-tier blocks.
    mmap: []align(page_size) u8,
    /// Free slot bitmap (one bit per block-sized slot).
    free_bitmap: std.DynamicBitSet,
    /// File descriptor for io_uring / cuFile operations.
    fd: std.posix.fd_t,
};
```

- **Linux**: `io_uring` for async NVMe I/O (batched, zero-copy with registered buffers)
- **macOS**: `dispatch_io` for async file I/O
- **GPUDirect Storage**: `cuFileRead`/`cuFileWrite` bypass CPU entirely (NVMe↔GPU)

---

## 21. Disaggregated Scheduling

### 21.1 Architecture

A disaggregated serving system has three components:

```
                    ┌───────────────┐
  Requests ───────► │   Scheduler   │ ◄─── capacity reports
                    │  (Router)     │
                    └──┬─────────┬──┘
                       │         │
              ┌────────▼──┐  ┌──▼────────┐
              │  Prefill   │  │  Decode    │
              │  Pool      │  │  Pool      │
              │  (GPU 0-3) │  │  (GPU 4-7) │
              └────────────┘  └────────────┘
```

The scheduler makes three decisions:
1. **Admission**: Accept or reject a request based on system capacity
2. **Prefill routing**: Which prefill device(s) handle this request
3. **Decode placement**: Which decode device(s) receive the KV cache and generate tokens

### 21.2 Scheduling Strategies

#### Strategy A: Eager Layer-Pipelined (Lowest TTFT)

```
Time ──────────────────────────────────────────────────►

Prefill GPU:  [L0 prefill][L1 prefill][L2 prefill]...
                  │            │            │
KV Transfer:      └─ L0 KV ──►└─ L1 KV ──►└─ L2 KV ──►
                                                    │
Decode GPU:   [idle..............................][L0 decode starts]
```

- Begin transferring each layer's KV as soon as prefill completes it
- Decode can begin layer 0 as soon as its KV arrives (before all layers are transferred)
- **Lowest TTFT** but requires tight coordination between prefill and decode

**Implementation**: The prefill model signals completion of each layer via a per-layer event. A transfer thread watches these events and initiates `cudaMemcpyPeerAsync` (same node) or RDMA write (cross-node) for each layer.

#### Strategy B: Batched Transfer (Highest Throughput)

```
Time ──────────────────────────────────────────────────►

Prefill GPU:  [full prefill for req A][full prefill for req B]
                                     │                      │
KV Transfer:                         └── all layers A ─────►│
                                                            └── all layers B ──►
Decode GPU:   [idle...][decode A tokens]          [decode B tokens]
```

- Wait for full prefill to complete, then transfer all KV at once
- Simpler implementation, allows DMA coalescing for maximum bandwidth
- **Higher TTFT** but better GPU utilization (fewer sync points)

#### Strategy C: Chunked Prefill (Hybrid, No Disaggregation)

An alternative to full disaggregation: run prefill and decode **on the same GPU** but interleave them in chunks. Process the prompt in chunks of 128-512 tokens, interleaving decode steps for other requests between chunks.

```
Time ──────────────────────────────────────────────────►

Same GPU:  [prefill chunk 0][decode req B][prefill chunk 1][decode req B]...
```

- No KV transfer at all (everything stays on one GPU)
- Avoids the network bottleneck entirely
- Decode latency increases slightly during prefill chunks, but bounded
- **Recommended when network bandwidth is limited** (PCIe, Thunderbolt, commodity Ethernet)

### 21.3 Pool Sizing and Dynamic Reassignment

**Static partitioning** (simple): Fix the prefill and decode pools at startup. E.g., in an 8-GPU node, 2 GPUs for prefill, 6 for decode.

**Dynamic reassignment** (adaptive): Monitor queue depths and reassign GPUs between pools based on workload:

```zig
pub const PoolManager = struct {
    prefill_gpus: std.ArrayList(DeviceId),
    decode_gpus: std.ArrayList(DeviceId),

    /// Called periodically to rebalance pools.
    pub fn rebalance(self: *PoolManager, metrics: Metrics) void {
        const prefill_pressure = metrics.prefill_queue_depth / self.prefill_gpus.items.len;
        const decode_pressure = metrics.decode_queue_depth / self.decode_gpus.items.len;

        if (prefill_pressure > 2 * decode_pressure and self.decode_gpus.items.len > min_decode) {
            // Move a GPU from decode to prefill
            const gpu = self.decode_gpus.pop();
            // Drain in-flight decode requests on this GPU first
            drainAndMigrate(gpu);
            self.prefill_gpus.append(gpu);
        }
        // ... symmetric case for decode pressure
    }
};
```

**Drain before reassignment**: A GPU being reassigned must finish its current requests. For decode→prefill, this means completing all active token generations (or migrating their KV cache to another decode GPU). For prefill→decode, it means completing current prefill and transferring KV.

### 21.4 Prefix-Aware Routing

When using RadixAttention, the scheduler should route prefill requests to the GPU most likely to have the prefix cached:

1. Hash the first `N` tokens of the prompt (the system prompt / prefix)
2. Consistent-hash to a preferred prefill GPU
3. If that GPU is overloaded, fall back to the least-loaded prefill GPU (but it will miss the cache)

This maximizes prefix cache hit rate across the prefill pool. The scheduler maintains a lightweight map of `prefix_hash → preferred_gpu`.

### 21.5 KV Transfer Scheduling Details

**Transfer queue**: Each prefill GPU has a transfer queue. Completed prefill requests are enqueued with their target decode GPU. A dedicated transfer thread processes the queue:

```zig
pub const TransferQueue = struct {
    queue: std.fifo(TransferRequest),

    pub const TransferRequest = struct {
        request_id: u64,
        source_kv: *PagedKvCache,
        source_blocks: []BlockId,
        target_device: DeviceId,
        target_kv: *PagedKvCache,
        /// Compression: quantize to FP8 before transfer.
        compress: bool = true,
        /// Layer-pipelined: signal each layer completion individually.
        layer_pipelined: bool = false,
    };
};
```

**Backpressure**: If the decode pool is full (all blocks allocated), the scheduler must either:
- **Preempt** a low-priority decode request (evict its KV, reuse blocks)
- **Queue** the prefill result on CPU DRAM until decode capacity frees up
- **Reject** the request (return HTTP 503 with retry-after)

### 21.6 Heterogeneous GPU Mixing

Different GPU types can be assigned to each pool based on their strengths:

| GPU | FLOPS (FP16) | HBM BW | Best for |
| :--- | :--- | :--- | :--- |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | Prefill (compute-bound) |
| H200 | 989 TFLOPS | 4.8 TB/s | Either (extra BW helps decode) |
| L40S | 362 TFLOPS | 864 GB/s | Decode (cost-effective BW) |
| A10G | 125 TFLOPS | 600 GB/s | Decode (budget) |
| MI300X | 1307 TFLOPS | 5.3 TB/s | Prefill or either |
| Apple M4 Ultra | ~56 TFLOPS | 819 GB/s | Either (UMA, no transfer cost) |

On Apple Silicon, disaggregation is less beneficial — UMA means the KV cache is already accessible to all compute units without transfer. Chunked prefill (Strategy C) is usually better.

---

## 22. RDMA Transport Layer: libibverbs vs libfabric

### 22.1 Overview

Two competing APIs exist for RDMA network access. The choice affects which network hardware and cloud environments we can support.

| | libibverbs | libfabric (OFI) |
| :--- | :--- | :--- |
| **Full name** | Linux InfiniBand Verbs | Open Fabrics Interfaces |
| **Abstraction level** | Low (close to hardware) | High (provider-based) |
| **Providers** | InfiniBand, RoCE v2 | InfiniBand, RoCE, Slingshot, EFA, SHM, TCP |
| **API style** | Queue pairs, work requests, completion queues | Endpoints, tagged messaging, RMA |
| **GPU RDMA** | Direct (`nvidia_peermem`, `amd_peer_direct`) | Via provider (GNI for Slingshot, EFA for AWS) |
| **Maturity** | Very mature, de facto standard | Mature, adopted by MPI implementations |
| **Complexity** | Higher (manual connection setup, MR management) | Lower (provider handles details) |
| **Cloud support** | Limited (need SR-IOV or bare metal) | AWS EFA, Azure IB (via verbs provider) |

### 22.2 libibverbs: Detailed API Surface

The verbs API we'd need to wrap via `dlopen("libibverbs.so")`:

**Connection setup (one-time):**
```
ibv_get_device_list()          → enumerate RDMA devices
ibv_open_device(dev)           → open device context
ibv_alloc_pd(ctx)              → allocate protection domain
ibv_create_cq(ctx, depth)      → create completion queue
ibv_create_qp(pd, &attr)       → create queue pair (RC or UD)
ibv_modify_qp(qp, &attr, mask) → transition QP: RESET→INIT→RTR→RTS
```

**Memory registration (at init, once per buffer):**
```
ibv_reg_mr(pd, addr, len, access_flags)  → register memory region
  // For GPU memory: addr = cuMemAlloc'd pointer
  // Requires nvidia_peermem kernel module for GPU MR
  // Returns lkey (local) and rkey (remote) for RDMA access
```

**Data transfer (hot path):**
```
ibv_post_send(qp, &wr, &bad_wr)  → post RDMA write/read/send
ibv_post_recv(qp, &wr, &bad_wr)  → post receive buffer
ibv_poll_cq(cq, num, &wc)        → poll for completions
```

**Zig wrapper sketch:**
```zig
pub const Verbs = struct {
    lib: std.DynLib,

    // Function pointers loaded via dlopen
    get_device_list: *const fn (**ibv_device, *c_int) callconv(.C) void,
    open_device: *const fn (*ibv_device) callconv(.C) ?*ibv_context,
    reg_mr: *const fn (*ibv_pd, *anyopaque, usize, c_int) callconv(.C) ?*ibv_mr,
    post_send: *const fn (*ibv_qp, *ibv_send_wr, **ibv_send_wr) callconv(.C) c_int,
    poll_cq: *const fn (*ibv_cq, c_int, *ibv_wc) callconv(.C) c_int,
    // ... ~15-20 functions total

    pub fn init() !Verbs {
        var lib = std.DynLib.open("libibverbs.so.1") catch return error.RdmaUnavailable;
        return .{
            .lib = lib,
            .get_device_list = lib.lookup(@TypeOf(get_device_list), "ibv_get_device_list") orelse return error.SymbolNotFound,
            // ...
        };
    }
};
```

**Pros**: Direct hardware control, maximum performance, well-documented for GPUDirect RDMA.
**Cons**: Linux-only, InfiniBand/RoCE-only, complex connection management.

### 22.3 libfabric: Detailed API Surface

libfabric wraps via `dlopen("libfabric.so")`:

**Connection setup:**
```
fi_getinfo(version, node, service, flags, hints, &info)  → discover providers
fi_fabric(info->fabric_attr, &fabric)                     → open fabric
fi_domain(fabric, info, &domain)                          → open domain
fi_endpoint(domain, info, &ep)                            → create endpoint
fi_ep_bind(ep, &cq->fid, flags)                           → bind CQ to endpoint
fi_enable(ep)                                              → enable endpoint
fi_connect(ep, dest_addr, param, paramlen)                → connect (for connected EPs)
```

**Memory registration:**
```
fi_mr_reg(domain, buf, len, access, offset, key, flags, &mr, context)
  // Provider-specific: some support GPU memory natively (GNI, EFA)
  // Others fall back to host-staged copy
```

**Data transfer:**
```
fi_send(ep, buf, len, desc, dest_addr, context)      → send message
fi_recv(ep, buf, len, desc, src_addr, context)        → post receive
fi_write(ep, buf, len, desc, dest_addr, rkey, ...)    → RDMA write
fi_read(ep, buf, len, desc, dest_addr, rkey, ...)     → RDMA read
fi_cq_read(cq, &entry, count)                         → poll completions
```

**Pros**: Portable across fabrics (IB, Slingshot, EFA, TCP fallback), cleaner API, single abstraction for all networks.
**Cons**: Extra indirection layer, GPU RDMA support varies by provider, less control over low-level tuning.

### 22.4 Recommendation

**Wrap both, with runtime selection:**

```zig
pub const RdmaTransport = union(enum) {
    verbs: VerbsTransport,    // InfiniBand/RoCE direct
    fabric: FabricTransport,  // libfabric (EFA, Slingshot, fallback)
    tcp: TcpTransport,        // Pure Zig TCP (no external deps, slowest)

    pub fn init() RdmaTransport {
        // Prefer verbs for bare metal / on-prem with IB
        if (VerbsTransport.probe()) |v| return .{ .verbs = v };
        // Fall back to libfabric for cloud / exotic networks
        if (FabricTransport.probe()) |f| return .{ .fabric = f };
        // Last resort: TCP
        return .{ .tcp = TcpTransport.init() };
    }
};
```

- **On-prem with InfiniBand**: `libibverbs` for maximum performance and GPUDirect RDMA
- **AWS (EFA)**: `libfabric` with EFA provider (verbs not available on EFA)
- **Cray/HPE (Slingshot)**: `libfabric` with GNI or CXI provider
- **Commodity Ethernet**: TCP fallback (pure Zig, no deps)
- Auto-detect at runtime: try `dlopen("libibverbs.so.1")` first, then `dlopen("libfabric.so.1")`, then TCP

The `CommOps` interface (§8.2) is transport-agnostic — `allReduce`, `send`, `recv` dispatch through `RdmaTransport` regardless of which backend is active.

### 22.5 GPUDirect RDMA Compatibility

| Transport | GPU RDMA | Mechanism |
| :--- | :--- | :--- |
| libibverbs + InfiniBand | Yes | `nvidia_peermem` / `amd_peer_direct` kernel module |
| libibverbs + RoCE v2 | Yes | Same kernel modules |
| libfabric + EFA (AWS) | Partial | EFA SRD supports GPU memory via `p2p` provider (requires `aws-neuron-driver` or CUDA 12.x `dmabuf`) |
| libfabric + Slingshot | Yes | GNI/CXI providers support GPU MR natively on Frontier/El Capitan |
| libfabric + TCP | No | Host-staged copy required |

When GPU RDMA isn't available, fall back to the two-copy path: GPU → `cudaMemcpy(D2H)` → host buffer → RDMA send → host buffer → `cudaMemcpy(H2D)` → GPU. This is transparent to the `CommOps` caller.

---

## 23. CXL Memory Pooling

### 23.1 What is CXL?

**Compute Express Link (CXL)** is a cache-coherent interconnect built on PCIe 5.0/6.0 physical layer. CXL 3.0 (ratified 2023, hardware shipping 2025-2026) introduces **shared memory pooling** — multiple hosts and accelerators can access a common memory pool with load/store semantics and hardware-enforced coherence.

```
Traditional (discrete memory per device):
  Host 0: [DDR 256GB]   GPU 0: [HBM 80GB]   GPU 1: [HBM 80GB]
  ──── PCIe ──── PCIe ──── NVLink ────

CXL 3.0 (shared memory pool):
  Host 0 ──┐                    ┌── GPU 0
  Host 1 ──┤── CXL Switch ─────┤── GPU 1
  Host 2 ──┤   [Memory Pool    ┤── GPU 2
            │    2 TB DDR5]     │
            └───────────────────┘
```

### 23.2 Relevance to Agave

CXL could fundamentally change how parallelism works for inference:

**KV cache in shared CXL memory:**
- All GPUs load/store KV cache from a shared pool — no explicit transfers
- Migration = changing a pointer in the block table (zero-copy)
- Prefill GPU writes KV to pool, decode GPU reads it — no DMA, no RDMA
- RadixAttention prefix sharing becomes trivial (all nodes see the same memory)

**Weight sharing:**
- Model weights in CXL pool, accessed by all GPUs
- No per-GPU weight replication (saves N× memory)
- Bandwidth is lower than HBM (~128 GB/s per CXL link vs 3+ TB/s HBM), so weights must still be cached in local HBM for hot-path access

**Disaggregated memory expansion:**
- A 2 TB CXL memory pool can hold KV cache for thousands of concurrent requests
- GPU HBM holds only active/hot KV blocks; cold blocks live in CXL pool
- Access latency: ~150-300 ns (vs ~100 ns for local DDR, ~10 ns for HBM)

### 23.3 CXL Hardware Landscape

| Product | Type | Capacity | Bandwidth | Status |
| :--- | :--- | :--- | :--- | :--- |
| Samsung CXL DRAM (CMM-D) | Type 3 Memory Expander | 128-512 GB | 64 GB/s (CXL 2.0) | Shipping |
| Micron CZ120 | Type 3 Memory Expander | 256 GB | 36 GB/s | Shipping |
| Samsung CMM-H (HBM+CXL) | Type 3 with HBM | 128 GB HBM | 256+ GB/s | Announced |
| Intel / AMD CXL switches | CXL 3.0 Switch | N/A | Multi-port | Sampling 2025-2026 |
| Astera Labs Aries | CXL Switch | N/A | PCIe 6.0 | Sampling |

### 23.4 Programming Model

CXL memory appears as a **NUMA node** to the OS. No special APIs needed for basic access:

```zig
// CXL memory exposed as NUMA node 2 (after node 0 = socket 0, node 1 = socket 1)
// Linux: same mbind/mmap interface as regular NUMA
const cxl_numa_node = 2;
const pool_mem = numaAlloc(pool_size, cxl_numa_node);

// GPU access: map CXL memory into GPU address space
// CUDA: cuMemMap with CXL-backed allocation
// This is where CXL 3.0 coherence matters — GPU cache snoops CXL
```

For our `PagedKvCache`, CXL integration would be:

```zig
pub const CacheBlock = struct {
    keys: []f32,
    values: []f32,
    used: u16 = 0,
    ref_count: u16 = 1,
    tier: StorageTier = .gpu,
    // ...
};

pub const StorageTier = enum {
    gpu,     // HBM — fastest, limited
    cpu,     // Local DDR — larger, slower
    cxl,     // CXL pool — shared, coherent, larger still
    nvme,    // NVMe SSD — largest, slowest
};
```

**Tiered access pattern:**
1. **Hot blocks** (current token ± sliding window): GPU HBM
2. **Warm blocks** (recent positions, likely to be accessed): CXL pool (shared, 150ns)
3. **Cold blocks** (old positions, unlikely to be accessed): CPU DRAM or NVMe

### 23.5 Impact on Design

If CXL memory pooling is available, several components simplify dramatically:

| Component | Without CXL | With CXL |
| :--- | :--- | :--- |
| KV cache migration | Explicit DMA/RDMA copy | Update block table pointer |
| Disaggregated prefill/decode | KV transfer is critical path | Zero-copy (shared pool) |
| RadixAttention cross-node | Distributed hash table + RDMA fetch | Direct memory access to shared trie |
| Prefix sharing | Per-node replication | Single shared copy |
| KV offloading | GPU→CPU→NVMe pipeline | GPU→CXL (always accessible) |

### 23.6 Limitations and Timeline

**Current limitations:**
- CXL 3.0 switches (required for true multi-host sharing) are still sampling; volume production expected late 2025-2026
- GPU support for CXL is limited — NVIDIA hasn't announced native CXL support in current architectures (Hopper/Blackwell use NVLink). AMD MI300X has CXL support but primarily for memory expansion, not shared pooling
- CXL bandwidth (~64-256 GB/s) is much lower than NVLink (900 GB/s) or HBM (3+ TB/s), so it's unsuitable for TP all-reduce. Best for KV cache and weight storage where access patterns are streaming, not reduction-heavy
- Cache coherence overhead adds latency vs raw RDMA

**Agave timeline recommendation:**
- **Now**: Design `StorageTier` enum and `PagedKvCache` offloading with CXL as a planned tier
- **Phase 5+**: When CXL 3.0 hardware is available, add CXL pool allocator that exposes CXL memory as a NUMA node. The rest of the system (block allocation, eviction, prefetch) works unchanged — CXL is just another tier with different latency/capacity characteristics
- **No code changes needed for CXL in the hot path**: load/store access means the SDPA kernel reads CXL-backed KV cache the same way it reads HBM-backed KV cache — the MMU handles the routing transparently. The only change is in the allocator and eviction policy.

---

## 24. Open Questions

1. **NCCL/RCCL dlopen interface**: The plan is to support both custom Zig ring all-reduce (default) and NCCL/RCCL via `dlopen` (preferred when available). The `dlopen` wrapper needs to cover: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclGroupStart/End`, `ncclCommDestroy`. Should we also wrap `ncclSend`/`ncclRecv` for PP activation transfer, or keep PP on raw CUDA/HIP P2P?

2. **Heterogeneous parallelism**: Can we do TP across mixed backends (e.g., Metal + CPU on Apple Silicon)? The CPU could handle overflow heads when `n_kv_heads` isn't evenly divisible by TP. Worth exploring for UMA platforms.

3. **Speculative decoding + parallelism**: If we add speculative decoding (draft model generates candidates, large model verifies), how does parallelism interact? The draft model likely runs on a single device while the large model is parallelized. KV cache management gets complex.

4. **Dynamic TP degree**: Can we adjust TP degree at runtime based on available devices? E.g., if a GPU becomes unavailable, fall back from TP=4 to TP=2 with model re-sharding. Low priority but interesting for resilience.

5. **Quantized communication**: For TP all-reduce, can we reduce in FP8/BF16 instead of F32? Each device produces F32 partial sums, but we could quantize before transfer and dequantize after. Saves 2-4× bandwidth at the cost of ~0.1% accuracy loss. This is increasingly standard in large-scale training; less explored for inference.

6. **Fault tolerance during parallel inference**: What happens when a device fails mid-forward-pass? Options: (a) abort the request and retry on remaining devices with lower TP degree, (b) checkpoint KV cache periodically for fast recovery, (c) replicate KV cache across devices for instant failover. The cost-benefit depends on failure frequency and SLA requirements.

7. **Profiling parallel overhead**: How to measure and attribute communication overhead vs compute in a distributed forward pass? Need per-layer timing that separates GEMV time from all-reduce time from idle/bubble time. Tracy integration (§12 in CLAUDE.md) should be extended with distributed trace correlation.
