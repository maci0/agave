# Agave — Distributed Inference

**Status**: Implemented  
**Scope**: Tensor Parallelism (TP), Pipeline Parallelism (PP), Hybrid TP+PP, Disaggregated Prefill/Decode  
**Transports**: TCP, POSIX Shared Memory, NCCL (RoCE RDMA)  
**Backends**: All GPU backends (Metal, CUDA, Vulkan, ROCm, WebGPU) + CPU

---

## Quick Start

```bash
# Same-node TP (auto-selects POSIX shm)
agave model.gguf --tp 2 --rank 0 --peers localhost "prompt"    # terminal 1
agave model.gguf --tp 2 --rank 1 --peers localhost "prompt"    # terminal 2

# Cross-node PP over TCP
agave model.gguf --pp 2 --rank 0 --peers 10.0.1.2 "prompt"    # node A
agave model.gguf --pp 2 --rank 1 --peers 10.0.1.1 "prompt"    # node B

# Cross-node TP over NCCL RoCE RDMA
agave model.gguf --tp 2 --rank 0 --peers 10.0.1.2 --transport nccl "prompt"
agave model.gguf --tp 2 --rank 1 --peers 10.0.1.1 --transport nccl "prompt"

# Hybrid TP+PP (4 GPUs: 2 TP groups × 2 PP stages)
agave model.gguf --tp 2 --pp 2 --rank 0 --peers 10.0.1.2 "prompt"

# Disaggregated prefill/decode
agave model.gguf --disagg --rank 0 --peers 10.0.1.2 "prompt"  # prefill node
agave model.gguf --disagg --rank 1 --peers 10.0.1.1 "prompt"  # decode node

# GPU selection
agave model.gguf --list-devices            # enumerate GPUs
agave model.gguf --backend vulkan --device 1   # select GPU by index
```

---

## Architecture

### Transport Layer

All distributed communication goes through `src/parallel/transport.zig`. Three transport backends:

| Transport | Mechanism | Best For | Bandwidth |
| :--- | :--- | :--- | :--- |
| **TCP** | BSD sockets, full send/recv loops | Cross-node, no RDMA | ~10 Gbps |
| **POSIX shm** | `shm_open` + `mmap`, atomic spin-wait | Same-node, zero-copy | Memory bandwidth |
| **NCCL** | `dlopen("libnccl.so.2")`, GPU-direct | CUDA multi-GPU, RoCE RDMA | Up to 400 Gbps |

Auto-selection (`--transport auto`): same-node peers (`localhost`/`127.0.0.1`) → shm, otherwise → tcp.

### Transport API

```
Transport.init(allocator, kind, rank, world_size)
Transport.allReduceAdd(buf, n)     — TP: sum partial results across ranks
Transport.sendBuf(buf, n)          — PP: send activation to next stage
Transport.recvBuf(buf, n)          — PP: receive activation from previous stage
```

All three operations dispatch to the active transport (TCP, shm, or NCCL) internally.

### NCCL Integration

NCCL is loaded at runtime via `std.DynLib` — no compile-time linking, no vendored C code. Function pointers resolved: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclSend`, `ncclRecv`, `ncclCommDestroy`.

**Initialization sequence:**
1. TCP connection established between ranks (standard `connect`/`accept`)
2. Rank 0 calls `ncclGetUniqueId`, sends 128-byte ID over TCP to rank 1
3. Both ranks call `ncclCommInitRank` — **deferred** to first `allReduceAdd` call (after CUDA kernels have initialized the primary context)
4. After `ncclCommInitRank`, restore CUDA context via `cuCtxSetCurrent` (NCCL may change active context)

**Critical requirement**: CUDA backend must use `cuDevicePrimaryCtxRetain` (not `cuCtxCreate`). NCCL's runtime API uses the primary context; a separate driver API context causes context corruption and wrong results.

**Device pointer optimization**: When the CUDA activation cache has data dirty on GPU (written by a kernel, not yet downloaded), `allReduceAdd` passes the device pointer directly to NCCL — no host↔device copy. When data is stale (CPU fallback wrote to host), uploads to a device staging buffer first, then calls `ncclAllReduce`, then downloads result.

### POSIX Shared Memory

Two shared memory regions per rank pair:
- Rank 0 creates `/agave_0to1` (send), opens `/agave_1to0` (recv)
- Rank 1 creates `/agave_1to0` (send), opens `/agave_0to1` (recv)

Each region: 16 MB data + 64-byte header (`ShmHeader` with atomic `ready` flag + `size`). Spin-wait with `std.atomic.spinLoopHint()` for synchronization. Zero-copy: data written directly to shared mapping, read directly from peer's mapping.

---

## Tensor Parallelism (TP)

### Weight Sharding

Follows the Megatron-LM pattern — 2 all-reduces per transformer layer:

**Attention block:**
- Q, K, V projections → **column-parallel** (each rank gets `n_heads/tp_degree` heads)
- Output projection (W_o) → **row-parallel** (each rank holds rows, results summed)
- One `allReduceAdd` after W_o

**FFN block (SwiGLU/GELU):**
- W_gate, W_up → **column-parallel** (each rank gets `n_ff/tp_degree` columns)
- W_down → **row-parallel**
- One `allReduceAdd` after W_down

**Implementation in model code** (e.g., `src/models/qwen35.zig`):
- `shardColumnWeight(tensor, n_rows, k)` — returns pointer offset for this rank's column shard
- `shardRowWeight(tensor, n, k_total, shard_buf)` — copies this rank's row shard into contiguous buffer, calls `be.invalidateWeight()` to evict stale GPU cache entry

TP degree must divide both `n_heads` and `n_kv_heads`. Embedding table is replicated (small relative to model). Final logits computed on each rank independently.

### Norms

RMS norm operates on the full hidden state. Current implementation: each rank computes norm on its local shard. This works because norm is applied **before** the sharded GEMV, when the hidden state is still full (after the previous layer's all-reduce).

---

## Pipeline Parallelism (PP)

Layers split into `pp_degree` contiguous stages. Stage assignment: `layer * pp_degree / n_layers`.

**Stage 0** (first): embedding lookup + first `n_layers/pp` layers  
**Stage N-1** (last): remaining layers + lm_head + argmax

**Activation transfer**: hidden state vector (`n_embd × f32` = 4-32 KB) sent via `sendBuf`/`recvBuf` between stages. Tiny relative to interconnect bandwidth.

**Decode loop:**
- Stage 0: forward through its layers → `sendBuf(hidden)` to stage 1 → `recvBuf(token)` from stage 1
- Stage 1: `recvBuf(hidden)` from stage 0 → forward through its layers → argmax → `sendBuf(token)` to stage 0

**Bubble**: During single-token decode, pipeline utilization is `1/pp_degree`. The primary benefit of PP is fitting larger models in memory, not throughput improvement.

---

## Disaggregated Prefill/Decode

`--disagg` mode splits the workload:
- **Rank 0 (prefill node)**: processes the prompt, builds KV cache, sends KV blocks to rank 1
- **Rank 1 (decode node)**: receives KV cache, runs autoregressive decode

KV cache transfer: sends metadata (n_blocks, block_size, n_layers) followed by per-block key/value f32 data via `sendBuf`/`recvBuf`.

---

## Benchmarks

Tested on dual NVIDIA GB10 (Blackwell sm_121) nodes with 4× ConnectX NICs each, RoCE RDMA.

| Model | Config | Transport | tok/s | vs Single GPU |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3.5 0.8B Q8_0 | PP=2 | TCP | 5.1 | 54% |
| Qwen3.5 0.8B Q8_0 | PP=2 | NCCL | 8.5 | 93% |
| Qwen3.5 0.8B Q8_0 | TP=2 | TCP | 3.2 | 34% |
| Qwen3.5 0.8B Q8_0 | TP=2 | NCCL | 5.1 | 56% |
| Qwen3.5 0.8B Q8_0 | Single | — | 9.2 | 100% |

PP=2 with NCCL achieves 93% of single-GPU throughput. TP=2 has higher overhead due to 2 all-reduces per layer.

---

## CLI Reference

| Flag | Description |
| :--- | :--- |
| `--tp N` | Tensor parallelism degree (split weights across N ranks) |
| `--pp N` | Pipeline parallelism degree (split layers across N stages) |
| `--rank N` | This node's rank (0-indexed) |
| `--peers HOST[:PORT]` | Peer address for distributed inference |
| `--transport TYPE` | Transport: `auto`, `tcp`, `shm`, `nccl` |
| `--disagg` | Disaggregated prefill/decode mode |
| `--list-devices` | Show available GPUs |
| `--device N` | Select GPU by index |

`--transport auto` selects shm for localhost peers, tcp otherwise. NCCL must be explicitly requested.

---

## Key Files

| File | Purpose |
| :--- | :--- |
| `src/parallel/transport.zig` | Transport layer: TCP, shm, NCCL |
| `src/main.zig` | CLI parsing, transport setup, NCCL wiring |
| `src/models/qwen35.zig` | TP/PP model integration (sharding, all-reduce, send/recv) |
| `src/backend/cuda.zig` | CUDA primary context, device pointer lookup for NCCL |
| `src/backend/backend.zig` | Backend dispatcher with `invalidateWeight()`, `getDevicePtr()` |
| `src/devices/discovery.zig` | GPU device enumeration (`--list-devices`) |

---

## Known Limitations

- **TP degree**: must divide `n_heads` and `n_kv_heads`
- **NCCL**: requires `libnccl.so.2` at runtime (not bundled)
- **NCCL primary context**: CUDA must use `cuDevicePrimaryCtxRetain` — `cuCtxCreate` will cause context corruption after `ncclCommInitRank`
- **K-quant CPU fallback on UMA**: Q4_K/Q5_K/Q6_K delegate to CPU on GB10 sm_121 (PTX register spilling). CPU allReduceAdd uploads to device staging buffer for NCCL
- **PP bubble**: single-token decode has `1/pp_degree` utilization; only worthwhile for fitting larger models
- **2 ranks only**: current transport supports rank 0 ↔ rank 1 pair. Multi-rank ring/tree not yet implemented

---

## Future Work

- Multi-rank ring all-reduce (>2 GPUs)
- Expert parallelism for MoE models
- Quantized communication (bf16/fp8 all-reduce)
- KV cache sharing over RDMA for disaggregated serving
- RCCL support for ROCm multi-GPU (same API as NCCL)
