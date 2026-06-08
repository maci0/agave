# Audit Batch 4 — Documentation vs. Source Code Cross-Reference

**Date**: 2026-06-03
**Auditor**: Documentation subagent
**Files audited**: PARALLELISM.md, BENCHMARKS.md, TEST_MATRIX.md, DOCUMENTATION.md, TODO.md

---

## Evidence Table

| # | Source | URL | Key claim / finding | Type | Confidence |
|---|--------|-----|---------------------|------|------------|
| 1 | src/parallel/transport.zig:27 | local | TransportKind = { tcp, shm, nccl, rccl } — 4 variants | primary | high |
| 2 | src/parallel/transport.zig:66-70 | local | ShmHeader: ready (atomic u32) + size (u32) + _pad ([56]u8) = 64 bytes | primary | high |
| 3 | src/parallel/transport.zig:12 | local | shm_buf_size = 16 MB | primary | high |
| 4 | src/parallel/transport.zig:268-271 | local | NCCL loaded via dlopen("libnccl.so.2") with fallbacks | primary | high |
| 5 | src/parallel/transport.zig:275-282 | local | NCCL resolves: ncclGetUniqueId, ncclCommInitRank, ncclAllReduce, ncclSend, ncclRecv, ncclCommDestroy, ncclGroupStart, ncclGroupEnd | primary | high |
| 6 | src/parallel/transport.zig:338-355 | local | ensureNcclComm() called in allReduceAdd, sendBuf, AND recvBuf | primary | high |
| 7 | src/parallel/discovery.zig:16 | local | discovery_port = 49460 | primary | high |
| 8 | src/parallel/discovery.zig:53,64,247,268 | local | Rank 0 binds port 49460 (recv), broadcasts to port 49461; workers bind 49461, reply to 49460 | primary | high |
| 9 | src/devices/discovery.zig:12 | local | BackendKind = { cpu, metal, cuda, rocm, vulkan } — no WebGPU | primary | high |
| 10 | src/main.zig:29 | local | TransportChoice = { auto, tcp, shm, nccl, rdma, udp, grpc } | primary | high |
| 11 | src/main.zig:360-368 | local | CLI specs: --backend, --device, --list-devices, --disagg, --tp, --pp, --peers, --rank, --transport | primary | high |
| 12 | src/main.zig:118 | local | Transport.init rejects rccl with error.NotImplemented | primary | high |
| 13 | src/backend/cuda.zig:424-432 | local | CUDA uses cuDevicePrimaryCtxRetain with cuCtxCreate fallback | primary | high |
| 14 | src/backend/backend.zig:888,901 | local | Backend has getDevicePtr() and invalidateWeight() | primary | high |
| 15 | docs/TEST_MATRIX.md | local | PP=2 0.8B = "8.5 tok/s, 93% of single GPU" | self-reported | high |
| 16 | docs/BENCHMARKS.md | local | PP=2 0.8B = "40.2 tok/s, 112%" | self-reported | high |
| 17 | docs/TEST_MATRIX.md | local | Qwen 3.5 0.8B Agave = 183.3 tok/s (llama.cpp comparison) | self-reported | medium |
| 18 | docs/BENCHMARKS.md | local | Qwen3.5 0.8B Agave Metal = 125† tok/s | self-reported | medium |
| 19 | src/parallel/transport.zig (no librccl) | local | No dlopen("librccl.so") anywhere in source | primary | high |
| 20 | mega_common.metal | local | 732 lines — matches doc claim | primary | high |
| 21 | megakernel files | local | 5,370 lines across 13 files — doc claims ~4,166 lines across 12 files | primary | high |
| 22 | ROCm kernels | local | 45 kernel files (excl. all.zig) — doc claims 44 | primary | medium |
| 23 | Vulkan shaders | local | 46 .comp files — doc claims 44 | primary | medium |

---

## Findings by File

### 1. PARALLELISM.md

#### [MEDIUM] PARALLELISM.md: NCCL function pointer list incomplete
  Doc claims: "Function pointers resolved: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclAllReduce`, `ncclSend`, `ncclRecv`, `ncclCommDestroy`."
  Source says: transport.zig:281-282 also resolves `ncclGroupStart` and `ncclGroupEnd`.
  Fix: Add `ncclGroupStart`, `ncclGroupEnd` to the function pointer list.

#### [LOW] PARALLELISM.md: NCCL init deferred description too narrow
  Doc claims: "Both ranks call `ncclCommInitRank` — **deferred** to first `allReduceAdd` call"
  Source says: `ensureNcclComm()` is called in `allReduceAdd` (transport.zig:377), `sendBuf` (transport.zig:464), AND `recvBuf` (transport.zig:538). It's deferred to the first NCCL operation, not specifically `allReduceAdd`.
  Fix: Change to "deferred to first NCCL operation (`allReduceAdd`, `sendBuf`, or `recvBuf`)"

#### [LOW] PARALLELISM.md: RCCL dlopen claim is speculative
  Doc claims: "RCCL ... will use `dlopen("librccl.so")` when available."
  Source says: No `librccl` string exists anywhere in the source [19]. The RCCL enum variant exists in TransportKind but there is zero implementation code for it. `Transport.init` rejects `.rccl` with `error.NotImplemented` (transport.zig:119).
  Fix: Change to "RCCL is declared in TransportKind but has no implementation. The planned library name is `librccl.so`." or remove the speculative dlopen claim entirely.

#### [LOW] PARALLELISM.md: Device discovery BackendKind missing WebGPU
  Doc claims: "**Backends**: All GPU backends (Metal, CUDA, Vulkan, ROCm, WebGPU) + CPU"
  Source says: `BackendKind` in `src/devices/discovery.zig:12` is `enum { cpu, metal, cuda, rocm, vulkan }` — WebGPU is not enumerable via `--list-devices`. WebGPU exists as a compute backend (`src/backend/webgpu.zig`) but is not in device discovery.
  Fix: Clarify that device discovery covers Metal/CUDA/Vulkan/ROCm, while WebGPU is a compute backend that doesn't participate in device enumeration.

#### [INFO] PARALLELISM.md: Transport API signature verified ✓
  `Transport.init(allocator, kind, rank, world_size)` matches source (transport.zig:118).
  `allReduceAdd(buf, n)`, `sendBuf(buf, n)`, `recvBuf(buf, n)` all match source.

#### [INFO] PARALLELISM.md: ShmHeader description verified ✓
  "64-byte header (`ShmHeader` with atomic `ready` flag + `size`)" — exact match (transport.zig:66-70).
  "16 MB data" matches `shm_buf_size = 16 * 1024 * 1024` (transport.zig:12).

#### [INFO] PARALLELISM.md: Shm naming convention verified ✓
  "Rank 0 creates `/agave_0to1` (send), opens `/agave_1to0` (recv)" — exact match (transport.zig:188-189).

#### [INFO] PARALLELISM.md: UDP discovery port verified ✓
  "port 49460" matches `discovery_port: u16 = 49460` (discovery.zig:16).

#### [INFO] PARALLELISM.md: CLI flags verified ✓
  All 8 flags (`--tp`, `--pp`, `--rank`, `--peers`, `--transport`, `--disagg`, `--device`, `--list-devices`) exist in the ArgSpec table (main.zig:360-368) with correct kinds (flags vs options).

#### [INFO] PARALLELISM.md: CUDA primary context claim verified ✓
  "CUDA backend must use `cuDevicePrimaryCtxRetain`" — source (cuda.zig:424-425) tries `cuDevicePrimaryCtxRetain` first, falls back to `cuCtxCreate`.

#### [INFO] PARALLELISM.md: Key Files table verified ✓
  All listed files exist: `src/parallel/transport.zig`, `src/parallel/tp.zig`, `src/parallel/discovery.zig`, `src/main.zig`, `src/models/qwen35.zig`, `src/devices/discovery.zig`, and the backend files.

#### [INFO] PARALLELISM.md: --transport auto behavior verified ✓
  "same-node peers → shm, otherwise → tcp" matches `resolveTransportKind()` (main.zig:1112-1127).

---

### 2. BENCHMARKS.md

#### [HIGH] BENCHMARKS.md + PARALLELISM.md: PP=2 0.8B benchmark internally inconsistent with TEST_MATRIX.md
  BENCHMARKS.md claims: "Qwen3.5 0.8B Q8_0 PP=2 NCCL RoCE: **40.2** tok/s, 112% vs Single GPU"
  PARALLELISM.md claims: Same data — "40.2 tok/s, 112%"
  TEST_MATRIX.md claims: "PP=2 dual GB10 NCCL RoCE 0.8B Q8_0: PASS, 8.5 tok/s, 93% of single GPU"
  Source says: These are contradictory across the docs. 40.2 tok/s is 437% of the 9.2 single-GPU baseline, making "112%" mathematically wrong regardless. TEST_MATRIX.md's 8.5 tok/s (93%) is the more physically plausible number for PP=2 single-token decode.
  Fix: Reconcile all three documents. If the correct value is 8.5 tok/s (93%), update BENCHMARKS.md and PARALLELISM.md. If 40.2 is real (perhaps from a batched/profiled scenario), clarify the methodology and fix the percentage.

#### [HIGH] BENCHMARKS.md vs TEST_MATRIX.md: Qwen 3.5 0.8B Metal throughput contradiction
  BENCHMARKS.md claims: "Qwen3.5 0.8B Q8_0 Agave Metal: 125† tok/s"
  TEST_MATRIX.md claims: "Qwen 3.5 0.8B Q8_0: 183.3 tok/s" (llama.cpp comparison table)
  Source says: These are from the same hardware (M4 Pro Metal) but show different numbers. The BENCHMARKS.md note "†Updated 2026-05-26 with sparse GEMV" suggests 125 is more recent.
  Fix: Update TEST_MATRIX.md's llama.cpp comparison table to use the 125 tok/s figure from the updated BENCHMARKS.md, or explain the discrepancy (different measurement methodology, different prompt lengths, etc.).

#### [MEDIUM] BENCHMARKS.md: Megakernel line count and file count outdated
  Doc claims: "Total megakernel code: ~4,166 lines across 12 files."
  Source says: Actual count is 5,370 lines across 13 files (5 Metal model-specific + mega_common.metal + megakernel.metal + 3 CUDA + 1 ROCm + megakernel.zig + mega_compose.zig).
  Fix: Update to "~5,370 lines across 13 files."

#### [LOW] BENCHMARKS.md: Vulkan shader count slightly off
  Doc (TODO.md): "44 shaders" for Vulkan
  Source says: 46 `.comp` shader files in `src/backend/kernels/vulkan/`, 92 total files (including pre-compiled `.spv`).
  Fix: Update to "46 shaders".

#### [LOW] BENCHMARKS.md: ROCm kernel count slightly off
  Doc (TODO.md): "44 kernels" for ROCm
  Source says: 45 kernel files (excluding `all.zig`) in `src/backend/kernels/rocm/`.
  Fix: Update to "45 kernels".

#### [INFO] BENCHMARKS.md: mega_common.metal line count verified ✓
  "732 lines" exactly matches `wc -l src/backend/kernels/metal/mega_common.metal` = 732.

#### [INFO] BENCHMARKS.md: CLI flags verified ✓
  `--megakernel` (main.zig:407), `--prefill-batch-size` (main.zig:372), `--kv-eviction` (main.zig:383) all exist with correct kinds.

---

### 3. TEST_MATRIX.md

#### [MEDIUM] TEST_MATRIX.md: "8/9 architectures" counts quant variants as architectures
  Doc claims: "8/9 architectures pass on Metal+CPU"
  Source says: Rows 6 and 7 are both Qwen 3.5 9B with different quants (Q4_K_M and Q8_0). That's one architecture, not two. Actual unique architectures tested: Gemma 4 26B-A4B, Gemma 4 E2B, Gemma 4 E4B, Gemma 3 27B, Qwen 3.5 0.8B, Qwen 3.5 9B, GLM-4.7, Nemotron-Nano 4B = 8 architectures (7 pass + 1 fail).
  Fix: Change to "7/8 architectures pass" or re-label as "8/9 test configs pass" since two configs test the same architecture.

#### [HIGH] TEST_MATRIX.md: PP=2 0.8B contradicts BENCHMARKS.md (see Finding above)
  Doc claims: "PP=2 dual GB10 NCCL RoCE 0.8B Q8_0: 8.5 tok/s, 93% of single GPU"
  BENCHMARKS.md claims: "40.2 tok/s, 112%"
  Fix: Reconcile across documents.

---

### 4. DOCUMENTATION.md

#### [INFO] DOCUMENTATION.md: All tutorial links verified ✓
  All 19 tutorial chapters + 4 appendices + README.md exist in `docs/tutorial/` with exact filename matches. No broken links.

#### [INFO] DOCUMENTATION.md: Product documentation links reference existing files ✓
  All referenced .md files (API.md, ARCHITECTURE.md, MODELS.md, KERNELS.md, MEGAKERNEL.md, BENCHMARKS.md, CONTRIBUTING.md, TEST_MATRIX.md, PARALLELISM.md, TODO.md, ../CHANGELOG.md) are referenced consistently.

---

### 5. TODO.md

#### [MEDIUM] TODO.md: CUDA kernel count says "56 kernels" but actual count is 58 files
  Doc claims: "CUDA: Complete — 56 kernels, fused FFN, 3 megakernels"
  Source says: `find src/backend/kernels/cuda/ -name '*.zig' | wc -l` = 58 files.
  Fix: Update to "58 kernels".

#### [MEDIUM] TODO.md: Vulkan shader count says "44 shaders" but actual is 46
  Doc claims: "Vulkan: Complete — 44 shaders, deferred dispatch"
  Source says: 46 `.comp` files in `src/backend/kernels/vulkan/`.
  Fix: Update to "46 shaders".

#### [MEDIUM] TODO.md: ROCm kernel count says "44 kernels" but actual is 45
  Doc claims: "ROCm: Complete — 44 kernels, GPTQ, 1 megakernel"
  Source says: 45 kernel files (excluding `all.zig`) in `src/backend/kernels/rocm/`.
  Fix: Update to "45 kernels".

#### [LOW] TODO.md: WebGPU shader count says "43 shaders" but actual is 45
  Doc claims: "WebGPU: Complete — 43 shaders, lazy readback"
  Source says: 45 files in `src/backend/kernels/webgpu/`.
  Fix: Update to "45 shaders".

#### [INFO] TODO.md: Sparse GEMV correctly marked as Done ✓
  Item #26 is struck through as done. Source confirms sparse GEMV exists: `sparse_threshold` in `src/backend/kernels/cpu/gemv.zig:15`, `isBlockSparse` used in gemv_f16.zig, gemv_fp8.zig.

#### [INFO] TODO.md: `agave pull` correctly marked as Done ✓
  Source confirms: `src/pull.zig` exists with full HuggingFace download implementation.

#### [INFO] TODO.md: Jump decoding correctly marked as Done ✓
  Source confirms: jump decode references in `src/server/server.zig:2636,4367`.

#### [INFO] TODO.md: Accelerate.framework correctly marked as Done ✓
  Source confirms: `src/backend/accelerate.zig` exists, imported in `src/backend/kernels/cpu/gemv.zig:77`.

---

## Summary of Issues by Severity

| Severity | Count | Issues |
|----------|:-----:|--------|
| HIGH | 2 | PP=2 benchmark contradiction (40.2 vs 8.5 tok/s), Qwen 0.8B Metal throughput contradiction (125 vs 183.3) |
| MEDIUM | 6 | NCCL function list incomplete, architecture count wrong, CUDA/Vulkan/ROCm/WebGPU kernel counts stale |
| LOW | 5 | NCCL deferred init description narrow, RCCL dlopen speculative, BackendKind missing WebGPU note, Vulkan/ROCm counts slightly off |
| INFO | 10 | Verified correct claims |

## Coverage Status

### Checked directly:
- ✅ All TransportKind enum variants against transport.zig
- ✅ All CLI flags (--tp, --pp, --rank, --peers, --transport, --disagg, --device, --list-devices) against main.zig ArgSpec
- ✅ ShmHeader struct layout and constants
- ✅ NCCL dlopen mechanism, function pointers, deferred init
- ✅ POSIX shm naming convention (/agave_0to1, /agave_1to0)
- ✅ UDP discovery port (49460) and protocol
- ✅ CUDA primary context (cuDevicePrimaryCtxRetain) usage
- ✅ Backend getDevicePtr()/invalidateWeight() existence
- ✅ shardColumnWeight/shardRowWeight existence in qwen35.zig
- ✅ All tutorial file links in DOCUMENTATION.md
- ✅ Key Files table (all files exist)
- ✅ Megakernel file counts, line counts
- ✅ Kernel counts for all backends (CUDA, Vulkan, ROCm, WebGPU, Metal)
- ✅ TODO items marked as Done (sparse GEMV, agave pull, jump decoding, Accelerate)
- ✅ Cross-document benchmark consistency (BENCHMARKS.md vs TEST_MATRIX.md vs PARALLELISM.md)

### Not checked (out of scope or blocked):
- ⬜ Actual benchmark numbers (require hardware replication)
- ⬜ Whether Metal pipeline count is truly "70+" (would need to count PSO creation calls, not just .metal files)
- ⬜ CUDA "56 PTX kernels loaded via sm_90 forward compatibility" — can't verify kernel loading without hardware
- ⬜ Correctness of Megatron-LM sharding pattern description (would need deep model code review)
