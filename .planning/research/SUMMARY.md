# Research Summary — Production LLM Inference Engine

**Project:** Agave LLM Inference Engine
**Research Completed:** 2026-03-21
**Scope:** GPU kernel optimization and production serving capabilities

---

## Executive Summary

This research investigated the technologies, features, architecture patterns, and pitfalls for transforming Agave from a high-performance single-request inference engine into a production-ready multi-tenant serving system with continuous batching and GPU kernel parity.

**Current State:** Agave has a solid foundation with pure Zig implementation, multi-backend support (CPU, Metal, CUDA, Vulkan), extensive quantization formats, and correct output on tested models. However, critical gaps exist: Metal/CUDA backends have CPU fallbacks for SDPA and several quantization formats, the HTTP server handles only serial single-request processing, and production features like continuous batching, PagedAttention integration, and observability are missing.

**Recommended Approach:** Adopt FlashAttention-2 tiling patterns (not FA-3, which requires H100+ and is beta-only), implement online softmax for numerical stability, integrate the existing PagedAttention implementation into the scheduler, add RadixAttention with frequency-cost eviction (not simple LRU), and build continuous batching with cache-aware scheduling. Complete GPU kernel parity by implementing missing CUDA quantized GEMV kernels (Q4_K/Q5_K/Q6_K/FP8) and debugging Metal GPU SDPA (currently falls back to CPU due to encoder switching overhead).

**Key Risks:** Three critical pitfalls threaten timelines: (1) Metal's encoder switching between compute and blit causes 150ms/layer stalls — all KV operations must use compute kernels, (2) CUDA Blackwell sm_121 has compiler bugs causing block-level reduction hangs — warp-only reductions required until CUDA 13.3+, (3) MoE router quantization overflow can silently produce garbage output — per-block scaling and overflow assertions are mandatory. Addressing these early prevents costly rewrites later.

---

## Key Findings by Research Area

### Stack (Technologies & Algorithms)

**Core GPU Kernel Algorithm:** FlashAttention-2 with online softmax
- **Why FA-2, not FA-3:** FlashAttention-3 requires Hopper H100+, CUDA 12.3+, and is beta-only with limited head dimension support. FA-2 is production-ready, cross-platform, and sufficient for target GPUs (M4 Pro, GB10 Blackwell).
- **Online softmax is mandatory:** Single-pass numerically stable softmax using running max+sum eliminates three-pass approach, reduces memory traffic by 1.33×. Standard in all modern attention kernels.
- **Tiling strategy:** Block sizes Q/K 128×128, shared dim 32. Auto-tuning recommended per input shape.

**Metal-Specific Critical Issues:**
- **GPU SDPA wrong output:** Likely race condition in compute-based KV append or incorrect threadgroup barrier placement. Must use `threadgroup_barrier(mem_flags::mem_threadgroup)` after shared memory writes.
- **Encoder switching overhead (CRITICAL):** Blit encoder for KV append → compute encoder for SDPA causes ~150ms/layer stalls. Solution: Fuse KV append into compute shader. **Never use blit commands in hot path.**
- **No native FP32 atomics:** Use int32 atomics + `as_type<float>` reinterpretation, or redesign to avoid atomics (use threadgroup reductions instead).

**CUDA-Specific Critical Issues:**
- **Blackwell sm_121 blockReduce hangs (CRITICAL):** Compiler codegen bug with dynamic shared memory. Use warp-level `__shfl_down_sync` reductions exclusively until CUDA toolkit fixes. Serial thread-0 softmax is current workaround.
- **Parallel softmax required for performance:** Two-tier reduction: warp shuffle for intra-warp max/sum, then shared memory (array size = num_warps, typically 8) for inter-warp reduction.

**CUDA Quantized GEMV Kernels (Missing — High Priority):**
- **Q4_K:** Most common quantization for 7B-13B models. Process 2-4 rows in parallel, share input vector via shared memory.
- **Q6_K:** Used for critical layers in mixed-quant models. Pre-compute scales, eliminate bounds checks.
- **Q5_K:** Between Q4/Q6 in accuracy. Unpack 5-bit → f32 using LUT or shift+mask.
- **FP8 E4M3/E5M2:** Use CUDA `__nv_fp8_e4m3` intrinsics on Ada/Hopper+, else 256-entry LUT. Tensor Core FP8 MMA on Hopper.

**Continuous Batching & Request Scheduling:**
- **Algorithm:** vLLM-style iteration-level scheduling. Process one token step across all active requests. Add/remove requests dynamically at each decode step.
- **Memory management:** PagedAttention block tables (16-token blocks). Reduces fragmentation from 60-80% to <4%.
- **Scheduler workflow:** Check queue → assemble batch → allocate KV blocks → dispatch GPU → update states → remove completed → repeat.
- **Key metrics:** Throughput (total tok/s), TTFT (<50ms target), GPU utilization (85-92% target), memory efficiency (>80% KV cache actively used).

**RadixAttention Prefix Caching:**
- **Structure:** Radix tree (prefix trie) mapping token sequences → KV cache blocks. Edges represent multi-token sequences.
- **Eviction policy (CRITICAL):** **NOT simple LRU.** Use frequency × compute_cost. Prioritize shared prefixes (ref_count > 1), evict last block of sequence first (reverse order).
- **Performance expectations:** 60-80% prefix hit rate for multi-turn conversations. 1.5-5× faster than vLLM for shared-prefix workloads.
- **Cache-aware scheduling:** Priority = -1 × (deadline + α × cached_prefix_length). Longer cached prefixes get higher priority.

**Production Serving Infrastructure:**
- **HTTP server:** Zig `std.http.Server` (native async I/O, zero external dependencies).
- **API format:** OpenAI-compatible `/v1/completions`, `/v1/chat/completions` (de facto standard).
- **Streaming:** Server-Sent Events (SSE) with `Content-Type: text/event-stream`.
- **Rate limiting:** Token bucket algorithm with atomics (lock-free).
- **Request timeouts:** Per-request deadline with cancellation via atomic flag.
- **Observability:** Prometheus `/metrics` endpoint (requests_total, request_duration_seconds, tokens_generated_total, kv_cache_hit_rate, gpu_memory_bytes_used).

**Source Quality:** HIGH. Research draws from production systems (vLLM, SGLang, TensorRT-LLM), 2025-2026 papers (FlashAttention-3, LiquidGEMM, FP8-Flow-MoE), and official documentation (Metal, CUDA, Vulkan).

---

### Features (Table Stakes, Differentiators, Anti-Features)

**Table Stakes (Missing = Unusable in Production):**

| Feature | Status | Complexity | Priority |
|---------|--------|------------|----------|
| **Continuous Batching** | Missing | High | P0 (Critical) |
| **PagedAttention** | Implemented but not integrated | Medium-High | P0 (Critical) |
| **OpenAI-Compatible API** | Partial (needs full schema compliance) | Low | P0 (Critical) |
| **SSE Streaming** | Present (verify format) | Low | P1 (High) |
| **Request Timeouts** | Missing | Low | P1 (High) |
| **Rate Limiting** | Missing | Medium | P1 (High) |
| **GPU GEMV for All Quant Formats** | CUDA gaps: Q4_K/Q5_K/Q6_K/FP8 | Medium | P0 (Critical) |
| **Metrics Endpoint** | Missing | Low | P1 (High) |
| **Authentication** | Missing | Low | P1 (High) |
| **Multi-Model Support** | Missing | Medium | P2 (Medium) |
| **Graceful Degradation** | Partial | Low-Medium | P1 (High) |
| **Concurrent Requests** | Missing (single-request serial) | Medium | P0 (Critical) |

**Differentiators (Competitive Advantage):**

| Feature | Value | Complexity | Priority |
|---------|-------|------------|----------|
| **RadixAttention** | 5× throughput on shared-prefix workloads | High | P1 (High — post-v1.0) |
| **Speculative Decoding** | 1.3-1.7× speedup | High | P3 (Defer to v2.0) |
| **Chunked Prefill** | Reduces head-of-line blocking | High | P3 (Defer to v1.3+) |
| **Structured Output** | Guaranteed valid JSON/regex | Medium-High | P2 (Defer to v1.2+) |
| **Multi-LoRA Serving** | Massive cost savings for multi-tenant | High | P3 (Evaluate demand) |
| **Native NVFP4/MXFP4 Support** | Early adopter advantage for Blackwell+ | Medium | P2 (Medium) |
| **Vision/Multimodal** | Expands use cases | Medium-High | P3 (Defer to v2.0+) |

**Anti-Features (Explicitly NOT Building):**
- Training/fine-tuning (different domain, massive scope creep)
- Custom templating DSL (keep chat templates as simple data)
- Built-in RAG/vector search (orthogonal concern, users use dedicated DBs)
- Model zoo/auto-download (maintenance nightmare, HF already does this)
- Web UI (scope creep, users choose their own UI)
- Windows support (use WSL2 instead)
- Prompt caching at app layer (RadixAttention handles this)
- Proprietary binary format (GGUF/SafeTensors are open standards)
- Built-in observability platform (expose /metrics, users use Prometheus/Grafana)
- Python bindings (focus on C API + HTTP API)
- Blockchain/crypto integration (zero technical value)

**MVP Recommendation (Production Readiness v1.0):**
1. Continuous Batching — Single biggest throughput multiplier
2. PagedAttention Integration — Already implemented, needs integration
3. OpenAI API Compatibility — Low effort, high compatibility value
4. Request Timeouts + Rate Limiting — Basic production hygiene
5. Metrics Endpoint — Observability is non-negotiable
6. Authentication — Required for any public deployment
7. GPU GEMV Completion — Can't ship with CPU fallbacks on common formats
8. Concurrent Request Handling — Foundation for continuous batching

**Defer to Post-v1.0:**
- RadixAttention (complex, add in v1.1+ after PagedAttention works)
- Multi-Model Support (nice-to-have, not critical for single-model deployments)
- Structured Output (trending, but requires grammar compiler)
- Speculative Decoding (high complexity, v2.0 feature)
- Vision/Multimodal (significant complexity, v2.0+)

**Source Quality:** HIGH. Feature landscape based on 2026 production standards from vLLM, SGLang, TensorRT-LLM, llama.cpp, with industry trend analysis.

---

### Architecture (Patterns & Component Boundaries)

**Recommended Architecture (5-Layer Separation):**

```
API Layer (HTTP/gRPC)
    ↓
Request Manager (Tokenization, Request Queue, AsyncStream)
    ↓
Scheduler (Continuous Batching, Cache-Aware Scheduling, Resource Allocation)
    ↓ ← KV Cache Manager (Block Allocation, RadixTree, LRU Eviction)
Executor (Model Forward, Batch Assembly, GPU Dispatch)
    ↓
Backend (GPU/CPU Kernels, Memory Management, Device Optimization)
```

**Component Boundaries:**

| Component | Responsibility | Owns | Communicates With |
|-----------|---------------|------|-------------------|
| **API Layer** | HTTP serving, protocol translation, SSE streaming | Connection state, HTTP buffers | Request Manager |
| **Request Manager** | Tokenization, request lifecycle, async streaming, timeout enforcement | Token buffers, request queues, AsyncStreams | API Layer, Scheduler |
| **Scheduler** | Continuous batching, cache-aware request ordering, resource allocation, preemption | Active batch state, scheduling policy, priority queues | Request Manager, KV Cache Manager, Executor |
| **KV Cache Manager** | Block allocation/deallocation, RadixTree maintenance, LRU eviction, prefix matching | KV block pool, RadixTree nodes, block tables | Scheduler, Executor |
| **Executor** | Model execution, batch assembly, GPU dispatch coordination | Model weights, activation buffers, batch metadata | Scheduler, KV Cache Manager, Backend |
| **Backend** | GPU kernels, memory allocation, device-specific optimization | Device memory, kernel state, BufCache | Executor |

**Key Patterns:**

1. **Iteration-Level Scheduling (Continuous Batching):** Process one token step at a time across all active requests. Insert new requests as soon as any completes. 2-3× throughput vs static batching.

2. **RadixAttention (Automatic Prefix Caching):** Maintain KV cache in radix tree. Automatically detect longest common prefix, reuse cached KV blocks, evict via frequency × cost (not simple LRU). 50-99% cache hit rates.

3. **PagedAttention (Block-Based KV Cache):** Partition KV cache into 16-token blocks. Map logical sequence positions to physical blocks via block tables. Reduces waste from 60-80% to <4%.

4. **Fused GPU Kernels:** Combine multiple operations into single GPU kernels (GEMV + norm, SDPA components, MLP + activation). 2-4× memory bandwidth reduction. 80-90% GPU utilization vs 30% unfused.

5. **Async Request Handling with Streaming:** Use async I/O for request ingress/egress. Stream tokens via SSE. Decouple API layer from inference loop via queues and AsyncStreams.

6. **Memory Pool Allocation:** Pre-allocate GPU memory into fixed-size pools (KV blocks, activation buffers). Use arena/slab allocators instead of per-request malloc.

**Anti-Patterns to Avoid:**
- Static batching (wait for full batch to complete)
- Contiguous KV allocation (wastes 60-80% memory)
- FIFO-only scheduling (misses cache hits)
- Per-op GPU sync (serializes pipeline)
- Unfused kernels on hot path (memory bandwidth bottleneck)

**Build Order & Dependencies:**

**Phase 1: Request Management Foundation (1-2 weeks)**
- AsyncStream (per-request output queue)
- RequestQueue (FIFO pending requests)
- RequestManager (tokenize, enqueue, detokenize)
- HTTP server integration (SSE streaming, timeout)

**Phase 2: Continuous Batching Scheduler (1-2 weeks)**
- Scheduler (step function, batch management)
- Batch (dynamic request list)
- Engine background loop (scheduler.step() → executor.execute())
- Request completion handling

**Phase 3: PagedAttention Integration (2-3 weeks)**
- KVCacheManager refactor (block allocation, block tables)
- BlockTable per request (logical → physical mapping)
- Executor integration (pass block tables to attention kernels)
- Preemption (evict low-priority requests on OOM)

**Phase 4: RadixAttention Prefix Caching (2-3 weeks)**
- RadixTree (prefix trie, node storage)
- Cache-aware scheduler (prioritize high hit-rate requests)
- KV block sharing (multiple requests reference same node)
- Frequency × cost eviction (not simple LRU)

**Phase 5: Fused Kernel Optimization (3-4 weeks, iterative)**
- Fused SDPA (debug Metal GPU kernel, implement CUDA parallel softmax)
- Fused RMSNorm (already done — fusedRmsNorm, addRmsNorm)
- Fused MLP (SwiGLU gate + activation in single kernel)
- Fused RoPE + Attention

**Critical path:** 1 → 2 → 3 → 4 (8-10 weeks sequential)
**Parallelization:** Phase 4 can start once Phase 3 block API is stable (saves 2 weeks). Phase 5 can start after Phase 2 (runs concurrently).
**Total calendar time (with parallelization):** 10-12 weeks

**Integration with Agave:**

**Leverage Existing:**
- Backend dispatcher → add fused SDPA, fused MLP kernels
- KV cache (PagedKvCache, RadixTree) → integrate with Scheduler allocation
- HTTP server → add SSE streaming, AsyncStream wiring
- ThreadPool → spawn scheduler loop thread
- Tokenizer → move to async context (no blocking)

**New Components Needed:**
- `src/scheduler.zig` — Continuous batching logic
- `src/request_manager.zig` — AsyncStream per request
- `src/kvcache/manager.zig` (refactor) — Unify Paged + Radix under single allocator

**Source Quality:** HIGH. Architecture patterns from vLLM, SGLang production codebases, official architecture documentation, and 2025-2026 research papers.

---

### Pitfalls (Critical, Moderate, Minor)

**Critical Pitfalls (Cause Rewrites or Broken Models):**

**Pitfall 1: Encoder Switching Overhead in Metal (Compute ↔ Blit)**
- **Impact:** 150ms/layer stalls. SDPA slower than CPU.
- **Cause:** Switching encoder types forces GPU synchronization.
- **Prevention:** **ALWAYS use compute kernels for memory operations in hot path.** Never switch to blit encoders during token generation.
- **Phase impact:** GPU Kernel Parity phase must establish this rule day 1.

**Pitfall 2: CUDA Blackwell (sm_121) Compiler Bugs in Shared Memory Reductions**
- **Impact:** Kernels hang indefinitely or produce silent data corruption.
- **Cause:** NVCC optimizer bug with `-O3` on sm_121.
- **Prevention:** Avoid shared memory reductions on sm_121 until CUDA 13.3+. Use warp-only reductions. Serial workarounds for now.
- **Phase impact:** GPU Kernel Parity phase MUST detect this early. Budget 2-3 weeks for workarounds.

**Pitfall 3: MoE Router Numerical Overflow with FP8/Low-Precision Quantization**
- **Impact:** Router scores explode to ±1e26. Model generates garbage output.
- **Cause:** Router weights quantized without per-block scaling. Outliers overflow FP8 E4M3 max value (448).
- **Prevention:** **Always use per-block quantization for router weights.** Implement overflow detection: `assert(score < 1e10 && score > -1e10)`.
- **Phase impact:** Model Verification phase must catch this. Golden tests mandatory for every MoE model.

**Pitfall 4: Pre-Dequantization vs In-Kernel Dequantization Performance Cliff**
- **Impact:** Q4_K GEMV slower than FP32 GEMV. VRAM usage 4× expected.
- **Cause:** Dequantizing entire weight matrix to FP32 before GEMV.
- **Prevention:** **Dequantization MUST happen inside the GEMV kernel.** Load quantized data, dequantize in registers, accumulate.
- **Phase impact:** GPU Kernel Parity phase must enforce this from the start.

**Pitfall 5: RadixAttention Block Eviction Order Causes Prefix Thrashing**
- **Impact:** Cache hit rate drops from 80% to 40%. Long system prompts repeatedly evicted and re-computed.
- **Cause:** Simple LRU treats all blocks equally. Shared prefixes evicted as often as private continuations.
- **Prevention:** **Evict based on minimum (frequency × compute_cost), not LRU.** Prioritize shared prefixes (ref_count > 1). Evict last block of sequence first.
- **Phase impact:** Production Serving phase must implement sophisticated eviction from day one.

**Moderate Pitfalls (Performance Regressions, Recoverable):**

**Pitfall 6: UMA Zero-Copy Memory Coherency Assumptions**
- **Impact:** Stale data reads. Intermittent corruption.
- **Prevention:** **Assume non-coherent unless proven otherwise.** After CPU writes: sync before GPU read. After GPU writes: sync before CPU read.

**Pitfall 7: GPU Kernel Numerical Tolerance Testing Without Dual Baselines**
- **Impact:** GPU kernel passes tests but model output diverges after 100 tokens.
- **Prevention:** **Dual-delta testing**: Compare both GPU and CPU to FP64 oracle. Verify GPU error ≤ 2× CPU error.

**Pitfall 8: Cross-Compilation with Musl Breaks dlopen of glibc Libraries**
- **Impact:** Segfaults when dlopen'ing libcuda.so from musl binary.
- **Prevention:** **Use `-gnu` target for cross-compilation** when you need to dlopen GPU drivers. Do NOT use musl for CUDA/ROCm backends.

**Pitfall 9: Thread Pool Futex Wake Latency Dominates Small GEMV Operations**
- **Impact:** Futex wake takes 5-10μs, GEMV work is 2μs/row → 50% overhead.
- **Prevention:** **Set minimum work size for parallel dispatch** (e.g., `parallel_min_rows = 32`).

**Pitfall 10: Metal newBufferWithBytesNoCopy Requires Page-Aligned Pointers**
- **Impact:** Metal silently creates copy instead of zero-copy mapping.
- **Prevention:** **Always wrap the enclosing page-aligned region.** Implement `getBufRef(ptr) -> {buf, offset}` that caches page-aligned buffers.

**Minor Pitfalls (Annoying but Easy to Fix):**
- Attention mask dimension mismatches with prompt tuning
- FP16 mixed precision overflow at 64k
- GGUF norm weights pre-baked with +1.0 (Gemma3)
- Continuous batching GPU memory fragmentation from `--gpu-memory-utilization=0.95`

**Pre-Deployment Checklist:**
- [ ] Metal SDPA: Zero blit encoder usage in hot path
- [ ] CUDA sm_121: Shared memory reductions tested with 1000-iteration stress test
- [ ] MoE models: Router scores logged, all values < 1e10
- [ ] Quantization: No FP32 scratch buffers allocated during GEMV
- [ ] RadixAttention: Eviction policy is frequency × cost, not LRU
- [ ] UMA platforms: Sync called after every CPU→GPU data transfer
- [ ] Numerical correctness: Dual-delta tests pass (GPU vs CPU vs FP64 oracle)
- [ ] Cross-compilation: Linux ARM64 build uses `-gnu` target (not musl)
- [ ] Thread pool: Parallel dispatch skips small inputs (rows < 32)
- [ ] Memory alignment: Metal buffers are page-aligned or use `getBufRef()`
- [ ] Continuous batching: `--gpu-memory-utilization ≤ 0.93`

**Source Quality:** HIGH. Pitfalls identified from production incident reports (llama.cpp, vLLM, CUTLASS GitHub issues), 2025-2026 research on numerical stability and GPU optimization, and official platform documentation (CUDA, Metal, Vulkan).

---

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: GPU Kernel Parity (Critical Foundation)**
**Duration:** 4-5 weeks
**Objective:** Eliminate all CPU fallbacks in Metal/CUDA backends. Achieve production-quality GPU kernels.

**Deliverables:**
- Metal GPU SDPA (debug encoder switching, fuse KV append into compute shader)
- CUDA quantized GEMV (Q4_K, Q5_K, Q6_K, FP8 E4M3/E5M2)
- CUDA parallel softmax (warp-only reductions for sm_121 compatibility)
- Dual-delta numerical tests for all kernels (GPU vs CPU vs FP64 oracle)
- Pre-dequantization pattern eliminated (in-kernel dequant enforcement)

**Why this order:** Can't build continuous batching on top of CPU fallbacks. GPU kernels must be correct and fast before scaling to multi-request serving.

**Which features:** None from FEATURES.md yet — this is pure backend work.

**Which pitfalls to avoid:**
- **CRITICAL:** Encoder switching (Metal SDPA)
- **CRITICAL:** Blackwell sm_121 reduction hangs
- **CRITICAL:** Pre-dequantization pattern
- UMA coherency assumptions
- Dual-baseline testing

**Research flags:** None needed — patterns are well-documented (FlashAttention-2, llama.cpp quantized kernels, vLLM SDPA).

---

**Phase 2: Request Management Foundation (Production Infrastructure)**
**Duration:** 2-3 weeks
**Objective:** Build async request handling, token streaming, basic queuing.

**Deliverables:**
- AsyncStream (per-request output queue with event-based wake/sleep)
- RequestQueue (FIFO pending requests with priority support)
- RequestManager (tokenize, enqueue, detokenize in background)
- HTTP server integration (SSE streaming, timeout enforcement)
- OpenAI API schema compliance (/v1/chat/completions, /v1/completions)
- Authentication (API key validation)

**Why this order:** Needs GPU kernels working (Phase 1 complete). Provides foundation for continuous batching (Phase 3).

**Which features:** SSE Streaming, Request Timeouts, OpenAI API Compatibility, Authentication (all table stakes).

**Which pitfalls to avoid:** None specific — this is mostly application logic, not GPU/numerical work.

**Research flags:** None needed — async patterns are standard (vLLM, SGLang examples exist).

---

**Phase 3: Continuous Batching Scheduler (Core Serving Logic)**
**Duration:** 2-3 weeks
**Objective:** Iteration-level scheduling, dynamic batch assembly, 2-3× throughput increase.

**Deliverables:**
- Scheduler (step function, batch management, priority queue)
- Batch (dynamic request list with add/remove per iteration)
- Engine background loop (scheduler.step() → executor.execute())
- Request completion handling (pop finished, pull new from queue)
- Metrics endpoint (Prometheus /metrics with throughput, latency, queue depth)

**Why this order:** Depends on RequestManager (Phase 2). Unlocks multi-request serving. Foundation for PagedAttention integration (Phase 4).

**Which features:** Continuous Batching, Concurrent Request Handling, Metrics Endpoint (all table stakes).

**Which pitfalls to avoid:**
- Static batching anti-pattern
- FIFO-only scheduling (add cache-awareness later in Phase 5)
- Continuous batching OOM (`--gpu-memory-utilization ≤ 0.93`)

**Research flags:** None needed — continuous batching patterns are standard (vLLM scheduler well-documented).

---

**Phase 4: PagedAttention Integration (Memory-Bounded Serving)**
**Duration:** 2-3 weeks
**Objective:** Block-based KV cache, memory-bounded serving, 2-4× more concurrent requests.

**Deliverables:**
- KVCacheManager refactor (block allocation, block tables per request)
- BlockTable (logical → physical mapping for indirect KV access)
- Executor integration (pass block tables to attention kernels)
- Preemption (evict low-priority requests on OOM)
- Rate limiting (token bucket algorithm, requests/min + tokens/min)

**Why this order:** Depends on Scheduler (Phase 3) to coordinate allocation. Agave already has PagedAttention implementation — this phase integrates it.

**Which features:** PagedAttention (table stakes), Rate Limiting (table stakes).

**Which pitfalls to avoid:**
- Contiguous KV allocation anti-pattern
- Per-op GPU sync (already avoided in Agave)

**Research flags:** None needed — PagedAttention is well-documented (vLLM, existing Agave implementation).

---

**Phase 5: RadixAttention Prefix Caching (Differentiation)**
**Duration:** 3-4 weeks
**Objective:** Automatic prefix detection, cache reuse, 1.5-5× throughput on shared-prefix workloads.

**Deliverables:**
- RadixTree (prefix trie, node storage, LCP matching)
- Cache-aware scheduler (prioritize high hit-rate requests: priority = -1 × (deadline + α × cached_prefix_length))
- KV block sharing (multiple requests reference same node via ref_count)
- Frequency × cost eviction (NOT simple LRU — shared prefixes prioritized, last block evicted first)
- Cache hit rate metrics (export to /metrics endpoint)

**Why this order:** Depends on PagedAttention (Phase 4) — radix tree uses paged blocks. Can start once Phase 4 block API is stable (parallelization opportunity).

**Which features:** RadixAttention (key differentiator).

**Which pitfalls to avoid:**
- **CRITICAL:** Prefix thrashing from simple LRU (must implement frequency × cost eviction from day 1)
- FIFO-only scheduling (integrate cache-aware priority queue)

**Research flags:** None needed — SGLang RadixAttention is well-documented with reference implementation.

---

**Phase 6: Fused Kernel Optimization (Performance Tuning)**
**Duration:** 3-4 weeks (iterative)
**Objective:** Minimize memory traffic, maximize GPU utilization (80%+ target).

**Deliverables:**
- Fused Metal SDPA (debug current GPU kernel or implement FlashAttention-2 tiling)
- Fused CUDA SDPA (implement FA-2 tiling with online softmax, warp-only reductions for sm_121)
- Fused MLP (SwiGLU gate + activation in single kernel)
- Fused RoPE + Attention (rotate + SDPA)
- Benchmark suite (throughput, memory bandwidth, GPU utilization per model/backend)

**Why this order:** Can start after Phase 2 (runs in parallel with Phases 3-5). Measures end-to-end impact incrementally.

**Which features:** None from table stakes — this is pure optimization.

**Which pitfalls to avoid:**
- **CRITICAL:** Metal encoder switching (compute-only in hot path)
- **CRITICAL:** CUDA sm_121 shared memory hangs (warp-only reductions)
- Unfused kernels on hot path

**Research flags:** FlashAttention-2 implementation details (community Metal ports, CUDA references exist).

---

### Phase Dependency Graph

```
Phase 1 (GPU Kernel Parity) ── MUST COMPLETE FIRST
  ↓
Phase 2 (Request Management)
  ↓
Phase 3 (Continuous Batching)
  ↓
Phase 4 (PagedAttention) ──┐
  ↓                        │ (can parallelize once block API stable)
Phase 5 (RadixAttention) ←─┘

Phase 6 (Fused Kernels) — can start after Phase 2, runs in parallel
```

**Critical path:** 1 → 2 → 3 → 4 → 5 (14-18 weeks sequential)
**Parallelization:** Phase 5 can overlap with Phase 4 (saves 2-3 weeks). Phase 6 runs concurrently (saves 3-4 weeks).
**Total calendar time (optimized):** 14-16 weeks (3.5-4 months)

---

### Research Flags (Which Phases Need Deeper Research)

**No additional research needed for Phases 1-5.** All patterns are well-documented with production references:
- FlashAttention-2: Papers, CUDA/Metal community implementations
- Continuous batching: vLLM scheduler architecture (detailed docs + code)
- PagedAttention: vLLM design docs, existing Agave implementation
- RadixAttention: SGLang paper, reference implementation, LMSYS blog

**Phase 6 (Fused Kernels) may need targeted research:**
- Metal FlashAttention-2 community ports (HuggingFace kernels-community has metal-flash-sdpa)
- CUDA warp-only softmax patterns (avoid block-level shared memory on sm_121)
- Online softmax numerical stability verification (standard algorithm, but test with adversarial inputs)

**Post-v1.0 features needing research:**
- Speculative decoding integration with RadixAttention (cache invalidation on rejected tokens)
- Structured output / constrained decoding (XGrammar library integration)
- Multi-LoRA serving (S-LoRA batching patterns, adapter memory management)
- Vision/multimodal (image encoder integration, cross-attention or early fusion)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack (GPU Kernels)** | **HIGH** | FlashAttention-2 is production-standard. CUDA quantized GEMV patterns well-documented. Metal encoder switching pitfall is known (Agave already hit this). CUDA sm_121 hang is documented in llama.cpp/vLLM issues. |
| **Stack (Serving Infrastructure)** | **HIGH** | Continuous batching, PagedAttention, RadixAttention are battle-tested in vLLM/SGLang with detailed architecture docs. OpenAI API is a stable standard. |
| **Features** | **HIGH** | Table stakes list matches 2026 production standards (vLLM, SGLang, TensorRT-LLM, llama.cpp all have these). Differentiator priorities based on SGLang benchmarks (RadixAttention) and industry trends (structured output, speculative decoding). |
| **Architecture** | **HIGH** | Patterns from vLLM/SGLang production codebases. Clear component boundaries prevent "spaghetti" imports. Build order validated against dependency graph. |
| **Pitfalls** | **MEDIUM-HIGH** | Critical pitfalls (encoder switching, sm_121 hangs, MoE overflow, pre-dequant) are all documented in production incident reports (GitHub issues, blog posts). Eviction policy pitfall is from vLLM RFC. Minor gaps: ROCm/Vulkan-specific issues less documented (assumed similar to CUDA/Metal). |

**Gaps Identified:**
1. **ROCm-specific kernel pitfalls:** Research focused on CUDA/Metal. ROCm (HIP runtime, AMDGCN compiler) likely has similar issues but different triggers. **Recommendation:** Investigate ROCm compiler bugs and HIP runtime quirks before Phase 1 if ROCm backend is priority.
2. **Vulkan driver differences:** Mesa vs proprietary NVIDIA vs AMDGPU-PRO. Agave Vulkan backend is near-complete, but cross-vendor testing may reveal driver-specific bugs. **Recommendation:** Test on all three drivers during Phase 1.
3. **KV cache quantization error accumulation:** Does Q8_0 KV cache cause divergence after 1000+ tokens? Research mentions KV quantization but not long-sequence stability. **Recommendation:** Golden tests with 2048+ token sequences during Phase 4.
4. **Speculative decoding + RadixAttention interaction:** Cache invalidation on rejected tokens. No detailed documentation found. **Recommendation:** Research this before implementing speculative decoding (post-v1.0).

**Overall Confidence for v1.0 Roadmap:** **HIGH**. All technologies, features, and patterns for Phases 1-5 are production-proven with detailed references. Timeline is realistic (14-16 weeks with parallelization). Pitfalls are known and preventable.

---

## Sources

### Technology Stack
- [FlashAttention & Paged Attention: GPU Sorcery](https://medium.com/@afafel/flashattention-paged-attention-gpu-sorcery-for-blazing-fast-transformers-9307df8a3f3f)
- [FlashAttention-3: Fast and Accurate Attention](https://arxiv.org/html/2407.08608v2)
- [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
- [Online Softmax Normalizer Optimization](https://www.emergentmind.com/papers/1805.02867)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
- [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel](https://arxiv.org/html/2509.01229v1)

### Feature Landscape
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [vLLM vs SGLang vs LMDeploy: Fastest LLM Inference Engine in 2026](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [OpenAI-Compatible Server - vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [LLM Structured Output in 2026](https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk)

### Architecture Patterns
- [Hugging Face: Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching)
- [Anyscale: Achieve 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [vLLM: Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [Aleksa Gordić: Inside vLLM](https://www.aleksagordic.com/blog/vllm)
- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)
- [FlashInfer Paper](https://arxiv.org/pdf/2501.01005)
- [How Fused Kernels Are Powering the LLM Revolution](https://medium.com/the-synaptic-stack/how-fused-kernels-are-powering-the-llm-revolution-and-why-you-should-care-1e232fa1ae70)

### Pitfalls
- [llama.cpp Issue #18331: CUDA MUL_MAT crash on Blackwell](https://github.com/ggml-org/llama.cpp/issues/18331)
- [CUTLASS Issue #3096: SM120 NVFP4 MoE Garbage Output](https://github.com/NVIDIA/cutlass/issues/3096)
- [vLLM Issue #36821: No sm_121 support on aarch64](https://github.com/vllm-project/vllm/issues/36821)
- [LMSYS: Unified FP8 for Stable MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM RFC: Frequency and Cost Aware Eviction Policy](https://github.com/vllm-project/vllm/issues/23641)
- [arXiv: Dual-Delta Testing for Mixed-Precision Computing](https://arxiv.org/html/2602.10605)
- [NVIDIA: Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

---

## Ready for Requirements

This research synthesis provides the foundation for detailed roadmap planning. The suggested phase structure (6 phases, 14-16 weeks total with parallelization) balances technical dependencies, risk mitigation, and incremental value delivery.

**Next Steps:**
1. Review this summary with project stakeholders
2. Prioritize phases based on business goals (production readiness vs differentiation)
3. Validate timeline assumptions (4-5 weeks for GPU Kernel Parity assumes 1-2 engineers)
4. Proceed to requirements definition for Phase 1 (GPU Kernel Parity)

**Critical Pre-Phase 1 Decisions:**
- Confirm CUDA Blackwell sm_121 as target (vs falling back to sm_120 compilation)
- Decide Metal SDPA approach: debug existing kernel vs rewrite with FA-2 tiling
- Validate CUDA quantized GEMV priority order (Q4_K → Q6_K → Q5_K → FP8)
- Agree on acceptable numerical tolerance for dual-delta tests (GPU error ≤ 2× CPU error)
