# Roadmap: Agave Production-Ready LLM Inference Engine

**Created:** 2026-03-21
**Granularity:** Coarse (3-5 phases, aggressive compression)
**Coverage:** 50/50 v1 requirements mapped

## Phases

- [ ] **Phase 1: Correctness Foundation** - All models produce correct output on all backends at full GPU speed
- [ ] **Phase 2: Production Serving** - Multi-tenant continuous batching with observability and safety
- [ ] **Phase 3: Memory Optimization** - RadixAttention prefix caching and tiered KV cache
- [ ] **Phase 4: Multi-GPU Parallelism** - Tensor, Pipeline, and Expert Parallelism for large models

## Phase Details

### Phase 1: Correctness Foundation
**Goal**: Every supported model produces correct output on every backend (CPU, Metal, CUDA, Vulkan) at full GPU speed with no unnecessary CPU fallbacks.

**Depends on**: Nothing (first phase)

**Requirements**: KERN-01 through KERN-10 (10 requirements), MODL-01 through MODL-09 (9 requirements)

**Success Criteria** (what must be TRUE):
1. Metal SDPA kernel produces identical output to CPU SDPA within numerical tolerance (dual-delta: GPU error ≤ 2x CPU error vs FP64 oracle)
2. CUDA backend supports Q4_K, Q5_K, Q6_K, and FP8 E4M3/E5M2 GEMV kernels with in-kernel dequantization (no pre-dequant CPU fallback)
3. All 6 models (Gemma3, Qwen3.5, Nemotron-Nano, GLM-4, GPT-OSS, Nemotron-H) pass golden tests on CPU, Metal, CUDA, and Vulkan backends
4. Nemotron Nano 30B produces coherent output with stable MoE router scores (all values < 1e10, no numerical overflow)
5. Automated CI runs golden tests comparing all models against reference implementations (llama.cpp or HuggingFace) with deterministic seed

**Plans**: TBD

---

### Phase 2: Production Serving
**Goal**: Multi-tenant HTTP server handles concurrent requests with continuous batching, PagedAttention, rate limiting, authentication, and full observability.

**Depends on**: Phase 1 (requires working GPU kernels and verified models)

**Requirements**: SERV-01, SERV-02, SERV-05 through SERV-12 (10 requirements)

**Success Criteria** (what must be TRUE):
1. Server processes 8+ concurrent requests with iteration-level continuous batching (2-3x throughput vs single-request serial)
2. PagedAttention integrated into model inference loop with block tables (16-token blocks, <4% memory fragmentation)
3. OpenAI-compatible /v1/chat/completions API accepts requests from OpenAI client libraries without modification (full schema compliance, SSE streaming with "data: [DONE]")
4. Authenticated requests enforce per-key rate limits (requests/min + tokens/min using token bucket algorithm)
5. Prometheus /metrics endpoint exports throughput, latency p50/p95/p99, queue depth, KV cache usage, and GPU memory
6. Server gracefully drains in-flight requests on SIGTERM/SIGINT (no aborted generations, max 30s drain timeout)

**Plans**: TBD

---

### Phase 3: Memory Optimization
**Goal**: Automatic prefix caching and tiered KV storage enable 1.5-5x throughput on shared-prefix workloads and memory-bounded serving at scale.

**Depends on**: Phase 2 (requires continuous batching scheduler and PagedAttention block allocator)

**Requirements**: SERV-03, SERV-04 (2 requirements), TIER-01 through TIER-07 (7 requirements)

**Success Criteria** (what must be TRUE):
1. RadixAttention radix tree automatically detects longest common prefix across requests and shares KV blocks (60-80% cache hit rate on multi-turn conversations)
2. Scheduler prioritizes requests by cache-aware score: priority = -1 × (deadline + α × cached_prefix_length)
3. Eviction policy uses frequency × cost metric (NOT simple LRU) — shared prefixes (ref_count > 1) prioritized, last block of sequence evicted first
4. KV pages automatically demote from VRAM to RAM when VRAM budget exceeded (configurable --kv-ram-budget), and promote back with LRU eviction
5. Zero-copy KV access works on all backends (Metal newBufferWithBytesNoCopy for RAM tier, UMA platforms avoid upload for RAM-resident KV)

**Plans**: TBD

---

### Phase 4: Multi-GPU Parallelism
**Goal**: Tensor Parallelism, Pipeline Parallelism, and Expert Parallelism distribute large models across multiple GPUs with zero-allocation communication in the hot path.

**Depends on**: Phase 1 (requires working backends and models)

**Requirements**: PARA-01 through PARA-12 (12 requirements)

**Success Criteria** (what must be TRUE):
1. DeviceGroup abstraction supports mixed backends (e.g., 4x Metal, 2x CUDA) and distributes weight shards at init
2. Tensor Parallelism splits Q/K/V projections column-parallel and output projection row-parallel across TP devices with all-reduce after attention
3. Pipeline Parallelism assigns contiguous layer ranges to PP stages with point-to-point activation transfer (stage N sends hidden state to stage N+1)
4. Expert Parallelism distributes MoE experts across EP devices with all-to-all token exchange (tokens routed to expert owner, results all-to-all gathered)
5. CLI flags --tp=N --pp=M auto-detect available devices and validate topology (e.g., TP × PP ≤ num_devices, hybrid TP+PP support)
6. Communication buffers pre-allocated at init (all-reduce scratch, activation transfer buffers) — zero allocations in decode hot path

**Plans**: TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Correctness Foundation | 0/? | Not started | - |
| 2. Production Serving | 0/? | Not started | - |
| 3. Memory Optimization | 0/? | Not started | - |
| 4. Multi-GPU Parallelism | 0/? | Not started | - |

---

## Coverage Validation

**Total v1 requirements:** 50

**Mapped:**
- Phase 1: 19 requirements (KERN-01 to KERN-10, MODL-01 to MODL-09)
- Phase 2: 10 requirements (SERV-01, SERV-02, SERV-05 to SERV-12)
- Phase 3: 9 requirements (SERV-03, SERV-04, TIER-01 to TIER-07)
- Phase 4: 12 requirements (PARA-01 to PARA-12)

**Total mapped:** 50

**Unmapped:** 0

**Coverage:** 100% ✓

---

## Notes

**Granularity calibration:** Config specifies "coarse" granularity. This roadmap compresses 5 natural requirement categories into 4 phases by combining GPU Kernel Parity + Model Verification into Phase 1 (both are correctness foundation), and keeping Production Serving infrastructure together despite large scope (10 requirements). Phase count at lower bound of coarse range (3-5 phases).

**Critical path:** Phase 1 MUST complete before Phase 2 (can't build serving on top of CPU fallbacks or broken models). Phase 2 MUST complete before Phase 3 (RadixAttention depends on PagedAttention block allocator and continuous batching scheduler). Phase 4 can begin after Phase 1 (parallelism only needs working backends and models, not serving infrastructure).

**Research flags:** None needed. All technologies are production-proven (FlashAttention-2, vLLM continuous batching, SGLang RadixAttention, NCCL-style collectives). Pitfalls documented in research/SUMMARY.md (Metal encoder switching, CUDA sm_121 reduction hangs, MoE overflow, eviction policy).

**Parallelization opportunity:** Phase 4 (Parallelism) can run concurrently with Phase 2-3 (Serving) after Phase 1 completes. Estimated timeline with parallelization: 14-18 weeks total (4-5 weeks Phase 1, then 10-13 weeks overlapping Phases 2+3+4).

---

*Roadmap created: 2026-03-21*
*Last updated: 2026-03-21*
