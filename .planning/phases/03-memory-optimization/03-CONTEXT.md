# Phase 3: Memory Optimization - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

RadixAttention prefix caching integrated into server with automatic prefix detection, frequency x cost eviction, and cache-aware scheduling. Tiered KV cache (VRAM → RAM → SSD) with automatic promotion/demotion, prefetching, and zero-copy per backend. Enables 1.5-5x throughput on shared-prefix workloads and memory-bounded serving at scale.

</domain>

<decisions>
## Implementation Decisions

### RadixAttention
- **D-01:** Scheduler owns RadixTree — queries RadixTree before block allocation to detect prefix reuse
- **D-02:** Cache-aware scheduling formula: `priority = -1 × (deadline + α × cached_prefix_length)` (SGLang-style)
- **D-03:** Eviction triggered when VRAM block pool drops below 10% free
- **D-04:** Export `kv_cache_hit_rate` and `prefix_reuse_ratio` on /metrics endpoint
- **D-05:** Eviction uses frequency × cost metric (NOT simple LRU) — shared prefixes (ref_count > 1) prioritized, last block of sequence evicted first

### Tiered KV Cache
- **D-06:** All 3 tiers in v1: VRAM + RAM + SSD
- **D-07:** Prefetch next 2 blocks from lower tiers during attention compute (overlap I/O with compute)
- **D-08:** RAM budget auto-detected: 50% of free system RAM (configurable via --kv-ram-budget)
- **D-09:** UMA platforms (Apple Silicon, GB10) use existing zero-copy patterns — no copy between VRAM and RAM tiers
- **D-10:** SSD uses async I/O for spill/restore (--kv-ssd-path, --kv-ssd-budget flags)
- **D-11:** Zero-copy access per backend: Metal newBufferWithBytesNoCopy for RAM, GPUDirect Storage for CUDA, VK_EXT_external_memory_host for Vulkan

### Claude's Discretion
- Block size for tiered pages (likely same 16-token blocks as PagedAttention)
- SSD page file format and naming
- Prefetch implementation (async thread vs dispatch queue)
- α coefficient for cache-aware priority formula

</decisions>

<canonical_refs>
## Canonical References

- `src/kvcache/manager.zig` — RadixTree + PagedKvCache implementations (ready for integration)
- `src/scheduler.zig` — Continuous batching scheduler (Phase 2, owns request lifecycle)
- `src/server.zig` — HTTP server with /metrics endpoint (Phase 2)
- `src/metrics.zig` — Prometheus metrics collector (Phase 2)
- `docs/IDEAS.md` — Tiered KV Cache section with architecture, zero-copy paths, CLI flags
- `.planning/research/STACK.md` — RadixAttention patterns, eviction policy
- `.planning/research/PITFALLS.md` — Prefix thrashing from simple LRU

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/kvcache/manager.zig`: RadixTree with insert, lookup, LCP matching already implemented
- `src/kvcache/manager.zig`: PagedKvCache with block allocation, free list, block tables
- `src/scheduler.zig`: Request lifecycle management, batch assembly
- `src/metrics.zig`: Prometheus metrics (add new gauges/counters for cache stats)
- `src/backend/metal.zig`: `newBufferWithBytesNoCopy` for zero-copy RAM access on UMA

### Established Patterns
- Block-based KV allocation (16-token blocks) — extend with tier tag
- Scheduler step() function — add cache-aware priority before batch assembly
- Backend sync pattern — prefetch must respect GPU sync points

### Integration Points
- RadixTree → Scheduler: prefix lookup before block allocation
- TieredCache → PagedKvCache: extend block struct with tier tag
- Metrics → /metrics: add cache hit rate, tier usage gauges
- CLI → main.zig: --kv-tiers, --kv-ram-budget, --kv-ssd-path, --kv-ssd-budget flags

</code_context>

<specifics>
## Specific Ideas

- CUDA test machine: maci@192.168.0.212 (DGX Spark)
- ROCm test machine: maci@192.168.0.205 (24GB VRAM)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-memory-optimization*
*Context gathered: 2026-03-22*
