# Phase 5: Integration Wiring (Gap Closure) - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire Phase 2-3 infrastructure (RequestManager scheduler, TieredKvCache, RadixTree block reuse) into the serving stack so they are active at runtime instead of dormant code.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/scheduler.zig` — RequestManager with continuous batching, Request struct, cache-aware priority scheduling
- `src/kvcache/tiered.zig` — TieredKvCache with 3-tier (VRAM/RAM/SSD) block management
- `src/kvcache/manager.zig` — RadixTree for prefix caching, PagedKvCache, flat KvCache
- `src/kvcache/prefetch.zig` — Prefetcher for async SSD→RAM promotion
- `src/server.zig` — HTTP server with per-connection threads, mutex-serialized inference

### Established Patterns
- Models use PagedKvCache directly (all 6 models import from kvcache/manager.zig)
- Server calls model.forward() directly under mutex — no scheduler involvement
- Backend dispatch via tagged union with `inline else`

### Integration Points
- `src/server.zig` — needs to route through RequestManager.enqueue() instead of direct model.forward()
- `src/models/*.zig` — need to accept TieredKvCache instead of PagedKvCache
- `src/scheduler.zig` — needs RadixTree.insert() call on request completion
- `src/main.zig` — CLI flags for --kv-tiers, --kv-ram-budget, --kv-ssd-path

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
