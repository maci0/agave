# Phase 2: Production Serving - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Multi-tenant HTTP server handles concurrent requests with continuous batching, PagedAttention, rate limiting, authentication, and full observability. Transforms Agave from single-request serial engine to production-viable serving system.

</domain>

<decisions>
## Implementation Decisions

### API Design
- **D-01:** Support both `/v1/completions` (legacy) and `/v1/chat/completions` — broader OpenAI compatibility
- **D-02:** Full OpenAI response object format (id, choices, usage, model, created)
- **D-03:** Accept and ignore unsupported params (tools, logprobs) with warning header — graceful degradation, not strict rejection
- **D-04:** SSE streaming with correct OpenAI format (`data: {"choices":[...]}`, `data: [DONE]`)

### Authentication & Rate Limiting
- **D-05:** Bearer token auth via `--api-key` CLI flag, `Authorization: Bearer <key>` header, 401 on invalid
- **D-06:** Per API key rate limiting (requests/min + tokens/min) using token bucket algorithm
- **D-07:** HTTP 429 with `Retry-After` header on rate limit exceeded
- **D-08:** Single API key via `--api-key` flag — sufficient for v1

### Batching & Scheduling
- **D-09:** Iteration-level continuous batching (vLLM-style) — process one decode step across all active requests
- **D-10:** Configurable `--max-batch-size` (default 8)
- **D-11:** No preemption in v1 — FIFO queue, reject with 503 when full
- **D-12:** Replace flat KV cache with PagedAttention blocks in model forward loop (not opt-in flag)

### Claude's Discretion
- Scheduler loop architecture (background thread vs async event loop)
- Request queue data structure (lock-free vs mutex-protected)
- Prometheus metrics format details (histogram buckets, label names)
- Health check response body format
- Graceful shutdown drain timeout value

</decisions>

<canonical_refs>
## Canonical References

- `src/server.zig` — Current HTTP server implementation (single-request serial)
- `src/kvcache/manager.zig` — PagedKvCache and RadixTree implementations
- `src/models/model.zig` — Model vtable interface (forward, resetCache, cancel)
- `src/thread_pool.zig` — Existing futex-based thread pool
- `.planning/research/ARCHITECTURE.md` — Continuous batching component boundaries
- `.planning/research/FEATURES.md` — Table stakes feature list
- `.planning/research/SUMMARY.md` — vLLM-style scheduler patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/server.zig`: HTTP server with SSE streaming, conversation management, web UI — extend, don't rewrite
- `src/kvcache/manager.zig`: PagedKvCache with block allocation, free list, block tables — ready for integration
- `src/thread_pool.zig`: Futex-based work-stealing pool — use for scheduler background loop
- `src/models/model.zig`: Model vtable with `forward()`, `resetCache()`, `cancel()` — scheduler calls these

### Established Patterns
- Per-connection threads in server.zig — need to add request queue between connection handler and inference
- `be.sync()` before CPU reads GPU data — scheduler must handle sync points
- `model.cancel()` for async cancellation — use for request timeouts

### Integration Points
- Server → RequestManager → Scheduler → Model.forward() pipeline
- PagedKvCache replaces flat allocKvCache in each model's init()
- Metrics exported via new `/metrics` endpoint in server.zig

</code_context>

<specifics>
## Specific Ideas

- CUDA test machine: maci@192.168.0.212 (DGX Spark)
- ROCm test machine: maci@192.168.0.205 (24GB VRAM)
- Existing server default port: 49453

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-production-serving*
*Context gathered: 2026-03-22*
