---
phase: 02-production-serving
plan: 03
subsystem: production-serving
completed: "2026-03-21T18:05:34Z"
tags:
  - openai-api
  - prometheus
  - observability
  - health-checks
  - graceful-shutdown
tech-stack:
  added:
    - Prometheus text exposition format
    - OpenAI Chat Completions API schema
    - SSE streaming with data: [DONE] terminator
    - POSIX signal handlers (SIGTERM/SIGINT)
  patterns:
    - Lock-free atomic metrics collection
    - Graceful shutdown with connection draining
    - Kubernetes liveness/readiness probes
key-files:
  created:
    - src/metrics.zig
  modified:
    - src/server.zig
decisions:
  - decision: Use lock-free atomics for metrics instead of mutex-protected counters
    rationale: Metrics recording happens in hot path (every request), must not block concurrent handlers
  - decision: Histogram buckets at 10/50/100/500ms, 1/5/10/30s
    rationale: Covers typical LLM latency distribution (TTFT 10-100ms, long completions 1-30s)
  - decision: 30-second drain timeout on graceful shutdown
    rationale: Balances clean shutdown vs delayed redeploy; matches Kubernetes terminationGracePeriodSeconds default
  - decision: Separate /health (liveness) and /ready (readiness) endpoints
    rationale: Kubernetes best practice — liveness never fails unless process stuck, readiness rejects traffic during shutdown
metrics:
  duration_seconds: 18
  tasks_completed: 3
  files_created: 1
  files_modified: 1
  commits: 3
dependency-graph:
  requires:
    - 02-01 (RequestManager for queue depth metrics)
  provides:
    - OpenAI-compatible API endpoints
    - Prometheus metrics for monitoring
    - Graceful shutdown for zero-downtime deploys
  affects:
    - src/main.zig (no changes, but server now fully OpenAI-compatible)
---

# Phase 02 Plan 03: OpenAI API + Metrics + Health Checks Summary

**One-liner:** Production-ready server with OpenAI-compatible /v1/chat/completions + /v1/completions, Prometheus /metrics endpoint, Kubernetes health probes, and SIGTERM/SIGINT graceful shutdown with 30s drain timeout.

## What Was Built

### 1. Prometheus Metrics Collector (src/metrics.zig)

**Lock-free atomic metrics tracking:**
- **Counters:** requests_total, requests_completed, requests_cancelled, tokens_generated_total
- **Gauges:** queue_depth, active_requests, kv_blocks_used, kv_blocks_total
- **Histogram:** Request duration buckets (10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s, 30s, +Inf)

**Key implementation details:**
- All operations use `std.atomic.Value` with `.monotonic` ordering — zero locks, safe for concurrent access
- `recordLatency()` updates both histogram bucket and running sum for average calculation
- `renderPrometheus()` outputs standard Prometheus text format with HELP/TYPE headers
- 210 lines including comprehensive unit tests (TDD RED-GREEN pattern)

**Test coverage:**
- Counter increment correctness
- Histogram bucket assignment (e.g., 250ms → latency_500ms)
- Prometheus text format compliance

### 2. OpenAI-Compatible API Endpoints (src/server.zig)

**Added endpoints:**
- `/v1/chat/completions` — Full OpenAI Chat Completions API with SSE streaming
- `/v1/completions` — Legacy OpenAI Completions API
- `/metrics` — Prometheus metrics exposition
- `/health` — Kubernetes liveness probe (always 200 if process alive)
- `/ready` — Kubernetes readiness probe (503 during shutdown)

**OpenAI schema compliance:**
- Chat completions: `{id: "chatcmpl-N", object: "chat.completion", created, model, choices: [{message: {role, content}, finish_reason}], usage: {prompt_tokens, completion_tokens, total_tokens}}`
- Completions: `{id: "cmpl-N", object: "text_completion", created, model, choices: [{text, index, finish_reason}], usage}`
- SSE streaming: `data: {id, object: "chat.completion.chunk", created, model, choices: [{delta: {content}, finish_reason: null}]}` chunks, terminated with `data: [DONE]`

**Metrics integration:**
- `recordRequest()` at handler start
- `recordLatency()` + `recordTokens()` + `recordCompletion()` at handler end
- Per-request timing via `std.time.milliTimestamp()`

### 3. Graceful Shutdown (src/server.zig)

**SIGTERM/SIGINT handlers:**
- `shutdown_requested` atomic flag set on signal
- Accept loop exits on shutdown_requested
- Drains active connections for up to 30 seconds
- Logs drain status and timeout warnings

**Connection tracking:**
- `active_connections` atomic counter
- Incremented on `handleConnection` entry, decremented on exit via `defer`
- Drain loop polls every 100ms until count reaches zero

**Readiness probe integration:**
- `/ready` returns 503 during shutdown, preventing new traffic from load balancers

## Deviations from Plan

**None** — plan executed exactly as written.

All tasks completed:
1. Created `src/metrics.zig` with Prometheus metrics collector (TDD RED-GREEN pattern)
2. Extended `src/server.zig` with OpenAI-compatible endpoints + metrics recording
3. Added /metrics, /health, /ready endpoints + SIGTERM/SIGINT graceful shutdown

## Key Decisions Made

1. **Lock-free atomics for metrics:** Hot path (every request) cannot afford mutex contention. Atomic operations with `.monotonic` ordering provide sufficient guarantees without blocking.

2. **Histogram bucket boundaries:** Chosen to cover typical LLM inference latency distribution:
   - 10-100ms: Time-to-first-token (TTFT) range for small batches
   - 500ms-5s: Short completions (10-50 tokens)
   - 10s-30s: Long completions (100+ tokens)
   - +Inf: Timeouts and outliers

3. **30-second drain timeout:** Balances clean shutdown (all requests complete normally) vs deployment velocity (don't wait forever for slow requests). Matches Kubernetes `terminationGracePeriodSeconds` default.

4. **Separate /health and /ready:** Kubernetes best practice:
   - `/health` (liveness): Only fails if process is stuck/deadlocked (never happens in practice)
   - `/ready` (readiness): Fails during shutdown to prevent new traffic, allows drain to complete

## Testing & Verification

**Automated tests passed:**
- `zig test src/metrics.zig` — 3 tests, all green
- `zig build` — clean compilation, no warnings/errors

**Human verification completed:**
- OpenAI /v1/chat/completions returns correct schema
- OpenAI /v1/completions returns correct schema
- SSE streaming outputs `data: {...}` chunks + `data: [DONE]` terminator
- /metrics outputs valid Prometheus text format
- /health returns 200 OK
- /ready returns 200 when running, 503 during shutdown
- Graceful shutdown drains active connections, logs status
- Metrics increment correctly (requests_total, tokens_generated_total, duration histogram)

## Known Issues / Limitations

None. All endpoints work as specified.

## Files Changed

**Created:**
- `src/metrics.zig` (210 lines) — Prometheus metrics collector with lock-free atomics

**Modified:**
- `src/server.zig` — Added:
  - `metrics: Metrics` field to Server struct
  - OpenAI-compatible response formatting (`sendChatCompletionResponse`, `sendCompletionResponse`)
  - SSE streaming helpers (`sseWriteChunk`, `sseWriteDone`)
  - `/v1/completions` handler
  - `/metrics`, `/health`, `/ready` handlers
  - SIGTERM/SIGINT signal handlers
  - Graceful shutdown drain loop
  - Metrics recording calls (recordRequest, recordLatency, recordTokens, recordCompletion)

## Requirements Completed

- **SERV-05:** OpenAI-compatible /v1/chat/completions endpoint with full schema compliance
- **SERV-06:** SSE streaming with correct OpenAI format (data: {...}, data: [DONE])
- **SERV-10:** Prometheus /metrics endpoint exports throughput, latency, queue depth, KV cache usage
- **SERV-11:** Kubernetes health probes (/health liveness, /ready readiness)
- **SERV-12:** Graceful shutdown drains in-flight requests on SIGTERM/SIGINT (30s timeout)

## Next Steps

**Immediate:**
- None — Phase 2 is now complete (3/3 plans)

**Future (Phase 3 — Memory Optimization):**
- RadixAttention prefix caching for shared-prefix workloads
- Tiered KV cache (VRAM → RAM) for memory-bounded serving
- Cache-aware scheduler priority scoring

## Integration Notes

**Server now provides:**
- Drop-in replacement for OpenAI API (clients can use official OpenAI SDKs unmodified)
- Full observability via Prometheus metrics (integrate with Grafana dashboards)
- Kubernetes-native health checks (liveness + readiness probes in pod spec)
- Zero-downtime deployments (graceful shutdown prevents aborted requests)

**Metrics to monitor:**
- `agave_requests_total` — Request throughput (rate)
- `agave_request_duration_seconds_bucket` — Latency percentiles (p50/p95/p99)
- `agave_queue_depth` — Scheduler backpressure
- `agave_kv_blocks_used / agave_kv_blocks_total` — KV cache pressure (add capacity before 80%)
- `agave_tokens_generated_total` — Token throughput (rate)

**Example Prometheus queries:**
```promql
# Request rate (requests/sec)
rate(agave_requests_total[1m])

# P95 latency
histogram_quantile(0.95, rate(agave_request_duration_seconds_bucket[5m]))

# KV cache utilization
agave_kv_blocks_used / agave_kv_blocks_total

# Average tokens per request
rate(agave_tokens_generated_total[1m]) / rate(agave_requests_completed[1m])
```

## Self-Check: PASSED

**Files exist:**
- [x] `/Users/mwysocki/Experiments/agave/src/metrics.zig` — FOUND
- [x] `/Users/mwysocki/Experiments/agave/src/server.zig` — FOUND (modified)

**Commits exist:**
- [x] `9ffdcc0` — FOUND (test(02-03): add failing tests for Prometheus metrics collector)
- [x] `e737b28` — FOUND (feat(02-03): add metrics recording to OpenAI endpoints)
- [x] `dad8e2a` — FOUND (feat(02-03): add graceful shutdown with SIGTERM/SIGINT drain)

**Grep verification:**
- [x] `grep "chatcmpl-" src/server.zig` — FOUND (OpenAI chat completion ID format)
- [x] `grep "cmpl-" src/server.zig` — FOUND (OpenAI completion ID format)
- [x] `grep "data: \[DONE\]" src/server.zig` — FOUND (SSE terminator)
- [x] `grep "/metrics" src/server.zig` — FOUND (Prometheus endpoint)
- [x] `grep "/health" src/server.zig` — FOUND (liveness probe)
- [x] `grep "/ready" src/server.zig` — FOUND (readiness probe)
- [x] `grep "shutdown_requested" src/server.zig` — FOUND (graceful shutdown flag)

**Build status:**
- [x] `zig build` — SUCCESS (no errors or warnings)

All verification checks passed.
