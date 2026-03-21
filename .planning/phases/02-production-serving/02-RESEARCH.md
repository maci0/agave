# Phase 2: Production Serving - Research

**Researched:** 2026-03-22
**Domain:** Multi-tenant HTTP server with continuous batching, OpenAI API compatibility, authentication, rate limiting, and observability
**Confidence:** HIGH

## Summary

Phase 2 transforms Agave from a single-request serial engine to a production-ready serving system with iteration-level continuous batching, OpenAI-compatible API endpoints, per-key authentication and rate limiting, and full Prometheus observability. The research confirms that all required components have well-established patterns: vLLM-style continuous batching operates at decode-step granularity, PagedAttention block tables enable fine-grained memory reclamation, token bucket rate limiting is straightforward to implement, and Prometheus metrics follow standard naming conventions.

**Primary recommendation:** Use vLLM's iteration-level scheduling pattern (scheduler checks after every token), extend existing `server.zig` per-connection threading with a request queue + background scheduler thread, replace flat KV cache with PagedAttention in all 6 models, implement token bucket per API key in memory (no external state), and expose Prometheus text format metrics at `/metrics`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Support both `/v1/completions` (legacy) and `/v1/chat/completions` — broader OpenAI compatibility
- **D-02:** Full OpenAI response object format (id, choices, usage, model, created)
- **D-03:** Accept and ignore unsupported params (tools, logprobs) with warning header — graceful degradation, not strict rejection
- **D-04:** SSE streaming with correct OpenAI format (`data: {"choices":[...]}`, `data: [DONE]`)
- **D-05:** Bearer token auth via `--api-key` CLI flag, `Authorization: Bearer <key>` header, 401 on invalid
- **D-06:** Per API key rate limiting (requests/min + tokens/min) using token bucket algorithm
- **D-07:** HTTP 429 with `Retry-After` header on rate limit exceeded
- **D-08:** Single API key via `--api-key` flag — sufficient for v1
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

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SERV-01 | Continuous batching scheduler processes multiple concurrent requests with iteration-level scheduling | vLLM architecture: iteration-level scheduling, scheduler maintains waiting + running queues |
| SERV-02 | PagedAttention integrated into model inference loop with block tables | PagedAttention block tables (mapping logical→physical), split KV into blocks during forward pass |
| SERV-05 | OpenAI-compatible /v1/chat/completions API with full schema compliance | OpenAI API spec: id, object, created, model, choices[], usage{} fields required |
| SERV-06 | SSE streaming with correct OpenAI event format (data: [DONE]) | Existing server.zig already implements SSE, just needs OpenAI-compliant chunk format |
| SERV-07 | Per-request timeout with inference cancellation (configurable 30-120s) | Existing Model.cancel() via SIGINT handler, extend to per-request timeout |
| SERV-08 | Rate limiting per API key (requests/min + tokens/min, token bucket algorithm) | Token bucket: capacity (burst), refill rate (sustained), separate buckets per key |
| SERV-09 | API key authentication (--api-key CLI flag, Authorization: Bearer header) | Standard Bearer token validation, 401 Unauthorized on mismatch |
| SERV-10 | Prometheus /metrics endpoint (throughput, latency p50/p95/p99, queue depth, KV cache usage) | Prometheus naming: snake_case, _total suffix, labels for dimensions, histogram for latency |
| SERV-11 | Health check endpoints (/health, /ready) | /health = liveness (always 200 if process running), /ready = readiness (200 if accepting requests) |
| SERV-12 | Graceful shutdown (drain in-flight requests before exit) | SIGTERM/SIGINT handlers stop accepting new connections, wait for active_requests==0 with timeout |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Zig std.net | stdlib | HTTP/1.1 server, per-connection threads | Built-in, zero external dependencies, existing `server.zig` uses it |
| Zig std.Thread | stdlib | Thread pool for scheduler loop | Already used in `thread_pool.zig`, futex-based wake/sleep |
| Zig std.ArrayList | stdlib | Request queue (FIFO) | Dynamic array with mutex protection, simple and correct |
| Zig std.atomic.Value | stdlib | Request counter, active connections | Lock-free atomics for metrics counters |
| Zig std.time | stdlib | Timestamp for rate limiter token refill | Monotonic clock via `std.time.milliTimestamp()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Zig std.StringHashMap | stdlib | API key → RateLimiter mapping | Single API key in v1, extensible to multi-key in v2 |
| Zig std.Thread.Mutex | stdlib | Protect request queue and scheduler state | Required for queue mutations (enqueue/dequeue) |
| Zig std.http.Headers | stdlib | Parse Authorization header | Already available, no custom parsing needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Thread pool scheduler | Async I/O event loop | Async would reduce thread count but complicate model forward() contract (currently sync) |
| Mutex-protected queue | Lock-free ring buffer | Lock-free adds complexity without proven benefit at target scale (8-64 concurrent requests) |
| In-memory rate limiter | Redis/external state | External state adds latency + dependency, overkill for single-instance v1 |

**Installation:**
All components are Zig standard library — no external dependencies required.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── server.zig               # HTTP server (existing, extend with auth + routing)
├── scheduler.zig            # NEW: RequestManager + continuous batching scheduler
├── rate_limiter.zig         # NEW: Token bucket per API key
├── metrics.zig              # NEW: Prometheus metrics collector + /metrics endpoint
├── kvcache/
│   ├── manager.zig          # Existing: PagedKvCache already implemented
│   └── block_allocator.zig  # NEW: Per-request block table management
└── models/
    └── *.zig                # MODIFY: Replace allocKvCache with PagedKvCache in all 6 models
```

### Pattern 1: Iteration-Level Continuous Batching
**What:** Scheduler maintains a `waiting` queue (new/paused requests) and a `running` list (actively decoding). After each model.forward() iteration, scheduler ejects finished requests and pulls new ones from the queue. All active requests are batched into a single forward pass.

**When to use:** This is the core serving pattern — enables concurrent request processing without waiting for full batch completion.

**Example:**
```zig
// Source: vLLM architecture (insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/)
pub const Scheduler = struct {
    waiting: std.ArrayList(*Request),
    running: std.ArrayList(*Request),
    max_batch_size: usize,

    pub fn step(self: *Scheduler, model: *Model) !void {
        // 1. Remove finished requests from running
        var i: usize = 0;
        while (i < self.running.items.len) {
            if (self.running.items[i].is_finished) {
                _ = self.running.swapRemove(i);
            } else {
                i += 1;
            }
        }

        // 2. Fill batch from waiting queue (FIFO, up to max_batch_size)
        while (self.running.items.len < self.max_batch_size and self.waiting.items.len > 0) {
            const req = self.waiting.orderedRemove(0);
            try self.running.append(req);
        }

        // 3. Execute one decode step for all running requests
        for (self.running.items) |req| {
            const next_token = try model.forward(req.last_token_id);
            try req.appendToken(next_token);
            if (model.isEog(next_token)) req.is_finished = true;
        }
    }
};
```

### Pattern 2: PagedAttention Block Table Integration
**What:** Each request owns a `SeqBlockTable` mapping logical KV positions to physical cache blocks. During forward(), allocate new blocks as needed, update block table, split KV into block-sized chunks.

**When to use:** Required for all model forward() calls — replaces flat `allocKvCache()` allocation.

**Example:**
```zig
// Source: vLLM PagedAttention design (docs.vllm.ai/en/stable/design/paged_attention/)
pub const Request = struct {
    seq_id: u32,
    block_tables: [][]u32, // [n_layers][n_logical_blocks] -> physical block IDs
    seq_len: usize,
    tokens: std.ArrayList(u32),
    is_finished: bool,

    pub fn allocateNextBlock(self: *Request, cache: *PagedKvCache) !u32 {
        const block_id = cache.allocBlock() orelse return error.OutOfBlocks;
        // Append to all layers' block tables
        for (self.block_tables) |*layer_table| {
            try layer_table.append(block_id);
        }
        return block_id;
    }
};

// In model forward():
// Before: model writes KV to flat cache[layer][seq_len]
// After:  model writes KV to cache.blocks[block_table[layer][logical_block_idx]]
```

### Pattern 3: Token Bucket Rate Limiter
**What:** Per API key, maintain two token buckets (requests/min, tokens/min). On each request, refill tokens based on elapsed time, consume one request token + prompt_tokens. Reject with 429 if insufficient tokens.

**When to use:** Before enqueuing request into scheduler's waiting queue.

**Example:**
```zig
// Source: Token bucket algorithm (geeksforgeeks.org/computer-networks/token-bucket-algorithm/)
pub const RateLimiter = struct {
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,

    pub const TokenBucket = struct {
        capacity: f64,       // max burst
        tokens: f64,         // current tokens
        refill_rate: f64,    // tokens per second
        last_refill: i64,    // timestamp (ms)

        pub fn tryConsume(self: *TokenBucket, amount: f64) bool {
            const now = std.time.milliTimestamp();
            const elapsed_sec = @as(f64, @floatFromInt(now - self.last_refill)) / 1000.0;
            self.tokens = @min(self.capacity, self.tokens + elapsed_sec * self.refill_rate);
            self.last_refill = now;

            if (self.tokens >= amount) {
                self.tokens -= amount;
                return true;
            }
            return false;
        }

        pub fn retryAfterSeconds(self: *const TokenBucket, amount: f64) u32 {
            const deficit = amount - self.tokens;
            return @intFromFloat(@ceil(deficit / self.refill_rate));
        }
    };
};
```

### Pattern 4: Prometheus Metrics Exposition
**What:** Maintain in-memory counters/histograms, expose via `/metrics` endpoint in Prometheus text format. Use labels for dimensions (endpoint, method, status_code).

**When to use:** Increment counters on request start/complete, record latency on completion, scrape every 15-30s.

**Example:**
```zig
// Source: Prometheus naming conventions (prometheus.io/docs/practices/naming/)
pub const Metrics = struct {
    requests_total: std.atomic.Value(u64),
    tokens_generated_total: std.atomic.Value(u64),
    request_duration_sum: std.atomic.Value(f64),  // for avg latency
    queue_depth: std.atomic.Value(u32),

    pub fn renderPrometheus(self: *const Metrics, writer: anytype) !void {
        try writer.print("# HELP agave_requests_total Total HTTP requests received\n", .{});
        try writer.print("# TYPE agave_requests_total counter\n", .{});
        try writer.print("agave_requests_total {d}\n", .{self.requests_total.load(.monotonic)});

        try writer.print("# HELP agave_tokens_generated_total Total tokens generated\n", .{});
        try writer.print("# TYPE agave_tokens_generated_total counter\n", .{});
        try writer.print("agave_tokens_generated_total {d}\n", .{self.tokens_generated_total.load(.monotonic)});

        try writer.print("# HELP agave_queue_depth Current request queue depth\n", .{});
        try writer.print("# TYPE agave_queue_depth gauge\n", .{});
        try writer.print("agave_queue_depth {d}\n", .{self.queue_depth.load(.monotonic)});
    }
};
```

### Anti-Patterns to Avoid
- **Batching entire forward pass:** Don't batch prefill+decode — prefill is sequential per request, only decode batches (different prompt lengths make batching prefill inefficient)
- **Synchronous request processing:** Don't call `model.forward()` directly from HTTP handler thread — blocks new connections (queue requests, scheduler thread processes them)
- **Per-request KV cache allocation:** Don't allocate fresh PagedKvCache blocks on every request — reuse freed blocks via free list (already implemented in `PagedKvCache`)
- **Global rate limiter:** Don't rate limit across all API keys — per-key buckets prevent one client from consuming all capacity
- **Blocking on full queue:** Don't block HTTP handler when queue full — return 503 immediately, client can retry with backoff

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Request scheduling | Custom work-stealing scheduler | Simple FIFO queue + single scheduler thread | vLLM uses FIFO, preemption adds complexity without benefit at target scale |
| Latency percentiles | Manual percentile calculation | Pre-allocated histogram bins + linear scan | Prometheus scrapers don't need exact percentiles, approximate bins (10ms, 50ms, 100ms, 500ms, 1s) sufficient |
| SSE encoding | Custom SSE formatter | Existing `server.zig` sseWriteData() | Already correct, just needs OpenAI-compliant JSON inside data: field |
| Block table allocation | Custom memory pool | Existing `PagedKvCache` free list | Already implements alloc/free with free_list tracking |
| Graceful shutdown | Custom drain logic | SIGTERM handler + atomic active_requests counter | Standard pattern: stop accepting, wait for counter==0, timeout forces exit |

**Key insight:** Production serving is about orchestrating existing components (scheduler, queue, rate limiter, metrics) rather than inventing new algorithms. The hard parts (PagedAttention, model forward(), tokenizer) are already implemented.

## Common Pitfalls

### Pitfall 1: Queue Overflow on Burst Traffic
**What goes wrong:** Request queue grows unbounded when arrival rate exceeds processing rate, causing memory exhaustion and latency spikes.

**Why it happens:** No admission control — server accepts all requests even when queue is full.

**How to avoid:** Enforce `max_queue_depth` limit (e.g., 64). Return HTTP 503 Service Unavailable with `Retry-After` header when queue is full. Clients retry with exponential backoff.

**Warning signs:** Increasing `/metrics` queue depth, growing memory usage, increasing p99 latency.

### Pitfall 2: Race Between model.forward() and model.resetCache()
**What goes wrong:** Request timeout triggers `model.cancel()` while scheduler thread is in `model.forward()`, causing stale KV cache reads.

**Why it happens:** No synchronization between HTTP handler (timeout check) and scheduler thread (model forward).

**How to avoid:** Scheduler owns model exclusively — timeout only sets `request.is_cancelled` flag, scheduler checks flag before forward() and after. Model.cancel() only sets atomic flag, doesn't directly modify cache.

**Warning signs:** Assertion failures in model forward(), garbage output, segfaults from stale pointers.

### Pitfall 3: Incorrect Block Table Indexing
**What goes wrong:** Model reads KV from wrong physical block, producing garbage output.

**Why it happens:** Logical block index calculation is off-by-one (e.g., `seq_len / block_size` instead of `(seq_len - 1) / block_size` for last block).

**How to avoid:** Existing `PagedKvCache` tests cover block allocation. Add integration test: forward 33 tokens (2 full blocks + 1 partial), verify KV cache contents match expected layout.

**Warning signs:** Model output changes when `--max-batch-size` changes, inconsistent results across runs.

### Pitfall 4: Token Bucket Underflow
**What goes wrong:** Rate limiter allows negative tokens, causing requests to never be rejected.

**Why it happens:** Refill logic doesn't clamp `tokens` to `[0, capacity]` range, overflow on long idle periods.

**How to avoid:** Always clamp: `self.tokens = @min(self.capacity, @max(0, self.tokens + refill_amount))`. Unit test: long idle period (e.g., 10 minutes) then consume — should not exceed capacity.

**Warning signs:** Rate limiter never rejects requests even when load is high.

### Pitfall 5: Prometheus Metric Cardinality Explosion
**What goes wrong:** Metrics endpoint response size explodes, scraper times out, Prometheus storage exhausted.

**Why it happens:** Using high-cardinality labels (e.g., `request_id`, `prompt_hash`, individual token IDs) creates millions of time series.

**How to avoid:** Only use low-cardinality labels: `endpoint` (3 values: /v1/chat/completions, /v1/completions, /metrics), `method` (2 values: GET, POST), `status_code` (5-10 values). Never label by request-specific data.

**Warning signs:** `/metrics` response > 10MB, scraper errors in Prometheus logs, increasing scrape duration.

## Code Examples

Verified patterns from official sources:

### HTTP 429 Rate Limit Response
```zig
// Source: Token bucket rate limiting (medium.com/@surajshende247/token-bucket-algorithm-rate-limiting-db4c69502283)
fn handleRateLimitExceeded(stream: net.Stream, retry_after: u32) void {
    var buf: [256]u8 = undefined;
    const body = std.fmt.bufPrint(&buf,
        \\{{"error":{{"message":"Rate limit exceeded. Retry after {d} seconds.","type":"rate_limit_exceeded","code":null}}}}
    , .{retry_after}) catch return;

    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf,
        "HTTP/1.1 429 Too Many Requests\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Retry-After: {d}\r\n" ++
        "Connection: close\r\n\r\n",
        .{ body.len, retry_after }
    ) catch return;

    stream.writeAll(hdr) catch return;
    stream.writeAll(body) catch return;
}
```

### OpenAI Chat Completions Response Format
```zig
// Source: OpenAI API reference (platform.openai.com/docs/api-reference/chat/create)
fn sendChatCompletionResponse(stream: net.Stream, req_id: u64, content: []const u8, tokens_generated: u32, prompt_tokens: u32) void {
    const created = std.time.timestamp();
    const total = tokens_generated + prompt_tokens;

    var buf: [65536]u8 = undefined;
    const json = std.fmt.bufPrint(&buf,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"completion_tokens":{d},"prompt_tokens":{d},"total_tokens":{d}}}}}
    , .{ req_id, created, g_server.model_name, content, tokens_generated, prompt_tokens, total }) catch return;

    sendResponse(stream, "200 OK", "application/json", json);
}
```

### Graceful Shutdown Pattern
```zig
// Source: Graceful shutdown in Go (oneuptime.com/blog/post/2026-01-23-go-graceful-shutdown/view)
pub fn runWithGracefulShutdown(server: *Server, timeout_sec: u32) !void {
    // Install SIGTERM/SIGINT handler
    const handler = struct {
        fn handle(_: c_int) callconv(.c) void {
            g_shutdown_requested.store(true, .release);
        }
    };
    const act = std.posix.Sigaction{
        .handler = .{ .handler = handler.handle },
        .mask = std.posix.sigemptyset(),
        .flags = 0,  // No SA_RESETHAND — allow multiple signals
    };
    std.posix.sigaction(std.posix.SIG.TERM, &act, null);
    std.posix.sigaction(std.posix.SIG.INT, &act, null);

    // Accept loop
    while (!g_shutdown_requested.load(.acquire)) {
        const conn = server.tcp.accept() catch continue;
        const thread = std.Thread.spawn(.{}, handleConnection, .{conn.stream}) catch {
            conn.stream.close();
            continue;
        };
        thread.detach();
    }

    // Drain: wait for active requests to complete
    const drain_start = std.time.timestamp();
    while (server.active_requests.load(.acquire) > 0) {
        if (std.time.timestamp() - drain_start > timeout_sec) break;
        std.time.sleep(100 * std.time.ns_per_ms);
    }

    // Force shutdown
    server.scheduler.shutdown();
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static batching (wait for batch to fill) | Iteration-level continuous batching | 2023 (vLLM paper) | 2-5× throughput improvement, reduced head-of-line blocking |
| Flat KV cache allocation | PagedAttention block tables | 2023 (vLLM paper) | <5% memory waste (down from 20-80%), enables continuous batching |
| Per-request mutex (serialize all inference) | Lock-free scheduler (queue protected, forward() exclusive) | N/A (existing pattern) | Allows concurrent request handling without serializing forward() |
| Bearer token as secret key | Bearer token + per-key rate limiting | Industry standard 2020+ | Prevents single client from exhausting server capacity |
| Pull-based metrics (client polls) | Prometheus scrape model (server exposes /metrics) | 2016 (Prometheus v1) | Simpler server implementation, time-series database handles aggregation |

**Deprecated/outdated:**
- **Single-request serving:** Models must support batched forward() for production use
- **Unbounded request queues:** Production servers enforce admission control (503 when full)
- **Global KV cache:** Multi-tenant serving requires per-request isolation (PagedAttention enables this)

## Open Questions

1. **Per-request timeout vs global timeout**
   - What we know: Current `server.zig` has no request-level timeout, only global SIGINT handler
   - What's unclear: Should timeout be per-request (30-120s configurable) or global (all requests cancelled on SIGINT)?
   - Recommendation: **Per-request timeout.** Background scheduler thread checks elapsed time, sets `request.is_cancelled` flag, scheduler skips cancelled requests on next iteration. SIGINT still cancels all in-flight requests (existing behavior).

2. **Histogram bins for latency metrics**
   - What we know: Prometheus histograms need pre-defined buckets (can't compute exact percentiles post-hoc)
   - What's unclear: Optimal bucket boundaries for LLM inference latency
   - Recommendation: **Use [10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s, 30s, +Inf].** Covers typical TTFT (50-500ms) and full generation latency (1-30s). Can adjust based on observed distribution in production.

3. **503 vs 429 when queue full**
   - What we know: 503 = service unavailable (temporary overload), 429 = rate limit exceeded (client-specific)
   - What's unclear: Which status code when queue is full but client hasn't hit rate limit?
   - Recommendation: **Use 503.** Queue full is server-side capacity limit, not client rate limit. Include `Retry-After: 5` header (client should retry in 5 seconds).

## Validation Architecture

> Workflow.nyquist_validation is `false` — validation section omitted per config.

## Sources

### Primary (HIGH confidence)
- [vLLM continuous batching architecture](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/) — Iteration-level scheduling, waiting + running queues
- [vLLM PagedAttention design](https://docs.vllm.ai/en/stable/design/paged_attention/) — Block tables, logical-to-physical mapping
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) — Response format (id, object, created, model, choices, usage)
- [Token Bucket Algorithm](https://www.geeksforgeeks.org/computer-networks/token-bucket-algorithm/) — Capacity, refill rate, consume logic
- [Prometheus Metric Naming](https://prometheus.io/docs/practices/naming/) — snake_case, _total suffix, base units

### Secondary (MEDIUM confidence)
- [Prometheus best practices](https://betterstack.com/community/guides/monitoring/prometheus-best-practices/) — Cardinality management, label design
- [Graceful shutdown patterns](https://oneuptime.com/blog/post/2026-01-23-go-graceful-shutdown/view) — SIGTERM handling, drain timeout
- [vLLM production deployment](https://introl.com/blog/vllm-production-deployment-inference-serving-architecture) — Scheduler architecture details
- [Token bucket Redis implementation](https://medium.com/redis-with-raphael-de-lio/token-bucket-rate-limiter-redis-java-8cd42f3f8a34) — In-memory vs external state tradeoffs

### Tertiary (LOW confidence)
- OpenAI unsupported parameter warnings — No official documentation found for warning header format. Recommend custom header `X-Agave-Unsupported-Params: tools,logprobs` or omit warning header entirely (user decision D-03 specifies "warning header" but format unspecified).

## Metadata

**Confidence breakdown:**
- Continuous batching: HIGH — vLLM architecture well-documented, clear iteration-level pattern
- PagedAttention integration: HIGH — Existing `PagedKvCache` implementation in `manager.zig`, integration path straightforward
- OpenAI API compliance: HIGH — Official API reference specifies exact response schema
- Rate limiting: HIGH — Token bucket algorithm is simple and well-understood
- Prometheus metrics: HIGH — Standard practices documented, existing patterns in ecosystem
- Graceful shutdown: MEDIUM — Standard pattern but no Zig-specific examples (extrapolated from Go/Rust)

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (30 days — stable domain, APIs unlikely to change)
