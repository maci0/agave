# Server observability

What the `--serve` runtime exposes, and how to pivot between the three signals.

Source of truth: `src/server/metrics.zig` (`renderPrometheus`), `src/server/server.zig`
(access log, health endpoints), `src/server/scheduler.zig` (queue, cache, prefill load),
`src/kvcache/tiered.zig` (tier demotion counters).

## Correlation

Every accepted connection gets `req=<n>`, a monotonic server-assigned ID, before
any log line for that connection is written. The same ID appears in:

- the access log (`[HH:MM:SS] req=<n> POST /v1/chat/completions -> 200 (1234ms)`),
  where a stream that failed or was cut short logs `500`, `504`, or `499`
  instead of the `200` its headers already sent,
- every `std.log` line from that handler thread,
- the `X-Request-Id` response header, so a client report carries the ID,
- scheduler logs, which use the HTTP ID passed to `RequestManager.enqueue`.

A client-supplied correlation header is echoed as `xid=<value>` on the access log
line only; the server ID stays authoritative.

## Endpoints

| Endpoint | Use |
|---|---|
| `GET /health` | Liveness plus model/backend identity. `503` only while shutting down. |
| `GET /ready` | Readiness. `503` when shutting down, under KV pressure, or when the error rate is high; the JSON `reason` field names which. |
| `GET /metrics` | Prometheus text format. Requires the API key when one is configured. |

Both health endpoints collapse details when the request is unauthenticated, so an
orchestrator probe sees `{"status":...}` only.

There is no `agave_ready` gauge: `/metrics` does not mirror the `/ready` decision,
so an alert has to probe `GET /ready` directly rather than re-implementing the
health formula as a metric query.

## Metrics

Rate and error signals (PromQL uses `rate()` or `increase()` over these):

| Metric | Meaning |
|---|---|
| `agave_requests_total` | Requests accepted, including rejected ones. Excludes `/health`, `/ready`, `/metrics`, `/favicon.ico`, and CORS preflights, so a Prometheus scrape does not sit in the request rate. |
| `agave_requests_completed_total` | Finished generation. |
| `agave_requests_cancelled_total` | Client disconnect or server timeout mid-generation. |
| `agave_requests_failed_total` | Server faults (5xx, inference failures). This is the numerator `/ready` uses. |
| `agave_requests_client_error_total` | Client-caused 4xx, tracked apart so error rate means server faults only. |
| `agave_requests_timeout_total` | Cancellations caused by the server-side deadline. |
| `agave_requests_auth_failed_total` | Invalid API key. |
| `agave_requests_rate_limited_total` | Rejected by the rate limiter. |
| `agave_connections_rejected_total` | Rejected at connection capacity. |
| `agave_scheduler_errors_total` | Scheduler step failures. |
| `agave_kv_promote_failures_total` | Block promotions to the VRAM tier that failed. Blocks stay on a slower tier, so a rising rate is the cause behind latency the demotion counters would otherwise read as cache pressure. One log line per scheduler step names the count and the first error. |
| `agave_conv_store_save_failures_total` | Conversation store writes that failed. Requests still succeed, so this counter is the only signal that user history is being dropped. |

Latency histograms (use `histogram_quantile` over `_bucket`):

- `agave_request_duration_seconds`: end-to-end request latency.
- `agave_ttft_seconds`: time to first token.
- `agave_request_prompt_tokens`, `agave_request_generation_tokens`: size distributions.
- `agave_time_per_output_token_seconds`, `agave_inter_token_latency_seconds`,
  `agave_request_queue_time_seconds`: decode, inter-token, and queue-wait time.

Saturation and cache:

| Metric | Meaning |
|---|---|
| `agave_queue_depth`, `agave_active_requests`, `agave_active_connections` | Load in flight. |
| `agave_kv_blocks_used` / `agave_kv_blocks_total` | KV occupancy; `agave_kv_cache_usage_perc` and `agave_gpu_cache_usage_perc` are the derived ratios. |
| `agave_kv_cache_tier_blocks{tier,state}` | Per-tier occupancy, `state` in `used`/`total`, `tier` in `vram`/`ram`/`ssd`. A full VRAM tier under a half-empty total is blocks spilling down the hierarchy. |
| `agave_kv_cache_demotions_vram_to_ram_total`, `agave_kv_cache_demotions_ram_to_ssd_total` | Blocks demoted out of VRAM and out of RAM under cache pressure. A sustained rate is the KV-pressure signal; a rising `ram_to_ssd` rate is the expensive one. |
| `agave_kv_cache_hits_total`, `agave_kv_cache_misses_total`, `agave_prefix_tokens_reused_total`, `agave_prefix_tokens_total`, `agave_prefix_cache_hit_rate` | Prefix-cache behavior. `agave_prefix_tokens_total` is the denominator behind `agave_prefix_cache_hit_rate`. |
| `agave_input_tokens_in_flight` | Prompt tokens still to prefill across running requests. Drops before `agave_active_requests` as prefill completes, so it is the load signal for routing decisions. |
| `agave_tokens_per_second`, `agave_avg_prompt_throughput_toks_per_s`, `agave_avg_generation_throughput_toks_per_s` | Throughput; the first is the last request, the other two are since-start averages. |
| `agave_tokens_generated_total`, `agave_prefill_tokens_total` | Token counters. |

Process state: `agave_up`, `agave_sleeping` (idle sleep mode, `--sleep-after`),
`agave_process_start_time_seconds`, `agave_build_info{version,backend,language}`,
`agave_cache_config_info{block_size,num_gpu_blocks}`.

No metric carries a path, model, or client label, so cardinality stays flat
regardless of client mix.

## Debugging a request

1. Symptom in a metric, for example `rate(agave_requests_failed_total[5m])` climbing.
2. `agave_requests_failed_total` counts server faults only; compare with
   `agave_requests_client_error_total` to see whether the traffic is at fault.
3. Grep the access log for `-> 5` to get the request IDs and their durations.
   Streaming requests that ended badly show `-> 499` (client disconnected) or
   `-> 504` (server-side deadline) rather than `-> 200`, because the SSE headers
   were already sent when the body failed.
4. Grep those IDs for the `std.log` line naming the failed dependency (tokenizer,
   prefill forward, scheduler enqueue, grammar setup).
5. `agave_scheduler_errors_total` and `agave_kv_cache_demotions_*_total` separate
   inference faults from cache pressure.

A panic aborts the process and prints `agave: panic req=<n>: <message>` before
the stack trace, so the crash names the request that was in flight. `req=0`
means the fault hit a thread with no request: startup, shutdown, or a background
worker.

Every path that increments `agave_requests_cancelled_total` emits a
`client disconnected during streaming` or `prefill cancelled` line with the same
`req=<n>`, so a cancellation spike can always be attributed to specific requests.
