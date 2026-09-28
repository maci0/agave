//! HTTP server with OpenAI-compatible and Anthropic-compatible API endpoints.
//! Provides /v1/chat/completions, /v1/completions, /v1/models, /v1/responses,
//! /v1/messages (Anthropic Messages API), /v1/embeddings, /v1/conversations,
//! /v1/chat (built-in web UI), /v1/chat/regenerate, /v1/tokenize, /v1/detokenize,
//! /health, /ready, and /metrics.
//! Supports both synchronous JSON responses and SSE streaming.
//! Uses `Io.net` with per-connection threads; inference is mutex-serialized.

const std = @import("std");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;

const Model = @import("../models/model.zig").Model;
const spec_decode = @import("../spec/spec_decode.zig");
const Tokenizer = @import("../tokenizer/tokenizer.zig").Tokenizer;
const chat_tmpl_mod = @import("../chat_template.zig");
const ChatTemplate = chat_tmpl_mod.ChatTemplate;
const Message = chat_tmpl_mod.Message;
const arch_mod = @import("../arch.zig");
const max_eog_ids = arch_mod.max_eog_ids;
const ImageTokens = arch_mod.ImageTokens;
const math_ops = @import("../ops/math.zig");
const scheduler = @import("scheduler.zig");
const ngram_mod = @import("../spec/ngram.zig");
const RateLimiter = @import("rate_limiter.zig").RateLimiter;
const metrics_mod = @import("metrics.zig");
const Metrics = metrics_mod.Metrics;
const json = @import("json.zig");
const http = @import("http.zig");
const conv_store = @import("conv_store.zig");
const Idempotency = @import("idempotency.zig");
const tools_mod = @import("tools.zig");
const SamplingParams = json.SamplingParams;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const VisionEncoder = @import("../models/model.zig").VisionEncoder;
const image_mod = @import("../image.zig");
const engine_version = @import("build_options").version;
const grammar_mod = @import("../grammar.zig");

const Mutex = Io.Mutex;

// ── Server constants ────────────────────────────────────────────
const slog_buf_size: usize = 4096;
const models_json_buf_size: usize = 1024;
const response_buf_size: usize = 65536;
/// Page sent when a response cannot be escaped (OOM). Recorded in the replay
/// ledger alongside the failed request: the turn is already stored, so a retry
/// has to collapse onto this page rather than append a second one.
const render_error_page: []const u8 = "<div class=\"msg assistant\">Error: could not render response</div>";
const cmd_buf_size: usize = 1024;
const gen_ids_buf_size: usize = 4096;
/// Hard cap on `max_tokens` per request. Equal to `gen_ids_buf_size` by design: the
/// generation ID buffer cannot hold more tokens than this (see docs/API.md).
const max_gen_tokens_cap: usize = gen_ids_buf_size;
const default_max_gen_tokens: usize = 512;
const system_fingerprint = "agave-v" ++ engine_version;

/// Clamp a max_tokens value to [1, max_gen_tokens_cap].
fn clampMaxTokens(raw: ?usize) usize {
    return @max(1, @min(raw orelse default_max_gen_tokens, max_gen_tokens_cap));
}

/// First present alias wins. Chat uses `max_tokens` / `max_completion_tokens`;
/// `/v1/responses` prefers OpenAI's `max_output_tokens`.
fn extractMaxGenTokens(body: []const u8, comptime names: []const []const u8) usize {
    inline for (names) |name| {
        if (json.extractIntField(body, name)) |v| return clampMaxTokens(v);
    }
    return clampMaxTokens(null);
}

/// Estimate bytes needed for `exportKvPrefix(n_tokens)`.
/// Uses model dims with headroom for per-layer KV variation; capped at `kv_export_max_bytes`.
fn estimateKvExportBytes(model: Model, n_tokens: usize) usize {
    const n_layers: usize = model.nLayers();
    const n_embd: usize = model.nEmbd();
    const n_head: usize = @max(model.nHead(), 1);
    const n_head_kv: usize = @max(model.nHeadKv(), 1);
    const head_dim = n_embd / n_head;
    const kvd = n_head_kv * head_dim;
    // K+V per layer × headroom for dual-attn / MLA dim spread across layers.
    const per_token = std.math.mul(usize, n_layers, kvd) catch return kv_export_max_bytes;
    const per_token2 = std.math.mul(usize, per_token, 2) catch return kv_export_max_bytes;
    const per_token3 = std.math.mul(usize, per_token2, @sizeOf(f32)) catch return kv_export_max_bytes;
    const per_token4 = std.math.mul(usize, per_token3, kv_export_dim_headroom) catch return kv_export_max_bytes;
    const raw = std.math.mul(usize, n_tokens, per_token4) catch return kv_export_max_bytes;
    return @min(if (raw == 0) kv_export_max_bytes else raw, kv_export_max_bytes);
}
const conv_title_max_len = conv_store.max_title_len;
const conv_list_buf_size: usize = 8192;
const conv_msgs_buf_size: usize = 65536;
const http_buf_size: usize = 1024 * 1024;
const hdr_buf_size: usize = 2048;
const short_hdr_buf_size: usize = 512;
const error_body_buf_size: usize = 512;
const max_log_path_len: usize = 256;
/// Max length of an inbound `X-Request-Id` copied into access logs as `xid=`.
const max_client_request_id_len: usize = 64;
const health_buf_size: usize = 768;
const metrics_render_buf_size: usize = 65536;
const stats_buf_size: usize = 512;
const sse_event_buf_size: usize = 1024;
/// Stack budget for decoding a single streamed token without allocating.
/// Tokens longer than this fall back to the allocating batch decode.
const stream_decode_buf_size: usize = 512;
const logprob_buf_size: usize = 4096;
const clear_response_buf_size: usize = 128;
/// Must not exceed http_buf_size, headers and body share the same read buffer.
const max_request_body_size: usize = http_buf_size;
/// Cap on `/v1/kv_cache` GET export buffer. Prevents unbounded alloc on huge `n_tokens`.
const kv_export_max_bytes: usize = 64 * 1024 * 1024;
/// Extra factor for per-layer KV dim variation (dual attention, MLA).
const kv_export_dim_headroom: usize = 2;
/// In-memory admission caps, taken from the store that persists them: a
/// conversation admitted here must survive the next `conv_store` save.
const max_conversations = conv_store.max_conversations;
const max_messages_per_conv = conv_store.max_messages_per_conv;
const max_message_len: usize = 100_000;
const max_concurrent_connections: u32 = 64;
/// Retry-After seconds advertised on 503 when at connection capacity or spawn fails.
const capacity_retry_after_sec: u32 = 1;
const capacity_503_body = "{\"error\":{\"message\":\"Server at capacity\",\"type\":\"server_error\",\"param\":null,\"code\":\"server_overloaded\"}}";
const spawn_fail_503_body = "{\"error\":{\"message\":\"Server unable to handle request\",\"type\":\"server_error\",\"param\":null,\"code\":\"server_overloaded\"}}";
const method_not_allowed_openai = "{\"error\":{\"message\":\"Method not allowed\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"method_not_allowed\"}}";
const method_not_allowed_anthropic = "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Method not allowed\"}}";
/// When only one of rpm/tpm is set via CLI, the unset bucket uses these
/// capacities so the configured limit is the effective constraint.
const rate_limit_unlimited_rpm: u32 = 1_000_000;
const rate_limit_unlimited_tpm: u32 = 100_000_000;
const scheduler_max_batch_size: usize = 8;
const scheduler_timeout_sec: u32 = 120;
const scheduler_poll_interval_ns: u64 = 1_000_000; // 1ms, matches scheduler_poll_ns in scheduler.zig
/// Allows Ctrl+C to interrupt the accept loop.
const accept_timeout_sec: i64 = 1;
const ms_per_second: f32 = 1000.0;

const stderr_file = Io.File.stderr();
const stdout_file = Io.File.stdout();

const sim_clock = @import("../sim_clock.zig");

/// Monotonic millisecond clock for all interval math (request latency,
/// prefill/generation durations, idle detection), injectable via sim_clock
/// for deterministic tests. Immune to NTP steps that would skew or negate
/// durations measured against REALTIME.
fn milliTimestamp() i64 {
    return sim_clock.monoMilli();
}

/// Nanosecond timestamp for seed generation (same injectable clock).
fn nanoTimestamp() i96 {
    return sim_clock.nanoNow();
}

/// PRNG seed from request params, falling back to the injectable clock.
fn prngSeedFromSampling(sampling: SamplingParams) u64 {
    return sampling.seed orelse @as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp()))));
}

/// 2^64 / φ, mixes request id into a clock-derived seed so concurrent
/// scheduler admits at the same virtual millisecond do not share one PRNG stream.
const prng_seed_mix_golden: u64 = 0x9E3779B97F4A7C15;

/// Effective scheduler PRNG seed: honor explicit `sampling.seed`, otherwise
/// derive from sim_clock and request id (not the enqueue-time id alone).
///
/// The request id is handed out per connection thread, so its value depends on
/// the order the kernel happened to accept connections in. Mixing it in makes
/// an auto-seed depend on host scheduling, which a replay cannot reproduce, so
/// under a clock override the seed is the virtual-clock value alone: a run then
/// samples from one stream seeded by the simulated timeline, and replaying the
/// same timeline replays the same tokens.
fn schedulerPrngSeed(req_id: u64, sampling: SamplingParams) u64 {
    const base = prngSeedFromSampling(sampling);
    if (sampling.seed != null) return base;
    if (sim_clock.isOverridden()) return base;
    return base ^ (req_id *% prng_seed_mix_golden);
}

/// True when a sampled request used a clock-derived seed that is not in
/// the request body. Logging it is what makes the run replayable.
fn shouldNoteAutoSeed(sampling: SamplingParams) bool {
    return sampling.temperature > 0 and sampling.seed == null;
}

/// Log the effective PRNG seed when the client omitted `seed`.
fn noteAutoSeed(sampling: SamplingParams, seed: u64) void {
    if (!shouldNoteAutoSeed(sampling)) return;
    std.log.info("req={d} seed={d} (auto; pass seed to reproduce)", .{ log_request_id, seed });
}

/// Sleep via sim_clock so virtual time advances under a clock override.
fn sleepNs(ns: u64) void {
    sim_clock.sleepNs(ns);
}

/// Seconds since epoch for log timestamps (injectable via sim_clock).
fn timestamp() i64 {
    return @divTrunc(sim_clock.milliNow(), 1000);
}

/// Process uptime in whole seconds from two monotonic-ms stamps.
///
/// `/health` `uptime_s` is elapsed process time, not a wall-clock difference:
/// an NTP step backward would otherwise report 0 (already clamped) and a
/// step forward would inflate uptime, which load balancers read as "this
/// process has been up a long time."
fn uptimeSeconds(start_mono_ms: i64, now_mono_ms: i64) i64 {
    return @divFloor(@max(0, now_mono_ms - start_mono_ms), std.time.ms_per_s);
}

/// Background sleep monitor: checks idle time every 10s.
/// Sets g_server.sleeping=true after sleep_after_s seconds of inactivity.
/// Flag-only signal for orchestrators via /health (docs/ARCHITECTURE.md,
/// tutorial 23): weights, KV cache, and prefix/ngram state stay resident so
/// the next request pays no wake-up prefill cost.
fn sleepMonitorLoop(shutdown: *const std.atomic.Value(bool)) void {
    const poll_interval_ns: u64 = 10 * std.time.ns_per_s; // check every 10s
    while (!shutdown.load(.acquire)) {
        sleepNs(poll_interval_ns);
        if (shutdown.load(.acquire)) break;
        const sleep_after_s = g_server.sleep_after_s;
        if (sleep_after_s == 0) continue;
        const last_ms = g_server.last_request_ms.load(.acquire);
        const now_ms = milliTimestamp();
        const idle_s: u32 = @intCast(@max(0, @divFloor(now_ms - last_ms, 1000)));
        if (!g_server.sleeping.load(.acquire) and idle_s >= sleep_after_s) {
            // Only enter sleep when truly idle: last_request_ms alone is wrong for
            // long-running generations that outlast sleep_after_s. Re-check
            // connections and last_request under the mutex (unlocked check races
            // with accept / wake-from-sleep on the handler path).
            var entered_sleep = false;
            {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                if (g_server.metrics.active_connections.load(.acquire) == 0 and
                    !g_server.sleeping.load(.acquire))
                {
                    const last_locked = g_server.last_request_ms.load(.acquire);
                    const now_locked = milliTimestamp();
                    const idle_locked: u32 = @intCast(@max(0, @divFloor(now_locked - last_locked, 1000)));
                    if (idle_locked >= sleep_after_s) {
                        g_server.sleeping.store(true, .release);
                        entered_sleep = true;
                    }
                }
            }
            if (entered_sleep) {
                g_server.metrics.updateSleeping(true);
                std.log.info("server: entering sleep mode after {d}s idle", .{idle_s});
            }
        }
    }
}

/// Copy per-request sampling onto a scheduler Request (enqueue defaults to greedy).
/// Publishes sampling_ready last so the scheduler cannot admit until fields are set.
fn configureSchedulerSampling(req: *scheduler.Request, sampling: SamplingParams) void {
    req.temperature = sampling.temperature;
    req.top_k = sampling.top_k;
    req.top_p = sampling.top_p;
    req.min_p = sampling.min_p;
    req.frequency_penalty = sampling.frequency_penalty;
    req.presence_penalty = sampling.presence_penalty;
    req.repetition_penalty = sampling.repetition_penalty;
    req.dry_multiplier = sampling.dry_multiplier;
    req.dry_allowed_length = sampling.dry_allowed_length;
    req.xtc_probability = sampling.xtc_probability;
    req.xtc_threshold = sampling.xtc_threshold;
    req.mirostat = sampling.mirostat;
    req.mirostat_tau = sampling.mirostat_tau;
    req.mirostat_eta = sampling.mirostat_eta;
    req.mirostat_mu = sampling.mirostat_tau * 2.0;
    const n_bias = @min(sampling.logit_bias_count, @as(u32, @intCast(req.logit_bias_ids.len)));
    req.logit_bias_count = n_bias;
    @memcpy(req.logit_bias_ids[0..n_bias], sampling.logit_bias_ids[0..n_bias]);
    @memcpy(req.logit_bias_vals[0..n_bias], sampling.logit_bias_vals[0..n_bias]);
    // Always re-seed: enqueue leaves a placeholder id-based PRNG; null seed must
    // follow prngSeedFromSampling (sim_clock) like the direct generation paths.
    const sched_seed = schedulerPrngSeed(req.id, sampling);
    noteAutoSeed(sampling, sched_seed);
    req.prng = std.Random.DefaultPrng.init(sched_seed);
    req.rebuildSampler();
    req.image_embeddings = pending_visual_embeddings;
    req.n_visual_tokens = pending_visual_tokens;
    req.image_pad_token_id = if (pending_visual_tokens > 0) g_server.image_pad_token_id else 0;
    req.sampling_ready.store(true, .release);
}

/// Install this handler's pending vision embeddings on the model.
/// Caller must hold `model_mutex` (direct-path generate, after
/// `lockModelWithScheduler`). Scheduler requests go through Request fields
/// instead so this thread never races `forward()`.
fn applyPendingVisionEmbeddings() void {
    if (pending_visual_tokens > 0) {
        g_server.model.setImageEmbeddings(pending_visual_embeddings, pending_visual_tokens, g_server.image_pad_token_id);
    } else {
        g_server.model.setImageEmbeddings(null, 0, 0);
    }
}

fn clearPendingVision() void {
    pending_visual_tokens = 0;
    pending_visual_embeddings = null;
}

/// Architecture image placeholder IDs from the running server config.
fn serverImageTokens() ImageTokens {
    return .{
        .start = g_server.image_start_token_id,
        .end = g_server.image_end_token_id,
        .pad = g_server.image_pad_token_id,
    };
}

/// Rolling UTF-8 window for stop-sequence matching (same size as direct path).
const scheduler_stop_buf_size: usize = 128;

fn appendStopWindow(buf: *[scheduler_stop_buf_size]u8, len: *usize, piece: []const u8) void {
    if (piece.len == 0) return;
    if (piece.len >= scheduler_stop_buf_size) {
        @memcpy(buf, piece[piece.len - scheduler_stop_buf_size ..]);
        len.* = scheduler_stop_buf_size;
    } else if (len.* + piece.len <= scheduler_stop_buf_size) {
        @memcpy(buf[len.*..][0..piece.len], piece);
        len.* += piece.len;
    } else {
        const keep = scheduler_stop_buf_size - piece.len;
        std.mem.copyForwards(u8, buf[0..keep], buf[len.* - keep .. len.*]);
        @memcpy(buf[keep..][0..piece.len], piece);
        len.* = scheduler_stop_buf_size;
    }
}

/// Decode newly visible scheduler tokens into a rolling window; on stop match
/// mark the request finished and return the token count to keep (inclusive).
fn pollSchedulerStop(
    req: *scheduler.Request,
    tok: *Tokenizer,
    sampling: SamplingParams,
    stop_buf: *[scheduler_stop_buf_size]u8,
    stop_len: *usize,
    checked_len: *usize,
    allocator: Allocator,
) ?u32 {
    if (!sampling.hasStop()) return null;
    const cur = req.visible_len.load(.acquire);
    var piece_buf: [stream_decode_buf_size]u8 = undefined;
    while (checked_len.* < cur) {
        const id = req.tokens.items[checked_len.*];
        // Fast path: decode into a stack buffer (no per-token heap traffic).
        if (tok.decodeOne(id, &piece_buf)) |piece| {
            appendStopWindow(stop_buf, stop_len, piece);
        } else {
            const piece = tok.decode(&[_]u32{id}) catch |err| {
                std.log.warn("req={d} stop-sequence decode failed (id={d}): {}", .{ req.id, id, err });
                checked_len.* += 1;
                continue;
            };
            defer allocator.free(piece);
            appendStopWindow(stop_buf, stop_len, piece);
        }
        checked_len.* += 1;
        if (sampling.matchesStop(stop_buf[0..stop_len.*])) {
            req.is_finished.store(true, .release);
            return @intCast(checked_len.*);
        }
    }
    return null;
}

/// Compute tokens-per-second from a token count and elapsed milliseconds.
/// Returns 0 if elapsed time is zero (avoids division by zero).
fn tokensPerSec(token_count: u32, time_ms: u64) f32 {
    return if (time_ms > 0) @as(f32, @floatFromInt(token_count)) / (@as(f32, @floatFromInt(time_ms)) / ms_per_second) else 0.0;
}

/// Authentication policy a route answers with. Applied by the dispatcher's
/// single chokepoint (`authorizedForPath`), never per handler.
const AuthPolicy = enum {
    /// Reject with 401 unless the request carries the configured API key.
    required,
    /// Serve unauthenticated, but let the handler trim its response for an
    /// unauthenticated caller (`/health`, `/ready`).
    optional,
    /// Serve unauthenticated with no server state behind it (`/favicon.ico`).
    public,
};

/// Known API endpoints with their allowed HTTP methods and error messages.
/// Shared by the CORS OPTIONS handler (path-specific Access-Control-Allow-Methods)
/// and the 405 Method Not Allowed handler. `auth` defaults to `.required` so a
/// new entry is protected unless it explicitly opts out.
const KnownEndpoint = struct {
    path: []const u8,
    allow: []const u8,
    msg: []const u8,
    is_anthropic: bool = false,
    auth: AuthPolicy = .required,
};
const known_endpoints = [_]KnownEndpoint{
    .{ .path = "/v1/chat/completions", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/completions", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/messages", .allow = "POST, OPTIONS", .msg = "Use POST.", .is_anthropic = true },
    .{ .path = "/v1/embeddings", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/responses", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/chat", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/chat/regenerate", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/conversations", .allow = "GET, POST, OPTIONS", .msg = "Use GET or POST." },
    .{ .path = "/v1/tokenize", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/detokenize", .allow = "POST, OPTIONS", .msg = "Use POST." },
    .{ .path = "/v1/models", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/v1/kv_cache", .allow = "GET, POST, OPTIONS", .msg = "Use GET or POST." },
    .{ .path = "/v1/kv_cache/info", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/health", .allow = "GET, OPTIONS", .msg = "Use GET.", .auth = .optional },
    .{ .path = "/ready", .allow = "GET, OPTIONS", .msg = "Use GET.", .auth = .optional },
    .{ .path = "/metrics", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/favicon.ico", .allow = "GET, OPTIONS", .msg = "Use GET.", .auth = .public },
};

/// Auth policy for a request path. A path absent from `known_endpoints` is
/// `.required`, so an unrecognized or newly added route cannot answer an
/// unauthenticated caller by omission.
fn authPolicyFor(path: []const u8) AuthPolicy {
    for (known_endpoints) |ep| {
        if (std.mem.eql(u8, path, ep.path)) return ep.auth;
    }
    return .required;
}

/// True when `path` addresses a route whose errors use the Anthropic envelope.
/// Used for failures raised before routing (malformed request, oversized body,
/// rejected Host, cross-origin request, failed auth) so a `/v1/messages` client
/// never receives an OpenAI-shaped error.
fn isAnthropicPath(path: []const u8) bool {
    for (known_endpoints) |ep| {
        if (ep.is_anthropic) return std.mem.eql(u8, path, ep.path);
    }
    return false;
}

/// Per-connection read timeout (seconds), prevents slow loris DoS attacks
/// where an attacker holds connections open by sending data one byte at a time.
const connection_read_timeout_sec: i64 = 30;
/// Poll interval (milliseconds) while draining active connections during shutdown.
const drain_poll_interval_ms: u64 = 100;
/// KV cache utilization percentage above which `/health` reports "degraded".
const kv_cache_degradation_pct: u32 = 90;
/// Minimum completed+failed requests before error rate check activates.
const error_rate_min_requests: u64 = 10;
/// Error rate percentage (failed / (completed + failed)) above which `/health` reports "degraded".
const error_rate_degradation_pct: u64 = 50;

/// Compact `/health`/`/ready` state for transition logs and `agave_ready`.
const HealthState = enum(u8) {
    ok = 0,
    kv_pressure = 1,
    high_error_rate = 2,
    kv_pressure_and_errors = 3,
    shutting_down = 4,
};
/// Seconds per minute, used for UTC time decomposition in request logs.
const seconds_per_minute: u64 = 60;
/// Seconds per hour, used for UTC time decomposition in request logs.
const seconds_per_hour: u64 = 3600;
/// Hours per day, used for UTC time decomposition in request logs.
const hours_per_day: u64 = 24;
const fnv1a_offset_basis: u64 = 14695981039346656037;
/// FNV-1a 64-bit prime.
const fnv1a_prime: u64 = 1099511628211;

/// A single conversation with its message history.
const Conversation = struct {
    id: u32,
    title_buf: [conv_title_max_len]u8 = undefined,
    title_len: u8 = 0,
    messages: std.ArrayList(Message) = .empty,

    fn titleSlice(self: *const Conversation) []const u8 {
        return self.title_buf[0..self.title_len];
    }

    fn setTitle(self: *Conversation, text: []const u8) void {
        @memset(&self.title_buf, 0);
        var len: usize = @min(text.len, conv_title_max_len);
        // Walk backwards to avoid truncating in the middle of a multi-byte UTF-8 sequence.
        while (len > 0) {
            const byte = text[len - 1];
            if (byte & 0x80 == 0) break; // ASCII, clean boundary
            if (byte & 0xC0 == 0xC0) {
                // Start byte of a multi-byte sequence, check if the full sequence fits.
                const seq_len = std.unicode.utf8ByteSequenceLength(byte) catch 1;
                if (len - 1 + seq_len > @min(text.len, conv_title_max_len)) {
                    // Sequence would be incomplete; drop it.
                    len -= 1;
                } else {
                    len = len - 1 + seq_len; // Full sequence fits, extend to include it.
                    break;
                }
                break;
            }
            // Continuation byte (10xxxxxx), keep walking back.
            len -= 1;
        }
        const safe_len: u8 = @intCast(len);
        @memcpy(self.title_buf[0..safe_len], text[0..safe_len]);
        self.title_len = safe_len;
    }

    /// Free one owned message: content and optional `tool_call_id`.
    /// Matches `conv_store.Snapshot.deinit` so loaded tool messages cannot leak.
    fn freeOwnedMessage(allocator: Allocator, msg: Message) void {
        const content = @constCast(msg.content);
        @memset(content, 0);
        allocator.free(content);
        if (msg.tool_call_id) |tcid| {
            const t = @constCast(tcid);
            @memset(t, 0);
            allocator.free(t);
        }
    }

    /// Put a held message back at the end of `self`. Used to undo a pop on a
    /// regenerate that fails before the reply is replaced: the conversation is
    /// then left exactly as one run left it, so the retry the released
    /// idempotency key invites has the same reply to work from.
    fn restoreHeldMessage(self: *Conversation, allocator: Allocator, msg: Message) void {
        self.messages.append(allocator, msg) catch {
            freeOwnedMessage(allocator, msg);
        };
    }

    fn freeMessageContents(self: *Conversation, allocator: Allocator) void {
        // Zero content before free so prompt/response text does not linger in the allocator freelist.
        for (self.messages.items) |msg| {
            freeOwnedMessage(allocator, msg);
        }
    }

    fn clearMessages(self: *Conversation, allocator: Allocator) void {
        self.freeMessageContents(allocator);
        self.messages.clearRetainingCapacity();
        @memset(&self.title_buf, 0);
        self.title_len = 0;
    }

    fn freeMessages(self: *Conversation, allocator: Allocator) void {
        self.freeMessageContents(allocator);
        self.messages.deinit(allocator);
        @memset(&self.title_buf, 0);
        self.title_len = 0;
    }
};

/// Server state, bundles all mutable state into a single struct
/// instead of scattered globals. Only g_server is a global (required
/// because the accept loop callback doesn't carry a context pointer).
const Server = struct {
    model: *Model,
    tokenizer: *Tokenizer,
    chat_template: ChatTemplate,
    model_name: []const u8,
    backend_name: []const u8,
    allocator: Allocator,
    bos_token_id: u32,
    /// End-of-generation token IDs (primary EOS + any additional EOG/EOT tokens).
    eog_ids: [max_eog_ids]u32 = undefined,
    eog_len: usize = 0,
    /// Conversation storage for the web UI. Persisted to `conv_store_path` when set.
    conversations: std.ArrayList(Conversation) = .empty,
    active_id: u32 = 0,
    next_id: u32 = 1,
    /// JSON conversation store path. Null disables persist/restore.
    conv_store_path: ?[]const u8 = null,
    /// Replay ledger for the mutating chat routes, keyed by client
    /// `X-Request-Id`. Guarded by `mutex`.
    idem: Idempotency.Ledger,
    /// Whether the KV cache matches the active conversation's state.
    kv_valid: bool = false,
    /// Cached prompt token IDs from the last API generation (for prefix reuse).
    cached_prompt_ids: []u32 = &.{},
    /// Lazily tokenized chat `user_prefix` (stable for the server lifetime).
    /// Avoids re-encoding the same prefix on every multimodal request.
    /// `null` = not yet computed; otherwise owned (possibly empty) slice.
    cached_user_prefix_ids: ?[]u32 = null,
    mutex: Mutex = .init,
    stdout_mutex: Mutex = .init,
    /// Serializes vision encode + inference for multimodal requests.
    /// Prevents concurrent processVisionImage calls from corrupting the
    /// shared vision encoder buffers and model embedding state.
    /// Lock ordering: vision_mutex → mutex (inference).
    vision_mutex: Mutex = .init,
    io: Io,
    /// Monotonically increasing request counter for unique response IDs.
    request_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Server start time (unix timestamp, set once in run()).
    start_time: i64 = 0,
    /// Monotonic milliseconds at process start, for `/health` `uptime_s`.
    /// Not comparable to `start_time` (epoch seconds) or to other machines.
    start_mono_ms: i64 = 0,
    /// Continuous batching scheduler (null = single-request mode).
    request_manager: ?*scheduler.RequestManager = null,
    /// Global rate limiter (null = no rate limiting).
    rate_limiter: ?*RateLimiter = null,
    /// API key for authentication (null = no auth).
    api_key: ?[]const u8 = null,
    /// Background scheduler thread (null when not using scheduler).
    scheduler_thread: ?std.Thread = null,
    /// Shutdown signal for scheduler loop.
    scheduler_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Prometheus metrics collector.
    metrics: Metrics = .{},
    /// Optional vision encoder for multimodal image support.
    vision_encoder: ?*VisionEncoder = null,
    /// Image pad token ID for multimodal prompt injection (architecture-specific).
    image_pad_token_id: u32 = 0,
    /// Image start token ID (e.g. <img> = 219 for Gemma).
    image_start_token_id: u32 = 0,
    /// Image end token ID (e.g. </img> = 230 for Gemma).
    image_end_token_id: u32 = 0,
    /// Draft model for speculative decoding (null = no spec dec).
    draft_model: ?*Model = null,
    spec_tokens: u32 = 5,
    tree_budget: u32 = 64,
    /// Graceful shutdown flag (set by SIGTERM/SIGINT).
    shutdown_requested: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Context window size (tokens).
    ctx_size: u32 = 0,
    /// Sleep mode: seconds of idle time before sleeping (0 = disabled).
    sleep_after_s: u32 = 0,
    /// Timestamp of last completed request (monotonic ms). 0 = no requests yet.
    last_request_ms: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),
    /// True when the server is in sleep mode (idle too long).
    sleeping: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Last published `/health` state (`HealthState`) for transition-only logs.
    health_state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(HealthState.ok)),
    /// Process-level tools (register/dispose). Request JSON tools overlay these.
    tool_registry: tools_mod.Registry = .{},
    /// gzip of `html_page`, filled once in `run()`. Empty if compression failed.
    html_gzip: []const u8 = &.{},

    fn getActiveConv(self: *Server) ?*Conversation {
        return self.getConvById(self.active_id);
    }

    fn getConvById(self: *Server, id: u32) ?*Conversation {
        for (self.conversations.items) |*conv| {
            if (conv.id == id) return conv;
        }
        return null;
    }

    /// Return tokenized chat user_prefix, computing and caching on first use.
    /// Thread-safe: multiple handler threads may call this concurrently.
    /// Lock ordering: may be called while holding vision_mutex (vision → mutex).
    fn userPrefixIds(self: *Server) []const u32 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.cached_user_prefix_ids) |ids| return ids;
        const ids = self.tokenizer.encode(self.chat_template.user_prefix) catch |err| {
            std.log.warn("req={d} user_prefix tokenize failed: {}", .{ log_request_id, err });
            return &.{};
        };
        self.cached_user_prefix_ids = ids;
        return ids;
    }

    /// Publish `token_ids` as the KV prefix memo for the next request. Single
    /// entry point so the read guard in `prefixCacheApplies` and this write
    /// cannot drift: a memo written by a vision request holds image placeholder
    /// IDs, which every image shares, so a later text request would match the
    /// prefix and read the earlier request's picture back out of the KV cache.
    /// Caller must hold self.mutex.
    fn publishCachedPromptIds(self: *Server, token_ids: []const u32, n_visual: u32) void {
        self.clearCachedPromptIds();
        if (!prefixCacheApplies(n_visual)) return;
        self.cached_prompt_ids = self.allocator.dupe(u32, token_ids) catch blk: {
            std.log.warn("req={d} prefix-cache OOM ({d} tokens); next request will re-prefill", .{ log_request_id, token_ids.len });
            break :blk &.{};
        };
    }

    /// Drop cached prompt-prefix token IDs. Must be called whenever the KV
    /// cache is wiped (`resetCache`) so the next request does not treat empty
    /// slots as a prefix-cache hit. Caller must hold self.mutex when other
    /// threads may read `cached_prompt_ids`.
    fn clearCachedPromptIds(self: *Server) void {
        if (self.cached_prompt_ids.len == 0) return;
        @memset(std.mem.sliceAsBytes(self.cached_prompt_ids), 0);
        self.allocator.free(self.cached_prompt_ids);
        self.cached_prompt_ids = &.{};
    }

    /// Create a new conversation. Caller must hold self.mutex.
    /// The caller persists with `persistConversations` once the lock is free.
    fn createConv(self: *Server) ?*Conversation {
        if (self.conversations.items.len >= max_conversations) return null;
        const id = self.next_id;
        self.next_id +%= 1;
        if (self.next_id == 0) self.next_id = 1;
        self.conversations.append(self.allocator, .{ .id = id }) catch {
            std.log.warn("req={d} conversation allocation failed", .{log_request_id});
            return null;
        };
        self.active_id = id;
        self.kv_valid = false;
        self.clearCachedPromptIds();
        if (ngram_mod.global_pool) |*pool| pool.clear();
        // Opaque title only, never store user message text (may contain PII).
        const conv = &self.conversations.items[self.conversations.items.len - 1];
        var title_buf: [24]u8 = undefined;
        const title = std.fmt.bufPrint(&title_buf, "Chat {d}", .{id}) catch "Chat";
        conv.setTitle(title);
        return conv;
    }

    /// Delete a conversation by ID. Caller must hold self.mutex.
    /// Always wipe the shared n-gram pool: it mixes tokens from every
    /// conversation, so one erasure cannot surgically remove its history.
    /// When the active conversation is deleted, also wipe KV / prefix cache
    /// so prompt-derived state does not survive (matches `/clear` / `/reset`).
    /// The caller persists with `persistConversations` once the lock is free.
    fn deleteConv(self: *Server, id: u32) void {
        var removed = false;
        for (self.conversations.items, 0..) |*conv, i| {
            if (conv.id == id) {
                conv.freeMessages(self.allocator);
                _ = self.conversations.orderedRemove(i);
                removed = true;
                break;
            }
        }
        if (removed) {
            if (ngram_mod.global_pool) |*pool| pool.clear();
        }
        if (self.active_id == id) {
            self.active_id = if (self.conversations.items.len > 0)
                self.conversations.items[self.conversations.items.len - 1].id
            else
                0;
            lockModelWithScheduler();
            defer unlockModelWithScheduler();
            self.model.resetCache();
            self.kv_valid = false;
            self.clearCachedPromptIds();
        }
    }

    /// Select a conversation by ID. Caller must hold self.mutex.
    /// Wipes the prefix memo and the shared n-gram pool alongside the KV flag:
    /// both hold the previous conversation's tokens, and neither can be
    /// trimmed to one conversation, so a switch carries state across.
    /// Returns true when the selection changed, so the caller knows to persist.
    fn selectConv(self: *Server, id: u32) bool {
        if (self.active_id != id) {
            self.active_id = id;
            self.kv_valid = false;
            self.clearCachedPromptIds();
            if (ngram_mod.global_pool) |*pool| pool.clear();
            return true;
        }
        return false;
    }

    /// Claim the caller's `X-Request-Id` in the replay ledger, scoped to
    /// `route`: the id alone is client-chosen and would replay across
    /// operations. A request without a client id has nothing to deduplicate on
    /// and always runs. A fresh claim's token and route are stashed in
    /// `log_idem_token` / `log_idem_route` for this thread.
    fn claimIdempotencyKey(self: *Server, route: []const u8) Idempotency.Claim {
        const key = log_client_rid[0..log_client_rid_len];
        if (key.len == 0) return .{ .fresh = 0 };
        log_idem_route_len = 0;
        if (route.len <= Idempotency.max_route_len) {
            @memcpy(log_idem_route[0..route.len], route);
            log_idem_route_len = route.len;
        }
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const claim = self.idem.claim(key, log_idem_route[0..log_idem_route_len], milliTimestamp());
        log_idem_token = if (claim == .fresh) claim.fresh else 0;
        return claim;
    }

    /// Record a mutating request's response so a retry replays it instead of
    /// applying the operation twice. `status_line` and `content_type` must be
    /// string literals or otherwise outlive the server.
    fn completeIdempotencyKey(self: *Server, status_line: []const u8, content_type: []const u8, body: []const u8) void {
        const key = log_client_rid[0..log_client_rid_len];
        if (key.len == 0 or log_idem_token == 0) return;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.idem.complete(key, log_idem_route[0..log_idem_route_len], log_idem_token, milliTimestamp(), status_line, content_type, body);
    }

    /// Drop an unfinished claim so a failed request does not block its retries.
    fn releaseIdempotencyKey(self: *Server) void {
        const key = log_client_rid[0..log_client_rid_len];
        if (key.len == 0 or log_idem_token == 0) return;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.idem.release(key, log_idem_route[0..log_idem_route_len], log_idem_token);
    }

    /// Deep-copy the conversation views into `arena` so they stay valid after
    /// `mutex` is released. Caller must hold self.mutex.
    fn snapshotConversationsLocked(self: *Server, arena: std.mem.Allocator) ![]conv_store.ConvView {
        const n = @min(self.conversations.items.len, max_conversations);
        const views = try arena.alloc(conv_store.ConvView, n);
        for (self.conversations.items[0..n], 0..) |*conv, i| {
            const msgs = try arena.alloc(Message, conv.messages.items.len);
            for (conv.messages.items, 0..) |m, j| {
                msgs[j] = .{ .role = m.role, .content = try arena.dupe(u8, m.content) };
            }
            views[i] = .{
                .id = conv.id,
                .title = try arena.dupe(u8, conv.titleSlice()),
                .messages = msgs,
            };
        }
        return views;
    }

    /// Encode `views` and write them to `path`, replacing the file
    /// durably. Takes no lock: `conv_store.save` runs two fsyncs, so it must
    /// not run while `mutex` is held, or every other request stalls on it.
    fn saveConversationSnapshot(
        self: *Server,
        path: []const u8,
        active_id: u32,
        next_id: u32,
        views: []const conv_store.ConvView,
    ) void {
        conv_store.save(self.allocator, path, active_id, next_id, views) catch |err| {
            self.metrics.recordConvStoreSaveFailure();
            std.log.err("conversation store save failed ({s}): {}; conversation history is not persisted", .{ path, err });
        };
    }

    /// Write conversations to `conv_store_path`. Caller must hold self.mutex.
    fn persistConversationsLocked(self: *Server) void {
        const path = self.conv_store_path orelse return;
        var views_buf: [max_conversations]conv_store.ConvView = undefined;
        const n = @min(self.conversations.items.len, max_conversations);
        for (self.conversations.items[0..n], 0..) |*conv, i| {
            views_buf[i] = .{
                .id = conv.id,
                .title = conv.titleSlice(),
                .messages = conv.messages.items,
            };
        }
        conv_store.save(self.allocator, path, self.active_id, self.next_id, views_buf[0..n]) catch |err| {
            // Reached from request threads and from startup/shutdown, so no
            // request ID is attached here; the counter carries the signal.
            self.metrics.recordConvStoreSaveFailure();
            std.log.err("conversation store save failed ({s}): {}; conversation history is not persisted", .{ path, err });
        };
    }

    /// Write conversations to `conv_store_path` from a request path, taking
    /// `mutex` only to snapshot the store. `conv_store.save` runs two fsyncs,
    /// so the write itself runs unlocked; holding `mutex` across it stalls
    /// every other request for the duration of the sync.
    fn persistConversations(self: *Server) void {
        if (self.conv_store_path == null) return;
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var views_buf: [max_conversations]conv_store.ConvView = undefined;
        var n: usize = 0;
        var active_id: u32 = 0;
        var next_id: u32 = 0;
        {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            n = @min(self.conversations.items.len, max_conversations);
            for (self.conversations.items[0..n], 0..) |*conv, i| {
                views_buf[i] = .{
                    .id = conv.id,
                    .title = arena.dupe(u8, conv.titleSlice()) catch {
                        self.metrics.recordConvStoreSaveFailure();
                        return;
                    },
                    .messages = arena.dupe(Message, conv.messages.items) catch {
                        self.metrics.recordConvStoreSaveFailure();
                        return;
                    },
                };
            }
            active_id = self.active_id;
            next_id = self.next_id;
        }
        self.saveConversationSnapshot(self.conv_store_path.?, active_id, next_id, views_buf[0..n]);
    }

    /// Restore conversations from `conv_store_path`. Caller must hold self.mutex
    /// (or be the only thread during startup). Missing file is a no-op.
    fn loadConversationsLocked(self: *Server) void {
        const path = self.conv_store_path orelse return;
        var snap = conv_store.load(self.allocator, path) catch |err| {
            if (err == error.FileNotFound) {
                std.log.info("conversation store: {s} (new)", .{path});
                return;
            }
            // A corrupt file is quarantined before this error returns, so
            // persisting a fresh store is safe. Any other failure, including a
            // store a newer agave wrote, left the live file in place: do not
            // overwrite it with an empty in-memory list.
            if (err == error.CorruptStore) {
                std.log.warn("conversation store load failed ({s}): {}", .{ path, err });
                return;
            }
            std.log.err("conversation store: disabling persist to avoid overwriting {s}: {}", .{
                path, err,
            });
            self.conv_store_path = null;
            return;
        };
        defer snap.deinit();

        for (snap.conversations) |*lc| {
            var conv = Conversation{ .id = lc.id };
            conv.setTitle(lc.title);
            conv.messages.items = lc.messages;
            conv.messages.capacity = lc.messages.len;
            lc.messages = &.{};
            self.conversations.append(self.allocator, conv) catch {
                conv.freeMessages(self.allocator);
                std.log.warn("conversation store: dropped remaining conversations (OOM)", .{});
                break;
            };
        }
        self.next_id = snap.next_id;
        var max_id: u32 = 0;
        for (self.conversations.items) |c| {
            if (c.id > max_id) max_id = c.id;
        }
        if (self.next_id <= max_id) self.next_id = max_id +% 1;
        if (self.next_id == 0) self.next_id = 1;
        self.active_id = snap.active_id;
        if (self.getConvById(self.active_id) == null) {
            self.active_id = if (self.conversations.items.len > 0)
                self.conversations.items[self.conversations.items.len - 1].id
            else
                0;
        }
        std.log.info("conversation store: loaded {d} conversation(s) from {s}", .{
            self.conversations.items.len, path,
        });
    }

    fn isEog(self: *const Server, token: u32) bool {
        for (self.eog_ids[0..self.eog_len]) |id| {
            if (token == id) return true;
        }
        return false;
    }
};

var g_server: *Server = undefined;

/// Mark KV-cache bookkeeping invalid after a partially-applied prefill
/// (cancel or forward error mid-loop). The KV cache then holds a partial
/// prompt that matches neither `cached_prompt_ids` nor any continuation
/// assumption, so the next request must fully reset and re-prefill.
/// Without this, a later request can skip prefilling tokens whose KV
/// entries were overwritten by the failed one, silently corrupting output.
/// Caller must hold g_server.mutex (all generate paths do).
fn invalidateKvBookkeeping() void {
    g_server.kv_valid = false;
    g_server.clearCachedPromptIds();
}

/// Whether the KV prefix memo applies to a request that carried `n_visual`
/// image embeddings. The memo is keyed on token IDs alone, and image
/// placeholders are the same IDs for every image, so a prompt with an image
/// must neither read nor publish it: two different images behind an identical
/// token sequence would share KV, and the answer would come from the earlier
/// request's picture. Text prompts keep the full prefix reuse.
fn prefixCacheApplies(n_visual: u32) bool {
    return n_visual == 0;
}

/// Acquire the scheduler's model mutex so direct-path forward loops
/// (grammar / json_mode fallbacks) cannot run concurrently with the
/// scheduler thread's Phase A/B forwards on the same KV cache.
/// No-op when no scheduler exists: direct paths are then serialized by
/// g_server.mutex alone. Must be called while holding g_server.mutex
/// (consistent order: server.mutex → model_mutex).
fn lockModelWithScheduler() void {
    if (g_server.request_manager) |rm| rm.model_mutex.lockUncancelable(g_server.io);
}

/// Release the scheduler's model mutex acquired by lockModelWithScheduler().
fn unlockModelWithScheduler() void {
    if (g_server.request_manager) |rm| rm.model_mutex.unlock(g_server.io);
}

/// Snapshot the model's KV sequence length with the model excluded.
///
/// `kv_seq_len` is a plain `usize` (no atomic). The scheduler thread writes it
/// inside its Phase A/B forward loops while holding only `model_mutex`, and the
/// direct-path generators write it under `server.mutex`. An unlocked read here
/// races with both, and the value is reported next to `kv_cache_used` /
/// `queue_depth` in the same response, so a torn pair is user-visible. Both
/// locks are taken in the documented order; callers that already hold
/// `server.mutex` must inline `lockModelWithScheduler` instead of calling this.
fn kvSeqLenSnapshot() usize {
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    return g_server.model.kvSeqLen();
}

/// Per-thread request ID for log correlation. Set at the start of each
/// handleRequest() call so all log lines from the same request (including
/// logGeneration calls deep in generate functions) share the same ID.
threadlocal var log_request_id: u64 = 0;
/// Sanitized inbound `X-Request-Id` (empty when the client omitted one).
threadlocal var log_client_rid: [max_client_request_id_len]u8 = undefined;
threadlocal var log_client_rid_len: usize = 0;
/// Replay-ledger claim token for this handler thread, or 0 when it does not
/// own a claim. Scopes `completeIdempotencyKey` and `releaseIdempotencyKey`
/// to the claim this request actually took, so a request that outlives the
/// in-flight TTL cannot complete or release the retry that replaced it.
threadlocal var log_idem_token: u64 = 0;
/// Route this handler thread claimed its replay key on. Set by
/// `claimIdempotencyKey` and read by the complete/release helpers, which have
/// no other handle on which route the token belongs to.
threadlocal var log_idem_route: [Idempotency.max_route_len]u8 = undefined;
threadlocal var log_idem_route_len: usize = 0;
/// Visual token count from processVisionImage for the current handler thread only.
/// Must not live on Server: concurrent text requests would observe another
/// connection's count and inject bogus image pad tokens into the prompt.
threadlocal var pending_visual_tokens: u32 = 0;
/// Borrowed encoder output for `pending_visual_tokens`. Valid while this
/// handler holds `vision_mutex`. Copied onto the scheduler Request (or
/// installed under `model_mutex` on the direct path); never written to
/// `Model.image_embeddings` from the handler thread.
threadlocal var pending_visual_embeddings: ?[]const f32 = null;

/// Write a formatted log message to stderr under the server stdout mutex,
/// ensuring concurrent handler threads do not interleave output.
fn slog(comptime fmt: []const u8, args: anytype) void {
    g_server.stdout_mutex.lockUncancelable(g_server.io);
    defer g_server.stdout_mutex.unlock(g_server.io);
    var buf: [slog_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch {
        const trunc = "[slog truncated]\n";
        _ = std.posix.system.write(stderr_file.handle, trunc.ptr, trunc.len);
        return;
    };
    _ = std.posix.system.write(stderr_file.handle, text.ptr, text.len);
}

fn elapsedMs(start: i64) u64 {
    return @intCast(@max(milliTimestamp() - start, 0));
}

/// Compute non-negative elapsed milliseconds between two timestamps.
fn elapsedBetween(start: i64, end: i64) u64 {
    return @intCast(@max(end - start, 0));
}

/// Wall-clock gap sampler for a streaming decode loop. One instance per
/// request: every emitted token records the gap since the previous one, so
/// `agave_inter_token_latency_seconds` covers every streaming endpoint and not
/// only the one whose loop samples it.
const ItlTracker = struct {
    last_ms: i64,

    fn init() ItlTracker {
        return .{ .last_ms = milliTimestamp() };
    }

    /// Record the gap since the previous sample, then advance to now.
    fn sample(self: *ItlTracker) void {
        const now = milliTimestamp();
        g_server.metrics.recordInterTokenLatency(elapsedBetween(self.last_ms, now));
        self.last_ms = now;
    }
};

/// Estimate prompt token count: use actual tokenized count when available,
/// fall back to byte-length estimate (1 byte = 1 token) to prevent rate
/// limiter bypass on tokenizer failure.
fn estimatePromptTokens(token_count: usize, text_len: usize) u32 {
    const n = if (token_count > 0) token_count else @max(1, text_len);
    return std.math.cast(u32, n) orelse std.math.maxInt(u32);
}

/// Capture inbound `X-Request-Id` for access-log correlation (`xid=`).
fn captureClientRequestId(headers: []const u8) void {
    log_client_rid_len = 0;
    log_idem_token = 0;
    const raw = http.getHeaderValue(headers, "x-request-id") orelse return;
    log_client_rid_len = http.sanitizeClientRequestId(raw, &log_client_rid);
}

/// Unauthenticated `--serve`: reject missing or non-loopback Host so a
/// DNS-rebound public hostname cannot drive the API (CWE-350).
fn isRebindHostUnauthenticated(headers: []const u8) bool {
    if (g_server.api_key != null) return false;
    const host = http.getHeaderValue(headers, "host") orelse return true;
    return !http.isLoopbackHttpHost(host);
}

/// Browser cross-origin call with no API key (CSRF / data theft via localhost).
/// Missing Origin (curl, probes) is allowed. Authenticated mode skips this
/// check (clients already present a secret).
fn isCrossOriginUnauthenticated(headers: []const u8) bool {
    if (g_server.api_key != null) return false;
    const origin = http.getHeaderValue(headers, "origin") orelse return false;
    const host = http.getHeaderValue(headers, "host") orelse return true;
    return !http.originMatchesHost(origin, host);
}

/// Broken-down UTC time for request log timestamps.
///
/// Every log line that carries this renders it as `[HH:MM:SSZ]`. The `Z` is
/// not decoration: the surrounding runtime (journald, `docker logs`, a
/// terminal prompt) prints local time, so a bare `HH:MM:SS` reads as local and
/// sits an hour off for half of every DST period, on the two transition
/// weekends and again whenever a host changes zone.
const TimeComponents = struct { hours: u64, minutes: u64, seconds: u64 };

/// Log-line timestamp prefix, UTC and labelled as such. Concatenated into
/// every slog format that prints a TimeComponents, so a line added later
/// cannot drop the marker.
const log_time_fmt = "[{d:0>2}:{d:0>2}:{d:0>2}Z]";

fn getTimeComponents() TimeComponents {
    const now = timestamp();
    // Floor division + Euclidean remainder so pre-epoch timestamps
    // (now < 0) still wrap into 00..23 / 00..59 instead of truncating
    // toward zero (`@divTrunc(-1, 3600) == 0` → 00:00:00 for 23:59:59 UTC).
    const sph: i64 = @intCast(seconds_per_hour);
    const spm: i64 = @intCast(seconds_per_minute);
    const hpd: i64 = @intCast(hours_per_day);
    return .{
        .hours = @intCast(@mod(@divFloor(now, sph), hpd)),
        .minutes = @intCast(@mod(@divFloor(now, spm), spm)),
        .seconds = @intCast(@mod(now, spm)),
    };
}

/// Zero heap bytes that may hold prompts, messages, or secrets, then free.
fn wipeFree(allocator: Allocator, buf: []u8) void {
    @memset(buf, 0);
    allocator.free(buf);
}

/// Zero token-ID slices (derived from prompts) then free.
fn wipeFreeTokens(allocator: Allocator, ids: []u32) void {
    @memset(std.mem.sliceAsBytes(ids), 0);
    allocator.free(ids);
}

/// Sanitize a string for safe terminal output: replace control codepoints
/// (C0 below space except space, DEL, and the C1 range 0x80-0x9F) with '?', and
/// replace bytes that are not valid UTF-8 the same way. Decoding per codepoint
/// is what makes the C1 range reachable: its UTF-8 form (C2 80-C2 9F) is two
/// ordinary-looking bytes, so a byte filter passes it and the terminal reads it
/// as CSI. Prevents log injection via terminal escape sequences (CWE-117).
///
/// Clips on a character boundary, so a path longer than `buf` never leaves half
/// a character in the log. The output is always valid UTF-8.
fn sanitizeForLog(input: []const u8, buf: []u8) []const u8 {
    var i: usize = 0;
    var out: usize = 0;
    while (i < input.len) {
        const seq_len = utf8SequenceLen(input, i) orelse {
            if (out < buf.len) {
                buf[out] = '?';
                out += 1;
            }
            i += 1;
            continue;
        };
        const cp = std.unicode.utf8Decode(input[i..][0..seq_len]) catch unreachable;
        const is_control = (cp < 0x20 and cp != ' ') or cp == 0x7F or (cp >= 0x80 and cp <= 0x9F);
        if (is_control or out + seq_len > buf.len) {
            if (out < buf.len) {
                buf[out] = '?';
                out += 1;
            }
        } else {
            @memcpy(buf[out..][0..seq_len], input[i..][0..seq_len]);
            out += seq_len;
        }
        i += seq_len;
    }
    return buf[0..out];
}

/// Length of the complete UTF-8 sequence at `i`, or null when those bytes are
/// not one valid, fully present character.
fn utf8SequenceLen(s: []const u8, i: usize) ?usize {
    const seq_len = std.unicode.utf8ByteSequenceLength(s[i]) catch return null;
    if (i + seq_len > s.len) return null;
    _ = std.unicode.utf8Decode(s[i..][0..seq_len]) catch return null;
    return seq_len;
}

fn logRequest(method: []const u8, path: []const u8) void {
    const t = getTimeComponents();
    var method_buf: [16]u8 = undefined;
    var path_buf: [max_log_path_len]u8 = undefined;
    const safe_method = sanitizeForLog(method, &method_buf);
    const safe_path = sanitizeForLog(path, &path_buf);
    const rid = log_request_id;
    if (log_client_rid_len > 0) {
        slog(log_time_fmt ++ " req={d} xid={s} {s} {s}\n", .{ t.hours, t.minutes, t.seconds, rid, log_client_rid[0..log_client_rid_len], safe_method, safe_path });
    } else {
        slog(log_time_fmt ++ " req={d} {s} {s}\n", .{ t.hours, t.minutes, t.seconds, rid, safe_method, safe_path });
    }
}

/// Log completion of a request with status code and duration.
fn logRequestDone(method: []const u8, path: []const u8, status: u16, duration_ms: u64) void {
    const t = getTimeComponents();
    var method_buf: [16]u8 = undefined;
    var path_buf: [max_log_path_len]u8 = undefined;
    const safe_method = sanitizeForLog(method, &method_buf);
    const safe_path = sanitizeForLog(path, &path_buf);
    const rid = log_request_id;
    if (log_client_rid_len > 0) {
        slog(log_time_fmt ++ " req={d} xid={s} {s} {s} -> {d} ({d}ms)\n", .{ t.hours, t.minutes, t.seconds, rid, log_client_rid[0..log_client_rid_len], safe_method, safe_path, status, duration_ms });
    } else {
        slog(log_time_fmt ++ " req={d} {s} {s} -> {d} ({d}ms)\n", .{ t.hours, t.minutes, t.seconds, rid, safe_method, safe_path, status, duration_ms });
    }
}

fn logGeneration(tokens: u32, time_ms: u64, tps: f32) void {
    const t = getTimeComponents();
    const rid = log_request_id;
    if (std.c.isatty(stderr_file.handle) != 0) {
        slog(log_time_fmt ++ " req={d} \x1b[32mGenerated {d} tokens in {d}ms ({d:.2} tok/s)\x1b[0m\n", .{ t.hours, t.minutes, t.seconds, rid, tokens, time_ms, tps });
    } else {
        slog(log_time_fmt ++ " req={d} Generated {d} tokens in {d}ms ({d:.2} tok/s)\n", .{ t.hours, t.minutes, t.seconds, rid, tokens, time_ms, tps });
    }
}

/// Web UI HTML page, assembled at comptime from src/web/ files.
const html_page = @embedFile("../web/head.html") ++
    @embedFile("../web/style.css") ++
    @embedFile("../web/body.html") ++
    @embedFile("../web/app.js") ++
    "\n</script></body></html>\n";

/// Wyhash walks the embedded page in 48-byte rounds; the default comptime
/// branch quota (1000) is far below the ~400 KiB of head+css+body+app.js that
/// the React bundle and the compiled Tailwind stylesheet add up to.
const html_page_hash_eval_quota: u32 = 4_000_000;

const html_page_hash = blk: {
    @setEvalBranchQuota(html_page_hash_eval_quota);
    break :blk std.hash.Wyhash.hash(0, html_page);
};
const html_etag = std.fmt.comptimePrint("\"{x}\"", .{html_page_hash});
const html_etag_gzip = std.fmt.comptimePrint("\"{x}-gzip\"", .{html_page_hash});
/// Return the current thread's request ID (set at start of handleRequest).
/// Used for API response IDs so they match log correlation IDs.
fn currentRequestId() u64 {
    return log_request_id;
}

// ── HTTP helpers ────────────────────────────────────────────────

/// Parse `{"tokens":[1,2,3]}`-style body into `out`.
/// Outcome of parsing the `tokens` array of a `/v1/detokenize` body.
const DetokenizeTokens = union(enum) {
    /// `tokens` absent, not an array, or an empty array.
    empty,
    /// An element is not a non-negative integer, or there are more elements
    /// than `out` can hold. Decoding a prefix would silently drop the rest.
    malformed,
    count: usize,
};

/// Parse the `tokens` array into `out`. A malformed element or an oversized
/// array is reported rather than truncated, so the caller returns 400.
fn parseDetokenizeTokens(body: []const u8, out: []u32) DetokenizeTokens {
    var n_toks: usize = 0;
    if (std.mem.indexOf(u8, body, "\"tokens\"")) |ti| {
        var di = ti + "\"tokens\"".len;
        while (di < body.len and (body[di] == ' ' or body[di] == ':')) : (di += 1) {}
        if (di < body.len and body[di] == '[') {
            di += 1;
            while (di < body.len) {
                while (di < body.len and (body[di] == ' ' or body[di] == ',' or body[di] == '\n')) : (di += 1) {}
                if (di >= body.len or body[di] == ']') break;
                const num_start = di;
                while (di < body.len and body[di] >= '0' and body[di] <= '9') : (di += 1) {}
                if (di == num_start) return .malformed;
                if (n_toks >= out.len) return .malformed;
                out[n_toks] = std.fmt.parseInt(u32, body[num_start..di], 10) catch return .malformed;
                n_toks += 1;
            }
        }
    }
    return if (n_toks == 0) .empty else .{ .count = n_toks };
}

/// Result of parsing a required positive integer query parameter.
const PositiveQueryParam = union(enum) {
    missing,
    invalid,
    value: usize,
};

/// Parse `key` from a query string as a required positive integer.
/// Empty, zero, and non-numeric values are `.invalid` (not `.missing`).
fn parsePositiveQueryParam(query: []const u8, key: []const u8) PositiveQueryParam {
    const raw = http.extractQueryParam(query, key) orelse return .missing;
    const n = std.fmt.parseInt(usize, raw, 10) catch return .invalid;
    if (n == 0) return .invalid;
    return .{ .value = n };
}

/// Result of parsing a conversation `id` form field.
const FormConversationId = union(enum) {
    missing,
    invalid,
    value: u32,
};

/// Parse form-encoded `id` for `/v1/conversations` select/delete.
/// Conversation IDs start at 1; 0, empty, and non-numeric values are `.invalid`.
fn parseFormConversationId(body: []const u8) FormConversationId {
    const raw = json.extractFormField(body, "id") orelse return .missing;
    if (raw.len == 0) return .invalid;
    const id = std.fmt.parseInt(u32, raw, 10) catch return .invalid;
    if (id == 0) return .invalid;
    return .{ .value = id };
}

/// 304 for the chat UI document. No `Content-Length`: RFC 9110 15.4.5 lets a 304
/// carry one only when it equals the length the matching 200 would have sent,
/// and this response has no body to measure. `Vary` and `Cache-Control` repeat
/// so a cache that stored the 200 revalidates it under the same key.
const not_modified_headers = http.security_headers_base ++ http.document_cache_headers;

/// Validate Authorization header against configured API key.
/// Supports both OpenAI-style `Authorization: Bearer <key>` and
/// Anthropic-style `x-api-key: <key>` headers.
/// Returns true if no auth configured or if token matches.
/// Uses constant-time comparison to prevent timing side-channel attacks.
/// Iterates header lines (not substring search) to prevent false matches
/// inside other header values (CWE-287).
fn validateAuth(server: *const Server, headers: []const u8) bool {
    const key = server.api_key orelse return true; // No auth configured
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        const name = line[0..colon];
        // Authorization: Bearer <key>
        if (colon == "authorization".len and std.ascii.eqlIgnoreCase(name, "authorization")) {
            const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
            const bearer = "bearer ";
            if (val.len > bearer.len and std.ascii.eqlIgnoreCase(val[0..bearer.len], bearer)) {
                const token = std.mem.trim(u8, val[bearer.len..], " \t");
                if (constantTimeEql(token, key)) return true;
            }
        }
        // x-api-key: <key>
        if (colon == "x-api-key".len and std.ascii.eqlIgnoreCase(name, "x-api-key")) {
            const token = std.mem.trim(u8, line[colon + 1 ..], " \t");
            if (constantTimeEql(token, key)) return true;
        }
    }
    return false;
}

/// Authentication decision for one request. The dispatcher's only chokepoint:
/// handlers must not re-check the key, and a route is protected unless its
/// `known_endpoints` entry opts out.
fn authorizedForPath(server: *const Server, path: []const u8, headers: []const u8) bool {
    return switch (authPolicyFor(path)) {
        .required => validateAuth(server, headers),
        .optional, .public => true,
    };
}

/// Constant-time byte comparison to prevent timing side-channel attacks on secrets.
/// Always iterates over the secret length (b) to avoid leaking key length.
/// Accumulates XOR differences into a single byte, the compiler cannot
/// short-circuit because the final result depends on every iteration.
fn constantTimeEql(a: []const u8, b: []const u8) bool {
    var diff: u8 = if (a.len == b.len) 0 else 1;
    // Always iterate over the full secret length (b) to avoid
    // leaking key length through timing. When a is shorter,
    // pad with zero bytes (length mismatch already captured in diff).
    for (0..b.len) |i| {
        const a_byte = if (i < a.len) a[i] else 0;
        diff |= a_byte ^ b[i];
    }
    return diff == 0;
}

/// Check rate limit for the given prompt token count.
/// Returns null if allowed, or retry-after seconds if rate limited.
fn checkRateLimit(server: *Server, prompt_tokens: u32) ?u32 {
    const rl = server.rate_limiter orelse return null;
    return rl.tryConsumeOrRetryAfter(prompt_tokens);
}

/// Write a complete HTTP response (status line + headers + body).
fn sendResponse(stream: http.TcpStream, status: []const u8, content_type: []const u8, body: []const u8) void {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ status, content_type, body.len, log_request_id, http.corsHeaders() }) catch {
        std.log.warn("req={d} response header overflow (body={d})", .{ log_request_id, body.len });
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} response write failed (headers): {}", .{ log_request_id, err });
        return;
    };
    stream.writeAll(body) catch |err| {
        std.log.warn("req={d} response write failed (body, {d} bytes): {}", .{ log_request_id, body.len, err });
        return;
    };
}

/// Send a 200 OK HTTP response with `application/json` content type.
fn sendJson(stream: http.TcpStream, body: []const u8) void {
    sendResponse(stream, "200 OK", "application/json", body);
}

/// Send a 200 OK HTTP response with `text/html; charset=utf-8` content type.
fn sendHtml(stream: http.TcpStream, body: []const u8) void {
    sendResponse(stream, "200 OK", "text/html; charset=utf-8", body);
}

/// Resolve a repeated mutating request against the replay ledger.
/// Returns true when the response was sent and the caller must not run the
/// operation again; false when this request owns the key and should proceed.
fn resolveIdempotency(stream: http.TcpStream, method: []const u8, path: []const u8, request_start: i64) bool {
    switch (g_server.claimIdempotencyKey(path)) {
        .fresh => return false,
        .duplicate => {
            sendJsonErrorEx(stream, "409 Conflict", "invalid_request_error", "A request with this X-Request-Id is still in flight", null, "duplicate_request");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 409, elapsedMs(request_start));
        },
        .replay => |r| {
            // The body is this thread's copy: the ledger's slot can be freed by
            // another request's claim while these bytes are still on the wire.
            defer r.deinit(g_server.allocator);
            // A streamed or oversized response is recorded as completed
            // without a body: the operation stays suppressed, but there are
            // no original bytes to send back.
            if (r.body.len == 0) {
                sendJsonErrorEx(stream, "409 Conflict", "invalid_request_error", "This X-Request-Id already ran; its response is not replayable", null, "duplicate_request");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 409, elapsedMs(request_start));
            } else {
                sendIdempotentReplay(stream, r);
                logRequestDone(method, path, 200, elapsedMs(request_start));
            }
        },
    }
    return true;
}

/// Re-send the response recorded for a repeated `X-Request-Id`. The ledger
/// has already refused to run the operation a second time, so this is the
/// answer the first execution produced.
fn sendIdempotentReplay(stream: http.TcpStream, replay: Idempotency.Replay) void {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nIdempotent-Replay: true\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ replay.status_line, replay.content_type, replay.body.len, log_request_id, http.corsHeaders() }) catch {
        std.log.warn("req={d} replay header overflow (body={d})", .{ log_request_id, replay.body.len });
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} replay write failed (headers): {}", .{ log_request_id, err });
        return;
    };
    stream.writeAll(replay.body) catch |err| {
        std.log.warn("req={d} replay write failed (body, {d} bytes): {}", .{ log_request_id, replay.body.len, err });
    };
}

fn sendDocumentNotModified(stream: http.TcpStream, etag: []const u8) void {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 304 Not Modified\r\nETag: {s}\r\nX-Request-Id: {d}\r\n{s}" ++
        not_modified_headers ++
        "Connection: close\r\n\r\n", .{ etag, log_request_id, http.corsHeaders() }) catch {
        std.log.warn("req={d} 304 header overflow", .{log_request_id});
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} 304 write failed: {}", .{ log_request_id, err });
    };
}

/// Serve the embedded chat UI. gzip when the client accepts it; 304 when ETag matches.
/// API JSON still uses `security_headers` (`Cache-Control: no-store`).
fn sendHtmlPage(stream: http.TcpStream, req_headers: []const u8) u16 {
    const gzip_body = g_server.html_gzip;
    const use_gzip = gzip_body.len > 0 and http.acceptsEncoding(req_headers, "gzip");
    const etag: []const u8 = if (use_gzip) html_etag_gzip else html_etag;
    if (http.ifNoneMatch(req_headers, etag)) {
        sendDocumentNotModified(stream, etag);
        return 304;
    }
    const body: []const u8 = if (use_gzip) gzip_body else html_page;
    const encoding_hdr: []const u8 = if (use_gzip) "Content-Encoding: gzip\r\n" else "";
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {d}\r\n{s}ETag: {s}\r\nX-Request-Id: {d}\r\n{s}" ++
        http.security_headers_base ++ http.document_cache_headers ++
        "Connection: close\r\n\r\n", .{ body.len, encoding_hdr, etag, log_request_id, http.corsHeaders() }) catch {
        sendHtml(stream, html_page);
        return 200;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} HTML write failed (headers): {}", .{ log_request_id, err });
        return 200;
    };
    stream.writeAll(body) catch |err| {
        std.log.warn("req={d} HTML write failed (body, {d} bytes): {}", .{ log_request_id, body.len, err });
        return 200;
    };
    return 200;
}

/// True when this request should inject tools and parse <tool_call> output.
fn toolsWanted(tp: *const json.ToolParams) bool {
    if (std.mem.eql(u8, tp.tool_choice, "none")) return false;
    return tp.tool_count > 0 or g_server.tool_registry.count() > 0;
}

/// Build system prompt with tool definitions injected.
fn buildToolSystemPrompt(allocator: Allocator, tmpl: ChatTemplate, tp: *const json.ToolParams, existing_system: ?[]const u8, registry: *const tools_mod.Registry) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    if (existing_system) |sys| {
        try buf.appendSlice(allocator, sys);
        try buf.appendSlice(allocator, "\n\n");
    }
    try buf.appendSlice(allocator, "You have access to the following tools:\n\n");

    var req_tools: [tools_mod.max_tools]?tools_mod.Tool = .{null} ** tools_mod.max_tools;
    var req_n: u32 = 0;
    for (tp.tools) |maybe| {
        const t = maybe orelse continue;
        if (req_n >= req_tools.len) break;
        req_tools[req_n] = .{
            .name = t.name,
            .description = t.description,
            .parameters_json = t.parameters_json,
        };
        req_n += 1;
    }
    var merged: [tools_mod.max_tools]?tools_mod.Tool = .{null} ** tools_mod.max_tools;
    const n = registry.mergeInto(req_tools[0..req_n], &merged);

    for (merged[0..n]) |maybe| {
        const tool = maybe orelse continue;
        try buf.appendSlice(allocator, "- ");
        try tmpl.writeUntrusted(allocator, &buf, tool.name);
        if (tool.description.len > 0) {
            try buf.appendSlice(allocator, ": ");
            try tmpl.writeUntrusted(allocator, &buf, tool.description);
        }
        try buf.appendSlice(allocator, "\n  Parameters: ");
        try tmpl.writeUntrusted(allocator, &buf, tool.parameters_json);
        try buf.appendSlice(allocator, "\n");
    }
    try buf.appendSlice(allocator,
        \\
        \\To call a tool, output a JSON object wrapped in <tool_call> tags:
        \\<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
        \\You may call multiple tools.
    );
    if (std.mem.eql(u8, tp.tool_choice, "required")) {
        try buf.appendSlice(allocator, " You MUST call at least one tool.");
    } else {
        try buf.appendSlice(allocator, " Only call tools when needed.");
    }
    return buf.toOwnedSlice(allocator);
}

/// Parse tool calls from model output. Looks for <tool_call>...</tool_call> patterns.
fn hasToolCalls(text: []const u8) bool {
    return std.mem.indexOf(u8, text, "<tool_call>") != null;
}

/// Drop tool-call tag markup from assistant text that is returned as plain
/// content. The tags are a control channel: when no declared tool parses, the
/// payload is unvalidated model output, and leaving the tags in place hands a
/// client that scans for them the same text the server just refused to trust.
/// Returns the input slice when there is no tag to remove.
fn stripToolCallTags(allocator: Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
    const tc_start_tag = "<tool_call>";
    const tc_end_tag = "</tool_call>";
    if (std.mem.indexOf(u8, text, tc_start_tag) == null) return text;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < text.len) {
        const start = std.mem.indexOfPos(u8, text, i, tc_start_tag) orelse {
            try out.appendSlice(allocator, text[i..]);
            break;
        };
        try out.appendSlice(allocator, text[i..start]);
        const body = start + tc_start_tag.len;
        const end = std.mem.indexOfPos(u8, text, body, tc_end_tag) orelse {
            i = body;
            continue;
        };
        i = end + tc_end_tag.len;
    }
    return out.toOwnedSlice(allocator) catch text;
}

fn toolCallNameAllowed(name: []const u8, tp: *const json.ToolParams, registry: *const tools_mod.Registry) bool {
    return tp.hasTool(name) or registry.hasName(name);
}

/// Advance `search_pos` through `text` to the next `<tool_call>` payload whose
/// `name` is in the request tools or the process registry. Returns the JSON
/// between the tags, or null when none remain. Undeclared names are skipped.
fn nextAllowedToolCall(text: []const u8, search_pos: *usize, tp: *const json.ToolParams, registry: *const tools_mod.Registry) ?[]const u8 {
    const tc_start_tag = "<tool_call>";
    const tc_end_tag = "</tool_call>";
    while (search_pos.* < text.len) {
        const tc_start = std.mem.indexOfPos(u8, text, search_pos.*, tc_start_tag) orelse return null;
        const json_start = tc_start + tc_start_tag.len;
        const tc_end = std.mem.indexOfPos(u8, text, json_start, tc_end_tag) orelse return null;
        const tc_json = text[json_start..tc_end];
        search_pos.* = tc_end + tc_end_tag.len;
        const name = json.extractField(tc_json, "name") orelse continue;
        if (!toolCallNameAllowed(name, tp, registry)) {
            if (!@import("builtin").is_test) {
                std.log.warn("req={d} dropping tool call with undeclared name", .{log_request_id});
            }
            continue;
        }
        return tc_json;
    }
    return null;
}

/// Split generated text into (reasoning, content) parts.
/// Detects <think>...</think> (DeepSeek R1, QwQ) and similar markers.
/// Returns reasoning slice and content slice, both reference the original text (no alloc).
const ThinkingSplit = struct { reasoning: []const u8, content: []const u8 };
fn splitThinkingContent(text: []const u8) ThinkingSplit {
    // Pattern: <think>REASONING</think>CONTENT
    const think_open = "<think>";
    const think_close = "</think>";
    if (std.mem.startsWith(u8, text, think_open)) {
        if (std.mem.indexOf(u8, text, think_close)) |end| {
            const reasoning = text[think_open.len..end];
            const content = blk2: {
                var s = text[end + think_close.len ..];
                while (s.len > 0 and (s[0] == 32 or s[0] == 9 or s[0] == 13 or s[0] == 10)) s = s[1..];
                break :blk2 s;
            };
            return .{ .reasoning = reasoning, .content = content };
        }
    }
    // Pattern: text contains <think>...</think> anywhere
    if (std.mem.indexOf(u8, text, think_open)) |start| {
        if (std.mem.indexOf(u8, text[start..], think_close)) |rel_end| {
            const reasoning = text[start + think_open.len .. start + rel_end];
            const after = text[start + rel_end + think_close.len ..];
            const content = blk2: {
                var s = after;
                while (s.len > 0 and (s[0] == 32 or s[0] == 9 or s[0] == 13 or s[0] == 10)) s = s[1..];
                break :blk2 s;
            };
            return .{ .reasoning = reasoning, .content = if (content.len > 0) content else text[0..start] };
        }
    }
    return .{ .reasoning = "", .content = text };
}

/// Build an Anthropic Messages API response from model output containing
/// <tool_call> tags. Emits one `tool_use` content block per parsed call with
/// stop_reason "tool_use". `input` must be a JSON object per the Anthropic
/// spec: object arguments are embedded verbatim (structurally validated by the
/// extractor); string arguments are unwrapped when they decode to an object,
/// otherwise replaced with `{}` and logged, an invalid `input` would break
/// spec-compliant clients. Returns "" when no <tool_call> payload parses;
/// callers fall back to a plain text response.
fn buildAnthropicToolCallResponse(buf: []u8, raw_text: []const u8, req_id: u64, prompt_tokens: u32, completion_tokens: u32, tp: *const json.ToolParams) []const u8 {
    var blocks_buf: [4096]u8 = undefined;
    var blocks_pos: usize = 0;
    var search_pos: usize = 0;
    var call_idx: usize = 0;

    while (nextAllowedToolCall(raw_text, &search_pos, tp, &g_server.tool_registry)) |tc_json| {
        const name = json.extractField(tc_json, "name") orelse continue;

        // Resolve input as a JSON object text (see resolveAnthropicToolInput).
        const resolved = resolveAnthropicToolInput(g_server.allocator, tc_json, call_idx);
        defer if (resolved.owned) |p| g_server.allocator.free(p);
        const args_obj = resolved.obj;

        // Escape name only, args are embedded as raw JSON (validated above).
        const escaped_name = json.jsonEscape(g_server.allocator, name) catch {
            std.log.warn("req={d} tool call name escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_name.ptr != name.ptr) g_server.allocator.free(escaped_name);

        const prefix: []const u8 = if (call_idx > 0) "," else "";
        const entry = std.fmt.bufPrint(blocks_buf[blocks_pos..], "{s}" ++
            \\{{"type":"tool_use","id":"toolu_{d}_{d}","name":"{s}","input":{s}}}
        , .{ prefix, req_id, call_idx, escaped_name, args_obj }) catch {
            std.log.warn("req={d} anthropic tool call response exceeded {d} byte buffer: dropped calls from index {d}", .{ log_request_id, blocks_buf.len, call_idx });
            break;
        };
        blocks_pos += entry.len;
        call_idx += 1;
    }

    if (call_idx == 0) return "";

    return std.fmt.bufPrint(buf,
        \\{{"id":"msg_{d}","type":"message","role":"assistant","content":[{s}],"model":"{s}","stop_reason":"tool_use","stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{ req_id, blocks_buf[0..blocks_pos], g_server.model_name, prompt_tokens, completion_tokens }) catch "";
}

/// Build tool_calls JSON response from model output containing <tool_call> tags.
/// Supports multiple tool calls. Arguments are JSON-escaped strings per OpenAI spec.
fn buildToolCallResponse(buf: []u8, raw_text: []const u8, req_id: u64, created: i64, prompt_tokens: u32, completion_tokens: u32, tp: *const json.ToolParams) []const u8 {
    const total = prompt_tokens + completion_tokens;

    // Build tool_calls array entries
    var tc_buf: [4096]u8 = undefined;
    var tc_pos: usize = 0;
    var search_pos: usize = 0;
    var call_idx: usize = 0;

    while (nextAllowedToolCall(raw_text, &search_pos, tp, &g_server.tool_registry)) |tc_json| {
        const name = json.extractField(tc_json, "name") orelse continue;
        const args = json.extractObjectField(tc_json, "arguments") orelse
            (json.extractField(tc_json, "arguments") orelse "{}");

        // Escape name and args, model output is untrusted (CWE-116).
        const escaped_name = json.jsonEscape(g_server.allocator, name) catch {
            std.log.warn("req={d} tool call name escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_name.ptr != name.ptr) g_server.allocator.free(escaped_name);
        const escaped_args = json.jsonEscape(g_server.allocator, args) catch {
            std.log.warn("req={d} tool call argument escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_args.ptr != args.ptr) g_server.allocator.free(escaped_args);

        const prefix: []const u8 = if (call_idx > 0) "," else "";
        const entry = std.fmt.bufPrint(tc_buf[tc_pos..], "{s}" ++
            \\{{"id":"call_{d}_{d}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
        , .{ prefix, req_id, call_idx, escaped_name, escaped_args }) catch {
            // Buffer full: remaining <tool_call> tags are dropped. Log loudly so
            // lost calls are diagnosable, the client still sees finish_reason
            // "tool_calls" for the calls that fit.
            std.log.warn("req={d} tool call response exceeded {d} byte buffer: dropped calls from index {d}", .{ log_request_id, tc_buf.len, call_idx });
            break;
        };
        tc_pos += entry.len;
        call_idx += 1;
    }

    if (call_idx == 0) return "";

    return std.fmt.bufPrint(buf,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":[{s}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{ req_id, created, g_server.model_name, system_fingerprint, tc_buf[0..tc_pos], prompt_tokens, completion_tokens, total }) catch "";
}

/// Build a Responses API payload from model output containing tool-call tags.
/// Emits one `function_call` output item per parsed call. Names are checked
/// against the request tools and the process registry, and both the name and
/// the arguments are JSON-escaped, so model output cannot inject extra output
/// items or fields. Returns "" when no tool-call payload parses; callers fall
/// back to a plain message item.
fn buildResponsesToolCallResponse(buf: []u8, raw_text: []const u8, req_id: u64, created: i64, prompt_tokens: u32, completion_tokens: u32, tp: *const json.ToolParams) []const u8 {
    var fc_buf: [4096]u8 = undefined;
    var fc_pos: usize = 0;
    var search_pos: usize = 0;
    var call_idx: usize = 0;
    const total = prompt_tokens + completion_tokens;

    while (nextAllowedToolCall(raw_text, &search_pos, tp, &g_server.tool_registry)) |tc_json| {
        const name = json.extractField(tc_json, "name") orelse continue;
        const args = json.extractObjectField(tc_json, "arguments") orelse
            (json.extractField(tc_json, "arguments") orelse "{}");
        // Escape name and args, model output is untrusted (CWE-116).
        const escaped_name = json.jsonEscape(g_server.allocator, name) catch {
            std.log.warn("req={d} tool call name escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_name.ptr != name.ptr) g_server.allocator.free(escaped_name);
        const escaped_args = json.jsonEscape(g_server.allocator, args) catch {
            std.log.warn("req={d} tool call argument escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_args.ptr != args.ptr) g_server.allocator.free(escaped_args);

        const prefix: []const u8 = if (call_idx > 0) "," else "";
        const entry = std.fmt.bufPrint(fc_buf[fc_pos..], "{s}" ++
            \\{{"type":"function_call","id":"fc_{d}_{d}","call_id":"call_{d}_{d}","name":"{s}","arguments":"{s}","status":"completed"}}
        , .{ prefix, req_id, call_idx, req_id, call_idx, escaped_name, escaped_args }) catch {
            std.log.warn("req={d} responses function_call exceeded {d} byte buffer: dropped calls from index {d}", .{ log_request_id, fc_buf.len, call_idx });
            break;
        };
        fc_pos += entry.len;
        call_idx += 1;
    }

    if (call_idx == 0) return "";

    return std.fmt.bufPrint(buf,
        \\{{"id":"resp-{d}","object":"response","created_at":{d},"status":"completed","model":"{s}","output":[{s}],"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}}}}
    , .{ req_id, created, g_server.model_name, fc_buf[0..fc_pos], prompt_tokens, completion_tokens, total }) catch "";
}

const openai_error_fallback = "{\"error\":{\"message\":\"Internal error\",\"type\":\"server_error\",\"param\":null,\"code\":null}}";

/// Send a JSON error response in OpenAI format. Escapes message and type to prevent injection (CWE-116).
fn sendJsonError(stream: http.TcpStream, status: []const u8, err_type: []const u8, message: []const u8) void {
    sendJsonErrorEx(stream, status, err_type, message, null, null);
}

/// True for static error `param`/`code` tokens (alnum, `_`, `-` only).
fn isSafeErrorToken(s: []const u8) bool {
    if (s.len == 0 or s.len > 40) return false;
    for (s) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    }
    return true;
}

/// Like `sendJsonError`, with optional `param` (field/query name) and `code` (machine-readable).
/// `param` and `code` must be static ASCII identifiers (not client-controlled).
fn sendJsonErrorEx(stream: http.TcpStream, status: []const u8, err_type: []const u8, message: []const u8, param: ?[]const u8, code: ?[]const u8) void {
    // Never fall back to unescaped input on OOM, that reintroduces injection.
    const escaped_msg = json.jsonEscape(g_server.allocator, message) catch {
        sendResponse(stream, status, "application/json", openai_error_fallback);
        return;
    };
    defer if (escaped_msg.ptr != message.ptr) g_server.allocator.free(escaped_msg);
    const escaped_type = json.jsonEscape(g_server.allocator, err_type) catch {
        sendResponse(stream, status, "application/json", openai_error_fallback);
        return;
    };
    defer if (escaped_type.ptr != err_type.ptr) g_server.allocator.free(escaped_type);
    var param_buf: [48]u8 = undefined;
    const param_json: []const u8 = if (param) |p| blk: {
        if (!isSafeErrorToken(p)) break :blk "null";
        break :blk (std.fmt.bufPrint(&param_buf, "\"{s}\"", .{p}) catch "null");
    } else "null";
    var code_buf: [48]u8 = undefined;
    const code_json: []const u8 = if (code) |c| blk: {
        if (!isSafeErrorToken(c)) break :blk "null";
        break :blk (std.fmt.bufPrint(&code_buf, "\"{s}\"", .{c}) catch "null");
    } else "null";
    var buf: [error_body_buf_size]u8 = undefined;
    const json_body = std.fmt.bufPrint(&buf,
        \\{{"error":{{"message":"{s}","type":"{s}","param":{s},"code":{s}}}}}
    , .{ escaped_msg, escaped_type, param_json, code_json }) catch {
        std.log.warn("req={d} error body overflow type={s}", .{ log_request_id, err_type });
        // Always respond, a hung client is worse than a generic body.
        sendResponse(stream, status, "application/json", openai_error_fallback);
        return;
    };
    sendResponse(stream, status, "application/json", json_body);
}

/// Send an error raised before routing, in the envelope of the route the
/// request addressed. A `/v1/messages` client parses the Anthropic shape only,
/// so the OpenAI body is an unparseable response there. `code` is the
/// OpenAI-envelope machine-readable code; the Anthropic envelope has no code
/// field, so only the message survives.
fn sendPreflightErrorEx(stream: http.TcpStream, path: []const u8, status: []const u8, code: []const u8, message: []const u8) void {
    if (isAnthropicPath(path)) {
        sendAnthropicError(stream, if (std.mem.eql(u8, status, "403 Forbidden")) "403" else status, "invalid_request_error", message);
        return;
    }
    sendJsonErrorEx(stream, status, "invalid_request_error", message, null, code);
}

/// Send 401 Unauthorized response for invalid API key.
fn send401(stream: http.TcpStream) void {
    g_server.metrics.recordAuthFailure();
    std.log.warn("req={d} authentication failed", .{log_request_id});
    const body = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\",\"param\":null,\"code\":\"invalid_api_key\"}}";
    sendResponse(stream, "401 Unauthorized", "application/json", body);
}

/// Write SSE response headers including X-Request-Id for log correlation.
/// Returns false if the write failed (client disconnected).
fn sendSseHeaders(stream: http.TcpStream) bool {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    // Cache-Control comes only from security_headers (no-store). Emitting a second
    // Cache-Control: no-cache here produced duplicate headers and ambiguous caching.
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Accel-Buffering: no\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: keep-alive\r\n\r\n", .{ log_request_id, http.corsHeaders() }) catch {
        std.log.warn("req={d} SSE header overflow ({d} bytes available)", .{ log_request_id, hdr_buf_size });
        return false;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} SSE header write failed: {}", .{ log_request_id, err });
        return false;
    };
    return true;
}

const rate_limit_fallback = "{\"error\":{\"message\":\"Rate limit exceeded\",\"type\":\"rate_limit_exceeded\",\"param\":null,\"code\":\"rate_limit_exceeded\"}}";

/// Send 429 Too Many Requests with Retry-After header.
fn send429(stream: http.TcpStream, retry_after: u32) void {
    g_server.metrics.recordRateLimit();
    std.log.warn("req={d} rate limited (retry_after={d}s)", .{ log_request_id, retry_after });
    var buf: [error_body_buf_size]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"Rate limit exceeded. Retry after {d} seconds.\",\"type\":\"rate_limit_exceeded\",\"param\":null,\"code\":\"rate_limit_exceeded\"}}}}", .{retry_after}) catch rate_limit_fallback;
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after, log_request_id, http.corsHeaders() }) catch {
        // Always respond, a hung client is worse than a response without Retry-After.
        std.log.warn("req={d} 429 header format failed, using fallback", .{log_request_id});
        sendResponse(stream, "429 Too Many Requests", "application/json", body);
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} 429 write failed (headers): {}", .{ log_request_id, err });
        return;
    };
    stream.writeAll(body) catch |err| {
        std.log.warn("req={d} 429 write failed (body): {}", .{ log_request_id, err });
        return;
    };
}

/// Send 503 with Retry-After (connection capacity / spawn failure).
fn send503Retry(stream: http.TcpStream, body: []const u8, retry_after: u32) void {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after, log_request_id, http.corsHeaders() }) catch {
        std.log.warn("req={d} 503 header format failed, using fallback", .{log_request_id});
        sendResponse(stream, "503 Service Unavailable", "application/json", body);
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} 503 write failed (headers): {}", .{ log_request_id, err });
        return;
    };
    stream.writeAll(body) catch |err| {
        std.log.warn("req={d} 503 write failed (body): {}", .{ log_request_id, err });
        return;
    };
}

// ── Request handler ─────────────────────────────────────────────

/// Snapshot of liveness/readiness inputs. Shared by `/health`, `/ready`,
/// `/metrics` (`agave_ready`), and health-transition logs so the formula
/// cannot drift across those surfaces.
const HealthView = struct {
    shutting_down: bool,
    kv_pressure: bool,
    high_error_rate: bool,
    kv_used: u32,
    kv_total: u32,
    queue: u32,
    completed: u64,
    failed: u64,
    cancelled: u64,
    sched_errs: u64,
    kv_demotions: u64,
    sleeping: bool,

    fn state(self: HealthView) HealthState {
        if (self.shutting_down) return .shutting_down;
        if (self.kv_pressure and self.high_error_rate) return .kv_pressure_and_errors;
        if (self.kv_pressure) return .kv_pressure;
        if (self.high_error_rate) return .high_error_rate;
        return .ok;
    }

    fn ready(self: HealthView) bool {
        return self.state() == .ok;
    }

    fn statusStr(self: HealthView) []const u8 {
        return switch (self.state()) {
            .ok => "ok",
            .shutting_down => "shutting_down",
            else => "degraded",
        };
    }

    fn reasonStr(self: HealthView) []const u8 {
        return switch (self.state()) {
            .ok => "none",
            .kv_pressure => "kv_pressure",
            .high_error_rate => "high_error_rate",
            .kv_pressure_and_errors => "kv_pressure,high_error_rate",
            .shutting_down => "shutting_down",
        };
    }
};

fn loadHealthView() HealthView {
    const m = &g_server.metrics;
    const kv_used = m.kv_blocks_used.load(.monotonic);
    const kv_total = m.kv_blocks_total.load(.monotonic);
    const completed = m.requests_completed.load(.monotonic);
    const failed = m.requests_failed.load(.monotonic);
    const settled: u64 = @as(u64, completed) + failed;
    return .{
        .shutting_down = g_server.shutdown_requested.load(.acquire),
        // Widen before the multiply: kv_total is a u32 block count, so kv_used * 100
        // wraps past 2^32 and reports "no pressure" on a full cache. failed needs
        // the same widening, and the sum needs it to keep settled non-zero.
        .kv_pressure = kv_total > 0 and @as(u64, kv_used) * 100 / kv_total >= kv_cache_degradation_pct,
        .high_error_rate = settled >= error_rate_min_requests and failed * 100 / settled >= error_rate_degradation_pct,
        .kv_used = kv_used,
        .kv_total = kv_total,
        .queue = m.queue_depth.load(.monotonic),
        .completed = completed,
        .failed = failed,
        .cancelled = m.requests_cancelled.load(.monotonic),
        .sched_errs = m.scheduler_errors.load(.monotonic),
        .kv_demotions = m.kv_demotions_vram_to_ram.load(.monotonic) + m.kv_demotions_ram_to_ssd.load(.monotonic),
        .sleeping = g_server.sleeping.load(.acquire),
    };
}

/// Log once when `/health`/`/ready` state changes (ok ↔ degraded/shutdown).
fn noteHealthTransition(view: HealthView) void {
    const next = @intFromEnum(view.state());
    const prev = g_server.health_state.swap(next, .acq_rel);
    if (prev == next) return;
    const prev_state: HealthState = if (prev <= @intFromEnum(HealthState.shutting_down))
        @enumFromInt(prev)
    else
        .ok;
    if (view.state() == .ok) {
        std.log.info("server: health recovered ({s} -> ok, kv={d}/{d} errors={d}/{d})", .{
            @tagName(prev_state), view.kv_used, view.kv_total, view.failed, view.completed + view.failed,
        });
    } else {
        std.log.warn("server: health {s} (was {s}, kv={d}/{d} errors={d}/{d} queue={d})", .{
            @tagName(view.state()), @tagName(prev_state), view.kv_used, view.kv_total, view.failed, view.completed + view.failed, view.queue,
        });
    }
}

const ImageReject = enum { none, missing_vision, decode_failed };

/// Cheap image-presence check used before generation. Decode still happens in
/// `processVisionImage`; this only rejects requests that would otherwise drop
/// the image on the floor (no encoder, or not a data URI).
fn classifyRequestImage(body: []const u8, form: bool) ImageReject {
    if (form) {
        if (json.extractFormField(body, "image") == null) return .none;
        if (g_server.vision_encoder == null) return .missing_vision;
        if (json.extractFormImage(body) == null) return .decode_failed;
        return .none;
    }
    if (!json.hasJsonImagePart(body)) return .none;
    if (g_server.vision_encoder == null) return .missing_vision;
    if (json.extractJsonImage(body) == null) return .decode_failed;
    return .none;
}

/// Send 400 for an unusable image attachment. Returns true when the caller
/// should return (response already written).
fn rejectBadImage(
    stream: http.TcpStream,
    body: []const u8,
    form: bool,
    anthropic: bool,
    request_start: i64,
    method: []const u8,
    path: []const u8,
) bool {
    const kind = classifyRequestImage(body, form);
    if (kind == .none) return false;
    const msg: []const u8 = switch (kind) {
        .none => unreachable,
        .missing_vision => "This model does not support vision",
        .decode_failed => "Failed to decode or encode attached image",
    };
    const code: []const u8 = switch (kind) {
        .none => unreachable,
        .missing_vision => "vision_not_supported",
        .decode_failed => "image_decode_failed",
    };
    if (anthropic) {
        sendAnthropicError(stream, "400", "invalid_request_error", msg);
    } else {
        sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", msg, "image", code);
    }
    g_server.metrics.recordClientError();
    logRequestDone(method, path, 400, elapsedMs(request_start));
    return true;
}

/// Main HTTP request dispatcher. Wakes the server from sleep mode if needed,
/// enforces CORS policy and authentication, then routes the request by method
/// and path to the appropriate endpoint handler (health, chat completions,
/// models, metrics, etc.).
fn handleRequest(stream: http.TcpStream, req: http.HttpRequest) void {
    const request_start = milliTimestamp();
    captureClientRequestId(req.headers);
    // Wake from sleep mode on any incoming request. Mutex serializes with
    // sleepMonitorLoop so a late sleep store cannot overwrite this wake.
    var woke_from_sleep = false;
    {
        g_server.mutex.lockUncancelable(g_server.io);
        defer g_server.mutex.unlock(g_server.io);
        g_server.last_request_ms.store(request_start, .release);
        if (g_server.sleeping.swap(false, .acq_rel)) {
            woke_from_sleep = true;
        }
    }
    if (woke_from_sleep) {
        g_server.metrics.updateSleeping(false);
        std.log.info("server: waking from sleep mode", .{});
    }
    const path = req.path;
    const method = req.method;
    const is_get = std.mem.eql(u8, method, "GET");
    const is_post = std.mem.eql(u8, method, "POST");

    // Unauthenticated servers only accept loopback Host. A public name that
    // DNS-rebinds to 127.0.0.1 would pass Origin==Host and skip the CSRF check.
    if (isRebindHostUnauthenticated(req.headers)) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        g_server.metrics.recordClientError();
        std.log.warn("req={d} non-loopback Host rejected (no API key)", .{log_request_id});
        sendPreflightErrorEx(stream, path, "403 Forbidden", "host_forbidden", "Host not allowed");
        logRequestDone(method, path, 403, elapsedMs(request_start));
        return;
    }

    // Block browser cross-origin calls when running without an API key so a
    // malicious page cannot drive inference or read conversation state on a
    // loopback --serve (CWE-352 / CWE-942).
    if (isCrossOriginUnauthenticated(req.headers)) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        g_server.metrics.recordClientError();
        std.log.warn("req={d} cross-origin request rejected (no API key)", .{log_request_id});
        sendPreflightErrorEx(stream, path, "403 Forbidden", "cross_origin_forbidden", "Cross-origin request rejected");
        logRequestDone(method, path, 403, elapsedMs(request_start));
        return;
    }

    // CORS preflight, return path-specific allowed methods
    if (std.mem.eql(u8, method, "OPTIONS")) {
        var allow_methods: []const u8 = "GET, POST, OPTIONS";
        for (known_endpoints) |ep| {
            if (std.mem.eql(u8, path, ep.path)) {
                allow_methods = ep.allow;
                break;
            }
        }
        var opts_buf: [hdr_buf_size]u8 = undefined;
        const opts_hdr = std.fmt.bufPrint(&opts_buf, "HTTP/1.1 204 No Content\r\n" ++
            "{s}" ++
            "Access-Control-Allow-Methods: {s}\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version\r\n" ++
            "Access-Control-Max-Age: " ++ http.cors_max_age_seconds ++ "\r\n" ++
            "X-Request-Id: {d}\r\n" ++
            http.security_headers ++
            "Content-Length: 0\r\n" ++
            "Connection: close\r\n\r\n", .{ http.corsHeaders(), allow_methods, log_request_id }) catch return;
        stream.writeAll(opts_hdr) catch return;
        return;
    }

    // Single authentication chokepoint. Every route is answered here first, so
    // a handler cannot be reachable without the key by omitting a check, and an
    // unknown path is indistinguishable from a known one to an unauthenticated
    // caller (401 before the 405 and 404 handlers run).
    if (!authorizedForPath(g_server, path, req.headers)) {
        logRequest(method, path);
        if (isAnthropicPath(path)) {
            // Anthropic clients need their error envelope; send401 emits the
            // OpenAI one, so the metric and log are recorded here.
            g_server.metrics.recordAuthFailure();
            std.log.warn("req={d} authentication failed", .{log_request_id});
            sendAnthropicError(stream, "401", "authentication_error", "Invalid API key");
        } else {
            send401(stream);
        }
        logRequestDone(method, path, 401, elapsedMs(request_start));
        return;
    }

    // Health check endpoint, lightweight, no mutex, no inference
    if (is_get and std.mem.eql(u8, path, "/health")) {
        var buf: [health_buf_size]u8 = undefined;
        const uptime: i64 = uptimeSeconds(g_server.start_mono_ms, milliTimestamp());
        const hv = loadHealthView();
        noteHealthTransition(hv);
        const status = hv.statusStr();
        const reason = hv.reasonStr();
        const http_status: []const u8 = if (hv.shutting_down) "503 Service Unavailable" else "200 OK";
        // When API key is configured and auth is not provided, return only
        // liveness status, omit model/version/backend to prevent fingerprinting.
        if (g_server.api_key != null and !validateAuth(g_server, req.headers)) {
            const minimal = std.fmt.bufPrint(&buf,
                \\{{"status":"{s}","reason":"{s}"}}
            , .{ status, reason }) catch return;
            sendResponse(stream, http_status, "application/json", minimal);
            return;
        }
        const kv_seq_len = kvSeqLenSnapshot();
        const json_body = std.fmt.bufPrint(&buf,
            \\{{"status":"{s}","reason":"{s}","version":"{s}","model":"{s}","backend":"{s}","uptime_s":{d},"active_connections":{d},"requests_total":{d},"requests_completed":{d},"requests_failed":{d},"requests_cancelled":{d},"queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d},"kv_seq_len":{d},"ctx_size":{d},"scheduler_errors":{d},"kv_demotions":{d},"sleeping":{s}}}
        , .{ status, reason, engine_version, g_server.model_name, g_server.backend_name, uptime, g_server.metrics.active_connections.load(.monotonic), g_server.metrics.requests_total.load(.monotonic), hv.completed, hv.failed, hv.cancelled, hv.queue, hv.kv_used, hv.kv_total, kv_seq_len, g_server.ctx_size, hv.sched_errs, hv.kv_demotions, if (hv.sleeping) "true" else "false" }) catch
            std.fmt.bufPrint(&buf, "{{\"status\":\"{s}\"}}", .{status}) catch return;
        sendResponse(stream, http_status, "application/json", json_body);
        return;
    }

    // Readiness check endpoint, returns 503 if shutting down, under KV cache pressure, or high error rate
    if (is_get and std.mem.eql(u8, path, "/ready")) {
        const hv = loadHealthView();
        noteHealthTransition(hv);
        // Unauthenticated probes omit queue/KV details; degraded still includes `reason`.
        const authed = g_server.api_key == null or validateAuth(g_server, req.headers);
        if (hv.shutting_down) {
            if (authed) {
                var sbuf: [health_buf_size]u8 = undefined;
                const sjson = std.fmt.bufPrint(&sbuf,
                    \\{{"status":"shutting_down","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ hv.queue, hv.kv_used, hv.kv_total }) catch "{\"status\":\"shutting_down\"}";
                sendResponse(stream, "503 Service Unavailable", "application/json", sjson);
            } else {
                sendResponse(stream, "503 Service Unavailable", "application/json", "{\"status\":\"shutting_down\"}");
            }
        } else if (!hv.ready()) {
            if (authed) {
                var rbuf: [health_buf_size]u8 = undefined;
                const rjson = std.fmt.bufPrint(&rbuf,
                    \\{{"status":"degraded","reason":"{s}","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ hv.reasonStr(), hv.queue, hv.kv_used, hv.kv_total }) catch "{\"status\":\"degraded\"}";
                sendResponse(stream, "503 Service Unavailable", "application/json", rjson);
            } else {
                var rbuf: [health_buf_size]u8 = undefined;
                const rjson = std.fmt.bufPrint(&rbuf,
                    \\{{"status":"degraded","reason":"{s}"}}
                , .{hv.reasonStr()}) catch "{\"status\":\"degraded\"}";
                sendResponse(stream, "503 Service Unavailable", "application/json", rjson);
            }
        } else {
            if (authed) {
                var rbuf: [health_buf_size]u8 = undefined;
                const rjson = std.fmt.bufPrint(&rbuf,
                    \\{{"status":"ready","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ hv.queue, hv.kv_used, hv.kv_total }) catch "{\"status\":\"ready\"}";
                sendJson(stream, rjson);
            } else {
                sendJson(stream, "{\"status\":\"ready\"}");
            }
        }
        return;
    }

    // Prometheus metrics endpoint (requires auth when API key configured)
    if (is_get and std.mem.eql(u8, path, "/metrics")) {
        logRequest(method, path);
        var buf: [metrics_render_buf_size]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&buf);
        g_server.metrics.renderPrometheus(&writer) catch {
            std.log.err("req={d} metrics render failed: buffer overflow ({d} bytes available)", .{ log_request_id, metrics_render_buf_size });
            g_server.metrics.recordFailure();
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Metrics rendering failed");
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        // Build info metric, standard Prometheus pattern for version tracking
        writer.print("# HELP agave_build_info Agave server version and configuration\n# TYPE agave_build_info gauge\nagave_build_info{{version=\"{s}\",backend=\"{s}\",language=\"zig\"}} 1\n", .{ engine_version, g_server.backend_name }) catch {
            std.log.warn("req={d} metrics buffer overflow: build_info metric truncated ({d} bytes available)", .{ log_request_id, metrics_render_buf_size });
        };
        // Ready gauge: same conditions as GET /ready so Prometheus alerts do
        // not have to reimplement /health JSON (agave_up is liveness only).
        const hv = loadHealthView();
        noteHealthTransition(hv);
        writer.print("# HELP agave_ready 1 when /ready would return 200, 0 when degraded or shutting down\n# TYPE agave_ready gauge\nagave_ready {d}\n", .{@as(u32, if (hv.ready()) 1 else 0)}) catch {
            std.log.warn("req={d} metrics buffer overflow: agave_ready metric truncated ({d} bytes available)", .{ log_request_id, metrics_render_buf_size });
        };
        sendResponse(stream, "200 OK", "text/plain; version=0.0.4; charset=utf-8", writer.buffered());
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/favicon.ico")) {
        var fav_buf: [short_hdr_buf_size]u8 = undefined;
        const fav_hdr = std.fmt.bufPrint(&fav_buf, "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ log_request_id, http.corsHeaders() }) catch return;
        stream.writeAll(fav_hdr) catch return;
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/")) {
        logRequest(method, path);
        const ui_status = sendHtmlPage(stream, req.headers);
        logRequestDone(method, path, ui_status, elapsedMs(request_start));
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/v1/models")) {
        logRequest(method, path);

        g_server.metrics.recordRequest();

        var buf: [models_json_buf_size]u8 = undefined;
        const kv_pos = kvSeqLenSnapshot();
        const has_vision = g_server.vision_encoder != null;
        const mtp_depth = g_server.model.getMtpDepth();
        const json_body = std.fmt.bufPrint(&buf,
            \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"agave","backend":"{s}","kv_seq_len":{d},"ctx_size":{d},"n_layers":{d},"n_embd":{d},"vocab_size":{d},"vision":{s},"mtp_depth":{d}}}]}}
        , .{ g_server.model_name, g_server.start_time, g_server.backend_name, kv_pos, g_server.ctx_size, g_server.model.nLayers(), g_server.model.nEmbd(), g_server.model.vocabSize(), if (has_vision) "true" else "false", mtp_depth }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json_body);
        g_server.metrics.recordCompletion();
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/chat/completions")) {
        logRequest(method, path);
        const req_start_time = milliTimestamp();

        // 1. Validate authentication
        g_server.metrics.recordRequest();

        const body = req.body;

        // Reject n > 1 (multiple completions not supported)
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available", "n", "n_not_supported");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const max_tokens = extractMaxGenTokens(body, &.{ "max_tokens", "max_completion_tokens" });
        var sampling = json.SamplingParams{};
        json.parseSampling(&sampling, body);

        // Do not log sampling.user, OpenAI "user" field often holds email/username (PII).

        // 2. Parse tools and extract messages
        const tool_params = json.parseTools(body);
        const extracted = json.extractMessages(body, g_server.allocator);
        defer if (extracted) |ex| ex.deinit(g_server.allocator);
        const fallback_raw = json.extractLastMessage(body);
        if (extracted == null and fallback_raw == null) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing or empty messages array", "messages", "missing_required_parameter");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        if (rejectBadImage(stream, body, false, false, request_start, method, path)) return;
        const fallback_str = fallback_raw orelse "";
        const fallback_content = json.jsonUnescape(g_server.allocator, fallback_str) catch @constCast(fallback_str);
        defer if (fallback_content.ptr != fallback_str.ptr) wipeFree(g_server.allocator, fallback_content);

        // Inject tool definitions into system prompt
        var tool_system: ?[]u8 = null;
        defer if (tool_system) |ts| wipeFree(g_server.allocator, ts);
        if (toolsWanted(&tool_params)) {
            tool_system = buildToolSystemPrompt(g_server.allocator, g_server.chat_template, &tool_params, if (extracted) |ex| ex.system else null, &g_server.tool_registry) catch |err| {
                std.log.err("req={d} tool system prompt build failed: {}", .{ log_request_id, err });
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to build tool definitions");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
        }
        const effective_system = if (tool_system) |ts| @as(?[]const u8, ts) else if (extracted) |ex| ex.system else null;

        // Format with full conversation context when available
        const formatted = if (extracted) |ex|
            g_server.chat_template.formatConversation(g_server.allocator, effective_system, ex.messages) catch
                g_server.chat_template.format(g_server.allocator, null, fallback_content) catch fallback_content
        else
            g_server.chat_template.format(g_server.allocator, null, fallback_content) catch fallback_content;
        defer if (formatted.ptr != fallback_content.ptr) wipeFree(g_server.allocator, @constCast(formatted));
        // encode("") allocates a zero-length slice; always free on success (not only when len > 0).
        const prompt_ids_owned = g_server.tokenizer.encode(formatted) catch |err| blk: {
            std.log.warn("req={d} tokenizer encode failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        defer if (prompt_ids_owned) |ids| wipeFreeTokens(g_server.allocator, ids);
        const prompt_ids = prompt_ids_owned orelse &[_]u32{};
        const prompt_tokens = estimatePromptTokens(prompt_ids.len, formatted.len);

        // 3. Check rate limit
        if (checkRateLimit(g_server, prompt_tokens)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        // 4. Check for base64 image in OpenAI content array format
        const completions_has_image = json.extractJsonImage(body) != null and g_server.vision_encoder != null;
        if (completions_has_image) g_server.vision_mutex.lockUncancelable(g_server.io);
        defer if (completions_has_image) g_server.vision_mutex.unlock(g_server.io);
        var completions_image_embedded = false;
        if (json.extractJsonImage(body)) |b64_data| {
            if (g_server.vision_encoder) |ve| {
                if (processVisionImage(b64_data, ve)) {
                    completions_image_embedded = true;
                } else {
                    std.log.warn("req={d} image attached but decode/encode failed (400, client error)", .{log_request_id});
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Failed to decode or encode attached image", "image", "image_decode_failed");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                }
            }
        }
        defer if (completions_image_embedded) {
            clearPendingVision();
        };

        if (json.extractBoolField(body, "stream")) {
            if (toolsWanted(&tool_params)) {
                startStreamWithTools(stream, formatted, max_tokens, sampling, &tool_params);
            } else {
                startStream(stream, formatted, true, false, max_tokens, sampling);
            }
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Formatted already computed above for token counting
        const gen = generateEscapedNPre(formatted, true, max_tokens, sampling, prompt_ids_owned);
        defer gen.deinit();

        // Generation error → 500 (don't return 200 with error content)
        if (std.mem.eql(u8, gen.finish_reason, "error")) {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Generation failed");
            g_server.metrics.recordFailure();
            g_server.metrics.recordLatency(elapsedMs(req_start_time));
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        }

        const req_id = currentRequestId();
        const created = timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        var resp_buf: [response_buf_size]u8 = undefined;

        const json_body = if (toolsWanted(&tool_params) and hasToolCalls(gen.raw)) blk: {
            const tc_resp = buildToolCallResponse(&resp_buf, gen.raw, req_id, created, gen.stats.prompt_tokens, gen.stats.tokens_generated, &tool_params);
            if (tc_resp.len > 0) break :blk tc_resp;
            // No declared tool parsed. Return the text without the tag markup:
            // the server just refused this payload, so a client that scans for
            // tags must not find it in the content either.
            const stripped = stripToolCallTags(g_server.allocator, gen.raw) catch gen.raw;
            defer if (stripped.ptr != gen.raw.ptr) g_server.allocator.free(@constCast(stripped));
            const content = json.jsonEscape(g_server.allocator, stripped) catch gen.escaped;
            defer if (content.ptr != stripped.ptr and content.ptr != gen.escaped.ptr) g_server.allocator.free(@constCast(content));
            break :blk std.fmt.bufPrint(&resp_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{ req_id, created, g_server.model_name, system_fingerprint, content, gen.finish_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
                std.log.warn("req={d} response buffer overflow: output {d} bytes exceeds {d} byte buffer", .{ log_request_id, gen.escaped.len, response_buf_size });
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
        } else blk: {
            // Split reasoning_content from content for thinking models (DeepSeek R1, QwQ, Gemma 4 12B).
            const split = splitThinkingContent(gen.raw);
            if (split.reasoning.len > 0) {
                // Emit separate reasoning_content field (DeepSeek/o1 compatible API)
                const reasoning_escaped = json.jsonEscape(g_server.allocator, split.reasoning) catch {
                    std.log.err("req={d} reasoning_content JSON escape OOM", .{log_request_id});
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to encode response");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
                defer if (reasoning_escaped.ptr != split.reasoning.ptr) g_server.allocator.free(reasoning_escaped);
                const content_escaped = json.jsonEscape(g_server.allocator, split.content) catch {
                    std.log.err("req={d} content JSON escape OOM", .{log_request_id});
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to encode response");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
                defer if (content_escaped.ptr != split.content.ptr) g_server.allocator.free(content_escaped);
                break :blk std.fmt.bufPrint(&resp_buf,
                    \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","reasoning_content":"{s}","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
                , .{ req_id, created, g_server.model_name, system_fingerprint, reasoning_escaped, content_escaped, gen.finish_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
                    std.log.warn("req={d} response buffer overflow (reasoning)", .{log_request_id});
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
            }
            break :blk std.fmt.bufPrint(&resp_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{ req_id, created, g_server.model_name, system_fingerprint, gen.escaped, gen.finish_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
                std.log.warn("req={d} response buffer overflow: output {d} bytes exceeds {d} byte buffer", .{ log_request_id, gen.escaped.len, response_buf_size });
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
        };
        sendJson(stream, json_body);

        // Record metrics
        g_server.metrics.recordLatency(elapsedMs(req_start_time));
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/completions")) {
        logRequest(method, path);
        const req_start_time = milliTimestamp();

        g_server.metrics.recordRequest();

        const body = req.body;
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available", "n", "n_not_supported");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const prompt_raw = json.extractField(body, "prompt") orelse {
            if (json.hasField(body, "prompt")) {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "prompt must be a string", "prompt", "invalid_value");
            } else {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing required field: prompt", "prompt", "missing_required_parameter");
            }
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const prompt = json.jsonUnescape(g_server.allocator, prompt_raw) catch @constCast(prompt_raw);
        defer if (prompt.ptr != prompt_raw.ptr) wipeFree(g_server.allocator, prompt);
        const max_tokens = extractMaxGenTokens(body, &.{"max_tokens"});
        var sampling_c = json.SamplingParams{};
        json.parseSampling(&sampling_c, body);

        // Rate limit check (estimate prompt tokens via encode)
        const prompt_ids_c_owned = g_server.tokenizer.encode(prompt) catch |err| blk: {
            std.log.warn("req={d} tokenizer encode failed for rate-limit estimate: {}", .{ log_request_id, err });
            break :blk null;
        };
        defer if (prompt_ids_c_owned) |ids| wipeFreeTokens(g_server.allocator, ids);
        const prompt_ids_c = prompt_ids_c_owned orelse &[_]u32{};
        const prompt_tokens_c = estimatePromptTokens(prompt_ids_c.len, prompt.len);
        if (checkRateLimit(g_server, prompt_tokens_c)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        if (json.extractBoolField(body, "stream")) {
            startStreamRaw(stream, prompt, max_tokens, sampling_c);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Completions endpoint: use prompt as-is (no chat template wrapping)
        const gen = generateEscapedNPre(prompt, true, max_tokens, sampling_c, prompt_ids_c_owned);
        defer gen.deinit();

        // Generation error → 500 (don't return 200 with error content)
        if (std.mem.eql(u8, gen.finish_reason, "error")) {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Generation failed");
            g_server.metrics.recordFailure();
            g_server.metrics.recordLatency(elapsedMs(req_start_time));
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        }

        const req_id = currentRequestId();
        const created = timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        var resp_buf: [response_buf_size]u8 = undefined;
        const json_body = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, g_server.model_name, system_fingerprint, gen.escaped, gen.finish_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
            std.log.warn("req={d} response buffer overflow: output {d} bytes exceeds {d} byte buffer", .{ log_request_id, gen.escaped.len, response_buf_size });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json_body);

        // Record metrics
        g_server.metrics.recordLatency(elapsedMs(req_start_time));
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/tokenize")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        // Accept: {"text":"..."} or {"content":"..."} (raw text)
        // Also accept: {"messages":[...]} (chat-completion format, apply template then tokenize)
        const body = req.body;
        const input_text: []const u8 = blk: {
            if (json.extractField(body, "text")) |t| break :blk t;
            if (json.extractField(body, "content")) |t| break :blk t;
            if (json.extractMessages(body, g_server.allocator)) |msgs| {
                defer msgs.deinit(g_server.allocator);
                const sys = msgs.system orelse "";
                const formatted = g_server.chat_template.formatConversation(g_server.allocator, sys, msgs.messages) catch |err| {
                    std.log.err("req={d} tokenize formatConversation failed: {}", .{ log_request_id, err });
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
                defer wipeFree(g_server.allocator, formatted);
                const tids = g_server.tokenizer.encode(formatted) catch |err| {
                    std.log.err("req={d} tokenize encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
                defer wipeFreeTokens(g_server.allocator, tids);
                var resp_buf2: [response_buf_size]u8 = undefined;
                const resp2 = std.fmt.bufPrint(&resp_buf2, "{{\"count\":{d},\"model\":\"{s}\"}}", .{ tids.len, g_server.model_name }) catch {
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                };
                sendJson(stream, resp2);
                g_server.metrics.recordCompletion();
                logRequestDone(method, path, 200, elapsedMs(request_start));
                return;
            }
            // Distinguish "field absent" from "field present but not a string",
            // the way /v1/completions and /v1/responses do for `prompt`/`input`.
            // Name the offending field so the caller knows which one to fix.
            const bad_text = json.hasField(body, "text") and json.extractField(body, "text") == null;
            const bad_content = json.hasField(body, "content") and json.extractField(body, "content") == null;
            const wrong_type = bad_text or bad_content;
            sendJsonErrorEx(
                stream,
                "400 Bad Request",
                "invalid_request_error",
                if (wrong_type) "text and content must be strings" else "Provide text, content, or messages",
                if (bad_text) "text" else if (bad_content) "content" else null,
                if (wrong_type) "invalid_value" else "missing_required_parameter",
            );
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const text = input_text;
        const unescaped = json.jsonUnescape(g_server.allocator, text) catch @constCast(text);
        defer if (unescaped.ptr != text.ptr) wipeFree(g_server.allocator, unescaped);
        const token_ids = g_server.tokenizer.encode(unescaped) catch |err| {
            std.log.err("req={d} tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, unescaped.len, err });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer wipeFreeTokens(g_server.allocator, token_ids);

        // Build JSON response with token count
        var resp_buf: [response_buf_size]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf,
            \\{{"count":{d},"model":"{s}"}}
        , .{ token_ids.len, g_server.model_name }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, resp);
        g_server.metrics.recordCompletion();
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/detokenize")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();

        // Parse token IDs from JSON array: {"tokens": [1, 2, 3]}
        var tok_ids: [gen_ids_buf_size]u32 = undefined;
        defer @memset(std.mem.sliceAsBytes(&tok_ids), 0);
        const n_toks: usize = switch (parseDetokenizeTokens(req.body, &tok_ids)) {
            .malformed => {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "tokens must be an array of non-negative integers of at most 4096 entries", "tokens", "invalid_value");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            },
            .empty => {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing or empty tokens array", "tokens", "missing_required_parameter");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            },
            .count => |n| n,
        };
        const decoded = g_server.tokenizer.decode(tok_ids[0..n_toks]) catch |err| {
            std.log.err("req={d} detokenizer decode failed ({d} tokens): {}", .{ log_request_id, n_toks, err });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Detokenization failed");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer wipeFree(g_server.allocator, decoded);

        const escaped = json.jsonEscape(g_server.allocator, decoded) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer if (escaped.ptr != decoded.ptr) wipeFree(g_server.allocator, escaped);
        var final_buf: [response_buf_size]u8 = undefined;
        defer @memset(&final_buf, 0);
        const resp = std.fmt.bufPrint(&final_buf,
            \\{{"text":"{s}","model":"{s}"}}
        , .{ escaped, g_server.model_name }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, resp);
        g_server.metrics.recordCompletion();
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    // ── /v1/kv_cache/info: lightweight KV metadata for external orchestrators ──
    // Must be checked before /v1/kv_cache so prefix matching cannot shadow it.
    // Returns: seq_len, prefix_len, kv_used, kv_total, prefix_hash
    if (is_get and std.mem.eql(u8, path, "/v1/kv_cache/info")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        const kv_used = g_server.metrics.kv_blocks_used.load(.monotonic);
        const kv_total = g_server.metrics.kv_blocks_total.load(.monotonic);
        // Snapshot seq_len + prefix under mutex, concurrent generateN may free/replace
        // cached_prompt_ids (UAF) or reset KV between unlocked reads (inconsistent pair).
        // server.mutex alone does not exclude the scheduler thread's forwards on
        // kv_seq_len (they hold model_mutex without server.mutex), so take both.
        var hash: u64 = fnv1a_offset_basis;
        var cached_prefix_len: usize = 0;
        var seq_len: usize = 0;
        {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);
            lockModelWithScheduler();
            defer unlockModelWithScheduler();
            seq_len = g_server.model.kvSeqLen();
            const prefix_ids = g_server.cached_prompt_ids;
            cached_prefix_len = prefix_ids.len;
            for (prefix_ids) |tid| {
                const b: [4]u8 = @bitCast(tid);
                for (b) |byte| {
                    hash ^= @as(u64, byte);
                    hash *%= fnv1a_prime;
                }
            }
        }
        var info_buf: [health_buf_size]u8 = undefined;
        const info_json = std.fmt.bufPrint(&info_buf,
            \\{{"seq_len":{d},"cached_prefix_len":{d},"prefix_hash":"{x}","kv_used":{d},"kv_total":{d}}}
        , .{ seq_len, cached_prefix_len, hash, kv_used, kv_total }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, info_json);
        g_server.metrics.recordCompletion();
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    // ── /v1/kv_cache: cross-instance KV prefix sharing (LMCache-style) ──────────
    // GET  /v1/kv_cache?n_tokens=<N> , export N-token prefix as binary blob
    // POST /v1/kv_cache?n_tokens=<N> , import N-token prefix from binary body
    if ((is_get or is_post) and std.mem.eql(u8, path, "/v1/kv_cache")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        // Parse n_tokens from query string: ?n_tokens=<N>
        const n_tokens: usize = switch (parsePositiveQueryParam(req.query, "n_tokens")) {
            .missing => {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n_tokens query parameter required (positive integer)", "n_tokens", "missing_required_parameter");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            },
            .invalid => {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n_tokens must be a positive integer", "n_tokens", "invalid_value");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            },
            .value => |n| n,
        };
        if (is_get) {
            // Export under inference mutex (not stdout_mutex): KV must not race generate.
            // Copy under lock, then release before network I/O so slow clients do not stall inference.
            const ExportOutcome = enum { ok, oom, unsupported, too_many };
            var export_buf: []u8 = &.{};
            var export_n: usize = 0;
            const outcome: ExportOutcome = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                // server.mutex alone does not exclude the scheduler thread's
                // Phase A/B forwards (they hold model_mutex without
                // server.mutex). Take model_mutex too, or exportKvPrefix reads
                // the KV cache mid-forward and hands out a torn prefix blob.
                lockModelWithScheduler();
                defer unlockModelWithScheduler();
                if (n_tokens > g_server.model.kvSeqLen()) break :blk .too_many;
                var buf_len = estimateKvExportBytes(g_server.model.*, n_tokens);
                var buf = g_server.allocator.alloc(u8, buf_len) catch break :blk .oom;
                var n_written = g_server.model.exportKvPrefix(buf, n_tokens);
                // Retry at cap if estimate undershot (per-layer kvd larger than average).
                if (n_written == 0 and buf_len < kv_export_max_bytes) {
                    g_server.allocator.free(buf);
                    buf_len = kv_export_max_bytes;
                    buf = g_server.allocator.alloc(u8, buf_len) catch break :blk .oom;
                    n_written = g_server.model.exportKvPrefix(buf, n_tokens);
                }
                if (n_written == 0) {
                    g_server.allocator.free(buf);
                    break :blk .unsupported;
                }
                export_buf = buf;
                export_n = n_written;
                break :blk .ok;
            };
            switch (outcome) {
                .oom => {
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "OOM");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                    return;
                },
                .too_many => {
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n_tokens exceeds current KV cache length", "n_tokens", "invalid_value");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                },
                .unsupported => {
                    sendJsonErrorEx(stream, "501 Not Implemented", "not_implemented", "Model does not support KV export", null, "not_implemented");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 501, elapsedMs(request_start));
                    return;
                },
                .ok => {
                    defer g_server.allocator.free(export_buf);
                    sendResponse(stream, "200 OK", "application/octet-stream", export_buf[0..export_n]);
                    g_server.metrics.recordCompletion();
                    logRequestDone(method, path, 200, elapsedMs(request_start));
                },
            }
        } else {
            // Import under inference mutex: concurrent generate must not read torn KV.
            // Clear prefix-cache IDs: blob has no token IDs, so bookkeeping would lie.
            const ok = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                // Same scheduler exclusion as the export path: importKvPrefix
                // writes the KV buffers a concurrent scheduled forward is
                // reading, which would silently corrupt its attention state.
                lockModelWithScheduler();
                defer unlockModelWithScheduler();
                if (!g_server.model.importKvPrefix(req.body, n_tokens)) break :blk false;
                g_server.clearCachedPromptIds();
                g_server.kv_valid = true;
                break :blk true;
            };
            if (!ok) {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "KV import failed (size mismatch or unsupported)", "n_tokens", "kv_import_failed");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            }
            var import_buf: [clear_response_buf_size]u8 = undefined;
            const import_resp = std.fmt.bufPrint(&import_buf, "{{\"imported\":{d}}}", .{n_tokens}) catch "{\"ok\":true}";
            sendJson(stream, import_resp);
            g_server.metrics.recordCompletion();
            logRequestDone(method, path, 200, elapsedMs(request_start));
        }
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/embeddings")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        sendJsonErrorEx(stream, "501 Not Implemented", "not_implemented", "Embeddings endpoint not implemented", null, "not_implemented");
        g_server.metrics.recordClientError();
        logRequestDone(method, path, 501, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/responses")) {
        logRequest(method, path);
        const req_start_time = milliTimestamp();

        g_server.metrics.recordRequest();

        const body = req.body;
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available", "n", "n_not_supported");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const input_raw = json.extractField(body, "input") orelse {
            if (json.hasField(body, "input")) {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "input must be a string", "input", "invalid_value");
            } else {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing required field: input", "input", "missing_required_parameter");
            }
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const input = json.jsonUnescape(g_server.allocator, input_raw) catch @constCast(input_raw);
        defer if (input.ptr != input_raw.ptr) wipeFree(g_server.allocator, input);
        const max_tokens = extractMaxGenTokens(body, &.{ "max_output_tokens", "max_tokens", "max_completion_tokens" });
        var sampling_r = json.SamplingParams{};
        json.parseSampling(&sampling_r, body);

        // Tool definitions, injected into the system prompt so the model can
        // emit tool-call tags this endpoint parses back with a name allowlist.
        const tool_params_r = json.parseToolsAnthropic(body);
        var tool_system_r: ?[]u8 = null;
        defer if (tool_system_r) |ts| wipeFree(g_server.allocator, ts);
        if (toolsWanted(&tool_params_r)) {
            tool_system_r = buildToolSystemPrompt(g_server.allocator, g_server.chat_template, &tool_params_r, null, &g_server.tool_registry) catch |err| {
                std.log.err("req={d} tool system prompt build failed: {}", .{ log_request_id, err });
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to build tool definitions");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
        }

        // Rate limit check
        const formatted_rl = g_server.chat_template.format(g_server.allocator, tool_system_r, input) catch input;
        defer if (formatted_rl.ptr != input.ptr) wipeFree(g_server.allocator, @constCast(formatted_rl));
        const prompt_ids_r_owned = g_server.tokenizer.encode(formatted_rl) catch |err| blk: {
            std.log.warn("req={d} tokenizer encode failed for rate-limit estimate: {}", .{ log_request_id, err });
            break :blk null;
        };
        defer if (prompt_ids_r_owned) |ids| wipeFreeTokens(g_server.allocator, ids);
        const prompt_ids_r = prompt_ids_r_owned orelse &[_]u32{};
        const prompt_tokens_r = estimatePromptTokens(prompt_ids_r.len, formatted_rl.len);
        if (checkRateLimit(g_server, prompt_tokens_r)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        if (json.extractBoolField(body, "stream")) {
            startResponsesStream(stream, input, max_tokens, sampling_r);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        const gen = generateEscapedNPre(formatted_rl, true, max_tokens, sampling_r, prompt_ids_r_owned);
        defer gen.deinit();

        // Generation error → 500 (don't return 200 with error content)
        if (std.mem.eql(u8, gen.finish_reason, "error")) {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Generation failed");
            g_server.metrics.recordFailure();
            g_server.metrics.recordLatency(elapsedMs(req_start_time));
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        }

        const req_id = currentRequestId();
        const created = timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        const resp_stop_reason: []const u8 = if (std.mem.eql(u8, gen.finish_reason, "length")) "max_tokens" else "stop";
        var resp_buf: [response_buf_size]u8 = undefined;
        // A tool-call tag that names no declared tool leaves call_idx at 0 and
        // returns "", so unparseable output degrades to a plain message item.
        const tc_body = if (toolsWanted(&tool_params_r) and hasToolCalls(gen.raw))
            buildResponsesToolCallResponse(&resp_buf, gen.raw, req_id, created, gen.stats.prompt_tokens, gen.stats.tokens_generated, &tool_params_r)
        else
            "";
        const json_body = if (tc_body.len > 0) tc_body else std.fmt.bufPrint(&resp_buf,
            \\{{"id":"resp-{d}","object":"response","created_at":{d},"status":"completed","model":"{s}","stop_reason":"{s}","output":[{{"type":"message","id":"msg_0","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}],"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, g_server.model_name, resp_stop_reason, gen.escaped, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
            std.log.warn("req={d} response buffer overflow: output {d} bytes exceeds {d} byte buffer", .{ log_request_id, gen.escaped.len, response_buf_size });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json_body);

        // Record metrics
        g_server.metrics.recordLatency(elapsedMs(req_start_time));
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    // ── Anthropic Messages API (/v1/messages) ───────────────────
    if (is_post and std.mem.eql(u8, path, "/v1/messages")) {
        logRequest(method, path);
        const req_start_time = milliTimestamp();

        g_server.metrics.recordRequest();

        const body = req.body;
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendAnthropicError(stream, "400", "invalid_request_error", "n > 1 is not supported; only single completions are available");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const max_tokens_m = extractMaxGenTokens(body, &.{"max_tokens"});
        var sampling_m = json.SamplingParams{};
        json.parseSampling(&sampling_m, body);

        // Tools (Anthropic flat format). Normalize Anthropic tool_choice values:
        // "any"/"tool" mean the model must call a tool, matching OpenAI "required".
        var tool_params_m = json.parseToolsAnthropic(body);
        if (std.mem.eql(u8, tool_params_m.tool_choice, "any") or std.mem.eql(u8, tool_params_m.tool_choice, "tool")) {
            tool_params_m.tool_choice = "required";
        }
        const want_tools_m = toolsWanted(&tool_params_m);

        // Anthropic: system message is a top-level field, not in messages
        // array, and is either a string or an array of text blocks.
        const system_msg = json.extractSystemOwned(g_server.allocator, body) catch |err| blk: {
            std.log.warn("req={d} anthropic system prompt decode failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        defer if (system_msg) |s| wipeFree(g_server.allocator, s);

        // Inject tool definitions into the system prompt ahead of the request's
        // own system message (same contract as /v1/chat/completions).
        var tool_system_m: ?[]u8 = null;
        defer if (tool_system_m) |ts| wipeFree(g_server.allocator, ts);
        if (want_tools_m) {
            tool_system_m = buildToolSystemPrompt(g_server.allocator, g_server.chat_template, &tool_params_m, system_msg, &g_server.tool_registry) catch |err| {
                std.log.err("req={d} anthropic tool system prompt build failed: {}", .{ log_request_id, err });
                sendAnthropicError(stream, "500", "api_error", "Failed to build tool definitions");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
        }
        const effective_system_m: ?[]const u8 = if (tool_system_m) |ts| ts else system_msg;

        // Extract full messages array for multi-turn conversations
        const extracted_m = json.extractMessages(body, g_server.allocator);
        defer if (extracted_m) |ex| ex.deinit(g_server.allocator);
        const fallback_raw_m = json.extractLastMessage(body);
        if (extracted_m == null and fallback_raw_m == null) {
            sendAnthropicError(stream, "400", "invalid_request_error", "Missing or empty messages array");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        if (rejectBadImage(stream, body, false, true, request_start, method, path)) return;
        const fallback_str_m = fallback_raw_m orelse "";
        const fallback_content_m = json.jsonUnescape(g_server.allocator, fallback_str_m) catch @constCast(fallback_str_m);
        defer if (fallback_content_m.ptr != fallback_str_m.ptr) wipeFree(g_server.allocator, fallback_content_m);

        // Format with full conversation context when available
        const formatted_m = if (extracted_m) |ex|
            g_server.chat_template.formatConversation(g_server.allocator, effective_system_m, ex.messages) catch
                g_server.chat_template.format(g_server.allocator, effective_system_m, fallback_content_m) catch fallback_content_m
        else
            g_server.chat_template.format(g_server.allocator, effective_system_m, fallback_content_m) catch fallback_content_m;
        defer if (formatted_m.ptr != fallback_content_m.ptr) wipeFree(g_server.allocator, @constCast(formatted_m));
        const prompt_ids_m_owned = g_server.tokenizer.encode(formatted_m) catch |err| blk: {
            std.log.warn("req={d} tokenizer encode failed for rate-limit estimate: {}", .{ log_request_id, err });
            break :blk null;
        };
        defer if (prompt_ids_m_owned) |ids| wipeFreeTokens(g_server.allocator, ids);
        const prompt_ids_m = prompt_ids_m_owned orelse &[_]u32{};
        // When tokenization fails, use conservative byte-count estimate (1 byte = 1 token)
        const prompt_tokens_m = estimatePromptTokens(prompt_ids_m.len, formatted_m.len);

        // Rate limit check
        if (checkRateLimit(g_server, prompt_tokens_m)) |retry| {
            sendAnthropic429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        // Vision: extract base64 image from content array (Anthropic format)
        const anthropic_has_image = json.extractJsonImage(body) != null and g_server.vision_encoder != null;
        if (anthropic_has_image) g_server.vision_mutex.lockUncancelable(g_server.io);
        defer if (anthropic_has_image) g_server.vision_mutex.unlock(g_server.io);
        var anthropic_image_embedded = false;
        if (json.extractJsonImage(body)) |b64_data| {
            if (g_server.vision_encoder) |ve| {
                if (processVisionImage(b64_data, ve)) {
                    anthropic_image_embedded = true;
                } else {
                    std.log.warn("req={d} anthropic image attached but decode/encode failed (400, client error)", .{log_request_id});
                    sendAnthropicError(stream, "400", "invalid_request_error", "Failed to decode or encode attached image");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                }
            }
        }
        defer if (anthropic_image_embedded) {
            clearPendingVision();
        };

        if (json.extractBoolField(body, "stream")) {
            if (want_tools_m) {
                startAnthropicStreamWithTools(stream, formatted_m, max_tokens_m, prompt_tokens_m, sampling_m, &tool_params_m);
            } else {
                startAnthropicStream(stream, formatted_m, max_tokens_m, prompt_tokens_m, sampling_m);
            }
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Non-streaming: generate and return Anthropic format
        const gen = generateEscapedNPre(formatted_m, true, max_tokens_m, sampling_m, prompt_ids_m_owned);
        defer gen.deinit();

        // Generation error → 500 (don't return 200 with error content)
        if (std.mem.eql(u8, gen.finish_reason, "error")) {
            sendAnthropicError(stream, "500", "api_error", "Generation failed");
            g_server.metrics.recordFailure();
            g_server.metrics.recordLatency(elapsedMs(req_start_time));
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        }

        const req_id = currentRequestId();
        const stop_reason: []const u8 = if (std.mem.eql(u8, gen.finish_reason, "length")) "max_tokens" else "end_turn";

        // Tool calls: emit Anthropic content blocks with stop_reason "tool_use".
        // Falls back to the plain text response when no <tool_call> payload
        // parses (small-model malformed output), mirroring /v1/chat/completions.
        var resp_buf_tc: [response_buf_size]u8 = undefined;
        if (want_tools_m and hasToolCalls(gen.raw)) {
            const anth_tc = buildAnthropicToolCallResponse(&resp_buf_tc, gen.raw, req_id, gen.stats.prompt_tokens, gen.stats.tokens_generated, &tool_params_m);
            if (anth_tc.len > 0) {
                sendJson(stream, anth_tc);
                g_server.metrics.recordLatency(elapsedMs(req_start_time));
                g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
                g_server.metrics.recordCompletion();
                logRequestDone(method, path, 200, elapsedMs(request_start));
                return;
            }
        }

        var resp_buf: [response_buf_size]u8 = undefined;
        const json_body = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"msg_{d}","type":"message","role":"assistant","content":[{{"type":"text","text":"{s}"}}],"model":"{s}","stop_reason":"{s}","stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
        , .{ req_id, gen.escaped, g_server.model_name, stop_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated }) catch {
            std.log.warn("req={d} response buffer overflow: output {d} bytes exceeds {d} byte buffer", .{ log_request_id, gen.escaped.len, response_buf_size });
            sendAnthropicError(stream, "500", "api_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json_body);

        // Record metrics
        g_server.metrics.recordLatency(elapsedMs(req_start_time));
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if ((is_get or is_post) and std.mem.eql(u8, path, "/v1/conversations")) {
        logRequest(method, path);

        g_server.metrics.recordRequest();

        if (is_get) {
            // Return list of conversations as JSON.
            // Format under mutex (reads conversation data), then release before I/O
            // to avoid blocking inference while sending to a slow client.
            var buf: [conv_list_buf_size]u8 = undefined;
            const response = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                var w: std.Io.Writer = .fixed(&buf);
                w.writeByte('[') catch break :blk @as(?[]const u8, null);
                for (g_server.conversations.items, 0..) |*conv, ci| {
                    if (ci > 0) w.writeByte(',') catch break :blk null;
                    const title = conv.titleSlice();
                    const escaped_title = json.jsonEscape(g_server.allocator, title) catch title[0..0];
                    defer if (escaped_title.ptr != title.ptr) g_server.allocator.free(escaped_title);
                    w.print(
                        \\{{"id":{d},"title":"{s}","active":{s},"count":{d}}}
                    , .{ conv.id, escaped_title, if (conv.id == g_server.active_id) "true" else "false", conv.messages.items.len }) catch break :blk null;
                }
                w.writeByte(']') catch break :blk null;
                break :blk @as(?[]const u8, w.buffered());
            };
            if (response) |json_data| {
                sendJson(stream, json_data);
                g_server.metrics.recordCompletion();
                logRequestDone(method, path, 200, elapsedMs(request_start));
            } else {
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Response buffer overflow");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
            }
            return;
        }
        // POST: action=new|select|delete
        // All conversation mutations must be mutex-protected to prevent
        // races with concurrent generate() calls that read kv_valid.
        const body = req.body;
        const action = json.extractFormField(body, "action") orelse "new";
        if (std.mem.eql(u8, action, "new")) {
            // `new` allocates an id and persists another conversation row, so a
            // retry after a lost response leaves two conversations for one user
            // action. `select` and `delete` converge on the same state and stay
            // out of the ledger.
            if (resolveIdempotency(stream, method, path, request_start)) return;
            const new_id: u32 = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                break :blk if (g_server.createConv()) |nc| nc.id else 0;
            };
            if (new_id == 0) {
                sendJsonErrorEx(stream, "503 Service Unavailable", "server_error", "Maximum conversation limit reached", null, "conversation_limit_reached");
                g_server.metrics.recordFailure();
                g_server.releaseIdempotencyKey();
                logRequestDone(method, path, 503, elapsedMs(request_start));
                return;
            }
            g_server.persistConversations();
            var nbuf: [clear_response_buf_size]u8 = undefined;
            const njson = std.fmt.bufPrint(&nbuf,
                \\{{"ok":true,"id":{d}}}
            , .{new_id}) catch "{\"ok\":true}";
            sendJson(stream, njson);
            g_server.completeIdempotencyKey("200 OK", "application/json", njson);
            g_server.metrics.recordCompletion();
            logRequestDone(method, path, 200, elapsedMs(request_start));
        } else if (std.mem.eql(u8, action, "select")) {
            const id: u32 = switch (parseFormConversationId(body)) {
                .missing => {
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing required field: id", "id", "missing_required_parameter");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                },
                .invalid => {
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "id must be a positive integer", "id", "invalid_value");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                },
                .value => |v| v,
            };
            var mbuf: [conv_msgs_buf_size]u8 = undefined;
            var mw: std.Io.Writer = .fixed(&mbuf);
            var switched = false;
            const select_result: enum { not_found, format_ok, format_fail } = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);

                const conv = g_server.getConvById(id) orelse break :blk .not_found;
                switched = g_server.selectConv(id);
                mw.writeAll("{\"messages\":[") catch break :blk .format_fail;
                for (conv.messages.items, 0..) |msg, mi| {
                    if (mi > 0) mw.writeByte(',') catch break :blk .format_fail;
                    const role_str: []const u8 = switch (msg.role) {
                        .user => "user",
                        .assistant => "assistant",
                        .tool => "tool",
                    };
                    const esc_content = json.jsonEscape(g_server.allocator, msg.content) catch break :blk .format_fail;
                    defer if (esc_content.ptr != msg.content.ptr) g_server.allocator.free(esc_content);
                    mw.print(
                        \\{{"role":"{s}","content":"{s}"}}
                    , .{ role_str, esc_content }) catch break :blk .format_fail;
                }
                mw.writeAll("]}") catch break :blk .format_fail;
                break :blk .format_ok;
            };
            switch (select_result) {
                .not_found => {
                    sendJsonErrorEx(stream, "404 Not Found", "invalid_request_error", "Conversation not found", "id", "conversation_not_found");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 404, elapsedMs(request_start));
                    return;
                },
                .format_ok => {
                    if (switched) g_server.persistConversations();
                    sendJson(stream, mw.buffered());
                    g_server.metrics.recordCompletion();
                    logRequestDone(method, path, 200, elapsedMs(request_start));
                },
                .format_fail => {
                    sendJsonError(stream, "500 Internal Server Error", "server_error", "Response buffer overflow");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 500, elapsedMs(request_start));
                },
            }
        } else if (std.mem.eql(u8, action, "delete")) {
            const id: u32 = switch (parseFormConversationId(body)) {
                .missing => {
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing required field: id", "id", "missing_required_parameter");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                },
                .invalid => {
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "id must be a positive integer", "id", "invalid_value");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                },
                .value => |v| v,
            };
            const delete_result: ?bool = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                if (g_server.getConvById(id) == null) break :blk null;
                const was_active = g_server.active_id == id;
                g_server.deleteConv(id);
                break :blk was_active;
            };
            if (delete_result == null) {
                sendJsonErrorEx(stream, "404 Not Found", "invalid_request_error", "Conversation not found", "id", "conversation_not_found");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 404, elapsedMs(request_start));
                return;
            }
            const was_active = delete_result.?;
            g_server.persistConversations();
            var dbuf: [clear_response_buf_size]u8 = undefined;
            const djson = std.fmt.bufPrint(&dbuf,
                \\{{"ok":true,"cleared":{s}}}
            , .{if (was_active) "true" else "false"}) catch "{\"ok\":true}";
            sendJson(stream, djson);
            g_server.metrics.recordCompletion();
            logRequestDone(method, path, 200, elapsedMs(request_start));
        } else {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Unknown conversation action", "action", "unknown_conversation_action");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
        }
        return;
    }

    // ── Chat regenerate endpoint ────────────────────────────────
    // Pops the last assistant message from the active conversation,
    // resets the KV cache, re-formats the full conversation, and
    // generates a new response. Supports SSE streaming.
    if (is_post and std.mem.eql(u8, path, "/v1/chat/regenerate")) {
        logRequest(method, path);

        g_server.metrics.recordRequest();

        // Each call pops one assistant message, so a retry after a lost
        // response would destroy a turn the first attempt already replaced.
        if (resolveIdempotency(stream, method, path, request_start)) return;

        const regen_body = req.body;
        var regen_sampling = json.SamplingParams{};
        json.parseFormSampling(&regen_sampling, regen_body);
        const regen_max_tokens = clampMaxTokens(json.extractFormInt(regen_body, "max_tokens"));

        // Extract optional system prompt (URL-decode since web UI sends encodeURIComponent)
        const regen_system_field = json.extractFormField(regen_body, "system");
        const regen_system_decoded = if (regen_system_field) |sf| (json.urlDecode(g_server.allocator, sf) catch |err| blk: {
            std.log.warn("req={d} system prompt URL decode failed: {}", .{ log_request_id, err });
            break :blk null;
        }) else null;
        defer if (regen_system_decoded) |sd| wipeFree(g_server.allocator, sd);
        const regen_system_prompt: ?[]const u8 = if (regen_system_decoded) |sd| blk: {
            const s = std.mem.trim(u8, sd, " \t\r\n");
            break :blk if (s.len > 0) s else null;
        } else null;

        const RegenPrepResult = struct { formatted: []const u8, msg_count: usize, prompt_ids: []u32 };
        const regen_prep: ?RegenPrepResult = blk: {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);

            const regen_conv = g_server.getActiveConv() orelse {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "No active conversation", null, "no_active_conversation");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                break :blk null;
            };

            // Remove the last assistant message (if any). It is held, not
            // freed, so the rate-limit check below can put it back when the
            // request is rejected: freeing first would make the 429 lose the
            // reply in memory and on disk.
            var removed_assistant: ?chat_tmpl_mod.Message = null;
            if (regen_conv.messages.items.len > 0) {
                const last_msg = regen_conv.messages.items[regen_conv.messages.items.len - 1];
                if (last_msg.role == .assistant) removed_assistant = regen_conv.messages.pop();
            }

            if (regen_conv.messages.items.len == 0) {
                if (removed_assistant) |m| regen_conv.restoreHeldMessage(g_server.allocator, m);
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "No user message to regenerate from", null, "no_user_message");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                break :blk null;
            }

            g_server.kv_valid = false;
            const regen_formatted = g_server.chat_template.formatConversation(
                g_server.allocator,
                regen_system_prompt,
                regen_conv.messages.items,
            ) catch |err| {
                std.log.err("req={d} conversation format failed: {}", .{ log_request_id, err });
                if (removed_assistant) |m| regen_conv.restoreHeldMessage(g_server.allocator, m);
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to format conversation");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            // Rate limit while the reply is still recoverable.
            const regen_ids_owned = g_server.tokenizer.encode(regen_formatted) catch |err| {
                std.log.err("req={d} tokenizer encode failed: {}", .{ log_request_id, err });
                wipeFree(g_server.allocator, @constCast(regen_formatted));
                // Put the popped reply back and answer, so the turn survives
                // and the client sees a status line instead of a bare close.
                if (removed_assistant) |m| regen_conv.restoreHeldMessage(g_server.allocator, m);
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            const regen_prompt_tokens = estimatePromptTokens(regen_ids_owned.len, regen_formatted.len);
            if (checkRateLimit(g_server, regen_prompt_tokens)) |retry| {
                if (removed_assistant) |m| regen_conv.restoreHeldMessage(g_server.allocator, m);
                wipeFree(g_server.allocator, @constCast(regen_formatted));
                wipeFreeTokens(g_server.allocator, regen_ids_owned);
                send429(stream, retry);
                logRequestDone(method, path, 429, elapsedMs(request_start));
                break :blk null;
            }
            if (removed_assistant) |m| Conversation.freeOwnedMessage(g_server.allocator, m);

            break :blk RegenPrepResult{ .formatted = regen_formatted, .msg_count = regen_conv.messages.items.len, .prompt_ids = regen_ids_owned };
        };
        if (regen_prep == null) {
            g_server.releaseIdempotencyKey();
            return;
        }
        g_server.persistConversations();
        const regen_formatted = regen_prep.?.formatted;
        defer wipeFree(g_server.allocator, @constCast(regen_formatted));
        const regen_msg_count = regen_prep.?.msg_count;
        const regen_prompt_ids_owned = regen_prep.?.prompt_ids;
        defer wipeFreeTokens(g_server.allocator, regen_prompt_ids_owned);

        std.log.info("req={d} regenerate from {d} messages", .{ log_request_id, regen_msg_count });

        // Always reset KV cache for regeneration (full re-prefill)
        const wants_stream_regen = json.extractFormBool(regen_body, "stream");
        if (wants_stream_regen) {
            if (!sendSseHeaders(stream)) {
                g_server.metrics.recordCancellation();
                // The assistant message is already popped and persisted, so the
                // claim cannot be released: a retry would pop a second reply.
                g_server.completeIdempotencyKey("200 OK", "text/event-stream", "");
                return;
            }
            const regen_result = chatStreamGeneratePre(stream, regen_formatted, true, regen_max_tokens, regen_sampling, regen_prompt_ids_owned);
            defer wipeFree(g_server.allocator, regen_result.data);
            storeConversationResponse(regen_result.data, regen_result.stats);
            // Streamed bytes are not buffered for replay, but the key still
            // has to be marked done so a retry does not pop another message.
            g_server.completeIdempotencyKey("200 OK", "text/event-stream", "");
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Non-streaming regeneration
        const regen_result = generateNPre(regen_formatted, true, regen_max_tokens, regen_sampling, regen_prompt_ids_owned);
        defer wipeFree(g_server.allocator, regen_result.data);

        storeConversationResponse(regen_result.data, regen_result.stats);

        g_server.metrics.recordLatency(regen_result.stats.time_ms);
        g_server.metrics.recordTokens(regen_result.stats.tokens_generated);
        if (std.mem.eql(u8, regen_result.finish_reason, "error")) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

        // Never fall back to unescaped model output, OOM must not enable XSS (CWE-79).
        const regen_escaped = json.htmlEscape(g_server.allocator, regen_result.data) catch {
            sendHtml(stream, render_error_page);
            g_server.completeIdempotencyKey("200 OK", "text/html; charset=utf-8", render_error_page);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        };
        defer if (regen_escaped.ptr != regen_result.data.ptr) g_server.allocator.free(regen_escaped);
        var regen_html_buf: [response_buf_size]u8 = undefined;
        const regen_html = std.fmt.bufPrint(&regen_html_buf,
            \\<div class="msg assistant" data-tokens="{d}" data-time="{d}" data-tps="{d:.2}" data-prefill-tokens="{d}" data-prefill-ms="{d}" data-prefill-tps="{d:.1}">{s}</div>
        , .{ regen_result.stats.tokens_generated, regen_result.stats.time_ms, regen_result.stats.tokens_per_sec, regen_result.stats.prompt_tokens, regen_result.stats.prefill_ms, regen_result.stats.prefill_tps, regen_escaped }) catch "<div class=\"msg assistant\">Error</div>";
        sendHtml(stream, regen_html);
        g_server.completeIdempotencyKey("200 OK", "text/html; charset=utf-8", regen_html);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/chat")) {
        logRequest(method, path);

        g_server.metrics.recordRequest();

        const body = req.body;
        const msg = json.extractFormField(body, "message") orelse {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Missing required field: message", "message", "missing_required_parameter");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        if (msg.len > max_message_len) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Message too long", "message", "message_too_long");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const decoded = json.urlDecode(g_server.allocator, msg) catch g_server.allocator.dupe(u8, msg) catch return;
        defer wipeFree(g_server.allocator, decoded);
        if (decoded.len > max_message_len) {
            sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Message too long", "message", "message_too_long");
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }

        // Log message receipt without content (avoid leaking prompts in shared deployments)
        std.log.info("req={d} user message ({d} chars)", .{ log_request_id, decoded.len });

        // Check for attached image data (base64-encoded data URI from web UI)
        // If a vision encoder is available, decode and encode the image into
        // visual token embeddings that the model injects during generation.
        if (rejectBadImage(stream, body, true, false, request_start, method, path)) return;
        const chat_has_image = json.extractFormImage(body) != null and g_server.vision_encoder != null;
        if (chat_has_image) g_server.vision_mutex.lockUncancelable(g_server.io);
        defer if (chat_has_image) g_server.vision_mutex.unlock(g_server.io);
        var image_embedded = false;
        if (json.extractFormImage(body)) |b64_data| {
            if (g_server.vision_encoder) |ve| {
                if (processVisionImage(b64_data, ve)) {
                    image_embedded = true;
                } else {
                    std.log.warn("req={d} image attached but decode/encode failed (400, client error)", .{log_request_id});
                    sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Failed to decode or encode attached image", "image", "image_decode_failed");
                    g_server.metrics.recordClientError();
                    logRequestDone(method, path, 400, elapsedMs(request_start));
                    return;
                }
            }
        }
        // Ensure image embeddings are cleared after generation
        defer if (image_embedded) {
            clearPendingVision();
        };

        // Handle REPL-style commands in the chat interface
        const trimmed = std.mem.trim(u8, decoded, " \t\r\n");
        if (trimmed.len > 0 and trimmed[0] == '/') {
            const cmd_html = handleChatCommand(trimmed);
            if (cmd_html) |html| {
                sendHtml(stream, html);
                g_server.metrics.recordCompletion();
                logRequestDone(method, path, 200, elapsedMs(request_start));
                return;
            }
        }

        // Extract optional system prompt from form data (URL-decode since web UI sends encodeURIComponent)
        const system_field = json.extractFormField(body, "system");
        const system_decoded = if (system_field) |sf| (json.urlDecode(g_server.allocator, sf) catch |err| blk: {
            std.log.warn("req={d} system prompt URL decode failed: {}", .{ log_request_id, err });
            break :blk null;
        }) else null;
        defer if (system_decoded) |sd| wipeFree(g_server.allocator, sd);
        const system_prompt: ?[]const u8 = if (system_decoded) |sd| blk: {
            const s = std.mem.trim(u8, sd, " \t\r\n");
            break :blk if (s.len > 0) s else null;
        } else null;

        // Get or create active conversation, add user message, format prompt
        //, all under mutex. Returns (need_reset, formatted) or null on failure.
        // The turn is appended and persisted, so a retry after a lost response
        // would leave a second user turn and a second assistant reply behind.
        if (resolveIdempotency(stream, method, path, request_start)) return;
        const ChatPrepResult = struct { need_reset: bool, formatted: []const u8, prompt_ids: []u32 };
        const prep_result: ?ChatPrepResult = blk: {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);

            const conv = g_server.getActiveConv() orelse g_server.createConv() orelse {
                sendJsonErrorEx(stream, "503 Service Unavailable", "server_error", "Maximum conversation limit reached", null, "conversation_limit_reached");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 503, elapsedMs(request_start));
                break :blk null;
            };

            if (conv.messages.items.len >= max_messages_per_conv) {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Conversation message limit reached", null, "conversation_message_limit");
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                break :blk null;
            }

            const user_content = g_server.allocator.dupe(u8, trimmed) catch {
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Out of memory");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            conv.messages.append(g_server.allocator, .{ .role = .user, .content = user_content }) catch {
                wipeFree(g_server.allocator, user_content);
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Out of memory");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            // Title is set at createConv time (opaque "Chat {id}"); never store message text.

            const need_reset = !g_server.kv_valid;
            const formatted = if (need_reset)
                g_server.chat_template.formatConversation(g_server.allocator, system_prompt, conv.messages.items) catch |err| fmt_err: {
                    std.log.warn("req={d} chat template formatting failed (OOM), using raw input: {}", .{ log_request_id, err });
                    break :fmt_err trimmed;
                }
            else
                g_server.chat_template.formatContinuation(g_server.allocator, trimmed) catch |err| fmt_err: {
                    std.log.warn("req={d} chat continuation formatting failed (OOM), using raw input: {}", .{ log_request_id, err });
                    break :fmt_err trimmed;
                };
            // Rate limit before the durable append. A 429 after persisting
            // leaves a user message that no assistant ever answered, and it
            // replays as context on the next turn.
            const prompt_ids_owned = g_server.tokenizer.encode(formatted) catch |err| {
                std.log.err("req={d} tokenizer encode failed: {}", .{ log_request_id, err });
                // Roll the turn back like the 429 path below so the stored
                // conversation is what one run leaves and the retry the
                // released key invites starts from the same state, then
                // answer: a request that ends with no response leaves the
                // client hanging on a closed connection with no status line.
                const appended = conv.messages.pop();
                if (appended) |dropped| Conversation.freeOwnedMessage(g_server.allocator, dropped);
                if (formatted.ptr != trimmed.ptr) wipeFree(g_server.allocator, @constCast(formatted));
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            // Ownership moves to the caller on the success path only.
            var prompt_ids_owned_handed_off = false;
            defer if (!prompt_ids_owned_handed_off) wipeFreeTokens(g_server.allocator, prompt_ids_owned);
            const prompt_tokens = estimatePromptTokens(prompt_ids_owned.len, formatted.len);
            if (checkRateLimit(g_server, prompt_tokens)) |retry| {
                const popped = conv.messages.pop();
                if (popped) |dropped| Conversation.freeOwnedMessage(g_server.allocator, dropped);
                if (formatted.ptr != trimmed.ptr) wipeFree(g_server.allocator, @constCast(formatted));
                send429(stream, retry);
                logRequestDone(method, path, 429, elapsedMs(request_start));
                break :blk null;
            }

            prompt_ids_owned_handed_off = true;
            break :blk ChatPrepResult{ .need_reset = need_reset, .formatted = formatted, .prompt_ids = prompt_ids_owned };
        };
        if (prep_result == null) {
            g_server.releaseIdempotencyKey();
            return;
        }
        g_server.persistConversations();
        const need_reset = prep_result.?.need_reset;
        const formatted = prep_result.?.formatted;
        defer if (formatted.ptr != trimmed.ptr) wipeFree(g_server.allocator, @constCast(formatted));
        const chat_prompt_ids_owned = prep_result.?.prompt_ids;
        defer wipeFreeTokens(g_server.allocator, chat_prompt_ids_owned);

        // Parse optional sampling parameters from form body
        var chat_sampling = json.SamplingParams{};
        json.parseFormSampling(&chat_sampling, body);
        const chat_max_tokens = clampMaxTokens(json.extractFormInt(body, "max_tokens"));

        // SSE streaming mode: stream tokens to the client in real-time
        const wants_stream = json.extractFormBool(body, "stream");
        if (wants_stream) {
            if (!sendSseHeaders(stream)) {
                g_server.metrics.recordCancellation();
                // The user turn is already appended and persisted, so the claim
                // cannot be released: a retry would append the turn a second
                // time. Same treatment as a stream that ran and was recorded.
                g_server.completeIdempotencyKey("200 OK", "text/event-stream", "");
                return;
            }
            const result = chatStreamGeneratePre(stream, formatted, need_reset, chat_max_tokens, chat_sampling, chat_prompt_ids_owned);
            defer wipeFree(g_server.allocator, result.data);
            storeConversationResponse(result.data, result.stats);
            // Streamed bytes are not buffered for replay, but the key still
            // has to be marked done so a retry does not append a second turn.
            g_server.completeIdempotencyKey("200 OK", "text/event-stream", "");
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        const result = generateNPre(formatted, need_reset, chat_max_tokens, chat_sampling, chat_prompt_ids_owned);
        defer wipeFree(g_server.allocator, result.data);
        storeConversationResponse(result.data, result.stats);

        // Record metrics
        g_server.metrics.recordLatency(result.stats.time_ms);
        g_server.metrics.recordTokens(result.stats.tokens_generated);
        if (std.mem.eql(u8, result.finish_reason, "error")) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

        // Never fall back to unescaped input, send a safe error page on OOM (CWE-79).
        const escaped_user = json.htmlEscape(g_server.allocator, decoded) catch {
            sendHtml(stream, render_error_page);
            g_server.completeIdempotencyKey("200 OK", "text/html; charset=utf-8", render_error_page);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        };
        defer if (escaped_user.ptr != decoded.ptr) wipeFree(g_server.allocator, escaped_user);
        const escaped_resp = json.htmlEscape(g_server.allocator, result.data) catch {
            sendHtml(stream, render_error_page);
            g_server.completeIdempotencyKey("200 OK", "text/html; charset=utf-8", render_error_page);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        };
        defer if (escaped_resp.ptr != result.data.ptr) wipeFree(g_server.allocator, escaped_resp);
        var html_buf: [response_buf_size]u8 = undefined;
        const html = std.fmt.bufPrint(&html_buf,
            \\<div class="msg user">{s}</div><div class="msg assistant" data-tokens="{d}" data-time="{d}" data-tps="{d:.2}" data-prefill-tokens="{d}" data-prefill-ms="{d}" data-prefill-tps="{d:.1}">{s}</div>
        , .{ escaped_user, result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec, result.stats.prompt_tokens, result.stats.prefill_ms, result.stats.prefill_tps, escaped_resp }) catch "<div class=\"msg assistant\">Error</div>";
        sendHtml(stream, html);
        g_server.completeIdempotencyKey("200 OK", "text/html; charset=utf-8", html);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    // Check for known paths with wrong method -> 405 with Allow header
    for (known_endpoints) |ep| {
        if (std.mem.eql(u8, path, ep.path)) {
            logRequest(method, path);
            g_server.metrics.recordRequest();
            var hdr_buf: [hdr_buf_size]u8 = undefined;
            var body_buf: [error_body_buf_size]u8 = undefined;
            const body = if (ep.is_anthropic)
                std.fmt.bufPrint(&body_buf,
                    \\{{"type":"error","error":{{"type":"invalid_request_error","message":"Method not allowed. {s}"}}}}
                , .{ep.msg}) catch method_not_allowed_anthropic
            else
                std.fmt.bufPrint(&body_buf,
                    \\{{"error":{{"message":"Method not allowed. {s}","type":"invalid_request_error","param":null,"code":"method_not_allowed"}}}}
                , .{ep.msg}) catch method_not_allowed_openai;
            const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAllow: {s}\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ body.len, ep.allow, log_request_id, http.corsHeaders() }) catch {
                // Prefer a body without Allow over a silent drop (hung client).
                sendResponse(stream, "405 Method Not Allowed", "application/json", body);
                g_server.metrics.recordClientError();
                logRequestDone(method, path, 405, elapsedMs(request_start));
                return;
            };
            stream.writeAll(hdr) catch |err| {
                std.log.warn("req={d} 405 write failed: {}", .{ log_request_id, err });
                return;
            };
            stream.writeAll(body) catch |err| {
                std.log.warn("req={d} 405 write failed (body): {}", .{ log_request_id, err });
                return;
            };
            g_server.metrics.recordClientError();
            logRequestDone(method, path, 405, elapsedMs(request_start));
            return;
        }
    }

    logRequest(method, path);
    g_server.metrics.recordRequest();
    sendJsonErrorEx(stream, "404 Not Found", "invalid_request_error", "Unknown endpoint", null, "unknown_endpoint");
    g_server.metrics.recordClientError();
    logRequestDone(method, path, 404, elapsedMs(request_start));
}

/// Thread-local buffer for `/model` command response formatting.
threadlocal var cmd_buf: [cmd_buf_size]u8 = undefined;

fn handleChatCommand(cmd: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, cmd, "/clear")) {
        {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);
            lockModelWithScheduler();
            defer unlockModelWithScheduler();
            g_server.model.resetCache();
            g_server.kv_valid = false;
            // KV wiped, drop prefix IDs or the next generate would setKvSeqLen
            // onto empty slots and skip re-prefilling (garbled output).
            g_server.clearCachedPromptIds();
            if (g_server.getActiveConv()) |conv| conv.clearMessages(g_server.allocator);
        }
        g_server.persistConversations();
        if (ngram_mod.global_pool) |*pool| pool.clear();
        std.log.info("req={d} chat command /clear", .{log_request_id});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">Conversation cleared.</div>";
    }
    if (std.mem.eql(u8, cmd, "/reset")) {
        {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);
            lockModelWithScheduler();
            defer unlockModelWithScheduler();
            g_server.model.resetCache();
            g_server.kv_valid = false;
            g_server.clearCachedPromptIds();
            if (g_server.getActiveConv()) |conv| conv.clearMessages(g_server.allocator);
        }
        g_server.persistConversations();
        if (ngram_mod.global_pool) |*pool| pool.clear();
        std.log.info("req={d} chat command /reset", .{log_request_id});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">Conversation cleared.</div>";
    }
    if (std.mem.eql(u8, cmd, "/model")) {
        std.log.info("req={d} chat command /model", .{log_request_id});
        const escaped_name = json.htmlEscape(g_server.allocator, g_server.model_name) catch return null;
        defer if (escaped_name.ptr != g_server.model_name.ptr) g_server.allocator.free(escaped_name);
        return std.fmt.bufPrint(&cmd_buf,
            \\<div class="msg assistant" data-tokens="0" data-time="0" data-tps="0">Model: {s}</div>
        , .{escaped_name}) catch null;
    }
    if (std.mem.eql(u8, cmd, "/help")) {
        std.log.info("req={d} chat command /help", .{log_request_id});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">/clear &mdash; Clear conversation and KV cache&lt;br&gt;/stats &mdash; Toggle generation statistics&lt;br&gt;/model &mdash; Show model name&lt;br&gt;/help &mdash; Show available commands</div>";
    }
    return null;
}

/// Process a base64-encoded image from a web UI form submission.
///
/// Decodes URL-encoded base64 image bytes, decodes PNG to RGB, resizes to
/// the vision encoder's expected input size, and stashes visual token
/// embeddings on this handler thread. The scheduler copies them onto the
/// model under `model_mutex`; the direct path does the same after locking.
///
/// HTTP vision accepts PNG only (JPEG is detected and rejected; PPM is
/// CLI/`--image` only). Returns true on success, false on any failure.
fn processVisionImage(b64_raw: []const u8, ve: *VisionEncoder) bool {
    const allocator = g_server.allocator;
    std.log.info("req={d} vision: encoding image ({d} bytes base64)", .{ log_request_id, b64_raw.len });

    // URL-decode the base64 data (form fields encode +, /, = as %2B, %2F, %3D)
    const url_decoded = json.urlDecode(allocator, b64_raw) catch |err| {
        std.log.warn("req={d} vision: URL decode failed: {}", .{ log_request_id, err });
        return false;
    };
    defer {
        @memset(url_decoded, 0);
        allocator.free(url_decoded);
    }

    // Base64 decode to raw image bytes
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(url_decoded) catch |err| {
        std.log.warn("req={d} vision: base64 size calc failed: {}", .{ log_request_id, err });
        return false;
    };
    const image_bytes = allocator.alloc(u8, decoded_size) catch |err| {
        std.log.warn("req={d} vision: alloc failed for {d} bytes: {}", .{ log_request_id, decoded_size, err });
        return false;
    };
    defer {
        @memset(image_bytes, 0);
        allocator.free(image_bytes);
    }
    std.base64.standard.Decoder.decode(image_bytes, url_decoded) catch |err| {
        std.log.warn("req={d} vision: base64 decode failed: {}", .{ log_request_id, err });
        return false;
    };

    // Detect format and decode to RGB pixels
    const format = image_mod.detectFormat(image_bytes);
    switch (format) {
        .png => {
            var png = image_mod.decodePng(allocator, image_bytes) catch |err| {
                std.log.warn("req={d} vision: PNG decode failed: {}", .{ log_request_id, err });
                return false;
            };
            defer png.deinit();

            // Resize to vision encoder's expected input size
            const target = ve.image_size;
            const resized = image_mod.resize(allocator, png.pixels, png.width, png.height, target, target) catch |err| {
                std.log.warn("req={d} vision: resize failed: {}", .{ log_request_id, err });
                return false;
            };
            defer {
                @memset(resized, 0);
                allocator.free(resized);
            }

            const visual_tokens = ve.encode(resized) catch |err| {
                std.log.warn("req={d} vision: encode failed: {}", .{ log_request_id, err });
                return false;
            };
            if (ve.projection_dim == 0) {
                std.log.warn("req={d} vision: projection_dim is 0", .{log_request_id});
                return false;
            }
            const n_vis: u32 = @intCast(visual_tokens.len / ve.projection_dim);
            pending_visual_embeddings = visual_tokens;
            pending_visual_tokens = n_vis;
            std.log.info("req={d} vision: encoded {d} visual tokens ({d}x{d})", .{ log_request_id, n_vis, png.width, png.height });
            return true;
        },
        .jpeg => {
            std.log.warn("req={d} vision: JPEG images not supported, convert to PNG", .{log_request_id});
            return false;
        },
        else => {
            std.log.warn("req={d} vision: unrecognized image format (first bytes: {x:0>2}{x:0>2}{x:0>2}{x:0>2})", .{
                log_request_id,
                if (image_bytes.len > 0) image_bytes[0] else 0,
                if (image_bytes.len > 1) image_bytes[1] else 0,
                if (image_bytes.len > 2) image_bytes[2] else 0,
                if (image_bytes.len > 3) image_bytes[3] else 0,
            });
            return false;
        },
    }
}

/// Result of a single generation request: decoded text and timing statistics.
const GenResult = struct {
    data: []u8,
    stats: Stats,
    finish_reason: []const u8 = "stop",
};

/// Timing and throughput statistics for a completed generation.
const Stats = struct {
    tokens_generated: u32,
    prompt_tokens: u32,
    time_ms: u64,
    tokens_per_sec: f32,
    prefill_ms: u64,
    prefill_tps: f32,

    const zero: Stats = .{ .tokens_generated = 0, .prompt_tokens = 0, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 };
};

/// Run generation, log stats, and JSON-escape the output in one step.
/// Caller must call `deinit()` on the returned value.
const GeneratedEscaped = struct {
    raw: []u8,
    escaped: []const u8,
    stats: Stats,
    finish_reason: []const u8,
    owns_escaped: bool,

    /// Free the owned raw and (optionally) escaped buffers.
    /// Zeros contents first so generated text does not linger in the freelist.
    pub fn deinit(self: GeneratedEscaped) void {
        if (self.owns_escaped) {
            const esc = @constCast(self.escaped);
            @memset(esc, 0);
            g_server.allocator.free(esc);
        }
        @memset(self.raw, 0);
        g_server.allocator.free(self.raw);
    }
};

/// Store a generation result in the active conversation under mutex.
/// Sets kv_valid, logs generation stats, and appends the trimmed response
/// as an assistant message. Used by both streaming and non-streaming chat/regen paths.
fn storeConversationResponse(result_data: []const u8, stats: Stats) void {
    var arena_state = std.heap.ArenaAllocator.init(g_server.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var views: []conv_store.ConvView = &.{};
    var active_id: u32 = 0;
    var next_id: u32 = 0;

    {
        g_server.mutex.lockUncancelable(g_server.io);
        defer g_server.mutex.unlock(g_server.io);
        g_server.kv_valid = true;
        logGeneration(stats.tokens_generated, stats.time_ms, stats.tokens_per_sec);
        const trimmed = std.mem.trimEnd(u8, result_data, " \t\r\n");
        if (trimmed.len == 0) return;
        const duped = g_server.allocator.dupe(u8, trimmed) catch {
            std.log.warn("req={d} OOM storing response ({d} bytes)", .{ log_request_id, trimmed.len });
            return;
        };
        const conv = g_server.getActiveConv() orelse {
            @memset(duped, 0);
            g_server.allocator.free(duped);
            return;
        };
        conv.messages.append(g_server.allocator, .{ .role = .assistant, .content = duped }) catch {
            std.log.warn("req={d} OOM appending response to conversation", .{log_request_id});
            @memset(duped, 0);
            g_server.allocator.free(duped);
            return;
        };
        if (g_server.conv_store_path) |_| {
            views = g_server.snapshotConversationsLocked(arena) catch |err| {
                std.log.err("req={d} conversation snapshot failed: {}; history is not persisted", .{ log_request_id, err });
                return;
            };
            active_id = g_server.active_id;
            next_id = g_server.next_id;
        }
    }

    if (g_server.conv_store_path) |path| {
        g_server.saveConversationSnapshot(path, active_id, next_id, views);
    }
}

fn generateEscapedN(prompt: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams) GeneratedEscaped {
    return generateEscapedNPre(prompt, reset, max_tokens, sampling, null);
}

/// Like generateEscapedN, but reuses `pre_ids` when the caller already tokenized
/// (avoids a second BPE encode after rate-limit estimation).
fn generateEscapedNPre(prompt: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams, pre_ids: ?[]const u32) GeneratedEscaped {
    const result = generateNPre(prompt, reset, max_tokens, sampling, pre_ids);
    logGeneration(result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec);
    const escaped = json.jsonEscape(g_server.allocator, result.data) catch |err| {
        std.log.err("req={d} JSON escape OOM ({d} bytes generated): {}", .{ log_request_id, result.data.len, err });
        // Fail closed: empty escaped body with a success finish_reason would look like
        // a valid empty completion to API clients.
        return .{
            .raw = result.data,
            .escaped = "",
            .stats = result.stats,
            .finish_reason = "error",
            .owns_escaped = false,
        };
    };
    return .{
        .raw = result.data,
        .escaped = escaped,
        .stats = result.stats,
        .finish_reason = result.finish_reason,
        .owns_escaped = escaped.ptr != result.data.ptr,
    };
}

/// Run inference on a pre-formatted prompt string. When `reset` is true,
/// the KV cache is cleared and BOS is sent (first turn). When false, the
/// existing KV cache is reused (continuation turn).
fn generate(formatted: []const u8, reset: bool) GenResult {
    return generateN(formatted, reset, default_max_gen_tokens, .{});
}

/// Run inference with a configurable max_tokens limit and optional sampling.
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and blocks until completion. Falls back to direct model.forward()
/// when no scheduler is running.
fn generateN(formatted: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams) GenResult {
    return generateNPre(formatted, reset, max_tokens, sampling, null);
}

/// Like generateN, but reuses caller-owned `pre_ids` when non-null (skips encode).
fn generateNPre(formatted: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams, pre_ids: ?[]const u32) GenResult {
    const tok = g_server.tokenizer;
    const zero_stats = Stats.zero;
    var owned_ids: ?[]u32 = null;
    const raw_token_ids: []const u32 = if (pre_ids) |ids| ids else blk: {
        const encoded = tok.encode(formatted) catch |err| {
            std.log.err("req={d} tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
            return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        owned_ids = encoded;
        break :blk encoded;
    };
    defer if (owned_ids) |ids| g_server.allocator.free(ids);

    // Inject image placeholder tokens if visual embeddings are pending.
    // The model's forward() checks for pad_token_id and replaces those
    // embeddings with visual data, so the tokenized prompt must contain
    // the [start, pad*N, end] sequence.
    var injected_ids: ?[]u32 = null;
    if (pending_visual_tokens > 0 and g_server.image_pad_token_id != 0) {
        // Find correct insertion point: after the user prefix tokens.
        const prefix_tokens = g_server.userPrefixIds();
        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(raw_token_ids, prefix_tokens);
        injected_ids = chat_tmpl_mod.injectImageTokens(
            g_server.allocator,
            raw_token_ids,
            insert_pos,
            serverImageTokens(),
            pending_visual_tokens,
        ) catch |err| {
            std.log.err("req={d} image token injection failed: {}", .{ log_request_id, err });
            // Fail closed: vision embeddings are set but prompt lacks placeholders,
            // so continuing would silently drop the image from generation.
            return .{ .data = g_server.allocator.dupe(u8, "[image inject error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }
    defer if (injected_ids) |ids| g_server.allocator.free(ids);
    var token_ids: []const u32 = if (injected_ids) |ids| ids else raw_token_ids;
    // Apply truncation_side: if prompt exceeds ctx_size, trim left or right.
    if (g_server.ctx_size > 0 and token_ids.len > @as(usize, g_server.ctx_size)) {
        const orig_len = token_ids.len;
        const limit = @as(usize, g_server.ctx_size);
        token_ids = switch (sampling.truncation_side) {
            .right => token_ids[0..limit], // keep prefix
            .left => token_ids[token_ids.len - limit ..], // keep suffix (recency)
        };
        std.log.warn("req={d} prompt truncated {d}->{d} tokens ({s} side)", .{
            log_request_id,                     orig_len, limit,
            @tagName(sampling.truncation_side),
        });
    }
    const prompt_token_count: u32 = @intCast(token_ids.len);

    // Scheduler path: enqueue and block until complete.
    // Grammar-constrained decoding and json_mode bypass the scheduler (grammar
    // state is per-request and the scheduler loop has no grammar/JSON support,
    // no first-token '{' forcing, no brace depth tracking).
    const use_grammar_pre = (sampling.grammar_string != null or sampling.json_schema != null) and !sampling.json_mode;
    if (g_server.request_manager != null and !use_grammar_pre and !sampling.json_mode) {
        const rm = g_server.request_manager.?;
        const gen_start = milliTimestamp();
        const req = rm.enqueue(token_ids, log_request_id) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            return .{ .data = g_server.allocator.dupe(u8, "[enqueue error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        configureSchedulerSampling(req, sampling);
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        var stop_buf: [scheduler_stop_buf_size]u8 = undefined;
        var stop_len: usize = 0;
        var checked_len: usize = 0;
        var stop_token_count: ?u32 = null;

        // Block until request completes
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (req.visible_len.load(.acquire) >= max_tokens) {
                req.is_cancelled.store(true, .release);
                break;
            }
            if (pollSchedulerStop(req, tok, sampling, &stop_buf, &stop_len, &checked_len, g_server.allocator)) |n| {
                stop_token_count = n;
                break;
            }
            sleepNs(scheduler_poll_interval_ns);
        }

        // Wait for the scheduler to finish touching this request before reading
        // tokens. Otherwise a cancel that races mid-appendToken can publish a
        // visible_len that the scheduler is still writing past, or we decode
        // while radix_tree.insert still holds the slice.
        while (!req.scheduler_done.load(.acquire))
            sleepNs(scheduler_poll_interval_ns);

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const raw_count: u32 = stop_token_count orelse req.visible_len.load(.acquire);
        const token_count: u32 = @min(raw_count, @as(u32, @intCast(max_tokens)));
        const tokens_per_sec: f32 = tokensPerSec(token_count, time_ms);

        const safe_tokens = req.tokens.items[0..token_count];
        const decoded = tok.decode(safe_tokens) catch |err| d: {
            std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, safe_tokens.len, err });
            break :d g_server.allocator.dupe(u8, "[decode error]") catch @as([]u8, &.{});
        };

        // Record TTFT from scheduler's per-request prefill timestamp
        var sched_prefill_ms: u64 = 0;
        if (req.prefill_done_at > 0) {
            sched_prefill_ms = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(sched_prefill_ms, prompt_token_count);
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        // TPOT is decode-only. gen_start precedes the enqueue here, so time_ms
        // carries queue wait and prefill; subtracting the prefill span keeps
        // this series comparable with the direct path.
        g_server.metrics.recordTPOT(token_count, time_ms -| sched_prefill_ms);
        g_server.metrics.recordPromptTokens(prompt_token_count);
        g_server.metrics.recordGenerationTokens(token_count);

        const finished = req.is_finished.load(.acquire);
        const timed_out = req.is_timed_out.load(.acquire);
        const finish_reason: []const u8 = if (finished or stop_token_count != null)
            "stop"
        else if (token_count >= max_tokens)
            "length"
        else if (timed_out) blk: {
            // Scheduler already called recordTimeout(); treat as length so callers
            // do not recordFailure and poison /ready error-rate degradation.
            std.log.warn("req={d} scheduler generation timed out (tokens={d})", .{
                log_request_id,
                token_count,
            });
            break :blk "length";
        } else blk: {
            // Cancelled for forward failure, capacity, or client abort,
            // scheduler already logged the cause; surface as error for handlers.
            std.log.warn("req={d} scheduler generation incomplete (tokens={d}, cancelled={})", .{
                log_request_id,
                token_count,
                req.is_cancelled.load(.acquire),
            });
            break :blk "error";
        };

        return .{
            .data = decoded,
            .finish_reason = finish_reason,
            .stats = .{
                .tokens_generated = token_count,
                .prompt_tokens = prompt_token_count,
                .time_ms = time_ms,
                .tokens_per_sec = tokens_per_sec,
                // Same span recordTTFT was fed, not zero: the caller renders
                // this as the request's prefill time.
                .prefill_ms = sched_prefill_ms,
                .prefill_tps = tokensPerSec(prompt_token_count, sched_prefill_ms),
            },
        };
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    const model = g_server.model;
    const n_visual = pending_visual_tokens;
    applyPendingVisionEmbeddings();

    // Re-check kv_valid under mutex, the caller's `reset` flag may be stale
    // if another thread invalidated the cache (e.g. /clear) between the caller's
    // unlock and this lock acquisition.
    const actual_reset = reset or !g_server.kv_valid;

    // Prompt prefix caching: find longest common prefix with cached state.
    // If the new prompt shares a prefix with the previous one, skip re-prefilling
    // those tokens (KV cache already has them). Roll back KV to the shared prefix
    // length and only process new tokens.
    var prefix_len: usize = 0;
    if (actual_reset and prefixCacheApplies(n_visual) and g_server.cached_prompt_ids.len > 0 and token_ids.len > 0) {
        const max_match = @min(g_server.cached_prompt_ids.len, token_ids.len);
        while (prefix_len < max_match and g_server.cached_prompt_ids[prefix_len] == token_ids[prefix_len]) {
            prefix_len += 1;
        }
        if (prefix_len > 0 and prefix_len < token_ids.len) {
            // Shared prefix found, rollback KV cache to prefix boundary
            const bos_offset: usize = if (g_server.bos_token_id > 0) 1 else 0;
            model.setKvSeqLen(prefix_len + bos_offset);
            std.log.info("req={d} prefix cache hit: {d}/{d} tokens reused", .{ log_request_id, prefix_len, token_ids.len });
        } else {
            prefix_len = 0;
        }
    }

    if (prefix_len == 0 and actual_reset) {
        model.resetCache();
        // Drop stale prefix IDs until prefill succeeds and replaces them below.
        // Otherwise a failed BOS/prefill leaves IDs pointing at a wiped cache.
        g_server.clearCachedPromptIds();
    }

    // BOS token, required by models like Gemma to initialize state correctly
    if (prefix_len == 0 and actual_reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            return .{ .data = g_server.allocator.dupe(u8, "[BOS forward error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }

    // Prefill phase, timed separately for TTFT stats.
    // Skip tokens already in KV cache (prefix_len > 0 = cache hit).
    const prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids[prefix_len..]) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                std.log.info("req={d} prefill cancelled", .{log_request_id});
                invalidateKvBookkeeping();
                return .{ .data = g_server.allocator.dupe(u8, "[cancelled]") catch &.{}, .finish_reason = "stop", .stats = .{ .tokens_generated = 0, .prompt_tokens = prompt_token_count, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 } };
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            return .{ .data = g_server.allocator.dupe(u8, "[prefill error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }

    // Cache the prompt token IDs for next request's prefix matching (zeros old IDs).
    g_server.publishCachedPromptIds(token_ids, n_visual);
    const prefill_ms: u64 = elapsedMs(prefill_start);
    const prefill_tps: f32 = tokensPerSec(prompt_token_count, prefill_ms);
    g_server.metrics.recordTTFT(prefill_ms, prompt_token_count);

    // Apply sampling to first generated token (from prefill's last forward call)
    const use_sampling = sampling.temperature > 0;
    const prng_seed = prngSeedFromSampling(sampling);
    noteAutoSeed(sampling, prng_seed);
    var prng = std.Random.Xoshiro256.init(prng_seed);
    var mirostat_mu: f32 = sampling.mirostat_tau * 2.0;
    var json_depth: i32 = 0;

    // Grammar-constrained decoding: parse GBNF and init state
    var grammar_storage: ?grammar_mod.Grammar = null;
    var grammar_state_storage: ?grammar_mod.GrammarState = null;
    defer {
        if (grammar_state_storage) |*gs| gs.deinit();
        if (grammar_storage) |*g| g.deinit();
    }
    const use_grammar = (sampling.grammar_string != null or sampling.json_schema != null) and !sampling.json_mode;
    if (sampling.json_schema) |schema| {
        grammar_storage = grammar_mod.Grammar.fromJsonSchema(g_server.allocator, schema) catch |err| blk: {
            std.log.err("req={d} json_schema grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (grammar_storage) |*g| grammar_state_storage = g.initState() catch |err| blk: {
            std.log.err("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    } else if (sampling.grammar_string) |gs| {
        // JSON-unescape the grammar string: extractField returns raw JSON content
        // with escape sequences intact (e.g., \" for literal quotes in GBNF).
        const unescaped_gs = json.jsonUnescape(g_server.allocator, gs) catch gs;
        defer if (unescaped_gs.ptr != gs.ptr) g_server.allocator.free(@constCast(unescaped_gs));
        grammar_storage = grammar_mod.Grammar.parse(g_server.allocator, unescaped_gs) catch |err| blk: {
            std.log.err("req={d} grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (grammar_storage) |*g| grammar_state_storage = g.initState() catch |err| blk: {
            std.log.err("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    }
    // Fail closed: never generate unconstrained output when the client asked for a grammar.
    if (use_grammar and (grammar_storage == null or grammar_state_storage == null)) {
        return .{ .data = g_server.allocator.dupe(u8, "[grammar error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
    }

    const vocab_texts = g_server.tokenizer.getVocabTexts();

    if (token_ids.len > 0) {
        const first_logits = model.getLogits();

        // Grammar masking on first token
        if (use_grammar) {
            if (grammar_storage) |*g| {
                if (grammar_state_storage) |*gs| {
                    g.maskLogits(gs, first_logits, vocab_texts) catch {
                        std.log.warn("req={d} grammar mask OOM", .{log_request_id});
                        return .{ .data = g_server.allocator.dupe(u8, "[grammar OOM]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
                    };
                }
            }
        }

        // JSON mode: force first token to start with {
        if (sampling.json_mode) {
            for (first_logits, 0..) |*logit, tid| {
                if (tid >= vocab_texts.len) break;
                const t = grammar_mod.Grammar.getEffectiveText(vocab_texts[tid]);
                if (t.len == 0 or t[0] != '{') logit.* = -std.math.inf(f32);
            }
        }
        if (use_sampling) {
            if (sampling.min_p > 0) math_ops.applyMinP(first_logits, sampling.min_p);
            first_gen_token = math_ops.sampleToken(first_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng.random());
        } else if (sampling.json_mode or use_grammar) {
            first_gen_token = math_ops.argmax(first_logits);
        }
        // Accept first token in grammar state
        if (use_grammar and grammar_state_storage != null) {
            // Use raw vocab text, NOT decoded text, so getEffectiveText strips
            // BPE prefixes consistently with maskLogits.
            const vt = g_server.tokenizer.getVocabTexts();
            const raw = if (first_gen_token < vt.len) vt[first_gen_token] else "";
            grammar_state_storage.?.acceptToken(raw);
        }
        // Track JSON depth for first token
        if (sampling.json_mode) {
            const tok_slice = [1]u32{first_gen_token};
            const text = g_server.tokenizer.decode(@constCast(&tok_slice)) catch |err| blk: {
                std.log.warn("req={d} json token decode failed (id={d}): {}", .{ log_request_id, first_gen_token, err });
                break :blk null;
            };
            defer if (text) |t| g_server.allocator.free(t);
            if (text) |t| {
                for (t) |ch| {
                    if (ch == '{' or ch == '[') json_depth += 1;
                    if (ch == '}' or ch == ']') json_depth -= 1;
                }
            }
        }
    }

    // Generation phase (timed), collect token IDs, batch-decode once at the end
    // to avoid per-token alloc/free overhead.
    const gen_start = milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    defer @memset(std.mem.sliceAsBytes(&gen_tokens), 0);
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;
    var cancelled = false;
    var g_in_think_block: bool = false;
    var g_n_think_tokens: u32 = 0;

    // Pre-tokenize </think> once, encoding inside the decode loop allocates every step.
    var close_think_owned: ?[]u32 = null;
    defer if (close_think_owned) |ids| g_server.allocator.free(ids);
    if (sampling.thinking_budget_tokens > 0) {
        close_think_owned = g_server.tokenizer.encode("</think>") catch null;
    }
    const close_think_ids: []const u32 = close_think_owned orelse &.{};
    const vocab_texts_gen = g_server.tokenizer.getVocabTexts();

    // Include first generated token (from last prefill forward)
    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
    }

    var hit_eog = first_is_eog;
    var forward_failed = false;
    const effective_max = @min(max_tokens, gen_ids_buf_size);

    // Rolling text buffer for stop sequence matching, avoids re-decoding
    // a window of tokens each iteration. 128 bytes covers stop sequences
    // that span up to ~32 tokens.
    const stop_buf_size: usize = 128;
    var stop_text_buf: [stop_buf_size]u8 = undefined;
    defer @memset(&stop_text_buf, 0);
    var stop_text_len: usize = 0;

    const has_draft = g_server.draft_model != null and !first_is_eog and token_ids.len > 0;
    var spec_state_storage: spec_decode.SpecState = undefined;
    var spec_state_valid = false;
    if (has_draft) {
        if (spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize())) |ok| {
            spec_state_storage = ok;
            spec_state_valid = true;
        } else |_| {
            spec_state_storage = spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        }
    }
    defer if (spec_state_valid) spec_state_storage.deinit(g_server.allocator);

    if (spec_state_valid) {
        var spec_state = &spec_state_storage;
        var draft_model = g_server.draft_model.?;

        // Prefill draft model if separate
        if (draft_model.ptr != model.ptr) {
            _ = draft_model.prefill(token_ids) catch |err| {
                std.log.warn("req={d} draft model prefill failed: {s}", .{ log_request_id, @errorName(err) });
            };
        }

        while (token_count < effective_max and !hit_eog) {
            const pre_draft_pos = model.kvSeqLen();
            const is_self = (model.ptr == draft_model.ptr);
            const spec_pen: spec_decode.Penalties = .{
                .history = gen_tokens[0..token_count],
                .repeat_penalty = sampling.repetition_penalty,
                .dry_multiplier = sampling.dry_multiplier,
                .dry_length = sampling.dry_allowed_length,
            };

            const n_drafted = if (is_self and !use_sampling)
                spec_decode.draft(spec_state, draft_model, last)
            else
                spec_decode.draftWithLogits(spec_state, draft_model, last);
            if (n_drafted == 0) break;

            const result = if (is_self)
                spec_decode.SpecResult{ .accepted = spec_state.n_draft, .next_token = blk: {
                    spec_state.recordRound(spec_state.n_draft);
                    const ld = spec_state.draft_tokens[spec_state.n_draft - 1];
                    break :blk model.forward(ld) catch ld;
                } }
            else if (use_sampling)
                spec_decode.verifySampling(spec_state, model, draft_model, last, pre_draft_pos, sampling.temperature, spec_pen, prng.random())
            else
                spec_decode.verifySequential(spec_state, model, draft_model, last, pre_draft_pos, spec_pen);

            for (0..result.accepted) |i| {
                const accepted_tok = spec_state.draft_tokens[i];
                if (g_server.isEog(accepted_tok)) {
                    hit_eog = true;
                    break;
                }
                if (token_count >= effective_max) break;
                gen_tokens[token_count] = accepted_tok;
                token_count += 1;
            }
            if (!hit_eog and token_count < effective_max) {
                if (g_server.isEog(result.next_token)) {
                    hit_eog = true;
                } else {
                    gen_tokens[token_count] = result.next_token;
                    token_count += 1;
                }
            }
            last = if (hit_eog) model.eosId() else result.next_token;
        }
    } else {
        // Standard single-token generation
        for (0..effective_max -| 1) |_| {
            if (first_is_eog or token_ids.len == 0) break;

            // Jump decoding: if grammar allows exactly one token, skip forward pass
            if (use_grammar and !use_sampling) {
                if (grammar_storage) |*g| {
                    if (grammar_state_storage) |*gs| {
                        if (g.singleValidToken(gs, vocab_texts)) |jump_tok| {
                            const jt_raw = if (jump_tok < vocab_texts.len) vocab_texts[jump_tok] else "";
                            gs.acceptToken(jt_raw);
                            gen_tokens[token_count] = jump_tok;
                            token_count += 1;
                            last = jump_tok;
                            if (gs.isComplete()) {
                                hit_eog = true;
                                break;
                            }
                            continue;
                        }
                    }
                }
            }

            var next = model.forward(last) catch |err| {
                if (err == error.Cancelled) {
                    cancelled = true;
                } else {
                    std.log.warn("req={d} generation forward failed: {}", .{ log_request_id, err });
                    forward_failed = true;
                }
                break;
            };
            const logits = model.getLogits();
            // Grammar masking before sampling
            if (use_grammar) {
                if (grammar_storage) |*g| {
                    if (grammar_state_storage) |*gs| {
                        g.maskLogits(gs, logits, vocab_texts) catch {
                            std.log.warn("req={d} grammar mask OOM", .{log_request_id});
                            forward_failed = true;
                            break;
                        };
                        next = math_ops.argmax(logits);
                    }
                }
            }
            if (sampling.logit_bias_count > 0) {
                math_ops.applyLogitBias(logits, &sampling.logit_bias_ids, &sampling.logit_bias_vals, sampling.logit_bias_count);
            }
            // Thinking budget: bias towards </think> when inside think block and budget exceeded.
            if (sampling.thinking_budget_tokens > 0 and g_n_think_tokens >= sampling.thinking_budget_tokens and g_in_think_block) {
                for (close_think_ids) |tid| {
                    if (tid < @as(u32, @intCast(logits.len))) logits[tid] += 100.0;
                }
            }
            if (sampling.repetition_penalty != 1.0 and token_count > 0) {
                math_ops.applyRepeatPenalty(logits, gen_tokens[0..token_count], sampling.repetition_penalty);
            }
            if (sampling.dry_multiplier > 0 and token_count > 0) {
                math_ops.applyDry(logits, gen_tokens[0..token_count], sampling.dry_multiplier, sampling.dry_allowed_length);
            }
            if (sampling.frequency_penalty != 0 or sampling.presence_penalty != 0) {
                math_ops.applyPenalties(logits, gen_tokens[0..token_count], sampling.frequency_penalty, sampling.presence_penalty);
            }
            if ((sampling.repetition_penalty != 1.0 or sampling.frequency_penalty != 0 or sampling.presence_penalty != 0 or sampling.dry_multiplier > 0) and !use_grammar and !use_sampling) {
                next = math_ops.argmax(logits);
            }
            if (use_sampling and !use_grammar) {
                if (sampling.mirostat >= 2) {
                    next = math_ops.sampleMirostat(logits, sampling.mirostat_tau, sampling.mirostat_eta, &mirostat_mu, sampling.temperature, prng.random());
                } else {
                    if (sampling.min_p > 0) math_ops.applyMinP(logits, sampling.min_p);
                    if (sampling.xtc_probability > 0) math_ops.applyXtc(logits, sampling.xtc_probability, sampling.xtc_threshold, prng.random());
                    next = math_ops.sampleToken(logits, sampling.temperature, sampling.top_k, sampling.top_p, prng.random());
                }
            }
            if (g_server.isEog(next)) {
                hit_eog = true;
                break;
            }
            gen_tokens[token_count] = next;
            // Decode token text for grammar/JSON/stop checks
            const needs_text = use_grammar or sampling.json_mode or sampling.hasStop();
            // Decode into a stack buffer; only a token whose text exceeds it
            // falls back to the allocating path, so the common case costs no
            // allocation per token.
            var tok_text_buf: [stream_decode_buf_size]u8 = undefined;
            var tok_text_alloc: ?[]u8 = null;
            var tok_decode_ok = true;
            var tok_text: []const u8 = "";
            if (needs_text) {
                if (g_server.tokenizer.decodeOne(next, &tok_text_buf)) |decoded| {
                    tok_text = decoded;
                } else {
                    const tok_slice = [1]u32{next};
                    tok_text_alloc = g_server.tokenizer.decode(@constCast(&tok_slice)) catch |err| blk2: {
                        std.log.warn("req={d} token decode failed (id={d}): {}", .{ log_request_id, next, err });
                        break :blk2 null;
                    };
                    tok_text = tok_text_alloc orelse "";
                    tok_decode_ok = tok_text_alloc != null;
                }
            }
            defer if (tok_text_alloc) |t| g_server.allocator.free(t);
            // Fail closed: accepting "" on decode failure corrupts grammar state.
            if (use_grammar and needs_text and !tok_decode_ok) {
                std.log.warn("req={d} grammar token decode failed, aborting generation", .{log_request_id});
                forward_failed = true;
                break;
            }
            // Accept token in grammar state, use raw vocab text for consistent BPE handling.
            if (use_grammar and grammar_state_storage != null) {
                const raw_tok = if (next < vocab_texts.len) vocab_texts[next] else "";
                grammar_state_storage.?.acceptToken(raw_tok);
                if (grammar_state_storage.?.isComplete()) {
                    token_count += 1;
                    hit_eog = true;
                    break;
                }
            }
            // JSON mode: stop at balanced braces
            if (sampling.json_mode) {
                for (tok_text) |ch| {
                    if (ch == '{' or ch == '[') json_depth += 1;
                    if (ch == '}' or ch == ']') json_depth -= 1;
                }
                if (json_depth <= 0) {
                    token_count += 1;
                    hit_eog = true;
                    break;
                }
            }
            // Stop sequence check using rolling text buffer (no per-token allocation)
            if (sampling.hasStop() and tok_text.len > 0) {
                if (tok_text.len >= stop_buf_size) {
                    @memcpy(&stop_text_buf, tok_text[tok_text.len - stop_buf_size ..]);
                    stop_text_len = stop_buf_size;
                } else if (stop_text_len + tok_text.len <= stop_buf_size) {
                    @memcpy(stop_text_buf[stop_text_len..][0..tok_text.len], tok_text);
                    stop_text_len += tok_text.len;
                } else {
                    const keep = stop_buf_size - tok_text.len;
                    std.mem.copyForwards(u8, stop_text_buf[0..keep], stop_text_buf[stop_text_len - keep .. stop_text_len]);
                    @memcpy(stop_text_buf[keep..][0..tok_text.len], tok_text);
                    stop_text_len = stop_buf_size;
                }
                if (sampling.matchesStop(stop_text_buf[0..stop_text_len])) {
                    token_count += 1;
                    hit_eog = true;
                    break;
                }
            }
            // Update thinking block state for budget tracking (vocab lookup, no per-token alloc).
            if (sampling.thinking_budget_tokens > 0 and next < vocab_texts_gen.len) {
                const tk_text = vocab_texts_gen[next];
                if (std.mem.indexOf(u8, tk_text, "<think>") != null) g_in_think_block = true;
                if (std.mem.indexOf(u8, tk_text, "</think>") != null) g_in_think_block = false;
                if (g_in_think_block) g_n_think_tokens += 1;
            }
            last = next;
            token_count += 1;
        }
    }

    const gen_end = milliTimestamp();
    const time_ms = elapsedBetween(gen_start, gen_end);
    const tokens_per_sec = tokensPerSec(token_count, time_ms);
    const finish_reason: []const u8 = if (cancelled) "stop" else if (forward_failed) "error" else if (hit_eog) "stop" else "length";
    g_server.metrics.recordThroughput(token_count, time_ms);
    g_server.metrics.recordTPOT(token_count, time_ms);
    g_server.metrics.recordPromptTokens(prompt_token_count);
    g_server.metrics.recordGenerationTokens(token_count);

    // Terminal metrics (recordFailure/recordCompletion) recorded by caller based on finish_reason.

    // Single batch decode, one alloc instead of N per-token allocs
    const decoded = tok.decode(gen_tokens[0..token_count]) catch |err| d: {
        std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, token_count, err });
        break :d g_server.allocator.dupe(u8, "[decode error]") catch @as([]u8, &.{});
    };

    return .{
        .data = decoded,
        .finish_reason = finish_reason,
        .stats = .{
            .tokens_generated = token_count,
            .prompt_tokens = prompt_token_count,
            .time_ms = time_ms,
            .tokens_per_sec = tokens_per_sec,
            .prefill_ms = prefill_ms,
            .prefill_tps = prefill_tps,
        },
    };
}

/// Run inference for the web UI chat with SSE token streaming.
/// Streams each decoded token as `data: {"t":"..."}` events.
/// Sends final stats as `data: {"done":true,...}` followed by `data: [DONE]`.
/// Returns GenResult with accumulated decoded text for conversation storage.
/// When the scheduler is active, routes through RequestManager.enqueue().
fn chatStreamGenerate(stream: http.TcpStream, formatted: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams) GenResult {
    return chatStreamGeneratePre(stream, formatted, reset, max_tokens, sampling, null);
}

/// Like chatStreamGenerate, but reuses caller-owned `pre_ids` when non-null.
fn chatStreamGeneratePre(stream: http.TcpStream, formatted: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams, pre_ids: ?[]const u32) GenResult {
    const tok = g_server.tokenizer;
    const zero_stats = Stats.zero;
    var owned_ids_cs: ?[]u32 = null;
    const raw_token_ids: []const u32 = if (pre_ids) |ids| ids else blk: {
        const encoded = tok.encode(formatted) catch |err| {
            std.log.err("req={d} chat stream tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "{\"t\":\"[encode error]\",\"done\":true}");
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        owned_ids_cs = encoded;
        break :blk encoded;
    };
    defer if (owned_ids_cs) |ids| g_server.allocator.free(ids);

    // Inject image placeholder tokens if visual embeddings are pending.
    var injected_ids_cs: ?[]u32 = null;
    if (pending_visual_tokens > 0 and g_server.image_pad_token_id != 0) {
        // Find correct insertion point: after the user prefix tokens.
        const cs_prefix = g_server.userPrefixIds();
        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(raw_token_ids, cs_prefix);
        injected_ids_cs = chat_tmpl_mod.injectImageTokens(
            g_server.allocator,
            raw_token_ids,
            insert_pos,
            serverImageTokens(),
            pending_visual_tokens,
        ) catch |err| {
            std.log.err("req={d} image token injection failed ({d} visual tokens): {}", .{ log_request_id, pending_visual_tokens, err });
            // Fail closed: same as generateNPre, do not silently drop the image.
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "{\"t\":\"[image inject error]\",\"done\":true}");
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "[image inject error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }
    defer if (injected_ids_cs) |ids| g_server.allocator.free(ids);
    const token_ids: []const u32 = if (injected_ids_cs) |ids| ids else raw_token_ids;
    const prompt_token_count: u32 = @intCast(token_ids.len);

    // Scheduler path: grammar/json_schema/json_mode requests bypass (no grammar/JSON support in scheduler).
    const use_grammar_cs = (sampling.grammar_string != null or sampling.json_schema != null) and !sampling.json_mode;
    if (g_server.request_manager != null and !use_grammar_cs and !sampling.json_mode) {
        const rm = g_server.request_manager.?;
        const gen_start = milliTimestamp();
        const req = rm.enqueue(token_ids, log_request_id) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        configureSchedulerSampling(req, sampling);
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        var streamed_count: usize = 0;
        var client_connected = true;
        var stop_buf: [scheduler_stop_buf_size]u8 = undefined;
        var stop_len: usize = 0;
        var checked_len: usize = 0;
        var hit_stop = false;
        var chat_hb = Utf8Holdback{};
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (pollSchedulerStop(req, tok, sampling, &stop_buf, &stop_len, &checked_len, g_server.allocator)) |_| {
                hit_stop = true;
                break;
            }
            // Stream only tokens already stop-checked (checked_len == visible when no stop).
            const stream_limit = if (sampling.hasStop()) checked_len else req.visible_len.load(.acquire);
            while (streamed_count < stream_limit) {
                if (!streamToken(stream, tok, req.tokens.items[streamed_count], &chat_hb)) {
                    client_connected = false;
                    req.is_cancelled.store(true, .release);
                    break;
                }
                streamed_count += 1;
                if (streamed_count >= max_tokens) {
                    req.is_cancelled.store(true, .release);
                    break;
                }
            }
            if (!client_connected or streamed_count >= max_tokens) break;
            sleepNs(scheduler_poll_interval_ns);
        }
        // Wait for scheduler to finish before draining/decoding (same race as generateN).
        while (!req.scheduler_done.load(.acquire))
            sleepNs(scheduler_poll_interval_ns);
        // Drain remaining tokens up to stop / max_tokens
        const final_len = if (hit_stop) checked_len else req.visible_len.load(.acquire);
        while (client_connected and streamed_count < final_len and streamed_count < max_tokens) {
            if (!streamToken(stream, tok, req.tokens.items[streamed_count], &chat_hb)) break;
            streamed_count += 1;
        }
        if (client_connected) _ = flushStreamHoldback(stream, &chat_hb, emitChatStreamEvent);

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const token_count: u32 = if (hit_stop) @intCast(checked_len) else req.visible_len.load(.acquire);
        const tps: f32 = tokensPerSec(token_count, time_ms);

        var stats_buf: [stats_buf_size]u8 = undefined;
        const stats_json = std.fmt.bufPrint(&stats_buf,
            \\{{"done":true,"n":{d},"ms":{d},"tps":{d:.2},"pn":{d},"pms":0,"ptps":0.0}}
        , .{ token_count, time_ms, tps, prompt_token_count }) catch "";
        if (stats_json.len > 0) _ = sseWriteData(stream, stats_json);
        _ = sseWriteData(stream, "[DONE]");

        g_server.metrics.recordLatency(time_ms);
        g_server.metrics.recordTokens(token_count);
        // Record TTFT from scheduler's per-request prefill timestamp
        var sched_prefill_ms: u64 = 0;
        if (req.prefill_done_at > 0) {
            sched_prefill_ms = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(sched_prefill_ms, prompt_token_count);
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        // Decode-only span: see the non-streaming scheduler path above.
        g_server.metrics.recordTPOT(token_count, time_ms -| sched_prefill_ms);
        g_server.metrics.recordPromptTokens(prompt_token_count);
        g_server.metrics.recordGenerationTokens(token_count);
        // Match other stream paths: max_tokens is successful completion, not failure.
        // Timeouts are already counted by recordTimeout(); do not poison requests_failed.
        if (!client_connected) {
            std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
            g_server.metrics.recordCancellation();
        } else if (req.is_finished.load(.acquire) or streamed_count >= max_tokens)
            g_server.metrics.recordCompletion()
        else if (req.is_timed_out.load(.acquire))
            std.log.warn("req={d} chat stream timed out (tokens={d})", .{ log_request_id, token_count })
        else {
            std.log.warn("req={d} chat stream incomplete (tokens={d}, cancelled={})", .{
                log_request_id,
                token_count,
                req.is_cancelled.load(.acquire),
            });
            g_server.metrics.recordFailure();
        }

        const sched_prefill_tps: f32 = tokensPerSec(prompt_token_count, sched_prefill_ms);
        // Bound by visible_len only, items.len is not synchronized with the
        // scheduler writer; visible_len's acquire load is the publication fence.
        const safe_cs_tokens = req.tokens.items[0..token_count];
        const decoded = tok.decode(safe_cs_tokens) catch |err| d: {
            std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, safe_cs_tokens.len, err });
            break :d g_server.allocator.dupe(u8, "") catch @as([]u8, &.{});
        };
        return .{
            .data = decoded,
            .stats = .{ .tokens_generated = token_count, .prompt_tokens = prompt_token_count, .time_ms = time_ms, .tokens_per_sec = tps, .prefill_ms = sched_prefill_ms, .prefill_tps = sched_prefill_tps },
        };
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    const model = g_server.model;
    applyPendingVisionEmbeddings();
    // Re-check kv_valid under mutex, caller's `reset` may be stale if another
    // thread invalidated the cache between the caller's unlock and this lock.
    const actual_reset = reset or !g_server.kv_valid;
    if (actual_reset) {
        model.resetCache();
        g_server.clearCachedPromptIds();
    }

    if (actual_reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }

    // Prefill
    const prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                std.log.info("req={d} chat stream prefill cancelled", .{log_request_id});
                invalidateKvBookkeeping();
                g_server.metrics.recordCancellation();
                _ = sseWriteData(stream, "[DONE]");
                return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .finish_reason = "stop", .stats = zero_stats };
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "{\"t\":\"[prefill error]\",\"done\":true}");
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "[prefill error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }
    const prefill_ms: u64 = elapsedMs(prefill_start);
    const prefill_tps: f32 = tokensPerSec(prompt_token_count, prefill_ms);
    g_server.metrics.recordTTFT(prefill_ms, prompt_token_count);

    // Apply sampling to first generated token (from prefill's last forward call)
    const use_sampling = sampling.temperature > 0;
    const prng_seed_cs = prngSeedFromSampling(sampling);
    noteAutoSeed(sampling, prng_seed_cs);
    var prng_cs = std.Random.Xoshiro256.init(prng_seed_cs);
    if (use_sampling and token_ids.len > 0) {
        const cs_logits = model.getLogits();
        if (sampling.min_p > 0) math_ops.applyMinP(cs_logits, sampling.min_p);
        first_gen_token = math_ops.sampleToken(cs_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng_cs.random());
    }

    // Generate and stream tokens
    const gen_start = milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    defer @memset(std.mem.sliceAsBytes(&gen_tokens), 0);
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    var client_disconnected = false;
    var chat_hb = Utf8Holdback{};
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
        // Stream first token
        client_disconnected = !streamToken(stream, tok, first_gen_token, &chat_hb);
    }

    var forward_failed = false;
    const effective_max = @min(max_tokens, gen_ids_buf_size);
    for (0..effective_max -| 1) |_| {
        if (client_disconnected or first_is_eog or token_ids.len == 0) break;
        var next = model.forward(last) catch |err| {
            if (err != error.Cancelled) {
                std.log.warn("req={d} generation forward failed: {}", .{ log_request_id, err });
                forward_failed = true;
            }
            break;
        };
        if (use_sampling) {
            const cs_next_logits = model.getLogits();
            if (sampling.min_p > 0) math_ops.applyMinP(cs_next_logits, sampling.min_p);
            next = math_ops.sampleToken(cs_next_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng_cs.random());
        }
        if (g_server.isEog(next)) break;
        gen_tokens[token_count] = next;
        last = next;
        token_count += 1;
        if (!streamToken(stream, tok, next, &chat_hb)) {
            client_disconnected = true;
            break;
        }
    }

    const gen_end = milliTimestamp();
    const time_ms = elapsedBetween(gen_start, gen_end);
    const tps: f32 = tokensPerSec(token_count, time_ms);
    g_server.metrics.recordThroughput(token_count, time_ms);
    g_server.metrics.recordTPOT(token_count, time_ms);
    g_server.metrics.recordPromptTokens(prompt_token_count);
    g_server.metrics.recordGenerationTokens(token_count);

    // Send final stats event
    if (!client_disconnected) _ = flushStreamHoldback(stream, &chat_hb, emitChatStreamEvent);
    var stats_buf: [stats_buf_size]u8 = undefined;
    const stats_json = std.fmt.bufPrint(&stats_buf,
        \\{{"done":true,"n":{d},"ms":{d},"tps":{d:.2},"pn":{d},"pms":{d},"ptps":{d:.1}}}
    , .{ token_count, time_ms, tps, prompt_token_count, prefill_ms, prefill_tps }) catch "";
    if (stats_json.len > 0) _ = sseWriteData(stream, stats_json);
    _ = sseWriteData(stream, "[DONE]");

    g_server.metrics.recordLatency(time_ms);
    g_server.metrics.recordTokens(token_count);
    if (client_disconnected) {
        std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
        g_server.metrics.recordCancellation();
    } else if (forward_failed) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

    // Decode accumulated text for conversation storage
    const decoded = tok.decode(gen_tokens[0..token_count]) catch |err| d: {
        std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, token_count, err });
        break :d g_server.allocator.dupe(u8, "") catch @as([]u8, &.{});
    };

    return .{
        .data = decoded,
        .stats = .{
            .tokens_generated = token_count,
            .prompt_tokens = prompt_token_count,
            .time_ms = time_ms,
            .tokens_per_sec = tps,
            .prefill_ms = prefill_ms,
            .prefill_tps = prefill_tps,
        },
    };
}

/// Maximum bytes in one UTF-8 sequence.
const max_utf8_seq_len: usize = 4;

/// True when every byte in `bytes` is a UTF-8 continuation byte (10xxxxxx).
fn allContinuationBytes(bytes: []const u8) bool {
    for (bytes) |b| {
        if (b & 0xC0 != 0x80) return false;
    }
    return true;
}

/// Length of the trailing incomplete-but-valid UTF-8 sequence prefix of `text`;
/// 0 when `text` ends on a character boundary or its tail is not a valid
/// sequence prefix (invalid bytes pass through rather than stalling a stream).
fn incompleteUtf8TailLen(text: []const u8) usize {
    const lookback_max: usize = @min(text.len, max_utf8_seq_len - 1);
    var k: usize = 0;
    while (k < lookback_max) : (k += 1) {
        const idx = text.len - 1 - k;
        const b = text[idx];
        if (b < 0x80) return 0; // ASCII terminates any sequence
        if (b & 0xC0 == 0x80) continue; // continuation byte, keep walking back
        // Lead byte at idx.
        const seq_len = std.unicode.utf8ByteSequenceLength(b) catch return 0;
        const have = text.len - idx;
        if (have >= seq_len) return 0; // sequence already complete
        if (!allContinuationBytes(text[idx + 1 ..])) return 0;
        return have;
    }
    return 0;
}

/// Cross-token UTF-8 holdback for SSE streaming.
///
/// Byte-level tokenizers can split one character across several tokens (for
/// example SPM byte fallback `<0xE4>` `<0xB8>` `<0x96>`); decoding each token
/// separately and sending it immediately would put invalid UTF-8 fragments on
/// the wire. A trailing incomplete sequence is held until its continuation
/// bytes arrive in a later token; `flush` releases anything still held when
/// generation ends so streamed output stays byte-identical to the
/// batch-decoded text returned to callers.
const Utf8Holdback = struct {
    /// Assembled storage for a character completed from held + fresh bytes.
    completed: [max_utf8_seq_len]u8 = undefined,
    /// Held leading bytes of an incomplete sequence.
    pending: [max_utf8_seq_len - 1]u8 = .{0} ** (max_utf8_seq_len - 1),
    pending_len: usize = 0,

    pub const Pieces = struct {
        /// Character completed from previously held bytes plus the head of
        /// this token's text. Points into `Utf8Holdback.completed`; valid
        /// until the next `feed` call. Empty when nothing was held.
        head: []const u8 = &.{},
        /// Ready-to-send body: all of this token's text except a newly held
        /// trailing fragment. Points into the `text` argument passed to feed.
        body: []const u8 = &.{},
    };

    /// Partition freshly decoded token text into ready-to-send pieces and
    /// update holdback state.
    pub fn feed(self: *Utf8Holdback, text: []const u8) Pieces {
        var rest = text;

        // Resolve a previously held partial character against the head of
        // `rest`. At most one character can complete per feed: the holdback
        // never spans more than a single sequence.
        var head: []const u8 = &.{};
        if (self.pending_len > 0 and rest.len > 0) {
            const seq_len = std.unicode.utf8ByteSequenceLength(self.pending[0]) catch {
                // Invalid held lead byte: release it raw rather than drop it,
                // keeping streamed output byte-identical to batch decode.
                const raw = self.pending[0..self.pending_len];
                self.pending_len = 0;
                return .{
                    .head = raw,
                    .body = self.feedTail(rest),
                };
            };
            const need = seq_len - self.pending_len;
            const take = @min(need, rest.len);
            if (!allContinuationBytes(rest[0..take])) {
                // Not a continuation: release held bytes raw and process the
                // whole token normally.
                const raw = self.pending[0..self.pending_len];
                self.pending_len = 0;
                return .{
                    .head = raw,
                    .body = self.feedTail(rest),
                };
            }
            if (take < need) {
                // Still incomplete after absorbing everything available.
                @memcpy(self.pending[self.pending_len..][0..take], rest[0..take]);
                self.pending_len += take;
                return .{};
            }
            @memcpy(self.completed[0..self.pending_len], self.pending[0..self.pending_len]);
            @memcpy(self.completed[self.pending_len..][0..take], rest[0..take]);
            head = self.completed[0..seq_len];
            self.pending_len = 0;
            rest = rest[take..];
        }

        return .{
            .head = head,
            .body = self.feedTail(rest),
        };
    }

    /// Hold any trailing partial sequence of `text`; return the emit-ready
    /// remainder.
    fn feedTail(self: *Utf8Holdback, text: []const u8) []const u8 {
        const hold = incompleteUtf8TailLen(text);
        if (hold > 0) {
            @memcpy(self.pending[0..hold], text[text.len - hold ..]);
        }
        self.pending_len = hold;
        return text[0 .. text.len - hold];
    }

    /// Bytes still held at end of generation. The holdback is reset either way.
    pub fn flush(self: *Utf8Holdback) []const u8 {
        const held = self.pending[0..self.pending_len];
        self.pending_len = 0;
        return held;
    }
};

/// Decode a single token ID to raw text. Returns null on decode failure or empty output.
fn decodeTokenText(tok: *Tokenizer, token_id: u32) ?[]u8 {
    const decoded = tok.decode(&[_]u32{token_id}) catch |err| {
        std.log.warn("req={d} stream decode failed (token_id={d}): {}", .{ log_request_id, token_id, err });
        return null;
    };
    if (decoded.len == 0) {
        g_server.allocator.free(decoded);
        return null;
    }
    return decoded;
}

/// JSON-escape `text` and write it as one chat-stream token event
/// (`data: {"t":"..."}`). Empty text emits nothing.
/// Returns false if the write failed (client disconnected).
fn emitChatStreamEvent(stream: http.TcpStream, text: []const u8) bool {
    if (text.len == 0) return true;
    // Fast path: escape into the event buffer without allocating. Falls back
    // to the allocating escape only when the escaped form does not fit.
    var buf: [sse_event_buf_size]u8 = undefined;
    const prefix = "data: {\"t\":\"";
    const suffix = "\"}\n\n";
    const budget = buf.len - prefix.len - suffix.len;
    if (text.len <= budget) {
        if (json.jsonEscapeInto(buf[prefix.len..][0..budget], text)) |escaped| {
            const event = buf[0 .. prefix.len + escaped.len + suffix.len];
            @memcpy(buf[0..prefix.len], prefix);
            @memcpy(event[prefix.len + escaped.len ..], suffix);
            stream.writeAll(event) catch return false;
            return true;
        }
    }
    const escaped = json.jsonEscape(g_server.allocator, text) catch return true;
    defer if (escaped.ptr != text.ptr) g_server.allocator.free(escaped);
    const event = std.fmt.bufPrint(&buf, "data: {{\"t\":\"{s}\"}}\n\n", .{escaped}) catch {
        std.log.warn("req={d} SSE token event exceeded buffer ({d} bytes escaped)", .{ log_request_id, escaped.len });
        return true;
    };
    stream.writeAll(event) catch return false;
    return true;
}

/// Release any bytes still held by `hb` as a final event at end of generation
/// so streamed output matches the batch-decoded text byte for byte.
fn flushStreamHoldback(stream: http.TcpStream, hb: *Utf8Holdback, comptime emitFn: fn (http.TcpStream, []const u8) bool) bool {
    const held = hb.flush();
    if (held.len == 0) return true;
    return emitFn(stream, held);
}

/// Stream a single decoded token as an SSE event, holding back any trailing
/// partial UTF-8 sequence until its continuation bytes arrive (byte-level
/// tokenizers can split one character across several tokens). Callers must
/// call flushStreamHoldback before the final [DONE].
/// Returns false if the write failed (client disconnected).
fn streamToken(stream: http.TcpStream, tok: *Tokenizer, token_id: u32, hb: *Utf8Holdback) bool {
    // Fast path: allocation-free single-token decode into a stack buffer.
    var buf: [stream_decode_buf_size]u8 = undefined;
    if (tok.decodeOne(token_id, &buf)) |decoded| {
        const pieces = hb.feed(decoded);
        if (!emitChatStreamEvent(stream, pieces.head)) return false;
        return emitChatStreamEvent(stream, pieces.body);
    }
    // Fallback for oversized tokens: allocating batch decode (same output).
    const decoded = decodeTokenText(tok, token_id) orelse return true;
    defer g_server.allocator.free(decoded);
    const pieces = hb.feed(decoded);
    if (!emitChatStreamEvent(stream, pieces.head)) return false;
    return emitChatStreamEvent(stream, pieces.body);
}

// ── Anthropic Messages API helpers ──────────────────────────────

const anthropic_error_fallback = "{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Internal error\"}}";

/// Map a numeric status code string to an HTTP status line for Anthropic errors.
fn anthropicStatusLine(status_code: []const u8) []const u8 {
    if (std.mem.eql(u8, status_code, "400")) return "400 Bad Request";
    if (std.mem.eql(u8, status_code, "401")) return "401 Unauthorized";
    if (std.mem.eql(u8, status_code, "403")) return "403 Forbidden";
    if (std.mem.eql(u8, status_code, "404")) return "404 Not Found";
    if (std.mem.eql(u8, status_code, "405")) return "405 Method Not Allowed";
    if (std.mem.eql(u8, status_code, "413")) return "413 Payload Too Large";
    if (std.mem.eql(u8, status_code, "429")) return "429 Too Many Requests";
    if (std.mem.eql(u8, status_code, "503")) return "503 Service Unavailable";
    return "500 Internal Server Error";
}

/// Send a JSON error response in Anthropic error format.
/// Message and type are JSON-escaped to prevent injection (CWE-116).
fn sendAnthropicError(stream: http.TcpStream, status_code: []const u8, err_type: []const u8, message: []const u8) void {
    const status = anthropicStatusLine(status_code);
    // Never fall back to unescaped input on OOM, that reintroduces injection.
    const escaped_msg = json.jsonEscape(g_server.allocator, message) catch {
        sendResponse(stream, status, "application/json", anthropic_error_fallback);
        return;
    };
    defer if (escaped_msg.ptr != message.ptr) g_server.allocator.free(escaped_msg);
    const escaped_type = json.jsonEscape(g_server.allocator, err_type) catch {
        sendResponse(stream, status, "application/json", anthropic_error_fallback);
        return;
    };
    defer if (escaped_type.ptr != err_type.ptr) g_server.allocator.free(escaped_type);
    var buf: [error_body_buf_size]u8 = undefined;
    const json_body = std.fmt.bufPrint(&buf,
        \\{{"type":"error","error":{{"type":"{s}","message":"{s}"}}}}
    , .{ escaped_type, escaped_msg }) catch {
        std.log.warn("req={d} anthropic error body overflow type={s}", .{ log_request_id, err_type });
        sendResponse(stream, status, "application/json", anthropic_error_fallback);
        return;
    };
    sendResponse(stream, status, "application/json", json_body);
}

const anthropic_rate_limit_fallback = "{\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Rate limit exceeded\"}}";

/// Send a 429 Too Many Requests response in Anthropic error format with Retry-After header.
fn sendAnthropic429(stream: http.TcpStream, retry_after: u32) void {
    g_server.metrics.recordRateLimit();
    std.log.warn("req={d} anthropic rate limited (retry_after={d}s)", .{ log_request_id, retry_after });
    var buf: [error_body_buf_size]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"type\":\"error\",\"error\":{{\"type\":\"rate_limit_error\",\"message\":\"Rate limit exceeded. Retry after {d} seconds.\"}}}}", .{retry_after}) catch anthropic_rate_limit_fallback;
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ http.security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after, log_request_id, http.corsHeaders() }) catch {
        // Always respond, a hung client is worse than a response without Retry-After.
        std.log.warn("req={d} anthropic 429 header format failed, using fallback", .{log_request_id});
        sendResponse(stream, "429 Too Many Requests", "application/json", body);
        return;
    };
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} anthropic 429 write failed (headers): {}", .{ log_request_id, err });
        return;
    };
    stream.writeAll(body) catch |err| {
        std.log.warn("req={d} anthropic 429 write failed (body): {}", .{ log_request_id, err });
        return;
    };
}

/// Send an SSE event with both event type and data (Anthropic streaming format).
fn sseWriteEvent(stream: http.TcpStream, event_type: []const u8, data: []const u8) bool {
    var event_buf: [response_buf_size + 64]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buf, "event: {s}\ndata: {s}\n\n", .{ event_type, data }) catch return false;
    stream.writeAll(event) catch return false;
    return true;
}

/// Start an Anthropic-format SSE streaming response for /v1/messages.
fn startAnthropicStream(stream: http.TcpStream, formatted: []const u8, max_tokens: usize, input_tokens: u32, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateAnthropicStream(stream, formatted, max_tokens, input_tokens, sampling);
}

/// Longest raw text piece handed to emitAnthropicDeltaPiece; its escaped form
/// must fit sse_event_buf_size.
const anthropic_delta_piece_len: usize = 256;

/// Anthropic SSE streaming with tools. Tool calls cannot be detected mid-stream
/// without holding back `<tool_call>` tokens, so, like the OpenAI tools path,
/// generation runs to completion first and the parsed result is emitted as
/// content blocks: one `tool_use` block per call (stop_reason "tool_use"), or
/// the raw text as a `text` block when nothing parses (small-model fallback).
fn startAnthropicStreamWithTools(stream: http.TcpStream, formatted: []const u8, max_tokens: usize, input_tokens: u32, sampling: SamplingParams, tp: *const json.ToolParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }

    const req_id = currentRequestId();
    const tool_stream_start = milliTimestamp();
    const gen = generateEscapedN(formatted, true, max_tokens, sampling);
    defer gen.deinit();

    // message_start event
    var msg_buf: [response_buf_size]u8 = undefined;
    const msg_start = std.fmt.bufPrint(&msg_buf,
        \\{{"type":"message_start","message":{{"id":"msg_{d}","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":0}}}}}}
    , .{ req_id, g_server.model_name, input_tokens }) catch return;
    if (!sseWriteEvent(stream, "message_start", msg_start)) return;

    if (std.mem.eql(u8, gen.finish_reason, "error")) {
        g_server.metrics.recordFailure();
        sendAnthropicFinalEvents(stream, "end_turn", 0);
        return;
    }

    var call_idx: usize = 0;
    if (hasToolCalls(gen.raw)) {
        var search_pos: usize = 0;

        while (nextAllowedToolCall(gen.raw, &search_pos, tp, &g_server.tool_registry)) |tc_json| {
            const name = json.extractField(tc_json, "name") orelse continue;
            const resolved_input = resolveAnthropicToolInput(g_server.allocator, tc_json, call_idx);
            defer if (resolved_input.owned) |p| g_server.allocator.free(p);
            const args_obj = resolved_input.obj;
            const escaped_name = json.jsonEscape(g_server.allocator, name) catch {
                std.log.warn("req={d} tool call name escaping failed (OOM), skipping tool call", .{log_request_id});
                continue;
            };
            defer if (escaped_name.ptr != name.ptr) g_server.allocator.free(escaped_name);

            var blk_buf: [sse_event_buf_size]u8 = undefined;
            const block_start = std.fmt.bufPrint(&blk_buf,
                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"tool_use","id":"toolu_{d}_{d}","name":"{s}"}}}}
            , .{ call_idx, req_id, call_idx, escaped_name }) catch {
                std.log.warn("req={d} stream anthropic tool call chunk overflow: skipping call {d}", .{ log_request_id, call_idx });
                continue;
            };
            if (!sseWriteEvent(stream, "content_block_start", block_start)) return;

            // input_json_delta carries the arguments as an escaped JSON string
            // per the Anthropic streaming spec.
            const escaped_args = json.jsonEscape(g_server.allocator, args_obj) catch "";
            defer if (escaped_args.ptr != args_obj.ptr and escaped_args.len > 0) g_server.allocator.free(escaped_args);
            if (escaped_args.len > 0) {
                var dbuf: [response_buf_size]u8 = undefined;
                const delta = std.fmt.bufPrint(&dbuf,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"input_json_delta","partial_json":"{s}"}}}}
                , .{ call_idx, escaped_args }) catch {
                    std.log.warn("req={d} anthropic tool input_json_delta exceeded buffer ({d} bytes), truncating call {d}", .{ log_request_id, escaped_args.len, call_idx });
                    continue;
                };
                if (!sseWriteEvent(stream, "content_block_delta", delta)) return;
            }

            var stop_buf: [64]u8 = undefined;
            const block_stop = std.fmt.bufPrint(&stop_buf,
                \\{{"type":"content_block_stop","index":{d}}}
            , .{call_idx}) catch continue;
            if (!sseWriteEvent(stream, "content_block_stop", block_stop)) return;
            call_idx += 1;
        }
    }

    var stop_reason: []const u8 = "tool_use";
    if (call_idx > 0) {
        // Each tool_use block already emitted its own content_block_stop;
        // close the message without another block event.
        sendAnthropicMessageEnd(stream, stop_reason, gen.stats.tokens_generated);
    } else {
        // No parseable tool call: degrade to a text block with the full output,
        // mirroring the non-streaming fallback and the OpenAI tools stream.
        stop_reason = if (std.mem.eql(u8, gen.finish_reason, "length")) "max_tokens" else "end_turn";
        if (!sseWriteEvent(stream, "content_block_start",
            \\{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
        )) return;
        var piece_start: usize = 0;
        while (piece_start < gen.raw.len) {
            var piece_end = @min(piece_start + anthropic_delta_piece_len, gen.raw.len);
            // Never split a UTF-8 sequence across events.
            if (piece_end < gen.raw.len) {
                while (piece_end > piece_start and (gen.raw[piece_end] & 0xC0) == 0x80) piece_end -= 1;
                // More continuation bytes than the piece is long leaves no lead
                // byte in range. Step forward so the walk always progresses.
                if (piece_end == piece_start) piece_end = piece_start + 1;
            }
            if (!emitAnthropicDeltaPiece(stream, gen.raw[piece_start..piece_end])) return;
            piece_start = piece_end;
        }
        sendAnthropicFinalEvents(stream, stop_reason, gen.stats.tokens_generated);
    }

    const tool_time_ms = elapsedMs(tool_stream_start);
    g_server.metrics.recordLatency(tool_time_ms);
    g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
    g_server.metrics.recordThroughput(gen.stats.tokens_generated, tool_time_ms);
    g_server.metrics.recordTPOT(gen.stats.tokens_generated, tool_time_ms);
    g_server.metrics.recordPromptTokens(input_tokens);
    g_server.metrics.recordGenerationTokens(gen.stats.tokens_generated);
    g_server.metrics.recordCompletion();
}

/// Result of resolving a <tool_call> payload's `arguments` into JSON object text.
/// `owned` must be freed by the caller when non-null; `obj` is valid while it is.
const ResolvedToolInput = struct {
    obj: []const u8,
    owned: ?[]u8 = null,
};

/// True when `s` is a single JSON object that closes on its last byte.
/// Braces inside strings do not count, and the object must end where the
/// brace depth returns to zero, so an object truncated mid-body is not
/// complete. `extractObjectField` returns the remaining text when a value
/// never closes, so its result needs this check before it goes on the wire.
fn isCompleteJsonObject(s: []const u8) bool {
    if (s.len < 2 or s[0] != '{') return false;
    var depth: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        switch (s[i]) {
            '"' => {
                i += 1;
                while (i < s.len and s[i] != '"') : (i += 1) {
                    if (s[i] == '\\' and i + 1 < s.len) i += 1;
                }
            },
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if (depth == 0) return i + 1 == s.len;
            },
            else => {},
        }
    }
    return false;
}

/// Resolve the `input` object text for a <tool_call> payload: object arguments
/// are embedded verbatim; string arguments are unwrapped when they decode to an
/// object, otherwise `{}` is substituted (with a warning). Anthropic requires
/// `input` to be a JSON object, an invalid value would break spec-compliant
/// clients, so unparseable arguments never reach the wire verbatim. A payload
/// the model truncated mid-object counts as unparseable.
fn resolveAnthropicToolInput(allocator: Allocator, tc_json: []const u8, call_idx: usize) ResolvedToolInput {
    if (json.extractObjectField(tc_json, "arguments")) |o| {
        if (isCompleteJsonObject(o)) return .{ .obj = o };
        std.log.warn("req={d} tool call {d}: arguments object is truncated, substituting {{}}", .{ log_request_id, call_idx });
        return .{ .obj = "{}" };
    }
    const s = json.extractField(tc_json, "arguments") orelse return .{ .obj = "{}" };
    const unescaped = json.jsonUnescape(allocator, s) catch s;
    if (unescaped.ptr == s.ptr) {
        // No allocation happened (nothing to unescape); borrow the input slice.
        const trimmed_borrowed = std.mem.trim(u8, unescaped, " \t\r\n");
        if (isCompleteJsonObject(trimmed_borrowed)) return .{ .obj = trimmed_borrowed };
        std.log.warn("req={d} tool call {d}: arguments not a JSON object, substituting {{}}", .{ log_request_id, call_idx });
        return .{ .obj = "{}" };
    }
    // Unescape allocated a new buffer; hand ownership to the caller.
    errdefer allocator.free(@constCast(unescaped));
    const trimmed = std.mem.trim(u8, unescaped, " \t\r\n");
    if (isCompleteJsonObject(trimmed)) return .{ .obj = trimmed, .owned = @constCast(unescaped) };
    std.log.warn("req={d} tool call {d}: arguments not a JSON object, substituting {{}}", .{ log_request_id, call_idx });
    allocator.free(@constCast(unescaped));
    return .{ .obj = "{}" };
}

/// Emit message_delta + message_stop for an already-closed content block set.
fn sendAnthropicMessageEnd(stream: http.TcpStream, stop_reason: []const u8, token_count: u32) void {
    var delta_buf: [response_buf_size]u8 = undefined;
    const delta = std.fmt.bufPrint(&delta_buf,
        \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":null}},"usage":{{"output_tokens":{d}}}}}
    , .{ stop_reason, token_count }) catch {
        std.log.warn("req={d} anthropic message_delta exceeded buffer", .{log_request_id});
        return;
    };
    _ = sseWriteEvent(stream, "message_delta", delta);
    _ = sseWriteEvent(stream, "message_stop",
        \\{"type":"message_stop"}
    );
}

/// Run generation and stream tokens as Anthropic-format SSE events.
/// Event sequence: message_start → content_block_start → content_block_delta* →
/// content_block_stop → message_delta → message_stop.
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and polls for generated tokens. Falls back to direct model.forward()
/// when no scheduler is running.
fn generateAnthropicStream(stream: http.TcpStream, formatted: []const u8, max_tokens: usize, input_tokens: u32, sampling_a: SamplingParams) void {
    const tok = g_server.tokenizer;
    const req_id = currentRequestId();

    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} anthropic streaming tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        sendAnthropicFinalEvents(stream, "end_turn", 0);
        return;
    };
    defer g_server.allocator.free(token_ids);

    // message_start event
    var msg_buf: [response_buf_size]u8 = undefined;
    const msg_start = std.fmt.bufPrint(&msg_buf,
        \\{{"type":"message_start","message":{{"id":"msg_{d}","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":0}}}}}}
    , .{ req_id, g_server.model_name, input_tokens }) catch return;
    if (!sseWriteEvent(stream, "message_start", msg_start)) return;

    // content_block_start
    if (!sseWriteEvent(stream, "content_block_start",
        \\{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
    )) return;

    // Scheduler path: grammar/json_mode requests bypass (no grammar/JSON support in scheduler).
    const use_grammar_anth = (sampling_a.grammar_string != null or sampling_a.json_schema != null) and !sampling_a.json_mode;
    if (g_server.request_manager != null and !use_grammar_anth and !sampling_a.json_mode) {
        const rm = g_server.request_manager.?;
        const req = rm.enqueue(token_ids, log_request_id) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            sendAnthropicFinalEvents(stream, "end_turn", 0);
            return;
        };
        configureSchedulerSampling(req, sampling_a);
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        const gen_start = milliTimestamp();
        var streamed_count: usize = 0;
        var token_count: u32 = 0;

        var anth_client_connected = true;
        var stop_buf: [scheduler_stop_buf_size]u8 = undefined;
        var stop_len: usize = 0;
        var checked_len: usize = 0;
        var hit_stop = false;
        var anth_hb = Utf8Holdback{};
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (pollSchedulerStop(req, tok, sampling_a, &stop_buf, &stop_len, &checked_len, g_server.allocator)) |_| {
                hit_stop = true;
                break;
            }
            const stream_limit = if (sampling_a.hasStop()) checked_len else req.visible_len.load(.acquire);
            while (streamed_count < stream_limit) {
                if (!streamAnthropicDelta(stream, tok, req.tokens.items[streamed_count], &anth_hb)) {
                    anth_client_connected = false;
                    req.is_cancelled.store(true, .release);
                    break;
                }
                streamed_count += 1;
                token_count += 1;
                if (token_count >= max_tokens) {
                    req.is_cancelled.store(true, .release);
                    break;
                }
            }
            if (!anth_client_connected or token_count >= max_tokens) break;
            sleepNs(scheduler_poll_interval_ns);
        }

        while (!req.scheduler_done.load(.acquire))
            sleepNs(scheduler_poll_interval_ns);

        // Drain remaining tokens
        const final_len = if (hit_stop) checked_len else req.visible_len.load(.acquire);
        while (anth_client_connected and streamed_count < final_len and token_count < max_tokens) {
            if (!streamAnthropicDelta(stream, tok, req.tokens.items[streamed_count], &anth_hb)) break;
            streamed_count += 1;
            token_count += 1;
        }
        if (hit_stop) token_count = @intCast(checked_len);

        if (anth_client_connected) {
            _ = flushStreamHoldback(stream, &anth_hb, emitAnthropicDeltaPiece);
            const stop_reason: []const u8 = if (token_count >= max_tokens) "max_tokens" else "end_turn";
            sendAnthropicFinalEvents(stream, stop_reason, token_count);
        }

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const tps: f32 = tokensPerSec(token_count, time_ms);
        logGeneration(token_count, time_ms, tps);
        g_server.metrics.recordLatency(time_ms);
        g_server.metrics.recordTokens(token_count);
        // Record TTFT from scheduler's per-request prefill timestamp
        if (req.prefill_done_at > 0) {
            const anth_ttft = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(anth_ttft, input_tokens);
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        g_server.metrics.recordTPOT(token_count, time_ms);
        g_server.metrics.recordPromptTokens(input_tokens);
        g_server.metrics.recordGenerationTokens(token_count);
        if (!anth_client_connected) {
            std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
            g_server.metrics.recordCancellation();
        } else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else if (req.is_timed_out.load(.acquire)) {
            std.log.warn("req={d} anthropic stream timed out (tokens={d})", .{ log_request_id, token_count });
        } else {
            std.log.warn("req={d} anthropic stream incomplete (tokens={d}, cancelled={})", .{
                log_request_id,
                token_count,
                req.is_cancelled.load(.acquire),
            });
            g_server.metrics.recordFailure();
        }
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    const model = g_server.model;
    applyPendingVisionEmbeddings();
    model.resetCache();
    g_server.clearCachedPromptIds();

    if (g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
            g_server.metrics.recordFailure();
            sendAnthropicFinalEvents(stream, "end_turn", 0);
            return;
        };
    }

    // Prefill
    const use_sampling_a = sampling_a.temperature > 0;
    const prng_seed_a = prngSeedFromSampling(sampling_a);
    noteAutoSeed(sampling_a, prng_seed_a);
    var prng_a = std.Random.Xoshiro256.init(prng_seed_a);
    const anth_prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                std.log.info("req={d} anthropic stream prefill cancelled", .{log_request_id});
                g_server.metrics.recordCancellation();
                sendAnthropicFinalEvents(stream, "end_turn", 0);
                return;
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            g_server.metrics.recordFailure();
            sendAnthropicFinalEvents(stream, "end_turn", 0);
            return;
        };
    }
    const anth_prefill_ms: u64 = @intCast(@max(milliTimestamp() - anth_prefill_start, 0));
    g_server.metrics.recordTTFT(anth_prefill_ms, @intCast(token_ids.len));
    if (use_sampling_a and token_ids.len > 0) {
        const a_logits = model.getLogits();
        if (sampling_a.min_p > 0) math_ops.applyMinP(a_logits, sampling_a.min_p);
        first_gen_token = math_ops.sampleToken(a_logits, sampling_a.temperature, sampling_a.top_k, sampling_a.top_p, prng_a.random());
    }

    // Generate and stream deltas
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    // Stream first generated token
    var anth_disconnected = false;
    var anth_hb = Utf8Holdback{};
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        anth_disconnected = !streamAnthropicDelta(stream, tok, first_gen_token, &anth_hb);
        last = first_gen_token;
        token_count = 1;
    }

    var anth_forward_failed = false;

    const a_has_draft = g_server.draft_model != null and token_ids.len > 0 and !(token_count == 0 and g_server.isEog(first_gen_token));
    var a_spec_storage: spec_decode.SpecState = undefined;
    var a_spec_valid = false;
    if (a_has_draft) {
        if (spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize())) |ok| {
            a_spec_storage = ok;
            a_spec_valid = true;
        } else |_| {
            a_spec_storage = spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        }
    }
    defer if (a_spec_valid) a_spec_storage.deinit(g_server.allocator);

    if (a_spec_valid) {
        var a_draft = g_server.draft_model.?;
        if (a_draft.ptr != model.ptr) {
            _ = a_draft.prefill(token_ids) catch |err| {
                std.log.warn("req={d} draft model prefill failed: {s}", .{ log_request_id, @errorName(err) });
            };
        }
        const a_spec = &a_spec_storage;
        // Token history for penalty tracking, mirroring the non-spec path below.
        var a_spec_tokens: [gen_ids_buf_size]u32 = undefined;
        defer @memset(std.mem.sliceAsBytes(&a_spec_tokens), 0);
        var a_spec_count: u32 = 0;
        if (token_count > 0) {
            a_spec_tokens[0] = first_gen_token;
            a_spec_count = 1;
        }
        while (token_count < max_tokens and !anth_disconnected) {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            const pre = model.kvSeqLen();
            const is_self = (model.ptr == a_draft.ptr);
            const pen_a: spec_decode.Penalties = .{
                .history = a_spec_tokens[0..a_spec_count],
                .repeat_penalty = sampling_a.repetition_penalty,
                .dry_multiplier = sampling_a.dry_multiplier,
                .dry_length = sampling_a.dry_allowed_length,
            };
            const nd = if (is_self and !use_sampling_a)
                spec_decode.draft(a_spec, a_draft, last)
            else
                spec_decode.draftWithLogits(a_spec, a_draft, last);
            if (nd == 0) break;
            const res = if (is_self)
                spec_decode.SpecResult{ .accepted = a_spec.n_draft, .next_token = blk: {
                    a_spec.recordRound(a_spec.n_draft);
                    const ld = a_spec.draft_tokens[a_spec.n_draft - 1];
                    break :blk model.forward(ld) catch ld;
                } }
            else if (use_sampling_a)
                spec_decode.verifySampling(a_spec, model, a_draft, last, pre, sampling_a.temperature, pen_a, prng_a.random())
            else
                spec_decode.verifySequential(a_spec, model, a_draft, last, pre, pen_a);

            for (0..res.accepted) |i| {
                const at = a_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (token_count >= max_tokens) break;
                if (!streamAnthropicDelta(stream, tok, at, &anth_hb)) {
                    anth_disconnected = true;
                    break;
                }
                if (a_spec_count < a_spec_tokens.len) {
                    a_spec_tokens[a_spec_count] = at;
                    a_spec_count += 1;
                }
                token_count += 1;
            }
            if (!anth_disconnected and !g_server.isEog(res.next_token)) {
                if (token_count < max_tokens) {
                    if (!streamAnthropicDelta(stream, tok, res.next_token, &anth_hb)) {
                        anth_disconnected = true;
                    }
                    if (a_spec_count < a_spec_tokens.len) {
                        a_spec_tokens[a_spec_count] = res.next_token;
                        a_spec_count += 1;
                    }
                    token_count += 1;
                }
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
        var itl = ItlTracker.init();
        for (0..max_tokens -| 1) |_| {
            if (anth_disconnected or token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            var next = model.forward(last) catch |err| {
                if (err != error.Cancelled) {
                    std.log.warn("req={d} generation forward failed: {}", .{ log_request_id, err });
                    anth_forward_failed = true;
                }
                break;
            };
            if (use_sampling_a) {
                const a_next_logits = model.getLogits();
                if (sampling_a.min_p > 0) math_ops.applyMinP(a_next_logits, sampling_a.min_p);
                next = math_ops.sampleToken(a_next_logits, sampling_a.temperature, sampling_a.top_k, sampling_a.top_p, prng_a.random());
            }
            if (g_server.isEog(next)) break;

            if (!streamAnthropicDelta(stream, tok, next, &anth_hb)) {
                anth_disconnected = true;
                break;
            }
            itl.sample();
            last = next;
            token_count += 1;
        }
    }

    if (!anth_disconnected) {
        _ = flushStreamHoldback(stream, &anth_hb, emitAnthropicDeltaPiece);
        const stop_reason: []const u8 = if (token_count >= max_tokens) "max_tokens" else "end_turn";
        sendAnthropicFinalEvents(stream, stop_reason, token_count);
    }

    const gen_end = milliTimestamp();
    const time_ms = elapsedBetween(gen_start, gen_end);
    const tps: f32 = tokensPerSec(token_count, time_ms);
    logGeneration(token_count, time_ms, tps);
    g_server.metrics.recordThroughput(token_count, time_ms);
    g_server.metrics.recordTPOT(token_count, time_ms);
    g_server.metrics.recordPromptTokens(@intCast(token_ids.len));
    g_server.metrics.recordGenerationTokens(token_count);
    g_server.metrics.recordLatency(time_ms);
    g_server.metrics.recordTokens(token_count);
    if (anth_disconnected) {
        std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
        g_server.metrics.recordCancellation();
    } else if (anth_forward_failed) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();
}

/// Send the Anthropic SSE final events: content_block_stop, message_delta, message_stop.
fn sendAnthropicFinalEvents(stream: http.TcpStream, stop_reason: []const u8, token_count: u32) void {
    _ = sseWriteEvent(stream, "content_block_stop",
        \\{"type":"content_block_stop","index":0}
    );
    var delta_buf: [response_buf_size]u8 = undefined;
    const delta = std.fmt.bufPrint(&delta_buf,
        \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":null}},"usage":{{"output_tokens":{d}}}}}
    , .{ stop_reason, token_count }) catch {
        std.log.warn("req={d} anthropic message_delta exceeded buffer", .{log_request_id});
        return;
    };
    _ = sseWriteEvent(stream, "message_delta", delta);
    _ = sseWriteEvent(stream, "message_stop",
        \\{"type":"message_stop"}
    );
}

/// JSON-escape `text` and write it as one Anthropic content_block_delta event.
/// Empty text emits nothing. Returns false on client disconnect.
fn emitAnthropicDeltaPiece(stream: http.TcpStream, text: []const u8) bool {
    if (text.len == 0) return true;
    // Fast path: escape into the event buffer without allocating, as the chat
    // stream path does. Falls back to the allocating escape only when the
    // escaped form does not fit.
    var buf: [sse_event_buf_size]u8 = undefined;
    const prefix = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"";
    const suffix = "\"}}";
    const budget = buf.len - prefix.len - suffix.len;
    if (text.len <= budget) {
        if (json.jsonEscapeInto(buf[prefix.len..][0..budget], text)) |escaped| {
            @memcpy(buf[0..prefix.len], prefix);
            @memcpy(buf[prefix.len + escaped.len ..][0..suffix.len], suffix);
            return sseWriteEvent(stream, "content_block_delta", buf[0 .. prefix.len + escaped.len + suffix.len]);
        }
    }
    const escaped = json.jsonEscape(g_server.allocator, text) catch return true;
    defer if (escaped.ptr != text.ptr) g_server.allocator.free(escaped);
    const data = std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, escaped, suffix }) catch {
        std.log.warn("req={d} anthropic SSE delta exceeded buffer ({d} bytes escaped)", .{ log_request_id, escaped.len });
        return true;
    };
    return sseWriteEvent(stream, "content_block_delta", data);
}

/// Stream a single decoded token as an Anthropic content_block_delta SSE
/// event, holding back trailing partial UTF-8 sequences across tokens.
/// Callers must flushStreamHoldback before the final events.
/// Returns false if the write failed (client disconnected).
fn streamAnthropicDelta(stream: http.TcpStream, tok: *Tokenizer, token_id: u32, hb: *Utf8Holdback) bool {
    var buf: [stream_decode_buf_size]u8 = undefined;
    if (tok.decodeOne(token_id, &buf)) |decoded| {
        const pieces = hb.feed(decoded);
        if (!emitAnthropicDeltaPiece(stream, pieces.head)) return false;
        return emitAnthropicDeltaPiece(stream, pieces.body);
    }
    const decoded = decodeTokenText(tok, token_id) orelse return true;
    defer g_server.allocator.free(decoded);
    const pieces = hb.feed(decoded);
    if (!emitAnthropicDeltaPiece(stream, pieces.head)) return false;
    return emitAnthropicDeltaPiece(stream, pieces.body);
}

// ── Responses API Streaming ─────────────────────────────────────

/// Start a Responses API SSE streaming response for /v1/responses.
fn startResponsesStream(stream: http.TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateResponsesStream(stream, prompt, max_tokens, sampling);
}

/// Send the Responses API setup events: response.created, response.output_item.added,
/// response.content_part.added.
fn sendResponsesStartEvents(stream: http.TcpStream, req_id: u64, created: i64) void {
    var buf: [response_buf_size]u8 = undefined;
    const created_evt = std.fmt.bufPrint(&buf,
        \\{{"type":"response.created","response":{{"id":"resp-{d}","object":"response","created_at":{d},"status":"in_progress","model":"{s}","output":[],"usage":null}}}}
    , .{ req_id, created, g_server.model_name }) catch return;
    _ = sseWriteEvent(stream, "response.created", created_evt);

    _ = sseWriteEvent(stream, "response.output_item.added",
        \\{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_0","status":"in_progress","role":"assistant","content":[]}}
    );

    _ = sseWriteEvent(stream, "response.content_part.added",
        \\{"type":"response.content_part.added","item_id":"msg_0","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}
    );
}

/// JSON-escape `text` and write it as one Responses API output_text.delta
/// event. Empty text emits nothing. Returns false on client disconnect.
fn emitResponsesDeltaPiece(stream: http.TcpStream, text: []const u8) bool {
    if (text.len == 0) return true;
    // Fast path: escape into the event buffer without allocating, as the chat
    // stream path does. Falls back to the allocating escape only when the
    // escaped form does not fit.
    var buf: [sse_event_buf_size]u8 = undefined;
    const prefix = "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_0\",\"output_index\":0,\"content_index\":0,\"delta\":\"";
    const suffix = "\"}";
    const budget = buf.len - prefix.len - suffix.len;
    if (text.len <= budget) {
        if (json.jsonEscapeInto(buf[prefix.len..][0..budget], text)) |escaped| {
            @memcpy(buf[0..prefix.len], prefix);
            @memcpy(buf[prefix.len + escaped.len ..][0..suffix.len], suffix);
            return sseWriteEvent(stream, "response.output_text.delta", buf[0 .. prefix.len + escaped.len + suffix.len]);
        }
    }
    const escaped = json.jsonEscape(g_server.allocator, text) catch return true;
    defer if (escaped.ptr != text.ptr) g_server.allocator.free(escaped);
    const data = std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, escaped, suffix }) catch {
        std.log.warn("req={d} responses SSE delta exceeded buffer ({d} bytes escaped)", .{ log_request_id, escaped.len });
        return true;
    };
    return sseWriteEvent(stream, "response.output_text.delta", data);
}

/// Stream a single decoded token as a Responses API output_text.delta event,
/// holding back trailing partial UTF-8 sequences across tokens.
/// Callers must flushStreamHoldback before the final events.
/// Returns false if the write failed (client disconnected).
fn streamResponsesDelta(stream: http.TcpStream, tok: *Tokenizer, token_id: u32, hb: *Utf8Holdback) bool {
    var buf: [stream_decode_buf_size]u8 = undefined;
    if (tok.decodeOne(token_id, &buf)) |decoded| {
        const pieces = hb.feed(decoded);
        if (!emitResponsesDeltaPiece(stream, pieces.head)) return false;
        return emitResponsesDeltaPiece(stream, pieces.body);
    }
    const decoded = decodeTokenText(tok, token_id) orelse return true;
    defer g_server.allocator.free(decoded);
    const pieces = hb.feed(decoded);
    if (!emitResponsesDeltaPiece(stream, pieces.head)) return false;
    return emitResponsesDeltaPiece(stream, pieces.body);
}

/// Send the Responses API final events: output_text.done, content_part.done,
/// output_item.done, response.completed.
fn sendResponsesFinalEvents(stream: http.TcpStream, req_id: u64, created: i64, stop_reason: []const u8, escaped_text: []const u8, input_tokens: u32, output_tokens: u32) void {
    var buf: [response_buf_size]u8 = undefined;
    const total = input_tokens + output_tokens;

    const text_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.output_text.done","item_id":"msg_0","output_index":0,"content_index":0,"text":"{s}"}}
    , .{escaped_text}) catch {
        std.log.warn("req={d} responses output_text.done exceeded buffer ({d} bytes text)", .{ log_request_id, escaped_text.len });
        return;
    };
    _ = sseWriteEvent(stream, "response.output_text.done", text_done);

    const part_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.content_part.done","item_id":"msg_0","output_index":0,"content_index":0,"part":{{"type":"output_text","text":"{s}"}}}}
    , .{escaped_text}) catch {
        std.log.warn("req={d} responses content_part.done exceeded buffer ({d} bytes text)", .{ log_request_id, escaped_text.len });
        return;
    };
    _ = sseWriteEvent(stream, "response.content_part.done", part_done);

    const item_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.output_item.done","output_index":0,"item":{{"type":"message","id":"msg_0","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}}}
    , .{escaped_text}) catch {
        std.log.warn("req={d} responses output_item.done exceeded buffer ({d} bytes text)", .{ log_request_id, escaped_text.len });
        return;
    };
    _ = sseWriteEvent(stream, "response.output_item.done", item_done);

    const completed = std.fmt.bufPrint(&buf,
        \\{{"type":"response.completed","response":{{"id":"resp-{d}","object":"response","created_at":{d},"status":"completed","model":"{s}","stop_reason":"{s}","output":[{{"type":"message","id":"msg_0","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}],"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}}}}}}
    , .{ req_id, created, g_server.model_name, stop_reason, escaped_text, input_tokens, output_tokens, total }) catch {
        std.log.warn("req={d} responses completed event exceeded buffer ({d} bytes text)", .{ log_request_id, escaped_text.len });
        return;
    };
    _ = sseWriteEvent(stream, "response.completed", completed);
}

/// Run generation and stream tokens as Responses API SSE events.
/// Event sequence: response.created → response.output_item.added →
/// response.content_part.added → response.output_text.delta* →
/// response.output_text.done → response.content_part.done →
/// response.output_item.done → response.completed.
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and polls for generated tokens. Falls back to direct model.forward()
/// when no scheduler is running.
fn generateResponsesStream(stream: http.TcpStream, prompt: []const u8, max_tokens: usize, sampling_r: SamplingParams) void {
    const tok = g_server.tokenizer;
    const req_id = currentRequestId();
    const created = timestamp();

    const formatted = g_server.chat_template.format(g_server.allocator, null, prompt) catch prompt;
    defer if (formatted.ptr != prompt.ptr) wipeFree(g_server.allocator, @constCast(formatted));
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} responses streaming tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        sendResponsesStartEvents(stream, req_id, created);
        sendResponsesFinalEvents(stream, req_id, created, "stop", "", 0, 0);
        return;
    };
    defer wipeFreeTokens(g_server.allocator, token_ids);
    const input_tokens: u32 = @intCast(token_ids.len);

    // Send setup events
    sendResponsesStartEvents(stream, req_id, created);

    // Scheduler path: grammar/json_mode requests bypass (no grammar/JSON support in scheduler).
    const use_grammar_resp = (sampling_r.grammar_string != null or sampling_r.json_schema != null) and !sampling_r.json_mode;
    if (g_server.request_manager != null and !use_grammar_resp and !sampling_r.json_mode) {
        const rm = g_server.request_manager.?;
        const req = rm.enqueue(token_ids, log_request_id) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            sendResponsesFinalEvents(stream, req_id, created, "stop", "", input_tokens, 0);
            return;
        };
        configureSchedulerSampling(req, sampling_r);
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        const gen_start = milliTimestamp();
        var streamed_count: usize = 0;
        var token_count: u32 = 0;

        var resp_client_connected = true;
        var stop_buf: [scheduler_stop_buf_size]u8 = undefined;
        var stop_len: usize = 0;
        var checked_len: usize = 0;
        var hit_stop = false;
        var resp_hb = Utf8Holdback{};
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (pollSchedulerStop(req, tok, sampling_r, &stop_buf, &stop_len, &checked_len, g_server.allocator)) |_| {
                hit_stop = true;
                break;
            }
            const stream_limit = if (sampling_r.hasStop()) checked_len else req.visible_len.load(.acquire);
            while (streamed_count < stream_limit) {
                if (!streamResponsesDelta(stream, tok, req.tokens.items[streamed_count], &resp_hb)) {
                    resp_client_connected = false;
                    req.is_cancelled.store(true, .release);
                    break;
                }
                streamed_count += 1;
                token_count += 1;
                if (token_count >= max_tokens) {
                    req.is_cancelled.store(true, .release);
                    break;
                }
            }
            if (!resp_client_connected or token_count >= max_tokens) break;
            sleepNs(scheduler_poll_interval_ns);
        }

        while (!req.scheduler_done.load(.acquire))
            sleepNs(scheduler_poll_interval_ns);

        // Drain remaining tokens
        const final_len = if (hit_stop) checked_len else req.visible_len.load(.acquire);
        while (resp_client_connected and streamed_count < final_len and token_count < max_tokens) {
            if (!streamResponsesDelta(stream, tok, req.tokens.items[streamed_count], &resp_hb)) break;
            streamed_count += 1;
            token_count += 1;
        }
        if (hit_stop) token_count = @intCast(checked_len);

        // Send final events, skip if client already disconnected
        if (resp_client_connected) {
            _ = flushStreamHoldback(stream, &resp_hb, emitResponsesDeltaPiece);
            const safe_resp_count: usize = token_count;
            // Bound by visible_len only (see chatStreamGenerate scheduler path).
            const safe_resp_tokens = req.tokens.items[0..safe_resp_count];
            const decoded = tok.decode(safe_resp_tokens) catch |err| d: {
                std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, safe_resp_tokens.len, err });
                break :d g_server.allocator.dupe(u8, "") catch @as([]u8, &.{});
            };
            defer g_server.allocator.free(decoded);
            const escaped = json.jsonEscape(g_server.allocator, decoded) catch |err| blk: {
                std.log.err("req={d} JSON escape OOM in responses stream ({d} bytes): {}", .{ log_request_id, decoded.len, err });
                break :blk decoded[0..0];
            };
            defer if (escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);

            const stop_reason: []const u8 = if (token_count >= max_tokens) "max_tokens" else "stop";
            sendResponsesFinalEvents(stream, req_id, created, stop_reason, escaped, input_tokens, token_count);
        }

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const tps: f32 = tokensPerSec(token_count, time_ms);
        logGeneration(token_count, time_ms, tps);
        g_server.metrics.recordLatency(time_ms);
        g_server.metrics.recordTokens(token_count);
        // Record TTFT from scheduler's per-request prefill timestamp
        if (req.prefill_done_at > 0) {
            const resp_ttft = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(resp_ttft, input_tokens);
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        g_server.metrics.recordTPOT(token_count, time_ms);
        g_server.metrics.recordPromptTokens(input_tokens);
        g_server.metrics.recordGenerationTokens(token_count);
        if (!resp_client_connected) {
            std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
            g_server.metrics.recordCancellation();
        } else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else if (req.is_timed_out.load(.acquire)) {
            std.log.warn("req={d} responses stream timed out (tokens={d})", .{ log_request_id, token_count });
        } else {
            std.log.warn("req={d} responses stream incomplete (tokens={d}, cancelled={})", .{
                log_request_id,
                token_count,
                req.is_cancelled.load(.acquire),
            });
            g_server.metrics.recordFailure();
        }
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    const model = g_server.model;
    applyPendingVisionEmbeddings();
    model.resetCache();
    g_server.clearCachedPromptIds();

    if (g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
            g_server.metrics.recordFailure();
            sendResponsesFinalEvents(stream, req_id, created, "stop", "", input_tokens, 0);
            return;
        };
    }

    // Prefill
    const use_sampling_r = sampling_r.temperature > 0;
    const prng_seed_r = prngSeedFromSampling(sampling_r);
    noteAutoSeed(sampling_r, prng_seed_r);
    var prng_r = std.Random.Xoshiro256.init(prng_seed_r);
    const resp_prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                std.log.info("req={d} responses stream prefill cancelled", .{log_request_id});
                g_server.metrics.recordCancellation();
                sendResponsesFinalEvents(stream, req_id, created, "stop", "", input_tokens, 0);
                return;
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            g_server.metrics.recordFailure();
            sendResponsesFinalEvents(stream, req_id, created, "stop", "", input_tokens, 0);
            return;
        };
    }
    const resp_prefill_ms: u64 = @intCast(@max(milliTimestamp() - resp_prefill_start, 0));
    g_server.metrics.recordTTFT(resp_prefill_ms, input_tokens);
    if (use_sampling_r and token_ids.len > 0) {
        const r_logits = model.getLogits();
        if (sampling_r.min_p > 0) math_ops.applyMinP(r_logits, sampling_r.min_p);
        first_gen_token = math_ops.sampleToken(r_logits, sampling_r.temperature, sampling_r.top_k, sampling_r.top_p, prng_r.random());
    }

    // Generate and stream deltas
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    defer @memset(std.mem.sliceAsBytes(&gen_tokens), 0);

    var resp_disconnected = false;
    var resp_hb = Utf8Holdback{};
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        resp_disconnected = !streamResponsesDelta(stream, tok, first_gen_token, &resp_hb);
        gen_tokens[0] = first_gen_token;
        last = first_gen_token;
        token_count = 1;
    }

    var resp_forward_failed = false;

    const r_has_draft = g_server.draft_model != null and token_ids.len > 0 and !(token_count == 0 and g_server.isEog(first_gen_token));
    var r_spec_storage: spec_decode.SpecState = undefined;
    var r_spec_valid = false;
    if (r_has_draft) {
        if (spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize())) |ok| {
            r_spec_storage = ok;
            r_spec_valid = true;
        } else |_| {
            r_spec_storage = spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        }
    }
    defer if (r_spec_valid) r_spec_storage.deinit(g_server.allocator);

    if (r_spec_valid) {
        var r_draft = g_server.draft_model.?;
        if (r_draft.ptr != model.ptr) {
            _ = r_draft.prefill(token_ids) catch |err| {
                std.log.warn("req={d} draft model prefill failed: {s}", .{ log_request_id, @errorName(err) });
            };
        }
        const r_spec = &r_spec_storage;
        while (token_count < max_tokens and !resp_disconnected) {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            const pre = model.kvSeqLen();
            const is_self = (model.ptr == r_draft.ptr);
            const pen_r: spec_decode.Penalties = .{
                .history = gen_tokens[0..token_count],
                .repeat_penalty = sampling_r.repetition_penalty,
                .dry_multiplier = sampling_r.dry_multiplier,
                .dry_length = sampling_r.dry_allowed_length,
            };
            const nd = if (is_self and !use_sampling_r)
                spec_decode.draft(r_spec, r_draft, last)
            else
                spec_decode.draftWithLogits(r_spec, r_draft, last);
            if (nd == 0) break;
            const res = if (is_self)
                spec_decode.SpecResult{ .accepted = r_spec.n_draft, .next_token = blk: {
                    r_spec.recordRound(r_spec.n_draft);
                    const ld = r_spec.draft_tokens[r_spec.n_draft - 1];
                    break :blk model.forward(ld) catch ld;
                } }
            else if (use_sampling_r)
                spec_decode.verifySampling(r_spec, model, r_draft, last, pre, sampling_r.temperature, pen_r, prng_r.random())
            else
                spec_decode.verifySequential(r_spec, model, r_draft, last, pre, pen_r);

            for (0..res.accepted) |i| {
                const at = r_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (token_count >= max_tokens) break;
                if (!streamResponsesDelta(stream, tok, at, &resp_hb)) {
                    resp_disconnected = true;
                    break;
                }
                if (token_count < gen_ids_buf_size) gen_tokens[token_count] = at;
                token_count += 1;
            }
            if (!resp_disconnected and !g_server.isEog(res.next_token)) {
                if (token_count < max_tokens) {
                    if (!streamResponsesDelta(stream, tok, res.next_token, &resp_hb)) {
                        resp_disconnected = true;
                    }
                    if (token_count < gen_ids_buf_size) gen_tokens[token_count] = res.next_token;
                    token_count += 1;
                }
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
        var itl = ItlTracker.init();
        for (0..max_tokens -| 1) |_| {
            if (resp_disconnected or token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            var next = model.forward(last) catch |err| {
                if (err != error.Cancelled) {
                    std.log.warn("req={d} generation forward failed: {}", .{ log_request_id, err });
                    resp_forward_failed = true;
                }
                break;
            };
            if (use_sampling_r) {
                const r_next_logits = model.getLogits();
                if (sampling_r.min_p > 0) math_ops.applyMinP(r_next_logits, sampling_r.min_p);
                next = math_ops.sampleToken(r_next_logits, sampling_r.temperature, sampling_r.top_k, sampling_r.top_p, prng_r.random());
            }
            if (g_server.isEog(next)) break;

            if (!streamResponsesDelta(stream, tok, next, &resp_hb)) {
                resp_disconnected = true;
                break;
            }
            itl.sample();
            gen_tokens[token_count] = next;
            last = next;
            token_count += 1;
        }
    }

    // Send final events, skip if client already disconnected
    if (!resp_disconnected) {
        _ = flushStreamHoldback(stream, &resp_hb, emitResponsesDeltaPiece);
        const decoded = tok.decode(gen_tokens[0..@min(token_count, gen_ids_buf_size)]) catch |err| d: {
            std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, token_count, err });
            break :d g_server.allocator.dupe(u8, "") catch @as([]u8, &.{});
        };
        defer g_server.allocator.free(decoded);
        const escaped = json.jsonEscape(g_server.allocator, decoded) catch |err| blk: {
            std.log.err("req={d} JSON escape OOM in responses stream ({d} bytes): {}", .{ log_request_id, decoded.len, err });
            break :blk decoded[0..0];
        };
        defer if (escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);

        const stop_reason: []const u8 = if (token_count >= max_tokens) "max_tokens" else "stop";
        sendResponsesFinalEvents(stream, req_id, created, stop_reason, escaped, input_tokens, token_count);
    }

    const gen_end = milliTimestamp();
    const time_ms = elapsedBetween(gen_start, gen_end);
    const tps: f32 = tokensPerSec(token_count, time_ms);
    logGeneration(token_count, time_ms, tps);
    g_server.metrics.recordThroughput(token_count, time_ms);
    g_server.metrics.recordTPOT(token_count, time_ms);
    g_server.metrics.recordPromptTokens(input_tokens);
    g_server.metrics.recordGenerationTokens(token_count);
    g_server.metrics.recordLatency(time_ms);
    g_server.metrics.recordTokens(token_count);
    if (resp_disconnected) {
        std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
        g_server.metrics.recordCancellation();
    } else if (resp_forward_failed) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();
}

// ── SSE Streaming ──────────────────────────────────────────────

/// Send an SSE data event. Returns false if the write failed (client disconnected).
/// Generic over the stream type so tests can collect output without sockets;
/// monomorphized at comptime, no dispatch cost.
fn sseWriteData(stream: anytype, data: []const u8) bool {
    var event_buf: [response_buf_size + 16]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buf, "data: {s}\n\n", .{data}) catch return false;
    stream.writeAll(event) catch return false;
    return true;
}

/// Start an SSE streaming response. Writes headers, generates tokens inline,
/// and writes each as an SSE frame. Runs synchronously on the handler thread.
fn startStream(stream: http.TcpStream, prompt: []const u8, is_chat: bool, format_prompt: bool, max_tokens: usize, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateStream(stream, prompt, currentRequestId(), timestamp(), is_chat, format_prompt, max_tokens, sampling);
}

/// Maximum escaped-content bytes per streamed content delta. Longer outputs are
/// split across multiple deltas rather than dropped when one frame overflows.
const stream_content_delta_max: usize = 16384;
/// Buffer for a quoted finish_reason token (`"stop"` / `"length"`).
const finish_reason_json_buf_size: usize = 16;

/// Format one streamed assistant content delta chunk.
/// `finish_reason_json` is the raw JSON value: `null` for intermediate chunks,
/// `"stop"`/`"length"` for the final one. Returns null when `buf` is too small.
fn formatContentDeltaChunk(buf: []u8, model_name: []const u8, req_id: u64, created: i64, content_piece: []const u8, finish_reason_json: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","content":"{s}"}},"finish_reason":{s}}}]}}
    , .{ req_id, created, model_name, content_piece, finish_reason_json }) catch null;
}

/// Emit `escaped_content` as one or more assistant content delta chunks,
/// splitting at `stream_content_delta_max` so oversized outputs stream fully
/// instead of being silently dropped when a single chunk overflows the buffer.
fn writeStreamedContent(stream: anytype, chunk_buf: []u8, model_name: []const u8, req_id: u64, created: i64, escaped_content: []const u8, finish_reason: []const u8) void {
    var pos: usize = 0;
    while (true) {
        const end = @min(pos + stream_content_delta_max, escaped_content.len);
        const piece = escaped_content[pos..end];
        const final = end == escaped_content.len;
        var fr_buf: [finish_reason_json_buf_size]u8 = undefined;
        const fr_json: []const u8 = if (final)
            std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{finish_reason}) catch "null"
        else
            "null";
        const chunk = formatContentDeltaChunk(chunk_buf, model_name, req_id, created, piece, fr_json) orelse {
            std.log.warn("req={d} stream content chunk overflow: dropping remainder from byte {d}", .{ log_request_id, pos });
            return;
        };
        _ = sseWriteData(stream, chunk);
        if (final) return;
        pos = end;
    }
}

/// Streaming with tool call support. Generates full output first, then emits
/// tool_calls delta chunks if tool calls detected, otherwise streams content.
fn startStreamWithTools(stream: http.TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams, tp: *const json.ToolParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }

    const req_id = currentRequestId();
    const created = timestamp();
    const tool_stream_start = milliTimestamp();
    const gen = generateEscapedN(prompt, true, max_tokens, sampling);
    defer gen.deinit();

    if (std.mem.eql(u8, gen.finish_reason, "error")) {
        _ = sseWriteData(stream, "{\"error\":\"Generation failed\"}");
        _ = sseWriteData(stream, "[DONE]");
        g_server.metrics.recordLatency(elapsedMs(tool_stream_start));
        g_server.metrics.recordFailure();
        return;
    }

    var chunk_buf: [response_buf_size]u8 = undefined;

    var call_idx: usize = 0;
    if (hasToolCalls(gen.raw)) {
        // Emit tool calls as delta chunks
        var search_pos: usize = 0;

        while (nextAllowedToolCall(gen.raw, &search_pos, tp, &g_server.tool_registry)) |tc_json| {
            const name = json.extractField(tc_json, "name") orelse continue;
            const args = json.extractObjectField(tc_json, "arguments") orelse
                (json.extractField(tc_json, "arguments") orelse "{}");
            // Escape name and args, model output is untrusted (CWE-116).
            const escaped_name = json.jsonEscape(g_server.allocator, name) catch {
                std.log.warn("req={d} tool call name escaping failed (OOM), skipping tool call", .{log_request_id});
                continue;
            };
            defer if (escaped_name.ptr != name.ptr) g_server.allocator.free(escaped_name);
            const escaped_args = json.jsonEscape(g_server.allocator, args) catch {
                std.log.warn("req={d} tool call argument escaping failed (OOM), skipping tool call", .{log_request_id});
                continue;
            };
            defer if (escaped_args.ptr != args.ptr) g_server.allocator.free(escaped_args);

            // First chunk: role + tool call header
            const role_chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","tool_calls":[{{"index":{d},"id":"call_{d}_{d}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]}},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, call_idx, req_id, call_idx, escaped_name, escaped_args }) catch {
                std.log.warn("req={d} stream tool call chunk overflow: skipping call {d}", .{ log_request_id, call_idx });
                continue;
            };
            _ = sseWriteData(stream, role_chunk);
            call_idx += 1;
        }
    }

    if (call_idx > 0) {
        // Final chunk with finish_reason
        const finish = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"tool_calls"}}]}}
        , .{ req_id, created, g_server.model_name }) catch "";
        if (finish.len > 0) {
            _ = sseWriteData(stream, finish);
        }
    } else {
        // No parseable tool call. `<tool_call>` tags whose payload fails to
        // parse (a common small-model failure) must degrade to the raw text as
        // content, mirroring the non-streaming path, instead of emitting an
        // empty assistant turn that claims finish_reason "tool_calls". The tag
        // markup itself is dropped, so a client scanning for tags cannot read
        // the payload the server just refused.
        const stripped = stripToolCallTags(g_server.allocator, gen.raw) catch gen.raw;
        defer if (stripped.ptr != gen.raw.ptr) g_server.allocator.free(@constCast(stripped));
        const content = json.jsonEscape(g_server.allocator, stripped) catch gen.escaped;
        defer if (content.ptr != stripped.ptr and content.ptr != gen.escaped.ptr) g_server.allocator.free(@constCast(content));
        writeStreamedContent(stream, &chunk_buf, g_server.model_name, req_id, created, content, gen.finish_reason);
    }

    // Usage chunk + DONE
    if (sampling.stream_include_usage) {
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        const usage = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, g_server.model_name, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch "";
        if (usage.len > 0) {
            _ = sseWriteData(stream, usage);
        }
    }
    _ = sseWriteData(stream, "[DONE]");

    const tool_time_ms = elapsedMs(tool_stream_start);
    g_server.metrics.recordLatency(tool_time_ms);
    g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
    g_server.metrics.recordThroughput(gen.stats.tokens_generated, tool_time_ms);
    g_server.metrics.recordTPOT(gen.stats.tokens_generated, tool_time_ms);
    g_server.metrics.recordPromptTokens(gen.stats.prompt_tokens);
    g_server.metrics.recordGenerationTokens(gen.stats.tokens_generated);
    g_server.metrics.recordCompletion();
}

/// Start an SSE streaming response without chat template wrapping (for /v1/completions).
fn startStreamRaw(stream: http.TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateStream(stream, prompt, currentRequestId(), timestamp(), false, false, max_tokens, sampling);
}

const max_top_logprobs = math_ops.max_top_logprobs;

/// Per-token logprob info for OpenAI API response.
const LogprobInfo = struct {
    token_logprob: f32 = 0,
    top_ids: [max_top_logprobs]u32 = undefined,
    top_logprobs: [max_top_logprobs]f32 = undefined,
    count: u32 = 0,
};

/// Compute logprobs for a token from logits.
/// `n_top == 0` still returns the sampled token's own logprob with an empty
/// `top_logprobs` list, so `logprobs: true` works without `top_logprobs`.
fn computeLogprobs(logits: []const f32, token_id: u32, n_top: u32) LogprobInfo {
    var info: LogprobInfo = .{};
    if (n_top > 0) info.count = math_ops.topLogProbs(logits, n_top, &info.top_ids, &info.top_logprobs);
    info.token_logprob = math_ops.tokenLogProb(logits, token_id);
    return info;
}

/// Format logprobs JSON into buffer. Returns slice or empty on overflow.
fn formatLogprobs(buf: []u8, tok: *Tokenizer, token_text: []const u8, info: LogprobInfo) []const u8 {
    var pos: usize = 0;
    var lp_buf: [32]u8 = undefined;
    const lp_s = json.formatFiniteF32(&lp_buf, info.token_logprob);
    const header = std.fmt.bufPrint(buf, "\"logprobs\":{{\"content\":[{{\"token\":\"{s}\",\"logprob\":{s},\"top_logprobs\":[", .{ token_text, lp_s }) catch return "";
    pos = header.len;
    for (0..info.count) |i| {
        const top_decoded = decodeTokenText(tok, info.top_ids[i]) orelse continue;
        defer g_server.allocator.free(top_decoded);
        const top_escaped = json.jsonEscape(g_server.allocator, top_decoded) catch continue;
        defer if (top_escaped.ptr != top_decoded.ptr) g_server.allocator.free(top_escaped);
        const prefix: []const u8 = if (i > 0) "," else "";
        var top_lp_buf: [32]u8 = undefined;
        const top_lp_s = json.formatFiniteF32(&top_lp_buf, info.top_logprobs[i]);
        const entry = std.fmt.bufPrint(buf[pos..], "{s}{{\"token\":\"{s}\",\"logprob\":{s}}}", .{ prefix, top_escaped, top_lp_s }) catch return "";
        pos += entry.len;
    }
    const tail = "]}]}";
    if (pos + tail.len > buf.len) return "";
    @memcpy(buf[pos..][0..tail.len], tail);
    pos += tail.len;
    return buf[0..pos];
}

/// Stream a single token as an SSE chunk in OpenAI format.
/// Returns false if the write failed (client disconnected).
fn streamChunk(stream: http.TcpStream, chunk_buf: *[response_buf_size]u8, tok: *Tokenizer, token_id: u32, req_id: u64, created: i64, is_chat: bool, hb: *Utf8Holdback) bool {
    return streamChunkLogprobs(stream, chunk_buf, tok, token_id, req_id, created, is_chat, null, hb);
}

/// Format one text piece as an OpenAI SSE chunk and write it.
/// Returns false if the write failed (client disconnected).
fn writeOpenAiChunk(
    stream: http.TcpStream,
    chunk_buf: *[response_buf_size]u8,
    tok: *Tokenizer,
    text: []const u8,
    req_id: u64,
    created: i64,
    is_chat: bool,
    lp_info: ?LogprobInfo,
) bool {
    if (text.len == 0) return true;
    const escaped = json.jsonEscape(g_server.allocator, text) catch return true;
    defer if (escaped.ptr != text.ptr) g_server.allocator.free(escaped);

    var lp_buf: [logprob_buf_size]u8 = undefined;
    const lp_json: []const u8 = if (lp_info) |info|
        formatLogprobs(&lp_buf, tok, escaped, info)
    else
        "";
    const has_lp = lp_json.len > 0;

    const chunk = if (is_chat) blk: {
        break :blk if (has_lp)
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},{s},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped, lp_json })
        else
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped });
    } else blk: {
        break :blk if (has_lp)
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,{s},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped, lp_json })
        else
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped });
    };

    if (chunk) |c| {
        return sseWriteData(stream, c);
    } else |_| {
        std.log.warn("req={d} SSE stream chunk exceeded buffer ({d} bytes escaped)", .{ log_request_id, escaped.len });
        return true;
    }
}

fn streamChunkLogprobs(stream: http.TcpStream, chunk_buf: *[response_buf_size]u8, tok: *Tokenizer, token_id: u32, req_id: u64, created: i64, is_chat: bool, lp_info: ?LogprobInfo, hb: *Utf8Holdback) bool {
    // Fast path: allocation-free single-token decode into a stack buffer.
    var dec_buf: [stream_decode_buf_size]u8 = undefined;
    var decoded: []const u8 = undefined;
    var heap_decoded: ?[]u8 = null;
    defer if (heap_decoded) |h| g_server.allocator.free(h);
    if (tok.decodeOne(token_id, &dec_buf)) |d| {
        decoded = d;
    } else {
        const h = decodeTokenText(tok, token_id) orelse return true;
        if (h.len <= dec_buf.len) {
            @memcpy(dec_buf[0..h.len], h);
            g_server.allocator.free(h);
            decoded = dec_buf[0..h.len];
        } else {
            heap_decoded = h;
            decoded = h;
        }
    }

    // A token split by the UTF-8 holdback may emit two chunks; logprobs
    // describe the whole token, so attach them to the first piece only.
    const pieces = hb.feed(decoded);
    var first_piece = true;
    for ([_][]const u8{ pieces.head, pieces.body }) |piece| {
        if (piece.len == 0) continue;
        const piece_lp: ?LogprobInfo = if (first_piece) lp_info else null;
        first_piece = false;
        if (!writeOpenAiChunk(stream, chunk_buf, tok, piece, req_id, created, is_chat, piece_lp)) return false;
    }
    return true;
}

/// Send a usage-only SSE chunk (OpenAI streaming format).
/// Emitted after the final chunk and before [DONE] so clients can track token usage.
fn sendUsageChunk(stream: http.TcpStream, chunk_buf: *[response_buf_size]u8, req_id: u64, created: i64, is_chat: bool, prompt_tokens: u32, completion_tokens: u32) void {
    const total = prompt_tokens + completion_tokens;
    const id_prefix: []const u8 = if (is_chat) "chatcmpl" else "cmpl";
    const obj_type: []const u8 = if (is_chat) "chat.completion.chunk" else "text_completion";
    const chunk = std.fmt.bufPrint(chunk_buf,
        \\{{"id":"{s}-{d}","object":"{s}","created":{d},"model":"{s}","choices":[],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{ id_prefix, req_id, obj_type, created, g_server.model_name, prompt_tokens, completion_tokens, total }) catch {
        std.log.warn("req={d} SSE usage chunk exceeded buffer", .{log_request_id});
        return;
    };
    _ = sseWriteData(stream, chunk);
}

/// Send the final SSE chunk with the given finish_reason ("stop" or "length").
fn sendFinalChunk(stream: http.TcpStream, chunk_buf: *[response_buf_size]u8, req_id: u64, created: i64, is_chat: bool, finish_reason: []const u8) void {
    const id_prefix: []const u8 = if (is_chat) "chatcmpl" else "cmpl";
    const obj_type: []const u8 = if (is_chat) "chat.completion.chunk" else "text_completion";
    const delta_or_text: []const u8 = if (is_chat)
        \\"delta":{}
    else
        \\"text":""
    ;
    const final = std.fmt.bufPrint(chunk_buf,
        \\{{"id":"{s}-{d}","object":"{s}","created":{d},"model":"{s}","choices":[{{"index":0,{s},"finish_reason":"{s}"}}]}}
    , .{ id_prefix, req_id, obj_type, created, g_server.model_name, delta_or_text, finish_reason }) catch {
        std.log.warn("req={d} SSE final chunk exceeded buffer", .{log_request_id});
        return;
    };
    _ = sseWriteData(stream, final);
}

/// Run generation and stream tokens as SSE events in OpenAI format.
/// Always resets the cache (completions API requests are stateless).
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and polls for generated tokens. Falls back to direct model.forward()
/// when no scheduler is running (CLI mode).
fn generateStream(stream: http.TcpStream, prompt: []const u8, req_id: u64, created: i64, is_chat: bool, format_prompt: bool, max_tokens: usize, sampling: SamplingParams) void {
    const tok = g_server.tokenizer;

    const formatted = if (format_prompt)
        g_server.chat_template.format(g_server.allocator, null, prompt) catch prompt
    else
        prompt;
    defer if (format_prompt and formatted.ptr != prompt.ptr) wipeFree(g_server.allocator, @constCast(formatted));
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} streaming tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        _ = sseWriteData(stream, "[DONE]");
        return;
    };
    defer wipeFreeTokens(g_server.allocator, token_ids);

    // Send initial chunk (role announcement for chat completions)
    var chunk_buf: [response_buf_size]u8 = undefined;
    // UTF-8 holdback shared by both streaming paths below.
    var chunk_hb = Utf8Holdback{};
    if (is_chat) {
        const initial = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}
        , .{ req_id, created, g_server.model_name }) catch "";
        if (initial.len > 0) _ = sseWriteData(stream, initial);
    }

    // Scheduler path: enqueue request and poll for tokens.
    // Grammar-constrained/json_mode decoding bypasses the scheduler (grammar state
    // is per-request and the scheduler loop has no grammar/JSON support).
    const use_grammar_stream = (sampling.grammar_string != null or sampling.json_schema != null) and !sampling.json_mode;
    if (g_server.request_manager != null and !use_grammar_stream and !sampling.json_mode) {
        const rm = g_server.request_manager.?;
        const req = rm.enqueue(token_ids, log_request_id) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return;
        };
        configureSchedulerSampling(req, sampling);
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        const gen_start = milliTimestamp();
        var streamed_count: usize = 0;
        var token_count: u32 = 0;

        var chunk_client_connected = true;
        var itl = ItlTracker.init();
        var stop_buf: [scheduler_stop_buf_size]u8 = undefined;
        var stop_len: usize = 0;
        var checked_len: usize = 0;
        var hit_stop = false;
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (pollSchedulerStop(req, tok, sampling, &stop_buf, &stop_len, &checked_len, g_server.allocator)) |_| {
                hit_stop = true;
                break;
            }
            const stream_limit = if (sampling.hasStop()) checked_len else req.visible_len.load(.acquire);
            while (streamed_count < stream_limit) {
                const token_id = req.tokens.items[streamed_count];
                if (!streamChunk(stream, &chunk_buf, tok, token_id, req_id, created, is_chat, &chunk_hb)) {
                    chunk_client_connected = false;
                    req.is_cancelled.store(true, .release);
                    break;
                }
                streamed_count += 1;
                token_count += 1;
                itl.sample();
                if (token_count >= max_tokens) {
                    req.is_cancelled.store(true, .release);
                    break;
                }
            }
            if (!chunk_client_connected or token_count >= max_tokens) break;
            sleepNs(scheduler_poll_interval_ns);
        }

        while (!req.scheduler_done.load(.acquire))
            sleepNs(scheduler_poll_interval_ns);

        // Drain any remaining tokens after completion
        const final_len = if (hit_stop) checked_len else req.visible_len.load(.acquire);
        while (chunk_client_connected and streamed_count < final_len and token_count < max_tokens) {
            const token_id = req.tokens.items[streamed_count];
            if (!streamChunk(stream, &chunk_buf, tok, token_id, req_id, created, is_chat, &chunk_hb)) break;
            streamed_count += 1;
            token_count += 1;
        }
        if (hit_stop) token_count = @intCast(checked_len);

        // Send final chunk, usage chunk, and [DONE], skip if client already disconnected
        if (chunk_client_connected) {
            const held = chunk_hb.flush();
            _ = writeOpenAiChunk(stream, &chunk_buf, tok, held, req_id, created, is_chat, null);
            const sched_finish: []const u8 = if (token_count >= max_tokens) "length" else "stop";
            sendFinalChunk(stream, &chunk_buf, req_id, created, is_chat, sched_finish);
            if (sampling.stream_include_usage)
                sendUsageChunk(stream, &chunk_buf, req_id, created, is_chat, @intCast(token_ids.len), token_count);
            _ = sseWriteData(stream, "[DONE]");
        }

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const tps: f32 = tokensPerSec(token_count, time_ms);
        logGeneration(token_count, time_ms, tps);
        g_server.metrics.recordLatency(time_ms);
        g_server.metrics.recordTokens(token_count);
        // Record TTFT from scheduler's per-request prefill timestamp
        if (req.prefill_done_at > 0) {
            const openai_ttft = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(openai_ttft, @intCast(token_ids.len));
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        g_server.metrics.recordTPOT(token_count, time_ms);
        g_server.metrics.recordPromptTokens(@intCast(token_ids.len));
        g_server.metrics.recordGenerationTokens(token_count);
        if (!chunk_client_connected) {
            std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
            g_server.metrics.recordCancellation();
        } else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else if (req.is_timed_out.load(.acquire)) {
            std.log.warn("req={d} openai stream timed out (tokens={d})", .{ log_request_id, token_count });
        } else {
            std.log.warn("req={d} openai stream incomplete (tokens={d}, cancelled={})", .{
                log_request_id,
                token_count,
                req.is_cancelled.load(.acquire),
            });
            g_server.metrics.recordFailure();
        }
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    lockModelWithScheduler();
    defer unlockModelWithScheduler();
    const model = g_server.model;
    const n_visual = pending_visual_tokens;
    applyPendingVisionEmbeddings();

    // Prompt prefix caching (streaming): reuse KV cache for shared prefix
    var s_prefix_len: usize = 0;
    if (prefixCacheApplies(n_visual) and g_server.cached_prompt_ids.len > 0 and token_ids.len > 0) {
        const s_max_match = @min(g_server.cached_prompt_ids.len, token_ids.len);
        while (s_prefix_len < s_max_match and g_server.cached_prompt_ids[s_prefix_len] == token_ids[s_prefix_len]) {
            s_prefix_len += 1;
        }
        if (s_prefix_len > 0 and s_prefix_len < token_ids.len) {
            const s_bos_off: usize = if (g_server.bos_token_id > 0) 1 else 0;
            model.setKvSeqLen(s_prefix_len + s_bos_off);
        } else {
            s_prefix_len = 0;
        }
    }
    if (s_prefix_len == 0) {
        model.resetCache();
        g_server.clearCachedPromptIds();
    }

    // BOS token, required by models like Gemma to initialize state correctly
    if (s_prefix_len == 0 and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.err("req={d} BOS forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return;
        };
    }

    // Grammar-constrained streaming
    var s_grammar: ?grammar_mod.Grammar = null;
    var s_grammar_state: ?grammar_mod.GrammarState = null;
    defer {
        if (s_grammar_state) |*gs| gs.deinit();
        if (s_grammar) |*g| g.deinit();
    }
    const use_grammar_s = (sampling.grammar_string != null or sampling.json_schema != null) and !sampling.json_mode;
    if (sampling.json_schema) |schema| {
        s_grammar = grammar_mod.Grammar.fromJsonSchema(g_server.allocator, schema) catch |err| blk: {
            std.log.err("req={d} json_schema grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (s_grammar) |*g| s_grammar_state = g.initState() catch |err| blk: {
            std.log.err("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    } else if (sampling.grammar_string) |gs| {
        const unescaped_gs_s = json.jsonUnescape(g_server.allocator, gs) catch gs;
        defer if (unescaped_gs_s.ptr != gs.ptr) g_server.allocator.free(@constCast(unescaped_gs_s));
        s_grammar = grammar_mod.Grammar.parse(g_server.allocator, unescaped_gs_s) catch |err| blk: {
            std.log.err("req={d} grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (s_grammar) |*g| s_grammar_state = g.initState() catch |err| blk: {
            std.log.err("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    }
    // Fail closed: never stream unconstrained tokens when the client asked for a grammar.
    if (use_grammar_s and (s_grammar == null or s_grammar_state == null)) {
        // The prefix rollback above already truncated KV seq_len; drop the
        // bookkeeping so the next request re-prefills instead of trusting it.
        invalidateKvBookkeeping();
        g_server.metrics.recordFailure();
        _ = sseWriteData(stream, "{\"error\":\"grammar setup failed\"}");
        _ = sseWriteData(stream, "[DONE]");
        return;
    }

    // Prefill, capture the last forward's return value (first generated token)
    const use_sampling_s = sampling.temperature > 0;
    const prng_seed_s = prngSeedFromSampling(sampling);
    noteAutoSeed(sampling, prng_seed_s);
    var prng_s = std.Random.Xoshiro256.init(prng_seed_s);
    var mirostat_mu_s: f32 = sampling.mirostat_tau * 2.0;
    const prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids[s_prefix_len..]) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                std.log.info("req={d} openai stream prefill cancelled", .{log_request_id});
                invalidateKvBookkeeping();
                g_server.metrics.recordCancellation();
                _ = sseWriteData(stream, "[DONE]");
                return;
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            invalidateKvBookkeeping();
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return;
        };
    }
    const stream_prefill_ms: u64 = elapsedMs(prefill_start);
    g_server.metrics.recordTTFT(stream_prefill_ms, @intCast(token_ids.len));
    const s_vocab_texts = g_server.tokenizer.getVocabTexts();
    // Grammar masking on first token
    if (use_grammar_s and token_ids.len > 0) {
        if (s_grammar) |*g| {
            if (s_grammar_state) |*gs| {
                const s_first_logits = model.getLogits();
                g.maskLogits(gs, s_first_logits, s_vocab_texts) catch {
                    std.log.warn("req={d} stream grammar mask OOM", .{log_request_id});
                    g_server.metrics.recordFailure();
                    _ = sseWriteData(stream, "{\"error\":\"grammar OOM\"}");
                    _ = sseWriteData(stream, "[DONE]");
                    return;
                };
                first_gen_token = math_ops.argmax(s_first_logits);
            }
        }
    } else if (use_sampling_s and token_ids.len > 0) {
        const s_first_logits = model.getLogits();
        if (sampling.mirostat >= 2) {
            first_gen_token = math_ops.sampleMirostat(s_first_logits, sampling.mirostat_tau, sampling.mirostat_eta, &mirostat_mu_s, sampling.temperature, prng_s.random());
        } else {
            if (sampling.min_p > 0) math_ops.applyMinP(s_first_logits, sampling.min_p);
            if (sampling.xtc_probability > 0) math_ops.applyXtc(s_first_logits, sampling.xtc_probability, sampling.xtc_threshold, prng_s.random());
            first_gen_token = math_ops.sampleToken(s_first_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng_s.random());
        }
    }
    // Accept first token in grammar, use raw vocab text for consistent BPE handling.
    if (use_grammar_s and s_grammar_state != null and token_ids.len > 0) {
        const ft_raw = if (first_gen_token < s_vocab_texts.len) s_vocab_texts[first_gen_token] else "";
        s_grammar_state.?.acceptToken(ft_raw);
    }

    // Generate and stream tokens
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    // Stream the first generated token (from last prefill forward)
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        if (!streamChunk(stream, &chunk_buf, tok, first_gen_token, req_id, created, is_chat, &chunk_hb)) {
            logGeneration(0, 0, 0);
            std.log.warn("req={d} client disconnected before first stream chunk", .{log_request_id});
            g_server.metrics.recordCancellation();
            return;
        }
        last = first_gen_token;
        token_count = 1;
    }

    var stream_forward_failed = false;
    var stream_disconnected = false;

    const s_has_draft = g_server.draft_model != null and token_ids.len > 0 and !(token_count == 0 and g_server.isEog(first_gen_token));
    var s_spec_storage: spec_decode.SpecState = undefined;
    var s_spec_valid = false;
    if (s_has_draft) {
        if (spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize())) |ok| {
            s_spec_storage = ok;
            s_spec_valid = true;
        } else |_| {
            s_spec_storage = spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        }
    }
    defer if (s_spec_valid) s_spec_storage.deinit(g_server.allocator);

    if (s_spec_valid) {
        // Speculative streaming: emit batches of accepted tokens
        var s_draft = g_server.draft_model.?;
        if (s_draft.ptr != model.ptr) {
            _ = s_draft.prefill(token_ids) catch |err| {
                std.log.warn("req={d} draft model prefill failed: {s}", .{ log_request_id, @errorName(err) });
            };
        }
        const s_spec = &s_spec_storage;
        // Token history for penalty tracking, mirroring the non-spec path below.
        var s_spec_tokens: [gen_ids_buf_size]u32 = undefined;
        defer @memset(std.mem.sliceAsBytes(&s_spec_tokens), 0);
        var s_spec_count: u32 = 0;
        if (token_count > 0) {
            s_spec_tokens[0] = first_gen_token;
            s_spec_count = 1;
        }
        while (token_count < max_tokens and !stream_disconnected) {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            const pre = model.kvSeqLen();
            const is_self = (model.ptr == s_draft.ptr);
            const pen_s: spec_decode.Penalties = .{
                .history = s_spec_tokens[0..s_spec_count],
                .repeat_penalty = sampling.repetition_penalty,
                .dry_multiplier = sampling.dry_multiplier,
                .dry_length = sampling.dry_allowed_length,
            };
            const nd = if (is_self and !use_sampling_s)
                spec_decode.draft(s_spec, s_draft, last)
            else
                spec_decode.draftWithLogits(s_spec, s_draft, last);
            if (nd == 0) break;
            const res = if (is_self)
                spec_decode.SpecResult{ .accepted = s_spec.n_draft, .next_token = blk: {
                    s_spec.recordRound(s_spec.n_draft);
                    const ld = s_spec.draft_tokens[s_spec.n_draft - 1];
                    break :blk model.forward(ld) catch ld;
                } }
            else if (use_sampling_s)
                spec_decode.verifySampling(s_spec, model, s_draft, last, pre, sampling.temperature, pen_s, prng_s.random())
            else
                spec_decode.verifySequential(s_spec, model, s_draft, last, pre, pen_s);

            for (0..res.accepted) |i| {
                const at = s_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (token_count >= max_tokens) break;
                if (!streamChunk(stream, &chunk_buf, tok, at, req_id, created, is_chat, &chunk_hb)) {
                    stream_disconnected = true;
                    break;
                }
                if (s_spec_count < s_spec_tokens.len) {
                    s_spec_tokens[s_spec_count] = at;
                    s_spec_count += 1;
                }
                token_count += 1;
            }
            if (!stream_disconnected and !g_server.isEog(res.next_token)) {
                if (token_count < max_tokens) {
                    if (!streamChunk(stream, &chunk_buf, tok, res.next_token, req_id, created, is_chat, &chunk_hb)) {
                        stream_disconnected = true;
                    }
                    if (s_spec_count < s_spec_tokens.len) {
                        s_spec_tokens[s_spec_count] = res.next_token;
                        s_spec_count += 1;
                    }
                    token_count += 1;
                }
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
        // Standard streaming, token history for penalty tracking
        const use_penalties_s = sampling.frequency_penalty != 0 or sampling.presence_penalty != 0 or sampling.repetition_penalty != 1.0 or sampling.dry_multiplier > 0;
        var s_gen_tokens: [gen_ids_buf_size]u32 = undefined;
        defer @memset(std.mem.sliceAsBytes(&s_gen_tokens), 0);
        var s_gen_count: u32 = 0;
        if (use_penalties_s and token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
            s_gen_tokens[0] = first_gen_token;
            s_gen_count = 1;
        }

        var itl = ItlTracker.init();

        // Thinking budget state (Anthropic-style: limit reasoning token count).
        const think_budget = sampling.thinking_budget_tokens;
        var think_token_count: u32 = 0;
        var in_think_block: bool = false;
        var close_think_owned_s: ?[]u32 = null;
        defer if (close_think_owned_s) |ids| g_server.allocator.free(ids);
        const vocab_texts_s = g_server.tokenizer.getVocabTexts();
        if (think_budget > 0) {
            close_think_owned_s = g_server.tokenizer.encode("</think>") catch null;
        }
        const close_think_ids_s: []const u32 = close_think_owned_s orelse &.{};
        // Check if the first token is the start of a thinking block
        if (think_budget > 0 and token_ids.len > 0 and !g_server.isEog(first_gen_token) and first_gen_token < vocab_texts_s.len) {
            const ft_text = vocab_texts_s[first_gen_token];
            if (std.mem.indexOf(u8, ft_text, "<think>") != null) in_think_block = true;
            if (std.mem.indexOf(u8, ft_text, "</think>") != null) in_think_block = false;
            if (in_think_block) think_token_count += 1;
        }

        for (0..max_tokens -| 1) |_| {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;

            // Jump decoding (streaming): skip forward pass when grammar has single valid token
            if (use_grammar_s and !use_sampling_s) {
                if (s_grammar) |*g| {
                    if (s_grammar_state) |*gs| {
                        if (g.singleValidToken(gs, s_vocab_texts)) |jump_tok| {
                            const jt_raw_s = if (jump_tok < s_vocab_texts.len) s_vocab_texts[jump_tok] else "";
                            gs.acceptToken(jt_raw_s);
                            if (!streamChunk(stream, &chunk_buf, tok, jump_tok, req_id, created, is_chat, &chunk_hb)) {
                                stream_disconnected = true;
                                break;
                            }
                            last = jump_tok;
                            token_count += 1;
                            if (gs.isComplete()) break;
                            continue;
                        }
                    }
                }
            }

            var next = model.forward(last) catch |err| {
                if (err != error.Cancelled) {
                    std.log.warn("req={d} generation forward failed: {}", .{ log_request_id, err });
                    stream_forward_failed = true;
                }
                break;
            };
            const s_logits = model.getLogits();
            if (sampling.logit_bias_count > 0) {
                math_ops.applyLogitBias(s_logits, &sampling.logit_bias_ids, &sampling.logit_bias_vals, sampling.logit_bias_count);
            }
            // Thinking budget: when in a think block and budget exhausted, heavily bias
            // towards the end-of-thinking token to force the model out of reasoning.
            if (think_budget > 0 and in_think_block and think_token_count >= think_budget) {
                for (close_think_ids_s) |tid| {
                    if (tid < @as(u32, @intCast(s_logits.len))) {
                        s_logits[tid] += 100.0;
                    }
                }
            }
            if (sampling.repetition_penalty != 1.0 and s_gen_count > 0) {
                math_ops.applyRepeatPenalty(s_logits, s_gen_tokens[0..s_gen_count], sampling.repetition_penalty);
            }
            if (sampling.dry_multiplier > 0 and s_gen_count > 0) {
                math_ops.applyDry(s_logits, s_gen_tokens[0..s_gen_count], sampling.dry_multiplier, sampling.dry_allowed_length);
            }
            if (use_penalties_s) {
                math_ops.applyPenalties(s_logits, s_gen_tokens[0..s_gen_count], sampling.frequency_penalty, sampling.presence_penalty);
            }
            if (use_grammar_s) {
                if (s_grammar) |*g| {
                    if (s_grammar_state) |*gs| {
                        g.maskLogits(gs, s_logits, s_vocab_texts) catch {
                            std.log.warn("req={d} stream grammar mask OOM", .{log_request_id});
                            stream_forward_failed = true;
                            _ = sseWriteData(stream, "{\"error\":\"grammar OOM\"}");
                            break;
                        };
                        next = math_ops.argmax(s_logits);
                    }
                }
            } else if (use_sampling_s) {
                if (sampling.mirostat >= 2) {
                    next = math_ops.sampleMirostat(s_logits, sampling.mirostat_tau, sampling.mirostat_eta, &mirostat_mu_s, sampling.temperature, prng_s.random());
                } else {
                    if (sampling.min_p > 0) math_ops.applyMinP(s_logits, sampling.min_p);
                    if (sampling.xtc_probability > 0) math_ops.applyXtc(s_logits, sampling.xtc_probability, sampling.xtc_threshold, prng_s.random());
                    next = math_ops.sampleToken(s_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng_s.random());
                }
            } else if (use_penalties_s or sampling.logit_bias_count > 0) {
                next = math_ops.argmax(s_logits);
            }
            // Compute logprobs before EOG/stop checks (logits still valid)
            const lp: ?LogprobInfo = if (sampling.logprobs) computeLogprobs(s_logits, next, sampling.top_logprobs) else null;

            if (g_server.isEog(next)) break;
            // Accept in grammar
            if (use_grammar_s and s_grammar_state != null) {
                const s_raw_tok = if (next < s_vocab_texts.len) s_vocab_texts[next] else "";
                s_grammar_state.?.acceptToken(s_raw_tok);
                if (s_grammar_state.?.isComplete()) {
                    if (!streamChunkLogprobs(stream, &chunk_buf, tok, next, req_id, created, is_chat, lp, &chunk_hb)) stream_disconnected = true;
                    token_count += 1;
                    break;
                }
            }
            // Stop sequence check (decode token, check trailing text)
            if (sampling.hasStop()) {
                var stop_text_buf: [stream_decode_buf_size]u8 = undefined;
                const stext_opt = g_server.tokenizer.decodeOne(next, &stop_text_buf);
                if (stext_opt != null and stext_opt.?.len > 0 and sampling.matchesStop(stext_opt.?)) {
                    token_count += 1;
                    break;
                }
                if (stext_opt == null) {
                    // Fallback for oversized tokens: allocating batch decode.
                    const stok = [1]u32{next};
                    const stext = g_server.tokenizer.decode(&stok) catch |err| blk: {
                        std.log.warn("req={d} stop seq decode failed (id={d}): {}", .{ log_request_id, next, err });
                        break :blk null;
                    };
                    defer if (stext) |st| g_server.allocator.free(st);
                    if (stext != null and stext.?.len > 0 and sampling.matchesStop(stext.?)) {
                        token_count += 1;
                        break;
                    }
                }
            }
            if (!streamChunkLogprobs(stream, &chunk_buf, tok, next, req_id, created, is_chat, lp, &chunk_hb)) {
                stream_disconnected = true;
                break;
            }
            if (use_penalties_s and s_gen_count < gen_ids_buf_size) {
                s_gen_tokens[s_gen_count] = next;
                s_gen_count += 1;
            }
            // Update thinking block state for next iteration (vocab lookup, no per-token alloc).
            if (think_budget > 0 and next < vocab_texts_s.len) {
                const tok_text = vocab_texts_s[next];
                if (std.mem.indexOf(u8, tok_text, "<think>") != null) in_think_block = true;
                if (std.mem.indexOf(u8, tok_text, "</think>") != null) in_think_block = false;
                if (in_think_block) think_token_count += 1;
            }

            // Record inter-token latency
            itl.sample();

            last = next;
            token_count += 1;
        }
    }

    // Send final chunk, usage chunk, and [DONE], skip if client already disconnected
    if (!stream_disconnected) {
        const held = chunk_hb.flush();
        _ = writeOpenAiChunk(stream, &chunk_buf, tok, held, req_id, created, is_chat, null);
        const direct_finish: []const u8 = if (stream_forward_failed) "error" else if (token_count >= max_tokens) "length" else "stop";
        sendFinalChunk(stream, &chunk_buf, req_id, created, is_chat, direct_finish);
        if (sampling.stream_include_usage)
            sendUsageChunk(stream, &chunk_buf, req_id, created, is_chat, @intCast(token_ids.len), token_count);
        _ = sseWriteData(stream, "[DONE]");
    }

    const gen_end = milliTimestamp();
    const time_ms = elapsedBetween(gen_start, gen_end);
    const tps: f32 = tokensPerSec(token_count, time_ms);
    logGeneration(token_count, time_ms, tps);
    g_server.metrics.recordThroughput(token_count, time_ms);
    g_server.metrics.recordTPOT(token_count, time_ms);
    g_server.metrics.recordPromptTokens(@intCast(token_ids.len));
    g_server.metrics.recordGenerationTokens(token_count);
    g_server.metrics.recordLatency(time_ms);
    g_server.metrics.recordTokens(token_count);
    if (stream_disconnected) {
        std.log.warn("req={d} client disconnected during streaming ({d} tokens sent)", .{ log_request_id, token_count });
        g_server.metrics.recordCancellation();
    } else if (stream_forward_failed) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

    // Update prompt prefix cache for next request (zeros old IDs).
    g_server.publishCachedPromptIds(token_ids, n_visual);
}

// JSON field extraction, encoding, and form-parsing utilities are in json.zig.

// ── Connection handler & server entry point ─────────────────────

fn handleConnection(stream: http.TcpStream) void {
    // Assign the correlation ID before any log so socket-option failures
    // can be grepped with the same req=N as the rest of the request.
    log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
    log_client_rid_len = 0;
    log_idem_token = 0;
    // Set read/write timeouts to prevent slow loris attacks, without this,
    // stream.read()/writeAll() block indefinitely and an attacker can exhaust
    // all max_concurrent_connections slots with incomplete requests or stalled reads.
    const timeout = std.posix.timeval{ .sec = connection_read_timeout_sec, .usec = 0 };
    std.posix.setsockopt(stream.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("req={d} failed to set connection read timeout: {}", .{ log_request_id, err });
    };
    std.posix.setsockopt(stream.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("req={d} failed to set connection write timeout: {}", .{ log_request_id, err });
    };
    // Disable Nagle's algorithm, SSE streaming writes small token chunks (~20-100 bytes)
    // that Nagle would buffer for up to 200ms waiting for ACK coalescing.
    const nodelay_val: c_int = 1;
    std.posix.setsockopt(stream.handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&nodelay_val)) catch |err| {
        std.log.warn("req={d} failed to set TCP_NODELAY: {}", .{ log_request_id, err });
    };

    // active_connections already incremented by accept loop before thread spawn.
    defer {
        _ = g_server.metrics.active_connections.fetchSub(1, .release);
        stream.close();
    }
    // Heap buffer: avoid ~1MB stack per connection thread (up to max_concurrent_connections).
    const buf = g_server.allocator.alloc(u8, http_buf_size) catch {
        g_server.metrics.recordRequest();
        g_server.metrics.recordFailure();
        const t = getTimeComponents();
        slog(log_time_fmt ++ " req={d} OOM allocating {d}-byte connection buffer -> 503\n", .{ t.hours, t.minutes, t.seconds, log_request_id, http_buf_size });
        sendJsonErrorEx(stream, "503 Service Unavailable", "server_error", "Out of memory", null, "server_overloaded");
        return;
    };
    // Wipe before free: buffer holds Authorization secrets and prompt/message bodies.
    defer {
        @memset(buf, 0);
        g_server.allocator.free(buf);
    }
    switch (http.readHttpRequest(stream, buf, max_request_body_size)) {
        .ok => |req| handleRequest(stream, req),
        .body_too_large => |ep_path| {
            g_server.metrics.recordRequest();
            g_server.metrics.recordClientError();
            const t = getTimeComponents();
            slog(log_time_fmt ++ " req={d} Rejected oversized request body (>{d} bytes) -> 413\n", .{ t.hours, t.minutes, t.seconds, log_request_id, max_request_body_size });
            if (isAnthropicPath(ep_path)) {
                sendAnthropicError(stream, "413", "request_too_large", "Request body too large");
            } else {
                sendJsonErrorEx(stream, "413 Payload Too Large", "invalid_request_error", "Request body too large", null, "request_too_large");
            }
        },
        .malformed => |ep_path| {
            g_server.metrics.recordRequest();
            g_server.metrics.recordClientError();
            const t = getTimeComponents();
            slog(log_time_fmt ++ " req={d} Malformed HTTP request -> 400\n", .{ t.hours, t.minutes, t.seconds, log_request_id });
            if (isAnthropicPath(ep_path)) {
                sendAnthropicError(stream, "400", "invalid_request_error", "Malformed HTTP request");
            } else {
                sendJsonErrorEx(stream, "400 Bad Request", "invalid_request_error", "Malformed HTTP request", null, "malformed_request");
            }
        },
        // Connection-level failures below are not client protocol errors: no
        // 4xx is produced (the peer is gone or unresponsive), so they are kept
        // out of requests_client_error to preserve that signal's meaning.
        // Probes/port scans make plain closes common; debug so production
        // ReleaseFast logs are not flooded.
        .connection_closed => {
            g_server.metrics.recordRequest();
            std.log.debug("req={d} connection closed before request completed", .{log_request_id});
        },
        .read_error => {
            g_server.metrics.recordRequest();
            const t = getTimeComponents();
            slog(log_time_fmt ++ " req={d} Request read failed (timeout or reset) before completion -> closing\n", .{ t.hours, t.minutes, t.seconds, log_request_id });
        },
    }
}

/// Configuration for the HTTP server.
pub const ServerConfig = struct {
    allocator: Allocator,
    model: *Model,
    tokenizer: *Tokenizer,
    chat_template: ChatTemplate,
    model_name: []const u8,
    backend_name: []const u8,
    port: u16,
    bos_token_id: u32,
    eog_ids: [max_eog_ids]u32,
    eog_len: usize,
    tiered_cache: ?*TieredKvCache,
    api_key: ?[]const u8,
    host: [4]u8,
    ctx_size: u32,
    vision_encoder: ?*VisionEncoder,
    image_pad_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    io: Io,
    draft_model: ?*Model = null,
    spec_tokens: u32 = 5,
    tree_budget: u32 = 64,
    /// Seconds of inactivity before entering sleep mode (0 = disabled).
    /// Flag-only: sets `/health` `sleeping` (and the metrics gauge). Weights, KV,
    /// and prefix/ngram state stay resident; wake is automatic on the next request.
    sleep_after_s: u32 = 0,
    /// Maximum number of requests to batch together in one scheduler cycle (default 8).
    max_batch_size: u32 = 8,
    /// Max requests per minute (0 = no request-rate limit). Enabling either
    /// rate_limit_rpm or rate_limit_tpm installs the token-bucket limiter.
    rate_limit_rpm: u32 = 0,
    /// Max prompt tokens per minute (0 = no token-rate limit).
    rate_limit_tpm: u32 = 0,
    /// Explicit conversation store path. Ignored when `no_conv_store` is set.
    conv_store_path: ?[]const u8 = null,
    /// Disable conversation persist/restore (in-memory only).
    no_conv_store: bool = false,
};

/// Start the HTTP server with OpenAI-compatible API endpoints.
/// Blocks until the server shuts down (via Ctrl+C).
pub fn run(config: ServerConfig) !void {
    const allocator = config.allocator;
    const io = config.io;
    const model = config.model;
    const tok = config.tokenizer;
    const chat_tmpl = config.chat_template;
    const model_name = config.model_name;
    const backend_name = config.backend_name;
    const port = config.port;
    const bos_token_id = config.bos_token_id;
    const eog_ids = config.eog_ids;
    const eog_len = config.eog_len;
    const tiered_cache = config.tiered_cache;
    const api_key = config.api_key;
    const host = config.host;
    const ctx_size = config.ctx_size;
    const vision_encoder = config.vision_encoder;
    const image_pad_token_id = config.image_pad_token_id;
    const image_start_token_id = config.image_start_token_id;
    const image_end_token_id = config.image_end_token_id;
    // Pre-sanitize model name for JSON safety (defense against crafted GGUF metadata).
    // Replace JSON-breaking characters (", \, control chars) with '_' so the name
    // can be safely embedded in JSON format strings without escaping per-call.
    var model_name_buf: ?[]u8 = null;
    defer if (model_name_buf) |b| allocator.free(b);
    const safe_model_name: []const u8 = blk: {
        for (model_name) |c| {
            if (http.isUnsafeJsonChar(c)) {
                const buf = allocator.alloc(u8, model_name.len) catch break :blk model_name;
                for (buf, model_name) |*d, sc| {
                    d.* = if (http.isUnsafeJsonChar(sc)) '_' else sc;
                }
                model_name_buf = buf;
                break :blk buf;
            }
        }
        break :blk model_name;
    };

    // Stack-allocate the Server struct. This is safe because run() blocks
    // until the server shuts down, so the frame stays alive.
    var server = Server{
        .model = model,
        .tokenizer = tok,
        .chat_template = chat_tmpl,
        .model_name = safe_model_name,
        .backend_name = backend_name,
        .allocator = allocator,
        .bos_token_id = bos_token_id,
        .eog_ids = eog_ids,
        .eog_len = eog_len,
        .ctx_size = ctx_size,
        .vision_encoder = vision_encoder,
        .image_pad_token_id = image_pad_token_id,
        .image_start_token_id = image_start_token_id,
        .image_end_token_id = image_end_token_id,
        .io = io,
        .idem = Idempotency.Ledger.init(allocator),
        .draft_model = config.draft_model,
        .spec_tokens = config.spec_tokens,
        .tree_budget = config.tree_budget,
        .sleep_after_s = config.sleep_after_s,
    };
    server.api_key = api_key;
    server.start_time = timestamp();
    server.start_mono_ms = milliTimestamp();

    var conv_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (config.no_conv_store) {
        server.conv_store_path = null;
    } else if (config.conv_store_path) |p| {
        server.conv_store_path = p;
    } else {
        server.conv_store_path = conv_store.defaultPath(&conv_path_buf);
        if (server.conv_store_path == null) {
            std.log.warn("conversation store disabled: no XDG_CACHE_HOME or HOME (pass --conv-store PATH)", .{});
        }
    }
    server.loadConversationsLocked();
    server.metrics.process_start_time.store(server.start_time, .monotonic);

    // Optional token-bucket rate limiter (null when both limits are 0).
    var rate_limiter_storage: RateLimiter = undefined;
    if (config.rate_limit_rpm > 0 or config.rate_limit_tpm > 0) {
        const rpm = if (config.rate_limit_rpm > 0) config.rate_limit_rpm else rate_limit_unlimited_rpm;
        const tpm = if (config.rate_limit_tpm > 0) config.rate_limit_tpm else rate_limit_unlimited_tpm;
        rate_limiter_storage = RateLimiter.init(rpm, tpm, io);
        server.rate_limiter = &rate_limiter_storage;
        std.log.info("server: rate limit enabled (rpm={d} tpm={d})", .{ rpm, tpm });
    }

    g_server = &server;

    server.html_gzip = http.gzipAlloc(allocator, html_page) catch |err| blk: {
        std.log.warn("server: gzip of chat UI failed ({s}), serving uncompressed", .{@errorName(err)});
        break :blk &.{};
    };
    defer if (server.html_gzip.len != 0) allocator.free(server.html_gzip);
    defer server.idem.deinit();

    // Initialize continuous batching scheduler and background thread.
    // The scheduler owns the model forward loop; HTTP handlers enqueue
    // requests and poll for results instead of calling model.forward() directly.
    const effective_batch_size: usize = if (config.max_batch_size > 0) config.max_batch_size else scheduler_max_batch_size;
    // Admission runs one request at a time while the model layer exposes a
    // single shared KV sequence (see scheduler.max_running_requests_single_sequence);
    // `--max-batch-size` becomes effective concurrent decoding once per-request
    // paged KV is wired end to end.
    const admission_limit = @min(effective_batch_size, scheduler.max_running_requests_single_sequence);
    var request_manager = try scheduler.RequestManager.init(allocator, &server.metrics, effective_batch_size, scheduler_timeout_sec, tiered_cache, io);
    defer request_manager.deinit();
    server.request_manager = &request_manager;

    const eog_slice = server.eog_ids[0..server.eog_len];
    const sched_thread = try std.Thread.spawn(.{}, scheduler.runSchedulerLoop, .{
        &request_manager,
        server.model,
        eog_slice,
        &server.scheduler_shutdown,
    });
    server.scheduler_thread = sched_thread;
    errdefer {
        server.scheduler_shutdown.store(true, .release);
        sched_thread.join();
        server.scheduler_thread = null;
    }

    // Sleep-mode monitor: background thread checks idle time every 10 seconds.
    // When the server has been idle for sleep_after_s seconds, sets sleeping=true
    // and logs a message. Wake-up happens automatically on the next request.
    var sleep_thread: ?std.Thread = null;
    if (config.sleep_after_s > 0) {
        server.last_request_ms.store(milliTimestamp(), .release);
        sleep_thread = std.Thread.spawn(.{}, sleepMonitorLoop, .{&server.scheduler_shutdown}) catch null;
        if (sleep_thread == null) std.log.warn("server: failed to start sleep monitor thread", .{});
        if (sleep_thread != null) std.log.info("server: sleep mode enabled (idle timeout: {d}s)", .{config.sleep_after_s});
    }
    // The monitor only exits once the shutdown flag is set, and it runs before
    // the errdefer that sets it, so set it here: joining first would hang.
    defer if (sleep_thread) |t| {
        server.scheduler_shutdown.store(true, .release);
        t.join();
    };

    const address = net.IpAddress{ .ip4 = .{ .bytes = host, .port = port } };
    var tcp = net.IpAddress.listen(&address, io, .{ .reuse_address = true }) catch |err| {
        var buf: [error_body_buf_size]u8 = undefined;
        // Port-in-use is by far the most common listen failure; give it an
        // hint naming the fix instead of a raw error name (matches pull/main error style).
        const msg = if (err == error.AddressInUse)
            std.fmt.bufPrint(&buf, "Error: port {d} is already in use (another server may be running).\n  Start on a different port with --port <PORT>.\n", .{port}) catch ""
        else
            std.fmt.bufPrint(&buf, "Error: failed to listen on port {d}: {s}\n", .{ port, @errorName(err) }) catch "";
        _ = std.posix.system.write(stderr_file.handle, msg.ptr, msg.len);
        return error.ListenError;
    };

    // Set accept timeout so signal handler can interrupt the loop.
    // Without this, accept() blocks indefinitely and Ctrl+C doesn't work on macOS.
    const timeout = std.posix.timeval{ .sec = accept_timeout_sec, .usec = 0 };
    std.posix.setsockopt(tcp.socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("server: failed to set accept timeout: {}", .{err});
    };
    defer tcp.deinit(io);

    const t = getTimeComponents();
    var buf: [hdr_buf_size]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "\n" ++ log_time_fmt ++ " agave server started on http://{d}.{d}.{d}.{d}:{d}\n  model={s} backend={s}\n  ctx_size={d} max_conn={d} batch={d} timeout={d}s auth={s} rate_limit={s}\nPress Ctrl+C to stop\n", .{ t.hours, t.minutes, t.seconds, host[0], host[1], host[2], host[3], port, model_name, backend_name, ctx_size, max_concurrent_connections, admission_limit, scheduler_timeout_sec, if (api_key != null) "yes" else "no", if (server.rate_limiter != null) "yes" else "no" }) catch "";
    _ = std.posix.system.write(stdout_file.handle, msg.ptr, msg.len);

    // Install graceful shutdown handlers for SIGTERM and SIGINT.
    // First signal: graceful shutdown (drain active connections).
    // Second signal: immediate process exit.
    const handler = struct {
        fn handle(_: std.posix.SIG) callconv(.c) void {
            if (g_server.shutdown_requested.load(.acquire)) {
                // Second signal, force immediate exit
                const force_msg = "\nForced shutdown.\n";
                _ = std.posix.system.write(stderr_file.handle, force_msg.ptr, force_msg.len);
                std.process.exit(1);
            }
            // First signal, write immediately (async-signal-safe)
            const shutdown_msg = "\nShutting down (Ctrl+C again to force)...\n";
            _ = std.posix.system.write(stderr_file.handle, shutdown_msg.ptr, shutdown_msg.len);
            g_server.shutdown_requested.store(true, .release);
            g_server.scheduler_shutdown.store(true, .release);
            g_server.model.cancel();
        }
    };
    const act = std.posix.Sigaction{
        .handler = .{ .handler = handler.handle },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.TERM, &act, null);
    std.posix.sigaction(std.posix.SIG.INT, &act, null);

    // Accept loop, each connection handled on its own thread.
    // Reject new connections when at the concurrency limit to prevent resource exhaustion.
    // Exit loop when graceful shutdown is requested.
    while (!g_server.shutdown_requested.load(.acquire)) {
        const net_stream = tcp.accept(io) catch |err| {
            if (g_server.shutdown_requested.load(.acquire)) break;
            // Timeout is expected, allows periodic shutdown check
            if (err == error.WouldBlock or err == error.Unexpected) continue;
            std.log.err("Accept failed: {}", .{err});
            continue;
        };
        const stream = http.TcpStream{ .handle = net_stream.socket.handle };
        // Atomically increment before capacity check to prevent TOCTOU race
        // where multiple accept() calls pass the check before any thread increments.
        const prev = g_server.metrics.active_connections.fetchAdd(1, .acquire);
        if (prev >= max_concurrent_connections) {
            _ = g_server.metrics.active_connections.fetchSub(1, .release);
            log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
            g_server.metrics.recordRequest();
            g_server.metrics.recordConnectionRejection();
            const tc = getTimeComponents();
            slog(log_time_fmt ++ " req={d} Connection rejected: at capacity ({d}/{d}) -> 503\n", .{ tc.hours, tc.minutes, tc.seconds, log_request_id, max_concurrent_connections, max_concurrent_connections });
            send503Retry(stream, capacity_503_body, capacity_retry_after_sec);
            stream.close();
            continue;
        }
        const thread = std.Thread.spawn(.{}, handleConnection, .{stream}) catch |err| {
            _ = g_server.metrics.active_connections.fetchSub(1, .release);
            log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
            g_server.metrics.recordRequest();
            g_server.metrics.recordFailure();
            std.log.err("Failed to spawn connection handler thread: {}", .{err});
            send503Retry(stream, spawn_fail_503_body, capacity_retry_after_sec);
            stream.close();
            continue;
        };
        thread.detach();
    }

    // Log shutdown (signal handler cannot safely log, do it here)
    {
        const tc = getTimeComponents();
        slog("\n" ++ log_time_fmt ++ " Server shutting down...\n", .{ tc.hours, tc.minutes, tc.seconds });
    }

    // Stop scheduler thread before draining connections
    server.scheduler_shutdown.store(true, .release);
    if (server.scheduler_thread) |sched_t| sched_t.join();
    server.scheduler_thread = null;

    // Drain active connections (wait up to 30 seconds). Monotonic clock:
    // a wall-clock step during shutdown must not truncate or extend the grace.
    const drain_timeout_ms: i64 = 30 * std.time.ms_per_s;
    const drain_start = milliTimestamp();
    const active_count = g_server.metrics.active_connections.load(.acquire);
    if (active_count > 0) {
        std.log.info("Draining {d} active connections...", .{active_count});
    }

    var drain_warned = false;
    while (g_server.metrics.active_connections.load(.acquire) > 0) {
        const elapsed = milliTimestamp() - drain_start;
        if (elapsed > drain_timeout_ms and !drain_warned) {
            // Detached handler threads still use g_server (stack Server in run()).
            // Freeing conversations / prefix IDs while they run is a use-after-free.
            // Keep waiting; connection read timeouts and the cancelled scheduler
            // make in-flight handlers exit.
            std.log.warn("Drain timeout after {d}ms, still {d} connection(s); waiting so handlers cannot use freed server state", .{
                elapsed,
                g_server.metrics.active_connections.load(.acquire),
            });
            drain_warned = true;
        }
        sleepNs(drain_poll_interval_ms * std.time.ns_per_ms);
    }

    // Free conversation storage under mutex. Handlers have all exited
    // (active_connections == 0), so they cannot still be reading these.
    {
        server.mutex.lockUncancelable(server.io);
        defer server.mutex.unlock(server.io);
        server.persistConversationsLocked();
        if (server.cached_prompt_ids.len > 0) {
            server.clearCachedPromptIds();
        }
        if (server.cached_user_prefix_ids) |ids| {
            @memset(std.mem.sliceAsBytes(ids), 0);
            allocator.free(ids);
            server.cached_user_prefix_ids = null;
        }
        for (server.conversations.items) |*conv| conv.freeMessages(allocator);
        server.conversations.deinit(allocator);
    }

    std.log.info("Graceful shutdown complete", .{});
}

// ── Tests ───────────────────────────────────────────────────────

test "uptimeSeconds is monotonic elapsed and never negative" {
    try std.testing.expectEqual(@as(i64, 0), uptimeSeconds(1000, 1000));
    try std.testing.expectEqual(@as(i64, 0), uptimeSeconds(1000, 1999));
    try std.testing.expectEqual(@as(i64, 5), uptimeSeconds(1000, 6000));
    // NTP/manual step backward on a wall clock would look like this; monotonic
    // duration must clamp rather than go negative.
    try std.testing.expectEqual(@as(i64, 0), uptimeSeconds(5000, 1000));
}

test "parseContentLength normal" {
    try std.testing.expectEqual(@as(?usize, 42), http.parseContentLength("Content-Length: 42\r\nHost: localhost"));
}

/// Test double satisfying the `writeAll` surface `sseWriteData` needs.
const CollectingStream = struct {
    list: *std.ArrayList(u8),
    allocator: Allocator,

    fn writeAll(self: *CollectingStream, data: []const u8) !void {
        try self.list.appendSlice(self.allocator, data);
    }
};

/// Count SSE data frames in collected output and return their payloads.
fn collectSseFrames(allocator: Allocator, raw: []const u8) !std.ArrayList([]const u8) {
    var frames = std.ArrayList([]const u8).empty;
    errdefer frames.deinit(allocator);
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, raw, pos, "data: ")) |start| {
        const payload_start = start + "data: ".len;
        const end = std.mem.indexOfPos(u8, raw, payload_start, "\n\n") orelse break;
        try frames.append(allocator, raw[payload_start..end]);
        pos = end + 2;
    }
    return frames;
}

test "formatContentDeltaChunk emits content and finish reason" {
    var buf: [512]u8 = undefined;
    const chunk = formatContentDeltaChunk(&buf, "m", 7, 1234, "hello", "\"stop\"").?;
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"content\":\"hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "\"finish_reason\":\"stop\"") != null);
    const pending = formatContentDeltaChunk(&buf, "m", 7, 1234, "x", "null").?;
    try std.testing.expect(std.mem.indexOf(u8, pending, "\"finish_reason\":null") != null);
    // Undersized buffer must be reported, not truncated.
    try std.testing.expect(formatContentDeltaChunk(buf[0..32], "m", 7, 1234, "hello", "\"stop\"") == null);
}

test "writeStreamedContent emits single delta for short output" {
    const allocator = std.testing.allocator;
    var collected = std.ArrayList(u8).empty;
    defer collected.deinit(allocator);
    var fake = CollectingStream{ .list = &collected, .allocator = allocator };
    var chunk_buf: [response_buf_size]u8 = undefined;
    writeStreamedContent(&fake, &chunk_buf, "test-model", 1, 2, "hi", "stop");
    var frames = try collectSseFrames(allocator, collected.items);
    defer frames.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), frames.items.len);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[0], "\"content\":\"hi\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[0], "\"finish_reason\":\"stop\"") != null);
}

test "writeStreamedContent splits oversized output across deltas" {
    const allocator = std.testing.allocator;
    var collected = std.ArrayList(u8).empty;
    defer collected.deinit(allocator);
    var fake = CollectingStream{ .list = &collected, .allocator = allocator };
    var chunk_buf: [response_buf_size]u8 = undefined;
    const total_len = stream_content_delta_max * 2 + 5;
    const content = try allocator.alloc(u8, total_len);
    defer allocator.free(content);
    @memset(content, 'y');
    writeStreamedContent(&fake, &chunk_buf, "test-model", 1, 2, content, "length");
    var frames = try collectSseFrames(allocator, collected.items);
    defer frames.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), frames.items.len);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[0], "\"finish_reason\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[1], "\"finish_reason\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[2], "\"finish_reason\":\"length\"") != null);
    // Reassembled pieces must equal the input exactly, no dropped bytes.
    var reassembled = std.ArrayList(u8).empty;
    defer reassembled.deinit(allocator);
    const fr_key = ",\"finish_reason\"";
    for (frames.items) |f| {
        const key = "\"content\":\"";
        const s = std.mem.indexOf(u8, f, key).? + key.len;
        const fr = std.mem.indexOf(u8, f, fr_key).?;
        const e = std.mem.lastIndexOf(u8, f[0..fr], "\"").?;
        try reassembled.appendSlice(allocator, f[s..e]);
    }
    try std.testing.expectEqualStrings(content, reassembled.items);
}

test "writeStreamedContent emits empty delta with finish reason" {
    const allocator = std.testing.allocator;
    var collected = std.ArrayList(u8).empty;
    defer collected.deinit(allocator);
    var fake = CollectingStream{ .list = &collected, .allocator = allocator };
    var chunk_buf: [response_buf_size]u8 = undefined;
    writeStreamedContent(&fake, &chunk_buf, "test-model", 1, 2, "", "stop");
    var frames = try collectSseFrames(allocator, collected.items);
    defer frames.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), frames.items.len);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[0], "\"content\":\"\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, frames.items[0], "\"finish_reason\":\"stop\"") != null);
}

test "getTimeComponents wraps pre-epoch timestamps" {
    defer sim_clock.setOverrideMs(null);
    // 1s before epoch → 1969-12-31 23:59:59 UTC. Truncating division used to
    // report 00:00:00.
    sim_clock.setOverrideMs(-1000);
    const before = getTimeComponents();
    try std.testing.expectEqual(@as(u64, 23), before.hours);
    try std.testing.expectEqual(@as(u64, 59), before.minutes);
    try std.testing.expectEqual(@as(u64, 59), before.seconds);

    sim_clock.setOverrideMs(0);
    const epoch = getTimeComponents();
    try std.testing.expectEqual(@as(u64, 0), epoch.hours);
    try std.testing.expectEqual(@as(u64, 0), epoch.minutes);
    try std.testing.expectEqual(@as(u64, 0), epoch.seconds);

    sim_clock.setOverrideMs(3_661_000); // 1h 1m 1s
    const later = getTimeComponents();
    try std.testing.expectEqual(@as(u64, 1), later.hours);
    try std.testing.expectEqual(@as(u64, 1), later.minutes);
    try std.testing.expectEqual(@as(u64, 1), later.seconds);
}

test "log timestamps render with an explicit UTC marker" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(3_661_000); // 1h 1m 1s
    const t = getTimeComponents();
    var buf: [32]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, log_time_fmt ++ " req={d}\n", .{ t.hours, t.minutes, t.seconds, @as(u64, 7) });
    try std.testing.expectEqualStrings("[01:01:01Z] req=7\n", line);
}

test "prngSeedFromSampling uses sim_clock when seed omitted" {
    defer sim_clock.setOverrideMs(null);
    sim_clock.setOverrideMs(1_700_000_000_000);
    const sampling = SamplingParams{};
    const seed = prngSeedFromSampling(sampling);
    const expected: u64 = @truncate(@as(u96, @bitCast(@as(i96, 1_700_000_000_000) * 1_000_000)));
    try std.testing.expectEqual(expected, seed);
    try std.testing.expectEqual(@as(u64, 99), schedulerPrngSeed(0, .{ .seed = 99 }));
    // Under an override the seed is the virtual clock alone, so two requests
    // admitted at the same virtual millisecond get the same replayable stream
    // regardless of connection arrival order.
    try std.testing.expectEqual(seed, schedulerPrngSeed(7, sampling));
    try std.testing.expectEqual(seed, schedulerPrngSeed(1024, sampling));

    // With the wall clock restored the request id decorrelates concurrent
    // requests, whose timestamps can collide. The wall clock keeps running
    // between the two reads, so only the decorrelation is checked, not the
    // exact value.
    sim_clock.setOverrideMs(null);
    const wall = SamplingParams{};
    try std.testing.expect(schedulerPrngSeed(7, wall) != schedulerPrngSeed(8, wall));
}

test "shouldNoteAutoSeed only when sampling and seed omitted" {
    try std.testing.expect(!shouldNoteAutoSeed(.{}));
    try std.testing.expect(shouldNoteAutoSeed(.{ .temperature = 0.7 }));
    try std.testing.expect(!shouldNoteAutoSeed(.{ .temperature = 0.7, .seed = 42 }));
    try std.testing.expect(!shouldNoteAutoSeed(.{ .temperature = 0, .seed = 42 }));
}

test "originMatchesHost accepts same-origin http" {
    try std.testing.expect(http.originMatchesHost("http://127.0.0.1:49453", "127.0.0.1:49453"));
    try std.testing.expect(http.originMatchesHost("https://Example.COM", "example.com"));
}

test "originMatchesHost rejects path userinfo and cross-origin" {
    try std.testing.expect(!http.originMatchesHost("http://127.0.0.1:49453/", "127.0.0.1:49453"));
    try std.testing.expect(!http.originMatchesHost("http://evil.com", "127.0.0.1:49453"));
    try std.testing.expect(!http.originMatchesHost("null", "127.0.0.1:49453"));
    try std.testing.expect(!http.originMatchesHost("http://user@127.0.0.1:49453", "127.0.0.1:49453"));
}

test "isLoopbackHttpHost accepts loopback Host values" {
    try std.testing.expect(http.isLoopbackHttpHost("127.0.0.1"));
    try std.testing.expect(http.isLoopbackHttpHost("127.0.0.1:49453"));
    try std.testing.expect(http.isLoopbackHttpHost("127.1.2.3:80"));
    try std.testing.expect(http.isLoopbackHttpHost("localhost"));
    try std.testing.expect(http.isLoopbackHttpHost("LocalHost:49453"));
    try std.testing.expect(http.isLoopbackHttpHost("localhost."));
    try std.testing.expect(http.isLoopbackHttpHost("[::1]"));
    try std.testing.expect(http.isLoopbackHttpHost("[::1]:49453"));
}

test "isLoopbackHttpHost rejects DNS-rebind and LAN Host values" {
    try std.testing.expect(!http.isLoopbackHttpHost("evil.com"));
    try std.testing.expect(!http.isLoopbackHttpHost("evil.com:49453"));
    try std.testing.expect(!http.isLoopbackHttpHost("127.0.0.1.nip.io"));
    try std.testing.expect(!http.isLoopbackHttpHost("localhost.evil.com"));
    try std.testing.expect(!http.isLoopbackHttpHost("0.0.0.0"));
    try std.testing.expect(!http.isLoopbackHttpHost("192.168.1.1:49453"));
    try std.testing.expect(!http.isLoopbackHttpHost(""));
    try std.testing.expect(!http.isLoopbackHttpHost("127.1"));
    try std.testing.expect(!http.isLoopbackHttpHost("127.0.0.1.2"));
}

test "getHeaderValue trims and rejects duplicates" {
    try std.testing.expectEqualStrings("127.0.0.1:49453", http.getHeaderValue("Host: 127.0.0.1:49453\r\n", "host").?);
    try std.testing.expect(http.getHeaderValue("Host: a\r\nHost: b\r\n", "host") == null);
    try std.testing.expect(http.getHeaderValue("Accept: */*\r\n", "origin") == null);
}

test "acceptsEncoding honors gzip q-values" {
    try std.testing.expect(http.acceptsEncoding("Accept-Encoding: gzip, deflate, br\r\n", "gzip"));
    try std.testing.expect(http.acceptsEncoding("accept-encoding: GZIP\r\n", "gzip"));
    try std.testing.expect(!http.acceptsEncoding("Accept-Encoding: gzip;q=0\r\n", "gzip"));
    try std.testing.expect(!http.acceptsEncoding("Accept-Encoding: deflate, br\r\n", "gzip"));
    try std.testing.expect(!http.acceptsEncoding("Host: localhost\r\n", "gzip"));
    try std.testing.expect(http.acceptsEncoding("Accept-Encoding: gzip;q=1.0, identity;q=0.5\r\n", "gzip"));
}

test "ifNoneMatch matches listed ETags" {
    try std.testing.expect(http.ifNoneMatch("If-None-Match: \"abc\"\r\n", "\"abc\""));
    try std.testing.expect(http.ifNoneMatch("If-None-Match: W/\"abc\"\r\n", "\"abc\""));
    try std.testing.expect(http.ifNoneMatch("If-None-Match: \"x\", \"abc\"\r\n", "\"abc\""));
    try std.testing.expect(http.ifNoneMatch("If-None-Match: *\r\n", "\"abc\""));
    try std.testing.expect(!http.ifNoneMatch("If-None-Match: \"xyz\"\r\n", "\"abc\""));
    try std.testing.expect(!http.ifNoneMatch("Host: localhost\r\n", "\"abc\""));
}

test "304 revalidates under the same cache key and claims no body" {
    // A 304 that drops Vary lets a shared cache answer a client that never
    // accepted gzip with the gzip entry, and one that carries
    // `Content-Length: 0` contradicts the length the 200 reported (RFC 9110
    // 15.4.5), which is how a revalidation turns into a truncated document.
    try std.testing.expect(std.mem.indexOf(u8, not_modified_headers, "Vary: Accept-Encoding\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, not_modified_headers, "Cache-Control: private, no-cache\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, not_modified_headers, "Content-Length") == null);
    // The same ETag names both the 200 and its 304, so a cache that stored
    // either one revalidates against the other.
    try std.testing.expect(std.mem.indexOf(u8, not_modified_headers, "ETag") == null);
}

test "gzipAlloc shrinks the chat UI and round-trips" {
    const gz = try http.gzipAlloc(std.testing.allocator, html_page);
    defer std.testing.allocator.free(gz);
    try std.testing.expect(gz.len < html_page.len);
    try std.testing.expectEqual(http.gzip_id1, gz[0]);
    try std.testing.expectEqual(http.gzip_id2, gz[1]);

    var reader: std.Io.Reader = .fixed(gz);
    var window_buf: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress = std.compress.flate.Decompress.init(&reader, .gzip, &window_buf);
    const out = try std.testing.allocator.alloc(u8, html_page.len + 1);
    defer std.testing.allocator.free(out);
    var writer: std.Io.Writer = .fixed(out);
    _ = try decompress.reader.streamRemaining(&writer);
    try std.testing.expectEqualSlices(u8, html_page, out[0..writer.end]);
}

test "chat UI head pulls no third-party asset" {
    // Every jsDelivr script is fetched on demand by the bundle (chat/markdown.ts
    // loadMarkdown / loadHighlightJs), so nothing third-party sits on the first
    // paint. The pinned URL is the marker that the deferred loader is shipped at
    // all: it may appear in the page, never in the head.
    const head_end = std.mem.indexOf(u8, html_page, "</head>").?;
    const head = html_page[0..head_end];
    try std.testing.expect(std.mem.indexOf(u8, head, "cdn.jsdelivr.net") == null);
    try std.testing.expect(std.mem.indexOf(u8, head, "rel=\"preconnect\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, head, "<script src=") == null);
    try std.testing.expect(std.mem.indexOf(u8, html_page, "https://cdn.jsdelivr.net/npm/marked@") != null);
}

test "chat UI streams by appending and times out a stalled CDN script" {
    // The committed src/web/app.js is a minified bundle (whitespace and syntax;
    // identifiers are kept so the artifact stays byte-reproducible for
    // scripts/check-web-artifacts.sh, but they are mangled past recognition).
    // Assert on the sources it is built from, which is where these two
    // mechanisms live.
    const message_source = @embedFile("../web/chat/components/message.tsx");
    const markdown_source = @embedFile("../web/chat/markdown.ts");
    // Each stream flush used to reset textContent, rewriting the whole message
    // every 60ms; the DOM keeps one text node and appends the delta instead.
    try std.testing.expect(std.mem.indexOf(u8, message_source, "node.appendData(") != null);
    try std.testing.expect(std.mem.indexOf(u8, message_source, "next.startsWith(painted)") != null);
    // A CDN that stalls fires no load or error event, so the load promise has
    // to give up on its own for the plain-text fallback to settle.
    try std.testing.expect(std.mem.indexOf(u8, markdown_source, "CDN_SCRIPT_TIMEOUT_MS") != null);
    try std.testing.expect(std.mem.indexOf(u8, markdown_source, "settle('timeout')") != null);
}

test "sanitizeClientRequestId accepts correlation tokens" {
    var buf: [max_client_request_id_len]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 3), http.sanitizeClientRequestId("abc", &buf));
    try std.testing.expectEqualStrings("abc", buf[0..3]);
    try std.testing.expectEqual(@as(usize, 36), http.sanitizeClientRequestId("550e8400-e29b-41d4-a716-446655440000", &buf));
    try std.testing.expectEqual(@as(usize, 0), http.sanitizeClientRequestId("", &buf));
    try std.testing.expectEqual(@as(usize, 0), http.sanitizeClientRequestId("has space", &buf));
    try std.testing.expectEqual(@as(usize, 0), http.sanitizeClientRequestId("bad\nid", &buf));
    try std.testing.expectEqual(@as(usize, 0), http.sanitizeClientRequestId("a" ** 65, &buf));
}

test "HealthView maps degradation reasons" {
    var hv = HealthView{
        .shutting_down = false,
        .kv_pressure = false,
        .high_error_rate = false,
        .kv_used = 0,
        .kv_total = 100,
        .queue = 0,
        .completed = 0,
        .failed = 0,
        .cancelled = 0,
        .sched_errs = 0,
        .kv_demotions = 0,
        .sleeping = false,
    };
    try std.testing.expect(hv.ready());
    try std.testing.expectEqualStrings("ok", hv.statusStr());
    try std.testing.expectEqualStrings("none", hv.reasonStr());
    hv.kv_pressure = true;
    try std.testing.expect(!hv.ready());
    try std.testing.expectEqual(HealthState.kv_pressure, hv.state());
    try std.testing.expectEqualStrings("kv_pressure", hv.reasonStr());
    hv.high_error_rate = true;
    try std.testing.expectEqualStrings("kv_pressure,high_error_rate", hv.reasonStr());
    hv.shutting_down = true;
    try std.testing.expectEqualStrings("shutting_down", hv.statusStr());
    try std.testing.expectEqualStrings("shutting_down", hv.reasonStr());
}

test "isSafeErrorToken" {
    try std.testing.expect(isSafeErrorToken("n_not_supported"));
    try std.testing.expect(!isSafeErrorToken("a\"b"));
    try std.testing.expect(!isSafeErrorToken(""));
}

test "computeLogprobs without top_logprobs still reports the sampled token" {
    const logits = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    const info = computeLogprobs(&logits, 2, 0);
    try std.testing.expectEqual(@as(u32, 0), info.count);
    const expected = math_ops.tokenLogProb(&logits, 2);
    try std.testing.expectApproxEqAbs(expected, info.token_logprob, 1e-6);
    try std.testing.expect(info.token_logprob < 0);
}

test "nextAllowedToolCall skips undeclared names" {
    var tp = json.ToolParams{};
    tp.tools[0] = .{ .name = "get_weather", .description = "", .parameters_json = "{}" };
    tp.tool_count = 1;
    var reg = tools_mod.Registry{};
    const text =
        \\<tool_call>{"name": "rm", "arguments": {"path": "/"}}</tool_call>
        \\<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
        \\<tool_call>{"name": "shell", "arguments": {}}</tool_call>
    ;
    var pos: usize = 0;
    const first = nextAllowedToolCall(text, &pos, &tp, &reg) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, first, "get_weather") != null);
    try std.testing.expect(std.mem.indexOf(u8, first, "rm") == null);
    try std.testing.expect(nextAllowedToolCall(text, &pos, &tp, &reg) == null);
}

test "stripToolCallTags removes tag markup and keeps surrounding text" {
    const alloc = std.testing.allocator;
    const text =
        \\before<tool_call>{"name": "rm"}</tool_call>after
    ;
    const out = try stripToolCallTags(alloc, text);
    defer if (out.ptr != text.ptr) alloc.free(@constCast(out));
    try std.testing.expectEqualStrings("beforeafter", out);
    // No tags: the input is borrowed, not copied.
    const plain = "plain";
    try std.testing.expectEqual(@intFromPtr(plain.ptr), @intFromPtr((try stripToolCallTags(alloc, plain)).ptr));
    // An unterminated tag drops the rest rather than looping.
    const unterminated = "keep<tool_call>{\"name\"";
    const open = try stripToolCallTags(alloc, unterminated);
    defer if (open.ptr != unterminated.ptr) alloc.free(@constCast(open));
    try std.testing.expectEqualStrings("keep", open);
}

test "buildResponsesToolCallResponse emits function_call items for declared tools" {
    var tp = json.ToolParams{};
    tp.tools[0] = .{ .name = "get_weather", .description = "", .parameters_json = "{}" };
    tp.tool_count = 1;
    const text =
        \\<tool_call>{"name": "rm", "arguments": {"path": "/"}}</tool_call>
        \\<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
    ;
    var buf: [response_buf_size]u8 = undefined;
    const out = buildResponsesToolCallResponse(&buf, text, 7, 100, 3, 4, &tp);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"type\":\"function_call\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"name\":\"get_weather\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"") != null);
    // The undeclared name never reaches the response.
    try std.testing.expect(std.mem.indexOf(u8, out, "rm") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"total_tokens\":7") != null);
}

test "buildResponsesToolCallResponse returns empty when no declared tool is called" {
    var tp = json.ToolParams{};
    const text = "<tool_call>{\"name\": \"rm\", \"arguments\": {}}</tool_call>";
    var buf: [response_buf_size]u8 = undefined;
    const out = buildResponsesToolCallResponse(&buf, text, 1, 1, 1, 1, &tp);
    try std.testing.expectEqualStrings("", out);
}

test "nextAllowedToolCall accepts registry names" {
    var tp = json.ToolParams{};
    var reg = tools_mod.Registry{};
    _ = try reg.register(.{ .name = "search", .description = "", .parameters_json = "{}" });
    const text =
        \\<tool_call>{"name": "search", "arguments": {}}</tool_call>
    ;
    var pos: usize = 0;
    const hit = nextAllowedToolCall(text, &pos, &tp, &reg) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, hit, "search") != null);
}

test "nextAllowedToolCall rejects payload without name" {
    var tp = json.ToolParams{};
    tp.tools[0] = .{ .name = "a", .description = "", .parameters_json = "{}" };
    tp.tool_count = 1;
    var reg = tools_mod.Registry{};
    const text =
        \\<tool_call>{"arguments": {}}</tool_call>
    ;
    var pos: usize = 0;
    try std.testing.expect(nextAllowedToolCall(text, &pos, &tp, &reg) == null);
}

test "resolveAnthropicToolInput keeps a complete object and drops a truncated one" {
    const allocator = std.testing.allocator;
    const good = resolveAnthropicToolInput(allocator, "{\"arguments\": {\"city\": \"Paris\"}}", 0);
    defer if (good.owned) |p| allocator.free(p);
    try std.testing.expectEqualStrings("{\"city\": \"Paris\"}", good.obj);

    // The model ran out of tokens mid-object: the payload's own closing
    // braces do not close the arguments value, and emitting the remainder
    // verbatim would hand the client a half-written tool_use block.
    const cut = resolveAnthropicToolInput(allocator, "{\"arguments\": {\"city\": {\"name\": \"Pa", 0);
    defer if (cut.owned) |p| allocator.free(p);
    try std.testing.expectEqualStrings("{}", cut.obj);

    // String arguments are unwrapped, and the unwrapped text is held to the
    // same rule.
    const uncut = resolveAnthropicToolInput(allocator, "{\"arguments\": \"{\\\"city\\\": \\\"Paris\\\"}\"}", 0);
    defer if (uncut.owned) |p| allocator.free(p);
    try std.testing.expectEqualStrings("{\"city\": \"Paris\"}", uncut.obj);

    const cut_str = resolveAnthropicToolInput(allocator, "{\"arguments\": \"{\\\"city\\\": \\\"Pa", 0);
    defer if (cut_str.owned) |p| allocator.free(p);
    try std.testing.expectEqualStrings("{}", cut_str.obj);

    // A complete object whose braces sit inside a string value is complete.
    const braced = resolveAnthropicToolInput(allocator, "{\"arguments\": {\"city\": \"{Paris}\"}}", 0);
    defer if (braced.owned) |p| allocator.free(p);
    try std.testing.expectEqualStrings("{\"city\": \"{Paris}\"}", braced.obj);
}

test "anthropicStatusLine maps known codes" {
    try std.testing.expectEqualStrings("400 Bad Request", anthropicStatusLine("400"));
    try std.testing.expectEqualStrings("401 Unauthorized", anthropicStatusLine("401"));
    try std.testing.expectEqualStrings("403 Forbidden", anthropicStatusLine("403"));
    try std.testing.expectEqualStrings("404 Not Found", anthropicStatusLine("404"));
    try std.testing.expectEqualStrings("405 Method Not Allowed", anthropicStatusLine("405"));
    try std.testing.expectEqualStrings("429 Too Many Requests", anthropicStatusLine("429"));
    try std.testing.expectEqualStrings("503 Service Unavailable", anthropicStatusLine("503"));
    try std.testing.expectEqualStrings("500 Internal Server Error", anthropicStatusLine("500"));
    try std.testing.expectEqualStrings("500 Internal Server Error", anthropicStatusLine("999"));
}

test "parseContentLength duplicate rejects" {
    try std.testing.expectEqual(@as(?usize, null), http.parseContentLength("Content-Length: 42\r\nContent-Length: 42"));
}

test "parseContentLength missing header returns zero" {
    try std.testing.expectEqual(@as(?usize, 0), http.parseContentLength("Host: localhost\r\nAccept: */*"));
}

test "parseContentLength non-numeric rejects" {
    try std.testing.expectEqual(@as(?usize, null), http.parseContentLength("Content-Length: abc\r\nHost: localhost"));
}

test "parseContentLength empty headers returns zero" {
    try std.testing.expectEqual(@as(?usize, 0), http.parseContentLength(""));
}

test "parseContentLength case insensitive" {
    try std.testing.expectEqual(@as(?usize, 99), http.parseContentLength("content-length: 99\r\nHost: x"));
    try std.testing.expectEqual(@as(?usize, 7), http.parseContentLength("CONTENT-LENGTH: 7\r\nHost: x"));
}

test "splitPathQuery strips query from path" {
    const no_q = http.splitPathQuery("/v1/kv_cache");
    try std.testing.expectEqualStrings("/v1/kv_cache", no_q.path);
    try std.testing.expectEqualStrings("", no_q.query);

    const with_q = http.splitPathQuery("/v1/kv_cache?n_tokens=512&foo=1");
    try std.testing.expectEqualStrings("/v1/kv_cache", with_q.path);
    try std.testing.expectEqualStrings("n_tokens=512&foo=1", with_q.query);

    const empty_q = http.splitPathQuery("/v1/models?");
    try std.testing.expectEqualStrings("/v1/models", empty_q.path);
    try std.testing.expectEqualStrings("", empty_q.query);
}

test "extractQueryParam reads values" {
    try std.testing.expectEqualStrings("512", http.extractQueryParam("n_tokens=512", "n_tokens").?);
    try std.testing.expectEqualStrings("512", http.extractQueryParam("foo=1&n_tokens=512&bar=2", "n_tokens").?);
    try std.testing.expect(http.extractQueryParam("foo=1&bar=2", "n_tokens") == null);
    try std.testing.expectEqualStrings("", http.extractQueryParam("n_tokens=&x=1", "n_tokens").?);
    try std.testing.expectEqualStrings("", http.extractQueryParam("n_tokens", "n_tokens").?);
    try std.testing.expect(http.extractQueryParam("n_tokens_extra=9", "n_tokens") == null);
}

test "parseRequestLine extracts method path query" {
    const a = http.parseRequestLine("GET /health HTTP/1.1").?;
    try std.testing.expectEqualStrings("GET", a.method);
    try std.testing.expectEqualStrings("/health", a.path);
    try std.testing.expectEqualStrings("", a.query);

    const b = http.parseRequestLine("POST /v1/kv_cache?n_tokens=64 HTTP/1.1").?;
    try std.testing.expectEqualStrings("POST", b.method);
    try std.testing.expectEqualStrings("/v1/kv_cache", b.path);
    try std.testing.expectEqualStrings("n_tokens=64", b.query);

    try std.testing.expect(http.parseRequestLine("GET /health") == null);
    try std.testing.expect(http.parseRequestLine("") == null);
}

test "parseDetokenizeTokens reads token id array" {
    var out: [8]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 3), parseDetokenizeTokens("{\"tokens\":[1, 2, 3]}", &out).count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
    try std.testing.expectEqual(@as(u32, 3), out[2]);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{}", &out)) == .empty);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[]}", &out)) == .empty);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":\"abc\"}", &out)) == .empty);
    try std.testing.expectEqual(@as(usize, 3), parseDetokenizeTokens("{\"tokens\":[10,20,30]}", out[0..3]).count);
    // More entries than `out` can hold is an error, not a truncated decode.
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[10,20,30]}", out[0..2])) == .malformed);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[1,2,3,4,5,6,7,8,9]}", &out)) == .malformed);
}

test "parseDetokenizeTokens rejects a non-numeric element instead of truncating" {
    var out: [8]u32 = undefined;
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[1,\"x\",3]}", &out)) == .malformed);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[1,-2,3]}", &out)) == .malformed);
    try std.testing.expect(std.meta.activeTag(parseDetokenizeTokens("{\"tokens\":[1,null]}", &out)) == .malformed);
}

test "parsePositiveQueryParam distinguishes missing and invalid" {
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("foo=1", "n_tokens")) == .missing);
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("n_tokens=0", "n_tokens")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("n_tokens=", "n_tokens")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("n_tokens", "n_tokens")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("n_tokens=abc", "n_tokens")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parsePositiveQueryParam("n_tokens=-1", "n_tokens")) == .invalid);
    const a = parsePositiveQueryParam("n_tokens=512", "n_tokens");
    try std.testing.expectEqual(@as(usize, 512), a.value);
    const b = parsePositiveQueryParam("foo=1&n_tokens=64&bar=2", "n_tokens");
    try std.testing.expectEqual(@as(usize, 64), b.value);
}

test "parseFormConversationId rejects missing zero and non-numeric" {
    try std.testing.expect(std.meta.activeTag(parseFormConversationId("action=select")) == .missing);
    try std.testing.expect(std.meta.activeTag(parseFormConversationId("action=select&id=")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parseFormConversationId("action=select&id=0")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parseFormConversationId("action=select&id=abc")) == .invalid);
    try std.testing.expect(std.meta.activeTag(parseFormConversationId("action=select&id=-1")) == .invalid);
    const a = parseFormConversationId("action=select&id=1");
    try std.testing.expectEqual(@as(u32, 1), a.value);
    const b = parseFormConversationId("id=42");
    try std.testing.expectEqual(@as(u32, 42), b.value);
}

test "extractMaxGenTokens prefers first present alias" {
    try std.testing.expectEqual(@as(usize, 10), extractMaxGenTokens(
        "{\"max_output_tokens\":10,\"max_tokens\":20}",
        &.{ "max_output_tokens", "max_tokens", "max_completion_tokens" },
    ));
    try std.testing.expectEqual(@as(usize, 20), extractMaxGenTokens(
        "{\"max_tokens\":20}",
        &.{ "max_output_tokens", "max_tokens", "max_completion_tokens" },
    ));
    try std.testing.expectEqual(@as(usize, 30), extractMaxGenTokens(
        "{\"max_completion_tokens\":30}",
        &.{ "max_tokens", "max_completion_tokens" },
    ));
    try std.testing.expectEqual(default_max_gen_tokens, extractMaxGenTokens("{}", &.{"max_tokens"}));
    try std.testing.expectEqual(@as(usize, 1), extractMaxGenTokens("{\"max_tokens\":0}", &.{"max_tokens"}));
    try std.testing.expectEqual(max_gen_tokens_cap, extractMaxGenTokens("{\"max_tokens\":99999}", &.{"max_tokens"}));
}

test "known_endpoints include kv_cache routes" {
    var found_cache = false;
    var found_info = false;
    var found_root = false;
    for (known_endpoints) |ep| {
        if (std.mem.eql(u8, ep.path, "/v1/kv_cache")) found_cache = true;
        if (std.mem.eql(u8, ep.path, "/v1/kv_cache/info")) found_info = true;
        if (std.mem.eql(u8, ep.path, "/")) found_root = true;
    }
    try std.testing.expect(found_cache);
    try std.testing.expect(found_info);
    try std.testing.expect(found_root);
}

test "isAnthropicPath selects the Anthropic error envelope" {
    try std.testing.expect(isAnthropicPath("/v1/messages"));
    try std.testing.expect(!isAnthropicPath("/v1/chat/completions"));
    try std.testing.expect(!isAnthropicPath("/v1/completions"));
    try std.testing.expect(!isAnthropicPath(""));
    // Pre-routing failures pass the request-target path, query already split off.
    try std.testing.expect(!isAnthropicPath("/v1/kv_cache?n_tokens=1"));
    // 413 must keep its status line in the Anthropic envelope, not fall back to 500.
    try std.testing.expectEqualStrings("413 Payload Too Large", anthropicStatusLine("413"));
}

test "incompleteUtf8TailLen holds only valid partial sequences" {
    // ASCII always ends on a boundary.
    try std.testing.expectEqual(@as(usize, 0), incompleteUtf8TailLen("hello"));
    // Complete 3-byte CJK char at the tail.
    try std.testing.expectEqual(@as(usize, 0), incompleteUtf8TailLen("\xe4\xb8\x96"));
    // Lead + one continuation of a 3-byte sequence: hold both.
    try std.testing.expectEqual(@as(usize, 2), incompleteUtf8TailLen("a\xe4\xb8"));
    // Bare lead byte of a 2-byte sequence: hold it.
    try std.testing.expectEqual(@as(usize, 1), incompleteUtf8TailLen("ab\xc3"));
    // Lone continuation byte with no lead in lookback range passes through.
    try std.testing.expectEqual(@as(usize, 0), incompleteUtf8TailLen("a\x80"));
    // Invalid lead byte (0xFF) passes through instead of stalling.
    try std.testing.expectEqual(@as(usize, 0), incompleteUtf8TailLen("a\xff"));
    // Empty input.
    try std.testing.expectEqual(@as(usize, 0), incompleteUtf8TailLen(""));
}

test "Utf8Holdback reassembles a character split across tokens" {
    const allocator = std.testing.allocator;
    var hb = Utf8Holdback{};
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    // "世" = E4 B8 96 split across three byte-fallback tokens.
    const p1 = hb.feed(&[_]u8{0xE4});
    try std.testing.expectEqualStrings("", p1.head);
    try std.testing.expectEqualStrings("", p1.body);
    const p2 = hb.feed(&[_]u8{0xB8});
    try std.testing.expectEqualStrings("", p2.head);
    try std.testing.expectEqualStrings("", p2.body);
    const p3 = hb.feed(&[_]u8{0x96});
    // Reassembled char arrives via `head` (built from held bytes), body empty.
    try std.testing.expectEqualStrings("\xe4\xb8\x96", p3.head);
    try std.testing.expectEqualStrings("", p3.body);

    // ASCII after a completed sequence flows straight through.
    const p4 = hb.feed("hi");
    try std.testing.expectEqualStrings("", p4.head);
    try std.testing.expectEqualStrings("hi", p4.body);

    // Held bytes at the end are released by flush.
    _ = hb.feed(&[_]u8{0xF0}); // lead of a 4-byte emoji
    const held = hb.flush();
    try std.testing.expectEqualStrings("\xf0", held);
    try std.testing.expectEqual(@as(usize, 0), hb.pending_len);
}

test "Utf8Holdback completes held bytes plus head of next token" {
    var hb = Utf8Holdback{};
    // First token carries the first two bytes of a 3-byte char.
    const p1 = hb.feed(&[_]u8{ 0xE7, 0x95 });
    try std.testing.expectEqualStrings("", p1.body);
    // Second token starts with the final byte then continues with ASCII.
    const p2 = hb.feed(&[_]u8{ 0x9C, 'x' });
    try std.testing.expectEqualStrings("\xe7\x95\x9c", p2.head); // 畜
    try std.testing.expectEqualStrings("x", p2.body);
}

test "Utf8Holdback releases invalid continuation raw like batch decode" {
    var hb = Utf8Holdback{};
    // Valid prefix E4 B8, then the model emits ASCII instead of continuing.
    _ = hb.feed(&[_]u8{ 0xE4, 0xB8 });
    const pieces = hb.feed("z");
    // Held bytes pass through raw rather than being dropped.
    try std.testing.expectEqualStrings("\xe4\xb8", pieces.head);
    try std.testing.expectEqualStrings("z", pieces.body);
}

test "prefixCacheApplies excludes prompts that carried an image" {
    // Image placeholders are the same token IDs for every image, so a
    // prompt with visual embeddings must never read or publish the KV
    // prefix memo; two different images would otherwise share KV.
    try std.testing.expect(prefixCacheApplies(0));
    try std.testing.expect(!prefixCacheApplies(1));
    try std.testing.expect(!prefixCacheApplies(256));
}

test "Conversation.freeMessages releases tool_call_id" {
    const allocator = std.testing.allocator;
    var conv = Conversation{ .id = 1 };
    const content = try allocator.dupe(u8, "ok");
    const tcid = try allocator.dupe(u8, "call_1");
    try conv.messages.append(allocator, .{
        .role = .tool,
        .content = content,
        .tool_call_id = tcid,
    });
    conv.freeMessages(allocator);
}

test "Conversation.clearMessages releases tool_call_id" {
    const allocator = std.testing.allocator;
    var conv = Conversation{ .id = 1 };
    defer conv.messages.deinit(allocator);
    const content = try allocator.dupe(u8, "ok");
    const tcid = try allocator.dupe(u8, "call_1");
    try conv.messages.append(allocator, .{
        .role = .tool,
        .content = content,
        .tool_call_id = tcid,
    });
    conv.clearMessages(allocator);
    try std.testing.expectEqual(@as(usize, 0), conv.messages.items.len);
}

test "Conversation.setTitle keeps trailing multi-byte characters" {
    var conv = Conversation{ .id = 1 };
    conv.setTitle("caf\xc3\xa9"); // café, trailing é is fully present
    try std.testing.expectEqualStrings("caf\xc3\xa9", conv.titleSlice());

    // Truncation that lands inside a multi-byte sequence drops the fragment.
    var conv2 = Conversation{ .id = 2 };
    const long = "\xe4\xb8\x96" ** 20; // 60 bytes of complete chars
    conv2.setTitle(long[0..47]); // cut splits the 16th char
    // Result must be valid UTF-8: ends on a character boundary.
    const t = conv2.titleSlice();
    try std.testing.expect(std.unicode.utf8ValidateSlice(t));
    try std.testing.expect(t.len < 47);
}

test "prefix memo: a vision request publishes nothing for the next text request" {
    // The memo is keyed on token IDs alone and image placeholders are the same
    // IDs for every image, so a published vision prompt would let the next text
    // request match its prefix and read back the earlier request's picture.
    var server = Server{
        .model = undefined,
        .tokenizer = undefined,
        .chat_template = ChatTemplate.chatml,
        .model_name = "test",
        .backend_name = "cpu",
        .allocator = std.testing.allocator,
        .bos_token_id = 0,
        .eog_ids = @splat(0),
        .eog_len = 0,
        .io = undefined,
        .idem = Idempotency.Ledger.init(std.testing.allocator),
    };
    defer server.idem.deinit();
    defer server.clearCachedPromptIds();

    const prompt = [_]u32{ 1, 2, 3, 4 };
    // A text request publishes, so the memo is live.
    server.publishCachedPromptIds(&prompt, 0);
    try std.testing.expectEqualSlices(u32, &prompt, server.cached_prompt_ids);

    // A vision request must leave no memo behind, not the placeholder IDs.
    server.publishCachedPromptIds(&prompt, 2);
    try std.testing.expectEqual(@as(usize, 0), server.cached_prompt_ids.len);
}

test "fuzz: all server functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── pub struct ServerConfig: verify all fields exist at comptime ──
            comptime {
                _ = @as(?*const ServerConfig, null);
                // Verify fields exist
                _ = @offsetOf(ServerConfig, "allocator");
                _ = @offsetOf(ServerConfig, "model");
                _ = @offsetOf(ServerConfig, "tokenizer");
                _ = @offsetOf(ServerConfig, "port");
                _ = @offsetOf(ServerConfig, "bos_token_id");
                _ = @offsetOf(ServerConfig, "eog_ids");
                _ = @offsetOf(ServerConfig, "eog_len");
                _ = @offsetOf(ServerConfig, "api_key");
                _ = @offsetOf(ServerConfig, "host");
                _ = @offsetOf(ServerConfig, "ctx_size");
                _ = @offsetOf(ServerConfig, "vision_encoder");
                _ = @offsetOf(ServerConfig, "draft_model");
                _ = @offsetOf(ServerConfig, "spec_tokens");
                _ = @offsetOf(ServerConfig, "tree_budget");
                _ = @offsetOf(ServerConfig, "rate_limit_rpm");
                _ = @offsetOf(ServerConfig, "rate_limit_tpm");
            }

            // ── pub fn run: comptime verify it exists (needs full server, cannot call) ──
            comptime {
                _ = &run;
            }

            // ── pub TcpStream methods: comptime verify (need real socket FDs) ──
            comptime {
                _ = &http.TcpStream.writeAll;
                _ = &http.TcpStream.read;
                _ = &http.TcpStream.close;
            }

            // ── pub GeneratedEscaped.deinit: comptime verify (needs g_server) ──
            comptime {
                _ = &GeneratedEscaped.deinit;
            }

            // ── Private pure helpers: fuzz with random inputs ──

            // clampMaxTokens
            {
                const raw_val = smith.valueWithHash(u16, 0x01);
                const result = clampMaxTokens(@as(?usize, @intCast(raw_val)));
                std.debug.assert(result >= 1);
                std.debug.assert(result <= max_gen_tokens_cap);
                // null case
                const null_result = clampMaxTokens(null);
                std.debug.assert(null_result == default_max_gen_tokens);
            }

            // extractMaxGenTokens: first present alias wins, missing → default
            {
                const a = extractMaxGenTokens("{\"max_output_tokens\":10,\"max_tokens\":20}", &.{ "max_output_tokens", "max_tokens" });
                std.debug.assert(a == 10);
                const b = extractMaxGenTokens("{\"max_tokens\":20}", &.{ "max_output_tokens", "max_tokens" });
                std.debug.assert(b == 20);
                const c = extractMaxGenTokens("{}", &.{"max_tokens"});
                std.debug.assert(c == default_max_gen_tokens);
            }

            // tokensPerSec
            {
                const count = smith.valueWithHash(u16, 0x02);
                const time_ms = smith.valueWithHash(u16, 0x03);
                const tps = tokensPerSec(@intCast(count), @intCast(time_ms));
                try std.testing.expect(tps >= 0);
                try std.testing.expect(!std.math.isNan(tps));
            }

            // estimatePromptTokens
            {
                const tok_count = smith.valueWithHash(u16, 0x04);
                const text_len = smith.valueWithHash(u16, 0x05);
                const result = estimatePromptTokens(@intCast(tok_count), @intCast(text_len));
                if (tok_count > 0) {
                    std.debug.assert(result == tok_count);
                } else {
                    std.debug.assert(result >= 1);
                }
            }

            // isUnsafeJsonChar
            {
                const c = smith.valueWithHash(u8, 0x06);
                const is_unsafe = http.isUnsafeJsonChar(c);
                // Control chars < 0x20 must be unsafe
                if (c < 0x20) std.debug.assert(is_unsafe);
                // Normal alphanum must be safe
                if (c >= 'A' and c <= 'Z') std.debug.assert(!is_unsafe);
            }

            // sanitizeForLog
            {
                var input_buf: [32]u8 = undefined;
                const len = @min(smith.valueWithHash(u5, 0x07), 31) + 1;
                for (0..len) |i| {
                    input_buf[i] = smith.valueWithHash(u8, @truncate(0x08 +% i));
                }
                var out_buf: [32]u8 = undefined;
                const result = sanitizeForLog(input_buf[0..len], &out_buf);
                std.debug.assert(result.len <= len);
                // Control chars, DEL, and invalid sequences are replaced; output
                // must be printable and valid UTF-8.
                for (result) |ch| {
                    try std.testing.expect(ch >= 0x20);
                    try std.testing.expect(ch != 0x7F);
                }
                try std.testing.expect(std.unicode.utf8ValidateSlice(result));
            }

            // hasHeader, fixed well-formed cases plus random header blobs
            {
                const has_ct = http.hasHeader("Content-Type: text/html\r\nHost: x", "Content-Type");
                std.debug.assert(has_ct);
                const has_missing = http.hasHeader("Content-Type: text/html\r\nHost: x", "Authorization");
                std.debug.assert(!has_missing);
                // Case-insensitive
                const has_ci = http.hasHeader("content-type: text/html\r\nHost: x", "Content-Type");
                std.debug.assert(has_ci);
                // Malformed / truncated / binary header lines must not crash
                var hdr_fuzz: [256]u8 = undefined;
                smith.bytesWithHash(&hdr_fuzz, 0x11);
                const hdr_len = smith.indexWithHash(hdr_fuzz.len + 1, 0x12);
                _ = http.hasHeader(hdr_fuzz[0..hdr_len], "Content-Type");
                _ = http.hasHeader(hdr_fuzz[0..hdr_len], "Authorization");
                _ = http.hasHeader(hdr_fuzz[0..hdr_len], "x-api-key");
            }

            // parseContentLength, well-formed + adversarial header blobs
            {
                const val = smith.valueWithHash(u16, 0x10);
                var hdr_buf: [64]u8 = undefined;
                const hdr = std.fmt.bufPrint(&hdr_buf, "Content-Length: {d}\r\nHost: x", .{val}) catch unreachable;
                const parsed = http.parseContentLength(hdr);
                std.debug.assert(parsed != null);
                std.debug.assert(parsed.? == @as(usize, val));
                // Duplicate / junk Content-Length must return null or a finite usize
                var cl_fuzz: [256]u8 = undefined;
                smith.bytesWithHash(&cl_fuzz, 0x13);
                const cl_len = smith.indexWithHash(cl_fuzz.len + 1, 0x14);
                _ = http.parseContentLength(cl_fuzz[0..cl_len]);
                // Explicit duplicate rejection path
                _ = http.parseContentLength("Content-Length: 10\r\nContent-Length: 20\r\n");
                _ = http.parseContentLength("Content-Length: not-a-number\r\n");
                _ = http.parseContentLength("Content-Length: \r\n");
            }

            // validateAuth, random Authorization / x-api-key header lines
            {
                var server: Server = undefined;
                server.api_key = "fuzz-secret-key";
                var auth_hdr: [256]u8 = undefined;
                smith.bytesWithHash(&auth_hdr, 0x15);
                const auth_len = smith.indexWithHash(auth_hdr.len + 1, 0x16);
                // Must never crash; must not accept unless constant-time match
                const accepted_junk = validateAuth(&server, auth_hdr[0..auth_len]);
                // Random bytes matching the exact secret are vanishingly unlikely
                std.debug.assert(!accepted_junk or std.mem.indexOf(u8, auth_hdr[0..auth_len], "fuzz-secret-key") != null);
                // Positive cases
                std.debug.assert(validateAuth(&server, "Authorization: Bearer fuzz-secret-key\r\n"));
                std.debug.assert(validateAuth(&server, "x-api-key: fuzz-secret-key\r\n"));
                std.debug.assert(!validateAuth(&server, "Authorization: Bearer wrong-key\r\n"));
                // No auth configured → always allow
                server.api_key = null;
                std.debug.assert(validateAuth(&server, auth_hdr[0..auth_len]));
            }

            // constantTimeEql
            {
                var a_buf: [8]u8 = undefined;
                var b_buf: [8]u8 = undefined;
                for (0..8) |i| {
                    a_buf[i] = smith.valueWithHash(u8, @truncate(0x20 +% i));
                    b_buf[i] = smith.valueWithHash(u8, @truncate(0x28 +% i));
                }
                // Same inputs must be equal
                std.debug.assert(constantTimeEql(&a_buf, &a_buf));
                // a == b iff they happen to be the same
                const eq = constantTimeEql(&a_buf, &b_buf);
                const mem_eq = std.mem.eql(u8, &a_buf, &b_buf);
                std.debug.assert(eq == mem_eq);
            }

            // hasToolCalls
            {
                std.debug.assert(hasToolCalls("<tool_call>{\"name\":\"foo\"}</tool_call>"));
                std.debug.assert(!hasToolCalls("No tools here"));
                // Random input
                var tc_buf: [16]u8 = undefined;
                for (0..16) |i| {
                    tc_buf[i] = smith.valueWithHash(u8, @truncate(0x30 +% i));
                }
                _ = hasToolCalls(&tc_buf);
            }

            // elapsedBetween
            {
                const a = smith.valueWithHash(i16, 0x40);
                const b = smith.valueWithHash(i16, 0x41);
                const start: i64 = @intCast(a);
                const end: i64 = @intCast(b);
                const result = elapsedBetween(start, end);
                if (end >= start) {
                    std.debug.assert(result == @as(u64, @intCast(end - start)));
                } else {
                    std.debug.assert(result == 0);
                }
            }

            // http.HttpReadResult union: verify layout at comptime
            comptime {
                _ = @as(?http.HttpReadResult, null);
                _ = http.HttpReadResult{ .malformed = "" };
                _ = http.HttpReadResult{ .body_too_large = "" };
            }

            // splitPathQuery + extractQueryParam, untrusted request-target / query string
            {
                var path_buf: [256]u8 = undefined;
                smith.bytesWithHash(&path_buf, 0x50);
                const path_len = smith.indexWithHash(path_buf.len + 1, 0x51);
                const pq = http.splitPathQuery(path_buf[0..path_len]);
                // Invariant: path + optional '?' + query reconstructs the input
                if (pq.query.len == 0 and path_len > 0 and path_buf[path_len - 1] != '?') {
                    std.debug.assert(std.mem.eql(u8, pq.path, path_buf[0..path_len]));
                } else if (std.mem.indexOf(u8, path_buf[0..path_len], "?")) |q| {
                    std.debug.assert(std.mem.eql(u8, pq.path, path_buf[0..q]));
                    std.debug.assert(std.mem.eql(u8, pq.query, path_buf[q + 1 .. path_len]));
                }
                // extractQueryParam on raw query bytes + structured key=value pairs
                var key_buf: [16]u8 = undefined;
                smith.bytesWithHash(&key_buf, 0x52);
                const key_len = smith.indexWithHash(key_buf.len + 1, 0x53);
                _ = http.extractQueryParam(pq.query, key_buf[0..key_len]);
                _ = http.extractQueryParam(path_buf[0..path_len], "n_tokens");

                var q_struct: [128]u8 = undefined;
                const qn = std.fmt.bufPrint(&q_struct, "n_tokens={d}&foo={s}&n_tokens={d}", .{
                    smith.valueWithHash(u16, 0x54),
                    key_buf[0..@min(key_len, 8)],
                    smith.valueWithHash(u16, 0x55),
                }) catch unreachable;
                // First match wins, must not crash on duplicate keys
                const ntok = http.extractQueryParam(qn, "n_tokens");
                std.debug.assert(ntok != null);
                _ = parsePositiveQueryParam(qn, "n_tokens");
                _ = parsePositiveQueryParam(path_buf[0..path_len], "n_tokens");
                std.debug.assert(std.meta.activeTag(parsePositiveQueryParam("foo=1", "n_tokens")) == .missing);
                std.debug.assert(std.meta.activeTag(parsePositiveQueryParam("n_tokens=0", "n_tokens")) == .invalid);
                std.debug.assert(std.meta.activeTag(parsePositiveQueryParam("n_tokens=512", "n_tokens")) == .value);
            }

            // parseRequestLine, untrusted HTTP request-line bytes
            {
                var line_buf: [256]u8 = undefined;
                smith.bytesWithHash(&line_buf, 0x60);
                const line_len = smith.indexWithHash(line_buf.len + 1, 0x61);
                _ = http.parseRequestLine(line_buf[0..line_len]);

                // Structure-aware well-formed request lines
                var good: [160]u8 = undefined;
                const gn = std.fmt.bufPrint(&good, "POST /v1/chat/completions?n_tokens={d} HTTP/1.1", .{
                    smith.valueWithHash(u16, 0x62),
                }) catch unreachable;
                const parsed = http.parseRequestLine(gn).?;
                std.debug.assert(std.mem.eql(u8, parsed.method, "POST"));
                std.debug.assert(std.mem.eql(u8, parsed.path, "/v1/chat/completions"));
                std.debug.assert(parsed.query.len > 0);

                std.debug.assert(http.parseRequestLine("GET /health") == null); // missing version SP
                std.debug.assert(http.parseRequestLine("") == null);
                std.debug.assert(http.parseRequestLine("NOSPACES") == null);
            }

            // parseDetokenizeTokens, untrusted /v1/detokenize JSON bodies
            {
                var body_buf: [256]u8 = undefined;
                smith.bytesWithHash(&body_buf, 0x70);
                const body_len = smith.indexWithHash(body_buf.len + 1, 0x71);
                var out_ids: [64]u32 = undefined;
                const rand_toks = parseDetokenizeTokens(body_buf[0..body_len], &out_ids);
                std.debug.assert(rand_toks == .empty or rand_toks == .malformed or
                    rand_toks.count <= out_ids.len);

                var good_body: [128]u8 = undefined;
                const bn = std.fmt.bufPrint(&good_body, "{{\"tokens\":[{d},{d},9999999999]}}", .{
                    smith.valueWithHash(u16, 0x72),
                    smith.valueWithHash(u32, 0x73),
                }) catch unreachable;
                const n_good = parseDetokenizeTokens(bn, &out_ids);
                // A value beyond u32 rejects the whole array rather than truncating it.
                std.debug.assert(n_good == .malformed or n_good.count >= 2);

                std.debug.assert(parseDetokenizeTokens("{}", &out_ids) == .empty);
                std.debug.assert(parseDetokenizeTokens("{\"tokens\":[]}", &out_ids) == .empty);
                std.debug.assert(parseDetokenizeTokens("{\"tokens\":[1,2,3]}", out_ids[0..2]) == .malformed);

                _ = parseFormConversationId(body_buf[0..body_len]);
                std.debug.assert(std.meta.activeTag(parseFormConversationId("action=select")) == .missing);
                std.debug.assert(std.meta.activeTag(parseFormConversationId("id=")) == .invalid);
                std.debug.assert(std.meta.activeTag(parseFormConversationId("id=0")) == .invalid);
                std.debug.assert(std.meta.activeTag(parseFormConversationId("id=1")) == .value);
            }

            // KnownEndpoint: verify at comptime
            comptime {
                std.debug.assert(known_endpoints.len > 0);
                for (known_endpoints) |ep| {
                    std.debug.assert(ep.path.len > 0);
                    std.debug.assert(ep.allow.len > 0);
                }
            }
        }
    }.f, .{});
}

test "fuzz: http.readHttpRequest over socket" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // Build the raw request bytes sent by the "client".
            var payload_buf: [512]u8 = undefined;
            smith.bytesWithHash(&payload_buf, 0);
            var payload_len = smith.indexWithHash(payload_buf.len + 1, 1);

            // Structure-aware seeding: half the time plant a well-formed
            // skeleton so deep paths (Content-Length parsing, body read loop,
            // split-header overlap scan) are reached instead of failing at the
            // request line.
            if (smith.valueWithHash(u8, 2) & 1 == 0) {
                const body_len: usize = smith.valueWithHash(u8, 3) % 16;
                const skeleton = std.fmt.bufPrint(
                    &payload_buf,
                    "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {d}\r\nHost: t\r\n\r\n",
                    .{body_len},
                ) catch unreachable;
                const body_total = @min(body_len, payload_buf.len - skeleton.len);
                for (payload_buf[skeleton.len..][0..body_total], 0..) |*b, i| {
                    b.* = smith.valueWithHash(u8, @truncate(4 +% i));
                }
                payload_len = skeleton.len + body_total;
            }
            const payload = payload_buf[0..payload_len];

            // AF_UNIX socketpair so readHttpRequest sees EOF after the payload:
            // reads return 0 instead of blocking, so no iteration can hang.
            var fds: [2]std.posix.fd_t = undefined;
            if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &fds) != 0) return;
            var client = http.TcpStream{ .handle = fds[0] };
            var conn = http.TcpStream{ .handle = fds[1] };
            defer client.close();
            defer conn.close();

            client.writeAll(payload) catch {};
            _ = std.c.shutdown(fds[0], std.posix.SHUT.WR);

            // Deliberately small buffer: oversized Content-Length must land in
            // `.body_too_large`, exercising that branch without a 1 MB array.
            var read_buf: [256]u8 = undefined;
            switch (http.readHttpRequest(conn, &read_buf, max_request_body_size)) {
                .ok => |req| {
                    // Pair assertion across the trust boundary: the declared
                    // Content-Length must equal the delivered body byte count.
                    try std.testing.expectEqual(http.parseContentLength(req.headers).?, req.body.len);
                    // Every parsed slice must live inside the read buffer.
                    // Empty slices may be static literals, only check real ones.
                    const slices = [_][]const u8{ req.method, req.path, req.query, req.headers, req.body };
                    for (slices) |s| {
                        if (s.len == 0) continue;
                        try std.testing.expect(@intFromPtr(s.ptr) >= @intFromPtr(&read_buf));
                        try std.testing.expect(@intFromPtr(s.ptr) + s.len <= @intFromPtr(&read_buf) + read_buf.len);
                    }
                    // The request-target is split on spaces before path/query exist.
                    try std.testing.expect(std.mem.indexOfScalar(u8, req.path, ' ') == null);
                },
                // Both are valid outcomes for arbitrary input.
                .malformed => {},
                .body_too_large => {},
                // EOF after a partial payload is the common fuzz outcome.
                .connection_closed => {},
                // EPIPE/reset from the socketpair teardown race.
                .read_error => {},
            }
        }
    }.f, .{});
}

test "fuzz: HTTP header and request-line helpers" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            var buf: [192]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const headers = buf[0..len];

            // getHeaderValue: any returned value must be a subslice of the
            // input, and must not contain the line separator.
            _ = http.acceptsEncoding(headers, "gzip");
            _ = http.ifNoneMatch(headers, "\"abc\"");
            if (http.getHeaderValue(headers, "host")) |val| {
                const v0 = @intFromPtr(val.ptr);
                const v1 = v0 + val.len;
                const h0 = @intFromPtr(headers.ptr);
                const h1 = h0 + headers.len;
                try std.testing.expect(v0 >= h0 and v1 <= h1);
                try std.testing.expect(std.mem.indexOf(u8, val, "\r\n") == null);
            }

            // parseRequestLine + splitPathQuery: parsed slices stay inside the
            // line; path/query cannot contain spaces (split happens first).
            if (http.parseRequestLine(headers)) |line| {
                const h0 = @intFromPtr(headers.ptr);
                const h1 = h0 + headers.len;
                for ([_][]const u8{ line.method, line.path, line.query }) |s| {
                    const s0 = @intFromPtr(s.ptr);
                    try std.testing.expect(s0 >= h0 and s0 + s.len <= h1);
                    try std.testing.expect(std.mem.indexOfScalar(u8, s, ' ') == null);
                }
            }

            // extractQueryParam: returned value is a query subslice.
            if (http.extractQueryParam(headers, "prompt")) |val| {
                const v0 = @intFromPtr(val.ptr);
                const h0 = @intFromPtr(headers.ptr);
                try std.testing.expect(v0 >= h0 and v0 + val.len <= h0 + headers.len);
            }

            // parseContentLength: duplicate Content-Length must be rejected
            // regardless of surrounding junk (RFC 7230 §3.3.3 smuggling guard).
            _ = http.parseContentLength(headers);
            var dup_buf: [128]u8 = undefined;
            const dup = std.fmt.bufPrint(&dup_buf, "Content-Length: 5\r\n{s}\r\ncontent-length: 7", .{headers[0..@min(headers.len, 64)]}) catch return;
            try std.testing.expect(http.parseContentLength(dup) == null);

            // originMatchesHost: a positive verdict implies scheme-stripped
            // case-insensitive equality with no delimiter characters.
            const sep = smith.indexWithHash(len + 1, 2);
            const origin = headers[0..sep];
            const host = headers[@min(sep, len)..];
            if (http.originMatchesHost(origin, host)) {
                const rest = if (std.mem.startsWith(u8, origin, "https://"))
                    origin["https://".len..]
                else
                    origin["http://".len..];
                try std.testing.expect(std.ascii.eqlIgnoreCase(rest, host));
                try std.testing.expect(std.mem.indexOfAny(u8, rest, "/@?#") == null);
            }

            // sanitizeClientRequestId: either rejects (0) or copies the whole
            // input; never panics or returns a length that would overrun `out`.
            var out: [max_client_request_id_len]u8 = undefined;
            const n = http.sanitizeClientRequestId(headers, &out);
            try std.testing.expect(n == 0 or n == headers.len);
            if (n > 0) try std.testing.expectEqualStrings(headers, out[0..n]);

            // isLoopbackHttpHost: a positive verdict is localhost, ::1, or 127/8.
            _ = http.isLoopbackHttpHost(host);
            _ = http.isLoopbackHttpHost(origin);
            if (http.isLoopbackHttpHost(host)) {
                const hn = http.hostnameFromHost(host);
                const name = if (hn.len > 0 and hn[hn.len - 1] == '.') hn[0 .. hn.len - 1] else hn;
                const loop = std.ascii.eqlIgnoreCase(name, "localhost") or
                    std.mem.eql(u8, name, "::1") or
                    (name.len >= 5 and std.mem.startsWith(u8, name, "127."));
                try std.testing.expect(loop);
            }
        }
    }.f, .{});
}

test "auth chokepoint denies every protected route" {
    var server: Server = undefined;
    server.api_key = "matrix-secret";
    const bad = "Authorization: Bearer wrong-key\r\n";
    const good = "Authorization: Bearer matrix-secret\r\n";

    var optional_paths: usize = 0;
    var public_paths: usize = 0;
    for (known_endpoints) |ep| {
        switch (ep.auth) {
            .required => {
                try std.testing.expect(!authorizedForPath(&server, ep.path, bad));
                try std.testing.expect(authorizedForPath(&server, ep.path, good));
            },
            // Probes stay reachable, but only these two, and only because their
            // handlers trim the body for a caller that failed validateAuth.
            .optional => {
                optional_paths += 1;
                try std.testing.expect(std.mem.eql(u8, ep.path, "/health") or std.mem.eql(u8, ep.path, "/ready"));
                try std.testing.expect(authorizedForPath(&server, ep.path, bad));
            },
            .public => {
                public_paths += 1;
                try std.testing.expectEqualStrings("/favicon.ico", ep.path);
                try std.testing.expect(authorizedForPath(&server, ep.path, bad));
            },
        }
    }
    try std.testing.expectEqual(@as(usize, 2), optional_paths);
    try std.testing.expectEqual(@as(usize, 1), public_paths);

    // A path with no table entry (typo, unrouted verb, future route) must not
    // be reachable unauthenticated by omission.
    try std.testing.expectEqual(AuthPolicy.required, authPolicyFor("/v1/nope"));
    try std.testing.expect(!authorizedForPath(&server, "/v1/nope", bad));
    try std.testing.expect(!authorizedForPath(&server, "/V1/MODELS", bad));
}

test "sanitizeForLog replaces C1 controls and invalid UTF-8" {
    var buf: [64]u8 = undefined;
    // U+009B (CSI) reaches the log as two ordinary bytes (C2 9B); a byte filter
    // would pass it and the terminal would run the escape that follows.
    const csi = sanitizeForLog("/x\xc2\x9b\x31m", &buf);
    try std.testing.expectEqualStrings("/x?1m", csi);
    try std.testing.expect(std.unicode.utf8ValidateSlice(csi));
}

test "sanitizeForLog clips on a character boundary" {
    var buf: [3]u8 = undefined;
    // "aaé" is 4 bytes; a 3-byte cap would leave half of the é behind.
    const s = sanitizeForLog("aa\xc3\xa9", &buf);
    try std.testing.expectEqualStrings("aa?", s);
    try std.testing.expect(std.unicode.utf8ValidateSlice(s));
}

test "sanitizeForLog keeps printable multibyte text intact" {
    var buf: [64]u8 = undefined;
    const s = sanitizeForLog("/模型/🌵", &buf);
    try std.testing.expectEqualStrings("/模型/🌵", s);
}

/// True when `part` points inside `whole` (both are borrowed from one buffer).
fn sliceWithin(whole: []const u8, part: []const u8) bool {
    const base = @intFromPtr(whole.ptr);
    const start = @intFromPtr(part.ptr);
    return start >= base and start + part.len <= base + whole.len;
}

test "fuzz: model output tool-call, thinking, and tool-input parsers" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            var noise: [256]u8 = undefined;
            smith.bytesWithHash(&noise, 0);
            const noise_len = smith.indexWithHash(noise.len + 1, 1);
            const n = noise[0..noise_len];

            var tp = json.ToolParams{};
            tp.tools[0] = .{ .name = "get_weather", .description = "", .parameters_json = "{}" };
            tp.tool_count = 1;
            var reg = tools_mod.Registry{};

            // Model output, not a request body: build payloads that name a
            // declared tool and carry the three shapes `arguments` takes on
            // the wire, so the parse runs on nested, truncated, and escaped
            // bodies instead of only unrelated bytes.
            var args_buf: [320]u8 = undefined;
            const args = switch (smith.indexWithHash(3, 2)) {
                0 => std.fmt.bufPrint(&args_buf, "{s}", .{n}) catch return,
                1 => std.fmt.bufPrint(&args_buf, "\"{{\\\"{s}\"", .{n}) catch return,
                else => std.fmt.bufPrint(&args_buf, "[\"{s}\"]", .{n}) catch return,
            };
            var payload_buf: [512]u8 = undefined;
            const payload = std.fmt.bufPrint(&payload_buf, "{{\"name\":\"get_weather\",\"arguments\":{s}}}", .{args}) catch return;

            var text_buf: [1024]u8 = undefined;
            const text = std.fmt.bufPrint(&text_buf, "{s}<tool_call>{s}</tool_call><think>{s}</think>{s}", .{ n, payload, n, n }) catch return;

            var pos: usize = 0;
            var accepted: usize = 0;
            while (true) {
                const before = pos;
                const tc_json = nextAllowedToolCall(text, &pos, &tp, &reg) orelse break;
                accepted += 1;
                // The scan must move forward every call, and a payload can
                // only be accepted once per occurrence in the output.
                try std.testing.expect(pos > before);
                try std.testing.expect(accepted <= text.len);
                try std.testing.expect(sliceWithin(text, tc_json));
                const name = json.extractField(tc_json, "name") orelse return error.TestUnexpectedResult;
                try std.testing.expect(tp.hasTool(name) or reg.hasName(name));

                // The Anthropic response embeds this text verbatim as
                // `input`, so it has to be a complete JSON object.
                const resolved = resolveAnthropicToolInput(allocator, tc_json, accepted);
                defer if (resolved.owned) |p| allocator.free(p);
                try std.testing.expect(isCompleteJsonObject(resolved.obj));
            }

            const split = splitThinkingContent(text);
            try std.testing.expect(sliceWithin(text, split.reasoning));
            try std.testing.expect(sliceWithin(text, split.content));
            try std.testing.expect(hasToolCalls(text));
        }
    }.f, .{});
}
