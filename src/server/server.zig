//! HTTP server with OpenAI-compatible and Anthropic-compatible API endpoints.
//! Provides /v1/chat/completions, /v1/completions, /v1/models, /v1/responses,
//! /v1/messages (Anthropic Messages API), /v1/embeddings, /v1/conversations,
//! /v1/chat (built-in web UI), /v1/chat/regenerate, /v1/tokenize, /v1/detokenize,
//! /health, /ready, and /metrics.
//! Supports both synchronous JSON responses and SSE streaming.
//! Uses std.net with per-connection threads; inference is mutex-serialized.

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
const RateLimiter = @import("rate_limiter.zig").RateLimiter;
const metrics_mod = @import("metrics.zig");
const Metrics = metrics_mod.Metrics;
const FixedBufStream = metrics_mod.FixedBufStream;
const json = @import("json.zig");
const SamplingParams = json.SamplingParams;
const TieredKvCache = @import("../kvcache/tiered.zig").TieredKvCache;
const VisionEncoder = @import("../models/model.zig").VisionEncoder;
const image_mod = @import("../image.zig");
const engine_version = @import("../display.zig").version;
const grammar_mod = @import("../grammar.zig");

const Mutex = Io.Mutex;

/// Lightweight wrapper providing writeAll/read/close over a raw socket fd.
const TcpStream = struct {
    handle: std.posix.fd_t,

    pub fn writeAll(self: TcpStream, data: []const u8) !void {
        var written: usize = 0;
        while (written < data.len) {
            const n = std.c.write(self.handle, data[written..].ptr, data[written..].len);
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                return error.BrokenPipe;
            }
            written += @intCast(n);
        }
    }

    pub fn read(self: TcpStream, buf: []u8) !usize {
        while (true) {
            const n = std.c.read(self.handle, buf.ptr, buf.len);
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                return error.ConnectionResetByPeer;
            }
            return @intCast(n);
        }
    }

    pub fn close(self: TcpStream) void {
        _ = std.c.close(self.handle);
    }
};

// ── Server constants ────────────────────────────────────────────
const slog_buf_size: usize = 4096;
const models_json_buf_size: usize = 1024;
const response_buf_size: usize = 65536;
const cmd_buf_size: usize = 1024;
const gen_ids_buf_size: usize = 4096;
const default_max_gen_tokens: usize = 512;
const system_fingerprint = "agave-v" ++ engine_version;

/// Clamp a max_tokens value to [1, gen_ids_buf_size].
fn clampMaxTokens(raw: ?usize) usize {
    return @max(1, @min(raw orelse default_max_gen_tokens, gen_ids_buf_size));
}
const conv_title_max_len: usize = 48;
const conv_list_buf_size: usize = 8192;
const conv_msgs_buf_size: usize = 65536;
const http_buf_size: usize = 1024 * 1024;
const hdr_buf_size: usize = 1024;
const short_hdr_buf_size: usize = 512;
const error_body_buf_size: usize = 512;
const max_log_path_len: usize = 256;
const health_buf_size: usize = 768;
const metrics_render_buf_size: usize = 65536;
const stats_buf_size: usize = 512;
const sse_event_buf_size: usize = 1024;
const logprob_buf_size: usize = 4096;
const clear_response_buf_size: usize = 128;
/// Must not exceed http_buf_size — headers and body share the same read buffer.
const max_request_body_size: usize = http_buf_size;
const max_conversations: usize = 100;
const max_messages_per_conv: usize = 1000;
const max_message_len: usize = 100_000;
const max_concurrent_connections: u32 = 64;
const scheduler_max_batch_size: usize = 8;
const scheduler_timeout_sec: u32 = 120;
const scheduler_poll_interval_ns: u64 = 1_000_000; // 1ms — matches scheduler_poll_ns in scheduler.zig
/// Allows Ctrl+C to interrupt the accept loop.
const accept_timeout_sec: i64 = 1;
const ms_per_second: f32 = 1000.0;
const stop_sequence_window: usize = 8;

const stderr_file = Io.File.stderr();
const stdout_file = Io.File.stdout();

/// Millisecond timestamp via raw clock_gettime (avoids Io dispatch overhead).
fn milliTimestamp() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000);
}

/// Nanosecond timestamp for seed generation.
fn nanoTimestamp() i96 {
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i96, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// Sleep for nanoseconds via C nanosleep.
fn sleepNs(ns: u64) void {
    const ts = std.posix.timespec{
        .sec = @intCast(ns / std.time.ns_per_s),
        .nsec = @intCast(ns % std.time.ns_per_s),
    };
    _ = std.c.nanosleep(&ts, null);
}

/// Seconds since epoch for log timestamps.
fn timestamp() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return ts.sec;
}

/// Background sleep monitor: checks idle time every 10s.
/// Sets g_server.sleeping=true after sleep_after_s seconds of inactivity.
/// The server remains in "sleep" state (serving requests at normal speed)
/// but signals external orchestrators via /health "sleeping": true.
/// Future: could release optional caches or defer prefetching here.
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
            g_server.sleeping.store(true, .release);
            std.log.info("server: entering sleep mode after {d}s idle", .{idle_s});
        }
    }
}

/// Compute tokens-per-second from a token count and elapsed milliseconds.
/// Returns 0 if elapsed time is zero (avoids division by zero).
fn tokensPerSec(token_count: u32, time_ms: u64) f32 {
    return if (time_ms > 0) @as(f32, @floatFromInt(token_count)) / (@as(f32, @floatFromInt(time_ms)) / ms_per_second) else 0.0;
}

/// Known API endpoints with their allowed HTTP methods and error messages.
/// Shared by the CORS OPTIONS handler (path-specific Access-Control-Allow-Methods)
/// and the 405 Method Not Allowed handler.
const KnownEndpoint = struct { path: []const u8, allow: []const u8, msg: []const u8, is_anthropic: bool = false };
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
    .{ .path = "/health", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/ready", .allow = "GET, OPTIONS", .msg = "Use GET." },
    .{ .path = "/metrics", .allow = "GET, OPTIONS", .msg = "Use GET." },
};
/// Per-connection read timeout (seconds) — prevents slow loris DoS attacks
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
/// Seconds per minute — used for UTC time decomposition in request logs.
const seconds_per_minute: u64 = 60;
/// Seconds per hour — used for UTC time decomposition in request logs.
const seconds_per_hour: u64 = 3600;
/// Hours per day — used for UTC time decomposition in request logs.
const hours_per_day: u64 = 24;
/// CORS preflight cache duration in seconds (24 hours).
const cors_max_age_seconds = "86400";
/// CORS response headers for cross-origin browser access (only sent when no API key configured).
/// Includes Access-Control-Expose-Headers so browsers can read X-Request-Id (log correlation)
/// and Retry-After (rate limiting) — these are non-safelisted headers hidden by default.
const cors_allow_headers = "Access-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: X-Request-Id, Retry-After\r\n";

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
        const len: u8 = @intCast(@min(text.len, conv_title_max_len));
        @memcpy(self.title_buf[0..len], text[0..len]);
        self.title_len = len;
    }

    fn freeMessageContents(self: *Conversation, allocator: Allocator) void {
        for (self.messages.items) |msg| allocator.free(@constCast(msg.content));
    }

    fn clearMessages(self: *Conversation, allocator: Allocator) void {
        self.freeMessageContents(allocator);
        self.messages.clearRetainingCapacity();
    }

    fn freeMessages(self: *Conversation, allocator: Allocator) void {
        self.freeMessageContents(allocator);
        self.messages.deinit(allocator);
    }
};

/// Server state — bundles all mutable state into a single struct
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
    /// Runtime-only conversation storage for the web UI.
    conversations: std.ArrayList(Conversation) = .empty,
    active_id: u32 = 0,
    next_id: u32 = 1,
    /// Whether the KV cache matches the active conversation's state.
    kv_valid: bool = false,
    /// Cached prompt token IDs from the last API generation (for prefix reuse).
    cached_prompt_ids: []u32 = &.{},
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
    /// Number of visual tokens from the last processVisionImage call.
    pending_visual_tokens: u32 = 0,
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

    fn getActiveConv(self: *Server) ?*Conversation {
        return self.getConvById(self.active_id);
    }

    fn getConvById(self: *Server, id: u32) ?*Conversation {
        for (self.conversations.items) |*conv| {
            if (conv.id == id) return conv;
        }
        return null;
    }

    /// Create a new conversation. Caller must hold self.mutex.
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
        if (self.cached_prompt_ids.len > 0) {
            self.allocator.free(self.cached_prompt_ids);
            self.cached_prompt_ids = &.{};
        }
        return &self.conversations.items[self.conversations.items.len - 1];
    }

    /// Delete a conversation by ID. Caller must hold self.mutex.
    fn deleteConv(self: *Server, id: u32) void {
        for (self.conversations.items, 0..) |*conv, i| {
            if (conv.id == id) {
                conv.freeMessages(self.allocator);
                _ = self.conversations.swapRemove(i);
                break;
            }
        }
        if (self.active_id == id) {
            self.active_id = if (self.conversations.items.len > 0)
                self.conversations.items[self.conversations.items.len - 1].id
            else
                0;
            self.kv_valid = false;
        }
    }

    /// Select a conversation by ID. Caller must hold self.mutex.
    fn selectConv(self: *Server, id: u32) void {
        if (self.active_id != id) {
            self.active_id = id;
            self.kv_valid = false;
        }
    }

    fn isEog(self: *const Server, token: u32) bool {
        for (self.eog_ids[0..self.eog_len]) |id| {
            if (token == id) return true;
        }
        return false;
    }
};

var g_server: *Server = undefined;

/// Per-thread request ID for log correlation. Set at the start of each
/// handleRequest() call so all log lines from the same request (including
/// logGeneration calls deep in generate functions) share the same ID.
threadlocal var log_request_id: u64 = 0;

fn slog(comptime fmt: []const u8, args: anytype) void {
    g_server.stdout_mutex.lockUncancelable(g_server.io);
    defer g_server.stdout_mutex.unlock(g_server.io);
    var buf: [slog_buf_size]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch {
        const trunc = "[slog truncated]\n";
        _ = std.c.write(stderr_file.handle, trunc.ptr, trunc.len);
        return;
    };
    _ = std.c.write(stderr_file.handle, text.ptr, text.len);
}

fn elapsedMs(start: i64) u64 {
    return @intCast(@max(milliTimestamp() - start, 0));
}

/// Compute non-negative elapsed milliseconds between two timestamps.
fn elapsedBetween(start: i64, end: i64) u64 {
    return @intCast(@max(end - start, 0));
}

/// Estimate prompt token count: use actual tokenized count when available,
/// fall back to byte-length estimate (1 byte = 1 token) to prevent rate
/// limiter bypass on tokenizer failure.
fn estimatePromptTokens(token_count: usize, text_len: usize) u32 {
    return if (token_count > 0) @intCast(token_count) else @intCast(@max(1, text_len));
}

/// Characters unsafe for direct embedding in JSON string values or HTML contexts.
/// Characters unsafe for direct embedding in JSON string values or HTML contexts.
fn isUnsafeJsonChar(c: u8) bool {
    return c == '"' or c == '\\' or c < 0x20 or c == '<' or c == '>' or c == '&';
}

/// CORS headers: include origin/expose headers when no API key is configured
/// (public mode); omit when auth is required (prevents wildcard origin leaks).
fn corsHeaders() []const u8 {
    return if (g_server.api_key != null) "" else cors_allow_headers;
}

/// Broken-down UTC time for request log timestamps.
const TimeComponents = struct { hours: u64, minutes: u64, seconds: u64 };

fn getTimeComponents() TimeComponents {
    const now = timestamp();
    return .{
        .hours = @intCast(@mod(@divTrunc(now, seconds_per_hour), hours_per_day)),
        .minutes = @intCast(@mod(@divTrunc(now, seconds_per_minute), seconds_per_minute)),
        .seconds = @intCast(@mod(now, seconds_per_minute)),
    };
}

/// Sanitize a string for safe terminal output by replacing control characters
/// (bytes < 0x20 except space, and DEL 0x7F) with '?'. Prevents log injection
/// via terminal escape sequences (CWE-117).
fn sanitizeForLog(input: []const u8, buf: []u8) []const u8 {
    const len = @min(input.len, buf.len);
    for (0..len) |i| {
        const c = input[i];
        buf[i] = if ((c < 0x20 and c != ' ') or c == 0x7F) '?' else c;
    }
    return buf[0..len];
}

fn logRequest(method: []const u8, path: []const u8) void {
    const t = getTimeComponents();
    var method_buf: [16]u8 = undefined;
    var path_buf: [max_log_path_len]u8 = undefined;
    const safe_method = sanitizeForLog(method, &method_buf);
    const safe_path = sanitizeForLog(path, &path_buf);
    const rid = log_request_id;
    slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} {s} {s}\n", .{ t.hours, t.minutes, t.seconds, rid, safe_method, safe_path });
}

/// Log completion of a request with status code and duration.
fn logRequestDone(method: []const u8, path: []const u8, status: u16, duration_ms: u64) void {
    const t = getTimeComponents();
    var method_buf: [16]u8 = undefined;
    var path_buf: [max_log_path_len]u8 = undefined;
    const safe_method = sanitizeForLog(method, &method_buf);
    const safe_path = sanitizeForLog(path, &path_buf);
    const rid = log_request_id;
    slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} {s} {s} -> {d} ({d}ms)\n", .{ t.hours, t.minutes, t.seconds, rid, safe_method, safe_path, status, duration_ms });
}

fn logGeneration(tokens: u32, time_ms: u64, tps: f32) void {
    const t = getTimeComponents();
    const rid = log_request_id;
    if (std.c.isatty(stderr_file.handle) != 0) {
        slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} \x1b[32mGenerated {d} tokens in {d}ms ({d:.2} tok/s)\x1b[0m\n", .{ t.hours, t.minutes, t.seconds, rid, tokens, time_ms, tps });
    } else {
        slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} Generated {d} tokens in {d}ms ({d:.2} tok/s)\n", .{ t.hours, t.minutes, t.seconds, rid, tokens, time_ms, tps });
    }
}

/// Web UI HTML page — assembled at comptime from src/web/ files.
const html_page = @embedFile("../web/head.html") ++
    @embedFile("../web/style.css") ++
    @embedFile("../web/body.html") ++
    @embedFile("../web/app.js") ++
    "\n</script></body></html>\n";

/// Return the current thread's request ID (set at start of handleRequest).
/// Used for API response IDs so they match log correlation IDs.
fn currentRequestId() u64 {
    return log_request_id;
}

// ── HTTP helpers ────────────────────────────────────────────────

/// Parsed HTTP request. Slices point into the read buffer.
const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    headers: []const u8,
    body: []const u8,
};

/// Result of reading an HTTP request — distinguishes malformed requests from
/// oversized bodies so the caller can return the correct status code.
const HttpReadResult = union(enum) {
    ok: HttpRequest,
    malformed,
    body_too_large,
};

/// Check whether a given header name is present in raw HTTP headers.
fn hasHeader(headers: []const u8, name: []const u8) bool {
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        if (colon == name.len and std.ascii.eqlIgnoreCase(line[0..name.len], name)) return true;
    }
    return false;
}

/// Parse Content-Length from raw HTTP headers.
/// Returns null on parse errors or duplicate headers (RFC 7230 §3.3.3),
/// 0 when no Content-Length header is present.
fn parseContentLength(headers: []const u8) ?usize {
    const header_name = "content-length";
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    var found: ?usize = null;
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        if (colon == header_name.len and std.ascii.eqlIgnoreCase(line[0..header_name.len], header_name)) {
            const val = std.fmt.parseInt(usize, std.mem.trim(u8, line[colon + 1 ..], " "), 10) catch return null;
            if (found != null) return null; // Duplicate Content-Length — reject
            found = val;
        }
    }
    return found orelse 0;
}

/// Read a complete HTTP/1.1 request from a TCP stream. Returns `.malformed`
/// on parse errors or connection close, `.body_too_large` when Content-Length
/// exceeds max_request_body_size (RFC 7231 §6.5.11).
fn readHttpRequest(stream: TcpStream, buf: []u8) HttpReadResult {
    var total: usize = 0;
    var hdr_end: usize = undefined;

    // Read until we have complete headers (\r\n\r\n).
    // Scan only the newly-received region (plus 3-byte overlap for split boundary).
    while (total < buf.len) {
        const n = stream.read(buf[total..]) catch return .malformed;
        if (n == 0) return .malformed;
        const scan_start = if (total >= 3) total - 3 else 0;
        total += n;
        if (std.mem.indexOf(u8, buf[scan_start..total], "\r\n\r\n")) |pos| {
            hdr_end = scan_start + pos;
            break;
        }
    } else return .malformed;

    // Parse request line: "GET /path HTTP/1.1"
    const req_line_end = std.mem.indexOf(u8, buf[0..hdr_end], "\r\n") orelse return .malformed;
    const req_line = buf[0..req_line_end];
    const sp1 = std.mem.indexOf(u8, req_line, " ") orelse return .malformed;
    const method = req_line[0..sp1];
    const rest = req_line[sp1 + 1 ..];
    const sp2 = std.mem.indexOf(u8, rest, " ") orelse return .malformed;
    const raw_path = rest[0..sp2];
    // Strip query string
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |q| raw_path[0..q] else raw_path;

    // Parse Content-Length (null = duplicate headers, reject per RFC 7230)
    const headers = buf[req_line_end + 2 .. hdr_end];

    // Reject Transfer-Encoding — this server only supports identity encoding.
    // Accepting chunked requests without parsing them enables HTTP request
    // smuggling (CWE-444) when behind a reverse proxy.
    if (hasHeader(headers, "transfer-encoding")) return .malformed;

    const content_length = parseContentLength(headers) orelse return .malformed;
    const body_start = hdr_end + 4;

    // Read remaining body bytes if needed
    if (content_length > 0) {
        if (content_length > max_request_body_size) return .body_too_large;
        const body_end = body_start + content_length;
        if (body_end > buf.len) return .body_too_large;
        while (total < body_end) {
            const n = stream.read(buf[total..body_end]) catch return .malformed;
            if (n == 0) return .malformed;
            total += n;
        }
        return .{ .ok = .{ .method = method, .path = path, .headers = headers, .body = buf[body_start..body_end] } };
    }

    return .{ .ok = .{ .method = method, .path = path, .headers = headers, .body = "" } };
}

/// Common security headers appended to every response.
const security_headers =
    "X-Content-Type-Options: nosniff\r\n" ++
    "X-Frame-Options: DENY\r\n" ++
    "Referrer-Policy: no-referrer\r\n" ++
    "Cache-Control: no-store\r\n" ++
    "Strict-Transport-Security: max-age=31536000; includeSubDomains\r\n" ++
    "Permissions-Policy: geolocation=(), microphone=(), camera=(), accelerometer=(), gyroscope=()\r\n" ++
    "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; img-src 'self' data: blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'\r\n";

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

/// Constant-time byte comparison to prevent timing side-channel attacks on secrets.
/// Always iterates over the secret length (b) to avoid leaking key length.
/// Accumulates XOR differences into a single byte — the compiler cannot
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
fn sendResponse(stream: TcpStream, status: []const u8, content_type: []const u8, body: []const u8) void {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: close\r\n\r\n", .{ status, content_type, body.len, log_request_id, corsHeaders() }) catch {
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

fn sendJson(stream: TcpStream, body: []const u8) void {
    sendResponse(stream, "200 OK", "application/json", body);
}

fn sendHtml(stream: TcpStream, body: []const u8) void {
    sendResponse(stream, "200 OK", "text/html; charset=utf-8", body);
}

/// Build system prompt with tool definitions injected.
fn buildToolSystemPrompt(allocator: Allocator, tp: *const json.ToolParams, existing_system: ?[]const u8) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    if (existing_system) |sys| {
        try buf.appendSlice(allocator, sys);
        try buf.appendSlice(allocator, "\n\n");
    }
    try buf.appendSlice(allocator, "You have access to the following tools:\n\n");
    for (0..tp.tool_count) |i| {
        if (tp.tools[i]) |tool| {
            try buf.appendSlice(allocator, "- ");
            try buf.appendSlice(allocator, tool.name);
            if (tool.description.len > 0) {
                try buf.appendSlice(allocator, ": ");
                try buf.appendSlice(allocator, tool.description);
            }
            try buf.appendSlice(allocator, "\n  Parameters: ");
            try buf.appendSlice(allocator, tool.parameters_json);
            try buf.appendSlice(allocator, "\n");
        }
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
/// Returns true if tool calls found and writes response. Otherwise returns false.
fn hasToolCalls(text: []const u8) bool {
    return std.mem.indexOf(u8, text, "<tool_call>") != null;
}

/// Split generated text into (reasoning, content) parts.
/// Detects <think>...</think> (DeepSeek R1, QwQ) and similar markers.
/// Returns reasoning slice and content slice — both reference the original text (no alloc).
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

/// Build tool_calls JSON response from model output containing <tool_call> tags.
/// Supports multiple tool calls. Arguments are JSON-escaped strings per OpenAI spec.
fn buildToolCallResponse(buf: []u8, raw_text: []const u8, req_id: u64, created: i64, prompt_tokens: u32, completion_tokens: u32) []const u8 {
    const tc_start_tag = "<tool_call>";
    const tc_end_tag = "</tool_call>";
    const total = prompt_tokens + completion_tokens;

    // Build tool_calls array entries
    var tc_buf: [4096]u8 = undefined;
    var tc_pos: usize = 0;
    var search_pos: usize = 0;
    var call_idx: usize = 0;

    while (search_pos < raw_text.len) {
        const tc_start = std.mem.indexOfPos(u8, raw_text, search_pos, tc_start_tag) orelse break;
        const json_start = tc_start + tc_start_tag.len;
        const tc_end = std.mem.indexOfPos(u8, raw_text, json_start, tc_end_tag) orelse break;
        const tc_json = raw_text[json_start..tc_end];
        search_pos = tc_end + tc_end_tag.len;

        const name = json.extractField(tc_json, "name") orelse continue;
        const args = json.extractObjectField(tc_json, "arguments") orelse
            (json.extractField(tc_json, "arguments") orelse "{}");

        const escaped_args = json.jsonEscape(g_server.allocator, args) catch {
            std.log.warn("req={d} tool call argument escaping failed (OOM), skipping tool call", .{log_request_id});
            continue;
        };
        defer if (escaped_args.ptr != args.ptr) g_server.allocator.free(escaped_args);

        if (call_idx > 0 and tc_pos < tc_buf.len) {
            tc_buf[tc_pos] = ',';
            tc_pos += 1;
        }
        const entry = std.fmt.bufPrint(tc_buf[tc_pos..],
            \\{{"id":"call_{d}_{d}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}
        , .{ req_id, call_idx, name, escaped_args }) catch break;
        tc_pos += entry.len;
        call_idx += 1;
    }

    if (call_idx == 0) return "";

    return std.fmt.bufPrint(buf,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":[{s}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{ req_id, created, g_server.model_name, system_fingerprint, tc_buf[0..tc_pos], prompt_tokens, completion_tokens, total }) catch "";
}

/// Send a JSON error response in OpenAI format. Escapes message and type to prevent injection (CWE-116).
fn sendJsonError(stream: TcpStream, status: []const u8, err_type: []const u8, message: []const u8) void {
    const escaped_msg = json.jsonEscape(g_server.allocator, message) catch message;
    defer if (escaped_msg.ptr != message.ptr) g_server.allocator.free(escaped_msg);
    const escaped_type = json.jsonEscape(g_server.allocator, err_type) catch err_type;
    defer if (escaped_type.ptr != err_type.ptr) g_server.allocator.free(escaped_type);
    var buf: [error_body_buf_size]u8 = undefined;
    const json_body = std.fmt.bufPrint(&buf,
        \\{{"error":{{"message":"{s}","type":"{s}","param":null,"code":null}}}}
    , .{ escaped_msg, escaped_type }) catch {
        std.log.warn("req={d} error body overflow type={s}", .{ log_request_id, err_type });
        return;
    };
    sendResponse(stream, status, "application/json", json_body);
}

/// Send 401 Unauthorized response for invalid API key.
fn send401(stream: TcpStream) void {
    g_server.metrics.recordAuthFailure();
    const body = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\",\"param\":null,\"code\":\"invalid_api_key\"}}";
    sendResponse(stream, "401 Unauthorized", "application/json", body);
}

/// Write SSE response headers including X-Request-Id for log correlation.
/// Returns false if the write failed (client disconnected).
fn sendSseHeaders(stream: TcpStream) bool {
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: keep-alive\r\n\r\n", .{ log_request_id, corsHeaders() }) catch return false;
    stream.writeAll(hdr) catch |err| {
        std.log.warn("req={d} SSE header write failed: {}", .{ log_request_id, err });
        return false;
    };
    return true;
}

/// Send 429 Too Many Requests with Retry-After header.
fn send429(stream: TcpStream, retry_after: u32) void {
    g_server.metrics.recordRateLimit();
    var buf: [error_body_buf_size]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"Rate limit exceeded. Retry after {d} seconds.\",\"type\":\"rate_limit_exceeded\",\"param\":null,\"code\":\"rate_limit_exceeded\"}}}}", .{retry_after}) catch {
        std.log.warn("req={d} 429 body format failed", .{log_request_id});
        return;
    };
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after, log_request_id, corsHeaders() }) catch {
        std.log.warn("req={d} 429 header format failed", .{log_request_id});
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

// ── Request handler ─────────────────────────────────────────────

fn handleRequest(stream: TcpStream, req: HttpRequest) void {
    const request_start = milliTimestamp();
    // Wake from sleep mode on any incoming request.
    g_server.last_request_ms.store(request_start, .release);
    if (g_server.sleeping.load(.acquire)) {
        g_server.sleeping.store(false, .release);
        std.log.info("server: waking from sleep mode", .{});
    }
    const path = req.path;
    const method = req.method;
    const is_get = std.mem.eql(u8, method, "GET");
    const is_post = std.mem.eql(u8, method, "POST");

    // CORS preflight — return path-specific allowed methods
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
            "Access-Control-Max-Age: " ++ cors_max_age_seconds ++ "\r\n" ++
            "X-Request-Id: {d}\r\n" ++
            security_headers ++
            "Content-Length: 0\r\n" ++
            "Connection: close\r\n\r\n", .{ corsHeaders(), allow_methods, log_request_id }) catch return;
        stream.writeAll(opts_hdr) catch return;
        return;
    }

    // Health check endpoint — lightweight, no mutex, no inference
    if (is_get and std.mem.eql(u8, path, "/health")) {
        var buf: [health_buf_size]u8 = undefined;
        const uptime: i64 = if (g_server.start_time > 0) timestamp() - g_server.start_time else 0;
        const queue = g_server.metrics.queue_depth.load(.monotonic);
        const kv_used = g_server.metrics.kv_blocks_used.load(.monotonic);
        const kv_total = g_server.metrics.kv_blocks_total.load(.monotonic);
        const completed = g_server.metrics.requests_completed.load(.monotonic);
        const failed = g_server.metrics.requests_failed.load(.monotonic);
        const cancelled = g_server.metrics.requests_cancelled.load(.monotonic);
        const is_shutting_down = g_server.shutdown_requested.load(.acquire);
        const kv_pressure = kv_total > 0 and kv_used * 100 / kv_total >= kv_cache_degradation_pct;
        const total_settled = completed + failed;
        const high_error_rate = total_settled >= error_rate_min_requests and failed * 100 / total_settled >= error_rate_degradation_pct;
        const status: []const u8 = if (is_shutting_down) "shutting_down" else if (kv_pressure or high_error_rate) "degraded" else "ok";
        const reason: []const u8 = if (is_shutting_down) "shutting_down" else if (kv_pressure and high_error_rate) "kv_pressure,high_error_rate" else if (kv_pressure) "kv_pressure" else if (high_error_rate) "high_error_rate" else "none";
        const http_status: []const u8 = if (is_shutting_down) "503 Service Unavailable" else "200 OK";
        // When API key is configured and auth is not provided, return only
        // liveness status — omit model/version/backend to prevent fingerprinting.
        if (g_server.api_key != null and !validateAuth(g_server, req.headers)) {
            const minimal = std.fmt.bufPrint(&buf,
                \\{{"status":"{s}","reason":"{s}"}}
            , .{ status, reason }) catch return;
            sendResponse(stream, http_status, "application/json", minimal);
            return;
        }
        const kv_seq_len = g_server.model.kvSeqLen();
        const sched_errs = g_server.metrics.scheduler_errors.load(.monotonic);
        const preemptions = g_server.metrics.preemptions_total.load(.monotonic);
        const sleeping = g_server.sleeping.load(.acquire);
        const json_body = std.fmt.bufPrint(&buf,
            \\{{"status":"{s}","reason":"{s}","version":"{s}","model":"{s}","backend":"{s}","uptime_s":{d},"active_connections":{d},"requests_total":{d},"requests_completed":{d},"requests_failed":{d},"requests_cancelled":{d},"queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d},"kv_seq_len":{d},"ctx_size":{d},"scheduler_errors":{d},"preemptions":{d},"sleeping":{s}}}
        , .{ status, reason, engine_version, g_server.model_name, g_server.backend_name, uptime, g_server.metrics.active_connections.load(.monotonic), g_server.metrics.requests_total.load(.monotonic), completed, failed, cancelled, queue, kv_used, kv_total, kv_seq_len, g_server.ctx_size, sched_errs, preemptions, if (sleeping) "true" else "false" }) catch
            std.fmt.bufPrint(&buf, "{{\"status\":\"{s}\"}}", .{status}) catch return;
        sendResponse(stream, http_status, "application/json", json_body);
        return;
    }

    // Readiness check endpoint — returns 503 if shutting down, under KV cache pressure, or high error rate
    if (is_get and std.mem.eql(u8, path, "/ready")) {
        const kv_used_r = g_server.metrics.kv_blocks_used.load(.monotonic);
        const kv_total_r = g_server.metrics.kv_blocks_total.load(.monotonic);
        const queue_r = g_server.metrics.queue_depth.load(.monotonic);
        const is_shutting_down_r = g_server.shutdown_requested.load(.acquire);
        const kv_pressure_r = kv_total_r > 0 and kv_used_r * 100 / kv_total_r >= kv_cache_degradation_pct;
        const completed_r = g_server.metrics.requests_completed.load(.monotonic);
        const failed_r = g_server.metrics.requests_failed.load(.monotonic);
        const total_settled_r = completed_r + failed_r;
        const high_error_rate_r = total_settled_r >= error_rate_min_requests and failed_r * 100 / total_settled_r >= error_rate_degradation_pct;
        // When API key is configured and auth missing, return only status to prevent fingerprinting.
        const authed = g_server.api_key == null or validateAuth(g_server, req.headers);
        if (is_shutting_down_r) {
            if (authed) {
                var sbuf: [health_buf_size]u8 = undefined;
                const sjson = std.fmt.bufPrint(&sbuf,
                    \\{{"status":"shutting_down","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ queue_r, kv_used_r, kv_total_r }) catch "{\"status\":\"shutting_down\"}";
                sendResponse(stream, "503 Service Unavailable", "application/json", sjson);
            } else {
                sendResponse(stream, "503 Service Unavailable", "application/json", "{\"status\":\"shutting_down\"}");
            }
        } else if (kv_pressure_r or high_error_rate_r) {
            if (authed) {
                const ready_reason: []const u8 = if (kv_pressure_r and high_error_rate_r) "kv_pressure,high_error_rate" else if (kv_pressure_r) "kv_pressure" else "high_error_rate";
                var rbuf: [health_buf_size]u8 = undefined;
                const rjson = std.fmt.bufPrint(&rbuf,
                    \\{{"status":"degraded","reason":"{s}","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ ready_reason, queue_r, kv_used_r, kv_total_r }) catch "{\"status\":\"degraded\"}";
                sendResponse(stream, "503 Service Unavailable", "application/json", rjson);
            } else {
                sendResponse(stream, "503 Service Unavailable", "application/json", "{\"status\":\"degraded\"}");
            }
        } else {
            if (authed) {
                var rbuf: [health_buf_size]u8 = undefined;
                const rjson = std.fmt.bufPrint(&rbuf,
                    \\{{"status":"ready","queue_depth":{d},"kv_cache_used":{d},"kv_cache_total":{d}}}
                , .{ queue_r, kv_used_r, kv_total_r }) catch "{\"status\":\"ready\"}";
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
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        var buf: [metrics_render_buf_size]u8 = undefined;
        var fbs = FixedBufStream.init(&buf);
        const writer = fbs.writer();
        g_server.metrics.renderPrometheus(writer) catch {
            std.log.err("req={d} metrics render failed: buffer overflow ({d} bytes available)", .{ log_request_id, metrics_render_buf_size });
            g_server.metrics.recordFailure();
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Metrics rendering failed");
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        // Build info metric — standard Prometheus pattern for version tracking
        writer.print("# HELP agave_build_info Agave server version and configuration\n# TYPE agave_build_info gauge\nagave_build_info{{version=\"{s}\",backend=\"{s}\",language=\"zig\"}} 1\n", .{ engine_version, g_server.backend_name }) catch {
            std.log.warn("metrics buffer overflow: build_info metric truncated ({d} bytes available)", .{metrics_render_buf_size});
        };
        sendResponse(stream, "200 OK", "text/plain; version=0.0.4; charset=utf-8", fbs.getWritten());
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/favicon.ico")) {
        var fav_buf: [short_hdr_buf_size]u8 = undefined;
        const fav_hdr = std.fmt.bufPrint(&fav_buf, "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: close\r\n\r\n", .{ log_request_id, corsHeaders() }) catch return;
        stream.writeAll(fav_hdr) catch return;
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/")) {
        logRequest(method, path);
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        sendHtml(stream, html_page);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/v1/models")) {
        logRequest(method, path);

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        var buf: [models_json_buf_size]u8 = undefined;
        const kv_pos = g_server.model.kvSeqLen();
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
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const body = req.body;
        // Reject n > 1 (multiple completions not supported)
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const max_tokens = clampMaxTokens(json.extractIntField(body, "max_tokens") orelse json.extractIntField(body, "max_completion_tokens"));
        const sampling = json.parseSampling(body);
        if (sampling.user) |u| {
            // Log presence and length only — user field may contain PII (email, username)
            slog("  req={d} user=({d} chars)\n", .{ log_request_id, u.len });
        }

        // 2. Parse tools and extract messages
        const tool_params = json.parseTools(body);
        const extracted = json.extractMessages(body, g_server.allocator);
        defer if (extracted) |ex| ex.deinit(g_server.allocator);
        const fallback_raw = json.extractLastMessage(body);
        if (extracted == null and fallback_raw == null) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Missing or empty messages array");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const fallback_str = fallback_raw orelse "";
        const fallback_content = json.jsonUnescape(g_server.allocator, fallback_str) catch @constCast(fallback_str);
        defer if (fallback_content.ptr != fallback_str.ptr) g_server.allocator.free(fallback_content);

        // Inject tool definitions into system prompt
        var tool_system: ?[]u8 = null;
        defer if (tool_system) |ts| g_server.allocator.free(ts);
        if (tool_params.tool_count > 0 and !std.mem.eql(u8, tool_params.tool_choice, "none")) {
            tool_system = buildToolSystemPrompt(g_server.allocator, &tool_params, if (extracted) |ex| ex.system else null) catch |err| blk: {
                std.log.warn("req={d} tool system prompt build failed: {}", .{ log_request_id, err });
                break :blk null;
            };
        }
        const effective_system = if (tool_system) |ts| @as(?[]const u8, ts) else if (extracted) |ex| ex.system else null;

        // Format with full conversation context when available
        const formatted = if (extracted) |ex|
            g_server.chat_template.formatConversation(g_server.allocator, effective_system, ex.messages) catch
                g_server.chat_template.format(g_server.allocator, null, fallback_content) catch fallback_content
        else
            g_server.chat_template.format(g_server.allocator, null, fallback_content) catch fallback_content;
        defer if (formatted.ptr != fallback_content.ptr) g_server.allocator.free(formatted);
        const prompt_ids = g_server.tokenizer.encode(formatted) catch |err| blk: {
            std.log.warn("req={d} tokenizer encode failed: {}", .{ log_request_id, err });
            break :blk &[_]u32{};
        };
        defer if (prompt_ids.len > 0) g_server.allocator.free(prompt_ids);
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
                    slog("  Image attached and encoded ({d} visual tokens)\n", .{if (ve.patch_size > 0) ve.image_size / ve.patch_size * (ve.image_size / ve.patch_size) else 0});
                } else {
                    slog("  Image attached but decode/encode failed\n", .{});
                }
            }
        }
        defer if (completions_image_embedded) {
            g_server.model.setImageEmbeddings(null, 0, 0);
            g_server.pending_visual_tokens = 0;
        };

        if (json.extractBoolField(body, "stream")) {
            if (tool_params.tool_count > 0) {
                startStreamWithTools(stream, formatted, max_tokens, sampling, &tool_params);
            } else {
                startStream(stream, formatted, true, false, max_tokens, sampling);
            }
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Formatted already computed above for token counting
        const gen = generateEscapedN(formatted, true, max_tokens, sampling);
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

        // Check if output contains tool calls
        const json_body = if (tool_params.tool_count > 0 and hasToolCalls(gen.raw)) blk: {
            const tc_resp = buildToolCallResponse(&resp_buf, gen.raw, req_id, created, gen.stats.prompt_tokens, gen.stats.tokens_generated);
            break :blk if (tc_resp.len > 0) tc_resp else std.fmt.bufPrint(&resp_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{ req_id, created, g_server.model_name, system_fingerprint, gen.escaped, gen.finish_reason, gen.stats.prompt_tokens, gen.stats.tokens_generated, total }) catch {
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
                const reasoning_escaped = json.jsonEscape(g_server.allocator, split.reasoning) catch split.reasoning;
                defer if (reasoning_escaped.ptr != split.reasoning.ptr) g_server.allocator.free(reasoning_escaped);
                const content_escaped = json.jsonEscape(g_server.allocator, split.content) catch split.content;
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

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const body = req.body;
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const prompt_raw = json.extractField(body, "prompt") orelse {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Missing required field: prompt");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const prompt = json.jsonUnescape(g_server.allocator, prompt_raw) catch @constCast(prompt_raw);
        defer if (prompt.ptr != prompt_raw.ptr) g_server.allocator.free(prompt);
        const max_tokens = clampMaxTokens(json.extractIntField(body, "max_tokens"));
        const sampling_c = json.parseSampling(body);

        // Rate limit check (estimate prompt tokens via encode)
        const prompt_ids_c = g_server.tokenizer.encode(prompt) catch &[_]u32{};
        defer if (prompt_ids_c.len > 0) g_server.allocator.free(prompt_ids_c);
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
        const gen = generateEscapedN(prompt, true, max_tokens, sampling_c);
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
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();
        // Accept: {"text":"..."} or {"content":"..."} (raw text)
        // Also accept: {"messages":[...]} (chat-completion format — apply template then tokenize)
        const body = req.body;
        const input_text: []const u8 = blk: {
            if (json.extractField(body, "text")) |t| break :blk t;
            if (json.extractField(body, "content")) |t| break :blk t;
            if (json.extractMessages(body, g_server.allocator)) |msgs| {
                defer msgs.deinit(g_server.allocator);
                const sys = msgs.system orelse "";
                const formatted = g_server.chat_template.formatConversation(g_server.allocator, sys, msgs.messages) catch break :blk "";
                defer g_server.allocator.free(formatted);
                const tids = g_server.tokenizer.encode(formatted) catch break :blk "";
                defer g_server.allocator.free(tids);
                var resp_buf2: [response_buf_size]u8 = undefined;
                const resp2 = std.fmt.bufPrint(&resp_buf2, "{{\"count\":{d},\"model\":\"{s}\"}}", .{ tids.len, g_server.model_name }) catch break :blk "";
                sendJson(stream, resp2);
                g_server.metrics.recordCompletion();
                logRequestDone(method, path, 200, elapsedMs(request_start));
                return;
            }
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Provide text, content, or messages");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const text = input_text;
        const unescaped = json.jsonUnescape(g_server.allocator, text) catch @constCast(text);
        defer if (unescaped.ptr != text.ptr) g_server.allocator.free(unescaped);
        const token_ids = g_server.tokenizer.encode(unescaped) catch |err| {
            std.log.err("req={d} tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, unescaped.len, err });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Tokenization failed");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer g_server.allocator.free(token_ids);

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
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        // Parse token IDs from JSON array: {"tokens": [1, 2, 3]}
        var tok_ids: [gen_ids_buf_size]u32 = undefined;
        var n_toks: usize = 0;
        if (std.mem.indexOf(u8, req.body, "\"tokens\"")) |ti| {
            var di = ti + "\"tokens\"".len;
            while (di < req.body.len and (req.body[di] == ' ' or req.body[di] == ':')) : (di += 1) {}
            if (di < req.body.len and req.body[di] == '[') {
                di += 1;
                while (di < req.body.len and n_toks < gen_ids_buf_size) {
                    while (di < req.body.len and (req.body[di] == ' ' or req.body[di] == ',' or req.body[di] == '\n')) : (di += 1) {}
                    if (di >= req.body.len or req.body[di] == ']') break;
                    const num_start = di;
                    while (di < req.body.len and req.body[di] >= '0' and req.body[di] <= '9') : (di += 1) {}
                    if (di > num_start) {
                        tok_ids[n_toks] = std.fmt.parseInt(u32, req.body[num_start..di], 10) catch break;
                        n_toks += 1;
                    } else break;
                }
            }
        }
        if (n_toks == 0) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Missing or empty tokens array");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const decoded = g_server.tokenizer.decode(tok_ids[0..n_toks]) catch |err| {
            std.log.err("req={d} detokenizer decode failed ({d} tokens): {}", .{ log_request_id, n_toks, err });
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Detokenization failed");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer g_server.allocator.free(decoded);

        const escaped = json.jsonEscape(g_server.allocator, decoded) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        defer if (escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);
        var final_buf: [response_buf_size]u8 = undefined;
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

    // ── /v1/kv_cache: cross-instance KV prefix sharing (LMCache-style) ──────────
    // GET  /v1/kv_cache?n_tokens=<N>  — export N-token prefix as binary blob
    // POST /v1/kv_cache?n_tokens=<N>  — import N-token prefix from binary body
    if ((is_get or is_post) and std.mem.startsWith(u8, path, "/v1/kv_cache")) {
        logRequest(method, path);
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        // Parse n_tokens from query string: ?n_tokens=<N>
        var n_tokens: usize = 0;
        if (std.mem.indexOf(u8, req.path, "n_tokens=")) |qi| {
            const start = qi + "n_tokens=".len;
            const end = std.mem.indexOfAnyPos(u8, req.path, start, "&# ") orelse req.path.len;
            n_tokens = std.fmt.parseInt(usize, req.path[start..end], 10) catch 0;
        }
        if (n_tokens == 0) {
            sendJsonError(stream, "400 Bad Request", "invalid_request", "n_tokens required");
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        if (is_get) {
            // Export: allocate buffer, fill KV data, send as binary
            g_server.stdout_mutex.lockUncancelable(g_server.io);
            defer g_server.stdout_mutex.unlock(g_server.io);
            // Estimate buffer size: n_tokens * n_layers * kvd * 2 (k+v) * 4 bytes
            // Use 64MB upper bound for safety
            const buf = g_server.allocator.alloc(u8, 64 * 1024 * 1024) catch {
                sendJsonError(stream, "500 Internal Server Error", "server_error", "OOM");
                logRequestDone(method, path, 500, elapsedMs(request_start));
                return;
            };
            defer g_server.allocator.free(buf);
            const n_written = g_server.model.exportKvPrefix(buf, n_tokens);
            if (n_written == 0) {
                sendJsonError(stream, "501 Not Implemented", "not_implemented", "Model does not support KV export");
                logRequestDone(method, path, 501, elapsedMs(request_start));
                return;
            }
            sendResponse(stream, "200 OK", "application/octet-stream", buf[0..n_written]);
            logRequestDone(method, path, 200, elapsedMs(request_start));
        } else {
            // Import: read body as binary KV blob
            if (!g_server.model.importKvPrefix(req.body, n_tokens)) {
                sendJsonError(stream, "400 Bad Request", "invalid_request", "KV import failed (size mismatch or unsupported)");
                logRequestDone(method, path, 400, elapsedMs(request_start));
                return;
            }
            sendJson(stream, std.fmt.allocPrint(g_server.allocator, "{{\"imported\":{d}}}", .{n_tokens}) catch "{}");
            logRequestDone(method, path, 200, elapsedMs(request_start));
        }
        return;
    }

    // ── /v1/kv_cache/info: lightweight KV metadata for external orchestrators ──
    // Returns: seq_len, prefix_len, kv_used, kv_total, prefix_hash
    // Useful for: routing decisions, cross-instance prefix matching, cache monitoring
    if (is_get and std.mem.eql(u8, path, "/v1/kv_cache/info")) {
        logRequest(method, path);
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        const seq_len = g_server.model.kvSeqLen();
        const kv_used = g_server.metrics.kv_blocks_used.load(.monotonic);
        const kv_total = g_server.metrics.kv_blocks_total.load(.monotonic);
        const prefix_ids = g_server.cached_prompt_ids;
        // FNV-1a hash of cached prefix token IDs for fast remote matching
        var hash: u64 = 14695981039346656037;
        for (prefix_ids) |tid| {
            const b: [4]u8 = @bitCast(tid);
            for (b) |byte| {
                hash ^= @as(u64, byte);
                hash *%= 1099511628211;
            }
        }
        const info_json = std.fmt.allocPrint(g_server.allocator,
            \\{{"seq_len":{d},"cached_prefix_len":{d},"prefix_hash":"{x}","kv_used":{d},"kv_total":{d}}}
        , .{ seq_len, prefix_ids.len, hash, kv_used, kv_total }) catch "{}";
        defer g_server.allocator.free(info_json);
        sendJson(stream, info_json);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/embeddings")) {
        logRequest(method, path);
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();
        sendJsonError(stream, "501 Not Implemented", "not_implemented", "Embeddings endpoint not implemented");
        g_server.metrics.recordFailure();
        logRequestDone(method, path, 501, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/responses")) {
        logRequest(method, path);
        const req_start_time = milliTimestamp();

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const body = req.body;
        if ((json.extractIntField(body, "n") orelse 1) > 1) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "n > 1 is not supported; only single completions are available");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const input_raw = json.extractField(body, "input") orelse {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Missing required field: input");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        const input = json.jsonUnescape(g_server.allocator, input_raw) catch @constCast(input_raw);
        defer if (input.ptr != input_raw.ptr) g_server.allocator.free(input);
        const max_tokens = clampMaxTokens(json.extractIntField(body, "max_tokens"));
        const sampling_r = json.parseSampling(body);

        // Rate limit check
        const formatted_rl = g_server.chat_template.format(g_server.allocator, null, input) catch input;
        defer if (formatted_rl.ptr != input.ptr) g_server.allocator.free(formatted_rl);
        const prompt_ids_r = g_server.tokenizer.encode(formatted_rl) catch &[_]u32{};
        defer if (prompt_ids_r.len > 0) g_server.allocator.free(prompt_ids_r);
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

        const gen = generateEscapedN(formatted_rl, true, max_tokens, sampling_r);
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
        const json_body = std.fmt.bufPrint(&resp_buf,
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

        if (!validateAuth(g_server, req.headers)) {
            g_server.metrics.recordAuthFailure();
            sendAnthropicError(stream, "401", "authentication_error", "Invalid API key");
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const body = req.body;
        const max_tokens_m = clampMaxTokens(json.extractIntField(body, "max_tokens"));
        const sampling_m = json.parseSampling(body);
        // Anthropic: system message is a top-level field, not in messages array
        const system_msg_raw = json.extractField(body, "system");
        const system_msg = if (system_msg_raw) |s| (json.jsonUnescape(g_server.allocator, s) catch @constCast(s)) else null;
        defer if (system_msg) |s| if (system_msg_raw) |r| {
            if (s.ptr != r.ptr) g_server.allocator.free(s);
        };

        // Extract full messages array for multi-turn conversations
        const extracted_m = json.extractMessages(body, g_server.allocator);
        defer if (extracted_m) |ex| ex.deinit(g_server.allocator);
        const fallback_raw_m = json.extractLastMessage(body);
        if (extracted_m == null and fallback_raw_m == null) {
            sendAnthropicError(stream, "400", "invalid_request_error", "Missing or empty messages array");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const fallback_str_m = fallback_raw_m orelse "";
        const fallback_content_m = json.jsonUnescape(g_server.allocator, fallback_str_m) catch @constCast(fallback_str_m);
        defer if (fallback_content_m.ptr != fallback_str_m.ptr) g_server.allocator.free(fallback_content_m);

        // Format with full conversation context when available
        const formatted_m = if (extracted_m) |ex|
            g_server.chat_template.formatConversation(g_server.allocator, system_msg, ex.messages) catch
                g_server.chat_template.format(g_server.allocator, system_msg, fallback_content_m) catch fallback_content_m
        else
            g_server.chat_template.format(g_server.allocator, system_msg, fallback_content_m) catch fallback_content_m;
        defer if (formatted_m.ptr != fallback_content_m.ptr) g_server.allocator.free(formatted_m);
        const prompt_ids_m = g_server.tokenizer.encode(formatted_m) catch &[_]u32{};
        defer if (prompt_ids_m.len > 0) g_server.allocator.free(prompt_ids_m);
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
                    slog("  Image attached and encoded\n", .{});
                }
            }
        }
        defer if (anthropic_image_embedded) {
            g_server.model.setImageEmbeddings(null, 0, 0);
            g_server.pending_visual_tokens = 0;
        };

        if (json.extractBoolField(body, "stream")) {
            startAnthropicStream(stream, formatted_m, max_tokens_m, prompt_tokens_m, sampling_m);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Non-streaming: generate and return Anthropic format
        const gen = generateEscapedN(formatted_m, true, max_tokens_m, sampling_m);
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

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        if (is_get) {
            // Return list of conversations as JSON.
            // Format under mutex (reads conversation data), then release before I/O
            // to avoid blocking inference while sending to a slow client.
            var buf: [conv_list_buf_size]u8 = undefined;
            const response = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                var fbs = FixedBufStream.init(&buf);
                const w = fbs.writer();
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
                break :blk @as(?[]const u8, fbs.getWritten());
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
            const new_id: u32 = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                break :blk if (g_server.createConv()) |nc| nc.id else 0;
            };
            if (new_id == 0) {
                sendJsonError(stream, "503 Service Unavailable", "server_error", "Maximum conversation limit reached");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 503, elapsedMs(request_start));
                return;
            }
            var nbuf: [clear_response_buf_size]u8 = undefined;
            const njson = std.fmt.bufPrint(&nbuf,
                \\{{"ok":true,"id":{d}}}
            , .{new_id}) catch "{\"ok\":true}";
            sendJson(stream, njson);
            g_server.metrics.recordCompletion();
            logRequestDone(method, path, 200, elapsedMs(request_start));
        } else if (std.mem.eql(u8, action, "select")) {
            const id_str = json.extractFormField(body, "id") orelse "0";
            const id = std.fmt.parseInt(u32, id_str, 10) catch 0;
            var mbuf: [conv_msgs_buf_size]u8 = undefined;
            var mfbs = FixedBufStream.init(&mbuf);
            const select_result: enum { not_found, format_ok, format_fail } = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);

                const conv = g_server.getConvById(id) orelse break :blk .not_found;
                g_server.selectConv(id);
                const mw = mfbs.writer();
                mw.writeAll("{\"messages\":[") catch break :blk .format_fail;
                for (conv.messages.items, 0..) |msg, mi| {
                    if (mi > 0) mw.writeByte(',') catch break :blk .format_fail;
                    const role_str: []const u8 = switch (msg.role) {
                        .user => "user",
                        .assistant => "assistant",
                        .tool => "tool",
                    };
                    const esc_content = json.jsonEscape(g_server.allocator, msg.content) catch msg.content;
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
                    sendJsonError(stream, "404 Not Found", "invalid_request_error", "Conversation not found");
                    g_server.metrics.recordFailure();
                    logRequestDone(method, path, 404, elapsedMs(request_start));
                    return;
                },
                .format_ok => {
                    sendJson(stream, mfbs.getWritten());
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
            const id_str = json.extractFormField(body, "id") orelse "0";
            const id = std.fmt.parseInt(u32, id_str, 10) catch 0;
            const delete_result: ?bool = blk: {
                g_server.mutex.lockUncancelable(g_server.io);
                defer g_server.mutex.unlock(g_server.io);
                if (g_server.getConvById(id) == null) break :blk null;
                const was_active = g_server.active_id == id;
                g_server.deleteConv(id);
                break :blk was_active;
            };
            if (delete_result == null) {
                sendJsonError(stream, "404 Not Found", "invalid_request_error", "Conversation not found");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 404, elapsedMs(request_start));
                return;
            }
            const was_active = delete_result.?;
            var dbuf: [clear_response_buf_size]u8 = undefined;
            const djson = std.fmt.bufPrint(&dbuf,
                \\{{"ok":true,"cleared":{s}}}
            , .{if (was_active) "true" else "false"}) catch "{\"ok\":true}";
            sendJson(stream, djson);
            g_server.metrics.recordCompletion();
            logRequestDone(method, path, 200, elapsedMs(request_start));
        } else {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Unknown conversation action");
            g_server.metrics.recordFailure();
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

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const regen_body = req.body;
        const regen_sampling = json.parseFormSampling(regen_body);
        const regen_max_tokens = clampMaxTokens(json.extractFormInt(regen_body, "max_tokens"));

        // Extract optional system prompt (URL-decode since web UI sends encodeURIComponent)
        const regen_system_field = json.extractFormField(regen_body, "system");
        const regen_system_decoded = if (regen_system_field) |sf| (json.urlDecode(g_server.allocator, sf) catch |err| blk: {
            std.log.warn("req={d} system prompt URL decode failed: {}", .{ log_request_id, err });
            break :blk null;
        }) else null;
        defer if (regen_system_decoded) |sd| g_server.allocator.free(sd);
        const regen_system_prompt: ?[]const u8 = if (regen_system_decoded) |sd| blk: {
            const s = std.mem.trim(u8, sd, " \t\r\n");
            break :blk if (s.len > 0) s else null;
        } else null;

        const RegenPrepResult = struct { formatted: []const u8, msg_count: usize };
        const regen_prep: ?RegenPrepResult = blk: {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);

            const regen_conv = g_server.getActiveConv() orelse {
                sendJsonError(stream, "400 Bad Request", "invalid_request_error", "No active conversation");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 400, elapsedMs(request_start));
                break :blk null;
            };

            // Remove the last assistant message (if any)
            if (regen_conv.messages.items.len > 0) {
                const last_msg = regen_conv.messages.items[regen_conv.messages.items.len - 1];
                if (last_msg.role == .assistant) {
                    g_server.allocator.free(@constCast(last_msg.content));
                    _ = regen_conv.messages.pop();
                }
            }

            if (regen_conv.messages.items.len == 0) {
                sendJsonError(stream, "400 Bad Request", "invalid_request_error", "No user message to regenerate from");
                g_server.metrics.recordFailure();
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
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Failed to format conversation");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };
            break :blk RegenPrepResult{ .formatted = regen_formatted, .msg_count = regen_conv.messages.items.len };
        };
        if (regen_prep == null) return;
        const regen_formatted = regen_prep.?.formatted;
        defer g_server.allocator.free(regen_formatted);
        const regen_msg_count = regen_prep.?.msg_count;

        slog("  [regenerate] Re-generating from {d} messages\n", .{regen_msg_count});

        // Rate limit check
        const regen_prompt_ids = g_server.tokenizer.encode(regen_formatted) catch &[_]u32{};
        defer if (regen_prompt_ids.len > 0) g_server.allocator.free(regen_prompt_ids);
        const regen_prompt_tokens = estimatePromptTokens(regen_prompt_ids.len, regen_formatted.len);
        if (checkRateLimit(g_server, regen_prompt_tokens)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        // Always reset KV cache for regeneration (full re-prefill)
        const wants_stream_regen = json.extractFormBool(regen_body, "stream");
        if (wants_stream_regen) {
            if (!sendSseHeaders(stream)) {
                g_server.metrics.recordCancellation();
                return;
            }
            const regen_result = chatStreamGenerate(stream, regen_formatted, true, regen_max_tokens, regen_sampling);
            defer g_server.allocator.free(regen_result.data);
            storeConversationResponse(regen_result.data, regen_result.stats);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Non-streaming regeneration
        const regen_result = generateN(regen_formatted, true, regen_max_tokens, regen_sampling);
        defer g_server.allocator.free(regen_result.data);

        storeConversationResponse(regen_result.data, regen_result.stats);

        g_server.metrics.recordLatency(regen_result.stats.time_ms);
        g_server.metrics.recordTokens(regen_result.stats.tokens_generated);
        if (std.mem.eql(u8, regen_result.finish_reason, "error")) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

        const regen_escaped = json.htmlEscape(g_server.allocator, regen_result.data) catch regen_result.data;
        defer if (regen_escaped.ptr != regen_result.data.ptr) g_server.allocator.free(regen_escaped);
        var regen_html_buf: [response_buf_size]u8 = undefined;
        const regen_html = std.fmt.bufPrint(&regen_html_buf,
            \\<div class="msg assistant" data-tokens="{d}" data-time="{d}" data-tps="{d:.2}" data-prefill-tokens="{d}" data-prefill-ms="{d}" data-prefill-tps="{d:.1}">{s}</div>
        , .{ regen_result.stats.tokens_generated, regen_result.stats.time_ms, regen_result.stats.tokens_per_sec, regen_result.stats.prompt_tokens, regen_result.stats.prefill_ms, regen_result.stats.prefill_tps, regen_escaped }) catch "<div class=\"msg assistant\">Error</div>";
        sendHtml(stream, regen_html);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/chat")) {
        logRequest(method, path);

        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }
        g_server.metrics.recordRequest();

        const body = req.body;
        const msg = json.extractFormField(body, "message") orelse {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Missing required field: message");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        };
        if (msg.len > max_message_len) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Message too long");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }
        const decoded = json.urlDecode(g_server.allocator, msg) catch g_server.allocator.dupe(u8, msg) catch return;
        defer g_server.allocator.free(decoded);
        if (decoded.len > max_message_len) {
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Message too long");
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            return;
        }

        // Log message receipt without content (avoid leaking prompts in shared deployments)
        slog("  User message ({d} chars)\n", .{decoded.len});

        // Check for attached image data (base64-encoded data URI from web UI)
        // If a vision encoder is available, decode and encode the image into
        // visual token embeddings that the model injects during generation.
        const chat_has_image = json.extractFormImage(body) != null and g_server.vision_encoder != null;
        if (chat_has_image) g_server.vision_mutex.lockUncancelable(g_server.io);
        defer if (chat_has_image) g_server.vision_mutex.unlock(g_server.io);
        var image_embedded = false;
        if (json.extractFormImage(body)) |b64_data| {
            if (g_server.vision_encoder) |ve| {
                if (processVisionImage(b64_data, ve)) {
                    image_embedded = true;
                    slog("  Image attached and encoded ({d} visual tokens)\n", .{if (ve.patch_size > 0) ve.image_size / ve.patch_size * (ve.image_size / ve.patch_size) else 0});
                } else {
                    slog("  Image attached but decode/encode failed\n", .{});
                }
            } else {
                slog("  Image attached (no vision encoder — ignored)\n", .{});
            }
        }
        // Ensure image embeddings are cleared after generation
        defer if (image_embedded) {
            g_server.model.setImageEmbeddings(null, 0, 0);
            g_server.pending_visual_tokens = 0;
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
        defer if (system_decoded) |sd| g_server.allocator.free(sd);
        const system_prompt: ?[]const u8 = if (system_decoded) |sd| blk: {
            const s = std.mem.trim(u8, sd, " \t\r\n");
            break :blk if (s.len > 0) s else null;
        } else null;

        // Get or create active conversation, add user message, format prompt
        // — all under mutex. Returns (need_reset, formatted) or null on failure.
        const ChatPrepResult = struct { need_reset: bool, formatted: []const u8 };
        const prep_result: ?ChatPrepResult = blk: {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);

            const conv = g_server.getActiveConv() orelse g_server.createConv() orelse {
                sendJsonError(stream, "503 Service Unavailable", "server_error", "Maximum conversation limit reached");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 503, elapsedMs(request_start));
                break :blk null;
            };

            if (conv.messages.items.len >= max_messages_per_conv) {
                sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Conversation message limit reached");
                g_server.metrics.recordFailure();
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
                g_server.allocator.free(user_content);
                sendJsonError(stream, "500 Internal Server Error", "server_error", "Out of memory");
                g_server.metrics.recordFailure();
                logRequestDone(method, path, 500, elapsedMs(request_start));
                break :blk null;
            };

            if (conv.title_len == 0) conv.setTitle(trimmed);

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
            break :blk ChatPrepResult{ .need_reset = need_reset, .formatted = formatted };
        };
        if (prep_result == null) return;
        const need_reset = prep_result.?.need_reset;
        const formatted = prep_result.?.formatted;
        defer if (formatted.ptr != trimmed.ptr) g_server.allocator.free(formatted);

        // Rate limit check (matches API endpoint pattern)
        const chat_prompt_ids = g_server.tokenizer.encode(formatted) catch &[_]u32{};
        defer if (chat_prompt_ids.len > 0) g_server.allocator.free(chat_prompt_ids);
        const chat_prompt_tokens = estimatePromptTokens(chat_prompt_ids.len, formatted.len);
        if (checkRateLimit(g_server, chat_prompt_tokens)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        // Parse optional sampling parameters from form body
        const chat_sampling = json.parseFormSampling(body);
        const chat_max_tokens = clampMaxTokens(json.extractFormInt(body, "max_tokens"));

        // SSE streaming mode: stream tokens to the client in real-time
        const wants_stream = json.extractFormBool(body, "stream");
        if (wants_stream) {
            if (!sendSseHeaders(stream)) {
                g_server.metrics.recordCancellation();
                return;
            }
            const result = chatStreamGenerate(stream, formatted, need_reset, chat_max_tokens, chat_sampling);
            defer g_server.allocator.free(result.data);
            storeConversationResponse(result.data, result.stats);
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        const result = generateN(formatted, need_reset, chat_max_tokens, chat_sampling);
        defer g_server.allocator.free(result.data);
        storeConversationResponse(result.data, result.stats);

        // Record metrics
        g_server.metrics.recordLatency(result.stats.time_ms);
        g_server.metrics.recordTokens(result.stats.tokens_generated);
        if (std.mem.eql(u8, result.finish_reason, "error")) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

        // Never fall back to unescaped input — send a safe error page on OOM (CWE-79).
        const escaped_user = json.htmlEscape(g_server.allocator, decoded) catch {
            sendHtml(stream, "<div class=\"msg assistant\">Error: could not render response</div>");
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        };
        defer if (escaped_user.ptr != decoded.ptr) g_server.allocator.free(escaped_user);
        const escaped_resp = json.htmlEscape(g_server.allocator, result.data) catch {
            sendHtml(stream, "<div class=\"msg assistant\">Error: could not render response</div>");
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        };
        defer if (escaped_resp.ptr != result.data.ptr) g_server.allocator.free(escaped_resp);
        var html_buf: [response_buf_size]u8 = undefined;
        const html = std.fmt.bufPrint(&html_buf,
            \\<div class="msg user">{s}</div><div class="msg assistant" data-tokens="{d}" data-time="{d}" data-tps="{d:.2}" data-prefill-tokens="{d}" data-prefill-ms="{d}" data-prefill-tps="{d:.1}">{s}</div>
        , .{ escaped_user, result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec, result.stats.prompt_tokens, result.stats.prefill_ms, result.stats.prefill_tps, escaped_resp }) catch "<div class=\"msg assistant\">Error</div>";
        sendHtml(stream, html);
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
                , .{ep.msg}) catch {
                    logRequestDone(method, path, 405, elapsedMs(request_start));
                    return;
                }
            else
                std.fmt.bufPrint(&body_buf,
                    \\{{"error":{{"message":"Method not allowed. {s}","type":"invalid_request_error","param":null,"code":null}}}}
                , .{ep.msg}) catch {
                    logRequestDone(method, path, 405, elapsedMs(request_start));
                    return;
                };
            const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAllow: {s}\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: close\r\n\r\n", .{ body.len, ep.allow, log_request_id, corsHeaders() }) catch return;
            stream.writeAll(hdr) catch |err| {
                std.log.warn("req={d} 405 write failed: {}", .{ log_request_id, err });
                return;
            };
            stream.writeAll(body) catch |err| {
                std.log.warn("req={d} 405 write failed (body): {}", .{ log_request_id, err });
                return;
            };
            g_server.metrics.recordFailure();
            logRequestDone(method, path, 405, elapsedMs(request_start));
            return;
        }
    }

    logRequest(method, path);
    g_server.metrics.recordRequest();
    sendJsonError(stream, "404 Not Found", "invalid_request_error", "Unknown endpoint");
    g_server.metrics.recordFailure();
    logRequestDone(method, path, 404, elapsedMs(request_start));
}

/// Thread-local buffer for `/model` command response formatting.
threadlocal var cmd_buf: [cmd_buf_size]u8 = undefined;

fn handleChatCommand(cmd: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, cmd, "/clear")) {
        {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);
            g_server.model.resetCache();
            g_server.kv_valid = false;
            if (g_server.getActiveConv()) |conv| conv.clearMessages(g_server.allocator);
        }
        slog("  [command] /clear\n", .{});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">Conversation cleared.</div>";
    }
    if (std.mem.eql(u8, cmd, "/reset")) {
        {
            g_server.mutex.lockUncancelable(g_server.io);
            defer g_server.mutex.unlock(g_server.io);
            g_server.model.resetCache();
            g_server.kv_valid = false;
            if (g_server.getActiveConv()) |conv| conv.clearMessages(g_server.allocator);
        }
        slog("  [command] /reset\n", .{});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">Conversation cleared.</div>";
    }
    if (std.mem.eql(u8, cmd, "/model")) {
        slog("  [command] /model\n", .{});
        const escaped_name = json.htmlEscape(g_server.allocator, g_server.model_name) catch g_server.model_name;
        defer if (escaped_name.ptr != g_server.model_name.ptr) g_server.allocator.free(escaped_name);
        return std.fmt.bufPrint(&cmd_buf,
            \\<div class="msg assistant" data-tokens="0" data-time="0" data-tps="0">Model: {s}</div>
        , .{escaped_name}) catch null;
    }
    if (std.mem.eql(u8, cmd, "/help")) {
        slog("  [command] /help\n", .{});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">/clear &mdash; Clear conversation and KV cache&lt;br&gt;/stats &mdash; Toggle generation statistics&lt;br&gt;/model &mdash; Show model name&lt;br&gt;/help &mdash; Show available commands</div>";
    }
    return null;
}

/// Process a base64-encoded image from a web UI form submission.
///
/// Decodes the URL-encoded base64 data, base64-decodes to raw bytes,
/// detects the image format, decodes to RGB pixels, resizes to
/// the vision encoder's expected input size, and encodes into visual
/// token embeddings. Sets the embeddings on the model so the next
/// forward pass injects visual tokens.
///
/// Returns true on success, false on any decode/encode failure.
fn processVisionImage(b64_raw: []const u8, ve: *VisionEncoder) bool {
    const allocator = g_server.allocator;
    slog("  vision: processing image ({d} bytes base64)\n", .{b64_raw.len});

    // URL-decode the base64 data (form fields encode +, /, = as %2B, %2F, %3D)
    const url_decoded = json.urlDecode(allocator, b64_raw) catch |err| {
        slog("  vision: URL decode failed: {}\n", .{err});
        return false;
    };
    defer allocator.free(url_decoded);
    slog("  vision: URL decoded ({d} bytes)\n", .{url_decoded.len});

    // Base64 decode to raw image bytes
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(url_decoded) catch |err| {
        slog("  vision: base64 size calc failed: {}\n", .{err});
        return false;
    };
    const image_bytes = allocator.alloc(u8, decoded_size) catch |err| {
        slog("  vision: alloc failed for {d} bytes: {}\n", .{ decoded_size, err });
        return false;
    };
    defer allocator.free(image_bytes);
    std.base64.standard.Decoder.decode(image_bytes, url_decoded) catch |err| {
        slog("  vision: base64 decode failed: {}\n", .{err});
        return false;
    };
    slog("  vision: decoded image ({d} bytes)\n", .{image_bytes.len});

    // Detect format and decode to RGB pixels
    const format = image_mod.detectFormat(image_bytes);
    switch (format) {
        .png => {
            slog("  vision: PNG detected, decoding...\n", .{});
            var png = image_mod.decodePng(allocator, image_bytes) catch |err| {
                slog("  vision: PNG decode failed: {}\n", .{err});
                return false;
            };
            defer png.deinit();
            slog("  vision: decoded {d}x{d} PNG\n", .{ png.width, png.height });

            // Resize to vision encoder's expected input size
            const target = ve.image_size;
            const resized = image_mod.resize(allocator, png.pixels, png.width, png.height, target, target) catch |err| {
                slog("  vision: resize failed: {}\n", .{err});
                return false;
            };
            defer allocator.free(resized);

            // Encode into visual token embeddings
            slog("  vision: running encoder...\n", .{});
            const visual_tokens = ve.encode(resized) catch |err| {
                slog("  vision: encode failed: {}\n", .{err});
                return false;
            };
            if (ve.projection_dim == 0) {
                slog("  vision: projection_dim is 0\n", .{});
                return false;
            }
            const n_vis: u32 = @intCast(visual_tokens.len / ve.projection_dim);
            g_server.model.setImageEmbeddings(visual_tokens, n_vis, g_server.image_pad_token_id);
            g_server.pending_visual_tokens = n_vis;
            slog("  vision: encoded {d} visual tokens\n", .{n_vis});
            return true;
        },
        .jpeg => {
            slog("  vision: JPEG images not supported — convert to PNG\n", .{});
            return false;
        },
        else => {
            slog("  vision: unrecognized image format (first bytes: {x:0>2}{x:0>2}{x:0>2}{x:0>2})\n", .{
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
    pub fn deinit(self: GeneratedEscaped) void {
        if (self.owns_escaped) g_server.allocator.free(@constCast(self.escaped));
        g_server.allocator.free(self.raw);
    }
};

/// Store a generation result in the active conversation under mutex.
/// Sets kv_valid, logs generation stats, and appends the trimmed response
/// as an assistant message. Used by both streaming and non-streaming chat/regen paths.
fn storeConversationResponse(result_data: []const u8, stats: Stats) void {
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
        g_server.allocator.free(duped);
        return;
    };
    conv.messages.append(g_server.allocator, .{ .role = .assistant, .content = duped }) catch {
        std.log.warn("req={d} OOM appending response to conversation", .{log_request_id});
        g_server.allocator.free(duped);
    };
}

fn generateEscapedN(prompt: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams) GeneratedEscaped {
    const result = generateN(prompt, reset, max_tokens, sampling);
    logGeneration(result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec);
    const escaped = json.jsonEscape(g_server.allocator, result.data) catch |err| blk: {
        std.log.err("req={d} JSON escape OOM ({d} bytes generated): {}", .{ log_request_id, result.data.len, err });
        break :blk result.data[0..0];
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
    const tok = g_server.tokenizer;
    const zero_stats = Stats.zero;
    const raw_token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
    };
    defer g_server.allocator.free(raw_token_ids);

    // Inject image placeholder tokens if visual embeddings are pending.
    // The model's forward() checks for pad_token_id and replaces those
    // embeddings with visual data, so the tokenized prompt must contain
    // the [start, pad*N, end] sequence.
    var injected_ids: ?[]u32 = null;
    if (g_server.pending_visual_tokens > 0 and g_server.image_pad_token_id != 0) {
        const img_toks = ImageTokens{
            .start = g_server.image_start_token_id,
            .end = g_server.image_end_token_id,
            .pad = g_server.image_pad_token_id,
        };
        // Find correct insertion point: after the user prefix tokens.
        // Tokenize the user prefix to find where the user content starts.
        const prefix_tokens = tok.encode(g_server.chat_template.user_prefix) catch &[_]u32{};
        defer if (prefix_tokens.len > 0) g_server.allocator.free(prefix_tokens);
        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(raw_token_ids, prefix_tokens);
        injected_ids = chat_tmpl_mod.injectImageTokens(
            g_server.allocator,
            raw_token_ids,
            insert_pos,
            img_toks,
            g_server.pending_visual_tokens,
        ) catch |err| blk: {
            std.log.warn("req={d} image token injection failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    }
    defer if (injected_ids) |ids| g_server.allocator.free(ids);
    var token_ids: []const u32 = if (injected_ids) |ids| ids else raw_token_ids;
    // Apply truncation_side: if prompt exceeds ctx_size, trim left or right.
    if (g_server.ctx_size > 0 and token_ids.len > @as(usize, g_server.ctx_size)) {
        const limit = @as(usize, g_server.ctx_size);
        token_ids = switch (sampling.truncation_side) {
            .right => token_ids[0..limit], // keep prefix
            .left => token_ids[token_ids.len - limit ..], // keep suffix (recency)
        };
        std.log.warn("req={d} prompt truncated {d}->{d} tokens ({s} side)", .{
            log_request_id,                     token_ids.len + (if (sampling.truncation_side == .right) 0 else g_server.ctx_size), limit,
            @tagName(sampling.truncation_side),
        });
    }
    const prompt_token_count: u32 = @intCast(token_ids.len);

    // Scheduler path: enqueue and block until complete
    if (g_server.request_manager) |rm| {
        const gen_start = milliTimestamp();
        const req = rm.enqueue(token_ids) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            return .{ .data = g_server.allocator.dupe(u8, "[enqueue error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        // Block until request completes
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            if (req.visible_len.load(.acquire) >= max_tokens) {
                req.is_cancelled.store(true, .release);
                break;
            }
            sleepNs(scheduler_poll_interval_ns);
        }

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const token_count: u32 = req.visible_len.load(.acquire);
        const tokens_per_sec: f32 = tokensPerSec(token_count, time_ms);

        const safe_tokens = req.tokens.items[0..token_count];
        const decoded = tok.decode(safe_tokens) catch |err| d: {
            std.log.warn("req={d} batch decode failed ({d} tokens): {}", .{ log_request_id, safe_tokens.len, err });
            break :d g_server.allocator.dupe(u8, "[decode error]") catch @as([]u8, &.{});
        };

        // Record TTFT from scheduler's per-request prefill timestamp
        if (req.prefill_done_at > 0) {
            const sched_ttft = elapsedBetween(req.enqueued_at, req.prefill_done_at);
            g_server.metrics.recordTTFT(sched_ttft, prompt_token_count);
        }
        g_server.metrics.recordThroughput(token_count, time_ms);
        g_server.metrics.recordTPOT(token_count, time_ms);
        g_server.metrics.recordPromptTokens(prompt_token_count);
        g_server.metrics.recordGenerationTokens(token_count);

        return .{
            .data = decoded,
            .finish_reason = if (req.is_finished.load(.acquire)) "stop" else if (token_count >= max_tokens) "length" else "error",
            .stats = .{
                .tokens_generated = token_count,
                .prompt_tokens = prompt_token_count,
                .time_ms = time_ms,
                .tokens_per_sec = tokens_per_sec,
                .prefill_ms = 0,
                .prefill_tps = 0,
            },
        };
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    const model = g_server.model;

    // Re-check kv_valid under mutex — the caller's `reset` flag may be stale
    // if another thread invalidated the cache (e.g. /clear) between the caller's
    // unlock and this lock acquisition.
    const actual_reset = reset or !g_server.kv_valid;

    // Prompt prefix caching: find longest common prefix with cached state.
    // If the new prompt shares a prefix with the previous one, skip re-prefilling
    // those tokens (KV cache already has them). Roll back KV to the shared prefix
    // length and only process new tokens.
    var prefix_len: usize = 0;
    if (actual_reset and g_server.cached_prompt_ids.len > 0 and token_ids.len > 0) {
        const max_match = @min(g_server.cached_prompt_ids.len, token_ids.len);
        while (prefix_len < max_match and g_server.cached_prompt_ids[prefix_len] == token_ids[prefix_len]) {
            prefix_len += 1;
        }
        if (prefix_len > 0 and prefix_len < token_ids.len) {
            // Shared prefix found — rollback KV cache to prefix boundary
            const bos_offset: usize = if (g_server.bos_token_id > 0) 1 else 0;
            model.setKvSeqLen(prefix_len + bos_offset);
            slog("  Prefix cache hit: {d}/{d} tokens reused\n", .{ prefix_len, token_ids.len });
        } else {
            prefix_len = 0;
        }
    }

    if (prefix_len == 0 and actual_reset) model.resetCache();

    // BOS token — required by models like Gemma to initialize state correctly
    if (prefix_len == 0 and actual_reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
            return .{ .data = g_server.allocator.dupe(u8, "[BOS forward error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }

    // Prefill phase — timed separately for TTFT stats.
    // Skip tokens already in KV cache (prefix_len > 0 = cache hit).
    const prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids[prefix_len..]) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                return .{ .data = g_server.allocator.dupe(u8, "[cancelled]") catch &.{}, .finish_reason = "error", .stats = .{ .tokens_generated = 0, .prompt_tokens = prompt_token_count, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 } };
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
            return .{ .data = g_server.allocator.dupe(u8, "[prefill error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
    }

    // Cache the prompt token IDs for next request's prefix matching
    if (g_server.cached_prompt_ids.len > 0) g_server.allocator.free(g_server.cached_prompt_ids);
    g_server.cached_prompt_ids = g_server.allocator.dupe(u32, token_ids) catch &.{};
    const prefill_ms: u64 = elapsedMs(prefill_start);
    const prefill_tps: f32 = tokensPerSec(prompt_token_count, prefill_ms);
    g_server.metrics.recordTTFT(prefill_ms, prompt_token_count);

    // Apply sampling to first generated token (from prefill's last forward call)
    const use_sampling = sampling.temperature > 0;
    const prng_seed = sampling.seed orelse @as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp()))));
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
            std.log.warn("req={d} json_schema grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (grammar_storage) |*g| grammar_state_storage = g.initState() catch |err| blk: {
            std.log.warn("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    } else if (sampling.grammar_string) |gs| {
        grammar_storage = grammar_mod.Grammar.parse(g_server.allocator, gs) catch |err| blk: {
            std.log.warn("req={d} grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (grammar_storage) |*g| grammar_state_storage = g.initState() catch |err| blk: {
            std.log.warn("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    }

    const vocab_texts = g_server.tokenizer.getVocabTexts();

    if (token_ids.len > 0) {
        const first_logits = model.getLogits();

        // Grammar masking on first token
        if (use_grammar) {
            if (grammar_storage) |*g| {
                if (grammar_state_storage) |*gs| {
                    g.maskLogits(gs, first_logits, vocab_texts);
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
            first_gen_token = math_ops.sampleToken(first_logits, sampling.temperature, sampling.top_k, sampling.top_p, prng.random());
        } else if (sampling.json_mode or use_grammar) {
            first_gen_token = math_ops.argmax(first_logits);
        }
        // Accept first token in grammar state
        if (use_grammar and grammar_state_storage != null) {
            const tok_slice = [1]u32{first_gen_token};
            const text = g_server.tokenizer.decode(@constCast(&tok_slice)) catch |err| blk: {
                std.log.warn("req={d} grammar token decode failed (id={d}): {}", .{ log_request_id, first_gen_token, err });
                break :blk null;
            };
            defer if (text) |t| g_server.allocator.free(t);
            grammar_state_storage.?.acceptToken(text orelse "");
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

    // Generation phase (timed) — collect token IDs, batch-decode once at the end
    // to avoid per-token alloc/free overhead.
    const gen_start = milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;
    var cancelled = false;
    var g_in_think_block: bool = false;
    var g_n_think_tokens: u32 = 0;

    // Include first generated token (from last prefill forward)
    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
    }

    var hit_eog = first_is_eog;
    var forward_failed = false;
    const effective_max = @min(max_tokens, gen_ids_buf_size);

    // Rolling text buffer for stop sequence matching — avoids re-decoding
    // a window of tokens each iteration. 128 bytes covers stop sequences
    // that span up to ~32 tokens.
    const stop_buf_size: usize = 128;
    var stop_text_buf: [stop_buf_size]u8 = undefined;
    var stop_text_len: usize = 0;

    const has_draft = g_server.draft_model != null and !first_is_eog and token_ids.len > 0;
    var spec_state_storage: spec_decode.SpecState = undefined;
    var spec_state_valid = false;
    if (has_draft) {
        spec_state_storage = spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize()) catch spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        spec_state_valid = spec_state_storage.k > 0;
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
                spec_decode.verifySampling(spec_state, model, draft_model, last, pre_draft_pos, sampling.temperature, prng.random())
            else
                spec_decode.verifySequential(spec_state, model, draft_model, last, pre_draft_pos);

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
                            const jt_slice = [1]u32{jump_tok};
                            const jt_text = g_server.tokenizer.decode(@constCast(&jt_slice)) catch |err| blk: {
                                std.log.warn("req={d} grammar jump decode failed (id={d}): {}", .{ log_request_id, jump_tok, err });
                                break :blk null;
                            };
                            defer if (jt_text) |t| g_server.allocator.free(t);
                            gs.acceptToken(jt_text orelse "");
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
                        g.maskLogits(gs, logits, vocab_texts);
                        next = math_ops.argmax(logits);
                    }
                }
            }
            if (sampling.logit_bias_count > 0) {
                math_ops.applyLogitBias(logits, &sampling.logit_bias_ids, &sampling.logit_bias_vals, sampling.logit_bias_count);
            }
            // Thinking budget: bias towards </think> token when inside think block and budget exceeded.
            if (sampling.thinking_budget_tokens > 0 and g_n_think_tokens >= sampling.thinking_budget_tokens and g_in_think_block) {
                const close_think = "</think>";
                const close_ids = g_server.tokenizer.encode(close_think) catch null;
                if (close_ids) |ids| {
                    defer g_server.allocator.free(ids);
                    for (ids) |tid| if (tid < @as(u32, @intCast(logits.len))) {
                        logits[tid] += 100.0;
                    };
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
            const tok_text_alloc: ?[]u8 = if (needs_text) blk: {
                const tok_slice = [1]u32{next};
                break :blk g_server.tokenizer.decode(@constCast(&tok_slice)) catch |err| blk2: {
                    std.log.warn("req={d} token decode failed (id={d}): {}", .{ log_request_id, next, err });
                    break :blk2 null;
                };
            } else null;
            defer if (tok_text_alloc) |t| g_server.allocator.free(t);
            const tok_text: []const u8 = tok_text_alloc orelse "";
            // Accept token in grammar state
            if (use_grammar and grammar_state_storage != null) {
                grammar_state_storage.?.acceptToken(tok_text);
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
            // Update thinking block state for budget tracking.
            if (sampling.thinking_budget_tokens > 0) {
                const tk_sl = [1]u32{next};
                if (g_server.tokenizer.decode(@constCast(&tk_sl)) catch null) |tk_text| {
                    defer g_server.allocator.free(tk_text);
                    if (std.mem.indexOf(u8, tk_text, "<think>") != null) g_in_think_block = true;
                    if (std.mem.indexOf(u8, tk_text, "</think>") != null) g_in_think_block = false;
                    if (g_in_think_block) g_n_think_tokens += 1;
                }
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

    // Single batch decode — one alloc instead of N per-token allocs
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
fn chatStreamGenerate(stream: TcpStream, formatted: []const u8, reset: bool, max_tokens: usize, sampling: SamplingParams) GenResult {
    const tok = g_server.tokenizer;
    const zero_stats = Stats.zero;
    const raw_token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} chat stream tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        _ = sseWriteData(stream, "{\"t\":\"[encode error]\",\"done\":true}");
        _ = sseWriteData(stream, "[DONE]");
        return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .finish_reason = "error", .stats = zero_stats };
    };
    defer g_server.allocator.free(raw_token_ids);

    // Inject image placeholder tokens if visual embeddings are pending.
    var injected_ids_cs: ?[]u32 = null;
    if (g_server.pending_visual_tokens > 0 and g_server.image_pad_token_id != 0) {
        const img_toks = ImageTokens{
            .start = g_server.image_start_token_id,
            .end = g_server.image_end_token_id,
            .pad = g_server.image_pad_token_id,
        };
        // Find correct insertion point: after the user prefix tokens.
        const cs_prefix = tok.encode(g_server.chat_template.user_prefix) catch &[_]u32{};
        defer if (cs_prefix.len > 0) g_server.allocator.free(cs_prefix);
        const insert_pos: usize = chat_tmpl_mod.findImageInsertPos(raw_token_ids, cs_prefix);
        injected_ids_cs = chat_tmpl_mod.injectImageTokens(
            g_server.allocator,
            raw_token_ids,
            insert_pos,
            img_toks,
            g_server.pending_visual_tokens,
        ) catch |err| blk: {
            std.log.warn("req={d} vision token injection failed ({d} visual tokens): {}", .{ log_request_id, g_server.pending_visual_tokens, err });
            break :blk null;
        };
    }
    defer if (injected_ids_cs) |ids| g_server.allocator.free(ids);
    const token_ids: []const u32 = if (injected_ids_cs) |ids| ids else raw_token_ids;
    const prompt_token_count: u32 = @intCast(token_ids.len);

    // Scheduler path: enqueue and poll for streamed tokens
    if (g_server.request_manager) |rm| {
        const gen_start = milliTimestamp();
        const req = rm.enqueue(token_ids) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .finish_reason = "error", .stats = zero_stats };
        };
        defer {
            while (!req.scheduler_done.load(.acquire))
                sleepNs(scheduler_poll_interval_ns);
            req.deinit();
            g_server.allocator.destroy(req);
        }

        var streamed_count: usize = 0;
        var client_connected = true;
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            const current_len = req.visible_len.load(.acquire);
            while (streamed_count < current_len) {
                if (!streamToken(stream, tok, req.tokens.items[streamed_count])) {
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
        // Drain remaining tokens (request is finished — scheduler no longer appending)
        const final_len = req.visible_len.load(.acquire);
        while (client_connected and streamed_count < final_len and streamed_count < max_tokens) {
            if (!streamToken(stream, tok, req.tokens.items[streamed_count])) break;
            streamed_count += 1;
        }

        const gen_end = milliTimestamp();
        const time_ms = elapsedBetween(gen_start, gen_end);
        const token_count: u32 = req.visible_len.load(.acquire);
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
        g_server.metrics.recordTPOT(token_count, time_ms);
        g_server.metrics.recordPromptTokens(prompt_token_count);
        g_server.metrics.recordGenerationTokens(token_count);
        if (!client_connected) g_server.metrics.recordCancellation() else if (req.is_finished.load(.acquire)) g_server.metrics.recordCompletion() else g_server.metrics.recordFailure();

        const sched_prefill_tps: f32 = tokensPerSec(prompt_token_count, sched_prefill_ms);
        const safe_cs_tokens = req.tokens.items[0..@min(token_count, @as(u32, @intCast(req.tokens.items.len)))];
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
    const model = g_server.model;
    // Re-check kv_valid under mutex — caller's `reset` may be stale if another
    // thread invalidated the cache between the caller's unlock and this lock.
    const actual_reset = reset or !g_server.kv_valid;
    if (actual_reset) model.resetCache();

    if (actual_reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("req={d} BOS forward failed: {}", .{ log_request_id, err });
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
                g_server.metrics.recordCancellation();
                _ = sseWriteData(stream, "[DONE]");
                return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .finish_reason = "error", .stats = zero_stats };
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
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
    var prng_cs = std.Random.Xoshiro256.init(@as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp())))));
    if (use_sampling and token_ids.len > 0) {
        first_gen_token = math_ops.sampleToken(model.getLogits(), sampling.temperature, sampling.top_k, sampling.top_p, prng_cs.random());
    }

    // Generate and stream tokens
    const gen_start = milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    var client_disconnected = false;
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
        // Stream first token
        client_disconnected = !streamToken(stream, tok, first_gen_token);
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
            next = math_ops.sampleToken(model.getLogits(), sampling.temperature, sampling.top_k, sampling.top_p, prng_cs.random());
        }
        if (g_server.isEog(next)) break;
        gen_tokens[token_count] = next;
        last = next;
        token_count += 1;
        if (!streamToken(stream, tok, next)) {
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

/// Decoded and JSON-escaped token text. Call `deinit()` to release memory.
const EscapedToken = struct {
    decoded: []u8,
    escaped: []u8,

    fn deinit(self: EscapedToken) void {
        if (self.escaped.ptr != self.decoded.ptr) g_server.allocator.free(self.escaped);
        g_server.allocator.free(self.decoded);
    }
};

/// Decode a token ID and JSON-escape its text. Returns null on decode failure or empty output.
fn decodeAndEscape(tok: *Tokenizer, token_id: u32) ?EscapedToken {
    const decoded = tok.decode(&[_]u32{token_id}) catch |err| {
        std.log.warn("req={d} stream decode failed (token_id={d}): {}", .{ log_request_id, token_id, err });
        return null;
    };
    if (decoded.len == 0) {
        g_server.allocator.free(decoded);
        return null;
    }
    const escaped = json.jsonEscape(g_server.allocator, decoded) catch {
        g_server.allocator.free(decoded);
        return null;
    };
    return .{ .decoded = decoded, .escaped = escaped };
}

/// Stream a single decoded token as an SSE event.
/// Returns false if the write failed (client disconnected).
fn streamToken(stream: TcpStream, tok: *Tokenizer, token_id: u32) bool {
    const dt = decodeAndEscape(tok, token_id) orelse return true;
    defer dt.deinit();

    var buf: [sse_event_buf_size]u8 = undefined;
    const event = std.fmt.bufPrint(&buf, "data: {{\"t\":\"{s}\"}}\n\n", .{dt.escaped}) catch {
        std.log.warn("SSE token event exceeded buffer ({d} bytes escaped)", .{dt.escaped.len});
        return true;
    };
    stream.writeAll(event) catch return false;
    return true;
}

// ── Anthropic Messages API helpers ──────────────────────────────

/// Send a JSON error response in Anthropic error format.
/// Message and type are JSON-escaped to prevent injection (CWE-116).
fn sendAnthropicError(stream: TcpStream, status_code: []const u8, err_type: []const u8, message: []const u8) void {
    const escaped_msg = json.jsonEscape(g_server.allocator, message) catch message;
    defer if (escaped_msg.ptr != message.ptr) g_server.allocator.free(escaped_msg);
    const escaped_type = json.jsonEscape(g_server.allocator, err_type) catch err_type;
    defer if (escaped_type.ptr != err_type.ptr) g_server.allocator.free(escaped_type);
    var buf: [error_body_buf_size]u8 = undefined;
    const json_body = std.fmt.bufPrint(&buf,
        \\{{"type":"error","error":{{"type":"{s}","message":"{s}"}}}}
    , .{ escaped_type, escaped_msg }) catch {
        std.log.warn("req={d} anthropic error body overflow type={s}", .{ log_request_id, err_type });
        return;
    };
    const status = if (std.mem.eql(u8, status_code, "400"))
        @as([]const u8, "400 Bad Request")
    else if (std.mem.eql(u8, status_code, "401"))
        @as([]const u8, "401 Unauthorized")
    else if (std.mem.eql(u8, status_code, "404"))
        @as([]const u8, "404 Not Found")
    else if (std.mem.eql(u8, status_code, "429"))
        @as([]const u8, "429 Too Many Requests")
    else if (std.mem.eql(u8, status_code, "503"))
        @as([]const u8, "503 Service Unavailable")
    else
        @as([]const u8, "500 Internal Server Error");
    sendResponse(stream, status, "application/json", json_body);
}

/// Send a 429 Too Many Requests response in Anthropic error format with Retry-After header.
fn sendAnthropic429(stream: TcpStream, retry_after: u32) void {
    g_server.metrics.recordRateLimit();
    var buf: [error_body_buf_size]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"type\":\"error\",\"error\":{{\"type\":\"rate_limit_error\",\"message\":\"Rate limit exceeded. Retry after {d} seconds.\"}}}}", .{retry_after}) catch {
        std.log.warn("req={d} anthropic 429 body format failed", .{log_request_id});
        return;
    };
    var hdr_buf: [hdr_buf_size]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\nX-Request-Id: {d}\r\n{s}" ++ security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after, log_request_id, corsHeaders() }) catch {
        std.log.warn("req={d} anthropic 429 header format failed", .{log_request_id});
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
fn sseWriteEvent(stream: TcpStream, event_type: []const u8, data: []const u8) bool {
    var event_buf: [response_buf_size + 64]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buf, "event: {s}\ndata: {s}\n\n", .{ event_type, data }) catch return false;
    stream.writeAll(event) catch return false;
    return true;
}

/// Start an Anthropic-format SSE streaming response for /v1/messages.
fn startAnthropicStream(stream: TcpStream, formatted: []const u8, max_tokens: usize, input_tokens: u32, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateAnthropicStream(stream, formatted, max_tokens, input_tokens, sampling);
}

/// Run generation and stream tokens as Anthropic-format SSE events.
/// Event sequence: message_start → content_block_start → content_block_delta* →
/// content_block_stop → message_delta → message_stop.
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and polls for generated tokens. Falls back to direct model.forward()
/// when no scheduler is running.
fn generateAnthropicStream(stream: TcpStream, formatted: []const u8, max_tokens: usize, input_tokens: u32, sampling_a: SamplingParams) void {
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

    // Scheduler path: enqueue request and poll for tokens
    if (g_server.request_manager) |rm| {
        const req = rm.enqueue(token_ids) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            sendAnthropicFinalEvents(stream, "end_turn", 0);
            return;
        };
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
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            const current_len = req.visible_len.load(.acquire);
            while (streamed_count < current_len) {
                if (!streamAnthropicDelta(stream, tok, req.tokens.items[streamed_count])) {
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

        // Drain remaining tokens
        const final_len = req.visible_len.load(.acquire);
        while (anth_client_connected and streamed_count < final_len and token_count < max_tokens) {
            if (!streamAnthropicDelta(stream, tok, req.tokens.items[streamed_count])) break;
            streamed_count += 1;
            token_count += 1;
        }

        if (anth_client_connected) {
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
        if (!anth_client_connected) g_server.metrics.recordCancellation() else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else g_server.metrics.recordFailure();
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    const model = g_server.model;
    model.resetCache();

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
    var prng_a = std.Random.Xoshiro256.init(@as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp())))));
    const anth_prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
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
        first_gen_token = math_ops.sampleToken(model.getLogits(), sampling_a.temperature, sampling_a.top_k, sampling_a.top_p, prng_a.random());
    }

    // Generate and stream deltas
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    // Stream first generated token
    var anth_disconnected = false;
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        anth_disconnected = !streamAnthropicDelta(stream, tok, first_gen_token);
        last = first_gen_token;
        token_count = 1;
    }

    var anth_forward_failed = false;

    const a_has_draft = g_server.draft_model != null and token_ids.len > 0 and !(token_count == 0 and g_server.isEog(first_gen_token));
    var a_spec_storage: spec_decode.SpecState = undefined;
    var a_spec_valid = false;
    if (a_has_draft) {
        a_spec_storage = spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize()) catch spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        a_spec_valid = a_spec_storage.k > 0;
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
        while (token_count < max_tokens and !anth_disconnected) {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            const pre = model.kvSeqLen();
            const is_self = (model.ptr == a_draft.ptr);
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
                spec_decode.verifySampling(a_spec, model, a_draft, last, pre, sampling_a.temperature, prng_a.random())
            else
                spec_decode.verifySequential(a_spec, model, a_draft, last, pre);

            for (0..res.accepted) |i| {
                const at = a_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (!streamAnthropicDelta(stream, tok, at)) {
                    anth_disconnected = true;
                    break;
                }
                token_count += 1;
            }
            if (!anth_disconnected and !g_server.isEog(res.next_token)) {
                if (!streamAnthropicDelta(stream, tok, res.next_token)) {
                    anth_disconnected = true;
                }
                token_count += 1;
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
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
                next = math_ops.sampleToken(model.getLogits(), sampling_a.temperature, sampling_a.top_k, sampling_a.top_p, prng_a.random());
            }
            if (g_server.isEog(next)) break;

            if (!streamAnthropicDelta(stream, tok, next)) {
                anth_disconnected = true;
                break;
            }
            last = next;
            token_count += 1;
        }
    }

    if (!anth_disconnected) {
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
fn sendAnthropicFinalEvents(stream: TcpStream, stop_reason: []const u8, token_count: u32) void {
    _ = sseWriteEvent(stream, "content_block_stop",
        \\{"type":"content_block_stop","index":0}
    );
    var delta_buf: [response_buf_size]u8 = undefined;
    const delta = std.fmt.bufPrint(&delta_buf,
        \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":null}},"usage":{{"output_tokens":{d}}}}}
    , .{ stop_reason, token_count }) catch {
        std.log.warn("Anthropic message_delta exceeded buffer", .{});
        return;
    };
    _ = sseWriteEvent(stream, "message_delta", delta);
    _ = sseWriteEvent(stream, "message_stop",
        \\{"type":"message_stop"}
    );
}

/// Stream a single decoded token as an Anthropic content_block_delta SSE event.
/// Returns false if the write failed (client disconnected).
fn streamAnthropicDelta(stream: TcpStream, tok: *Tokenizer, token_id: u32) bool {
    const dt = decodeAndEscape(tok, token_id) orelse return true;
    defer dt.deinit();

    var buf: [sse_event_buf_size]u8 = undefined;
    const data = std.fmt.bufPrint(&buf,
        \\{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{s}"}}}}
    , .{dt.escaped}) catch {
        std.log.warn("Anthropic SSE delta exceeded buffer ({d} bytes escaped)", .{dt.escaped.len});
        return true;
    };
    return sseWriteEvent(stream, "content_block_delta", data);
}

// ── Responses API Streaming ─────────────────────────────────────

/// Start a Responses API SSE streaming response for /v1/responses.
fn startResponsesStream(stream: TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateResponsesStream(stream, prompt, max_tokens, sampling);
}

/// Send the Responses API setup events: response.created, response.output_item.added,
/// response.content_part.added.
fn sendResponsesStartEvents(stream: TcpStream, req_id: u64, created: i64) void {
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

/// Stream a single decoded token as a Responses API output_text.delta event.
/// Returns false if the write failed (client disconnected).
fn streamResponsesDelta(stream: TcpStream, tok: *Tokenizer, token_id: u32) bool {
    const dt = decodeAndEscape(tok, token_id) orelse return true;
    defer dt.deinit();

    var buf: [sse_event_buf_size]u8 = undefined;
    const data = std.fmt.bufPrint(&buf,
        \\{{"type":"response.output_text.delta","item_id":"msg_0","output_index":0,"content_index":0,"delta":"{s}"}}
    , .{dt.escaped}) catch {
        std.log.warn("Responses SSE delta exceeded buffer ({d} bytes escaped)", .{dt.escaped.len});
        return true;
    };
    return sseWriteEvent(stream, "response.output_text.delta", data);
}

/// Send the Responses API final events: output_text.done, content_part.done,
/// output_item.done, response.completed.
fn sendResponsesFinalEvents(stream: TcpStream, req_id: u64, created: i64, stop_reason: []const u8, escaped_text: []const u8, input_tokens: u32, output_tokens: u32) void {
    var buf: [response_buf_size]u8 = undefined;
    const total = input_tokens + output_tokens;

    const text_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.output_text.done","item_id":"msg_0","output_index":0,"content_index":0,"text":"{s}"}}
    , .{escaped_text}) catch {
        std.log.warn("Responses output_text.done exceeded buffer ({d} bytes text)", .{escaped_text.len});
        return;
    };
    _ = sseWriteEvent(stream, "response.output_text.done", text_done);

    const part_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.content_part.done","item_id":"msg_0","output_index":0,"content_index":0,"part":{{"type":"output_text","text":"{s}"}}}}
    , .{escaped_text}) catch {
        std.log.warn("Responses content_part.done exceeded buffer ({d} bytes text)", .{escaped_text.len});
        return;
    };
    _ = sseWriteEvent(stream, "response.content_part.done", part_done);

    const item_done = std.fmt.bufPrint(&buf,
        \\{{"type":"response.output_item.done","output_index":0,"item":{{"type":"message","id":"msg_0","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}}}
    , .{escaped_text}) catch {
        std.log.warn("Responses output_item.done exceeded buffer ({d} bytes text)", .{escaped_text.len});
        return;
    };
    _ = sseWriteEvent(stream, "response.output_item.done", item_done);

    const completed = std.fmt.bufPrint(&buf,
        \\{{"type":"response.completed","response":{{"id":"resp-{d}","object":"response","created_at":{d},"status":"completed","model":"{s}","stop_reason":"{s}","output":[{{"type":"message","id":"msg_0","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"{s}"}}]}}],"usage":{{"input_tokens":{d},"output_tokens":{d},"total_tokens":{d}}}}}}}
    , .{ req_id, created, g_server.model_name, stop_reason, escaped_text, input_tokens, output_tokens, total }) catch {
        std.log.warn("Responses completed event exceeded buffer ({d} bytes text)", .{escaped_text.len});
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
fn generateResponsesStream(stream: TcpStream, prompt: []const u8, max_tokens: usize, sampling_r: SamplingParams) void {
    const tok = g_server.tokenizer;
    const req_id = currentRequestId();
    const created = timestamp();

    const formatted = g_server.chat_template.format(g_server.allocator, null, prompt) catch prompt;
    defer if (formatted.ptr != prompt.ptr) g_server.allocator.free(formatted);
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} responses streaming tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        sendResponsesStartEvents(stream, req_id, created);
        sendResponsesFinalEvents(stream, req_id, created, "stop", "", 0, 0);
        return;
    };
    defer g_server.allocator.free(token_ids);
    const input_tokens: u32 = @intCast(token_ids.len);

    // Send setup events
    sendResponsesStartEvents(stream, req_id, created);

    // Scheduler path: enqueue request and poll for tokens
    if (g_server.request_manager) |rm| {
        const req = rm.enqueue(token_ids) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            sendResponsesFinalEvents(stream, req_id, created, "stop", "", input_tokens, 0);
            return;
        };
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
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            const current_len = req.visible_len.load(.acquire);
            while (streamed_count < current_len) {
                if (!streamResponsesDelta(stream, tok, req.tokens.items[streamed_count])) {
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

        // Drain remaining tokens
        const final_len = req.visible_len.load(.acquire);
        while (resp_client_connected and streamed_count < final_len and token_count < max_tokens) {
            if (!streamResponsesDelta(stream, tok, req.tokens.items[streamed_count])) break;
            streamed_count += 1;
            token_count += 1;
        }

        // Send final events — skip if client already disconnected
        if (resp_client_connected) {
            const safe_resp_count = req.visible_len.load(.acquire);
            const safe_resp_tokens = req.tokens.items[0..@min(safe_resp_count, @as(u32, @intCast(req.tokens.items.len)))];
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
        } else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else g_server.metrics.recordFailure();
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    const model = g_server.model;
    model.resetCache();

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
    var prng_r = std.Random.Xoshiro256.init(@as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp())))));
    const resp_prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
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
        first_gen_token = math_ops.sampleToken(model.getLogits(), sampling_r.temperature, sampling_r.top_k, sampling_r.top_p, prng_r.random());
    }

    // Generate and stream deltas
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;

    var resp_disconnected = false;
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        resp_disconnected = !streamResponsesDelta(stream, tok, first_gen_token);
        gen_tokens[0] = first_gen_token;
        last = first_gen_token;
        token_count = 1;
    }

    var resp_forward_failed = false;

    const r_has_draft = g_server.draft_model != null and token_ids.len > 0 and !(token_count == 0 and g_server.isEog(first_gen_token));
    var r_spec_storage: spec_decode.SpecState = undefined;
    var r_spec_valid = false;
    if (r_has_draft) {
        r_spec_storage = spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize()) catch spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        r_spec_valid = r_spec_storage.k > 0;
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
                spec_decode.verifySampling(r_spec, model, r_draft, last, pre, sampling_r.temperature, prng_r.random())
            else
                spec_decode.verifySequential(r_spec, model, r_draft, last, pre);

            for (0..res.accepted) |i| {
                const at = r_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (!streamResponsesDelta(stream, tok, at)) {
                    resp_disconnected = true;
                    break;
                }
                if (token_count < gen_ids_buf_size) gen_tokens[token_count] = at;
                token_count += 1;
            }
            if (!resp_disconnected and !g_server.isEog(res.next_token)) {
                if (!streamResponsesDelta(stream, tok, res.next_token)) {
                    resp_disconnected = true;
                }
                if (token_count < gen_ids_buf_size) gen_tokens[token_count] = res.next_token;
                token_count += 1;
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
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
                next = math_ops.sampleToken(model.getLogits(), sampling_r.temperature, sampling_r.top_k, sampling_r.top_p, prng_r.random());
            }
            if (g_server.isEog(next)) break;

            if (!streamResponsesDelta(stream, tok, next)) {
                resp_disconnected = true;
                break;
            }
            gen_tokens[token_count] = next;
            last = next;
            token_count += 1;
        }
    }

    // Send final events — skip if client already disconnected
    if (!resp_disconnected) {
        const decoded = tok.decode(gen_tokens[0..token_count]) catch |err| d: {
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
fn sseWriteData(stream: TcpStream, data: []const u8) bool {
    var event_buf: [response_buf_size + 16]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buf, "data: {s}\n\n", .{data}) catch return false;
    stream.writeAll(event) catch return false;
    return true;
}

/// Start an SSE streaming response. Writes headers, generates tokens inline,
/// and writes each as an SSE frame. Runs synchronously on the handler thread.
fn startStream(stream: TcpStream, prompt: []const u8, is_chat: bool, format_prompt: bool, max_tokens: usize, sampling: SamplingParams) void {
    if (!sendSseHeaders(stream)) {
        g_server.metrics.recordCancellation();
        return;
    }
    generateStream(stream, prompt, currentRequestId(), timestamp(), is_chat, format_prompt, max_tokens, sampling);
}

/// Streaming with tool call support. Generates full output first, then emits
/// tool_calls delta chunks if tool calls detected, otherwise streams content.
fn startStreamWithTools(stream: TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams, tp: *const json.ToolParams) void {
    _ = tp;
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

    if (hasToolCalls(gen.raw)) {
        // Emit tool calls as delta chunks
        const tc_start_tag = "<tool_call>";
        const tc_end_tag = "</tool_call>";
        var search_pos: usize = 0;
        var call_idx: usize = 0;

        while (search_pos < gen.raw.len) {
            const tc_start = std.mem.indexOfPos(u8, gen.raw, search_pos, tc_start_tag) orelse break;
            const json_start = tc_start + tc_start_tag.len;
            const tc_end = std.mem.indexOfPos(u8, gen.raw, json_start, tc_end_tag) orelse break;
            const tc_json = gen.raw[json_start..tc_end];
            search_pos = tc_end + tc_end_tag.len;

            const name = json.extractField(tc_json, "name") orelse continue;
            const args = json.extractObjectField(tc_json, "arguments") orelse
                (json.extractField(tc_json, "arguments") orelse "{}");
            const escaped_args = json.jsonEscape(g_server.allocator, args) catch {
                std.log.warn("req={d} tool call argument escaping failed (OOM), skipping tool call", .{log_request_id});
                continue;
            };
            defer if (escaped_args.ptr != args.ptr) g_server.allocator.free(escaped_args);

            // First chunk: role + tool call header
            const role_chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","tool_calls":[{{"index":{d},"id":"call_{d}_{d}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]}},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, call_idx, req_id, call_idx, name, escaped_args }) catch continue;
            _ = sseWriteData(stream, role_chunk);
            call_idx += 1;
        }

        // Final chunk with finish_reason
        const finish = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"tool_calls"}}]}}
        , .{ req_id, created, g_server.model_name }) catch "";
        if (finish.len > 0) {
            _ = sseWriteData(stream, finish);
        }
    } else {
        // No tool calls — emit content as single delta chunk
        const chunk = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}]}}
        , .{ req_id, created, g_server.model_name, gen.escaped, gen.finish_reason }) catch "";
        if (chunk.len > 0) {
            _ = sseWriteData(stream, chunk);
        }
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
fn startStreamRaw(stream: TcpStream, prompt: []const u8, max_tokens: usize, sampling: SamplingParams) void {
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

/// Compute logprobs for a token from logits. Returns null if not requested.
fn computeLogprobs(logits: []const f32, token_id: u32, n_top: u32) ?LogprobInfo {
    if (n_top == 0) return null;
    var info: LogprobInfo = .{};
    info.count = math_ops.topLogProbs(logits, n_top, &info.top_ids, &info.top_logprobs);
    info.token_logprob = math_ops.tokenLogProb(logits, token_id);
    return info;
}

/// Format logprobs JSON into buffer. Returns slice or empty on overflow.
fn formatLogprobs(buf: []u8, tok: *Tokenizer, token_text: []const u8, info: LogprobInfo) []const u8 {
    var pos: usize = 0;
    const header = std.fmt.bufPrint(buf, "\"logprobs\":{{\"content\":[{{\"token\":\"{s}\",\"logprob\":{d:.6},\"top_logprobs\":[", .{ token_text, info.token_logprob }) catch return "";
    pos = header.len;
    for (0..info.count) |i| {
        const dt = decodeAndEscape(tok, info.top_ids[i]) orelse continue;
        defer dt.deinit();
        const prefix: []const u8 = if (i > 0) "," else "";
        const entry = std.fmt.bufPrint(buf[pos..], "{s}{{\"token\":\"{s}\",\"logprob\":{d:.6}}}", .{ prefix, dt.escaped, info.top_logprobs[i] }) catch return "";
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
fn streamChunk(stream: TcpStream, chunk_buf: *[response_buf_size]u8, tok: *Tokenizer, token_id: u32, req_id: u64, created: i64, is_chat: bool) bool {
    return streamChunkLogprobs(stream, chunk_buf, tok, token_id, req_id, created, is_chat, null);
}

fn streamChunkLogprobs(stream: TcpStream, chunk_buf: *[response_buf_size]u8, tok: *Tokenizer, token_id: u32, req_id: u64, created: i64, is_chat: bool, lp_info: ?LogprobInfo) bool {
    const dt = decodeAndEscape(tok, token_id) orelse return true;
    defer dt.deinit();

    var lp_buf: [logprob_buf_size]u8 = undefined;
    const lp_json: []const u8 = if (lp_info) |info|
        formatLogprobs(&lp_buf, tok, dt.escaped, info)
    else
        "";
    const has_lp = lp_json.len > 0;

    const chunk = if (is_chat) blk: {
        break :blk if (has_lp)
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},{s},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, dt.escaped, lp_json })
        else
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, dt.escaped });
    } else blk: {
        break :blk if (has_lp)
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,{s},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, dt.escaped, lp_json })
        else
            std.fmt.bufPrint(chunk_buf,
                \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, dt.escaped });
    };

    if (chunk) |c| {
        return sseWriteData(stream, c);
    } else |_| {
        std.log.warn("SSE stream chunk exceeded buffer ({d} bytes escaped)", .{dt.escaped.len});
        return true;
    }
}

/// Send a usage-only SSE chunk (OpenAI streaming format).
/// Emitted after the final chunk and before [DONE] so clients can track token usage.
fn sendUsageChunk(stream: TcpStream, chunk_buf: *[response_buf_size]u8, req_id: u64, created: i64, is_chat: bool, prompt_tokens: u32, completion_tokens: u32) void {
    const total = prompt_tokens + completion_tokens;
    const id_prefix: []const u8 = if (is_chat) "chatcmpl" else "cmpl";
    const obj_type: []const u8 = if (is_chat) "chat.completion.chunk" else "text_completion";
    const chunk = std.fmt.bufPrint(chunk_buf,
        \\{{"id":"{s}-{d}","object":"{s}","created":{d},"model":"{s}","choices":[],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{ id_prefix, req_id, obj_type, created, g_server.model_name, prompt_tokens, completion_tokens, total }) catch {
        std.log.warn("SSE usage chunk exceeded buffer", .{});
        return;
    };
    _ = sseWriteData(stream, chunk);
}

/// Send the final SSE chunk with the given finish_reason ("stop" or "length").
fn sendFinalChunk(stream: TcpStream, chunk_buf: *[response_buf_size]u8, req_id: u64, created: i64, is_chat: bool, finish_reason: []const u8) void {
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
        std.log.warn("SSE final chunk exceeded buffer", .{});
        return;
    };
    _ = sseWriteData(stream, final);
}

/// Run generation and stream tokens as SSE events in OpenAI format.
/// Always resets the cache (completions API requests are stateless).
/// When the scheduler is active, routes through RequestManager.enqueue()
/// and polls for generated tokens. Falls back to direct model.forward()
/// when no scheduler is running (CLI mode).
fn generateStream(stream: TcpStream, prompt: []const u8, req_id: u64, created: i64, is_chat: bool, format_prompt: bool, max_tokens: usize, sampling: SamplingParams) void {
    const tok = g_server.tokenizer;

    const formatted = if (format_prompt)
        g_server.chat_template.format(g_server.allocator, null, prompt) catch prompt
    else
        prompt;
    defer if (format_prompt and formatted.ptr != prompt.ptr) g_server.allocator.free(formatted);
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("req={d} streaming tokenizer encode failed ({d} bytes input): {}", .{ log_request_id, formatted.len, err });
        g_server.metrics.recordFailure();
        _ = sseWriteData(stream, "[DONE]");
        return;
    };
    defer g_server.allocator.free(token_ids);

    // Send initial chunk (role announcement for chat completions)
    var chunk_buf: [response_buf_size]u8 = undefined;
    if (is_chat) {
        const initial = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}
        , .{ req_id, created, g_server.model_name }) catch "";
        if (initial.len > 0) _ = sseWriteData(stream, initial);
    }

    // Scheduler path: enqueue request and poll for tokens
    if (g_server.request_manager) |rm| {
        const req = rm.enqueue(token_ids) catch |err| {
            std.log.warn("req={d} scheduler enqueue failed ({d} tokens): {}", .{ log_request_id, token_ids.len, err });
            g_server.metrics.recordFailure();
            _ = sseWriteData(stream, "[DONE]");
            return;
        };
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
        while (!req.is_finished.load(.acquire) and !req.is_cancelled.load(.acquire)) {
            // Stream any new tokens since last poll
            const current_len = req.visible_len.load(.acquire);
            while (streamed_count < current_len) {
                const token_id = req.tokens.items[streamed_count];
                if (!streamChunk(stream, &chunk_buf, tok, token_id, req_id, created, is_chat)) {
                    chunk_client_connected = false;
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
            if (!chunk_client_connected or token_count >= max_tokens) break;
            sleepNs(scheduler_poll_interval_ns);
        }

        // Drain any remaining tokens after completion
        const final_len = req.visible_len.load(.acquire);
        while (chunk_client_connected and streamed_count < final_len and token_count < max_tokens) {
            const token_id = req.tokens.items[streamed_count];
            if (!streamChunk(stream, &chunk_buf, tok, token_id, req_id, created, is_chat)) break;
            streamed_count += 1;
            token_count += 1;
        }

        // Send final chunk, usage chunk, and [DONE] — skip if client already disconnected
        if (chunk_client_connected) {
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
        if (!chunk_client_connected) g_server.metrics.recordCancellation() else if (req.is_finished.load(.acquire) or token_count >= max_tokens) g_server.metrics.recordCompletion() else g_server.metrics.recordFailure();
        return;
    }

    // Direct forward path (fallback when scheduler is not active)
    g_server.mutex.lockUncancelable(g_server.io);
    defer g_server.mutex.unlock(g_server.io);
    const model = g_server.model;

    // Prompt prefix caching (streaming): reuse KV cache for shared prefix
    var s_prefix_len: usize = 0;
    if (g_server.cached_prompt_ids.len > 0 and token_ids.len > 0) {
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
    if (s_prefix_len == 0) model.resetCache();

    // BOS token — required by models like Gemma to initialize state correctly
    if (s_prefix_len == 0 and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.err("req={d} BOS forward failed: {}", .{ log_request_id, err });
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
            std.log.warn("req={d} json_schema grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (s_grammar) |*g| s_grammar_state = g.initState() catch |err| blk: {
            std.log.warn("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    } else if (sampling.grammar_string) |gs| {
        s_grammar = grammar_mod.Grammar.parse(g_server.allocator, gs) catch |err| blk: {
            std.log.warn("req={d} grammar parse failed: {}", .{ log_request_id, err });
            break :blk null;
        };
        if (s_grammar) |*g| s_grammar_state = g.initState() catch |err| blk: {
            std.log.warn("req={d} grammar state init failed: {}", .{ log_request_id, err });
            break :blk null;
        };
    }

    // Prefill — capture the last forward's return value (first generated token)
    const use_sampling_s = sampling.temperature > 0;
    const prng_seed_s = sampling.seed orelse @as(u64, @truncate(@as(u96, @bitCast(nanoTimestamp()))));
    var prng_s = std.Random.Xoshiro256.init(prng_seed_s);
    var mirostat_mu_s: f32 = sampling.mirostat_tau * 2.0;
    const prefill_start = milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids[s_prefix_len..]) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                g_server.metrics.recordCancellation();
                _ = sseWriteData(stream, "[DONE]");
                return;
            }
            std.log.warn("req={d} prefill forward failed: {}", .{ log_request_id, err });
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
                g.maskLogits(gs, s_first_logits, s_vocab_texts);
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
    // Accept first token in grammar
    if (use_grammar_s and s_grammar_state != null and token_ids.len > 0) {
        const ft_slice = [1]u32{first_gen_token};
        const ft_text = g_server.tokenizer.decode(@constCast(&ft_slice)) catch |err| blk: {
            std.log.warn("req={d} stream grammar decode failed (id={d}): {}", .{ log_request_id, first_gen_token, err });
            break :blk null;
        };
        defer if (ft_text) |t| g_server.allocator.free(t);
        s_grammar_state.?.acceptToken(ft_text orelse "");
    }

    // Generate and stream tokens
    const gen_start = milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    // Stream the first generated token (from last prefill forward)
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        if (!streamChunk(stream, &chunk_buf, tok, first_gen_token, req_id, created, is_chat)) {
            logGeneration(0, 0, 0);
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
        s_spec_storage = spec_decode.SpecState.init(g_server.allocator, g_server.spec_tokens, model.vocabSize()) catch spec_decode.SpecState{ .k = 0, .vocab_size = 0 };
        s_spec_valid = s_spec_storage.k > 0;
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
        while (token_count < max_tokens and !stream_disconnected) {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
            const pre = model.kvSeqLen();
            const is_self = (model.ptr == s_draft.ptr);
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
                spec_decode.verifySampling(s_spec, model, s_draft, last, pre, sampling.temperature, prng_s.random())
            else
                spec_decode.verifySequential(s_spec, model, s_draft, last, pre);

            for (0..res.accepted) |i| {
                const at = s_spec.draft_tokens[i];
                if (g_server.isEog(at)) break;
                if (!streamChunk(stream, &chunk_buf, tok, at, req_id, created, is_chat)) {
                    stream_disconnected = true;
                    break;
                }
                token_count += 1;
            }
            if (!stream_disconnected and !g_server.isEog(res.next_token)) {
                if (!streamChunk(stream, &chunk_buf, tok, res.next_token, req_id, created, is_chat)) {
                    stream_disconnected = true;
                }
                token_count += 1;
            }
            last = res.next_token;
            if (g_server.isEog(res.next_token)) break;
        }
    } else {
        // Standard streaming — token history for penalty tracking
        const use_penalties_s = sampling.frequency_penalty != 0 or sampling.presence_penalty != 0 or sampling.repetition_penalty != 1.0 or sampling.dry_multiplier > 0;
        var s_gen_tokens: [gen_ids_buf_size]u32 = undefined;
        var s_gen_count: u32 = 0;
        if (use_penalties_s and token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
            s_gen_tokens[0] = first_gen_token;
            s_gen_count = 1;
        }

        var last_token_time: i64 = milliTimestamp();

        // Thinking budget state (Anthropic-style: limit reasoning token count).
        const think_budget = sampling.thinking_budget_tokens;
        var think_token_count: u32 = 0;
        var in_think_block: bool = false;
        // Check if the first token is the start of a thinking block
        if (think_budget > 0 and token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
            const ft_sl = [1]u32{first_gen_token};
            if (g_server.tokenizer.decode(@constCast(&ft_sl)) catch null) |ft_text| {
                defer g_server.allocator.free(ft_text);
                if (std.mem.indexOf(u8, ft_text, "<think>") != null) in_think_block = true;
                if (std.mem.indexOf(u8, ft_text, "</think>") != null) in_think_block = false;
                if (in_think_block) think_token_count += 1;
            }
        }

        for (0..max_tokens -| 1) |_| {
            if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;

            // Jump decoding (streaming): skip forward pass when grammar has single valid token
            if (use_grammar_s and !use_sampling_s) {
                if (s_grammar) |*g| {
                    if (s_grammar_state) |*gs| {
                        if (g.singleValidToken(gs, s_vocab_texts)) |jump_tok| {
                            const jt_s = [1]u32{jump_tok};
                            const jt_text = g_server.tokenizer.decode(@constCast(&jt_s)) catch |err| blk: {
                                std.log.warn("req={d} stream jump decode failed (id={d}): {}", .{ log_request_id, jump_tok, err });
                                break :blk null;
                            };
                            defer if (jt_text) |t| g_server.allocator.free(t);
                            gs.acceptToken(jt_text orelse "");
                            if (!streamChunk(stream, &chunk_buf, tok, jump_tok, req_id, created, is_chat)) {
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
                const close_think = "</think>";
                const close_ids = g_server.tokenizer.encode(close_think) catch null;
                if (close_ids) |ids| {
                    defer g_server.allocator.free(ids);
                    // Apply strong positive bias to all </think> sub-tokens
                    for (ids) |tid| {
                        if (tid < @as(u32, @intCast(s_logits.len))) {
                            s_logits[tid] += 100.0;
                        }
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
                        g.maskLogits(gs, s_logits, s_vocab_texts);
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
            const lp = if (sampling.logprobs) computeLogprobs(s_logits, next, sampling.top_logprobs) else null;

            if (g_server.isEog(next)) break;
            // Accept in grammar
            if (use_grammar_s and s_grammar_state != null) {
                const stok_slice = [1]u32{next};
                const stext = g_server.tokenizer.decode(@constCast(&stok_slice)) catch |err| blk: {
                    std.log.warn("req={d} stream grammar decode failed (id={d}): {}", .{ log_request_id, next, err });
                    break :blk null;
                };
                defer if (stext) |st| g_server.allocator.free(st);
                s_grammar_state.?.acceptToken(stext orelse "");
                if (s_grammar_state.?.isComplete()) {
                    if (!streamChunkLogprobs(stream, &chunk_buf, tok, next, req_id, created, is_chat, lp)) stream_disconnected = true;
                    token_count += 1;
                    break;
                }
            }
            // Stop sequence check (decode token, check trailing text)
            if (sampling.hasStop()) {
                const stok = [1]u32{next};
                const stext = g_server.tokenizer.decode(@constCast(&stok)) catch |err| blk: {
                    std.log.warn("req={d} stop seq decode failed (id={d}): {}", .{ log_request_id, next, err });
                    break :blk null;
                };
                defer if (stext) |st| g_server.allocator.free(st);
                if (stext != null and stext.?.len > 0 and sampling.matchesStop(stext.?)) {
                    token_count += 1;
                    break;
                }
            }
            if (!streamChunkLogprobs(stream, &chunk_buf, tok, next, req_id, created, is_chat, lp)) {
                stream_disconnected = true;
                break;
            }
            if (use_penalties_s and s_gen_count < gen_ids_buf_size) {
                s_gen_tokens[s_gen_count] = next;
                s_gen_count += 1;
            }
            // Update thinking block state for next iteration
            if (think_budget > 0) {
                const tok_sl = [1]u32{next};
                if (g_server.tokenizer.decode(@constCast(&tok_sl)) catch null) |tok_text| {
                    defer g_server.allocator.free(tok_text);
                    if (std.mem.indexOf(u8, tok_text, "<think>") != null) in_think_block = true;
                    if (std.mem.indexOf(u8, tok_text, "</think>") != null) in_think_block = false;
                    if (in_think_block) think_token_count += 1;
                }
            }

            // Record inter-token latency
            const now_itl = milliTimestamp();
            const itl_ms: u64 = @intCast(@max(now_itl - last_token_time, 0));
            g_server.metrics.recordInterTokenLatency(itl_ms);
            last_token_time = now_itl;

            last = next;
            token_count += 1;
        }
    }

    // Send final chunk, usage chunk, and [DONE] — skip if client already disconnected
    if (!stream_disconnected) {
        const direct_finish: []const u8 = if (token_count >= max_tokens) "length" else "stop";
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
    if (stream_disconnected) g_server.metrics.recordCancellation() else if (stream_forward_failed) g_server.metrics.recordFailure() else g_server.metrics.recordCompletion();

    // Update prompt prefix cache for next request
    if (g_server.cached_prompt_ids.len > 0) g_server.allocator.free(g_server.cached_prompt_ids);
    g_server.cached_prompt_ids = g_server.allocator.dupe(u32, token_ids) catch &.{};
}

// JSON field extraction, encoding, and form-parsing utilities are in json.zig.

// ── Connection handler & server entry point ─────────────────────

fn handleConnection(stream: TcpStream) void {
    // Set read/write timeouts to prevent slow loris attacks — without this,
    // stream.read()/writeAll() block indefinitely and an attacker can exhaust
    // all max_concurrent_connections slots with incomplete requests or stalled reads.
    const timeout = std.posix.timeval{ .sec = connection_read_timeout_sec, .usec = 0 };
    std.posix.setsockopt(stream.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("Failed to set connection read timeout: {}", .{err});
    };
    std.posix.setsockopt(stream.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("Failed to set connection write timeout: {}", .{err});
    };
    // Disable Nagle's algorithm — SSE streaming writes small token chunks (~20-100 bytes)
    // that Nagle would buffer for up to 200ms waiting for ACK coalescing.
    const nodelay_val: c_int = 1;
    std.posix.setsockopt(stream.handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&nodelay_val)) catch |err| {
        std.log.warn("Failed to set TCP_NODELAY: {}", .{err});
    };

    // active_connections already incremented by accept loop before thread spawn.
    defer {
        _ = g_server.metrics.active_connections.fetchSub(1, .release);
        stream.close();
    }
    log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
    var buf: [http_buf_size]u8 = undefined;
    switch (readHttpRequest(stream, &buf)) {
        .ok => |req| handleRequest(stream, req),
        .body_too_large => {
            g_server.metrics.recordRequest();
            g_server.metrics.recordFailure();
            const t = getTimeComponents();
            slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} Rejected oversized request body (>{d} bytes) -> 413\n", .{ t.hours, t.minutes, t.seconds, log_request_id, max_request_body_size });
            sendJsonError(stream, "413 Payload Too Large", "invalid_request_error", "Request body too large");
        },
        .malformed => {
            g_server.metrics.recordRequest();
            g_server.metrics.recordFailure();
            const t = getTimeComponents();
            slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} Malformed HTTP request -> 400\n", .{ t.hours, t.minutes, t.seconds, log_request_id });
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Malformed HTTP request");
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
    /// In sleep mode the server suspends CPU-side speculative work and logs
    /// a "sleeping" state in /health. Wake-up happens automatically on next request.
    sleep_after_s: u32 = 0,
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
            if (isUnsafeJsonChar(c)) {
                const buf = allocator.alloc(u8, model_name.len) catch break :blk model_name;
                for (buf, model_name) |*d, sc| {
                    d.* = if (isUnsafeJsonChar(sc)) '_' else sc;
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
        .draft_model = config.draft_model,
        .spec_tokens = config.spec_tokens,
        .tree_budget = config.tree_budget,
        .sleep_after_s = config.sleep_after_s,
    };
    server.api_key = api_key;
    server.start_time = timestamp();
    server.metrics.process_start_time.store(server.start_time, .monotonic);
    g_server = &server;

    // Initialize continuous batching scheduler and background thread.
    // The scheduler owns the model forward loop; HTTP handlers enqueue
    // requests and poll for results instead of calling model.forward() directly.
    var request_manager = try scheduler.RequestManager.init(allocator, &server.metrics, scheduler_max_batch_size, scheduler_timeout_sec, tiered_cache, io);
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
    defer if (sleep_thread) |t| t.join();

    const address = net.IpAddress{ .ip4 = .{ .bytes = host, .port = port } };
    var tcp = net.IpAddress.listen(&address, io, .{ .reuse_address = true }) catch |err| {
        var buf: [error_body_buf_size]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Error: failed to listen on port {d}: {s}\n", .{ port, @errorName(err) }) catch "";
        _ = std.c.write(stderr_file.handle, msg.ptr, msg.len);
        return error.ListenError;
    };

    // Set accept timeout so signal handler can interrupt the loop.
    // Without this, accept() blocks indefinitely and Ctrl+C doesn't work on macOS.
    const timeout = std.posix.timeval{ .sec = accept_timeout_sec, .usec = 0 };
    std.posix.setsockopt(tcp.socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch |err| {
        std.log.warn("Failed to set accept timeout: {}", .{err});
    };
    defer tcp.deinit(io);

    const t = getTimeComponents();
    var buf: [hdr_buf_size]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "\n[{d:0>2}:{d:0>2}:{d:0>2}] agave server started on http://{d}.{d}.{d}.{d}:{d}\n  model={s} backend={s}\n  ctx_size={d} max_conn={d} batch={d} timeout={d}s auth={s} rate_limit={s}\nPress Ctrl+C to stop\n", .{ t.hours, t.minutes, t.seconds, host[0], host[1], host[2], host[3], port, model_name, backend_name, ctx_size, max_concurrent_connections, scheduler_max_batch_size, scheduler_timeout_sec, if (api_key != null) "yes" else "no", if (server.rate_limiter != null) "yes" else "no" }) catch "";
    _ = std.c.write(stdout_file.handle, msg.ptr, msg.len);

    // Install graceful shutdown handlers for SIGTERM and SIGINT.
    // First signal: graceful shutdown (drain active connections).
    // Second signal: immediate process exit.
    const handler = struct {
        fn handle(_: std.posix.SIG) callconv(.c) void {
            if (g_server.shutdown_requested.load(.acquire)) {
                // Second signal — force immediate exit
                const force_msg = "\nForced shutdown.\n";
                _ = std.c.write(stderr_file.handle, force_msg.ptr, force_msg.len);
                std.process.exit(1);
            }
            // First signal — write immediately (async-signal-safe)
            const shutdown_msg = "\nShutting down (Ctrl+C again to force)...\n";
            _ = std.c.write(stderr_file.handle, shutdown_msg.ptr, shutdown_msg.len);
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

    // Accept loop — each connection handled on its own thread.
    // Reject new connections when at the concurrency limit to prevent resource exhaustion.
    // Exit loop when graceful shutdown is requested.
    while (!g_server.shutdown_requested.load(.acquire)) {
        const net_stream = tcp.accept(io) catch |err| {
            if (g_server.shutdown_requested.load(.acquire)) break;
            // Timeout is expected — allows periodic shutdown check
            if (err == error.WouldBlock or err == error.Unexpected) continue;
            std.log.err("Accept failed: {}", .{err});
            continue;
        };
        const stream = TcpStream{ .handle = net_stream.socket.handle };
        // Atomically increment before capacity check to prevent TOCTOU race
        // where multiple accept() calls pass the check before any thread increments.
        const prev = g_server.metrics.active_connections.fetchAdd(1, .acquire);
        if (prev >= max_concurrent_connections) {
            _ = g_server.metrics.active_connections.fetchSub(1, .release);
            log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
            g_server.metrics.recordRequest();
            g_server.metrics.recordConnectionRejection();
            g_server.metrics.recordFailure();
            const tc = getTimeComponents();
            slog("[{d:0>2}:{d:0>2}:{d:0>2}] req={d} Connection rejected: at capacity ({d}/{d}) -> 503\n", .{ tc.hours, tc.minutes, tc.seconds, log_request_id, max_concurrent_connections, max_concurrent_connections });
            sendResponse(stream, "503 Service Unavailable", "application/json", "{\"error\":{\"message\":\"Server at capacity\",\"type\":\"server_error\",\"param\":null,\"code\":\"server_overloaded\"}}");
            stream.close();
            continue;
        }
        const thread = std.Thread.spawn(.{}, handleConnection, .{stream}) catch |err| {
            _ = g_server.metrics.active_connections.fetchSub(1, .release);
            log_request_id = g_server.request_counter.fetchAdd(1, .monotonic);
            g_server.metrics.recordRequest();
            g_server.metrics.recordFailure();
            std.log.err("Failed to spawn connection handler thread: {}", .{err});
            sendResponse(stream, "503 Service Unavailable", "application/json", "{\"error\":{\"message\":\"Server unable to handle request\",\"type\":\"server_error\",\"param\":null,\"code\":\"server_overloaded\"}}");
            stream.close();
            continue;
        };
        thread.detach();
    }

    // Log shutdown (signal handler cannot safely log — do it here)
    {
        const tc = getTimeComponents();
        slog("\n[{d:0>2}:{d:0>2}:{d:0>2}] Server shutting down...\n", .{ tc.hours, tc.minutes, tc.seconds });
    }

    // Stop scheduler thread before draining connections
    server.scheduler_shutdown.store(true, .release);
    if (server.scheduler_thread) |sched_t| sched_t.join();
    server.scheduler_thread = null;

    // Drain active connections (wait up to 30 seconds)
    const drain_timeout_sec: i64 = 30;
    const drain_start = timestamp();
    const active_count = g_server.metrics.active_connections.load(.acquire);
    if (active_count > 0) {
        std.log.info("Draining {d} active connections...", .{active_count});
    }

    while (g_server.metrics.active_connections.load(.acquire) > 0) {
        const elapsed = timestamp() - drain_start;
        if (elapsed > drain_timeout_sec) {
            std.log.warn("Drain timeout after {d}s, forcing shutdown", .{elapsed});
            break;
        }
        sleepNs(drain_poll_interval_ms * std.time.ns_per_ms);
    }

    // Free conversation storage under mutex — handler threads may still
    // be running (drain timeout exceeded) and accessing conversations.
    {
        server.mutex.lockUncancelable(server.io);
        defer server.mutex.unlock(server.io);
        if (server.cached_prompt_ids.len > 0) {
            allocator.free(server.cached_prompt_ids);
            server.cached_prompt_ids = &.{};
        }
        for (server.conversations.items) |*conv| conv.freeMessages(allocator);
        server.conversations.deinit(allocator);
    }

    std.log.info("Graceful shutdown complete", .{});
}

// ── Tests ───────────────────────────────────────────────────────

test "parseContentLength normal" {
    try std.testing.expectEqual(@as(?usize, 42), parseContentLength("Content-Length: 42\r\nHost: localhost"));
}

test "parseContentLength duplicate rejects" {
    try std.testing.expectEqual(@as(?usize, null), parseContentLength("Content-Length: 42\r\nContent-Length: 42"));
}

test "parseContentLength missing header returns zero" {
    try std.testing.expectEqual(@as(?usize, 0), parseContentLength("Host: localhost\r\nAccept: */*"));
}

test "parseContentLength non-numeric rejects" {
    try std.testing.expectEqual(@as(?usize, null), parseContentLength("Content-Length: abc\r\nHost: localhost"));
}

test "parseContentLength empty headers returns zero" {
    try std.testing.expectEqual(@as(?usize, 0), parseContentLength(""));
}

test "parseContentLength case insensitive" {
    try std.testing.expectEqual(@as(?usize, 99), parseContentLength("content-length: 99\r\nHost: x"));
    try std.testing.expectEqual(@as(?usize, 7), parseContentLength("CONTENT-LENGTH: 7\r\nHost: x"));
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
            }

            // ── pub fn run: comptime verify it exists (needs full server, cannot call) ──
            comptime {
                _ = &run;
            }

            // ── pub TcpStream methods: comptime verify (need real socket FDs) ──
            comptime {
                _ = &TcpStream.writeAll;
                _ = &TcpStream.read;
                _ = &TcpStream.close;
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
                std.debug.assert(result <= gen_ids_buf_size);
                // null case
                const null_result = clampMaxTokens(null);
                std.debug.assert(null_result == default_max_gen_tokens);
            }

            // tokensPerSec
            {
                const count = smith.valueWithHash(u16, 0x02);
                const time_ms = smith.valueWithHash(u16, 0x03);
                const tps = tokensPerSec(@intCast(count), @intCast(time_ms));
                std.debug.assert(tps >= 0 or tps == 0);
                std.debug.assert(!std.math.isNan(tps));
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
                const is_unsafe = isUnsafeJsonChar(c);
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
                std.debug.assert(result.len == len);
                // Check no control chars in output (except space)
                for (result) |ch| {
                    std.debug.assert(ch >= 0x20 or ch == ' ');
                    std.debug.assert(ch != 0x7F);
                }
            }

            // hasHeader
            {
                const has_ct = hasHeader("Content-Type: text/html\r\nHost: x", "Content-Type");
                std.debug.assert(has_ct);
                const has_missing = hasHeader("Content-Type: text/html\r\nHost: x", "Authorization");
                std.debug.assert(!has_missing);
                // Case-insensitive
                const has_ci = hasHeader("content-type: text/html\r\nHost: x", "Content-Type");
                std.debug.assert(has_ci);
            }

            // parseContentLength
            {
                const val = smith.valueWithHash(u16, 0x10);
                var hdr_buf: [64]u8 = undefined;
                const hdr = std.fmt.bufPrint(&hdr_buf, "Content-Length: {d}\r\nHost: x", .{val}) catch unreachable;
                const parsed = parseContentLength(hdr);
                std.debug.assert(parsed != null);
                std.debug.assert(parsed.? == @as(usize, val));
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

            // HttpReadResult union: verify layout at comptime
            comptime {
                _ = @as(?HttpReadResult, null);
                _ = HttpReadResult.malformed;
                _ = HttpReadResult.body_too_large;
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
