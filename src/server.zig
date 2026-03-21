//! HTTP server with OpenAI-compatible API endpoints.
//! Provides /v1/chat/completions, /v1/completions, /v1/models, /v1/responses,
//! /v1/conversations, and /v1/chat (built-in web UI). Supports both synchronous
//! JSON responses and SSE streaming. Uses std.net with per-connection threads;
//! inference is mutex-serialized.

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;

const Model = @import("models/model.zig").Model;
const Tokenizer = @import("tokenizer/tokenizer.zig").Tokenizer;
const chat_tmpl_mod = @import("chat_template.zig");
const ChatTemplate = chat_tmpl_mod.ChatTemplate;
const Message = chat_tmpl_mod.Message;
const max_eog_ids = @import("arch.zig").max_eog_ids;
const scheduler = @import("scheduler.zig");
const RateLimiter = @import("rate_limiter.zig").RateLimiter;
const Metrics = @import("metrics.zig").Metrics;

// ── Server constants ────────────────────────────────────────────
const slog_buf_size: usize = 4096;
const models_json_buf_size: usize = 1024;
const response_buf_size: usize = 65536;
const msg_preview_buf_size: usize = 100;
const msg_preview_max_len: usize = 80;
const cmd_buf_size: usize = 1024;
/// Buffer for collecting generated token IDs. Sized to accommodate
/// the default_max_gen_tokens and typical API request max_tokens values.
const gen_ids_buf_size: usize = 4096;
/// Default maximum tokens per server generation request (matches CLI default).
const default_max_gen_tokens: usize = 512;
const extract_field_buf_size: usize = 256;
const conv_title_max_len: usize = 48;
const conv_list_buf_size: usize = 8192;
const conv_msgs_buf_size: usize = 65536;
const http_buf_size: usize = 65536;
/// Maximum HTTP request body size (1 MB).
const max_request_body_size: usize = 1_000_000;
/// Maximum number of concurrent conversations.
const max_conversations: usize = 100;
/// Maximum messages per conversation.
const max_messages_per_conv: usize = 1000;
/// Maximum decoded message length (100 KB).
const max_message_len: usize = 100_000;
/// Maximum concurrent HTTP handler threads.
const max_concurrent_connections: u32 = 64;

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

    fn clearMessages(self: *Conversation, allocator: Allocator) void {
        for (self.messages.items) |msg| allocator.free(@constCast(msg.content));
        self.messages.clearRetainingCapacity();
    }

    fn freeMessages(self: *Conversation, allocator: Allocator) void {
        for (self.messages.items) |msg| allocator.free(@constCast(msg.content));
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
    mutex: std.Thread.Mutex = .{},
    stdout_mutex: std.Thread.Mutex = .{},
    /// Monotonically increasing request counter for unique response IDs.
    request_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Number of currently active HTTP connections (for health/monitoring).
    active_connections: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    /// Server start time (unix timestamp, set once in run()).
    start_time: i64 = 0,
    /// Continuous batching scheduler (null = single-request mode).
    request_manager: ?*scheduler.RequestManager = null,
    /// Per-API-key rate limiter (null = no rate limiting).
    rate_limiter: ?*RateLimiter = null,
    /// API key for authentication (null = no auth).
    api_key: ?[]const u8 = null,
    /// Background scheduler thread (null when not using scheduler).
    scheduler_thread: ?std.Thread = null,
    /// Shutdown signal for scheduler loop.
    scheduler_shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Prometheus metrics collector.
    metrics: Metrics = .{},
    /// Graceful shutdown flag (set by SIGTERM/SIGINT).
    shutdown_requested: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn getActiveConv(self: *Server) ?*Conversation {
        for (self.conversations.items) |*conv| {
            if (conv.id == self.active_id) return conv;
        }
        return null;
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
        self.conversations.append(self.allocator, .{ .id = self.next_id }) catch return null;
        self.active_id = self.next_id;
        self.next_id += 1;
        self.kv_valid = false;
        return &self.conversations.items[self.conversations.items.len - 1];
    }

    /// Delete a conversation by ID. Caller must hold self.mutex.
    fn deleteConv(self: *Server, id: u32) void {
        for (self.conversations.items, 0..) |*conv, i| {
            if (conv.id == id) {
                conv.freeMessages(self.allocator);
                _ = self.conversations.orderedRemove(i);
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

fn slog(comptime fmt: []const u8, args: anytype) void {
    g_server.stdout_mutex.lock();
    defer g_server.stdout_mutex.unlock();
    var buf: [slog_buf_size]u8 = undefined;
    std.fs.File.stdout().writeAll(std.fmt.bufPrint(&buf, fmt, args) catch return) catch {};
}

fn elapsedMs(start: i64) u64 {
    return @intCast(@max(std.time.milliTimestamp() - start, 0));
}

/// Broken-down UTC time for request log timestamps.
const TimeComponents = struct { hours: u64, minutes: u64, seconds: u64 };

fn getTimeComponents() TimeComponents {
    const now = std.time.timestamp();
    return .{
        .hours = @intCast(@mod(@divTrunc(now, 3600), 24)),
        .minutes = @intCast(@mod(@divTrunc(now, 60), 60)),
        .seconds = @intCast(@mod(now, 60)),
    };
}

fn logRequest(method: []const u8, path: []const u8) void {
    const t = getTimeComponents();
    slog("[{d:0>2}:{d:0>2}:{d:0>2}] {s} {s}\n", .{ t.hours, t.minutes, t.seconds, method, path });
}

/// Log completion of a request with status code and duration.
fn logRequestDone(method: []const u8, path: []const u8, status: u16, duration_ms: u64) void {
    const t = getTimeComponents();
    slog("[{d:0>2}:{d:0>2}:{d:0>2}] {s} {s} -> {d} ({d}ms)\n", .{ t.hours, t.minutes, t.seconds, method, path, status, duration_ms });
}

fn logGeneration(tokens: u32, time_ms: u64, tps: f32) void {
    const t = getTimeComponents();
    if (std.posix.isatty(std.fs.File.stdout().handle)) {
        slog("[{d:0>2}:{d:0>2}:{d:0>2}] \x1b[32mGenerated {d} tokens in {d}ms ({d:.2} tok/s)\x1b[0m\n", .{ t.hours, t.minutes, t.seconds, tokens, time_ms, tps });
    } else {
        slog("[{d:0>2}:{d:0>2}:{d:0>2}] Generated {d} tokens in {d}ms ({d:.2} tok/s)\n", .{ t.hours, t.minutes, t.seconds, tokens, time_ms, tps });
    }
}

const html_page =
    \\<!DOCTYPE html><html lang="en"><head>
    \\<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    \\<title>agave</title>
\\<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#127797;</text></svg>">
    \\<link rel="preconnect" href="https://fonts.googleapis.com">
    \\<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    \\<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js" integrity="sha384-zbcZAIxlvJtNE3Dp5nxLXdXtXyxwOdnILY1TDPVmKFhl4r4nSUG1r8bcFXGVa4Te" crossorigin="anonymous"></script>
    \\<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.8/dist/purify.min.js" integrity="sha384-vdScihEZCfbPnBQf+lc7LgXUdJVYyhC3yWHUW5C5P5GpHRqVnaM6HJELJxT6IqwM" crossorigin="anonymous"></script>
    \\<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js" integrity="sha384-F/bZzf7p3Joyp5psL90p/p89AZJsndkSoGwRpXcZhleCWhd8SnRuoYo4d0yirjJp" crossorigin="anonymous"></script>
    \\<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/monokai-sublime.min.css" integrity="sha384-fPmRQ3GXo2iBk8Sgxizmu0IOOv+vAPjYXqCHNut2KSkSBleqjCd7FpBJwrfyCfGc" crossorigin="anonymous">
    \\<style>
    \\:root{--bg:#1a1714;--surface:#231f1c;--elevated:#2c2825;--border:#3d3833;--border-hl:#544e48;--text:#ebe3db;--text-2:#a09890;--text-3:#8a847e;--accent:#d4a574;--accent-hover:#e0b88a;--accent-bg:rgba(212,165,116,0.1);--green:#8faa7b;--red:#c75050;--red-bg:rgba(199,80,80,0.1);--mono:'IBM Plex Mono',ui-monospace,'Cascadia Code',Menlo,monospace;--sans:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;--max-w:860px;--r:8px;--sidebar-w:260px}
    \\*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    \\body{font-family:var(--sans);background:var(--bg);color:var(--text);height:100vh;overflow:hidden;line-height:1.6;display:flex;flex-direction:column}
    \\body::before{content:'';position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 50% -20%,rgba(212,165,116,0.07) 0%,transparent 60%);z-index:0}
    \\.layout{display:flex;flex:1;overflow:hidden;position:relative;z-index:1}
    \\.sidebar{width:var(--sidebar-w);background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;z-index:20}
    \\.new-chat-btn{font-family:var(--mono);font-size:12px;font-weight:500;padding:6px 12px;border-radius:var(--r);border:1px solid var(--accent);background:var(--accent-bg);color:var(--accent);cursor:pointer;transition:all .2s}
    \\.new-chat-btn:hover{background:var(--accent);color:var(--bg)}
    \\.conv-list{flex:1;overflow-y:auto;padding:8px}
    \\.conv-list::-webkit-scrollbar{width:4px}
    \\.conv-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
    \\.conv-item{padding:10px 12px;border-radius:6px;cursor:pointer;display:flex;align-items:center;gap:8px;margin-bottom:2px;transition:background .12s;border:1px solid transparent;outline:none}
\\.conv-item:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
    \\.conv-item:hover{background:var(--elevated)}
    \\.conv-item.active{background:var(--accent-bg);border-color:rgba(212,165,116,0.25)}
    \\.conv-title{flex:1;font-size:13px;color:var(--text-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    \\.conv-item.active .conv-title{color:var(--text)}
    \\.conv-del{background:none;border:none;color:var(--text-3);cursor:pointer;font-size:14px;padding:2px 4px;border-radius:3px;opacity:0;transition:all .15s;flex-shrink:0}
    \\.conv-item:hover .conv-del{opacity:1}
    \\.conv-del:hover{color:var(--red);background:var(--red-bg)}
    \\.conv-empty{padding:20px 12px;text-align:center;color:var(--text-3);font-size:12px;font-family:var(--mono)}
    \\.main{flex:1;display:flex;flex-direction:column;min-width:0}
    \\header{z-index:10;padding:12px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:var(--surface);box-shadow:0 1px 3px rgba(0,0,0,0.2);flex-shrink:0}
    \\.logo{display:flex;align-items:center;gap:10px}
    \\.logo h1{font-family:var(--mono);font-size:18px;font-weight:600;color:var(--accent);letter-spacing:-0.5px}
    \\.model-badge{font-family:var(--mono);font-size:12px;color:var(--text-2);background:var(--elevated);padding:4px 10px;border-radius:var(--r);border:1px solid var(--border)}
    \\.hdr-btns{display:flex;gap:6px}
    \\.hdr-btn{font-family:var(--mono);background:none;border:1px solid var(--border);color:var(--text-3);padding:6px 12px;border-radius:var(--r);cursor:pointer;font-size:12px;transition:all .2s}
    \\.hdr-btn:hover{background:var(--elevated);color:var(--text);border-color:var(--border-hl)}
    \\#chat{position:relative;z-index:1;flex:1;overflow-y:auto;padding:24px 24px 8px;display:flex;flex-direction:column;gap:24px;scroll-behavior:smooth}
    \\#chat::-webkit-scrollbar{width:6px}
    \\#chat::-webkit-scrollbar-track{background:transparent}
    \\#chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    \\#chat::-webkit-scrollbar-thumb:hover{background:var(--border-hl)}
    \\#empty{margin:auto;text-align:center;padding:40px 20px;animation:fadeUp .5s ease-out}
    \\#empty .icon{font-size:48px;margin-bottom:16px;filter:grayscale(0.3)}
    \\#empty h2{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--accent);letter-spacing:-1px;margin-bottom:8px}
    \\#empty p{color:var(--text-2);font-size:15px;margin-bottom:24px}
    \\#empty .hints{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}
    \\#empty .hint{font-family:var(--mono);font-size:11px;color:var(--text-3);background:var(--surface);border:1px solid var(--border);padding:4px 10px;border-radius:var(--r)}
    \\.msg-wrap{display:flex;flex-direction:column;gap:4px;max-width:var(--max-w);width:100%;margin:0 auto;animation:fadeUp .3s ease-out}
    \\.msg-wrap.user{align-items:flex-end}
    \\.msg-wrap.assistant{align-items:flex-start}
    \\.role{font-family:var(--mono);font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.5px;padding:0 4px}
    \\.role.user{color:var(--text-3)}
    \\.role.assistant{color:var(--accent)}
    \\.msg{position:relative;padding:14px 18px;border-radius:var(--r);max-width:100%;word-wrap:break-word;overflow-wrap:break-word}
    \\.msg.user{background:var(--surface);border:1px solid var(--border);border-bottom-right-radius:2px}
    \\.msg.assistant{background:var(--elevated);border:1px solid var(--border);border-bottom-left-radius:2px}
    \\.msg p{margin:0 0 10px}.msg p:last-child{margin:0}
    \\.msg pre{background:var(--bg);padding:14px;border-radius:6px;overflow-x:auto;margin:12px 0;position:relative;border:1px solid var(--border)}
    \\.msg pre code{font-family:var(--mono);font-size:13px;line-height:1.5;background:none!important;padding:0;border:none;color:inherit}
    \\.msg code:not(pre code){font-family:var(--mono);background:var(--surface);padding:2px 6px;border-radius:3px;font-size:13px;color:var(--accent)}
    \\.msg ul,.msg ol{margin:8px 0 8px 24px}
    \\.msg li{margin:4px 0}
    \\.msg blockquote{border-left:3px solid var(--accent);padding-left:16px;color:var(--text-2);margin:12px 0;font-style:italic}
    \\.msg h1,.msg h2,.msg h3{color:var(--accent);margin:16px 0 8px}
    \\.msg h1{font-size:1.4em}.msg h2{font-size:1.2em}.msg h3{font-size:1.1em}
    \\.msg a{color:var(--accent);text-decoration:none}
    \\.msg a:hover{text-decoration:underline}
    \\.msg table{border-collapse:collapse;margin:12px 0;width:100%}
    \\.msg th,.msg td{border:1px solid var(--border);padding:8px 12px;text-align:left}
    \\.msg th{background:var(--surface);font-weight:600}
    \\.code-lang{position:absolute;top:6px;left:12px;font-family:var(--mono);font-size:10px;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px}
    \\.copy-btn{position:absolute;top:6px;right:8px;font-family:var(--mono);background:var(--surface);border:1px solid var(--border);color:var(--text-3);padding:3px 8px;border-radius:4px;cursor:pointer;font-size:11px;opacity:0;transition:all .2s}
    \\.msg pre:hover .copy-btn{opacity:1}
    \\.copy-btn:hover{background:var(--border);color:var(--text)}
    \\.msg-copy{position:absolute;top:8px;right:8px;font-family:var(--mono);background:none;border:none;color:var(--text-3);cursor:pointer;font-size:11px;opacity:0;transition:opacity .2s;padding:4px}
    \\.msg:hover .msg-copy{opacity:1}
    \\.msg-copy:hover{color:var(--accent)}
    \\.stats{font-family:var(--mono);font-size:11px;color:var(--text-3);margin-top:10px;padding-top:10px;border-top:1px solid var(--border);display:flex;gap:16px;flex-wrap:wrap}
    \\.stats .val{color:var(--accent);font-weight:500}
    \\.streaming{display:flex;align-items:center;gap:6px;padding:4px 0}
    \\.typing-dots{display:flex;gap:4px;align-items:center}
    \\.typing-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);opacity:.3;animation:pulse 1.4s ease-in-out infinite}
    \\.typing-dot:nth-child(2){animation-delay:.2s}
    \\.typing-dot:nth-child(3){animation-delay:.4s}
    \\.error-msg{background:var(--red-bg);border:1px solid var(--red);color:#e8a0a0;padding:12px 18px;border-radius:var(--r);margin:8px 0;font-size:14px}
    \\#chat-form{position:relative;z-index:10;padding:16px 24px 20px;border-top:1px solid var(--border);background:var(--surface);box-shadow:0 -1px 3px rgba(0,0,0,0.2)}
    \\.input-row{max-width:var(--max-w);margin:0 auto;display:flex;gap:10px;align-items:flex-end}
    \\#msg{flex:1;font-family:var(--sans);padding:12px 16px;border-radius:var(--r);border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:15px;outline:none;resize:none;min-height:48px;max-height:200px;line-height:1.5;transition:border-color .2s}
    \\#msg:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-bg)}
    \\#msg::placeholder{color:var(--text-3)}
    \\.send-btn{font-family:var(--mono);padding:12px 20px;border-radius:var(--r);border:1px solid var(--accent);background:var(--accent-bg);color:var(--accent);font-size:14px;font-weight:500;cursor:pointer;transition:all .2s;white-space:nowrap}
    \\.send-btn:hover{background:var(--accent);color:var(--bg)}
    \\.send-btn:disabled{opacity:.3;cursor:not-allowed}
    \\.send-btn:disabled:hover{background:var(--accent-bg);color:var(--accent)}
    \\.stop-btn{font-family:var(--mono);padding:12px 20px;border-radius:var(--r);border:1px solid var(--red);background:var(--red-bg);color:var(--red);font-size:14px;font-weight:500;cursor:pointer;transition:all .2s;white-space:nowrap;display:none}
    \\.stop-btn:hover{background:var(--red);color:white}
    \\.input-hint{font-family:var(--mono);font-size:11px;color:var(--text-3);text-align:center;margin-top:8px}
    \\.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
    \\.modal-overlay.show{display:flex}
    \\.modal{background:var(--elevated);border:1px solid var(--border);border-radius:12px;padding:28px;max-width:480px;width:90%;box-shadow:0 24px 48px rgba(0,0,0,.4)}
    \\.modal-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
    \\.modal-title{font-family:var(--mono);font-size:16px;font-weight:600;color:var(--accent)}
    \\.modal-close{background:none;border:none;color:var(--text-3);cursor:pointer;font-size:20px;padding:4px 8px;border-radius:4px}
    \\.modal-close:hover{background:var(--surface);color:var(--text)}
    \\.modal-body{color:var(--text-2);line-height:1.8}
    \\.modal-body h3{font-family:var(--mono);color:var(--accent);margin:16px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.5px}
    \\.modal-body ul{margin:4px 0;padding-left:20px}
    \\.modal-body li{margin:4px 0;font-size:14px}
    \\.info-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:13px}
    \\.info-row:last-child{border:none}
    \\.info-label{color:var(--text-3)}
    \\.info-val{font-family:var(--mono);color:var(--text)}
    \\.modal-body kbd{font-family:var(--mono);background:var(--surface);padding:2px 6px;border-radius:3px;border:1px solid var(--border);font-size:12px}
    \\.hdr-btn:focus-visible,.new-chat-btn:focus-visible,.send-btn:focus-visible,.stop-btn:focus-visible,.modal-close:focus-visible,.conv-del:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
    \\@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
    \\@keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
    \\@media(prefers-reduced-motion:reduce){.msg-wrap,.typing-dot,#empty{animation:none!important}.typing-dot{opacity:.6}}
    \\@media(max-width:700px){.sidebar{position:fixed;left:0;top:0;bottom:0;transform:translateX(-100%);transition:transform .2s;z-index:50}.sidebar.open{transform:translateX(0)}.sidebar-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:40}.sidebar-overlay.show{display:block}header{padding:12px 16px}#chat{padding:16px 16px 8px}#chat-form{padding:12px 16px 16px}#msg{font-size:16px}.btn-text{display:none}.msg{padding:12px 14px}#empty h2{font-size:24px}.input-hint{display:none}.menu-btn{display:flex!important}}
    \\</style></head><body>
    \\<div class="sidebar-overlay" id="sidebar-overlay" onclick="toggleSidebar()"></div>
    \\<header>
    \\<div class="logo">
    \\<button class="hdr-btn menu-btn" onclick="toggleSidebar()" style="display:none" title="Menu" aria-label="Toggle sidebar" aria-expanded="false" id="menu-btn">&#9776;</button>
    \\<h1>&#127797; agave</h1><span class="model-badge" id="model-name">loading...</span></div>
    \\<div class="hdr-btns">
    \\<button class="new-chat-btn" onclick="newConv()" aria-label="New conversation">+ New</button>
    \\<button class="hdr-btn" onclick="clearChat()" title="Clear chat" aria-label="Clear conversation"><span class="btn-text">Clear </span>&#10005;</button>
    \\<button class="hdr-btn" onclick="showInfo()" title="About" aria-label="About Agave"><span class="btn-text">Info </span>&#9432;</button>
    \\</div></header>
    \\<div class="layout">
    \\<aside class="sidebar" id="sidebar" role="navigation" aria-label="Conversations">
    \\<div class="conv-list" id="conv-list"><div class="conv-empty">No conversations yet</div></div>
    \\</aside>
    \\<div class="main" role="main">
    \\<div id="chat" aria-live="polite" aria-relevant="additions"><div id="empty">
    \\<div class="icon">&#127797;</div><h2>agave</h2>
    \\<p>High-performance LLM inference engine</p>
    \\<div class="hints"><span class="hint">Type a message to start</span><span class="hint">/help for commands</span><span class="hint">Shift+Enter for new line</span></div>
    \\</div></div>
    \\<div class="modal-overlay" id="info-modal" onclick="if(event.target===this)hideInfo()" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    \\<div class="modal"><div class="modal-head">
    \\<span class="modal-title" id="modal-title">&#127797; About Agave</span>
    \\<button class="modal-close" onclick="hideInfo()" aria-label="Close dialog">&#10005;</button>
    \\</div><div class="modal-body">
    \\<p><strong>Agave LLM Inference Engine</strong></p>
    \\<h3>System</h3>
    \\<div class="info-row"><span class="info-label">Model</span><span class="info-val" id="info-model">-</span></div>
    \\<div class="info-row"><span class="info-label">Backend</span><span class="info-val" id="info-backend">-</span></div>
    \\<div class="info-row"><span class="info-label">API</span><span class="info-val">OpenAI-compatible</span></div>
    \\<h3>Features</h3>
    \\<ul><li>Multi-backend: CPU, Metal, Vulkan, CUDA</li><li>GGUF &amp; SafeTensors support</li><li>SSE streaming responses</li><li>Zero-copy weight loading via mmap</li></ul>
    \\<h3>Shortcuts</h3>
    \\<div class="info-row"><span class="info-label">Send</span><span class="info-val"><kbd>Enter</kbd></span></div>
    \\<div class="info-row"><span class="info-label">New line</span><span class="info-val"><kbd>Shift+Enter</kbd></span></div>
    \\<div class="info-row"><span class="info-label">Stop</span><span class="info-val"><kbd>Escape</kbd></span></div>
    \\</div></div></div>
    \\<form id="chat-form" onsubmit="return onSubmit(event)">
    \\<div class="input-row">
    \\<textarea id="msg" placeholder="Send a message..." rows="1" aria-label="Message input"></textarea>
    \\<button type="submit" class="send-btn" id="send-btn" aria-label="Send message">Send</button>
    \\<button type="button" class="stop-btn" id="stop-btn" onclick="stopGen()" aria-label="Stop generation">Stop</button>
    \\</div>
    \\<div class="input-hint">Enter to send &middot; Shift+Enter for new line &middot; Escape to stop</div>
    \\</form>
    \\</div></div>
    \\<script>
    \\marked.setOptions({breaks:true,gfm:true});
    \\var chat=document.getElementById('chat'),inp=document.getElementById('msg'),sendBtn=document.getElementById('send-btn'),stopBtn=document.getElementById('stop-btn');
    \\var modelName='',abortCtrl=null,isStreaming=false,autoScroll=true,renderTimer=null;
    \\var backendName='';
\\fetch('/v1/models').then(function(r){return r.json()}).then(function(d){if(d.data&&d.data[0]){modelName=d.data[0].id;backendName=d.data[0].backend||'';document.getElementById('model-name').textContent=modelName}}).catch(function(){document.getElementById('model-name').textContent='offline'});
    \\function autoResize(){inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,200)+'px'}
    \\inp.addEventListener('input',autoResize);
    \\inp.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('chat-form').requestSubmit()}});
    \\document.addEventListener('keydown',function(e){if(e.key==='Escape'){if(document.getElementById('info-modal').classList.contains('show'))hideInfo();else if(isStreaming)stopGen()}});
    \\chat.addEventListener('scroll',function(){autoScroll=chat.scrollHeight-chat.scrollTop-chat.clientHeight<80});
    \\function scrollBottom(){if(autoScroll)chat.scrollTop=chat.scrollHeight}
    \\function setStreaming(s){isStreaming=s;sendBtn.style.display=s?'none':'';stopBtn.style.display=s?'':'none';inp.disabled=s}
    \\function addUser(text){
    \\  var e=document.getElementById('empty');if(e)e.remove();
    \\  var w=document.createElement('div');w.className='msg-wrap user';
    \\  var r=document.createElement('span');r.className='role user';r.textContent='You';
    \\  var m=document.createElement('div');m.className='msg user';m.textContent=text;
    \\  w.appendChild(r);w.appendChild(m);chat.appendChild(w);scrollBottom()}
    \\function addAssistant(){
    \\  var e=document.getElementById('empty');if(e)e.remove();
    \\  var w=document.createElement('div');w.className='msg-wrap assistant';
    \\  var r=document.createElement('span');r.className='role assistant';r.textContent='agave';
    \\  var m=document.createElement('div');m.className='msg assistant';
    \\  m.textContent='\u2026';
    \\  w.appendChild(r);w.appendChild(m);chat.appendChild(w);scrollBottom();return m}
    \\function processCode(el){
    \\  el.querySelectorAll('pre code').forEach(function(b){
    \\    hljs.highlightElement(b);var pre=b.parentElement,lang=(b.className.match(/language-(\w+)/)||[])[1]||'';
    \\    if(lang){var l=document.createElement('span');l.className='code-lang';l.textContent=lang;pre.appendChild(l)}
    \\    var c=document.createElement('button');c.className='copy-btn';c.textContent='Copy';
    \\    c.onclick=function(){navigator.clipboard.writeText(b.textContent);c.textContent='Copied!';setTimeout(function(){c.textContent='Copy'},2000)};
    \\    pre.appendChild(c)})}
    \\function renderContent(el,content,final){
    \\  if(renderTimer&&!final)return;
    \\  var doRender=function(){
    \\    el.textContent='';var dc=content.replace(/<think>([\s\S]*?)<\/think>\s*/g,function(_,p){var t=p.trim();return t?'\n> '+t.replace(/\n/g,'\n> ')+'\n\n':''});if(dc.indexOf('<think>')===0)dc=dc.substring(7);var parsed=marked.parse(dc);
    \\    var sanitized=typeof DOMPurify!=='undefined'?DOMPurify.sanitize(parsed):parsed;
    \\    var container=document.createElement('div');container.innerHTML=sanitized;
    \\    while(container.firstChild)el.appendChild(container.firstChild);
    \\    processCode(el);
    \\    if(final){var cb=document.createElement('button');cb.className='msg-copy';cb.textContent='Copy';
    \\      cb.onclick=function(){navigator.clipboard.writeText(content);cb.textContent='Copied!';setTimeout(function(){cb.textContent='Copy'},2000)};el.appendChild(cb)}
    \\    scrollBottom();renderTimer=null};
    \\  if(final){if(renderTimer)clearTimeout(renderTimer);doRender()}else{renderTimer=setTimeout(doRender,60)}}
    \\function mkStat(label,val,unit){var sp=document.createElement('span');sp.textContent=label+' ';var v=document.createElement('span');v.className='val';v.textContent=val;sp.appendChild(v);if(unit){var u=document.createTextNode(' '+unit);sp.appendChild(u)}return sp}
    \\function addStats(el,s){
    \\  var d=document.createElement('div');d.className='stats';
    \\  var total=parseInt(s.time)+(parseInt(s.pfMs)||0);
    \\  d.appendChild(mkStat('decode ',s.tokens+' tok @ '+s.tps,'tok/s'));
    \\  if(s.pfTok&&s.pfTok!=='0')d.appendChild(mkStat('prefill ',s.pfTok+' tok @ '+s.pfTps,'tok/s'));
    \\  if(s.pfMs&&s.pfMs!=='0')d.appendChild(mkStat('TTFT ',s.pfMs,'ms'));
    \\  d.appendChild(mkStat('total ',String(total),'ms'));
    \\  el.appendChild(d)}
    \\function sendMessage(text){
    \\  var el=addAssistant();
    \\  setStreaming(true);abortCtrl=new AbortController();var content='';
    \\  fetch('/v1/chat',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    \\    body:'message='+encodeURIComponent(text)+'&stream=1',signal:abortCtrl.signal})
    \\  .then(function(resp){if(!resp.ok)throw new Error('Server error: '+resp.status);
    \\    var reader=resp.body.getReader(),decoder=new TextDecoder(),buf='';
    \\    function read(){return reader.read().then(function(r){if(r.done){renderContent(el,content||'*(no response)*',true);loadConvs();return}
    \\      buf+=decoder.decode(r.value,{stream:true});var lines=buf.split('\n');buf=lines.pop()||'';
    \\      for(var i=0;i<lines.length;i++){var ln=lines[i];if(ln.indexOf('data: ')!==0)continue;var d=ln.substring(6);
    \\        if(d==='[DONE]'){renderContent(el,content||'*(no response)*',true);loadConvs();return}
    \\        try{var o=JSON.parse(d);if(o.t){content+=o.t;renderContent(el,content,false)}
    \\          if(o.done)addStats(el,{tokens:String(o.n),tps:o.tps.toFixed(2),time:String(o.ms),pfTok:String(o.pn),pfMs:String(o.pms),pfTps:o.ptps.toFixed(1)})}catch(e){}}
    \\      return read()})}return read()})
    \\  .catch(function(e){
    \\    if(e.name==='AbortError'){renderContent(el,content||'*Stopped*',true)}
    \\    else{var err=document.createElement('div');err.className='error-msg';err.textContent='Failed to get response: '+e.message+'. Check that the server is running.';el.textContent='';el.appendChild(err)}})
    \\  .finally(function(){abortCtrl=null;setStreaming(false);inp.focus()})}
    \\function handleCommand(cmd){
    \\  if(cmd==='/help'){var el=addAssistant();renderContent(el,'**Commands:**\n- `/clear` \u2014 Clear conversation and KV cache\n- `/model` \u2014 Show model name\n- `/help` \u2014 Show this help\n\n**Shortcuts:**\n- `Enter` \u2014 Send message\n- `Shift+Enter` \u2014 New line\n- `Escape` \u2014 Stop generation or close dialog',true);return}
    \\  if(cmd==='/model'){var el2=addAssistant();renderContent(el2,'Model: **'+(modelName||'unknown')+'**',true);return}
    \\  fetch('/v1/chat',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'message='+encodeURIComponent(cmd)})
    \\  .then(function(resp){return resp.text()}).then(function(html){
    \\    var tmp=document.createElement('div');tmp.innerHTML=html;
    \\    var msgEl=tmp.querySelector('.msg.assistant');var msg=msgEl?msgEl.textContent:'Done';
    \\    var el3=addAssistant();renderContent(el3,msg,true)})
    \\  .catch(function(){var el4=addAssistant();var err=document.createElement('div');err.className='error-msg';err.textContent='Command failed';el4.textContent='';el4.appendChild(err)})}
    \\function onSubmit(e){e.preventDefault();var text=inp.value.trim();if(!text||isStreaming)return false;
    \\  inp.value='';autoResize();addUser(text);
    \\  if(text.charAt(0)==='/')handleCommand(text);else sendMessage(text);return false}
    \\function stopGen(){if(abortCtrl)abortCtrl.abort()}
    \\function showEmpty(){
    \\  while(chat.firstChild)chat.removeChild(chat.firstChild);
    \\  var empty=document.createElement('div');empty.id='empty';
    \\  var emptyHtml='<div class="icon">&#127797;</div><h2>agave</h2><p>High-performance LLM inference engine</p><div class="hints"><span class="hint">Type a message to start</span><span class="hint">/help for commands</span></div>';
    \\  empty.innerHTML=typeof DOMPurify!=='undefined'?DOMPurify.sanitize(emptyHtml):emptyHtml;
    \\  chat.appendChild(empty)}
    \\function clearChat(){
    \\  fetch('/v1/chat',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'message=%2Fclear'}).then(function(){loadConvs();showEmpty()}).catch(function(){showEmpty()})}
    \\function toggleSidebar(){
    \\  var sb=document.getElementById('sidebar'),btn=document.getElementById('menu-btn');
    \\  sb.classList.toggle('open');document.getElementById('sidebar-overlay').classList.toggle('show');
    \\  if(btn)btn.setAttribute('aria-expanded',sb.classList.contains('open'))}
    \\function loadConvs(){
    \\  fetch('/v1/conversations').then(function(r){return r.json()}).then(function(convs){
    \\    var list=document.getElementById('conv-list');
    \\    while(list.firstChild)list.removeChild(list.firstChild);
    \\    if(!convs.length){var em=document.createElement('div');em.className='conv-empty';em.textContent='No conversations yet';list.appendChild(em);return}
    \\    convs.forEach(function(c){
    \\      var item=document.createElement('div');item.className='conv-item'+(c.active?' active':'');item.tabIndex=0;item.setAttribute('role','button');
    \\      item.onclick=function(){selectConv(c.id)};item.onkeydown=function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();selectConv(c.id)}};
    \\      var title=document.createElement('span');title.className='conv-title';title.textContent=c.title||'New chat';if(c.title&&c.title.length>30)title.title=c.title;
    \\      var del=document.createElement('button');del.className='conv-del';del.textContent='\u00d7';del.setAttribute('aria-label','Delete conversation');
    \\      del.onclick=function(e){e.stopPropagation();deleteConv(c.id)};
    \\      item.appendChild(title);item.appendChild(del);list.appendChild(item)})
    \\  }).catch(function(){})}
    \\function newConv(){
    \\  fetch('/v1/conversations',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'action=new'})
    \\  .then(function(){loadConvs();showEmpty();inp.focus()}).catch(function(){})}
    \\function selectConv(id){
    \\  fetch('/v1/conversations',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'action=select&id='+id})
    \\  .then(function(r){return r.json()}).then(function(data){
    \\    while(chat.firstChild)chat.removeChild(chat.firstChild);
    \\    if(!data.messages||!data.messages.length){showEmpty();loadConvs();return}
    \\    var e=document.getElementById('empty');if(e)e.remove();
    \\    data.messages.forEach(function(m){
    \\      if(m.role==='user'){addUser(m.content)}
    \\      else{var el=addAssistant();renderContent(el,m.content,true)}});
    \\    loadConvs();scrollBottom()}).catch(function(){})}
    \\function deleteConv(id){
    \\  fetch('/v1/conversations',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'action=delete&id='+id})
    \\  .then(function(r){return r.json()}).then(function(data){
    \\    loadConvs();if(data.cleared)showEmpty()}).catch(function(){})}
    \\function showInfo(){var m=document.getElementById('info-modal');m.classList.add('show');document.getElementById('info-model').textContent=modelName||'-';document.getElementById('info-backend').textContent=backendName||'-';var cb=m.querySelector('.modal-close');if(cb)cb.focus()}
    \\function hideInfo(){document.getElementById('info-modal').classList.remove('show');inp.focus()}
    \\loadConvs();
    \\</script></body></html>
;

/// Generate a unique request ID (monotonically increasing counter).
fn nextRequestId() u64 {
    return g_server.request_counter.fetchAdd(1, .monotonic);
}

// ── HTTP helpers ────────────────────────────────────────────────

/// Parsed HTTP request. Slices point into the read buffer.
const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    headers: []const u8,
    body: []const u8,
};

/// Parse Content-Length from raw HTTP headers.
fn parseContentLength(headers: []const u8) usize {
    const header_name = "content-length";
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        if (colon == header_name.len and std.ascii.eqlIgnoreCase(line[0..header_name.len], header_name)) {
            return std.fmt.parseInt(usize, std.mem.trim(u8, line[colon + 1 ..], " "), 10) catch 0;
        }
    }
    return 0;
}

/// Read a complete HTTP/1.1 request from a TCP stream. Returns null on
/// malformed input or connection close.
fn readHttpRequest(stream: net.Stream, buf: []u8) ?HttpRequest {
    var total: usize = 0;

    // Read until we have complete headers (\r\n\r\n)
    while (total < buf.len) {
        const n = stream.read(buf[total..]) catch return null;
        if (n == 0) return null;
        total += n;
        if (std.mem.indexOf(u8, buf[0..total], "\r\n\r\n")) |_| break;
    }

    const hdr_end = std.mem.indexOf(u8, buf[0..total], "\r\n\r\n") orelse return null;

    // Parse request line: "GET /path HTTP/1.1"
    const req_line_end = std.mem.indexOf(u8, buf[0..hdr_end], "\r\n") orelse return null;
    const req_line = buf[0..req_line_end];
    const sp1 = std.mem.indexOf(u8, req_line, " ") orelse return null;
    const method = req_line[0..sp1];
    const rest = req_line[sp1 + 1 ..];
    const sp2 = std.mem.indexOf(u8, rest, " ") orelse return null;
    const raw_path = rest[0..sp2];
    // Strip query string
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |q| raw_path[0..q] else raw_path;

    // Parse Content-Length
    const headers = buf[req_line_end + 2 .. hdr_end];
    const content_length = parseContentLength(headers);
    const body_start = hdr_end + 4;

    // Read remaining body bytes if needed
    if (content_length > 0) {
        if (content_length > max_request_body_size) return null; // body too large
        const body_end = body_start + content_length;
        if (body_end > buf.len) return null; // exceeds read buffer
        while (total < body_end) {
            const n = stream.read(buf[total..body_end]) catch return null;
            if (n == 0) return null;
            total += n;
        }
        return .{ .method = method, .path = path, .headers = headers, .body = buf[body_start..body_end] };
    }

    return .{ .method = method, .path = path, .headers = headers, .body = "" };
}

/// Common security headers appended to every response.
const security_headers =
    "X-Content-Type-Options: nosniff\r\n" ++
    "X-Frame-Options: DENY\r\n" ++
    "Referrer-Policy: no-referrer\r\n" ++
    "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; img-src 'self'; frame-ancestors 'none'\r\n";

/// Validate Authorization header against configured API key.
/// Returns true if no auth configured or if token matches.
fn validateAuth(server: *const Server, headers: []const u8) bool {
    if (server.api_key == null) return true; // No auth configured
    const needle = "authorization: bearer ";
    const idx = std.ascii.indexOfIgnoreCase(headers, needle) orelse return false;
    const token_start = idx + needle.len;
    const token_end = std.mem.indexOfScalarPos(u8, headers, token_start, '\r') orelse headers.len;
    const token = headers[token_start..token_end];
    return std.mem.eql(u8, token, server.api_key.?);
}

/// Check rate limit for the given prompt token count.
/// Returns null if allowed, or retry-after seconds if rate limited.
fn checkRateLimit(server: *Server, prompt_tokens: u32) ?u32 {
    if (server.rate_limiter == null) return null;
    if (server.rate_limiter.?.tryConsumeRequest(prompt_tokens)) return null;
    return server.rate_limiter.?.retryAfter(prompt_tokens);
}

/// Write a complete HTTP response (status line + headers + body).
fn sendResponse(stream: net.Stream, status: []const u8, content_type: []const u8, body: []const u8) void {
    var hdr_buf: [1024]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\n" ++ security_headers ++ "Connection: close\r\n\r\n", .{ status, content_type, body.len }) catch return;
    stream.writeAll(hdr) catch return;
    stream.writeAll(body) catch return;
}

fn sendJson(stream: net.Stream, body: []const u8) void {
    sendResponse(stream, "200 OK", "application/json", body);
}

fn sendHtml(stream: net.Stream, body: []const u8) void {
    sendResponse(stream, "200 OK", "text/html; charset=utf-8", body);
}

/// Send a JSON error response following the OpenAI error format.
fn sendJsonError(stream: net.Stream, status: []const u8, err_type: []const u8, message: []const u8) void {
    var buf: [response_buf_size]u8 = undefined;
    const json = std.fmt.bufPrint(&buf,
        \\{{"error":{{"message":"{s}","type":"{s}","code":null}}}}
    , .{ message, err_type }) catch return;
    sendResponse(stream, status, "application/json", json);
}

/// Send 401 Unauthorized response for invalid API key.
fn send401(stream: net.Stream) void {
    const body = "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}";
    sendResponse(stream, "401 Unauthorized", "application/json", body);
}

/// Send 429 Too Many Requests with Retry-After header.
fn send429(stream: net.Stream, retry_after: u32) void {
    var buf: [256]u8 = undefined;
    const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"Rate limit exceeded. Retry after {d} seconds.\",\"type\":\"rate_limit_exceeded\",\"code\":null}}}}", .{retry_after}) catch return;
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nRetry-After: {d}\r\n" ++ security_headers ++ "Connection: close\r\n\r\n", .{ body.len, retry_after }) catch return;
    stream.writeAll(hdr) catch return;
    stream.writeAll(body) catch return;
}

// ── Request handler ─────────────────────────────────────────────

fn handleRequest(stream: net.Stream, req: HttpRequest) void {
    const request_start = std.time.milliTimestamp();
    const path = req.path;
    const method = req.method;
    const is_get = std.mem.eql(u8, method, "GET");
    const is_post = std.mem.eql(u8, method, "POST");

    // CORS preflight
    if (std.mem.eql(u8, method, "OPTIONS")) {
        stream.writeAll("HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 86400\r\n" ++ security_headers ++ "Content-Length: 0\r\nConnection: close\r\n\r\n") catch return;
        return;
    }

    // Health check endpoint — lightweight, no mutex, no inference
    if (is_get and std.mem.eql(u8, path, "/health")) {
        var buf: [512]u8 = undefined;
        const uptime: i64 = if (g_server.start_time > 0) std.time.timestamp() - g_server.start_time else 0;
        const json = std.fmt.bufPrint(&buf,
            \\{{"status":"ok","model":"{s}","backend":"{s}","uptime_s":{d},"active_connections":{d},"requests_served":{d}}}
        , .{ g_server.model_name, g_server.backend_name, uptime, g_server.active_connections.load(.monotonic), g_server.request_counter.load(.monotonic) }) catch return;
        sendJson(stream, json);
        return;
    }

    // Readiness check endpoint — returns 503 if shutting down
    if (is_get and std.mem.eql(u8, path, "/ready")) {
        if (g_server.shutdown_requested.load(.acquire)) {
            const body = "{\"status\":\"shutting_down\"}";
            stream.writeAll("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: ") catch return;
            var len_buf: [20]u8 = undefined;
            const len_str = std.fmt.bufPrint(&len_buf, "{d}", .{body.len}) catch return;
            stream.writeAll(len_str) catch return;
            stream.writeAll("\r\nConnection: close\r\n\r\n") catch return;
            stream.writeAll(body) catch return;
        } else {
            const body = "{\"status\":\"ready\"}";
            sendJson(stream, body);
        }
        return;
    }

    // Prometheus metrics endpoint
    if (is_get and std.mem.eql(u8, path, "/metrics")) {
        var buf: [16384]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const writer = fbs.writer();
        g_server.metrics.renderPrometheus(writer) catch return;
        const body = fbs.getWritten();
        stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: ") catch return;
        var len_buf: [20]u8 = undefined;
        const len_str = std.fmt.bufPrint(&len_buf, "{d}", .{body.len}) catch return;
        stream.writeAll(len_str) catch return;
        stream.writeAll("\r\nConnection: close\r\n\r\n") catch return;
        stream.writeAll(body) catch return;
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/favicon.ico")) {
        stream.writeAll("HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n") catch return;
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/")) {
        logRequest(method, path);
        sendHtml(stream, html_page);
        return;
    }

    if (is_get and std.mem.eql(u8, path, "/v1/models")) {
        logRequest(method, path);
        var buf: [models_json_buf_size]u8 = undefined;
        const json = std.fmt.bufPrint(&buf,
            \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"agave","backend":"{s}"}}]}}
        , .{ g_server.model_name, std.time.timestamp(), g_server.backend_name }) catch return;
        sendJson(stream, json);
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/chat/completions")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        const req_start_time = std.time.milliTimestamp();

        // 1. Validate authentication
        if (!validateAuth(g_server, req.headers)) {
            send401(stream);
            logRequestDone(method, path, 401, elapsedMs(request_start));
            return;
        }

        const body = req.body;
        const content = extractLastMessage(body) orelse "Hello!";
        const max_tokens = extractIntField(body, "max_tokens") orelse default_max_gen_tokens;

        // 2. Format prompt and get token count for rate limiting
        const formatted = g_server.chat_template.format(g_server.allocator, null, content) catch content;
        defer if (formatted.ptr != content.ptr) g_server.allocator.free(formatted);
        const prompt_ids = g_server.tokenizer.encode(formatted) catch &[_]u32{};
        defer if (prompt_ids.len > 0) g_server.allocator.free(prompt_ids);
        const prompt_tokens: u32 = @intCast(prompt_ids.len);

        // 3. Check rate limit
        if (checkRateLimit(g_server, prompt_tokens)) |retry| {
            send429(stream, retry);
            logRequestDone(method, path, 429, elapsedMs(request_start));
            return;
        }

        if (extractBoolField(body, "stream")) {
            startStream(stream, content, true, max_tokens);
            // Note: metrics for streaming recorded inside startStream
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Formatted already computed above for token counting
        const gen = generateEscapedN(formatted, true, max_tokens);
        defer gen.deinit();
        const req_id = nextRequestId();
        const created = std.time.timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        var resp_buf: [response_buf_size]u8 = undefined;
        const json = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"completion_tokens":{d},"prompt_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, g_server.model_name, gen.escaped, gen.finish_reason, gen.stats.tokens_generated, gen.stats.prompt_tokens, total }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json);

        // Record metrics
        const req_end_time = std.time.milliTimestamp();
        const duration_ms = @as(u64, @intCast(req_end_time - req_start_time));
        g_server.metrics.recordLatency(duration_ms);
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/completions")) {
        logRequest(method, path);
        g_server.metrics.recordRequest();
        const req_start_time = std.time.milliTimestamp();

        const body = req.body;
        const prompt = extractField(body, "prompt") orelse "Hello";
        const max_tokens = extractIntField(body, "max_tokens") orelse default_max_gen_tokens;

        if (extractBoolField(body, "stream")) {
            startStreamRaw(stream, prompt, max_tokens);
            // Note: metrics for streaming recorded inside startStreamRaw
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        // Completions endpoint: use prompt as-is (no chat template wrapping)
        const gen = generateEscapedN(prompt, true, max_tokens);
        defer gen.deinit();
        const req_id = nextRequestId();
        const created = std.time.timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        var resp_buf: [response_buf_size]u8 = undefined;
        const json = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":"{s}"}}],"usage":{{"completion_tokens":{d},"prompt_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, g_server.model_name, gen.escaped, gen.finish_reason, gen.stats.tokens_generated, gen.stats.prompt_tokens, total }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json);

        // Record metrics
        const req_end_time = std.time.milliTimestamp();
        const duration_ms = @as(u64, @intCast(req_end_time - req_start_time));
        g_server.metrics.recordLatency(duration_ms);
        g_server.metrics.recordTokens(@intCast(gen.stats.tokens_generated));
        g_server.metrics.recordCompletion();

        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/embeddings")) {
        logRequestDone(method, path, 501, elapsedMs(request_start));
        sendJsonError(stream, "501 Not Implemented", "not_implemented", "Embeddings endpoint not implemented");
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/responses")) {
        logRequest(method, path);
        const body = req.body;
        const input = extractField(body, "input") orelse "Hello";
        const max_tokens = extractIntField(body, "max_tokens") orelse default_max_gen_tokens;
        const formatted_i = g_server.chat_template.format(g_server.allocator, null, input) catch input;
        defer if (formatted_i.ptr != input.ptr) g_server.allocator.free(formatted_i);
        const gen = generateEscapedN(formatted_i, true, max_tokens);
        defer gen.deinit();
        const req_id = nextRequestId();
        const created = std.time.timestamp();
        const total = gen.stats.tokens_generated + gen.stats.prompt_tokens;
        var resp_buf: [response_buf_size]u8 = undefined;
        const json = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"resp-{d}","object":"response","created":{d},"output":[{{"type":"message","content":[{{"type":"output_text","text":"{s}"}}]}}],"usage":{{"completion_tokens":{d},"prompt_tokens":{d},"total_tokens":{d}}}}}
        , .{ req_id, created, gen.escaped, gen.stats.tokens_generated, gen.stats.prompt_tokens, total }) catch {
            sendJsonError(stream, "500 Internal Server Error", "server_error", "Response too large");
            logRequestDone(method, path, 500, elapsedMs(request_start));
            return;
        };
        sendJson(stream, json);
        logRequestDone(method, path, 200, elapsedMs(request_start));
        return;
    }

    if ((is_get or is_post) and std.mem.eql(u8, path, "/v1/conversations")) {
        logRequest(method, path);
        if (is_get) {
            // Return list of conversations as JSON (mutex-protected to
            // prevent races with concurrent create/delete/select operations).
            g_server.mutex.lock();
            defer g_server.mutex.unlock();
            var buf: [conv_list_buf_size]u8 = undefined;
            var fbs = std.io.fixedBufferStream(&buf);
            const w = fbs.writer();
            w.writeByte('[') catch return;
            for (g_server.conversations.items, 0..) |*conv, ci| {
                if (ci > 0) w.writeByte(',') catch return;
                const title = conv.titleSlice();
                const escaped_title = jsonEscape(g_server.allocator, title) catch title;
                defer if (escaped_title.ptr != title.ptr) g_server.allocator.free(escaped_title);
                std.fmt.format(w,
                    \\{{"id":{d},"title":"{s}","active":{s},"count":{d}}}
                , .{ conv.id, escaped_title, if (conv.id == g_server.active_id) "true" else "false", conv.messages.items.len }) catch return;
            }
            w.writeByte(']') catch return;
            sendJson(stream, fbs.getWritten());
            return;
        }
        // POST: action=new|select|delete
        // All conversation mutations must be mutex-protected to prevent
        // races with concurrent generate() calls that read kv_valid.
        const body = req.body;
        const action = extractFormField(body, "action") orelse "new";
        if (std.mem.eql(u8, action, "new")) {
            g_server.mutex.lock();
            _ = g_server.createConv();
            g_server.mutex.unlock();
            sendJson(stream, "{\"ok\":true}");
        } else if (std.mem.eql(u8, action, "select")) {
            const id_str = extractFormField(body, "id") orelse "0";
            const id = std.fmt.parseInt(u32, id_str, 10) catch 0;
            g_server.mutex.lock();
            g_server.selectConv(id);
            // Return messages for the selected conversation
            const conv = g_server.getConvById(id);
            var mbuf: [conv_msgs_buf_size]u8 = undefined;
            var mfbs = std.io.fixedBufferStream(&mbuf);
            const mw = mfbs.writer();
            mw.writeAll("{\"messages\":[") catch {
                g_server.mutex.unlock();
                return;
            };
            if (conv) |c| {
                for (c.messages.items, 0..) |msg, mi| {
                    if (mi > 0) mw.writeByte(',') catch {
                        g_server.mutex.unlock();
                        return;
                    };
                    const role_str: []const u8 = switch (msg.role) {
                        .user => "user",
                        .assistant => "assistant",
                    };
                    const esc_content = jsonEscape(g_server.allocator, msg.content) catch msg.content;
                    defer if (esc_content.ptr != msg.content.ptr) g_server.allocator.free(esc_content);
                    std.fmt.format(mw,
                        \\{{"role":"{s}","content":"{s}"}}
                    , .{ role_str, esc_content }) catch {
                        g_server.mutex.unlock();
                        return;
                    };
                }
            }
            mw.writeAll("]}") catch {
                g_server.mutex.unlock();
                return;
            };
            g_server.mutex.unlock();
            sendJson(stream, mfbs.getWritten());
        } else if (std.mem.eql(u8, action, "delete")) {
            const id_str = extractFormField(body, "id") orelse "0";
            const id = std.fmt.parseInt(u32, id_str, 10) catch 0;
            g_server.mutex.lock();
            const was_active = g_server.active_id == id;
            g_server.deleteConv(id);
            g_server.mutex.unlock();
            var dbuf: [128]u8 = undefined;
            const djson = std.fmt.bufPrint(&dbuf,
                \\{{"ok":true,"cleared":{s}}}
            , .{if (was_active) "true" else "false"}) catch "{\"ok\":true}";
            sendJson(stream, djson);
        } else {
            sendJson(stream, "{\"ok\":false}");
        }
        return;
    }

    if (is_post and std.mem.eql(u8, path, "/v1/chat")) {
        logRequest(method, path);
        const body = req.body;
        const msg = extractFormField(body, "message") orelse "Hello";
        if (msg.len > max_message_len) {
            logRequestDone(method, path, 400, elapsedMs(request_start));
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Message too long");
            return;
        }
        const decoded = urlDecode(g_server.allocator, msg) catch g_server.allocator.dupe(u8, msg) catch return;
        defer g_server.allocator.free(decoded);
        if (decoded.len > max_message_len) {
            logRequestDone(method, path, 400, elapsedMs(request_start));
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Message too long");
            return;
        }

        // Log the user's message (truncate if too long)
        var msg_preview: [msg_preview_buf_size]u8 = undefined;
        const preview = if (decoded.len > msg_preview_max_len)
            std.fmt.bufPrint(&msg_preview, "{s}...", .{decoded[0..msg_preview_max_len]}) catch decoded
        else
            decoded;
        slog("  User: {s}\n", .{preview});

        // Handle REPL-style commands in the chat interface
        const trimmed = std.mem.trim(u8, decoded, " \t\r\n");
        if (trimmed.len > 0 and trimmed[0] == '/') {
            const cmd_html = handleChatCommand(trimmed);
            if (cmd_html) |html| {
                sendHtml(stream, html);
                return;
            }
        }

        // Get or create active conversation (under mutex for kv_valid coherency)
        g_server.mutex.lock();
        const conv = g_server.getActiveConv() orelse g_server.createConv() orelse {
            g_server.mutex.unlock();
            return;
        };

        // Enforce per-conversation message limit
        if (conv.messages.items.len >= max_messages_per_conv) {
            g_server.mutex.unlock();
            logRequestDone(method, path, 400, elapsedMs(request_start));
            sendJsonError(stream, "400 Bad Request", "invalid_request_error", "Conversation message limit reached");
            return;
        }

        // Add user message to conversation
        const user_content = g_server.allocator.dupe(u8, trimmed) catch {
            g_server.mutex.unlock();
            return;
        };
        conv.messages.append(g_server.allocator, .{ .role = .user, .content = user_content }) catch {
            g_server.allocator.free(user_content);
            g_server.mutex.unlock();
            return;
        };

        // Set title from first user message
        if (conv.title_len == 0) conv.setTitle(trimmed);

        // Format prompt based on KV cache validity (still under mutex)
        const need_reset = !g_server.kv_valid;
        const formatted = if (need_reset)
            g_server.chat_template.formatConversation(g_server.allocator, null, conv.messages.items) catch trimmed
        else
            g_server.chat_template.formatContinuation(g_server.allocator, trimmed) catch trimmed;
        defer if (formatted.ptr != trimmed.ptr) g_server.allocator.free(formatted);
        // Release mutex before generate() — it acquires the mutex internally.
        g_server.mutex.unlock();

        // SSE streaming mode: stream tokens to the client in real-time
        const wants_stream = extractFormField(body, "stream") != null;
        if (wants_stream) {
            stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n" ++ security_headers ++ "Connection: keep-alive\r\n\r\n") catch return;
            const result = chatStreamGenerate(stream, formatted, need_reset);
            defer g_server.allocator.free(result.data);

            g_server.mutex.lock();
            g_server.kv_valid = true;
            logGeneration(result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec);
            const resp_trimmed_s = std.mem.trimRight(u8, result.data, " \t\r\n");
            if (resp_trimmed_s.len > 0) {
                const resp_s = g_server.allocator.dupe(u8, resp_trimmed_s) catch null;
                if (resp_s) |rc| {
                    const active_conv = g_server.getActiveConv();
                    if (active_conv) |ac| {
                        ac.messages.append(g_server.allocator, .{ .role = .assistant, .content = rc }) catch {
                            g_server.allocator.free(rc);
                        };
                    } else {
                        g_server.allocator.free(rc);
                    }
                }
            }
            g_server.mutex.unlock();
            logRequestDone(method, path, 200, elapsedMs(request_start));
            return;
        }

        const result = generate(formatted, need_reset);
        defer g_server.allocator.free(result.data);

        // Re-acquire mutex to update conversation state
        g_server.mutex.lock();
        g_server.kv_valid = true;
        logGeneration(result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec);

        // Store assistant response in conversation
        const resp_trimmed = std.mem.trimRight(u8, result.data, " \t\r\n");
        if (resp_trimmed.len > 0) {
            const resp_content = g_server.allocator.dupe(u8, resp_trimmed) catch null;
            if (resp_content) |rc| {
                conv.messages.append(g_server.allocator, .{ .role = .assistant, .content = rc }) catch {
                    g_server.allocator.free(rc);
                };
            }
        }
        g_server.mutex.unlock();

        const escaped_user = htmlEscape(g_server.allocator, decoded) catch decoded;
        defer if (escaped_user.ptr != decoded.ptr) g_server.allocator.free(escaped_user);
        const escaped_resp = htmlEscape(g_server.allocator, result.data) catch result.data;
        defer if (escaped_resp.ptr != result.data.ptr) g_server.allocator.free(escaped_resp);
        var html_buf: [response_buf_size]u8 = undefined;
        const html = std.fmt.bufPrint(&html_buf,
            \\<div class="msg user">{s}</div><div class="msg assistant" data-tokens="{d}" data-time="{d}" data-tps="{d:.2}" data-prefill-tokens="{d}" data-prefill-ms="{d}" data-prefill-tps="{d:.1}">{s}</div>
        , .{ escaped_user, result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec, result.stats.prompt_tokens, result.stats.prefill_ms, result.stats.prefill_tps, escaped_resp }) catch "<div class=\"msg assistant\">Error</div>";
        sendHtml(stream, html);
        return;
    }

    // Check for known paths with wrong method -> 405 with Allow header
    const KnownEndpoint = struct { path: []const u8, allow: []const u8, msg: []const u8 };
    const known_endpoints = [_]KnownEndpoint{
        .{ .path = "/v1/chat/completions", .allow = "POST, OPTIONS", .msg = "Use POST." },
        .{ .path = "/v1/completions", .allow = "POST, OPTIONS", .msg = "Use POST." },
        .{ .path = "/v1/embeddings", .allow = "POST, OPTIONS", .msg = "Use POST." },
        .{ .path = "/v1/responses", .allow = "POST, OPTIONS", .msg = "Use POST." },
        .{ .path = "/v1/chat", .allow = "POST, OPTIONS", .msg = "Use POST." },
        .{ .path = "/v1/conversations", .allow = "GET, POST, OPTIONS", .msg = "Use GET or POST." },
        .{ .path = "/v1/models", .allow = "GET, OPTIONS", .msg = "Use GET." },
        .{ .path = "/health", .allow = "GET, OPTIONS", .msg = "Use GET." },
    };
    for (known_endpoints) |ep| {
        if (std.mem.eql(u8, path, ep.path)) {
            logRequestDone(method, path, 405, elapsedMs(request_start));
            var hdr_buf: [1024]u8 = undefined;
            var body_buf: [256]u8 = undefined;
            const body = std.fmt.bufPrint(&body_buf,
                \\{{"error":{{"message":"Method not allowed. {s}","type":"invalid_request_error","code":null}}}}
            , .{ep.msg}) catch return;
            const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAllow: {s}\r\nAccess-Control-Allow-Origin: *\r\n" ++ security_headers ++ "Connection: close\r\n\r\n", .{ body.len, ep.allow }) catch return;
            stream.writeAll(hdr) catch return;
            stream.writeAll(body) catch return;
            return;
        }
    }

    logRequestDone(method, path, 404, elapsedMs(request_start));
    sendJsonError(stream, "404 Not Found", "invalid_request_error", "Unknown endpoint");
}

/// Thread-local buffer for `/model` command response formatting.
threadlocal var cmd_buf: [cmd_buf_size]u8 = undefined;

fn handleChatCommand(cmd: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, cmd, "/clear")) {
        g_server.mutex.lock();
        g_server.model.resetCache();
        g_server.kv_valid = false;
        if (g_server.getActiveConv()) |conv| conv.clearMessages(g_server.allocator);
        g_server.mutex.unlock();
        slog("  [command] /clear\n", .{});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">Conversation cleared.</div>";
    }
    if (std.mem.eql(u8, cmd, "/model")) {
        slog("  [command] /model\n", .{});
        const escaped_name = htmlEscape(g_server.allocator, g_server.model_name) catch g_server.model_name;
        defer if (escaped_name.ptr != g_server.model_name.ptr) g_server.allocator.free(escaped_name);
        return std.fmt.bufPrint(&cmd_buf,
            \\<div class="msg assistant" data-tokens="0" data-time="0" data-tps="0">Model: {s}</div>
        , .{escaped_name}) catch null;
    }
    if (std.mem.eql(u8, cmd, "/help")) {
        slog("  [command] /help\n", .{});
        return "<div class=\"msg assistant\" data-tokens=\"0\" data-time=\"0\" data-tps=\"0\">/clear &mdash; Clear conversation and KV cache&lt;br&gt;/model &mdash; Show model name&lt;br&gt;/help &mdash; Show available commands</div>";
    }
    return null;
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

fn generateEscaped(prompt: []const u8, reset: bool) GeneratedEscaped {
    return generateEscapedN(prompt, reset, default_max_gen_tokens);
}

fn generateEscapedN(prompt: []const u8, reset: bool, max_tokens: usize) GeneratedEscaped {
    const result = generateN(prompt, reset, max_tokens);
    logGeneration(result.stats.tokens_generated, result.stats.time_ms, result.stats.tokens_per_sec);
    const escaped = jsonEscape(g_server.allocator, result.data) catch result.data;
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
    return generateN(formatted, reset, default_max_gen_tokens);
}

/// Run inference with a configurable max_tokens limit.
fn generateN(formatted: []const u8, reset: bool, max_tokens: usize) GenResult {
    // Mutex serialises inference requests. This is intentional: the model
    // holds mutable state (KV cache, activation buffers) that cannot be
    // shared across concurrent requests without full session isolation.
    g_server.mutex.lock();
    defer g_server.mutex.unlock();
    const model = g_server.model;
    const tok = g_server.tokenizer;
    if (reset) model.resetCache();
    const zero_stats = Stats{ .tokens_generated = 0, .prompt_tokens = 0, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 };
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("tokenizer encode failed: {}", .{err});
        return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .stats = zero_stats };
    };
    defer g_server.allocator.free(token_ids);
    const prompt_token_count: u32 = @intCast(token_ids.len);

    // BOS token — required by models like Gemma to initialize state correctly
    if (reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.warn("BOS forward failed: {}", .{err});
            return .{ .data = g_server.allocator.dupe(u8, "[BOS forward error]") catch &.{}, .stats = zero_stats };
        };
    }

    // Prefill phase — timed separately for TTFT stats.
    // Capture the return value of the last forward() — it's the first generated token.
    const prefill_start = std.time.milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                const cancelled_stats = Stats{ .tokens_generated = 0, .prompt_tokens = prompt_token_count, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 };
                return .{ .data = g_server.allocator.dupe(u8, "[cancelled]") catch &.{}, .stats = cancelled_stats };
            }
            break;
        };
    }
    const prefill_ms: u64 = @intCast(@max(std.time.milliTimestamp() - prefill_start, 0));
    const prefill_tps: f32 = if (prefill_ms > 0) @as(f32, @floatFromInt(prompt_token_count)) / (@as(f32, @floatFromInt(prefill_ms)) / 1000.0) else 0.0;

    // Generation phase (timed) — collect token IDs, batch-decode once at the end
    // to avoid per-token alloc/free overhead.
    const gen_start = std.time.milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;
    var cancelled = false;

    // Include first generated token (from last prefill forward)
    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
    }

    var hit_eog = first_is_eog;
    const effective_max = @min(max_tokens, gen_ids_buf_size);
    for (0..effective_max -| 1) |_| {
        if (first_is_eog or token_ids.len == 0) break;
        const next = model.forward(last) catch |err| {
            if (err == error.Cancelled) cancelled = true;
            break;
        };
        if (g_server.isEog(next)) { hit_eog = true; break; }
        gen_tokens[token_count] = next;
        last = next;
        token_count += 1;
    }

    const gen_end = std.time.milliTimestamp();
    const time_ms = @as(u64, @intCast(gen_end - gen_start));
    const tokens_per_sec = if (time_ms > 0) @as(f32, @floatFromInt(token_count)) / (@as(f32, @floatFromInt(time_ms)) / 1000.0) else 0.0;
    const finish_reason: []const u8 = if (cancelled) "stop" else if (hit_eog) "stop" else "length";

    if (cancelled) {
        return .{
            .data = g_server.allocator.dupe(u8, "[cancelled]") catch &.{},
            .stats = .{ .tokens_generated = token_count, .prompt_tokens = prompt_token_count, .time_ms = time_ms, .tokens_per_sec = tokens_per_sec, .prefill_ms = prefill_ms, .prefill_tps = prefill_tps },
        };
    }

    // Single batch decode — one alloc instead of N per-token allocs
    const decoded = tok.decode(gen_tokens[0..token_count]) catch
        g_server.allocator.dupe(u8, "[decode error]") catch @constCast("");

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
fn chatStreamGenerate(stream: net.Stream, formatted: []const u8, reset: bool) GenResult {
    g_server.mutex.lock();
    defer g_server.mutex.unlock();
    const model = g_server.model;
    const tok = g_server.tokenizer;
    if (reset) model.resetCache();
    const zero_stats = Stats{ .tokens_generated = 0, .prompt_tokens = 0, .time_ms = 0, .tokens_per_sec = 0, .prefill_ms = 0, .prefill_tps = 0 };
    const token_ids = tok.encode(formatted) catch {
        _ = sseWriteData(stream, "{\"t\":\"[encode error]\",\"done\":true}");
        _ = sseWriteData(stream, "[DONE]");
        return .{ .data = g_server.allocator.dupe(u8, "[encode error]") catch &.{}, .stats = zero_stats };
    };
    defer g_server.allocator.free(token_ids);
    const prompt_token_count: u32 = @intCast(token_ids.len);

    if (reset and g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch {
            _ = sseWriteData(stream, "[DONE]");
            return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .stats = zero_stats };
        };
    }

    // Prefill
    const prefill_start = std.time.milliTimestamp();
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                _ = sseWriteData(stream, "[DONE]");
                return .{ .data = g_server.allocator.dupe(u8, "") catch &.{}, .stats = zero_stats };
            }
            break;
        };
    }
    const prefill_ms: u64 = @intCast(@max(std.time.milliTimestamp() - prefill_start, 0));
    const prefill_tps: f32 = if (prefill_ms > 0) @as(f32, @floatFromInt(prompt_token_count)) / (@as(f32, @floatFromInt(prefill_ms)) / 1000.0) else 0.0;

    // Generate and stream tokens
    const gen_start = std.time.milliTimestamp();
    var gen_tokens: [gen_ids_buf_size]u32 = undefined;
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    const first_is_eog = token_ids.len > 0 and g_server.isEog(first_gen_token);
    if (!first_is_eog and token_ids.len > 0) {
        gen_tokens[0] = first_gen_token;
        token_count = 1;
        // Stream first token
        streamToken(stream, tok, first_gen_token);
    }

    const effective_max = @min(default_max_gen_tokens, gen_ids_buf_size);
    for (0..effective_max -| 1) |_| {
        if (first_is_eog or token_ids.len == 0) break;
        const next = model.forward(last) catch break;
        if (g_server.isEog(next)) break;
        gen_tokens[token_count] = next;
        last = next;
        token_count += 1;
        streamToken(stream, tok, next);
    }

    const gen_end = std.time.milliTimestamp();
    const time_ms = @as(u64, @intCast(gen_end - gen_start));
    const tps: f32 = if (time_ms > 0) @as(f32, @floatFromInt(token_count)) / (@as(f32, @floatFromInt(time_ms)) / 1000.0) else 0.0;

    // Send final stats event
    var stats_buf: [512]u8 = undefined;
    const stats_json = std.fmt.bufPrint(&stats_buf,
        \\{{"done":true,"n":{d},"ms":{d},"tps":{d:.2},"pn":{d},"pms":{d},"ptps":{d:.1}}}
    , .{ token_count, time_ms, tps, prompt_token_count, prefill_ms, prefill_tps }) catch "";
    if (stats_json.len > 0) _ = sseWriteData(stream, stats_json);
    _ = sseWriteData(stream, "[DONE]");

    // Decode accumulated text for conversation storage
    const decoded = tok.decode(gen_tokens[0..token_count]) catch
        g_server.allocator.dupe(u8, "") catch @constCast("");

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

/// Stream a single decoded token as an SSE event.
fn streamToken(stream: net.Stream, tok: *Tokenizer, token_id: u32) void {
    const decoded = tok.decode(&[_]u32{token_id}) catch return;
    defer g_server.allocator.free(decoded);
    if (decoded.len == 0) return;

    const escaped = jsonEscape(g_server.allocator, decoded) catch return;
    defer if (escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);

    var buf: [1024]u8 = undefined;
    const event = std.fmt.bufPrint(&buf, "data: {{\"t\":\"{s}\"}}\n\n", .{escaped}) catch return;
    stream.writeAll(event) catch {};
}

// ── SSE Streaming ──────────────────────────────────────────────

/// Send an SSE data event. Returns false if the write failed (client disconnected).
fn sseWriteData(stream: net.Stream, data: []const u8) bool {
    var event_buf: [response_buf_size + 16]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buf, "data: {s}\n\n", .{data}) catch return false;
    stream.writeAll(event) catch return false;
    return true;
}

/// Start an SSE streaming response. Writes headers, generates tokens inline,
/// and writes each as an SSE frame. Runs synchronously on the handler thread.
fn startStream(stream: net.Stream, prompt: []const u8, is_chat: bool, max_tokens: usize) void {
    stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n" ++ security_headers ++ "Connection: keep-alive\r\n\r\n") catch return;
    generateStream(stream, prompt, nextRequestId(), std.time.timestamp(), is_chat, true, max_tokens);
}

/// Start an SSE streaming response without chat template wrapping (for /v1/completions).
fn startStreamRaw(stream: net.Stream, prompt: []const u8, max_tokens: usize) void {
    stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n" ++ security_headers ++ "Connection: keep-alive\r\n\r\n") catch return;
    generateStream(stream, prompt, nextRequestId(), std.time.timestamp(), false, false, max_tokens);
}

/// Run generation and stream tokens as SSE events in OpenAI format.
/// Always resets the cache (completions API requests are stateless).
fn generateStream(stream: net.Stream, prompt: []const u8, req_id: u64, created: i64, is_chat: bool, format_prompt: bool, max_tokens: usize) void {
    g_server.mutex.lock();
    defer g_server.mutex.unlock();
    const model = g_server.model;
    const tok = g_server.tokenizer;
    model.resetCache();

    const formatted = if (format_prompt)
        g_server.chat_template.format(g_server.allocator, null, prompt) catch prompt
    else
        prompt;
    defer if (format_prompt and formatted.ptr != prompt.ptr) g_server.allocator.free(formatted);
    const token_ids = tok.encode(formatted) catch |err| {
        std.log.err("streaming tokenizer encode failed: {}", .{err});
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

    // BOS token — required by models like Gemma to initialize state correctly
    if (g_server.bos_token_id > 0) {
        _ = model.forward(g_server.bos_token_id) catch |err| {
            std.log.err("BOS forward failed: {}", .{err});
            _ = sseWriteData(stream, "[DONE]");
            return;
        };
    }

    // Prefill — capture the last forward's return value (first generated token)
    var first_gen_token: u32 = 0;
    for (token_ids) |tid| {
        first_gen_token = model.forward(tid) catch |err| {
            if (err == error.Cancelled) {
                _ = sseWriteData(stream, "[DONE]");
                return;
            }
            break;
        };
    }

    // Generate and stream tokens
    const gen_start = std.time.milliTimestamp();
    var last: u32 = first_gen_token;
    var token_count: u32 = 0;

    // Stream the first generated token (from last prefill forward)
    if (token_ids.len > 0 and !g_server.isEog(first_gen_token)) {
        const decoded = tok.decode(&[_]u32{first_gen_token}) catch "";
        defer if (decoded.len > 0) g_server.allocator.free(decoded);

        if (decoded.len > 0) {
            const escaped = jsonEscape(g_server.allocator, decoded) catch "";
            defer if (escaped.len > 0 and escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);

            const chunk = if (is_chat)
                std.fmt.bufPrint(&chunk_buf,
                    \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
                , .{ req_id, created, g_server.model_name, escaped })
            else
                std.fmt.bufPrint(&chunk_buf,
                    \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":null}}]}}
                , .{ req_id, created, g_server.model_name, escaped });

            if (chunk) |c| {
                if (!sseWriteData(stream, c)) {
                    logGeneration(0, 0, 0);
                    return;
                }
            } else |_| {}
        }
        last = first_gen_token;
        token_count = 1;
    }

    for (0..max_tokens -| 1) |_| {
        if (token_ids.len == 0 or (token_count == 0 and g_server.isEog(first_gen_token))) break;
        const next = model.forward(last) catch break;
        if (g_server.isEog(next)) break;

        const decoded = tok.decode(&[_]u32{next}) catch continue;
        defer g_server.allocator.free(decoded);

        const escaped = jsonEscape(g_server.allocator, decoded) catch continue;
        defer if (escaped.ptr != decoded.ptr) g_server.allocator.free(escaped);

        const chunk = if (is_chat)
            std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped })
        else
            std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"text":"{s}","index":0,"finish_reason":null}}]}}
            , .{ req_id, created, g_server.model_name, escaped });

        if (chunk) |c| {
            if (!sseWriteData(stream, c)) break; // client disconnected
        } else |_| {}

        last = next;
        token_count += 1;
    }

    // Send final chunk with finish_reason
    const id_prefix: []const u8 = if (is_chat) "chatcmpl" else "cmpl";
    const obj_type: []const u8 = if (is_chat) "chat.completion.chunk" else "text_completion";
    const delta_or_text: []const u8 = if (is_chat)
        \\"delta":{}
    else
        \\"text":""
    ;
    const final = std.fmt.bufPrint(&chunk_buf,
        \\{{"id":"{s}-{d}","object":"{s}","created":{d},"model":"{s}","choices":[{{"index":0,{s},"finish_reason":"stop"}}]}}
    , .{ id_prefix, req_id, obj_type, created, g_server.model_name, delta_or_text }) catch "";
    if (final.len > 0) _ = sseWriteData(stream, final);

    _ = sseWriteData(stream, "[DONE]");

    const gen_end = std.time.milliTimestamp();
    const time_ms = @as(u64, @intCast(@max(gen_end - gen_start, 0)));
    const tps: f32 = if (time_ms > 0) @as(f32, @floatFromInt(token_count)) / (@as(f32, @floatFromInt(time_ms)) / 1000.0) else 0.0;
    logGeneration(token_count, time_ms, tps);
}

/// Check if a JSON body contains `"field": true`.
fn extractBoolField(json: []const u8, field: []const u8) bool {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return false;
    const pos = std.mem.indexOf(u8, json, needle) orelse return false;
    var i = pos + needle.len;
    while (i < json.len and (json[i] == ':' or json[i] == ' ')) : (i += 1) {}
    return i + 4 <= json.len and std.mem.eql(u8, json[i..][0..4], "true");
}

/// Extract an integer field value from a JSON body (e.g., `"max_tokens": 128`).
fn extractIntField(json: []const u8, field: []const u8) ?usize {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
    const pos = std.mem.indexOf(u8, json, needle) orelse return null;
    var i = pos + needle.len;
    while (i < json.len and (json[i] == ':' or json[i] == ' ')) : (i += 1) {}
    const start = i;
    while (i < json.len and json[i] >= '0' and json[i] <= '9') : (i += 1) {}
    if (i == start) return null;
    return std.fmt.parseInt(usize, json[start..i], 10) catch null;
}

/// Scan past a JSON string value starting at `start` (just after the opening `"`).
/// Returns the index of the closing `"`, or `json.len` if unterminated.
fn findJsonStringEnd(json: []const u8, start: usize) usize {
    var i = start;
    while (i < json.len and json[i] != '"') : (i += 1) {
        if (json[i] == '\\' and i + 1 < json.len) i += 1;
    }
    return i;
}

/// Skip to the start of a JSON string value after a field key match.
/// Returns the index just past the opening `"`, or null if no string follows.
fn skipToJsonValue(json: []const u8, pos: usize) ?usize {
    var i = pos;
    while (i < json.len and (json[i] == ':' or json[i] == ' ')) : (i += 1) {}
    if (i >= json.len or json[i] != '"') return null;
    return i + 1;
}

fn extractField(json: []const u8, field: []const u8) ?[]const u8 {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
    const pos = std.mem.indexOf(u8, json, needle) orelse return null;
    const start = skipToJsonValue(json, pos + needle.len) orelse return null;
    const end = findJsonStringEnd(json, start);
    return json[start..end];
}

fn extractLastMessage(json: []const u8) ?[]const u8 {
    var last: ?[]const u8 = null;
    var pos: usize = 0;
    const content_key = "\"content\"";
    while (pos < json.len) {
        const idx = std.mem.indexOf(u8, json[pos..], content_key) orelse break;
        const abs = pos + idx + content_key.len;
        if (skipToJsonValue(json, abs)) |start| {
            const end = findJsonStringEnd(json, start);
            last = json[start..end];
        }
        pos = abs + 1;
    }
    return last;
}

fn extractFormField(body: []const u8, field: []const u8) ?[]const u8 {
    var parts = std.mem.splitScalar(u8, body, '&');
    while (parts.next()) |part| {
        const eq = std.mem.indexOf(u8, part, "=") orelse continue;
        if (std.mem.eql(u8, part[0..eq], field)) return part[eq + 1 ..];
    }
    return null;
}

fn urlDecode(allocator: Allocator, input: []const u8) ![]u8 {
    // Decoded output is always <= input length (%XX → 1 byte).
    // Pre-allocate with ensureTotalCapacity to avoid per-byte realloc.
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(allocator);
    try result.ensureTotalCapacity(allocator, input.len);
    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '+') {
            result.appendAssumeCapacity(' ');
            i += 1;
        } else if (input[i] == '%' and i + 2 < input.len) {
            const hi = hexVal(input[i + 1]);
            const lo = hexVal(input[i + 2]);
            if (hi != null and lo != null) {
                result.appendAssumeCapacity(hi.? * 16 + lo.?);
                i += 3;
            } else {
                result.appendAssumeCapacity(input[i]);
                i += 1;
            }
        } else {
            result.appendAssumeCapacity(input[i]);
            i += 1;
        }
    }
    return result.toOwnedSlice(allocator);
}

fn hexVal(c: u8) ?u8 {
    if (c >= '0' and c <= '9') return c - '0';
    if (c >= 'a' and c <= 'f') return c - 'a' + 10;
    if (c >= 'A' and c <= 'F') return c - 'A' + 10;
    return null;
}

/// Generic character escaper: for each byte, `escape_fn` returns a replacement
/// string or null (pass through). Used by jsonEscape and htmlEscape.
fn escapeWith(allocator: Allocator, input: []const u8, comptime escape_fn: fn (u8) ?[]const u8) ![]u8 {
    // First pass: count output size to avoid reallocations
    var out_len: usize = 0;
    var needs_escape = false;
    for (input) |c| {
        if (escape_fn(c)) |replacement| {
            out_len += replacement.len;
            needs_escape = true;
        } else {
            out_len += 1;
        }
    }
    if (!needs_escape) return allocator.dupe(u8, input);

    // Second pass: write directly into pre-sized buffer
    const buf = try allocator.alloc(u8, out_len);
    var pos: usize = 0;
    for (input) |c| {
        if (escape_fn(c)) |replacement| {
            @memcpy(buf[pos..][0..replacement.len], replacement);
            pos += replacement.len;
        } else {
            buf[pos] = c;
            pos += 1;
        }
    }
    return buf;
}

fn jsonEscapeChar(c: u8) ?[]const u8 {
    return switch (c) {
        '"' => "\\\"",
        '\\' => "\\\\",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
        0x08 => "\\b",
        0x0C => "\\f",
        // Remaining control chars (0x00-0x07, 0x0E-0x1F) are escaped as
        // \\uXXXX by the caller via escapeWith's fallback. We handle them
        // here with a fixed-table approach for the most common ones.
        0x00 => "\\u0000",
        0x01 => "\\u0001",
        0x02 => "\\u0002",
        0x03 => "\\u0003",
        0x04 => "\\u0004",
        0x05 => "\\u0005",
        0x06 => "\\u0006",
        0x07 => "\\u0007",
        0x0B => "\\u000b",
        0x0E => "\\u000e",
        0x0F => "\\u000f",
        0x10 => "\\u0010",
        0x11 => "\\u0011",
        0x12 => "\\u0012",
        0x13 => "\\u0013",
        0x14 => "\\u0014",
        0x15 => "\\u0015",
        0x16 => "\\u0016",
        0x17 => "\\u0017",
        0x18 => "\\u0018",
        0x19 => "\\u0019",
        0x1A => "\\u001a",
        0x1B => "\\u001b",
        0x1C => "\\u001c",
        0x1D => "\\u001d",
        0x1E => "\\u001e",
        0x1F => "\\u001f",
        else => null,
    };
}

fn htmlEscapeChar(c: u8) ?[]const u8 {
    return switch (c) {
        '<' => "&lt;",
        '>' => "&gt;",
        '&' => "&amp;",
        '"' => "&quot;",
        else => null,
    };
}

fn jsonEscape(allocator: Allocator, input: []const u8) ![]u8 {
    return escapeWith(allocator, input, jsonEscapeChar);
}

fn htmlEscape(allocator: Allocator, input: []const u8) ![]u8 {
    return escapeWith(allocator, input, htmlEscapeChar);
}

// ── Connection handler & server entry point ─────────────────────

fn handleConnection(stream: net.Stream) void {
    _ = g_server.active_connections.fetchAdd(1, .monotonic);
    defer {
        _ = g_server.active_connections.fetchSub(1, .monotonic);
        stream.close();
    }
    var buf: [http_buf_size]u8 = undefined;
    const req = readHttpRequest(stream, &buf) orelse return;
    handleRequest(stream, req);
}

/// Start the HTTP server with OpenAI-compatible API endpoints.
/// Blocks until the server shuts down (via Ctrl+C).
///
/// Parameters:
///   - allocator: General-purpose allocator for request handling.
///   - model: Initialized model for inference.
///   - tok: Initialized tokenizer for encode/decode.
///   - chat_tmpl: Chat template for prompt formatting.
///   - model_name: Display name for /v1/models endpoint.
///   - backend_name: Backend name for /v1/models endpoint.
///   - port: TCP port to listen on.
///   - bos_token_id: Beginning-of-sequence token ID (0 to skip).
///   - eog_ids: End-of-generation token IDs (slice of up to max_eog_ids).
///   - eog_len: Number of valid entries in eog_ids.
pub fn run(allocator: Allocator, model: *Model, tok: *Tokenizer, chat_tmpl: ChatTemplate, model_name: []const u8, backend_name: []const u8, port: u16, bos_token_id: u32, eog_ids: [max_eog_ids]u32, eog_len: usize) !void {
    // Stack-allocate the Server struct. This is safe because run() blocks
    // until the server shuts down, so the frame stays alive.
    var server = Server{
        .model = model,
        .tokenizer = tok,
        .chat_template = chat_tmpl,
        .model_name = model_name,
        .backend_name = backend_name,
        .allocator = allocator,
        .bos_token_id = bos_token_id,
        .eog_ids = eog_ids,
        .eog_len = eog_len,
    };
    server.start_time = std.time.timestamp();
    g_server = &server;

    const address = net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var tcp = address.listen(.{ .reuse_address = true }) catch {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Error: port {d} already in use. Try:\n  agave model.gguf --serve --port {d}\n", .{ port, port + 1 }) catch "";
        std.fs.File.stderr().writeAll(msg) catch {};
        return error.ListenError;
    };
    defer tcp.deinit();

    const t = getTimeComponents();
    var buf: [512]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "\n[{d:0>2}:{d:0>2}:{d:0>2}] agave server started on http://0.0.0.0:{d} (http://localhost:{d})\n  model={s} backend={s}\nPress Ctrl+C to stop\n", .{ t.hours, t.minutes, t.seconds, port, port, model_name, backend_name }) catch "";
    std.fs.File.stdout().writeAll(msg) catch {};

    // Install graceful shutdown handlers for SIGTERM and SIGINT.
    // Sets shutdown flag and signals scheduler to stop.
    const handler = struct {
        fn handle(_: c_int) callconv(.c) void {
            g_server.shutdown_requested.store(true, .release);
            if (g_server.scheduler_shutdown.load(.acquire) == false) {
                g_server.scheduler_shutdown.store(true, .release);
            }
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
        const conn = tcp.accept() catch |err| {
            if (g_server.shutdown_requested.load(.acquire)) break;
            std.log.err("Accept failed: {}", .{err});
            continue;
        };
        if (g_server.active_connections.load(.monotonic) >= max_concurrent_connections) {
            conn.stream.close();
            continue;
        }
        const thread = std.Thread.spawn(.{}, handleConnection, .{conn.stream}) catch {
            conn.stream.close();
            continue;
        };
        thread.detach();
    }

    // Drain active connections (wait up to 30 seconds)
    const drain_timeout_sec: i64 = 30;
    const drain_start = std.time.timestamp();
    const active_count = g_server.active_connections.load(.acquire);
    if (active_count > 0) {
        std.log.info("Draining {d} active connections...", .{active_count});
    }

    while (g_server.active_connections.load(.acquire) > 0) {
        const elapsed = std.time.timestamp() - drain_start;
        if (elapsed > drain_timeout_sec) {
            std.log.warn("Drain timeout after {d}s, forcing shutdown", .{elapsed});
            break;
        }
        std.Thread.sleep(100 * std.time.ns_per_ms);
    }

    std.log.info("Graceful shutdown complete", .{});
}
