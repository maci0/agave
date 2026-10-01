//! HTTP/1.1 wire protocol: socket wrapper, request parsing, header lookup,
//! host and origin policy, and the response header policy constants.
//!
//! Owns no server state. Everything here is a pure function of its arguments
//! (plus the raw socket), so the request and response envelope can be read and
//! tested without the model, scheduler, or auth policy in `server.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Lightweight wrapper providing writeAll/read/close over a raw socket fd.
pub const TcpStream = struct {
    handle: std.posix.fd_t,

    /// Writes the entire contents of `data` to the socket, retrying on EINTR.
    pub fn writeAll(self: TcpStream, data: []const u8) !void {
        var written: usize = 0;
        while (written < data.len) {
            const n = std.posix.system.write(self.handle, data[written..].ptr, data[written..].len);
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                return error.BrokenPipe;
            }
            written += @intCast(n);
        }
    }

    /// Reads up to `buf.len` bytes from the socket, retrying on EINTR.
    /// Returns the number of bytes read (0 signals EOF).
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

    /// Closes the underlying socket file descriptor.
    pub fn close(self: TcpStream) void {
        _ = std.c.close(self.handle);
    }
};

/// CORS preflight cache duration in seconds (24 hours).
pub const cors_max_age_seconds = "86400";

/// Characters unsafe for direct embedding in JSON string values or HTML contexts.
pub fn isUnsafeJsonChar(c: u8) bool {
    return c == '"' or c == '\\' or c < 0x20 or c == '<' or c == '>' or c == '&';
}

/// CORS allow-origin headers. Always empty: the embedded UI is same-origin
/// (no CORS needed). Wildcard ACAO with no API key enabled cross-site
/// read/CSRF against local servers (CWE-942); authenticated mode already
/// omitted CORS. Cross-origin browser clients should use a reverse proxy.
pub fn corsHeaders() []const u8 {
    return "";
}

/// Return the first header value for `name` (case-insensitive), trimmed.
/// Returns null when missing or when the header appears more than once.
pub fn getHeaderValue(headers: []const u8, name: []const u8) ?[]const u8 {
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    var found: ?[]const u8 = null;
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        if (colon == name.len and std.ascii.eqlIgnoreCase(line[0..name.len], name)) {
            if (found != null) return null;
            found = std.mem.trim(u8, line[colon + 1 ..], " \t");
        }
    }
    return found;
}

/// True when `Accept-Encoding` lists `name` with q > 0.
pub fn acceptsEncoding(headers: []const u8, name: []const u8) bool {
    const ae = getHeaderValue(headers, "accept-encoding") orelse return false;
    var tokens = std.mem.splitScalar(u8, ae, ',');
    while (tokens.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t");
        if (item.len == 0) continue;
        var parts = std.mem.splitScalar(u8, item, ';');
        const coding = std.mem.trim(u8, parts.next() orelse continue, " \t");
        var q: f32 = 1.0;
        while (parts.next()) |param| {
            const p = std.mem.trim(u8, param, " \t");
            if (p.len >= 2 and std.ascii.eqlIgnoreCase(p[0..2], "q=")) {
                q = std.fmt.parseFloat(f32, std.mem.trim(u8, p[2..], " \t")) catch 0;
            }
        }
        if (std.ascii.eqlIgnoreCase(coding, name) and q > 0) return true;
    }
    return false;
}

/// True when `If-None-Match` is `*` or lists `etag` (strong or weak).
pub fn ifNoneMatch(headers: []const u8, etag: []const u8) bool {
    const inm = getHeaderValue(headers, "if-none-match") orelse return false;
    const trimmed = std.mem.trim(u8, inm, " \t");
    if (std.mem.eql(u8, trimmed, "*")) return true;
    var iter = std.mem.splitScalar(u8, trimmed, ',');
    while (iter.next()) |part| {
        var tag = std.mem.trim(u8, part, " \t");
        if (std.mem.startsWith(u8, tag, "W/")) {
            tag = std.mem.trim(u8, tag[2..], " \t");
        }
        if (std.mem.eql(u8, tag, etag)) return true;
    }
    return false;
}

/// Copy `raw` into `buf` when it is a safe correlation token (alnum, `-`, `_`, `.`).
/// Returns 0 (ignore) on empty, oversized, or illegal characters (CWE-117).
pub fn sanitizeClientRequestId(raw: []const u8, buf: []u8) usize {
    if (raw.len == 0 or raw.len > buf.len) return 0;
    for (raw, 0..) |c, i| {
        const ok = std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.';
        if (!ok) return 0;
        buf[i] = c;
    }
    return raw.len;
}

/// True when `Origin` is `http(s)://` + Host (no path/userinfo). CWE-346.
pub fn originMatchesHost(origin: []const u8, host: []const u8) bool {
    const rest = if (std.mem.startsWith(u8, origin, "https://"))
        origin["https://".len..]
    else if (std.mem.startsWith(u8, origin, "http://"))
        origin["http://".len..]
    else
        return false;
    if (rest.len == 0) return false;
    if (std.mem.indexOfAny(u8, rest, "/@?#")) |_| return false;
    return std.ascii.eqlIgnoreCase(rest, host);
}

/// True when the `Sec-Fetch-Site` header marks a browser cross-site request.
/// Every current browser attaches it to every request, including ones that
/// carry no `Origin` (a cross-site HTML form POST with `enctype=text/plain`
/// sends `Origin` in some browsers and omits it in others, and a cross-site
/// form GET never carries one), so `cross-site` is a cross-site drive-by even
/// when `Origin` is absent (CWE-352). `same-site` is a different registrable
/// domain, `none` is a user-typed navigation, and `same-origin` is the
/// embedded UI. A missing header is a non-browser client (curl, SDKs, health
/// probes) with no ambient credentials to abuse, so it stays allowed.
pub fn isCrossSiteFetch(headers: []const u8) bool {
    const site = getHeaderValue(headers, "sec-fetch-site") orelse return false;
    return std.ascii.eqlIgnoreCase(site, "cross-site");
}

/// Hostname from a Host header: strip `:port` or `[ipv6]:port`.
pub fn hostnameFromHost(host: []const u8) []const u8 {
    if (host.len == 0) return host;
    if (host[0] == '[') {
        if (std.mem.indexOfScalar(u8, host, ']')) |end| {
            if (end > 1) return host[1..end];
        }
        return host;
    }
    if (std.mem.lastIndexOfScalar(u8, host, ':')) |colon| {
        const port = host[colon + 1 ..];
        if (port.len > 0) {
            for (port) |c| {
                if (c < '0' or c > '9') return host;
            }
            return host[0..colon];
        }
    }
    return host;
}

/// True when Host is loopback (`localhost`, `::1`, `127.0.0.0/8`).
/// DNS rebinding points a public name at 127.0.0.1 so Origin equals Host and
/// the same-origin check would allow the request (CWE-350).
pub fn isLoopbackHttpHost(host: []const u8) bool {
    var name = hostnameFromHost(host);
    if (name.len > 0 and name[name.len - 1] == '.') name = name[0 .. name.len - 1];
    if (name.len == 0) return false;
    if (std.ascii.eqlIgnoreCase(name, "localhost")) return true;
    if (std.mem.eql(u8, name, "::1")) return true;
    if (name.len < 5 or !std.mem.startsWith(u8, name, "127.")) return false;
    // Remaining three octets of 127.0.0.0/8.
    var octets: u32 = 1;
    var acc: u32 = 0;
    var saw_digit = false;
    for (name["127.".len..]) |c| {
        if (c == '.') {
            if (!saw_digit or acc > 255 or octets >= 4) return false;
            octets += 1;
            acc = 0;
            saw_digit = false;
        } else if (c >= '0' and c <= '9') {
            acc = std.math.mul(u32, acc, 10) catch return false;
            acc = std.math.add(u32, acc, c - '0') catch return false;
            if (acc > 255) return false;
            saw_digit = true;
        } else {
            return false;
        }
    }
    return saw_digit and octets == 3 and acc <= 255;
}

pub const gzip_id1: u8 = 0x1f;

pub const gzip_id2: u8 = 0x8b;

/// gzip `src` at best effort. Caller owns the slice.
/// Returns `error.NoCompressionGain` when the result is not smaller than `src`.
pub fn gzipAlloc(allocator: Allocator, src: []const u8) ![]u8 {
    var aw = try std.Io.Writer.Allocating.initCapacity(allocator, @max(src.len / 2, 16));
    errdefer aw.deinit();
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var compressor = try std.compress.flate.Compress.init(&aw.writer, &window, .gzip, .best);
    try compressor.writer.writeAll(src);
    try compressor.finish();
    const out = try aw.toOwnedSlice();
    if (out.len >= src.len) {
        allocator.free(out);
        return error.NoCompressionGain;
    }
    return out;
}

/// Parsed HTTP request. Slices point into the read buffer.
pub const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    /// Query string without leading `?` (empty when absent).
    query: []const u8,
    headers: []const u8,
    body: []const u8,
};

/// Split a raw request-target into path and query (without leading `?`).
pub fn splitPathQuery(raw_path: []const u8) struct { path: []const u8, query: []const u8 } {
    if (std.mem.indexOf(u8, raw_path, "?")) |q| {
        return .{ .path = raw_path[0..q], .query = raw_path[q + 1 ..] };
    }
    return .{ .path = raw_path, .query = "" };
}

/// Parse an HTTP/1.1 request-line body (`METHOD SP request-target SP HTTP-version`).
/// Returns null if the line lacks two spaces (malformed).
pub fn parseRequestLine(req_line: []const u8) ?struct { method: []const u8, path: []const u8, query: []const u8 } {
    const sp1 = std.mem.indexOf(u8, req_line, " ") orelse return null;
    const method = req_line[0..sp1];
    const rest = req_line[sp1 + 1 ..];
    const sp2 = std.mem.indexOf(u8, rest, " ") orelse return null;
    const raw_path = rest[0..sp2];
    const pq = splitPathQuery(raw_path);
    return .{ .method = method, .path = pq.path, .query = pq.query };
}

/// Extract a single query parameter value (`key=value`). Returns null if absent.
pub fn extractQueryParam(query: []const u8, key: []const u8) ?[]const u8 {
    var iter = std.mem.splitScalar(u8, query, '&');
    while (iter.next()) |pair| {
        if (pair.len == 0) continue;
        if (std.mem.indexOf(u8, pair, "=")) |eq| {
            if (std.mem.eql(u8, pair[0..eq], key)) return pair[eq + 1 ..];
        } else if (std.mem.eql(u8, pair, key)) {
            return "";
        }
    }
    return null;
}

/// Result of reading an HTTP request, distinguishes malformed requests from
/// oversized bodies so the caller can return the correct status code.
/// Connection failures (`connection_closed`, `read_error`) are kept separate
/// from `malformed` so logs and client-error metrics do not blame the request
/// content when the peer vanished before sending a complete request.
pub const HttpReadResult = union(enum) {
    ok: HttpRequest,
    /// Request line or headers were unreadable. Payload is the request path
    /// when the request line parsed, "" otherwise, so the caller can pick the
    /// error envelope of the route the client addressed.
    malformed: []const u8,
    /// Payload is the request path, so an oversized body to `/v1/messages`
    /// still gets the Anthropic error shape.
    body_too_large: []const u8,
    /// Peer closed the connection before a complete request arrived
    /// (probes, port scans, health checks dialing the raw port).
    connection_closed,
    /// Socket read failed (timeout or reset) before a complete request arrived.
    read_error,
};

/// Check whether a given header name is present in raw HTTP headers.
pub fn hasHeader(headers: []const u8, name: []const u8) bool {
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
pub fn parseContentLength(headers: []const u8) ?usize {
    const header_name = "content-length";
    var iter = std.mem.splitSequence(u8, headers, "\r\n");
    var found: ?usize = null;
    while (iter.next()) |line| {
        const colon = std.mem.indexOf(u8, line, ":") orelse continue;
        if (colon == header_name.len and std.ascii.eqlIgnoreCase(line[0..header_name.len], header_name)) {
            // OWS is SP and HTAB (RFC 9110 5.6.3); trimming SP only rejected a
            // tab-separated value every other header parser here accepts.
            const val = std.fmt.parseInt(usize, std.mem.trim(u8, line[colon + 1 ..], " \t"), 10) catch return null;
            if (found != null) return null; // Duplicate Content-Length, reject
            found = val;
        }
    }
    return found orelse 0;
}

/// Read a complete HTTP/1.1 request from a TCP stream. Returns `.malformed`
/// on parse errors, `.connection_closed`/`.read_error` when the peer vanished
/// or the socket failed before a complete request arrived, `.body_too_large`
/// when Content-Length exceeds `max_body` (RFC 7231 §6.5.11).
/// Both failure variants carry the request path (empty when the request line
/// did not parse) so the caller can answer in the addressed route's envelope.
pub fn readHttpRequest(stream: TcpStream, buf: []u8, max_body: usize) HttpReadResult {
    var total: usize = 0;
    var hdr_end: usize = undefined;

    // Read until we have complete headers (\r\n\r\n).
    // Scan only the newly-received region (plus 3-byte overlap for split boundary).
    while (total < buf.len) {
        const n = stream.read(buf[total..]) catch return .read_error;
        if (n == 0) return .connection_closed;
        const scan_start = if (total >= 3) total - 3 else 0;
        total += n;
        if (std.mem.indexOf(u8, buf[scan_start..total], "\r\n\r\n")) |pos| {
            hdr_end = scan_start + pos;
            break;
        }
    } else return .{ .malformed = "" };

    // Parse request line: "GET /path HTTP/1.1"
    const req_line_end = std.mem.indexOf(u8, buf[0..hdr_end], "\r\n") orelse return .{ .malformed = "" };
    const req_line = buf[0..req_line_end];
    const parsed_line = parseRequestLine(req_line) orelse return .{ .malformed = "" };
    const method = parsed_line.method;
    const path = parsed_line.path;
    const query = parsed_line.query;

    // Parse Content-Length (null = duplicate headers, reject per RFC 7230)
    const headers = buf[req_line_end + 2 .. hdr_end];

    // Reject Transfer-Encoding, this server only supports identity encoding.
    // Accepting chunked requests without parsing them enables HTTP request
    // smuggling (CWE-444) when behind a reverse proxy.
    if (hasHeader(headers, "transfer-encoding")) return .{ .malformed = path };

    const content_length = parseContentLength(headers) orelse return .{ .malformed = path };
    const body_start = hdr_end + 4;

    // Read remaining body bytes if needed
    if (content_length > 0) {
        if (content_length > max_body) return .{ .body_too_large = path };
        const body_end = std.math.add(usize, body_start, content_length) catch return .{ .body_too_large = path };
        if (body_end > buf.len) return .{ .body_too_large = path };
        while (total < body_end) {
            const n = stream.read(buf[total..body_end]) catch return .read_error;
            if (n == 0) return .connection_closed;
            total += n;
        }
        return .{ .ok = .{ .method = method, .path = path, .query = query, .headers = headers, .body = buf[body_start..body_end] } };
    }

    return .{ .ok = .{ .method = method, .path = path, .query = query, .headers = headers, .body = "" } };
}

/// Security headers without Cache-Control. API/SSE/error responses append
/// `Cache-Control: no-store` via `security_headers`. The chat UI document
/// uses `private, no-cache` plus ETag instead (see `sendHtmlPage`).
pub const security_headers_base =
    "X-Content-Type-Options: nosniff\r\n" ++
    "X-Frame-Options: DENY\r\n" ++
    "Referrer-Policy: no-referrer\r\n" ++
    "Strict-Transport-Security: max-age=31536000; includeSubDomains\r\n" ++
    "Permissions-Policy: geolocation=(), microphone=(), camera=(), accelerometer=(), gyroscope=()\r\n" ++
    "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; object-src 'none'; worker-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'\r\n";

/// Common security headers appended to every API/SSE/error response.
pub const security_headers = security_headers_base ++ "Cache-Control: no-store\r\n";

pub const document_cache_headers =
    "Cache-Control: private, no-cache\r\n" ++
    "Vary: Accept-Encoding\r\n";
