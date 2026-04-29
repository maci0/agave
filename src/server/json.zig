//! JSON field extraction, encoding, and form-parsing utilities for the HTTP server.
//! Pure functions with no server state dependencies — extracted from server.zig
//! to keep each module single-concern.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Message = @import("../chat_template.zig").Message;

// ── Constants ───────────────────────────────────────────────────

const extract_field_buf_size: usize = 256;
/// Maximum messages extractable from a single API request body.
const max_api_messages: usize = 128;
/// Maximum valid sampling temperature (prevents numerical instability in softmax).
const max_temperature: f32 = 100.0;
/// Maximum valid top_k value (larger values are clamped, not rejected).
const max_top_k: u32 = 1024;

// ── Types ───────────────────────────────────────────────────────

/// Per-request sampling parameters. Defaults match greedy decoding.
/// Values are clamped to safe ranges by `parseSampling()`.
pub const SamplingParams = struct {
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
};

/// Result of extracting messages from an OpenAI/Anthropic-format JSON body.
pub const ExtractedMessages = struct {
    messages: []Message,
    system: ?[]const u8,

    pub fn deinit(self: ExtractedMessages, allocator: Allocator) void {
        for (self.messages) |msg| allocator.free(@constCast(msg.content));
        if (self.system) |sys| allocator.free(@constCast(sys));
        allocator.free(self.messages);
    }
};

// ── JSON field extraction ───────────────────────────────────────

/// Check if a JSON body contains `"field": true`.
pub fn extractBoolField(json: []const u8, field: []const u8) bool {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return false;
    var search_start: usize = 0;
    const i = findFieldValuePos(json, needle, &search_start) orelse return false;
    return i + 4 <= json.len and std.mem.eql(u8, json[i..][0..4], "true");
}

/// Extract an integer field value from a JSON body (e.g., `"max_tokens": 128`).
pub fn extractIntField(json: []const u8, field: []const u8) ?usize {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
    var search_start: usize = 0;
    while (findFieldValuePos(json, needle, &search_start)) |val_pos| {
        var end = val_pos;
        while (end < json.len and json[end] >= '0' and json[end] <= '9') : (end += 1) {}
        if (end == val_pos) continue;
        return std.fmt.parseInt(usize, json[val_pos..end], 10) catch continue;
    }
    return null;
}

/// Extract a floating-point field value from a JSON body (e.g., `"temperature": 0.7`).
pub fn extractFloatField(json: []const u8, field: []const u8) ?f32 {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
    var search_start: usize = 0;
    while (findFieldValuePos(json, needle, &search_start)) |val_pos| {
        var end = val_pos;
        while (end < json.len and (json[end] == '.' or (json[end] >= '0' and json[end] <= '9') or json[end] == '-' or json[end] == 'e' or json[end] == 'E' or json[end] == '+')) : (end += 1) {}
        if (end == val_pos) continue;
        return std.fmt.parseFloat(f32, json[val_pos..end]) catch continue;
    }
    return null;
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
/// Returns the index just past the opening `"`, or null if no colon+string follows.
/// Requires a colon to distinguish JSON keys from false matches inside values.
fn skipToJsonValue(json: []const u8, pos: usize) ?usize {
    var i = pos;
    var saw_colon = false;
    while (i < json.len and (json[i] == ':' or json[i] == ' ')) : (i += 1) {
        if (json[i] == ':') saw_colon = true;
    }
    if (!saw_colon) return null;
    if (i >= json.len or json[i] != '"') return null;
    return i + 1;
}

/// Locate the start of a non-string JSON value for the given field key.
/// Skips false matches where the needle appears inside a string value (no colon follows).
/// Advances `search_start` past each match for retry on parse failure.
/// Returns the index of the first non-whitespace character after the colon, or null.
fn findFieldValuePos(json_buf: []const u8, needle: []const u8, search_start: *usize) ?usize {
    while (search_start.* < json_buf.len) {
        const rel = std.mem.indexOf(u8, json_buf[search_start.*..], needle) orelse return null;
        const after = search_start.* + rel + needle.len;
        search_start.* = after;
        var i = after;
        var saw_colon = false;
        while (i < json_buf.len and (json_buf[i] == ':' or json_buf[i] == ' ')) : (i += 1) {
            if (json_buf[i] == ':') saw_colon = true;
        }
        if (saw_colon) return i;
    }
    return null;
}

/// Extract the string value of a JSON field by key name.
/// Returns the unescaped content between quotes, or null if the field is missing.
/// Handles false matches inside string values by requiring a colon after the key.
pub fn extractField(json: []const u8, field: []const u8) ?[]const u8 {
    var buf: [extract_field_buf_size]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
    // Retry loop: the needle may match inside a JSON string value (e.g.,
    // content containing "\"system\""). skipToJsonValue rejects those
    // (no colon follows), so we advance past false matches.
    var search_start: usize = 0;
    while (search_start < json.len) {
        const rel = std.mem.indexOf(u8, json[search_start..], needle) orelse return null;
        const after = search_start + rel + needle.len;
        const start = skipToJsonValue(json, after) orelse {
            search_start = after;
            continue;
        };
        const end = findJsonStringEnd(json, start);
        return json[start..end];
    }
    return null;
}

/// Extract the last "content" field value from a JSON body.
/// Scans for all "content" keys and returns the final match, or null if none found.
pub fn extractLastMessage(json: []const u8) ?[]const u8 {
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

/// Parse and clamp sampling parameters from a JSON request body.
/// Negative temperature is treated as 0 (greedy). top_p is clamped to [0, 1].
pub fn parseSampling(body: []const u8) SamplingParams {
    const raw_temp = extractFloatField(body, "temperature") orelse 0;
    const raw_top_p = extractFloatField(body, "top_p") orelse 1.0;
    const raw_top_k = extractIntField(body, "top_k") orelse 0;
    return .{
        .temperature = if (std.math.isFinite(raw_temp)) std.math.clamp(raw_temp, 0, max_temperature) else 0,
        .top_k = @intCast(@min(raw_top_k, max_top_k)),
        .top_p = if (std.math.isFinite(raw_top_p)) std.math.clamp(raw_top_p, 0, 1.0) else 1.0,
    };
}

/// Extract all messages from an OpenAI-format `"messages"` JSON array.
/// Returns conversation messages (user/assistant) and an optional system message.
/// Message content slices point into the original JSON body — valid for the request lifetime.
pub fn extractMessages(json: []const u8, allocator: Allocator) ?ExtractedMessages {
    const msgs_key = "\"messages\"";
    const msgs_pos = std.mem.indexOf(u8, json, msgs_key) orelse return null;
    var i = msgs_pos + msgs_key.len;

    // Skip to array start
    while (i < json.len and (json[i] == ':' or json[i] == ' ' or json[i] == '\n' or json[i] == '\r' or json[i] == '\t')) : (i += 1) {}
    if (i >= json.len or json[i] != '[') return null;
    i += 1;

    var messages_buf: [max_api_messages]Message = undefined;
    var count: usize = 0;
    var system_msg: ?[]const u8 = null;

    while (i < json.len and count < max_api_messages) {
        // Skip whitespace and commas
        while (i < json.len and (json[i] == ' ' or json[i] == '\n' or json[i] == '\r' or json[i] == '\t' or json[i] == ',')) : (i += 1) {}
        if (i >= json.len or json[i] == ']') break;
        if (json[i] != '{') break;

        // Find end of this object (handle nested braces and strings)
        var depth: usize = 1;
        const obj_start = i + 1;
        i += 1;
        while (i < json.len and depth > 0) : (i += 1) {
            if (json[i] == '{') {
                depth += 1;
            } else if (json[i] == '}') {
                depth -= 1;
            } else if (json[i] == '"') {
                i += 1;
                while (i < json.len and json[i] != '"') : (i += 1) {
                    if (json[i] == '\\' and i + 1 < json.len) i += 1;
                }
            }
        }
        // Guard against malformed JSON (unmatched brace or empty object).
        const obj_end = if (i > 0) i - 1 else 0;
        if (obj_end < obj_start) continue;
        const obj_slice = json[obj_start..obj_end];

        const role_str = extractField(obj_slice, "role") orelse continue;
        const content = extractField(obj_slice, "content") orelse continue;
        const owned_content = jsonUnescapeOwned(allocator, content) catch continue;

        if (std.mem.eql(u8, role_str, "system")) {
            if (system_msg) |prev_sys| allocator.free(@constCast(prev_sys));
            system_msg = owned_content;
        } else if (std.mem.eql(u8, role_str, "user")) {
            messages_buf[count] = .{ .role = .user, .content = owned_content };
            count += 1;
        } else if (std.mem.eql(u8, role_str, "assistant")) {
            messages_buf[count] = .{ .role = .assistant, .content = owned_content };
            count += 1;
        } else {
            allocator.free(owned_content);
        }
    }

    if (count == 0) {
        if (system_msg) |sys| allocator.free(@constCast(sys));
        return null;
    }

    const messages = allocator.alloc(Message, count) catch {
        for (messages_buf[0..count]) |msg| allocator.free(@constCast(msg.content));
        if (system_msg) |sys| allocator.free(@constCast(sys));
        return null;
    };
    @memcpy(messages, messages_buf[0..count]);

    return .{ .messages = messages, .system = system_msg };
}

// ── Form field extraction ───────────────────────────────────────

/// Extract a value from a URL-encoded form body (e.g. "key=value&key2=value2").
/// Returns the raw value string after the `=`, or null if the field is not present.
pub fn extractFormField(body: []const u8, field: []const u8) ?[]const u8 {
    var parts = std.mem.splitScalar(u8, body, '&');
    while (parts.next()) |part| {
        const eq = std.mem.indexOf(u8, part, "=") orelse continue;
        if (std.mem.eql(u8, part[0..eq], field)) return part[eq + 1 ..];
    }
    return null;
}

/// Extract a boolean value from a URL-encoded form field.
/// Returns true for values "1", "true", "yes", "on" (case-insensitive).
/// Returns false if the field is missing, empty, or has any other value.
pub fn extractFormBool(body: []const u8, field: []const u8) bool {
    const raw = extractFormField(body, field) orelse return false;
    if (raw.len == 0) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (raw.len == 4 and std.ascii.eqlIgnoreCase(raw, "true")) return true;
    if (raw.len == 3 and std.ascii.eqlIgnoreCase(raw, "yes")) return true;
    if (raw.len == 2 and std.ascii.eqlIgnoreCase(raw, "on")) return true;
    return false;
}

/// Extract a float value from a URL-encoded form field.
/// Returns null if the field is missing or cannot be parsed.
pub fn extractFormFloat(body: []const u8, field: []const u8) ?f32 {
    const raw = extractFormField(body, field) orelse return null;
    if (raw.len == 0) return null;
    return std.fmt.parseFloat(f32, raw) catch null;
}

/// Extract an unsigned integer value from a URL-encoded form field.
/// Returns null if the field is missing or cannot be parsed.
pub fn extractFormInt(body: []const u8, field: []const u8) ?usize {
    const raw = extractFormField(body, field) orelse return null;
    if (raw.len == 0) return null;
    return std.fmt.parseInt(usize, raw, 10) catch null;
}

/// Parse and clamp sampling parameters from a URL-encoded form body.
/// Negative temperature is treated as 0 (greedy). top_p is clamped to [0, 1].
pub fn parseFormSampling(body: []const u8) SamplingParams {
    const raw_temp = extractFormFloat(body, "temperature") orelse 0;
    const raw_top_p = extractFormFloat(body, "top_p") orelse 1.0;
    const raw_top_k = extractFormInt(body, "top_k") orelse 0;
    return .{
        .temperature = if (std.math.isFinite(raw_temp)) std.math.clamp(raw_temp, 0, max_temperature) else 0,
        .top_k = @intCast(@min(raw_top_k, max_top_k)),
        .top_p = if (std.math.isFinite(raw_top_p)) std.math.clamp(raw_top_p, 0, 1.0) else 1.0,
    };
}

/// Extract base64 image data from a URL-encoded form body.
/// Looks for field "image" with a data URI value (e.g., "data:image/png;base64,...").
/// Returns the raw base64 string (after the "base64," prefix), or null if
/// no image field is present or the data URI format is unrecognized.
/// The returned slice points into the original body — valid for the request lifetime.
pub fn extractFormImage(body: []const u8) ?[]const u8 {
    const field_val = extractFormField(body, "image") orelse return null;
    // URL-encoded form values encode ',' as '%2C', so check both variants.
    const marker = "base64,";
    if (std.mem.indexOf(u8, field_val, marker)) |idx| {
        return field_val[idx + marker.len ..];
    }
    // Try URL-encoded comma
    const encoded_marker = "base64%2C";
    if (std.mem.indexOf(u8, field_val, encoded_marker)) |idx| {
        return field_val[idx + encoded_marker.len ..];
    }
    return null;
}

/// Extract base64 image data from an OpenAI-format JSON body.
/// Searches for a data URI pattern "data:image/...;base64,..." inside the JSON,
/// typically within an "image_url" content part. Returns the raw base64 string
/// between the "base64," marker and the next quote, or null if not found.
/// The returned slice points into the original body — valid for the request lifetime.
pub fn extractJsonImage(body: []const u8) ?[]const u8 {
    const marker = "data:image/";
    const idx = std.mem.indexOf(u8, body, marker) orelse return null;
    const after = body[idx + marker.len ..];
    const b64_marker = ";base64,";
    const b64_idx = std.mem.indexOf(u8, after, b64_marker) orelse return null;
    const start = idx + marker.len + b64_idx + b64_marker.len;
    const remaining = body[start..];
    // Find end — next quote (JSON string boundary) or end-of-string
    const end = std.mem.indexOfScalar(u8, remaining, '"') orelse return null;
    if (end == 0) return null;
    return body[start .. start + end];
}

// ── URL decoding ────────────────────────────────────────────────

/// Decode a URL-encoded (percent-encoded) string. `+` becomes space, `%XX` becomes the byte.
/// Caller owns the returned slice.
pub fn urlDecode(allocator: Allocator, input: []const u8) ![]u8 {
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
                const byte = hi.? * 16 + lo.?;
                if (byte == 0) {
                    i += 3;
                    continue;
                } // Strip null bytes
                result.appendAssumeCapacity(byte);
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

// ── JSON / HTML escaping ────────────────────────────────────────

/// Generic character escaper: for each byte, `escape_fn` returns a replacement
/// string or null (pass through). Used by jsonEscape and htmlEscape.
/// IMPORTANT: When no escaping is needed, returns a cast of `input` (no allocation).
/// Callers must compare `result.ptr != input.ptr` before freeing the result.
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
    if (!needs_escape) return @constCast(input);

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

/// Escape a string for safe embedding in JSON (quotes, backslashes, control chars).
/// Returns the input pointer unchanged (no allocation) when no escaping is needed.
pub fn jsonEscape(allocator: Allocator, input: []const u8) ![]u8 {
    return escapeWith(allocator, input, jsonEscapeChar);
}

/// Decode JSON string escape sequences (\\n → newline, \\\" → quote, etc.).
/// Returns the input unchanged (via @constCast) when no escapes are present.
/// Caller must check ptr equality to determine if the result was allocated.
pub fn jsonUnescape(allocator: Allocator, input: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, input, "\\") == null) return @constCast(input);

    const buf = try allocator.alloc(u8, input.len);
    var out: usize = 0;
    var i: usize = 0;

    while (i < input.len) {
        if (input[i] == '\\' and i + 1 < input.len) {
            i += 1;
            switch (input[i]) {
                '"' => {
                    buf[out] = '"';
                    out += 1;
                    i += 1;
                },
                '\\' => {
                    buf[out] = '\\';
                    out += 1;
                    i += 1;
                },
                '/' => {
                    buf[out] = '/';
                    out += 1;
                    i += 1;
                },
                'n' => {
                    buf[out] = '\n';
                    out += 1;
                    i += 1;
                },
                'r' => {
                    buf[out] = '\r';
                    out += 1;
                    i += 1;
                },
                't' => {
                    buf[out] = '\t';
                    out += 1;
                    i += 1;
                },
                'b' => {
                    buf[out] = 0x08;
                    out += 1;
                    i += 1;
                },
                'f' => {
                    buf[out] = 0x0C;
                    out += 1;
                    i += 1;
                },
                'u' => {
                    if (i + 5 <= input.len) {
                        const cp = std.fmt.parseInt(u21, input[i + 1 .. i + 5], 16) catch {
                            buf[out] = '\\';
                            out += 1;
                            buf[out] = 'u';
                            out += 1;
                            i += 1;
                            continue;
                        };
                        // Handle UTF-16 surrogates (CWE-176): decode surrogate
                        // pairs into a valid codepoint; emit U+FFFD for lone surrogates.
                        if (cp >= 0xD800 and cp <= 0xDFFF) {
                            if (cp <= 0xDBFF and i + 11 <= input.len and input[i + 5] == '\\' and input[i + 6] == 'u') {
                                // High surrogate — try to read low surrogate
                                const lo = std.fmt.parseInt(u21, input[i + 7 .. i + 11], 16) catch 0;
                                if (lo >= 0xDC00 and lo <= 0xDFFF) {
                                    // Valid surrogate pair — decode to codepoint (U+10000..U+10FFFF)
                                    const full: u21 = 0x10000 + (@as(u21, cp - 0xD800) << 10) + (lo - 0xDC00);
                                    buf[out] = @intCast(0xF0 | (full >> 18));
                                    buf[out + 1] = @intCast(0x80 | ((full >> 12) & 0x3F));
                                    buf[out + 2] = @intCast(0x80 | ((full >> 6) & 0x3F));
                                    buf[out + 3] = @intCast(0x80 | (full & 0x3F));
                                    out += 4;
                                    i += 11;
                                    continue;
                                }
                            }
                            // Lone surrogate — emit U+FFFD replacement character
                            buf[out] = 0xEF;
                            buf[out + 1] = 0xBF;
                            buf[out + 2] = 0xBD;
                            out += 3;
                            i += 5;
                            continue;
                        }
                        if (cp < 0x80) {
                            buf[out] = @intCast(cp);
                            out += 1;
                        } else if (cp < 0x800) {
                            buf[out] = @intCast(0xC0 | (cp >> 6));
                            buf[out + 1] = @intCast(0x80 | (cp & 0x3F));
                            out += 2;
                        } else {
                            buf[out] = @intCast(0xE0 | (cp >> 12));
                            buf[out + 1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
                            buf[out + 2] = @intCast(0x80 | (cp & 0x3F));
                            out += 3;
                        }
                        i += 5;
                    } else {
                        buf[out] = '\\';
                        out += 1;
                        buf[out] = 'u';
                        out += 1;
                        i += 1;
                    }
                },
                else => {
                    buf[out] = '\\';
                    out += 1;
                    buf[out] = input[i];
                    out += 1;
                    i += 1;
                },
            }
        } else {
            buf[out] = input[i];
            out += 1;
            i += 1;
        }
    }

    if (out == input.len) {
        allocator.free(buf);
        return @constCast(input);
    }

    return allocator.realloc(buf, out) catch buf[0..out];
}

/// Unescape a JSON string and return an owned copy (always allocated).
/// Caller must always free the result.
pub fn jsonUnescapeOwned(allocator: Allocator, input: []const u8) ![]u8 {
    const unescaped = try jsonUnescape(allocator, input);
    if (unescaped.ptr == input.ptr) {
        return try allocator.dupe(u8, input);
    }
    return unescaped;
}

/// Escape a string for safe embedding in HTML (`<`, `>`, `&`, `"`).
/// Returns the input pointer unchanged (no allocation) when no escaping is needed.
pub fn htmlEscape(allocator: Allocator, input: []const u8) ![]u8 {
    return escapeWith(allocator, input, htmlEscapeChar);
}

// ── Tests ───────────────────────────────────────────────────────

test "extractField skips false matches in string values" {
    // "system" appears as a value before it appears as a key
    const json = "{\"role\": \"system\", \"system\": \"You are helpful\"}";
    const result = extractField(json, "system");
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("You are helpful", result.?);
}

test "extractField handles normal case" {
    const json = "{\"model\": \"gpt-4\", \"prompt\": \"hello\"}";
    try std.testing.expectEqualStrings("gpt-4", extractField(json, "model").?);
    try std.testing.expectEqualStrings("hello", extractField(json, "prompt").?);
}

test "extractField returns null for missing field" {
    const json = "{\"model\": \"gpt-4\"}";
    try std.testing.expect(extractField(json, "prompt") == null);
}

test "extractIntField skips false matches" {
    // "max_tokens" as a string value before it as a key
    const json = "{\"name\": \"max_tokens\", \"max_tokens\": 256}";
    const result = extractIntField(json, "max_tokens");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 256), result.?);
}

test "extractBoolField skips false matches" {
    const json = "{\"label\": \"stream\", \"stream\": true}";
    try std.testing.expect(extractBoolField(json, "stream"));
}

test "extractBoolField returns false when value is false" {
    const json = "{\"stream\": false}";
    try std.testing.expect(!extractBoolField(json, "stream"));
}

test "extractFloatField skips false matches" {
    const json = "{\"label\": \"temperature\", \"temperature\": 0.7}";
    const result = extractFloatField(json, "temperature");
    try std.testing.expect(result != null);
    try std.testing.expect(@abs(result.? - 0.7) < 0.001);
}

test "extractBoolField with spaces around colon" {
    const json = "{\"stream\" : true}";
    try std.testing.expect(extractBoolField(json, "stream"));
}

test "jsonUnescape basic escapes" {
    const allocator = std.testing.allocator;

    // No escapes — returns input unchanged
    const plain = try jsonUnescape(allocator, "hello world");
    try std.testing.expectEqualStrings("hello world", plain);
    // ptr should be the same (no allocation)
    try std.testing.expect(plain.ptr == "hello world".ptr);

    // Newline and tab
    const nl = try jsonUnescape(allocator, "line1\\nline2\\ttab");
    defer allocator.free(nl);
    try std.testing.expectEqualStrings("line1\nline2\ttab", nl);

    // Escaped quotes and backslash
    const quotes = try jsonUnescape(allocator, "say \\\"hello\\\"");
    defer allocator.free(quotes);
    try std.testing.expectEqualStrings("say \"hello\"", quotes);

    const bs = try jsonUnescape(allocator, "path\\\\to\\\\file");
    defer allocator.free(bs);
    try std.testing.expectEqualStrings("path\\to\\file", bs);
}

test "jsonUnescape \\uXXXX" {
    const allocator = std.testing.allocator;

    // ASCII range: \u0041 = 'A'
    const ascii = try jsonUnescape(allocator, "\\u0041BC");
    defer allocator.free(ascii);
    try std.testing.expectEqualStrings("ABC", ascii);

    // BMP: \u00e9 = 'é' (UTF-8: 0xC3 0xA9)
    const bmp = try jsonUnescape(allocator, "caf\\u00e9");
    defer allocator.free(bmp);
    try std.testing.expectEqualStrings("café", bmp);

    // CJK: \u4e16 = '世' (UTF-8: 0xE4 0xB8 0x96)
    const cjk = try jsonUnescape(allocator, "\\u4e16\\u754c");
    defer allocator.free(cjk);
    try std.testing.expectEqualStrings("世界", cjk);
}

test "jsonUnescapeOwned always allocates" {
    const allocator = std.testing.allocator;

    // Even without escapes, returns an owned copy
    const owned = try jsonUnescapeOwned(allocator, "hello");
    defer allocator.free(owned);
    try std.testing.expectEqualStrings("hello", owned);
    // Must be a different allocation
    try std.testing.expect(owned.ptr != "hello".ptr);
}

test "extractFormFloat parses valid floats" {
    const body = "temperature=0.7&top_p=0.9&message=hello";
    const temp = extractFormFloat(body, "temperature");
    try std.testing.expect(temp != null);
    try std.testing.expect(@abs(temp.? - 0.7) < 0.001);
    const top_p = extractFormFloat(body, "top_p");
    try std.testing.expect(top_p != null);
    try std.testing.expect(@abs(top_p.? - 0.9) < 0.001);
}

test "extractFormFloat returns null for missing or invalid" {
    try std.testing.expect(extractFormFloat("message=hello", "temperature") == null);
    try std.testing.expect(extractFormFloat("temperature=abc", "temperature") == null);
    try std.testing.expect(extractFormFloat("temperature=", "temperature") == null);
}

test "extractFormInt parses valid integers" {
    const body = "max_tokens=256&stream=1";
    const mt = extractFormInt(body, "max_tokens");
    try std.testing.expect(mt != null);
    try std.testing.expectEqual(@as(usize, 256), mt.?);
}

test "extractFormInt returns null for missing or invalid" {
    try std.testing.expect(extractFormInt("message=hello", "max_tokens") == null);
    try std.testing.expect(extractFormInt("max_tokens=abc", "max_tokens") == null);
    try std.testing.expect(extractFormInt("max_tokens=", "max_tokens") == null);
}

test "extractFormImage with data URI" {
    const body = "message=hello&image=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo&stream=1";
    const result = extractFormImage(body);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("iVBORw0KGgo", result.?);
}

test "extractFormImage returns null when missing" {
    try std.testing.expect(extractFormImage("message=hello&stream=1") == null);
}

test "extractFormBool returns true for truthy values" {
    try std.testing.expect(extractFormBool("stream=1&message=hi", "stream"));
    try std.testing.expect(extractFormBool("stream=true&message=hi", "stream"));
    try std.testing.expect(extractFormBool("stream=TRUE&message=hi", "stream"));
    try std.testing.expect(extractFormBool("stream=yes", "stream"));
    try std.testing.expect(extractFormBool("stream=on", "stream"));
}

test "extractFormBool returns false for falsy or missing values" {
    try std.testing.expect(!extractFormBool("stream=0&message=hi", "stream"));
    try std.testing.expect(!extractFormBool("stream=false&message=hi", "stream"));
    try std.testing.expect(!extractFormBool("stream=&message=hi", "stream"));
    try std.testing.expect(!extractFormBool("message=hi", "stream"));
    try std.testing.expect(!extractFormBool("stream=no", "stream"));
}

test "extractFormImage with unencoded comma" {
    const body = "image=data:image/png;base64,AAAA&message=hi";
    const result = extractFormImage(body);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("AAAA", result.?);
}

test "parseFormSampling clamps values" {
    // Within range
    const s1 = parseFormSampling("temperature=0.8&top_p=0.95&top_k=50");
    try std.testing.expect(@abs(s1.temperature - 0.8) < 0.001);
    try std.testing.expect(@abs(s1.top_p - 0.95) < 0.001);
    try std.testing.expectEqual(@as(u32, 50), s1.top_k);

    // Defaults when missing
    const s2 = parseFormSampling("message=hello");
    try std.testing.expectEqual(@as(f32, 0), s2.temperature);
    try std.testing.expectEqual(@as(f32, 1.0), s2.top_p);
    try std.testing.expectEqual(@as(u32, 0), s2.top_k);

    // Negative temperature clamped to 0
    const s3 = parseFormSampling("temperature=-1.0");
    try std.testing.expectEqual(@as(f32, 0), s3.temperature);

    // top_p > 1 clamped to 1
    const s4 = parseFormSampling("top_p=2.0");
    try std.testing.expectEqual(@as(f32, 1.0), s4.top_p);

    // top_k clamped to max_top_k (1024)
    const s5 = parseFormSampling("top_k=9999");
    try std.testing.expectEqual(@as(u32, 1024), s5.top_k);
}

test "extractJsonImage with OpenAI format" {
    const body =
        \\{"messages":[{"role":"user","content":[{"type":"text","text":"What?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo"}}]}]}
    ;
    const result = extractJsonImage(body);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("iVBORw0KGgo", result.?);
}

test "extractJsonImage returns null when missing" {
    const body =
        \\{"messages":[{"role":"user","content":"hello"}]}
    ;
    try std.testing.expect(extractJsonImage(body) == null);
}

test "extractJsonImage with jpeg" {
    const body =
        \\{"content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQ"}}]}
    ;
    const result = extractJsonImage(body);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("/9j/4AAQ", result.?);
}

// ── Error-path tests ──────────────────────────────────────────────

test "extractField handles empty JSON" {
    try std.testing.expect(extractField("", "key") == null);
    try std.testing.expect(extractField("{}", "key") == null);
}

test "extractField handles malformed JSON" {
    try std.testing.expect(extractField("{\"key\":", "key") == null);
    try std.testing.expect(extractField("{\"key\" \"val\"}", "key") == null);
}

test "extractIntField handles negative and zero" {
    try std.testing.expect(extractIntField("{\"n\": 0}", "n") != null);
    try std.testing.expectEqual(@as(usize, 0), extractIntField("{\"n\": 0}", "n").?);
    // Negative numbers are not valid for usize
    try std.testing.expect(extractIntField("{\"n\": -1}", "n") == null);
}

test "extractFloatField handles edge values" {
    try std.testing.expect(extractFloatField("{\"t\": 0.0}", "t") != null);
    try std.testing.expect(extractFloatField("{\"t\": not_a_number}", "t") == null);
}

test "extractJsonImage handles truncated base64 marker" {
    // Has "base64," but no data after it — closing quote immediately follows,
    // so end == 0 and extractJsonImage returns null.
    const body =
        \\{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}
    ;
    try std.testing.expect(extractJsonImage(body) == null);
}

test "extractFormFloat handles boundary values" {
    const zero = extractFormFloat("v=0", "v");
    try std.testing.expect(zero != null);
    try std.testing.expect(@abs(zero.? - 0.0) < 0.001);
    const large = extractFormFloat("v=1e10", "v");
    try std.testing.expect(large != null);
    try std.testing.expect(@abs(large.? - 1e10) < 1e6);
    const half = extractFormFloat("v=.5", "v");
    try std.testing.expect(half != null);
    try std.testing.expect(@abs(half.? - 0.5) < 0.001);
}

test "parseFormSampling handles extreme values" {
    // Temperature > max_temperature (100) clamped to exactly max_temperature
    const s = parseFormSampling("temperature=999.0");
    try std.testing.expectEqual(max_temperature, s.temperature);
}
