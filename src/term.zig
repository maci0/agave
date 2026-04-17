//! Self-contained terminal input/output primitives.
//! Replaces the vaxis dependency with pure Zig — no libc calls,
//! no external deps. Uses only Zig builtins and posix syscalls.

const std = @import("std");

// ── ANSI Escape Sequences ────────────────────────────────────

/// ANSI control sequences for terminal styling.
/// Matches the subset of vaxis.ctlseqs that agave actually uses.
pub const ctlseqs = struct {
    pub const bold_set = "\x1b[1m";
    pub const dim_set = "\x1b[2m";
    pub const sgr_reset = "\x1b[m";
    /// Standard ANSI foreground color: "\x1b[3{d}m" where {d} is 0-7.
    pub const fg_base = "\x1b[3{d}m";
};

// ── Key Representation ───────────────────────────────────────

/// Lightweight key event, API-compatible with vaxis.Key for the subset
/// that readline.zig and display.zig use (codepoint, mods, text, matches).
pub const Key = struct {
    codepoint: u21 = 0,
    text: ?[]const u8 = null,
    mods: Modifiers = .{},

    pub const Modifiers = packed struct(u8) {
        shift: bool = false,
        alt: bool = false,
        ctrl: bool = false,
        _pad: u5 = 0,

        pub fn eql(self: Modifiers, other: Modifiers) bool {
            return @as(u8, @bitCast(self)) == @as(u8, @bitCast(other));
        }
    };

    // Named key constants — use the same values as vaxis (Kitty private-use area)
    // so that CSI-encoded sequences decode correctly.
    pub const enter: u21 = 0x0D;
    pub const backspace: u21 = 0x7F;
    pub const escape: u21 = 0x1B;
    pub const delete: u21 = 57349;
    pub const left: u21 = 57350;
    pub const right: u21 = 57351;
    pub const up: u21 = 57352;
    pub const down: u21 = 57353;
    pub const home: u21 = 57356;
    pub const end: u21 = 57357;

    /// Loose key matching — checks codepoint and the ctrl/alt/shift modifiers.
    pub fn matches(self: Key, cp: u21, mods: Modifiers) bool {
        // Exact codepoint + modifier match (ignoring padding bits)
        var self_m = self.mods;
        self_m._pad = 0;
        var tgt_m = mods;
        tgt_m._pad = 0;
        if (self.codepoint == cp and self_m.eql(tgt_m)) return true;

        // Text-based match: if the key generated text, compare UTF-8 encoding
        // of cp against text, consuming Shift from the comparison.
        if (self.text) |text| {
            var sm = self.mods;
            sm._pad = 0;
            sm.shift = false;
            var tm = mods;
            tm._pad = 0;
            tm.shift = false;
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cp, &buf) catch return false;
            if (std.mem.eql(u8, text, buf[0..n]) and sm.eql(tm)) return true;
        }

        return false;
    }
};

// ── VT100/xterm Escape Sequence Parser ───────────────────────

/// Minimal VT100/xterm escape sequence parser.
/// Handles: CSI cursor/function keys, SS3 keys, Alt+key, Ctrl+key, UTF-8 text.
/// API-compatible with vaxis.Parser for the subset used by readline.zig.
pub const Parser = struct {
    /// Result of parsing one event from the input buffer.
    pub const Result = struct {
        n: usize = 0,
        event: ?Event = null,
    };

    pub const Event = union(enum) {
        key_press: Key,
    };

    /// Parse one key event from the input buffer.
    /// Returns .n=0 if more data is needed (incomplete sequence).
    /// The `grapheme_data` parameter exists for API compatibility and is ignored.
    pub fn parse(_: *Parser, buf: []const u8, _: ?*const anyopaque) !Result {
        if (buf.len == 0) return .{};

        const b = buf[0];

        // ESC sequences
        if (b == 0x1b) {
            if (buf.len < 2) return .{}; // need more data
            if (buf[1] == '[') {
                // CSI sequence: ESC [ ... final_byte
                return parseCsi(buf);
            }
            if (buf[1] == 'O') {
                // SS3 sequence (arrow keys on some terminals)
                if (buf.len < 3) return .{};
                const key: u21 = switch (buf[2]) {
                    'A' => Key.up,
                    'B' => Key.down,
                    'C' => Key.right,
                    'D' => Key.left,
                    'H' => Key.home,
                    'F' => Key.end,
                    else => 0,
                };
                if (key != 0) return .{ .n = 3, .event = .{ .key_press = .{ .codepoint = key } } };
                return .{ .n = 3 };
            }
            // Alt+key
            if (buf[1] >= 0x20) {
                return .{ .n = 2, .event = .{ .key_press = .{
                    .codepoint = buf[1],
                    .mods = .{ .alt = true },
                } } };
            }
            // Bare ESC
            return .{ .n = 1, .event = .{ .key_press = .{ .codepoint = Key.escape } } };
        }

        // Ctrl+letter (0x01-0x1a = Ctrl+a through Ctrl+z, excluding 0x0d=Enter and 0x09=Tab)
        if (b >= 1 and b <= 26 and b != '\r' and b != '\t') {
            return .{ .n = 1, .event = .{ .key_press = .{
                .codepoint = @as(u21, b) + 0x60,
                .mods = .{ .ctrl = true },
            } } };
        }

        // Enter
        if (b == '\r' or b == '\n') {
            return .{ .n = 1, .event = .{ .key_press = .{ .codepoint = Key.enter } } };
        }

        // Backspace (DEL)
        if (b == 0x7f) {
            return .{ .n = 1, .event = .{ .key_press = .{ .codepoint = Key.backspace } } };
        }

        // UTF-8 multi-byte
        if (b >= 0x80) {
            const len = std.unicode.utf8ByteSequenceLength(b) catch return .{ .n = 1 };
            if (buf.len < len) return .{}; // need more data
            const cp = std.unicode.utf8Decode(buf[0..len]) catch return .{ .n = 1 };
            return .{ .n = len, .event = .{ .key_press = .{
                .codepoint = cp,
                .text = buf[0..len],
            } } };
        }

        // Printable ASCII
        if (b >= 0x20 and b < 0x7f) {
            return .{ .n = 1, .event = .{ .key_press = .{
                .codepoint = b,
                .text = buf[0..1],
            } } };
        }

        // Other control characters — consume and ignore
        return .{ .n = 1 };
    }

    fn parseCsi(buf: []const u8) Result {
        // CSI: ESC [ (params) final_byte
        // Find the final byte (0x40-0x7e range)
        var i: usize = 2;
        while (i < buf.len) : (i += 1) {
            if (buf[i] >= 0x40 and buf[i] <= 0x7e) {
                const final = buf[i];
                const consumed = i + 1;
                const key: u21 = switch (final) {
                    'A' => Key.up,
                    'B' => Key.down,
                    'C' => Key.right,
                    'D' => Key.left,
                    'H' => Key.home,
                    'F' => Key.end,
                    '~' => blk: {
                        // Numeric CSI: ESC [ N ~ or ESC [ N N ~ etc.
                        // Parse the numeric parameter between '[' and '~'
                        const param_slice = buf[2..i];
                        // Find parameter before any ';' (modifier follows ';')
                        const semi = std.mem.indexOfScalar(u8, param_slice, ';');
                        const num_slice = if (semi) |s| param_slice[0..s] else param_slice;
                        const num = std.fmt.parseUnsigned(u16, num_slice, 10) catch break :blk @as(u21, 0);
                        break :blk switch (num) {
                            2 => Key.delete, // Insert on some terminals, but we map common ones
                            3 => Key.delete,
                            5 => 0, // Page Up — not used
                            6 => 0, // Page Down — not used
                            7 => Key.home,
                            8 => Key.end,
                            else => 0,
                        };
                    },
                    else => 0,
                };
                if (key != 0) return .{ .n = consumed, .event = .{ .key_press = .{ .codepoint = key } } };
                return .{ .n = consumed };
            }
        }
        return .{}; // incomplete CSI
    }
};

// ── Display Width ────────────────────────────────────────────

/// Compute the display width of a UTF-8 string.
/// Handles ASCII (width 1), CJK fullwidth (width 2), zero-width combining marks.
/// Pure Zig — no libc wcwidth.
pub fn displayWidth(s: []const u8) usize {
    var w: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const len = std.unicode.utf8ByteSequenceLength(s[i]) catch {
            i += 1;
            w += 1; // replacement char
            continue;
        };
        if (i + len > s.len) break;
        const cp = std.unicode.utf8Decode(s[i..][0..len]) catch {
            i += 1;
            w += 1;
            continue;
        };
        i += len;
        w += codepointWidth(cp);
    }
    return w;
}

/// Also expose via the name used by vaxis: gwidth.gwidth(slice, .unicode).
/// This enables minimal changes at call sites.
pub const gwidth = struct {
    pub const Method = enum { unicode };
    pub fn gwidth(s: []const u8, _: Method) u16 {
        return @intCast(displayWidth(s));
    }
};

/// Width of a single codepoint. CJK fullwidth = 2, combining = 0, most = 1.
fn codepointWidth(cp: u21) usize {
    // Zero-width: combining marks, zero-width space/joiner, variation selectors
    if (cp >= 0x0300 and cp <= 0x036F) return 0; // Combining Diacriticals
    if (cp >= 0x1AB0 and cp <= 0x1AFF) return 0; // Combining Diacriticals Extended
    if (cp >= 0x1DC0 and cp <= 0x1DFF) return 0; // Combining Diacriticals Supplement
    if (cp >= 0x20D0 and cp <= 0x20FF) return 0; // Combining for Symbols
    if (cp >= 0xFE00 and cp <= 0xFE0F) return 0; // Variation Selectors
    if (cp >= 0xFE20 and cp <= 0xFE2F) return 0; // Combining Half Marks
    if (cp == 0x200B or cp == 0x200C or cp == 0x200D or cp == 0xFEFF) return 0; // ZW spaces
    if (cp >= 0xE0100 and cp <= 0xE01EF) return 0; // Variation Selectors Supplement

    // Fullwidth: CJK Unified, CJK Compatibility, Hangul, fullwidth forms
    if (cp >= 0x1100 and cp <= 0x115F) return 2; // Hangul Jamo
    if (cp >= 0x2E80 and cp <= 0x303E) return 2; // CJK Radicals + Symbols
    if (cp >= 0x3040 and cp <= 0x33BF) return 2; // Hiragana + Katakana + CJK compat
    if (cp >= 0x3400 and cp <= 0x4DBF) return 2; // CJK Extension A
    if (cp >= 0x4E00 and cp <= 0x9FFF) return 2; // CJK Unified
    if (cp >= 0xA960 and cp <= 0xA97F) return 2; // Hangul Jamo Extended-A
    if (cp >= 0xAC00 and cp <= 0xD7AF) return 2; // Hangul Syllables
    if (cp >= 0xF900 and cp <= 0xFAFF) return 2; // CJK Compatibility Ideographs
    if (cp >= 0xFE30 and cp <= 0xFE6F) return 2; // CJK Compatibility Forms
    if (cp >= 0xFF01 and cp <= 0xFF60) return 2; // Fullwidth forms
    if (cp >= 0xFFE0 and cp <= 0xFFE6) return 2; // Fullwidth symbols
    if (cp >= 0x20000 and cp <= 0x2FA1F) return 2; // CJK Extension B-F + Supplement
    if (cp >= 0x30000 and cp <= 0x323AF) return 2; // CJK Extension G-I

    // Control characters
    if (cp < 0x20 or cp == 0x7F) return 0;

    return 1;
}

// ── Gap Buffer (line editor) ─────────────────────────────────

/// Minimal gap buffer for line editing. API-compatible with vaxis TextInput
/// for the subset used by readline.zig (init, deinit, update, insertSliceAtCursor,
/// clearRetainingCapacity, toOwnedSlice, buf.firstHalf/secondHalf/realLength).
pub const TextInput = struct {
    buf: Buffer,

    pub fn init(allocator: std.mem.Allocator) TextInput {
        return .{ .buf = Buffer.init(allocator) };
    }

    pub fn deinit(self: *TextInput) void {
        self.buf.deinit();
    }

    /// Insert text at cursor position.
    pub fn insertSliceAtCursor(self: *TextInput, text: []const u8) !void {
        try self.buf.insertSliceAtCursor(text);
    }

    /// Clear all text, retain allocated capacity.
    pub fn clearRetainingCapacity(self: *TextInput) void {
        self.buf.clearRetainingCapacity();
    }

    /// Get owned copy of the text and clear the buffer.
    pub fn toOwnedSlice(self: *TextInput) ![]const u8 {
        return self.buf.toOwnedSlice();
    }

    /// Handle a key press (cursor movement, delete, backspace, text insertion).
    pub fn update(self: *TextInput, event: Parser.Event) !void {
        switch (event) {
            .key_press => |key| {
                if (key.matches(Key.backspace, .{})) {
                    self.deleteBeforeCursor();
                } else if (key.matches(Key.delete, .{}) or key.matches('d', .{ .ctrl = true })) {
                    self.deleteAfterCursor();
                } else if (key.matches(Key.left, .{}) or key.matches('b', .{ .ctrl = true })) {
                    self.cursorLeft();
                } else if (key.matches(Key.right, .{}) or key.matches('f', .{ .ctrl = true })) {
                    self.cursorRight();
                } else if (key.matches('a', .{ .ctrl = true }) or key.matches(Key.home, .{})) {
                    self.buf.moveGapLeft(self.buf.firstHalf().len);
                } else if (key.matches('e', .{ .ctrl = true }) or key.matches(Key.end, .{})) {
                    self.buf.moveGapRight(self.buf.secondHalf().len);
                } else if (key.matches('k', .{ .ctrl = true })) {
                    // Kill to end of line
                    self.buf.growGapRight(self.buf.secondHalf().len);
                } else if (key.matches('u', .{ .ctrl = true })) {
                    // Kill to start of line
                    self.buf.growGapLeft(self.buf.cursor);
                } else if (key.matches('w', .{ .ctrl = true }) or key.matches(Key.backspace, .{ .alt = true })) {
                    // Kill word backward (whitespace-delimited)
                    const first_half = self.buf.firstHalf();
                    var pos = first_half.len;
                    while (pos > 0 and first_half[pos - 1] == ' ') pos -= 1;
                    while (pos > 0 and first_half[pos - 1] != ' ') pos -= 1;
                    const to_delete = self.buf.cursor - pos;
                    self.buf.moveGapLeft(to_delete);
                    self.buf.growGapRight(to_delete);
                } else if (key.text) |text| {
                    if (text.len > 0 and text[0] >= 0x20) {
                        try self.insertSliceAtCursor(text);
                    }
                }
            },
        }
    }

    fn cursorLeft(self: *TextInput) void {
        const fh = self.buf.firstHalf();
        if (fh.len == 0) return;
        // Walk back over one UTF-8 character
        var pos = fh.len - 1;
        while (pos > 0 and (fh[pos] & 0xC0) == 0x80) pos -= 1;
        self.buf.moveGapLeft(fh.len - pos);
    }

    fn cursorRight(self: *TextInput) void {
        const sh = self.buf.secondHalf();
        if (sh.len == 0) return;
        const len = std.unicode.utf8ByteSequenceLength(sh[0]) catch 1;
        self.buf.moveGapRight(@min(len, sh.len));
    }

    fn deleteBeforeCursor(self: *TextInput) void {
        const fh = self.buf.firstHalf();
        if (fh.len == 0) return;
        var pos = fh.len - 1;
        while (pos > 0 and (fh[pos] & 0xC0) == 0x80) pos -= 1;
        self.buf.growGapLeft(fh.len - pos);
    }

    fn deleteAfterCursor(self: *TextInput) void {
        const sh = self.buf.secondHalf();
        if (sh.len == 0) return;
        const len = std.unicode.utf8ByteSequenceLength(sh[0]) catch 1;
        self.buf.growGapRight(@min(len, sh.len));
    }

    /// The underlying gap buffer — exposed for readline.zig which accesses
    /// `input.buf.firstHalf()`, `input.buf.secondHalf()`, `input.buf.realLength()`.
    pub const Buffer = struct {
        allocator: std.mem.Allocator,
        buffer: []u8,
        cursor: usize,
        gap_size: usize,

        pub fn init(allocator: std.mem.Allocator) Buffer {
            return .{
                .allocator = allocator,
                .buffer = &.{},
                .cursor = 0,
                .gap_size = 0,
            };
        }

        pub fn deinit(self: *Buffer) void {
            if (self.buffer.len > 0) self.allocator.free(self.buffer);
        }

        pub fn firstHalf(self: Buffer) []const u8 {
            return self.buffer[0..self.cursor];
        }

        pub fn secondHalf(self: Buffer) []const u8 {
            return self.buffer[self.cursor + self.gap_size ..];
        }

        pub fn realLength(self: *const Buffer) usize {
            return self.firstHalf().len + self.secondHalf().len;
        }

        pub fn insertSliceAtCursor(self: *Buffer, slice: []const u8) std.mem.Allocator.Error!void {
            if (slice.len == 0) return;
            if (self.gap_size <= slice.len) try self.grow(slice.len);
            @memcpy(self.buffer[self.cursor .. self.cursor + slice.len], slice);
            self.cursor += slice.len;
            self.gap_size -= slice.len;
        }

        pub fn moveGapLeft(self: *Buffer, n: usize) void {
            const new_idx = self.cursor -| n;
            const dst = self.buffer[new_idx + self.gap_size ..];
            const src = self.buffer[new_idx..self.cursor];
            std.mem.copyForwards(u8, dst, src);
            self.cursor = new_idx;
        }

        pub fn moveGapRight(self: *Buffer, n: usize) void {
            const new_idx = self.cursor + n;
            const dst = self.buffer[self.cursor..];
            const src = self.buffer[self.cursor + self.gap_size .. new_idx + self.gap_size];
            std.mem.copyForwards(u8, dst, src);
            self.cursor = new_idx;
        }

        pub fn growGapLeft(self: *Buffer, n: usize) void {
            self.gap_size += n;
            self.cursor -|= n;
        }

        pub fn growGapRight(self: *Buffer, n: usize) void {
            self.gap_size = @min(self.gap_size + n, self.buffer.len - self.cursor);
        }

        pub fn clearRetainingCapacity(self: *Buffer) void {
            self.cursor = 0;
            self.gap_size = self.buffer.len;
        }

        pub fn toOwnedSlice(self: *Buffer) std.mem.Allocator.Error![]const u8 {
            const fh = self.firstHalf();
            const sh = self.secondHalf();
            const out = try self.allocator.alloc(u8, fh.len + sh.len);
            @memcpy(out[0..fh.len], fh);
            @memcpy(out[fh.len..], sh);
            self.clearAndFree();
            return out;
        }

        fn clearAndFree(self: *Buffer) void {
            self.cursor = 0;
            if (self.buffer.len > 0) self.allocator.free(self.buffer);
            self.buffer = &.{};
            self.gap_size = 0;
        }

        /// Growth factor for the gap buffer.
        const growth_increment = 512;

        fn grow(self: *Buffer, n: usize) std.mem.Allocator.Error!void {
            const new_size = self.buffer.len + n + growth_increment;
            const new_memory = try self.allocator.alloc(u8, new_size);
            @memcpy(new_memory[0..self.cursor], self.firstHalf());
            const sh = self.secondHalf();
            @memcpy(new_memory[new_size - sh.len ..], sh);
            if (self.buffer.len > 0) self.allocator.free(self.buffer);
            self.buffer = new_memory;
            self.gap_size = new_size - sh.len - self.cursor;
        }
    };
};

// ── Tests ────────────────────────────────────────────────────

test "displayWidth ascii" {
    try std.testing.expectEqual(@as(usize, 5), displayWidth("hello"));
}

test "displayWidth middot" {
    // "a · b" — the middot (U+00B7) is 2 bytes in UTF-8 but occupies 1 terminal column
    try std.testing.expectEqual(@as(usize, 5), displayWidth("a \xc2\xb7 b"));
}

test "displayWidth CJK" {
    // Two CJK characters = 4 columns
    try std.testing.expectEqual(@as(usize, 4), displayWidth("\xe4\xb8\xad\xe6\x96\x87"));
}

test "displayWidth combining" {
    // 'a' + combining acute = 1 column (combining mark has zero width)
    try std.testing.expectEqual(@as(usize, 1), displayWidth("a\xcc\x81"));
}

test "parser: printable ascii" {
    var parser: Parser = .{};
    const result = try parser.parse("a", null);
    try std.testing.expectEqual(@as(usize, 1), result.n);
    try std.testing.expectEqual(@as(u21, 'a'), result.event.?.key_press.codepoint);
}

test "parser: ctrl+a" {
    var parser: Parser = .{};
    const result = try parser.parse("\x01", null);
    try std.testing.expectEqual(@as(usize, 1), result.n);
    try std.testing.expectEqual(@as(u21, 'a'), result.event.?.key_press.codepoint);
    try std.testing.expect(result.event.?.key_press.mods.ctrl);
}

test "parser: enter" {
    var parser: Parser = .{};
    const result = try parser.parse("\r", null);
    try std.testing.expectEqual(@as(usize, 1), result.n);
    try std.testing.expectEqual(Key.enter, result.event.?.key_press.codepoint);
}

test "parser: escape sequence arrow up" {
    var parser: Parser = .{};
    const result = try parser.parse("\x1b[A", null);
    try std.testing.expectEqual(@as(usize, 3), result.n);
    try std.testing.expectEqual(Key.up, result.event.?.key_press.codepoint);
}

test "parser: SS3 arrow down" {
    var parser: Parser = .{};
    const result = try parser.parse("\x1bOB", null);
    try std.testing.expectEqual(@as(usize, 3), result.n);
    try std.testing.expectEqual(Key.down, result.event.?.key_press.codepoint);
}

test "parser: alt+a" {
    var parser: Parser = .{};
    const result = try parser.parse("\x1ba", null);
    try std.testing.expectEqual(@as(usize, 2), result.n);
    try std.testing.expectEqual(@as(u21, 'a'), result.event.?.key_press.codepoint);
    try std.testing.expect(result.event.?.key_press.mods.alt);
}

test "parser: delete key" {
    var parser: Parser = .{};
    const result = try parser.parse("\x1b[3~", null);
    try std.testing.expectEqual(@as(usize, 4), result.n);
    try std.testing.expectEqual(Key.delete, result.event.?.key_press.codepoint);
}

test "parser: utf8 multi-byte" {
    var parser: Parser = .{};
    const input = "\xc3\xa9"; // é
    const result = try parser.parse(input, null);
    try std.testing.expectEqual(@as(usize, 2), result.n);
    try std.testing.expectEqual(@as(u21, 0xe9), result.event.?.key_press.codepoint);
}

test "parser: incomplete escape returns zero" {
    var parser: Parser = .{};
    const result = try parser.parse("\x1b", null);
    try std.testing.expectEqual(@as(usize, 0), result.n);
}

test "key matches basic" {
    const key: Key = .{ .codepoint = 'a', .text = "a" };
    try std.testing.expect(key.matches('a', .{}));
    try std.testing.expect(!key.matches('b', .{}));
    try std.testing.expect(!key.matches('a', .{ .ctrl = true }));
}

test "key matches ctrl" {
    const key: Key = .{ .codepoint = 'c', .mods = .{ .ctrl = true } };
    try std.testing.expect(key.matches('c', .{ .ctrl = true }));
    try std.testing.expect(!key.matches('c', .{}));
}

test "gap buffer basics" {
    var buf = TextInput.Buffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.insertSliceAtCursor("abc");
    try std.testing.expectEqualStrings("abc", buf.firstHalf());
    try std.testing.expectEqualStrings("", buf.secondHalf());

    buf.moveGapLeft(1);
    try std.testing.expectEqualStrings("ab", buf.firstHalf());
    try std.testing.expectEqualStrings("c", buf.secondHalf());

    try buf.insertSliceAtCursor(" ");
    try std.testing.expectEqualStrings("ab ", buf.firstHalf());
    try std.testing.expectEqualStrings("c", buf.secondHalf());

    buf.growGapLeft(1);
    try std.testing.expectEqualStrings("ab", buf.firstHalf());
    try std.testing.expectEqualStrings("c", buf.secondHalf());
}

test "text input update" {
    var input = TextInput.init(std.testing.allocator);
    defer input.deinit();

    // Type "hello"
    for ("hello") |c| {
        try input.update(.{ .key_press = .{ .codepoint = c, .text = &.{c} } });
    }
    try std.testing.expectEqual(@as(usize, 5), input.buf.realLength());
    try std.testing.expectEqualStrings("hello", input.buf.firstHalf());

    // Backspace
    try input.update(.{ .key_press = .{ .codepoint = Key.backspace } });
    try std.testing.expectEqual(@as(usize, 4), input.buf.realLength());
    try std.testing.expectEqualStrings("hell", input.buf.firstHalf());
}
