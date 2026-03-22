//! Line editor with history, reverse search (Ctrl-R), and Unicode support.
//! Supports Ctrl-C (double-tap to quit), Ctrl-L (clear screen), and Ctrl-D (EOF).
//! Uses vaxis for input parsing, key matching, and text editing (gap buffer).

const std = @import("std");
const posix = std.posix;
const vaxis = @import("vaxis");
const Key = vaxis.Key;
const TextInput = vaxis.widgets.TextInput;

const max_history = 256;
const search_buf_size = 256;
const leftover_buf_size = 128;
const input_read_buf_size = 256;
const ctrl_c_double_tap_ms: i64 = 1000;

/// Interactive line editor with history and reverse-search (Ctrl-R).
pub const LineEditor = struct {
    history: [max_history][]const u8 = undefined,
    hist_len: usize = 0,
    allocator: std.mem.Allocator,
    fd: posix.fd_t,
    orig_termios: ?posix.termios = null,
    last_ctrl_c_ms: i64 = 0,

    /// Create a line editor backed by stdin with empty history.
    pub fn init(allocator: std.mem.Allocator) LineEditor {
        return .{ .allocator = allocator, .fd = std.fs.File.stdin().handle };
    }

    /// Free all owned history entries.
    pub fn deinit(self: *LineEditor) void {
        for (self.history[0..self.hist_len]) |h| self.allocator.free(h);
    }

    /// Append a line to history (deduplicates consecutive entries).
    pub fn addHistory(self: *LineEditor, line: []const u8) void {
        if (line.len == 0) return;
        if (self.hist_len > 0 and std.mem.eql(u8, self.history[self.hist_len - 1], line)) return;

        const dupe = self.allocator.dupe(u8, line) catch return;
        if (self.hist_len < max_history) {
            self.history[self.hist_len] = dupe;
            self.hist_len += 1;
        } else {
            self.allocator.free(self.history[0]);
            std.mem.copyForwards([]const u8, self.history[0 .. max_history - 1], self.history[1..max_history]);
            self.history[max_history - 1] = dupe;
        }
    }

    /// Read a line with editing support. Returns owned slice or null on EOF/Ctrl-D.
    pub fn readline(self: *LineEditor, prompt: []const u8) ?[]const u8 {
        if (!posix.isatty(self.fd)) return self.readSimple();

        self.enableRaw() catch return self.readSimple();
        defer self.disableRaw();

        var input = TextInput.init(self.allocator);
        defer input.deinit();

        var hist_idx: usize = self.hist_len;
        var saved_text: ?[]const u8 = null;
        defer if (saved_text) |s| self.allocator.free(s);

        var searching = false;
        var search_buf: [search_buf_size]u8 = undefined;
        var search_len: usize = 0;
        var search_match: ?usize = null;

        const prompt_w = displayWidth(prompt);
        self.writeAll(prompt);

        var parser: vaxis.Parser = .{};
        var leftover: [leftover_buf_size]u8 = undefined;
        var leftover_len: usize = 0;

        while (true) {
            var read_buf: [input_read_buf_size]u8 = undefined;
            @memcpy(read_buf[0..leftover_len], leftover[0..leftover_len]);
            const n = posix.read(self.fd, read_buf[leftover_len..]) catch return null;
            if (n == 0) {
                if (input.buf.realLength() == 0) return null;
                break;
            }
            const total = leftover_len + n;
            leftover_len = 0;

            var i: usize = 0;
            while (i < total) {
                const result = parser.parse(read_buf[i..total], null) catch {
                    // Possibly incomplete UTF-8 at end of buffer
                    const remaining = total - i;
                    if (remaining < 4 and i > 0) {
                        @memcpy(leftover[0..remaining], read_buf[i..total]);
                        leftover_len = remaining;
                        break;
                    }
                    i += 1;
                    continue;
                };
                if (result.n == 0) {
                    const remaining = total - i;
                    @memcpy(leftover[0..remaining], read_buf[i..total]);
                    leftover_len = remaining;
                    break;
                }
                i += result.n;

                const event = result.event orelse continue;
                switch (event) {
                    .key_press => |key| {
                        if (searching) {
                            self.handleSearch(key, &searching, &search_buf, &search_len, &search_match, &input, prompt, prompt_w);
                            continue;
                        }

                        if (key.codepoint != 'c' or !key.mods.ctrl) self.last_ctrl_c_ms = 0;

                        if (key.matches(Key.enter, .{})) {
                            self.writeAll("\r\n");
                            return input.toOwnedSlice() catch null;
                        }
                        if (key.matches('d', .{ .ctrl = true })) {
                            if (input.buf.realLength() == 0) return null;
                            input.update(.{ .key_press = key }) catch {};
                            self.refreshLine(prompt, prompt_w, &input);
                            continue;
                        }
                        if (key.matches('c', .{ .ctrl = true })) {
                            const now = std.time.milliTimestamp();
                            if (self.last_ctrl_c_ms != 0 and now - self.last_ctrl_c_ms < ctrl_c_double_tap_ms) {
                                self.writeAll("\r\n");
                                return null;
                            }
                            self.last_ctrl_c_ms = now;
                            input.clearRetainingCapacity();
                            self.clearLine();
                            self.writeAll("Press Ctrl+C again to quit");
                            _ = self.pollInput(ctrl_c_double_tap_ms);
                            self.clearLine();
                            self.writeAll(prompt);
                            continue;
                        }
                        if (key.matches('l', .{ .ctrl = true })) {
                            self.writeAll("\x1b[2J\x1b[H");
                            self.refreshLine(prompt, prompt_w, &input);
                            continue;
                        }
                        if (key.matches('r', .{ .ctrl = true })) {
                            searching = true;
                            search_len = 0;
                            search_match = null;
                            self.showSearch(search_buf[0..0], null);
                            continue;
                        }
                        if (key.matches(Key.up, .{})) {
                            if (hist_idx > 0) {
                                if (hist_idx == self.hist_len) {
                                    if (saved_text) |s| self.allocator.free(s);
                                    saved_text = input.toOwnedSlice() catch null;
                                }
                                hist_idx -= 1;
                                input.clearRetainingCapacity();
                                input.insertSliceAtCursor(self.history[hist_idx]) catch {};
                                self.refreshLine(prompt, prompt_w, &input);
                            }
                            continue;
                        }
                        if (key.matches(Key.down, .{})) {
                            if (hist_idx < self.hist_len) {
                                hist_idx += 1;
                                input.clearRetainingCapacity();
                                if (hist_idx == self.hist_len) {
                                    if (saved_text) |s| {
                                        input.insertSliceAtCursor(s) catch {};
                                        self.allocator.free(s);
                                        saved_text = null;
                                    }
                                } else {
                                    input.insertSliceAtCursor(self.history[hist_idx]) catch {};
                                }
                                self.refreshLine(prompt, prompt_w, &input);
                            }
                            continue;
                        }

                        // All other keys — delegate to TextInput (handles cursor movement,
                        // word-wise ops, kill-line, backspace, delete, text insertion, etc.)
                        input.update(.{ .key_press = key }) catch {};
                        self.refreshLine(prompt, prompt_w, &input);
                    },
                    else => {},
                }
            }
        }

        return input.toOwnedSlice() catch null;
    }

    // ── Search ──────────────────────────────────────────────────

    fn handleSearch(
        self: *LineEditor,
        key: Key,
        searching: *bool,
        search_buf: *[256]u8,
        search_len: *usize,
        search_match: *?usize,
        input: *TextInput,
        prompt: []const u8,
        prompt_w: usize,
    ) void {
        if (key.matches('r', .{ .ctrl = true })) {
            if (search_match.*) |mi| {
                search_match.* = self.searchBack(search_buf[0..search_len.*], if (mi > 0) mi - 1 else null);
            }
            self.showSearch(search_buf[0..search_len.*], search_match.*);
            return;
        }
        if (key.matches('c', .{ .ctrl = true }) or key.matches('g', .{ .ctrl = true }) or key.matches(Key.escape, .{})) {
            searching.* = false;
            self.refreshLine(prompt, prompt_w, input);
            return;
        }
        if (key.matches(Key.enter, .{})) {
            searching.* = false;
            if (search_match.*) |mi| {
                input.clearRetainingCapacity();
                input.insertSliceAtCursor(self.history[mi]) catch {};
            }
            self.refreshLine(prompt, prompt_w, input);
            return;
        }
        if (key.matches(Key.backspace, .{})) {
            if (search_len.* > 0) search_len.* -= 1;
            search_match.* = self.searchBack(search_buf[0..search_len.*], null);
            self.showSearch(search_buf[0..search_len.*], search_match.*);
            return;
        }
        if (key.text) |text| {
            if (text.len > 0 and text[0] >= 0x20 and search_len.* + text.len <= search_buf.len) {
                @memcpy(search_buf[search_len.*..][0..text.len], text);
                search_len.* += text.len;
                search_match.* = self.searchBack(search_buf[0..search_len.*], null);
                self.showSearch(search_buf[0..search_len.*], search_match.*);
            }
        }
    }

    fn showSearch(self: *LineEditor, query: []const u8, match_idx: ?usize) void {
        self.clearLine();
        self.writeAll("(reverse-i-search)`");
        self.writeAll(query);
        self.writeAll("': ");
        if (match_idx) |mi| {
            self.writeAll(self.history[mi]);
        }
    }

    fn searchBack(self: *LineEditor, query: []const u8, start: ?usize) ?usize {
        if (query.len == 0 or self.hist_len == 0) return null;
        var i: usize = start orelse self.hist_len - 1;
        while (true) {
            if (std.mem.indexOf(u8, self.history[i], query) != null) return i;
            if (i == 0) break;
            i -= 1;
        }
        return null;
    }

    // ── Rendering ───────────────────────────────────────────────

    fn refreshLine(self: *LineEditor, prompt: []const u8, prompt_w: usize, input: *TextInput) void {
        self.clearLine();
        self.writeAll(prompt);
        self.writeAll(input.buf.firstHalf());
        self.writeAll(input.buf.secondHalf());
        const cursor_w = displayWidth(input.buf.firstHalf());
        self.moveCursorTo(prompt_w + cursor_w);
    }

    fn clearLine(self: *LineEditor) void {
        self.writeAll("\r\x1b[K");
    }

    fn moveCursorTo(self: *LineEditor, col: usize) void {
        if (col == 0) {
            self.writeAll("\r");
            return;
        }
        var cbuf: [32]u8 = undefined;
        const s = std.fmt.bufPrint(&cbuf, "\r\x1b[{d}C", .{col}) catch return;
        self.writeAll(s);
    }

    /// Write data to stdout (for terminal echo and prompt display).
    fn writeAll(_: *LineEditor, data: []const u8) void {
        _ = posix.write(posix.STDOUT_FILENO, data) catch {};
    }

    // ── Terminal ────────────────────────────────────────────────

    fn readSimple(self: *LineEditor) ?[]const u8 {
        var buf: [4096]u8 = undefined;
        var len: usize = 0;
        while (len < buf.len) {
            var b: [1]u8 = undefined;
            const nr = posix.read(self.fd, &b) catch return null;
            if (nr == 0) return if (len > 0) self.allocator.dupe(u8, buf[0..len]) catch null else null;
            if (b[0] == '\n') return self.allocator.dupe(u8, buf[0..len]) catch null;
            buf[len] = b[0];
            len += 1;
        }
        return self.allocator.dupe(u8, buf[0..len]) catch null;
    }

    fn enableRaw(self: *LineEditor) !void {
        self.orig_termios = try posix.tcgetattr(self.fd);
        var raw = self.orig_termios.?;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.ISIG = false;
        raw.lflag.IEXTEN = false;
        raw.iflag.IXON = false;
        raw.iflag.ICRNL = false;
        raw.cc[@intFromEnum(posix.system.V.MIN)] = 1;
        raw.cc[@intFromEnum(posix.system.V.TIME)] = 0;
        try posix.tcsetattr(self.fd, .FLUSH, raw);
    }

    fn disableRaw(self: *LineEditor) void {
        if (self.orig_termios) |orig| {
            posix.tcsetattr(self.fd, .FLUSH, orig) catch {};
        }
    }

    /// Poll stdin for input with a timeout in milliseconds. Returns true if input is available.
    fn pollInput(self: *LineEditor, timeout_ms: i32) bool {
        var fds = [1]posix.pollfd{.{
            .fd = self.fd,
            .events = 0x0001, // POLLIN
            .revents = 0,
        }};
        const nr = posix.poll(&fds, timeout_ms) catch return false;
        return nr > 0;
    }

    /// Display width using vaxis grapheme-aware width calculation.
    fn displayWidth(s: []const u8) usize {
        // Filter out ANSI escape sequences for prompt width calculation
        var cols: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            if (s[i] == 0x1b) {
                i += 1;
                if (i < s.len and s[i] == '[') {
                    i += 1;
                    while (i < s.len and s[i] >= 0x20 and s[i] < 0x40) i += 1;
                    if (i < s.len) i += 1;
                }
            } else if (s[i] < 0x20) {
                i += 1;
            } else {
                // Measure the next grapheme cluster
                const cp_len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
                const end = @min(i + cp_len, s.len);
                cols += @as(usize, vaxis.gwidth.gwidth(s[i..end], .unicode));
                i = end;
            }
        }
        return cols;
    }
};
