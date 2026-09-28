//! Self-contained terminal input/output primitives.
//! Replaces the vaxis dependency with pure Zig, no libc calls,
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

        /// Compare two modifier sets for equality via bitwise cast.
        pub fn eql(self: Modifiers, other: Modifiers) bool {
            return @as(u8, @bitCast(self)) == @as(u8, @bitCast(other));
        }
    };

    // Named key constants, use the same values as vaxis (Kitty private-use area)
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

    /// Loose key matching, checks codepoint and the ctrl/alt/shift modifiers.
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
            // Alt+key (ESC + character). Multi-byte UTF-8 after ESC must be
            // consumed as one event: ESC+C3 A9 (Alt+é) is 3 bytes, not Alt
            // of the lead byte leaving a dangling continuation.
            if (buf[1] >= 0x20) {
                if (buf[1] >= 0x80) {
                    const seq_len = std.unicode.utf8ByteSequenceLength(buf[1]) catch
                        return .{ .n = 2, .event = .{ .key_press = .{
                            .codepoint = buf[1],
                            .mods = .{ .alt = true },
                        } } };
                    if (buf.len < 1 + seq_len) return .{}; // need more data
                    const cp = std.unicode.utf8Decode(buf[1..][0..seq_len]) catch
                        return .{ .n = 2, .event = .{ .key_press = .{
                            .codepoint = buf[1],
                            .mods = .{ .alt = true },
                        } } };
                    return .{ .n = 1 + seq_len, .event = .{ .key_press = .{
                        .codepoint = cp,
                        .mods = .{ .alt = true },
                    } } };
                }
                return .{ .n = 2, .event = .{ .key_press = .{
                    .codepoint = buf[1],
                    .mods = .{ .alt = true },
                } } };
            }
            // Bare ESC
            return .{ .n = 1, .event = .{ .key_press = .{ .codepoint = Key.escape } } };
        }

        // Ctrl+letter (0x01-0x1a = Ctrl+a through Ctrl+z, excluding 0x0a=LF,
        // 0x0d=CR=Enter and 0x09=Tab). 0x0a must be excluded or a bare LF is
        // decoded as Ctrl+j and the Enter arm below never fires.
        if (b >= 1 and b <= 26 and b != '\n' and b != '\r' and b != '\t') {
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

        // Other control characters, consume and ignore
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
                            2 => 0, // Insert key, ignore (not used)
                            3 => Key.delete,
                            5 => 0, // Page Up, not used
                            6 => 0, // Page Down, not used
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
/// Widths are counted per extended grapheme cluster, not per codepoint, so a
/// ZWJ emoji sequence, a flag, or a base plus combining marks takes the columns
/// the terminal actually gives it.
/// Pure Zig, no libc wcwidth.
pub fn displayWidth(s: []const u8) usize {
    var w: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const cluster = nextCluster(s, i);
        w += cluster.width;
        i = cluster.end;
    }
    return w;
}

/// One step of the grapheme walk: the cluster's leading codepoint, the byte
/// index just past the cluster, and the terminal columns it occupies.
pub const Cluster = struct {
    cp: u21,
    end: usize,
    width: usize,
};

/// Decode the codepoint at `i`, or null when the bytes there are not one valid
/// complete sequence (bad lead byte, overlong form, surrogate, truncated tail).
fn decodeAt(s: []const u8, i: usize) ?struct { cp: u21, end: usize } {
    const seq_len = std.unicode.utf8ByteSequenceLength(s[i]) catch return null;
    if (i + seq_len > s.len) return null;
    const cp = std.unicode.utf8Decode(s[i..][0..seq_len]) catch return null;
    return .{ .cp = cp, .end = i + seq_len };
}

/// Advance past one extended grapheme cluster starting at `i`.
///
/// A cluster is the leading codepoint plus whatever attaches to it without
/// advancing the cursor: trailing combining marks, variation selectors, skin
/// tones, a ZWJ-joined chain, or a paired regional indicator. Bytes that are not
/// valid UTF-8 step one at a time at one column, so invalid input still makes
/// progress instead of stalling the walk.
pub fn nextCluster(s: []const u8, i: usize) Cluster {
    const first = decodeAt(s, i) orelse return .{ .cp = 0xFFFD, .end = i + 1, .width = 1 };
    const cp = first.cp;
    var end = first.end;

    // A regional indicator pair renders as one 2-column flag; a lone one is
    // still 2 columns on its own.
    if (cp >= cp_ri_lo and cp <= cp_ri_hi) {
        if (decodeAt(s, end)) |second| {
            if (second.cp >= cp_ri_lo and second.cp <= cp_ri_hi) {
                return .{ .cp = cp, .end = second.end, .width = 2 };
            }
        }
        return .{ .cp = cp, .end = end, .width = 2 };
    }

    // The joined characters keep the leading element's width: 👨‍👩‍👧 is one
    // 2-column glyph, not three.
    while (end < s.len) {
        const d = decodeAt(s, end) orelse break;
        if (d.cp == cp_zwj) {
            // ZWJ only glues pictographs (width 2 here). A ZWJ before an
            // ordinary letter is a zero-width mark like any other, so it must
            // not swallow the letter into the previous cluster.
            const joined = decodeAt(s, d.end) orelse break;
            if (codepointWidth(joined.cp) != 2) break;
            end = joined.end;
            continue;
        }
        if (codepointWidth(d.cp) != 0) break;
        end = d.end;
    }
    return .{ .cp = cp, .end = end, .width = codepointWidth(cp) };
}

/// Longest prefix of `s` with byte length ≤ `max_bytes` that does not split a
/// UTF-8 sequence. Invalid trailing bytes that cannot complete a character are
/// dropped. Used by display truncation and metadata sanitization.
pub fn utf8BytePrefix(s: []const u8, max_bytes: usize) []const u8 {
    const cap = @min(s.len, max_bytes);
    var len: usize = cap;
    while (len > 0) {
        const b = s[len - 1];
        if (b & 0x80 == 0) break; // ASCII, clean cut
        if (b & 0xC0 == 0xC0) {
            // Lead byte: keep the sequence only if every byte fits in `cap`.
            const seq_len = std.unicode.utf8ByteSequenceLength(b) catch 1;
            const start = len - 1;
            if (start + seq_len > cap) {
                len = start;
            } else {
                len = start + seq_len;
            }
            break;
        }
        // Continuation byte (10xxxxxx): walk back to the lead.
        len -= 1;
    }
    return s[0..len];
}

/// Zero-width joiner: glues the characters around it into one cluster.
const cp_zwj: u21 = 0x200D;
/// Regional indicators: a pair of them is one flag glyph.
const cp_ri_lo: u21 = 0x1F1E6;
const cp_ri_hi: u21 = 0x1F1FF;
/// Fitzpatrick emoji skin-tone modifiers (👋🏻..👋🏿). Combine with the preceding emoji.
const cp_emoji_skin_tone_lo: u21 = 0x1F3FB;
const cp_emoji_skin_tone_hi: u21 = 0x1F3FF;
/// Hangul Jungseong (medial vowels) through Jongseong (finals). Combining in NFD syllables.
const cp_hangul_jungseong_lo: u21 = 0x1160;
const cp_hangul_jamo_hi: u21 = 0x11FF;
/// Hangul Jamo Extended-B (additional jungseong/jongseong). Combining, width 0.
const cp_hangul_jamo_ext_b_lo: u21 = 0xD7B0;
const cp_hangul_jamo_ext_b_hi: u21 = 0xD7FF;

/// Zero-width codepoints as sorted, non-overlapping (lo, hi) pairs: every
/// codepoint of general category Mn, Me, or Cf. That is the combining marks of
/// Arabic, Hebrew, Devanagari, Thai and the other scripts that attach to the
/// preceding letter, plus the bidi controls and zero-width spaces. Without them
/// a tashkeeled Arabic word or a Thai sentence counts one column per mark, so
/// the cursor drifts while it is typed and centred output lands off-axis.
/// Emoji skin tones and Hangul jamo are not in these categories, so
/// `isZeroWidth` keeps explicit checks for them. Unicode 16.0.
const zero_width_pairs = [_]u21{
    0x00AD,  0x00AD,  0x0300,  0x036F,  0x0483,  0x0489,  0x0591,  0x05BD,
    0x05BF,  0x05BF,  0x05C1,  0x05C2,  0x05C4,  0x05C5,  0x05C7,  0x05C7,
    0x0600,  0x0605,  0x0610,  0x061A,  0x061C,  0x061C,  0x064B,  0x065F,
    0x0670,  0x0670,  0x06D6,  0x06DD,  0x06DF,  0x06E4,  0x06E7,  0x06E8,
    0x06EA,  0x06ED,  0x070F,  0x070F,  0x0711,  0x0711,  0x0730,  0x074A,
    0x07A6,  0x07B0,  0x07EB,  0x07F3,  0x07FD,  0x07FD,  0x0816,  0x0819,
    0x081B,  0x0823,  0x0825,  0x0827,  0x0829,  0x082D,  0x0859,  0x085B,
    0x0890,  0x0891,  0x0897,  0x089F,  0x08CA,  0x0902,  0x093A,  0x093A,
    0x093C,  0x093C,  0x0941,  0x0948,  0x094D,  0x094D,  0x0951,  0x0957,
    0x0962,  0x0963,  0x0981,  0x0981,  0x09BC,  0x09BC,  0x09C1,  0x09C4,
    0x09CD,  0x09CD,  0x09E2,  0x09E3,  0x09FE,  0x09FE,  0x0A01,  0x0A02,
    0x0A3C,  0x0A3C,  0x0A41,  0x0A42,  0x0A47,  0x0A48,  0x0A4B,  0x0A4D,
    0x0A51,  0x0A51,  0x0A70,  0x0A71,  0x0A75,  0x0A75,  0x0A81,  0x0A82,
    0x0ABC,  0x0ABC,  0x0AC1,  0x0AC5,  0x0AC7,  0x0AC8,  0x0ACD,  0x0ACD,
    0x0AE2,  0x0AE3,  0x0AFA,  0x0AFF,  0x0B01,  0x0B01,  0x0B3C,  0x0B3C,
    0x0B3F,  0x0B3F,  0x0B41,  0x0B44,  0x0B4D,  0x0B4D,  0x0B55,  0x0B56,
    0x0B62,  0x0B63,  0x0B82,  0x0B82,  0x0BC0,  0x0BC0,  0x0BCD,  0x0BCD,
    0x0C00,  0x0C00,  0x0C04,  0x0C04,  0x0C3C,  0x0C3C,  0x0C3E,  0x0C40,
    0x0C46,  0x0C48,  0x0C4A,  0x0C4D,  0x0C55,  0x0C56,  0x0C62,  0x0C63,
    0x0C81,  0x0C81,  0x0CBC,  0x0CBC,  0x0CBF,  0x0CBF,  0x0CC6,  0x0CC6,
    0x0CCC,  0x0CCD,  0x0CE2,  0x0CE3,  0x0D00,  0x0D01,  0x0D3B,  0x0D3C,
    0x0D41,  0x0D44,  0x0D4D,  0x0D4D,  0x0D62,  0x0D63,  0x0D81,  0x0D81,
    0x0DCA,  0x0DCA,  0x0DD2,  0x0DD4,  0x0DD6,  0x0DD6,  0x0E31,  0x0E31,
    0x0E34,  0x0E3A,  0x0E47,  0x0E4E,  0x0EB1,  0x0EB1,  0x0EB4,  0x0EBC,
    0x0EC8,  0x0ECE,  0x0F18,  0x0F19,  0x0F35,  0x0F35,  0x0F37,  0x0F37,
    0x0F39,  0x0F39,  0x0F71,  0x0F7E,  0x0F80,  0x0F84,  0x0F86,  0x0F87,
    0x0F8D,  0x0F97,  0x0F99,  0x0FBC,  0x0FC6,  0x0FC6,  0x102D,  0x1030,
    0x1032,  0x1037,  0x1039,  0x103A,  0x103D,  0x103E,  0x1058,  0x1059,
    0x105E,  0x1060,  0x1071,  0x1074,  0x1082,  0x1082,  0x1085,  0x1086,
    0x108D,  0x108D,  0x109D,  0x109D,  0x135D,  0x135F,  0x1712,  0x1714,
    0x1732,  0x1733,  0x1752,  0x1753,  0x1772,  0x1773,  0x17B4,  0x17B5,
    0x17B7,  0x17BD,  0x17C6,  0x17C6,  0x17C9,  0x17D3,  0x17DD,  0x17DD,
    0x180B,  0x180F,  0x1885,  0x1886,  0x18A9,  0x18A9,  0x1920,  0x1922,
    0x1927,  0x1928,  0x1932,  0x1932,  0x1939,  0x193B,  0x1A17,  0x1A18,
    0x1A1B,  0x1A1B,  0x1A56,  0x1A56,  0x1A58,  0x1A5E,  0x1A60,  0x1A60,
    0x1A62,  0x1A62,  0x1A65,  0x1A6C,  0x1A73,  0x1A7C,  0x1A7F,  0x1A7F,
    0x1AB0,  0x1ACE,  0x1B00,  0x1B03,  0x1B34,  0x1B34,  0x1B36,  0x1B3A,
    0x1B3C,  0x1B3C,  0x1B42,  0x1B42,  0x1B6B,  0x1B73,  0x1B80,  0x1B81,
    0x1BA2,  0x1BA5,  0x1BA8,  0x1BA9,  0x1BAB,  0x1BAD,  0x1BE6,  0x1BE6,
    0x1BE8,  0x1BE9,  0x1BED,  0x1BED,  0x1BEF,  0x1BF1,  0x1C2C,  0x1C33,
    0x1C36,  0x1C37,  0x1CD0,  0x1CD2,  0x1CD4,  0x1CE0,  0x1CE2,  0x1CE8,
    0x1CED,  0x1CED,  0x1CF4,  0x1CF4,  0x1CF8,  0x1CF9,  0x1DC0,  0x1DFF,
    0x200B,  0x200F,  0x202A,  0x202E,  0x2060,  0x2064,  0x2066,  0x206F,
    0x20D0,  0x20F0,  0x2CEF,  0x2CF1,  0x2D7F,  0x2D7F,  0x2DE0,  0x2DFF,
    0x302A,  0x302D,  0x3099,  0x309A,  0xA66F,  0xA672,  0xA674,  0xA67D,
    0xA69E,  0xA69F,  0xA6F0,  0xA6F1,  0xA802,  0xA802,  0xA806,  0xA806,
    0xA80B,  0xA80B,  0xA825,  0xA826,  0xA82C,  0xA82C,  0xA8C4,  0xA8C5,
    0xA8E0,  0xA8F1,  0xA8FF,  0xA8FF,  0xA926,  0xA92D,  0xA947,  0xA951,
    0xA980,  0xA982,  0xA9B3,  0xA9B3,  0xA9B6,  0xA9B9,  0xA9BC,  0xA9BD,
    0xA9E5,  0xA9E5,  0xAA29,  0xAA2E,  0xAA31,  0xAA32,  0xAA35,  0xAA36,
    0xAA43,  0xAA43,  0xAA4C,  0xAA4C,  0xAA7C,  0xAA7C,  0xAAB0,  0xAAB0,
    0xAAB2,  0xAAB4,  0xAAB7,  0xAAB8,  0xAABE,  0xAABF,  0xAAC1,  0xAAC1,
    0xAAEC,  0xAAED,  0xAAF6,  0xAAF6,  0xABE5,  0xABE5,  0xABE8,  0xABE8,
    0xABED,  0xABED,  0xFB1E,  0xFB1E,  0xFE00,  0xFE0F,  0xFE20,  0xFE2F,
    0xFEFF,  0xFEFF,  0xFFF9,  0xFFFB,  0x101FD, 0x101FD, 0x102E0, 0x102E0,
    0x10376, 0x1037A, 0x10A01, 0x10A03, 0x10A05, 0x10A06, 0x10A0C, 0x10A0F,
    0x10A38, 0x10A3A, 0x10A3F, 0x10A3F, 0x10AE5, 0x10AE6, 0x10D24, 0x10D27,
    0x10D69, 0x10D6D, 0x10EAB, 0x10EAC, 0x10EFC, 0x10EFF, 0x10F46, 0x10F50,
    0x10F82, 0x10F85, 0x11001, 0x11001, 0x11038, 0x11046, 0x11070, 0x11070,
    0x11073, 0x11074, 0x1107F, 0x11081, 0x110B3, 0x110B6, 0x110B9, 0x110BA,
    0x110BD, 0x110BD, 0x110C2, 0x110C2, 0x110CD, 0x110CD, 0x11100, 0x11102,
    0x11127, 0x1112B, 0x1112D, 0x11134, 0x11173, 0x11173, 0x11180, 0x11181,
    0x111B6, 0x111BE, 0x111C9, 0x111CC, 0x111CF, 0x111CF, 0x1122F, 0x11231,
    0x11234, 0x11234, 0x11236, 0x11237, 0x1123E, 0x1123E, 0x11241, 0x11241,
    0x112DF, 0x112DF, 0x112E3, 0x112EA, 0x11300, 0x11301, 0x1133B, 0x1133C,
    0x11340, 0x11340, 0x11366, 0x1136C, 0x11370, 0x11374, 0x113BB, 0x113C0,
    0x113CE, 0x113CE, 0x113D0, 0x113D0, 0x113D2, 0x113D2, 0x113E1, 0x113E2,
    0x11438, 0x1143F, 0x11442, 0x11444, 0x11446, 0x11446, 0x1145E, 0x1145E,
    0x114B3, 0x114B8, 0x114BA, 0x114BA, 0x114BF, 0x114C0, 0x114C2, 0x114C3,
    0x115B2, 0x115B5, 0x115BC, 0x115BD, 0x115BF, 0x115C0, 0x115DC, 0x115DD,
    0x11633, 0x1163A, 0x1163D, 0x1163D, 0x1163F, 0x11640, 0x116AB, 0x116AB,
    0x116AD, 0x116AD, 0x116B0, 0x116B5, 0x116B7, 0x116B7, 0x1171D, 0x1171D,
    0x1171F, 0x1171F, 0x11722, 0x11725, 0x11727, 0x1172B, 0x1182F, 0x11837,
    0x11839, 0x1183A, 0x1193B, 0x1193C, 0x1193E, 0x1193E, 0x11943, 0x11943,
    0x119D4, 0x119D7, 0x119DA, 0x119DB, 0x119E0, 0x119E0, 0x11A01, 0x11A0A,
    0x11A33, 0x11A38, 0x11A3B, 0x11A3E, 0x11A47, 0x11A47, 0x11A51, 0x11A56,
    0x11A59, 0x11A5B, 0x11A8A, 0x11A96, 0x11A98, 0x11A99, 0x11C30, 0x11C36,
    0x11C38, 0x11C3D, 0x11C3F, 0x11C3F, 0x11C92, 0x11CA7, 0x11CAA, 0x11CB0,
    0x11CB2, 0x11CB3, 0x11CB5, 0x11CB6, 0x11D31, 0x11D36, 0x11D3A, 0x11D3A,
    0x11D3C, 0x11D3D, 0x11D3F, 0x11D45, 0x11D47, 0x11D47, 0x11D90, 0x11D91,
    0x11D95, 0x11D95, 0x11D97, 0x11D97, 0x11EF3, 0x11EF4, 0x11F00, 0x11F01,
    0x11F36, 0x11F3A, 0x11F40, 0x11F40, 0x11F42, 0x11F42, 0x11F5A, 0x11F5A,
    0x13430, 0x13440, 0x13447, 0x13455, 0x1611E, 0x16129, 0x1612D, 0x1612F,
    0x16AF0, 0x16AF4, 0x16B30, 0x16B36, 0x16F4F, 0x16F4F, 0x16F8F, 0x16F92,
    0x16FE4, 0x16FE4, 0x1BC9D, 0x1BC9E, 0x1BCA0, 0x1BCA3, 0x1CF00, 0x1CF2D,
    0x1CF30, 0x1CF46, 0x1D167, 0x1D169, 0x1D173, 0x1D182, 0x1D185, 0x1D18B,
    0x1D1AA, 0x1D1AD, 0x1D242, 0x1D244, 0x1DA00, 0x1DA36, 0x1DA3B, 0x1DA6C,
    0x1DA75, 0x1DA75, 0x1DA84, 0x1DA84, 0x1DA9B, 0x1DA9F, 0x1DAA1, 0x1DAAF,
    0x1E000, 0x1E006, 0x1E008, 0x1E018, 0x1E01B, 0x1E021, 0x1E023, 0x1E024,
    0x1E026, 0x1E02A, 0x1E08F, 0x1E08F, 0x1E130, 0x1E136, 0x1E2AE, 0x1E2AE,
    0x1E2EC, 0x1E2EF, 0x1E4EC, 0x1E4EF, 0x1E5EE, 0x1E5EF, 0x1E8D0, 0x1E8D6,
    0x1E944, 0x1E94A, 0xE0001, 0xE0001, 0xE0020, 0xE007F, 0xE0100, 0xE01EF,
};

/// Whether `cp` takes no columns: one of the zero-width ranges, an emoji
/// skin-tone modifier, or a Hangul jamo.
fn isZeroWidth(cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = zero_width_pairs.len / 2;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < zero_width_pairs[mid * 2]) {
            hi = mid;
        } else if (cp > zero_width_pairs[mid * 2 + 1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return (cp >= cp_emoji_skin_tone_lo and cp <= cp_emoji_skin_tone_hi) or
        (cp >= cp_hangul_jungseong_lo and cp <= cp_hangul_jamo_hi) or
        (cp >= cp_hangul_jamo_ext_b_lo and cp <= cp_hangul_jamo_ext_b_hi);
}

/// Width of a single codepoint. CJK fullwidth = 2, combining = 0, most = 1.
fn codepointWidth(cp: u21) usize {
    // Zero-width: combining marks, bidi controls, variation selectors
    if (isZeroWidth(cp)) return 0;

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

    // Emoji: misc technical, misc symbols, dingbats, pictographs, supplemental
    if ((cp >= 0x231A and cp <= 0x23FF) or
        (cp >= 0x2600 and cp <= 0x27BF) or
        (cp >= 0x1F300 and cp <= 0x1F9FF) or
        (cp >= 0x1FA00 and cp <= 0x1FAFF)) return 2;

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

    /// Create a new `TextInput` backed by a gap buffer using the given allocator.
    pub fn init(allocator: std.mem.Allocator) TextInput {
        return .{ .buf = Buffer.init(allocator) };
    }

    /// Free the underlying gap buffer memory.
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

    /// The underlying gap buffer, exposed for readline.zig which accesses
    /// `input.buf.firstHalf()`, `input.buf.secondHalf()`, `input.buf.realLength()`.
    pub const Buffer = struct {
        allocator: std.mem.Allocator,
        buffer: []u8,
        cursor: usize,
        gap_size: usize,

        /// Create an empty gap buffer. No memory is allocated until the first insert.
        pub fn init(allocator: std.mem.Allocator) Buffer {
            return .{
                .allocator = allocator,
                .buffer = &.{},
                .cursor = 0,
                .gap_size = 0,
            };
        }

        /// Free the backing allocation, if any.
        pub fn deinit(self: *Buffer) void {
            if (self.buffer.len > 0) self.allocator.free(self.buffer);
        }

        /// Return the content before the cursor (gap start).
        pub fn firstHalf(self: Buffer) []const u8 {
            return self.buffer[0..self.cursor];
        }

        /// Return the content after the cursor (gap end).
        pub fn secondHalf(self: Buffer) []const u8 {
            return self.buffer[self.cursor + self.gap_size ..];
        }

        /// Logical content length excluding the internal gap.
        pub fn realLength(self: *const Buffer) usize {
            return self.firstHalf().len + self.secondHalf().len;
        }

        /// Insert a byte slice at the current cursor position.
        pub fn insertSliceAtCursor(self: *Buffer, slice: []const u8) std.mem.Allocator.Error!void {
            if (slice.len == 0) return;
            if (self.gap_size <= slice.len) try self.grow(slice.len);
            @memcpy(self.buffer[self.cursor .. self.cursor + slice.len], slice);
            self.cursor += slice.len;
            self.gap_size -= slice.len;
        }

        /// Move the cursor left by n positions.
        pub fn moveGapLeft(self: *Buffer, n: usize) void {
            const new_idx = self.cursor -| n;
            const dst = self.buffer[new_idx + self.gap_size ..];
            const src = self.buffer[new_idx..self.cursor];
            std.mem.copyForwards(u8, dst, src);
            self.cursor = new_idx;
        }

        /// Move the cursor right by n positions. Callers must not move past
        /// the second half; there is no room to grow the gap from the right.
        pub fn moveGapRight(self: *Buffer, n: usize) void {
            const new_idx = self.cursor + n;
            std.debug.assert(new_idx + self.gap_size <= self.buffer.len);
            const dst = self.buffer[self.cursor..];
            const src = self.buffer[self.cursor + self.gap_size .. new_idx + self.gap_size];
            std.mem.copyForwards(u8, dst, src);
            self.cursor = new_idx;
        }

        /// Delete n characters to the left of the cursor.
        pub fn growGapLeft(self: *Buffer, n: usize) void {
            self.gap_size += n;
            self.cursor -|= n;
        }

        /// Delete n characters to the right of the cursor.
        pub fn growGapRight(self: *Buffer, n: usize) void {
            self.gap_size = @min(self.gap_size + n, self.buffer.len - self.cursor);
        }

        /// Reset the buffer content without freeing memory.
        pub fn clearRetainingCapacity(self: *Buffer) void {
            self.cursor = 0;
            self.gap_size = self.buffer.len;
        }

        /// Compact the gap and return the content as an owned slice.
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
    // "a · b", the middot (U+00B7) is 2 bytes in UTF-8 but occupies 1 terminal column
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

test "displayWidth arabic with tashkeel" {
    // مَرحبا: five letters, the fatha over the mīm takes no column
    try std.testing.expectEqual(@as(usize, 5), displayWidth("\xd9\x85\xd9\x8e\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7"));
    // שָׁל: two letters, qamats and dagesh are marks
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xd7\xa9\xd6\xb8\xd7\x81\xd7\x9c"));
}

test "displayWidth thai and devanagari marks" {
    // ภาษาไทย: seven letters, none of the marks add a column
    try std.testing.expectEqual(@as(usize, 7), displayWidth("\xe0\xb8\xa0\xe0\xb8\xb2\xe0\xb8\xa9\xe0\xb8\xb2\xe0\xb9\x84\xe0\xb8\x97\xe0\xb8\xa2"));
    // नमस्ते: four letters, the virama and the vowel sign are marks
    try std.testing.expectEqual(@as(usize, 4), displayWidth("\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87"));
}

test "displayWidth bidi controls are zero width" {
    // LRM, RLM and ALM carry no column of their own.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("a\xe2\x80\x8e\xe2\x80\x8f\xd8\x9cb"));
    // The isolate controls around a quoted latin run.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xe2\x81\xa8ok\xe2\x81\xa9"));
}

test "displayWidth cjk ideographic tone mark" {
    // U+302A sits on the preceding kanji, which keeps both its columns.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xe4\xb8\x80\xe3\x80\xaa"));
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

test "parser: alt+utf8 consumes the full character" {
    var parser: Parser = .{};
    // ESC + é (U+00E9, UTF-8 C3 A9). Must consume all 3 bytes.
    const result = try parser.parse("\x1b\xc3\xa9", null);
    try std.testing.expectEqual(@as(usize, 3), result.n);
    try std.testing.expectEqual(@as(u21, 0xe9), result.event.?.key_press.codepoint);
    try std.testing.expect(result.event.?.key_press.mods.alt);

    // Incomplete: ESC + lead byte only, wait for continuation.
    const incomplete = try parser.parse("\x1b\xc3", null);
    try std.testing.expectEqual(@as(usize, 0), incomplete.n);
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

test "displayWidth control characters" {
    // Control chars (< 0x20) should have zero width
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x00));
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x1F));
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x7F));
}

test "codepointWidth zero-width joiner" {
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x200B)); // ZWSP
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x200C)); // ZWNJ
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x200D)); // ZWJ
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0xFEFF)); // BOM
}

test "codepointWidth variation selectors" {
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0xFE00));
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0xFE0F));
}

test "codepointWidth emoji skin tone modifier is zero" {
    // U+1F3FB..U+1F3FF must not add columns; 👋🏻 is one emoji, width 2, not 4.
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x1F3FB));
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x1F3FF));
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xf0\x9f\x91\x8b\xf0\x9f\x8f\xbb")); // 👋🏻
}

test "codepointWidth NFD Hangul jamo combines to syllable width" {
    // NFC 가 (U+AC00) is width 2; NFD 가 (U+1100 + U+1161) must match, not 2+1.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xea\xb0\x80")); // 가
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xe1\x84\x80\xe1\x85\xa1")); // 가
    try std.testing.expectEqual(@as(usize, 0), codepointWidth(0x1161));
}

test "utf8BytePrefix does not split multi-byte characters" {
    const cafe = "caf\xc3\xa9"; // café
    try std.testing.expectEqualStrings("caf", utf8BytePrefix(cafe, 4));
    try std.testing.expectEqualStrings(cafe, utf8BytePrefix(cafe, 5));
    try std.testing.expectEqualStrings(cafe, utf8BytePrefix(cafe, 64));

    // 16 CJK chars = 48 bytes; a 47-byte cap must drop the split last char.
    const cjk = "\xe4\xb8\x96" ** 16;
    const clipped = utf8BytePrefix(cjk, 47);
    try std.testing.expect(std.unicode.utf8ValidateSlice(clipped));
    try std.testing.expectEqual(@as(usize, 45), clipped.len);

    try std.testing.expectEqualStrings("", utf8BytePrefix("\xc3\xa9", 1));
    try std.testing.expectEqualStrings("", utf8BytePrefix("", 8));
}

test "displayWidth ZWJ emoji sequence is one glyph" {
    // 👨‍👩‍👧‍👦 is four emoji joined by three ZWJ: 2 columns, not 8.
    const family = "\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7\xe2\x80\x8d\xf0\x9f\x91\xa6";
    try std.testing.expectEqual(@as(usize, 2), displayWidth(family));
    try std.testing.expectEqual(@as(usize, 3), displayWidth("a" ++ family));
    // A ZWJ before an ordinary letter glues nothing: the letter is its own cluster.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("a\xe2\x80\x8d" ++ "b"));
}

test "displayWidth regional indicator pair is one flag" {
    // 🇯🇵 (U+1F1EF U+1F1F5): one 2-column glyph.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\xf0\x9f\x87\xaf\xf0\x9f\x87\xb5"));
    // A lone indicator is still 2 columns; the next one is not absorbed.
    try std.testing.expectEqual(@as(usize, 3), displayWidth("\xf0\x9f\x87\xafx"));
}

test "nextCluster never loops on invalid bytes" {
    const bad = "a\xff\x80b";
    var i: usize = 0;
    var cols: usize = 0;
    while (i < bad.len) {
        const c = nextCluster(bad, i);
        try std.testing.expect(c.end > i);
        cols += c.width;
        i = c.end;
    }
    try std.testing.expectEqual(@as(usize, 4), cols);
}

test "parser: backspace" {
    var parser: Parser = .{};
    const result = try parser.parse("\x7f", null);
    try std.testing.expectEqual(@as(usize, 1), result.n);
    try std.testing.expectEqual(Key.backspace, result.event.?.key_press.codepoint);
}

test "parser: home and end CSI" {
    var parser: Parser = .{};
    const home_result = try parser.parse("\x1b[H", null);
    try std.testing.expectEqual(Key.home, home_result.event.?.key_press.codepoint);

    const end_result = try parser.parse("\x1b[F", null);
    try std.testing.expectEqual(Key.end, end_result.event.?.key_press.codepoint);
}

test "parser: empty buffer returns zero" {
    var parser: Parser = .{};
    const result = try parser.parse("", null);
    try std.testing.expectEqual(@as(usize, 0), result.n);
    try std.testing.expect(result.event == null);
}

test "Key.Modifiers.eql" {
    const m1: Key.Modifiers = .{ .ctrl = true };
    const m2: Key.Modifiers = .{ .ctrl = true };
    const m3: Key.Modifiers = .{ .alt = true };
    try std.testing.expect(m1.eql(m2));
    try std.testing.expect(!m1.eql(m3));
}

test "gap buffer clearRetainingCapacity" {
    var buf = TextInput.Buffer.init(std.testing.allocator);
    defer buf.deinit();

    try buf.insertSliceAtCursor("hello world");
    try std.testing.expect(buf.realLength() > 0);

    buf.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), buf.realLength());
    try std.testing.expectEqual(@as(usize, 0), buf.cursor);
    // Buffer memory is still allocated (capacity retained)
    try std.testing.expect(buf.buffer.len > 0);
}

test "fuzz: all term functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── 1. Key.Modifiers.eql ──
            const mod_a: Key.Modifiers = @bitCast(smith.valueWithHash(u8, 0));
            const mod_b: Key.Modifiers = @bitCast(smith.valueWithHash(u8, 1));
            const eq = mod_a.eql(mod_b);
            // reflexivity: a mod must equal itself
            try std.testing.expect(mod_a.eql(mod_a));
            _ = eq;

            // ── 2. Key.matches ──
            const cp_a = smith.valueWithHash(u21, 2) % 0x110000;
            const cp_b = smith.valueWithHash(u21, 3) % 0x110000;
            const key: Key = .{ .codepoint = cp_a, .mods = mod_a };
            _ = key.matches(cp_b, mod_b);

            // ── 3. displayWidth ──
            // Build a small random buffer as input
            var dw_buf: [16]u8 = undefined;
            for (&dw_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(100 + i));
            const dw_len = smith.valueWithHash(u4, 50);
            const dw_slice = dw_buf[0..dw_len];
            const w = displayWidth(dw_slice);
            try std.testing.expect(w <= dw_slice.len * 2); // max 2 columns per byte

            // utf8BytePrefix: never longer than cap, never splits a sequence
            const cap = smith.valueWithHash(u4, 51);
            const prefix = utf8BytePrefix(dw_slice, cap);
            try std.testing.expect(prefix.len <= @min(dw_slice.len, cap));
            if (std.unicode.utf8ValidateSlice(dw_slice)) {
                try std.testing.expect(std.unicode.utf8ValidateSlice(prefix));
            }

            // ── 4. Parser.parse ──
            var parser: Parser = .{};
            var parse_buf: [8]u8 = undefined;
            for (&parse_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(200 + i));
            const parse_len = smith.valueWithHash(u4, 60);
            const result = parser.parse(parse_buf[0..parse_len], null) catch return;
            try std.testing.expect(result.n <= parse_len);

            // ── 5-10. TextInput (init, deinit, insertSliceAtCursor,
            //          clearRetainingCapacity, toOwnedSlice, update) ──
            var input = TextInput.init(std.testing.allocator);
            defer input.deinit();

            // insertSliceAtCursor with random slice
            var ins_buf: [8]u8 = undefined;
            for (&ins_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(300 + i));
            const ins_len = smith.valueWithHash(u3, 70);
            input.insertSliceAtCursor(ins_buf[0..ins_len]) catch return;

            // update with a random key event
            const ev_cp = smith.valueWithHash(u21, 80) % 0x110000;
            const ev_mod: Key.Modifiers = @bitCast(smith.valueWithHash(u8, 81));
            input.update(.{ .key_press = .{
                .codepoint = ev_cp,
                .mods = ev_mod,
            } }) catch return;

            // clearRetainingCapacity
            input.clearRetainingCapacity();
            try std.testing.expectEqual(@as(usize, 0), input.buf.realLength());

            // Re-insert so toOwnedSlice has something to return
            input.insertSliceAtCursor("fz") catch return;
            const owned = input.toOwnedSlice() catch return;
            std.testing.allocator.free(owned);

            // ── 11-22. TextInput.Buffer (init, deinit, firstHalf, secondHalf,
            //           realLength, insertSliceAtCursor, moveGapLeft, moveGapRight,
            //           growGapLeft, growGapRight, clearRetainingCapacity,
            //           toOwnedSlice) ──
            var buf = TextInput.Buffer.init(std.testing.allocator);
            defer buf.deinit();

            // insertSliceAtCursor
            var buf_data: [6]u8 = undefined;
            for (&buf_data, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(400 + i));
            const buf_ins_len = smith.valueWithHash(u3, 90);
            buf.insertSliceAtCursor(buf_data[0..buf_ins_len]) catch return;

            // firstHalf / secondHalf / realLength
            const fh = buf.firstHalf();
            const sh = buf.secondHalf();
            try std.testing.expectEqual(fh.len + sh.len, buf.realLength());

            // moveGapLeft, clamp to firstHalf length
            const ml = smith.valueWithHash(u3, 91);
            buf.moveGapLeft(@min(ml, buf.firstHalf().len));

            // moveGapRight, clamp to secondHalf length
            const mr = smith.valueWithHash(u3, 92);
            buf.moveGapRight(@min(mr, buf.secondHalf().len));

            // growGapLeft, clamp to cursor
            const gl = smith.valueWithHash(u3, 93);
            buf.growGapLeft(@min(gl, buf.cursor));

            // growGapRight, clamp to secondHalf length
            const gr = smith.valueWithHash(u3, 94);
            buf.growGapRight(@min(gr, buf.secondHalf().len));

            // clearRetainingCapacity
            buf.clearRetainingCapacity();
            try std.testing.expectEqual(@as(usize, 0), buf.realLength());

            // Re-insert for toOwnedSlice
            buf.insertSliceAtCursor("ab") catch return;
            const buf_owned = buf.toOwnedSlice() catch return;
            try std.testing.expectEqual(@as(usize, 2), buf_owned.len);
            std.testing.allocator.free(buf_owned);
        }
    }.f, .{});
}
