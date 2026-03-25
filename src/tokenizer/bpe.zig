//! Byte-level BPE tokenizer supporting BPE, SPM, and SPM-no-dummy modes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tok_iface = @import("tokenizer.zig");
const TokenizerIface = tok_iface.Tokenizer;
const TokenizerKind = @import("../arch.zig").TokenizerKind;

const max_path_buf_size: usize = 4096;
const max_vocab_file_size: usize = 64 * 1024 * 1024;
const max_config_file_size: usize = 1024 * 1024;
const merge_key_buf_size: usize = 512;
const max_spm_token_len: usize = 64;
const fallback_unknown_token_id: u32 = 3;
/// SPM word-initial marker: U+2581 LOWER ONE EIGHTH BLOCK (▁), UTF-8 encoded.
const spm_prefix = "\xe2\x96\x81";
/// Default Qwen EOS token ID, used when tokenizer.json doesn't specify one.
const qwen_default_eos_id: u32 = 151645;
/// Default Qwen BOS token ID, used when tokenizer.json doesn't specify one.
const qwen_default_bos_id: u32 = 151643;

// ── GPT-2 byte-to-unicode mapping ranges (OpenAI BPE specification) ──
/// First printable ASCII codepoint (maps 1:1 in GPT-2 byte encoder).
const gpt2_printable_min: u8 = 33; // '!'
/// Last printable ASCII codepoint (maps 1:1 in GPT-2 byte encoder).
const gpt2_printable_max: u8 = 126; // '~'
/// Start of Latin-1 Supplement passthrough range (maps 1:1 in GPT-2 byte encoder).
const gpt2_latin1_min: u8 = 161; // '¡'
/// End of first Latin-1 Supplement passthrough sub-range (soft hyphen excluded).
const gpt2_latin1_mid: u8 = 172; // '¬'
/// Start of second Latin-1 Supplement passthrough sub-range (after soft hyphen).
const gpt2_latin1_resume: u8 = 174; // '®'

/// Byte-level BPE tokenizer supporting both BPE (with merges) and SPM (greedy longest-match) modes.
pub const BpeTokenizer = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.ArrayList([]const u8) = .empty,
    special_tokens: std.StringHashMap(u32),
    merge_map: std.StringHashMap(u32),
    byte_to_unicode: [256][]const u8 = [_][]const u8{&.{}} ** 256,
    unicode_to_byte: std.StringHashMap(u8),
    byte_mappings_init: bool = false,
    vocab_size: u32 = 0,
    eos_token_id: u32 = qwen_default_eos_id,
    bos_token_id: u32 = qwen_default_bos_id,
    tok_kind: TokenizerKind = .bpe,
    allocator: Allocator,
    // Owned memory for duped strings
    owned_strings: std.ArrayList([]const u8) = .empty,

    /// Return the generic Tokenizer interface backed by this BPE tokenizer.
    pub fn tokenizer(self: *BpeTokenizer) TokenizerIface {
        return .{ .ptr = self, .vtable = &tok_vtable };
    }
    const tok_vtable = TokenizerIface.VTable{
        .encode = @ptrCast(&tokEncode),
        .decode = @ptrCast(&tokDecode),
        .get_vocab_size = @ptrCast(&tokGetVocabSize),
    };
    fn tokEncode(self: *BpeTokenizer, text: []const u8) tok_iface.TokenizerError![]u32 {
        return switch (self.tok_kind) {
            .bpe => self.encode(text),
            .spm => self.encodeSpm(text),
            .spm_no_dummy => self.encodeSpmNoDummy(text),
        };
    }
    fn tokDecode(self: *BpeTokenizer, tokens: []const u32) tok_iface.TokenizerError![]u8 {
        return switch (self.tok_kind) {
            .spm, .spm_no_dummy => self.decodeSpm(tokens),
            .bpe => self.decode(tokens),
        };
    }
    fn tokGetVocabSize(self: *BpeTokenizer) u32 {
        return self.vocab_size;
    }

    /// Create a new BPE tokenizer. Caller must call deinit() when done.
    pub fn init(allocator: Allocator) BpeTokenizer {
        return .{
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .special_tokens = std.StringHashMap(u32).init(allocator),
            .merge_map = std.StringHashMap(u32).init(allocator),
            .unicode_to_byte = std.StringHashMap(u8).init(allocator),
            .allocator = allocator,
        };
    }

    /// Free all owned memory (vocab, merges, byte mappings).
    pub fn deinit(self: *BpeTokenizer) void {
        // Free byte_to_unicode mappings allocated by initByteMappings
        if (self.byte_mappings_init) {
            for (&self.byte_to_unicode) |s| {
                // Only free heap-allocated slices (allocated via self.allocator in initByteMappings).
                if (s.len > 0) self.allocator.free(s);
            }
        }
        self.token_to_id.deinit();
        self.id_to_token.deinit(self.allocator);
        self.special_tokens.deinit();
        self.merge_map.deinit();
        self.unicode_to_byte.deinit();
        for (self.owned_strings.items) |s| self.allocator.free(s);
        self.owned_strings.deinit(self.allocator);
    }

    fn own(self: *BpeTokenizer, s: []const u8) ![]const u8 {
        const d = try self.allocator.dupe(u8, s);
        errdefer self.allocator.free(d);
        try self.owned_strings.append(self.allocator, d);
        return d;
    }

    /// Load tokenizer data (vocabulary, merge rules, and config) from a directory.
    pub fn loadFromDir(self: *BpeTokenizer, dir: []const u8) !void {
        var vp: [max_path_buf_size]u8 = undefined;
        const vocab_path = try std.fmt.bufPrint(&vp, "{s}/vocab.txt", .{dir});
        var mp: [max_path_buf_size]u8 = undefined;
        const merges_path = try std.fmt.bufPrint(&mp, "{s}/merges.txt", .{dir});
        var cp: [max_path_buf_size]u8 = undefined;
        const config_path = try std.fmt.bufPrint(&cp, "{s}/tokenizer_config.txt", .{dir});

        // Load vocab
        {
            const file = try std.fs.cwd().openFile(vocab_path, .{});
            defer file.close();
            const content = try file.readToEndAlloc(self.allocator, max_vocab_file_size);
            defer self.allocator.free(content);
            var id: u32 = 0;
            var lines = std.mem.splitScalar(u8, content, '\n');
            while (lines.next()) |raw_line| {
                var line = raw_line;
                if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
                const tok = try self.own(line);
                try self.token_to_id.put(tok, id);
                try self.id_to_token.append(self.allocator, tok);
                if (tok.len > 0 and tok[0] == '<' and tok[tok.len - 1] == '>') {
                    try self.special_tokens.put(tok, id);
                }
                id += 1;
            }
            self.vocab_size = id;
        }

        // Load merges
        {
            const file = try std.fs.cwd().openFile(merges_path, .{});
            defer file.close();
            const content = try file.readToEndAlloc(self.allocator, max_vocab_file_size);
            defer self.allocator.free(content);
            var priority: u32 = 0;
            var lines = std.mem.splitScalar(u8, content, '\n');
            while (lines.next()) |raw_line| {
                var line = raw_line;
                if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
                if (line.len == 0 or line[0] == '#') continue;
                const sp = std.mem.indexOf(u8, line, " ") orelse continue;
                const first = line[0..sp];
                const second = line[sp + 1 ..];
                // Key: "first\x00second"
                var key_buf: [merge_key_buf_size]u8 = undefined;
                const key_len = first.len + 1 + second.len;
                if (key_len > key_buf.len) continue;
                @memcpy(key_buf[0..first.len], first);
                key_buf[first.len] = 0;
                @memcpy(key_buf[first.len + 1 ..][0..second.len], second);
                const key = try self.own(key_buf[0..key_len]);
                if (!self.merge_map.contains(key)) {
                    try self.merge_map.put(key, priority);
                }
                priority += 1;
            }
        }

        // Load tokenizer config
        {
            const file = std.fs.cwd().openFile(config_path, .{}) catch return;
            defer file.close();
            const content = file.readToEndAlloc(self.allocator, max_config_file_size) catch return;
            defer self.allocator.free(content);
            var lines = std.mem.splitScalar(u8, content, '\n');
            while (lines.next()) |raw_line| {
                const line = std.mem.trim(u8, raw_line, " \t\r");
                if (line.len == 0 or line[0] == '#') continue;
                const eq = std.mem.indexOf(u8, line, "=") orelse continue;
                const key = std.mem.trim(u8, line[0..eq], " \t");
                const val = std.mem.trim(u8, line[eq + 1 ..], " \t");
                if (std.mem.eql(u8, key, "eos_token_id")) self.eos_token_id = std.fmt.parseInt(u32, val, 10) catch continue;
                // Qwen uses pad_token_id as BOS token in tokenizer_config.txt
                if (std.mem.eql(u8, key, "pad_token_id")) self.bos_token_id = std.fmt.parseInt(u32, val, 10) catch continue;
            }
        }

        try self.initByteMappings();
    }

    fn initByteMappings(self: *BpeTokenizer) !void {
        if (self.byte_mappings_init) return;
        self.byte_mappings_init = true;
        errdefer {
            // Clean up partially-allocated byte_to_unicode entries on error.
            for (&self.byte_to_unicode) |*s| {
                if (s.len > 0) {
                    self.allocator.free(s.*);
                    s.* = &.{};
                }
            }
            self.byte_mappings_init = false;
        }
        var unicode_start: u21 = 256;
        for (0..256) |b| {
            const byte: u8 = @intCast(b);
            if ((byte >= gpt2_printable_min and byte <= gpt2_printable_max)) {
                self.byte_to_unicode[b] = try self.allocator.dupe(u8, &[_]u8{byte});
            } else if ((byte >= gpt2_latin1_min and byte <= gpt2_latin1_mid) or byte >= gpt2_latin1_resume) {
                // 2-byte UTF-8 for codepoints 161-255
                var buf: [2]u8 = undefined;
                buf[0] = 0xC0 | (byte >> 6);
                buf[1] = 0x80 | (byte & 0x3F);
                self.byte_to_unicode[b] = try self.allocator.dupe(u8, &buf);
            } else {
                // Map to unicode_start++
                var buf: [3]u8 = undefined;
                const cp = unicode_start;
                unicode_start += 1;
                if (cp < 0x800) {
                    buf[0] = @intCast(0xC0 | (cp >> 6));
                    buf[1] = @intCast(0x80 | (cp & 0x3F));
                    self.byte_to_unicode[b] = try self.allocator.dupe(u8, buf[0..2]);
                } else {
                    buf[0] = @intCast(0xE0 | (cp >> 12));
                    buf[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
                    buf[2] = @intCast(0x80 | (cp & 0x3F));
                    self.byte_to_unicode[b] = try self.allocator.dupe(u8, buf[0..3]);
                }
            }
            // HashMap put can fail on OOM; log and continue (non-fatal for tokenization).
            self.unicode_to_byte.put(self.byte_to_unicode[b], byte) catch |err| {
                std.log.warn("unicode_to_byte put failed: {}", .{err});
            };
        }
    }

    fn bytesToUnicode(self: *const BpeTokenizer, text: []const u8) ![]u8 {
        var result = std.ArrayList(u8).empty;
        for (text) |byte| {
            try result.appendSlice(self.allocator, self.byte_to_unicode[byte]);
        }
        return result.toOwnedSlice(self.allocator);
    }

    fn unicodeToBytes(self: *const BpeTokenizer, text: []const u8) ![]u8 {
        var result = std.ArrayList(u8).empty;
        var i: usize = 0;
        while (i < text.len) {
            var char_len: usize = 1;
            if ((text[i] & 0x80) == 0) {
                char_len = 1;
            } else if ((text[i] & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((text[i] & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((text[i] & 0xF8) == 0xF0) {
                char_len = 4;
            }
            if (i + char_len > text.len) char_len = 1;
            const uc = text[i .. i + char_len];
            if (self.unicode_to_byte.get(uc)) |byte| {
                try result.append(self.allocator, byte);
            } else if (uc.len == 1 and uc[0] < 128) {
                try result.append(self.allocator, uc[0]);
            } else {
                try result.append(self.allocator, '?');
            }
            i += char_len;
        }
        return result.toOwnedSlice(self.allocator);
    }

    fn splitUtfChars(self: *const BpeTokenizer, text: []const u8) !std.ArrayList([]const u8) {
        var chars: std.ArrayList([]const u8) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            var cl: usize = 1;
            if ((text[i] & 0x80) == 0) cl = 1 else if ((text[i] & 0xE0) == 0xC0) cl = 2 else if ((text[i] & 0xF0) == 0xE0) cl = 3 else if ((text[i] & 0xF8) == 0xF0) cl = 4;
            if (i + cl > text.len) cl = 1;
            try chars.append(self.allocator, text[i .. i + cl]);
            i += cl;
        }
        return chars;
    }

    fn findBestMerge(self: *const BpeTokenizer, tokens: []const []const u8) struct { pos: i32, priority: u32 } {
        var best_pos: i32 = -1;
        var best_pri: u32 = std.math.maxInt(u32);
        if (tokens.len < 2) return .{ .pos = -1, .priority = best_pri };
        var key_buf: [merge_key_buf_size]u8 = undefined;
        for (0..tokens.len - 1) |i| {
            const kl = tokens[i].len + 1 + tokens[i + 1].len;
            if (kl > key_buf.len) continue;
            @memcpy(key_buf[0..tokens[i].len], tokens[i]);
            key_buf[tokens[i].len] = 0;
            @memcpy(key_buf[tokens[i].len + 1 ..][0..tokens[i + 1].len], tokens[i + 1]);
            if (self.merge_map.get(key_buf[0..kl])) |pri| {
                if (pri < best_pri) {
                    best_pri = pri;
                    best_pos = @intCast(i);
                }
            }
        }
        return .{ .pos = best_pos, .priority = best_pri };
    }

    fn applyBpe(self: *const BpeTokenizer, chars: []const []const u8) !std.ArrayList([]const u8) {
        var current: std.ArrayList([]const u8) = .empty;
        var allocated: std.ArrayList([]const u8) = .empty;
        try current.appendSlice(self.allocator, chars);
        while (current.items.len > 1) {
            const m = self.findBestMerge(current.items);
            if (m.pos < 0) break;
            const pos: usize = @intCast(m.pos);
            const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ current.items[pos], current.items[pos + 1] });
            try allocated.append(self.allocator, merged);
            current.items[pos] = merged;
            _ = current.orderedRemove(pos + 1);
        }
        // Free intermediate merged strings that aren't in the final result
        for (allocated.items) |s| {
            var still_used = false;
            for (current.items) |t| {
                if (s.ptr == t.ptr) {
                    still_used = true;
                    break;
                }
            }
            if (!still_used) self.allocator.free(s);
        }
        allocated.deinit(self.allocator);
        return current;
    }

    /// Encode text to token IDs using byte-level BPE with merge rules.
    pub fn encode(self: *const BpeTokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);
        var result: std.ArrayList(u32) = .empty;

        // Split by special tokens first
        var segments: std.ArrayList([]const u8) = .empty;
        defer segments.deinit(self.allocator);
        var is_special: std.ArrayList(bool) = .empty;
        defer is_special.deinit(self.allocator);

        var start: usize = 0;
        while (start < text.len) {
            var best_pos: usize = text.len;
            var best_len: usize = 0;
            var best_tok: ?[]const u8 = null;
            var it = self.special_tokens.iterator();
            while (it.next()) |entry| {
                const st = entry.key_ptr.*;
                if (std.mem.indexOf(u8, text[start..], st)) |p| {
                    if (start + p < best_pos) {
                        best_pos = start + p;
                        best_len = st.len;
                        best_tok = st;
                    }
                }
            }
            if (best_tok != null and best_pos < text.len) {
                if (best_pos > start) {
                    try segments.append(self.allocator, text[start..best_pos]);
                    try is_special.append(self.allocator, false);
                }
                try segments.append(self.allocator, text[start + (best_pos - start) ..][0..best_len]);
                try is_special.append(self.allocator, true);
                start = best_pos + best_len;
            } else {
                if (start < text.len) {
                    try segments.append(self.allocator, text[start..]);
                    try is_special.append(self.allocator, false);
                }
                break;
            }
        }

        for (segments.items, 0..) |seg, si| {
            if (is_special.items[si]) {
                if (self.special_tokens.get(seg)) |id| {
                    try result.append(self.allocator, id);
                }
            } else {
                // Byte-level BPE
                const unicode_text = try self.bytesToUnicode(seg);
                defer self.allocator.free(unicode_text);
                var chars = try self.splitUtfChars(unicode_text);
                defer chars.deinit(self.allocator);
                var bpe_tokens = try self.applyBpe(chars.items);
                defer {
                    // Free merged strings (those not pointing into unicode_text)
                    for (bpe_tokens.items) |s| {
                        if (@intFromPtr(s.ptr) < @intFromPtr(unicode_text.ptr) or
                            @intFromPtr(s.ptr) >= @intFromPtr(unicode_text.ptr) + unicode_text.len)
                        {
                            self.allocator.free(s);
                        }
                    }
                    bpe_tokens.deinit(self.allocator);
                }
                for (bpe_tokens.items) |tok| {
                    if (self.token_to_id.get(tok)) |id| {
                        try result.append(self.allocator, id);
                    } else {
                        try result.append(self.allocator, 0); // unk
                    }
                }
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Decode token IDs back to text using byte-level BPE mappings.
    pub fn decode(self: *const BpeTokenizer, tokens: []const u32) ![]u8 {
        var unicode_result = std.ArrayList(u8).empty;
        for (tokens) |id| {
            if (id >= self.id_to_token.items.len) continue;
            const tok = self.id_to_token.items[id];
            try unicode_result.appendSlice(self.allocator, tok);
        }
        const unicode_str = try unicode_result.toOwnedSlice(self.allocator);
        defer self.allocator.free(unicode_str);
        return self.unicodeToBytes(unicode_str);
    }

    /// Load vocabulary and merge rules from GGUF-embedded tokenizer data.
    pub fn loadFromGGUF(self: *BpeTokenizer, vocab: []const []const u8, merges: []const []const u8, eos_id: u32) !void {
        self.eos_token_id = eos_id;
        var special_count: usize = 0;
        for (vocab, 0..) |tok, i| {
            const owned_tok = try self.own(tok);
            try self.token_to_id.put(owned_tok, @intCast(i));
            try self.id_to_token.append(self.allocator, owned_tok);
            if (tok.len > 0 and tok[0] == '<' and tok[tok.len - 1] == '>') {
                special_count += 1;
                if (std.mem.indexOf(u8, tok, "im_start") != null or std.mem.indexOf(u8, tok, "im_end") != null) {
                    std.log.warn("[bpe] Found ChatML special token: '{s}' = {}", .{ tok, i });
                }
                try self.special_tokens.put(owned_tok, @intCast(i));
            }
        }
        self.vocab_size = @intCast(vocab.len);
        std.log.warn("[bpe] Loaded {} special tokens from GGUF vocab. Token 10='{s}', Token 11='{s}'", .{ special_count, if (10 < self.id_to_token.items.len) self.id_to_token.items[10] else "", if (11 < self.id_to_token.items.len) self.id_to_token.items[11] else "" });
        var priority: u32 = 0;
        for (merges) |merge_line| {
            if (merge_line.len == 0 or merge_line[0] == '#') continue;
            const sp = std.mem.indexOf(u8, merge_line, " ") orelse continue;
            const first = merge_line[0..sp];
            const second = merge_line[sp + 1 ..];
            var key_buf: [merge_key_buf_size]u8 = undefined;
            const key_len = first.len + 1 + second.len;
            if (key_len > key_buf.len) continue;
            @memcpy(key_buf[0..first.len], first);
            key_buf[first.len] = 0;
            @memcpy(key_buf[first.len + 1 ..][0..second.len], second);
            const key = try self.own(key_buf[0..key_len]);
            if (!self.merge_map.contains(key)) {
                try self.merge_map.put(key, priority);
            }
            priority += 1;
        }
        try self.initByteMappings();
    }

    /// Load vocabulary from GGUF for SPM-style tokenizer (no merges).
    /// Uses greedy longest-match encoding instead of BPE merges.
    pub fn loadFromGGUFSpm(self: *BpeTokenizer, vocab: []const []const u8, eos_id: u32) !void {
        self.eos_token_id = eos_id;
        var special_count: usize = 0;
        for (vocab, 0..) |tok, i| {
            const owned_tok = try self.own(tok);
            try self.token_to_id.put(owned_tok, @intCast(i));
            try self.id_to_token.append(self.allocator, owned_tok);
            if (tok.len > 0 and tok[0] == '<' and tok[tok.len - 1] == '>') {
                special_count += 1;
                try self.special_tokens.put(owned_tok, @intCast(i));
            }
        }
        self.vocab_size = @intCast(vocab.len);
        std.log.warn("[bpe] Loaded {} special tokens from GGUF vocab (SPM mode)", .{special_count});
        // No merges for SPM — encode uses greedy longest match
        // No byte mappings needed — SPM tokens are raw UTF-8
    }

    /// Greedy longest-match encoding for SPM tokenizers (no BPE merges).
    /// Spaces are consumed and represented as ▁ (U+2581) prefix on the following
    /// word, matching the SentencePiece convention the model was trained with.
    /// A dummy ▁ prefix is prepended to the input (add_dummy_prefix=true).
    pub fn encodeSpm(self: *const BpeTokenizer, text: []const u8) ![]u32 {
        return self.encodeSpmInner(text, true);
    }

    /// Like encodeSpm but without add_dummy_prefix — used by tokenizers
    /// (like Gemma) where ▁ prefix only appears for actual spaces.
    pub fn encodeSpmNoDummy(self: *const BpeTokenizer, text: []const u8) ![]u32 {
        return self.encodeSpmInner(text, false);
    }

    fn encodeSpmInner(self: *const BpeTokenizer, text: []const u8, add_dummy_prefix: bool) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);
        var result: std.ArrayList(u32) = .empty;

        // When add_dummy_prefix is true (traditional SPM), the first word and
        // every word after whitespace/special tokens/newlines gets a ▁ prefix.
        // When false (Gemma), ▁ only appears for actual space characters.
        var word_start = add_dummy_prefix;

        var start: usize = 0;
        while (start < text.len) {
            // Try to match special tokens first (longest match wins)
            var best_sp_len: usize = 0;
            var best_sp_id: u32 = 0;
            var sp_it = self.special_tokens.iterator();
            while (sp_it.next()) |entry| {
                const st = entry.key_ptr.*;
                if (st.len > best_sp_len and start + st.len <= text.len and
                    std.mem.eql(u8, text[start..][0..st.len], st))
                {
                    best_sp_len = st.len;
                    best_sp_id = entry.value_ptr.*;
                }
            }
            if (best_sp_len > 0) {
                try result.append(self.allocator, best_sp_id);
                start += best_sp_len;
                // After a special token: traditional SPM adds ▁, Gemma doesn't
                word_start = add_dummy_prefix;
                continue;
            }

            // Consume spaces — they become ▁ prefix on the next token
            if (text[start] == ' ') {
                word_start = true;
                start += 1;
                continue;
            }

            // Limit greedy match to not cross a special token boundary.
            // Scan for the nearest '<' that starts a known special token.
            var max_reach: usize = text.len - start;
            {
                var scan: usize = start + 1;
                while (scan < start + max_reach) : (scan += 1) {
                    if (text[scan] == '<') {
                        // Check if any special token starts here
                        var sp2 = self.special_tokens.iterator();
                        while (sp2.next()) |entry| {
                            const st = entry.key_ptr.*;
                            if (scan + st.len <= text.len and
                                std.mem.eql(u8, text[scan..][0..st.len], st))
                            {
                                max_reach = scan - start;
                                break;
                            }
                        }
                        if (max_reach == scan - start) break;
                    }
                }
            }

            var best_len: usize = 0;
            var best_id: u32 = 0;

            // SPM uses ▁ (U+2581, 3 bytes: 0xE2 0x96 0x81) as word separator.
            // At word boundaries, try ▁-prefixed tokens first.
            if (word_start) {
                var buf: [spm_prefix.len + max_spm_token_len]u8 = undefined;
                const max_try = @min(max_reach, max_spm_token_len);
                var tl: usize = max_try;
                while (tl > 0) : (tl -= 1) {
                    if (spm_prefix.len + tl > buf.len) continue;
                    @memcpy(buf[0..spm_prefix.len], spm_prefix);
                    @memcpy(buf[spm_prefix.len..][0..tl], text[start..][0..tl]);
                    if (self.token_to_id.get(buf[0 .. spm_prefix.len + tl])) |id| {
                        best_len = tl;
                        best_id = id;
                        break;
                    }
                }
                // If no ▁-prefixed token matched, emit standalone ▁ as a token
                if (best_len == 0) {
                    if (self.token_to_id.get(spm_prefix)) |sp_id| {
                        try result.append(self.allocator, sp_id);
                    }
                    word_start = false;
                    // Don't advance start — re-process current char without ▁
                    continue;
                }
            }

            // Fallback: greedy longest match without ▁ prefix
            if (best_len == 0) {
                const max_tok_len = @min(max_reach, max_spm_token_len);
                var try_len: usize = max_tok_len;
                while (try_len > 0) : (try_len -= 1) {
                    if (self.token_to_id.get(text[start..][0..try_len])) |id| {
                        best_len = try_len;
                        best_id = id;
                        break;
                    }
                }
            }

            if (best_len > 0) {
                try result.append(self.allocator, best_id);
                // After a newline: traditional SPM adds ▁, Gemma doesn't
                word_start = add_dummy_prefix and (text[start + best_len - 1] == '\n');
                start += best_len;
            } else {
                // Fall back to single byte as unknown
                // Try to find the byte as a hex token like <0xNN>
                var hex_buf: [6]u8 = undefined;
                const hex = std.fmt.bufPrint(&hex_buf, "<0x{X:0>2}>", .{text[start]}) catch {
                    start += 1;
                    continue;
                };
                if (self.token_to_id.get(hex)) |id| {
                    try result.append(self.allocator, id);
                } else {
                    try result.append(self.allocator, fallback_unknown_token_id); // unknown token
                }
                word_start = add_dummy_prefix and (text[start] == '\n');
                start += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Decode for SPM tokenizer — tokens are raw UTF-8, ▁ maps to space
    pub fn decodeSpm(self: *const BpeTokenizer, tokens: []const u32) ![]u8 {
        var result = std.ArrayList(u8).empty;
        for (tokens) |id| {
            if (id >= self.id_to_token.items.len) continue;
            const tok = self.id_to_token.items[id];
            // Replace ▁ (U+2581) with space
            var i: usize = 0;
            while (i < tok.len) {
                if (i + spm_prefix.len <= tok.len and std.mem.eql(u8, tok[i..][0..spm_prefix.len], spm_prefix)) {
                    try result.append(self.allocator, ' ');
                    i += spm_prefix.len;
                } else {
                    try result.append(self.allocator, tok[i]);
                    i += 1;
                }
            }
        }
        return result.toOwnedSlice(self.allocator);
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "BpeTokenizer SPM encode/decode roundtrip" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    // Build a minimal SPM vocabulary with ▁-prefixed word tokens
    const vocab = [_][]const u8{ "\xe2\x96\x81hello", "\xe2\x96\x81world", "h", "e", "l", "o", "w", "r", "d" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    // SPM encode: "hello world" → [▁hello, ▁world]
    const ids = try tok.encodeSpm("hello world");
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 2), ids.len);

    // First token should be "▁hello" (id 0), second "▁world" (id 1)
    try std.testing.expectEqual(@as(u32, 0), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
}

test "BpeTokenizer SPM decode produces text" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "\xe2\x96\x81hello", "\xe2\x96\x81world" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    // Decode token id 0 → " hello" (▁ maps to space)
    const decoded = try tok.decodeSpm(&.{0});
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(" hello", decoded);
}

test "BpeTokenizer SPM decode multiple tokens" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "\xe2\x96\x81hello", "\xe2\x96\x81", "\xe2\x96\x81world" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    // Decode [▁hello, ▁world] → " hello world"
    const decoded = try tok.decodeSpm(&.{ 0, 2 });
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(" hello world", decoded);
}

test "BpeTokenizer empty encode" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{"a"};
    var vocab_slice: [vocab.len][]const u8 = undefined;
    vocab_slice[0] = vocab[0];
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    const ids = try tok.encodeSpm("");
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 0), ids.len);
}

test "BpeTokenizer vocabSize" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "a", "b", "c", "d", "e" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 4);

    try std.testing.expectEqual(@as(u32, 5), tok.vocab_size);
    try std.testing.expectEqual(@as(u32, 4), tok.eos_token_id);
}

test "BpeTokenizer decode out of range token" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{"a"};
    var vocab_slice: [vocab.len][]const u8 = undefined;
    vocab_slice[0] = vocab[0];
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    // Token id 999 is out of range — should be skipped
    const decoded = try tok.decodeSpm(&.{999});
    defer allocator.free(decoded);
    try std.testing.expectEqual(@as(usize, 0), decoded.len);
}

test "BpeTokenizer interface via VTable" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "hi", "!" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 0);

    // Use the VTable interface
    var iface = tok.tokenizer();
    try std.testing.expectEqual(@as(u32, 2), iface.vocabSize());
}
