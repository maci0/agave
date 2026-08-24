//! Byte-level BPE tokenizer supporting BPE, SPM, and SPM-no-dummy modes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tok_iface = @import("tokenizer.zig");
const TokenizerIface = tok_iface.Tokenizer;
const TokenizerKind = @import("tokenizer.zig").TokenizerKind;

const merge_key_buf_size: usize = 512;
const max_spm_token_len: usize = 64;
const fallback_unknown_token_id: u32 = 3;
/// SPM word-initial marker: U+2581 LOWER ONE EIGHTH BLOCK (▁), UTF-8 encoded.
const spm_prefix = "\xe2\x96\x81";
/// Default Qwen EOS token ID, used when tokenizer.json doesn't specify one.
const qwen_default_eos_id: u32 = 151645;
/// Default Qwen BOS token ID, used when tokenizer.json doesn't specify one.
const qwen_default_bos_id: u32 = 151643;
/// Max bytes per cached BPE segment (longer segments skip the cache).
const max_word_cache_seg_bytes: usize = 4096;
/// Cap on word_cache entries to bound memory in long-lived --serve processes.
const max_word_cache_entries: usize = 8192;

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
    // Word-level BPE cache: maps pretoken bytes → token ID slice.
    // Avoids re-running bytesToUnicode + applyBpe for recurring words.
    // Keys and values are owned by this allocator.
    // Atomic spinlock guards concurrent encode from server connection threads
    // (Zig 0.16 Mutex lives on Io; tokenizer has no Io context).
    word_cache: std.StringHashMapUnmanaged([]u32) = .{},
    word_cache_lock: std.atomic.Value(u8) = .init(0),

    /// Return the generic Tokenizer interface backed by this BPE tokenizer.
    pub fn tokenizer(self: *BpeTokenizer) TokenizerIface {
        return .{ .ptr = self, .vtable = &tok_vtable };
    }
    const tok_vtable = TokenizerIface.VTable{
        .encode = @ptrCast(&tokEncode),
        .decode = @ptrCast(&tokDecode),
        .get_vocab_size = @ptrCast(&tokGetVocabSize),
        .get_vocab_texts = @ptrCast(&tokGetVocabTexts),
        .decode_one = @ptrCast(&tokDecodeOne),
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
    fn tokDecodeOne(self: *BpeTokenizer, token_id: u32, buf: []u8) ?[]const u8 {
        return self.decodeOne(token_id, buf);
    }
    fn tokGetVocabSize(self: *BpeTokenizer) u32 {
        return self.vocab_size;
    }
    fn tokGetVocabTexts(self: *BpeTokenizer) []const []const u8 {
        return self.id_to_token.items;
    }

    /// True when `id` is one of the tokenizer's special tokens (chat-template
    /// markers such as <|im_start|> or <start_of_turn>). Consults the loaded
    /// special-token table rather than assuming specials occupy the top of the
    /// ID range, which does not hold for every vocab (e.g. Gemma's
    /// <start_of_turn> sits at 105). Init-path cost only: linear scan over the
    /// special-token set.
    pub fn isSpecialId(self: *const BpeTokenizer, id: u32) bool {
        var it = self.special_tokens.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* == id) return true;
        }
        return false;
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

    fn lockWordCache(self: *BpeTokenizer) void {
        while (self.word_cache_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlockWordCache(self: *BpeTokenizer) void {
        self.word_cache_lock.store(0, .release);
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
        // Free word cache: keys and values are both owned.
        self.lockWordCache();
        defer self.unlockWordCache();
        var it = self.word_cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.word_cache.deinit(self.allocator);
    }

    /// Duplicate `s` into owned memory and track it for cleanup in `deinit`.
    fn own(self: *BpeTokenizer, s: []const u8) ![]const u8 {
        const d = try self.allocator.dupe(u8, s);
        errdefer self.allocator.free(d);
        try self.owned_strings.append(self.allocator, d);
        return d;
    }

    /// Build the GPT-2 byte↔unicode lookup tables (`byte_to_unicode` / `unicode_to_byte`).
    /// Idempotent — subsequent calls are no-ops. On error, partially-allocated
    /// entries are freed and `byte_mappings_init` is reset so the caller can retry.
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
            try self.unicode_to_byte.put(self.byte_to_unicode[b], byte);
        }
    }

    /// Map raw bytes to their GPT-2 unicode representations.
    /// Returns a caller-owned UTF-8 slice that must be freed with `self.allocator`.
    fn bytesToUnicode(self: *const BpeTokenizer, text: []const u8) ![]u8 {
        var result = std.ArrayList(u8).empty;
        for (text) |byte| {
            try result.appendSlice(self.allocator, self.byte_to_unicode[byte]);
        }
        return result.toOwnedSlice(self.allocator);
    }

    /// Reverse the GPT-2 unicode mapping: convert unicode-encoded token text back
    /// to raw bytes. Unmapped multi-byte sequences are replaced with `'?'`.
    /// Returns a caller-owned slice that must be freed with `self.allocator`.
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

    /// Split a UTF-8 string into individual codepoint slices (each pointing into `text`).
    /// Invalid lead bytes are treated as single-byte sequences.
    fn splitUtfChars(self: *const BpeTokenizer, text: []const u8) !std.ArrayList([]const u8) {
        var chars: std.ArrayList([]const u8) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            const raw_cl: usize = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            const cl: usize = if (i + raw_cl > text.len) 1 else raw_cl;
            try chars.append(self.allocator, text[i .. i + cl]);
            i += cl;
        }
        return chars;
    }

    /// Scan adjacent token pairs and return the position and priority of the
    /// highest-priority (lowest numeric priority) merge. Returns `pos = -1`
    /// when no applicable merge exists. Uses a stack buffer for the merge key.
    fn findBestMerge(self: *const BpeTokenizer, tokens: []const []const u8) struct { pos: i32, priority: u32 } {
        var best_pos: i32 = -1;
        var best_pri: u32 = std.math.maxInt(u32);
        if (tokens.len < 2) return .{ .pos = -1, .priority = best_pri };
        var key_buf: [merge_key_buf_size]u8 = undefined;
        for (0..tokens.len - 1) |i| {
            const kl = std.math.add(usize, std.math.add(usize, tokens[i].len, 1) catch continue, tokens[i + 1].len) catch continue;
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

    /// Iteratively apply BPE merges to a list of unicode character slices until
    /// no more merges are possible. Returns the merged token list; intermediate
    /// allocations that are not part of the final result are freed before return.
    fn applyBpe(self: *const BpeTokenizer, chars: []const []const u8) !std.ArrayList([]const u8) {
        var current: std.ArrayList([]const u8) = .empty;
        errdefer current.deinit(self.allocator);
        var allocated: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (allocated.items) |s| self.allocator.free(s);
            allocated.deinit(self.allocator);
        }
        try current.appendSlice(self.allocator, chars);
        while (current.items.len > 1) {
            const m = self.findBestMerge(current.items);
            if (m.pos < 0) break;
            const pos: usize = @intCast(m.pos);
            const a = current.items[pos];
            const b = current.items[pos + 1];
            const merged_len = std.math.add(usize, a.len, b.len) catch break;
            const merged = try self.allocator.alloc(u8, merged_len);
            @memcpy(merged[0..a.len], a);
            @memcpy(merged[a.len..], b);
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
    /// Word-level cache: segments seen before skip bytesToUnicode + applyBpe entirely.
    pub fn encode(self: *BpeTokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);
        var result: std.ArrayList(u32) = .empty;
        errdefer result.deinit(self.allocator);

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
            // All special tokens start with '<' (enforced at load time).
            // Scan for '<' positions and check special tokens only there,
            // avoiding O(n_special × text_len) substring searches.
            {
                var scan = start;
                while (scan < text.len) {
                    const rel = std.mem.indexOfScalar(u8, text[scan..], '<') orelse break;
                    scan += rel;
                    var it = self.special_tokens.iterator();
                    while (it.next()) |entry| {
                        const st = entry.key_ptr.*;
                        if (st.len > best_len and scan + st.len <= text.len and
                            std.mem.eql(u8, text[scan..][0..st.len], st))
                        {
                            best_pos = scan;
                            best_len = st.len;
                            best_tok = st;
                        }
                    }
                    if (best_tok != null) break;
                    scan += 1;
                }
            }
            if (best_tok != null and best_pos < text.len) {
                if (best_pos > start) {
                    try segments.append(self.allocator, text[start..best_pos]);
                    try is_special.append(self.allocator, false);
                }
                try segments.append(self.allocator, text[best_pos..][0..best_len]);
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
                // Byte-level BPE — check word cache first.
                self.lockWordCache();
                const cached_opt = self.word_cache.get(seg);
                if (cached_opt) |cached_ids| {
                    // Copy data under lock so concurrent deinit cannot free the
                    // slice between unlock and read. Stack buffer handles the
                    // common case; oversized entries (>1024 tokens, extremely rare)
                    // are duplicated under lock to avoid use-after-free.
                    var tmp_ids: [1024]u32 = undefined;
                    const n_ids = cached_ids.len;
                    if (n_ids <= tmp_ids.len) {
                        @memcpy(tmp_ids[0..n_ids], cached_ids);
                        self.unlockWordCache();
                        try result.appendSlice(self.allocator, tmp_ids[0..n_ids]);
                    } else {
                        // Allocate and copy under lock — the slice must not be
                        // read after unlock (concurrent deinit could free it).
                        const owned = self.allocator.dupe(u32, cached_ids) catch {
                            self.unlockWordCache();
                            return error.OutOfMemory;
                        };
                        self.unlockWordCache();
                        defer self.allocator.free(owned);
                        try result.appendSlice(self.allocator, owned);
                    }
                } else {
                    self.unlockWordCache();
                    const unicode_text = try self.bytesToUnicode(seg);
                    defer self.allocator.free(unicode_text);
                    var chars = try self.splitUtfChars(unicode_text);
                    defer chars.deinit(self.allocator);
                    var bpe_tokens = try self.applyBpe(chars.items);
                    defer {
                        for (bpe_tokens.items) |s| {
                            if (@intFromPtr(s.ptr) < @intFromPtr(unicode_text.ptr) or
                                @intFromPtr(s.ptr) >= @intFromPtr(unicode_text.ptr) + unicode_text.len)
                            {
                                self.allocator.free(s);
                            }
                        }
                        bpe_tokens.deinit(self.allocator);
                    }
                    // Collect IDs for this segment.
                    const seg_start = result.items.len;
                    for (bpe_tokens.items) |tok| {
                        if (self.token_to_id.get(tok)) |id| {
                            try result.append(self.allocator, id);
                        } else {
                            try result.append(self.allocator, 0); // unk
                        }
                    }
                    // Store in cache: key = owned copy of seg, value = owned copy of IDs.
                    const seg_ids = result.items[seg_start..];
                    if (seg_ids.len > 0 and seg.len <= max_word_cache_seg_bytes) {
                        self.lockWordCache();
                        defer self.unlockWordCache();
                        if (self.word_cache.count() < max_word_cache_entries and !self.word_cache.contains(seg)) {
                            const owned_key = try self.allocator.dupe(u8, seg);
                            errdefer self.allocator.free(owned_key);
                            const owned_val = try self.allocator.dupe(u32, seg_ids);
                            errdefer self.allocator.free(owned_val);
                            try self.word_cache.put(self.allocator, owned_key, owned_val);
                        }
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

    /// Decode a single token ID into `buf` without allocating.
    /// Byte-for-byte equivalent to decoding a one-element slice via `decode`
    /// (BPE mode: reverse byte↔unicode mapping; SPM modes: `<0xNN>` hex-byte
    /// tokens and ▁→space). Returns a slice of `buf`, or null when the token
    /// is out of range or its text does not fit — callers then fall back to
    /// the allocating `decode`.
    pub fn decodeOne(self: *const BpeTokenizer, token_id: u32, buf: []u8) ?[]const u8 {
        if (token_id >= self.id_to_token.items.len) return null;
        const text = self.id_to_token.items[token_id];
        return switch (self.tok_kind) {
            .bpe => self.decodeOneBpe(text, buf),
            .spm, .spm_no_dummy => decodeOneSpm(text, buf),
        };
    }

    /// Allocation-free single-token reverse of the GPT-2 unicode mapping.
    /// Mirrors `unicodeToBytes` char-walk and fallback behavior exactly.
    fn decodeOneBpe(self: *const BpeTokenizer, text: []const u8, out: []u8) ?[]const u8 {
        var n: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            var char_len: usize = 1;
            if ((text[i] & 0x80) != 0) {
                if ((text[i] & 0xE0) == 0xC0) {
                    char_len = 2;
                } else if ((text[i] & 0xF0) == 0xE0) {
                    char_len = 3;
                } else if ((text[i] & 0xF8) == 0xF0) {
                    char_len = 4;
                }
            }
            if (i + char_len > text.len) char_len = 1;
            const uc = text[i .. i + char_len];
            const byte: u8 = if (self.unicode_to_byte.get(uc)) |b|
                b
            else if (uc.len == 1 and uc[0] < 128)
                uc[0]
            else
                '?';
            if (n >= out.len) return null;
            out[n] = byte;
            n += 1;
            i += char_len;
        }
        return out[0..n];
    }

    /// Allocation-free single-token SPM decode: mirrors the per-token branch
    /// of `decodeSpm` (`<0xNN>` hex-byte tokens, ▁ → space, raw copy).
    fn decodeOneSpm(text: []const u8, out: []u8) ?[]const u8 {
        // Handle <0xNN> hex-byte tokens — emit raw byte.
        if (text.len == 6 and text[0] == '<' and text[1] == '0' and text[2] == 'x' and text[5] == '>') {
            if (std.fmt.parseUnsigned(u8, text[3..5], 16)) |byte| {
                if (out.len < 1) return null;
                out[0] = byte;
                return out[0..1];
            } else |_| {}
        }
        // Replace ▁ (U+2581) with space; copy the rest verbatim.
        var n: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (i + spm_prefix.len <= text.len and std.mem.eql(u8, text[i..][0..spm_prefix.len], spm_prefix)) {
                if (n >= out.len) return null;
                out[n] = ' ';
                n += 1;
                i += spm_prefix.len;
            } else {
                const remaining = text[i..];
                const next = std.mem.indexOf(u8, remaining, spm_prefix) orelse remaining.len;
                if (n + next > out.len) return null;
                @memcpy(out[n..][0..next], remaining[0..next]);
                n += next;
                i += next;
            }
        }
        return out[0..n];
    }

    /// Load vocabulary and merge rules from GGUF-embedded tokenizer data.
    pub fn loadFromGGUF(self: *BpeTokenizer, vocab: []const []const u8, merges: []const []const u8, eos_id: u32) !void {
        self.eos_token_id = eos_id;
        // Reject vocab/merge counts that overflow u32 (token IDs are u32).
        const vocab_len: u32 = std.math.cast(u32, vocab.len) orelse return error.VocabTooLarge;
        const merges_len: u32 = std.math.cast(u32, merges.len) orelse return error.VocabTooLarge;
        // Pre-allocate maps to avoid repeated rehashing during bulk insert.
        try self.token_to_id.ensureTotalCapacity(vocab_len);
        try self.id_to_token.ensureTotalCapacity(self.allocator, vocab.len);
        try self.merge_map.ensureTotalCapacity(merges_len);
        var special_count: usize = 0;
        for (vocab, 0..) |tok, i| {
            const id: u32 = std.math.cast(u32, i) orelse break;
            const owned_tok = try self.own(tok);
            try self.token_to_id.put(owned_tok, id);
            try self.id_to_token.append(self.allocator, owned_tok);
            if (tok.len > 0 and ((tok[0] == '<' and tok[tok.len - 1] == '>') or
                (tok[0] == '[' and tok[tok.len - 1] == ']')))
            {
                special_count += 1;
                if (std.mem.indexOf(u8, tok, "im_start") != null or std.mem.indexOf(u8, tok, "im_end") != null) {
                    std.log.info("[bpe] Found ChatML special token: '{s}' = {}", .{ tok, i });
                }
                try self.special_tokens.put(owned_tok, id);
            }
        }
        self.vocab_size = vocab_len;
        std.log.info("[bpe] Loaded {} special tokens from GGUF vocab. Token 10='{s}', Token 11='{s}'", .{ special_count, if (10 < self.id_to_token.items.len) self.id_to_token.items[10] else "", if (11 < self.id_to_token.items.len) self.id_to_token.items[11] else "" });
        var priority: u32 = 0;
        for (merges) |merge_line| {
            if (merge_line.len == 0 or merge_line[0] == '#') continue;
            const sp = std.mem.indexOf(u8, merge_line, " ") orelse continue;
            const first = merge_line[0..sp];
            const second = merge_line[sp + 1 ..];
            var key_buf: [merge_key_buf_size]u8 = undefined;
            const kl1 = std.math.add(usize, first.len, 1) catch continue;
            const key_len = std.math.add(usize, kl1, second.len) catch continue;
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
        // Reject vocab counts that overflow u32 (token IDs are u32).
        const vocab_len: u32 = std.math.cast(u32, vocab.len) orelse return error.VocabTooLarge;
        // Pre-allocate maps to avoid repeated rehashing during bulk insert.
        try self.token_to_id.ensureTotalCapacity(vocab_len);
        try self.id_to_token.ensureTotalCapacity(self.allocator, vocab.len);
        var special_count: usize = 0;
        for (vocab, 0..) |tok, i| {
            const id: u32 = std.math.cast(u32, i) orelse break;
            const owned_tok = try self.own(tok);
            try self.token_to_id.put(owned_tok, id);
            try self.id_to_token.append(self.allocator, owned_tok);
            if (tok.len > 0 and ((tok[0] == '<' and tok[tok.len - 1] == '>') or
                (tok[0] == '[' and tok[tok.len - 1] == ']')))
            {
                special_count += 1;
                try self.special_tokens.put(owned_tok, id);
            }
        }
        self.vocab_size = vocab_len;
        std.log.info("[bpe] Loaded {} special tokens from GGUF vocab (SPM mode)", .{special_count});
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
        errdefer result.deinit(self.allocator);

        // When add_dummy_prefix is true (traditional SPM), the first word and
        // every word after whitespace/special tokens/newlines gets a ▁ prefix.
        // When false (Gemma), ▁ only appears for actual space characters.
        var word_start = add_dummy_prefix;

        var start: usize = 0;
        while (start < text.len) {
            // Try to match special tokens first (longest match wins).
            // All special tokens start with '<' — skip scan otherwise.
            var best_sp_len: usize = 0;
            var best_sp_id: u32 = 0;
            if (text[start] == '<') {
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
            // Jump to each '<' via indexOfScalar (SIMD-accelerated) instead of byte-by-byte.
            var max_reach: usize = text.len - start;
            {
                var scan: usize = start + 1;
                const scan_end = start + max_reach;
                while (scan < scan_end) {
                    const rel = std.mem.indexOfScalar(u8, text[scan..scan_end], '<') orelse break;
                    scan += rel;
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
                    scan += 1;
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
            // Handle <0xNN> hex-byte tokens — emit raw byte
            if (tok.len == 6 and tok[0] == '<' and tok[1] == '0' and tok[2] == 'x' and tok[5] == '>') {
                const byte = std.fmt.parseUnsigned(u8, tok[3..5], 16) catch {
                    try result.appendSlice(self.allocator, tok);
                    continue;
                };
                try result.append(self.allocator, byte);
                continue;
            }
            // Replace ▁ (U+2581) with space, bulk-copy non-prefix segments
            var i: usize = 0;
            while (i < tok.len) {
                if (i + spm_prefix.len <= tok.len and std.mem.eql(u8, tok[i..][0..spm_prefix.len], spm_prefix)) {
                    try result.append(self.allocator, ' ');
                    i += spm_prefix.len;
                } else {
                    const remaining = tok[i..];
                    const next = std.mem.indexOf(u8, remaining, spm_prefix) orelse remaining.len;
                    try result.appendSlice(self.allocator, remaining[0..next]);
                    i += next;
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

test "isSpecialId uses loaded special-token table, not id range" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    // Gemma-like layout: specials at the BOTTOM of the vocab, content above.
    const vocab = [_][]const u8{ "<pad>", "<bos>", "hello", "world" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    try tok.loadFromGGUFSpm(&vocab_slice, 1);

    // Specials detected by table membership regardless of id value.
    try std.testing.expect(tok.isSpecialId(0)); // <pad>
    try std.testing.expect(tok.isSpecialId(1)); // <bos>
    // Content ids stay content even when numerically adjacent.
    try std.testing.expect(!tok.isSpecialId(2));
    try std.testing.expect(!tok.isSpecialId(3));
    try std.testing.expect(!tok.isSpecialId(9999));
}

test "BPE encode with merge rules" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    // Vocab in unicode-mapped form (ASCII printable chars map 1:1).
    // "a"=0, "b"=1, "c"=2, "ab"=3
    const vocab = [_][]const u8{ "a", "b", "c", "ab" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;

    // Merge rule: "a b" → priority 0 (merge "a"+"b" into "ab")
    const merges = [_][]const u8{"a b"};
    var merges_slice: [merges.len][]const u8 = undefined;
    for (&merges, 0..) |m, i| merges_slice[i] = m;

    try tok.loadFromGGUF(&vocab_slice, &merges_slice, 0);

    // Encode "ab": bytesToUnicode→"ab", splitUtfChars→["a","b"],
    // applyBpe merges at pos 0→["ab"], lookup→3
    const ids = try tok.encode("ab");
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 1), ids.len);
    try std.testing.expectEqual(@as(u32, 3), ids[0]);
}

test "BPE encode no merge match falls back to individual tokens" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "a", "b", "c", "ab" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    const merges = [_][]const u8{"a b"};
    var merges_slice: [merges.len][]const u8 = undefined;
    for (&merges, 0..) |m, i| merges_slice[i] = m;

    try tok.loadFromGGUF(&vocab_slice, &merges_slice, 0);

    // Encode "c": no merge rules for "c" → stays as single char → id 2
    const ids = try tok.encode("c");
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 1), ids.len);
    try std.testing.expectEqual(@as(u32, 2), ids[0]);
}

test "BPE decode reverses encode" {
    const allocator = std.testing.allocator;
    var tok = BpeTokenizer.init(allocator);
    defer tok.deinit();

    const vocab = [_][]const u8{ "a", "b", "c", "ab" };
    var vocab_slice: [vocab.len][]const u8 = undefined;
    for (&vocab, 0..) |v, i| vocab_slice[i] = v;
    const merges = [_][]const u8{"a b"};
    var merges_slice: [merges.len][]const u8 = undefined;
    for (&merges, 0..) |m, i| merges_slice[i] = m;

    try tok.loadFromGGUF(&vocab_slice, &merges_slice, 0);

    // Decode [3, 2] → "ab" + "c" → unicodeToBytes → "abc"
    const decoded = try tok.decode(&.{ 3, 2 });
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("abc", decoded);
}

test "decodeOne matches single-token decode in all modes" {
    const allocator = std.testing.allocator;
    var buf: [256]u8 = undefined;

    // BPE mode: mapped bytes, raw ASCII, and an unmapped codepoint ('?' fallback).
    {
        var tok = BpeTokenizer.init(allocator);
        defer tok.deinit();
        const vocab = [_][]const u8{ "a", "\xc3\xa9", "\xe2\x96\x81", "ab" };
        var vocab_slice: [vocab.len][]const u8 = undefined;
        for (&vocab, 0..) |v, i| vocab_slice[i] = v;
        try tok.loadFromGGUF(&vocab_slice, &.{}, 0);

        for (0..vocab.len) |id| {
            const batch = try tok.decode(&.{@intCast(id)});
            defer allocator.free(batch);
            const one = tok.decodeOne(@intCast(id), &buf) orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(batch, one);
        }
        // Out-of-range ID behaves like batch decode (empty).
        const oob_batch = try tok.decode(&.{99});
        defer allocator.free(oob_batch);
        try std.testing.expectEqual(@as(usize, 0), oob_batch.len);
        try std.testing.expect(tok.decodeOne(99, &buf) == null);
    }

    // SPM mode: hex-byte tokens, ▁ prefix, plain text, malformed hex.
    {
        var tok = BpeTokenizer.init(allocator);
        defer tok.deinit();
        const vocab = [_][]const u8{ "<0x41>", "\xe2\x96\x81hello", "world", "<0xZZ>" };
        var vocab_slice: [vocab.len][]const u8 = undefined;
        for (&vocab, 0..) |v, i| vocab_slice[i] = v;
        try tok.loadFromGGUFSpm(&vocab_slice, 0);
        tok.tok_kind = .spm; // set by callers after loading (see main.zig)

        for (0..vocab.len) |id| {
            const batch = try tok.decodeSpm(&.{@intCast(id)});
            defer allocator.free(batch);
            const one = tok.decodeOne(@intCast(id), &buf) orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(batch, one);
        }
    }

    // Undersized buffer must return null (caller falls back), never truncate.
    {
        var tok = BpeTokenizer.init(allocator);
        defer tok.deinit();
        const long_tok = "x" ** 100;
        const vocab = [_][]const u8{ long_tok, "y" };
        var vocab_slice: [vocab.len][]const u8 = undefined;
        for (&vocab, 0..) |v, i| vocab_slice[i] = v;
        try tok.loadFromGGUFSpm(&vocab_slice, 0);
        tok.tok_kind = .spm;
        var small: [4]u8 = undefined;
        try std.testing.expect(tok.decodeOne(0, &small) == null);
        const ok = tok.decodeOne(1, &small) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings("y", ok);
    }
}

test "fuzz: all bpe functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // --- init / deinit ---
            var tok = BpeTokenizer.init(allocator);
            defer tok.deinit();

            // --- loadFromGGUFSpm ---
            // Build a small vocab with SPM-prefixed and special tokens
            const vocab = [_][]const u8{
                "<unk>",
                "<s>",
                "</s>",
                "\xe2\x96\x81hello",
                "\xe2\x96\x81world",
                "\xe2\x96\x81",
                "h",
                "e",
                "l",
                "o",
                "<0x0A>",
            };
            var vocab_slice: [vocab.len][]const u8 = undefined;
            for (&vocab, 0..) |v, i| vocab_slice[i] = v;
            const eos_id = smith.valueWithHash(u32, 0) % @as(u32, vocab.len);
            tok.loadFromGGUFSpm(&vocab_slice, eos_id) catch return;

            // --- vocab_size ---
            const vs = tok.vocab_size;
            try std.testing.expect(vs == vocab.len);

            // --- tokenizer (vtable interface) ---
            var iface = tok.tokenizer();
            try std.testing.expect(iface.vocabSize() == vs);

            // --- encodeSpm with random text ---
            var text_buf: [32]u8 = undefined;
            const text_len = smith.valueWithHash(u5, 1) | 1; // 1..31
            for (text_buf[0..text_len]) |*b| {
                b.* = smith.valueWithHash(u8, 2);
            }
            const spm_ids = tok.encodeSpm(text_buf[0..text_len]) catch return;
            defer allocator.free(spm_ids);
            // Invariant: every ID must be a valid vocab index (unk is 0).
            for (spm_ids) |tid| try std.testing.expect(tid < vocab.len);

            // --- decodeSpm with the encoded ids ---
            const spm_decoded = tok.decodeSpm(spm_ids) catch return;
            defer allocator.free(spm_decoded);

            // --- decodeSpm with random token ids (including out-of-range) ---
            var rand_ids: [4]u32 = undefined;
            for (&rand_ids, 0..) |*r, i| {
                r.* = smith.valueWithHash(u32, @intCast(10 + i));
            }
            const spm_dec2 = tok.decodeSpm(&rand_ids) catch return;
            defer allocator.free(spm_dec2);

            // --- encodeSpmNoDummy ---
            const nodummy_ids = tok.encodeSpmNoDummy(text_buf[0..text_len]) catch return;
            defer allocator.free(nodummy_ids);

            // --- encodeSpm / encodeSpmNoDummy with empty string ---
            const empty_spm = tok.encodeSpm("") catch return;
            defer allocator.free(empty_spm);
            try std.testing.expect(empty_spm.len == 0);

            const empty_nodummy = tok.encodeSpmNoDummy("") catch return;
            defer allocator.free(empty_nodummy);
            try std.testing.expect(empty_nodummy.len == 0);

            // --- Now test BPE mode: create a second tokenizer with merges ---
            var tok2 = BpeTokenizer.init(allocator);
            defer tok2.deinit();

            const bpe_vocab = [_][]const u8{ "a", "b", "c", "ab", "bc", "abc" };
            var bpe_vocab_slice: [bpe_vocab.len][]const u8 = undefined;
            for (&bpe_vocab, 0..) |v, i| bpe_vocab_slice[i] = v;

            const bpe_merges = [_][]const u8{ "a b", "ab c" };
            var bpe_merges_slice: [bpe_merges.len][]const u8 = undefined;
            for (&bpe_merges, 0..) |m, i| bpe_merges_slice[i] = m;

            // --- loadFromGGUF ---
            const bpe_eos = smith.valueWithHash(u32, 3) % @as(u32, bpe_vocab.len);
            tok2.loadFromGGUF(&bpe_vocab_slice, &bpe_merges_slice, bpe_eos) catch return;

            // --- encode (BPE mode) with mixed vocab chars, special-token
            // markers, and hostile bytes. Pure a/b/c input never reaches the
            // '<' special-token scan or the byte→unicode mapping of controls
            // and invalid UTF-8, so bias the generator toward those.
            var bpe_text: [8]u8 = undefined;
            const bpe_len = (smith.valueWithHash(u3, 4) | 1); // 1..7
            for (bpe_text[0..bpe_len]) |*b| {
                switch (smith.valueWithHash(u8, 5) % 4) {
                    0 => b.* = 'a' + (smith.valueWithHash(u8, 6) % 3), // a, b, or c — merge logic
                    1 => b.* = '<', // triggers special-token scanning
                    else => b.* = smith.valueWithHash(u8, 7), // arbitrary bytes incl. invalid UTF-8
                }
            }
            const bpe_ids = tok2.encode(bpe_text[0..bpe_len]) catch return;
            defer allocator.free(bpe_ids);
            // Invariant: every ID must be a valid vocab index (unk is 0).
            for (bpe_ids) |tid| try std.testing.expect(tid < bpe_vocab.len);

            // --- decode (BPE mode) ---
            const bpe_decoded = tok2.decode(bpe_ids) catch return;
            defer allocator.free(bpe_decoded);

            // --- decode with random ids ---
            var rand_bpe_ids: [3]u32 = undefined;
            for (&rand_bpe_ids, 0..) |*r, i| {
                r.* = smith.valueWithHash(u32, @intCast(20 + i));
            }
            const bpe_dec2 = tok2.decode(&rand_bpe_ids) catch return;
            defer allocator.free(bpe_dec2);

            // --- encode / decode empty ---
            const empty_enc = tok2.encode("") catch return;
            defer allocator.free(empty_enc);
            try std.testing.expect(empty_enc.len == 0);

            const empty_dec = tok2.decode(&.{}) catch return;
            defer allocator.free(empty_dec);
            try std.testing.expect(empty_dec.len == 0);
        }
    }.f, .{});
}
