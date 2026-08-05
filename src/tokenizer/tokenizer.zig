//! Tokenizer interface.
//! Implementations: bpe.zig (byte-level BPE, SentencePiece, and SentencePiece-no-dummy modes).
//!
//! `BpeTokenizer` is re-exported here as the package barrel. That creates a
//! compile-time import edge with `bpe.zig` (interface ←→ impl); callers should
//! keep importing via this module, not reach into `bpe.zig` directly.

const std = @import("std");

/// Error set for encode/decode operations via the VTable interface.
/// Loading operations may return additional errors.
pub const TokenizerError = error{OutOfMemory};

/// Generic tokenizer interface dispatching encode/decode via VTable.
pub const Tokenizer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        encode: *const fn (self: *anyopaque, text: []const u8) TokenizerError![]u32,
        decode: *const fn (self: *anyopaque, tokens: []const u32) TokenizerError![]u8,
        get_vocab_size: *const fn (self: *anyopaque) u32,
        get_vocab_texts: *const fn (self: *anyopaque) []const []const u8,
    };

    /// Encode text into a sequence of token IDs.
    pub fn encode(self: Tokenizer, text: []const u8) TokenizerError![]u32 {
        return self.vtable.encode(self.ptr, text);
    }
    /// Decode a sequence of token IDs back into text.
    pub fn decode(self: Tokenizer, tokens: []const u32) TokenizerError![]u8 {
        return self.vtable.decode(self.ptr, tokens);
    }
    /// Return the vocabulary size.
    pub fn vocabSize(self: Tokenizer) u32 {
        return self.vtable.get_vocab_size(self.ptr);
    }
    /// Return per-token text strings indexed by token ID.
    pub fn getVocabTexts(self: Tokenizer) []const []const u8 {
        return self.vtable.get_vocab_texts(self.ptr);
    }
};

/// Tokenizer mode: BPE (byte-pair merges), SPM (SentencePiece greedy), or SPM without dummy prefix.
pub const TokenizerKind = enum { bpe, spm, spm_no_dummy };

/// BPE tokenizer implementation (supports BPE and SPM modes) — re-exported so callers use tokenizer.zig as the single import.
pub const BpeTokenizer = @import("bpe.zig").BpeTokenizer;

// ── Tests ─────────────────────────────────────────────────────────

test "Tokenizer encode error propagates through VTable" {
    const S = struct {
        fn encode(_: *anyopaque, _: []const u8) TokenizerError![]u32 {
            return error.OutOfMemory;
        }
        fn decode(_: *anyopaque, _: []const u32) TokenizerError![]u8 {
            return error.OutOfMemory;
        }
        fn getVocabSize(_: *anyopaque) u32 {
            return 0;
        }
        fn getVocabTexts(_: *anyopaque) []const []const u8 {
            return &.{};
        }
    };
    const vtable = Tokenizer.VTable{ .encode = S.encode, .decode = S.decode, .get_vocab_size = S.getVocabSize, .get_vocab_texts = S.getVocabTexts };
    var dummy: u8 = 0;
    const tok = Tokenizer{ .ptr = @ptrCast(&dummy), .vtable = &vtable };
    try std.testing.expectError(error.OutOfMemory, tok.encode("test"));
    try std.testing.expectError(error.OutOfMemory, tok.decode(&.{0}));
}

test "Tokenizer VTable dispatch" {
    // Verify encode/decode/vocabSize dispatch through VTable correctly
    const S = struct {
        vocab_size: u32 = 42,

        fn encode(_: *anyopaque, _: []const u8) TokenizerError![]u32 {
            return error.OutOfMemory; // stub
        }
        fn decode(_: *anyopaque, _: []const u32) TokenizerError![]u8 {
            return error.OutOfMemory; // stub
        }
        fn getVocabSize(ptr: *anyopaque) u32 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return self.vocab_size;
        }
        fn getVocabTexts(_: *anyopaque) []const []const u8 {
            return &.{};
        }
    };
    const vtable = Tokenizer.VTable{
        .encode = S.encode,
        .decode = S.decode,
        .get_vocab_size = S.getVocabSize,
        .get_vocab_texts = S.getVocabTexts,
    };
    var impl = S{};
    const tok = Tokenizer{
        .ptr = @ptrCast(&impl),
        .vtable = &vtable,
    };
    try std.testing.expectEqual(@as(u32, 42), tok.vocabSize());
}

test "Tokenizer getVocabTexts dispatch" {
    const S = struct {
        texts: []const []const u8 = &.{ "hello", "world" },

        fn encode(_: *anyopaque, _: []const u8) TokenizerError![]u32 {
            return error.OutOfMemory;
        }
        fn decode(_: *anyopaque, _: []const u32) TokenizerError![]u8 {
            return error.OutOfMemory;
        }
        fn getVocabSize(_: *anyopaque) u32 {
            return 2;
        }
        fn getVocabTexts(ptr: *anyopaque) []const []const u8 {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return self.texts;
        }
    };
    const vtable = Tokenizer.VTable{
        .encode = S.encode,
        .decode = S.decode,
        .get_vocab_size = S.getVocabSize,
        .get_vocab_texts = S.getVocabTexts,
    };
    var impl = S{};
    const tok = Tokenizer{ .ptr = @ptrCast(&impl), .vtable = &vtable };
    const texts = tok.getVocabTexts();
    try std.testing.expectEqual(@as(usize, 2), texts.len);
    try std.testing.expectEqualStrings("hello", texts[0]);
    try std.testing.expectEqualStrings("world", texts[1]);
}

test "TokenizerKind enum variants" {
    try std.testing.expect(TokenizerKind.bpe != TokenizerKind.spm);
    try std.testing.expect(TokenizerKind.spm != TokenizerKind.spm_no_dummy);
    try std.testing.expect(TokenizerKind.bpe != TokenizerKind.spm_no_dummy);
    // Verify there are exactly 3 variants
    const fields = @typeInfo(TokenizerKind).@"enum".fields;
    try std.testing.expectEqual(@as(usize, 3), fields.len);
}

test "fuzz: all tokenizer functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // -- Mock VTable that returns deterministic results based on input --
            const Mock = struct {
                val: u32,

                fn encode(ptr: *anyopaque, _: []const u8) TokenizerError![]u32 {
                    _ = ptr;
                    return error.OutOfMemory; // stub: no allocator available
                }
                fn decode(ptr: *anyopaque, _: []const u32) TokenizerError![]u8 {
                    _ = ptr;
                    return error.OutOfMemory; // stub: no allocator available
                }
                fn getVocabSize(ptr: *anyopaque) u32 {
                    const self: *@This() = @ptrCast(@alignCast(ptr));
                    return self.val;
                }
                fn getVocabTexts(_: *anyopaque) []const []const u8 {
                    return &.{};
                }
            };

            const vtable = Tokenizer.VTable{
                .encode = Mock.encode,
                .decode = Mock.decode,
                .get_vocab_size = Mock.getVocabSize,
                .get_vocab_texts = Mock.getVocabTexts,
            };

            var mock = Mock{ .val = smith.valueWithHash(u32, 0) };
            const tok = Tokenizer{ .ptr = @ptrCast(&mock), .vtable = &vtable };

            // Exercise Tokenizer.encode with random-length text
            const text_len = smith.valueWithHash(u8, 1) % 64;
            var text_buf: [64]u8 = undefined;
            for (text_buf[0..text_len]) |*b| b.* = smith.valueWithHash(u8, 2);
            _ = tok.encode(text_buf[0..text_len]) catch |e| {
                try std.testing.expectEqual(TokenizerError.OutOfMemory, e);
            };

            // Exercise Tokenizer.decode with random token IDs
            const tok_len = smith.valueWithHash(u8, 3) % 16;
            var tok_buf: [16]u32 = undefined;
            for (tok_buf[0..tok_len]) |*t| t.* = smith.valueWithHash(u32, 4);
            _ = tok.decode(tok_buf[0..tok_len]) catch |e| {
                try std.testing.expectEqual(TokenizerError.OutOfMemory, e);
            };

            // Exercise Tokenizer.vocabSize — must return mock val
            const vs = tok.vocabSize();
            try std.testing.expectEqual(mock.val, vs);

            // Exercise Tokenizer.getVocabTexts — must return empty slice from mock
            const texts = tok.getVocabTexts();
            try std.testing.expectEqual(@as(usize, 0), texts.len);

            // Exercise TokenizerKind — convert random u8 to enum variant
            const kind_idx = smith.valueWithHash(u8, 5) % 3;
            const kind: TokenizerKind = @enumFromInt(kind_idx);
            try std.testing.expect(@intFromEnum(kind) < 3);

            // Exercise TokenizerError — verify it is a valid error set
            comptime {
                const info = @typeInfo(TokenizerError);
                _ = info;
            }

            // Exercise BpeTokenizer — comptime verify the re-exported type exists
            comptime {
                _ = BpeTokenizer;
            }

            // Exercise Tokenizer.VTable — verify field accessibility
            comptime {
                _ = @TypeOf(Tokenizer.VTable{
                    .encode = Mock.encode,
                    .decode = Mock.decode,
                    .get_vocab_size = Mock.getVocabSize,
                    .get_vocab_texts = Mock.getVocabTexts,
                });
            }
        }
    }.f, .{});
}
