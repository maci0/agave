//! Tokenizer interface.
//! Implementations: bpe.zig (byte-level BPE, SentencePiece, and SentencePiece-no-dummy modes).

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
    };
    const vtable = Tokenizer.VTable{ .encode = S.encode, .decode = S.decode, .get_vocab_size = S.getVocabSize };
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
    };
    const vtable = Tokenizer.VTable{
        .encode = S.encode,
        .decode = S.decode,
        .get_vocab_size = S.getVocabSize,
    };
    var impl = S{};
    const tok = Tokenizer{
        .ptr = @ptrCast(&impl),
        .vtable = &vtable,
    };
    try std.testing.expectEqual(@as(u32, 42), tok.vocabSize());
}
