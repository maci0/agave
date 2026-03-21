//! Shared model architecture enum and tokenizer kind.

const std = @import("std");
const build_options = @import("build_options");
const ChatTemplate = @import("chat_template.zig").ChatTemplate;

/// Supported model architectures — used for dispatch, display, and build-time toggles.
pub const Arch = enum {
    qwen35,
    gemma3,
    gpt_oss,
    nemotron_h,
    nemotron_nano,
    glm4,

    /// Detect model architecture from GGUF/SafeTensors arch string.
    pub fn detect(name: []const u8) ?Arch {
        const map = .{
            .{ "gemma3", .gemma3 },
            .{ "gemma3_text", .gemma3 },
            .{ "gemma2", .gemma3 },
            .{ "qwen35moe", .qwen35 },
            .{ "qwen35", .qwen35 },
            .{ "qwen3", .qwen35 },
            .{ "qwen2", .qwen35 },
            .{ "gpt-oss", .gpt_oss },
            .{ "gptoss", .gpt_oss },
            .{ "nemotron_h", .nemotron_h },
            .{ "nemotron-h", .nemotron_h },
            .{ "nemotron", .nemotron_h },
            .{ "nemotron_nano", .nemotron_nano },
            .{ "nemotron-nano", .nemotron_nano },
            .{ "glm4_moe_lite", .glm4 },
            .{ "glm4", .glm4 },
        };
        inline for (map) |entry| {
            if (std.mem.eql(u8, name, entry[0])) return entry[1];
        }
        return null;
    }

    /// Human-readable model name for banner display.
    pub fn displayName(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "Gemma 3",
            .qwen35 => "Qwen 3.5",
            .gpt_oss => "GPT-OSS",
            .nemotron_h => "Nemotron-H",
            .nemotron_nano => "Nemotron-Nano",
            .glm4 => "GLM-4",
        };
    }

    /// Default chat template for this architecture.
    pub fn chatTemplate(self: Arch) ChatTemplate {
        return switch (self) {
            .gemma3 => ChatTemplate.gemma,
            .gpt_oss => ChatTemplate.gpt_oss,
            .qwen35 => ChatTemplate.qwen35,
            else => ChatTemplate.chatml,
        };
    }

    /// Short name of the chat template for this architecture (for display).
    pub fn templateName(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "gemma",
            .gpt_oss => "gpt-oss",
            .qwen35 => "qwen35",
            else => "chatml",
        };
    }

    /// Returns whether this model architecture was enabled at compile time.
    pub fn isEnabled(self: Arch) bool {
        return switch (self) {
            .gemma3 => build_options.enable_gemma3,
            .qwen35 => build_options.enable_qwen35,
            .gpt_oss => build_options.enable_gpt_oss,
            .nemotron_h => build_options.enable_nemotron_h,
            .nemotron_nano => build_options.enable_nemotron_nano,
            .glm4 => build_options.enable_glm4,
        };
    }

    /// Returns the CLI build flag name for this architecture (e.g. "gpt-oss").
    pub fn buildFlag(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "gemma3",
            .qwen35 => "qwen35",
            .gpt_oss => "gpt-oss",
            .nemotron_h => "nemotron-h",
            .nemotron_nano => "nemotron-nano",
            .glm4 => "glm4",
        };
    }
};

/// Tokenizer mode: BPE (byte-pair merges), SPM (SentencePiece greedy), or SPM without dummy prefix.
pub const TokenizerKind = enum { bpe, spm, spm_no_dummy };

// ── Shared token ID defaults ─────────────────────────────────────

/// Fallback EOS token ID for Gemma models (used when metadata is missing).
pub const gemma_fallback_eos: u32 = 1;
/// Qwen-family fallback EOS token ID (used when metadata is missing).
pub const default_fallback_eos: u32 = 248046;
/// Default BOS token ID when metadata is missing (SentencePiece convention).
pub const default_bos_id: u32 = 2;
/// Maximum end-of-generation token IDs tracked simultaneously.
pub const max_eog_ids: usize = 8;

test "Arch.detect known names" {
    try std.testing.expectEqual(Arch.gemma3, Arch.detect("gemma3").?);
    try std.testing.expectEqual(Arch.gemma3, Arch.detect("gemma2").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen35").?);
    try std.testing.expectEqual(Arch.gpt_oss, Arch.detect("gpt-oss").?);
    try std.testing.expect(Arch.detect("unknown_model") == null);
}

test "Arch.displayName" {
    try std.testing.expectEqualStrings("Gemma 3", Arch.gemma3.displayName());
    try std.testing.expectEqualStrings("Qwen 3.5", Arch.qwen35.displayName());
    try std.testing.expectEqualStrings("GPT-OSS", Arch.gpt_oss.displayName());
    try std.testing.expectEqualStrings("Nemotron-H", Arch.nemotron_h.displayName());
    try std.testing.expectEqualStrings("Nemotron-Nano", Arch.nemotron_nano.displayName());
    try std.testing.expectEqualStrings("GLM-4", Arch.glm4.displayName());
}
