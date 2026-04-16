//! Shared model architecture enum and tokenizer kind.

const std = @import("std");
const build_options = @import("build_options");
const ChatTemplate = @import("chat_template.zig").ChatTemplate;

/// Supported model architectures — used for dispatch, display, and build-time toggles.
pub const Arch = enum {
    qwen35,
    gemma3,
    gemma4,
    gpt_oss,
    nemotron_h,
    nemotron_nano,
    glm4,

    /// Detect model architecture from GGUF/SafeTensors arch string.
    pub fn detect(name: []const u8) ?Arch {
        const map = .{
            .{ "gemma4", .gemma4 },
            .{ "gemma4_text", .gemma4 },
            .{ "gemma3", .gemma3 },
            .{ "gemma3_text", .gemma3 },
            .{ "gemma2", .gemma3 }, // Gemma 2 uses same architecture path as Gemma 3
            .{ "qwen3_5_text", .qwen35 },
            .{ "qwen35moe", .qwen35 },
            .{ "qwen35", .qwen35 },
            .{ "qwen3_5", .qwen35 },
            .{ "qwen3", .qwen35 },
            .{ "qwen2", .qwen35 },
            .{ "gpt-oss", .gpt_oss },
            .{ "gpt_oss", .gpt_oss },
            .{ "gptoss", .gpt_oss },
            .{ "nemotron_h", .nemotron_h },
            .{ "nemotron-h", .nemotron_h },
            .{ "nemotron", .nemotron_h },
            .{ "nemotron_nano", .nemotron_nano },
            .{ "nemotron-nano", .nemotron_nano },
            .{ "glm4_moe_lite", .glm4 },
            .{ "glm4", .glm4 },
            .{ "deepseek2", .glm4 },
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
            .gemma4 => "Gemma 4",
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
            .gemma4 => ChatTemplate.gemma4,
            .gpt_oss => ChatTemplate.gpt_oss,
            .qwen35 => ChatTemplate.qwen35,
            .glm4 => ChatTemplate.glm4,
            else => ChatTemplate.chatml,
        };
    }

    /// Short name of the chat template for this architecture (for display).
    pub fn templateName(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "gemma",
            .gemma4 => "gemma4",
            .gpt_oss => "gpt-oss",
            .qwen35 => "qwen35",
            .glm4 => "glm4",
            else => "chatml",
        };
    }

    /// Returns whether this model architecture was enabled at compile time.
    pub fn isEnabled(self: Arch) bool {
        return switch (self) {
            .gemma3 => build_options.enable_gemma3,
            .gemma4 => build_options.enable_gemma4,
            .qwen35 => build_options.enable_qwen35,
            .gpt_oss => build_options.enable_gpt_oss,
            .nemotron_h => build_options.enable_nemotron_h,
            .nemotron_nano => build_options.enable_nemotron_nano,
            .glm4 => build_options.enable_glm4,
        };
    }

    /// Fallback BOS token ID when metadata is missing.
    /// Returns null for architectures that don't prepend BOS (GPT-2 family).
    pub fn defaultBos(self: Arch) ?u32 {
        return switch (self) {
            .glm4 => glm4_fallback_bos,
            .qwen35, .gpt_oss, .nemotron_h, .nemotron_nano => null,
            .gemma3, .gemma4 => default_bos_id,
        };
    }

    /// Fallback EOS token ID when metadata is missing.
    pub fn defaultEos(self: Arch) u32 {
        return switch (self) {
            .gemma3, .gemma4 => gemma_fallback_eos,
            else => default_fallback_eos,
        };
    }

    /// Returns image token IDs for multimodal architectures, or null for text-only.
    pub fn imageTokens(self: Arch) ?ImageTokens {
        return switch (self) {
            .gemma4 => .{ .start = gemma4_image_start, .end = gemma4_image_start, .pad = gemma4_image_start },
            .gemma3 => .{ .start = gemma3_image_start, .end = gemma3_image_start, .pad = gemma3_image_start },
            .qwen35 => .{ .start = qwen35_image_start, .end = qwen35_image_end, .pad = qwen35_image_pad },
            else => null,
        };
    }

    /// Returns the CLI build flag name for this architecture (e.g. "gpt-oss").
    pub fn buildFlag(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "gemma3",
            .gemma4 => "gemma4",
            .qwen35 => "qwen35",
            .gpt_oss => "gpt-oss",
            .nemotron_h => "nemotron-h",
            .nemotron_nano => "nemotron-nano",
            .glm4 => "glm4",
        };
    }
};

// ── Shared token ID defaults ─────────────────────────────────────

/// Fallback EOS token ID for Gemma models (used when metadata is missing).
pub const gemma_fallback_eos: u32 = 1;
/// Qwen-family fallback EOS token ID (used when metadata is missing).
pub const default_fallback_eos: u32 = 248046;
/// GLM-4 fallback BOS token ID (`[gMASK]`, used when metadata is missing).
pub const glm4_fallback_bos: u32 = 154822;
/// Default BOS token ID when metadata is missing (SentencePiece convention).
pub const default_bos_id: u32 = 2;
/// Maximum end-of-generation token IDs tracked simultaneously.
pub const max_eog_ids: usize = 8;

// ── Image token IDs for multimodal models ─────────────────────

/// Image token IDs for multimodal models.
/// These are special tokens in the vocabulary that serve as placeholders
/// for visual embeddings during forward passes.
pub const ImageTokens = struct {
    /// Start-of-image token ID (e.g. `<img>`, `<|vision_start|>`).
    start: u32,
    /// End-of-image token ID (e.g. `</img>`, `<|vision_end|>`).
    end: u32,
    /// Placeholder token ID repeated n_visual_tokens times between start/end.
    pad: u32,
};

/// Gemma 4 uses <|image|> (258880) as the image placeholder token.
/// Note: 255999 is <|image> (without trailing |) — different token.
/// 219 is <img> — used in Gemma 3, not Gemma 4.
const gemma4_image_start: u32 = 258880;
/// Gemma 3 uses the same SentencePiece token IDs as Gemma 4.
const gemma3_image_start: u32 = 219;
/// Qwen 3.5 VL image token IDs.
const qwen35_image_start: u32 = 248053;
const qwen35_image_end: u32 = 248054;
const qwen35_image_pad: u32 = 248056;

test "Arch.detect known names" {
    try std.testing.expectEqual(Arch.gemma4, Arch.detect("gemma4").?);
    try std.testing.expectEqual(Arch.gemma4, Arch.detect("gemma4_text").?);
    try std.testing.expectEqual(Arch.gemma3, Arch.detect("gemma3").?);
    try std.testing.expectEqual(Arch.gemma3, Arch.detect("gemma3_text").?);
    try std.testing.expectEqual(Arch.gemma3, Arch.detect("gemma2").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen35").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_5_text").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen35moe").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_5").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen2").?);
    try std.testing.expectEqual(Arch.gpt_oss, Arch.detect("gpt-oss").?);
    try std.testing.expectEqual(Arch.gpt_oss, Arch.detect("gpt_oss").?);
    try std.testing.expectEqual(Arch.gpt_oss, Arch.detect("gptoss").?);
    try std.testing.expectEqual(Arch.nemotron_h, Arch.detect("nemotron_h").?);
    try std.testing.expectEqual(Arch.nemotron_h, Arch.detect("nemotron-h").?);
    try std.testing.expectEqual(Arch.nemotron_h, Arch.detect("nemotron").?);
    try std.testing.expectEqual(Arch.nemotron_nano, Arch.detect("nemotron_nano").?);
    try std.testing.expectEqual(Arch.nemotron_nano, Arch.detect("nemotron-nano").?);
    try std.testing.expectEqual(Arch.glm4, Arch.detect("glm4").?);
    try std.testing.expectEqual(Arch.glm4, Arch.detect("deepseek2").?);
    try std.testing.expectEqual(Arch.glm4, Arch.detect("glm4_moe_lite").?);
    try std.testing.expect(Arch.detect("unknown_model") == null);
}

test "Arch.displayName" {
    try std.testing.expectEqualStrings("Gemma 3", Arch.gemma3.displayName());
    try std.testing.expectEqualStrings("Gemma 4", Arch.gemma4.displayName());
    try std.testing.expectEqualStrings("Qwen 3.5", Arch.qwen35.displayName());
    try std.testing.expectEqualStrings("GPT-OSS", Arch.gpt_oss.displayName());
    try std.testing.expectEqualStrings("Nemotron-H", Arch.nemotron_h.displayName());
    try std.testing.expectEqualStrings("Nemotron-Nano", Arch.nemotron_nano.displayName());
    try std.testing.expectEqualStrings("GLM-4", Arch.glm4.displayName());
}

test "Arch.defaultBos" {
    try std.testing.expectEqual(@as(?u32, 2), Arch.gemma3.defaultBos());
    try std.testing.expectEqual(@as(?u32, 2), Arch.gemma4.defaultBos());
    try std.testing.expectEqual(@as(?u32, 154822), Arch.glm4.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.qwen35.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.gpt_oss.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.nemotron_h.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.nemotron_nano.defaultBos());
}

test "Arch.defaultEos" {
    try std.testing.expectEqual(@as(u32, 1), Arch.gemma3.defaultEos());
    try std.testing.expectEqual(@as(u32, 1), Arch.gemma4.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.qwen35.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.gpt_oss.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.glm4.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.nemotron_h.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.nemotron_nano.defaultEos());
}

test "Arch.imageTokens multimodal" {
    // Gemma 4 should return image tokens (258880 = <|image|>)
    const g4 = Arch.gemma4.imageTokens().?;
    try std.testing.expectEqual(@as(u32, 258880), g4.start);
    try std.testing.expectEqual(@as(u32, 258880), g4.end);
    try std.testing.expectEqual(@as(u32, 258880), g4.pad);

    // Gemma 3 should return image tokens (219 = <img>)
    const g3 = Arch.gemma3.imageTokens().?;
    try std.testing.expectEqual(@as(u32, 219), g3.start);
    try std.testing.expectEqual(@as(u32, 219), g3.end);

    // Qwen 3.5 should return image tokens
    const qw = Arch.qwen35.imageTokens().?;
    try std.testing.expectEqual(@as(u32, 248053), qw.start);
    try std.testing.expectEqual(@as(u32, 248054), qw.end);
    try std.testing.expectEqual(@as(u32, 248056), qw.pad);

    // Text-only architectures should return null
    try std.testing.expect(Arch.gpt_oss.imageTokens() == null);
    try std.testing.expect(Arch.nemotron_h.imageTokens() == null);
    try std.testing.expect(Arch.nemotron_nano.imageTokens() == null);
    try std.testing.expect(Arch.glm4.imageTokens() == null);
}
