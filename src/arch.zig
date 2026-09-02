//! Shared model architecture enum, chat-template mapping, and multimodal defaults.
//!
//! Depends on `chat_template.zig` for template structs and on `image_tokens.zig`
//! for the leaf `ImageTokens` type (re-exported here for a stable public path).

const std = @import("std");
const build_options = @import("build_options");
const ChatTemplate = @import("chat_template.zig").ChatTemplate;
/// Re-exported `ImageTokens` type from `image_tokens.zig` for a stable public path.
pub const ImageTokens = @import("image_tokens.zig").ImageTokens;

/// Supported model architectures, used for dispatch, display, and build-time toggles.
pub const Arch = enum {
    qwen35,
    qwen4_exp,
    gemma3,
    gemma4,
    diffusion_gemma,
    gpt_oss,
    nemotron_h,
    nemotron_nano,
    glm4,
    deepseek4,
    llama4,
    dflash2,

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
            .{ "qwen3_5_moe", .qwen35 },
            .{ "qwen3_5_moe_text", .qwen35 },
            .{ "qwen36", .qwen35 },
            .{ "qwen3_6", .qwen35 },
            .{ "qwen38", .qwen35 },
            .{ "qwen3_8", .qwen35 },
            .{ "qwen3_8_text", .qwen35 },
            .{ "qwen35", .qwen35 },
            .{ "qwen3_5", .qwen35 },
            .{ "qwen3", .qwen35 },
            .{ "qwen2", .qwen35 },
            .{ "qwen4_exp", .qwen4_exp },
            .{ "qwen4_exp_text", .qwen4_exp },
            .{ "qwen4exp", .qwen4_exp },
            .{ "qwen4_text", .qwen4_exp },
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
            .{ "deepseek_v4", .deepseek4 },
            .{ "deepseek4", .deepseek4 },
            .{ "dflash", .deepseek4 }, // DeepSeek V4 Flash DSpark draft model
            .{ "dflash2", .dflash2 }, // DFlash2 block-diffusion drafter (Inco AI)
            .{ "llama4", .llama4 },
            .{ "llama4_text", .llama4 },
            .{ "diffusion_gemma", .diffusion_gemma },
            .{ "diffusion_gemma_text", .diffusion_gemma },
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
            .diffusion_gemma => "DiffusionGemma",
            .qwen35 => "Qwen 3.5/3.8",
            .qwen4_exp => "Qwen4-Exp",
            .gpt_oss => "GPT-OSS",
            .nemotron_h => "Nemotron-H",
            .nemotron_nano => "Nemotron-Nano",
            .glm4 => "GLM-4",
            .deepseek4 => "DeepSeek V4 Flash",
            .llama4 => "Llama 4",
            .dflash2 => "DFlash2 Drafter",
        };
    }

    /// Default chat template for this architecture.
    pub fn chatTemplate(self: Arch) ChatTemplate {
        return switch (self) {
            .gemma3 => ChatTemplate.gemma,
            .gemma4, .diffusion_gemma => ChatTemplate.gemma4,
            .gpt_oss => ChatTemplate.gpt_oss,
            .qwen35, .qwen4_exp => ChatTemplate.qwen35,
            .glm4 => ChatTemplate.glm4,
            .deepseek4 => ChatTemplate.deepseek4,
            .llama4 => ChatTemplate.llama4,
            else => ChatTemplate.chatml,
        };
    }

    /// Layer-count-aware template selection. Gemma 4 12B (48 layers) uses
    /// `gemma4_unified` with a thinking-channel prefix; E2B/E4B use plain `gemma4`.
    pub fn chatTemplateForLayers(self: Arch, n_layers: u32) ChatTemplate {
        if (self == .gemma4 and n_layers >= 48) return ChatTemplate.gemma4_unified;
        return self.chatTemplate();
    }

    /// Short name of the chat template for this architecture (for display).
    pub fn templateName(self: Arch) []const u8 {
        return switch (self) {
            .gemma3 => "gemma",
            .gemma4, .diffusion_gemma => "gemma4",
            .gpt_oss => "gpt-oss",
            .qwen35, .qwen4_exp => "qwen35",
            .glm4 => "glm4",
            .deepseek4 => "deepseek4",
            .llama4 => "llama4",
            else => "chatml",
        };
    }

    /// Returns whether this model architecture was enabled at compile time.
    pub fn isEnabled(self: Arch) bool {
        return switch (self) {
            .gemma3 => build_options.enable_gemma3,
            .gemma4 => build_options.enable_gemma4,
            .diffusion_gemma => build_options.enable_diffusion_gemma,
            .qwen35 => build_options.enable_qwen35,
            .qwen4_exp => build_options.enable_qwen4_exp,
            .gpt_oss => build_options.enable_gpt_oss,
            .nemotron_h => build_options.enable_nemotron_h,
            .nemotron_nano => build_options.enable_nemotron_nano,
            .glm4 => build_options.enable_glm4,
            .deepseek4 => build_options.enable_deepseek4,
            .llama4 => build_options.enable_llama4,
            .dflash2 => build_options.enable_dflash2,
        };
    }

    /// Fallback BOS token ID when metadata is missing.
    /// Returns null for architectures that don't prepend BOS (GPT-2 family).
    pub fn defaultBos(self: Arch) ?u32 {
        return switch (self) {
            .glm4, .deepseek4 => glm4_fallback_bos,
            .qwen35, .qwen4_exp, .gpt_oss, .nemotron_h, .nemotron_nano, .dflash2 => null,
            .llama4 => llama4_fallback_bos,
            .gemma3, .gemma4, .diffusion_gemma => default_bos_id,
        };
    }

    /// Fallback EOS token ID when metadata is missing.
    pub fn defaultEos(self: Arch) u32 {
        return switch (self) {
            .gemma3, .gemma4, .diffusion_gemma => gemma_fallback_eos,
            .llama4 => llama4_fallback_eos,
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
            .diffusion_gemma => "diffusion-gemma",
            .qwen35 => "qwen35",
            .qwen4_exp => "qwen4-exp",
            .gpt_oss => "gpt-oss",
            .nemotron_h => "nemotron-h",
            .nemotron_nano => "nemotron-nano",
            .glm4 => "glm4",
            .deepseek4 => "deepseek4",
            .llama4 => "llama4",
            .dflash2 => "dflash2",
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
/// Llama 4 fallback BOS token ID.
pub const llama4_fallback_bos: u32 = 128000;
/// Llama 4 fallback EOS token ID.
pub const llama4_fallback_eos: u32 = 128009;
/// Default BOS token ID when metadata is missing (SentencePiece convention).
pub const default_bos_id: u32 = 2;
/// Maximum end-of-generation token IDs tracked simultaneously.
pub const max_eog_ids: usize = 8;

// ── Image token IDs for multimodal models ─────────────────────

/// Gemma 4 uses <|image|> (258880) as the image placeholder token.
/// Note: 255999 is <|image> (without trailing |), different token.
/// 219 is <img>, used in Gemma 3, not Gemma 4.
const gemma4_image_start: u32 = 258880;
/// Gemma 3 uses <img> (219) as the image placeholder token.
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
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_5_moe").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_5_moe_text").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen36").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_6").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen38").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_8").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_8_text").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3_5").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen3").?);
    try std.testing.expectEqual(Arch.qwen35, Arch.detect("qwen2").?);
    try std.testing.expectEqual(Arch.qwen4_exp, Arch.detect("qwen4_exp").?);
    try std.testing.expectEqual(Arch.qwen4_exp, Arch.detect("qwen4_exp_text").?);
    try std.testing.expectEqual(Arch.qwen4_exp, Arch.detect("qwen4exp").?);
    try std.testing.expectEqual(Arch.qwen4_exp, Arch.detect("qwen4_text").?);
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
    try std.testing.expectEqual(Arch.deepseek4, Arch.detect("deepseek4").?);
    try std.testing.expectEqual(Arch.deepseek4, Arch.detect("deepseek_v4").?);
    try std.testing.expectEqual(Arch.deepseek4, Arch.detect("dflash").?);
    try std.testing.expectEqual(Arch.dflash2, Arch.detect("dflash2").?);
    try std.testing.expectEqual(Arch.llama4, Arch.detect("llama4").?);
    try std.testing.expectEqual(Arch.llama4, Arch.detect("llama4_text").?);
    try std.testing.expectEqual(Arch.diffusion_gemma, Arch.detect("diffusion_gemma").?);
    try std.testing.expectEqual(Arch.diffusion_gemma, Arch.detect("diffusion_gemma_text").?);
    try std.testing.expectEqual(@as(?Arch, null), Arch.detect("unknown_model"));
}

test "Arch.displayName" {
    try std.testing.expectEqualStrings("Gemma 3", Arch.gemma3.displayName());
    try std.testing.expectEqualStrings("Gemma 4", Arch.gemma4.displayName());
    try std.testing.expectEqualStrings("DiffusionGemma", Arch.diffusion_gemma.displayName());
    try std.testing.expectEqualStrings("Qwen 3.5/3.8", Arch.qwen35.displayName());
    try std.testing.expectEqualStrings("Qwen4-Exp", Arch.qwen4_exp.displayName());
    try std.testing.expectEqualStrings("GPT-OSS", Arch.gpt_oss.displayName());
    try std.testing.expectEqualStrings("Nemotron-H", Arch.nemotron_h.displayName());
    try std.testing.expectEqualStrings("Nemotron-Nano", Arch.nemotron_nano.displayName());
    try std.testing.expectEqualStrings("GLM-4", Arch.glm4.displayName());
    try std.testing.expectEqualStrings("DeepSeek V4 Flash", Arch.deepseek4.displayName());
    try std.testing.expectEqualStrings("Llama 4", Arch.llama4.displayName());
    try std.testing.expectEqualStrings("DFlash2 Drafter", Arch.dflash2.displayName());
}

test "Arch.defaultBos" {
    try std.testing.expectEqual(@as(?u32, 2), Arch.gemma3.defaultBos());
    try std.testing.expectEqual(@as(?u32, 2), Arch.gemma4.defaultBos());
    try std.testing.expectEqual(@as(?u32, 2), Arch.diffusion_gemma.defaultBos());
    try std.testing.expectEqual(@as(?u32, 154822), Arch.glm4.defaultBos());
    try std.testing.expectEqual(@as(?u32, 154822), Arch.deepseek4.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.qwen35.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.qwen4_exp.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.gpt_oss.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.nemotron_h.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.nemotron_nano.defaultBos());
    try std.testing.expectEqual(@as(?u32, null), Arch.dflash2.defaultBos());
    try std.testing.expectEqual(@as(?u32, 128000), Arch.llama4.defaultBos());
}

test "Arch.defaultEos" {
    try std.testing.expectEqual(@as(u32, 1), Arch.gemma3.defaultEos());
    try std.testing.expectEqual(@as(u32, 1), Arch.gemma4.defaultEos());
    try std.testing.expectEqual(@as(u32, 1), Arch.diffusion_gemma.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.qwen35.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.qwen4_exp.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.gpt_oss.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.glm4.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.deepseek4.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.nemotron_h.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.nemotron_nano.defaultEos());
    try std.testing.expectEqual(@as(u32, 248046), Arch.dflash2.defaultEos());
    try std.testing.expectEqual(@as(u32, 128009), Arch.llama4.defaultEos());
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
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.gpt_oss.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.nemotron_h.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.nemotron_nano.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.glm4.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.qwen4_exp.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.deepseek4.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.llama4.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.diffusion_gemma.imageTokens());
    try std.testing.expectEqual(@as(?ImageTokens, null), Arch.dflash2.imageTokens());
}

test "Arch.chatTemplate returns correct template per arch" {
    // Verify each arch returns its expected template
    try std.testing.expectEqual(ChatTemplate.gemma, Arch.gemma3.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.gemma4, Arch.gemma4.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.gemma4, Arch.diffusion_gemma.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.qwen35, Arch.qwen35.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.qwen35, Arch.qwen4_exp.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.gpt_oss, Arch.gpt_oss.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.glm4, Arch.glm4.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.deepseek4, Arch.deepseek4.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.llama4, Arch.llama4.chatTemplate());
    // nemotron variants and dflash2 use chatml default
    try std.testing.expectEqual(ChatTemplate.chatml, Arch.nemotron_h.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.chatml, Arch.nemotron_nano.chatTemplate());
    try std.testing.expectEqual(ChatTemplate.chatml, Arch.dflash2.chatTemplate());
}

test "Arch.chatTemplateForLayers gemma4 12B uses unified" {
    try std.testing.expectEqual(ChatTemplate.gemma4_unified, Arch.gemma4.chatTemplateForLayers(48));
    try std.testing.expectEqual(ChatTemplate.gemma4_unified, Arch.gemma4.chatTemplateForLayers(49));
    try std.testing.expectEqual(ChatTemplate.gemma4, Arch.gemma4.chatTemplateForLayers(47));
    try std.testing.expectEqual(ChatTemplate.gemma4, Arch.gemma4.chatTemplateForLayers(0));
    try std.testing.expectEqual(ChatTemplate.gemma, Arch.gemma3.chatTemplateForLayers(48));
    try std.testing.expectEqual(ChatTemplate.gemma4, Arch.diffusion_gemma.chatTemplateForLayers(48));
}

test "Arch.templateName returns non-empty strings" {
    try std.testing.expectEqualStrings("gemma", Arch.gemma3.templateName());
    try std.testing.expectEqualStrings("gemma4", Arch.gemma4.templateName());
    try std.testing.expectEqualStrings("gemma4", Arch.diffusion_gemma.templateName());
    try std.testing.expectEqualStrings("qwen35", Arch.qwen35.templateName());
    try std.testing.expectEqualStrings("qwen35", Arch.qwen4_exp.templateName());
    try std.testing.expectEqualStrings("gpt-oss", Arch.gpt_oss.templateName());
    try std.testing.expectEqualStrings("glm4", Arch.glm4.templateName());
    try std.testing.expectEqualStrings("deepseek4", Arch.deepseek4.templateName());
    try std.testing.expectEqualStrings("llama4", Arch.llama4.templateName());
    try std.testing.expectEqualStrings("chatml", Arch.nemotron_h.templateName());
    try std.testing.expectEqualStrings("chatml", Arch.nemotron_nano.templateName());
    try std.testing.expectEqualStrings("chatml", Arch.dflash2.templateName());
}

test "Arch.buildFlag returns valid flag names" {
    try std.testing.expectEqualStrings("gemma3", Arch.gemma3.buildFlag());
    try std.testing.expectEqualStrings("gemma4", Arch.gemma4.buildFlag());
    try std.testing.expectEqualStrings("diffusion-gemma", Arch.diffusion_gemma.buildFlag());
    try std.testing.expectEqualStrings("qwen35", Arch.qwen35.buildFlag());
    try std.testing.expectEqualStrings("qwen4-exp", Arch.qwen4_exp.buildFlag());
    try std.testing.expectEqualStrings("gpt-oss", Arch.gpt_oss.buildFlag());
    try std.testing.expectEqualStrings("nemotron-h", Arch.nemotron_h.buildFlag());
    try std.testing.expectEqualStrings("nemotron-nano", Arch.nemotron_nano.buildFlag());
    try std.testing.expectEqualStrings("glm4", Arch.glm4.buildFlag());
    try std.testing.expectEqualStrings("deepseek4", Arch.deepseek4.buildFlag());
    try std.testing.expectEqualStrings("llama4", Arch.llama4.buildFlag());
    try std.testing.expectEqualStrings("dflash2", Arch.dflash2.buildFlag());
}

test "Arch.isEnabled matches compile-time build flags" {
    try std.testing.expectEqual(build_options.enable_gemma3, Arch.gemma3.isEnabled());
    try std.testing.expectEqual(build_options.enable_gemma4, Arch.gemma4.isEnabled());
    try std.testing.expectEqual(build_options.enable_diffusion_gemma, Arch.diffusion_gemma.isEnabled());
    try std.testing.expectEqual(build_options.enable_qwen35, Arch.qwen35.isEnabled());
    try std.testing.expectEqual(build_options.enable_qwen4_exp, Arch.qwen4_exp.isEnabled());
    try std.testing.expectEqual(build_options.enable_gpt_oss, Arch.gpt_oss.isEnabled());
    try std.testing.expectEqual(build_options.enable_nemotron_h, Arch.nemotron_h.isEnabled());
    try std.testing.expectEqual(build_options.enable_nemotron_nano, Arch.nemotron_nano.isEnabled());
    try std.testing.expectEqual(build_options.enable_glm4, Arch.glm4.isEnabled());
    try std.testing.expectEqual(build_options.enable_deepseek4, Arch.deepseek4.isEnabled());
    try std.testing.expectEqual(build_options.enable_llama4, Arch.llama4.isEnabled());
    try std.testing.expectEqual(build_options.enable_dflash2, Arch.dflash2.isEnabled());
}

test "fuzz: all arch functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const num_archs = @typeInfo(Arch).@"enum".fields.len;

            const detect_len = smith.valueWithHash(u4, 0);
            var detect_buf: [16]u8 = undefined;
            for (detect_buf[0..detect_len]) |*b| b.* = smith.valueWithHash(u8, 1);
            _ = Arch.detect(detect_buf[0..detect_len]);

            const arch_idx = smith.valueWithHash(u8, 2) % num_archs;
            const arch: Arch = @enumFromInt(arch_idx);

            const dn = arch.displayName();
            try std.testing.expect(dn.len > 0);

            _ = arch.chatTemplate();

            const tn = arch.templateName();
            try std.testing.expect(tn.len > 0);

            const enabled = arch.isEnabled();
            try std.testing.expect(enabled or !enabled);

            if (arch.defaultBos()) |bos| {
                try std.testing.expect(bos > 0);
            }

            const eos = arch.defaultEos();
            try std.testing.expect(eos > 0);

            if (arch.imageTokens()) |img| {
                try std.testing.expect(img.pad > 0);
            }

            const bf = arch.buildFlag();
            try std.testing.expect(bf.len > 0);

            try std.testing.expect(gemma_fallback_eos == 1);
            try std.testing.expect(default_fallback_eos == 248046);
            try std.testing.expect(glm4_fallback_bos == 154822);
            try std.testing.expect(llama4_fallback_bos == 128000);
            try std.testing.expect(llama4_fallback_eos == 128009);
            try std.testing.expect(default_bos_id == 2);
            try std.testing.expect(max_eog_ids == 8);

            const it = ImageTokens{
                .start = smith.valueWithHash(u32, 3),
                .end = smith.valueWithHash(u32, 4),
                .pad = smith.valueWithHash(u32, 5),
            };
            _ = it;
        }
    }.f, .{});
}
