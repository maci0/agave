//! Recipes, optional proven-default configurations for model + hardware combos.
//!
//! A recipe bundles sampling parameters, context size, and generation limits
//! that are known to work well together. Users can override
//! any individual field via CLI flags; recipes only provide defaults.
//!
//! Usage:
//!   const recipe = Recipe.match(arch, backend_name, quant) orelse Recipe.default;
//!   // Apply recipe defaults, then overlay user CLI flags on top.

const std = @import("std");

/// A recipe is a set of proven-default parameters for a specific scenario.
/// All fields are optional, `null` means "use the CLI default / model default".
pub const Recipe = struct {
    /// Human-readable name for this recipe.
    name: []const u8 = "default",
    /// Sampling temperature (null = CLI default, typically 0 = greedy).
    temperature: ?f32 = null,
    /// Nucleus sampling threshold.
    top_p: ?f32 = null,
    /// Top-k sampling cutoff.
    top_k: ?u32 = null,
    /// Repetition penalty multiplier.
    repeat_penalty: ?f32 = null,
    /// Maximum tokens to generate.
    max_tokens: ?u32 = null,
    /// Context window size (0 or null = model default).
    ctx_size: ?u32 = null,

    /// The universal fallback, all nulls, changes nothing.
    pub const default = Recipe{};

    /// Apply recipe defaults under user-provided CLI values.
    /// User CLI flags always win; recipe fills in only where the user didn't specify.
    pub fn applyDefaults(
        self: Recipe,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        max_tokens: u32,
        ctx_size: u32,
        /// Whether each CLI arg was explicitly set by the user.
        user_set: Overrides,
    ) Applied {
        return .{
            .temperature = if (user_set.temperature) temperature else self.temperature orelse temperature,
            .top_p = if (user_set.top_p) top_p else self.top_p orelse top_p,
            .top_k = if (user_set.top_k) top_k else self.top_k orelse top_k,
            .repeat_penalty = if (user_set.repeat_penalty) repeat_penalty else self.repeat_penalty orelse repeat_penalty,
            .max_tokens = if (user_set.max_tokens) max_tokens else self.max_tokens orelse max_tokens,
            .ctx_size = if (user_set.ctx_size) ctx_size else self.ctx_size orelse ctx_size,
        };
    }

    /// Tracks which CLI args the user explicitly set (so recipes don't override them).
    pub const Overrides = struct {
        temperature: bool = false,
        top_p: bool = false,
        top_k: bool = false,
        repeat_penalty: bool = false,
        max_tokens: bool = false,
        ctx_size: bool = false,
    };

    /// Resolved parameter set after applying recipe + user overrides.
    pub const Applied = struct {
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repeat_penalty: f32,
        max_tokens: u32,
        ctx_size: u32,
    };

    /// Try to match a recipe for the given arch + backend + quantization.
    /// Returns null if no specific recipe exists (caller should use `Recipe.default`).
    pub fn match(arch: []const u8, backend: []const u8, quant: []const u8) ?Recipe {
        // Exact matches first, then progressively looser.
        for (presets) |p| {
            if (p.matches(arch, backend, quant)) return p.recipe;
        }
        return null;
    }

    // ── Preset recipes ──────────────────────────────────────────

    /// A preset binds a `Recipe` to a filter triple (architecture, backend, quantization).
    /// Each filter field is a prefix match (or empty string to match any value).
    /// `match()` iterates the `presets` array in order and returns the first hit,
    /// so more-specific presets must come before broader ones.
    const Preset = struct {
        /// Architecture name prefix to match (e.g. "gemma3"). Empty matches any arch.
        arch_prefix: []const u8,
        /// Backend name to match exactly (e.g. "Metal"). Empty matches any backend.
        backend: []const u8,
        /// Quantization prefix to match (e.g. "Q4_K"). Empty matches any quant.
        quant: []const u8,
        /// The recipe to apply when this preset matches.
        recipe: Recipe,

        /// Returns true if the given arch/backend/quant satisfies all non-empty filters.
        fn matches(self: Preset, arch: []const u8, be: []const u8, q: []const u8) bool {
            if (self.arch_prefix.len > 0 and !std.mem.startsWith(u8, arch, self.arch_prefix)) return false;
            if (self.backend.len > 0 and !std.mem.eql(u8, be, self.backend)) return false;
            if (self.quant.len > 0 and !std.mem.startsWith(u8, q, self.quant)) return false;
            return true;
        }
    };

    /// Ordered list of preset recipes. First match wins, so place more-specific
    /// entries (exact arch + backend + quant) before broader wildcards.
    const presets = [_]Preset{
        // ── Small models on Metal, responsive chat defaults ──
        .{
            .arch_prefix = "qwen3",
            .backend = "Metal",
            .quant = "Q4",
            .recipe = .{
                .name = "Qwen3.5 Q4 Metal",
                .temperature = 0.6,
                .top_p = 0.9,
                .repeat_penalty = 1.1,
                .max_tokens = 1024,
            },
        },
        // ── Qwen 2 on Metal Q4, older Qwen variant, similar tuning ──
        .{
            .arch_prefix = "qwen2",
            .backend = "Metal",
            .quant = "Q4",
            .recipe = .{
                .name = "Qwen2 Q4 Metal",
                .temperature = 0.7,
                .top_p = 0.8,
                .repeat_penalty = 1.1,
                .max_tokens = 1024,
            },
        },
        .{
            .arch_prefix = "gemma",
            .backend = "Metal",
            .quant = "Q4",
            .recipe = .{
                .name = "Gemma Q4 Metal",
                .temperature = 0.7,
                .top_p = 0.95,
                .repeat_penalty = 1.05,
                .max_tokens = 1024,
            },
        },
        // ── Large MoE on Metal, conservative to avoid OOM ──
        .{
            .arch_prefix = "gpt",
            .backend = "Metal",
            .quant = "",
            .recipe = .{
                .name = "GPT-OSS Metal",
                .temperature = 0.5,
                .top_p = 0.9,
                .max_tokens = 512,
                .ctx_size = 2048,
            },
        },
        // ── GLM-4, needs repeat penalty to avoid greedy loops ──
        .{
            .arch_prefix = "glm4",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "GLM-4 generic",
                .temperature = 0.7,
                .repeat_penalty = 1.1,
                .max_tokens = 1024,
            },
        },
        // ── DeepSeek V4 Flash, no V2 repeat penalty (greedy / official sampling) ──
        .{
            .arch_prefix = "deepseek4",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "DeepSeek V4 Flash",
                .repeat_penalty = 1.0,
            },
        },
        .{
            .arch_prefix = "deepseek_v4",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "DeepSeek V4 Flash",
                .repeat_penalty = 1.0,
            },
        },
        .{
            .arch_prefix = "dflash",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "DeepSeek V4 Flash",
                .repeat_penalty = 1.0,
            },
        },
        // ── DeepSeek V2, shares inference path with GLM-4, same repeat penalty ──
        .{
            .arch_prefix = "deepseek",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "DeepSeek V2 generic",
                .repeat_penalty = 1.1,
            },
        },
        // ── Llama 4, iRoPE + chunked attention, standard chat penalty ──
        .{
            .arch_prefix = "llama4",
            .backend = "",
            .quant = "",
            .recipe = .{
                .name = "Llama 4 generic",
                .repeat_penalty = 1.1,
            },
        },
        // ── CPU-only, larger batches, lower context ──
        .{
            .arch_prefix = "",
            .backend = "CPU",
            .quant = "",
            .recipe = .{
                .name = "CPU generic",
                .max_tokens = 256,
                .ctx_size = 2048,
            },
        },
    };
};

// ── Tests ─────────────────────────────────────────────────────────

test "recipe match exact" {
    const r = Recipe.match("qwen35", "Metal", "Q4_K") orelse Recipe.default;
    try std.testing.expectEqualStrings("Qwen3.5 Q4 Metal", r.name);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), r.temperature.?, 0.001);
}

test "recipe match glm4 gets GLM-4 recipe" {
    const r = Recipe.match("glm4", "CPU", "Q4_0") orelse Recipe.default;
    try std.testing.expectEqualStrings("GLM-4 generic", r.name);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), r.repeat_penalty.?, 0.001);
}

test "recipe match deepseek4 does not use V2 penalty" {
    const r4 = Recipe.match("deepseek4", "CPU", "Q4") orelse Recipe.default;
    try std.testing.expectEqualStrings("DeepSeek V4 Flash", r4.name);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r4.repeat_penalty.?, 0.001);
    const rv4 = Recipe.match("deepseek_v4", "Metal", "MLX") orelse Recipe.default;
    try std.testing.expectEqualStrings("DeepSeek V4 Flash", rv4.name);
    const r2 = Recipe.match("deepseek", "CPU", "Q4") orelse Recipe.default;
    try std.testing.expectEqualStrings("DeepSeek V2 generic", r2.name);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), r2.repeat_penalty.?, 0.001);
}

test "recipe match falls through to CPU generic" {
    const r = Recipe.match("unknown_cpu_arch", "CPU", "Q4_0") orelse Recipe.default;
    try std.testing.expectEqualStrings("CPU generic", r.name);
}

test "recipe no match returns null" {
    const r = Recipe.match("unknown_arch", "Vulkan", "F32");
    try std.testing.expect(r == null);
}

test "applyDefaults user override wins" {
    const recipe = Recipe{
        .name = "test",
        .temperature = 0.8,
        .top_p = 0.85,
        .top_k = 50,
        .repeat_penalty = 1.2,
        .max_tokens = 2048,
        .ctx_size = 4096,
    };
    const applied = recipe.applyDefaults(
        0.0, // CLI temperature (user-set)
        1.0, // CLI top_p
        0, // CLI top_k
        1.0, // CLI repeat_penalty
        512, // CLI max_tokens
        0, // CLI ctx_size
        .{ .temperature = true }, // user explicitly set only temperature
    );
    // User set temperature=0.0 explicitly → recipe's 0.8 does NOT override
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), applied.temperature, 0.001);
    // User did NOT set these → recipe values apply
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), applied.top_p, 0.001);
    try std.testing.expectEqual(@as(u32, 50), applied.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), applied.repeat_penalty, 0.001);
    try std.testing.expectEqual(@as(u32, 2048), applied.max_tokens);
    try std.testing.expectEqual(@as(u32, 4096), applied.ctx_size);
}

test "applyDefaults no recipe values uses CLI defaults" {
    const recipe = Recipe{ .name = "empty" };
    const applied = recipe.applyDefaults(0.7, 0.95, 40, 1.1, 256, 2048, .{});
    // No recipe overrides → CLI values pass through
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), applied.temperature, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), applied.top_p, 0.001);
    try std.testing.expectEqual(@as(u32, 40), applied.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), applied.repeat_penalty, 0.001);
    try std.testing.expectEqual(@as(u32, 256), applied.max_tokens);
    try std.testing.expectEqual(@as(u32, 2048), applied.ctx_size);
}

test "fuzz: all recipe functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // ── Recipe.default ──
            const def = Recipe.default;
            try std.testing.expect(def.temperature == null);

            // ── Recipe.Overrides ── (exercise struct with random bools)
            const overrides = Recipe.Overrides{
                .temperature = smith.valueWithHash(bool, 0),
                .top_p = smith.valueWithHash(bool, 1),
                .top_k = smith.valueWithHash(bool, 2),
                .repeat_penalty = smith.valueWithHash(bool, 3),
                .max_tokens = smith.valueWithHash(bool, 4),
                .ctx_size = smith.valueWithHash(bool, 5),
            };

            // ── Recipe.applyDefaults ── (exercises Applied return type too)
            const temp = smith.valueWithHash(f32, 10);
            const top_p = smith.valueWithHash(f32, 11);
            const top_k = smith.valueWithHash(u32, 12);
            const rep = smith.valueWithHash(f32, 13);
            const max_tok = smith.valueWithHash(u32, 14);
            const ctx = smith.valueWithHash(u32, 15);

            const recipe = Recipe{
                .temperature = 0.7,
                .top_p = 0.9,
                .top_k = 50,
                .repeat_penalty = 1.1,
                .max_tokens = 1024,
                .ctx_size = 4096,
            };
            const applied = recipe.applyDefaults(temp, top_p, top_k, rep, max_tok, ctx, overrides);

            // ── Recipe.Applied ── verify fields are populated
            _ = applied.temperature;
            _ = applied.top_p;
            _ = applied.top_k;
            _ = applied.repeat_penalty;
            _ = applied.max_tokens;
            _ = applied.ctx_size;

            // When user overrides temperature, CLI value wins
            if (overrides.temperature) {
                try std.testing.expectEqual(temp, applied.temperature);
            } else {
                try std.testing.expectApproxEqAbs(@as(f32, 0.7), applied.temperature, 0.001);
            }

            // ── Recipe.match ── with random arch/backend/quant strings
            const archs = [_][]const u8{ "qwen35", "gemma3", "gpt", "glm4", "unknown", "" };
            const backends = [_][]const u8{ "Metal", "CPU", "Vulkan", "WebGPU", "" };
            const quants = [_][]const u8{ "Q4_K", "Q4_0", "Q8_0", "F32", "" };

            const ai = smith.valueWithHash(u8, 20) % archs.len;
            const bi = smith.valueWithHash(u8, 21) % backends.len;
            const qi = smith.valueWithHash(u8, 22) % quants.len;

            const matched = Recipe.match(archs[ai], backends[bi], quants[qi]);
            if (matched) |r| {
                try std.testing.expect(r.name.len > 0);
            }

            // Also exercise match with default fallback
            const r2 = Recipe.match(archs[ai], backends[bi], quants[qi]) orelse Recipe.default;
            try std.testing.expect(r2.name.len > 0);
        }
    }.f, .{});
}
