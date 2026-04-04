//! Recipes — optional proven-default configurations for model + hardware combos.
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
/// All fields are optional — `null` means "use the CLI default / model default".
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

    /// The universal fallback — all nulls, changes nothing.
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

    const Preset = struct {
        arch_prefix: []const u8, // e.g. "gemma3", "" = any
        backend: []const u8, // e.g. "Metal", "" = any
        quant: []const u8, // e.g. "Q4_K", "" = any
        recipe: Recipe,

        fn matches(self: Preset, arch: []const u8, be: []const u8, q: []const u8) bool {
            if (self.arch_prefix.len > 0 and !std.mem.startsWith(u8, arch, self.arch_prefix)) return false;
            if (self.backend.len > 0 and !std.mem.eql(u8, be, self.backend)) return false;
            if (self.quant.len > 0 and !std.mem.startsWith(u8, q, self.quant)) return false;
            return true;
        }
    };

    const presets = [_]Preset{
        // ── Small models on Metal — responsive chat defaults ──
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
        // ── Large MoE on Metal — conservative to avoid OOM ──
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
        // ── GLM-4 — needs repeat penalty to avoid greedy loops ──
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
        // ── CPU-only — larger batches, lower context ──
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
