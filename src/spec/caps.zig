//! Speculative-mode provider requirements (spatial composability).
//!
//! Each `--spec-mode` declares the providers it needs. Activation is a table
//! lookup: a missing provider stays unsatisfied with a named wait instead of
//! crashing later in `forward_tree` / `mtpForward`.
//!
//! Self-draft modes (ddtree/standard/self/ngram/...) are always satisfiable.
//! Modes that need a separate GGUF or in-weight heads wait for that provider.

const std = @import("std");

/// Speculative decoding strategy. CLI aliases (`medusa` → `mtp`) normalize
/// at parse time so call sites never branch on synonym variants.
pub const SpecMode = enum {
    none,
    standard,
    ddtree,
    self_spec,
    ngram,
    suffix,
    mtp,
    eagle,
    eagle3,
    mlp,
    lookahead,
    pflash,
    dspark,
    dflash2,
};

/// A runtime capability a spec mode may require.
pub const Provider = enum {
    /// Separate `--draft-model` GGUF (not the target).
    draft,
    /// Target model `getMtpDepth() > 0`.
    mtp,
};

/// What the process currently provides.
pub const Caps = struct {
    draft: bool = false,
    mtp: bool = false,

    pub fn has(self: Caps, p: Provider) bool {
        return switch (p) {
            .draft => self.draft,
            .mtp => self.mtp,
        };
    }
};

/// Providers required to activate `mode`. Empty slice: always active.
pub fn required(mode: SpecMode) []const Provider {
    return switch (mode) {
        .none, .standard, .ddtree, .self_spec, .ngram, .suffix, .lookahead, .dspark => &.{},
        .eagle, .eagle3, .mlp, .pflash, .dflash2 => &.{.draft},
        .mtp => &.{.mtp},
    };
}

/// First missing provider for `mode`, or null if Caps satisfy it.
pub fn unsatisfied(mode: SpecMode, have: Caps) ?Provider {
    for (required(mode)) |p| {
        if (!have.has(p)) return p;
    }
    return null;
}

/// Short name used in "waiting for {s}" messages.
pub fn providerName(p: Provider) []const u8 {
    return switch (p) {
        .draft => "draft",
        .mtp => "mtp",
    };
}

/// How to satisfy a missing provider (CLI hint).
pub fn howToProvide(p: Provider) []const u8 {
    return switch (p) {
        .draft => "provide --draft-model <path>",
        .mtp => "model has no MTP heads (n_mtp_layers == 0)",
    };
}

test "self-draft modes need no providers" {
    const modes = [_]SpecMode{ .none, .standard, .ddtree, .self_spec, .ngram, .suffix, .lookahead, .dspark };
    const empty = Caps{};
    for (modes) |m| {
        try std.testing.expectEqual(@as(usize, 0), required(m).len);
        try std.testing.expect(unsatisfied(m, empty) == null);
    }
}

test "dflash2 waits for draft" {
    try std.testing.expectEqual(Provider.draft, unsatisfied(.dflash2, Caps{}).?);
    try std.testing.expect(unsatisfied(.dflash2, Caps{ .draft = true }) == null);
}

test "eagle waits for draft" {
    try std.testing.expectEqual(Provider.draft, unsatisfied(.eagle, Caps{}).?);
    try std.testing.expectEqual(Provider.draft, unsatisfied(.eagle3, Caps{}).?);
    try std.testing.expectEqual(Provider.draft, unsatisfied(.mlp, Caps{}).?);
    try std.testing.expectEqual(Provider.draft, unsatisfied(.pflash, Caps{}).?);
    try std.testing.expect(unsatisfied(.eagle, Caps{ .draft = true }) == null);
}

test "mtp waits for mtp heads" {
    try std.testing.expectEqual(Provider.mtp, unsatisfied(.mtp, Caps{ .draft = true }).?);
    try std.testing.expect(unsatisfied(.mtp, Caps{ .mtp = true }) == null);
}

test "providerName matches wait token" {
    try std.testing.expectEqualStrings("draft", providerName(.draft));
    try std.testing.expectEqualStrings("mtp", providerName(.mtp));
}
