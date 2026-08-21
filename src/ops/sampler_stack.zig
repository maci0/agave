//! Per-request sampling interceptor stack.
//!
//! Processors mutate logits in registration order. `dispose` runs LIFO and
//! clears the stack (request end). Fixed array, no heap on the generate path.

const std = @import("std");
const math_ops = @import("math.zig");

/// Maximum interceptors on one request (bias, repeat, dry, penalties, min_p, xtc).
pub const max_processors: usize = 8;

/// Logit-mutating processor kind. Order on the stack is apply order.
pub const Kind = enum {
    bias,
    repeat,
    dry,
    penalties,
    min_p,
    xtc,
};

/// Inputs for `Stack.apply`. Slices are borrowed; no ownership.
pub const Params = struct {
    tokens: []const u32 = &.{},
    repetition_penalty: f32 = 1.0,
    dry_multiplier: f32 = 0,
    dry_allowed_length: u32 = 2,
    frequency_penalty: f32 = 0,
    presence_penalty: f32 = 0,
    min_p: f32 = 0,
    xtc_probability: f32 = 0,
    xtc_threshold: f32 = 0.1,
    logit_bias_ids: []const u32 = &.{},
    logit_bias_vals: []const f32 = &.{},
    logit_bias_count: u32 = 0,
    temperature: f32 = 0,
    rng: ?std.Random = null,
};

/// Fixed-capacity interceptor stack.
pub const Stack = struct {
    kinds: [max_processors]Kind = undefined,
    len: u8 = 0,

    /// Append a processor. Extra pushes past `max_processors` are ignored.
    pub fn push(self: *Stack, kind: Kind) void {
        if (self.len >= max_processors) return;
        self.kinds[self.len] = kind;
        self.len += 1;
    }

    /// Apply processors in registration order.
    pub fn apply(self: *const Stack, logits: []f32, params: Params) void {
        for (self.kinds[0..self.len]) |k| {
            switch (k) {
                .bias => {
                    if (params.logit_bias_count > 0)
                        math_ops.applyLogitBias(logits, params.logit_bias_ids, params.logit_bias_vals, params.logit_bias_count);
                },
                .repeat => {
                    if (params.repetition_penalty != 1.0 and params.tokens.len > 0)
                        math_ops.applyRepeatPenalty(logits, params.tokens, params.repetition_penalty);
                },
                .dry => {
                    if (params.dry_multiplier > 0 and params.tokens.len > 0)
                        math_ops.applyDry(logits, params.tokens, params.dry_multiplier, params.dry_allowed_length);
                },
                .penalties => {
                    if ((params.frequency_penalty != 0 or params.presence_penalty != 0) and params.tokens.len > 0)
                        math_ops.applyPenalties(logits, params.tokens, params.frequency_penalty, params.presence_penalty);
                },
                .min_p => {
                    if (params.temperature != 0 and params.min_p > 0)
                        math_ops.applyMinP(logits, params.min_p);
                },
                .xtc => {
                    if (params.temperature != 0 and params.xtc_probability > 0) {
                        if (params.rng) |rng|
                            math_ops.applyXtc(logits, params.xtc_probability, params.xtc_threshold, rng);
                    }
                },
            }
        }
    }

    /// LIFO teardown: drop every processor. Idempotent.
    pub fn dispose(self: *Stack) void {
        self.len = 0;
    }
};

test "empty stack apply is no-op" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    const stack = Stack{};
    stack.apply(&logits, .{});
    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 2.0), logits[1]);
}

test "bias processor boosts id" {
    var stack = Stack{};
    stack.push(.bias);
    var logits = [_]f32{ 1.0, 5.0, 2.0 };
    const ids = [_]u32{0};
    const vals = [_]f32{10.0};
    stack.apply(&logits, .{
        .logit_bias_ids = &ids,
        .logit_bias_vals = &vals,
        .logit_bias_count = 1,
    });
    try std.testing.expectEqual(@as(f32, 11.0), logits[0]);
    try std.testing.expectEqual(@as(f32, 5.0), logits[1]);
}

test "dispose clears so later apply is no-op" {
    var stack = Stack{};
    stack.push(.bias);
    stack.dispose();
    try std.testing.expectEqual(@as(u8, 0), stack.len);
    var logits = [_]f32{ 1.0, 5.0 };
    const ids = [_]u32{0};
    const vals = [_]f32{10.0};
    stack.apply(&logits, .{
        .logit_bias_ids = &ids,
        .logit_bias_vals = &vals,
        .logit_bias_count = 1,
    });
    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
}

test "push order is apply order" {
    var stack = Stack{};
    stack.push(.bias);
    stack.push(.repeat);
    try std.testing.expectEqual(Kind.bias, stack.kinds[0]);
    try std.testing.expectEqual(Kind.repeat, stack.kinds[1]);
    try std.testing.expectEqual(@as(u8, 2), stack.len);
}
