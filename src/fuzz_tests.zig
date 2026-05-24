//! Fuzz tests for security/correctness-critical parsers and samplers.
//!
//! Run with: zig build test --fuzz
//! These tests exercise parsers with random inputs to find crashes,
//! hangs, out-of-bounds, and undefined behavior.

const std = @import("std");
const Smith = std.testing.Smith;
const math_ops = @import("ops/math.zig");
const json = @import("server/json.zig");
const grammar_mod = @import("grammar.zig");
const kv_quant = @import("ops/kv_quant.zig");

// ── JSON Parser Fuzzing ─────────────────────────────────────────

test "fuzz: JSON field extraction" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const input = buf[0..smith.indexWithHash(buf.len + 1, 1)];
            // Invariant: extracted value, if any, must be a substring of input
            inline for (.{ "model", "temperature", "messages" }) |field| {
                if (json.extractField(input, field)) |val| {
                    try std.testing.expect(val.len <= input.len);
                }
            }
        }
    }.f, .{});
}

test "fuzz: JSON sampling params parser" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const s = json.parseSampling(buf[0..len]);
            // Invariant: all numeric fields must be finite and clamped
            try std.testing.expect(std.math.isFinite(s.temperature) and s.temperature >= 0);
            try std.testing.expect(std.math.isFinite(s.top_p) and s.top_p >= 0 and s.top_p <= 1.0);
            try std.testing.expect(s.top_k <= 1024);
        }
    }.f, .{});
}

test "fuzz: JSON escape" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const input = buf[0..len];
            const escaped = json.jsonEscape(std.testing.allocator, input) catch return;
            defer if (escaped.ptr != input.ptr) std.testing.allocator.free(escaped);
            // Roundtrip invariant: unescape(escape(x)) == x
            const unescaped = json.jsonUnescape(std.testing.allocator, escaped) catch return;
            defer if (unescaped.ptr != escaped.ptr) std.testing.allocator.free(unescaped);
            try std.testing.expectEqualSlices(u8, input, unescaped);
        }
    }.f, .{});
}

// ── Grammar Parser Fuzzing ──────────────────────────────────────

test "fuzz: GBNF grammar parser" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            var g = grammar_mod.Grammar.parse(std.testing.allocator, buf[0..len]) catch return;
            defer g.deinit();
            // Invariant: every parsed rule must have a non-empty name and elements
            for (g.rules) |rule| {
                try std.testing.expect(rule.name.len > 0);
                try std.testing.expect(rule.elements.len > 0);
            }
        }
    }.f, .{});
}

test "fuzz: JSON schema to grammar" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            var g = grammar_mod.Grammar.fromJsonSchema(std.testing.allocator, buf[0..len]) catch return;
            defer g.deinit();
            // Invariant: every parsed rule must have a non-empty name and elements
            for (g.rules) |rule| {
                try std.testing.expect(rule.name.len > 0);
                try std.testing.expect(rule.elements.len > 0);
            }
        }
    }.f, .{});
}

// ── Sampler Fuzzing ─────────────────────────────────────────────

test "fuzz: sampleToken no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [64]f32 = undefined;
            for (&logits) |*v| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(@as(u32, @bitCast(v.*))))));
            const temp: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 2))) / 100.0;
            const top_k: u32 = smith.valueWithHash(u8, 3);
            const top_p: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 4))) / 255.0;
            var prng = std.Random.Xoshiro256.init(smith.valueWithHash(u64, 5));
            const result = math_ops.sampleToken(&logits, temp, top_k, top_p, prng.random());
            try std.testing.expect(result < 64);
            // Sampling must not corrupt logits to NaN
            for (logits) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: applyMinP no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const min_p: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 100))) / 255.0;
            math_ops.applyMinP(&logits, min_p);
            // Invariant: max logit always survives — at least one token must remain
            var n_alive: u32 = 0;
            for (logits) |v| {
                if (v != -std.math.inf(f32)) n_alive += 1;
            }
            try std.testing.expect(n_alive >= 1);
        }
    }.f, .{});
}

test "fuzz: applyXtc no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const prob: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 100))) / 255.0;
            const thresh: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 101))) / 255.0;
            var prng = std.Random.Xoshiro256.init(smith.valueWithHash(u64, 102));
            math_ops.applyXtc(&logits, prob, thresh, prng.random());
            // Invariant: XTC always keeps at least one token
            var n_alive: u32 = 0;
            for (logits) |v| {
                if (v != -std.math.inf(f32)) n_alive += 1;
            }
            try std.testing.expect(n_alive >= 1);
        }
    }.f, .{});
}

test "fuzz: applyDry no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [16]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            var history: [32]u32 = undefined;
            for (&history, 0..) |*v, i| v.* = smith.valueWithHash(u8, @truncate(i + 100));
            const len = smith.indexWithHash(history.len + 1, 200);
            const mult: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 201))) / 50.0;
            const allowed: u32 = smith.valueWithHash(u4, 202);
            math_ops.applyDry(&logits, history[0..len], mult, allowed);
            // Invariant: DRY only subtracts penalties — no NaN/Inf introduced
            for (logits) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: sampleMirostat no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            var mu: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 100))) / 10.0;
            const tau: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 101))) / 10.0 + 0.1;
            const eta: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 102))) / 255.0 + 0.01;
            const temp: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 103))) / 100.0 + 0.1;
            var prng = std.Random.Xoshiro256.init(smith.valueWithHash(u64, 104));
            const result = math_ops.sampleMirostat(&logits, tau, eta, &mu, temp, prng.random());
            try std.testing.expect(result < 32);
            // Invariant: mu must remain finite after update
            try std.testing.expect(std.math.isFinite(mu));
        }
    }.f, .{});
}

// ── KV Cache Quantization Fuzzing ────────────────────────────────

test "fuzz: kvStore + kvDot roundtrip" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n: usize = 32;
            var src: [n]f32 = undefined;
            for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;

            const kv_types = [_]kv_quant.KvQuantType{ .f16, .q8_0, .fp8_e4m3 };
            const kv_type = kv_types[smith.indexWithHash(kv_types.len, 100)];

            var kv_buf: [256]u8 = undefined;
            const needed = kv_quant.kvSliceBytes(kv_type, n);
            if (needed > kv_buf.len) return;
            kv_quant.kvStore(&kv_buf, &src, n, kv_type);

            var query: [n]f32 = undefined;
            for (&query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            const dot = kv_quant.kvDot(&query, &kv_buf, n, kv_type);
            try std.testing.expect(std.math.isFinite(dot));
            // Reference f32 dot product for approximate correctness
            var ref: f32 = 0;
            for (0..n) |i| ref += query[i] * src[i];
            // Quantization error scales with magnitude; allow 20% relative or 1.0 absolute
            const err = @abs(dot - ref);
            const threshold = @max(1.0, @abs(ref) * 0.2);
            try std.testing.expect(err < threshold);
        }
    }.f, .{});
}

test "fuzz: kvDot with random bytes" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n: usize = 16;
            var kv_buf: [128]u8 = undefined;
            smith.bytesWithHash(&kv_buf, 0);

            var query: [n]f32 = undefined;
            for (&query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;

            // Invariant: result must not be NaN (garbage data may produce large values but not NaN)
            const dot_f16 = kv_quant.kvDot(&query, &kv_buf, n, .f16);
            try std.testing.expect(!std.math.isNan(dot_f16));
            const dot_q8 = kv_quant.kvDot(&query, &kv_buf, n, .q8_0);
            try std.testing.expect(!std.math.isNan(dot_q8));
        }
    }.f, .{});
}

// ── N-gram Fuzzing ──────────────────────────────────────────────

test "fuzz: ngram propose no crash" {
    const ngram_mod = @import("spec/ngram.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var state = ngram_mod.NgramState{};
            const n_tokens = smith.valueWithHash(u8, 0);
            for (0..n_tokens) |i| {
                state.push(smith.valueWithHash(u16, @truncate(i + 1)));
            }
            var draft: [16]u32 = undefined;
            const n = state.propose(smith.valueWithHash(u4, 200), &draft);
            try std.testing.expect(n <= 16);
        }
    }.f, .{});
}
