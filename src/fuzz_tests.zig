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
            _ = json.extractField(input, "model");
            _ = json.extractField(input, "temperature");
            _ = json.extractField(input, "messages");
        }
    }.f, .{});
}

test "fuzz: JSON sampling params parser" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            _ = json.parseSampling(buf[0..len]);
        }
    }.f, .{});
}

test "fuzz: JSON escape" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const result = json.jsonEscape(std.testing.allocator, buf[0..len]) catch return;
            std.testing.allocator.free(result);
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
            g.deinit();
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
            g.deinit();
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
        }
    }.f, .{});
}

// ── N-gram Fuzzing ──────────────────────────────────────────────

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

            // Should not crash even with garbage KV data
            _ = kv_quant.kvDot(&query, &kv_buf, n, .f16);
            _ = kv_quant.kvDot(&query, &kv_buf, n, .q8_0);
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
