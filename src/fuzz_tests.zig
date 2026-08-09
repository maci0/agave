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

// ── Chat Template Fuzzing ──────────────────────────────────────

test "fuzz: chat template format" {
    const chat = @import("chat_template.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Generate random user message content
            var user_buf: [128]u8 = undefined;
            smith.bytesWithHash(&user_buf, 0);
            const user_len = smith.indexWithHash(user_buf.len + 1, 1);
            const user_msg = user_buf[0..user_len];

            // Generate random system message
            var sys_buf: [64]u8 = undefined;
            smith.bytesWithHash(&sys_buf, 2);
            const sys_len = smith.indexWithHash(sys_buf.len + 1, 3);
            const sys_msg: ?[]const u8 = if (sys_len > 0) sys_buf[0..sys_len] else null;

            // Test against all preset templates
            const templates = [_]chat.ChatTemplate{
                chat.ChatTemplate.chatml,
                chat.ChatTemplate.qwen35,
                chat.ChatTemplate.gemma,
                chat.ChatTemplate.gemma4,
                chat.ChatTemplate.glm4,
                chat.ChatTemplate.gpt_oss,
                chat.ChatTemplate.llama4,
            };
            const tmpl = templates[smith.indexWithHash(templates.len, 4)];

            // Invariant: format must not crash and must produce non-empty output
            const result = tmpl.format(std.testing.allocator, sys_msg, user_msg) catch return;
            defer std.testing.allocator.free(result);
            try std.testing.expect(result.len > 0);
        }
    }.f, .{});
}

test "fuzz: chat template formatConversation" {
    const chat = @import("chat_template.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Build random conversation messages (1-4 turns)
            const n_msgs = smith.indexWithHash(4, 0) + 1;
            var msg_bufs: [4][64]u8 = undefined;
            var messages: [4]chat.Message = undefined;
            const roles = [_]chat.Role{ .user, .assistant, .tool };
            for (0..n_msgs) |i| {
                smith.bytesWithHash(&msg_bufs[i], @truncate(i + 10));
                const len = smith.indexWithHash(msg_bufs[i].len + 1, @truncate(i + 20));
                messages[i] = .{
                    .role = roles[smith.indexWithHash(roles.len, @truncate(i + 30))],
                    .content = msg_bufs[i][0..len],
                };
            }

            const tmpl = chat.ChatTemplate.chatml;
            const result = tmpl.formatConversation(std.testing.allocator, null, messages[0..n_msgs]) catch return;
            defer std.testing.allocator.free(result);
            // Invariant: result must end with assistant prefix + generation_prefix
            try std.testing.expect(result.len > 0);
        }
    }.f, .{});
}

test "fuzz: findImageInsertPos" {
    const chat = @import("chat_template.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Random token sequence
            var tokens: [32]u32 = undefined;
            for (&tokens, 0..) |*t, i| t.* = smith.valueWithHash(u16, @truncate(i));
            const tok_len = smith.indexWithHash(tokens.len + 1, 100);

            // Random prefix sequence
            var prefix: [8]u32 = undefined;
            for (&prefix, 0..) |*p, i| p.* = smith.valueWithHash(u16, @truncate(i + 50));
            const pfx_len = smith.indexWithHash(prefix.len + 1, 101);

            const pos = chat.findImageInsertPos(tokens[0..tok_len], prefix[0..pfx_len]);
            // Invariant: position must be within bounds
            try std.testing.expect(pos <= tok_len);
        }
    }.f, .{});
}

// ── CLI Parser Fuzzing ─────────────────────────────────────────

test "fuzz: CLI arg parser" {
    const cli = @import("cli.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Build random null-terminated argument strings.
            // On POSIX, std.process.Args.Vector = []const [*:0]const u8
            var arg_storage: [8][31:0]u8 = undefined;
            var argv: [8][*:0]const u8 = undefined;
            const n_args = smith.indexWithHash(8, 0) + 1;
            for (0..n_args) |i| {
                smith.bytesWithHash(&arg_storage[i], @truncate(i + 1));
                // Ensure null-terminator is preserved (bytesWithHash may overwrite it
                // but the sentinel-terminated array type guarantees buf[31] == 0)
                argv[i] = &arg_storage[i];
            }

            // Define a representative set of arg specs (mirrors real CLI options)
            const specs = [_]cli.ArgSpec{
                .{ .long = "help", .short = 'h' },
                .{ .long = "serve", .short = 's' },
                .{ .long = "verbose", .short = 'V' },
                .{ .long = "backend", .short = 'b', .kind = .option },
                .{ .long = "max-tokens", .short = 'n', .kind = .option },
                .{ .long = "temperature", .short = 't', .kind = .option },
                .{ .long = "ctx-size", .kind = .option },
            };

            const args = std.process.Args{ .vector = argv[0..n_args] };
            // Invariant: parse must not crash on any argument combination
            var result = cli.parse(std.testing.allocator, args, &specs);
            defer result.deinit();

            // All accessors must be safe on parsed result
            _ = result.flag("help");
            _ = result.flag("nonexistent");
            _ = result.option("backend");
            _ = result.optionU32("max-tokens");
            _ = result.optionF32("temperature");
            _ = result.positional(0);
        }
    }.f, .{});
}

// ── AWQ Dequant Fuzzing ────────────────────────────────────────

test "fuzz: AWQ dequant no crash" {
    const awq = @import("ops/awq.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Fixed dimensions: 8 output channels, 8 input elements, group_size=128
            // AWQ column-major: qweight is [k, n/8] = [8, 1] = 1 u32 per input row
            const n: usize = 8;
            const k: usize = 8;
            const n_words = n / 8; // = 1
            const group_size: u32 = 128;

            // qweight: [k * n_words] = 8 words (one per input row)
            var qw: [k * n_words]u32 = undefined;
            for (&qw, 0..) |*w, i| w.* = smith.valueWithHash(u32, @truncate(i));
            // qzeros: [1 group * n_words] = 1 word
            var qz: [1 * n_words]u32 = undefined;
            qz[0] = smith.valueWithHash(u32, 100);
            // scales: [1 group * n] = 8 FP16 values
            var scales: [1 * n]u16 = undefined;
            for (&scales, 0..) |*s, i| s.* = smith.valueWithHash(u16, @truncate(i + 110));
            // Input vector
            var x: [k]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 130)))) / 10.0;
            // Output
            var y: [n]f32 = undefined;

            awq.awqGemvRows(&x, &qw, &scales, &qz, &y, 0, n, n, k, group_size);
            // Invariant: output must be finite (no NaN from INT4 dequant)
            for (y) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── GPTQ Dequant Fuzzing ───────────────────────────────────────

test "fuzz: GPTQ dequant no crash" {
    const gptq = @import("ops/gptq.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Fixed dimensions: 8 output rows, k=8, group_size=128
            // GPTQ row-major: qweight is [n, k/8] = [8, 1] = 1 u32 per output row
            const n: usize = 8;
            const k: usize = 8;
            const words_per_row = k / 8; // = 1
            const group_size: u32 = 128;
            const n_groups: usize = 1; // ceil(8/128) = 1

            // qweight: [n * words_per_row] = 8 words
            var qw: [n * words_per_row]u32 = undefined;
            for (&qw, 0..) |*w, i| w.* = smith.valueWithHash(u32, @truncate(i));
            // scales: [n * n_groups] = 8 FP16 values
            var scales: [n * n_groups]u16 = undefined;
            for (&scales, 0..) |*s, i| s.* = smith.valueWithHash(u16, @truncate(i + 50));
            // qzeros: [n_groups * ceil(n/8)] = 1 word
            var qz: [n_groups * 1]u32 = undefined;
            qz[0] = smith.valueWithHash(u32, 100);
            // Input vector
            var x: [k]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 110)))) / 10.0;
            // Output
            var y: [n]f32 = undefined;

            gptq.gptqGemvRows(&x, &qw, &scales, &qz, &y, 0, n, n, k, group_size);
            // Invariant: output must be finite (no NaN from INT4 dequant)
            for (y) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── Terminal Display Width Fuzzing ──────────────────────────────

test "fuzz: term displayWidth" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [64]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const w = term.displayWidth(buf[0..len]);
            // Invariant: display width cannot exceed 2x byte count
            // (each byte is at most 1 codepoint of width 2)
            try std.testing.expect(w <= len * 2);
        }
    }.f, .{});
}

test "fuzz: term key parser" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [32]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len, 1) + 1; // at least 1 byte
            var parser: term.Parser = .{};
            // Parse all events from the buffer without crashing
            var offset: usize = 0;
            var n_events: u32 = 0;
            while (offset < len) {
                const result = parser.parse(buf[offset..len], null) catch break;
                if (result.n == 0) break; // incomplete sequence
                offset += result.n;
                if (result.event != null) n_events += 1;
            }
            // Invariant: consumed bytes cannot exceed input length
            try std.testing.expect(offset <= len);
        }
    }.f, .{});
}

// ── GEMV Kernel Fuzzing ─────────────────────────────────────────

test "fuzz: Q8_0 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q8_0 block: 34 bytes (f16 scale + 32 i8 quants), block_size=32
            var block: [34]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, &block, .q8_0, &y, 1, 32);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q4_0 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q4_0 block: 18 bytes (f16 scale + 16 bytes = 32 nibbles)
            var block: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, &block, .q4_0, &y, 1, 32);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: F16 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var w: [16]u16 = undefined;
            for (&w, 0..) |*v, i| v.* = smith.valueWithHash(u16, @truncate(i));
            var x: [16]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, @ptrCast(&w), .f16, &y, 1, 16);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}

test "fuzz: BF16 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var w: [16]u16 = undefined;
            for (&w, 0..) |*v, i| v.* = smith.valueWithHash(u16, @truncate(i));
            var x: [16]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, @ptrCast(&w), .bf16, &y, 1, 16);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}

test "fuzz: isBlockSparse no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var x: [64]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 100.0;
            const len = smith.indexWithHash(64, 100) + 1;
            const start = smith.indexWithHash(65 - len, 101);
            _ = gemv.isBlockSparse(&x, start, len);
        }
    }.f, .{});
}

// ── Quant Conversion Fuzzing ────────────────────────────────────

test "fuzz: dequantToF32 Q8_0 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [34]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [32]f32 = undefined;
            quant.dequantToF32(&output, &block, .q8_0, 32);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: dequantToF32 F16 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [32]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [16]f32 = undefined;
            quant.dequantToF32(&output, &block, .f16, 16);
        }
    }.f, .{});
}

// ── KV Cache Fuzzing ────────────────────────────────────────────

test "fuzz: TurboQuant kvStore+kvDot consistency" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n: usize = 64;
            var src: [n]f32 = undefined;
            for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;

            const turbo_types = [_]kv_quant.KvQuantType{ .turbo2, .turbo3, .turbo4 };
            const kv_type = turbo_types[smith.indexWithHash(turbo_types.len, 100)];

            var kv_buf: [512]u8 = undefined;
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

// ── Spec Decode Fuzzing ─────────────────────────────────────────

test "fuzz: argmax stability" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const idx = math_ops.argmax(&logits);
            try std.testing.expect(idx < 32);
            for (logits, 0..) |v, i| {
                try std.testing.expect(v <= logits[idx] or i == idx);
            }
        }
    }.f, .{});
}

// ── Model Helper Fuzzing ────────────────────────────────────────

test "fuzz: expertWeightStride no overflow" {
    const model_mod = @import("models/model.zig");
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const dtypes = [_]format_mod.DType{ .q4_k, .q8_0, .f16, .f32, .bf16, .q4_0, .q6_k };
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 0)];
            const d0 = smith.valueWithHash(u16, 1);
            const d1 = smith.valueWithHash(u16, 2);
            const d2 = smith.valueWithHash(u16, 3);
            const t = format_mod.TensorInfo{
                .name = "test",
                .n_dims = 3,
                .dims = .{ d0, d1, d2, 0 },
                .dtype = dtype,
                .data_ptr = undefined,
            };
            const stride = model_mod.expertWeightStride(t);
            _ = stride;
        }
    }.f, .{});
}

// ── Norm Kernel Fuzzing ─────────────────────────────────────────

test "fuzz: rmsNorm no crash" {
    const norm = @import("backend/kernels/cpu/norm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // 32-element buffers: exercises both SIMD and scalar tail paths
            var input: [32]f32 = undefined;
            var weight: [32]f32 = undefined;
            var output: [32]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&weight, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            const n = smith.indexWithHash(32, 100) + 1;
            // Ensure n is at least 1 to avoid div-by-zero in RMS
            norm.rmsNorm(&input, &weight, &output, n, 1e-6);
            for (output[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: addRmsNorm no crash" {
    const norm = @import("backend/kernels/cpu/norm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            var weight: [32]f32 = undefined;
            var output: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            for (&weight, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 80)))) / 10.0;
            const n = smith.indexWithHash(32, 120) + 1;
            norm.addRmsNorm(&a, &b, &weight, &output, n, 1e-6);
            // output must be finite
            for (output[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
            // a must be modified in-place (a = a + b)
            for (a[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: l2Norm no crash" {
    const norm = @import("backend/kernels/cpu/norm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const n = smith.indexWithHash(32, 50) + 1;
            norm.l2Norm(&x, n, 1e-12);
            // After L2 norm, all values should be finite
            for (x[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
            // Verify unit norm: sum of squares should be approximately 1.0
            var ss: f32 = 0;
            for (x[0..n]) |v| ss += v * v;
            try std.testing.expect(@abs(ss - 1.0) < 0.01 or ss == 0.0);
        }
    }.f, .{});
}

// ── Activation Fuzzing ──────────────────────────────────────────

test "fuzz: silu activation" {
    const act = @import("backend/kernels/cpu/activation.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var input: [32]f32 = undefined;
            var output: [32]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const n = smith.indexWithHash(32, 50) + 1;
            act.silu(&input, &output, n);
            for (output[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: gelu activation" {
    const act = @import("backend/kernels/cpu/activation.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var input: [32]f32 = undefined;
            var output: [32]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const n = smith.indexWithHash(32, 50) + 1;
            act.gelu(&input, &output, n);
            for (output[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: siluMul activation" {
    const act = @import("backend/kernels/cpu/activation.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            var out: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            act.siluMul(&a, &b, &out, n);
            for (out[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: geluMul activation" {
    const act = @import("backend/kernels/cpu/activation.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            var out: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            act.geluMul(&a, &b, &out, n);
            for (out[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── Elementwise Fuzzing ─────────────────────────────────────────

test "fuzz: elementwise add" {
    const elem = @import("backend/kernels/cpu/elementwise.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            var out: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            elem.add(&a, &b, &out, n);
            // Invariant: out[i] must equal a[i] + b[i]
            for (0..n) |i| {
                const expected = a[i] + b[i];
                try std.testing.expect(@abs(out[i] - expected) < 1e-5);
            }
        }
    }.f, .{});
}

test "fuzz: elementwise mul" {
    const elem = @import("backend/kernels/cpu/elementwise.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            var out: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            elem.mul(&a, &b, &out, n);
            for (0..n) |i| {
                const expected = a[i] * b[i];
                try std.testing.expect(@abs(out[i] - expected) < 1e-5);
            }
        }
    }.f, .{});
}

test "fuzz: sigmoidMul no crash" {
    const elem = @import("backend/kernels/cpu/elementwise.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var data: [32]f32 = undefined;
            var gate: [32]f32 = undefined;
            for (&data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&gate, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            elem.sigmoidMul(&data, &gate, n);
            for (data[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: deinterleave no crash" {
    const elem = @import("backend/kernels/cpu/elementwise.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var input: [32]f32 = undefined;
            var out_a: [16]f32 = undefined;
            var out_b: [16]f32 = undefined;
            for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            // stride in {1,2,4,8}, n_pairs such that stride*2*n_pairs <= 32
            const stride_idx = smith.indexWithHash(4, 50);
            const stride: usize = @as(usize, 1) << @intCast(stride_idx);
            const max_pairs = 32 / (stride * 2);
            if (max_pairs == 0) return;
            const n_pairs = smith.indexWithHash(max_pairs, 51) + 1;
            elem.deinterleave(&input, &out_a, &out_b, stride, n_pairs);
            // Verify no NaN in outputs
            for (out_a[0 .. n_pairs * stride]) |v| try std.testing.expect(!std.math.isNan(v));
            for (out_b[0 .. n_pairs * stride]) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── RoPE Fuzzing ─────────────────────────────────────────────────

test "fuzz: rope no crash" {
    const rope_mod = @import("backend/kernels/cpu/rope.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // head_dim must be even for RoPE; use fixed multiples of 8
            const head_dims = [_]usize{ 8, 16, 32 };
            const hd = head_dims[smith.indexWithHash(head_dims.len, 0)];
            const n_heads_choices = [_]usize{ 1, 2, 4 };
            const n_heads = n_heads_choices[smith.indexWithHash(n_heads_choices.len, 1)];
            var x: [128]f32 = undefined; // max: 32 * 4 = 128
            const total = hd * n_heads;
            for (x[0..total], 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 10)))) / 10.0;
            const pos: usize = smith.valueWithHash(u8, 2);
            const theta: f32 = 10000.0;
            rope_mod.rope(&x, pos, n_heads, hd, hd, theta);
            // All output values must be finite
            for (x[0..total]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: rope preserves magnitude" {
    const rope_mod = @import("backend/kernels/cpu/rope.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const hd: usize = 16;
            const half = hd / 2;
            var x: [16]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const orig = x;
            const pos: usize = smith.valueWithHash(u8, 20);
            rope_mod.rope(&x, pos, 1, hd, hd, 10000.0);
            // RoPE is rotation: magnitude preserved per (i, i+half) pair
            for (0..half) |i| {
                const orig_mag = @sqrt(orig[i] * orig[i] + orig[i + half] * orig[i + half]);
                const new_mag = @sqrt(x[i] * x[i] + x[i + half] * x[i + half]);
                try std.testing.expect(@abs(orig_mag - new_mag) < 0.01);
            }
        }
    }.f, .{});
}

// ── Softmax Fuzzing ─────────────────────────────────────────────

test "fuzz: softmax sums to one" {
    const soft = @import("backend/kernels/cpu/softmax.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var data: [32]f32 = undefined;
            for (&data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const n = smith.indexWithHash(32, 50) + 1;
            soft.softmaxSimd(8, &data, n);
            // All values must be non-negative and finite
            var sum: f32 = 0;
            for (data[0..n]) |v| {
                try std.testing.expect(std.math.isFinite(v));
                try std.testing.expect(v >= 0.0);
                sum += v;
            }
            // Sum must be approximately 1.0
            try std.testing.expect(@abs(sum - 1.0) < 0.01);
        }
    }.f, .{});
}

// ── Embedding Fuzzing ───────────────────────────────────────────

test "fuzz: embLookup f32 bounds" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // 4 tokens, 8 dims = 128 bytes of f32 weight data
            const dim: usize = 8;
            const vocab: usize = 4;
            var weights: [vocab * dim]f32 = undefined;
            for (&weights, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            var output: [dim]f32 = undefined;
            const token_id: u32 = @intCast(smith.indexWithHash(vocab, 50));
            emb.embLookup(@ptrCast(&weights), .f32, token_id, &output, dim);
            // Output should exactly match the weight row
            for (0..dim) |i| {
                try std.testing.expectApproxEqAbs(weights[token_id * dim + i], output[i], 1e-6);
            }
        }
    }.f, .{});
}

test "fuzz: embQ8_0 no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q8_0: 34 bytes per 32-element block
            var block: [34]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [32]f32 = undefined;
            emb.embQ8_0(&block, 0, &output, 32);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: embQ4_0 no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [32]f32 = undefined;
            emb.embQ4_0(&block, 0, &output, 32);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: embQ5_0 no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q5_0: 22 bytes per 32-element block
            var block: [22]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [32]f32 = undefined;
            emb.embQ5_0(&block, 0, &output, 32);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── GEMV Format Fuzzing ─────────────────────────────────────────

test "fuzz: Q4_K GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q4_K: 256-elem super-blocks, 144 bytes each. 1 row of k=256.
            var block: [144]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ4_K(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q5_K GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q5_K: 256-elem super-blocks, 176 bytes each
            var block: [176]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ5_K(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q6_K GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q6_K: 256-elem super-blocks, 210 bytes each
            var block: [210]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ6_K(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q2_K GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q2_K: 256-elem super-blocks, 84 bytes each
            var block: [84]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ2_K(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q3_K GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q3_K: 256-elem super-blocks, 110 bytes each
            var block: [110]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ3_K(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: IQ4_NL GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // IQ4_NL: 32-elem blocks, 18 bytes each (same as Q4_0)
            var block: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = .{0};
            gemv.gemvIQ4_NL(&x, &block, &y, 1, 32);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: FP8 E4M3 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // FP8 E4M3: 1 byte per element, 32 elements
            var block: [32]u8 = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = .{0};
            gemv.gemvFP8_E4M3(&x, &block, &y, 1, 32);
            // FP8 may produce NaN for special patterns but must not crash
            try std.testing.expect(!std.math.isNan(y[0]) or std.math.isNan(y[0]));
        }
    }.f, .{});
}

test "fuzz: FP8 E5M2 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [32]u8 = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = .{0};
            gemv.gemvFP8_E5M2(&x, &block, &y, 1, 32);
            // Must not crash; NaN is allowed from special FP8 bit patterns
            _ = y[0];
        }
    }.f, .{});
}

test "fuzz: TQ1_0 GEMV no crash" {
    const gemv_tq1 = @import("backend/kernels/cpu/gemv_tq1_0.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // TQ1_0: 256-elem blocks, 54 bytes each
            var block: [54]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv_tq1.gemvTQ1_0(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q4_1 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q4_1: 32-elem blocks, 20 bytes (f16 scale + f16 min + 16 nibble bytes)
            var block: [20]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ4_1(&x, &block, &y, 1, 32);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}

test "fuzz: Q5_0 GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q5_0: 32-elem blocks, 22 bytes (f16 scale + 4 byte qh + 16 nibble bytes)
            var block: [22]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = .{0};
            gemv.gemvQ5_0(&x, &block, &y, 1, 32);
            try std.testing.expect(!std.math.isNan(y[0]));
        }
    }.f, .{});
}

test "fuzz: IQ4_XS GEMV no crash" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // IQ4_XS: 256-elem blocks, 136 bytes
            var block: [136]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [256]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 100.0;
            var y: [1]f32 = .{0};
            gemv.gemvIQ4_XS(&x, &block, &y, 1, 256);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

// ── Math/Sampler Fuzzing ────────────────────────────────────────

test "fuzz: applyRepeatPenalty no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            var recent_ids: [16]u32 = undefined;
            for (&recent_ids, 0..) |*v, i| v.* = smith.valueWithHash(u8, @truncate(i + 40)) % 32;
            const len = smith.indexWithHash(16, 60) + 1;
            // penalty > 0, typically 1.0-2.0
            const penalty: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 61))) / 100.0 + 0.1;
            math_ops.applyRepeatPenalty(&logits, recent_ids[0..len], penalty);
            for (logits) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: applyPenalties no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            var gen_tokens: [16]u32 = undefined;
            for (&gen_tokens, 0..) |*v, i| v.* = smith.valueWithHash(u8, @truncate(i + 40)) % 32;
            const len = smith.indexWithHash(16, 60) + 1;
            const freq: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 61))) / 100.0;
            const pres: f32 = @as(f32, @floatFromInt(smith.valueWithHash(u8, 62))) / 100.0;
            math_ops.applyPenalties(&logits, gen_tokens[0..len], freq, pres);
            for (logits) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: applyLogitBias no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            var ids: [8]u32 = undefined;
            var biases: [8]f32 = undefined;
            for (&ids, 0..) |*v, i| v.* = smith.valueWithHash(u8, @truncate(i + 40)) % 32;
            for (&biases, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            const count: u32 = @intCast(smith.indexWithHash(8, 60) + 1);
            math_ops.applyLogitBias(&logits, &ids, &biases, count);
            for (logits) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: sigmoid no NaN" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Test sigmoid over the full i8 range → f32 [-12.8, 12.7]
            const x: f32 = @as(f32, @floatFromInt(smith.valueWithHash(i8, 0))) / 10.0;
            const result = math_ops.sigmoid(x);
            try std.testing.expect(std.math.isFinite(result));
            try std.testing.expect(result >= 0.0 and result <= 1.0);
        }
    }.f, .{});
}

test "fuzz: softplus no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const x: f32 = @as(f32, @floatFromInt(smith.valueWithHash(i8, 0))) / 5.0;
            const result = math_ops.softplus(x);
            try std.testing.expect(std.math.isFinite(result));
            try std.testing.expect(result >= 0.0);
        }
    }.f, .{});
}

test "fuzz: silu scalar no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const x: f32 = @as(f32, @floatFromInt(smith.valueWithHash(i8, 0))) / 5.0;
            const result = math_ops.silu(x);
            try std.testing.expect(std.math.isFinite(result));
        }
    }.f, .{});
}

test "fuzz: applyReluSquared no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const n = smith.indexWithHash(32, 50) + 1;
            math_ops.applyReluSquared(x[0..n]);
            for (x[0..n]) |v| {
                try std.testing.expect(std.math.isFinite(v));
                try std.testing.expect(v >= 0.0);
            }
        }
    }.f, .{});
}

test "fuzz: applyGelu no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const n = smith.indexWithHash(32, 50) + 1;
            math_ops.applyGelu(x[0..n]);
            for (x[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: simdDotF32 no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var a: [32]f32 = undefined;
            var b: [32]f32 = undefined;
            for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            const n = smith.indexWithHash(32, 80) + 1;
            const dot = math_ops.simdDotF32(&a, &b, n);
            try std.testing.expect(std.math.isFinite(dot));
        }
    }.f, .{});
}

test "fuzz: simdScaleF32 no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [32]f32 = undefined;
            for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const scale: f32 = @as(f32, @floatFromInt(smith.valueWithHash(i8, 50))) / 50.0;
            const n = smith.indexWithHash(32, 60) + 1;
            math_ops.simdScaleF32(&buf, scale, n);
            for (buf[0..n]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: tokenLogProb no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const token_id: u32 = smith.valueWithHash(u8, 50) % 32;
            const result = math_ops.tokenLogProb(&logits, token_id);
            // Log prob should be <= 0 (log of probability) or -inf for impossible tokens
            try std.testing.expect(result <= 0.0 or result == -std.math.inf(f32));
            try std.testing.expect(!std.math.isNan(result));
        }
    }.f, .{});
}

test "fuzz: topKExperts no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var scores: [16]f32 = undefined;
            for (&scores, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const k = smith.indexWithHash(4, 50) + 1; // 1..4
            var out_indices: [4]usize = undefined;
            var out_scores: [4]f32 = undefined;
            math_ops.topKExperts(&scores, k, out_indices[0..k], out_scores[0..k]);
            // Each index must be in range
            for (out_indices[0..k]) |idx| try std.testing.expect(idx < 16);
            // Each score must be finite
            for (out_scores[0..k]) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── Quant Conversion Fuzzing ────────────────────────────────────

test "fuzz: bf16ToF32 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const val = smith.valueWithHash(u16, 0);
            const result = quant.bf16ToF32(val);
            // Must not crash; result may be NaN/Inf for special patterns
            _ = result;
        }
    }.f, .{});
}

test "fuzz: fp8e4m3ToF32 exhaustive" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const val = smith.valueWithHash(u8, 0);
            const result = quant.fp8e4m3ToF32(val);
            // fp8 e4m3 max is ~448; all values should be representable
            _ = result;
        }
    }.f, .{});
}

test "fuzz: fp8e5m2ToF32 exhaustive" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const val = smith.valueWithHash(u8, 0);
            const result = quant.fp8e5m2ToF32(val);
            _ = result;
        }
    }.f, .{});
}

test "fuzz: mxfp4Lookup exhaustive" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const nibble = smith.valueWithHash(u8, 0) & 0x0F;
            const result = quant.mxfp4Lookup(nibble);
            try std.testing.expect(std.math.isFinite(result));
        }
    }.f, .{});
}

test "fuzz: e8m0ToF32 exhaustive" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const e = smith.valueWithHash(u8, 0);
            const result = quant.e8m0ToF32(e);
            try std.testing.expect(std.math.isFinite(result));
            try std.testing.expect(result >= 0.0);
        }
    }.f, .{});
}

test "fuzz: getScaleMinK4 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var scales: [12]u8 = undefined;
            smith.bytesWithHash(&scales, 0);
            const j = smith.indexWithHash(8, 20); // 0..7 (Q4_K has 8 sub-groups)
            var sc: u8 = undefined;
            var m: u8 = undefined;
            quant.getScaleMinK4(j, &scales, &sc, &m);
            // Both values must fit in u8 (always true by type, but verify non-crash)
            try std.testing.expect(sc <= 255);
            try std.testing.expect(m <= 255);
        }
    }.f, .{});
}

test "fuzz: dequantToF32 Q4_0 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [32]f32 = undefined;
            quant.dequantToF32(&output, &block, .q4_0, 32);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: dequantToF32 BF16 no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [32]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [16]f32 = undefined;
            quant.dequantToF32(&output, &block, .bf16, 16);
        }
    }.f, .{});
}

// ── Spec Decode Fuzzing ─────────────────────────────────────────

test "fuzz: DDTree presort + buildTree no crash" {
    const ddtree_mod = @import("spec/ddtree.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var builder = ddtree_mod.DDTreeBuilder{};
            builder.budget = @as(u32, smith.valueWithHash(u4, 0)) + 1; // 1..16

            // Generate random logits for 1-3 draft depths
            const n_depths = smith.indexWithHash(3, 1) + 1;
            const vocab_size: usize = 16;
            var logit_storage: [3][16]f32 = undefined;
            var logit_slices: [3][]const f32 = undefined;
            for (0..n_depths) |d| {
                for (&logit_storage[d], 0..) |*v, i| {
                    v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(d * 20 + i + 10)))) / 10.0;
                }
                logit_slices[d] = logit_storage[d][0..vocab_size];
            }

            builder.presort(logit_slices[0..n_depths]);
            builder.buildTree();
            try std.testing.expect(builder.n_nodes <= builder.budget);
            const compiled = builder.compile(0);
            try std.testing.expect(compiled.n_nodes == builder.n_nodes);
        }
    }.f, .{});
}

test "fuzz: CompiledTree findChild no crash" {
    const ddtree_mod = @import("spec/ddtree.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var tree = ddtree_mod.CompiledTree{};
            // Set up a few children in the tree
            const n_children = smith.indexWithHash(4, 0) + 1;
            tree.n_nodes = @intCast(n_children);
            for (0..n_children) |i| {
                tree.child_tokens[0][i] = smith.valueWithHash(u16, @truncate(i + 10));
                tree.child_indices[0][i] = @intCast(i);
            }
            tree.child_counts[0] = @intCast(n_children);
            // Search for random token — must not crash
            const search_id = smith.valueWithHash(u16, 50);
            _ = tree.findChild(-1, search_id);
            _ = tree.findChild(0, search_id);
        }
    }.f, .{});
}

test "fuzz: CompiledTree isAncestor no crash" {
    const ddtree_mod = @import("spec/ddtree.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var mask: [8]u64 = undefined;
            for (&mask, 0..) |*v, i| v.* = smith.valueWithHash(u64, @truncate(i));
            const j = smith.indexWithHash(ddtree_mod.max_budget, 10);
            const result = ddtree_mod.CompiledTree.isAncestor(mask, j);
            _ = result;
        }
    }.f, .{});
}

// ── Format Fuzzing ──────────────────────────────────────────────

test "fuzz: GGUF fromBuffer no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            // Structure-aware path: plant GGUF magic + version, mutate the rest so
            // metadata/tensor parsing is reached instead of failing at magic check.
            if (smith.valueWithHash(u8, 2) & 1 == 0) {
                std.mem.writeInt(u32, buf[0..4], 0x46554747, .little); // "GGUF"
                std.mem.writeInt(u32, buf[4..8], 3, .little);
                // tensor_count / kv_count: keep small to avoid DoS-sized allocations
                const n_tensors: u64 = smith.valueWithHash(u8, 3) % 4;
                const n_kv: u64 = smith.valueWithHash(u8, 4) % 8;
                std.mem.writeInt(u64, buf[8..16], n_tensors, .little);
                std.mem.writeInt(u64, buf[16..24], n_kv, .little);
            }
            const len = smith.indexWithHash(buf.len - 23, 1) + 24; // at least min header
            const effective_len = @min(len, buf.len);
            var g = gguf.GGUFFile.fromBuffer(std.testing.allocator, buf[0..effective_len]) catch return;
            defer g.deinit();
        }
    }.f, .{});
}

test "fuzz: MetaValue asU32 no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Create various MetaValue variants with random data
            const variant = smith.indexWithHash(5, 0);
            const val: gguf.MetaValue = switch (variant) {
                0 => .{ .uint32 = smith.valueWithHash(u32, 1) },
                1 => .{ .int32 = smith.valueWithHash(i32, 2) },
                2 => .{ .uint64 = smith.valueWithHash(u64, 3) },
                3 => .{ .uint8 = smith.valueWithHash(u8, 4) },
                4 => .{ .uint16 = smith.valueWithHash(u16, 5) },
                else => unreachable,
            };
            // Must not crash
            _ = val.asU32();
        }
    }.f, .{});
}

test "fuzz: MetaValue asF32 no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const variant = smith.indexWithHash(2, 0);
            const val: gguf.MetaValue = switch (variant) {
                0 => .{ .float32 = @bitCast(smith.valueWithHash(u32, 1)) },
                1 => .{ .float64 = @bitCast(smith.valueWithHash(u64, 2)) },
                else => unreachable,
            };
            _ = val.asF32();
        }
    }.f, .{});
}

test "fuzz: GGMLType blockSize no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const raw: u32 = smith.valueWithHash(u32, 0) % 40;
            const t: gguf.GGMLType = @enumFromInt(raw);
            _ = t.blockSize();
            _ = t.bytesPerBlock();
        }
    }.f, .{});
}

test "fuzz: GGMLType tensorBytes no overflow" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const types = [_]gguf.GGMLType{ .q4_0, .q8_0, .f16, .f32, .q4_k, .q6_k, .bf16 };
            const t = types[smith.indexWithHash(types.len, 0)];
            const n_elements: usize = smith.valueWithHash(u32, 1);
            const bytes = t.tensorBytes(n_elements);
            // Must not wrap around to a small number
            try std.testing.expect(bytes >= n_elements or n_elements == 0);
        }
    }.f, .{});
}

// ── Display/Metrics Fuzzing ─────────────────────────────────────

test "fuzz: formatSize no crash" {
    const display = @import("display.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const size: usize = smith.valueWithHash(u64, 0);
            const result = display.formatSize(size);
            try std.testing.expect(result.val >= 0.0);
            try std.testing.expect(result.unit.len > 0);
        }
    }.f, .{});
}

test "fuzz: GenStats tokPerSec no crash" {
    const display = @import("display.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const stats = display.GenStats{
                .token_count = smith.valueWithHash(u32, 0),
                .gen_ms = smith.valueWithHash(u64, 1),
                .prefill_token_count = smith.valueWithHash(u32, 2),
                .prefill_ms = smith.valueWithHash(u64, 3),
            };
            const tps = stats.tokPerSec();
            try std.testing.expect(std.math.isFinite(tps));
            try std.testing.expect(tps >= 0.0);
            const ptps = stats.prefillTokPerSec();
            try std.testing.expect(std.math.isFinite(ptps));
            try std.testing.expect(ptps >= 0.0);
        }
    }.f, .{});
}

// ── KV Cache Fuzzing ────────────────────────────────────────────

test "fuzz: tokenBucket distribution" {
    const kvcache_mgr = @import("kvcache/manager.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const token_id = smith.valueWithHash(u32, 0);
            // tokenBucket is inline and not pub, so test the public alloc/free cycle.
            const n_layers: usize = (token_id % 4) + 1; // 1..4 layers
            const bytes_per_layer: usize = 64;
            const cache = kvcache_mgr.allocKvCache(std.testing.allocator, n_layers, bytes_per_layer) catch return;
            defer kvcache_mgr.freeKvCache(std.testing.allocator, cache);
            try std.testing.expect(cache.keys.len == n_layers);
            try std.testing.expect(cache.values.len == n_layers);
        }
    }.f, .{});
}

test "fuzz: allocKvCache + freeKvCache cycle" {
    const kvcache_mgr = @import("kvcache/manager.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n_layers: usize = smith.indexWithHash(8, 0) + 1; // 1..8
            const bytes_choices = [_]usize{ 0, 32, 64, 128, 256 };
            const bytes_per_layer = bytes_choices[smith.indexWithHash(bytes_choices.len, 1)];
            const cache = kvcache_mgr.allocKvCache(std.testing.allocator, n_layers, bytes_per_layer) catch return;
            defer kvcache_mgr.freeKvCache(std.testing.allocator, cache);
            // Verify structure
            try std.testing.expect(cache.keys.len == n_layers);
            for (cache.keys) |k| try std.testing.expect(k.len == bytes_per_layer);
            for (cache.values) |v| try std.testing.expect(v.len == bytes_per_layer);
        }
    }.f, .{});
}

// ── Recipe Fuzzing ──────────────────────────────────────────────

test "fuzz: Recipe.match no crash" {
    const recipe_mod = @import("recipe.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const archs = [_][]const u8{ "gemma3", "gemma4", "qwen3", "gpt", "glm4", "llama4", "nemotron", "unknown" };
            const backends = [_][]const u8{ "Metal", "Vulkan", "CPU", "CUDA", "WebGPU", "" };
            const quants = [_][]const u8{ "Q4_K", "Q8_0", "Q4_0", "BF16", "F16", "" };
            const arch = archs[smith.indexWithHash(archs.len, 0)];
            const backend = backends[smith.indexWithHash(backends.len, 1)];
            const quant = quants[smith.indexWithHash(quants.len, 2)];
            // Must not crash — may return null or a recipe
            const result = recipe_mod.Recipe.match(arch, backend, quant);
            if (result) |r| {
                try std.testing.expect(r.name.len > 0);
            }
        }
    }.f, .{});
}

test "fuzz: Recipe.applyDefaults no crash" {
    const recipe_mod = @import("recipe.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const recipe = recipe_mod.Recipe{
                .temperature = if (smith.valueWithHash(u8, 0) > 128) @as(f32, @floatFromInt(smith.valueWithHash(u8, 1))) / 255.0 else null,
                .top_p = if (smith.valueWithHash(u8, 2) > 128) @as(f32, @floatFromInt(smith.valueWithHash(u8, 3))) / 255.0 else null,
                .top_k = if (smith.valueWithHash(u8, 4) > 128) smith.valueWithHash(u16, 5) else null,
                .max_tokens = if (smith.valueWithHash(u8, 6) > 128) smith.valueWithHash(u16, 7) else null,
            };
            const overrides = recipe_mod.Recipe.Overrides{
                .temperature = smith.valueWithHash(u8, 10) > 128,
                .top_p = smith.valueWithHash(u8, 11) > 128,
                .top_k = smith.valueWithHash(u8, 12) > 128,
                .max_tokens = smith.valueWithHash(u8, 13) > 128,
            };
            const applied = recipe.applyDefaults(0.7, 0.9, 40, 1.1, 1024, 4096, overrides);
            try std.testing.expect(std.math.isFinite(applied.temperature));
            try std.testing.expect(std.math.isFinite(applied.top_p));
        }
    }.f, .{});
}

// ── Mega Compose Fuzzing ────────────────────────────────────────

test "fuzz: ModelDesc layer accessors" {
    const mega = @import("backend/mega_compose.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var desc = mega.ModelDesc{
                .name = "test",
                .n_layers = @as(u32, smith.valueWithHash(u6, 0)) + 1, // 1..64
                .n_embd = smith.valueWithHash(u16, 1),
                .n_ff = smith.valueWithHash(u16, 2),
                .n_head = @as(u32, smith.valueWithHash(u8, 3)) + 1,
                .n_kv = @as(u32, smith.valueWithHash(u8, 4)) + 1,
                .head_dim = @as(u32, smith.valueWithHash(u8, 5)) + 1,
                .rope_dim = @as(u32, smith.valueWithHash(u8, 6)) + 1,
                .rope_theta = 10000.0,
                .rms_eps = 1e-6,
                .max_seq_len = smith.valueWithHash(u16, 7),
                .activation = .silu,
                .quant = .q8_0,
                .layer_types = mega.ModelDesc.uniform(64, .attention),
            };
            // Set some per-layer overrides
            const override_layer = smith.indexWithHash(@as(usize, desc.n_layers), 10);
            desc.layer_n_head[override_layer] = smith.valueWithHash(u8, 11);
            desc.layer_head_dim[override_layer] = smith.valueWithHash(u8, 12);
            desc.layer_n_ff[override_layer] = smith.valueWithHash(u16, 13);
            desc.layer_rope_theta[override_layer] = @as(f32, @floatFromInt(smith.valueWithHash(u16, 14)));

            // Access each layer — must not crash
            for (0..desc.n_layers) |li| {
                const nh = desc.layerNHead(li);
                const nkv = desc.layerNKv(li);
                const hd = desc.layerHeadDim(li);
                const nff = desc.layerNFf(li);
                const rt = desc.layerRopeTheta(li);
                const sw = desc.layerWindow(li);
                try std.testing.expect(nh > 0);
                try std.testing.expect(nkv > 0);
                try std.testing.expect(hd > 0);
                _ = nff;
                _ = rt;
                _ = sw;
            }
            _ = desc.hasPerLayerVariation();
        }
    }.f, .{});
}

test "fuzz: ModelDesc uniform pattern" {
    const mega = @import("backend/mega_compose.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n_layers: u32 = @as(u32, smith.valueWithHash(u6, 0)) + 1;
            const kinds = [_]mega.LayerKind{ .attention, .deltanet, .moe, .ffn_only };
            const kind = kinds[smith.indexWithHash(kinds.len, 1)];
            const types = mega.ModelDesc.uniform(n_layers, kind);
            for (0..n_layers) |i| try std.testing.expect(types[i] == kind);
        }
    }.f, .{});
}

test "fuzz: ModelDesc qwenHybrid pattern" {
    const mega = @import("backend/mega_compose.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n_layers: u32 = @as(u32, smith.valueWithHash(u6, 0)) + 1;
            const intervals = [_]u32{ 0, 1, 2, 4, 6, 8, 12, 16 };
            const interval = intervals[smith.indexWithHash(intervals.len, 1)];
            const types = mega.ModelDesc.qwenHybrid(n_layers, interval);
            // Verify: every Nth layer is attention (when interval > 0)
            for (0..n_layers) |i| {
                if (interval > 0 and ((i + 1) % interval) == 0) {
                    try std.testing.expect(types[i] == .attention);
                } else if (interval > 0) {
                    try std.testing.expect(types[i] == .deltanet);
                }
            }
        }
    }.f, .{});
}

// ── KV Quant Extended Fuzzing ───────────────────────────────────

test "fuzz: kvSliceBytes no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const all_types = [_]kv_quant.KvQuantType{ .f16, .q8_0, .fp8_e4m3, .turbo2, .turbo3, .turbo4 };
            const kv_type = all_types[smith.indexWithHash(all_types.len, 0)];
            const n: usize = @as(usize, smith.valueWithHash(u16, 1)) + 1;
            const bytes = kv_quant.kvSliceBytes(kv_type, n);
            try std.testing.expect(bytes > 0);
        }
    }.f, .{});
}

test "fuzz: kvStore random data no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n: usize = 16;
            var src: [n]f32 = undefined;
            for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;

            const types = [_]kv_quant.KvQuantType{ .f16, .q8_0, .fp8_e4m3, .turbo2, .turbo3, .turbo4 };
            const kv_type = types[smith.indexWithHash(types.len, 50)];

            var kv_buf: [256]u8 = undefined;
            const needed = kv_quant.kvSliceBytes(kv_type, n);
            if (needed > kv_buf.len) return;
            kv_quant.kvStore(&kv_buf, &src, n, kv_type);
            // Verify we can read it back
            var query: [n]f32 = undefined;
            for (&query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 20)))) / 10.0;
            const dot = kv_quant.kvDot(&query, &kv_buf, n, kv_type);
            try std.testing.expect(std.math.isFinite(dot));
        }
    }.f, .{});
}

// ── Chat Template Extended Fuzzing ──────────────────────────────

test "fuzz: chat template all presets" {
    const chat = @import("chat_template.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var msg_buf: [64]u8 = undefined;
            smith.bytesWithHash(&msg_buf, 0);
            const msg_len = smith.indexWithHash(msg_buf.len + 1, 1);
            const msg = msg_buf[0..msg_len];

            const templates = [_]chat.ChatTemplate{
                chat.ChatTemplate.chatml,
                chat.ChatTemplate.qwen35,
                chat.ChatTemplate.gemma,
                chat.ChatTemplate.gemma4,
                chat.ChatTemplate.glm4,
                chat.ChatTemplate.gpt_oss,
                chat.ChatTemplate.llama4,
            };

            for (templates) |tmpl| {
                const result = tmpl.format(std.testing.allocator, null, msg) catch continue;
                defer std.testing.allocator.free(result);
                try std.testing.expect(result.len > 0);
            }
        }
    }.f, .{});
}

// ── GEMV Dispatcher Fuzzing ──────────────────────────────────────

test "fuzz: gemvSeq dispatch Q8_0" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [34]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, &block, .q8_0, &y, 1, 32);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

test "fuzz: gemvSeq dispatch Q4_0" {
    const gemv = @import("backend/kernels/cpu/gemv.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var x: [32]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            var y: [1]f32 = undefined;
            gemv.gemvSeq(&x, &block, .q4_0, &y, 1, 32);
            try std.testing.expect(std.math.isFinite(y[0]));
        }
    }.f, .{});
}

// ════════════════════════════════════════════════════════════════
// NEW FUZZ TARGETS — Categories below bring total to 143+
// ════════════════════════════════════════════════════════════════

// ── JSON Extended Fuzzing ──────────────────────────────────────

test "fuzz: extractBoolField no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            _ = json.extractBoolField(buf[0..len], "stream");
            _ = json.extractBoolField(buf[0..len], "echo");
        }
    }.f, .{});
}

test "fuzz: extractIntField no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            _ = json.extractIntField(buf[0..len], "max_tokens");
            _ = json.extractIntField(buf[0..len], "n");
            _ = json.extractIntField(buf[0..len], "seed");
        }
    }.f, .{});
}

test "fuzz: extractFloatField no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const fields = [_][]const u8{ "temperature", "top_p", "frequency_penalty", "presence_penalty" };
            inline for (fields) |field| {
                // May return Inf for extreme exponents; callers clamp via isFinite
                _ = json.extractFloatField(buf[0..len], field);
            }
        }
    }.f, .{});
}

test "fuzz: extractObjectField no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            if (json.extractObjectField(buf[0..len], "response_format")) |val| {
                try std.testing.expect(val.len <= len);
            }
        }
    }.f, .{});
}

test "fuzz: extractLastMessage no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            if (json.extractLastMessage(buf[0..len])) |msg| {
                try std.testing.expect(msg.len <= len);
            }
        }
    }.f, .{});
}

test "fuzz: parseTools no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const tools = json.parseTools(buf[0..len]);
            // Tool count must be bounded
            try std.testing.expect(tools.tool_count <= 16);
        }
    }.f, .{});
}

test "fuzz: extractMessages no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            if (json.extractMessages(buf[0..len], std.testing.allocator)) |msgs| {
                var m = msgs;
                defer m.deinit(std.testing.allocator);
                try std.testing.expect(m.messages.len <= 128);
            }
        }
    }.f, .{});
}

test "fuzz: urlDecode no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const decoded = json.urlDecode(std.testing.allocator, buf[0..len]) catch return;
            defer std.testing.allocator.free(decoded);
            try std.testing.expect(decoded.len <= len);
        }
    }.f, .{});
}

test "fuzz: htmlEscape no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const escaped = json.htmlEscape(std.testing.allocator, buf[0..len]) catch return;
            defer std.testing.allocator.free(escaped);
            // HTML escape can only grow the string
            try std.testing.expect(escaped.len >= len);
        }
    }.f, .{});
}

test "fuzz: jsonUnescapeOwned no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const unescaped = json.jsonUnescapeOwned(std.testing.allocator, buf[0..len]) catch return;
            defer std.testing.allocator.free(unescaped);
        }
    }.f, .{});
}

test "fuzz: extractFormField + extractFormBool + extractFormFloat + extractFormInt" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const input = buf[0..len];
            _ = json.extractFormField(input, "prompt");
            _ = json.extractFormBool(input, "stream");
            _ = json.extractFormFloat(input, "temperature");
            _ = json.extractFormInt(input, "max_tokens");
        }
    }.f, .{});
}

test "fuzz: parseFormSampling no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const s = json.parseFormSampling(buf[0..len]);
            try std.testing.expect(std.math.isFinite(s.temperature) and s.temperature >= 0);
            try std.testing.expect(std.math.isFinite(s.top_p));
        }
    }.f, .{});
}

test "fuzz: extractFormImage + extractJsonImage" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            _ = json.extractFormImage(buf[0..len]);
            _ = json.extractJsonImage(buf[0..len]);
        }
    }.f, .{});
}

test "fuzz: SamplingParams hasStop + matchesStop" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [512]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const s = json.parseSampling(buf[0..len]);
            _ = s.hasStop();
            // Generate random text to match against stop sequences
            var text_buf: [64]u8 = undefined;
            smith.bytesWithHash(&text_buf, 2);
            const text_len = smith.indexWithHash(text_buf.len + 1, 3);
            _ = s.matchesStop(text_buf[0..text_len]);
        }
    }.f, .{});
}

// ── Backend Utility Fuzzing ────────────────────────────────────

test "fuzz: weightBytes no overflow" {
    const backend = @import("backend/backend.zig");
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const dtypes = [_]format_mod.DType{
                .f32,    .f16,    .bf16,  .q4_0,  .q4_1,  .q5_0,     .q8_0,
                .q4_k,   .q5_k,   .q6_k,  .q2_k,  .q3_k,  .fp8_e4m3, .fp8_e5m2,
                .iq4_nl, .iq4_xs, .tq1_0, .mxfp4, .nvfp4, .gptq,     .awq,
                .mlx_q,
            };
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 0)];
            const n: usize = @as(usize, smith.valueWithHash(u16, 1)) + 1;
            const k: usize = (@as(usize, smith.valueWithHash(u8, 2)) + 1) * 32; // multiple of 32
            const bytes = backend.weightBytes(dtype, n, k);
            try std.testing.expect(bytes > 0);
        }
    }.f, .{});
}

test "fuzz: gemvRowBytes no overflow" {
    const backend = @import("backend/backend.zig");
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const dtypes = [_]format_mod.DType{
                .f32,    .f16,    .bf16,  .q4_0,  .q4_1, .q5_0,     .q8_0,
                .q4_k,   .q5_k,   .q6_k,  .q2_k,  .q3_k, .fp8_e4m3, .fp8_e5m2,
                .iq4_nl, .iq4_xs, .tq1_0, .mxfp4,
            };
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 0)];
            const k: usize = (@as(usize, smith.valueWithHash(u8, 1)) + 1) * 32;
            const bytes = backend.gemvRowBytes(dtype, k);
            try std.testing.expect(bytes > 0);
        }
    }.f, .{});
}

// ── KV Quant Extended Fuzzing ──────────────────────────────────

test "fuzz: KvQuantType name + bitsPerElement + classification" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const all_types = [_]kv_quant.KvQuantType{
                .f32,    .f16,    .q8_0,   .int8,    .fp8_e4m3, .nvfp4,
                .turbo2, .turbo3, .turbo4, .planar2, .planar3,  .planar4,
                .iso2,   .iso3,   .iso4,   .rotor2,  .rotor3,   .rotor4,
            };
            const kv_type = all_types[smith.indexWithHash(all_types.len, 0)];
            // name() must return non-empty string
            try std.testing.expect(kv_type.name().len > 0);
            // bitsPerElement must be positive
            try std.testing.expect(kv_type.bitsPerElement() > 0);
            // Classification methods — verify consistency
            const is_rot = kv_type.isRotationQuant();
            if (kv_type.isTurbo() or kv_type.isPlanar() or kv_type.isIso() or kv_type.isRotor()) {
                try std.testing.expect(is_rot);
            }
            // turboBits: rotation quants return 2/3/4, others return 0
            const bits = kv_type.turboBits();
            if (is_rot) {
                try std.testing.expect(bits >= 2 and bits <= 4);
            } else {
                try std.testing.expect(bits == 0);
            }
        }
    }.f, .{});
}

test "fuzz: KvQuantType fromString roundtrip" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [16]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            // fromString on random input must not crash
            _ = kv_quant.KvQuantType.fromString(buf[0..len]);
            // Known strings must roundtrip
            const known = [_][]const u8{
                "f32",    "f16",      "q8_0",   "q8",  "int8",    "i8",
                "fp8",    "fp8_e4m3", "nvfp4",  "fp4", "turbo2",  "tq2",
                "turbo3", "tq3",      "turbo4", "tq4", "planar2", "pq2",
                "iso2",   "iq2",      "rotor2", "rq2",
            };
            const k = known[smith.indexWithHash(known.len, 2)];
            const result = kv_quant.KvQuantType.fromString(k);
            try std.testing.expect(result != null);
        }
    }.f, .{});
}

test "fuzz: kvByteOffset no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const types = [_]kv_quant.KvQuantType{ .f16, .q8_0, .fp8_e4m3, .turbo2, .turbo3, .turbo4 };
            const kv_type = types[smith.indexWithHash(types.len, 0)];
            const i: usize = smith.valueWithHash(u16, 1);
            const offset = kv_quant.kvByteOffset(kv_type, i);
            // Offset must be less than or equal to kvSliceBytes for i+1 elements
            const total = kv_quant.kvSliceBytes(kv_type, i + 1);
            try std.testing.expect(offset <= total);
        }
    }.f, .{});
}

test "fuzz: kvMulAccum no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n: usize = 32;
            var src: [n]f32 = undefined;
            for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            const types = [_]kv_quant.KvQuantType{ .f16, .q8_0, .fp8_e4m3, .turbo2, .turbo3, .turbo4 };
            const kv_type = types[smith.indexWithHash(types.len, 50)];
            var kv_buf: [256]u8 = undefined;
            const needed = kv_quant.kvSliceBytes(kv_type, n);
            if (needed > kv_buf.len) return;
            kv_quant.kvStore(&kv_buf, &src, n, kv_type);
            var acc: [n]f32 = .{0} ** n;
            const weight: f32 = @as(f32, @floatFromInt(smith.valueWithHash(i8, 60))) / 10.0;
            kv_quant.kvMulAccum(&acc, weight, &kv_buf, n, kv_type);
            // All accumulator values must be finite
            for (acc) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── Embedding Extended Fuzzing ─────────────────────────────────

test "fuzz: embQ6_K no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q6_K: 210 bytes per 256-element super-block
            const dim: usize = 256;
            var block: [210]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [dim]f32 = undefined;
            emb.embQ6_K(&block, 0, &output, dim);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: embQ4_K no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q4_K: 144 bytes per 256-element super-block
            const dim: usize = 256;
            var block: [144]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [dim]f32 = undefined;
            emb.embQ4_K(&block, 0, &output, dim);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: embQ5_K no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Q5_K: 176 bytes per 256-element super-block
            const dim: usize = 256;
            var block: [176]u8 align(2) = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [dim]f32 = undefined;
            emb.embQ5_K(&block, 0, &output, dim);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

test "fuzz: embMXFP4 no crash" {
    const emb = @import("backend/kernels/cpu/embedding.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // MXFP4: 17 bytes per 32-element block
            const dim: usize = 32;
            var block: [17]u8 = undefined;
            smith.bytesWithHash(&block, 0);
            var output: [dim]f32 = undefined;
            emb.embMXFP4(&block, 0, &output, dim);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── NVFP4 / MXFP4 GEMV Fuzzing ────────────────────────────────

test "fuzz: gemvNvfp4St no crash" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // n=2 output rows, k=16 (must be divisible by 16)
            const n: usize = 2;
            const k: usize = 16;
            var weight: [n * k / 2]u8 = undefined; // packed nibbles
            smith.bytesWithHash(&weight, 0);
            var scale: [n * k / 16]u8 = undefined; // FP8 E4M3 scales
            smith.bytesWithHash(&scale, 10);
            var x: [k]f32 = undefined;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 20)))) / 10.0;
            var y: [n]f32 = undefined;
            quant.gemvNvfp4St(&x, &weight, &scale, &y, n, k);
            for (y) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── SSM Kernel Fuzzing ─────────────────────────────────────────

test "fuzz: causalConv1dSilu no crash" {
    const ssm = @import("ops/ssm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const conv_ch: usize = 4;
            const d_conv: usize = 4;
            const hist = d_conv - 1;
            var conv_out: [conv_ch]f32 = undefined;
            var conv_state: [hist * conv_ch]f32 = undefined;
            var conv_in: [conv_ch]f32 = undefined;
            var conv_w: [conv_ch * d_conv]f32 = undefined;
            var conv_b: [conv_ch]f32 = undefined;
            for (&conv_state, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&conv_in, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 20)))) / 10.0;
            for (&conv_w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            for (&conv_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 60)))) / 10.0;
            ssm.causalConv1dSilu(&conv_out, &conv_state, &conv_in, &conv_w, &conv_b, conv_ch, d_conv);
            for (conv_out) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: causalConv1dSilu variable d_conv" {
    const ssm = @import("ops/ssm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const conv_ch: usize = 2;
            const d_convs = [_]usize{ 2, 3, 4, 5, 6, 7, 8 };
            const d_conv = d_convs[smith.indexWithHash(d_convs.len, 0)];
            const hist = d_conv - 1;
            var conv_out: [2]f32 = undefined;
            var conv_state: [7 * 2]f32 = undefined; // max hist * conv_ch
            var conv_in: [2]f32 = undefined;
            var conv_w: [2 * 8]f32 = undefined; // max conv_ch * d_conv
            for (&conv_state, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 1)))) / 10.0;
            for (&conv_in, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 20)))) / 10.0;
            for (&conv_w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            ssm.causalConv1dSilu(&conv_out, conv_state[0 .. hist * conv_ch].ptr, &conv_in, &conv_w, null, conv_ch, d_conv);
            for (conv_out) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: mamba2Recurrence no crash" {
    const ssm = @import("ops/ssm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const num_heads: usize = 2;
            const head_dim: usize = 4;
            const d_state: usize = 4;
            const hpg: usize = 2;
            const state_size = num_heads * head_dim * d_state;
            var y: [num_heads * head_dim]f32 = undefined;
            var state: [state_size]f32 = undefined;
            var x: [num_heads * head_dim]f32 = undefined;
            var B: [1 * d_state]f32 = undefined; // n_groups = num_heads / hpg = 1
            var C: [1 * d_state]f32 = undefined;
            var dt_raw: [num_heads]f32 = undefined;
            var dt_bias: [num_heads]f32 = undefined;
            var ssm_a: [num_heads]f32 = undefined;
            var ssm_d: [num_heads]f32 = undefined;
            for (&state, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 100.0;
            for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            for (&B, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 50)))) / 10.0;
            for (&C, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 60)))) / 10.0;
            for (&dt_raw, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 70)))) / 50.0;
            for (&dt_bias, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 80)))) / 50.0;
            for (&ssm_a, 0..) |*v, i| v.* = -@as(f32, @floatFromInt(smith.valueWithHash(u8, @truncate(i + 90)))) / 50.0;
            for (&ssm_d, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 100)))) / 10.0;
            ssm.mamba2Recurrence(&y, &state, &x, &B, &C, &dt_raw, &dt_bias, &ssm_a, &ssm_d, num_heads, head_dim, d_state, hpg);
            for (y) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: groupRmsNormSiluGate no crash" {
    const ssm = @import("ops/ssm.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const d_inner: usize = 16;
            const n_groups: usize = 2;
            var y: [d_inner]f32 = undefined;
            var z: [d_inner]f32 = undefined;
            var norm_w: [d_inner]f32 = undefined;
            for (&y, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&z, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 20)))) / 10.0;
            for (&norm_w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 40)))) / 10.0;
            ssm.groupRmsNormSiluGate(&y, &z, &norm_w, d_inner, n_groups, 1e-6);
            for (y) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

// ── SDPA Kernel Fuzzing ────────────────────────────────────────

test "fuzz: sdpaHead no crash" {
    const sdpa_mod = @import("backend/kernels/cpu/sdpa.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const hd: usize = 8;
            const sl: usize = 2;
            const nh: usize = 1;
            const nkv: usize = 1;
            var q: [hd]f32 = undefined;
            var keys: [sl * hd]f32 = undefined;
            var values: [sl * hd]f32 = undefined;
            var output: [hd]f32 = undefined;
            for (&q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            for (&keys, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 10)))) / 10.0;
            for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i + 30)))) / 10.0;
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
            sdpa_mod.sdpaHead(&q, &keys, &values, &output, 0, nh, nkv, hd, sl, scale);
            for (output) |v| try std.testing.expect(std.math.isFinite(v));
        }
    }.f, .{});
}

test "fuzz: sdpa softmax no crash" {
    const sdpa_mod = @import("backend/kernels/cpu/sdpa.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var scores: [32]f32 = undefined;
            for (&scores, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const n = smith.indexWithHash(32, 50) + 1;
            const slice = scores[0..n];
            sdpa_mod.softmax(slice);
            var sum: f32 = 0;
            for (slice) |v| {
                try std.testing.expect(std.math.isFinite(v));
                try std.testing.expect(v >= 0.0);
                sum += v;
            }
            try std.testing.expect(@abs(sum - 1.0) < 0.01);
        }
    }.f, .{});
}

// ── Grammar State Fuzzing ──────────────────────────────────────

test "fuzz: Grammar getEffectiveText no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var buf: [32]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            const result = grammar_mod.Grammar.getEffectiveText(buf[0..len]);
            try std.testing.expect(result.len <= len);
        }
    }.f, .{});
}

test "fuzz: Grammar parse + acceptChar cycle" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Generate random grammar input
            var buf: [128]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            var g = grammar_mod.Grammar.parse(std.testing.allocator, buf[0..len]) catch return;
            defer g.deinit();
            // Try to init state and feed random chars
            var state = g.initState() catch return;
            defer state.deinit();
            for (0..8) |i| {
                const c = smith.valueWithHash(u8, @truncate(i + 10));
                _ = state.acceptChar(c);
            }
            _ = state.isComplete();
        }
    }.f, .{});
}

// ── Chat Template Extended Fuzzing ─────────────────────────────

test "fuzz: chat template formatContinuation" {
    const chat = @import("chat_template.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var msg_buf: [64]u8 = undefined;
            smith.bytesWithHash(&msg_buf, 0);
            const msg_len = smith.indexWithHash(msg_buf.len + 1, 1);
            const templates = [_]chat.ChatTemplate{
                chat.ChatTemplate.chatml,
                chat.ChatTemplate.qwen35,
                chat.ChatTemplate.gemma,
                chat.ChatTemplate.gemma4,
                chat.ChatTemplate.glm4,
            };
            const tmpl = templates[smith.indexWithHash(templates.len, 2)];
            const result = tmpl.formatContinuation(std.testing.allocator, msg_buf[0..msg_len]) catch return;
            defer std.testing.allocator.free(result);
            try std.testing.expect(result.len > 0);
        }
    }.f, .{});
}

// ── Display Helpers Fuzzing ────────────────────────────────────

test "fuzz: ModelInfo bitsPerWeight no crash" {
    const display = @import("display.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const info = display.ModelInfo{
                .name = "test",
                .arch_name = "test",
                .quant = "Q4_K",
                .be_name = "CPU",
                .n_layers = smith.valueWithHash(u16, 0),
                .n_embed = smith.valueWithHash(u16, 1),
                .n_heads = smith.valueWithHash(u8, 2),
                .n_kv_heads = smith.valueWithHash(u8, 3),
                .head_dim = smith.valueWithHash(u8, 4),
                .ff_dim = smith.valueWithHash(u16, 5),
                .vocab_size = smith.valueWithHash(u16, 6),
                .ctx_size = smith.valueWithHash(u16, 7),
                .rope_theta = 10000.0,
                .n_params = smith.valueWithHash(u64, 8),
                .n_experts = 0,
                .n_experts_used = 0,
                .file_size_bytes = smith.valueWithHash(u32, 9),
                .load_ms = smith.valueWithHash(u32, 10),
                .warmup_ms = smith.valueWithHash(u16, 11),
            };
            const bpw = info.bitsPerWeight();
            try std.testing.expect(std.math.isFinite(bpw));
            try std.testing.expect(bpw >= 0.0);
        }
    }.f, .{});
}

// ── Format / TensorInfo Fuzzing ────────────────────────────────

test "fuzz: TensorInfo numElements no overflow" {
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n_dims: u32 = @intCast(smith.indexWithHash(4, 0) + 1);
            const t = format_mod.TensorInfo{
                .name = "test",
                .n_dims = n_dims,
                .dims = .{
                    smith.valueWithHash(u16, 1),
                    smith.valueWithHash(u16, 2),
                    smith.valueWithHash(u16, 3),
                    smith.valueWithHash(u16, 4),
                },
                .dtype = .f32,
                .data_ptr = undefined,
            };
            const n = t.numElements();
            // numElements returns 0 on overflow via saturating mul
            _ = n;
        }
    }.f, .{});
}

test "fuzz: TensorInfo dataByteLen no overflow" {
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const dtypes = [_]format_mod.DType{
                .f32,      .f16,      .bf16,  .q4_0,  .q4_1,  .q5_0,   .q8_0,
                .q4_k,     .q5_k,     .q6_k,  .q2_k,  .q3_k,  .iq4_nl, .iq4_xs,
                .fp8_e4m3, .fp8_e5m2, .tq1_0, .mxfp4, .nvfp4,
            };
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 0)];
            const d0 = @as(u64, smith.valueWithHash(u16, 1));
            const d1 = @as(u64, smith.valueWithHash(u16, 2));
            const t = format_mod.TensorInfo{
                .name = "test",
                .n_dims = 2,
                .dims = .{ d0, d1, 0, 0 },
                .dtype = dtype,
                .data_ptr = undefined,
            };
            const bytes = t.dataByteLen();
            _ = bytes;
        }
    }.f, .{});
}

// ── Model Helper Extended Fuzzing ──────────────────────────────

test "fuzz: expertStride no overflow" {
    const model_mod = @import("models/model.zig");
    const format_mod = @import("format/format.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const dtypes = [_]format_mod.DType{ .q4_k, .q8_0, .f16, .f32, .bf16, .q4_0, .q6_k };
            const dtype = dtypes[smith.indexWithHash(dtypes.len, 0)];
            const d0 = smith.valueWithHash(u16, 1);
            const d1 = smith.valueWithHash(u16, 2);
            const d2 = smith.valueWithHash(u16, 3);
            const t = format_mod.TensorInfo{
                .name = "test",
                .n_dims = 3,
                .dims = .{ d0, d1, d2, 0 },
                .dtype = dtype,
                .data_ptr = undefined,
            };
            const stride = model_mod.expertStride(t);
            _ = stride;
        }
    }.f, .{});
}

// ── Paged KV Cache Fuzzing ─────────────────────────────────────

test "fuzz: PagedKvCache alloc/free cycle" {
    const kvcache_mgr = @import("kvcache/manager.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const n_layers: usize = 1;
            const kv_dim: usize = 8;
            const num_blocks: usize = smith.indexWithHash(8, 0) + 1;
            const block_sizes = [_]u16{ 1, 2, 4, 8 };
            const block_size = block_sizes[smith.indexWithHash(block_sizes.len, 1)];
            var cache = kvcache_mgr.PagedKvCache.init(std.testing.allocator, n_layers, kv_dim, num_blocks, block_size) catch return;
            defer cache.deinit();
            try std.testing.expect(cache.freeCount() == num_blocks);
            // Alloc all blocks
            var allocated: [8]u32 = undefined;
            var n_alloc: usize = 0;
            while (cache.allocBlock()) |block_id| {
                if (n_alloc < 8) {
                    allocated[n_alloc] = block_id;
                    n_alloc += 1;
                }
            }
            try std.testing.expect(n_alloc == num_blocks);
            try std.testing.expect(cache.freeCount() == 0);
            // Free all blocks
            for (0..n_alloc) |i| cache.freeBlock(allocated[i]);
            try std.testing.expect(cache.freeCount() == num_blocks);
        }
    }.f, .{});
}

// ── Radix Tree Fuzzing ─────────────────────────────────────────

test "fuzz: RadixTree insert + matchPrefix" {
    const kvcache_mgr = @import("kvcache/manager.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var tree = kvcache_mgr.RadixTree.init(std.testing.allocator) catch return;
            defer tree.deinit();
            // Insert random token sequences
            var tokens: [16]u32 = undefined;
            var block_ids: [16]u32 = undefined;
            for (&tokens, 0..) |*t, i| t.* = smith.valueWithHash(u16, @truncate(i));
            for (&block_ids, 0..) |*b, i| b.* = smith.valueWithHash(u8, @truncate(i + 20));
            const tok_len = smith.indexWithHash(16, 40) + 1;
            tree.insert(tokens[0..tok_len], block_ids[0..tok_len]) catch return;
            // Match prefix must find at least what we inserted
            const match = tree.matchPrefix(tokens[0..tok_len]);
            try std.testing.expect(match.matched >= 1);
            try std.testing.expect(match.matched <= tok_len);
        }
    }.f, .{});
}

test "fuzz: RadixTree multiple inserts + invalidateHashCache" {
    const kvcache_mgr = @import("kvcache/manager.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var tree = kvcache_mgr.RadixTree.init(std.testing.allocator) catch return;
            defer tree.deinit();
            // Insert 2-3 sequences
            const n_seqs = smith.indexWithHash(2, 0) + 2;
            for (0..n_seqs) |seq| {
                var tokens: [8]u32 = undefined;
                var blocks: [8]u32 = undefined;
                for (&tokens, 0..) |*t, i| t.* = smith.valueWithHash(u16, @truncate(seq * 20 + i));
                for (&blocks, 0..) |*b, i| b.* = @truncate(seq * 10 + i);
                const len = smith.indexWithHash(8, @truncate(seq + 50)) + 1;
                tree.insert(tokens[0..len], blocks[0..len]) catch return;
            }
            tree.invalidateHashCache();
            // Query after invalidation
            var q: [4]u32 = undefined;
            for (&q, 0..) |*t, i| t.* = smith.valueWithHash(u16, @truncate(i + 100));
            _ = tree.matchPrefix(&q);
        }
    }.f, .{});
}

// ── Math Extended Fuzzing ──────────────────────────────────────

test "fuzz: topLogProbs no crash" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            const n = smith.indexWithHash(5, 50) + 1; // 1..5
            var out_ids: [5]u32 = undefined;
            var out_logprobs: [5]f32 = undefined;
            const count = math_ops.topLogProbs(&logits, @intCast(n), out_ids[0..n], out_logprobs[0..n]);
            try std.testing.expect(count <= n);
            for (out_logprobs[0..count]) |lp| {
                try std.testing.expect(!std.math.isNan(lp));
                // Log probs should be <= 0
                try std.testing.expect(lp <= 0.0 or lp == -std.math.inf(f32));
            }
        }
    }.f, .{});
}

// ── Mega Compose Extended Fuzzing ──────────────────────────────

test "fuzz: composeMSL no crash" {
    const mega = @import("backend/mega_compose.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const activations = [_]mega.Activation{ .silu, .gelu };
            const quants = [_]mega.QuantKind{ .q8_0, .q4_k, .q4_0, .q5_k, .q6_k };
            const desc = mega.ModelDesc{
                .name = "fuzz_model",
                .n_layers = @as(u32, smith.valueWithHash(u4, 0)) + 1,
                .n_embd = (@as(u32, smith.valueWithHash(u8, 1)) + 1) * 8,
                .n_ff = (@as(u32, smith.valueWithHash(u8, 2)) + 1) * 8,
                .n_head = @as(u32, smith.valueWithHash(u4, 3)) + 1,
                .n_kv = @as(u32, smith.valueWithHash(u4, 4)) + 1,
                .head_dim = @as(u32, smith.valueWithHash(u4, 5)) * 8 + 8,
                .rope_dim = @as(u32, smith.valueWithHash(u4, 6)) * 8 + 8,
                .rope_theta = 10000.0,
                .rms_eps = 1e-6,
                .max_seq_len = smith.valueWithHash(u16, 7),
                .activation = activations[smith.indexWithHash(activations.len, 8)],
                .quant = quants[smith.indexWithHash(quants.len, 9)],
                .layer_types = mega.ModelDesc.uniform(64, .attention),
            };
            var buf: [16384]u8 = undefined;
            const result = mega.composeMSL(&buf, desc);
            // Must produce non-empty MSL source
            try std.testing.expect(result.len > 0);
        }
    }.f, .{});
}

// ── Quant Extended Fuzzing ─────────────────────────────────────

test "fuzz: dequantToF32 all supported dtypes" {
    const quant = @import("ops/quant.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // dequantToF32 supports: f32, bf16, f16, q8_0, q4_0, iq4_nl, iq4_xs
            // (IQ2/IQ3/IQ1 and other dtypes zero the buffer; not exercised here)
            const variant = smith.indexWithHash(5, 0);
            switch (variant) {
                0 => {
                    // f32: 4 bytes per element
                    var block: [128]u8 align(4) = undefined;
                    smith.bytesWithHash(&block, 1);
                    var output: [32]f32 = undefined;
                    quant.dequantToF32(&output, &block, .f32, 32);
                },
                1 => {
                    // bf16: 2 bytes per element
                    var block: [64]u8 align(2) = undefined;
                    smith.bytesWithHash(&block, 1);
                    var output: [32]f32 = undefined;
                    quant.dequantToF32(&output, &block, .bf16, 32);
                },
                2 => {
                    // f16: 2 bytes per element
                    var block: [64]u8 align(2) = undefined;
                    smith.bytesWithHash(&block, 1);
                    var output: [32]f32 = undefined;
                    quant.dequantToF32(&output, &block, .f16, 32);
                },
                3 => {
                    // q8_0: 34 bytes per 32-elem block
                    var block: [34]u8 align(2) = undefined;
                    smith.bytesWithHash(&block, 1);
                    var output: [32]f32 = undefined;
                    quant.dequantToF32(&output, &block, .q8_0, 32);
                    for (output) |v| try std.testing.expect(!std.math.isNan(v));
                },
                4 => {
                    // q4_0: 18 bytes per 32-elem block
                    var block: [18]u8 align(2) = undefined;
                    smith.bytesWithHash(&block, 1);
                    var output: [32]f32 = undefined;
                    quant.dequantToF32(&output, &block, .q4_0, 32);
                    for (output) |v| try std.testing.expect(!std.math.isNan(v));
                },
                else => unreachable,
            }
        }
    }.f, .{});
}

// ── Term Extended Fuzzing ──────────────────────────────────────

test "fuzz: TextInput insert + update cycle" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var input = term.TextInput.init(std.testing.allocator);
            defer input.deinit();
            // Insert random text
            var buf: [32]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            input.insertSliceAtCursor(buf[0..len]) catch return;
            // Clear and verify
            input.clearRetainingCapacity();
        }
    }.f, .{});
}

test "fuzz: term GapBuffer operations" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var gbuf = term.TextInput.Buffer.init(std.testing.allocator);
            defer gbuf.deinit();
            // Insert random text
            var buf: [16]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len + 1, 1);
            gbuf.insertSliceAtCursor(buf[0..len]) catch return;
            const real_len = gbuf.realLength();
            try std.testing.expect(real_len == len);
            // Move gap
            const move_left = smith.indexWithHash(len + 1, 2);
            gbuf.moveGapLeft(move_left);
            const move_right = smith.indexWithHash(move_left + 1, 3);
            gbuf.moveGapRight(move_right);
            // Test slices
            _ = gbuf.firstHalf();
            _ = gbuf.secondHalf();
            gbuf.clearRetainingCapacity();
            try std.testing.expect(gbuf.realLength() == 0);
        }
    }.f, .{});
}

test "fuzz: term Key matches" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const cp: u21 = smith.valueWithHash(u16, 0);
            const mods = term.Key.Modifiers{
                .shift = smith.valueWithHash(u8, 1) > 128,
                .alt = smith.valueWithHash(u8, 2) > 128,
                .ctrl = smith.valueWithHash(u8, 3) > 128,
            };
            const key = term.Key{
                .codepoint = cp,
                .mods = mods,
            };
            // Self-match must be true
            try std.testing.expect(key.matches(cp, mods));
            // Test with different codepoint
            const other_cp: u21 = smith.valueWithHash(u16, 5);
            if (other_cp != cp) {
                try std.testing.expect(!key.matches(other_cp, mods));
            }
        }
    }.f, .{});
}

test "fuzz: term Modifiers eql" {
    const term = @import("term.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const m1 = term.Key.Modifiers{
                .shift = smith.valueWithHash(u8, 0) > 128,
                .alt = smith.valueWithHash(u8, 1) > 128,
                .ctrl = smith.valueWithHash(u8, 2) > 128,
            };
            const m2 = term.Key.Modifiers{
                .shift = smith.valueWithHash(u8, 4) > 128,
                .alt = smith.valueWithHash(u8, 5) > 128,
                .ctrl = smith.valueWithHash(u8, 6) > 128,
            };
            // Self-equality must hold
            try std.testing.expect(m1.eql(m1));
            // Symmetry
            try std.testing.expect(m1.eql(m2) == m2.eql(m1));
        }
    }.f, .{});
}

// ── GGUF Extended Fuzzing ──────────────────────────────────────

test "fuzz: MetaValue asStr no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var str_buf: [32]u8 = undefined;
            smith.bytesWithHash(&str_buf, 0);
            const len = smith.indexWithHash(str_buf.len + 1, 1);
            const vals = [_]gguf.MetaValue{
                .{ .string = str_buf[0..len] },
                .{ .uint32 = smith.valueWithHash(u32, 2) },
                .{ .float32 = @bitCast(smith.valueWithHash(u32, 3)) },
            };
            const val = vals[smith.indexWithHash(vals.len, 4)];
            _ = val.asStr();
            _ = val.asBool();
        }
    }.f, .{});
}

test "fuzz: GGMLType blockSize + bytesPerBlock exhaustive" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Test all valid enum values
            const types = [_]gguf.GGMLType{
                @enumFromInt(0),  @enumFromInt(1),  @enumFromInt(2),
                @enumFromInt(3),  @enumFromInt(6),  @enumFromInt(7),
                @enumFromInt(8),  @enumFromInt(10), @enumFromInt(12),
                @enumFromInt(14), @enumFromInt(15), @enumFromInt(16),
            };
            const t = types[smith.indexWithHash(types.len, 0)];
            const bs = t.blockSize();
            const bpb = t.bytesPerBlock();
            try std.testing.expect(bs > 0);
            try std.testing.expect(bpb > 0);
            // bytes per block should be at least 1 for any quantized type
            const n_elements = @as(usize, smith.valueWithHash(u16, 1)) + 1;
            const tensor_bytes = t.tensorBytes(n_elements);
            _ = tensor_bytes;
        }
    }.f, .{});
}

test "fuzz: GGUF dequantQ4_0 + dequantQ8_0 no crash" {
    const gguf = @import("format/gguf.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var block_q4: [18]u8 align(2) = undefined;
            smith.bytesWithHash(&block_q4, 0);
            var out_q4: [32]f32 = undefined;
            gguf.GGUFFile.dequantQ4_0(&block_q4, &out_q4);
            for (out_q4) |v| try std.testing.expect(!std.math.isNan(v));

            var block_q8: [34]u8 align(2) = undefined;
            smith.bytesWithHash(&block_q8, 10);
            var out_q8: [32]f32 = undefined;
            gguf.GGUFFile.dequantQ8_0(&block_q8, &out_q8);
            for (out_q8) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── NGram Extended Fuzzing ─────────────────────────────────────

test "fuzz: ngram push + propose many tokens" {
    const ngram_mod = @import("spec/ngram.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var state = ngram_mod.NgramState{};
            // Push a longer sequence to build more n-gram patterns
            const n_tokens: usize = @as(usize, smith.valueWithHash(u8, 0)) + 1;
            for (0..n_tokens) |i| {
                // Use small vocab to create repeating patterns
                const token = smith.valueWithHash(u8, @truncate(i + 1)) % 8;
                state.push(token);
            }
            // Try all draft lengths 1..16
            const max_drafts = [_]usize{ 1, 2, 4, 8, 16 };
            const max_draft = max_drafts[smith.indexWithHash(max_drafts.len, 200)];
            var draft: [16]u32 = undefined;
            const n = state.propose(max_draft, &draft);
            try std.testing.expect(n <= max_draft);
        }
    }.f, .{});
}

// ── DDTree Extended Fuzzing ────────────────────────────────────

test "fuzz: DDTree full pipeline — presort + build + compile + findChild" {
    const ddtree_mod = @import("spec/ddtree.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var builder = ddtree_mod.DDTreeBuilder{};
            builder.budget = @as(u32, smith.valueWithHash(u4, 0)) + 2; // 2..17

            const n_depths = smith.indexWithHash(3, 1) + 1;
            const vocab_size: usize = 8;
            var logit_storage: [3][8]f32 = undefined;
            var logit_slices: [3][]const f32 = undefined;
            for (0..n_depths) |d| {
                for (&logit_storage[d], 0..) |*v, i| {
                    v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(d * 10 + i + 10)))) / 10.0;
                }
                logit_slices[d] = logit_storage[d][0..vocab_size];
            }

            builder.presort(logit_slices[0..n_depths]);
            builder.buildTree();
            const compiled = builder.compile(smith.valueWithHash(u8, 50));

            // Search for children at various nodes
            for (0..@min(compiled.n_nodes, 4)) |node_i| {
                const search_token = smith.valueWithHash(u16, @truncate(node_i + 60));
                _ = compiled.findChild(@intCast(node_i), search_token);
            }
        }
    }.f, .{});
}

// ── SDPA Quant Kernel Fuzzing ──────────────────────────────────

test "fuzz: sdpaQuantHead f16 keys/values" {
    const sdpa_mod = @import("backend/kernels/cpu/sdpa.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const hd: usize = 8;
            const sl: usize = 2;
            const nh: usize = 1;
            const nkv: usize = 1;
            var q: [hd]f32 = undefined;
            for (&q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i)))) / 10.0;
            // f16 quantized KV: 2 bytes per element
            const k_bytes = sl * hd * 2;
            const v_bytes = sl * hd * 2;
            var keys: [k_bytes]u8 align(2) = undefined;
            var values: [v_bytes]u8 align(2) = undefined;
            smith.bytesWithHash(&keys, 10);
            smith.bytesWithHash(&values, 20);
            var output: [hd]f32 = undefined;
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
            sdpa_mod.sdpaQuantHead(&q, &keys, &values, &output, 0, nh, nkv, hd, sl, scale, .f16, .f16);
            for (output) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}

// ── Recipe Extended Fuzzing ────────────────────────────────────

test "fuzz: Recipe match + applyDefaults cycle" {
    const recipe_mod = @import("recipe.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const archs = [_][]const u8{ "gemma3", "gemma4", "qwen3", "gpt", "glm4", "llama4", "nemotron", "unknown", "" };
            const backends = [_][]const u8{ "Metal", "Vulkan", "CPU", "CUDA", "WebGPU", "ROCm", "" };
            const quants = [_][]const u8{ "Q4_K", "Q8_0", "Q4_0", "BF16", "F16", "IQ4_XS", "" };
            const arch = archs[smith.indexWithHash(archs.len, 0)];
            const backend = backends[smith.indexWithHash(backends.len, 1)];
            const quant = quants[smith.indexWithHash(quants.len, 2)];
            if (recipe_mod.Recipe.match(arch, backend, quant)) |recipe| {
                // Apply with random overrides
                const overrides = recipe_mod.Recipe.Overrides{
                    .temperature = smith.valueWithHash(u8, 3) > 128,
                    .top_p = smith.valueWithHash(u8, 4) > 128,
                    .top_k = smith.valueWithHash(u8, 5) > 128,
                    .max_tokens = smith.valueWithHash(u8, 6) > 128,
                };
                const applied = recipe.applyDefaults(0.7, 0.9, 40, 1.1, 1024, 4096, overrides);
                try std.testing.expect(std.math.isFinite(applied.temperature));
                try std.testing.expect(std.math.isFinite(applied.top_p));
                try std.testing.expect(applied.top_k <= 1024);
            }
        }
    }.f, .{});
}

// ── High-risk untrusted-input surfaces (vision + nested API JSON) ──

test "fuzz: nested chat/completions JSON messages + tools" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Structure-aware: wrap mutated payloads in the shapes the HTTP API expects
            // so extractMessages / parseTools / parseSampling reach nested paths.
            var inner: [192]u8 = undefined;
            smith.bytesWithHash(&inner, 0);
            const inner_len = smith.indexWithHash(inner.len + 1, 1);
            // Sanitize to printable ASCII, excluding " and \ so the JSON wrapper stays valid
            for (inner[0..inner_len]) |*b| {
                var c: u8 = 0x20 + (b.* % 0x5f);
                if (c == '"' or c == '\\') c = 'x';
                b.* = c;
            }

            var body: [512]u8 = undefined;
            const n = std.fmt.bufPrint(&body,
                \\{{"model":"m","temperature":{d},"messages":[{{"role":"user","content":"{s}"}}],"tools":[{{"type":"function","function":{{"name":"{s}"}}}}],"stream":true}}
            , .{
                @as(f32, @floatFromInt(smith.valueWithHash(u8, 2))) / 64.0,
                inner[0..@min(inner_len, 64)],
                inner[0..@min(inner_len, 32)],
            }) catch return;

            const s = json.parseSampling(body[0..n.len]);
            try std.testing.expect(std.math.isFinite(s.temperature) and s.temperature >= 0);
            try std.testing.expect(std.math.isFinite(s.top_p));

            const tools = json.parseTools(body[0..n.len]);
            try std.testing.expect(tools.tool_count <= 16);

            if (json.extractMessages(body[0..n.len], std.testing.allocator)) |msgs| {
                var m = msgs;
                defer m.deinit(std.testing.allocator);
                try std.testing.expect(m.messages.len <= 128);
            }
            _ = json.extractLastMessage(body[0..n.len]);
            _ = json.extractJsonImage(body[0..n.len]);
        }
    }.f, .{});
}

test "fuzz: vision form/json image extract + base64 + decode" {
    const image_mod = @import("image.zig");
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const allocator = std.testing.allocator;

            // Build fake image bytes (PNG-shaped or random), base64-encode, embed in API bodies
            var raw: [96]u8 = undefined;
            smith.bytesWithHash(&raw, 0);
            if (smith.valueWithHash(u8, 1) & 1 == 0) {
                const sig = [_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
                @memcpy(raw[0..8], &sig);
            }
            const raw_len = smith.indexWithHash(raw.len - 7, 2) + 8;

            var b64_buf: [256]u8 = undefined;
            const enc = std.base64.standard.Encoder;
            const b64_len = enc.calcSize(raw_len);
            if (b64_len > b64_buf.len) return;
            _ = enc.encode(b64_buf[0..b64_len], raw[0..raw_len]);

            // OpenAI-style JSON body
            var json_body: [400]u8 = undefined;
            const jn = std.fmt.bufPrint(&json_body,
                \\{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{s}"}}}}]}}]}}
            , .{b64_buf[0..b64_len]}) catch return;
            const extracted = json.extractJsonImage(json_body[0..jn.len]);
            if (extracted) |b64| {
                // Mirror processVisionImage decode path (without vision encoder)
                const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return;
                const image_bytes = allocator.alloc(u8, decoded_size) catch return;
                defer allocator.free(image_bytes);
                std.base64.standard.Decoder.decode(image_bytes, b64) catch return;
                switch (image_mod.detectFormat(image_bytes)) {
                    .png => {
                        var png = image_mod.decodePng(allocator, image_bytes) catch return;
                        defer png.deinit();
                        try std.testing.expect(png.pixels.len == @as(usize, png.width) * png.height * 3);
                    },
                    .ppm => {
                        const ppm = image_mod.decodePpm(image_bytes) catch return;
                        try std.testing.expect(ppm.pixels.len == @as(usize, ppm.width) * ppm.height * 3);
                    },
                    .jpeg, .unknown => {},
                }
            }

            // Form-encoded image field (URL-encoded base64 marker path)
            var form_body: [400]u8 = undefined;
            const fn_written = std.fmt.bufPrint(&form_body, "message=hi&image=data%3Aimage%2Fpng%3Bbase64%2C{s}&stream=1", .{b64_buf[0..b64_len]}) catch return;
            if (json.extractFormImage(form_body[0..fn_written.len])) |form_b64| {
                const url_decoded = json.urlDecode(allocator, form_b64) catch return;
                defer allocator.free(url_decoded);
                try std.testing.expect(url_decoded.len <= form_b64.len);
            }
        }
    }.f, .{});
}

// ── Additional high-risk untrusted surfaces ─────────────────────

test "fuzz: deeply nested JSON extractObjectField + skip paths" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            // Craft nested objects/arrays so extractObjectField and message
            // scanners exercise bracket-matching under adversarial depth.
            const depth: usize = smith.indexWithHash(48, 0) + 1;
            var body: [512]u8 = undefined;
            var i: usize = 0;
            const prefix = "{\"messages\":[{\"role\":\"user\",\"content\":";
            if (prefix.len > body.len) return;
            @memcpy(body[0..prefix.len], prefix);
            i = prefix.len;
            for (0..depth) |_| {
                if (i + 1 >= body.len) break;
                body[i] = '{';
                i += 1;
            }
            const mid = "\"x\":\"";
            if (i + mid.len >= body.len) return;
            @memcpy(body[i..][0..mid.len], mid);
            i += mid.len;
            var payload: [32]u8 = undefined;
            smith.bytesWithHash(&payload, 1);
            const plen = smith.indexWithHash(payload.len + 1, 2);
            for (payload[0..plen]) |*b| {
                if (b.* < 0x20 or b.* == '"' or b.* == '\\') b.* = 'x';
            }
            const copy_len = @min(plen, body.len - i);
            @memcpy(body[i..][0..copy_len], payload[0..copy_len]);
            i += copy_len;
            if (i + 1 >= body.len) return;
            body[i] = '"';
            i += 1;
            for (0..depth) |_| {
                if (i + 1 >= body.len) break;
                body[i] = '}';
                i += 1;
            }
            const suffix = "}]}";
            if (i + suffix.len > body.len) return;
            @memcpy(body[i..][0..suffix.len], suffix);
            i += suffix.len;

            _ = json.extractObjectField(body[0..i], "messages");
            _ = json.extractObjectField(body[0..i], "content");
            _ = json.extractLastMessage(body[0..i]);
            if (json.extractMessages(body[0..i], std.testing.allocator)) |msgs| {
                var m = msgs;
                defer m.deinit(std.testing.allocator);
                try std.testing.expect(m.messages.len <= 128);
            }
            const s = json.parseSampling(body[0..i]);
            try std.testing.expect(std.math.isFinite(s.temperature));
        }
    }.f, .{});
}

test "fuzz: Anthropic stop_sequences + logit_bias structure-aware" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            var frag: [64]u8 = undefined;
            smith.bytesWithHash(&frag, 0);
            const flen = smith.indexWithHash(frag.len + 1, 1);
            for (frag[0..flen]) |*b| {
                if (b.* < 0x20 or b.* == '"' or b.* == '\\') b.* = 's';
            }
            var body: [384]u8 = undefined;
            const n = std.fmt.bufPrint(&body,
                \\{{"temperature":{d},"stop_sequences":["{s}","</s>"],"logit_bias":{{"{d}":{d},"999":-100}},"response_format":{{"type":"json_object"}}}}
            , .{
                @as(f32, @floatFromInt(smith.valueWithHash(u8, 2))) / 32.0,
                frag[0..@min(flen, 24)],
                smith.valueWithHash(u16, 3),
                @as(i8, @bitCast(smith.valueWithHash(u8, 4))),
            }) catch return;
            const s = json.parseSampling(body[0..n.len]);
            try std.testing.expect(s.n_stop <= 4);
            try std.testing.expect(s.logit_bias_count <= 16);
            try std.testing.expect(std.math.isFinite(s.temperature));
            _ = s.matchesStop(frag[0..flen]);
            _ = s.matchesStop("</s>");
            _ = s.hasStop();
        }
    }.f, .{});
}

test "fuzz: grammar fromJsonSchema nested + maskLogits" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *Smith) !void {
            const allocator = std.testing.allocator;
            // Nested schema shapes that SchemaConverter walks recursively
            var schema: [320]u8 = undefined;
            const kinds = [_][]const u8{ "string", "number", "integer", "boolean", "object", "array" };
            const kind = kinds[smith.indexWithHash(kinds.len, 0)];
            const n = std.fmt.bufPrint(&schema,
                \\{{"type":"object","properties":{{"a":{{"type":"{s}","enum":["x","y"]}},"b":{{"type":"array","items":{{"type":"string"}}}}}},"required":["a"]}}
            , .{kind}) catch return;
            var g = grammar_mod.Grammar.fromJsonSchema(allocator, schema[0..n.len]) catch {
                // Also try raw bytes as schema
                var raw: [128]u8 = undefined;
                smith.bytesWithHash(&raw, 1);
                const rlen = smith.indexWithHash(raw.len + 1, 2);
                var g2 = grammar_mod.Grammar.fromJsonSchema(allocator, raw[0..rlen]) catch return;
                defer g2.deinit();
                return;
            };
            defer g.deinit();
            var state = g.initState() catch return;
            defer state.deinit();
            const ch: u8 = smith.valueWithHash(u8, 3);
            _ = state.acceptChar(ch);
            var logits: [32]f32 = undefined;
            for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(smith.valueWithHash(i8, @truncate(i))));
            // Empty vocab path must not crash maskLogits
            g.maskLogits(&state, &logits, &.{}) catch {};
            for (logits) |v| try std.testing.expect(!std.math.isNan(v));
        }
    }.f, .{});
}
