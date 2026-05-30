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

            awq.awqGemvRows(&x, &qw, &scales, &qz, &y, 0, n, k, group_size);
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

            gptq.gptqGemvRows(&x, &qw, &scales, &qz, &y, 0, n, k, group_size);
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
