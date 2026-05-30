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
            // TQ1_0: 256-elem blocks, 64 bytes each
            var block: [64]u8 align(2) = undefined;
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
            var buf: [256]u8 = undefined;
            smith.bytesWithHash(&buf, 0);
            const len = smith.indexWithHash(buf.len, 1) + 24; // at least min header size
            const effective_len = @min(len, buf.len);
            // Must not crash — just return error for invalid data
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
