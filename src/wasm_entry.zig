//! WASM entry point for browser-based inference.
//!
//! Exports functions callable from JavaScript:
//! - agave_init(model_ptr, model_len) → context handle
//! - agave_generate(ctx, prompt_ptr, prompt_len, max_tokens) → token count
//! - agave_get_output(ctx, buf_ptr, buf_len) → bytes written
//! - agave_free(ctx)
//!
//! Model data is passed as a byte buffer from JS (no file I/O).
//! Uses CPU backend only (no GPU in WASM freestanding).
//!
//! NOTE: Full forward pass is blocked by a Zig 0.16 + LLVM 21 wasm32
//! codegen bug (Invalid cast in SIMD vector lowering). Model init,
//! GGUF parsing, and tokenization work. Inference requires Zig update.

const std = @import("std");

pub const std_options: std.Options = .{
    .logFn = wasmLogFn,
};
fn wasmLogFn(comptime _: std.log.Level, comptime _: @TypeOf(.enum_literal), comptime _: []const u8, _: anytype) void {}

const format_mod = @import("format/format.zig");
const GGUFFile = format_mod.GGUFFile;
const Format = format_mod.Format;
const model_mod = @import("models/model.zig");
const ModelStorage = model_mod.ModelStorage;
const Arch = @import("arch.zig").Arch;
const backend_mod = @import("backend/backend.zig");
const Backend = backend_mod.Backend;
const CpuBackend = backend_mod.CpuBackend;
const tok_mod = @import("tokenizer/tokenizer.zig");
const BpeTokenizer = tok_mod.BpeTokenizer;

var gpa = std.heap.page_allocator;

const max_output_bytes = 16384;

const InferenceContext = struct {
    gguf: GGUFFile,
    tok: BpeTokenizer,
    cpu_be: CpuBackend = .{},
    mdl: ?ModelStorage = null,
    output_buf: [max_output_bytes]u8 = undefined,
    output_len: usize = 0,
    ready: bool = false,
    eos_id: u32 = 0,
    bos_id: u32 = 0,
    arch: Arch = .gemma3,
    n_layers: u32 = 0,
    n_embd: u32 = 0,
    vocab_size: u32 = 0,
    arch_name: []const u8 = "",
    model_name: []const u8 = "",
};

export fn agave_init(model_ptr: [*]const u8, model_len: usize) usize {
    const ctx = gpa.create(InferenceContext) catch return 0;
    ctx.* = .{
        .gguf = GGUFFile.fromBuffer(gpa, model_ptr[0..model_len]) catch |e| {
            const msg = std.fmt.bufPrint(&ctx.output_buf, "GGUF parse error: {s}", .{@errorName(e)}) catch "";
            ctx.output_len = msg.len;
            return @intFromPtr(ctx);
        },
        .tok = BpeTokenizer.init(gpa),
    };

    const fmt = ctx.gguf.format();
    const arch_str = fmt.getMetaStr("general.architecture") orelse "unknown";
    ctx.arch = Arch.detect(arch_str) orelse {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Unsupported arch: {s}", .{arch_str}) catch "";
        ctx.output_len = msg.len;
        return @intFromPtr(ctx);
    };
    ctx.arch_name = arch_str;
    ctx.model_name = fmt.getMetaStr("general.name") orelse arch_str;

    // Load tokenizer
    const vocab = fmt.getVocab() orelse {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "No vocab in GGUF", .{}) catch "";
        ctx.output_len = msg.len;
        return @intFromPtr(ctx);
    };
    ctx.eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse ctx.arch.defaultEos();
    ctx.bos_id = fmt.getMetaU32("tokenizer.ggml.bos_token_id") orelse 0;
    ctx.vocab_size = @intCast(vocab.len);

    const merges = fmt.getMerges();
    if (merges == null or ctx.arch == .gemma3 or ctx.arch == .gemma4) {
        ctx.tok.loadFromGGUFSpm(vocab, ctx.eos_id) catch |e| {
            const msg = std.fmt.bufPrint(&ctx.output_buf, "Tok error: {s}", .{@errorName(e)}) catch "";
            ctx.output_len = msg.len;
            return @intFromPtr(ctx);
        };
    } else {
        ctx.tok.loadFromGGUF(vocab, merges.?, ctx.eos_id) catch |e| {
            const msg = std.fmt.bufPrint(&ctx.output_buf, "Tok error: {s}", .{@errorName(e)}) catch "";
            ctx.output_len = msg.len;
            return @intFromPtr(ctx);
        };
    }
    ctx.tok.bos_token_id = ctx.bos_id;

    // Init model
    const be: Backend = .{ .cpu = &ctx.cpu_be };
    const capped_ctx: u32 = @min(
        fmt.getArchU32(arch_str, "context_length") orelse 2048,
        2048,
    );
    ctx.n_layers = fmt.getArchU32(arch_str, "block_count") orelse 0;
    ctx.n_embd = fmt.getArchU32(arch_str, "embedding_length") orelse 0;

    ctx.mdl = ModelStorage.initFromArch(ctx.arch, gpa, fmt, be, capped_ctx, .f16, .f16, 0, 0, null, 0, 1) catch |e| {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Model init error: {s}", .{@errorName(e)}) catch "";
        ctx.output_len = msg.len;
        return @intFromPtr(ctx);
    };

    ctx.ready = true;
    const msg = std.fmt.bufPrint(&ctx.output_buf, "Loaded: {s} ({d} layers, {d}D, vocab={d})", .{
        ctx.model_name,
        ctx.n_layers,
        ctx.n_embd,
        ctx.vocab_size,
    }) catch "";
    ctx.output_len = msg.len;
    return @intFromPtr(ctx);
}

export fn agave_generate(ctx_ptr: usize, prompt_ptr: [*]const u8, prompt_len: usize, max_tokens: u32) u32 {
    if (ctx_ptr == 0) return 0;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    if (!ctx.ready) {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Model not initialized", .{}) catch "";
        ctx.output_len = msg.len;
        return 0;
    }

    const prompt = prompt_ptr[0..prompt_len];
    _ = max_tokens;

    // Tokenize prompt to verify pipeline
    const tok_iface = ctx.tok.tokenizer();
    const tmpl = ctx.arch.chatTemplate();
    const formatted_owned = tmpl.format(gpa, null, prompt) catch |err| blk: {
        std.log.warn("chat template format failed: {s}, using raw prompt", .{@errorName(err)});
        break :blk null;
    };
    defer if (formatted_owned) |f| gpa.free(f);
    const formatted = formatted_owned orelse prompt;

    const token_ids = tok_iface.encode(formatted) catch {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Tokenize error", .{}) catch "";
        ctx.output_len = msg.len;
        return 0;
    };
    defer gpa.free(token_ids);

    // Report tokenization (forward pass blocked by Zig wasm32 LLVM bug)
    const msg = std.fmt.bufPrint(&ctx.output_buf, "[{s}] Tokenized {d} tokens from prompt. " ++
        "Model: {d} layers, {d}D. " ++
        "Forward pass pending Zig wasm32 codegen fix.", .{
        ctx.model_name,
        token_ids.len,
        ctx.n_layers,
        ctx.n_embd,
    }) catch "";
    ctx.output_len = msg.len;
    return @intCast(token_ids.len);
}

export fn agave_get_output(ctx_ptr: usize, buf_ptr: [*]u8, buf_len: usize) usize {
    if (ctx_ptr == 0) return 0;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    const copy_len = @min(ctx.output_len, buf_len);
    @memcpy(buf_ptr[0..copy_len], ctx.output_buf[0..copy_len]);
    return copy_len;
}

export fn agave_free(ctx_ptr: usize) void {
    if (ctx_ptr == 0) return;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    if (ctx.mdl) |*m| m.deinit();
    ctx.tok.deinit();
    ctx.gguf.deinit();
    gpa.destroy(ctx);
}

export fn agave_alloc(len: usize) usize {
    const buf = gpa.alloc(u8, len) catch return 0;
    return @intFromPtr(buf.ptr);
}

export fn agave_dealloc(ptr: usize, len: usize) void {
    if (ptr == 0 or len == 0) return;
    const slice: [*]u8 = @ptrFromInt(ptr);
    gpa.free(slice[0..len]);
}

// ── Tests ──────────────────────────────────────────────────────────

test "wasmLogFn is a no-op" {
    comptime { _ = &wasmLogFn; }
    wasmLogFn(.debug, .wasm, "test {}", .{42});
}

test "agave_alloc zero returns null pointer" {
    try std.testing.expectEqual(@as(usize, 0), agave_alloc(0));
}

test "agave_alloc and agave_dealloc round-trip" {
    const ptr = agave_alloc(64);
    try std.testing.expect(ptr != 0);
    agave_dealloc(ptr, 64);
}

test "agave_dealloc zero ptr is safe" {
    agave_dealloc(0, 0);
    agave_dealloc(0, 16);
}

test "fuzz: wasm entry pure functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            comptime {
                _ = &wasmLogFn;
                _ = &agave_init;
                _ = &agave_generate;
                _ = &agave_get_output;
                _ = &agave_free;
                _ = &agave_alloc;
                _ = &agave_dealloc;
            }
            // Test alloc/dealloc cycle with random sizes
            const len = smith.valueWithHash(u16, 0);
            if (len == 0) {
                try std.testing.expectEqual(@as(usize, 0), agave_alloc(0));
            } else {
                const ptr = agave_alloc(len);
                if (ptr != 0) agave_dealloc(ptr, len);
            }
        }
    }.f, .{});
}
