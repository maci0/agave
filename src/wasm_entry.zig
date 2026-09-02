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

const gpa = std.heap.page_allocator;

const max_output_bytes = 16384;

const InferenceContext = struct {
    gguf: GGUFFile,
    tok: BpeTokenizer,
    cpu_be: CpuBackend = .{},
    mdl: ?ModelStorage = null,
    output_buf: [max_output_bytes]u8 = undefined,
    output_len: usize = 0,
    ready: bool = false,
    gguf_valid: bool = false,
    eos_id: u32 = 0,
    bos_id: u32 = 0,
    arch: Arch = .gemma3,
    n_layers: u32 = 0,
    n_embd: u32 = 0,
    vocab_size: u32 = 0,
    arch_name: []const u8 = "",
    model_name: []const u8 = "",
};

/// Loads a GGUF model from a raw byte buffer provided by the JS host.
/// Returns a context pointer (as `usize`) on success, or 0 on allocation failure.
/// On parse/init errors the context is still returned with a diagnostic in the output buffer.
export fn agave_init(model_ptr: [*]const u8, model_len: usize) usize {
    const ctx = gpa.create(InferenceContext) catch return 0;
    // Initialize with safe defaults BEFORE attempting fallible parse.
    // gpa.create returns uninitialized memory, if fromBuffer fails and we
    // return early, agave_free would deinit garbage fields.
    ctx.* = .{
        .gguf = undefined,
        .tok = BpeTokenizer.init(gpa),
    };
    ctx.gguf = GGUFFile.fromBuffer(gpa, model_ptr[0..model_len]) catch |e| {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "GGUF parse error: {s}", .{@errorName(e)}) catch "";
        ctx.output_len = msg.len;
        return @intFromPtr(ctx);
    };
    ctx.gguf_valid = true;

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

/// Runs text generation for the given prompt using the loaded model context.
/// Tokenizes the prompt, applies the model's chat template, and returns the
/// number of tokens produced. The textual output is written into the context's
/// internal buffer and can be retrieved with `agave_get_output`. Returns 0 on error.
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
    defer if (formatted_owned) |f| {
        @memset(f, 0);
        gpa.free(f);
    };
    const formatted = formatted_owned orelse prompt;

    const token_ids = tok_iface.encode(formatted) catch {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Tokenize error", .{}) catch "";
        ctx.output_len = msg.len;
        return 0;
    };
    defer {
        @memset(std.mem.sliceAsBytes(token_ids), 0);
        gpa.free(token_ids);
    }

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

/// Copies the context's output string into the caller-supplied buffer.
/// Returns the number of bytes actually written (capped to `buf_len`).
/// Returns 0 if `ctx_ptr` is null.
export fn agave_get_output(ctx_ptr: usize, buf_ptr: [*]u8, buf_len: usize) usize {
    if (ctx_ptr == 0) return 0;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    const copy_len = @min(ctx.output_len, buf_len);
    @memcpy(buf_ptr[0..copy_len], ctx.output_buf[0..copy_len]);
    return copy_len;
}

/// Frees an inference context previously returned by `agave_init`.
/// Releases the model, tokenizer, GGUF data, and the context allocation itself.
/// Safe to call with `ctx_ptr == 0` (no-op).
export fn agave_free(ctx_ptr: usize) void {
    if (ctx_ptr == 0) return;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    if (ctx.mdl) |*m| m.deinit();
    ctx.tok.deinit();
    if (ctx.gguf_valid) ctx.gguf.deinit();
    gpa.destroy(ctx);
}

/// Allocates `len` bytes of linear memory for the WASM host (e.g. to pass a model buffer).
/// Returns the address as `usize`, or 0 if `len` is zero or allocation fails.
/// The caller must free the allocation with `agave_dealloc` using the same length.
export fn agave_alloc(len: usize) usize {
    if (len == 0) return 0;
    const buf = gpa.alloc(u8, len) catch return 0;
    return @intFromPtr(buf.ptr);
}

/// Frees a WASM linear-memory allocation previously obtained from `agave_alloc`.
/// `ptr` and `len` must match the original allocation. Safe to call with `ptr == 0`.
export fn agave_dealloc(ptr: usize, len: usize) void {
    if (ptr == 0 or len == 0) return;
    const slice: [*]u8 = @ptrFromInt(ptr);
    gpa.free(slice[0..len]);
}

// ── Tests ──────────────────────────────────────────────────────────

test "wasmLogFn is a no-op" {
    comptime {
        _ = &wasmLogFn;
    }
    // Must complete without trapping; scope/level are accepted but ignored.
    wasmLogFn(.debug, .wasm, "test {}", .{42});
    wasmLogFn(.err, .wasm, "err {s}", .{"x"});
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
    // Still able to allocate after null deallocs.
    const ptr = agave_alloc(8);
    try std.testing.expect(ptr != 0);
    agave_dealloc(ptr, 8);
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
