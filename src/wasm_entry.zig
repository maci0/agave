//! WASM entry point for browser-based inference.
//!
//! Exports functions callable from JavaScript:
//! - agave_init(model_ptr, model_len) → context handle
//! - agave_generate(ctx, prompt_ptr, prompt_len, max_tokens) → token count
//! - agave_get_output(ctx, buf_ptr, buf_len) → bytes written
//! - agave_last_error(ctx) → WasmError as i32 (0 = ok)
//! - agave_free(ctx) — also frees the model buffer passed to agave_init
//!
//! Model data is passed as a byte buffer from JS (no file I/O). The context
//! takes ownership of that buffer and releases it in `agave_free`.
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

/// Stable codes for `agave_last_error`. Host glue maps these to `AgaveError.code`.
pub const WasmError = enum(i32) {
    ok = 0,
    not_ready = 1,
    tokenize = 2,
    gguf_parse = 3,
    unsupported_arch = 4,
    no_vocab = 5,
    tokenizer = 6,
    model_init = 7,
    invalid_handle = 8,
};

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
    /// Host-allocated model bytes (`agave_alloc`). Freed in `agave_free`.
    model_buf: []u8 = &.{},
    last_error: i32 = 0,
};

fn writeErr(ctx: *InferenceContext, code: WasmError, comptime fmt: []const u8, args: anytype) void {
    ctx.last_error = @intFromEnum(code);
    const msg = std.fmt.bufPrint(&ctx.output_buf, fmt, args) catch "";
    ctx.output_len = msg.len;
}

fn fail(ctx: *InferenceContext, code: WasmError, comptime fmt: []const u8, args: anytype) usize {
    writeErr(ctx, code, fmt, args);
    return @intFromPtr(ctx);
}

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
        .model_buf = @constCast(model_ptr[0..model_len]),
    };
    ctx.gguf = GGUFFile.fromBuffer(gpa, ctx.model_buf) catch |e| {
        return fail(ctx, .gguf_parse, "GGUF parse error: {s}", .{@errorName(e)});
    };
    ctx.gguf_valid = true;

    const fmt = ctx.gguf.format();
    const arch_str = fmt.getMetaStr("general.architecture") orelse "unknown";
    ctx.arch = Arch.detect(arch_str) orelse {
        return fail(ctx, .unsupported_arch, "Unsupported arch: {s}", .{arch_str});
    };
    ctx.arch_name = arch_str;
    ctx.model_name = fmt.getMetaStr("general.name") orelse arch_str;

    // Load tokenizer
    const vocab = fmt.getVocab() orelse {
        return fail(ctx, .no_vocab, "No vocab in GGUF", .{});
    };
    ctx.eos_id = fmt.getMetaU32("tokenizer.ggml.eos_token_id") orelse ctx.arch.defaultEos();
    ctx.bos_id = fmt.getMetaU32("tokenizer.ggml.bos_token_id") orelse 0;
    ctx.vocab_size = @intCast(vocab.len);

    const merges = fmt.getMerges();
    if (merges == null or ctx.arch == .gemma3 or ctx.arch == .gemma4) {
        ctx.tok.loadFromGGUFSpm(vocab, ctx.eos_id) catch |e| {
            return fail(ctx, .tokenizer, "Tok error: {s}", .{@errorName(e)});
        };
    } else {
        ctx.tok.loadFromGGUF(vocab, merges.?, ctx.eos_id) catch |e| {
            return fail(ctx, .tokenizer, "Tok error: {s}", .{@errorName(e)});
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
        return fail(ctx, .model_init, "Model init error: {s}", .{@errorName(e)});
    };

    ctx.ready = true;
    ctx.last_error = 0;
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
export fn agave_generate(ctx_ptr: usize, prompt_ptr: [*]allowzero const u8, prompt_len: usize, max_tokens: u32) u32 {
    if (ctx_ptr == 0) return 0;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    if (!ctx.ready) {
        writeErr(ctx, .not_ready, "Model not initialized", .{});
        return 0;
    }

    // len 0 must not slice a null host pointer: JS `generate("")` passes ptr=0.
    if (prompt_len != 0 and @intFromPtr(prompt_ptr) == 0) {
        writeErr(ctx, .tokenize, "Tokenize error", .{});
        return 0;
    }
    const prompt: []const u8 = if (prompt_len == 0) &.{} else @as([*]const u8, @ptrCast(prompt_ptr))[0..prompt_len];
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
        writeErr(ctx, .tokenize, "Tokenize error", .{});
        return 0;
    };
    defer {
        @memset(std.mem.sliceAsBytes(token_ids), 0);
        gpa.free(token_ids);
    }

    // Report tokenization (forward pass blocked by Zig wasm32 LLVM bug)
    ctx.last_error = 0;
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

/// Last `WasmError` for `ctx_ptr` as i32. `0` means the previous init/generate
/// succeeded. Null handle returns `invalid_handle` so the host can distinguish
/// "no context" from "ok" without string-matching the output buffer.
export fn agave_last_error(ctx_ptr: usize) i32 {
    if (ctx_ptr == 0) return @intFromEnum(WasmError.invalid_handle);
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    return ctx.last_error;
}

/// Frees an inference context previously returned by `agave_init`.
/// Releases the model, tokenizer, GGUF data, the host model buffer, and the
/// context allocation itself. Safe to call with `ctx_ptr == 0` (no-op).
export fn agave_free(ctx_ptr: usize) void {
    if (ctx_ptr == 0) return;
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    if (ctx.mdl) |*m| m.deinit();
    ctx.tok.deinit();
    if (ctx.gguf_valid) ctx.gguf.deinit();
    // GGUF borrows this buffer; free it only after gguf.deinit().
    if (ctx.model_buf.len > 0) gpa.free(ctx.model_buf);
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

test "agave_last_error null handle is invalid_handle" {
    try std.testing.expectEqual(
        @intFromEnum(WasmError.invalid_handle),
        agave_last_error(0),
    );
}

test "agave_generate null handle returns 0" {
    const dummy: [1]u8 = .{0};
    try std.testing.expectEqual(@as(u32, 0), agave_generate(0, &dummy, dummy.len, 8));
    try std.testing.expectEqual(
        @intFromEnum(WasmError.invalid_handle),
        agave_last_error(0),
    );
}

test "agave_get_output null handle returns 0" {
    var out: [8]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), agave_get_output(0, &out, out.len));
}

test "agave_generate on unready context sets not_ready" {
    const ctx = try std.testing.allocator.create(InferenceContext);
    defer std.testing.allocator.destroy(ctx);
    ctx.* = .{
        .gguf = undefined,
        .tok = BpeTokenizer.init(std.testing.allocator),
        .ready = false,
    };
    defer ctx.tok.deinit();
    const dummy: [1]u8 = .{0};
    try std.testing.expectEqual(@as(u32, 0), agave_generate(@intFromPtr(ctx), &dummy, dummy.len, 8));
    try std.testing.expectEqual(@intFromEnum(WasmError.not_ready), agave_last_error(@intFromPtr(ctx)));
    const null_ptr: [*]allowzero const u8 = @ptrFromInt(0);
    try std.testing.expectEqual(@as(u32, 0), agave_generate(@intFromPtr(ctx), null_ptr, 0, 8));
    try std.testing.expectEqual(@intFromEnum(WasmError.not_ready), agave_last_error(@intFromPtr(ctx)));
    var out: [64]u8 = undefined;
    const n = agave_get_output(@intFromPtr(ctx), &out, out.len);
    try std.testing.expectEqualStrings("Model not initialized", out[0..n]);
}

test "agave_init parse failure is recoverable and agave_free owns the buffer" {
    const ptr = agave_alloc(64);
    try std.testing.expect(ptr != 0);
    const slice: [*]u8 = @ptrFromInt(ptr);
    @memset(slice[0..64], 0);
    const ctx = agave_init(slice, 64);
    try std.testing.expect(ctx != 0);
    try std.testing.expectEqual(@intFromEnum(WasmError.gguf_parse), agave_last_error(ctx));
    var out: [64]u8 = undefined;
    const n = agave_get_output(ctx, &out, out.len);
    try std.testing.expect(std.mem.startsWith(u8, out[0..n], "GGUF parse error:"));
    // Must not agave_dealloc(ptr): init took ownership. Double-free would trap.
    agave_free(ctx);
}

test "fuzz: wasm entry pure functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            comptime {
                _ = &wasmLogFn;
                _ = &agave_init;
                _ = &agave_generate;
                _ = &agave_get_output;
                _ = &agave_last_error;
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
