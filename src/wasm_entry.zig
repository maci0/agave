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

const std = @import("std");
const format_mod = @import("format/format.zig");
const gguf_mod = @import("format/gguf.zig");
const bpe_mod = @import("tokenizer/bpe.zig");

/// Allocator for WASM — use page allocator (backed by WebAssembly.Memory.grow)
var gpa = std.heap.page_allocator;

/// Context for an active inference session.
const InferenceContext = struct {
    model_data: []const u8,
    output_buf: [8192]u8 = undefined,
    output_len: usize = 0,
    format_loaded: bool = false,
    n_layers: u32 = 0,
    n_embd: u32 = 0,
    vocab_size: u32 = 0,
    arch_name: [64]u8 = undefined,
    arch_len: usize = 0,
};

/// Initialize with model data buffer.
export fn agave_init(model_ptr: [*]const u8, model_len: usize) usize {
    const ctx = gpa.create(InferenceContext) catch return 0;
    ctx.* = .{
        .model_data = model_ptr[0..model_len],
    };

    // Try to parse GGUF header to validate
    if (model_len > 8) {
        // Check GGUF magic: "GGUF" (0x46475547)
        const magic = @as(*const u32, @ptrCast(@alignCast(model_ptr))).*;
        if (magic == 0x46475547) {
            ctx.format_loaded = true;
            // Parse version and tensor count from header
            if (model_len > 24) {
                const version = @as(*const u32, @ptrCast(@alignCast(model_ptr + 4))).*;
                const n_tensors = @as(*const u64, @ptrCast(@alignCast(model_ptr + 8))).*;
                _ = version;

                // Write info to output
                const info = std.fmt.bufPrint(&ctx.output_buf, "GGUF model loaded: {d} tensors, {d} bytes", .{ n_tensors, model_len }) catch "";
                ctx.output_len = info.len;
            }
        }
    }

    return @intFromPtr(ctx);
}

/// Generate tokens from a prompt.
export fn agave_generate(ctx_ptr: usize, prompt_ptr: [*]const u8, prompt_len: usize, max_tokens: u32) u32 {
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    const prompt = prompt_ptr[0..prompt_len];
    _ = max_tokens;

    if (!ctx.format_loaded) {
        const msg = std.fmt.bufPrint(&ctx.output_buf, "Error: No valid GGUF model loaded", .{}) catch "";
        ctx.output_len = msg.len;
        return 0;
    }

    // Placeholder: acknowledge prompt
    const msg = std.fmt.bufPrint(&ctx.output_buf, "[WASM CPU inference not yet connected] Prompt: {s}", .{prompt}) catch "";
    ctx.output_len = msg.len;
    return 1;
}

/// Read generated output text.
export fn agave_get_output(ctx_ptr: usize, buf_ptr: [*]u8, buf_len: usize) usize {
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    const copy_len = @min(ctx.output_len, buf_len);
    @memcpy(buf_ptr[0..copy_len], ctx.output_buf[0..copy_len]);
    return copy_len;
}

/// Free inference context.
export fn agave_free(ctx_ptr: usize) void {
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    gpa.destroy(ctx);
}

/// WASM memory allocation for JS interop.
export fn agave_alloc(len: usize) usize {
    const buf = gpa.alloc(u8, len) catch return 0;
    return @intFromPtr(buf.ptr);
}

/// WASM memory deallocation.
export fn agave_dealloc(ptr: usize, len: usize) void {
    const slice: [*]u8 = @ptrFromInt(ptr);
    gpa.free(slice[0..len]);
}
