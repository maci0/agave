//! WASM entry point for browser-based inference.
//!
//! Exports functions callable from JavaScript:
//! - agave_init(model_ptr, model_len) → context handle
//! - agave_generate(ctx, prompt_ptr, prompt_len, max_tokens) → token count
//! - agave_get_output(ctx, buf_ptr, buf_len) → bytes written
//! - agave_free(ctx)
//!
//! The JS glue (web/agave.js) loads the WASM module, provides WebGPU
//! API imports, and manages model data via fetch or IndexedDB.

const std = @import("std");

/// Allocator for WASM — use page allocator (backed by WebAssembly.Memory.grow)
var gpa = std.heap.page_allocator;

/// Context for an active inference session.
const InferenceContext = struct {
    model_data: []const u8,
    output_buf: [4096]u8 = undefined,
    output_len: usize = 0,
};

/// Initialize with model data buffer.
/// Returns opaque context pointer (as usize for JS interop).
export fn agave_init(model_ptr: [*]const u8, model_len: usize) usize {
    const ctx = gpa.create(InferenceContext) catch return 0;
    ctx.* = .{
        .model_data = model_ptr[0..model_len],
    };
    return @intFromPtr(ctx);
}

/// Generate tokens from a prompt.
/// Returns number of tokens generated.
export fn agave_generate(ctx_ptr: usize, prompt_ptr: [*]const u8, prompt_len: usize, max_tokens: u32) u32 {
    const ctx: *InferenceContext = @ptrFromInt(ctx_ptr);
    const prompt = prompt_ptr[0..prompt_len];

    // Placeholder: echo prompt back as output
    const copy_len = @min(prompt.len, ctx.output_buf.len);
    @memcpy(ctx.output_buf[0..copy_len], prompt[0..copy_len]);
    ctx.output_len = copy_len;
    _ = max_tokens;

    return 1; // placeholder token count
}

/// Read generated output text.
/// Returns number of bytes written to buf.
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
