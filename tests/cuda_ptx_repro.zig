//! Minimal reproducer for Zig 0.16.0 nvptx64 LLVM aliasee bug.
//!
//! Bug: `export fn ... callconv(.kernel)` triggers:
//!   LLVM ERROR: NVPTX aliasee must be a non-kernel function definition
//!
//! Workaround: use callconv(.c) and post-process PTX (.func → .entry).
//!
//! To test: zig build-obj -target nvptx64-cuda -mcpu sm_90 -OReleaseFast \
//!          tests/cuda_ptx_repro.zig -fno-emit-bin -femit-asm=repro.ptx
//!
//! Expected: LLVM ERROR with callconv(.kernel)
//! With .c: compiles, but generates .func (not .entry) — needs post-processing

// Test 1: This CRASHES with callconv(.kernel)
// Uncomment to reproduce:
// export fn crash_kernel(x: [*]f32, n: u32) callconv(.kernel) void {
//     _ = x; _ = n;
// }

// Test 2: This WORKS with callconv(.c) but generates .func instead of .entry
export fn workaround_kernel(x: [*]f32, n: u32) callconv(.c) void {
    _ = x;
    _ = n;
}

// Test 3: Kernel with PTX inline asm (common pattern)
fn threadIdx() u32 {
    return asm ("mov.u32 %[ret], %tid.x;"
        : [ret] "=r" (-> u32),
    );
}

export fn asm_kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.c) void {
    const idx = threadIdx();
    if (idx >= n) return;
    output[idx] = input[idx] + 1.0;
}
