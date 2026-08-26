//! Minimal reproducer for Zig 0.16.0 nvptx64 LLVM aliasee bug.
//!
//! Bug: `export fn ... callconv(.kernel)` triggers:
//!   LLVM ERROR: NVPTX aliasee must be a non-kernel function definition
//!
//! Upstream: Not yet filed on github.com/ziglang/zig (NVPTX is Tier 4)
//!   LLVM root cause: PR #81170 (github.com/llvm/llvm-project/pull/81170)
//!     Added alias support to NVPTX but rejects aliases to kernel functions.
//!     emitAliasDeclaration in NVPTXAsmPrinter.cpp checks isKernelFunction()
//!     and calls report_fatal_error. Zig's export mechanism creates an alias
//!     from clean name → mangled name, triggering the check.
//!   Root cause: LLVM 21 NVPTX backend rejects .alias directives that
//!   reference kernel (.entry) functions. Zig's export mechanism creates
//!   an alias from the clean name to the mangled name, which triggers
//!   the error when the function has callconv(.kernel).
//!
//! Workaround applied in agave:
//!   1. Use callconv(.c) instead of callconv(.kernel), compiles to .func
//!   2. Post-process PTX: rename .func definitions to .entry for kernel functions
//!   3. Remove .alias directives for kernel names
//!   4. Ensure forward declarations also use .entry (must match definitions)
//!
//! Additional finding: CUDA 13.0 driver on GB10 (sm_121) rejects sm_100 PTX
//!   (error 218) but accepts sm_90 (forward compat) and sm_120 (native).
//!
//! To reproduce:
//!   zig build-obj -target nvptx64-cuda -mcpu sm_90 -OReleaseFast \
//!       tests/cuda_ptx_repro.zig -fno-emit-bin -femit-asm=repro.ptx
//!
//! Test 1 (uncomment): crashes with callconv(.kernel)
//! Test 2: compiles with callconv(.c), generates .func, needs post-processing

// Test 1: CRASHES, uncomment to reproduce
// export fn crash_kernel(x: [*]f32, n: u32) callconv(.kernel) void {
//     _ = x; _ = n;
// }

// Test 2: WORKS with callconv(.c), generates .func instead of .entry
fn threadIdx() u32 {
    return asm ("mov.u32 %[ret], %tid.x;"
        : [ret] "=r" (-> u32),
    );
}

fn kern_body(input: [*]const f32, output: [*]f32, n: u32) callconv(.c) void {
    const idx = threadIdx();
    if (idx < n) output[idx] = input[idx] + 1.0;
}

comptime {
    @export(&kern_body, .{ .name = "kern" });
}
