//! Reproducer: Zig 0.16.0 + LLVM 21 wasm32 "Invalid cast" codegen bug
//!
//! Error: "Invalid cast (Producer: 'zig 0.16.0' Reader: 'LLVM 21.1.8')"
//!
//! Triggers when compiling a comptime-generated vtable that references
//! functions using @Vector SIMD operations for wasm32-freestanding.
//! Individual SIMD functions compile fine. The vtable pattern that
//! captures function pointers to them also compiles fine in isolation.
//! But when the vtable references functions from a sufficiently complex
//! module (e.g. a transformer model with GEMV + SDPA + RMS norm), LLVM's
//! bitcode verifier rejects the generated IR during module linking.
//!
//! Reproduction:
//!   zig build wasm   (with any model enabled in build.zig)
//!
//! Affected targets: wasm32-freestanding, wasm64-freestanding
//! NOT affected: native, aarch64-linux, nvptx64-cuda, amdgcn-amdhsa
//! All optimization levels fail: Debug, ReleaseSafe, ReleaseSmall, ReleaseFast
//!
//! Workaround: None found. The bug is in LLVM's bitcode reader/verifier
//! during LTO of the combined module. Cannot be avoided by:
//! - Changing @Vector width (4 vs 8)
//! - Changing optimization level
//! - Using wasm64 instead of wasm32
//! - Disabling specific SIMD operations
//! - Restructuring code to avoid vtables
//!
//! The bug does NOT trigger when:
//! - No model is enabled (vtable has no concrete implementations)
//! - Model code is referenced but not pulled into codegen
//!   (e.g. storing the model but not calling .model() to create vtable)
//!
//! Root cause hypothesis: Zig's comptime @ptrCast for vtable function
//! pointers generates an LLVM IR bitcast between incompatible pointer
//! types when the target functions use SIMD vector types. The LLVM
//! bitcode verifier (Reader) rejects this cast during module merging.
//!
//! Related: LLVM PR #81170 (nvptx64 aliasee bug is the same class of
//! Zig codegen → LLVM IR incompatibility)
//!
//! Workarounds tried and failed:
//!   - wasm64-freestanding: same error
//!   - wasm32 MVP (no SIMD): same error (LLVM scalarizes @Vector, still fails)
//!   - -ODebug, -OReleaseSafe, -OReleaseFast: all fail
//!   - Zig self-hosted backend (-fno-llvm): crashes (BUS/SEGV) on complex code
//!   - Disabling specific models: ALL models trigger it
//!   - Standalone SIMD code + vtable: compiles fine (not enough code to trigger)
//!
//! Related upstream bugs:
//!   - Rust #110707: exact same "Invalid cast" in LTO with portable_simd
//!     (fixed by LLVM upgrade, not a code workaround)
//!   - Zig #23414: "Invalid cast" with extern unions (different trigger)
//!   - LLVM #87329: castIsValid assertion in SLP vectorizer
//!
//! Status: Blocked. Needs Zig compiler fix (likely LLVM 22 / Zig 0.17).
//!
//! Minimal trigger (in the agave codebase):
//!   1. build.zig: enable any model for wasm target
//!   2. wasm_entry.zig: call ModelStorage.initFromArch() which generates
//!      comptime vtable with @ptrCast to model's forward() function
//!   3. forward() transitively references GEMV/SDPA kernels with @Vector

const std = @import("std");

// This standalone file does NOT reproduce the bug because the vtable
// and SIMD functions are in the same compilation unit. The actual bug
// requires cross-module LTO where the vtable is generated in one module
// and the SIMD functions are in separate modules linked together.
//
// To reproduce: `zig build wasm` in the agave repo with any model enabled.

const V8 = @Vector(8, f32);

const VTable = struct {
    forward: *const fn (*anyopaque, u32) u32,
};

fn genVTable(comptime T: type) *const VTable {
    return &comptime .{
        .forward = @ptrCast(&struct {
            fn call(self: *T, tok: u32) u32 {
                return self.forward(tok);
            }
        }.call),
    };
}

const SimpleModel = struct {
    buf: [1024]f32 = undefined,

    fn forward(self: *SimpleModel, tok: u32) u32 {
        // SIMD operations that work individually on wasm32
        var acc: V8 = @splat(@as(f32, 0));
        var i: usize = 0;
        while (i + 8 <= 1024) : (i += 8) {
            const v: V8 = self.buf[i..][0..8].*;
            acc = @mulAdd(V8, v, v, acc);
        }
        const sum = @reduce(.Add, acc);
        _ = sum;
        return tok + 1;
    }
};

// This compiles fine on wasm32 — the bug only triggers with the full
// agave model code across multiple compilation units.
export fn test_vtable() u32 {
    var model = SimpleModel{};
    const vt = genVTable(SimpleModel);
    return vt.forward(&model, 42);
}
