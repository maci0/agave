# Coding Conventions

**Analysis Date:** 2026-03-21

## Naming Patterns

**Files:**
- `snake_case` for all files and directories (e.g., `cpu.zig`, `gemv.zig`, `thread_pool.zig`)

**Functions:**
- `camelCase` for all functions and methods (e.g., `rmsNorm()`, `rope()`, `bf16ToF32()`, `mxfp4Lookup()`, `gemv()`, `dequantToF32()`)
- Dispatch functions use `inline else` pattern: `pub fn gemv(self: *Backend, ...)` dispatches via `switch(self.*) { inline else => |*be| be.gemv(...) }`
- Getter functions don't use `get` prefix unless there's significant computation (e.g., `backendInfo()` not `getBackendInfo()`)

**Variables:**
- `snake_case` for local variables and struct fields (e.g., `kv_bytes_per_layer`, `n_workers`, `task_total`, `local_gen`, `block_size`)
- Field names are descriptive: `mapped_data`, `current_offset`, `init_count`, `avail_mem`
- Loop variables: short names acceptable in tight loops (`i`, `j`, `k`, `n`, `b`, `g`)

**Types:**
- `PascalCase` for structs, enums, unions, error sets (e.g., `TensorData`, `GGMLType`, `Backend`, `ThreadPool`, `KvCache`, `CacheBlock`, `SeqBlockTable`)
- Type functions (returning `type`) also use `PascalCase`

**Constants:**
- `snake_case` for all `const` declarations, including module-level tuning constants (e.g., `quant_block_elems`, `print_buf_size`, `default_port`, `max_workers`, `min_grain`, `softmax_cpu_threshold`)
- Named constants MUST be extracted from inline literals — no magic numbers allowed
- Constants are placed at module level for visibility and configurability

## Code Style

**Formatting:**
- Zig stdlib default: 4-space indentation (consistent across all files)
- Max line length: no hard limit; clarity is preferred over wrapping
- Use blank lines between logical sections (imports, constants, types, functions)

**Linting:**
- No external linter enforced; rely on `zig fmt` for auto-formatting
- `zig build` runs `zig fmt` check on changed files

**Explicit Types:**
- Always specify types explicitly in function signatures (no inference at boundaries)
- Use explicit type annotations for quantized data: `f32`, `f16`, `bf16`, `i8`, `u8`, etc.
- Example: `pub fn bf16ToF32(val: u16) f32` — types clear at call site

## Import Organization

**Order:**
1. Standard library: `const std = @import("std");`
2. Builtin: `const builtin = @import("builtin");`
3. Build options: `const build_options = @import("build_options");`
4. Project imports (ordered by relative path depth or logical grouping)

**Path Aliases:**
- No global path aliases; use relative `@import` paths
- Dispatcher pattern: high-level modules import the dispatcher (e.g., `@import("backend/backend.zig")`), never implementation directly
- Re-exports used to avoid exposing internal types: `pub const DType = @import("../format/format.zig").DType;`

**Example (from `src/backend/backend.zig`):**
```zig
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const TensorData = struct { data: [*]const u8, dtype: DType };
pub const DType = @import("../format/format.zig").DType;
pub const detectSystemMem = @import("cpu.zig").detectSystemMem;
```

## Error Handling

**Patterns:**
- Use explicit error sets: `error.DeviceOutOfMemory`, `error.OffsetOutOfBounds`, `error.FileTooSmall`
- Propagate errors up with `try`: `const val = try self.readU32(off);`
- Use `catch` with logging for non-critical paths: `catch |err| { std.log.err("detail: {}", .{err}); return err; }`
- Silent error swallowing (`catch {}`) is forbidden except in shutdown paths

**Error Set Definitions:**
- Define at module or function level, near where they're used
- Example (from `gguf.zig`): return statements use inferred error types `!u64` from `return error.OffsetOutOfBounds;`

**Try/Catch Pattern:**
- `try` is preferred for error propagation in regular paths
- Explicit catch blocks for recovery: `const x = foo() catch |err| { cleanup(); return err; };`
- Never swallow with `catch {}` in production code

## Logging

**Framework:** `std.log` from Zig stdlib

**Patterns:**
- Log levels: `.debug`, `.info`, `.warn`, `.err`
- Scoped logs use `@import("std").log.scoped()` or implicit scope from file context
- Performance-critical paths use `.perf` scope (gated by `if (g_debug)` at callsite)
- Example: `std.log.warn("warning detail: {}", .{val});`

**Output Control:**
- Global flags in `main.zig`: `g_debug`, `g_quiet`, `g_verbose`, `g_color`, `g_tty`
- Use `eprint()` helper for immediate stderr (for CLI feedback)
- Use `print()` helper for stdout with custom formatting buffer
- Tests use `std.testing.expectEqual()`, not print statements

## Comments

**When to Comment:**
- Document non-obvious algorithm details (e.g., bit-packing semantics, quantization block structure)
- Explain invariants and constraints (e.g., "zero allocations in hot path")
- DON'T comment obvious code ("increment i by 1" not needed)

**Doc Comments (`///`):**
- Required on all public functions and structs
- Include: purpose, parameter semantics (especially ownership), return value meaning, error conditions
- Example (from `quant.zig`):
  ```zig
  /// Convert a BF16 value (stored as u16) to f32.
  /// BF16 shares f32's exponent range; conversion is a 16-bit left shift.
  pub inline fn bf16ToF32(val: u16) f32 {
      return @bitCast(@as(u32, val) << 16);
  }
  ```

**Inline Comments (`//`):**
- Use for implementation details that aren't self-documenting
- Example: `// Wake workers by bumping generation`
- Keep comments close to the code they describe

**Module-Level Comments (`//!`):**
- First lines of each file explain purpose and role
- Example (from `thread_pool.zig`): `//! Lightweight thread pool for parallel-for workloads.`

## Function Design

**Size:**
- Small is preferred; aim for <50 lines per function
- Dispatch wrappers (inline else) can be exceptions
- Large functions should be broken into helpers with clear names

**Parameters:**
- Explicit names; no positional inference (e.g., `fn rope(x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32)`)
- Use structs for >4 related parameters (e.g., `DeltaNetParams`, `BackendInfo`)
- Pass allocators explicitly; never hide allocators in globals
- Output buffers are `[*]T` (pointers) or `[]T` (slices) — be consistent

**Return Values:**
- Error types explicit: `!Result` for fallible, `Result` for infallible
- Use error propagation (`try`) unless recovery needed
- Multiple return values via struct or named return type
- Example: `fn readMetaValue(self: *GGUFFile, off: usize) !struct { val: MetaValue, len: usize }`

**Inline Keyword:**
- `inline fn` or `inline` for hot-path math helpers (e.g., `bf16ToF32()`, `mxfp4Lookup()`, `fp8e4m3ToF32()`)
- Use for comptime dispatch: `inline else` in backend switch statements
- Avoid on large functions or one-time init code (wastes code size)

## Memory Management

**Allocator Passing:**
- All functions requiring memory accept `allocator: Allocator` as explicit parameter
- Never use global or hidden allocators
- No `std.heap.page_allocator` in hot paths (initialization only)

**Cleanup Patterns:**
- `defer` immediately after resource acquisition: `var obj = try init(allocator); defer obj.deinit();`
- `errdefer` for partial cleanup on error paths: `var temp = try allocator.alloc(u8, n); errdefer allocator.free(temp);`
- Never rely on manual cleanup in catch blocks (error-prone)

**Structs with Allocation:**
- Require `deinit()` method: `pub fn deinit(self: *MyType) void`
- Clear ownership semantics: `init(allocator)` vs `initCopy(allocator, data)`
- Document transfer of ownership in doc comments

**Example (from `kvcache/manager.zig`):**
```zig
pub fn allocKvCache(allocator: Allocator, n_layers: usize, kv_bytes_per_layer: usize) !KvCache {
    const keys = try allocator.alloc([]u8, n_layers);
    errdefer allocator.free(keys);
    const values = try allocator.alloc([]u8, n_layers);
    errdefer allocator.free(values);

    var init_count: usize = 0;
    errdefer {
        for (0..init_count) |i| {
            allocator.free(keys[i]);
            allocator.free(values[i]);
        }
    }

    for (0..n_layers) |i| {
        keys[i] = try allocator.alloc(u8, kv_bytes_per_layer);
        values[i] = try allocator.alloc(u8, kv_bytes_per_layer);
        init_count = i + 1;
    }

    return .{ .keys = keys, .values = values };
}
```

## Module Design

**Exports:**
- Use `pub` only for intended API surface; keep internals private
- Re-export dependencies for convenience (e.g., `pub const DType = @import("format.zig").DType;`)
- Hide backend-specific types (e.g., `CUcontext` private to `cuda.zig`)

**Dispatcher Pattern:**
- High-level modules (main.zig, models) import dispatcher, never implementations
- Dispatcher uses tagged union with `inline else` for zero-overhead dispatch
- Each backend implements the same interface (duck-typing + comptime)

**Example Pattern (from `backend.zig`):**
```zig
pub const Backend = union(Enum) {
    cpu: CpuBackend,
    metal: MetalBackend,
    // ... other variants

    pub fn gemv(self: *Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        switch (self.*) {
            inline else => |*be| be.gemv(x, w, y, n, k),
        }
    }
};
```

**Zig Builtins Preference:**
- Use `@Vector`, `@reduce`, `@splat` for SIMD instead of manual loops
- Use `@memcpy`, `@memset` for bulk ops
- Use `@bitCast`, `@intCast` for type conversions
- Use `@exp`, `@sqrt`, `@mulAdd` for math (avoid libcalls on nvptx)
- Example (from `quant.zig`): `return @bitCast(@as(u32, val) << 16);`

## Concurrency

**Constraints:**
- No manual `std.Thread.spawn()` in inference code (use centralized thread pool)
- No locks in hot paths; prefer atomics: `std.atomic.Value(T)`
- Use `std.Thread.Futex` for sleep/wake synchronization

**Thread Pool Pattern:**
- All parallel work goes through `ThreadPool` in `src/thread_pool.zig`
- Example (from `thread_pool.zig`): `parallelFor(total, grain, ctx, func)` splits work into chunks
- Workers capture pool by pointer (must be at final memory location before `spawn()`)

## Comptime Usage

**Patterns:**
- Backend/model selection: `comptime` checks on `build_options` and `builtin`
- Cost-amortized dispatch: `comptime` in if/else branches to eliminate dead code
- Compile-time lookup tables: `const table = blk: { ... break :blk table; };`
- Format detection: `if (comptime builtin.os.tag == .macos)` gates platform-specific code

**Examples:**
- `fp8e4m3_lut`: 256-entry f32 table computed at comptime
- `iq4nl_table`: i8 dequant LUT (hardcoded, comptime-verified)
- Backend dispatch: `inline else` on tagged union eliminates VTable calls

## Hot-Path Constraints

**Zero-Cost Axioms:**
- No allocations (`allocator.alloc()`) in token generation loop
- No syscalls in inner loops
- No locks in inference path
- All I/O (model loading, streaming) happens outside the hot path

**Verification:**
- Mark hot-path functions with comments: `// Hot path: no allocs, locks, syscalls`
- Use `std.debug.assert()` for invariants
- Benchmark any changes to backend/model forward() functions

---

*Convention analysis: 2026-03-21*
