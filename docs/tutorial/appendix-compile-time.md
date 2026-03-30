# Appendix: Compile-Time Optimization

Zig's `comptime` feature executes code **at compile time**, generating optimized runtime code with zero overhead. Agave uses this extensively for lookup tables, feature detection, and type-specialized dispatch.

## comptime Basics

**comptime** means "computed at compile time". The compiler evaluates the expression during compilation, and the result is baked into the binary.

```zig
const table_size = 256;  // Regular constant
const doubled = comptime table_size * 2;  // Computed at compile time (512)

// The binary contains the value 512, not the multiplication
```

**When to use comptime:**

- Building lookup tables
- Feature detection based on target platform
- Type-level computations
- Format string validation

## Lookup Tables

Pre-computing values at compile time eliminates runtime arithmetic.

### FP8 E4M3 Dequantization Table

**Naive approach** (runtime conversion):

```zig
pub fn fp8e4m3ToF32(val: u8) f32 {
    // Extract sign, exponent, mantissa from 8-bit value
    const sign = (val >> 7) & 1;
    const exp = (val >> 3) & 0xF;
    const mant = val & 0x7;

    // Compute float value
    const bias = 7;
    const sign_mult = if (sign == 1) -1.0 else 1.0;

    if (exp == 0) {
        // Subnormal
        return sign_mult * (@as(f32, @floatFromInt(mant)) / 8.0) * std.math.pow(f32, 2.0, 1 - bias);
    } else {
        // Normal
        const frac = 1.0 + (@as(f32, @floatFromInt(mant)) / 8.0);
        return sign_mult * frac * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(exp)) - bias);
    }
}
```

**Cost per call:** ~20-30 instructions (bit shifts, branches, floating-point arithmetic, `pow()` call).

**Optimized approach** (comptime lookup table):

```zig
// Build 256-entry lookup table at compile time
const fp8e4m3_lut: [256]f32 = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| {
        table[i] = fp8ToF32Internal(@intCast(i));  // Computed once at compile time
    }
    break :blk table;
};

// Runtime dequantization is a single array lookup
pub inline fn fp8e4m3ToF32(val: u8) f32 {
    return fp8e4m3_lut[val];
}
```

**Cost per call:** 1 instruction (load from `.rodata` section).

**Speedup:** 20-30× faster for the dequantization itself. In a full GEMV, this saves ~5-10% total time.

### comptime Block Syntax

```zig
const table = blk: {
    var result: [N]T = undefined;
    // ... compute result ...
    break :blk result;  // Return from comptime block
};
```

**Key points:**

- `blk:` is a labeled block
- `break :blk value` returns from the block
- The entire block runs at compile time
- `result` becomes a compile-time constant

### IQ4_NL Dequantization Table

**IQ4_NL** uses a fixed dequantization table (not computed, but verified at comptime):

```zig
const iq4nl_values = [_]i8{
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
};

pub fn iq4nlToF32(nibble: u4, scale: f32) f32 {
    return @as(f32, @floatFromInt(iq4nl_values[nibble])) * scale;
}
```

**Why a table?** IQ4_NL uses **non-linear quantization** — the step sizes aren't uniform. Small values have fine steps, large values have coarse steps. This gives better accuracy than linear Q4.

**comptime verification:**

```zig
comptime {
    std.debug.assert(iq4nl_values.len == 16);  // 4-bit = 16 values
    for (iq4nl_values, 0..) |v, i| {
        if (i > 0) {
            std.debug.assert(v > iq4nl_values[i - 1]);  // Strictly increasing
        }
    }
}
```

This runs at compile time. If the table is malformed, **compilation fails**.

## Feature Detection

Zig's `builtin` module provides platform information at comptime.

### Target OS Detection

```zig
const builtin = @import("builtin");

pub fn initBackend() !Backend {
    if (comptime builtin.os.tag == .macos) {
        return Backend{ .metal = try MetalBackend.init() };
    } else if (comptime builtin.os.tag == .linux) {
        return Backend{ .vulkan = try VulkanBackend.init() };
    } else {
        return Backend{ .cpu = try CpuBackend.init() };
    }
}
```

**Dead code elimination:** The compiler generates **only the code for the target platform**. If compiling for macOS, the Linux and CPU branches are **completely removed** from the binary.

### CPU Feature Detection

```zig
const has_avx2 = comptime builtin.cpu.features.isEnabled(@import("std").Target.x86.Feature.avx2);

pub fn gemv(...) void {
    if (comptime has_avx2) {
        gemvAVX2(...);  // 256-bit SIMD
    } else {
        gemvSSE2(...);  // 128-bit SIMD fallback
    }
}
```

**Benefit:** No runtime CPU detection overhead. The compiler knows at build time which CPU features are available (based on `-mcpu` flag or target triple).

### Build Options

```zig
// build.zig
const backend_options = b.addOptions();
backend_options.addOption(bool, "enable_metal", true);
backend_options.addOption(bool, "enable_cuda", false);

// backend.zig
const build_options = @import("build_options");

pub const MetalBackend = if (build_options.enable_metal)
    @import("metal.zig").MetalBackend
else
    NullBackend;
```

**Effect:** If `enable_metal=false`, the Metal backend is **not compiled at all** — `@import("metal.zig")` never happens, reducing binary size and compile time.

## @embedFile for Kernel Source

Shader source code can be embedded directly into the binary at compile time.

### Metal Shader Embedding

```zig
// Concatenate all MSL files at compile time
const msl_source = @embedFile("kernels/metal/common.metal") ++
    @embedFile("kernels/metal/elementwise.metal") ++
    @embedFile("kernels/metal/norm.metal") ++
    @embedFile("kernels/metal/rope.metal") ++
    @embedFile("kernels/metal/gemv.metal") ++
    @embedFile("kernels/metal/sdpa.metal");

pub fn init(allocator: Allocator) !MetalBackend {
    // Compile MSL source at runtime (driver compiles to GPU bytecode)
    const library = device.newLibraryWithSource(msl_source, null, &err);
    // ...
}
```

**Benefits:**

1. **Single binary:** No need to ship separate `.metal` files
2. **No file I/O:** No `std.fs.cwd().openFile()` at runtime
3. **Compile-time concatenation:** Multiple files merged into one string at zero cost

**Alternative (runtime file loading):**

```zig
// BAD: Runtime file I/O
const file = try std.fs.cwd().openFile("shaders/gemv.metal", .{});
defer file.close();
const source = try file.readToEndAlloc(allocator, 1024 * 1024);
defer allocator.free(source);
```

**Problems:**

- Requires shipping shader files alongside binary
- File path resolution (where is the binary run from?)
- Runtime allocation + I/O
- Error handling (file not found, permission denied)

**@embedFile eliminates all of these.**

### SPIR-V Binary Embedding

Vulkan uses pre-compiled SPIR-V bytecode:

```zig
const gemv_spirv = @embedFile("kernels/vulkan/gemv.spv");

pub fn init() !VulkanBackend {
    const shader_module = vk.createShaderModule(device, .{
        .code_size = gemv_spirv.len,
        .code = @ptrCast(gemv_spirv.ptr),
    });
    // ...
}
```

**SPIR-V is binary data** — `@embedFile` works with any file type, not just text.

## Type-Specialized Functions

Generate different code for each type at compile time.

### Generic Dequantization

```zig
pub fn dequantize(comptime T: type, quant: []const u8, output: []f32) void {
    switch (T) {
        Q4_0 => dequantizeQ4_0(quant, output),
        Q8_0 => dequantizeQ8_0(quant, output),
        BF16 => dequantizeBF16(quant, output),
        else => @compileError("Unsupported quantization type"),
    }
}

// Usage:
dequantize(Q4_0, quant_data, f32_output);  // Compiles to direct call to dequantizeQ4_0
```

**No runtime dispatch** — the switch is resolved at compile time, and only the relevant function is called.

### Tagged Union Dispatch (inline else)

```zig
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    // ...

    pub fn gemv(self: Backend, ...) void {
        switch (self) {
            inline else => |be| be.gemv(...),  // Expands to separate case per variant
        }
    }
};
```

**What `inline else` does:**

```zig
// Expands to:
switch (self) {
    .cpu => |be| be.gemv(...),
    .metal => |be| be.gemv(...),
    .vulkan => |be| be.gemv(...),
    .cuda => |be| be.gemv(...),
    .rocm => |be| be.gemv(...),
}
```

**Benefit:** Compiler sees all calls, can inline them. No function pointer indirection.

## Format String Validation

Compile-time format string checking prevents runtime errors.

```zig
// GOOD: Format string validated at compile time
std.log.info("Temperature: {d}, Tokens: {d}", .{temp, n_tokens});

// BAD: Wrong number of arguments — compile error!
std.log.info("Temperature: {d}, Tokens: {d}", .{temp});
// error: expected 2 format arguments, found 1

// BAD: Wrong type specifier — compile error!
std.log.info("Temperature: {d}", .{"0.5"});
// error: cannot format string with 'd' (expected number)
```

**C comparison:**

```c
printf("Temperature: %d, Tokens: %d\n", temp);  // Runtime crash or garbage
```

Zig catches this at compile time.

## Comptime Assertions

Validate assumptions at compile time.

### Array Size Validation

```zig
const quant_block_elems = 32;
const Q4_0_Block = extern struct {
    scale: f16,
    quants: [16]u8,  // 16 bytes = 32 nibbles
};

comptime {
    std.debug.assert(@sizeOf(Q4_0_Block) == 18);  // 2 + 16 = 18 bytes
    std.debug.assert(16 * 2 == quant_block_elems);  // 16 bytes × 2 nibbles/byte
}
```

**Effect:** If you change `quants` to `[15]u8`, compilation fails with an assertion error.

### Alignment Validation

```zig
comptime {
    std.debug.assert(@alignOf(KVCache) == 64);  // Must be cache-line aligned
}
```

### Type Size Checks

```zig
comptime {
    std.debug.assert(@sizeOf(f32) == 4);
    std.debug.assert(@sizeOf(bf16) == 2);
    std.debug.assert(@sizeOf(V8) == 32);  // 8 × f32
}
```

**Why?** If porting to a weird platform where `f32` isn't 32 bits, these fail at compile time instead of producing silent data corruption at runtime.

## Practical Examples

### MXFP4 Lookup Table

```zig
// MXFP4 uses E2M1 format (2-bit exponent, 1-bit mantissa)
// 4-bit nibble → 16 possible values
const mxfp4_lut = comptime blk: {
    var table: [16]f32 = undefined;
    for (0..16) |nibble| {
        const exp = (nibble >> 1) & 0x3;  // 2 bits
        const mant = nibble & 0x1;        // 1 bit

        // E2M1 decoding
        if (exp == 0 and mant == 0) {
            table[nibble] = 0.0;
        } else {
            const frac = 1.0 + @as(f32, @floatFromInt(mant));  // 1.0 or 1.5
            table[nibble] = frac * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(exp)) - 1.0);
        }
    }
    break :blk table;
};

pub fn mxfp4ToF32(nibble: u4, scale_fp8: u8) f32 {
    const scale = fp8e4m3ToF32(scale_fp8);  // Also a comptime LUT!
    return mxfp4_lut[nibble] * scale;
}
```

**Two-level lookup:** nibble → base value (comptime), scale (comptime) → final value (runtime multiply).

### Quantization Block Sizes

```zig
pub fn blockBytes(comptime dtype: DType) usize {
    return switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
        .q4_0 => 18,
        .q8_0 => 34,
        .q4_k => 144,
        .q6_k => 210,
        .mxfp4 => 17,
        // ...
    };
}

// Usage: computed at compile time
const bytes_per_block = comptime blockBytes(.q4_0);  // 18
```

**Benefit:** Function is `comptime`, so it can be used in other comptime expressions:

```zig
const num_blocks = (total_bytes + comptime blockBytes(dtype) - 1) / comptime blockBytes(dtype);
```

## Performance Impact

**FP8 dequantization** (measured on Apple M4):

| Method | Cycles/call | Speedup |
| ------ | ----------- | ------- |
| Runtime computation | ~30 cycles | 1× |
| Comptime LUT | ~1 cycle | 30× |

**Binary size impact:**

| Feature | Binary size increase |
| ------- | -------------------- |
| FP8 E4M3 LUT (256 × 4 bytes) | +1 KB |
| MXFP4 LUT (16 × 4 bytes) | +64 bytes |
| IQ4_NL LUT (16 × 1 byte) | +16 bytes |
| Embedded Metal shaders (~50 KB source) | +50 KB |

**Trade-off:** Small binary size increase for significant runtime speedup.

## Common Patterns

### Conditional Compilation

```zig
const use_simd = comptime builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64;

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    if (comptime use_simd) {
        return dotProductSIMD(a, b);
    } else {
        return dotProductScalar(a, b);
    }
}
```

### Type-Generic Containers

```zig
pub fn RingBuffer(comptime T: type, comptime size: usize) type {
    return struct {
        data: [size]T,
        head: usize = 0,

        pub fn push(self: *@This(), item: T) void {
            self.data[self.head] = item;
            self.head = (self.head + 1) % size;
        }
    };
}

// Usage:
var conv_state = RingBuffer(f32, 4).init();  // 4-element f32 ring buffer
```

**Each instantiation** (`RingBuffer(f32, 4)`, `RingBuffer(u32, 8)`) generates **separate specialized code**.

### Compile-Time String Manipulation

```zig
const kernel_name = "gemv_" ++ dtype_name;  // Comptime string concat

pub fn loadKernel(comptime dtype: DType) !Pipeline {
    const name = comptime kernelName(dtype);  // e.g., "gemv_q4_0"
    return library.newFunctionWithName(name);
}

fn kernelName(comptime dtype: DType) []const u8 {
    return "gemv_" ++ @tagName(dtype);  // "gemv_" + "q4_0" → "gemv_q4_0"
}
```

## Anti-Patterns

### Don't Overuse comptime

**BAD:** Using comptime for simple runtime values

```zig
const temperature = comptime 0.7;  // Pointless — it's already a constant
```

**GOOD:** Just use `const`

```zig
const temperature: f32 = 0.7;
```

### Don't Compute Heavy Things at Comptime

**BAD:** Large nested loops at comptime slow down compilation

```zig
const huge_table = comptime blk: {
    var table: [1000000]f32 = undefined;
    for (0..1000000) |i| {
        table[i] = expensiveComputation(i);  // Runs at compile time!
    }
    break :blk table;
};
```

**Effect:** Compilation takes minutes instead of seconds.

**Better:** Use codegen (separate script generates the table, output checked into repo) or load from file at runtime.

### Don't Use comptime for Mutable State

**WRONG:** This doesn't work

```zig
var comptime_counter: usize = 0;  // Error: comptime variables can't be var

pub fn getNextId() usize {
    comptime {
        comptime_counter += 1;  // Error: comptime mutation not allowed
        return comptime_counter;
    }
}
```

**comptime is for constants**, not mutable state.

## Best Practices

1. **Use comptime for lookup tables** when the table is small (<10 KB) and frequently accessed
2. **Use comptime for feature detection** to eliminate dead code
3. **Use @embedFile for resources** that ship with the binary
4. **Use comptime assertions** to validate invariants
5. **Don't use comptime for runtime configuration** — use `const` or runtime parameters instead

---

**In the code:** [src/ops/quant.zig](../../src/ops/quant.zig) (fp8e4m3_lut, iq4nl_values), [src/backend/metal.zig](../../src/backend/metal.zig) (@embedFile for MSL shaders), [src/backend/backend.zig](../../src/backend/backend.zig) (inline else dispatch), [build.zig](../../build.zig) (build_options)

**Related:** [Zig Language Reference — comptime](https://ziglang.org/documentation/master/#comptime), [Chapter 9: CPU SIMD Optimization](09-cpu-simd-optimization.md#real-world-example-rmsnorm) (uses comptime LUTs)

**Back:** [Chapter 16: Recipe System ←](16-recipe-system.md)
