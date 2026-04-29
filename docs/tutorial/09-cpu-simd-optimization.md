# Chapter 9: CPU SIMD Optimization

When a GPU isn't available, the CPU backend needs to be fast. Modern CPUs have **SIMD** (Single Instruction Multiple Data) units that can process 4-8 values in parallel with a single instruction. Zig provides portable SIMD via `@Vector` — the same code generates **NEON** on ARM (Apple Silicon, Raspberry Pi) and **AVX2/AVX-512** on x86_64 (Intel, AMD).

## The @Vector Type

A vector is a fixed-size array that maps to hardware SIMD registers:

```zig
const V8 = @Vector(8, f32);  // 8 × f32 = 256 bits (AVX2 register or 2 NEON registers)

var a: V8 = .{1, 2, 3, 4, 5, 6, 7, 8};
var b: V8 = .{2, 2, 2, 2, 2, 2, 2, 2};
var c = a + b;  // Compiles to 1 instruction: vadd or vaddps
// c = {3, 4, 5, 6, 7, 8, 9, 10}
```

**Why 8 elements?** AVX2 (Intel/AMD) has 256-bit registers = 8 f32s. NEON (ARM) has 128-bit registers = 4 f32s, so the compiler uses 2 registers. This is the sweet spot for portable code.

### Loading from Memory

Vectors load from slices using array syntax:

```zig
const x: [*]const f32 = ...;  // Input data
var i: usize = 0;

while (i + 8 <= n) : (i += 8) {
    const xv: V8 = x[i..][0..8].*;  // Load 8 consecutive f32s
    // xv now contains x[i], x[i+1], ..., x[i+7]
}
```

**Memory alignment matters:** SIMD loads are fastest when the address is aligned to 32 bytes (AVX2) or 16 bytes (NEON). Agave relies on the allocator providing sufficient alignment — `std.heap.page_allocator` guarantees this for large allocations.

## Core SIMD Operations

### @splat — Broadcast a Scalar

```zig
const v: V8 = @splat(2.5);  // All 8 elements = 2.5
// v = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5}
```

Used to initialize accumulators to zero:

```zig
const v8zero: V8 = @splat(0.0);
var acc: V8 = v8zero;
```

### @reduce — Horizontal Sum

```zig
const v: V8 = .{1, 2, 3, 4, 5, 6, 7, 8};
const sum = @reduce(.Add, v);  // sum = 36.0 (1+2+3+4+5+6+7+8)
```

Compiles to a **reduction tree** (pair-wise adds that preserve precision better than sequential accumulation):

```
{1,2,3,4,5,6,7,8}
→ {1+2, 3+4, 5+6, 7+8} = {3, 7, 11, 15}
→ {3+7, 11+15}          = {10, 26}
→ 10+26                 = 36
```

On NEON: `vaddvq_f32` (horizontal add). On AVX2: `vhaddps` + scalar extract.

### @mulAdd — Fused Multiply-Add (FMA)

**The single most important SIMD operation for inference.**

```zig
acc = @mulAdd(V8, a, b, acc);
// Equivalent to: acc += a * b
// But compiles to 1 instruction instead of 2
```

Maps to hardware FMA:
- **NEON**: `vfma` or `vmlaq_f32` (1 cycle latency, 2× throughput)
- **AVX2**: `vfmadd231ps` (1 instruction vs separate `vmulps` + `vaddps`)

**Why FMA matters:**
- **Fewer instructions**: 1 instead of 2 → 2× fewer instruction fetches
- **Better precision**: `a*b+c` computed as one operation → no intermediate rounding
- **Higher throughput**: FMA units are separate from regular ALUs on modern CPUs

Example from f32 GEMV (dot product):

```zig
var acc: V8 = v8zero;
var i: usize = 0;
while (i + 8 <= k) : (i += 8) {
    const xv: V8 = x[i..][0..8].*;
    const wv: V8 = w[row*k + i ..][0..8].*;
    acc = @mulAdd(V8, xv, wv, acc);  // acc += xv * wv
}
const dot = @reduce(.Add, acc);
```

**Performance:** On Apple M4, this achieves **~70% of peak memory bandwidth** — the bottleneck is loading `x` and `w`, not arithmetic.

## Multi-Row GEMV Batching

The problem: loading `x` from memory is expensive. Each row of the matrix needs the same `x` vector. **Reuse it across multiple rows before evicting from cache.**

### 4-Row Batching Pattern

```zig
pub fn gemvF32(x: [*]const f32, w: [*]const f32, y: [*]f32, n: usize, k: usize) void {
    var row: usize = 0;

    // Process 4 rows at a time
    while (row + 4 <= n) : (row += 4) {
        var acc0: V8 = v8zero;
        var acc1: V8 = v8zero;
        var acc2: V8 = v8zero;
        var acc3: V8 = v8zero;

        const r0 = row * k;       // Offset to row 0
        const r1 = r0 + k;        // Offset to row 1
        const r2 = r1 + k;        // Offset to row 2
        const r3 = r2 + k;        // Offset to row 3

        var i: usize = 0;
        while (i + 8 <= k) : (i += 8) {
            const xv: V8 = x[i..][0..8].*;  // Load x ONCE

            // Reuse xv for all 4 rows
            acc0 = @mulAdd(V8, xv, @as(V8, w[r0+i..][0..8].*), acc0);
            acc1 = @mulAdd(V8, xv, @as(V8, w[r1+i..][0..8].*), acc1);
            acc2 = @mulAdd(V8, xv, @as(V8, w[r2+i..][0..8].*), acc2);
            acc3 = @mulAdd(V8, xv, @as(V8, w[r3+i..][0..8].*), acc3);
        }

        // Tail loop for remaining elements (if k not multiple of 8)
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        while (i < k) : (i += 1) {
            const xv = x[i];
            t0 = @mulAdd(f32, xv, w[r0+i], t0);
            t1 = @mulAdd(f32, xv, w[r1+i], t1);
            t2 = @mulAdd(f32, xv, w[r2+i], t2);
            t3 = @mulAdd(f32, xv, w[r3+i], t3);
        }

        // Reduce and store
        y[row]     = @reduce(.Add, acc0) + t0;
        y[row + 1] = @reduce(.Add, acc1) + t1;
        y[row + 2] = @reduce(.Add, acc2) + t2;
        y[row + 3] = @reduce(.Add, acc3) + t3;
    }

    // Remainder rows (< 4 remaining)
    while (row < n) : (row += 1) {
        var acc: V8 = v8zero;
        var tail: f32 = 0.0;
        const roff = row * k;
        var i: usize = 0;
        while (i + 8 <= k) : (i += 8) {
            acc = @mulAdd(V8, @as(V8, x[i..][0..8].*), @as(V8, w[roff+i..][0..8].*), acc);
        }
        while (i < k) : (i += 1) {
            tail = @mulAdd(f32, x[i], w[roff+i], tail);
        }
        y[row] = @reduce(.Add, acc) + tail;
    }
}
```

**Key insights:**

1. **`xv` loaded once, used 4 times** — amortizes memory latency
2. **4 independent accumulators** — allows CPU to **pipeline** FMAs (execute multiple in parallel)
3. **Tail loop** — handles `k` not divisible by 8 (common with quantized blocks)
4. **Remainder loop** — handles `n` not divisible by 4

**Performance gain:** 2-3× faster than 1-row-at-a-time on bandwidth-bound workloads (most GEMV cases).

**Why not 8 rows?** Register pressure. 8 accumulators x V8 = 32 SIMD registers (NEON only has 32, AVX2 has 16). Compiler starts spilling to stack -> slower. 4 rows is the sweet spot.

**NR=2 for quantized formats:** In practice, all CPU GEMV kernels for quantized formats (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, etc.) use **NR=2** (2-row batching). The dequantization logic per block consumes more registers than f32 GEMV, so 2 rows is the optimal trade-off between input vector reuse and register pressure. The same NR multi-row pattern is applied across GPU backends as well (Metal, CUDA, ROCm) with NR values tuned per format and hardware.

## Handling Quantized Data

Quantized GEMV must **dequantize inside the loop** to avoid materializing the full f32 matrix.

### Example: Q4_0 GEMV (4-bit with f16 scale)

Q4_0 layout: 32 elements per block = 16 bytes (nibbles) + 2 bytes (f16 scale) = 18 bytes/block.

```zig
pub fn gemvQ4_0(x: [*]const f32, w: [*]const u8, y: [*]f32, n: usize, k: usize) void {
    const block_size = 32;
    const nb = (k + block_size - 1) / block_size;  // Blocks per row

    var row: usize = 0;
    while (row < n) : (row += 1) {
        var sum: f32 = 0.0;
        const row_offset = row * nb * 18;  // 18 bytes per Q4_0 block

        for (0..nb) |ib| {
            const block_offset = row_offset + ib * 18;

            // Decode scale (first 2 bytes, f16 format)
            const scale_ptr = @as(*const f16, @ptrCast(@alignCast(&w[block_offset])));
            const scale: f32 = @floatCast(scale_ptr.*);

            // Dequantize and accumulate 32 elements
            const quant_data = w[block_offset + 2 ..];
            const x_offset = ib * block_size;

            var block_sum: f32 = 0.0;
            for (0..16) |j| {  // 16 bytes = 32 nibbles (2 per byte)
                const byte = quant_data[j];
                const q0 = @as(i8, @intCast(byte & 0xF)) - 8;  // Low nibble
                const q1 = @as(i8, @intCast(byte >> 4)) - 8;   // High nibble

                block_sum += @as(f32, @floatFromInt(q0)) * x[x_offset + j*2];
                block_sum += @as(f32, @floatFromInt(q1)) * x[x_offset + j*2 + 1];
            }

            sum += scale * block_sum;  // Apply scale once per block
        }
        y[row] = sum;
    }
}
```

**Optimization notes:**

- **Scalar loop** shown for clarity — production code uses V8 SIMD for the 32-element block
- **Scale applied once per block** — not per element (32× fewer multiplies)
- **Nibble extraction** via bit shifts — no lookup tables needed
- **Signed offset** (`-8`) centers the quantized range at zero

For the full SIMD-optimized version, see [src/backend/kernels/cpu/gemv_q4_0.zig](../../src/backend/kernels/cpu/gemv_q4_0.zig).

## Common Patterns

### Zeroing an Accumulator

```zig
const v8zero: V8 = @splat(0.0);
var acc: V8 = v8zero;
```

### Element-wise Operations

```zig
// Element-wise multiply
const a: V8 = ...;
const b: V8 = ...;
const c = a * b;  // c[i] = a[i] * b[i]

// Element-wise add
const sum = a + b;

// Multiply by scalar (broadcast)
const scaled = a * @as(V8, @splat(2.0));
```

### Conditional Operations (Masking)

```zig
// Select elements based on condition
const mask = a > @as(V8, @splat(0.0));  // Boolean vector
const result = @select(f32, mask, a, v8zero);  // result[i] = mask[i] ? a[i] : 0.0
```

Used in ReLU (Rectified Linear Unit — max(0, x)):

```zig
pub fn relu(x: [*]f32, n: usize) void {
    const v8zero: V8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const xv: V8 = x[i..][0..8].*;
        const result = @max(xv, v8zero);  // Element-wise max
        x[i..][0..8].* = result;
    }
    while (i < n) : (i += 1) {
        x[i] = @max(x[i], 0.0);
    }
}
```

### Transcendental Functions

Zig provides SIMD-vectorized math builtins:

```zig
const v: V8 = ...;
const exp_v = @exp(v);    // Element-wise e^x
const sqrt_v = @sqrt(v);  // Element-wise √x
const log_v = @log(v);    // Element-wise ln(x)
```

Used in SoftPlus activation (`log(1 + e^x)`):

```zig
pub inline fn softplus(x: f32) f32 {
    return @log(1.0 + @exp(x));
}

// Vectorized version
pub fn softplusVec(x: [*]f32, n: usize) void {
    const v8one: V8 = @splat(1.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const xv: V8 = x[i..][0..8].*;
        const result = @log(v8one + @exp(xv));
        x[i..][0..8].* = result;
    }
    while (i < n) : (i += 1) {
        x[i] = softplus(x[i]);
    }
}
```

**Note:** On CUDA/Metal, avoid `@exp` in GPU kernels — it compiles to a slow `libcall`. Use native GPU intrinsics instead (e.g., MSL `exp()`, CUDA `__expf()`).

## Performance Considerations

### Cache Locality

Process data in the order it's laid out in memory. Row-major matrices should iterate rows → columns:

```zig
// GOOD: Sequential memory access
for (0..n_rows) |row| {
    for (0..n_cols) |col| {
        process(matrix[row * n_cols + col]);
    }
}

// BAD: Strided access (cache misses)
for (0..n_cols) |col| {
    for (0..n_rows) |row| {
        process(matrix[row * n_cols + col]);
    }
}
```

### Alignment

Aligned loads are faster (1 cycle vs 2-3 cycles for unaligned on some CPUs):

```zig
// Let the allocator handle alignment
const data = try allocator.alloc(f32, n);  // Typically 16-byte aligned

// For explicit control:
const data = try allocator.alignedAlloc(f32, 32, n);  // Force 32-byte alignment
```

### Prefetching

For large sequential scans, hint the CPU to prefetch:

```zig
@prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
```

Agave doesn't use explicit prefetching — the CPU's hardware prefetcher does well enough for sequential GEMV access.

### Avoid Branching in Inner Loops

Branches inside SIMD loops can **serialize** (force sequential execution, losing SIMD parallelism). Use `@select` or `@max`/`@min` instead:

```zig
// BAD: Branch per element (serializes)
for (0..n) |i| {
    if (x[i] > 0) {
        y[i] = x[i];
    } else {
        y[i] = 0;
    }
}

// GOOD: SIMD-friendly (no branches)
var i: usize = 0;
const v8zero: V8 = @splat(0.0);
while (i + 8 <= n) : (i += 8) {
    const xv: V8 = x[i..][0..8].*;
    const yv = @max(xv, v8zero);
    y[i..][0..8].* = yv;
}
```

## Real-World Example: RMSNorm

RMSNorm is a two-pass reduction: compute RMS, then normalize.

```zig
pub fn rmsNorm(input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
    // Pass 1: Compute mean of squares
    var sum_sq: f32 = 0.0;
    {
        var acc: V8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            const xv: V8 = input[i..][0..8].*;
            acc = @mulAdd(V8, xv, xv, acc);  // acc += xv * xv
        }
        sum_sq = @reduce(.Add, acc);
        while (i < n) : (i += 1) {
            sum_sq = @mulAdd(f32, input[i], input[i], sum_sq);
        }
    }

    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
    const scale = 1.0 / rms;

    // Pass 2: Normalize and apply weight
    {
        const scale_v: V8 = @splat(scale);
        var i: usize = 0;
        while (i + 8 <= n) : (i += 8) {
            const xv: V8 = input[i..][0..8].*;
            const wv: V8 = weight[i..][0..8].*;
            const normalized = xv * scale_v;
            const weighted = normalized * wv;
            output[i..][0..8].* = weighted;
        }
        while (i < n) : (i += 1) {
            output[i] = (input[i] * scale) * weight[i];
        }
    }
}
```

**Optimizations:**

- **FMA for squares** — `@mulAdd(V8, xv, xv, acc)` is 1 instruction
- **Horizontal sum** — `@reduce(.Add, acc)` for final sum
- **Broadcast scale** — `@splat(scale)` once, reuse for all elements
- **Fused normalize+weight** — both in one loop (cache-friendly)

**Alternative:** GPU backends can fuse both passes into a single kernel using **threadgroup reductions** (parallel sum across threads, not sequential).

---

**In the code:** [src/backend/kernels/cpu/gemv_f32.zig](../../src/backend/kernels/cpu/gemv_f32.zig), [src/backend/kernels/cpu/gemv_bf16.zig](../../src/backend/kernels/cpu/gemv_bf16.zig), [src/backend/kernels/cpu/norm.zig](../../src/backend/kernels/cpu/norm.zig), [src/ops/mlx.zig](../../src/ops/mlx.zig) (MLX GEMV with factored dequant)

**Next:** [Chapter 10: Memory Safety →](10-memory-safety.md) | **Back:** [Chapter 8: Backends ←](08-backends.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)
