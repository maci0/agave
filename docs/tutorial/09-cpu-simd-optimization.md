# Chapter 9: CPU SIMD Optimization

When a GPU isn't available, the CPU backend needs to be fast. Modern CPUs have **SIMD** (Single Instruction Multiple Data) units that can process 4-8 values in parallel with a single instruction. Zig provides portable SIMD via `@Vector` — the same code generates **NEON** on ARM (Apple Silicon, Raspberry Pi) and **AVX2/AVX-512** on x86_64 (Intel, AMD).

## Code Flow

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d

    X["x, w\n(k floats each)"]:::setup
    Load["load 8-wide chunk\n@Vector(8, f32)"]:::sync
    FMA["@mulAdd(V8, xv, wv, acc)\nSIMD accumulate"]:::sync
    More{"i + 8 <= k?"}
    Reduce["@reduce(.Add, acc)\nhorizontal sum"]:::migration
    Tail["scalar tail loop\nfor k % 8 remainder"]:::migration
    Y["y[row]"]:::success

    X --> Load --> FMA --> More
    More -- "yes" --> Load
    More -- "no" --> Reduce --> Tail --> Y
```

One GEMV row: load 8 floats from `x` and `w`, fuse-multiply-add into a SIMD accumulator, repeat until fewer than 8 elements remain, then reduce the accumulator to a scalar and finish the leftover elements one at a time. The rest of this chapter builds up to that loop and beyond it.

## The @Vector Type

A single SIMD instruction operates on an entire register of values at once, turning N sequential operations into 1 parallel operation.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Scalar["Scalar (8 instructions)"]
        direction TB
        s1["a[0]+b[0]"]:::sync --> r1["c[0]"]:::success
        s2["a[1]+b[1]"]:::sync --> r2["c[1]"]:::success
        s3["a[2]+b[2]"]:::sync --> r3["c[2]"]:::success
        s4["..."]:::sync --> r4["..."]:::success
    end

    subgraph SIMD["SIMD @Vector(8, f32) (1 instruction)"]
        direction TB
        reg_a["AVX2 register\n1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0"]:::setup
        reg_b["AVX2 register\n2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0"]:::setup
        op["vaddps (1 cycle)"]:::sync
        reg_c["AVX2 register\n3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0"]:::success
        reg_a --> op
        reg_b --> op
        op --> reg_c
    end

    Scalar -- "8x slower" --- SIMD
```

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

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Input["Input vector (8 lanes)"]
        direction LR
        v1["1"]:::setup --- v2["2"]:::setup --- v3["3"]:::setup --- v4["4"]:::setup --- v5["5"]:::setup --- v6["6"]:::setup --- v7["7"]:::setup --- v8["8"]:::setup
    end
    p1["1+2 = 3"]:::sync
    p2["3+4 = 7"]:::sync
    p3["5+6 = 11"]:::sync
    p4["7+8 = 15"]:::sync
    q1["3+7 = 10"]:::migration
    q2["11+15 = 26"]:::migration
    total["10+26 = 36"]:::success

    v1 & v2 --> p1
    v3 & v4 --> p2
    v5 & v6 --> p3
    v7 & v8 --> p4
    p1 & p2 --> q1
    p3 & p4 --> q2
    q1 & q2 --> total
```

```
{1,2,3,4,5,6,7,8}
→ {1+2, 3+4, 5+6, 7+8} = {3, 7, 11, 15}
→ {3+7, 11+15}          = {10, 26}
→ 10+26                 = 36
```

On NEON: `vaddvq_f32` (horizontal add). On AVX2: `vhaddps` + scalar extract.

### @mulAdd — Fused Multiply-Add (FMA)

FMA collapses a multiply and an add into one instruction, firing through a dedicated hardware unit. The CPU dispatches it alongside other work in the same cycle.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Unfused["Without FMA (2 instructions, 2 cycles min)"]
        direction TB
        u_a["a (weight)"]:::setup
        u_b["b (input)"]:::setup
        u_acc["acc"]:::setup
        mul["vmulps\n1 cycle"]:::sync
        tmp["tmp = a * b\n(intermediate, rounded)"]:::migration
        add["vaddps\n1 cycle"]:::sync
        u_out["acc + tmp"]:::success
        u_a --> mul
        u_b --> mul
        mul --> tmp
        tmp --> add
        u_acc --> add
        add --> u_out
    end

    subgraph Fused["With FMA — @mulAdd (1 instruction, 1 cycle)"]
        direction TB
        f_a["a (weight)"]:::setup
        f_b["b (input)"]:::setup
        f_acc["acc"]:::setup
        fma["vfmadd231ps\n1 cycle, dedicated FMA unit"]:::sync
        f_out["acc + a*b\n(no intermediate rounding)"]:::success
        f_a --> fma
        f_b --> fma
        f_acc --> fma
        fma --> f_out
    end
```

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

Loading the input vector `x` is expensive. With 4-row batching, `x` is loaded once and reused across 4 rows of the weight matrix before being evicted from cache.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    x_mem["x vector\n(k floats in L1/L2 cache)"]:::setup

    subgraph Weight["Weight matrix W (4 rows)"]
        w0["row 0: w[0..k]"]:::setup
        w1["row 1: w[k..2k]"]:::setup
        w2["row 2: w[2k..3k]"]:::setup
        w3["row 3: w[3k..4k]"]:::setup
    end

    subgraph FMA["FMA units (4 independent accumulators)"]
        acc0["acc0 += xv * w0[i..i+8]"]:::sync
        acc1["acc1 += xv * w1[i..i+8]"]:::sync
        acc2["acc2 += xv * w2[i..i+8]"]:::sync
        acc3["acc3 += xv * w3[i..i+8]"]:::sync
    end

    subgraph Output["y (4 output values)"]
        y0["y[row]"]:::success
        y1["y[row+1]"]:::success
        y2["y[row+2]"]:::success
        y3["y[row+3]"]:::success
    end

    x_mem -- "load ONCE\nreuse 4x" --> acc0
    x_mem --> acc1
    x_mem --> acc2
    x_mem --> acc3

    w0 --> acc0
    w1 --> acc1
    w2 --> acc2
    w3 --> acc3

    acc0 -- "@reduce(.Add)" --> y0
    acc1 -- "@reduce(.Add)" --> y1
    acc2 -- "@reduce(.Add)" --> y2
    acc3 -- "@reduce(.Add)" --> y3
```

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

**Why not 8 rows?** Register pressure. 8 accumulators + 8 row-weight vectors + the xv broadcast = ~17 SIMD registers on AVX2, which only has 16 YMM registers, forcing spills to the stack. 4 rows is the sweet spot.

**NR=2 for K-quant formats:** Q4_K, Q5_K, and Q6_K use **NR=2**; Q4_0, Q8_0, BF16, and F16 use **NR=4** (same as the f32 kernel). The heavier per-block dequantization in K-quant formats is what reduces the optimal row count to 2. The same NR multi-row pattern is applied across GPU backends as well (Metal, CUDA, ROCm) with NR values tuned per format and hardware.

### Performance (from BENCHMARKS.md)

Measured 2026-03-24 on Apple M4 Pro (14-core CPU), full methodology in [BENCHMARKS.md](../BENCHMARKS.md).

| Claim | Source |
|-------|--------|
| Qwen3.5 9B Q8_0 CPU decode: 11.3 tok/s | BENCHMARKS Decode Throughput, M4 Pro |
| Gemma 3 12B Q8_0 CPU decode: 6.3 tok/s | BENCHMARKS Decode Throughput, M4 Pro |
| CPU numbers use all 14 threads (Agave) vs 10 threads (llama.cpp default) | BENCHMARKS Notes |

## Handling Quantized Data

Quantized GEMV must **dequantize inside the loop** to avoid materializing the full f32 matrix.

### Q4_0 Block Memory Layout

Each Q4_0 block encodes 32 elements into 18 bytes: a 2-byte f16 scale followed by 16 bytes of packed nibbles (two 4-bit values per byte).

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Block["Q4_0 block — 18 bytes total\nencode 32 elements"]
        direction LR
        scale["bytes 0-1\nf16 scale\napplied to all 32 elements"]:::setup
        nibbles["bytes 2-17\n16 packed bytes\n= 32 nibbles (4 bits each)"]:::setup
        scale --- nibbles
    end

    subgraph Byte["Single nibble byte (byte j)"]
        direction TB
        lo["low nibble [3:0]\nelement j\nvalue = (byte & 0xF) - 8"]:::migration
        hi["high nibble [7:4]\nelement j+16\nvalue = (byte >> 4) - 8"]:::migration
    end

    subgraph Decode["Dequantize element"]
        direction TB
        d1["q = nibble - 8\n(center at zero)"]:::sync
        d2["f32 = scale * q\n(apply block scale once)"]:::success
        d1 --> d2
    end

    nibbles -- "byte j" --> Byte
    Byte -- "q0, q1" --> Decode
    scale -- "scale" --> Decode
```

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
                const q0 = @as(i8, @intCast(byte & 0xF)) - 8;  // Low nibble -> element j
                const q1 = @as(i8, @intCast(byte >> 4)) - 8;   // High nibble -> element j + 16
                block_sum += @as(f32, @floatFromInt(q0)) * x[x_offset + j];
                block_sum += @as(f32, @floatFromInt(q1)) * x[x_offset + j + 16];
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

Process data in the order it's laid out in memory. Row-major matrices should iterate rows then columns. Sequential access keeps data in L1/L2 cache; column-major access on a row-major matrix jumps by `n_cols` bytes between loads, thrashing cache lines.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    subgraph Matrix["Row-major matrix in memory\n(4 cols × 3 rows = 12 floats, 48 bytes)"]
        direction LR
        m00["[0,0]"]:::setup --- m01["[0,1]"]:::setup --- m02["[0,2]"]:::setup --- m03["[0,3]"]:::setup --- m10["[1,0]"]:::setup --- m11["[1,1]"]:::setup --- m12["[1,2]"]:::setup --- m13["[1,3]"]:::setup --- m20["[2,0]"]:::setup --- m21["[2,1]"]:::setup --- m22["[2,2]"]:::setup --- m23["[2,3]"]:::setup
    end

    subgraph Good["Row-major iteration (sequential)"]
        direction LR
        g1["[0,0] [0,1] [0,2] [0,3]"]:::success -- "cache line hit" --> g2["[1,0] [1,1] [1,2] [1,3]"]:::success -- "cache line hit" --> g3["[2,0] [2,1] [2,2] [2,3]"]:::success
    end

    subgraph Bad["Column-major iteration (strided)"]
        direction LR
        b1["[0,0]"]:::danger -- "stride 4 floats\ncache miss" --> b2["[1,0]"]:::danger -- "stride 4 floats\ncache miss" --> b3["[2,0]"]:::danger -- "stride 4 floats\ncache miss" --> b4["..."]:::danger
    end

    Matrix --> Good
    Matrix --> Bad
```

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

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    input["input[0..n]\n(n f32 values)"]:::setup
    weight["weight[0..n]\n(learned gain per element)"]:::setup

    subgraph Pass1["Pass 1 — sum of squares"]
        direction LR
        p1a["@mulAdd(V8, xv, xv, acc)\nfor each 8-element chunk"]:::sync --> p1b["@reduce(.Add, acc)\nhorizontal sum"]:::sync --> p1c["sum_sq\n= Σ input[i]²"]:::migration
    end

    subgraph Scalar["Scalar normalization"]
        direction LR
        s1["rms = sqrt(sum_sq / n + eps)"]:::migration --> s2["scale = 1.0 / rms"]:::migration
    end

    subgraph Pass2["Pass 2 — normalize + weight"]
        direction LR
        p2a["@splat(scale)\nbroadcast scalar"]:::sync --> p2b["xv * scale_v\nnormalize chunk"]:::sync --> p2c["normalized * wv\napply weight"]:::sync --> p2d["output[i..i+8]"]:::success
    end

    input --> Pass1
    Pass1 --> Scalar
    Scalar --> Pass2
    input --> Pass2
    weight --> Pass2
```

(simplified for clarity -- the real implementation in norm.zig uses 4-accumulator unrolling with a stride-32 inner loop to hide FMA latency)

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

## Activation Sparsity (Sparse GEMV)

After SiLU activation in FFN layers, ~40% of output values are near-zero (magnitude < 0.005). The down-projection GEMV multiplies these near-zero values by weight blocks — wasting ~40% of compute. Sparse GEMV skips these blocks entirely:

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    gate["gate_proj output\n(n values)"]:::setup
    up["up_proj output\n(n values)"]:::setup

    subgraph SiLU["SiLU activation\nx * sigmoid(x)"]
        direction LR
        act["gate * sigmoid(gate)\nelement-wise"]:::sync
    end

    subgraph Sparsity["~40% near-zero after SiLU\n(magnitude < threshold)"]
        direction LR
        sparse["block 0: |max| = 0.001\nSKIP"]:::danger -.- dense1["block 1: |max| = 0.83\ncompute"]:::success -.- sparse2["block 2: |max| = 0.002\nSKIP"]:::danger -.- dense2["block 3: |max| = 1.2\ncompute"]:::success
    end

    subgraph Check["isBlockSparse (SIMD max-abs)"]
        direction LR
        c1["@reduce(.Max, @abs(xv))\n~1 cycle per 8 elements"]:::sync --> c2{"< threshold?"}
        c2 -- "yes" --> skip["continue\n(skip dequant + dot)"]:::danger
        c2 -- "no" --> compute["normal GEMV block\n(dequant + FMA)"]:::success
    end

    gate --> SiLU
    up --> SiLU
    SiLU -- "element-wise product" --> Sparsity
    Sparsity --> Check
```

```zig
// Before processing each weight block, check if input is negligible
for (0..nb) |b| {
    if (isBlockSparse(x, b * block_size, block_size)) continue;
    // ... normal dequant + MAC ...
}
```

`isBlockSparse` uses SIMD max-abs reduction (~1 cycle per 8 elements) to check if all block inputs are below threshold. If so, the entire block (dequant + dot product) is skipped.

**Measured speedup (Qwen3.5 on M4 Pro CPU):**
- Q8_0: +21% decode throughput
- Q4_K_M: +23% decode throughput
- Output identical — threshold only controls whether to compute, not what values to use

This is inspired by [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) and [TurboSparse](https://arxiv.org/abs/2406.05955), which exploit activation sparsity for 2-5× speedup on ReLU models (90%+ sparsity). SiLU models have lower sparsity (~40%) but still benefit significantly.

**Why CPU only?** GPU kernels are bandwidth-bound (waiting for memory, not compute). Adding branch checks to GPU shaders causes thread divergence which hurts performance. CPU GEMV is compute-bound (sequential dot products), so skipping blocks is pure win.

---

**In the code:** [src/backend/kernels/cpu/gemv.zig](../../src/backend/kernels/cpu/gemv.zig) (`isBlockSparse`, `sparse_threshold`), [src/backend/kernels/cpu/gemv_f32.zig](../../src/backend/kernels/cpu/gemv_f32.zig), [src/backend/kernels/cpu/gemv_bf16.zig](../../src/backend/kernels/cpu/gemv_bf16.zig), [src/backend/kernels/cpu/norm.zig](../../src/backend/kernels/cpu/norm.zig), [src/ops/mlx.zig](../../src/ops/mlx.zig) (MLX GEMV with factored dequant)

**Next:** [Chapter 10: Memory Safety →](10-memory-safety.md) | **Back:** [Chapter 8: Backends ←](08-backends.md) | **Product docs:** [Architecture](../ARCHITECTURE.md) · [Models](../MODELS.md)

---

## Glossary

**@mulAdd (FMA)** — Fused Multiply-Add; a single instruction computing a×b+c with no intermediate rounding, mapped to hardware FMA units.

**@reduce** — Zig builtin that collapses a SIMD vector to a scalar via a specified operation (e.g., `.Add` for horizontal sum).

**@splat** — Zig builtin that broadcasts a scalar value to all lanes of a SIMD vector.

**@Vector** — Zig's portable SIMD type mapping to hardware vector registers; e.g., `@Vector(8, f32)` is 8 packed f32 values.

**activation sparsity** — The phenomenon where ~40% of activation values are near-zero after SiLU, allowing those GEMV blocks to be skipped.

**AVX2 (Advanced Vector Extensions 2)** — Intel/AMD 256-bit SIMD instruction set providing 8-wide f32 operations.

**cache locality** — Accessing memory sequentially to maximize CPU cache hits and minimize cache misses.

**isBlockSparse** — A SIMD max-abs check that determines whether all input values in a block are below a threshold, enabling the block to be skipped.

**multi-row GEMV batching** — Processing multiple output rows simultaneously (NR=2 or NR=4) to amortize the cost of loading the input vector.

**NEON** — ARM's SIMD instruction set providing 128-bit vector operations on aarch64 processors.

**NR (Number of Rows)** — The number of output rows computed per batch in a multi-row GEMV kernel (e.g., NR=4 for Q8_0 on CPU).

**register pressure** — The constraint from having finite hardware SIMD registers; exceeding capacity causes spills to slower stack memory.

**tail loop** — A scalar cleanup loop handling remaining elements when data length is not a multiple of the SIMD vector width.
