# Chapter 13: Batched Dispatch and Fusion

Every GPU kernel dispatch has overhead: setting up the pipeline state, binding buffers, launching threadgroups, and inserting memory barriers. When operations share the same input vector or can be combined into a single pass, **batching** and **fusion** eliminate redundant dispatches.

## The Dispatch Overhead Problem

A typical attention layer does:

```zig
be.gemv(x, w_q, q, n_q, k);    // Q projection: 1 dispatch + 1 barrier
be.gemv(x, w_k, k_buf, n_k, k); // K projection: 1 dispatch + 1 barrier
be.gemv(x, w_v, v, n_v, k);    // V projection: 1 dispatch + 1 barrier
// Total: 3 dispatches, 3 barriers
```

**Sequential Dispatch Visualization:**

```
Timeline (unfused):

CPU:   [dispatch Q] [wait] [dispatch K] [wait] [dispatch V] [wait]
                       ▲                   ▲                   ▲
                    barrier             barrier             barrier
GPU:        [Q GEMV]   │   [K GEMV]       │   [V GEMV]       │
                       │                  │                  │
            load x ────┘   load x ────────┘   load x ────────┘
            (3× redundant memory loads — x loaded from VRAM 3 times)

Timeline (batched via gemvMulti):

CPU:   [dispatch Q,K,V together]           [wait once]
                                              ▲
                                           barrier
GPU:        [Q GEMV] [K GEMV] [V GEMV]      │
             ▲        ▲        ▲             │
             └────────┴────────┘─────────────┘
            load x once, reuse in registers

Overhead saved: 2 dispatches, 2 barriers, 2× redundant x loads
```

**Problem:** All three GEMVs use the same `x` input vector. The GPU loads `x` from memory **three times** (once per dispatch), even though it could load it once and reuse it.

**Overhead per dispatch** (measured on Apple M4 Metal):
- Pipeline state setup: ~5-10 µs
- Memory barrier: ~0 µs (Apple Silicon overlaps work)
- Total per dispatch: ~5-10 µs

For a 27B model with ~210 GEMVs per token, that's **1-2 ms of pure overhead** per token.

## Batched GEMV: gemvMulti

**Idea:** Dispatch all GEMVs that share the same input vector in a **single kernel launch**.

### GemvOp Structure

```zig
pub const GemvOp = struct {
    w: TensorData,      // Weight matrix (quantized)
    y: [*]f32,          // Output buffer
    n: usize,           // Number of output rows
    // Optional MLX companions (for MLX quantized weights)
    mlx_scales: ?[*]const u8 = null,
    mlx_biases: ?[*]const u8 = null,
    mlx_bits: u32 = 0,
};
```

### Backend Interface

```zig
pub inline fn gemvMulti(self: Backend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
    switch (self) {
        inline else => |be| be.gemvMulti(x, ops, k),
    }
}
```

### Usage Example

```zig
// Attention Q/K/V projection (all share input x)
const ops = [_]GemvOp{
    .{ .w = w_q, .y = q_buf, .n = n_q * nh },
    .{ .w = w_k, .y = k_buf, .n = n_kv * nh },
    .{ .w = w_v, .y = v_buf, .n = n_kv * nh },
};
be.gemvMulti(x, &ops, n_embd);  // 1 dispatch instead of 3
```

### Metal Implementation

```zig
pub fn gemvMulti(self: *MetalBackend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
    for (ops) |op| {
        // Determine pipeline based on dtype and MLX companions
        const pipeline = if (op.mlx_scales != null) blk: {
            if (op.mlx_bits == 4) break :blk self.pipe_gemv_mlx_q4;
            if (op.mlx_bits == 6) break :blk self.pipe_gemv_mlx_q6;
            if (op.mlx_bits == 8) break :blk self.pipe_gemv_mlx_q8;
            @panic("Unsupported MLX bit width");
        } else switch (op.w.dtype) {
            .f32 => self.pipe_gemv_f32,
            .bf16 => self.pipe_gemv_bf16,
            .q4_0 => self.pipe_gemv_q4_0,
            .q8_0 => self.pipe_gemv_q8_0,
            // ... other dtypes
        };

        // Encode this GEMV (reuses active encoder)
        self.encode(pipeline, &[_]BufRef{
            self.getBufRef(@ptrCast(x), k * @sizeOf(f32)),
            self.getBufRef(@ptrCast(op.w.data), weightBytes(op.w.dtype, op.n, k)),
            self.getBufRef(@ptrCast(op.y), op.n * @sizeOf(f32)),
            // ... MLX companions if present
        }, grid);
    }
    // Single barrier at the end (outside the loop)
}
```

**Key insight:** All dispatches use the same command encoder. The GPU can overlap them, and only **one barrier** is inserted after all ops complete.

### CPU Implementation

```zig
pub fn gemvMulti(self: *CpuBackend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
    for (ops) |op| {
        self.gemv(x, op.w, op.y, op.n, k);  // Sequential on CPU
    }
}
```

**CPU doesn't batch** — sequential execution is fine. The API uniformity is the benefit.

### Performance Impact

**Qwen3.5 27B MLX** (Apple M4 Pro):
- Before gemvMulti: 930 barriers/token
- After gemvMulti: 690 barriers/token
- Throughput change: 0% (barriers are free on Apple Silicon)

**But:** On discrete GPUs (NVIDIA, AMD), barriers flush PCIe, so this would be a 20-30% win.

## Fused Operations

**Fusion** combines sequential operations into a single kernel to eliminate intermediate memory writes.

### Why Fusion Matters

```zig
// Unfused: 2 dispatches, 2 memory round-trips
be.add(residual, ffn_out, temp, n_embd);       // Write temp to VRAM
be.rmsNorm(temp, norm_w, normalized, n_embd);  // Read temp from VRAM

// Memory traffic: residual (read) + ffn_out (read) + temp (write+read) + normalized (write)
//               = 4 memory ops
```

```zig
// Fused: 1 dispatch, 1 memory round-trip
be.addRmsNorm(residual, ffn_out, norm_w, normalized, n_embd);

// Memory traffic: residual (read) + ffn_out (read) + normalized (write)
//               = 3 memory ops (25% reduction)
```

**Savings:** Eliminate `temp` write and read → 2× memory bandwidth saved for the intermediate result.

**GPU implementation:** `temp` computed in registers, never written to VRAM.

### Common Fused Operations in Agave

#### addRmsNorm: Residual + Normalization

```zig
pub inline fn addRmsNorm(
    self: Backend,
    a: [*]f32,              // Residual (modified in-place)
    b: [*]const f32,        // Input to add
    weight: [*]const f32,   // Norm weight
    output: [*]f32,         // Normalized output
    n: usize,
    eps: f32,
) void
```

**Metal kernel:**

```metal
kernel void add_rms_norm_fused(
    device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Phase 1: Add (in-place)
    if (tid < n) {
        a[tid] += b[tid];
    }
    threadgroup_barrier(mem_flags::mem_device);  // Ensure all adds complete

    // Phase 2: RMSNorm (reads from a, writes to output)
    // ... (same as standalone rmsNorm)
}
```

**Usage:** After every FFN sub-block:

```zig
// Before: residual += ffn(x); x = rmsNorm(residual)
be.add(residual, ffn_out, residual, n_embd);
be.rmsNorm(residual, norm_w, x_normed, n_embd, eps);

// After: fused
be.addRmsNorm(residual, ffn_out, norm_w, x_normed, n_embd, eps);
```

**Impact:** Qwen3.5 saved **128 dispatches/token** (64 layers × 2 residual+norm per layer).

#### siluMul: SwiGLU Activation

```zig
pub inline fn siluMul(
    self: Backend,
    a: [*]const f32,  // Gate input
    b: [*]const f32,  // Up input
    out: [*]f32,      // Output
    n: usize,
) void
```

**Formula:** `out[i] = silu(a[i]) * b[i]` where `silu(x) = x * sigmoid(x)`

**Metal kernel:**

```metal
kernel void silu_mul(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const float x = a[tid];
    const float silu_x = x / (1.0 + exp(-x));  // SiLU in-register
    out[tid] = silu_x * b[tid];                 // Multiply without storing silu_x
}
```

**Unfused equivalent:**

```zig
be.silu(gate, temp, n);  // Write temp
be.mul(temp, up, out, n); // Read temp, write out
```

**Fused:** No `temp` buffer needed → saves 1 allocation + 2 memory transfers.

**Usage:** SwiGLU FFN:

```zig
// gate_out = silu(gate_proj(x))
// up_out = up_proj(x)
// ffn_out = gate_out * up_out

be.gemv(x, w_gate, gate_buf, ff_dim, n_embd);
be.gemv(x, w_up, up_buf, ff_dim, n_embd);
be.siluMul(gate_buf, up_buf, ffn_out, ff_dim);  // Fused
```

#### splitQGate: Q+Gate Deinterleaving (GPU Kernel)

**Problem:** DeltaNet (Qwen3.5) stores Q and gate interleaved per head:

```
[Q0, G0, Q1, G1, Q2, G2, ..., Q_{hd-1}, G_{hd-1}] × nh heads
```

Needs to split into:
```
Q: [Q0..Q_{hd-1}] × nh heads
G: [G0..G_{hd-1}] × nh heads
```

**Naive CPU implementation:**

```zig
// CPU: requires be.sync() round-trip (GPU → CPU → GPU)
be.sync();  // Flush GPU writes to qg_buf
for (0..nh) |h| {
    const src = h * hd * 2;
    const dst = h * hd;
    @memcpy(q_out[dst..][0..hd], qg[src..][0..hd]);
    @memcpy(g_out[dst..][0..hd], qg[src+hd..][0..hd]);
}
// q_out and g_out now contain CPU-copied data
// Next GPU op must re-upload them → 2 more syncs!
```

**Cost:** 16 syncs/token (one per DeltaNet layer) × ~200 µs/sync = **3.2 ms/token overhead**.

**Fused GPU kernel:**

```metal
kernel void split_qgate(
    const device float* qg [[buffer(0)]],
    device float* q_out [[buffer(1)]],
    device float* g_out [[buffer(2)]],
    constant uint& hd [[buffer(3)]],
    constant uint& nh [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint h = tid / hd;           // Head index
    const uint i = tid % hd;           // Element within head
    const uint src = h * hd * 2 + i;   // Interleaved source
    const uint dst = h * hd + i;       // Contiguous dest

    q_out[dst] = qg[src];              // First half
    g_out[dst] = qg[src + hd];         // Second half
}
```

**Dispatch:**

```zig
be.splitQGate(qg_buf, q_buf, g_buf, hd, nh);  // 1 dispatch, no sync needed
```

**Impact:** Eliminated 16 syncs/token → Qwen3.5 throughput **12.3 → 14.1 tok/s** (+15%).

**Key insight:** Moving data manipulation from CPU to GPU eliminates sync points. Even a trivial operation (memcpy) is worth a GPU kernel if it avoids a round-trip.

#### addScaled: MoE Expert Accumulation

```zig
pub inline fn addScaled(
    self: Backend,
    src: [*]const f32,  // Expert output
    dst: [*]f32,        // Accumulator (modified in-place)
    scale: f32,         // Expert weight
    n: usize,
) void
```

**Formula:** `dst[i] += src[i] * scale`

**Usage:** Mixture of Experts:

```zig
// Zero accumulator
@memset(moe_out, 0.0);

// Dispatch experts
for (active_experts) |expert_id, i| {
    be.gemv(x, expert_weights[expert_id], expert_out, ff_dim, n_embd);
    const weight = expert_weights[i];
    be.addScaled(expert_out, moe_out, weight, ff_dim);  // Accumulate
}

// No sync needed — moe_out stays on GPU throughout
```

**Alternative (unfused):**

```zig
for (active_experts) |expert_id, i| {
    be.gemv(x, expert_weights[expert_id], expert_out, ff_dim, n_embd);
    be.sync();  // BAD: Force GPU → CPU
    const weight = expert_weights[i];
    for (0..ff_dim) |j| {
        moe_out[j] += expert_out[j] * weight;  // CPU accumulation
    }
}
```

**Cost:** `n_experts` syncs per MoE layer → 8 experts × 20 MoE layers = **160 syncs/token**.

**Fused:** Zero syncs. All accumulation happens on GPU.

## When to Fuse vs When to Keep Separate

### Fuse when:

✅ **Sequential dependency:** Output of A is input to B
✅ **Intermediate is temporary:** No other consumer needs it
✅ **Memory-bound:** Eliminating the intermediate write/read is the bottleneck
✅ **Small overhead:** Fusion logic is simple (not a massive kernel)

**Examples:** addRmsNorm (residual + norm), siluMul (activation + multiply)

### Don't fuse when:

❌ **Intermediate is reused:** Other ops need the intermediate result
❌ **Complex control flow:** Fusion makes the kernel hard to understand/debug
❌ **Compute-bound:** The bottleneck is arithmetic, not memory
❌ **Different thread counts:** A needs 256 threads/block, B needs 1024

**Example:** Don't fuse GEMV + RoPE — RoPE only operates on a subset of GEMV output, and they have different grid sizes.

## Batched Independent Operations

When operations are **independent** (no data dependency), batch them to suppress intermediate barriers.

### beginBatch / endBatch Pattern

```zig
// Normalize Q and K (independent — can run in parallel)
be.beginBatch();
  be.rmsNormMulti(q_buf, norm_w, nh_q, hd, eps);   // No barrier after
  be.rmsNormMulti(k_buf, norm_w, nh_kv, hd, eps);  // No barrier after
be.endBatch();  // Single barrier here
```

**Metal implementation:**

```zig
pub fn beginBatch(self: *MetalBackend) void {
    self.batch_mode = true;
}

pub fn endBatch(self: *MetalBackend) void {
    self.batch_mode = false;
    if (self.active_enc) |enc| {
        objc.msgSend(void, enc, objc.sel("memoryBarrierWithScope:"), .{
            MTLBarrierScopeBuffers,
        });
    }
}

fn encode(...) void {
    // ... dispatch kernel ...

    // Suppress barrier in batch mode
    if (!self.batch_mode) {
        objc.msgSend(void, enc, objc.sel("memoryBarrierWithScope:"), .{...});
    }
}
```

**When to use:**

- Multiple normalizations on different buffers
- RoPE on Q and K (both modify their input, but independently)
- Parallel GEMV (using gemvMulti is better, but batching is an alternative)

**Impact:** Qwen3.5 used batching for RoPE(Q) + RoPE(K) → saved ~64 barriers/token.

## Real-World Example: Qwen3.5 Optimization Journey

**Initial (naive):**
- 16 DeltaNet layers × 1 sync per Q/gate split = **16 syncs/token**
- No gemvMulti → 3 dispatches for Q/K/V projection = **~600 extra dispatches**
- No addRmsNorm → 128 extra dispatches for residual+norm
- **Throughput:** 12.3 tok/s

**Optimizations applied:**

1. **splitQGate GPU kernel** → eliminated 16 syncs
2. **gemvMulti for Q/K/V** → reduced dispatches by ~200
3. **addRmsNorm fusion** → reduced dispatches by 128
4. **Batch mode for independent norms/RoPE** → reduced barriers by 240

**Final:**
- 1 sync/token (only for final argmax)
- 690 barriers/token (down from 930)
- 994 dispatches/token
- **Throughput:** 14.1 tok/s (+15%)

**Key insight:** Even though barriers are free on Apple Silicon, reducing dispatches and syncs improves throughput by reducing CPU-side overhead and GPU command buffer size.

## Best Practices

### API Design

1. **Batched variants for common patterns:** gemvMulti, rmsNormMulti, ropeBatched
2. **Fused ops for common sequences:** addRmsNorm, siluMul
3. **CPU fallback must match semantics:** Batched CPU = sequential execution of same ops

### Implementation

1. **Profile before optimizing:** Use `--profile` to see dispatch/barrier/sync counts
2. **Benchmark impact:** Some "optimizations" don't help (e.g., barriers on Apple Silicon)
3. **Keep unfused fallback:** For debugging, keep the sequential version

### Debugging

1. **Validate output:** Fused kernel must match unfused output exactly
2. **Test edge cases:** Single element, non-multiple-of-8 sizes
3. **Check all backends:** Fusion bug on Metal but not CPU? Check threadgroup barriers.

---

**In the code:** [src/backend/backend.zig](../../src/backend/backend.zig) (gemvMulti, siluMul, addRmsNorm interfaces), [src/backend/metal.zig](../../src/backend/metal.zig) (Metal implementations), [src/backend/kernels/metal/elementwise.metal](../../src/backend/kernels/metal/elementwise.metal) (fused kernels), [src/models/qwen35.zig](../../src/models/qwen35.zig) (usage examples)

**Related:** [Chapter 11: Metal Backend Internals](11-metal-backend-internals.md#batch-mode-suppressing-intermediate-barriers)

**Back:** [Chapter 12: CPU Parallelism ←](12-cpu-parallelism.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)
