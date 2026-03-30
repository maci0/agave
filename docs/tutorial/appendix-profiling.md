# Appendix: Profiling and Debugging

Performance regressions are silent — the model still runs, but slower. **Profiling** makes performance visible. Agave has built-in instrumentation for dispatch counts, barriers, syncs, and per-operation timing.

## --profile Flag

**Enable profiling:** Add `--profile` to any inference command.

```bash
./zig-out/bin/agave model.gguf --profile "Test prompt"
```

**Output per token:**

```
Token 15: "world" (151ms)
  embedLookup: 0.2ms
  layer 0: 8.1ms (rmsNorm: 0.3ms, gemv×3: 6.2ms, rope: 0.1ms, sdpa: 1.3ms, ...)
  layer 1: 8.0ms
  ...
  layer 31: 8.1ms
  final_norm: 0.3ms
  lm_head_gemv: 12.1ms

Metal counters:
  Dispatches: 994
  Barriers: 690
  Syncs: 1
```

**What profiling adds:**

- **Per-operation timing:** Each gemv, rmsNorm, sdpa, etc. timed individually
- **Backend counters:** Dispatch/barrier/sync counts (Metal, CUDA, ROCm)
- **Total time per layer:** Aggregated time for each transformer layer

**Cost of profiling:** ~50% throughput loss due to additional GPU syncs (timing requires flushing command buffers).

**When to use:**

- ✅ Debugging performance regressions
- ✅ Identifying bottlenecks (which op is slow?)
- ✅ Verifying optimizations (did gemvMulti reduce dispatches?)
- ❌ Production inference (too slow)

## Profiling Implementation

### Timing Individual Operations

```zig
// src/perf.zig
pub const PerfCounters = struct {
    gemv_time: i64 = 0,
    rmsNorm_time: i64 = 0,
    rope_time: i64 = 0,
    sdpa_time: i64 = 0,
    // ...

    pub fn record(self: *PerfCounters, op: Operation, elapsed_ns: i64) void {
        switch (op) {
            .gemv => self.gemv_time += elapsed_ns,
            .rmsNorm => self.rmsNorm_time += elapsed_ns,
            // ...
        }
    }
};
```

### Instrumented Operation

```zig
pub fn gemv(self: *Model, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
    const start = if (g_profile) std.time.nanoTimestamp() else 0;

    self.be.gemv(x, w, y, n, k);

    if (g_profile) {
        self.be.sync();  // Flush GPU work (ensures timing is accurate)
        const elapsed = std.time.nanoTimestamp() - start;
        self.perf.record(.gemv, elapsed);
    }
}
```

**Key:** GPU work is deferred. Without `sync()`, you'd measure only the CPU dispatch time (~5 µs), not the actual GPU execution time.

**Trade-off:** `sync()` per operation serializes execution → 50% throughput loss. This is why profiling is only enabled by the `--profile` flag.

### Per-Layer Aggregation

```zig
pub fn forward(self: *Model, token_id: u32) !u32 {
    // ... embedding lookup ...

    for (0..self.n_layers) |layer| {
        const layer_start = if (g_profile) std.time.nanoTimestamp() else 0;

        // ... layer ops ...

        if (g_profile) {
            const layer_time = std.time.nanoTimestamp() - layer_start;
            std.log.info("  layer {d}: {d:.1}ms", .{layer, @as(f64, @floatFromInt(layer_time)) / 1e6});
        }
    }

    // ... final projection ...
}
```

## Backend Dispatch Counters

### Metal Counters

```zig
pub const MetalBackend = struct {
    dispatch_count: u32 = 0,
    barrier_count: u32 = 0,
    sync_count: u32 = 0,
    profile_counters: bool = false,
    // ...
};

fn encode(...) void {
    // ... dispatch kernel ...
    if (self.profile_counters) self.dispatch_count += 1;

    // ... insert barrier ...
    if (!self.batch_mode) {
        // ... barrier ...
        if (self.profile_counters) self.barrier_count += 1;
    }
}

fn flush() void {
    // ... commit command buffer ...
    if (self.profile_counters) self.sync_count += 1;
}
```

**Reset per token:**

```zig
pub fn resetCounters(self: *MetalBackend) void {
    if (self.profile_counters) {
        self.dispatch_count = 0;
        self.barrier_count = 0;
        self.sync_count = 0;
    }
}
```

**Print at end of token:**

```zig
if (g_profile) {
    const info = be.backendInfo();
    std.log.info("Metal: {d} dispatches, {d} barriers, {d} syncs",
        .{backend.metal.dispatch_count, backend.metal.barrier_count, backend.metal.sync_count});
}
```

### Interpreting Counts

**Dispatch count:**

- **High (>1000):** Many small kernels, dispatch overhead may dominate
- **Optimal (300-600):** Batched/fused ops, minimal overhead
- **Too low (<100):** Likely missing parallelism opportunities

**Barrier count:**

- **High (>1000):** Serialized execution, GPU can't overlap work
- **Optimal (300-700):** Batching used where possible
- **Too low (<100):** Risky — may be missing necessary synchronization

**Sync count:**

- **High (>10):** Excessive CPU/GPU round-trips, throughput loss
- **Optimal (1-3):** Only at necessary points (argmax, embedding lookup)
- **Zero:** Suspicious — CPU likely reading stale GPU data

**Example:** Qwen3.5 optimization reduced syncs from 18 → 1 per token (+15% throughput).

## Missing Kernel Policy

**Golden rule:** GPU backends must **never silently fall back to CPU**. Missing kernels must `@panic` with a clear error message.

### Enforcement

```zig
pub fn gemvMlxQ(self: *MetalBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
    const pipeline = switch (bits) {
        4 => self.pipe_gemv_mlx_q4,
        8 => self.pipe_gemv_mlx_q8,
        6 => @panic("Metal MLX 6-bit GEMV not implemented — use --backend cpu or convert to 4-bit"),
        else => @panic("Unsupported MLX bit width"),
    };
    // ... dispatch ...
}
```

**Error message requirements:**

1. **What's missing:** "Metal MLX 6-bit GEMV not implemented"
2. **Workaround:** "use --backend cpu"
3. **Alternative:** "or convert to 4-bit"

### Why @panic?

**Alternative (silent fallback):**

```zig
pub fn gemvMlxQ(...) void {
    if (bits == 6) {
        // Silently fall back to CPU
        self.be.sync();  // Flush GPU
        cpuGemvMlxQ(...);  // Run on CPU
        return;
    }
    // ... GPU path ...
}
```

**Problem:** User expects GPU performance, gets CPU performance, **doesn't realize** until they profile. Silent regressions are the worst kind.

**With @panic:**

```
$ ./agave model-6bit-mlx.gguf "Hello"
thread 1 panic: Metal MLX 6-bit GEMV not implemented — use --backend cpu or convert to 4-bit
```

User **immediately knows** there's an issue and has clear next steps.

### CPU Fallback Exceptions

**Only two cases allow CPU fallback:**

#### 1. embLookup (Single-Row Read)

```zig
pub fn embLookup(self: *MetalBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
    // Fallback to CPU: single-row lookup is faster on CPU than GPU dispatch overhead
    dequantToF32(table.data[token_id * rowBytes(dim, table.dtype) ..], output, dim);
}
```

**Why CPU is faster:**

- GPU dispatch overhead: ~10 µs
- Single-row dequant on CPU: ~2 µs (SIMD)
- GPU would be faster for batch embedding lookup, but not single-token decode

#### 2. Tiny Softmax (Below Threshold)

```zig
const softmax_cpu_threshold: usize = 128;

pub fn softmax(self: *MetalBackend, data: [*]f32, n: usize) void {
    if (n < softmax_cpu_threshold) {
        // CPU fallback: dispatch overhead dominates for tiny softmax
        cpuSoftmax(data, n);
        return;
    }
    // ... GPU path ...
}
```

**Why threshold?**

- GPU dispatch: ~10 µs base cost
- Softmax(128): ~2 µs on CPU SIMD
- Softmax(1024): ~15 µs on CPU, ~3 µs on GPU (worth the dispatch)

**Both exceptions are documented** with comments explaining the performance justification.

## Debugging Performance Regressions

### Workflow

1. **Establish baseline:** Run with `--profile` on main branch

   ```bash
   git checkout main
   ./zig-out/bin/agave model.gguf --profile "Test" > baseline.txt
   ```

2. **Test change:** Run with `--profile` on feature branch

   ```bash
   git checkout feature
   ./zig-out/bin/agave model.gguf --profile "Test" > feature.txt
   ```

3. **Compare:**

   ```bash
   diff baseline.txt feature.txt
   ```

   **Look for:**

   - Increased dispatch/barrier/sync counts
   - Slower individual operations
   - New operations (unexpected CPU fallbacks?)

4. **Isolate:** Comment out parts of the change to identify the culprit

5. **Fix:** Once identified, fix the regression

6. **Verify:** Re-run profile, confirm counters match baseline

### Example: Identifying a Regression

**Before (baseline):**

```
Metal: 690 dispatches, 690 barriers, 1 sync
Token time: 71ms (14.1 tok/s)
```

**After (regression):**

```
Metal: 706 dispatches, 930 barriers, 17 syncs
Token time: 83ms (12.0 tok/s)
```

**Analysis:**

- +16 dispatches → something new is being dispatched
- +240 barriers → batching was removed somewhere
- +16 syncs → **major red flag** — CPU/GPU round-trips added

**Investigation:** 16 syncs = 16 DeltaNet layers. Check DeltaNet code.

**Root cause:** Q/gate split moved from GPU kernel to CPU memcpy:

```zig
// REGRESSION: CPU memcpy requires sync before and after
self.be.sync();  // Sync 1 (GPU → CPU)
for (0..nh) |h| {
    @memcpy(...);  // CPU memcpy
}
// Next GPU op needs data → sync 2 (CPU → GPU)
```

**Fix:** Move split to GPU kernel (eliminates 16 syncs/token).

## Tracy Integration

Agave doesn't currently use Tracy, but here's how you'd integrate it:

### Build with Tracy

```zig
// build.zig
const tracy = b.dependency("tracy", .{});
exe.linkLibrary(tracy.artifact("tracy"));
exe.addCSourceFile(.{ .file = tracy.path("public/TracyClient.cpp"), .flags = &.{"-DTRACY_ENABLE"} });
```

### Instrument Code

```zig
const tracy = @cImport(@cInclude("tracy/Tracy.hpp"));

pub fn gemv(...) void {
    const zone = tracy.ZoneScoped();
    defer tracy.ZoneEnd(zone);

    // ... operation ...
}
```

### View Results

```bash
./tracy-profiler  # GUI shows flamegraph, GPU timelines, memory allocations
```

**Benefits:**

- Visual timeline (see parallelism, gaps)
- GPU queue visualization
- Memory allocation tracking

**Cost:** ~5-10% overhead (lower than `--profile` because no forced syncs).

## Common Profiling Patterns

### Bottleneck Identification

```bash
./agave model.gguf --profile "Test" 2>&1 | grep "ms" | sort -rn -k2
```

**Output (sorted by time):**

```
  lm_head_gemv: 12.1ms
  layer 15: 8.2ms
  layer 0: 8.1ms
  gemv×3: 6.2ms
  sdpa: 1.3ms
  rmsNorm: 0.3ms
```

**Interpretation:** `lm_head_gemv` is the bottleneck (vocab projection, large matrix).

### Regression Detection (CI)

```bash
# In CI pipeline
./agave model.gguf --profile "Test" > current.txt
./agave-baseline model.gguf --profile "Test" > baseline.txt

# Extract sync count
current_syncs=$(grep "syncs" current.txt | awk '{print $NF}')
baseline_syncs=$(grep "syncs" baseline.txt | awk '{print $NF}')

if [ "$current_syncs" -gt "$baseline_syncs" ]; then
    echo "Regression: sync count increased from $baseline_syncs to $current_syncs"
    exit 1
fi
```

**Prevents:** Silent performance regressions from merging.

### Comparative Profiling

```bash
# Compare two quantization formats
./agave model-q4.gguf --profile "Test" | grep "Token time"
./agave model-mlx.gguf --profile "Test" | grep "Token time"

# Compare two backends
./agave model.gguf --backend Metal --profile "Test" | grep "dispatches"
./agave model.gguf --backend CPU --profile "Test" | grep "layer 0"
```

## Performance Debugging Checklist

When investigating slow performance:

- [ ] Run with `--profile` to get baseline numbers
- [ ] Check sync count (should be ≤3 per token)
- [ ] Check dispatch count (should be 300-600 for typical model)
- [ ] Identify slowest operation (sort profiling output)
- [ ] Compare against expected performance (other quantization formats, backends)
- [ ] Check for unexpected CPU fallbacks (CPU time in GPU-expected ops)
- [ ] Verify batching is used (gemvMulti, rmsNormMulti, etc.)
- [ ] Check for missing fusion opportunities (sequential ops that could be fused)

## Best Practices

### Development

1. **Profile before optimizing:** Measure first, optimize second
2. **One change at a time:** Isolate what caused the improvement/regression
3. **Keep baseline numbers:** Document expected performance for each model+backend combo

### CI/CD

1. **Benchmark on merge:** Run performance suite on every PR
2. **Regression threshold:** Fail CI if throughput drops >5%
3. **Track over time:** Graph performance trends (detect gradual degradation)

### Production

1. **Never use --profile in production:** 50% throughput loss
2. **Use metrics instead:** Log tokens/sec, TTFT, latency percentiles
3. **A/B test optimizations:** Roll out changes to subset of traffic first

---

**In the code:** [src/perf.zig](../../src/perf.zig) (profiling infrastructure), [src/backend/metal.zig](../../src/backend/metal.zig) (dispatch counters), [src/main.zig](../../src/main.zig) (--profile flag handling)

**Related:** [Chapter 11: Metal Backend Internals](11-metal-backend-internals.md#profiling-counters), [Chapter 13: Batched Dispatch and Fusion](13-batched-dispatch-and-fusion.md#real-world-example-qwen35-optimization-journey)

**Back:** [Appendix: Compile-Time Optimization ←](appendix-compile-time.md)
