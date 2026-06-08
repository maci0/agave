# Review Batch 6: Appendix Tutorials

Reviewed files:
1. `docs/tutorial/appendix-atomics.md`
2. `docs/tutorial/appendix-compile-time.md`
3. `docs/tutorial/appendix-math.md`
4. `docs/tutorial/appendix-profiling.md`

Cross-referenced against: `src/thread_pool.zig`, `src/perf.zig`, `src/backend/backend.zig`, `src/backend/metal.zig`, `src/models/qwen35.zig`, `src/ops/math.zig`, `src/ops/quant.zig`, `src/backend/kernels/cpu/softmax.zig`, `src/backend/kernels/cpu/sdpa.zig`

---

## Issues Found

### appendix-profiling.md

[ERROR] appendix-profiling.md: PerfCounters.end() code snippet
  Tutorial claims: `"const elapsed: u64 = @intCast(@divFloor(std.time.nanoTimestamp() - t0, 1000));"`
  Source says: `const elapsed: u64 = @intCast(@divFloor(nanoTimestamp() - t0, 1000));` (src/perf.zig:73). The module uses a private `nanoTimestamp()` helper that calls `std.c.clock_gettime(CLOCK.REALTIME)` directly, explicitly to avoid `std.time.nanoTimestamp()` / Io virtual dispatch overhead. The tutorial even correctly documents this optimization in the `start()` comment but then shows the wrong function in `end()`.
  Fix: Change `std.time.nanoTimestamp()` to `nanoTimestamp()` in the `end()` snippet.

[ERROR] appendix-profiling.md: Metal counter printing guard condition
  Tutorial claims: `"if (self.be == .metal and self.kv_seq_len == 1) {"`
  Source says: The actual code (src/models/qwen35.zig:1656-1661) is:
  ```zig
  if (self.perf.enabled and self.kv_seq_len == 1) {
      if (comptime builtin.os.tag == .macos) switch (self.be) {
          .metal => |be| std.log.warn("Metal stats: {d} dispatches, {d} barriers, {d} syncs", .{
              be.dispatch_count, be.barrier_count, be.sync_count,
          }),
          else => {},
      };
  }
  ```
  The outer guard checks `self.perf.enabled` (not `self.be == .metal`), and the Metal check is a comptime-conditional switch inside. The tutorial's version omits the profiling-enabled requirement and uses a runtime tagged-union comparison that differs from the actual comptime-guarded switch pattern.
  Fix: Replace with the actual two-level guard pattern, or at minimum add the `self.perf.enabled` check.

[WARNING] appendix-profiling.md: gemvMlxQ code structure
  Tutorial claims: Code uses `switch (bits) { 4 => self.pipe_gemv_mlx_q4, 6 => self.pipe_gemv_mlx_q6, 8 => self.pipe_gemv_mlx_q8, else => @panic(...) }`
  Source says: The actual code (src/backend/metal.zig:1880) uses an `if` guard: `if (bits != 4 and bits != 6 and bits != 8) @panic("Metal MLX GEMV: unsupported bit width");`. There is no pipeline-selection switch; the switch is for `wpg` (words per group). The tutorial invents a code structure that doesn't exist.
  Fix: Replace with the actual `if` guard pattern, or mark the code as a simplified illustration rather than presenting it as direct source.

[WARNING] appendix-profiling.md: Instrumented operation code pattern
  Tutorial claims: The model code pattern is `self.be.gemv(x, w, y, n, k); self.be.sync();`
  Source says: The actual code uses `self.syncProfile()` (src/models/qwen35.zig:635-637), a helper that only calls `self.be.sync()` when `self.perf.enabled` is true. The tutorial's inline `self.be.sync()` would always sync, destroying throughput even without `--profile`.
  Fix: Show `self.syncProfile()` or note that sync is conditional on profiling being enabled.

### appendix-compile-time.md

[ERROR] appendix-compile-time.md: MetalBackend conditional compilation
  Tutorial claims: `"pub const MetalBackend = if (build_options.enable_metal) @import("metal.zig").MetalBackend else NullBackend;"`
  Source says: `pub const MetalBackend = if (build_options.enable_metal and builtin.os.tag == .macos) @import("metal.zig").MetalBackend else NullBackend;` (src/backend/backend.zig:259-261). The tutorial omits the `builtin.os.tag == .macos` guard, which is critical — it's the exact kind of platform-specific comptime check this appendix is supposed to teach.
  Fix: Add `and builtin.os.tag == .macos` to the condition.

[WARNING] appendix-compile-time.md: @embedFile list incomplete
  Tutorial claims: Shows 8 MSL files in the `@embedFile` concatenation, ending with `deltanet.metal`.
  Source says: The actual concatenation (src/backend/metal.zig:24-40) includes 17 files: the 8 shown plus `sdpa_tree.metal`, `gemv_tiled.metal`, `megakernel.metal`, `mega_common.metal`, `mega_qwen35_q8.metal`, `mega_gemma_q4k.metal`, `mega_gemma_q8.metal`, `mega_qwen35_q4k.metal`, and `mega_nemotron_h_q8.metal`. The tutorial says "Concatenate **all** MSL files" but omits more than half of them.
  Fix: Either add `// ... more files ...` with a comment indicating truncation, or remove the word "all".

### appendix-math.md

[WARNING] appendix-math.md: Top-K sampling description step 3
  Tutorial claims: `"3. In a single SIMD pass: mask tokens below that threshold to −∞ and apply exp\n3. Renormalize by dividing by the accumulated sum"`
  Source says: The actual `sampleToken` (src/ops/math.zig) uses unnormalized sampling. Step 5 explicitly comments: `"// 5. Weighted random sampling (unnormalized — scale threshold by sum)"` and does `const sample_threshold = rng.float(f32) * sum;`. No renormalization pass exists — the code scales the random threshold by the sum instead of dividing all logits.
  Fix: Replace step 3 with: "mask tokens below that threshold to −∞, apply exp, and accumulate the sum (no renormalization — sampling scales the threshold instead)".

[WARNING] appendix-math.md: MXFP4 lookup table value at index 8
  Tutorial claims: The MXFP4 table entry at index 8 is `0.0`
  Source says: Index 8 is `-0.0` (negative zero) in `src/ops/quant.zig:44`. In IEEE 754, `-0.0` has the sign bit set, which correctly represents the E2M1 value with sign=1, exponent=0, mantissa=0. While `-0.0 == 0.0` evaluates to true in Zig, the bit representation differs, and the tutorial's context is about exact E2M1 format representation.
  Fix: Change `0.0` to `-0.0` at index 8 in the lookup table.

---

## Sections Verified Correct (no issues)

- **appendix-atomics.md**: All thread pool code snippets (task_counter, generation, active, shutdown), memory ordering choices, CAS pattern, workerLoop, parallelFor, spinLoopHint. Verified against src/thread_pool.zig. The "never `.seq_cst`" claim confirmed — grep found zero `.seq_cst` usages across the entire codebase.
- **appendix-atomics.md**: `std.atomic.Value(T)` API usage (fetchAdd, fetchSub, cmpxchgStrong, load, store) all matches Zig 0.16 API.
- **appendix-compile-time.md**: fp8e4m3_lut comptime pattern, iq4nl_table values, Backend tagged union `inline else` dispatch, quantization block byte sizes (q4_0=18, q8_0=34, q4_k=144, q6_k=210).
- **appendix-math.md**: Argmax two-pass implementation, softmax numerical stability, GEMV count calculation (28 layers × 7 + 1 = 197), dot product / GEMV / outer product math, activation function definitions, RMSNorm formula.
- **appendix-profiling.md**: PerfCounters struct fields and Op enum, Metal counter fields (dispatch_count, barrier_count, sync_count), resetCounters implementation, softmax_cpu_threshold=128, embLookup CPU fallback.
