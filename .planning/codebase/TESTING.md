# Testing Patterns

**Analysis Date:** 2026-03-21

## Test Framework

**Runner:**
- Zig built-in `test` blocks (no external test framework)
- Run with `zig build test`
- All tests in `.zig` source files, placed at bottom of each file

**Assertion Library:**
- `std.testing` module from stdlib
- Common functions: `expectEqual()`, `expectApproxEqAbs()`, `allocator` (detects leaks)

**Run Commands:**
```bash
zig build test              # Run all tests
zig build test -Denable-glm4=false  # Run tests excluding disabled models
zig test src/thread_pool.zig # Single-file test (for development)
```

## Test File Organization

**Location:**
- In-place: test blocks at the bottom of each `.zig` source file
- Tests are in the same file as the code they test (e.g., `src/thread_pool.zig` contains `test "ThreadPool..."`)
- No separate `tests/` directory; one test block per test concept

**Naming:**
- `test "short descriptive name"` format
- Names are human-readable phrases describing what's tested
- Examples: `"ThreadPool basic parallelFor"`, `"bf16ToF32"`, `"softmax max element gets highest probability"`

**Structure:**
- Test blocks are `pub fn` conceptually; they're just blocks that run at test time
- Each test is independent and can be run alone
- Common: create local struct with context if needed (see thread pool tests)

## Test Structure

**Suite Organization:**
```zig
test "ThreadPool basic parallelFor" {
    // 1. Setup
    var pool = ThreadPool.init(3);
    pool.spawn();
    defer pool.deinit();  // Always defer cleanup

    var results: [100]f32 = undefined;

    // 2. Create context struct for work function
    const Ctx = struct {
        out: *[100]f32,
        fn work(ctx_ptr: *anyopaque, start: usize, end: usize) void {
            const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
            for (start..end) |i| {
                ctx.out[i] = @floatFromInt(i * i);
            }
        }
    };
    var ctx = Ctx{ .out = &results };

    // 3. Execute
    pool.parallelFor(100, 8, @ptrCast(&ctx), Ctx.work);

    // 4. Assert
    for (0..100) |i| {
        try std.testing.expectApproxEqAbs(
            @as(f32, @floatFromInt(i * i)),
            results[i],
            0.001,
        );
    }
}
```

**Patterns:**
- Setup: allocate resources, initialize test data
- Teardown: `defer` cleanup immediately after setup (guaranteed even on assertion failure)
- Assertions: use `try std.testing.expectXxx()` — `try` propagates failures
- Context structs: nested `struct` with `fn` for function pointers (used by thread pool, dispatch tests)

## Mocking

**Framework:**
- No mocking framework used; data-driven testing preferred
- For backend tests: call backend directly with known inputs, verify outputs

**Patterns:**
- **Quantization tests**: call `dequantToF32()` with known bytes, verify f32 output
  ```zig
  test "bf16ToF32" {
      try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
  }
  ```
- **Kernel tests**: create small tensors, call kernel, verify results (no GPU needed for CPU backend)
- **Dispatch tests**: tag union dispatch verified by calling backend methods
  ```zig
  test "cpu backend rms_norm via tagged union dispatch" {
      var be: Backend = .{ .cpu = try CpuBackend.init(allocator) };
      defer be.cpu.deinit();
      be.rmsNorm(input_ptr, weight_ptr, output_ptr, n, eps);
      // Verify output matches expected
  }
  ```

**What to Mock:**
- Nothing; use concrete implementations with test data
- GPU backends can be skipped with `if (builtin.os.tag == .macos)` guards

**What NOT to Mock:**
- Backend implementations themselves (test via direct calls)
- Allocators (use `std.testing.allocator` which detects leaks)

## Fixtures and Factories

**Test Data:**
- Hardcoded small tensors inline in test blocks
- Example (from `softmax.zig`):
  ```zig
  test "softmax uniform" {
      var input: [8]f32 = .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
      var output: [8]f32 = undefined;
      softmax(&input, &output, 8);
      // Assert all output values are 0.125
  }
  ```

**Location:**
- Test data created locally within each test block
- No shared test data files (keep tests self-contained)
- Reference implementations (for golden tests) in `research/kernels/` outside main build

**Allocator:**
- Tests use `std.testing.allocator` which detects memory leaks
- Example (from `kvcache/manager.zig` test):
  ```zig
  test "KVCache allocation and cleanup" {
      const allocator = std.testing.allocator;
      var cache = try KVCache.init(allocator, 2048);
      defer cache.deinit();
      // ... operations ...
  }
  ```

## Coverage

**Requirements:**
- No formal coverage percentage enforced
- Core paths must have tests: quantization, backend dispatch, memory allocation
- Edge cases: empty inputs, single elements, non-power-of-2 sizes

**View Coverage:**
- No built-in coverage tool; manual review of untested code
- Mark future gaps with `// TODO: test` comments

**Gap Identification:**
- CPU backends: most core ops tested (gemv, softmax, norm, rope, activation)
- GPU backends: tested via dispatch + CPU fallback verification
- Models: integration tests via end-to-end inference (not unit tests)
- Format parsers: tested via actual GGUF/SafeTensors files

## Test Types

**Unit Tests:**
- Scope: Single function or small kernel
- Approach: Direct function call with known inputs, verify outputs
- Examples:
  - `test "bf16ToF32"` — quantization conversion
  - `test "softmax uniform"` — mathematical correctness
  - `test "ThreadPool basic parallelFor"` — thread pool work distribution

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Backend dispatch test (loads CPU backend, calls multiple ops)
- Example (from `main.zig`):
  ```zig
  test "cpu backend rms_norm via tagged union dispatch" {
      // Loads CpuBackend, calls rmsNorm through Backend union
      // Verifies output matches naive Zig computation
  }
  ```

**E2E Tests:**
- Scope: Full inference pipeline (not in unit test suite)
- Approach: Run CLI on actual model file, compare output to reference (golden test)
- Location: Implicit (via running `./zig-out/bin/agave model.gguf "prompt"`)
- Reference: `research/kernels/` contains Python scripts for generating golden outputs

**Correctness Tests:**
- Quantization tests verify dequant output matches reference within tolerance
- Example (from `ops/quant.zig`):
  ```zig
  test "bf16ToF32" {
      // Verify against known value
      try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
  }
  ```

## Common Patterns

**Async Testing:**
- N/A for CPU tests (synchronous by nature)
- GPU backend tests skip on non-target platform: `if (comptime builtin.os.tag == .macos)`

**Error Testing:**
- Test error propagation via `try`:
  ```zig
  test "Tokenizer encode error propagates through VTable" {
      const result = tokenizer.encode(allocator, "test");
      try std.testing.expectError(error.InvalidInput, result);
  }
  ```

**Dispatch Testing:**
- Verify `inline else` works correctly:
  ```zig
  test "cpu backend softmax via tagged union dispatch" {
      var be: Backend = .{ .cpu = try CpuBackend.init(allocator) };
      defer be.cpu.deinit();
      be.softmax(x_ptr, y_ptr, n);
      // Verify y matches reference
  }
  ```

**Parallel Testing:**
- Thread pool tests verify work distribution:
  ```zig
  test "ThreadPool basic parallelFor" {
      // Spawn 3 workers, do work in parallel, verify all chunks executed
  }
  ```

**Memory Leak Detection:**
- Automatic via `std.testing.allocator`:
  ```zig
  const allocator = std.testing.allocator;
  var cache = try KVCache.init(allocator, n_layers);
  defer cache.deinit();
  // allocator automatically fails test if any bytes leaked
  ```

## Target-Specific Testing

**Platform Guards:**
- Use `if (comptime builtin.os.tag == .macos)` for Metal tests
- Use `if (comptime builtin.os.tag == .linux)` for Vulkan/CUDA tests
- Gated imports: backend tests only run on supported platforms

**Example (from `cpu.zig`):**
```zig
if (comptime builtin.os.tag == .macos) {
    // macOS-only sysctl call
    const rc = std.c.sysctlbyname("machdep.cpu.brand_string", ...);
}
```

## Test Coverage Priorities

**High Priority (tested):**
- Backend dispatch: all backends must implement same interface
- Quantization: all dequant formats must handle edge cases (0, min, max values)
- Memory: allocation/deallocation via `std.testing.allocator`
- Softmax, RMSNorm: mathematical correctness
- Threading: work distribution and synchronization

**Medium Priority (spot-checked):**
- Format parsing: a few real GGUF/SafeTensors files, not exhaustive
- Tokenizer: a few encode/decode roundtrips

**Low Priority (implicit via E2E):**
- Full model inference: tested by running CLI, not isolated
- Attention: tested via backend dispatch (GPU backend verification)
- GEMV: core CPU tests exist; GPU tests via numerical correctness

## Running Tests in CI

**Matrix:**
- CPU on Linux x86_64, aarch64, macOS aarch64
- Metal on macOS (aarch64)
- Vulkan on Linux (x86_64, aarch64)
- (Future) CUDA/ROCm on GPU runners

**Local Development:**
```bash
zig build test                           # All tests
zig build test -Denable-glm4=false      # Disable GLM4 model tests
zig test src/backend/kernels/cpu/softmax.zig  # Single kernel tests
```

---

*Testing analysis: 2026-03-21*
