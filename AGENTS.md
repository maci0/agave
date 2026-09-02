# Agave Engineering Standards

Zig LLM inference engine. No C/C++ ML libraries. Kernels, quants, and models are written here.

`CLAUDE.md` is a symlink to this file. Do not fork a second copy.

## Commands

```bash
zig build                          # agave (ReleaseFast, stripped) + agave-debug (ReleaseSafe)
zig build test                     # unit tests at ReleaseSafe so asserts fire. Does not build agave-bench.
zig build check                    # fmt-check + docs hygiene + unit tests (local CI gate)
zig build lint-web                 # oxlint + tsc (CI lint-web; needs bun 1.4.0)
zig build fmt                      # apply zig fmt to the paths CI checks
zig build fmt-check                # check formatting without writing
bun run lint                       # oxlint (web TypeScript; blocking in CI)
bun run typecheck                  # tsc --noEmit for src/web and web
zig build bench                    # agave-bench micro-benchmarks
./zig-out/bin/agave model.gguf "prompt"
./zig-out/bin/agave model.gguf --serve
./zig-out/bin/agave model.gguf --backend cpu|webgpu|vulkan|cuda|rocm|metal
zig build -Denable-<model>=false   # gemma3 gemma4 diffusion-gemma qwen35 qwen4-exp gpt-oss nemotron-h nemotron-nano glm4 deepseek4 llama4 dflash2
zig build -Denable-<backend>=false # cpu cuda metal rocm vulkan webgpu
zig build -Denable-debug=false     # skip agave-debug
zig build -Denable-bench=false     # skip installing agave-bench
```

`--spec-mode`: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp (alias medusa), eagle, eagle3, mlp, pflash, dspark, dflash2. Flag list: `src/main.zig` `cli_specs`, or `agave --help`.

After backend or model interface changes run `zig build`, not only `zig build test`.

Docs: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md). Dispatchers: `src/backend/backend.zig`, `src/models/model.zig`, `src/format/format.zig`, `src/tokenizer/tokenizer.zig`. `--serve` UI is `src/web/` (`scripts/build-web.sh` refreshes `app.js`). Browser WASM shell is `web/`, not `src/web/`.

## Invariants

Non-negotiable. Every change must respect all of them.

### Hot path (token generation)
- Zero allocations, zero syscalls, zero locks. No exceptions.
- `inline` small math helpers in tight loops (`silu`, `bf16ToF32`, `sigmoid`). Not large or init-only functions.
- Data-parallel CPU work uses `ThreadPool.parallelFor()`. Do not `std.Thread.spawn` for compute. Server and KV-prefetch workers may spawn.
- Pool signaling: atomics (`std.atomic.Value(T)`), not mutexes.

### Comptime
- If a type or hardware feature is known at compile time, it is a `comptime` parameter.
- Backend dispatch is a tagged union with `inline else`. No vtable in the hot path.
- Prefer Zig `@` builtins: `@Vector`/`@reduce`/`@splat`, `@exp`/`@sqrt`/`@mulAdd`, `@memcpy`/`@memset`, `@bitCast`/`@intCast`.

### Memory
- Functions that allocate take `std.mem.Allocator`. No global allocators.
- `defer obj.deinit()` immediately after acquire. `errdefer` for error-path-only cleanup. No manual cleanup in `catch`.
- `std.heap.page_allocator` only in one-time init (and page-aligned weight buffers). Forbidden in the hot path.
- Tests use `std.testing.allocator`.

### Dispatcher
- High-level code and models import `backend/backend.zig`, never `cuda.zig` / `metal.zig` / other implementations. Test-only `_ = @import` in `main.zig` is the exception.
- Backend-specific types (`CUcontext`, `hsa_queue_t`) stay private to their backend file.
- Same pattern: `models/model.zig`, `tokenizer/tokenizer.zig`, `format/format.zig`.

### GPU backends
- Missing kernels `@panic`. Never silently fall through to CPU.
- Exceptions: `embLookup` (single-row CPU read is faster) and Metal `softmax` when `n < softmax_cpu_threshold` (128). Must have a performance-justification comment.
- `--allow-cpu-fallback` is a stub (warns, does nothing). Do not add CPU fallbacks behind it.

### Quantization
- Dequant happens inside the kernel via comptime-unrolled paths. No full f32 conversion on the hot path.
- Explicit precision: `f32`, `f16`, `bf16`, `i8`. Mixed precision: tagged union or `comptime` type parameter.

### Naming
- `camelCase` functions, `snake_case` variables/fields/params, `PascalCase` types, `snake_case` files.
- No magic numbers. Thresholds are named module-level `const`s.

### Build
- Build system is `build.zig` + `build.zig.zon` only. Do not add a Makefile or C/C++ inference libraries. `scripts/` is profiling, docs, and the web bundle — not the build.
- `build.zig.zon` has zero Zig package dependencies. Keep it that way. CLI is `cli.zig`. Terminal I/O is `term.zig` (posix + `std.unicode`, no libc, no `wcwidth`, no terminal frameworks).
- Cross-compile must keep working: Linux x86_64, Linux aarch64, macOS aarch64.
- Production is ReleaseFast and stripped (unstripped binaries embed host paths). `agave-debug` and tests are ReleaseSafe: Debug optimize mode breaks linking with GCC 16 `.sframe`. Do not switch tests to ReleaseFast — that no-ops `std.debug.assert`.
- 11 model architectures: Gemma3, Gemma4, DiffusionGemma, Qwen3.5, Qwen4-Exp, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4, DeepSeek V4, Llama 4. DFlash2 is a block-diffusion drafter (`-Denable-dflash2`).

### Errors, docs, tests
- Explicit error sets and `try`/`catch`. Never `catch {}` except shutdown. Never `catch undefined`.
- `std.debug.assert` for internal invariants. `pub` only for intended API.
- Public functions and structs get `///` (purpose, ownership, returns, errors). Files get `//!`.
- `test` blocks at the bottom of the relevant file. Backend tests use target guards.
- Changes under `src/backend/`, `src/models/`, `src/kvcache/` include benchmarks (throughput, TTFT, VRAM, bandwidth). A >5% regression needs a written justification in the change.
- Research prototypes (Triton/CUTLASS/TVM) stay in `research/kernels/`. Port to native Zig + target IR before merging into `src/`.

### Models
- New models: `megakernel_enabled` field so `setMegakernel()` vtable dispatch works; `ModelDesc` in `mega_compose.zig` ([docs/MEGAKERNEL.md](docs/MEGAKERNEL.md) Tier 3).
- Fused FFN: `inline else => |be|` plus `comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUp...")` so Metal methods do not compile on Linux `NullBackend`. Pattern: `qwen35.zig` `mlpLayer`.
- Chat templates for prompt formatting. No hardcoded role markers.

## Gotchas

**GPU sync before argmax.** GPU writes logits. CPU argmax must `be.sync()` first or UMA platforms read stale data.

**Metal threadgroup memory ≤ 32KB.** Sum `q_local + kv_block + out_acc + scores + shared`. `makePipeline` fails silently without its error logging.

**Kernel targets.** NVIDIA `nvptx64-cuda`, AMD `amdgcn-amdhsa`. Vulkan = GLSL compute → embedded SPIR-V. WebGPU = WGSL. No OpenCL or PAL.

**macOS Vulkan** uses the KosmicKrisp ICD:
`VK_ICD_FILENAMES=$(brew --prefix)/share/android-commandlinetools/emulator/lib64/vulkan/libkosmickrisp_icd.json`

## Zig 0.16

- `main()` takes `std.process.Init`: `init.io`, `init.gpa`, `init.minimal.args`. Thread `io` through all I/O.
- Files: `Io.Dir.cwd().openFile(io, path, .{})`, `file.close(io)`, `file.readPositionalAll(io, buf, offset)`.
- Stdout/stderr: `Io.File.stdout()` / `Io.File.stderr()`, write with `posix.system.write(file.handle, ...)`.
- Durations: `sim_clock.monoMilli` / `sim_clock.monoNano` (CLOCK_MONOTONIC; override drives tests). `sim_clock.milliNow` is REALTIME: logs, seeds, epoch only. Raw `clock_gettime` stays in `sim_clock.zig` and micro-benchmarks only.
- Futex: `io.futexWaitUncancelable(u32, &atomic.raw, expected)`, `io.futexWake(u32, &atomic.raw, count)`.
- Mutex: `Io.Mutex`, `lockUncancelable(io)` / `unlock(io)`. No custom spinlocks.
- Allocators: `init.gpa`, or `std.heap.DebugAllocator` in standalone tools.
- Build: `mod.link_libc = true`, `mod.linkFramework("Metal", .{})` on Module, not Step.Compile.
- `@Type()` is gone: `@Int()`, `@Enum()`, `@Struct()`, `@Union()`.
- ArrayList: `.empty`, pass allocator to every method (`list.append(allocator, val)`).
