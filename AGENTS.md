# Agave Engineering Standards

Zig LLM inference engine. No C/C++ ML libraries. Kernels, quants, and models are written here.

`CLAUDE.md` is a symlink to this file. Do not fork a second copy.

## Commands

```bash
zig build                          # agave (ReleaseFast, stripped) + agave-debug (ReleaseSafe) + agave-bench
zig build test                     # unit tests at ReleaseSafe so asserts fire. Does not build agave-bench.
zig build ci                        # full local CI gate: check + lint-web (incl. check-web) + lint-shell + lint-python
zig build check                    # fmt-check + docs hygiene + pin consistency + unit tests + conv-store backup self-test (local CI gate)
zig build check-pins               # Zig/Docker reproducibility pins, uv.lock freshness, and exact third-party version pins agree (CI fmt-check job)
zig build check-reproducible       # build twice from different paths and byte-compare the binaries (CI reproducible-build job; compiles the engine twice)
zig build docs-check               # docs link and count hygiene (scripts/check-docs.py)
zig build conv-store-backup-test   # conversation store backup + restore self-test (docs/DURABILITY.md)
zig build lint-web                 # oxlint + tsc + ignorePatterns ratchet (CI lint-web; needs bun 1.4.0)
zig build lint-shell               # shellcheck on scripts/*.sh (CI lint-shell)
zig build lint-python              # ruff on scripts/, tests/, tools/, research/ Python (CI lint-python)
zig build check-web                # committed bundles and stylesheets match a fresh bun + Tailwind build (CI lint-web)
zig build fmt                      # apply zig fmt to the paths CI checks
zig build fmt-check                # check formatting without writing
bun run lint                       # oxlint (web TypeScript; blocking in CI)
bun run typecheck                  # tsc --noEmit for src/web and web
zig build bench                    # agave-bench micro-benchmarks
zig build run -- model.gguf "hi"   # run the ReleaseFast binary without installing
./zig-out/bin/agave model.gguf "prompt"
./zig-out/bin/agave model.gguf --serve
./zig-out/bin/agave model.gguf --backend cpu|webgpu|vulkan|cuda|rocm|metal
zig build -Denable-<model>=false   # one per model; the flag slugs are not the display names, see Build
zig build -Denable-<backend>=false # cpu cuda metal rocm vulkan webgpu
zig build -Denable-debug=false     # skip agave-debug
zig build -Denable-bench=false     # skip installing agave-bench
zig build test -Dtest-filter=<str> # only tests whose name contains <str>. Repeat to AND filters. A filter matching no `test "..."` name aborts the build (an empty match would otherwise exit 0).
zig build wasm                     # browser WASM module (web/), not src/web/. Compile only, see Gotchas.
zig build validate                 # every GPU kernel against the CPU backend. Needs the GPU; -Dvalidate-backend= picks it (default rocm).
zig build ptx                      # CUDA kernels to zig-out/ptx/*.ptx (see Build: commit them)
zig build amdgcn -Drocm-arch=gfx1100  # ROCm kernels to zig-out/rocm/kernels.hsaco (see Build)
```

`--spec-mode`: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp (alias medusa), eagle, eagle3, mlp, pflash, dspark, dflash2 (alias dflash). Flag list: `src/main.zig` `cli_specs`, or `agave --help`.

After backend or model interface changes run `zig build`, not only `zig build test`.

`zig build ci` is what a workstation reproduces. CI also runs, with no local step covering them: the macOS test job, the Docker build, the cross-compile matrix, the wasm build, PTX freshness, a bounded fuzz pass. See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

Docs: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md). Dispatchers: `src/backend/backend.zig`, `src/models/model.zig`, `src/format/format.zig`, `src/tokenizer/tokenizer.zig`. `--serve` UI is `src/web/` (React in `app.tsx`, Tailwind 4 in `app.css`, shadcn primitives in `ui/`; `scripts/build-web.sh` bundles them into the committed `src/web/app.js` and `src/web/style.css`). Browser WASM shell is `web/` (React in `shell.tsx`, `shell.css`), not `src/web/`.

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
- Build system is `build.zig` + `build.zig.zon` only. Do not add a Makefile or C/C++ inference libraries. `scripts/` holds the profiling, lint, pin, and web-bundle helpers; `build.zig` shells out to its lint, pin, docs, and artifact checks, but the build graph itself is Zig-only. `tools/` is offline model-prep utilities (`tools/README.md`), never linked into `agave`; anything the build or CI must run goes in `scripts/`.
- `build.zig.zon` has zero Zig package dependencies. Keep it that way. CLI is `src/cli.zig`. Terminal I/O is `src/term.zig` (posix + `std.unicode`, no libc, no `wcwidth`, no terminal frameworks).
- Cross-compile must keep working, matching `.github/workflows/ci.yml`: `x86_64-linux-gnu`, `aarch64-linux-gnu`, `aarch64-macos` (Metal off), `x86_64-linux-musl`, `aarch64-linux-musl` (static, CPU-only), plus the separate `wasm32-freestanding` build.
- Production is ReleaseFast and stripped (unstripped binaries embed host paths). `agave-debug` and tests are ReleaseSafe: Debug optimize mode breaks linking with GCC 16 `.sframe`. Do not switch tests to ReleaseFast — that no-ops `std.debug.assert`.
- 12 model architectures: Gemma3, Gemma4, DiffusionGemma, Qwen3.5, Qwen 3.8 Flash-Next GGUF (`qwen4exp`), Qwen4-Exp SafeTensors (`qwen4_exp`), GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4, DeepSeek V4, Llama 4. DFlash2 is a block-diffusion drafter (`-Denable-dflash2`).
- The model build flags are slugs, not the display names: `gemma3`, `gemma4`, `diffusion-gemma`, `qwen35`, `qwen4exp`, `qwen4-exp`, `gpt-oss`, `nemotron-h`, `nemotron-nano`, `glm4`, `deepseek4`, `llama4`, `dflash2`.
- Committed GPU kernel artifacts (`src/backend/kernels/**/*.ptx`, `.spv`, `.hsaco`, `.metal`, `.wgsl`) are `@embedFile`d and are *not* rebuilt by `zig build`. Editing a kernel source without regenerating its artifact ships stale GPU code, and CI's kernel-freshness job (`scripts/check-shader-artifacts.sh --ptx-only`) fails on PTX drift. Regenerate and commit:
  - PTX: `zig build ptx`, copy `zig-out/ptx/*.ptx` into `src/backend/kernels/cuda/`.
  - SPIR-V: `glslangValidator -V --target-env vulkan1.1` per `.comp` in `src/backend/kernels/vulkan/`.
  - ROCm: `zig build amdgcn`, copy `zig-out/rocm/kernels.hsaco` to `src/backend/kernels/rocm/kernels.hsaco`. The step also installs `zig-out/rocm/kernels.o` (relocatable ELF, for manual linking only); it is not the HSACO.
  - Metal and WGSL are hand-written; no compile step.

### Errors, docs, tests
- Explicit error sets and `try`/`catch`. Never `catch undefined`. `catch {}` only where the failure provably cannot affect state (advisory syscalls like `madvise`, thread affinity, best-effort cleanup) or in test cleanup; never to swallow an error that loses data or hides a failed operation.
- `std.debug.assert` for internal invariants. `pub` only for intended API.
- Public functions and structs get `///` (purpose, ownership, returns, errors). Files get `//!`.
- `test` blocks at the bottom of the relevant file. Backend tests use target guards.
- Changes under `src/backend/`, `src/models/`, `src/kvcache/` ship before/after numbers (throughput, TTFT, VRAM, bandwidth) for the touched op. Measure with `agave model.gguf "prompt" --profile` (`--profile` is a boolean flag) and `zig build bench` (see docs/CONTRIBUTING.md, "How to Debug Performance Regressions"). A >5% regression needs a written justification in the change.
- Kernel research and reference implementations stay in `research/kernels/` (Python/torch tooling plus `tilelang/` experiments). Port to native Zig + target IR before merging into `src/`.

### Models
- New models: `megakernel_enabled` field so `setMegakernel()` vtable dispatch works; `ModelDesc` in `src/backend/mega_compose.zig` ([docs/MEGAKERNEL.md](docs/MEGAKERNEL.md) Tier 3).
- Fused FFN: `inline else => |be|` plus `comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUpSiluQ5K")` (exact decl for the quant dispatched on) so Metal methods do not compile on Linux `NullBackend`. Pattern: `qwen35.zig` `mlpLayer`.
- Chat templates for prompt formatting. No hardcoded role markers.

## Gotchas

**GPU sync before argmax.** GPU writes logits. CPU argmax must `be.sync()` first or UMA platforms read stale data.

**Metal threadgroup memory ≤ 32KB.** Sum `q_local + kv_block + out_acc + scores + shared`. `makePipeline` fails silently without its error logging.

**WASM runs init, parse, and tokenize only.** A Zig 0.16 + LLVM 21 wasm32 codegen bug (invalid cast in SIMD vector lowering) blocks the full forward pass, so `agave_generate` does not run. `zig build wasm` compiles the module with Gemma3 only; every other arch is off there.

**Kernel targets.** NVIDIA `nvptx64-cuda`, AMD `amdgcn-amdhsa`. Vulkan = GLSL compute → embedded SPIR-V. WebGPU = WGSL. No OpenCL or PAL.

**macOS Vulkan** uses the KosmicKrisp ICD:
`VK_ICD_FILENAMES=$(brew --prefix)/share/android-commandlinetools/emulator/lib64/vulkan/libkosmickrisp_icd.json`

## Zig 0.16

- `main()` takes `std.process.Init`: `init.io`, `init.gpa`, `init.minimal.args`. Thread `io` through all I/O.
- Files: `Io.Dir.cwd().openFile(io, path, .{})`, `file.close(io)`, `file.readPositionalAll(io, buf, offset)`.
- Stdout/stderr: `Io.File.stdout()` / `Io.File.stderr()`, write with `posix.system.write(file.handle, ...)`.
- Durations: `sim_clock.monoMilli` / `sim_clock.monoNano` (CLOCK_MONOTONIC; override drives tests, and `--sim-clock-ms` drives a whole run). `sim_clock.milliNow` is REALTIME: logs, seeds, epoch only. Raw `clock_gettime` stays in `src/sim_clock.zig` and micro-benchmarks only.
- Futex: `io.futexWaitUncancelable(u32, &atomic.raw, expected)`, `io.futexWake(u32, &atomic.raw, count)`.
- Mutex: `Io.Mutex`, `lockUncancelable(io)` / `unlock(io)`, where an `Io` is in scope. Code without one (tokenizer, tiered KV cache, one-time init) locks with a `std.atomic.Value` spinlock.
- Allocators: `init.gpa`, or `std.heap.DebugAllocator` in standalone tools.
- Build: `mod.link_libc = true`, `mod.linkFramework("Metal", .{})` on Module, not Step.Compile.
- `@Type()` is gone: `@Int()`, `@Enum()`, `@Struct()`, `@Union()`.
- ArrayList: `.empty`, pass allocator to every method (`list.append(allocator, val)`).
