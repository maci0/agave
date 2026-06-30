# Agave Engineering Standards

High-performance LLM inference engine in Zig. Zero external ML libraries — all kernels, quantization, and model logic from scratch.

---

## Quick Reference

```bash
zig build                                      # Build (ReleaseFast + Debug)
zig build test                                 # Run all tests
./zig-out/bin/agave model.gguf "prompt"        # Run inference
./zig-out/bin/agave model.gguf --serve         # HTTP server
./zig-out/bin/agave model.gguf --backend cpu     # Force CPU backend
./zig-out/bin/agave model.gguf --backend webgpu  # Force WebGPU backend
./zig-out/bin/agave model.gguf --prefill-batch-size 256  # Chunked prefill
zig build -Denable-glm4=false                  # Disable a model at compile time
zig build -Denable-webgpu=false                # Disable WebGPU backend
```

```bash
# Speculative decoding
agave target.gguf --draft-model draft.gguf "prompt"              # Separate draft model
agave model.gguf --spec-mode ddtree "prompt"                     # DDTree self-draft
agave model.gguf --spec-mode self --draft-layers 9 "prompt"      # Self-speculative (layer skip)
agave model.gguf --spec-mode self --kv-type turbo2 "prompt"      # QuantSpec approximation: quantized KV draft (Apple ICML 2025)
agave model-medusa.gguf --spec-mode medusa "prompt"              # Medusa MLP heads (alias for mtp)
agave model.gguf --spec-mode ngram "prompt"                      # N-gram (history-based, no draft model)
agave model.gguf --spec-mode suffix "prompt"                     # Suffix decoding (cross-request exact match, dynamic k)
agave model.gguf --spec-mode lookahead "prompt"                  # Lookahead / Jacobi parallel draft (novel tokens)
agave target.gguf --draft-model eagle.gguf --spec-mode eagle "prompt"  # EAGLE (post-norm hidden-state conditioned, chains draft hidden)
agave target.gguf --draft-model eagle.gguf --spec-mode eagle3 "prompt" # EAGLE-3 (pre-output-norm hidden, richer conditioning signal)
agave target.gguf --draft-model mlp.gguf --spec-mode mlp "prompt"    # MLP Speculator (frozen target hidden, no chain)
agave target.gguf --draft-model draft.gguf --spec-token-map freq.txt --spec-mode ddtree "prompt"  # FR-Spec vocab truncation
agave model-mtp.gguf --spec-mode mtp "prompt"                    # Multi-token prediction heads
agave model.gguf --ctx-size auto "prompt"                        # Auto-fit context to available memory
agave model.gguf -t 0.8 --mirostat-mode 2 "prompt"              # Mirostat target-entropy sampling
agave model.gguf -t 0.8 --dry-multiplier 1.5 "prompt"           # DRY n-gram repetition penalty
agave target.gguf --draft-model draft.gguf --spec-mode ddtree \
  --spec-tokens 5 --tree-budget 64 "prompt"                      # Full DDTree with draft model
agave model.gguf --draft-model draft.gguf --spec-mode pflash "prompt"                             # PFlash prefill + DDTree decode
agave model.gguf --draft-model draft.gguf --pflash-scorer scorer.gguf --spec-mode pflash "prompt" # separate scorer model
agave model.gguf --draft-model draft.gguf --spec-mode pflash --pflash-alpha 0.7 "prompt"          # aggressive block compression
agave target.gguf --draft-model draft.gguf --spec-mode dspark "prompt"                            # DSpark: confidence-scheduled verification (Cheng et al. 2026)
```

```bash
agave model.gguf --megakernel "prompt"         # Fused FFN megakernel (3→1 dispatch)
```

```bash
# Multimodal (vision + video)
agave model.gguf --mmproj mmproj.gguf --image pic.png "Describe this image"
agave model.gguf --mmproj mmproj.gguf --video clip.mp4 "What happens in this video?"
agave model.gguf --mmproj mmproj.gguf --video clip.mp4 --video-fps 2 "Summarize"
```

```bash
# DiffusionGemma (block diffusion generation)
agave diffusiongemma-26B-A4B-it/ "Describe the future of AI"        # default 16 steps
agave diffusiongemma-26B-A4B-it/ --diffusion-steps 48 "..."         # more steps for quality
agave diffusiongemma-26B-A4B-it/ --diffusion-confidence 0.8 "..."   # higher acceptance bar
agave diffusiongemma-26B-A4B-it/ --diffusion-canvas 128 "..."       # smaller canvas
```

```bash
# LoRA adapters
agave model.gguf --lora adapter.gguf "prompt"                        # Merge LoRA at load time
```

```bash
# Server features
agave model.gguf --serve --sleep-after 300     # Enter sleep-mode after 5min idle
agave model.gguf --serve --no-kv-cache         # Prefill-only / embedding server
agave model.gguf --serve --max-batch-size 16   # Higher throughput for concurrent workloads
```

```bash
# Vulkan on macOS (KosmicKrisp — install via android-commandlinetools)
VK_ICD_FILENAMES=$(brew --prefix)/share/android-commandlinetools/emulator/lib64/vulkan/libkosmickrisp_icd.json \
DYLD_LIBRARY_PATH=$(brew --prefix)/lib \
agave model.gguf --backend vulkan "prompt"                         # First run ~5min (LLVM JIT)
```

```bash
# Distributed inference
agave model.gguf --list-devices                                    # Show available GPUs
agave model.gguf --backend vulkan --device 1                       # Select GPU by index
agave model.gguf --tp 2 --rank 0 --peers localhost "prompt"        # Same-node TP (shm)
agave model.gguf --pp 2 --rank 0 --peers 10.0.1.2 --transport nccl  # PP over NCCL RoCE RDMA
agave model.gguf --tp 2 --rank 0 --peers 10.0.1.2 --transport nccl  # TP over NCCL RoCE RDMA
agave model.gguf --pp 2 --rank 0 --peers 192.168.0.2 "prompt"     # Cross-node PP (TCP)
agave model.gguf --tp 2 --pp 2 --rank 0 --peers 192.168.0.2       # Hybrid TP+PP
agave model.gguf --disagg --rank 0 --peers 192.168.0.2             # Disagg prefill/decode
```

**Key files:** [build.zig](build.zig), [src/backend/backend.zig](src/backend/backend.zig) (dispatcher), [src/backend/webgpu.zig](src/backend/webgpu.zig) (WebGPU backend), [src/backend/accelerate.zig](src/backend/accelerate.zig) (Apple Accelerate.framework BLAS), [src/models/model.zig](src/models/model.zig) (model vtable), [src/models/llama4.zig](src/models/llama4.zig) (Llama 4: iRoPE, chunked attention, MoE), [src/ops/quant.zig](src/ops/quant.zig), [src/ops/attention.zig](src/ops/attention.zig), [src/ops/ssm.zig](src/ops/ssm.zig), [src/chat_template.zig](src/chat_template.zig), [src/recipe.zig](src/recipe.zig), [src/backend/mega_compose.zig](src/backend/mega_compose.zig) (megakernel composer), [src/backend/megakernel.zig](src/backend/megakernel.zig) (weight offsets), [src/term.zig](src/term.zig) (terminal I/O, key parsing, display width), [src/cli.zig](src/cli.zig) (CLI argument parser), [src/spec/ddtree.zig](src/spec/ddtree.zig) (DDTree speculative decoding), [src/spec/spec_decode.zig](src/spec/spec_decode.zig) (spec decode orchestrator), [src/spec/dspark.zig](src/spec/dspark.zig) (DSpark: confidence-scheduled verification, Markov/RNN head, SPS scheduler), [src/spec/ngram.zig](src/spec/ngram.zig) (n-gram speculative decoding), [src/grammar.zig](src/grammar.zig) (GBNF parser, JSON schema→grammar, constrained decoding), [src/server/server.zig](src/server/server.zig) (HTTP server, tool calling, vision API, prompt prefix caching), [src/server/json.zig](src/server/json.zig) (JSON parsing, sampling params, tool/vision extraction), [src/ops/math.zig](src/ops/math.zig) (sampling, penalties, logprobs), [src/ops/kv_quant.zig](src/ops/kv_quant.zig) (KV cache quantization: TurboQuant/PlanarQuant/IsoQuant/RotorQuant), [src/wasm_entry.zig](src/wasm_entry.zig) (WASM browser entry point), [src/kvcache/manager.zig](src/kvcache/manager.zig) (paged KV cache, PagedKvView), [src/kvcache/tiered.zig](src/kvcache/tiered.zig) (tiered KV cache, TransferCallback), [src/parallel/transport.zig](src/parallel/transport.zig) (TCP + shm + NCCL transport), [src/parallel/discovery.zig](src/parallel/discovery.zig) (UDP peer discovery), [src/devices/discovery.zig](src/devices/discovery.zig) (GPU device enumeration), [src/pull.zig](src/pull.zig) (HF Hub model download)

**Docs:** [KERNELS.md](docs/KERNELS.md) (kernel status per backend), [MODELS.md](docs/MODELS.md) (model params), [MEGAKERNEL.md](docs/MEGAKERNEL.md) (megakernel system), [CONTRIBUTING.md](docs/CONTRIBUTING.md) (how to add backends/models/quants), [DOCUMENTATION.md](docs/DOCUMENTATION.md) (tutorials index), [API.md](docs/API.md) (HTTP API reference)

---

## Core Rules

These rules are non-negotiable. Every change must respect all of them.

### Hot Path (token generation loop)
- **Zero allocations, zero syscalls, zero locks.** No exceptions.
- `inline` for small math helpers in tight loops (`silu`, `bf16ToF32`, `sigmoid`). Avoid on large or init-only functions.
- All parallel CPU work goes through the centralized `ThreadPool` — never spawn threads manually.
- Prefer atomics (`std.atomic.Value(T)`) over mutexes for cross-thread communication.

### Comptime & Zero-Cost Abstraction
- If a type or hardware feature is known at compile time, it **must** be a `comptime` parameter.
- Backend dispatch uses tagged union with `inline else` — zero VTable overhead.
- Use `@import("builtin")` for CPU/GPU feature detection at comptime.
- Prefer Zig `@` builtins: `@Vector`/`@reduce`/`@splat` (SIMD), `@exp`/`@sqrt`/`@mulAdd` (math), `@memcpy`/`@memset` (bulk), `@bitCast`/`@intCast` (conversions).

### Memory Management
- Functions requiring memory **must** accept `std.mem.Allocator` as parameter. No global allocators.
- `defer obj.deinit()` immediately after acquisition. `errdefer` for error-path-only cleanup.
- `std.heap.page_allocator` only in one-time initialization, forbidden in hot path.
- Use `std.testing.allocator` in tests — it detects leaks automatically.

### Dispatcher Pattern
- High-level code (`main.zig`, models) imports the dispatcher (`backend/backend.zig`), **never** the implementation (`cuda.zig`, `metal.zig`).
- Backend-specific types (`CUcontext`, `hsa_queue_t`) stay private to their backend file.
- Same pattern for models (`model.zig`), tokenizer (`tokenizer.zig`), format (`format.zig`).

### No CPU Fallbacks in GPU Backends
- GPU backends must `@panic` on missing kernels — never silently delegate to CPU.
- **Only exceptions**: `embLookup` (single-row read, CPU faster) and `softmax` below threshold (Metal only). Must have performance-justification comment.

### Quantization
- All dequantization happens **inside the kernel** via comptime-unrolled paths — never full f32 conversion in the hot path.
- Use explicit precision types everywhere: `f32`, `f16`, `bf16`, `i8`.
- Use tagged unions or `comptime` type parameters for mixed-precision dispatch.

### Naming
- `camelCase` functions, `snake_case` variables/fields/params, `PascalCase` types, `snake_case` files.
- **No magic numbers.** All thresholds and tuning constants must be named module-level `const` declarations.

### Build System
- Pure `build.zig` + `build.zig.zon`. No Makefiles, no shell scripts, no external C/C++ libs.
- Zero external dependencies. CLI parsing is self-contained (`cli.zig`). Terminal I/O is self-contained (`term.zig`).
- Every change must maintain cross-compilation for all targets (Linux x86_64/aarch64, macOS aarch64).
- `ReleaseFast` is production. `Debug` retains full safety checks.
- Model toggles: `-Denable-<model>=false` (gemma3, gemma4, diffusion-gemma, qwen35, gpt-oss, nemotron-h, nemotron-nano, glm4, llama4).
- Backend maturity: Level 0 (CPU) → Level 1 (Metal + Vulkan, current goal) → Level 2 (CUDA + ROCm optimized). WebGPU has 43 shaders (most core ops).

### Error Handling & Safety
- Use explicit Zig error sets and `try`/`catch`. Never swallow errors with `catch {}` except in shutdown paths.
- Use `std.debug.assert` for internal invariants.
- `pub` only for intended API surface.

### Documentation
- Every public function and struct gets `///` doc comments: purpose, parameter semantics (especially ownership), return value, error conditions.
- File-level `//!` doc comment for each module's purpose.

### Testing
- `test` blocks at bottom of relevant file. Use target guards for backend-specific tests.
- Golden tests against reference implementations (llama.cpp, HuggingFace).
- Categories: unit, integration, quantization accuracy, memory leak detection.

### Benchmarking
- Changes to `src/backend/`, `src/models/`, `src/kvcache/` must include benchmarks (throughput, TTFT, VRAM, bandwidth).
- **>5% regression requires explanation and approval.**
- `zig build test` does NOT compile `agave-bench` — always run `zig build` (full build) after changing backend/model interfaces.

### External Prototypes
- Triton/CUTLASS/TVM/etc. are **research-only** (keep in `research/kernels/`). Every prototype must be ported to native Zig + target IR before merging into `src/`.

---

## Gotchas

**GPU sync before argmax:** Final logits are written by the GPU. CPU argmax must call `be.sync()` first. Missing this reads stale data on UMA platforms.

**Metal threadgroup memory limit:** Must stay ≤ 32KB. Calculate total: `q_local + kv_block + out_acc + scores + shared`. Pipeline creation fails silently without the error logging in `makePipeline`.

**Zig 0.16.0 — idiomatic patterns:**
- **`main()` accepts `std.process.Init`** — provides `init.io` (Io context), `init.gpa` (allocator), `init.minimal.args` (CLI args). Thread `io` to all I/O code.
- **File I/O via `Io`** — `Io.Dir.cwd().openFile(io, path, .{})`, `file.close(io)`, `file.readPositionalAll(io, buf, offset)`.
- **Stdout/stderr via `Io.File`** — `Io.File.stdout()`, `Io.File.stderr()`. Write via `posix.system.write(file.handle, ...)`.
- **Timestamps** — `posix.system.clock_gettime(.REALTIME, &ts)` for hot-path timing (perf counters). Use `Io.Clock.real.now(io)` for non-hot-path.
- **Futex** — `io.futexWaitUncancelable(u32, &atomic.raw, expected)` and `io.futexWake(u32, &atomic.raw, count)`. No raw `__ulock_wait`/`linux.futex_wait`.
- **Mutex** — `Io.Mutex` with `lockUncancelable(io)`/`unlock(io)`. No custom spinlocks.
- **Allocators** — `init.gpa` from Init, or `std.heap.DebugAllocator` for standalone tools.
- **Build system** — `mod.link_libc = true`, `mod.linkFramework("Metal", .{})`. Methods on Module, not Step.Compile.
- **Type creation** — `@Type()` removed → `@Int()`, `@Enum()`, `@Struct()`, `@Union()`.
- **ArrayList** — `.empty` initializer, pass allocator to every method: `list.append(allocator, val)`.
- **Terminal I/O** — `src/term.zig` provides key parsing, display width, ANSI sequences. Pure Zig, no libc, no external deps.
- **No external deps** — vaxis/uucode/zigimg/clap removed. Zero external dependencies.

**Megakernel composability:** When adding a new model, define a `ModelDesc` in `mega_compose.zig` to auto-generate the megakernel MSL. No hand-written .metal files needed. See [MEGAKERNEL.md](docs/MEGAKERNEL.md) Tier 3.

**Cross-platform megakernel dispatch:** Use `inline else => |be|` with `comptime @hasDecl(@TypeOf(be.*), "fusedFfnGateUp...")` to avoid compiling Metal-specific fused FFN methods on Linux (where Metal backend is NullBackend). See `qwen35.zig` mlpLayer for the pattern.

**Kernel targets:** NVIDIA = `nvptx64-cuda`, AMD = `amdgcn-amdhsa`. Vulkan kernels are GLSL compute shaders pre-compiled to embedded SPIR-V. WebGPU kernels are WGSL compute shaders. Do not use OpenCL or PAL variants.

---

## Anti-Patterns (one-line reminders)

- **No `page_allocator` in hot paths** — pass allocator as parameter, use page_allocator only in init
- **No direct backend imports** — `@import("backend/backend.zig")`, never `@import("backend/cuda.zig")`
- **No manual thread spawn** — use `thread_pool.parallelFor()`
- **No pre-dequantization** — pass quantized tensors to `be.gemv()`, dequant happens in-kernel
- **No silent error swallowing** — `try` to propagate, `catch` with logging, never `catch undefined`
- **No inline magic numbers** — extract to named `const` at module level
- **No manual error-path cleanup** — use `defer`/`errdefer`, never manual cleanup in catch blocks
- **No compat wrappers** — use native Zig 0.16 `std.Io` APIs. Thread `io` from `main(Init)`
- **No external deps for terminal** — use `term.zig`, not vaxis or other terminal frameworks
- **No libc for terminal I/O** — `term.zig` uses `posix.read`/`posix.system.write` and `std.unicode`, no `wcwidth`/libc

---

## Agent Meta-Instructions

1. **Analyze the Hot Path:** Before suggesting code for the inference loop, verify zero syscalls, zero locks, zero allocations.
2. **Strict Target Adherence:** Check target `cpu` and `os` before writing code. No Metal for Linux, no CUDA when working on ROCm.
3. **Comptime First:** Backend dispatchers use `comptime` to switch implementations — no runtime vtable overhead.
4. **No Leaks:** Backend-specific types (CUDA stream handles, Metal command buffers) must not appear in `main.zig`.

---

## Code Review Checklist

Before approving any PR, verify:
- [ ] Hot path is allocation-free, lock-free, and syscall-free
- [ ] All memory explicitly passed via allocator + proper `deinit()`
- [ ] `defer` and `errdefer` used correctly (no manual cleanup paths)
- [ ] Comptime used aggressively for dispatch and type specialization
- [ ] Zig `@` builtins used where applicable
- [ ] Backend code only accessed through dispatcher
- [ ] No magic numbers — all thresholds are named module-level constants
- [ ] Chat templates used for prompt formatting (no hardcoded role markers)
- [ ] Benchmarks included, >5% regression explained
- [ ] Cross-compilation still works
- [ ] Quantization stays in-kernel (including NVFP4, MXFP4, bf16 paths)
- [ ] KV cache strategy documented and bounded
- [ ] All kernels are native Zig / IR (no unported prototypes)
- [ ] No CPU fallbacks in GPU backends unless provably faster
- [ ] Megakernel fused FFN uses `comptime @hasDecl` for cross-platform dispatch
- [ ] New models have `megakernel_enabled` field for `setMegakernel()` vtable dispatch

---

## Project

**Agave — Production-Ready LLM Inference Engine**

Supports 6 backends (Metal, CUDA, Vulkan, ROCm, WebGPU, CPU), 9 model architectures (Gemma3/4, DiffusionGemma, Qwen3.5/Nex-N2-Pro, GPT-OSS, Nemotron-H/Nano, GLM-4, Llama 4), and extensive quantization (Q2-Q8, FP8, bf16, NVFP4, MXFP4, MLX, TurboQuant KV). Megakernel system with composable building blocks for fused GPU dispatch. CLI and HTTP server interfaces. Built with Zig 0.16.0. Zero external dependencies.

**Core Value:** Every supported model must produce correct output on every backend at full GPU speed.

**Constraints:**
- **Pure Zig**: No external C/C++ inference libraries. All kernels native Zig, MSL, PTX, SPIR-V, WGSL
- **Hot Path**: Zero allocations, zero syscalls, zero locks in token generation loop
- **Cross-Platform**: Must cross-compile for Linux x86_64, Linux aarch64, macOS aarch64
- **No Regressions**: Performance changes must be benchmarked. >5% regression requires justification

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.

## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
