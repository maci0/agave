# Contributing to Agave

Templates and step-by-step guides for extending the inference engine.

## Development setup

Install **Zig 0.16.0** from https://ziglang.org/download/ (pin: [`.zigversion`](../.zigversion), also `build.zig.zon` `.minimum_zig_version`). `zig build` exits if the running compiler does not match that pin. GPU backends `dlopen` drivers at runtime; no GPU SDK is needed to compile.

`zig build check` also needs **Python 3.11+** (`scripts/check-docs.py`). TypeScript gates need **bun 1.4.0** (`package.json` `packageManager`, enforced exactly by `scripts/lint-web.sh`) and `bun install --frozen-lockfile`. The remaining `zig build ci` halves need **shellcheck** (`zig build lint-shell`) and **ruff** (`zig build lint-python`). `scripts/lint-python.sh` runs a `ruff` on PATH when it already reports the `ruff.toml` pin, and otherwise runs the pinned ruff ephemerally through `uvx`, the same way CI does, so no global tool install is required; with neither available it names both ways to get the pin.

```bash
zig version          # must print 0.16.0
zig build            # agave (ReleaseFast) + agave-debug (ReleaseSafe)
zig build test       # unit tests
zig build --help     # all steps
```

Before a PR, run `zig build ci`, the local half of the blocking `ci-pass` gate: `check` (format check + docs hygiene + pin consistency + unit tests), `lint-web` (oxlint + tsc + web artifact freshness + the oxlint `ignorePatterns` ratchet), `lint-shell` (shellcheck) and `lint-python` (ruff). Run the halves separately when one toolchain is not installed: `check` needs only Python 3.11+, `lint-web` needs bun 1.4.0. `check` covers the Zig jobs (fmt-check, pin consistency, unit tests, and docs-check), `lint-web` is the blocking TypeScript job, `lint-shell` is the blocking shell job, and `lint-python` is the blocking Python job. CI jobs that cannot run on a workstation (Docker, cross-compile, wasm, PTX freshness) are listed below, as are the extra jobs that fire on specific surfaces:

| You changed | Also run |
|---|---|
| CUDA kernel sources under `src/backend/kernels/cuda/` | `scripts/check-shader-artifacts.sh --ptx-only` (then regenerate with `zig build ptx -Dcuda-sm=sm_120` if it drifts) |
| WASM / `src/wasm_entry.zig` / `web/` | `zig build wasm` |
| Docs, changelog, version pins | `python3 scripts/check-docs.py` (also part of `zig build check`) |
| `Dockerfile`, `.zigversion`, `ruff.toml`, or `package.json` (bumping a pin) | `zig build check-pins` (also part of `zig build check`; the Zig version, the Dockerfile's Zig download checksums, the Debian snapshot day, `SOURCE_DATE_EPOCH`, apt source isolation, the listen port, the ruff and bun versions, the committed `uv.lock` files, the vendored anti-slop source against `tools/oxlint/anti-slop/VENDORED.sha256`, and every exact third-party version in `package.json` and the `pyproject.toml` files must agree) |
| `tools/oxlint/anti-slop/**` (re-vendoring) | `zig build check-pins` after regenerating `tools/oxlint/anti-slop/VENDORED.sha256`; record the upstream commit in that directory's `README.md` and copy upstream's LICENSE file there |
| Built-in chat UI and browser shell (React + Tailwind 4) | `scripts/build-web.sh` (needs bun 1.4.0 and `bun install --frozen-lockfile`) |
| `src/web/` / `web/` TypeScript | `zig build lint-web` and `scripts/check-web-artifacts.sh` (both part of the blocking CI job `lint-web`) |
| `.oxlintrc.json` | `bash scripts/check-web-lint-scope.sh` (an `ignorePatterns` entry that is not known debt fails; entries may leave the list, never join it) |
| `scripts/*.sh` | `zig build lint-shell` (blocking CI job `lint-shell`) |
| `build.zig` or anything that can put a path, timestamp or hostname in a binary | `zig build check-reproducible` (blocking CI job `reproducible-build`; builds twice from different paths and byte-compares) |
| `scripts/`, `tests/`, `tools/`, `research/` Python | `zig build lint-python` (blocking CI job `lint-python`) |
| `Dockerfile`, `docker-compose.yml` | `AGAVE_API_KEY=local docker compose config --quiet`, then build the image the way the `docker-build` job does (its `build-args` in `.github/workflows/ci.yml`, tagged `agave:ci`) and run `bash scripts/check-docker-image.sh agave:ci`. The job runs both; neither has a `zig build` step, so copy the build args rather than inventing a subset |

Dependabot (`.github/dependabot.yml`) opens weekly bumps for the GitHub
Actions, Docker, npm and Python ecosystems. Two of them cannot be merged on
their own, because `zig build check-pins` deliberately fails until a human
completes the pin:

- A `debian:bookworm-YYYYMMDD-slim` bump also needs `DEBIAN_SNAPSHOT` (both
  stages) and `SOURCE_DATE_EPOCH` (both stages) set to the same new day.
- A `package.json` bump of the `bun` version also needs `engines.bun` and
  `lint-web`'s `bun-version` to match; `scripts/check-pins.sh` and
  `scripts/check-docs.py` check the same three.

A Python bump under `tests/` or `research/kernels/` comes from the `uv`
ecosystem, which relocks `uv.lock`; `check-pins` rejects a manifest that has
moved without its lock.

Weights for golden and e2e tests go in a local `./models` directory (gitignored). Do not commit a symlink.

## Golden Tests (manual, weights required)

`golden_tests.yml` is `workflow_dispatch` only: GitHub-hosted runners have no model weights, so the CPU, Metal, and Vulkan matrix entries all fail on a hosted runner by design. Dispatch it with the `runner` input set to a self-hosted label whose checkout has `./models` populated (and, for the Vulkan entry, a working Vulkan loader; the LunarG SDK install step is skipped on self-hosted):

```bash
gh workflow run golden_tests.yml -f runner=self-hosted
```

Leaving `runner` empty keeps the matrix on the GitHub-hosted labels, where the run stops at the `Require local GGUF models` step. Without `runner` there is no way to reach a machine that has weights, so the workflow always fails.

`ci-pass` also requires the `fuzz-smoke`, `docker-build`, `cross-compile-check`, `wasm-build`, `kernel-artifacts`, and `reproducible-build` jobs. `zig build ci` does not cover them. Fuzz smoke runs anywhere (`zig build test --fuzz=1000 --summary all`); the rest need Docker, cross toolchains, or `glslangValidator` (SPIR-V freshness, see `scripts/check-shader-artifacts.sh`). A green `zig build ci` can still go red on those jobs after push.

Every shell step in `.github/workflows/ci.yml` is a script in `scripts/`, so `zig build lint-shell` analyses the shell CI depends on: `scripts/check-ci-pass.sh` (the `ci-pass` gate, `NEEDS_JSON=${{ toJSON(needs) }}`), `scripts/check-docker-image.sh` (the `docker-build` smoke test), `scripts/build-cross-target.sh` (the `cross-compile-check` matrix), and `scripts/build-wasm.sh` (the `wasm-build` output check). A new `run:` block with logic in it belongs in one of those, not in the YAML.

`reproducible-build` runs `scripts/check-reproducible.sh` (`zig build check-reproducible` locally). It builds the same source twice, from two different directories, with different install prefixes, cache dirs, `SOURCE_DATE_EPOCH`, timezone and locale, and requires the two `agave` binaries to be byte-identical. A mismatch means the build root, the clock or the locale is leaking into a release artifact; the script names the differing SHA256 and, when `diffoscope` is installed, what changed. Run it after touching `build.zig`, the release module options, or anything that can put a path, timestamp or hostname into a binary.

## Where New Code Goes

| Kind of change | Put it in |
|---|---|
| Compute backend / kernels | `src/backend/` (+ `kernels/<backend>/`) |
| Model architecture | `src/models/` |
| Shared math / quant / attention ops | `src/ops/` |
| Weight file formats | `src/format/` |
| KV cache policy | `src/kvcache/` |
| Speculative decoding | `src/spec/` |
| HTTP API / scheduler / metrics | `src/server/` |
| Built-in `--serve` chat UI | `src/web/` (React in `app.tsx`, state in `chat/`, shadcn primitives in `ui/`; `scripts/build-web.sh` refreshes `app.js` and `style.css`, needs `bun`) |
| Browser WASM shell | `web/` (not `src/web/`; TypeScript in `agave.ts`, React in `shell.tsx`, same build script) |
| Tokenizer | `src/tokenizer/` |
| Distributed TP/PP transport | `src/parallel/` |
| Local GPU enumeration | `src/devices/` |
| CLI flags / REPL wiring | `src/cli.zig`, `src/main.zig` |
| Architecture enum / chat templates | `src/arch.zig`, `src/chat_template.zig` |
| Directional steering | `src/steering.zig` (CLI via `main.zig`) |
| NLL eval / MoE expert profile+cache | `src/eval.zig`, `src/expert_profile.zig`, `src/expert_cache.zig` (library; wire CLI in `main.zig` when ready) |
| Image placeholder token IDs | `src/image_tokens.zig` (shared by `arch.zig` / `chat_template.zig`) |

Import through package dispatchers (`backend/backend.zig`, `models/model.zig`, `format/format.zig`, `tokenizer/tokenizer.zig`). Do not import concrete backend or model files from outside their package.

## How to Add a New Backend

Existing backends: CPU (`cpu.zig`), Metal (`metal.zig`), Vulkan (`vulkan.zig`), CUDA (`cuda.zig`), ROCm (`rocm.zig`), WebGPU (`webgpu.zig`).

1. Create `src/backend/yourbackend.zig`
2. Implement the full backend interface, core ops (`gemv`, `rmsNorm`, `rope`, `sdpa`, `sync`, etc.) plus fused variants (`siluMul`, `addRmsNorm`, `sdpaPrefill`, `gemvMulti`, ...). See `src/backend/backend.zig` for the complete dispatch interface; every function must be implemented or `@panic` on unsupported ops
3. Add variant to the `Backend` tagged union in `src/backend/backend.zig`
4. Add backend-specific tests in your implementation file
5. Update `build.zig` with target-specific compilation flags
6. Add entry to `docs/KERNELS.md`
7. Add GPU kernels in `src/backend/kernels/yourbackend/`, shader format depends on backend (MSL for Metal, SPIR-V for Vulkan, PTX for CUDA, HSACO for ROCm, WGSL for WebGPU)

**Template:**
```zig
// src/backend/yourbackend.zig
const std = @import("std");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;

pub const YourBackend = struct {
    pub fn init(allocator: std.mem.Allocator) !YourBackend { }
    pub fn deinit(self: *YourBackend) void { }

    pub fn gemv(self: *YourBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void { }
    pub fn rmsNorm(self: *YourBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void { }
    pub fn rope(self: *YourBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void { }
    pub fn sdpa(self: *YourBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void { }
    pub fn sync(self: *YourBackend) void { }
    // ... see cpu.zig for remaining functions
};
```

## How to Add a New Model Architecture

1. Create `src/models/yourmodel.zig`
2. Implement the model interface (init, forward, prefill, deinit)
3. Add to `src/models/model.zig` (conditional import gated by `build_options.enable_yourmodel`)
4. Add `enable-yourmodel` build flag in `build.zig` (both `b.option()` and `backend_options.addOption()`)
5. Add variant to `Arch` enum in `src/arch.zig` with `detect`, `displayName`, `chatTemplate`, `isEnabled`, `buildFlag` methods
6. Add to `initAndRun` switch in `src/main.zig`
7. Add weight loading in your model's `init()` using the `Format` interface (`getTensor`, `layerTensor`, `getMetaU32`, etc.)
8. Add golden test against reference implementation
9. Add `ARG ENABLE_YOURMODEL=true` to the Dockerfile and pass `-Denable-yourmodel` in the image `zig build` (defaults are on; omitting this makes the flag undisableable)
10. Turn it off in the Gemma3-only local image: `docker-compose.yml`, CI `docker-build` (`ENABLE_YOURMODEL=false`), and the README "Minimal build" `--build-arg` list (`scripts/check-docs.py` enforces this)

**Required interface** (see `src/models/model.zig` for the vtable contract):
```zig
pub const YourModel = struct {
    // Required fields (read by model.zig vtable):
    eos_token_id: u32,
    vocab_size: u32,
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    logits_buf: []f32,
    kv_seq_len: usize = 0,

    // Implementation fields:
    fmt: Format,
    be: Backend,
    allocator: Allocator,

    // Optional fields (vtable detects via @hasField/@hasDecl):
    // image_embeddings: ?[]const f32 = null,   // multimodal vision
    // n_visual_tokens: u32 = 0,
    // image_pad_token_id: u32 = 0,
    // visual_token_idx: u32 = 0,
    // n_mtp_layers: u32 = 0,                   // MTP speculation
    // mtp_logits_buf: []f32 = &.{},
    // layer_skip_start: u32 = 0,               // self-speculative
    // layer_skip_end: u32 = 0,
    // megakernel_enabled: bool = false,         // fused FFN dispatch

    pub fn init(allocator: Allocator, fmt: Format, be: Backend, ctx_size: u32, kv_type_k: KvQuantType, kv_type_v: KvQuantType, tiered_cache: ?*TieredKvCache) !YourModel { }
    pub fn deinit(self: *YourModel) void { }
    pub fn forward(self: *YourModel, token_id: u32) ForwardError!u32 { }
    pub fn prefill(self: *YourModel, token_ids: []const u32) ForwardError!u32 { }
    pub fn resetCache(self: *YourModel) void { }
    pub fn cancel(self: *YourModel) void { }
    pub fn getBlockTable(self: *YourModel) []const u32 { }
};
```

## How to Add Megakernel Support for a New Model

The composable megakernel generator (`src/backend/mega_compose.zig`) auto-generates model-specific Metal megakernels from metadata. No MSL or shader code is needed -- just define a `ModelDesc`.

1. In your model's `init()`, populate a `ModelDesc` from model metadata:
   ```zig
   const mega_compose = @import("backend/mega_compose.zig");
   const desc = mega_compose.ModelDesc{
       .name = "yourmodel",
       .n_layers = fmt.getMetaU32("num_hidden_layers"),
       .n_embd = fmt.getMetaU32("hidden_size"),
       .n_ff = fmt.getMetaU32("intermediate_size"),
       .n_head = fmt.getMetaU32("num_attention_heads"),
       .n_kv = fmt.getMetaU32("num_key_value_heads"),
       .head_dim = n_embd / n_head,
       .rope_dim = head_dim,
       .rope_theta = fmt.getMetaF32("rope_theta"),
       .rms_eps = fmt.getMetaF32("rms_norm_eps"),
       .max_seq_len = ctx_size,
       .activation = .silu,    // or .gelu, .relu_squared
       .quant = .q4_k,         // detected from weight tensors
       .layer_types = mega_compose.ModelDesc.uniform(n_layers, .attention),
   };
   ```
2. Generate MSL and compile:
   ```zig
   var buf: [32768]u8 = undefined;
   const msl = mega_compose.composeMSL(&buf, desc);
   try metal_be.compileComposedMegakernel(msl);
   ```
3. Dispatch in `forward()`:
   ```zig
   metal_be.dispatchMegakernelAuto(params);
   ```

**Layer type helpers:**
- `ModelDesc.uniform(n, .attention)` -- all attention layers (Gemma 3, dense models)
- `ModelDesc.qwenHybrid(n, interval)` -- DeltaNet + attention hybrid (Qwen 3.5)
- Custom: populate `layer_types` array directly for mixed architectures (Nemotron-H, DeepSeek V4 with MLA + hyper connections + MoE)

**Model-specific Metal kernels:** Models with unique inter-layer operations can add dedicated MSL kernels (e.g., `ds4.metal` + `ds4_fused.metal` for DeepSeek V4). Use `comptime @hasDecl` to dispatch to model-specific Metal functions and fall back to CPU for non-Metal backends.

**Dedicated CpuBackend bypass:** DeepSeek V4 Flash keeps a model-owned `CpuBackend` for CSA/HCA, LID, HC mixing, rmsNorm, and q8_0 SDPA (those ops read host f32). Vulkan and WebGPU dispatch MLX-Q attention GEMV and MXFP4 expert GEMV through native shaders, then download so the CPU path sees fresh activations. Metal/CUDA/ROCm still route the full hot path through the dedicated CpuBackend. See `deepseek4.zig` `computeBackend()` / `gemvBackend()`.

**Optional flags:** `has_gate`, `has_qk_norm`, `has_post_attn_norm`, `fuse_residual` -- set these for model-specific structural variations.

The composer selects the correct GEMV, activation, residual pattern, and SDPA building blocks automatically. See [MEGAKERNEL.md](MEGAKERNEL.md) for the full three-tier architecture.

## How to Add a New Quantization Scheme

1. Add variant to `DType` enum in `src/format/format.zig` and wire up byte-size calculation in `src/backend/backend.zig` (`weightBytes()`)
2. Implement GEMV kernel: CPU SIMD in `src/backend/kernels/cpu/` and native GPU versions per backend (no CPU fallback in GPU backends). Dequantization happens in-kernel, never pre-dequant to f32
3. Add conversion helpers in `src/ops/quant.zig` if the format needs custom type conversions (e.g., `fp8e4m3ToF32`)
4. Update backend dispatch to include new format (add GEMV variant in `backend.zig`)
5. Add GEMM kernel for batched prefill: Metal in `gemm.metal` (reuse the `<format>_block_dot` helpers from `gemv.metal`), pipeline in `metal.zig`, dispatch in `gemm()`. Pattern: one threadgroup per output row, loop over n_tok tokens
6. For compressed-tensors formats (NVFP4, etc.): add fusion logic in `safetensors.zig` `fuseNvfp4Experts()` to combine per-expert weight_packed/weight_scale/weight_global_scale into GGUF-named entries
7. Benchmark against existing formats
8. Add to Quantization Types table in `docs/ARCHITECTURE.md`
9. Golden tests against reference implementation

## How to Add CLI Arguments

CLI arguments are parsed by `src/cli.zig` (self-contained, zero dependencies). To add a new flag or option:

1. Add an `ArgSpec` entry to the `cli_specs` array in `src/main.zig`
2. For flags (bool): `.{ .long = "my-flag", .short = 'f', .kind = .flag }`
3. For options (string): `.{ .long = "my-option", .kind = .option }`
4. Access in `parseCli()`: `res.flag("my-flag")`, `res.option("my-option")`, `res.optionU32("my-option")`
5. Add to `printUsage()` help text **and** the CLI Options block in `README.md`
6. If the flag is user-facing, mention it in Features (README) when it is a major capability

## Docs update checklist (after CHANGELOG entries)

When shipping a user-visible change, update in the same PR when possible:

1. `CHANGELOG.md` entry
2. `README.md` Features and/or CLI Options if the surface changed
3. `printUsage()` in `src/main.zig` (keep in sync with `cli_specs`)
4. `docs/KERNELS.md` pipeline counts from `n_pipelines` / `n_kernels` if kernels changed
5. `docs/MODELS.md` / `docs/ARCHITECTURE.md` for new arches or modules
6. `docs/TEST_MATRIX.md` date + cells if correctness coverage changed
7. `docs/BENCHMARKS.md` only for measured numbers (do not invent parallel tok/s tables elsewhere)
8. Tutorial nav (Next/Back) if a new chapter was added

Run `python3 scripts/check-docs.py` (when present) for link and count hygiene.

## Diagram Palette

Mermaid blocks and the SVGs rendered from them share one palette, derived from
the chat UI tokens in `src/web/ui/theme.css` and the wordmark in `docs/logo.svg`.
Copy these classDef lines into a new diagram rather than picking colors:

```text
classDef setup     fill:#e2e5e8,stroke:#5f7480,color:#25333a
classDef sync      fill:#d0dac9,stroke:#7d9b6a,color:#2f3f26
classDef success   fill:#a9ba9e,stroke:#5f7f4c,color:#1e2b18
classDef migration fill:#f9f3ea,stroke:#c4823a,color:#4a3116
classDef danger    fill:#e8c3c3,stroke:#c05c5c,color:#4a1c1c
classDef optional  fill:#d2bcb6,stroke:#a97f75,color:#3d2520
```

Role meanings are unchanged from the original set: `setup` is a prerequisite,
`sync` an incremental update, `success` a finished state, `migration` work in
flight, `danger` a failure, `optional` something shared. Unstyled nodes use the
neutral ink ramp.

Two rules hold the palette together, so keep them if you edit it:

- Fills sit on a geometric luminance ladder, so neighboring roles separate by at
  least 1.15 in luminance and a diagram still reads without color. `sync` and
  `success` share the agave green hue and are told apart by lightness alone.
- Every fill clears 4.5:1 for its text and every stroke clears 3:1 against the
  white canvas (WCAG 1.4.3 and 1.4.11).

To re-render `docs/diagrams/`, see `docs/render-diagrams.mjs`.

## How to Add a New Chat Template

1. Add a `pub const` to `src/chat_template.zig` with role prefixes/suffixes and EOG token names
2. Map arch → template in `src/arch.zig: Arch.chatTemplate()`
3. Add format test verifying correct prompt assembly

## How to Add a New Recipe

1. Add a `Preset` entry to the `presets` array in `src/recipe.zig`
2. Set match criteria: `arch_prefix`, `backend`, `quant` (empty string = "any")
3. Only set fields that differ from CLI defaults (null = don't override)
4. Run `zig test src/recipe.zig` to verify matching

**Key principle**: User CLI flags always override recipe defaults.

### How to Add Vision Support

1. Add a variant to `VisionVariant` enum in `src/models/vision.zig`
2. Implement `patchEmbed`, `projectToLlm`, and any variant-specific steps (e.g., pooling, learned positional embeddings)
3. Add image token IDs (`image_pad_token_id`, `image_start_token_id`, etc.) to the arch config in `src/arch.zig`
4. Wire `setImageEmbeddings` in the `model.zig` vtable (detected via `@hasField`)
5. Add `forwardImageBatch` to the model for non-causal (bidirectional) attention over vision tokens
6. Ensure the model's `forward`/`prefill` replaces image pad tokens with vision embeddings before the main transformer pass

**Template fields** (add to your model struct):
```zig
// Optional vision fields, vtable detects these via @hasField
image_embeddings: ?[]const f32 = null,
n_visual_tokens: u32 = 0,
image_pad_token_id: u32 = 0,
visual_token_idx: u32 = 0,
```

## How to Add a KV Cache Quantization Type

KV cache quantization compresses stored K/V vectors. The pipeline is: normalize → rotate → quantize → pack. To add a new rotation-based scheme (like PlanarQuant, IsoQuant, RotorQuant):

1. Add variant to `KvQuantType` enum in `src/ops/kv_quant.zig`
2. Add entries to: `name()`, `bitsPerElement()`, `turboBits()`, `fromString()`, `kvSliceBytes()`, `kvByteOffset()`
3. Implement `myStore()`: normalize → forward rotation → Lloyd-Max quantize → pack indices
4. Implement `myDot()`: forward rotation on query → dot with codebook values (K cache path)
5. Implement `myMulAccum()`: unpack → codebook → **inverse rotation** → accumulate (V cache path, inverse rotation critical for correctness)
6. Wire into `kvStore()`, `kvDot()`, `kvMulAccum()` switch statements
7. All rotation-based types share the same storage format (f16 norm + packed indices) and Lloyd-Max codebook

Hybrid layouts (not rotation-based): `nvfp4_ds_mla` stores DeepSeek MLA caches as NVFP4 on the NoPE prefix (448 elems) and f16 on the RoPE tail (64 elems). Implement `store`/`dot`/`mulAccum` plus `kvSliceBytes`/`kvByteOffset` for 512-d records. GPU SDPA has no kernel for this layout; backends route `cpuSdpaOnly()` types to CPU `kvDot`/`kvMulAccum`.

Existing examples: `turboStore/turboDot/turboMulAccum` (WHT), `planarStore/planarDot/planarMulAccum` (Givens 2D), `isoStore/isoDot/isoMulAccum` (quaternion 4D), `rotorStore/rotorDot/rotorMulAccum` (Clifford rotor).

## How to Add a Grammar/Structured Output Format

1. For new GBNF features: extend `Parser` and `ElementType` in `src/grammar.zig`
2. For new schema types: extend `SchemaConverter.emitRule()` in `src/grammar.zig`
3. CLI: add flag to `cli_specs` in `main.zig`, wire in grammar init section
4. Server API: parse from `SamplingParams` in `src/server/json.zig`, apply in generation loop in `src/server/server.zig` (both streaming and non-streaming paths)

## How to Add Speculative Decoding Support to a New Model

Layer skip for self-speculative mode is automatic, the `layer_skip_start`/`layer_skip_end` fields and the skip check in `forward()` are required in every model. The pattern is:

```zig
// In model struct:
layer_skip_start: u32 = 0,
layer_skip_end: u32 = 0,

// In forward() layer loop:
for (0..self.n_layers) |li| {
    const l: u32 = @intCast(li);
    if (l >= self.layer_skip_start and l < self.layer_skip_end) continue;
    // ... layer computation
}
```

The `Model` VTable provides `setLayerSkip(start, end)` via `@hasField` detection, no manual wiring needed.

For tree attention support (`forwardTree`), implement a batch forward that:
1. Processes B queries through all layers with position IDs (not sequential)
2. Uses `be.sdpaTree()` instead of `be.sdpa()` for ancestor-masked attention
3. Writes B logit vectors to a tree logits buffer

See `src/spec/ddtree.zig` for the tree construction algorithm and `src/backend/kernels/cpu/sdpa_tree.zig` for the tree SDPA kernel.

## How to Debug Performance Regressions

1. **Profile per-op timing**: `./zig-out/bin/agave model.gguf "prompt" --profile` (`--profile` is a boolean flag; the prompt is positional, and profiling adds GPU syncs for ~50% throughput loss)
2. **Micro-benchmarks**: `zig build bench && ./zig-out/bin/agave-bench gemv_f32 --n=4096 --k=4096 --backend=metal`
3. **Research kernels**: `cd research/kernels && uv run run.py bench sdpa --backend cpu`
4. **Check allocations**: Use `std.testing.allocator` in tests (detects leaks automatically)
5. **Verify comptime dispatch**: Ensure `inline else` dispatch is still used in `backend.zig`

## Kernel Artifact Regeneration

The GPU kernel binaries under `src/backend/kernels/` are **generated artifacts checked into git** and embedded via `@embedFile`. `zig build` never rebuilds them, so editing a kernel source without regenerating ships stale GPU code silently. Canonical commands:

| Backend | Source | Regenerate | Verify |
|---|---|---|---|
| CUDA | `src/backend/kernels/cuda/*.zig` | `zig build ptx -Dcuda-sm=sm_120`, then copy `zig-out/ptx/*.ptx` to `src/backend/kernels/cuda/` | `scripts/check-shader-artifacts.sh --ptx-only` |
| Vulkan | `src/backend/kernels/vulkan/*.comp` | `glslangValidator -V --target-env vulkan1.1 foo.comp -o foo.spv` (per shader) | `scripts/check-shader-artifacts.sh` |
| ROCm | `src/backend/kernels/rocm/all.zig` | `zig build amdgcn` on Linux with ROCm + `ld.lld`; copy `zig-out/rocm/kernels.hsaco` to `src/backend/kernels/rocm/kernels.hsaco` | manual (needs HIP runtime) |
| Metal | `src/backend/kernels/metal/*.metal` | none, MSL is compiled from source at runtime | n/a |
| WebGPU | `src/backend/kernels/webgpu/*.wgsl` | none, WGSL is the source of truth | n/a |

Notes:
- The committed PTX targets `sm_120` (see docs/KERNELS.md). `zig build ptx` defaults to `sm_120` so a bare run matches CI (`scripts/check-shader-artifacts.sh --ptx-only`).
- CI enforces the PTX comparison as a blocking job (`kernel-artifacts`, part of `ci-pass`); regenerate committed PTX whenever kernel sources change.
- SPIR-V byte-compares are only exact for the glslang release that produced the commit; treat cross-version diffs as suspect.

## How to Run Tests

```bash
# Everything CI runs on a workstation, in one command. Needs Python 3.11+ and
# bun 1.4.0 with `bun install --frozen-lockfile`.
zig build ci

# Local CI gate (format + docs hygiene + pin consistency + unit tests +
# conversation store backup self-test). Needs Python 3.11+; run this before pushing.
zig build check

# Web TypeScript (blocking CI job lint-web). Needs bun 1.4.0 + `bun install --frozen-lockfile`.
zig build lint-web
# equivalent: bash scripts/lint-web.sh (bun pin check, oxlint ignorePatterns ratchet, oxlint, tsc)

# Shell scripts (blocking CI job lint-shell). Needs shellcheck.
zig build lint-shell
# equivalent: bash scripts/lint-shell.sh

# Format check only (same paths as .github/workflows/ci.yml fmt-check)
zig build fmt-check
# Apply formatting
zig build fmt

# Run all tests (includes leak detection via std.testing.allocator)
zig build test

# Full build (needed after changing backend/model interfaces: test target doesn't build agave-bench)
zig build

# Run only tests whose name contains a substring (repeatable flag to AND filters)
zig build test -Dtest-filter=wht32
# A filter matching no `test "..."` name under src/ or tests/ aborts the build
# before compiling: the test runner would otherwise report "All 0 tests
# passed." and exit 0, so a typo would read as a green run that tested nothing.
# Candidates: rg -n '^test "' src/ tests/

# Run with a specific backend (tests that need GPU use target guards)
zig build test -Denable-webgpu=false    # skip WebGPU tests

# Fuzz smoke (CI fuzz-smoke job)
zig build test --fuzz=1000 --summary all

# Golden tests (need ./zig-out/bin/agave and weights under ./models; skipped if missing)
zig build
zig test tests/models/test_gemma3.zig --test-filter CPU
# One backend over every model: zig test takes a single root file, so loop.
for f in tests/models/test_*.zig; do zig test "$f" --test-filter CPU; done
```

`zig test` accepts one root source file: `zig test tests/models/test_*.zig` fails with
`found another zig file ... after root source file`, so sweep a backend across models with a loop.

**Test categories:**
- **Unit tests**: `test` blocks at the bottom of each source file (run via `zig build test`)
- **Leak detection**: All tests use `std.testing.allocator`, any unfreed allocation fails the test
- **Golden tests**: `zig test tests/models/test_*.zig` is a loop, not one command (see above); skipped without weights under `./models`
- **Model × Backend matrix**: See [TEST_MATRIX.md](TEST_MATRIX.md)

### End-to-End Test Harness

`tests/harness.py` runs end-to-end correctness tests against real model files: golden reference comparison, architecture detection, multi-backend validation, and regression detection. Requires Python 3.11+ and [`rich`](https://github.com/Textualize/rich) `15.0.0`.

`tests/uv.lock` is committed, so `rich` and its three transitive packages resolve to reviewed versions with recorded hashes.

```bash
uv sync --frozen --directory tests
tests/.venv/bin/python tests/harness.py --model-dir ./models
```

---

## Code Examples

### Proper Resource Management
```zig
// GOOD: Both defer and errdefer used correctly
pub fn processRequest(allocator: Allocator, config: Config) !Result {
    var buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(buffer);

    var cache = try KVCache.init(allocator, config.max_tokens);
    errdefer cache.deinit(); // Only cleanup on error path

    try populateCache(cache, buffer);
    return Result{ .cache = cache }; // ownership transferred to caller
}
```

### Dispatcher Pattern
```zig
// src/backend/backend.zig, tagged union with inline else dispatch
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    // ...

    pub inline fn gemv(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemv(x, w, y, n, k),
        }
    }
};
```

```zig
// main.zig, GOOD: use dispatcher
be.gemv(x, weight, output, n, k);

// BAD: never import implementations directly
// const cuda = @import("backend/cuda.zig"); // WRONG
```

### Memory-Safe Test
```zig
test "KVCache allocation and cleanup" {
    const allocator = std.testing.allocator;
    var cache = try KVCache.init(allocator, 2048);
    defer cache.deinit(); // Will detect leaks automatically
    try cache.insert(0, test_key, test_value);
    try std.testing.expectEqual(1, cache.num_entries);
}
```

## How to Add a New Transport

Existing transports: TCP, POSIX shared memory (shm), NCCL (RoCE RDMA).

Transports are implemented in `src/parallel/transport.zig`. Each transport must implement:

1. **allReduceAdd(buf, n)**, Sum partial results across ranks (TP)
2. **sendBuf(buf, n)**, Point-to-point send (PP)
3. **recvBuf(buf, n)**, Point-to-point receive (PP)

### Transport Selection

Transports are selected via `--transport auto|tcp|shm|nccl`:
- `auto`: shm for localhost, tcp for remote
- `nccl`: NCCL over RoCE RDMA (requires libnccl2, ConnectX NICs)

`TransportKind.rccl` (AMD's NCCL equivalent) is declared in `src/parallel/transport.zig` but has no CLI choice or implementation yet.

UDP peer discovery (`src/parallel/peer_discovery.zig`) is a separate mechanism: rank 0 broadcasts a beacon on port 49460, other ranks discover it automatically on the same subnet.

### NCCL Architecture

NCCL uses the CUDA primary context (`cuDevicePrimaryCtxRetain`), NOT `cuCtxCreate`. This ensures NCCL's internal runtime API operations share the same context as our driver API kernel launches.

```
setupNccl():
  1. dlopen("libnccl.so.2")
  2. Resolve: ncclGetUniqueId, ncclCommInitRank, ncclAllReduce, ncclSend, ncclRecv
  3. Exchange unique ID over TCP (rank 0 generates, sends to rank 1)
  4. Defer ncclCommInitRank to first allReduceAdd call (lazy init)

allReduceAdd(buf, n):
  1. Lazy init ncclCommInitRank if first call
  2. Get device pointer via getDevicePtr(buf)
  3. If dirty (GPU data current): ncclAllReduce directly on device pointer
  4. If stale (CPU fallback): upload to staging → ncclAllReduce → download
```

### Adding a New Transport

1. Add variant to `TransportKind` enum
2. Add `--transport yourname` to `TransportChoice` in `main.zig`
3. Implement setup function (e.g., `setupYourTransport`)
4. Add transport-specific paths in `allReduceAdd`, `sendBuf`, `recvBuf`
5. Wire in `resolveTransportKind` and `setupTransport` in `main.zig`

## How to Add a New Sampler

1. **Implement** in `src/ops/math.zig`: add `pub fn applyYourSampler(logits, params)` or `pub fn sampleYourMethod(logits, params, rng)`. Operate on pre-softmax logits in-place.

2. **API params**: add fields to `SamplingParams` in `src/server/json.zig`. Add parsing in `parseSampling()`.

3. **Server wiring**: apply in ALL generation paths in `src/server/server.zig`:
   - Non-streaming: `generateEscapedN()` (search for `applyRepeatPenalty` in the function)
   - Streaming: `generateStream()` (search for `applyRepeatPenalty` in the function)
   Order: repeat_penalty → DRY → grammar mask → mirostat OR (min_p → XTC → sampleToken)

4. **CLI flags**: add to `cli_specs` array in `main.zig`, add fields to CLI struct, parse in init block. Wire into:
   - Standard decode loop (search for `applyRepeatPenalty` in `main.zig`)
   - First-token sampling (search for `sampleToken` calls near `first_target`)
   - Spec decode fallback paths (cooldown + ngram no-match)

5. **Tests**: add unit tests in `src/ops/math.zig`, fuzz test in `src/fuzz_tests.zig`

6. **Docs**: update `docs/API.md` (parameter table), `docs/tutorial/07-sampling.md`, `--help` text, `README.md`

## Versioning & Releases

Agave is a CLI + HTTP server. The consumer contract is the binary behavior, CLI
flags, and the HTTP API in `docs/API.md`, not a Zig package API.

### SemVer (0.x)

- Product version: **0.6.0**, reported by `agave --version`, `/health` `version`,
  Prometheus `agave_build_info`, and OpenAI `system_fingerprint` (`agave-v0.6.0`).
- On **0.x**, breaking changes are allowed without bumping the major digit, but
  they must be called out in `CHANGELOG.md` under **Breaking** (or **Changed**
  with an explicit compatibility note) before merge.
- At **1.0.0**, treat removed/renamed CLI flags, HTTP fields, defaults that alter
  existing request results, and on-disk format changes as major bumps; new
  features as minor; fixes as patch.
- Git tag `v1.0` (2026-03-22) is a **milestone name only**. It is not product
  SemVer `1.0.0`. Prefer tags that match the product version (for example
  `v0.6.0`) for future releases.

### Single sources of truth

| Field | Location |
|-------|----------|
| Product SemVer string | `build.zig.zon` `.version` only (injected as `build_options.version`; `display.version` re-exports it) |
| Minimum Zig | `build.zig.zon` `.minimum_zig_version` and `.zigversion` |
| Zig release the Docker image downloads | `Dockerfile` `ZIG_CHECKSUMS_FOR`, which must equal `.zigversion` (`zig build check-pins`) |
| User-facing history | `CHANGELOG.md` (Keep a Changelog-style sections; date stamps for historical entries) |

Bump `.version` in `build.zig.zon` in the release commit. Do not publish a tag
whose name disagrees with that string.

`CHANGELOG.md`, `docs/API.md`, `docs/CONTRIBUTING.md`, `README.md`,
`SECURITY.md`, and `docs/DOCUMENTATION.md` each restate the product version for
readers. `python3 scripts/check-docs.py` fails when any of them disagrees with
`build.zig.zon`; `python3 scripts/test_check_docs.py` covers that guard. A
release commit updates all of them in the same change.

### Changelog requirements

For every user-facing change, add an entry under `## [Unreleased]` before merge:

- **Breaking**: removed/renamed flag or API field, default change that alters
  results, fail-closed behavior that used to succeed, wire/format changes
- **Added**: new flags, endpoints, models, backends, quant types
- **Fixed**: correctness or crash fixes consumers would notice
- **Changed**: non-breaking behavior or docs that affect upgrade decisions

Write for operators and API clients (what breaks, what to do), not for
maintainers (avoid commit hashes as the only description).

### Release checklist

1. Move `## [Unreleased]` items into a dated/versioned section; bump
   `build.zig.zon` `.version` when cutting a release. Add the
   `[X.Y.Z]:` link definition for the new section and repoint `[unreleased]`
   at the new tag (`compare/vX.Y.Z...HEAD`).
2. Confirm `CHANGELOG.md` and `build.zig.zon` `.version` agree (`agave --version`).
3. Tag `vX.Y.Z` matching the product version (do not reuse or mutate tags).
4. Smoke: `agave --version`, one short CPU inference, and `GET /health` if serving.
5. Note minimum Zig (`.zigversion`) in release notes when raised (breaking for
   builders).
6. Run `python3 scripts/check-docs.py` (SemVer string must match across
   `build.zig.zon`, `CHANGELOG.md`, `docs/API.md`, `docs/CONTRIBUTING.md`,
   `README.md`, `SECURITY.md`, and `docs/DOCUMENTATION.md`;
   every released `[X.Y.Z]` section must have a link definition and
   `[unreleased]` must compare against the current tag;
   a HEAD tag matching `v[0-9]+.[0-9]+.[0-9]+` must equal `.version`
   (a milestone name like `v1.0` is exempt);
   every `cli_specs` flag must appear in the README CLI Options block; every
   model `-Denable-*` must appear in README, `Dockerfile`, and the Gemma3-only
   Compose override).

### Deprecation

Prefer deprecate-then-remove for CLI flags and HTTP fields: warn for at least
one release (or document a removal date), name the replacement, update
examples/docs so they stop recommending the old path, then remove.

### Support and lifecycle (0.x)

Until **1.0.0**, there is no multi-version support matrix and no promised LTS:

- **Fixes and security**: applied on current `main` / the latest product tag that
  matches `build.zig.zon` `.version`. Older tags are not maintained.
- **Backports**: none by default. Cherry-picks are case-by-case only.
- **Minimum Zig**: `.zigversion` / `build.zig.zon` `.minimum_zig_version`. Raising
  it is a **Breaking** changelog entry for anyone building from source.
- **Supported (opt-in)**: `--rate-limit-rpm` / `--rate-limit-tpm` token-bucket
  limits (default off). Treat removals or default changes as **Breaking**.
- **Experimental / incomplete**: endpoints that return `501 Not Implemented` in
  `docs/API.md` (for example `/v1/embeddings`), the unversioned
  `/v1/kv_cache` HTTP blob, and `--kv-tiers` (unified-memory backends only). Still
  changelog user-visible breaks; do not assume long-term wire stability.
