# Contributing to Agave

Templates and step-by-step guides for extending the inference engine.

## How to Add a New Backend

Existing backends: CPU (`cpu.zig`), Metal (`metal.zig`), Vulkan (`vulkan.zig`), CUDA (`cuda.zig`), ROCm (`rocm.zig`), WebGPU (`webgpu.zig`).

1. Create `src/backend/yourbackend.zig`
2. Implement the full backend interface — core ops (`gemv`, `rmsNorm`, `rope`, `sdpa`, `sync`, etc.) plus fused variants (`siluMul`, `addRmsNorm`, `sdpaPrefill`, `gemvMulti`, ...). See `src/backend/backend.zig` for the complete dispatch interface; every function must be implemented or `@panic` on unsupported ops
3. Add variant to the `Backend` tagged union in `src/backend/backend.zig`
4. Add backend-specific tests in your implementation file
5. Update `build.zig` with target-specific compilation flags
6. Add entry to `docs/KERNELS.md`
7. Add GPU kernels in `src/backend/kernels/yourbackend/` — shader format depends on backend (MSL for Metal, SPIR-V for Vulkan, PTX for CUDA, HSACO for ROCm, WGSL for WebGPU)

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
7. Update `src/format/format.zig` for weight loading
8. Add golden test against reference implementation

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

    // Optional fields for multimodal (vtable detects via @hasField):
    // image_embeddings: ?[]const f32 = null,
    // n_visual_tokens: u32 = 0,
    // image_pad_token_id: u32 = 0,
    // visual_token_idx: u32 = 0,

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
- Custom: populate `layer_types` array directly for mixed architectures (Nemotron-H)

**Optional flags:** `has_gate`, `has_qk_norm`, `has_post_attn_norm`, `fuse_residual` -- set these for model-specific structural variations.

The composer selects the correct GEMV, activation, residual pattern, and SDPA building blocks automatically. See [MEGAKERNEL.md](MEGAKERNEL.md) for the full three-tier architecture.

## How to Add a New Quantization Scheme

1. Add variant to `DType` enum in `src/format/format.zig` and wire up byte-size calculation in `src/backend/backend.zig` (`weightBytes()`)
2. Implement GEMV kernel: CPU SIMD in `src/backend/kernels/cpu/` and native GPU versions per backend (no CPU fallback in GPU backends). Dequantization happens in-kernel — never pre-dequant to f32
3. Add conversion helpers in `src/ops/quant.zig` if the format needs custom type conversions (e.g., `fp8e4m3ToF32`)
4. Update backend dispatch to include new format (add GEMV variant in `backend.zig`)
5. Add GEMM kernel for batched prefill: Metal in `gemm.metal` (reuse `block_dot` from GEMV), pipeline in `metal.zig`, dispatch in `gemm()`. Pattern: one threadgroup per output row, loop over n_tok tokens
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
5. Add to `printUsage()` help text

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
// Optional vision fields — vtable detects these via @hasField
image_embeddings: ?[]const f32 = null,
n_visual_tokens: u32 = 0,
image_pad_token_id: u32 = 0,
visual_token_idx: u32 = 0,
```

## How to Add Speculative Decoding Support to a New Model

Layer skip for self-speculative mode is automatic — the `layer_skip_start`/`layer_skip_end` fields and the skip check in `forward()` are required in every model. The pattern is:

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

The `Model` VTable provides `setLayerSkip(start, end)` via `@hasField` detection — no manual wiring needed.

For tree attention support (`forwardTree`), implement a batch forward that:
1. Processes B queries through all layers with position IDs (not sequential)
2. Uses `be.sdpaTree()` instead of `be.sdpa()` for ancestor-masked attention
3. Writes B logit vectors to a tree logits buffer

See `src/spec/ddtree.zig` for the tree construction algorithm and `src/backend/kernels/cpu/sdpa_tree.zig` for the tree SDPA kernel.

## How to Debug Performance Regressions

1. **Profile per-op timing**: `./zig-out/bin/agave model.gguf --profile "prompt"` (adds GPU syncs, ~50% throughput loss)
2. **Micro-benchmarks**: `zig build bench && ./zig-out/bin/agave-bench gemv_f32 --n=4096 --k=4096 --backend=metal`
3. **Research kernels**: `cd research/kernels && uv run run.py bench sdpa --backend cpu`
4. **Check allocations**: Use `std.testing.allocator` in tests (detects leaks automatically)
5. **Verify comptime dispatch**: Ensure `inline else` dispatch is still used in `backend.zig`

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
// src/backend/backend.zig — tagged union with inline else dispatch
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
// main.zig — GOOD: use dispatcher
be.gemv(x, weight, output, n, k);

// BAD: never import implementations directly!
// const cuda = @import("backend/cuda.zig"); // WRONG!
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
