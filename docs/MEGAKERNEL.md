# Megakernel Implementation

Fused GPU kernels that eliminate per-layer dispatch overhead. Three-tier system:
1. **Fused FFN** (active): 3→1 dispatch per FFN layer. Up to +93% short decode.
2. **True Megakernel** (hand-written): single dispatch for ALL layers. 5 Metal + 3 CUDA + 1 ROCm.
3. **Composed Megakernel** (auto-generated): `mega_compose.zig` generates MSL from model metadata at runtime.

Enable with `--megakernel` flag.

Inspired by [Luce Megakernel](https://github.com/Luce-Org/luce-megakernel) which achieved 3.4× prefill and 1.55× decode speedup on Qwen 3.5-0.8B.

## CLI

```bash
agave model.gguf --megakernel "prompt"     # Fused FFN (active)
agave model.gguf "prompt"                  # Standard dispatch (default)
```

Supported: Qwen 3.5, Gemma 3/4, GLM-4 on Metal. Qwen 3.5 on CUDA.

## Tier 1: Fused FFN Kernels (Active)

Fuses gate GEMV + up GEMV + activation into a single dispatch per FFN layer. Saves 2 dispatches per layer (48-84 saved per token depending on model).

### Metal — 12 kernels in `megakernel.metal`

| Kernel | Activation | Quant | Models |
|--------|-----------|-------|--------|
| `fused_ffn_gate_up_silu_q8` | SiLU | Q8_0 | Qwen 3.5 |
| `fused_ffn_gate_up_silu_q4_k` | SiLU | Q4_K | Qwen 3.5 |
| `fused_ffn_gate_up_silu_q5_k` | SiLU | Q5_K | Qwen 3.5 |
| `fused_ffn_gate_up_silu_q6_k` | SiLU | Q6_K | Qwen 3.5 |
| `fused_ffn_gate_up_silu_q4_0` | SiLU | Q4_0 | Qwen 3.5 |
| `fused_ffn_gate_up_silu_mlx_q4` | SiLU | MLX-Q4 | GLM-4 |
| `fused_ffn_gate_up_gelu_q8` | GELU | Q8_0 | Gemma 3/4 |
| `fused_ffn_gate_up_gelu_q4_k` | GELU | Q4_K | Gemma 3/4 |
| `fused_ffn_gate_up_gelu_q5_k` | GELU | Q5_K | Gemma 3/4 |
| `fused_ffn_gate_up_gelu_q6_k` | GELU | Q6_K | Gemma 3/4 |
| `fused_ffn_gate_up_gelu_q4_0` | GELU | Q4_0 | Gemma 3/4 |

### CUDA — 1 kernel in `fused_ffn_q8_0.zig`

Q8_0 SiLU variant. PTX compiled and appended to `all.ptx`.

### Performance (M4 Pro 48GB)

| Model | Quant | Standard | Megakernel | Delta |
|-------|-------|----------|------------|-------|
| Qwen 3.5 0.8B | Q8_0 | 111.7 tok/s | 116.3 tok/s | **+4%** |
| Qwen 3.5 0.8B | Q8_0 (profiled) | 23.8 tok/s | 25.5 tok/s | **+7%** |
| Gemma 4 E2B | Q4_K_M (short) | 9.9 tok/s | 19.1 tok/s | **+93%** |
| Gemma 4 E2B | Q4_K_M (100 tok) | 12.4 tok/s | 12.7 tok/s | +2% |
| Gemma 4 E2B | Q4_K_M (prefill) | 2206 ms | 1702 ms | **−23%** |

The +93% short-decode result comes from Q4_K_M mixed quant — with Q5_K/Q6_K fused kernels, ALL FFN layers use the fused path. Largest effect at short sequences where dispatch overhead dominates.

## Tier 2: True Megakernels (Architecture Built)

Single GPU dispatch for ALL layers. Uses composable building blocks with atomic grid sync between stages.

### Composable Building Blocks (`mega_common.metal`, 732 lines)

| Block | Purpose |
|-------|---------|
| `mega_grid_sync` | Atomic counter barrier (Metal `memory_order_relaxed`) |
| `mega_rms_norm` / `mega_add_rms_norm` | Multi-TG cooperative norm with atomic sum-of-squares |
| `mega_gemv_q8` / `mega_gemv_q4k` / `mega_gemv_q4_0` / `mega_gemv_q5k` / `mega_gemv_q6k` | Per-TG GEMV stages (all K-quant formats) |
| `mega_silu_mul` / `mega_gelu_mul` / `mega_relu_squared` / `mega_silu_mul_clamp` | All activation types |
| `mega_rope` | Rotary position encoding |
| `mega_add` | Residual addition |
| `mega_kv_append_f32` / `mega_kv_append_tq` | KV cache append (f32 + TurboQuant WHT encode) |
| `mega_sdpa_inline` | Full inline SDPA with TQ+ dequant, sparse V (1e-6), GQA, online softmax |

### True Megakernel Files

| Backend | File | Model | Quant |
|---------|------|-------|-------|
| Metal | `mega_qwen35_q8.metal` | Qwen 3.5 | Q8_0 |
| Metal | `mega_qwen35_q4k.metal` | Qwen 3.5 | Q4_K |
| Metal | `mega_gemma_q4k.metal` | Gemma 3/4 | Q4_K |
| Metal | `mega_gemma_q8.metal` | Gemma 3/4 | Q8_0 |
| Metal | `mega_nemotron_h_q8.metal` | Nemotron-H | Q8_0 |
| CUDA | `mega_qwen35_q8.zig` | Qwen 3.5 | Q8_0 |
| CUDA | `mega_gemma_q4k.zig` | Gemma 3/4 | Q4_K |
| CUDA | `mega_gemma_q8.zig` | Gemma 3/4 | Q8_0 |
| ROCm | `mega_qwen35_q8.zig` | Qwen 3.5 | Q8_0 |

### TurboQuant+ in Megakernels

The `mega_sdpa_inline` block supports:
- **Asymmetric K/V types**: different bits_k and bits_v per call
- **Sparse V threshold** (1e-6): skip positions with negligible softmax weight
- **Boundary V**: caller passes per-layer bits_v (f16 for first/last N layers)
- **TurboQuant 2/3/4-bit**: WHT inverse + Lloyd-Max codebook dequantization
- **Online softmax**: FlashAttention-2 incremental max/sum tracking

### Integration Status

The true megakernels compile and dispatch correctly. SDPA inline is wired into the layer loops. `--megakernel` uses single-dispatch for the full forward pass (all layers in 1 GPU dispatch).

Known limitations of hand-written megakernels:
- Qwen 3.5: Q+gate deinterleave, per-head QK norms, sigmoid gate not yet in megakernel
- Gemma 3/4: Sliding-window vs global attention layers have different head dims
- Paged KV cache incompatible with flat KV arrays in megakernel (needs flat allocation mode)
- DeltaNet SSM: sequential recurrence dependencies don't parallelize across TGs

## Tier 3: Composed Megakernels (Auto-Generated)

Tier 3 eliminates the need to hand-write per-model megakernel files. The `mega_compose.zig` module generates model-specific MSL source code at runtime from model metadata, then the Metal backend compiles it via `newLibraryWithSource`.

### Architecture

```
Model Metadata (GGUF/SafeTensors)
        |
        v
    ModelDesc struct         ← populated from format metadata at model init
        |
        v
    composeMSL(buf, desc)    ← generates MSL source string in a stack buffer
        |
        v
    MSL source               ← references building blocks from mega_common.metal
        |
        v
    compileComposedMegakernel()  ← Metal runtime compile (newLibraryWithSource)
        |
        v
    dispatchMegakernelAuto()     ← single GPU dispatch for all layers
```

### Key Files

| File | Role |
|------|------|
| `src/backend/mega_compose.zig` | `ModelDesc` struct, `composeMSL()` function, helper constructors |
| `src/backend/kernels/metal/mega_common.metal` | 18 composable building blocks (732 lines) |
| `src/backend/metal.zig` | `compileComposedMegakernel()`, `dispatchMegakernelAuto()` |

### What the Composer Handles Automatically

The composer reads the `ModelDesc` and selects the correct building block for each stage:

- **Quant dispatch**: Q8_0, Q4_K, Q5_K, Q6_K, Q4_0 -- selects the correct `mega_gemv_*` function
- **Activation**: SiLU, GELU, ReLU-squared -- selects correct activation call (`mega_silu_mul`, `mega_gelu_mul`, `mega_relu_squared`)
- **Layer types**: attention, DeltaNet, MoE, FFN-only -- attention layers get SDPA, others skip
- **Residual pattern**: fused (Qwen `addRmsNorm`) or separate (Gemma `add` + `norm`)
- **Post-attention norm**: optional fused `mega_add_rms_norm`
- **Inline SDPA**: KV cache append via `mega_kv_append_f32` or `mega_kv_append_tq`, online softmax, sparse V, GQA
- **TurboQuant+**: via `mega_kv_append_tq` and `mega_sdpa_inline` building blocks

### Adding a New Model

Adding megakernel support for a new model requires only a `ModelDesc` -- no MSL or shader code:

```zig
const mega_compose = @import("backend/mega_compose.zig");
const ModelDesc = mega_compose.ModelDesc;

const desc = ModelDesc{
    .name = "new_model",
    .n_layers = 32,
    .n_embd = 4096,
    .n_ff = 11008,
    .n_head = 32,
    .n_kv = 8,
    .head_dim = 128,
    .rope_dim = 128,
    .rope_theta = 10000.0,
    .rms_eps = 1e-6,
    .max_seq_len = 4096,
    .activation = .silu,
    .quant = .q4_k,
    .layer_types = ModelDesc.uniform(32, .attention),
    // Per-layer overrides (0 = use default). For models with varying dims:
    // .layer_n_head = ...,      // per-layer Q head count
    // .layer_n_kv = ...,        // per-layer KV head count
    // .layer_head_dim = ...,    // per-layer head dimension
    // .layer_n_ff = ...,        // per-layer FFN dimension
    // .layer_rope_theta = ...,  // per-layer RoPE theta
    // .layer_sliding_window = ..., // per-layer sliding window size
};

// Generate MSL and compile at init
var buf: [32768]u8 = undefined;
const msl = mega_compose.composeMSL(&buf, desc);
try metal_be.compileComposedMegakernel(msl);

// Dispatch in forward pass
metal_be.dispatchMegakernelAuto(params);
```

Helper constructors simplify common patterns:

| Constructor | Pattern |
|-------------|---------|
| `ModelDesc.uniform(n, .attention)` | All layers are attention (Gemma 3, Gemma 4 dense) |
| `ModelDesc.qwenHybrid(n, interval)` | DeltaNet except every Nth layer is attention (Qwen 3.5) |

### ModelDesc Flags

| Flag | Default | Effect |
|------|---------|--------|
| `has_gate` | false | Q projection includes interleaved gate (Qwen 3.5) |
| `has_qk_norm` | false | Per-head Q/K RMS norms after projection |
| `has_post_attn_norm` | false | Post-attention fused addRmsNorm |
| `fuse_residual` | false | Deferred residual pattern (Qwen addRmsNorm) |

### Composed vs Hand-Written

| Aspect | Hand-Written (Tier 2) | Composed (Tier 3) |
|--------|----------------------|-------------------|
| Adding a model | Write ~200-400 lines MSL/PTX | Define a `ModelDesc` struct |
| Quant variants | Separate file per quant | Automatic from `desc.quant` |
| Optimization level | Maximum (hand-tuned) | Good (building block composition) |
| Maintenance | Per-file updates | Update building blocks once |
| Backend support | Metal + CUDA + ROCm | Metal only (runtime MSL compile) |

## Model Coverage

| Model | Fused FFN | True Megakernel | Composed Megakernel | Notes |
|-------|:---------:|:---------------:|:-------------------:|-------|
| Qwen 3.5 | SiLU (all quants) | Q8_0 + Q4_K | Yes (SiLU, fused residual) | DeltaNet breaks out to standard dispatch |
| Gemma 3 | GELU (all quants) | Q4_K + Q8_0 | Yes (GELU, uniform) | Cleanest megakernel |
| Gemma 4 (dense) | GELU (all quants) | Q4_K + Q8_0 | Yes (GELU, uniform) | SL/GL attention complicates true megakernel |
| Gemma 4 (MoE) | GELU (per-expert) | — | — | MoE routing is CPU-side |
| GLM-4 | SiLU (MLX-Q4) | — | — | MLA attention is CPU-side |
| Nemotron-H | — | Q8_0 (attn+FFN) | Yes (ReLU-squared) | SSM breaks out, FFN uses ReLU² (no gate) |
| Nemotron-Nano | — | — | — | MoE + SSM, all CPU-side |
| GPT-OSS | — | — | — | Bias+clamp between GEMV and activation |

## Pipeline/Kernel Counts

| Backend | Pipelines/Kernels | Megakernel Files | Composed |
|---------|:-----------------:|:----------------:|:--------:|
| Metal | 70+ | 7 (5 true + megakernel.metal + mega_common.metal) | Yes (runtime MSL) |
| CUDA | 41 | 4 (3 true + fused FFN) | No |
| ROCm | 28+ | 1 (true Qwen Q8_0) | No |

The composed megakernel (`mega_compose.zig`) generates an additional Metal pipeline at runtime via `compileComposedMegakernel()`. This pipeline is not counted in the static file totals above.

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/backend/mega_compose.zig` | ~780 | Composable megakernel generator (ModelDesc, composeMSL) |
| `src/backend/megakernel.zig` | — | Weight offset computation for fused FFN megakernels |
| `src/backend/kernels/metal/mega_common.metal` | 732 | 18 composable building blocks |
| `src/backend/kernels/metal/megakernel.metal` | — | 12 fused FFN kernels (Tier 1) |
| `src/backend/kernels/metal/mega_*.metal` | — | 5 hand-written true megakernels (Tier 2) |
| `src/backend/metal.zig` | — | `compileComposedMegakernel()`, `dispatchMegakernelAuto()` |

## References

- [Luce Megakernel](https://github.com/Luce-Org/luce-megakernel) — Qwen 3.5-0.8B CUDA megakernel, 3.4× prefill speedup
- [Cooperative Groups (CUDA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) — Grid-level synchronization
- [Metal Atomic Operations](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — `atomic_fetch_add_explicit` with `memory_order_relaxed`
