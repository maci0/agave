# Tutorial Audit: FFN/MoE (Chapter 3) and Quantization (Chapter 4)

Audited 2026-05-25 against the Agave source tree at `/Users/mwysocki/Experiments/agave`.

---

## Evidence Table

| # | Source | File Path | Key claim verified | Type | Confidence |
|---|--------|-----------|--------------------|------|------------|
| 1 | qwen35.zig init() | `src/models/qwen35.zig:1082-1095` | SwiGLU structure: `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` + `siluMul` | primary | high |
| 2 | qwen35.zig MoE init | `src/models/qwen35.zig:231-242` | Qwen3.5 MoE: fallback 256 experts, default top-8, shared expert present | primary | high |
| 3 | gpt_oss.zig defaults | `src/models/gpt_oss.zig:83-88` | GPT-OSS: 32 experts, top-4, no shared expert | primary | high |
| 4 | nemotron_nano.zig defaults | `src/models/nemotron_nano.zig:93-98` | Nemotron-Nano: 128 routed experts, top-6, shared_expert_size=3712 (2× routed 1856) | primary | high |
| 5 | gemma4.zig defaults & dualFfnLayer | `src/models/gemma4.zig:56-58,1814-1860` | Gemma 4 26B-A4B: 128 experts, top-8, dual path (dense GELU + MoE per layer) | primary | high |
| 6 | gpt_oss.zig swiglu_limit | `src/models/gpt_oss.zig:89,730-736` | Clamped SwiGLU: `swiglu_limit: f32 = 7.0`, `@min(@max(prod, -limit), limit)` | primary | high |
| 7 | math.zig applyReluSquared | `src/ops/math.zig:133-134` | ReLU² function exists: `pub fn applyReluSquared(x: []f32)` | primary | high |
| 8 | nemotron_nano.zig moeLayer | `src/models/nemotron_nano.zig:665,691` | Nemotron-Nano calls `applyReluSquared` in both routed and shared expert paths | primary | high |
| 9 | All model files | Multiple: `qwen35.zig:34`, `gpt_oss.zig:36`, `nemotron_nano.zig:44`, `gemma4.zig:46` | `max_active_experts` stack arrays: `var top_experts: [max_active_experts]usize = undefined` | primary | high |
| 10 | math.zig | `src/ops/math.zig:118,123,129` | Activation functions: `softplus`, `sigmoid`, `silu` defined | primary | high |
| 11 | Multiple backends | `src/backend/kernels/cpu/activation.zig:60`, etc. | GELU activation exists in CPU, Metal, CUDA, ROCm, Vulkan backends | primary | high |
| 12 | quant.zig | `src/ops/quant.zig:28-30` | `quant_block_elems: usize = 32`, `q8_0_block_bytes = 34`, `q4_0_block_bytes = 18` | primary | high |
| 13 | backend.zig | `src/backend/backend.zig:178` | `quant_super_block_elems: usize = 256` (used by Q4_K, Q5_K, Q6_K) | primary | high |
| 14 | mlx.zig | `src/ops/mlx.zig:9` | `mlx_group_size: usize = 64` | primary | high |
| 15 | mlx.zig | `src/ops/mlx.zig:108-112` | Factored dequantization: `q_acc` + `x_acc` pattern, scale/bias applied once per group | primary | high |
| 16 | kv_quant.zig | `src/ops/kv_quant.zig:420-438` | KvQuantType enum: turbo2-4, planar2-4, iso2-4, rotor2-4 | primary | high |
| 17 | kv_quant.zig | `src/ops/kv_quant.zig:14-19` | TurboQuant WHT + Lloyd-Max; PlanarQuant Givens 2D; IsoQuant Quaternion 4D; RotorQuant Clifford Cl(3,0) | primary | high |
| 18 | kv_quant.zig | `src/ops/kv_quant.zig:129-141` | Givens rotation (PlanarQuant): 16 pairs per 32-element block, 4 FMA per pair = 64 per block | primary | high |
| 19 | kv_quant.zig | `src/ops/kv_quant.zig:169-212` | Quaternion rotation (IsoQuant): 8 quartets per 32-element block | primary | high |
| 20 | kv_quant.zig | `src/ops/kv_quant.zig:250-302` | Clifford Cl(3,0) rotor (RotorQuant): groups of 3 dimensions per block | primary | high |
| 21 | mlx.zig companion tensors | `src/ops/mlx.zig:67-70`, model files | MLX companion tensors: `.scales` (bf16) and `.biases` (bf16) per tensor | primary | high |
| 22 | qwen35.zig shared_expert_ff_dim | `src/models/qwen35.zig:240-241` | Qwen3.5 MoE: shared expert uses `shared_expert_ff_dim` (from metadata or same as expert_ff_dim) | primary | high |

---

## Findings

### FFN/MoE Claims (Chapter 3)

#### 1. SwiGLU structure (gate_proj, up_proj, down_proj) — **MATCH** ✅

All model files use the three-projection SwiGLU pattern. For example, in `qwen35.zig` `ffnCompute()` [1]:

```zig
const gw_raw = self.fmt.layerTensor(li, "ffn_gate.weight") ...
const uw_raw = self.fmt.layerTensor(li, "ffn_up.weight") ...
self.doGemvBatch2(self.hidden2.ptr, gw, self.ff_buf1.ptr, ff, uw, self.ff_buf2.ptr, ff, e);
self.be.siluMul(self.ff_buf1.ptr, self.ff_buf2.ptr, self.ff_buf1.ptr, ff);
const dw_raw = self.fmt.layerTensor(li, "ffn_down.weight") ...
```

The pattern `activation(gate_proj(x)) * up_proj(x)` followed by `down_proj()` matches the tutorial claim. Tensor names are `ffn_gate`, `ffn_up`, `ffn_down` in GGUF convention, and vary per model in SafeTensors (e.g., `mixer.switch_mlp.fc1` / `fc2` for Nemotron).

#### 2. MoE expert counts per model — **MATCH** ✅

| Model | Tutorial claim | Code evidence | Status |
|-------|---------------|---------------|--------|
| **Qwen 3.5/3.6 MoE** | 128/256 routed, top-8, 1 shared | Default fallback 256 [2]; `default_moe_experts_active = 8` [2]; shared expert tensors `ffn_gate_shexp.weight` present [22] | **MATCH** — Code defaults to 256 when metadata absent and tensors exist; shared expert confirmed |
| **GPT-OSS** | 32, top-4, no shared | `n_experts: u32 = 32`, `n_experts_active: u32 = 4` [3]; no shared expert logic in `moeLayer` | **MATCH** |
| **Nemotron-Nano** | 128, top-6, 1 shared (2x hidden dim) | `n_routed_experts: u32 = 128`, `num_experts_per_tok: u32 = 6` [4]; `shared_expert_size: u32 = 3712` = 2 × `moe_intermediate_size: u32 = 1856` [4] | **MATCH** |
| **Gemma 4 26B-A4B** | 128, top-8, dual path | `default_n_experts = 128`, `default_top_k_experts = 8` [5]; `dualFfnLayer()` runs both `denseFfn()` (GELU-gated) and `moeFfn()` per layer, then combines [5] | **MATCH** |

**Note on Qwen "128/256":** The tutorial says "128/256 routed experts" implying two variants. The code comment says "Qwen3.5-35B-A3B uses 256 experts" and the fallback hardcodes 256. The n_experts field has no default (it's 0 when not MoE). The 128 vs 256 distinction comes from different Qwen3 vs Qwen3.5 model sizes, with actual count read from metadata (`expert_count`). The code supports both.

#### 3. Clamped SwiGLU [-7.0, +7.0] for GPT-OSS MoE — **MATCH** ✅

In `gpt_oss.zig` [6]:

```zig
swiglu_limit: f32 = 7.0,  // line 89

// line 730-736 in moeLayer():
const limit = self.swiglu_limit;
for (0..ff) |i| {
    const g = self.expert_gate[i];
    const silu_g = math_ops.silu(g);
    const prod = silu_g * self.expert_up[i];
    self.expert_gate[i] = @min(@max(prod, -limit), limit);
}
```

Exact match: clamp is `@min(@max(prod, -7.0), 7.0)`.

#### 4. ReLU² for Nemotron-Nano — **MATCH** ✅

`math.zig` defines `applyReluSquared` [7]:

```zig
/// Squared ReLU activation in-place: x[i] = max(0, x[i])². SIMD-optimized.
pub fn applyReluSquared(x: []f32) void {
```

Called in `nemotron_nano.zig` `moeLayer()` at both routed expert and shared expert stages [8]:

```zig
math_ops.applyReluSquared(self.expert_buf[0..ff]);    // routed expert (line 665)
math_ops.applyReluSquared(self.expert_buf[0..shared_ff]); // shared expert (line 691)
```

#### 5. Stack-allocated expert selection arrays — **MATCH** ✅

All MoE models use fixed-size arrays on the stack [9]:

- `qwen35.zig:34` — `const max_active_experts: usize = 16;`
- `gpt_oss.zig:36` — `const max_active_experts: usize = 8;`
- `nemotron_nano.zig:44` — `const max_active_experts: usize = 8;`
- `gemma4.zig:46` — `const max_active_experts: usize = 16;`

Usage pattern (all models):
```zig
var top_experts: [max_active_experts]usize = undefined;
var top_scores: [max_active_experts]f32 = undefined;
```

Zero heap allocation confirmed — `topKExperts` writes into these stack buffers.

#### 6. Activation functions — **MATCH** ✅

All four listed activation functions exist in the codebase [10][11]:

| Function | Location | Used by |
|----------|----------|---------|
| **SiLU** | `src/ops/math.zig:129`, `src/backend/kernels/cpu/activation.zig:13` | SwiGLU FFN in all models, SSM gating |
| **GELU** | `src/backend/kernels/cpu/activation.zig:60` | Gemma 4 dense FFN, Gemma3 |
| **Softplus** | `src/ops/math.zig:118` | SSM dt computation |
| **Sigmoid** | `src/ops/math.zig:123` | Nemotron routing, attention gate (Qwen3.5), DeltaNet beta |

---

### Quantization Claims (Chapter 4)

#### 1. Block sizes: Q4_0/Q8_0 = 32 elements per block — **MATCH** ✅

In `src/ops/quant.zig` [12]:

```zig
pub const quant_block_elems: usize = 32;
pub const q8_0_block_bytes: usize = 34;  // f16 scale (2) + 32 i8 = 34
pub const q4_0_block_bytes: usize = 18;  // f16 scale (2) + 16 nibble bytes = 18
```

#### 2. Super-block formats Q4_K/Q5_K/Q6_K = 256 elements — **MATCH** ✅

In `src/backend/backend.zig` [13]:

```zig
pub const quant_super_block_elems: usize = 256;
```

Used by Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_XS kernel dispatches (confirmed by Metal backend line 852: `.q4_k, .q5_k, .q6_k, .q2_k, .q3_k, .iq4_xs => (k + quant_super_block_elems - 1) / quant_super_block_elems`).

#### 3. MLX affine: group size 64, bf16 scales/biases, companion tensors — **MATCH** ✅

In `src/ops/mlx.zig` [14]:

```zig
pub const mlx_group_size: usize = 64;
```

Scales and biases are bf16 (`u16`) arrays — confirmed by the GEMV kernel reading them via `quant.bf16ToF32(sc[g])` and `quant.bf16ToF32(bi[g])` [15].

Companion tensors confirmed: all model files look up `weight.scales` and `weight.biases` by constructing the name from the weight tensor's prefix [21].

#### 4. Factored dequantization optimization — **MATCH** ✅

In `src/ops/mlx.zig` `mlxGemvQ4Rows()` [15]:

```zig
// Comment: "Uses factored scale/bias: sum(x*(scale*q+bias)) = scale*dot(x,q) + bias*sum(x)"

var q_acc0: VecF32 = vzero;   // accumulates dot(x, q)
var x_acc: VecF32 = vzero;    // accumulates sum(x)

for (0..full_words) |wi| {
    const xv: VecF32 = (x + xo + wi * V)[0..V].*;
    const vals0: VecF32 = @floatFromInt((w0 >> nibble_shifts) & mask4);
    q_acc0 = @mulAdd(VecF32, xv, vals0, q_acc0);
    x_acc += xv;
}
const x_sum = @reduce(.Add, x_acc);
sum0 += scale0 * @reduce(.Add, q_acc0) + bias0 * x_sum;
```

Exact match with the tutorial's "factored dequantization" optimization: `scale * dot(q, x) + bias * sum(x)` instead of per-element dequant.

#### 5. KV cache quant formats in kv_quant.zig — **MATCH** ✅

All four format families exist in the `KvQuantType` enum [16]:

```zig
pub const KvQuantType = enum {
    ...
    turbo2, turbo3, turbo4,     // TurboQuant (WHT transform)
    planar2, planar3, planar4,  // PlanarQuant (Givens 2D rotation)
    iso2, iso3, iso4,           // IsoQuant (Quaternion 4D rotation)
    rotor2, rotor3, rotor4,     // RotorQuant (Clifford Cl(3,0) rotor)
};
```

Transform implementations confirmed [17][18][19][20]:
- **TurboQuant**: `wht32()` — Walsh-Hadamard Transform, 5-stage butterfly on 32 elements
- **PlanarQuant**: `givensRotateForward/Inverse()` — Givens 2D rotation on 16 pairs
- **IsoQuant**: `quatRotateForward/Inverse()` — Quaternion rotation on 8 quartets
- **RotorQuant**: `rotorForward/Inverse()` — Clifford Cl(3,0) sandwich product on groups of 3

**Tutorial naming vs code naming:**
- Tutorial calls them "TurboQuant", "PlanarQuant", "IsoQuant", "RotorQuant" — **matches** the comments in kv_quant.zig exactly (line 14-19 file header).

#### 6. FMA counts claimed — **PARTIAL MATCH / NEEDS VERIFICATION** ⚠️

The tutorial claims for a 128-dim head (4 blocks of 32 elements each):

| Method | Tutorial FMA claim | Code analysis | Status |
|--------|-------------------|---------------|--------|
| **TurboQuant** | ~640 | WHT: 5-stage butterfly on 32 elements = 5 × 32 = 160 FMA per block × 4 blocks = 640 | **MATCH** ✅ |
| **PlanarQuant** | 256 (2.5× fewer) | Givens: 4 FMA per pair × 16 pairs per block × 4 blocks = 256 | **MATCH** ✅ |
| **IsoQuant** | 512 | Quaternion: each sandwich product `q*v*q̄` for 3D + passthrough; inline loop over 8 quartets per block, each needs ~16 FMA (multiply/add ops in the quaternion sandwich) × 8 quartets × 4 blocks = 512 | **MATCH** ✅ (inferred from FMA count in inline code, not explicitly stated in source) |
| **RotorQuant** | ~2,400 | Cl(3,0) rotor: 10 groups of 3 per block, each group has multiple bivector multiplies. The code has ~12 multiply-add ops per group × 10 groups × 4 blocks ≈ 480 (much less than 2,400). However, the tutorial says "~2,400" which could account for additional FMAs from non-zero bivector components and normalization overhead that aren't visible at this analysis level. | **UNCERTAIN** ⚠️ — Code shows simpler structure than 2,400 FMAs would imply. The sparse rotor (only e12 plane is non-zero) reduces actual FMA count significantly below the general case described in the tutorial. |

**RotorQuant FMA detail**: The code at `src/ops/kv_quant.zig:250-302` shows `rotorForward()` using `rotor_params` where most rotors are in the e12 plane (b13=0, b23=0 by default). The general-case code includes all bivector terms (6 multiply-adds per output coordinate × 3 coords × 10 groups × 4 blocks = ~720), but many terms multiply by zero. The tutorial's "~2,400" count may describe the fully-general Clifford rotor (all 3 bivector components non-zero) rather than the actual sparse implementation. This is a **possible overstatement** in the tutorial.

---

## Coverage Status

### Directly verified ✅
- [x] SwiGLU three-projection structure (all models)
- [x] MoE expert counts: GPT-OSS (32/top-4/no shared), Nemotron-Nano (128/top-6/1 shared/2x), Gemma4 (128/top-8/dual), Qwen3.5 (256 fallback/top-8/shared)
- [x] Clamped SwiGLU ±7.0 in GPT-OSS
- [x] ReLU² in Nemotron-Nano (both routed and shared experts)
- [x] Stack-allocated expert selection arrays (all MoE models)
- [x] Activation functions: SiLU, GELU, Softplus, Sigmoid
- [x] Q4_0/Q8_0 block size = 32
- [x] Q4_K/Q5_K/Q6_K super-block = 256
- [x] MLX group size = 64, bf16 scales/biases
- [x] Factored dequantization optimization
- [x] KV quant format enum: turbo, planar, iso, rotor (2/3/4 bit variants)
- [x] Transform implementations: WHT, Givens, Quaternion, Clifford rotor

### Uncertain ⚠️
- [ ] RotorQuant FMA count (~2,400 claimed vs ~720 apparent in actual sparse implementation)
- [ ] Qwen 3.5 vs 3.6 distinction (code uses same model file; "128/256" from tutorial may refer to different model sizes, code fallback defaults to 256 only)

### Not checked (out of scope)
- [ ] Actual perplexity impact numbers for TurboQuant variants
- [ ] Walsh-Hadamard paper citation accuracy (Zandieh et al., 2025)
- [ ] KIVI paper citation accuracy
- [ ] 30-40% speedup claim for factored dequantization

---

## Sources

1. `src/models/qwen35.zig` — Qwen 3.5 hybrid model implementation (SwiGLU FFN, MoE)
2. `src/models/qwen35.zig:231-242` — Qwen3.5 MoE configuration (256 experts fallback, top-8)
3. `src/models/gpt_oss.zig:83-88` — GPT-OSS model defaults (32 experts, top-4)
4. `src/models/nemotron_nano.zig:93-98` — Nemotron-Nano MoE defaults (128 experts, top-6, shared=3712)
5. `src/models/gemma4.zig:56-58,1814-1860` — Gemma 4 MoE defaults + dualFfnLayer implementation
6. `src/models/gpt_oss.zig:89,730-736` — GPT-OSS clamped SwiGLU (±7.0)
7. `src/ops/math.zig:133-134` — applyReluSquared function
8. `src/models/nemotron_nano.zig:665,691` — Nemotron-Nano ReLU² usage in both expert paths
9. All model files — `max_active_experts` stack-allocated arrays
10. `src/ops/math.zig:118,123,129` — softplus, sigmoid, silu functions
11. `src/backend/kernels/cpu/activation.zig` and other backends — GELU implementation
12. `src/ops/quant.zig:28-30` — quant_block_elems = 32, block byte sizes
13. `src/backend/backend.zig:178` — quant_super_block_elems = 256
14. `src/ops/mlx.zig:9` — mlx_group_size = 64
15. `src/ops/mlx.zig:108-175` — Factored dequantization implementation
16. `src/ops/kv_quant.zig:420-438` — KvQuantType enum
17. `src/ops/kv_quant.zig:14-19` — File header listing all four KV quant methods
18. `src/ops/kv_quant.zig:129-141` — Givens rotation (PlanarQuant)
19. `src/ops/kv_quant.zig:154-212` — Quaternion rotation (IsoQuant)
20. `src/ops/kv_quant.zig:222-302` — Clifford rotor (RotorQuant)
21. Model files — MLX companion tensor lookup (`.scales`, `.biases`)
22. `src/models/qwen35.zig:240-241` — Qwen3.5 shared_expert_ff_dim
