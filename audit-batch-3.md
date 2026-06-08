# Audit Batch 3: MEGAKERNEL.md & CONTRIBUTING.md

Cross-reference of every factual claim against source code.

---

## MEGAKERNEL.md Issues

### [HIGH] MEGAKERNEL.md: Pipeline/Kernel Counts table — Metal pipeline count
  Doc claims: "Metal | 70+"
  Source says: `src/backend/metal.zig:463` defines `pub const n_pipelines: u32 = 83;`
  Note: The test at `metal.zig:3046` expects 70 and is itself stale vs the code constant.
  Fix: Change "70+" to "83" (or "80+" if approximate is acceptable).

### [HIGH] MEGAKERNEL.md: Pipeline/Kernel Counts table — CUDA kernel count
  Doc claims: "CUDA | 56"
  Source says: `src/backend/cuda.zig:285` defines `pub const n_kernels: u32 = 43;`
  Fix: Change "56" to "43".

### [HIGH] MEGAKERNEL.md: Pipeline/Kernel Counts table — ROCm kernel count
  Doc claims: "ROCm | 44"
  Source says: `src/backend/rocm.zig:198` defines `pub const n_kernels: u32 = 28;`
  Fix: Change "44" to "28".

### [HIGH] MEGAKERNEL.md: Pipeline/Kernel Counts table — CUDA megakernel file count
  Doc claims: "CUDA | 4 (3 true + fused FFN)"
  Source says: There are 3 true megakernel files (`mega_qwen35_q8.zig`, `mega_gemma_q4k.zig`, `mega_gemma_q8.zig`) PLUS 4 fused FFN files (`fused_ffn_q4_k.zig`, `fused_ffn_q5_k.zig`, `fused_ffn_q6_k.zig`, `fused_ffn_q8_0.zig`) = 7 total megakernel-related files.
  Fix: Change to "7 (3 true + 4 fused FFN)" or clarify counting methodology.

### [MEDIUM] MEGAKERNEL.md: Key Source Files — mega_compose.zig line count
  Doc claims: "`src/backend/mega_compose.zig` | ~780 | Composable megakernel generator"
  Source says: `wc -l src/backend/mega_compose.zig` = 1036 lines
  Fix: Change "~780" to "~1036".

### [MEDIUM] MEGAKERNEL.md: Composable Building Blocks table — missing `mega_sync_reset`
  Doc claims: Table lists building blocks starting with `mega_grid_sync` for "Atomic counter barrier (Metal `memory_order_relaxed`)"
  Source says: `mega_common.metal` has 18 `inline void mega_*` functions. The doc table only accounts for 17. The function `mega_sync_reset` (line 47, resets the atomic counter after grid sync) is not listed in the table.
  Fix: Add `mega_sync_reset` row: "Reset atomic counter after sync barrier"

### [MEDIUM] MEGAKERNEL.md: Luce Megakernel reference — prefill speedup claim
  Doc claims: "Inspired by Luce Megakernel which achieved 3.4× prefill and 1.55× decode speedup on Qwen 3.5-0.8B."
  Source says: Luce RESULTS.md shows RTX 3090 pp520: 21,347 tok/s megakernel vs 11,247 tok/s llama.cpp = **1.9× prefill** (vs llama.cpp) or 2.82× vs PyTorch. The 1.55× decode is correct. The DGX Spark shows 3.6× vs PyTorch, but that's on a different platform with NVFP4, not the Qwen 3.5-0.8B claim.
  Fix: Change "3.4× prefill" to "1.9× prefill" (vs llama.cpp) or "2.8× prefill (vs PyTorch)" with clarification.

### [LOW] MEGAKERNEL.md: ModelDesc Flags table — incomplete flags
  Doc claims: Table lists 4 flags: `has_gate`, `has_qk_norm`, `has_post_attn_norm`, `fuse_residual`
  Source says: `mega_compose.zig` ModelDesc also has `embd_scale: bool = false` ("Gemma-style embedding scaling") and `logit_softcap: f32 = 0` ("Logit softcap value, Gemma uses 30.0").
  Fix: Add `embd_scale` and `logit_softcap` to the flags table.

### [LOW] MEGAKERNEL.md: Code sample — dispatchMegakernelAuto simplified signature
  Doc claims: Comment says `dispatchMegakernelAuto(weights, weights_size, layer_offsets, ...)`
  Source says: `metal.zig:1580` shows the full signature has 19 parameters including separate `_size` params for each buffer pointer.
  Fix: This is a deliberate simplification with "..." — acceptable but could note it's simplified.

### [INFO] MEGAKERNEL.md: Helper constructors table
  Doc claims: Two constructors: `ModelDesc.uniform(n, .attention)` and `ModelDesc.qwenHybrid(n, interval)`
  Source says: Both exist exactly as documented in `mega_compose.zig` lines (uniform at ~line 154, qwenHybrid at ~line 165). ✅ Verified correct.

### [INFO] MEGAKERNEL.md: Kernel count in megakernel.metal
  Doc claims: "11 kernels in `megakernel.metal`"
  Source says: `grep -c '^kernel void' src/backend/kernels/metal/megakernel.metal` = 11. ✅ Verified correct.

### [INFO] MEGAKERNEL.md: Kernel names table
  Doc lists 11 kernel names. Source shows exactly:
  `fused_ffn_gate_up_silu_q8`, `fused_ffn_gate_up_gelu_q8`, `fused_ffn_gate_up_silu_q4_k`, `fused_ffn_gate_up_silu_q4_0`, `fused_ffn_gate_up_gelu_q4_k`, `fused_ffn_gate_up_gelu_q4_0`, `fused_ffn_gate_up_silu_q6_k`, `fused_ffn_gate_up_gelu_q6_k`, `fused_ffn_gate_up_silu_q5_k`, `fused_ffn_gate_up_gelu_q5_k`, `fused_ffn_gate_up_silu_mlx_q4`.
  ✅ All 11 match.

### [INFO] MEGAKERNEL.md: Building block count
  Doc claims: "18 composable building blocks (732 lines)"
  Source says: 18 `inline void mega_*` functions in 732 lines. ✅ Verified correct.

### [INFO] MEGAKERNEL.md: True Megakernel Files table — Metal files
  Doc lists 5 Metal files: `mega_qwen35_q8.metal`, `mega_qwen35_q4k.metal`, `mega_gemma_q4k.metal`, `mega_gemma_q8.metal`, `mega_nemotron_h_q8.metal`.
  Source shows exactly these 5 files (plus `mega_common.metal`). ✅ Verified correct.

### [INFO] MEGAKERNEL.md: True Megakernel Files table — CUDA files
  Doc lists 3 CUDA files: `mega_qwen35_q8.zig`, `mega_gemma_q4k.zig`, `mega_gemma_q8.zig`.
  Source shows exactly these 3 files. ✅ Verified correct.

### [INFO] MEGAKERNEL.md: True Megakernel Files table — ROCm files
  Doc lists 1 ROCm file: `mega_qwen35_q8.zig`.
  Source shows exactly this 1 file. ✅ Verified correct.

### [INFO] MEGAKERNEL.md: sparse_v_threshold value
  Doc claims: "Sparse V threshold (1e-6)"
  Source says: `sdpa.metal:23` defines `constant float sparse_v_threshold = 1e-6f;` ✅ Verified correct.

### [INFO] MEGAKERNEL.md: Weight offset types in megakernel.zig
  Doc claims: "Weight offset computation for fused FFN megakernels"
  Source says: `megakernel.zig` defines `LayerOffsets` struct with 20 fields (attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output, attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_a, ssm_dt_bias, ssm_conv1d, ssm_norm, ssm_out, post_attn_norm, ffn_gate, ffn_up, ffn_down) and `WeightPack` struct. ✅ Verified correct.

---

## CONTRIBUTING.md Issues

### [HIGH] CONTRIBUTING.md: Backend list
  Doc claims: "Existing backends: CPU (`cpu.zig`), Metal (`metal.zig`), Vulkan (`vulkan.zig`), CUDA (`cuda.zig`), ROCm (`rocm.zig`), WebGPU (`webgpu.zig`)."
  Source says: `backend.zig:492-498` Backend union has: `cpu`, `metal`, `vulkan`, `cuda`, `rocm`, `webgpu`. ✅ Verified correct — all 6 match.

### [MEDIUM] CONTRIBUTING.md: Quant scheme step 6 reference
  Doc claims: "For compressed-tensors formats (NVFP4, etc.): add fusion logic in `safetensors.zig` `fuseNvfp4Experts()`"
  Source says: `safetensors.zig:779` has `fn fuseNvfp4Experts(...)`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: Arch enum methods
  Doc claims: "Add variant to `Arch` enum in `src/arch.zig` with `detect`, `displayName`, `chatTemplate`, `isEnabled`, `buildFlag` methods"
  Source says: `arch.zig` has all five methods: `detect` (line 19), `displayName` (line 57), `chatTemplate` (line 71), `isEnabled` (line 97), `buildFlag` (line 141). ✅ Verified correct.

### [INFO] CONTRIBUTING.md: Model registration pattern
  Doc claims: "Add to `src/models/model.zig` (conditional import gated by `build_options.enable_yourmodel`)"
  Source says: `model.zig` uses pattern: `const XModel = if (build_options.enable_x) @import("x.zig").XModel else void;` for all 8 models. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: Build flags
  Doc claims: "Add `enable-yourmodel` build flag in `build.zig`"
  Source says: `build.zig` lines 17-24 have `enable-gemma3`, `enable-qwen35`, `enable-gpt-oss`, `enable-nemotron-h`, `enable-nemotron-nano`, `enable-glm4`, `enable-gemma4`, `enable-llama4`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: Backend tagged union pattern
  Doc shows: `pub const Backend = union(enum) { cpu: *CpuBackend, metal: *MetalBackend, ... }` with `inline else` dispatch.
  Source says: `backend.zig:492` shows exact pattern with `switch (self) { inline else => |be| be.gemv(x, w, y, n, k) }`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: DType enum location
  Doc claims: "Add variant to `DType` enum in `src/format/format.zig`"
  Source says: `format.zig:8` defines `pub const DType = enum { ... }`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: weightBytes location
  Doc claims: "wire up byte-size calculation in `src/backend/backend.zig` (`weightBytes()`)"
  Source says: `backend.zig:222` defines `pub fn weightBytes(dtype: DType, n: usize, k: usize) usize`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: KvQuantType location
  Doc claims: "`KvQuantType` enum in `src/ops/kv_quant.zig`"
  Source says: `kv_quant.zig:420` defines `pub const KvQuantType = enum { ... }`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: VisionVariant location
  Doc claims: "Add a variant to `VisionVariant` enum in `src/models/vision.zig`"
  Source says: `vision.zig:91` defines `const VisionVariant = enum { ... }`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: Transport methods
  Doc claims: Transport must implement `allReduceAdd(buf, n)`, `sendBuf(buf, n)`, `recvBuf(buf, n)`
  Source says: `transport.zig` has `allReduceAdd` (line 375), `sendBuf` (line 461), `recvBuf` (line 535). ✅ Verified correct.

### [INFO] CONTRIBUTING.md: gemm.metal exists
  Doc claims: "Metal in `gemm.metal` (reuse `block_dot` from GEMV)"
  Source says: `src/backend/kernels/metal/gemm.metal` exists and uses `q8_0_block_dot`, `q4_k_block_dot`, `q6_k_block_dot`. ✅ Verified correct.

### [INFO] CONTRIBUTING.md: fp8e4m3ToF32 helper
  Doc claims: "add conversion helpers in `src/ops/quant.zig` if the format needs custom type conversions (e.g., `fp8e4m3ToF32`)"
  Source says: `quant.zig` has `pub inline fn fp8e4m3ToF32(val: u8) f32`. ✅ Verified correct.

---

## Summary

### Severity Counts

| Severity | Count | Files |
|----------|-------|-------|
| HIGH | 4 | MEGAKERNEL.md (4) |
| MEDIUM | 3 | MEGAKERNEL.md (3) |
| LOW | 2 | MEGAKERNEL.md (2) |
| INFO (verified correct) | 16 | Both files |

### Critical Fixes Needed

1. **Metal pipeline count**: 70+ → 83 (`metal.zig:463`)
2. **CUDA kernel count**: 56 → 43 (`cuda.zig:285`)
3. **ROCm kernel count**: 44 → 28 (`rocm.zig:198`)
4. **CUDA megakernel file count**: 4 → 7 (3 true + 4 fused FFN files)
5. **mega_compose.zig line count**: ~780 → ~1036
6. **Missing building block `mega_sync_reset`** in table
7. **Luce prefill speedup**: 3.4× → 1.9× (vs llama.cpp) or 2.8× (vs PyTorch)

### Also Note

- `metal.zig` test at line 3046 expects `n_pipelines = 70` but the actual constant is 83. This is a separate test staleness issue, not a doc issue.
- CONTRIBUTING.md is remarkably accurate — every source file path, method name, and pattern was verified correct against the codebase.

---

## Coverage Status

- **Checked directly**: All source files specified in the task (mega_compose.zig, megakernel.zig, megakernel.metal, mega_common.metal, metal.zig, build.zig, backend.zig, model.zig, format.zig, arch.zig, transport.zig, vision.zig, quant.zig, safetensors.zig, gemm.metal)
- **Checked externally**: Luce Megakernel GitHub repo (RESULTS.md, README.md) for speedup claims
- **Not checked**: CUDA fused FFN kernel contents (verified file existence only), Vulkan/WebGPU backend files (not referenced in docs under audit)
- **Uncertain**: Whether the metal.zig test staleness (expecting 70 vs actual 83) is intentional or a bug — likely a bug since the constant was updated but the test wasn't
