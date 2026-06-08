# Audit Batch 2: MODELS.md and KERNELS.md Cross-Reference

**Auditor:** Technical documentation subagent  
**Date:** 2026-06-03  
**Scope:** Every factual claim in `docs/MODELS.md` and `docs/KERNELS.md` cross-referenced against actual source code.

---

## Summary

- **MODELS.md:** 0 critical errors, 0 moderate errors. All verifiable model parameter defaults match source code exactly. Model variant-specific parameters (Gemma3 4B/12B/27B, Qwen3.5 0.8B, Qwen3.6 35B-A3B, Gemma4 E2B/E4B, Llama4 Scout) are loaded from GGUF metadata, not from source fallback defaults — these cannot be verified from source alone but the documented fallback defaults are all correct.
- **KERNELS.md:** 4 issues found (1 moderate, 3 low severity).

---

## MODELS.md Audit

### Model Parameter Table Verification

Each row checked against struct field defaults in the corresponding source file.

#### Gemma3 (src/models/gemma3.zig)

The doc's "Gemma3 1B" row matches the source fallback defaults exactly:

| Param | Doc | Source (gemma3.zig) | Status |
|-------|-----|---------------------|--------|
| n_embd | 1152 | `default_n_embd: u32 = 1152` (line ~42) | ✅ |
| n_heads | 4 | `default_n_head: u32 = 4` (line ~48) | ✅ |
| n_kv | 1 | `default_n_head_kv: u32 = 1` (line ~50) | ✅ |
| head_dim | 256 | `default_head_dim: u32 = 256` (line ~44) | ✅ |
| ff_dim | 6912 | `default_n_ff: u32 = 6912` (line ~46) | ✅ |
| n_layers | 26 | `default_n_layers: u32 = 26` (line ~40) | ✅ |
| theta | 1M | `default_rope_freq_base: f32 = 1_000_000.0` (line ~34) | ✅ |
| rope_dim | 256 | `rope_dim = head_dim` (set in init, line ~125) | ✅ |

**Gemma3 4B/12B/27B rows:** Values come from GGUF metadata, not from source defaults. Cannot verify from source code alone, but the pattern is architecturally consistent (decreasing head_dim for 27B, increasing layers/heads for larger variants).

#### Gemma4 (src/models/gemma4.zig)

The doc's "Gemma4 26B-A4B" row matches the source defaults:

| Param | Doc | Source (gemma4.zig) | Status |
|-------|-----|---------------------|--------|
| n_embd | 2816 | `default_n_embd: u32 = 2816` | ✅ |
| n_heads (sl) | 16 | `default_sl_n_head: u32 = 16` | ✅ |
| n_kv (sl) | 8 | `default_sl_n_kv_head: u32 = 8` | ✅ |
| head_dim (sl) | 256 | `default_sl_head_dim: u32 = 256` | ✅ |
| head_dim (gl) | 512 | `default_gl_head_dim: u32 = 512` | ✅ |
| ff_dim dense | 2816 | `default_dense_ff_dim: u32 = 2816` | ✅ |
| ff_dim MoE | 704/expert | `default_moe_intermediate: u32 = 704` | ✅ |
| n_layers | 30 | `default_n_layers: u32 = 30` | ✅ |
| theta (sl) | 10K | `default_sl_rope_theta: f32 = 10_000.0` | ✅ |
| theta (gl) | 1M | `default_gl_rope_theta: f32 = 1_000_000.0` | ✅ |
| n_experts | 128 | `default_n_experts: u32 = 128` | ✅ |
| top_k | 8 | `default_top_k_experts: u32 = 8` | ✅ |

**Gemma4 E2B (n_embd=2304, n_heads=8, n_kv=4, n_layers=35) and E4B (n_embd=2816, n_heads=16, n_kv=8, n_layers=42):** Values from GGUF metadata. Cannot verify from source defaults. However, the doc correctly notes E2B and E4B are dense (no MoE) — the source detects this from `has_expert_tensors`.

#### Qwen3.5 (src/models/qwen35.zig)

The source defaults represent a base/generic model (likely the 9B variant):

| Param | Source Default | Status |
|-------|---------------|--------|
| n_layers | 32 | Struct default |
| n_embd | 4096 | Struct default |
| n_head | 16 | Struct default |
| n_head_kv | 4 | Struct default |
| head_dim | 256 | Struct default |
| n_ff | 12288 | Struct default |
| vocab_size | 248320 | Struct default |
| rope_theta | 10,000,000.0 (10M) | Struct default |
| rope_dim | 64 | Struct default |

**Doc "Qwen3.5 0.8B" row** (n_embd=1536, n_heads=16, n_kv=4, head_dim=128, n_ff=4096, n_layers=64, theta=10M, rope_dim=64): These values come from GGUF metadata for the 0.8B variant. Only `theta=10M` and `rope_dim=64` can be verified from the source defaults — both match ✅.

**Doc "Qwen3.6 35B-A3B" row** (n_embd=2048, n_heads=16, n_kv=2, head_dim=256, ff_dim=512 MoE×256, n_layers=40, theta=10M, rope_dim=64): Values from GGUF metadata. The MoE defaults in source are `default_moe_experts_active = 8` (top-8) and `default_moe_expert_ff_dim = 512`. Doc says "ff_dim = 512 (MoE×256)" which matches per-expert = 512. The doc's "256 experts" claim for Qwen3.6 matches the source detection pattern that defaults to 256 when expert tensors are present (`@as(u32, 256)` in init). ✅

#### GPT-OSS (src/models/gpt_oss.zig)

| Param | Doc | Source (gpt_oss.zig) | Status |
|-------|-----|----------------------|--------|
| n_embd | 2880 | `n_embd: u32 = 2880` | ✅ |
| n_heads | 64 | `n_head: u32 = 64` | ✅ |
| n_kv | 8 | `n_head_kv: u32 = 8` | ✅ |
| head_dim | 64 | `head_dim: u32 = 64` | ✅ |
| n_ff (MoE) | 2880 | `n_ff: u32 = 2880` | ✅ |
| n_layers | 24 | `n_layers: u32 = 24` | ✅ |
| theta | 150K | `rope_theta: f32 = 150000.0` | ✅ |
| rope_dim | 64 | Implicit: `hd` used for RoPE = 64 | ✅ |
| n_experts | 32 | `n_experts: u32 = 32` | ✅ |
| n_active | 4 (top-4) | `n_experts_active: u32 = 4` | ✅ |

#### Nemotron-H (src/models/nemotron_h.zig)

| Param | Doc | Source (nemotron_h.zig) | Status |
|-------|-----|------------------------|--------|
| n_embd | 3136 | `n_embd: u32 = 3136` | ✅ |
| n_heads | 40 | `n_head: u32 = 40` | ✅ |
| n_kv | 8 | `n_head_kv: u32 = 8` | ✅ |
| head_dim | 128 | `head_dim: u32 = 128` | ✅ |
| n_ff | 12544 | `n_ff: u32 = 12544` | ✅ |
| n_layers | 42 | `n_layers: u32 = 42` | ✅ |
| theta | 10K | `rope_theta: f32 = 10000.0` | ✅ |
| rope_dim | 78 | `rope_dim: u32 = 78` | ✅ |

#### Nemotron Nano (src/models/nemotron_nano.zig)

| Param | Doc | Source (nemotron_nano.zig) | Status |
|-------|-----|--------------------------|--------|
| n_embd | 2688 | `n_embd: u32 = 2688` | ✅ |
| n_heads | 32 | `n_head: u32 = 32` | ✅ |
| n_kv | 2 | `n_head_kv: u32 = 2` | ✅ |
| head_dim | 128 | `head_dim: u32 = 128` | ✅ |
| n_ff (MoE) | 1856 | `moe_intermediate_size: u32 = 1856` | ✅ |
| n_layers | 52 | `n_layers: u32 = 52` | ✅ |
| theta | 10K | `rope_theta: f32 = 10000.0` | ✅ |
| rope_dim | 128 | `rope_dim: u32 = 128` | ✅ |
| n_routed_experts | 128 | `n_routed_experts: u32 = 128` | ✅ |
| top_k | 6 | `num_experts_per_tok: u32 = 6` | ✅ |

#### GLM-4 (src/models/glm4.zig)

| Param | Doc | Source (glm4.zig) | Status |
|-------|-----|-------------------|--------|
| n_embd | 2048 | `n_embd: u32 = 2048` | ✅ |
| n_heads | 20 | `n_head: u32 = 20` | ✅ |
| n_kv | 20 (MLA) | `n_head_kv: u32 = 20` | ✅ |
| head_dim | 256 (qk_nope=192 + qk_rope=64) | `v_head_dim: u32 = 256`, `qk_nope_head_dim: u32 = 192`, `qk_rope_head_dim: u32 = 64` | ✅ |
| ff_dim (dense) | 10240 | `intermediate_size: u32 = 10240` | ✅ |
| ff_dim (MoE) | 1536, 64 experts top-4 | `moe_intermediate_size: u32 = 1536`, `n_routed_experts: u32 = 64`, `num_experts_per_tok: u32 = 4` | ✅ |
| n_layers | 47 | `n_layers: u32 = 47` | ✅ |
| theta | 1M | `rope_theta: f32 = 1000000.0` | ✅ |
| rope_dim | 64 | `qk_rope_head_dim: u32 = 64` | ✅ |
| Routing | sigmoid | Source confirms sigmoid routing in comments and code | ✅ |

#### Llama 4 Scout (src/models/llama4.zig)

| Param | Doc | Source (llama4.zig) | Status |
|-------|-----|---------------------|--------|
| n_embd | 5120 | `default_n_embd: u32 = 5120` | ✅ |
| n_heads | 40 | `default_n_head: u32 = 40` | ✅ |
| n_kv | 8 | `default_n_head_kv: u32 = 8` | ✅ |
| head_dim | 128 | `default_head_dim: u32 = 128` | ✅ |
| n_ff | 14336 | `default_n_ff: u32 = 14336` | ✅ |
| n_layers | 48 | `default_n_layers: u32 = 48` | ✅ |
| theta | 500K | `default_rope_theta: f32 = 500_000.0` | ✅ |
| rope_dim | 128 | Implicit: `head_dim = 128` | ✅ |
| MoE | top-1 + shared | `n_experts_active` defaults to 1 from metadata | ✅ |

### Model-Specific Details Verification

| Doc Claim | Source Verification | Status |
|-----------|-------------------|--------|
| Gemma3: "GELU + SwiGLU" | `geluMul` called in gemma3.zig | ✅ |
| Gemma3: "Embeddings scaled by sqrt(n_embd)" | `embd_scale = @sqrt(@floatFromInt(n_embd))` | ✅ |
| Qwen3.5: "SiLU + SwiGLU" | `siluMul` called in qwen35.zig | ✅ |
| Qwen3.5: "DeltaNet SSM hybrid" | `deltaNetLayer` function present, `full_attn_interval: u32 = 4` | ✅ |
| GPT-OSS: "SwiGLU clamped [-7.0, +7.0]" | `swiglu_limit: f32 = 7.0` | ✅ |
| GPT-OSS: "Even layers = sliding window, odd = full" | `is_sliding = (li % 2 == 0)` in attentionLayer | ✅ |
| Nemotron-H: "Squared ReLU for FFN-only layers" | Comment confirms in source | ✅ |
| Nemotron-Nano: "ReLU² MoE" | `math_ops.applyReluSquared` called | ✅ |
| GLM-4: "Sigmoid routing" | Confirmed in source comments and code | ✅ |
| Llama4: "iRoPE (local+global, chunked)" | `nope_interval: u32` and `chunk_size: u32` present | ✅ |

---

## KERNELS.md Audit

### Issue 1: Missing kernel file entries for AWQ and TQ1_0 across all backends

[MODERATE] docs/KERNELS.md: Kernel File Locations section

**Doc claims:** Lists specific kernel files per backend, but **omits** `gemv_awq` and `gemv_tq1_0` entries for ALL backends (CPU, Metal, Vulkan, CUDA, ROCm, WebGPU).

**Source says:** Every backend has AWQ and TQ1_0 kernel files:
- CPU: `gemv_tq1_0.zig` exists in `src/backend/kernels/cpu/`
- Metal: AWQ/TQ1_0 handled via `gemv.metal` dispatch
- Vulkan: `gemv_awq.comp`, `gemv_tq1_0.comp` exist
- CUDA: `gemv_awq.zig`, `gemv_tq1_0.zig` exist
- ROCm: `gemv_awq.zig`, `gemv_tq1_0.zig` exist
- WebGPU: `gemv_awq.wgsl`, `gemv_tq1_0.wgsl` exist

**Fix:** Add `gemv_awq` and `gemv_tq1_0` to each backend's file listing. Add AWQ row to the GEMV by Data Type table. Add tq1_0 row to the GEMV by Data Type table.

Additionally, the CPU listing omits `gemv_fp4.zig`, `gemv_fp8.zig`, `gemv_iq4.zig`, `gemv_q_small.zig`, `gemv_tq1_0.zig` — these are actual files in the cpu directory.

### Issue 2: Kernel file and pipeline count discrepancies

[LOW] docs/KERNELS.md: "Pipeline/kernel counts" line

**Doc claims:** "Vulkan 44 shaders, WebGPU 43 shaders"  
**Source says:** Vulkan has 46 `.comp` files; WebGPU has 45 `.wgsl` files (the difference is exactly the 2 missing AWQ/TQ1_0 entries per backend).

**Fix:** Update to "Vulkan 46 shaders, WebGPU 45 shaders" (or add the missing file entries and re-count).

### Issue 3: mega_compose.zig line count discrepancy

[LOW] docs/KERNELS.md: "~773 lines in `mega_compose.zig`"

**Doc claims:** "~773 lines in `mega_compose.zig` (composable generator)"  
**Source says:** `wc -l src/backend/mega_compose.zig` → **1036 lines**

**Fix:** Update to "~1036 lines in `mega_compose.zig`"

### Issue 4: Megakernel file count and line count discrepancy

[LOW] docs/KERNELS.md: "Total megakernel code: ~4,640 lines across 16 files"

**Doc claims:** "~4,640 lines across 16 files"  
**Source says:** 15 megakernel/fused-FFN files totaling 4,478 lines (counted via `find + wc -l` for all files matching `mega_*`, `megakernel*`, `fused_ffn*` in all kernel directories).

**Fix:** Update to "~4,478 lines across 15 files"

### Verified KERNELS.md Claims (No Issues)

| Claim | Verification | Status |
|-------|-------------|--------|
| Metal megakernel.metal: 11 fused FFN kernels | `grep -c 'kernel void'` = 11 | ✅ |
| mega_common.metal: 18 primitives, 732 lines | `grep -c '^inline void mega_'` = 18; `wc -l` = 732 | ✅ |
| 5 Metal true megakernels | 5 `mega_*.metal` files (excluding mega_common.metal) | ✅ |
| 3 CUDA true megakernels | 3 `mega_*.zig` files in cuda/ | ✅ |
| 1 ROCm true megakernel | 1 `mega_*.zig` file in rocm/ | ✅ |
| CUDA 56 kernels | 58 .zig files minus all.zig + common.zig = 56 | ✅ |
| ROCm 44 kernels | 46 .zig files minus all.zig + common.zig = 44 | ✅ |
| ROCm: no add_scaled kernel | No `add_scaled.zig` in rocm/ | ✅ |
| CUDA fused SiLU: Q8_0, Q4_K, Q5_K, Q6_K | Files exist: fused_ffn_{q8_0,q4_k,q5_k,q6_k}.zig | ✅ |
| CUDA fused GELU: Q8_0 only | Only fused_ffn_q8_0.zig has GELU kernel | ✅ |
| mega_compose.zig location | `src/backend/mega_compose.zig` (not in kernels/) | ✅ |

---

## Coverage Status

### Directly verified from source code:
- ✅ All fallback/default model parameters for all 8 model architectures (Gemma3, Gemma4, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4, Llama4)
- ✅ Activation function types (GELU vs SiLU vs ReLU²) for each model
- ✅ MoE configuration defaults (expert count, top-k, routing type)
- ✅ RoPE theta values for all models
- ✅ All kernel file existence claims across all 6 backends
- ✅ Kernel file counts per backend
- ✅ Megakernel file counts and primitive counts
- ✅ Metal mega_common.metal line count and primitive count

### Cannot verify from source (metadata-dependent):
- Gemma3 4B/12B/27B parameters (loaded from GGUF metadata at runtime)
- Qwen3.5 0.8B parameters (loaded from GGUF metadata at runtime)
- Qwen3.6 35B-A3B parameters (loaded from GGUF metadata at runtime)
- Gemma4 E2B and E4B parameters (loaded from GGUF metadata at runtime)
- Gemma4 26B-A4B global KV head count of 2 (loaded from per-layer metadata array)
- Llama 4 Scout expert count (loaded from GGUF metadata at runtime; source defaults to 0)
- Performance numbers (tok/s benchmarks — require runtime measurement)

### Needs follow-up: none
All verifiable claims have been checked. No blocked items.
