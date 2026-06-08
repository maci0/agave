# Agave Documentation–Codebase Audit

**Date:** 2026-06-03  
**Scope:** All 11 top-level docs in `docs/` cross-referenced against Zig source  
**Method:** Parallel researcher subagents + manual spot-checks on every disputed claim  

---

## Summary

| Severity | Count |
|----------|-------|
| **ERROR** | 8 |
| **WARNING** | 11 |
| **Clean files** | 2 of 11 |

| File | Errors | Warnings |
|------|--------|----------|
| API.md | 0 | 0 |
| ARCHITECTURE.md | 1 | 2 |
| MODELS.md | 0 | 0 |
| KERNELS.md | 1 | 3 |
| MEGAKERNEL.md | 3 | 3 |
| CONTRIBUTING.md | 0 | 0 |
| PARALLELISM.md | 0 | 4 |
| BENCHMARKS.md | 2 | 1 |
| TEST_MATRIX.md | 1 | 1 |
| DOCUMENTATION.md | 0 | 0 |
| TODO.md | 0 | 4 |

---

## ERRORS (8)

### ARCHITECTURE.md

**[ERROR] GLM-4 generation_prefix**  
Doc claims: "`[gMASK]<sop>` prefix, `</think>` generation prefix"  
Source says: `src/chat_template.zig:224` — `.generation_prefix = ""` (empty string).  
Fix: Remove the `</think>` generation prefix claim. GLM-4 has no generation prefix.

### MEGAKERNEL.md

**[ERROR] Metal pipeline count**  
Doc claims: "Metal | 70+"  
Source says: `src/backend/metal.zig:463` — `pub const n_pipelines: u32 = 83;`  
Fix: Change "70+" to "83".

**[ERROR] CUDA kernel count**  
Doc claims: "CUDA | 56"  
Source says: `src/backend/cuda.zig:285` — `pub const n_kernels: u32 = 43;`  
Fix: Change "56" to "43".

**[ERROR] ROCm kernel count**  
Doc claims: "ROCm | 44"  
Source says: `src/backend/rocm.zig:198` — `pub const n_kernels: u32 = 28;`  
Fix: Change "44" to "28".

### BENCHMARKS.md

**[ERROR] PP=2 0.8B throughput contradicts TEST_MATRIX.md**  
Doc claims: "Qwen3.5 0.8B Q8_0 PP=2 NCCL RoCE: **40.2** tok/s, 112% vs Single GPU"  
TEST_MATRIX.md claims: Same config: "8.5 tok/s, 93% of single GPU"  
Analysis: 40.2 tok/s would be 437% of the 9.2 single-GPU baseline (BENCHMARKS.md row), making "112%" mathematically impossible. The TEST_MATRIX.md figure of 8.5 tok/s (93%) is physically plausible for PP=2 single-token decode.  
Fix: Reconcile across documents. If 8.5 tok/s is correct, update BENCHMARKS.md and PARALLELISM.md. If 40.2 is from a batched scenario, clarify methodology and fix the percentage.

**[ERROR] Qwen 0.8B Metal throughput internally inconsistent**  
BENCHMARKS.md claims: "Qwen3.5 0.8B Q8_0 Agave Metal: 125† tok/s" (line 15, updated 2026-05-26)  
TEST_MATRIX.md claims: Same model: "183.3 tok/s" (llama.cpp comparison table, line 100)  
BENCHMARKS.md also claims: "140.4 tok/s" in the CPU column (line 15)  
Analysis: The "125†" is the most recent measurement (post sparse-GEMV). The 183.3 figure appears stale.  
Fix: Update TEST_MATRIX.md llama.cpp comparison to use 125 tok/s, or explain the discrepancy (prompt length, measurement method, etc.).

### KERNELS.md

**[ERROR] mega_compose.zig line count**  
Doc claims: "~773 lines in `mega_compose.zig`"  
Source says: `wc -l src/backend/mega_compose.zig` = **1036 lines**  
Fix: Change "~773" to "~1036".

### TEST_MATRIX.md

**[ERROR] PP=2 benchmark contradicts BENCHMARKS.md**  
(Same issue as BENCHMARKS.md above — these two docs give conflicting numbers for the same test.)

---

## WARNINGS (11)

### ARCHITECTURE.md

**[WARNING] Recipe table — Gemma Q4 Metal row incomplete**  
Doc shows: "temp=0.7, top_p=0.95"  
Source says: `src/recipe.zig:130-137` also sets `repeat_penalty=1.05, max_tokens=1024`.  
Fix: Add `repeat=1.05` to match other rows that include repeat_penalty.

**[WARNING] Recipe table — GPT-OSS Metal row incomplete**  
Doc shows: "temp=0.5, ctx=2048"  
Source says: `src/recipe.zig:142-149` also sets `top_p=0.9, max_tokens=512`.  
Fix: Add `top_p=0.9, max_tokens=512` for consistency.

### KERNELS.md

**[WARNING] Missing AWQ/TQ1_0 kernel entries**  
Doc omits `gemv_awq` and `gemv_tq1_0` entries for all backends.  
Source: These kernel files exist across CPU, Metal, Vulkan, CUDA, ROCm, and WebGPU.  
Fix: Add AWQ and TQ1_0 rows to the GEMV data-type table and kernel file listings.

**[WARNING] Vulkan/WebGPU shader counts stale**  
Doc claims: "Vulkan 44 shaders, WebGPU 43 shaders"  
Source: Vulkan has 46 `.comp` files; WebGPU has 45 `.wgsl` files.  
Fix: Update to "Vulkan 46, WebGPU 45".

**[WARNING] Megakernel total line/file count stale**  
Doc claims: "~4,640 lines across 16 files"  
Source: Actual count differs (varies by counting method, but mega_compose.zig alone grew by ~260 lines).  
Fix: Recount and update.

### MEGAKERNEL.md

**[WARNING] mega_compose.zig line count**  
Doc claims: "~780 lines"  
Source says: 1036 lines  
Fix: Change to "~1036".

**[WARNING] Missing `mega_sync_reset` in building blocks table**  
Doc lists 17 building blocks. Source (`mega_common.metal`) has 18 `inline void mega_*` functions. `mega_sync_reset` is missing from the table.  
Fix: Add row: `mega_sync_reset` — "Reset atomic counter after sync barrier".

**[WARNING] Luce Megakernel prefill speedup**  
Doc claims: "3.4× prefill speedup"  
Source (Luce RESULTS.md): RTX 3090 pp520 shows 21,347 vs 11,247 tok/s = **1.9× vs llama.cpp** (or 2.8× vs PyTorch). The 3.4× figure does not correspond to any published result.  
Fix: Change to "1.9× prefill (vs llama.cpp)" or "2.8× (vs PyTorch)" with citation.

### PARALLELISM.md

**[WARNING] NCCL function pointer list incomplete**  
Doc lists 6 function pointers.  
Source says: `transport.zig:281-282` also resolves `ncclGroupStart` and `ncclGroupEnd`.  
Fix: Add `ncclGroupStart`, `ncclGroupEnd` to the list.

**[WARNING] NCCL deferred init description too narrow**  
Doc claims: "deferred to first `allReduceAdd` call"  
Source: `ensureNcclComm()` is called in `allReduceAdd`, `sendBuf`, **and** `recvBuf`.  
Fix: Change to "deferred to first NCCL operation".

**[WARNING] RCCL dlopen claim is speculative**  
Doc claims: "will use `dlopen(\"librccl.so\")`"  
Source: No `librccl` string exists anywhere. `Transport.init` rejects `.rccl` with `error.NotImplemented`.  
Fix: Mark RCCL as "declared but unimplemented" — remove the speculative dlopen claim.

**[WARNING] Device discovery missing WebGPU note**  
Doc claims: "All GPU backends (Metal, CUDA, Vulkan, ROCm, WebGPU) + CPU"  
Source: `BackendKind` in `src/devices/discovery.zig:12` = `{ cpu, metal, cuda, rocm, vulkan }` — no WebGPU.  
Fix: Note that WebGPU is a compute backend but not enumerable via `--list-devices`.

### TODO.md

**[WARNING] Kernel counts stale (×4)**  
- CUDA: doc says "56 kernels", source constant is 43, file count is 58
- Vulkan: doc says "44 shaders", actual is 46
- ROCm: doc says "44 kernels", source constant is 28, file count is 46
- WebGPU: doc says "43 shaders", actual is 45

Fix: Update all four counts using the authoritative `n_kernels` / `n_pipelines` constants from each backend's source file.

### BENCHMARKS.md

**[WARNING] Megakernel line/file count stale**  
Doc claims: "~4,166 lines across 12 files"  
Source: Count has grown since last update (mega_compose.zig alone is now 1036 lines vs documented ~780).  
Fix: Recount and update.

---

## Clean Files (no issues found)

1. **API.md** — All 30+ checked claims verified: endpoint paths, JSON fields, defaults (temperature=0, top_p=1.0, min_p=0, max_tokens=512, port 49453, body limit 1MB, logit_bias max 16, top_logprobs 0-20, CORS behavior, health/ready status codes, system_fingerprint format, all sampling parameter ranges).

2. **MODELS.md** — Every model parameter default verified against struct fields for all 8 architectures (Gemma3, Gemma4, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4, Llama4). n_embd, n_heads, n_kv, head_dim, n_ff, n_layers, vocab_size, rope_theta, MoE params — all correct.

3. **CONTRIBUTING.md** — Every source file path, method name, enum location, and registration pattern verified. All 6 backend files, Arch enum methods (detect, displayName, chatTemplate, isEnabled, buildFlag), build flags, DType enum, KvQuantType, VisionVariant — all correct.

4. **DOCUMENTATION.md** — All 24 tutorial links and 10 product doc links verified against actual files.

---

## Priority Triage

### Must fix (data integrity)
1. **BENCHMARKS.md ↔ TEST_MATRIX.md PP=2 contradiction** — 40.2 vs 8.5 tok/s for the same test. One is wrong.
2. **BENCHMARKS.md ↔ TEST_MATRIX.md 0.8B Metal throughput** — 125 vs 183.3 tok/s. Stale data in one doc.
3. **MEGAKERNEL.md kernel counts** — Metal 70→83, CUDA 56→43, ROCm 44→28. All three are wrong.

### Should fix (stale numbers)
4. **KERNELS.md** mega_compose line count: 773→1036
5. **MEGAKERNEL.md** mega_compose line count: 780→1036
6. **TODO.md** all four backend kernel counts stale
7. **ARCHITECTURE.md** GLM-4 generation_prefix: `</think>` → empty
8. **Luce prefill speedup**: 3.4× → 1.9×

### Nice to fix (completeness)
9. **KERNELS.md** missing AWQ/TQ1_0 entries
10. **MEGAKERNEL.md** missing `mega_sync_reset` building block
11. **PARALLELISM.md** NCCL function list, deferred init scope, RCCL status, WebGPU device discovery
12. **ARCHITECTURE.md** recipe table incomplete columns

---

## Methodology Notes

- **Model parameters** that come from GGUF metadata (not source defaults) are marked as unverifiable from source alone. The fallback defaults for all 8 architectures were fully verified.
- **Benchmark numbers** are self-reported and cannot be verified without hardware. Only internal consistency across docs was checked.
- **Kernel file counts** vs `n_kernels` constants: these can differ because file counts include `all.zig`/`common.zig` utility files that aren't standalone kernels. The authoritative count is the `n_kernels`/`n_pipelines` constant defined in each backend's source file.

---

## Sources

| Source | Location |
|--------|----------|
| Metal pipeline count | `src/backend/metal.zig:463` |
| CUDA kernel count | `src/backend/cuda.zig:285` |
| ROCm kernel count | `src/backend/rocm.zig:198` |
| mega_compose.zig | `src/backend/mega_compose.zig` (1036 lines) |
| GLM-4 chat template | `src/chat_template.zig:224` |
| NCCL function pointers | `src/parallel/transport.zig:275-282` |
| BackendKind enum | `src/devices/discovery.zig:12` |
| Server defaults | `src/server/json.zig:62-75`, `src/server/server.zig:76-101` |
| Recipe presets | `src/recipe.zig:118-170` |
| All model defaults | `src/models/{gemma3,gemma4,qwen35,gpt_oss,nemotron_h,nemotron_nano,glm4,llama4}.zig` |
| Kernel directories | `src/backend/kernels/{cpu,metal,cuda,rocm,vulkan,webgpu}/` |
