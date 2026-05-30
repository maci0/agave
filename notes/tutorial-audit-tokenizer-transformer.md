# Tutorial Audit: Tokenizer (Ch.1) & Transformer Architecture (Ch.2)

Audited: 2026-05-25
Files examined: `docs/tutorial/01-tokens-and-text.md`, `docs/tutorial/02-the-transformer.md`, and corresponding source files.

---

## Evidence Table

| # | Source | File:Line | Key Claim | Type | Confidence |
|---|--------|-----------|-----------|------|------------|
| 1 | `src/tokenizer/bpe.zig` | :1 | BPE+SPM modes exist ("Byte-level BPE tokenizer supporting BPE, SPM, and SPM-no-dummy modes") | primary | high |
| 2 | `src/tokenizer/tokenizer.zig` | :41 | `TokenizerKind = enum { bpe, spm, spm_no_dummy }` | primary | high |
| 3 | `src/tokenizer/bpe.zig` | :59–63 | tokEncode dispatches to encode/encodeSpm/encodeSpmNoDummy | primary | high |
| 4 | `src/tokenizer/bpe.zig` | :449–452 | encodeSpmNoDummy exists, docstring says "like Gemma" | primary | high |
| 5 | `src/models/qwen35.zig` | :61 | Qwen3.5 default vocab_size = 248,320 | primary | high |
| 6 | `src/models/gemma4.zig` | :57 | Gemma4 default_vocab_size = 262,144 | primary | high |
| 7 | `src/models/gpt_oss.zig` | :77 | GPT-OSS default vocab_size = 201,088 | primary | high |
| 8 | `src/models/glm4.zig` | :48 | GLM-4 default vocab_size = 154,880 | primary | high |
| 9 | `src/grammar.zig` | :235 | getEffectiveText() exists in grammar.zig (NOT bpe.zig) | primary | high |
| 10 | `src/grammar.zig` | :233–234 | "Strip BPE byte-level encoding prefix to get actual text" | primary | high |
| 11 | `src/models/gemma4.zig` | :2 | Comment: "E2B (dense, 35 layers)" | primary | high |
| 12 | HuggingFace config.json | google/gemma-4-E2B | E2B: hidden_size=1536, num_hidden_layers=35, vocab_size=262,144 | primary | high |
| 13 | `src/models/gemma4.zig` | :55–56 | Code defaults: n_layers=30, n_embd=2816 (for 26B-A4B, not E2B) | primary | high |
| 14 | HuggingFace model card | Qwen/Qwen3.5-0.8B | Qwen3.5 0.8B: 24 layers, hidden_dim=1024 | primary | high |
| 15 | `src/models/qwen35.zig` | :55 | Code default n_layers=32 (loaded from metadata) | primary | high |
| 16 | `src/backend/kernels/cpu/rope.zig` | :1–47 | Full RoPE CPU kernel (split-complex layout, SIMD) | primary | high |
| 17 | `src/backend/cpu.zig` | :421 | `pub fn rope()` on CpuBackend | primary | high |
| 18 | `src/models/llama4.zig` | :4,71,724 | iRoPE: alternating RoPE/NoPE layers with `is_nope` check | primary | high |
| 19 | `src/ops/attention.zig` | :44,95 | GQA via `nkv` param, `hpg = nh / nkv`, `kvh = h / hpg` mapping | primary | high |
| 20 | `src/ops/attention.zig` | :404–441 | Test: "sdpa multi-token with GQA" — 2 Q heads, 1 KV head | primary | high |
| 21 | `src/backend/kernels/cpu/norm.zig` | :7–41 | rmsNorm kernel: `output[i] = input[i] * weight[i] / rms(input)` | primary | high |
| 22 | `src/backend/cpu.zig` | :304 | `pub fn rmsNorm()` dispatches to norm_kernel.rmsNorm | primary | high |

---

## Findings

### Tokenizer Claims (Chapter 1)

#### 1. BPE + SPM modes in bpe.zig — ✅ MATCH

The file header [1] explicitly states "Byte-level BPE tokenizer supporting BPE, SPM, and SPM-no-dummy modes." The `TokenizerKind` enum [2] defines all three variants: `{ bpe, spm, spm_no_dummy }`. The `tokEncode` dispatch [3] routes to `encode()`, `encodeSpm()`, or `encodeSpmNoDummy()` based on the kind. Tutorial claim is accurate.

#### 2. Vocab sizes — ❌ MISMATCH (3 of 4 are wrong)

| Model | Tutorial Claim | Code Default | Verdict |
|-------|---------------|-------------|---------|
| Qwen 3.5 | 151,936 | **248,320** [5] | ❌ MISMATCH — 151,936 is the Qwen 2/2.5 vocab. Qwen 3.5 expanded it. |
| Gemma 3/4 | 262,144 | **262,144** [6] | ✅ MATCH |
| GPT-OSS | 200,064 | **201,088** [7] | ❌ MISMATCH — off by 1,024 (likely padding) |
| GLM-4 | 151,552 | **154,880** [8] | ❌ MISMATCH — off by 3,328 |

Note: The code defaults may themselves be overridden at runtime by metadata from the model file, but the tutorial's "reference" values don't match the code's hardcoded defaults.

#### 3. getEffectiveText() function — ⚠️ PARTIAL MATCH (wrong file attribution)

The function exists [9] and does strip BPE byte-level encoding prefixes [10], which is exactly what the tutorial describes. However, it's in `src/grammar.zig`, not in `src/tokenizer/bpe.zig`. The tutorial says "Grammar-constrained decoding must strip these prefixes via `getEffectiveText()`" — the purpose is correctly described, but the implication is that it's part of the tokenizer subsystem. It's actually part of the grammar module. The tutorial's "In the code" section at the bottom correctly links to `bpe.zig` for the tokenizer but doesn't explicitly locate `getEffectiveText()`.

#### 4. SPM 'no dummy' variant for Gemma — ✅ MATCH

`encodeSpmNoDummy()` exists at bpe.zig:451 [4] with docstring: "Like encodeSpm but without add_dummy_prefix — used by tokenizers (like Gemma) where ▁ prefix only appears for actual spaces." This exactly matches the tutorial's description.

---

### Transformer Claims (Chapter 2)

#### 5. Gemma4 E2B: 2304 hidden dim, 28 layers, 262,144 vocab — ❌ MISMATCH (2 of 3 wrong)

| Parameter | Tutorial Claim | Actual (HF config [12]) | Code [13] | Verdict |
|-----------|---------------|------------------------|-----------|---------|
| hidden_size | 2304 | **1536** | default=2816 (26B-A4B) | ❌ MISMATCH — E2B is 1536, not 2304 |
| num_layers | 28 | **35** | default=30 (26B-A4B); docstring says "E2B (dense, 35 layers)" [11] | ❌ MISMATCH — E2B is 35, not 28 |
| vocab_size | 262,144 | **262,144** | 262,144 [6] | ✅ MATCH |

The tutorial's worked example ("Gemma4 E2B, 2.6B parameters") uses 2304 hidden dim and 28 layers, but the actual Gemma4 E2B model has hidden_size=1536 and 35 layers. The "2.6B parameters" count also seems off — Google's docs call it "E2B" (Effective 2 Billion), not 2.6B. The code's own docstring [11] correctly identifies E2B as having 35 layers.

#### 6. Qwen3.5 0.8B: 64 layers — ❌ MISMATCH (confirmed suspicious)

The tutorial states "64 for Qwen3.5 0.8B". The HuggingFace model card [14] states Qwen3.5 0.8B has **24 layers**. The code default is `n_layers: u32 = 32` [15], which is for the larger variant. **64 layers for a 0.8B model is implausible** — that would be an extremely narrow, very deep model. For reference, 64 layers is typical for 30B+ models. The tutorial claim is wrong.

#### 7. RoPE implementation — ✅ MATCH

RoPE is implemented as a dedicated CPU kernel in `src/backend/kernels/cpu/rope.zig` [16] with full split-complex layout and SIMD vectorization. It's exposed as `pub fn rope()` on CpuBackend [17] and all other backends (Metal:1786, CUDA:1480, Vulkan:1763, ROCm:752, WebGPU:993). The tutorial correctly references `src/backend/kernels/cpu/rope.zig`.

The implementation matches the formula in the tutorial:
```
freq = exp(-log(theta) * 2i / rope_dim)
angle = pos * freq
x'[i]      = x[i] * cos(angle) - x[i+half] * sin(angle)
x'[i+half] = x[i] * sin(angle) + x[i+half] * cos(angle)
```

#### 8. iRoPE for Llama4 in llama4.zig — ✅ MATCH

The file header [18] declares "iRoPE: alternating RoPE (local) and NoPE (global) attention layers." The implementation at llama4.zig:724 [18] conditionally applies RoPE: `if (!is_nope) { self.be.rope(...) }`. The NoPE interval is configurable via `nope_interval` (default 4, meaning every 4th layer is NoPE/global). This exactly matches the tutorial's description of iRoPE.

#### 9. GQA (Grouped Query Attention) — ✅ MATCH

GQA is implemented in `src/ops/attention.zig` [19]. The key line `hpg = nh / nkv` computes heads-per-group, and `kvh = h / hpg` maps each query head to its KV head. The function signature [19] takes separate `nh` (query heads) and `nkv` (KV heads) parameters. A dedicated test [20] validates GQA with 2 Q heads sharing 1 KV head. The tutorial correctly describes GQA and its implementation.

#### 10. RMSNorm implementation — ✅ MATCH

RMSNorm is implemented in `src/backend/kernels/cpu/norm.zig` [21] with the formula `output[i] = input[i] * weight[i] / rms(input)`. The kernel uses 4x unrolled SIMD accumulation for the sum-of-squares pass, then a second pass to apply `input * inv * weight`. Dispatched via `CpuBackend.rmsNorm()` [22]. Also available as fused `addRmsNorm()`, `rmsNormMulti()` (per-head), and `rmsNormBatched()` variants. All backends (CPU, Metal, CUDA, Vulkan, ROCm, WebGPU) have implementations. Tutorial description is accurate.

---

## Summary

| Claim | Verdict | Severity |
|-------|---------|----------|
| BPE + SPM modes | ✅ MATCH | — |
| Vocab sizes | ❌ 3 of 4 MISMATCH | **High** — factual errors in reference table |
| getEffectiveText() | ⚠️ PARTIAL — wrong module implied | Low |
| SPM no-dummy for Gemma | ✅ MATCH | — |
| Gemma4 E2B dims (2304, 28 layers) | ❌ MISMATCH (1536, 35 layers) | **High** — worked example uses wrong numbers |
| Qwen3.5 0.8B 64 layers | ❌ MISMATCH (24 layers) | **High** — clearly wrong, order-of-magnitude |
| RoPE implementation | ✅ MATCH | — |
| iRoPE for Llama4 | ✅ MATCH | — |
| GQA implementation | ✅ MATCH | — |
| RMSNorm implementation | ✅ MATCH | — |

### Critical Issues (3)

1. **Gemma4 E2B worked example** in Chapter 2 uses hidden_size=2304 and 28 layers. Actual E2B is hidden_size=1536 and 35 layers. The code's own docstring says 35 layers. The tutorial's numbers don't correspond to any known Gemma4 variant.

2. **Qwen3.5 0.8B layer count** claimed as 64. Actual is 24. This is likely a confusion with a larger model (Qwen3.5 35B or similar).

3. **Vocab sizes table** has 3 incorrect entries:
   - Qwen 3.5: 151,936 → should be 248,320 (code default)
   - GPT-OSS: 200,064 → should be 201,088 (code default)
   - GLM-4: 151,552 → should be 154,880 (code default)

---

## Sources

1. `src/tokenizer/bpe.zig` line 1 — file header docstring
2. `src/tokenizer/tokenizer.zig` line 41 — TokenizerKind enum
3. `src/tokenizer/bpe.zig` lines 59–63 — tokEncode dispatch
4. `src/tokenizer/bpe.zig` lines 449–452 — encodeSpmNoDummy
5. `src/models/qwen35.zig` line 61 — vocab_size default
6. `src/models/gemma4.zig` line 57 — default_vocab_size
7. `src/models/gpt_oss.zig` line 77 — vocab_size default
8. `src/models/glm4.zig` line 48 — vocab_size default
9. `src/grammar.zig` line 235 — getEffectiveText function
10. `src/grammar.zig` lines 233–234 — docstring for getEffectiveText
11. `src/models/gemma4.zig` line 2 — docstring mentioning "E2B (dense, 35 layers)"
12. HuggingFace google/gemma-4-E2B config.json — https://huggingface.co/google/gemma-4-E2B/blob/main/config.json
13. `src/models/gemma4.zig` lines 55–56 — default_n_layers, default_n_embd
14. HuggingFace Qwen/Qwen3.5-0.8B model card — https://huggingface.co/Qwen/Qwen3.5-0.8B
15. `src/models/qwen35.zig` line 55 — n_layers default
16. `src/backend/kernels/cpu/rope.zig` lines 1–47 — RoPE kernel
17. `src/backend/cpu.zig` line 421 — rope() method
18. `src/models/llama4.zig` lines 4, 71, 724 — iRoPE implementation
19. `src/ops/attention.zig` lines 44, 95 — GQA in SDPA
20. `src/ops/attention.zig` lines 404–441 — GQA test
21. `src/backend/kernels/cpu/norm.zig` lines 7–41 — rmsNorm kernel
22. `src/backend/cpu.zig` line 304 — rmsNorm dispatch

---

## Coverage Status

- ✅ All 10 claims checked directly against source code
- ✅ External verification via HuggingFace for Gemma4 E2B and Qwen3.5 0.8B architecture specs
- ✅ Every claim has file:line evidence
- No remaining tasks or blocked items
