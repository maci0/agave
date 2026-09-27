# DS V4 Flash MTP Implementation Design

**Status**: implemented (shared-expert FFN only). Weights load from a caller-supplied safetensors file via `--mtp-model`; they are not bundled in GGUF. Canonical CLI: `src/main.zig` (`--mtp-model`), loader: `src/models/ds4_mtp.zig`, forward: `Ds4Model.mtpForward` (`src/models/deepseek4.zig`).

**Last updated**: 2026-09-27 (status and performance note checked against `mtpForward`).

**Implementation note:** `mtpForward` currently runs MTP layers 0–2 on every call and does not use `depth` to select a single layer. It also runs `main_proj` and `main_norm` once, ahead of the layer loop, and always fills the middle 4096-wide slot of the input from `mtp_hidden_buf` rather than zeros at depth 0. The per-depth sketch below is the intended v1 shape; do not treat the loop-all-layers path as a superseding decision.

## Architecture

The DS V4 Flash model has 3 MTP (Multi-Token Prediction) layers that predict
the next 1-3 tokens in parallel with the target model. Each MTP layer is a
complete DS V4 decoder layer (MLA attention + MoE FFN + hyper connections).

### MTP Forward Pass (per depth d=0,1,2)

```
Input construction:
  if d == 0:
    input = main_proj(concat(target_hidden[4096], zeros[4096], embed(token)[4096]))
  else:
    input = main_proj(concat(target_hidden[4096], prev_mtp_hidden[4096], embed(token)[4096]))
  # main_proj: [4096, 12288] FP8 → projects 3×4096=12288 → 4096

Layer computation (same as main model layer):
  1. main_norm(input) 
  2. hcPre(attn) → attention → hcPost(attn)
  3. hcPre(ffn) → shared expert FFN → hcPost(ffn)  [skip routed experts for v1]
  4. hidden → lm_head → argmax → draft token

Output:
  - draft_token (u32)
  - mtp_hidden (saved for next depth)
```

### Weight Loading

MTP weights live in a separate safetensors file (on the order of 595MB for Flash 0731).
Pass the path with `--mtp-model`; the loader mmaps the file. GGUF checkpoints omit these tensors.

### Tensor Names

`MtpWeights.load` keys every tensor by its checkpoint (HF) name verbatim, so lookups in
`mtpForward` and its helpers use those names directly. There is no separate internal name.
Every FP8 weight has a matching `.scale` (E8M0) tensor, also read by name.

| Name | Shape | Type | Read by `mtpForward` |
|------|-------|------|:--------------------:|
| mtp.{d}.main_proj.weight / .scale | [4096, 12288] | FP8 | yes (depth 0 only) |
| mtp.{d}.main_norm.weight | [4096] | BF16 | yes (depth 0 only) |
| mtp.{d}.attn_norm.weight | [4096] | BF16 | yes |
| mtp.{d}.attn.q_norm.weight | [4096] | BF16 | yes |
| mtp.{d}.attn.kv_norm.weight | [512] | BF16 | yes |
| mtp.{d}.attn.wq_a.weight / .scale | [1024, 4096] | FP8 | yes |
| mtp.{d}.attn.wq_b.weight / .scale | [32768, 1024] | FP8 | yes |
| mtp.{d}.attn.wkv.weight / .scale | [512, 4096] | FP8 | yes (depth 0 weights also fill the MTP KV cache) |
| mtp.{d}.attn.wo_a.weight / .scale | [8192, 4096] | FP8 | yes |
| mtp.{d}.attn.wo_b.weight / .scale | [4096, 8192] | FP8 | yes |
| mtp.{d}.ffn_norm.weight | [4096] | BF16 | yes |
| mtp.{d}.ffn.shared_experts.w1.weight / .scale | [2048, 4096] | FP8 | yes |
| mtp.{d}.ffn.shared_experts.w2.weight / .scale | [4096, 2048] | FP8 | yes |
| mtp.{d}.ffn.shared_experts.w3.weight / .scale | [2048, 4096] | FP8 | yes |
| mtp.{d}.hc_{attn,ffn}_{fn,base,scale} | various | F32 | yes (no `.weight` suffix) |
| mtp.2.norm.weight | [4096] | BF16 | yes |
| mtp.2.confidence_head.proj.weight | [1, 4352] | BF16 | no |
| mtp.2.markov_head.markov_w1.weight | [129280, 256] | BF16 | no |
| mtp.2.markov_head.markov_w2.weight | [129280, 256] | BF16 | no |
| mtp.2.hc_head_{fn,base,scale} | various | F32 | no |

The last four rows are loaded (the loader takes every `mtp.*` tensor) but no code path
reads them; the output head uses `mtp.2.norm.weight` followed by the main model's shared
`output.weight` LM head.

### Memory Layout

MTP tensors are mmap'd from the safetensors file (595MB).
On 48GB system: 595MB fits easily alongside the 155GB main model page cache.
No SSD streaming needed for MTP non-expert weights.

### Performance Estimate (unmeasured projection)

The numbers below are a pre-implementation estimate, not a measurement: MTP has
never been benchmarked (`docs/BENCHMARKS.md` has no MTP entry). Treat the
throughput and acceptance figures as a hypothesis to test, not a result.

Each MTP forward:
- main_proj GEMV: [4096, 12288] × [12288] → ~50M FLOPs
- Attention GEMVs: ~5 × [~4K, ~4K] → ~80M FLOPs
- Shared expert FFN: 3 × [2048, 4096] → ~50M FLOPs
- HC: negligible
- Total: ~180M FLOPs per MTP depth
- At 10 GFLOPS (CPU with 14 threads): ~18ms per MTP depth

3 MTP depths: ~54ms per target token
Target token: ~770ms
MTP overhead: 54/770 = 7%
Assumed draft acceptance: ~60% (3 drafts → ~1.8 accepted)
Projected throughput: 2.8 tokens per 824ms = 3.4 tok/s
