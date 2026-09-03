# DS V4 Flash MTP Implementation Design

**Status**: implemented (shared-expert FFN only). Weights load from a caller-supplied safetensors file via `--mtp-model`; they are not bundled in GGUF. Canonical CLI: `src/main.zig` (`--mtp-model`), loader: `src/models/ds4_mtp.zig`, forward: `Ds4Model.mtpForward`.

**Implementation note:** `mtpForward` currently runs MTP layers 0–2 on every call and does not use `depth` to select a single layer. The per-depth sketch below is the intended v1 shape; do not treat the loop-all-layers path as a superseding decision.

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

### Tensor Name Mapping (HF → Internal)

| HF name | Internal name | Shape | Type |
|---------|--------------|-------|------|
| mtp.{d}.main_proj.weight | mtp.{d}.main_proj | [4096, 12288] | FP8 |
| mtp.{d}.main_proj.scale | mtp.{d}.main_proj_scale | [32, 96] | E8M0 |
| mtp.{d}.main_norm.weight | mtp.{d}.main_norm | [4096] | BF16 |
| mtp.{d}.attn_norm.weight | mtp.{d}.attn_norm | [4096] | BF16 |
| mtp.{d}.attn.wq_a.weight | mtp.{d}.attn_q_a | [1024, 4096] | FP8 |
| mtp.{d}.attn.wq_b.weight | mtp.{d}.attn_q_b | [32768, 1024] | FP8 |
| mtp.{d}.attn.wkv.weight | mtp.{d}.attn_kv | [512, 4096] | FP8 |
| mtp.{d}.attn.wo_a.weight | mtp.{d}.attn_output_a | [8192, 4096] | FP8 |
| mtp.{d}.attn.wo_b.weight | mtp.{d}.attn_output_b | [4096, 8192] | FP8 |
| mtp.{d}.ffn_norm.weight | mtp.{d}.ffn_norm | [4096] | BF16 |
| mtp.{d}.ffn.shared_experts.w1 | mtp.{d}.ffn_gate_shexp | [2048, 4096] | FP8 |
| mtp.{d}.ffn.shared_experts.w2 | mtp.{d}.ffn_down_shexp | [4096, 2048] | FP8 |
| mtp.{d}.ffn.shared_experts.w3 | mtp.{d}.ffn_up_shexp | [2048, 4096] | FP8 |
| mtp.{d}.hc_*.{fn/base/scale} | mtp.{d}.hc_* | various | F32 |
| mtp.2.confidence_head.proj.weight | mtp.2.confidence | [1, 4352] | BF16 |
| mtp.2.markov_head.markov_w1.weight | mtp.2.markov_w1 | [129280, 256] | BF16 |
| mtp.2.markov_head.markov_w2.weight | mtp.2.markov_w2 | [129280, 256] | BF16 |
| mtp.2.norm.weight | mtp.2.output_norm | [4096] | BF16 |
| mtp.2.hc_head_{fn/base/scale} | mtp.2.hc_head_* | various | F32 |

### Memory Layout

MTP tensors are mmap'd from the safetensors file (595MB).
On 48GB system: 595MB fits easily alongside the 155GB main model page cache.
No SSD streaming needed for MTP non-expert weights.

### Performance Estimate

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
Expected draft acceptance: ~60% (3 drafts → ~1.8 accepted)
Net throughput: 2.8 tokens per 824ms = 3.4 tok/s (2.6× improvement!)
