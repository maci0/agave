# Supported Models

Download models directly from Hugging Face Hub:

```bash
agave pull Qwen/Qwen3.5-9B-GGUF --quant Q4_K_M    # download specific quant
agave pull google/gemma-4-4b-it-gguf --list          # list available files
```

## Overview

| Model | Arch ID | Attention | FFN | Special |
|-------|---------|-----------|-----|---------|
| **Qwen 3.5/3.6/3.8** | `qwen35` | GQA (every 4th layer) | SiLU + SwiGLU | DeltaNet SSM hybrid, MoE (3.5-35B, 3.6-35B, Nex-N2-Pro 512-expert), dense 3.8-27B, MTP heads, attn_output_gate, 3.8 native vision |
| **Qwen 3.8 Flash-Next GGUF** | `qwen4exp` | QSA GQA every 4th layer (indexer top-k) | SiLU SwiGLU MoE | llama.cpp GGUF; 125B MoE (512 experts, top-10 + shared), 4-stream HC, GDN sigmoid output gate, n-gram PLE (mmap, no GPU upload) |
| **Gemma 4** | `gemma4` | GQA + QK norm + post-norms | GELU + SwiGLU | MoE (top-8) or dense, PLE (E2B/E4B), vision (SigLIP-2), Q4_K/Q5_K/Q6_K GEMM |
| **DiffusionGemma** | `diffusion_gemma` | GQA + bidirectional canvas | SiLU + SwiGLU | Block diffusion: 256-token canvas, 128 MoE experts top-8, BF16 SafeTensors only |
| **DeepSeek V4 Flash** | `deepseek4` | MLA (K=V compressed) | SiLU + SwiGLU | 4-stream HC, CSA/HCA compressors, LID, hash+sqrt_softplus routing, 256 experts top-6, output LoRA |
| **Qwen4-Exp** | `qwen4_exp` | Gated DeltaNet + full attention (runs the `qwen35` implementation) | SiLU + SwiGLU | 512 experts top-10, NVFP4. PLE ngram, HC and the QSA indexer are **not** implemented on this path |
| **Llama 4** | `llama4` | iRoPE (local+global, chunked) | SiLU + SwiGLU | MoE (top-1) + shared expert, temperature scaling, 10M context |

## Speculative Decoding Support

| Model | DDTree | Self-Spec | EAGLE/EAGLE-3 | MTP | N-gram | Suffix | Lookahead | PFlash | DSpark | Notes |
|-------|--------|-----------|---------------|-----|--------|--------|-----------|--------|--------|-------|
| Gemma 4 | ❌ | ✅ | ❌/✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | KV export/import for cross-instance sharing |
| Qwen 3.5 | ❌ | ✅ | ✅/❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | SSM state save/restore for rollback |
| Qwen 3.8 Flash-Next | ❌ | ❌ | ❌/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | GGUF `qwen4exp`: no MTP/megakernel; auto `--mmap`; IQ expert GEMV on CPU |
| DeepSeek V4 | ⚠️ causal `forwardTree` | ✅ `setLayerSkip` | ❌/❌ | ✅ `--mtp-model` | ✅ | ✅ | ✅ | ✅ | ✅ | `forwardTree` ignores ancestor masks (no HC); dedicated MTP weights (`ds4_mtp.zig`) |
| Llama 4 | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Qwen4-Exp | ❌ | ✅ | ✅/❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Runs the `qwen35` implementation |
| DiffusionGemma | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Block diffusion (not autoregressive) |

All autoregressive models support standard draft-verify, n-gram, suffix, lookahead, PFlash, and DSpark modes. Ancestor-masked DDTree verification (`be.sdpaTree`) has no implementation: Gemma 3 was its only user and has been removed. DeepSeek V4 implements `forwardTree`/`treeLogits`, but with standard causal attention (the ancestor-mask argument is unused). EAGLE-3 requires `hidden_pre_norm` (Gemma 4, DiffusionGemma). MTP requires dedicated MTP heads in the model weights.

**DFlash2** is a block-diffusion *drafter* (`arch` `dflash2`, `-Denable-dflash2`), not a chat model. Load it with `--draft-model` and `--spec-mode dflash2` (alias `dflash`). The checkpoint has no embeddings or LM head; both bind from the target at runtime. See `src/models/dflash2.zig`.


## Model Parameters

| Model | n_embd | n_heads | n_kv_heads | head_dim | ff_dim | n_layers | theta | rope_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3.5 0.8B | 1536 | 16 | 4 | 128 | 4096 | 64 | 10M | 64 |
| Qwen3.8 27B | 5120 | 24 | 4 | 256 | 17408 | 64 | 10M | 64 |
| Qwen3.6 35B-A3B | 2048 | 16 | 2 | 256 | 512 (MoE×256) | 40 | 10M | 64 |
| Qwen3.8 Flash-Next | 2560 | 24 | 2 | 256 | 640 (MoE×512, top-10) | 48 | 10M | 64 |
| Gemma4 E2B | 2304 | 8 | 4 | 256 | 9216 | 35 | 10K | 256 |
| Gemma4 E4B | 2816 | 16 | 8 | 256 | 11264 | 42 | 10K | 256 |
| Gemma4 12B | 2304 | 8 | 8/1 (sl/gl) | 256/512 (sl/gl) | 9216 | 48 | 10K | 256/128 (sl/gl) |
| Gemma4 26B-A4B | 2816 | 16 | 8/2 (sl/gl) | 256/512 (sl/gl) | 2816 + 704/expert (MoE) | 30 | 10K/1M (sl/gl) | 256/128 (sl/gl) |
| DeepSeek V4 Flash | 4096 | 64 | 1 (MLA) | 512 (kv_lora=512 + rope=64) | 2048 (MoE, 256 experts top-6 + 1 shared) | 43 | 10K | 64 |
| Llama 4 Scout | 5120 | 40 | 8 | 128 | 14336 (MoE top-1 + shared) | 48 | 500K | 128 |

## Model-Specific Details


**Qwen 3.5/3.6/3.8**: Hybrid architecture alternating DeltaNet SSM and full attention layers (every 4th layer is full attention). DeltaNet uses causal conv1d → delta rule state recurrence with learned decay (alpha) and update strength (beta). Full attention Q-gate is `attn * sigmoid(gate)` (HuggingFace `Qwen3_5Attention`). Config `output_gate_type: "swish"` is the DeltaNet RMSNormGated z-activation (already SiLU), not the full-attention gate. HuggingFace RMSNorm is `(1+w)*rms(x)`; GGUF converters and MLX `sanitize()` bake the +1 (detected via conv1d `[C,K,1]`). Do not bake again on those checkpoints. Qwen 3.6-35B-A3B uses same arch with 40 layers, 256 experts (top-8 + shared), hidden_size 2048. **Qwen 3.8-27B** (`Qwen/Qwen3.8-27B`) is dense (no MoE): hidden 5120, 24 Q / 4 KV heads, head_dim 256 (not n_embd/n_head), rope_dim 64, FFN 17408, 48 V / 16 K DeltaNet heads (`ssm_d_inner=6144`, conv channels 10240). Chat EOS is top-level `eos_token_id[0]` (`<|im_end|>` = 248046); `text_config.eos_token_id` is pad/EOT and must not overwrite it. MTP lives in `mtp.*` on SafeTensors (`mtp.fc`, `mtp.layers.0`, `mtp.norm`) and `blk.{n_layers}.*` on GGUF. Native vision is in the same checkpoint (`model.visual.*` / `vision_tower.*`, ViT depth 27, hidden 1152, patch 16, spatial merge 2, Conv3d temporal_patch_size 2, image_size 768). K/Q grouping is HuggingFace/llama.cpp `repeat_interleave` (`kh = h * n_k / n_v`), not modulo. **nex-agi/Nex-N2-Pro**: same `qwen35moe` arch, 60 layers (3 DeltaNet + 1 full_attention × 15), 512 experts (top-10), hidden_size 4096, full-attention output gate (`attn_output_gate`), MTP head. Expert count is auto-detected from weight tensor dimensions. Formats: GGUF (Q4_K_M, Q8_0), SafeTensors (BF16, MLX-4bit). Supports `--megakernel` (fused FFN SiLU, true megakernel Q8/Q4K on Metal+CUDA).

**Qwen 3.8 Flash-Next GGUF** (`qwen4exp`): separate architecture from `qwen35` and from SafeTensors `qwen4_exp`. llama.cpp split GGUF uses this arch string. 48 layers, hidden 2560, 4-stream hyper-connections (rank 320), GDN on non-QSA layers with a sigmoid output gate (not SiLU), QSA every 4th layer (24Q/2KV, head_dim 256, indexer 4Q/1K top-k 2048), n-gram PLE on layer 1 (3-gram, 8 heads/ngram, head_dim 160, dilated conv). MoE is 512 experts, top-10 plus a sigmoid-gated shared expert, ff=640. The PLE table `per_layer_token_embd.weight` is tens of GB: Agave auto-enables `--mmap`, marks that tensor `MADV_RANDOM`, and gathers rows on the CPU (iq4_nl). IQ2/IQ3 expert GEMV also stays on a dedicated CpuBackend because GPU kernels panic on those dtypes. No megakernel, vision, or MTP in v1. Chat template is Qwen 3.5. Fallback EOS is 248046.

**Qwen4-Exp Flash-Next** (`qwen4_exp`, `Qwen/Qwen3.8-Flash-Next`): SafeTensors `model_type qwen4_exp`. Runs on the Qwen3.5 implementation (`src/models/qwen35.zig`) — the DeltaNet + full-attention layer stack and the MoE FFN. Config comes from the HF keys through the `gguf_hf_meta_map` translation in `src/format/safetensors.zig`, so no model-side key handling is needed. `-Denable-qwen4-exp` therefore also requires `-Denable-qwen35`.

The checkpoint itself is 48 layers, 2560 hidden, 24/2 heads head_dim 256 (irregular: 256*24=6144≠2560, so head_dim must come from metadata, never from `n_embd/n_head`), 512 experts top-10 640 + shared 640. NVFP4 is `modelopt` group 16 (`weight_packed/weight_scale/weight_global_scale`) via `fuseNvfp4Experts`; attention/BF16 stays in the ignore-list. Chat template is Qwen3.5 (`im_start`/`im_end`, eos 248046).

Not implemented for this arch: the PLE ngram table (20M vocab, 128 shards `model-plefp8-*.safetensors`, FP8 E4M3), Gated Residual hyper-connections (4×320), the QSA sparse indexer, and MTP-1. The checkpoint carries the config and weights for all four; the forward pass ignores them. `NgramCache` (`src/ngram_cache.zig`) exists but is not wired to `--ssd-streaming`. For a Flash-Next model that *does* implement PLE and hyper-connections, use the split-GGUF `qwen4exp` arch above.




**Gemma 4**: Four variants, E2B and E4B are dense (no MoE), 12B is dense, 26B-A4B uses MoE (128 experts, top-8 softmax) + dense FFN path. All variants use dual attention (sliding-window + global layers) and PLE (Per-Layer Embeddings). Shared KV cache for trailing layers. Channel-based chat template. Vision supported via SigLIP-2 encoder. Supports `--megakernel` (fused FFN GELU for dense+MoE, true megakernel Q4K/Q8 on Metal+CUDA). 26B MoE now produces correct output after fixing the expert stride calculation (was computing `dims[0] * dims[1]` instead of `dims[1] * dims[2]` for 3D expert tensors).

The 12B variant has 48 layers with a global attention layer every 6 layers (layers 5, 11, 17, ...). Unlike the 26B which stores a scalar `attention.head_count_kv`, the 12B GGUF stores a per-layer `head_count_kv` array: SWA layers use nkv=8 with head_dim=256, global layers use nkv=1 with head_dim=512. Global layers also omit the V projection (tied K=V: copy K to V after `k_norm`, not before). When loading, read `attention.key_length_global` before `attention.key_length` to detect the global head dimension, if the key is absent, fall back to `attention.key_length`. Sliding window size: 4096 tokens. Maximum context: 128K.


**DeepSeek V4 Flash 0731** (GGUF): Modified MLA where K=V share a single compressed head (no separate V projection). 4-stream hyper connections (HC) with Sinkhorn-normalized combination matrices mix information across streams at each layer boundary. Routing: layers 0–2 use hash routing (deterministic expert assignment), layers 3+ use sqrt_softplus scoring with learned bias. Output uses grouped LoRA (8 groups × 1024 rank) instead of a single dense output projection. Every layer attends a 128-token raw sliding window. Compressed (CSA/HCA) layers also attend completed compressed groups and learned per-head attention sinks. KV compressors: CSA (ratio=4, 21 layers) and HCA (ratio=128, 20 layers) compress KV cache with per-ratio APE and group compression. Lightning Indexer (LID) scores compressed blocks via multi-head ReLU dot-product and selects top-k for sparse attention when block count exceeds `index_topk`. KV cache defaults to Q8_0; `--kv-type nvfp4_ds_mla` packs NoPE as NVFP4 and keeps the 64-d RoPE tail in f16. GGUF tensor prefix: `blk.N.*`. **Metal path**: 10 MSL kernels (9 in `ds4.metal` + `ds4_fused_attn_proj` in `ds4_fused.metal`) for HC mixing, RoPE, SDPA hd=512 turbo, batched MoE, GPU routing, fused attention megakernel. **CUDA path**: GEMV/MLX-Q/MXFP4 and clamped SiLU run on the CUDA backend (Metal still uses the dedicated `CpuBackend` bypass). Multi-node: `--pp 2` ships the 4-stream HC state between stages; `--tp 2` is expert-parallel (routed expert `eid % 2`) with NCCL `allReduceAdd`. Combine with `--transport nccl --spec-mode dspark`. For MLX-Q SafeTensors on Metal: dedicated `CpuBackend` bypass produces bit-identical output to `--backend cpu` at 10.7-21.2 tok/s with suffix speculation. GPU kernels activate for GGUF models with native GPU GEMV types.

**DiffusionGemma** (SafeTensors BF16 only): Google's block-autoregressive discrete text diffusion model (26B-A4B). Built on Gemma 4 26B backbone but generates text in 256-token blocks via iterative denoising. Uses *uniform state diffusion*: instead of a special [MASK] token, noisy positions are replaced with random vocabulary tokens. Each denoising step runs bidirectional attention across the entire canvas, scores each position's confidence, and locks high-confidence tokens. Up to 48 steps supported; typically converges in 12-16. Tensor prefix: `model.decoder.layers.N.` with fused `experts.gate_up_proj` per-layer. Reported up to 4x faster than autoregressive on H200 at FP8. See `--diffusion-steps`, `--diffusion-canvas`, `--diffusion-confidence`.

**Llama 4** (GGUF): iRoPE architecture alternating local RoPE (chunked attention, 8K window) and global NoPE (temperature-scaled) layers. NoPE interval = 4 (layers 3,7,11,... are global). MoE with top-1 expert routing + optional shared expert; some layers are dense. Per-head QK RMSNorm applied after RoPE on local layers. Batched prefill with chunked GEMM.

## Performance

Canonical numbers live in [BENCHMARKS.md](BENCHMARKS.md). The table below is a convenience snapshot and may lag re-benches.

### Apple M4 Pro (48 GB)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Qwen3.5 0.8B | Q8_0 | Metal | 125† |
| Qwen3.5 0.8B | Q4_0 | Metal | 110 |
| Qwen3.5 9B | Q4_K_M | Metal | 7.2 |
| Qwen3.5 9B | MLX-4bit | Metal | 12.7 |
| Qwen3.5 9B | Q4_0 | Metal | 34.5 |
| Gemma4 E2B | Q4_K_M | Metal | 21.8 |
| Gemma4 E4B | Q4_K_M | Metal | 14.4 |
| Gemma4 26B-A4B | Q4_K_M | Metal | 4.2 |

### NVIDIA GB10 (Blackwell, UMA)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
