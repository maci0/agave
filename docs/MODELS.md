# Supported Models

Download models directly from Hugging Face Hub:

```bash
agave pull Qwen/Qwen3.5-9B-GGUF --quant Q4_K_M    # download specific quant
agave pull google/gemma-4-4b-it-gguf --list          # list available files
```

## Overview

| Model | Arch ID | Attention | FFN | Special |
|-------|---------|-----------|-----|---------|
| **Gemma 3** | `gemma3` | GQA + QK norm + post-norms | GELU + SwiGLU | Embedding scaling, logit softcap, vision (SigLIP) |
| **Qwen 3.5/3.6/3.8** | `qwen35` | GQA (every 4th layer) | SiLU + SwiGLU | DeltaNet SSM hybrid, MoE (3.5-35B, 3.6-35B, Nex-N2-Pro 512-expert), dense 3.8-27B, MTP heads, attn_output_gate, 3.8 native vision |
| **Qwen 3.8 Flash-Next** | `qwen4exp` | QSA GQA every 4th layer (indexer top-k) | SiLU SwiGLU MoE | 125B MoE (512 experts, top-10 + shared), 4-stream HC, GDN with sigmoid output gate, n-gram PLE (mmap, no GPU upload) |
| **GPT-OSS** | `gpt_oss` | GQA + sliding window + sinks | SiLU + SwiGLU | MoE (top-4 of 32 experts) |
| **Nemotron-H** | `nemotron_h` | GQA (sparse layers) | SiLU + SwiGLU | Mamba-2 SSM hybrid (GGUF) |
| **Nemotron Nano** | `nemotron_nano` | GQA (sparse layers) | ReLU² MoE | SSM + MoE + attention hybrid (NVFP4) |
| **Gemma 4** | `gemma4` | GQA + QK norm + post-norms | GELU + SwiGLU | MoE (top-8) or dense, PLE (E2B/E4B), vision (SigLIP-2), Q4_K/Q5_K/Q6_K GEMM |
| **DiffusionGemma** | `diffusion_gemma` | GQA + bidirectional canvas | SiLU + SwiGLU | Block diffusion: 256-token canvas, 128 MoE experts top-8, BF16 SafeTensors only |
| **DeepSeek V4 Flash** | `deepseek4` | MLA (K=V compressed) | SiLU + SwiGLU | 4-stream HC, CSA/HCA compressors, LID, hash+sqrt_softplus routing, 256 experts top-6, output LoRA |
| **GLM-4 / DeepSeek V3** | `glm4` | MLA (compressed KV) | SiLU + SwiGLU | MoE (64/256 experts, top-4/top-8, sigmoid routing) |
| **Llama 4** | `llama4` | iRoPE (local+global, chunked) | SiLU + SwiGLU | MoE (top-1) + shared expert, temperature scaling, 10M context |

## Speculative Decoding Support

| Model | DDTree | Self-Spec | EAGLE/EAGLE-3 | MTP | N-gram | Suffix | Lookahead | PFlash | DSpark | Notes |
|-------|--------|-----------|---------------|-----|--------|--------|-----------|--------|--------|-------|
| Gemma 3 | ✅ `forwardTree` | ✅ | ✅/⚠️² | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | Only model with native tree verification |
| Gemma 4 | ❌ | ✅ | ❌/✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | KV export/import for cross-instance sharing |
| Qwen 3.5 | ❌ | ✅ | ✅/❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | SSM state save/restore for rollback |
| Qwen 3.8 Flash-Next | ❌ | ❌ | ❌/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | v1: no MTP/megakernel; auto `--mmap`; IQ expert GEMV on CPU |
| DeepSeek V4 | ❌ | ✅ `setLayerSkip` | ❌/❌ | ✅ `--mtp-model` | ✅ | ✅ | ✅ | ✅ | ✅ | Layer-skip self-speculative; dedicated MTP weights (`ds4_mtp.zig`) |
| GLM-4 | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| GPT-OSS | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Nemotron-H | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Nemotron Nano | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| Llama 4 | ❌ | ✅ | ✅/❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| DiffusionGemma | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Block diffusion (not autoregressive) |

All autoregressive models support standard draft-verify, n-gram, suffix, lookahead, PFlash, and DSpark modes. DDTree tree verification requires `forwardTree`/`treeLogits` (currently Gemma 3 only). EAGLE-3 requires `hidden_pre_norm` (Gemma 4, DiffusionGemma). MTP requires dedicated MTP heads in the model weights.

² Gemma 3 lacks `hidden_pre_norm`, so EAGLE-3 falls back to post-norm hidden state (equivalent to regular EAGLE, no additional benefit).

## Model Parameters

| Model | n_embd | n_heads | n_kv_heads | head_dim | ff_dim | n_layers | theta | rope_dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma3 1B | 1152 | 4 | 1 | 256 | 6912 | 26 | 1M | 256 |
| Gemma3 4B | 2560 | 8 | 4 | 256 | 10240 | 34 | 1M | 256 |
| Gemma3 12B | 3840 | 16 | 8 | 256 | 15360 | 48 | 1M | 256 |
| Gemma3 27B | 4608 | 32 | 16 | 128 | 36864 | 62 | 1M | 128 |
| Qwen3.5 0.8B | 1536 | 16 | 4 | 128 | 4096 | 64 | 10M | 64 |
| Qwen3.8 27B | 5120 | 24 | 4 | 256 | 17408 | 64 | 10M | 64 |
| Qwen3.6 35B-A3B | 2048 | 16 | 2 | 256 | 512 (MoE×256) | 40 | 10M | 64 |
| Qwen3.8 Flash-Next | 2560 | 24 | 2 | 256 | 640 (MoE×512, top-10) | 48 | 10M | 64 |
| GPT-OSS | 2880 | 64 | 8 | 64 | 2880 (MoE) | 24 | 150K | 64 |
| Nemotron-H | 3136 | 40 | 8 | 128 | 12544 | 42 | 10K | 78 |
| Nemotron-Nano | 2688 | 32 | 2 | 128 | 1856 (MoE) | 52 | 10K | 128 |
| Gemma4 E2B | 2304 | 8 | 4 | 256 | 9216 | 35 | 10K | 256 |
| Gemma4 E4B | 2816 | 16 | 8 | 256 | 11264 | 42 | 10K | 256 |
| Gemma4 12B | 2304 | 8 | 8/1 (sl/gl) | 256/512 (sl/gl) | 9216 | 48 | 10K | 256/128 (sl/gl) |
| Gemma4 26B-A4B | 2816 | 16 | 8/2 (sl/gl) | 256/512 (sl/gl) | 2816 + 704/expert (MoE) | 30 | 10K/1M (sl/gl) | 256/128 (sl/gl) |
| GLM-4 | 2048 | 20 | 20 (MLA) | 256 (qk_nope=192 + qk_rope=64) | 10240 (dense) / 1536 (MoE, 64 experts top-4) | 47 | 1M | 64 |
| DeepSeek V4 Flash | 4096 | 64 | 1 (MLA) | 512 (kv_lora=512 + rope=64) | 2048 (MoE, 256 experts top-6 + 1 shared) | 43 | 10K | 64 |
| Llama 4 Scout | 5120 | 40 | 8 | 128 | 14336 (MoE top-1 + shared) | 48 | 500K | 128 |

## Model-Specific Details

**Gemma 3**: GGUF converter bakes +1.0 into RMS norm weights (don't add again). Embeddings scaled by `sqrt(n_embd)`. Uses SPM tokenizer (no merges). Tied output embeddings. Vision supported via SigLIP encoder. Supports `--megakernel` (fused FFN GELU, true megakernel Q4K/Q8 on Metal+CUDA).

**Qwen 3.5/3.6/3.8**: Hybrid architecture alternating DeltaNet SSM and full attention layers (every 4th layer is full attention). DeltaNet uses causal conv1d → delta rule state recurrence with learned decay (alpha) and update strength (beta). Full attention Q-gate is `attn * sigmoid(gate)` (HuggingFace `Qwen3_5Attention`). Config `output_gate_type: "swish"` is the DeltaNet RMSNormGated z-activation (already SiLU), not the full-attention gate. HuggingFace RMSNorm is `(1+w)*rms(x)`; GGUF converters and MLX `sanitize()` bake the +1 (detected via conv1d `[C,K,1]`). Do not bake again on those checkpoints. Qwen 3.6-35B-A3B uses same arch with 40 layers, 256 experts (top-8 + shared), hidden_size 2048. **Qwen 3.8-27B** (`Qwen/Qwen3.8-27B`) is dense (no MoE): hidden 5120, 24 Q / 4 KV heads, head_dim 256 (not n_embd/n_head), rope_dim 64, FFN 17408, 48 V / 16 K DeltaNet heads (`ssm_d_inner=6144`, conv channels 10240). Chat EOS is top-level `eos_token_id[0]` (`<|im_end|>` = 248046); `text_config.eos_token_id` is pad/EOT and must not overwrite it. MTP lives in `mtp.*` on SafeTensors (`mtp.fc`, `mtp.layers.0`, `mtp.norm`) and `blk.{n_layers}.*` on GGUF. Native vision is in the same checkpoint (`model.visual.*` / `vision_tower.*`, ViT depth 27, hidden 1152, patch 16, spatial merge 2, Conv3d temporal_patch_size 2, image_size 768). K/Q grouping is HuggingFace/llama.cpp `repeat_interleave` (`kh = h * n_k / n_v`), not modulo. **nex-agi/Nex-N2-Pro**: same `qwen35moe` arch, 60 layers (3 DeltaNet + 1 full_attention × 15), 512 experts (top-10), hidden_size 4096, full-attention output gate (`attn_output_gate`), MTP head. Expert count is auto-detected from weight tensor dimensions. Formats: GGUF (Q4_K_M, Q8_0), SafeTensors (BF16, MLX-4bit). Supports `--megakernel` (fused FFN SiLU, true megakernel Q8/Q4K on Metal+CUDA+ROCm).

**Qwen 3.8 Flash-Next** (`qwen4exp`): separate architecture from `qwen35`. 48 layers, hidden 2560, 4-stream hyper-connections (rank 320), GDN on non-QSA layers with a sigmoid output gate (not SiLU), QSA every 4th layer (24Q/2KV, head_dim 256, indexer 4Q/1K top-k 2048), n-gram PLE on layer 1 (3-gram, 8 heads/ngram, head_dim 160, dilated conv). MoE is 512 experts, top-10 plus a sigmoid-gated shared expert, ff=640. The PLE table `per_layer_token_embd.weight` is tens of GB: Agave auto-enables `--mmap`, marks that tensor `MADV_RANDOM`, and gathers rows on the CPU (iq4_nl). IQ2/IQ3 expert GEMV also stays on a dedicated CpuBackend because GPU kernels panic on those dtypes. No megakernel, vision, or MTP in v1. Chat template is Qwen 3.5.

**GPT-OSS**: Even layers = 128-token sliding window, odd = full sequence. Learned attention sinks per head. Clamped SwiGLU `[-7.0, +7.0]` in MoE experts.

**Nemotron-H** (GGUF): Mamba-2 SSM with per-group RMS normalization. Layer types (SSM/attention/FFN-only) detected from tensor presence. Squared ReLU for FFN-only layers. Supports `--megakernel` (true megakernel Q8 on Metal).

**Nemotron Nano** (SafeTensors NVFP4): 52-layer hybrid with `hybrid_override_pattern` (M=SSM, E=MoE, *=attention). Mixed quant, most layers NVFP4, 6 SSM layers use BF16. 128 routed experts, top-6 + shared expert.

**Gemma 4**: Four variants, E2B and E4B are dense (no MoE), 12B is dense, 26B-A4B uses MoE (128 experts, top-8 softmax) + dense FFN path. All variants use dual attention (sliding-window + global layers) and PLE (Per-Layer Embeddings). Shared KV cache for trailing layers. Channel-based chat template. Vision supported via SigLIP-2 encoder. Supports `--megakernel` (fused FFN GELU for dense+MoE, true megakernel Q4K/Q8 on Metal+CUDA). 26B MoE now produces correct output after fixing the expert stride calculation (was computing `dims[0] * dims[1]` instead of `dims[1] * dims[2]` for 3D expert tensors).

The 12B variant has 48 layers with a global attention layer every 6 layers (layers 5, 11, 17, ...). Unlike the 26B which stores a scalar `attention.head_count_kv`, the 12B GGUF stores a per-layer `head_count_kv` array: SWA layers use nkv=8 with head_dim=256, global layers use nkv=1 with head_dim=512. Global layers also omit the V projection (tied K=V: copy K to V after `k_norm`, not before). When loading, read `attention.key_length_global` before `attention.key_length` to detect the global head dimension, if the key is absent, fall back to `attention.key_length`. Sliding window size: 4096 tokens. Maximum context: 128K.

**GLM-4 / DeepSeek V2/V3** (MLX + GGUF): MLA (Multi-head Latent Attention) compresses K/V into a low-rank latent space via `kv_a_proj_with_mqa` → latent → per-head `kv_b_proj`. Q also uses low-rank factorization (`q_a_proj` + `q_b_proj`). Sigmoid routing for MoE (independent expert gates, not competing). GLM-4 uses MLX 4/6/8-bit affine quantization. DeepSeek V3 uses GGUF format (`arch=deepseek2`), with tensor names `blk.N.attn_q_a.weight`, `blk.N.attn_kv_a_mqa.weight` etc, both GGUF and SafeTensors now supported. MLA params (q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim) auto-detected from GGUF metadata. Supports `--megakernel` (fused FFN SiLU on Metal).

**DeepSeek V4 Flash 0731** (GGUF): Modified MLA where K=V share a single compressed head (no separate V projection). 4-stream hyper connections (HC) with Sinkhorn-normalized combination matrices mix information across streams at each layer boundary. Routing: layers 0–2 use hash routing (deterministic expert assignment), layers 3+ use sqrt_softplus scoring with learned bias. Output uses grouped LoRA (8 groups × 1024 rank) instead of a single dense output projection. Every layer attends a 128-token raw sliding window. Compressed (CSA/HCA) layers also attend completed compressed groups and learned per-head attention sinks. KV compressors: CSA (ratio=4, 21 layers) and HCA (ratio=128, 20 layers) compress KV cache with per-ratio APE and group compression. Lightning Indexer (LID) scores compressed blocks via multi-head ReLU dot-product and selects top-k for sparse attention when block count exceeds `index_topk`. KV cache defaults to Q8_0; `--kv-type nvfp4_ds_mla` packs NoPE as NVFP4 and keeps the 64-d RoPE tail in f16. GGUF tensor prefix: `blk.N.*`. **Metal path**: 14 MSL kernels (`ds4.metal` + `ds4_fused.metal`) for HC mixing, RoPE, SDPA hd=512 turbo, batched MoE, GPU routing, fused attention megakernel. **CUDA path**: GEMV/MLX-Q/MXFP4 and clamped SiLU run on the CUDA backend (Metal still uses the dedicated `CpuBackend` bypass). Multi-node: `--pp 2` ships the 4-stream HC state between stages; `--tp 2` is expert-parallel (routed expert `eid % 2`) with NCCL `allReduceAdd`. Combine with `--transport nccl --spec-mode dspark`. For MLX-Q SafeTensors on Metal: dedicated `CpuBackend` bypass produces bit-identical output to `--backend cpu` at 10.7-21.2 tok/s with suffix speculation. GPU kernels activate for GGUF models with native GPU GEMV types.

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
| Gemma3 27B | QAT 4-bit | Metal | 11.6 |
| Gemma3 27B | QAT 4-bit | CPU | 3.2 |

### NVIDIA GB10 (Blackwell, UMA)

| Model | Quant | Backend | tok/s |
|-------|-------|---------|-------|
| Gemma3 1B | Q4_0 | CUDA | 40 |
| Gemma3 1B | Q4_0 | CPU | 5.7 |
