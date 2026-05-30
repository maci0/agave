# Audit Plan: docs/tutorial ↔ Codebase

## Target
22 tutorial files (~8500 lines) in `docs/tutorial/` — the "LLM Inference From Scratch" series for Agave.

## Scope
Compare tutorial claims (architecture, algorithms, defaults, data types, parameter names, code patterns, performance numbers) against the actual Zig source code in `src/`.

## Claim Categories to Check

### 1. Tokenizer (Ch 1)
- BPE + SPM modes in `src/tokenizer/bpe.zig`
- Vocab sizes per model (Qwen 151,936; Gemma 262,144; GPT-OSS 200,064; GLM-4 151,552)
- Byte-level BPE encoding details, `getEffectiveText()` function

### 2. Transformer Architecture (Ch 2)
- Gemma4 E2B claimed dims: 2304 hidden, 28 layers, 262,144 vocab
- Qwen3.5 0.8B: 64 layers claim
- RoPE, GQA, RMSNorm implementations
- iRoPE for Llama4

### 3. FFN / MoE (Ch 3)
- SwiGLU structure: gate_proj, up_proj, down_proj
- MoE expert counts per model (Qwen 128/256, GPT-OSS 32, Nemotron-Nano 128, Gemma4 128)
- Top-K per model (Qwen 8, GPT-OSS 4, Nemotron-Nano 6, Gemma4 8)
- Clamped SwiGLU [-7.0, +7.0] for GPT-OSS
- ReLU² for Nemotron-Nano
- Stack-allocated expert selection

### 4. Quantization (Ch 4)
- Block sizes (Q4_0/Q8_0 = 32, super-blocks = 256)
- MLX affine: group size 64, bf16 scales/biases
- Factored dequantization algebra
- KV cache quant formats (TurboQuant, PlanarQuant, IsoQuant, RotorQuant)

### 5. Memory & Caching (Ch 5)
- KV cache math (Qwen3.5 9B: 64 layers × 4 KV heads × 128 dim → 128KB/token)
- PagedAttention block size default = 16
- Paged SDPA, RadixAttention claims
- Boundary V protection (first/last 2 layers)
- Sparse V dequant threshold (1e-6), +22.8% decode throughput claim

### 6. Sampling (Ch 7)
- Temperature, top-k, top-p, min-p, repeat penalty
- XTC, DRY, Mirostat, logit bias, grammar-constrained decoding
- Default values for each parameter

### 7. Backends (Ch 8)
- 6 backends: CPU, CUDA, Metal, Vulkan, ROCm, WebGPU
- Tagged union dispatch pattern in `backend.zig`
- Kernel naming conventions

### 8. CPU SIMD (Ch 9)
- @Vector(8, f32) usage
- Multi-row GEMV batching (2-4× speedup claim)
- gemvMulti pattern

### 9. Metal Backend (Ch 11)
- Zero-copy UMA buffer wrapping
- Buffer cache by host pointer
- Threadgroup memory ≤ 32KB limit
- Dispatch overhead: 5-10µs per dispatch claim

### 10. Batched Dispatch & Megakernel (Ch 13)
- GemvOp struct shape
- Megakernel 3-tier architecture
- ModelDesc in mega_compose.zig

### 11. Speculative Decoding (Ch 17)
- DDTree (Ringel & Romano, 2026) - verify reference
- 4 modes: draft-model, ddtree, self, ngram
- Default spec-tokens = 5, tree-budget = 64
- Self-spec default = 50% layers skipped
- N-gram: searches last 2048 tokens, n=3..10

### 12. MTP (Ch 18)
- MTP head architecture: hnorm, enorm, eh_proj, transformer block, shared head
- +1 offset norm
- 70-85% acceptance rate claim
- ~5-10% cost of full forward pass claim

### 13. Cross-cutting
- File paths referenced in tutorials match actual codebase
- Function/struct names match actual code
- CLI flags match cli.zig
- Default values match code defaults

## Method
1. Delegate to researcher subagents for parallel evidence gathering across source files
2. Verify specific claims against code with grep/read
3. Flag: mismatches, missing code, stale references, unverifiable performance claims
4. Produce single audit artifact at `outputs/tutorial-code-audit-audit.md`
