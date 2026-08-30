# Architecture

Project structure, module reference, and inference pipeline for the Agave inference engine.

**Tutorial:** [Getting Started](tutorial/00-getting-started.md)

For conceptual background, see the [tutorial series](tutorial/README.md).

## Quick Start

```bash
zig build                                          # Build (ReleaseFast + Debug)
./zig-out/bin/agave model.gguf                     # Interactive REPL
./zig-out/bin/agave model.gguf "What is 2+2?"      # Single prompt
./zig-out/bin/agave model.gguf --serve              # HTTP server (OpenAI + Anthropic API)
./zig-out/bin/agave model.gguf -q "Hello" > out.txt # Quiet mode (pipe-friendly)
./zig-out/bin/agave model.gguf --backend cpu        # Force CPU backend
./zig-out/bin/agave model.gguf --megakernel "Hi"    # Fused FFN megakernel (Metal/CUDA)
```

`zig build` produces three binaries:
- `zig-out/bin/agave`: ReleaseFast (optimized; size varies with enabled models/backends)
- `zig-out/bin/agave-debug`: ReleaseSafe (safety checks, leak detection)
- `zig-out/bin/agave-bench`: ReleaseFast micro-benchmark tool (`src/micro_bench.zig`)

## Project Structure

```
agave/
├── build.zig              # Build config (ReleaseFast default + Debug)
├── build.zig.zon          # Package metadata (zero external dependencies)
├── src/
│   ├── main.zig           # CLI: arg parsing, format detection, model init, REPL, recipe application
│   ├── cli.zig            # Self-contained CLI argument parser (zero deps)
│   ├── arch.zig           # Architecture enum, detection, chat template mapping
│   ├── pull.zig           # Model download from HuggingFace Hub (agave pull <org/repo>)
│   ├── server/
│   │   ├── server.zig     # HTTP server (OpenAI + Anthropic API + chat UI)
│   │   ├── scheduler.zig  # Continuous batching request scheduler
│   │   ├── tools.zig      # Process-level tool registry (register / dispose)
│   │   ├── metrics.zig    # Prometheus metrics collector
│   │   ├── rate_limiter.zig # Token bucket rate limiter
│   │   ├── json.zig        # JSON field extraction, encoding, and form-parsing
│   │   └── fixed_buf_stream.zig # Allocation-free fixed buffer writer (server responses)
│   ├── display.zig        # Rich CLI output (banner, stats, progress)
│   ├── chat_template.zig  # Data-driven chat prompt templates (ChatML, Gemma, Gemma 4, Qwen35, GLM-4, GPT-OSS, Llama 4)
│   ├── recipe.zig         # Optional preset configs per model/hardware/quant combo
│   ├── grammar.zig        # GBNF parser, JSON schema -> grammar converter, constrained decoding
│   ├── calibrate.zig      # TriAttention calibration subcommand (agave calibrate)
│   ├── steering.zig       # Directional steering (--dir-steering-file); activation projection
│   ├── eval.zig           # Token NLL scoring library (scoreCase; no --eval CLI yet)
│   ├── expert_profile.zig # MoE expert activation profiler (library; no CLI yet)
│   ├── expert_cache.zig   # SSD expert LRU streaming cache (--ssd-streaming CLI)
│   ├── image_tokens.zig   # Multimodal image placeholder token IDs (shared by arch + chat_template)
│   ├── test_exports.zig   # Test bridge re-exporting backend types for out-of-tree tests
│   ├── thread_pool.zig    # Futex-based work-stealing thread pool
│   ├── sim_clock.zig      # Injectable wall clock (deterministic tests / future sim harness)
│   ├── perf.zig           # Performance timer utilities
│   ├── readline.zig       # Line editor for interactive REPL
│   ├── term.zig           # Terminal I/O: key parser, ANSI sequences, display width (pure Zig, no libc)
│   ├── image.zig          # PNG/PPM image decoder and resize for multimodal inference
│   ├── wasm_entry.zig     # Browser inference entry point, GGUF parsing from buffer
│   ├── micro_bench.zig    # Standalone micro-benchmark binary
│   ├── fuzz_tests.zig     # Fuzz tests (tokenizer, grammar, JSON parser, quantization)
│   ├── format/
│   │   ├── format.zig     # Format interface (getTensor, getMetaStr, ...)
│   │   ├── gguf.zig       # GGUF v2/v3 parser with mmap
│   │   └── safetensors.zig# Multi-shard SafeTensors loader with config.json
│   ├── lora.zig           # LoRA adapter merge; Handle.dispose unmerges
│   ├── models/
│   │   ├── model.zig      # Model interface (forward, prefill, resetCache, cancel)
│   │   ├── gemma3.zig     # Gemma 3 (GQA, GELU, post-norms)
│   │   ├── gemma4.zig     # Gemma 4 (dual attention, MoE/dense variants, PLE)
│   │   ├── diffusion_gemma.zig # DiffusionGemma (block diffusion, bidirectional canvas)
│   │   ├── qwen35.zig     # Qwen 3.5 (hybrid DeltaNet SSM + attention)
│   │   ├── qwen4exp.zig   # Qwen 3.8 Flash-Next (HC, GDN sigmoid gate, QSA, n-gram PLE, 512-expert MoE)
│   │   ├── gpt_oss.zig    # GPT-OSS (MoE, sliding window, attention sinks)
│   │   ├── nemotron_h.zig # Nemotron-H (Mamba-2 + attention hybrid)
│   │   ├── glm4.zig       # GLM-4 MoE Lite (MLA (DeepSeek-V2) + MoE, MLX 4/6/8-bit)
│   │   ├── nemotron_nano.zig # Nemotron Nano (SSM + MoE + attention, NVFP4)
│   │   ├── deepseek4.zig    # DeepSeek V4 Flash (HC, MLA, CSA/HCA, LID; Vulkan/WebGPU GEMV shaders; CpuBackend for rms/SDPA/HC)
│   │   ├── llama4.zig       # Llama 4 (iRoPE, chunked attention, top-1 MoE)
│   │   └── vision.zig       # Vision encoder (SigLIP-2, SigLIP, Qwen VL) for multimodal models
│   ├── ops/
│   │   ├── attention.zig  # Shared SDPA kernel (SIMD, sliding window, backend dispatch)
│   │   ├── sparse_attn.zig # Block sparse attention (PFlash scorer path)
│   │   ├── math.zig       # argmax, GELU, sampling (top-k/p, min-p, XTC, Mirostat, DRY)
│   │   ├── sampler_stack.zig # Per-request logit interceptor stack (LIFO dispose)
│   │   ├── ssm.zig        # SSM ops: causal conv1d, Mamba-2 recurrence, group norm+gate
│   │   ├── quant.zig      # Quantization helpers (bf16, mxfp4, fp8, iq4nl, nvfp4_st)
│   │   ├── kv_quant.zig   # KV cache quantization (f32/f16/q8_0/int8/fp8/nvfp4/nvfp4_ds_mla/turbo/planar/iso/rotor)
│   │   ├── mlx.zig        # MLX 4/6/8-bit dequant (mlxGemvRaw, mlxGemvRows, mlxEmbLookup)
│   │   ├── gptq.zig       # GPTQ INT4 GEMV kernel (row-major packed u32, per-group scales/qzeros)
│   │   ├── awq.zig        # AWQ INT4 GEMV kernel (column-major, GEMM-order nibble interleave)
│   │   ├── hqq.zig        # HQQ 4-bit GEMV kernel (uint8 2-nibble, float meta.scale/meta.zero)
│   │   ├── kv_evict.zig   # KV eviction: norm-based scoring, cache compaction
│   │   └── split_attention.zig # Split-attention: async CPU-GPU KV cache offloading
│   ├── backend/
│   │   ├── backend.zig    # Backend interface (gemv, rmsNorm, softmax, ...)
│   │   ├── cpu.zig        # CPU: V8 SIMD, 4-row GEMV, precomputed RoPE
│   │   ├── metal.zig      # Metal: MSL kernels, simd_sum reduction, buffer cache
│   │   ├── vulkan.zig     # Vulkan: SPIR-V shaders, subgroup reductions, buffer cache
│   │   ├── cuda.zig       # CUDA: PTX kernels from Zig, deferred execution, Driver API
│   │   ├── rocm.zig       # ROCm: HIP Runtime API, HSACO kernels, deferred execution
│   │   ├── webgpu.zig     # WebGPU: WGSL shaders, browser + native (wgpu/Dawn)
│   │   ├── megakernel.zig # Weight offset computation for fused FFN megakernels
│   │   ├── mega_compose.zig # Composable megakernel generator (ModelDesc → MSL at runtime)
│   │   ├── accelerate.zig # Apple Accelerate.framework BLAS bindings (AMX-accelerated SGEMM)
│   │   ├── objc.zig       # Objective-C runtime bridge for Metal API
│   │   └── kernels/       # Kernel source files
│   │       ├── cpu/       # CPU SIMD kernels (gemv_*.zig, sdpa.zig, softmax.zig, norm.zig, rope.zig, ...)
│   │       ├── metal/     # MSL compute shaders (incl. megakernel.metal, mega_common.metal, mega_*.metal)
│   │       ├── vulkan/    # GLSL compute shaders → compiled SPIR-V (.spv)
│   │       ├── cuda/      # Zig kernels compiled to PTX (incl. fused_ffn_q8_0.zig, mega_*.zig)
│   │       ├── rocm/      # Zig kernels compiled to HSACO via amdgcn-amdhsa target (incl. mega_*.zig)
│   │       └── webgpu/    # WGSL compute shaders
│   ├── parallel/
│   │   ├── transport.zig  # Distributed transport: TCP, POSIX shm, NCCL (RoCE RDMA)
│   │   ├── tp.zig         # CPU tensor parallelism coordinator (rank-0 only; GPU TP uses NCCL via transport.zig)
│   │   └── peer_discovery.zig # UDP peer discovery (LAN broadcast, auto-connect; not devices/discovery)
│   ├── spec/
│   │   ├── spec_decode.zig # Speculative decoding orchestrator (draft, verify, accept)
│   │   ├── caps.zig       # Spec-mode provider table (named wait if unsatisfied)
│   │   ├── ddtree.zig     # DDTree tree construction (best-first heap, compile, walk)
│   │   ├── ngram.zig      # N-gram / suffix / lookahead (history-based, no draft model)
│   │   ├── pflash.zig     # PFlash speculative prefill (block scoring, alpha threshold)
│   │   └── dspark.zig     # DSpark confidence-scheduled verification (trim + SPS)
│   ├── devices/
│   │   └── discovery.zig  # Local GPU/CPU enumeration (--list-devices, --device N; not peer discovery)
│   ├── kvcache/
│   │   ├── manager.zig    # KV cache alloc/free, PagedKvCache, RadixTree
│   │   ├── block_allocator.zig # Block allocation for paged KV cache
│   │   ├── tiered.zig     # Tiered KV cache (VRAM + RAM + SSD)
│   │   ├── prefetch.zig   # Async block prefetching for tiered cache
│   │   └── checkpoint.zig # KV checkpoint header encode/validate (payload I/O not wired yet)
│   ├── web/
│   │   ├── app.ts         # Chat UI TypeScript (SSE streaming, conversation management)
│   │   ├── app.js         # Generated classic script; embedded by server.zig
│   │   ├── body.html      # Chat UI HTML body
│   │   ├── head.html      # Chat UI HTML head (meta, styles)
│   │   └── style.css      # Chat UI stylesheet
│   └── tokenizer/
│       ├── tokenizer.zig  # Tokenizer interface
│       └── bpe.zig        # BPE + SPM tokenizer with byte-level encoding
├── web/                   # Browser WASM shell (distinct from src/web server chat UI)
│   ├── index.html         # Standalone WASM demo page
│   ├── agave.ts           # Typed glue for agave.wasm (AgaveEngine)
│   ├── agave.js           # Generated classic script
│   ├── shell.ts           # Demo page logic
│   └── shell.js           # Generated classic script
```

## Design Decisions

Irreversible or high-cost choices. Rationale lives here so they are not re-litigated casually.

| Decision | Choice | Why | Revisit when |
|----------|--------|-----|--------------|
| Dependencies | Zero external ML or CLI libs; pure Zig + OS GPU APIs | Hot-path control, cross-compile, no ABI churn | A platform requires a vendor SDK that cannot be `dlopen`'d |
| Backend dispatch | Tagged union + `inline else` (not vtable) | Zero indirect-call cost on every GEMV/SDPA | Dynamic plugin backends become a hard requirement |
| Model dispatch | Comptime-generated vtable (`Model.from`) | Architectures differ too much for one tagged union; optional methods (EAGLE, MTP, SSM snapshot) need soft no-ops | VTable surface exceeds ~40 methods and most models leave half unused |
| Quantization | Dequant inside kernels; no full f32 weight materialization on hot path | Bandwidth-bound decode; full dequant would dominate | A backend cannot express in-kernel dequant for a new format |
| Weight I/O | mmap GGUF (SafeTensors secondary); no full materialize at load | UMA zero-copy; peak RSS ≈ working set | Discrete-GPU direct-to-VRAM (`cuFile`) becomes the common path |
| KV memory model | Paged blocks + optional tiered VRAM/RAM/SSD | Prefix sharing, preemption, and demotion need non-contiguous layout | A target requires fully contiguous device-resident KV only |
| HTTP surface | OpenAI + Anthropic shapes on one server | Clients already speak those protocols; one binary | A third incompatible protocol becomes a first-class requirement |
| Scheduling | Continuous batching scheduler; admission serialized to one request at a time until per-request paged KV is wired | The model layer exposes a single shared KV sequence (scalar `kv_seq_len`, one `seq_table` row); interleaving requests corrupts attention state silently. Serialization matches vLLM-class *interfaces* while keeping output correct | `Request.block_table` is plumbed through the model vtable as a per-request sequence row (raise `scheduler.max_running_requests_single_sequence`) |
| Spec CLI aliases | Normalize at parse (`medusa` → `mtp`); domain enum has no synonyms | Call sites must not re-branch on marketing names | A “alias” gains a divergent inference path |
| Wall clock | `sim_clock` for server/scheduler/rate-limiter/tiered KV; MONOTONIC for interval timers (`perf`, `pull`, benches) | One injectable clock for deterministic timeout/refill tests; MONOTONIC avoids NTP skew in elapsed timing | Multi-threaded tests need per-thread virtual clocks |
| Device discovery `BackendKind` | `cpu/metal/cuda/rocm/vulkan` only (no `webgpu`) | `--list-devices` / TP-PP target discrete GPUs; WebGPU is a single logical adapter (browser or wgpu), not multi-device topology | WebGPU multi-adapter or peer groups become real |
| Parallel transport topology | Fixed 2-rank pair (`rank 0 ↔ 1`); CLI rejects `--tp > 2` and `--pp > 2` | SHM region names, `tcp_fds[0]`, and `sendBuf` peer encoding are pair-shaped; multi-rank ring/tree not built | Ring/tree all-reduce and multi-stage PP ship |
| Server sleep mode | Flag in `/health` only; weights stay resident | Orchestrators need an idle signal without cold-start latency | Memory pressure requires actual weight unload / sleep-to-disk |
| GPU missing kernels | `@panic` (fail closed), except documented cases (`embLookup`, small Metal softmax) | Silent CPU fallback hides broken builds and destroys latency | A new op is proven faster on CPU on UMA (must comment why) |
| `max_tokens` cap | Tied to `gen_ids_buf_size` (4096) | Generation ID buffer cannot hold more tokens than the clamp | Streaming without a fixed ID buffer needs a higher cap |
| HTTP KV prefix blob | Unversioned f32 layout: `layer0_K\|layer0_V\|…` (not `checkpoint.KVC`) | Hot path for LMCache-style fleet transfer; uniform-dim checkpoint header cannot express dual-attn / MLA per-layer `kvd` | Wire format gains magic/version + token IDs for safe API prefix reuse |
| KV export implementors | Soft vtable stubs; only Gemma4 implements today | Avoid forcing every arch to stub; 501 when unsupported | A second architecture needs fleet KV transfer |
| KV import vs prefix cache | Import clears `cached_prompt_ids`, sets `kv_valid` | Blob carries no token IDs; keeping old IDs would lie to `/info` and prefix matching | Blob (or sidecar) includes prompt token IDs so API `reset=true` can skip re-prefill |

## The Inference Pipeline

When you run `agave model.gguf "Hello"`:

```
1. LOAD        model.gguf → mmap → Format interface
2. DETECT      "general.architecture" = "gemma3" → Gemma3Model
3. BACKEND     macOS → Metal GPU (auto), --backend cpu → CPU fallback
4. RECIPE      Match arch + backend + quant → apply proven defaults
5. TEMPLATE    arch → ChatTemplate → format prompt with role markers
6. TOKENIZE    formatted prompt → [BOS, 15496, ...] (BPE/SPM encode)
7. PREFILL     model.prefill(prompt_tokens) → fills KV cache (batched)
8. GENERATE    Loop: next = model.forward(last) → sample/argmax → decode → print
9. STATS       "5 tok, 10.4 tok/s, prefill 200ms, gen 480ms"
```

## Module Reference

### Format (`src/format/`)

| Method | Description |
|--------|-------------|
| `getTensor(name)` | Look up tensor by name → `{data_ptr, dtype, dims}` |
| `getMetaStr(key)` | String metadata (architecture name, model name) |
| `getMetaU32(key)` | Integer metadata (num_layers, hidden_size) |
| `getMetaF32(key)` | Float metadata (rope_theta, rms_norm_eps) |
| `getVocab()` | Tokenizer vocabulary array |
| `getMerges()` | BPE merge rules array |
| `layerTensor(li, suffix)` | Shorthand for `getTensor("blk.{li}.{suffix}")` |

### Backend (`src/backend/`)

| Operation | Description | Hot path? |
|-----------|-------------|-----------|
| **Core** | | |
| `gemv(x, W, y, n, k)` | y = W @ x with dequantization | Yes (95% of time) |
| `gemm(x, W, y, n_tok, n_out, n_in)` | Batched matrix multiply (prefill, BF16 on Metal) | Prefill only |
| `rmsNorm(in, w, out, n, eps)` | RMS normalization | Yes |
| `sdpa(q, keys, vals, ...)` | Scaled dot-product attention | Yes |
| `softmax(data, n)` | In-place softmax | Yes |
| `rope(x, pos, nh, hd, rd, θ)` | Rotary position encoding | Yes |
| `silu(in, out, n)` | SiLU activation | Yes |
| `gelu(in, out, n)` | GELU activation | Yes |
| `add(a, b, out, n)` | Element-wise add | Yes |
| `mul(a, b, out, n)` | Element-wise multiply | Yes |
| `l2Norm(x, n, eps)` | L2 normalization (DeltaNet) | Yes |
| `embLookup(table, id, out, d)` | Embedding with dequant | Once per token |
| **Fused** | | |
| `addRmsNorm(a, b, w, out, n, eps)` | Fused add + RMS norm | Yes |
| `siluMul(a, b, out, n)` | Fused SiLU(a) × b (SwiGLU gate) | Yes |
| `geluMul(a, b, out, n)` | Fused GELU(a) × b (Gemma FFN) | Yes |
| `sigmoidMul(data, gate, n)` | In-place data × sigmoid(gate) | Yes |
| `addScaled(src, dst, scale, n)` | dst += src × scale (MoE accumulation) | Yes |
| **Batched (prefill)** | | |
| `rmsNormBatched(in, w, out, n_tok, dim, eps)` | Per-row RMS norm for n_tok rows | Prefill only |
| `ropeBatched(x, positions, n_tok, ...)` | RoPE for n_tok vectors | Prefill only |
| `sdpaPrefill(q, k, v, ...)` | Causal self-attention for n_tok tokens | Prefill only |
| `gemvMulti(x, ops, k)` | Batched GEMV dispatch (fused kernel launch) | Yes |
| **Specialized** | | |
| `gemvT(x, W, y, out_dim, in_dim)` | Transposed GEMV for Q8_0 3D weights (MLA) | Yes |
| `gemvNvfp4St(x, w, scale, y, n, k)` | NVFP4 SafeTensors GEMV (separate scale tensor) | Yes |
| `gemvMlxQ(x, w, scales, biases, y, n, k, bits, group_size)` | MLX affine quantized GEMV (2/4/6/8-bit, variable group_size) | Yes |
| `gemvMxfp4St(x, w, scale, y, n, k)` | MXFP4 SafeTensors GEMV | Yes |
| `rmsNormMulti(data, w, n_heads, hd, eps)` | Per-head RMS norm (QK norm) | Yes |
| `deinterleave(in, out_a, out_b, stride, n)` | Split interleaved Q/K pairs | Yes |
| `splitQGate(qg, q, g, hd, nh)` | Split concatenated Q+gate (Qwen3.5) | Yes |
| `deltaNet(...)` | DeltaNet SSM recurrence | Yes |
| `sdpaWithStats(q, keys, vals, ..., max, sum)` | SDPA returning softmax stats (split-attention) | Yes |
| `sdpaPaged(q, page_table, kv_pool, ...)` | Paged SDPA with block table indirection (256-token blocks) | Yes |
| **Infrastructure** | | |
| `sync()` | Flush GPU work | At sync points |
| `beginBatch()` / `endBatch()` | Suppress/restore GPU memory barriers | GPU only |
| `backendInfo()` | Device name, VRAM, library version | Init only |

### Chat Templates (`src/chat_template.zig`)

| Preset | Models | EOG Tokens | Notes |
|--------|--------|------------|-------|
| `chatml` | Nemotron-H, Nemotron-Nano | `<\|im_end\|>`, `<\|endoftext\|>` | Standard ChatML |
| `qwen35` | Qwen 3.5 | `<\|im_end\|>`, `<\|endoftext\|>` | ChatML + `<think>\n\n</think>\n\n` generation prefix (disables reasoning) |
| `gemma` | Gemma 3, Gemma 2 | `<end_of_turn>`, `<eos>` | |
| `gemma4` | Gemma 4 | `<turn\|>`, `<eos>`, `<channel\|>`, `<\|endoftext\|>`, `<\|end\|>` | `<\|channel>0\n<channel\|>` generation prefix |
| `glm4` | GLM-4 | `<\|endoftext\|>`, `<\|user\|>`, `<\|observation\|>` | `[gMASK]<sop>` prefix, no generation prefix |
| `gpt_oss` | GPT-OSS Harmony | `<\|end\|>`, `<\|endoftext\|>` | Includes default system prompt + developer role override |
| `llama4` | Llama 4 | `<\|eot\|>`, `<\|end_of_text\|>` | Default system prompt |
| `deepseek4` | DeepSeek V4 | `<｜end▁of▁sentence｜>` | BOS prefix, `</think>` generation prefix |

### Recipes (`src/recipe.zig`)

| Recipe | Arch | Backend | Key Defaults |
|--------|------|---------|--------------|
| Qwen3.5 Q4 Metal | qwen3* | Metal | temp=0.6, top_p=0.9, repeat=1.1 |
| Gemma Q4 Metal | gemma* | Metal | temp=0.7, top_p=0.95 |
| GPT-OSS Metal | gpt* | Metal | temp=0.5, ctx=2048 |
| GLM-4 generic | glm4* | any | temp=0.7, repeat=1.1 |
| CPU generic | any | CPU | max_tokens=256, ctx=2048 |

User CLI flags always override recipe defaults.

### Server (`src/server/`)

HTTP server activated via `--serve` (default port 49453, override with `--port`). Provides a full OpenAI-compatible API plus health and metrics endpoints.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming SSE or batch) |
| `/v1/completions` | POST | Text completions |
| `/v1/responses` | POST | Responses API |
| `/v1/messages` | POST | Anthropic Messages API format |
| `/v1/models` | GET | List available models |
| `/v1/chat/regenerate` | POST | Regenerate last assistant response |
| `/v1/conversations` | GET/POST | List or create conversations |
| `/v1/tokenize` | POST | Count tokens in text |
| `/v1/detokenize` | POST | Detokenize token IDs to text |
| `/v1/chat` | POST | Built-in chat web UI endpoint |
| `/v1/embeddings` | POST | Text embeddings (501 stub) |
| `/v1/kv_cache` | GET/POST | KV prefix export/import |
| `/v1/kv_cache/info` | GET | KV cache metadata |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check (model loaded) |
| `/metrics` | GET | Prometheus metrics (tokens/s, latency, queue depth) |

**Sampling parameters** (accepted in chat/completions request body):
- `temperature` -- sampling temperature (0 = greedy)
- `top_k` -- top-K filtering
- `top_p` -- nucleus sampling threshold
- `min_p` -- minimum probability cutoff
- `frequency_penalty` -- penalize tokens by frequency in generated text
- `presence_penalty` -- penalize tokens already present in generated text
- `repetition_penalty` -- multiplicative repetition penalty
- `seed` -- deterministic sampling seed
- `stop` -- stop sequences (string or array)
- `xtc_probability` -- XTC sampling probability
- `xtc_threshold` -- XTC sampling threshold
- `dry_multiplier` -- DRY n-gram repetition penalty multiplier (0 = disabled)
- `dry_allowed_length` -- DRY minimum n-gram length before penalty applies
- `mirostat` -- Mirostat mode (0 = disabled, 2 = Mirostat 2.0)
- `mirostat_tau` -- Mirostat target entropy
- `mirostat_eta` -- Mirostat learning rate
- `logit_bias` -- Token ID to bias map (e.g. `{"123": 5.0}`)
- `logprobs` -- Return log probabilities of output tokens
- `top_logprobs` -- Number of top log probabilities to return per token
- `grammar` -- GBNF grammar string for constrained decoding
- `json_schema` -- JSON Schema object for structured output (converted to GBNF internally)
- `tools` -- Array of tool/function definitions (OpenAI-compatible)
- `tool_choice` -- Tool selection mode (`"auto"`, `"required"`, `"none"`)

**Architecture**: `server.zig` handles HTTP parsing and routing, `scheduler.zig` implements continuous batching for concurrent requests, `rate_limiter.zig` provides token-bucket rate limiting, and `metrics.zig` collects Prometheus-format telemetry. A built-in chat UI is served from `src/web/`.

**Tool/function calling**: OpenAI-compatible tool use is supported via system prompt injection and output parsing. When `tools` are provided in a request, tool definitions are injected into the system prompt and the model's output is parsed for `<tool_call>` tags. Multiple tool calls per response are supported. Controlled by the `tool_choice` parameter (`"auto"`, `"required"`, `"none"`).

### Shared Ops (`src/ops/`)

| Function | File | Description |
|----------|------|-------------|
| `scaledDotProductAttention` | attention.zig | Full SDPA with KV cache, GQA, sliding window |
| `sampleToken` | math.zig | Temperature + top-k + top-p nucleus sampling |
| `Stack.apply` | sampler_stack.zig | Ordered logit interceptors; `dispose` clears LIFO |
| `causalConv1dSilu` | ssm.zig | Causal conv1d with ring buffer + SiLU |
| `mamba2Recurrence` | ssm.zig | Mamba-2 per-head state update + output |
| `groupRmsNormSiluGate` | ssm.zig | Group RMS norm followed by SiLU gate |
| `expertWeightStride` | model.zig | Byte stride between experts in packed weights |

### Grammar (`src/grammar.zig`)

Constrained decoding via GBNF grammars and JSON schema. The module provides:

| Component | Description |
|-----------|-------------|
| GBNF Parser | Parses GBNF grammar strings into rule sets with alternation, repetition, and character classes |
| JSON Schema Converter | Converts JSON Schema objects into equivalent GBNF grammars for structured output |
| Decoding State Machine | Tracks valid next-token sets during generation, masking logits to enforce grammar constraints |

Activated via `--grammar <file.gbnf>` or `--json-schema <schema>` CLI flags, or via the server API's `grammar` / `json_schema` sampling parameters.

### WASM (`src/wasm_entry.zig`)

Browser inference entry point for running Agave in WebAssembly environments. Provides GGUF parsing from an in-memory buffer and connects to the WebGPU backend for GPU-accelerated inference in the browser.

### Speculative Decoding (`src/spec/`)

Agave supports 14 speculative decoding modes via `--spec-mode` (including `auto`):

| Module | Description |
|--------|-------------|
| `spec_decode.zig` | Orchestrator: all draft/verify modes, adaptive K, FR-Spec masking, EAGLE/MLP/Lookahead drafting, DSpark confidence trim |
| `caps.zig` | Per-mode provider requirements; missing provider is a named wait, not a crash in `forward_tree` |
| `ddtree.zig` | DDTree tree construction: best-first heap, compile, acceptance walk |
| `ngram.zig` | N-gram, SharedNgramPool (server cross-request), SuffixState (10k cache), LookaheadState (Jacobi) |
| `pflash.zig` | PFlash speculative prefill: block scoring, alpha-threshold selection, compressed prefill |
| `dspark.zig` | DSpark: confidence-scheduled verification, hardware-aware prefix scheduler, Markov/RNN sequential head, SPS profiling |

| Backend Kernel | Description |
|----------------|-------------|
| `sdpa_tree.zig` | Tree-masked SDPA: ancestor bitmask attention for tree verification |

**Mode summary:**

| `--spec-mode` | Draft source | Draft model required? |
|---|---|---|
| `auto` | DDTree with `--draft-model`, else n-gram | Conditional |
| `standard` | Separate draft model, greedy | Optional (self-draft) |
| `ddtree` | Separate draft model, tree-based | Optional (self-draft) |
| `self` | Target model with layer skip | No |
| `ngram` | Output history ring buffer | No |
| `suffix` | Cross-request suffix cache (10k tokens) | No |
| `lookahead` | Jacobi parallel branch exploration | No |
| `mtp` | Built-in MTP prediction heads | MTP heads (`waiting for mtp` if missing) |
| `medusa` | Built-in Medusa MLP heads (MTP alias) | MTP heads |
| `eagle` | Hidden-state conditioned draft (chained) | Yes (`waiting for draft`) |
| `eagle3` | Pre-output-norm hidden-state conditioned draft | Yes (`waiting for draft`) |
| `mlp` | Hidden-state conditioned draft (frozen) | Yes (`waiting for draft`) |
| `pflash` | Block-scored speculative prefill | Yes (`waiting for draft`) |
| `dspark` | Confidence-scheduled verification (any drafter) | Optional |

EAGLE uses `get_hidden_state` + `eagle_forward` vtable methods on the target model to extract last residual hidden state and feed it to the draft model. FR-Spec (`--spec-token-map`) restricts draft logits to a frequency-ranked token subset.

DSpark (Cheng et al., 2026) applies confidence-scheduled verification on top of any existing drafter. After drafting, `dsparkTrimDraft()` uses per-position acceptance history as a survival-probability proxy and trims the verification block to tokens with positive expected return. In server mode, the full `scheduleVerification()` (Algorithm 1) jointly optimises all concurrent requests against a pre-profiled `SpsProfile` throughput curve. The `dspark.zig` module also implements the Markov head (`B(x_{k-1},·) = W1[x_{k-1}]W2`, low-rank V×V), RNN head (gated recurrent sequential stage), and confidence head (`c_k = σ(w^T [h_k; W1[x_{k-1}]])`), ready for inference when a trained DSpark checkpoint is loaded.

### Block Sparse Attention (`src/ops/sparse_attn.zig`)

BigBird-style block sparsity for long-context inference. Reduces attention complexity from O(n²) to O(n) by computing QK dot products only for attended block pairs.

**Sparsity pattern:**

```
Blocks:   [0] [1] [2] [3] [4] [5] [6] [7]
Query 0:  G   G   W               W         G = global block (attends all)
Query 1:  G   G   W   W   W                 W = sliding window block
Query 2:  G   G   W   W   W   W             . = masked (not computed)
Query 3:  G   G       W   W   W   W
Query 4:  G   G           W   W   W   W
```

Two components determine which blocks are computed:

- **Global blocks**: the first N blocks attend to and are attended by every other block. They capture long-range information (BOS, task prefix, system prompt).
- **Sliding window**: each block attends to the ±window blocks around it, preserving local context without O(n²) cost.

The CPU SDPA kernel in `sparse_attn.zig` iterates over query blocks and skips the inner KV loop entirely for masked block pairs. For a 128K-token sequence with block size 64 and window 2, this reduces dot-product work by roughly 98%.

### PFlash: Speculative Prefill (`src/spec/pflash.zig`)

PFlash accelerates prefill for long prompts (128K+ tokens) by having a cheap scorer model identify which KV blocks carry the most information. Only those blocks are forwarded through the full target model.

**Pipeline:**

```
Prompt (128K tokens)
    |
    v
Scorer model runs forward pass
(block-sparse attention, O(n) cost)
    |
    v
Score each KV block: [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.85, ...]
    |
    v
Adaptive threshold: keep block if score > alpha * mean(scores)
[    ##       ##        ##  ##  ]  <- selected (~5-15% of blocks)
    |
    v
Target model prefills compressed prompt (~6-13K tokens)
    |
    v
DDTree speculative decode -> output tokens
```

**Adaptive PFlash** uses a data-dependent threshold: `alpha * mean(block_scores)` rather than a fixed top-K count. This adapts to prompt structure -- dense technical content selects more blocks than sparse narrative text -- and avoids the need to tune K per prompt length.

**Composability:** PFlash handles prefill; DDTree handles decode. They compose naturally: PFlash fills the KV cache with the compressed prompt, then the normal DDTree speculative decode loop takes over for generation. Use `--spec-mode pflash` to activate both.

**Scorer model:** By default, the `--draft-model` is used as the scorer. For highest throughput, pass a separate, smaller `--pflash-scorer` model. The scorer only needs to identify important blocks -- it does not need to produce high-quality token predictions.

**CLI parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--spec-mode pflash` | -- | Activate PFlash prefill (requires `--draft-model`) |
| `--pflash-alpha` | 0.85 | Block selection threshold multiplier |
| `--pflash-block-size` | 64 | Block size in tokens |
| `--pflash-scorer` | (draft model) | Separate model for block scoring |

### Distributed Inference (`src/parallel/`)

| Module | Description |
|--------|-------------|
| `transport.zig` | Transport layer: TCP (cross-node), POSIX shm (same-node zero-copy), NCCL (GPU-optimized RoCE RDMA) |

**Modes**: Tensor Parallelism (`--tp N`), Pipeline Parallelism (`--pp N`), Hybrid TP+PP, Disaggregated Prefill/Decode (`--disagg`). Transport auto-selects shm for localhost, tcp otherwise; NCCL via `--transport nccl`. NCCL loaded at runtime via `dlopen("libnccl.so.2")`, no compile-time dependencies. Device pointer allReduceAdd passes GPU activation cache pointers directly to NCCL when data is dirty on device. See [PARALLELISM.md](PARALLELISM.md).

### Quantization Types

| DType | Bits/val | Block | Models |
|-------|----------|-------|--------|
| `f32` | 32 | 1 | Reference |
| `f16` | 16 | 1 | Embeddings |
| `bf16` | 16 | 1 | Gemma3, Nemotron SSM layers |
| `q8_0` | 8.5 | 32 | General |
| `q6_k` | 6.6 | 256 | General |
| `q5_k` | 5.5 | 256 | General |
| `q4_k` | 4.8 | 256 | General |
| `q5_0` | 5.5 | 32 | Nemotron-H |
| `q4_0` | 4.5 | 32 | General |
| `q4_1` | 5.0 | 32 | General |
| `q3_k` | 3.4 | 256 | Compact |
| `q2_k` | 2.6 | 256 | Ultra-compact |
| `iq4_nl` | 4.5 | 32 | CPU-optimized (lookup table) |
| `iq4_xs` | 4.3 | 256 | CPU-optimized (super-block) |
| `fp8_e4m3` | 8 | 1 | KV cache, weights |
| `fp8_e5m2` | 8 | 1 | Weights only |
| `nvfp4` | 4.25 | 16 | Blackwell+ (GGUF) |
| `mxfp4` | 4.25 | 32 | Microscaled FP4 |
| `tq1_0` | 1.7 | 256 | Ternary {-1,0,+1}, base-3 packed (5 trits/byte), all 6 backends |
| `tq2_0` | 2.0 | 256 | Ternary {-1,0,+1}, 2-bit packed (4 values/byte), all 6 backends |
| `mlx_q` | 4-8 | 64 | MLX models (affine: scale × uint + bias) |
| `gptq` | 4.25 | 32-128 | GPTQ INT4 (row-major packed u32, per-group scales/qzeros) |
| `awq` | 4.25 | 32-128 | AWQ INT4 (column-major packed u32, GEMM-order interleave [0,2,4,6,1,3,5,7]) |
| `hqq` | 4.0 | 64 | HQQ INT4 (uint8 2-nibble packed, float meta.scale/meta.zero, CPU only) |

**KV Cache Quantization Types** (see `src/ops/kv_quant.zig`):

| KvQuantType | Bits/val | Notes |
|-------------|----------|-------|
| `f32` | 32 | Full precision |
| `f16` | 16 | Half precision |
| `q8_0` | 8.5 | Block-quantized |
| `int8` | 8 | Symmetric INT8 |
| `fp8_e4m3` | 8 | FP8 E4M3 |
| `nvfp4` | 4.25 | NVFP4 microscaled |
| `nvfp4_ds_mla` | 5.94 | DeepSeek MLA: NVFP4 on 448 NoPE dims, f16 on 64 RoPE dims (380 B/token) |
| `turbo2` | 2.5 | TurboQuant 2-bit (WHT-32 + Lloyd-Max codebook) |
| `turbo3` | 3.5 | TurboQuant 3-bit (WHT-32 + Lloyd-Max codebook) |
| `turbo4` | 4.5 | TurboQuant 4-bit (WHT-32 + Lloyd-Max codebook) |
| `pq2` | 2.5 | PlanarQuant 2-bit (Givens 2D rotation, 256 FMAs per block) |
| `pq3` | 3.5 | PlanarQuant 3-bit (Givens 2D rotation, 256 FMAs per block) |
| `pq4` | 4.5 | PlanarQuant 4-bit (Givens 2D rotation, 256 FMAs per block) |
| `iq2` | 2.5 | IsoQuant 2-bit (quaternion 4D rotation, 512 FMAs per block) |
| `iq3` | 3.5 | IsoQuant 3-bit (quaternion 4D rotation, 512 FMAs per block) |
| `iq4` | 4.5 | IsoQuant 4-bit (quaternion 4D rotation, 512 FMAs per block) |
| `rq2` | 2.5 | RotorQuant 2-bit (Cl(3,0) rotor 3D rotation, ~2400 FMAs per block) |
| `rq3` | 3.5 | RotorQuant 3-bit (Cl(3,0) rotor 3D rotation, ~2400 FMAs per block) |
| `rq4` | 4.5 | RotorQuant 4-bit (Cl(3,0) rotor 3D rotation, ~2400 FMAs per block) |

**TurboQuant+ features:**
- **`turbo` preset** (`--kv-type turbo`): asymmetric K=q8\_0, V=turbo4. Higher K precision protects attention routing accuracy while V compression (3.8x) is nearly free.
- **Boundary V protection**: the turbo preset automatically keeps first/last 2 layers at f16 V to preserve input embedding fidelity and final output quality.
- **Sparse V dequantization**: SDPA kernels dequantize V blocks on-the-fly inside the attention loop rather than pre-expanding, saving memory bandwidth.
- **Native GPU SDPA kernels**: TurboQuant dequantization is fused into SDPA kernels on all GPU backends (Metal, CUDA, Vulkan, ROCm).

**Tiered KV cache and split-attention** (`src/ops/split_attention.zig`):
- Enabled via `--kv-tiers vram+ram`. KV cache blocks are partitioned by tier (GPU-resident VRAM vs CPU-resident RAM).
- **Fast path**: when all blocks are on GPU, dispatches a single `be.sdpa()` with zero overhead.
- **Mixed path**: GPU SDPA with softmax statistics runs concurrently with CPU SDPA on the thread pool, then partial outputs are merged via [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) online softmax correction (exact, no approximation).
- **CPU-only path**: falls back to CPU SDPA on the thread pool when all blocks have been offloaded.

**Paged KV cache and paged SDPA** (`src/kvcache/manager.zig`, `src/backend/kernels/cpu/sdpa.zig`):
- KV cache is organized into 256-token blocks managed by `PagedKvCache` with `RadixTree` prefix sharing and `BlockAllocator` for efficient allocation.
- `PagedKvView` provides block table indirection, translating logical token positions to physical block locations.
- `sdpaPagedHeads` computes attention over paged blocks with thread-pool parallelism across heads.
- Every backend implements `sdpaPaged()` natively. GPU paths gather scattered host blocks into a flat staging buffer, then run a GPU paged SDPA kernel (no silent CPU compute fallback). Staging gather is host-side by design: the paged pool lives in CPU-visible memory for prefix sharing and tier demotion.

### KV Cache Eviction

When context grows beyond `--kv-budget`, eviction compresses the cache in-place to stay within budget. Two policies are available:

| Policy | Flag | Calibration | Description |
|--------|------|:-----------:|-------------|
| **Norm** | `--kv-eviction norm` | No | Evicts entries with the smallest K vector L2 norm |
| **Tri** | `--kv-eviction tri` | Yes (`.cal` file) | Trigonometric frequency-domain scoring from [TriAttention (Mao et al., 2025)](https://arxiv.org/abs/2604.04921) |

**Shared behavior:**
- **Attention sink preservation**: the first 4 positions are never evicted (they accumulate disproportionate attention mass in causal models).
- **Recent window**: the most recent positions are always retained regardless of score.
- **Periodic compression**: eviction runs every 128 tokens once the cache exceeds `--kv-budget`.

**Calibration (`agave calibrate`):** The `tri` policy requires per-head Q/K frequency statistics stored in a `.cal` file alongside the model. Run `agave calibrate model.gguf` to generate this data. The calibration pass processes a representative prompt and records the dominant frequency components per attention head.

**Stacking with TurboQuant:** Eviction reduces the *number* of KV entries while TurboQuant reduces the *bits per entry*. Combined, they can achieve ~40x KV memory reduction vs f16 baseline.

## Vision / Multimodal

Vision support is implemented in `src/models/vision.zig` with three auto-detected encoder variants loaded from mmproj GGUF files. The encoder variant is detected at init by probing available tensor names.

### Encoder Architectures

**Gemma 4 [SigLIP-2 (Tschannen et al., 2025)](https://arxiv.org/abs/2502.14786)** (`gemma4_siglip2`):
- 768x768 input, 16x16 patches -> 2304 patches, 3x3 average pooling -> 256 output tokens.
- Conv2D patch embedding (no bias), learned 2D position encoding `[embd_dim, max_pos, 2]`.
- ViT blocks with per-head QK RMSNorm, post-attention/FFN RMSNorm, SwiGLU FFN.
- Input standardization (`scale * x + bias`, replaces CLIP mean/std normalization).
- Single linear projection (`mm.input_projection.weight`) to LLM hidden dimension.

**Gemma 3 [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)** (`gemma3_siglip`):
- 896x896 input, 14x14 patches -> 4096 patches, no spatial merge.
- Conv2D patch embedding (with bias), learned 1D position embedding `[embd_dim, n_patches]`.
- ViT blocks with LayerNorm (with bias), GELU FFN (up+down, no gate), no QK norms.
- Post-encoder LayerNorm (`v.post_ln`), then `mm.soft_emb_norm` + `mm.input_projection`.

**Qwen VL** (`qwen_vl`):
- Dual Conv2D patch embedding (with bias), learned 1D position embedding.
- ViT blocks with fused QKV projection, LayerNorm (with bias), GELU FFN, no QK norms.
- Post-encoder LayerNorm (`v.post_ln`), then 4x MLP merge projector (`mm.0` + GELU + `mm.2`) -> n\_patches/4 output tokens.
- M-RoPE (multi-dimensional rotary position embedding): 4 sections `[temporal, height, height, width]` with theta=10000.

### Vision Pipeline

```
1. PREPROCESS    decode image -> resize to encoder input size -> normalize pixels
2. PATCH EMBED   Conv2D: [H, W, 3] -> [n_patches, embd_dim]
3. POS EMBED     add learned position embeddings (1D or 2D depending on variant)
4. VIT BLOCKS    N transformer blocks: LayerNorm/RMSNorm -> SDPA -> FFN -> residuals
5. POOL          spatial merge (Gemma 4: 3x3 avg pool, Qwen VL: 4x MLP, Gemma 3: none)
6. STANDARDIZE   input standardization (Gemma 4 only: scale * x + bias)
7. PROJECT       linear projection to LLM hidden dimension
8. NORMALIZE     RMSNorm / soft_emb_norm on projected tokens
```

### LLM Integration

Vision tokens are injected into the LLM via `forwardImageBatch` (Gemma 4). Image embeddings replace placeholder tokens in the input sequence, then the model runs a dedicated forward pass over the image batch using **non-causal (bidirectional) attention** -- image tokens attend to all other image tokens without the causal mask. After this pass, the KV cache contains the image context and subsequent text tokens attend to it normally via causal attention.
