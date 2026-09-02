<p align="center">
  <img src="docs/logo.svg" alt="Agave" width="480">
</p>

<p align="center">
  A high-performance LLM inference engine written in Zig.<br>
  Zero external ML libraries, all kernels, quantization, and model logic from scratch.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="docs/CONTRIBUTING.md">Contributing</a> •
  <a href="docs/DOCUMENTATION.md">Docs</a>
</p>

---

## Why Agave

The usual way to run a model locally is a C++ engine with a large dependency
graph: BLAS, a vendor math library per GPU, a build system that has to find all
of them. Agave has none. Every kernel, quantizer, tokenizer and model is written
here, in Zig, and the only thing you need to build it is a Zig compiler.

That buys two things. Cross-compiling to another OS or CPU is one flag, because
there is no native toolchain to satisfy on the other side. And a quantization
format or a new architecture can be added without negotiating with an upstream
tensor library, which is why the backend and quant matrices below are as wide as
they are.

The cost is honest: this is a 0.x project, several backends still have
correctness gaps (see [Benchmarks](#benchmarks) and
[docs/TEST_MATRIX.md](docs/TEST_MATRIX.md)), and llama.cpp supports far more
architectures. Use Agave if you want a readable, dependency-free engine to build
on. Use llama.cpp if you want maximum model coverage today.

## It Works

```console
$ ./zig-out/bin/agave qwen2.5-1.5b-instruct-q4k.gguf --backend cpu -n 60 --seed 42 \
    "Explain what a KV cache is, in two sentences."
agave qwen2.5-1.5b-instruct · Qwen 3.5/3.8 · Q4_K · 1.0GB · CPU
system: Linux 7.2.0-1-cachyos (x86_64) · CPU · AMD Ryzen 9 9950X 16-Core Processor · 32 threads
loading 1.0 GB... done (39ms)
recipe: CPU generic
context: 2048 (model supports 32768, use --ctx-size to increase)
loaded: GGUF v3 · 339 tensors · bpe tokenizer · 151K vocab · eos=151645 bos=151643 · qwen35 template

A KV cache is a type of data storage system that stores key-value pairs, allowing for quick retrieval of data.

23 tok · 4.6 tok/s · 4810ms prefill
```

## Features

- **11 Model Architectures**: Gemma 3, Gemma 4, DiffusionGemma, Qwen 3.5, Qwen4-Exp, GPT-OSS, Nemotron-H, Nemotron Nano, GLM-4, DeepSeek V4, Llama 4 (plus DFlash2 as a block-diffusion drafter, not a chat model)
- **6 Backends**: CPU (SIMD-optimized, Accelerate.framework on macOS), Metal GPU (Apple Silicon), Vulkan, CUDA, ROCm, WebGPU, individually toggleable at build time
- **Compile-Time Model Selection**: Disable unused model architectures to reduce binary size
- **2 Formats**: GGUF, SafeTensors (multi-shard, MLX quantized, NVFP4)
- **20+ Quantization Types**: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, TQ1_0, IQ4_XS, IQ4_NL, FP8 E4M3, FP8 E5M2, NVFP4, MXFP4, MLX 4/6/8-bit, GPTQ
- **19 KV Cache Quantization Types**: F32, F16, Q8_0, INT8, FP8, NVFP4, NVFP4-MLA, TurboQuant 2/3/4-bit, PlanarQuant 2/3/4-bit, IsoQuant 2/3/4-bit, RotorQuant 2/3/4-bit, with asymmetric K/V support and paged SDPA
- **Tiered KV Cache**: VRAM + RAM + SSD offloading with async prefetch (`--kv-tiers vram+ram+ssd`)
- **Chat Templates**: Data-driven per-architecture prompt formatting (ChatML, Gemma, Gemma 4, Qwen 3.5, GLM-4, GPT-OSS, Llama 4)
- **Recipes**: Optional proven-default configs per model/hardware/quant combo
- **Model Download**: `agave pull <org/repo>`, download GGUF models from HuggingFace Hub with auto quant selection
- **Interactive REPL**: Multi-turn chat with `/help`, `/clear`, `/stats`, `/model`, `/quit`
- **HTTP Server**: OpenAI + Anthropic API compatible, built-in chat UI, Prometheus metrics, Bearer token auth
- **Multimodal**: Image (`--image`) and video frames (`--video`, `--video-fps`) via Gemma 4 SigLIP-2, Gemma 3 SigLIP, and Qwen VL encoders; also HTTP API
- **Structured Output**: GBNF grammar (`--grammar-string`, `--grammar`), JSON schema (`--json-schema`), JSON mode (`--json-output`), server `response_format: json_object/json_schema`
- **Full Sampling**: CLI: temperature, top-k, top-p, min-p, repeat penalty, DRY, XTC, Mirostat, seed. HTTP API also: frequency/presence penalties, stop sequences
- **Batched Prefill**: Chunked GEMM + fused FlashAttention-2 for fast prompt processing
- **Distributed Inference**: Tensor parallelism (TP), pipeline parallelism (PP), disaggregated prefill/decode. Same-node multi-GPU via POSIX shm (zero-copy IPC), cross-node via TCP. Heterogeneous: mix CUDA + Vulkan + CPU across x86_64 + aarch64
- **Speculative Decoding**: Modes: auto, standard, ddtree, self, ngram, suffix, lookahead, mtp/medusa, eagle, eagle3, mlp, pflash, dspark, dflash2; plus FR-Spec vocab map and LoRA (`--lora`)
- **Fused Megakernels**: Composable GPU megakernels, gate+up+SiLU fused into single dispatch (3→1)
- **Sparse GEMV**: Skip near-zero FFN activation blocks (~40% sparsity from SiLU). CPU +21%, Metal +12%, all GPU backends. Inspired by PowerInfer/TurboSparse
- **SSD Expert Streaming**: `--ssd-streaming` demand-pages MoE experts (and Qwen4-Exp PLE ngrams) from disk via an LRU cache; `--vram-budget` caps resident GPU weights
- **~125 tok/s** on Qwen3.5 0.8B Q8_0 Metal (M4 Pro; see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) as the source of truth), **24.9 tok/s** on Qwen3.5 9B MLX-4bit

## Quick Start

```bash
# Requires Zig 0.16.0 (pin: .zigversion). https://ziglang.org/download/
# Build (produces both ReleaseFast and Debug binaries)
zig build

# Download a model from HuggingFace
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF

# Interactive REPL
./zig-out/bin/agave model.gguf

# Single prompt
./zig-out/bin/agave model.gguf "What is the capital of France?"

# HTTP server
./zig-out/bin/agave model.gguf --serve

# Quiet mode (pipe-friendly, no banner/stats)
./zig-out/bin/agave model.gguf -q "Hello" > output.txt

# Force CPU backend
./zig-out/bin/agave model.gguf --backend cpu

# SafeTensors directory (MLX models)
./zig-out/bin/agave models/mlx-community/gemma-3-4b-it-qat-4bit

# TurboQuant KV cache (2/3/4-bit quantization for longer contexts)
./zig-out/bin/agave model.gguf --kv-type turbo4

# KV cache eviction (extend context past --ctx-size limit)
./zig-out/bin/agave model.gguf --kv-eviction norm --kv-budget 2048
./zig-out/bin/agave model.gguf --kv-eviction tri   # requires .cal file

# Generate TriAttention calibration data
./zig-out/bin/agave calibrate model.gguf

# Vision: describe an image (requires mmproj or built-in vision encoder)
./zig-out/bin/agave model.gguf --image photo.png "Describe this image"

# Override recipe defaults (user flags always win)
./zig-out/bin/agave model.gguf -t 0.9 --top-p 0.95 "Tell me a story"

# Structured output: force JSON
./zig-out/bin/agave model.gguf --json-output "Generate a user profile with name and age"

# Grammar-constrained decoding (GBNF format)
./zig-out/bin/agave model.gguf --grammar-string 'root ::= "yes" | "no"' "Is the sky blue?"

# JSON schema → structured output
./zig-out/bin/agave model.gguf --json-schema '{"type":"object","properties":{"name":{"type":"string"}}}' "User info"

# Sampling parameters
./zig-out/bin/agave model.gguf -t 0.7 --top-p 0.9 --min-p 0.05 "Tell me a story"

# GPU device selection
./zig-out/bin/agave model.gguf --list-devices                      # Show available GPUs
./zig-out/bin/agave model.gguf --backend vulkan --device 1          # Use second GPU

# Speculative decoding
./zig-out/bin/agave target.gguf --draft-model draft.gguf "prompt"   # Separate draft model
./zig-out/bin/agave model.gguf --spec-mode self --draft-layers 9    # Self-speculative
./zig-out/bin/agave model.gguf --spec-mode ddtree "prompt"          # DDTree self-draft

# Fused megakernel (3→1 GPU dispatch for FFN)
./zig-out/bin/agave model.gguf --megakernel "prompt"
```

## Distributed Inference

Split models across multiple GPUs or machines via tensor parallelism (TP) and pipeline parallelism (PP).

```bash
# Same-node multi-GPU (shared memory IPC, zero-copy)
# Terminal 1: rank 0 on GPU 0
./zig-out/bin/agave model.gguf --backend vulkan --device 0 --pp 2 --rank 0 --peers localhost "prompt"
# Terminal 2: rank 1 on GPU 1
./zig-out/bin/agave model.gguf --backend vulkan --device 1 --pp 2 --rank 1 --peers localhost "prompt"

# Cross-node pipeline parallelism (TCP transport)
# Machine A (first half of layers):
./zig-out/bin/agave model.gguf --backend cuda --pp 2 --rank 0 --peers 192.168.0.2 "prompt"
# Machine B (second half + logits):
./zig-out/bin/agave model.gguf --backend cpu --pp 2 --rank 1 --peers 192.168.0.1 "prompt"

# Distributed tensor parallelism (weight sharding + all-reduce)
# Machine A:
./zig-out/bin/agave model.gguf --tp 2 --rank 0 --peers 192.168.0.2 "prompt"
# Machine B:
./zig-out/bin/agave model.gguf --tp 2 --rank 1 --peers 192.168.0.1 "prompt"
```

Supports heterogeneous setups: different backends (CUDA + Vulkan + CPU), architectures (aarch64 + x86_64), and GPU vendors (NVIDIA + AMD) in the same cluster. When `--peers` is `localhost` or `127.0.0.1`, POSIX shared memory is used instead of TCP for zero-copy IPC.

## Supported Models

| Model | Sizes | Status | Quant Types | Notes |
|-------|-------|--------|-------------|-------|
| Gemma 3 | 1B, 4B, 12B, 27B | Working | BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, MLX 4-bit | SPM tokenizer, GELU activation, batched prefill |
| Gemma 4 | E2B, E4B, 26B-A4B | Working | Q8_0, Q4_K, MLX 4-bit | MoE (top-8), channel-based chat template, multimodal vision (SigLIP-2) |
| Qwen 3.5 | 0.8B, 9B, 27B, 35B | Working | Q4_0, Q4_K_M, Q8_0, BF16, MLX 4-bit | Hybrid DeltaNet SSM + attention |
| Qwen4-Exp | Flash-Next | Working | NVFP4, BF16, MLX 4-bit | Gated DeltaNet 36× + QSA 12×, PLE 51B ngram (SSD), HC 4×320, 512 experts |
| GPT-OSS | 20B | Partial | Q4_0 | MoE, sliding window, attention sinks (poor output quality) |
| Nemotron-H | n/a | Partial | Q5_0 | Mamba-2 + attention hybrid, GGUF (poor output quality) |
| Nemotron Nano | 30B | Partial | MLX 4-bit, NVFP4 | SSM + MoE + attention hybrid, SafeTensors (poor output quality) |
| GLM-4 MoE Lite | 4.7B | Partial | MLX 4/6/8-bit | MLA + MoE (GGUF compatibility issue, poor output quality) |
| DiffusionGemma | 26B-A4B | Working | BF16 | Block diffusion: 256-token canvas, MoE top-8, SafeTensors only |
| DeepSeek V4 Flash | 0731 | Working | Q4_K, Q8_0 | MLA, 4-stream HC, CSA/HCA compressors, LID, 256 MoE experts top-6, MTP heads (`--mtp-model`) |
| Llama 4 | Scout | Working | Q4_K, Q8_0 | iRoPE, chunked attention, MoE top-1 + shared expert, batched prefill |

## Model Download

Download GGUF models from HuggingFace Hub with automatic quantization selection:

```bash
# Download best available quantization (prefers Q4_K_M)
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF

# Request specific quantization
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF --quant Q8_0

# List available GGUF files without downloading
./zig-out/bin/agave pull Qwen/Qwen3.5-0.8B-GGUF --list

# Private repos
HF_TOKEN=hf_xxxxx ./zig-out/bin/agave pull org/private-model
```

Downloads are stored in the standard HuggingFace cache layout with an agave convenience symlink. Supports resume on interrupted downloads.

## Calibration

Generate TriAttention calibration data for frequency-domain KV eviction:

```bash
# Run calibration (produces model.cal alongside model.gguf)
./zig-out/bin/agave calibrate model.gguf
```

The calibration pass records per-head Q/K frequency statistics used by the `--kv-eviction tri` policy. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Browser WASM

In-browser inference uses `web/agave.ts` (`AgaveEngine`) against `agave.wasm`
(`zig build wasm`). The engine is a small library, not the HTTP `/v1` API.

```js
const agave = new AgaveEngine();
await agave.init(); // or agave.init(wasmBytes)
await agave.loadModel('https://example.com/model.gguf');
try {
  const output = await agave.generate('What is 2+2?', { maxTokens: 100 });
} catch (e) {
  if (e instanceof AgaveError && e.code === 'no_model') {
    // load a model, then retry
  }
  throw e;
}
agave.destroy();
```

`AgaveError.code` is the supported way to distinguish failures (`not_initialized`,
`no_model`, `alloc_failed`, `wasm_fetch_failed`, `wasm_invalid`, `download_failed`,
`gguf_parse`, `unsupported_arch`, `no_vocab`, `tokenizer`, `init_failed`,
`generate_failed`). Serve `web/` as a static directory after `zig build wasm`.
Forward-pass generation in WASM is still blocked by a Zig wasm32 codegen bug;
load and tokenize work.

## HTTP Server

Start with `--serve`. Supports both synchronous JSON and SSE streaming.

```bash
# Prefer AGAVE_API_KEY over --api-key (CLI args appear in process listings)
AGAVE_API_KEY=sk-mykey ./zig-out/bin/agave model.gguf --serve
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI chat completion API |
| `/v1/completions` | POST | OpenAI text completion API |
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/models` | GET | List loaded models |
| `/v1/embeddings` | POST | Embedding generation (stub, returns 501) |
| `/v1/chat` | POST | Built-in web chat UI |
| `/v1/chat/regenerate` | POST | Regenerate last assistant response |
| `/v1/conversations` | GET, POST | Conversation management |
| `/v1/tokenize` | POST | Count tokens in text |
| `/v1/detokenize` | POST | Convert token IDs to text |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

Server features: up to 64 concurrent connections, request scheduler (batch up to 8, 120s timeout), 30s connection read timeout, Bearer token auth, CORS support.

## Interactive REPL

Launch without a prompt argument for multi-turn chat:

```bash
./zig-out/bin/agave model.gguf
```

**Commands:**

| Command | Description |
|---------|-------------|
| `/clear`, `/reset` | Clear conversation history and KV cache |
| `/context`, `/ctx` | Show context window usage (tokens used / max) |
| `/system <text>` | Set system prompt (clears conversation) |
| `/system` | Show current system prompt |
| `/stats` | Toggle generation statistics display |
| `/verbose` | Toggle technical details (params, EOG tokens) |
| `/debug` | Toggle debug logging (token IDs, layer timing) |
| `/model` | Show model information |
| `/help` | Show REPL help |
| `/quit`, `/exit`, `/q` | Exit |

Keyboard shortcuts: `Ctrl+C` cancel, `Ctrl+D` quit, `Ctrl+L` clear screen, `Ctrl+R` reverse search.

## Benchmarks

Measured on Apple M4 Pro (48 GB unified memory). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology.

| Model | Quant | Backend | Decode (tok/s) | vs llama.cpp |
|-------|-------|---------|---------------:|-------------:|
| Qwen3.5 0.8B | Q8_0 | Metal | 125† | n/a |
| Qwen3.5 9B | Q8_0 | Metal | 41.7 | **1.67x** |
| Gemma 3 4B | MLX-Q4 | Metal | 78.1 | n/a |
| Gemma 3 12B | Q8_0 | Metal | 22.3 | **1.19x** |
| Gemma 4 E2B | Q4_K_M | Metal | 21.8 | n/a |
| Gemma 4 E4B | Q4_K_M | Metal | 14.4 | n/a |
| Gemma 4 26B-A4B | Q4_K_M | Metal | 4.2 | n/a |
| Gemma 3 27B | QAT 4-bit | Metal | 6.3 | n/a |
| Qwen3.5 9B | MLX-4bit | Metal | 24.9 | n/a |

### Multi-Backend (Qwen3.5 0.8B Q8_0)

| Backend | Hardware | Decode (tok/s) | Output correct |
|---------|----------|---------------:|----------------|
| Metal | Apple M4 Pro | 125† | yes |
| ROCm | AMD RX 7900 XTX | 50.8 | **no**, see below |
| CPU | Ryzen 9 9950X (32T) | 44 | yes |
| CUDA | NVIDIA GB10 (aarch64) | 35 | yes |
| Vulkan | AMD RX 7900 XTX | 2.7 | **no**, see below |

> **Known bug (2026-08-26):** on AMD RX 7900 XTX, Qwen 3.5 GGUF decodes to
> incoherent text on both ROCm and Vulkan while CPU is correct for the same
> model, prompt and seed. Both backends emit the same wrong tokens, so the
> fault is in a path they share, not in two separate kernels. Treat the ROCm
> and Vulkan throughput above as speed-only measurements, not working
> configurations. Tracked in [docs/TODO.md](docs/TODO.md).

### Distributed Inference (dual NVIDIA GB10 over RoCE RDMA)

| Model | Config | Transport | Decode (tok/s) |
|-------|--------|-----------|---------------:|
| 9B Q8_0 | Single GPU | n/a | 9.1 |
| 9B Q8_0 | PP=2 | NCCL RoCE | **8.5** |
| 9B Q8_0 | TP=2 | NCCL RoCE | 5.1 |
| 9B Q8_0 | TP=2 | TCP RoCE | 4.9 |
| 27B Q4_K_M | Single GPU | n/a | 2.2 |
| 27B Q4_K_M | PP=2 | NCCL RoCE | 2.2 |
| 27B Q4_K_M | TP=2 | NCCL RoCE | 1.7 |

†Canonical decode numbers from [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (2026-05-26 sparse GEMV + Accelerate). Other tables may reflect older runs.

All quant formats supported on all backends: Q8_0 (GPU), Q4_0/Q4_K/Q5_K/Q6_K (GPU or CPU fallback on UMA). See [docs/KERNELS.md](docs/KERNELS.md) for details.

## Prerequisites

- **Zig 0.16.0** (pin in `.zigversion`; must match `build.zig.zon` `.minimum_zig_version`). Download: https://ziglang.org/download/
- macOS (Metal backend) / Linux (Vulkan, CUDA, ROCm) / any platform (CPU, WebGPU backends)
- GPU backends load drivers at runtime via dlopen, no SDK needed at build time
- Contributors: `zig build check` is the local CI gate (format + docs hygiene + unit tests). TypeScript under `src/web/` and `web/` also needs `bun run lint` and `bun run typecheck`. See [Contributing](docs/CONTRIBUTING.md).

## CLI Options

```
agave [OPTIONS] <model> [prompt]

  -h, --help               Show help
  -v, --version            Print version
  -q, --quiet              Suppress banner and stats
  -s, --serve              Start HTTP server
  -p, --port <PORT>        Server port [default: 49453]
  -n, --max-tokens <N>     Max tokens to generate [default: 512]
  -t, --temperature <T>    Sampling temperature, 0 = greedy [default: 0]
      --top-p <P>          Nucleus sampling threshold [default: 1.0]
      --top-k <K>          Top-k sampling, 0 = disabled [default: 0]
      --min-p <P>          Min-p sampling threshold [default: 0]
      --repeat-penalty <R> Repetition penalty [default: 1.0]
      --dry-multiplier <M> DRY n-gram repetition penalty [default: 0]
      --dry-length <N>     DRY minimum n-gram length [default: 2]
      --xtc-probability <P> XTC diversity sampling [default: 0]
      --xtc-threshold <T>  XTC probability threshold [default: 0.1]
      --mirostat-mode <N>  Mirostat target-entropy sampling: 0=off, 2=on [default: 0]
      --mirostat-tau <T>   Mirostat target entropy [default: 5.0]
      --mirostat-eta <E>   Mirostat learning rate [default: 0.1]
      --system <TEXT>      System prompt for chat formatting
      --backend <BE>       auto, cpu, metal, vulkan, cuda, rocm, webgpu [default: auto]
      --ctx-size <N|auto>  Context window size [default: min(model, 4096), 0 = model max, auto = fit to memory]
      --seed <N>           Random seed for sampling [default: random]
      --grammar <FILE>     GBNF grammar file for constrained decoding
      --grammar-string <G> Inline GBNF grammar string
      --json-schema <S>    JSON schema for structured output
      --json-output        Constrain generation to valid JSON via grammar (not output format; see --json)
      --kv-type <TYPE>     KV cache quantization: f32, f16, q8_0/q8, int8/i8, fp8/fp8_e4m3, nvfp4/fp4, nvfp4_ds_mla, turbo2/tq2, turbo3/tq3, turbo4/tq4, planar2/pq2 through planar4/pq4, iso2/iq2 through iso4/iq4, rotor2/rq2 through rotor4/rq4, turbo (preset: K=q8_0, V=turbo4) [default: f16]
      --kv-tiers <TIERS>   Enable tiered KV cache: vram+ram, vram+ram+ssd [default: off]
      --kv-ram-budget <GB> RAM tier budget in GB, requires --kv-tiers [default: 50% of free RAM]
      --kv-ssd-path <PATH> SSD tier file path, requires --kv-tiers with ssd
      --kv-ssd-budget <GB> SSD tier budget in GB, requires --kv-tiers with ssd [default: 10]
      --host <ADDR>        Server bind address [default: 127.0.0.1]
      --api-key <KEY>      API key for server auth (prefer AGAVE_API_KEY; env wins if both set)
      --prefill-batch-size <N> Prefill chunk size in tokens [default: 512]
      --no-color           Disable colored output (same as --color=never)
      --color <MODE>       Color mode: auto, always, never [default: auto]
      --kv-type-k <TYPE>   KV key quantization (overrides --kv-type)
      --kv-type-v <TYPE>   KV value quantization (overrides --kv-type)
      --cache-type-k <TYPE> Alias for --kv-type-k
      --cache-type-v <TYPE> Alias for --kv-type-v
  -V, --verbose            Show technical details (params, load times, EOG)
      --allow-cpu-fallback Not implemented; GPU backends fail closed (flag only warns)
  -d, --debug              Enable debug logging (token IDs, layer timing); implies --verbose
      --json               Output results as JSON (implies --quiet)
      --model-info         Print model metadata and exit (combine with --json)
      --profile            Profile per-op timing (halves throughput)
      --benchmark          Run decode benchmark with built-in prompt
      --frontier-bench     Frontier benchmark: tok/s at each --frontier-ctx length
      --frontier-ctx <LIST> Comma-separated context lengths [default: 512,2048,8192]
      --mmproj <PATH>      Path to vision projector GGUF (mmproj file)
      --image <PATH>       Path to image file for multimodal inference (PNG or PPM)
      --kv-eviction <MODE> KV cache eviction policy: none, norm, tri [default: none]
      --kv-budget <N>      Max KV entries to retain after eviction [default: 80% of ctx-size]
      --mmap               Use lazy mmap instead of preloading weights into RAM
      --megakernel         Enable fused FFN megakernels (3→1 dispatch per layer)
      --power <N>          Target GPU utilisation percent 1-100 [default: 100]
      --draft-model <PATH> Draft model GGUF for speculative decoding
      --mtp-model <PATH>   MTP weight file (safetensors) for multi-token prediction
      --spec-mode <MODE>   Speculative mode: auto, standard, ddtree, self, ngram, suffix,
                           lookahead, mtp, medusa, eagle, eagle3, mlp, pflash, dspark, dflash2
  -K, --spec-tokens <N>    Draft tokens per speculation round [default: 5]
      --tree-budget <N>    DDTree node budget [default: 64]
      --draft-layers <N>   Layers for self-speculative draft [default: auto]
      --spec-token-map <F> FR-Spec token frequency map for vocab truncation
      --pflash-alpha <F>   PFlash block selection threshold [default: 0.85]
      --pflash-block-size <N>  PFlash scoring block size [default: 64]
      --pflash-scorer <P>  Separate model for PFlash scoring
      --lora <PATH>        Merge LoRA adapter GGUF at load time
      --dir-steering-file <PATH>  Directional steering f32 vector (n_layers × n_embd)
      --dir-steering-ffn <F>      Steering scale for FFN outputs [default: 1.0 with file]
      --dir-steering-attn <F>     Steering scale for attention outputs [default: 0]
      --video <PATH>       Video file for multimodal (frames extracted via ffmpeg)
      --video-fps <N>      Video frame sampling rate [default: 1]
      --diffusion-steps <N>  DiffusionGemma denoising steps [default: 16]
      --diffusion-canvas <N> DiffusionGemma canvas size [default: 256]
      --diffusion-confidence <F>  Diffusion acceptance threshold [default: 0.5]
      --sleep-after <N>    Server sleep after N seconds idle (0=off)
      --max-batch-size <N> Server concurrent batch size [default: 8] (admission is one-at-a-time until per-request paged KV is wired)
      --rate-limit-rpm <N> Server max requests/min (0=unlimited)
      --rate-limit-tpm <N> Server max prompt tokens/min (0=unlimited)
      --conv-store <PATH>  Persist web-UI conversations [default: $XDG_CACHE_HOME/agave/conversations.json]
      --no-conv-store      Do not persist or restore web-UI conversations
      --no-kv-cache        Prefill-only / embedding server mode
      --list-devices       List available compute devices and exit
      --device <N>         GPU device index for CUDA/ROCm/Vulkan [default: 0]
      --tp <N>             Tensor parallelism degree [default: 1]
      --pp <N>             Pipeline parallelism stages [default: 1]
      --peers <ADDR>       Peer address for distributed inference
      --rank <N>           This node's rank [default: 0]
      --transport <TYPE>   IPC transport: auto, tcp, shm, nccl [default: auto]
      --disagg             Disaggregated prefill/decode
      --power <N>          Target GPU utilisation percent (1-100) [default: 100]
      --vram-budget <GIB>  Cap GPU memory for cached weights (GiB, e.g. 20, 0.5, or auto)
      --vram-budget-policy <P>  Weight eviction when full: mru (default) or lru
      --ssd-streaming      Stream MoE experts from SSD
      --ssd-cache-slots <N> LRU expert cache size [default: 256]
      --expert-profile-out <FILE>  Save expert activation profile
      --expert-profile-in <FILE>   Load expert activation profile for cache warming
      --frontier-bench     Frontier benchmark (snapshot KV at each context)
      --frontier-ctx <LIST> Comma-separated context lengths for frontier bench
```

## Build Options

All backends and models are enabled by default. Disable individually to reduce binary size or avoid unwanted dependencies.

```bash
# Disable specific backends
zig build -Denable-vulkan=false
zig build -Denable-cuda=false -Denable-rocm=false

# CPU-only build (no GPU backends)
zig build -Denable-metal=false -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false

# GPU-only (disable CPU fallback: compile error if GPU init fails)
zig build -Denable-cpu=false

# Disable specific model architectures
zig build -Denable-glm4=false

# Minimal build: single model (Gemma 3) + single backend (Metal)
zig build -Denable-gemma4=false -Denable-qwen35=false -Denable-qwen4-exp=false \
  -Denable-gpt-oss=false -Denable-nemotron-h=false -Denable-nemotron-nano=false \
  -Denable-glm4=false -Denable-llama4=false -Denable-diffusion-gemma=false \
  -Denable-deepseek4=false -Denable-dflash2=false \
  -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false

# Override GPU architecture targets
zig build -Dcuda-sm=sm_120        # Blackwell
zig build -Drocm-arch=gfx942      # MI300X

# Cross-compile
zig build -Dtarget=aarch64-linux-gnu -Denable-metal=false
```

**Backend Options:**

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable-cpu` | bool | true | CPU backend |
| `enable-metal` | bool | true | Metal backend (macOS only) |
| `enable-vulkan` | bool | true | Vulkan backend (runtime dlopen) |
| `enable-cuda` | bool | true | CUDA backend (runtime dlopen) |
| `enable-rocm` | bool | true | ROCm backend (runtime dlopen) |
| `enable-webgpu` | bool | true | WebGPU backend (runtime dlopen, WGSL) |
| `enable-debug` | bool | true | Build the `agave-debug` (ReleaseSafe) binary |
| `cuda-sm` | enum | sm_90 | CUDA SM target (sm_50..sm_120, plus sm_121) |
| `rocm-arch` | enum | gfx1100 | ROCm GFX target (gfx90a..gfx1151) |

**Model Options:**

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable-gemma3` | bool | true | Gemma 3 model support |
| `enable-gemma4` | bool | true | Gemma 4 model support |
| `enable-diffusion-gemma` | bool | true | DiffusionGemma model support |
| `enable-qwen35` | bool | true | Qwen 3.5 model support |
| `enable-qwen4-exp` | bool | true | Qwen4-Exp / Qwen3.8-Flash-Next |
| `enable-gpt-oss` | bool | true | GPT-OSS model support |
| `enable-nemotron-h` | bool | true | Nemotron-H model support |
| `enable-nemotron-nano` | bool | true | Nemotron Nano model support |
| `enable-glm4` | bool | true | GLM-4 model support |
| `enable-deepseek4` | bool | true | DeepSeek V4 Flash model support |
| `enable-llama4` | bool | true | Llama 4 model support |
| `enable-dflash2` | bool | true | DFlash2 block-diffusion drafter support |

## Recipes

Recipes are optional preset configurations matched by architecture + backend + quantization. They provide proven defaults (temperature, top-p, context size, etc.) while allowing full user override via CLI flags.

```
# Recipe auto-applied, shown in banner:
🌵 agave Qwen3.5-0.8B Q4_0 Metal 32L/4096E/16H (45ms)
recipe: Qwen3.5 Q4 Metal

# User flags always take priority over recipe defaults:
./zig-out/bin/agave model.gguf -t 0  # overrides recipe temperature
```

Current presets: Qwen3.5 Q4 Metal, Qwen2 Q4 Metal, Gemma Q4 Metal, GPT-OSS Metal, GLM-4 generic, DeepSeek V4 Flash, DeepSeek V2 generic, Llama 4 generic, CPU generic. Add new recipes in `src/recipe.zig`.

## Project Structure

The annotated source tree lives in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#project-structure), together with the
inference pipeline and the reasoning behind each layer. In short: `src/backend/`
holds one file per backend behind a comptime dispatcher, `src/models/` one file
per architecture behind a vtable, `src/ops/` the shared math and quantization,
and `research/kernels/` prototypes that are not part of the build.

## Docker

Preferred local server path: copy `.env.example` to `.env`, set `AGAVE_API_KEY` and model paths, then `docker compose up --build`. Compose publishes on `127.0.0.1` by default (override with `AGAVE_HOST_BIND`). Conversations, the Vulkan pipeline cache, and Hub downloads live in a named volume `agave-cache` at `/home/agave/.cache` (override with `AGAVE_CACHE_DIR`). `docker compose down -v` deletes that volume.

Build multi-platform images (x86_64 + aarch64) using `docker buildx`:

```bash
# Build for both platforms (all GPU backends enabled, glibc)
docker buildx build --platform linux/amd64,linux/arm64 -t agave .

# Build and load for current platform only
docker buildx build --load -t agave .

# Release build: stamp the OCI version label from build.zig.zon (the image
# validates it against .version; plain builds label as "dev" but still ship
# /usr/share/agave/version)
docker buildx build --load -t agave \
  --build-arg AGAVE_VERSION="$(sed -n 's/^[[:space:]]*\.version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' build.zig.zon | head -n1)" .

# CPU-only build (static musl binary, smaller image)
docker buildx build --load -t agave \
  --build-arg ENABLE_VULKAN=false \
  --build-arg ENABLE_CUDA=false \
  --build-arg ENABLE_ROCM=false \
  --build-arg ENABLE_WEBGPU=false .

# Minimal build: single model + CPU only
docker buildx build --load -t agave \
  --build-arg ENABLE_VULKAN=false \
  --build-arg ENABLE_CUDA=false \
  --build-arg ENABLE_ROCM=false \
  --build-arg ENABLE_WEBGPU=false \
  --build-arg ENABLE_QWEN35=false \
  --build-arg ENABLE_QWEN4_EXP=false \
  --build-arg ENABLE_GPT_OSS=false \
  --build-arg ENABLE_NEMOTRON_H=false \
  --build-arg ENABLE_NEMOTRON_NANO=false \
  --build-arg ENABLE_GLM4=false \
  --build-arg ENABLE_GEMMA4=false \
  --build-arg ENABLE_DIFFUSION_GEMMA=false \
  --build-arg ENABLE_DEEPSEEK4=false \
  --build-arg ENABLE_LLAMA4=false \
  --build-arg ENABLE_DFLASH2=false .

# One-shot inference (--no-healthcheck: image HEALTHCHECK expects --serve /ready)
docker run --rm --no-healthcheck -v /path/to/models:/models agave /models/model.gguf "Hello"

# HTTP server (AGAVE_API_KEY required: image binds 0.0.0.0 inside the container)
# Prefer loopback publish; HEALTHCHECK reads AGAVE_PORT (keep -p and -e aligned).
docker run --rm -p 127.0.0.1:49453:49453 -e AGAVE_API_KEY \
  -v /path/to/models:/models agave /models/model.gguf --serve

# Override Zig version at build time
docker buildx build --build-arg ZIG_VERSION=0.16.0 -t agave .
```

Dlopen backends (CUDA, Vulkan, ROCm, WebGPU) load native libraries at runtime and require glibc. When all four are disabled, the Docker build switches to musl for a fully static binary. Zig cross-compiles natively, no QEMU emulation needed during build.

### Static musl builds

For environments where a fully static, dependency-free binary is needed (Alpine containers, embedded systems, minimal distros), disable all dlopen backends:

```bash
# Static musl binary (CPU backend only)
zig build -Dtarget=x86_64-linux-musl \
  -Denable-metal=false -Denable-vulkan=false \
  -Denable-cuda=false -Denable-rocm=false \
  -Denable-webgpu=false

# Cross-compile static ARM64 binary
zig build -Dtarget=aarch64-linux-musl \
  -Denable-metal=false -Denable-vulkan=false \
  -Denable-cuda=false -Denable-rocm=false \
  -Denable-webgpu=false
```

**Note:** Static musl builds only work with the CPU backend. Dlopen backends (CUDA, Vulkan, ROCm, WebGPU) need glibc. Loading a glibc-linked `.so` from a musl binary will segfault.

## Documentation

- **[Tutorial: LLM Inference From Scratch](docs/tutorial/README.md)**: 25-chapter progressive tutorial + 5 appendixes
- **[Architecture](docs/ARCHITECTURE.md)**: Project structure, module reference, inference pipeline
- **[Models](docs/MODELS.md)**: Supported models, parameters, per-model details
- **[Benchmarks](docs/BENCHMARKS.md)**: Performance comparisons vs llama.cpp
- **[Kernel Status](docs/KERNELS.md)**: Per-backend kernel implementation status
- **[Distributed Inference](docs/PARALLELISM.md)**: TP, PP, disaggregated prefill/decode
- **[Contributing](docs/CONTRIBUTING.md)**: How to add backends, models, quantization; [versioning & releases](docs/CONTRIBUTING.md#versioning--releases)
- **[Changelog](CHANGELOG.md)**: User-facing history (product version `0.2.0`, 0.x SemVer)
- **[API Reference](docs/API.md)**: HTTP API endpoints, request/response formats
- **[Megakernel System](docs/MEGAKERNEL.md)**: Composable fused GPU dispatch
- **[CLAUDE.md](CLAUDE.md)**: Engineering standards for contributors
- **[research/kernels/](research/kernels/)**: Kernel research tools (benchmarks, golden tests)

## License

GNU General Public License v3.0 or later
