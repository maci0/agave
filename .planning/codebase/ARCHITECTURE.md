# Architecture

**Analysis Date:** 2025-03-21

## Pattern Overview

**Overall:** Dispatcher/Implementation (tagged union with `inline else` dispatch)

**Key Characteristics:**
- Zero-overhead backend selection via tagged union dispatch (no VTable indirection)
- Comptime vtable generation for model abstraction (duck-typing)
- Deferred GPU execution on UMA (Unified Memory Architecture) platforms
- Multi-model support with build-time architecture toggles
- Strict boundary enforcement: implementations accessed only through dispatchers

## Layers

**Presentation Layer:**
- Purpose: CLI interface, REPL, HTTP server, output formatting
- Location: `src/main.zig` (CLI + generation logic), `src/server.zig` (HTTP API), `src/display.zig` (formatting), `src/readline.zig` (interactive input)
- Contains: Argument parsing, generation loop, prompt formatting, token streaming
- Depends on: Model interface, Tokenizer interface, Format interface
- Used by: Entry point only

**Model Layer:**
- Purpose: Unified interface to all supported architectures (Gemma3, Qwen3.5, GPT-OSS, Nemotron-H, Nemotron-Nano, GLM-4)
- Location: `src/models/model.zig` (interface), `src/models/{gemma3,qwen35,gpt_oss,nemotron_h,nemotron_nano,glm4}.zig` (implementations)
- Contains: Forward pass, KV cache management, layer dispatch, quantization handling
- Depends on: Backend interface, Format interface, KV cache managers, Shared ops (attention, SSM, quantization)
- Used by: Main (via `Model.from()` interface)

**Backend Layer:**
- Purpose: Compute abstraction for CPU, Metal, Vulkan, CUDA, ROCm backends
- Location: `src/backend/backend.zig` (dispatcher), `src/backend/{cpu,metal,vulkan,cuda,rocm}.zig` (implementations)
- Contains: Kernel launchers, memory management, sync points, deferred execution handling
- Depends on: Kernel implementations (in `src/backend/kernels/`), system APIs (Metal Framework, CUDA Runtime, Vulkan, etc.)
- Used by: Model layer, Shared ops layer

**Operations Layer (Shared):**
- Purpose: Algorithm implementations reused across models
- Location: `src/ops/`
- Contains:
  - `attention.zig` - SDPA with optional sliding window and KV quantization
  - `ssm.zig` - SSM operations (causal conv1d, Mamba-2 recurrence, group RMS+gate)
  - `math.zig` - Shared math (GELU, softplus, sigmoid, sampling, argmax)
  - `quant.zig` - Quantization helpers (dequant, bf16, FP8, NVFP4, MXFP4, IQ4)
  - `kv_quant.zig` - KV cache quantization interface
  - `mlx.zig` - MLX affine quantization (SafeTensors support)
- Depends on: Backend interface
- Used by: Model layer

**Format Layer:**
- Purpose: Model file loading (GGUF, SafeTensors)
- Location: `src/format/format.zig` (interface), `src/format/{gguf,safetensors}.zig` (implementations)
- Contains: Tensor lookup, metadata extraction, weight loading, format detection
- Depends on: None (filesystem only)
- Used by: Main, Model layer

**Tokenizer Layer:**
- Purpose: Text ↔ token ID conversion
- Location: `src/tokenizer/tokenizer.zig` (interface), `src/tokenizer/bpe.zig` (implementation)
- Contains: BPE encoding/decoding, SentencePiece (SPM) encoding, special tokens
- Depends on: None
- Used by: Main, Server

**KV Cache Layer:**
- Purpose: Manage key/value cache across sequences
- Location: `src/kvcache/manager.zig`
- Contains: Flat allocation, PagedAttention (block-based), RadixAttention (prefix tree), LRU eviction
- Depends on: None
- Used by: Backend layer, Model layer

**Supporting:**
- `src/chat_template.zig` - Role markers, EOG tokens per model family (data-driven)
- `src/recipe.zig` - Optional preset configurations (arch + backend + quant matching)
- `src/thread_pool.zig` - Futex-based work-stealing thread pool for CPU parallelism
- `src/arch.zig` - Model architecture enum with detection and build flags
- `src/perf.zig` - Per-layer profiling instrumentation

## Data Flow

**Model Load → Initialization:**
1. `main.zig` detects file format (directory → SafeTensors, else → GGUF)
2. Format dispatcher opens file, parses metadata, detects architecture
3. Architecture enum routes to concrete model type
4. Model `init()` loads weights via format interface, allocates KV cache, initializes backend
5. All subsequent operations use opaque `Model` and `Backend` interfaces

**Token Generation (Inference Loop):**
1. User provides prompt → `main.zig` encodes to token IDs via tokenizer
2. Prefill phase: for each prompt token, call `model.forward(token_id)`
   - Model processes through all layers (via Backend dispatcher)
   - Each layer reads/writes activations (on GPU: deferred, no sync needed)
   - Last forward call's logits used for first generated token
3. Decode phase: repeatedly call `model.forward(next_token)`
   - Each forward advances KV cache position (single token)
   - Logits sampled or selected (via `math_ops.sampleToken`)
   - Stop conditions: EOG token, repetition limit, or max_tokens reached
4. Token IDs decoded back to text via tokenizer, streamed to stdout/socket

**Generation → Output:**
- Tokens decoded in small batches (4 for TTY, 32 for pipe) for responsive streaming
- Optional stats printed: TTFT, throughput, prefill/decode breakdown
- JSON mode outputs structured result with metadata

**GPU Sync Points:**
- `be.sync()` called only when CPU reads GPU data (e.g., before argmax, before embedding lookup)
- UMA platforms: sync flushes command buffer but no D2H copy needed (memory is shared)
- Discrete GPU: sync also downloads activations to host
- Models minimize sync calls: SDPA handles its own internal sync, final logits stay on GPU until argmax

## Key Abstractions

**Backend Union:**
- Type: `union(enum) { cpu, metal, vulkan, cuda, rocm }`
- Dispatch: `inline else` switch — compiler resolves at compile-time for each backend variant
- Zero overhead: no function pointer table, no runtime type checks
- Interface: All backends implement same set of operations (gemv, rmsNorm, rope, sdpa, silu, gelu, add, mul, softmax, embLookup, l2Norm, etc.)
- Pattern: Operations that differ per-backend (e.g., GPU-accelerated) dispatch to backend; ops that are universal (e.g., CPU-only math) call shared helpers

**Model Vtable:**
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Generated: Comptime vtable creation via `Model.from(ConcreteType, &instance)`
- Interface: `forward(token_id) !u32`, `resetCache()`, `cancel()`, field accessors
- Benefit: High-level code (`main.zig`, `server.zig`) works with any model type without generics

**Format Interface:**
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Methods: `getTensor(name)`, `getMetaStr/U32/F32(key)`, `getVocab()`, `getMerges()`
- Implementations: GGUF (mmap'd file with index), SafeTensors (sharded tensors with JSON metadata)

**Tokenizer Interface:**
- Type: `struct { ptr: *anyopaque, vtable: *const VTable }`
- Methods: `encode(text)`, `decode(tokens)`, `vocabSize()`
- Modes: BPE (byte-pair merges), SPM (SentencePiece greedy), SPM-no-dummy (Gemma3)

**KV Cache:**
- Flat: Simple per-layer byte slices (single-sequence, no sharing)
- Paged: Block-based allocation per vLLM PagedAttention spec (multi-sequence, shared blocks)
- Radix: Prefix tree for longest-common-prefix detection and copy-on-write sharing (SGLang-style)

## Entry Points

**Interactive REPL (`src/main.zig: runRepl()`):**
- Location: `src/main.zig` lines 1114–1245
- Triggers: No prompt or server flag provided
- Responsibilities:
  - Read user input line-by-line with history
  - Format messages (system + history + new message) via `ChatTemplate`
  - Call `generateAndPrintInner()` for each turn
  - Clear KV cache on `/clear` command
  - Multi-turn support via conversation history

**One-Shot Generation (`src/main.zig: generateAndPrint()`):**
- Location: `src/main.zig` lines 1249–1265
- Triggers: Prompt provided (CLI arg or piped stdin)
- Responsibilities:
  - Format prompt via chat template
  - Run `generateAndPrintInner()` once
  - Print stats if `--verbose` or `--json`

**HTTP Server (`src/server.zig`):**
- Location: `src/server.zig`
- Triggers: `--serve` flag
- Responsibilities:
  - Listen on port (default 49453)
  - Accept OpenAI-compatible requests (`POST /v1/chat/completions`)
  - Dispatch to model, stream response as SSE or JSON
  - Request queuing and batching (if implemented)

**Core Generation (`src/main.zig: generateAndPrintInner()`):**
- Location: `src/main.zig` lines 1272–1451
- Triggers: Called by all three entry points
- Responsibilities:
  - Encode prompt to token IDs (BPE or SPM)
  - Send BOS token if needed
  - Prefill: process all prompt tokens, capture first generated token
  - Decode: loop calling `model.forward()`, sample tokens, check stop conditions
  - Stream output in batches (TTY vs pipe sizing)
  - Return full response text for REPL/server use

## Error Handling

**Strategy:** Explicit error sets with context-specific handling

**Patterns:**

```zig
// Model initialization errors
pub const ForwardError = error{
    MissingTensor,      // Weight not found in model file
    KVCacheFull,        // Context limit reached
    Cancelled,          // Async cancellation
    OutOfMemory,        // Allocation failure
};

// Format loading errors
GGUFFile.open() -> error{ FileNotFound, InvalidMagic, UnsupportedVersion, FileTooSmall, OutOfMemory }
SafeTensorsDir.open() -> error{ FileNotFound, NotDir, InvalidFormat, OutOfMemory }

// Tokenizer errors
TokenizerError = error{ OutOfMemory }

// Backend errors
Backend.init() -> error{ BackendDisabled, OutOfMemory, ... (platform-specific) }
```

**Propagation:**
- Main catches all initialization errors, prints diagnostic, exits
- Generation errors (forward fails) logged and generation aborts
- Tokenizer errors (encode/decode fails) logged and prompt/output skipped
- Server catches all request errors, returns HTTP 500 with error message

## Cross-Cutting Concerns

**Logging:**
- CLI: `eprint()` for errors/warnings, `print()` for output
- Debug mode: `dbg()` macro (conditional on `g_debug` flag)
- Server: logs to stderr with request context
- No global logger; output via `std.fs.File.stderr()` and `std.fs.File.stdout()`

**Validation:**
- Format: Metadata consistency checks (architecture name, tensor names)
- Model: Weight shapes validated during load, layer count asserted
- Tokenizer: Vocabulary size must match model vocab
- Backend: Device availability checked at init, graceful fallback to CPU

**Authentication:**
- Models loaded from local filesystem only (no remote URLs)
- Server: No authentication (localhost only by default, or behind proxy)
- No secrets in code (env vars for API keys, not committed)

**Concurrency:**
- Single-threaded generation (one token at a time through model)
- CPU backend parallelizes GEMV rows via thread pool (`src/thread_pool.zig`)
- Metal/Vulkan/CUDA: GPU handles parallelism; CPU thread is event loop (no explicit threading)
- Server: Each request queued sequentially (future: continuous batching could interleave multiple requests)

**Configuration:**
- Build-time toggles: `-Denable-gemma3=false` disables model at compile time
- Runtime CLI args: backend, context size, sampling params, KV quantization
- Recipes: Optional preset configs matched by arch + backend + quant (user CLI overrides)
- Chat templates: Data-driven per-architecture (no hardcoded prompts)

**Performance:**
- Hot path (token generation): Zero allocations, zero syscalls, no locks
- Memory: KV cache allocation happens once at init; activations reused across tokens
- GPU: Deferred dispatch (command buffers); CPU reads data only at sync points
- Quantization: Dequantization happens in-kernel (on GPU or SIMD on CPU), not pre-converted to f32
