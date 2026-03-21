# External Integrations

**Analysis Date:** 2026-03-21

## APIs & External Services

**OpenAI-Compatible API:**
- Endpoint: `http://0.0.0.0:{port}` (default 49453)
- Compatible with: OpenAI client libraries, LM Studio, Ollama clients
- Implementation: `src/server.zig` (pure Zig HTTP/1.1 server)

**Provided Endpoints:**
- `POST /v1/chat/completions` - Chat completion with streaming support
  - Request format: JSON with `messages`, `max_tokens`, `stream`
  - Response: OpenAI-compatible JSON or SSE stream

- `POST /v1/completions` - Text completion (raw prompt, no chat template)
  - Request format: JSON with `prompt`, `max_tokens`, `stream`
  - Response: OpenAI-compatible JSON or SSE stream

- `POST /v1/responses` - Alternative response endpoint
  - Request format: Form-encoded `input`, `max_tokens`

- `GET /v1/models` - List available models
  - Response: JSON with model metadata

- `POST /v1/conversations` - Conversation management (web UI only)
  - Actions: `new` (create), `select` (activate), `delete` (remove)
  - Stored in-memory per server instance

- `POST /v1/chat` - Web UI chat endpoint with streaming
  - Request format: Form-encoded `message`, optional `stream`
  - Response: HTML (non-streaming) or SSE (streaming)

- `GET /health` - Health check
  - Returns: JSON with status, model, backend, uptime, active connections

- `GET /` - Built-in web UI
  - Returns: Single-page HTML application
  - Features: Chat interface, conversation management, Markdown rendering

**CORS & Security:**
- All endpoints: `Access-Control-Allow-Origin: *` (CORS enabled)
- Headers: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, CSP
- Max request body: 1 MB
- Max message length: 100 KB
- Max concurrent connections: 64

## Data Storage

**Model Weights (Read-Only):**
- **GGUF Format** (.gguf files)
  - Location: User-specified file path
  - Loading: Memory-mapped via `std.os.mmap` (zero-copy on all platforms)
  - Metadata: Key-value pairs for model config, tokenizer vocab
  - Tensors: Named weight matrices with quantization metadata
  - Implementation: `src/format/gguf.zig`

- **SafeTensors Format** (directory structure)
  - Location: User-specified directory with `.safetensors` shard files
  - Structure:
    - `model.safetensors.index.json` - Tensor → shard mapping
    - `*.safetensors` - Shard files (memory-mapped)
    - `config.json` - Model architecture metadata (JSON)
    - `tokenizer.json` - Vocabulary and merge rules (JSON)
  - Loading: Sequential mmap per shard (zero-copy)
  - Implementation: `src/format/safetensors.zig`

**KV Cache (Volatile, Per-Inference):**
- **In-Memory Storage**: Allocated by backend, freed on model reset
  - CPU: `std.heap.page_allocator` allocation
  - Metal/Vulkan/CUDA/ROCm: GPU device memory or unified memory
- **Strategies**:
  - **Flat**: Single contiguous buffer per layer (simple, fixed size)
  - **Paged**: Block-based allocation with page tables (better reuse, continuation support)
  - **Radix Tree**: Prefix-sharing with automatic LRU eviction (production-grade, multi-request support)
- **Quantization Types** (configurable via CLI):
  - f32 (default: full precision)
  - f16 (half precision)
  - q8_0 (8-bit quantization)
  - fp8_e4m3, fp8_e5m2 (8-bit floating point)
  - nvfp4 (4-bit NVIDIA microscaling, Blackwell+)
  - mxfp4 (4-bit standard microscaling)

**Tokenizer Data:**
- **BPE Vocabulary** (token ID → string mapping)
  - Source: GGUF metadata or `tokenizer.json`
  - Loaded into memory as string array at init
  - Used for: Encode (text → token IDs), decode (token IDs → text)

- **Merge Rules** (BPE subword merges)
  - Format: Array of "A B" merge instructions
  - Used during tokenization for merging subword units

**Conversation Storage (Web UI Only):**
- In-memory `ArrayList` per active conversation
  - Max conversations: 100
  - Max messages per conversation: 1000
  - Cleared on server restart (volatile)
- Format: Structured Message objects (role + content)

**File System Access:**
- **mmap (read-only)**: Model weights via `std.os.mmap`
- **madvise**: Sequential/random access hints to kernel for prefetch optimization
- **sendfile** (future): Zero-copy network serving if enabled
- **No persistent storage**: All data is in-memory or from mmap'd files

## Authentication & Identity

**Auth Provider:**
- **None** - No authentication layer
- Access: Open HTTP server (suitable for localhost/trusted networks)
- Recommended: Network isolation via firewall or reverse proxy with auth (nginx, Caddy, etc.)

**Future Consideration:**
- OpenAI API key validation could be added to `src/server.zig` at endpoint entry points
- Would require request header parsing: `Authorization: Bearer sk-...`

## Monitoring & Observability

**Error Tracking:**
- **Logging**: `std.log` with scoped output (e.g., `.perf` scope for performance metrics)
- **stderr**: Errors and warnings written to stderr
- **stdout**: Inference output and generation stats
- No external error service integration

**Performance Metrics (Instrumentation):**
- Per-request timing: `std.time.milliTimestamp()` for prefill/decode/total duration
- Per-token throughput: tokens/sec calculated inline
- Optional profiling: `--profile` CLI flag instruments individual GPU operations
- Memory usage: Tracked by allocator (during development with `std.testing.allocator`)
- No external APM service integration

**Logs:**
- Destination: stderr and stdout via `std.fs.File`
- Buffering: `std.io.BufferedWriter` for batched writes
- Request logging: HTTP method, path, duration, status code

## CI/CD & Deployment

**Hosting:**
- **Self-Hosted Only** - Pure Zig binary, no cloud dependencies
- Deployment: `zig build` produces standalone executable
- Distribution: Single binary `zig-out/bin/agave`

**Build System:**
- **Zig Build** (`build.zig`) - Cross-compilation supported
  - CPU architecture detection via `std.Target.Cpu.Feature`
  - Conditional kernel compilation (CUDA, ROCm, Vulkan)

**CI Pipeline:**
- **Not defined in codebase** - No GitHub Actions/GitLab CI configuration
- Manual testing via local `zig build` and `zig build test`
- Cross-compilation tested: Linux x86_64, Linux aarch64, macOS aarch64

## Environment Configuration

**Required Environment Variables:**
- None (all config via CLI arguments)

**Optional Environment Variables:**
- CUDA:
  - `CUDA_PATH` (if non-standard CUDA install location)
  - `LD_LIBRARY_PATH` (for libcuda.so discovery)
- ROCm:
  - `HIP_PATH` (if non-standard HIP install location)
  - `LD_LIBRARY_PATH` (for libamdhip64.so discovery)
- Vulkan:
  - `VK_LAYER_PATH` (for validation layers in debug builds)
  - `LD_LIBRARY_PATH` (for libvulkan.so discovery)

**Secrets Location:**
- No secrets stored — all credentials/API keys are external (user's responsibility)
- Example: If using reverse proxy with auth, credentials are in proxy config, not agave config

## Webhooks & Callbacks

**Incoming Webhooks:**
- None - Server is request-response only, no inbound webhook support

**Outgoing Webhooks:**
- None - Server does not make external requests during inference
- HTTP server is fully self-contained

**Signal Handling:**
- `SIGINT` (Ctrl+C): Gracefully cancels the current inference operation
  - Implemented via `std.posix.sigaction()` in `src/server.zig`
  - Handler calls `model.cancel()` to set a cancellation flag
- `SIGTERM`: Default handler (process termination)

## Model Format & Loading

**GGUF Metadata:**
- Read via VTable interface: `format.getMetaStr()`, `format.getMetaU32()`, etc.
- Parsed fields:
  - Model architecture (gemma, qwen, llama, etc.)
  - Quantization type per tensor
  - Tokenizer metadata (vocab size, special tokens)
  - Chat template (role markers, EOG tokens)
- Implementation: `src/format/gguf.zig` (binary parsing)

**SafeTensors Metadata:**
- Loaded from `config.json` and `tokenizer.json` (JSON format)
- Supports multimodal model configs (nested text_config)
- HuggingFace-compatible name mapping via `ggufToHfName()`
- Implementation: `src/format/safetensors.zig` (JSON parsing)

**Dynamic Model Detection:**
- Architecture inferred from metadata keys: `*.block_count`, `*.hidden_size`, etc.
- Falls back to filename heuristics if metadata unavailable
- Implementation: `src/arch.zig` and model-specific loaders

## Compute Backend Backends & Their Integrations

**CPU Backend (Always Available):**
- Uses system SIMD: `@Vector` intrinsics for vectorization
- Thread pool for multi-core parallelization
- No external library dependency

**Metal Backend (macOS):**
- **Integration Method**: ObjC runtime via custom bindings in `src/backend/objc.zig`
- **Metal Framework**: Linked at compile-time via `linkFramework("Metal", "Foundation")`
- Kernel compilation: MSL source embedded in binary, compiled at runtime
- Device selection: Automatic (default GPU device via Metal API)

**CUDA Backend (Linux/Windows):**
- **Integration Method**: Dynamic library loading via `std.DynLib.open("libcuda.so")`
- **NVIDIA Driver**: Must be installed; no link-time dependency
- **PTX Kernels**: Cross-compiled from Zig source to PTX via `zig build ptx`, embedded in binary
- **Runtime Compilation**: PTX loaded by NVIDIA driver at runtime
- Device selection: First available GPU

**ROCm Backend (Linux):**
- **Integration Method**: Dynamic library loading via `std.DynLib.open("libamdhip64.so")`
- **AMD Driver**: Must be installed; no link-time dependency
- **HSACO Kernels**: Cross-compiled from Zig to AMDGCN, linked into HSACO object file
- **Runtime Loading**: HSACO loaded via HIP runtime at runtime
- Device selection: First available GPU

**Vulkan Backend (Cross-Platform):**
- **Integration Method**: Dynamic library loading via `std.DynLib.open()`
  - `libvulkan.so` on Linux, `libMoltenVK.dylib` on macOS (MoltenVK), `vulkan-1.dll` on Windows
- **Vulkan SDK**: Optional at runtime; if unavailable, graceful fallback to CPU
- **SPIR-V Shaders**: Precompiled and embedded as binary constants
- Device selection: First available Vulkan device

---

*Integration audit: 2026-03-21*
