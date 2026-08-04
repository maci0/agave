# Chapter 11: Metal Backend Internals

**Prerequisites:** [Chapter 8: Backends](08-backends.md) (UMA, dispatcher pattern)

**Time:** ~19 min

The Metal backend is Agave's primary GPU path on Apple Silicon. It's designed around **zero-copy UMA** (Unified Memory Architecture — CPU and GPU share the same physical RAM), **deferred dispatch** (batching operations without blocking), and **cache-aware resource management** (reusing GPU buffer wrappers to avoid ObjC allocation overhead).

## Unified Memory Architecture (UMA)

On Apple Silicon (M1, M2, M3, M4), the CPU and GPU share the **same physical DRAM** — there's no separate VRAM. This is different from discrete GPUs (NVIDIA, AMD) where data must be copied between host RAM and GPU memory.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    H1["Host RAM\n(CPU allocation)"]:::danger
    V1["VRAM\n(GPU memory)"]:::danger
    Shared["Shared DRAM\n(~400 GB/s)"]:::success
    CPU["CPU"]:::setup
    GPU["GPU"]:::setup

    subgraph Discrete["Discrete GPU (NVIDIA/AMD)"]
        direction LR
        H1 -->|"memcpy D2H/H2D\n(PCIe, ~32 GB/s)"| V1
    end

    subgraph UMA["Apple Silicon UMA"]
        direction LR
        CPU -->|"pointer read/write"| Shared
        GPU -->|"pointer read/write"| Shared
    end
```

**Implications:**

- **Zero-copy buffer wrapping:** CPU allocations can be used directly by the GPU via `MTLBuffer.newBufferWithBytesNoCopy()`
- **No D2H transfers:** When the GPU writes data, the CPU sees it immediately (after `sync()` flushes the command buffer)
- **Shared bandwidth:** CPU and GPU compete for the same memory bus (~400 GB/s on M4 Pro)

**Metal buffer creation:**

```objc
// Wrap existing CPU allocation (zero copy)
id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                  length:len
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
```

**Storage modes:**

- `MTLResourceStorageModeShared` — CPU and GPU both access the same memory (UMA)
- `MTLResourceStorageModePrivate` — GPU-only (used for scratch buffers)
- `MTLResourceStorageModeManaged` — Discrete GPU mode (not used on Apple Silicon)

Agave wraps all model weights (mmap'd from GGUF/SafeTensors) and activation buffers as **shared** Metal buffers.

## Buffer Caching

Creating a `MTLBuffer` wrapper involves ObjC allocation and reference counting. Doing this **every dispatch** (800+ times per token) adds 10-15% overhead.

**Solution:** Cache `MTLBuffer` objects by their host pointer address.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Call["getBufRef(ptr, len)"]:::setup
    Hash["Compute addr = @intFromPtr(ptr)"]:::migration
    Return["Return cached MTLBuffer\n+ recomputed offset"]:::success
    Align["Align ptr down to\npage boundary"]:::migration
    Wrap["newBufferWithBytesNoCopy\n(zero-copy, Shared mode)"]:::sync
    Store["buf_cache.put(aligned_base, info)"]:::migration
    Kernel["Pass BufRef{buf, offset}\nto Metal compute encoder"]:::success

    Call --> Hash
    Hash --> Lookup{"buf_cache.get(aligned_base)?"}

    Lookup -->|"Hit (stable weight pointer)"| Return
    Lookup -->|"Miss (first access)"| Align
    Align --> Wrap
    Wrap --> Store
    Store --> Return

    Return --> Kernel
```

### Cache Structure

```text
type BufferInfo:
    metal_buf: MTLBuffer
    len: usize

buf_cache: HashMap(host_pointer -> BufferInfo)
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`BufferInfo`, `buf_cache`)

### Lookup Pattern

```text
getBufRef(ptr, len):
    addr = intFromPtr(ptr)
    aligned_base = addr & ~(page_size - 1)
    offset = addr - aligned_base

    if buf_cache.get(aligned_base) is Some(cached):
        return BufRef{ buf: cached.metal_buf, offset: offset }

    # not cached: wrap from the aligned base (zero-copy)
    aligned_ptr = ptrFromInt(aligned_base)
    aligned_len = roundUp(offset + len, page_size)
    buf = device.newBufferWithBytesNoCopy(aligned_ptr, aligned_len, StorageModeShared)
          or panic("Metal buffer creation failed")

    buf_cache.put(aligned_base, { metal_buf: buf, len: aligned_len })
    return BufRef{ buf: buf, offset: offset }
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`getBufRef`)

**When to cache:**

- ✅ **Stable pointers:** mmap'd model weights, model activation buffers (allocated once at init)
- ❌ **Temporary allocations:** Scratch buffers that get freed and reallocated

### Page Alignment Requirement

`newBufferWithBytesNoCopy` requires the pointer to be **page-aligned** (multiple of 16384 bytes (16 KB) on Apple Silicon (aarch64 macOS)). If you pass a mid-page pointer, buffer creation fails.

**Workaround:** Wrap the **entire page range** and use an offset:

```text
page_size = 16384                              # Apple Silicon page size
page_mask = ~(page_size - 1)

page_base = intFromPtr(ptr) & page_mask         # round down to page boundary
offset    = intFromPtr(ptr) - page_base         # offset within page

aligned_ptr = ptrFromInt(page_base)
aligned_len = roundUp(offset + len, page_size)

buf = device.newBufferWithBytesNoCopy(aligned_ptr, aligned_len, ...)
return BufRef{ buf: buf, offset: offset }
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`getBufRef`)

**Example:**

- Pointer: `0x100008800` (not page-aligned, within a 16384-byte page)
- Page base: `0x100004000` (aligned to 16384)
- Offset: `0x4800` (18432 bytes into the page)
- Metal sees: buffer starting at `0x100004000`, length = rounded up to next 16 KB boundary
- Kernel access: `buffer[offset + idx]` to skip the leading bytes

**Cost:** Negligible — Metal doesn't copy the page, just references it with an offset.

## Command Buffer Batching

Metal kernels are dispatched via **command buffers** — sequences of GPU operations that execute together. Creating a new command buffer for every kernel would serialize execution and waste CPU time.

**Pattern:** Maintain a **persistent command buffer** and **compute encoder** across multiple dispatches.

```mermaid
sequenceDiagram
    participant Model as Model (Zig)
    participant BE as MetalBackend
    participant GPU as GPU Hardware

    Note over Model,GPU: Token generation forward pass

    Model->>BE: gemv(x, W_q, ...)
    BE->>BE: active_enc == nil: create CommandBuffer + Encoder
    BE->>BE: encode kernel #1 (GEMV)
    BE->>BE: insert memoryBarrier

    Model->>BE: gemv(x, W_k, ...)
    BE->>BE: reuse active_enc
    BE->>BE: encode kernel #2 (GEMV)
    BE->>BE: insert memoryBarrier

    Note over Model,BE: ...20+ more dispatches, same command buffer...

    Model->>BE: sync()
    BE->>BE: endEncoding()
    BE->>GPU: commit() — submit all work at once
    GPU-->>BE: waitUntilCompleted()
    BE->>BE: active_enc = nil, active_cmd = nil

    Model->>Model: argmax(logits) — safe to read GPU output
```

### Active Command Buffer State

```text
type MetalBackend:
    queue:      MTLCommandQueue
    active_cmd: MTLCommandBuffer?
    active_enc: MTLComputeCommandEncoder?
    ...
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`MetalBackend`, `active_cmd`, `active_enc`)

### Encode Pattern

```text
encode(self, pipeline, buffers, grid):
    if self.active_enc is None:                       # lazy creation on first dispatch
        self.active_cmd = queue.commandBuffer() or panic
        self.active_enc = active_cmd.computeCommandEncoder() or panic

    enc = self.active_enc
    enc.setComputePipelineState(pipeline)
    for i, buf_ref in enumerate(buffers):
        enc.setBuffer(buf_ref.buf, buf_ref.offset, atIndex: i)

    enc.dispatchThreadgroups(grid.threadgroups, grid.threads_per_group)

    if not self.batch_mode:                            # barrier unless batching
        enc.memoryBarrierWithScope(Buffers)
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`encode`)

**Key points:**

- **Lazy creation:** Command buffer + encoder created on first `encode()` call
- **Persistent encoder:** Stays open across multiple kernel dispatches
- **Memory barriers:** Inserted between dispatches to ensure writes are visible to subsequent reads

### Flush (Commit and Wait)

```text
flush(self):
    if self.active_enc is Some(enc):
        enc.endEncoding()
        self.active_cmd.commit()               # submit to GPU
        self.active_cmd.waitUntilCompleted()    # block until done
        self.active_enc = None
        self.active_cmd = None
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`flush`)

**When to flush:**

- **Before CPU reads GPU data:** `be.sync()` calls `flush()` before argmax, embedding lookup, etc.
- **Not after every dispatch:** Batching is the whole point

**Example:** Forward pass for one token (20+ dispatches) uses **1 command buffer**, flushed once at the end before argmax.

## Batch Mode: Suppressing Intermediate Barriers

Metal's memory barrier (`memoryBarrierWithScope`) ensures write visibility but **serializes execution** — the GPU must finish all prior work before starting the next kernel.

**Problem:** Independent operations (e.g., normalizing Q and K in parallel) don't need a barrier between them.

**Solution:** `beginBatch()` / `endBatch()` to suppress barriers and insert one at the end.

```mermaid
flowchart LR
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    A1["rmsNorm Q"]:::danger
    A2["rmsNorm K"]:::danger
    A3["rmsNorm V"]:::danger
    A4["next op"]:::danger
    B1["rmsNorm Q"]:::sync
    B2["rmsNorm K"]:::sync
    B3["rmsNorm V"]:::sync
    B4["next op"]:::success

    subgraph Sequential["Without beginBatch\n(many barriers/token)"]
        direction LR
        A1 -->|barrier| A2 -->|barrier| A3 -->|barrier| A4
    end

    subgraph Batched["With beginBatch/endBatch\n(one barrier for the group)"]
        direction LR
        B1 & B2 & B3 -->|"single barrier\n(endBatch)"| B4
    end
```

### API

```text
beginBatch(self):
    self.batch_mode = true

endBatch(self):
    self.batch_mode = false
    if self.active_enc is Some(enc):
        enc.memoryBarrierWithScope(Buffers)     # single barrier for the group
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`beginBatch`, `endBatch`)

### Usage Example

```text
# Normalize Q and K in parallel (independent operations)
be.beginBatch()
  be.rmsNormMulti(q_buf, norm_w, nh_q, hd, eps)    # no barrier after
  be.rmsNormMulti(k_buf, norm_w, nh_kv, hd, eps)   # no barrier after
be.endBatch()                                      # single barrier here

# vs sequential (default):
be.rmsNormMulti(q_buf, norm_w, nh_q, hd, eps)    # barrier after
be.rmsNormMulti(k_buf, norm_w, nh_kv, hd, eps)   # barrier after
```

**Impact:**

- **Qwen3.5-class models:** Barrier count drops when independent norms/RoPE share one `endBatch` (illustrative `--profile` counters, not a `BENCHMARKS.md` row)
- **Throughput change:** often near zero on Apple Silicon (GPUs overlap work even with barriers; they are essentially free)

**Why track it anyway?**

- **Diagnostics:** Profiling counters show barrier count (`--profile` flag)
- **Future platforms:** On some GPUs (Intel, older AMD), barriers are expensive
- **Documentation:** Makes the independence explicit in the code

## Deferred Dispatch Contract

**Key principle:** GPU operations are **deferred** — they're encoded into the command buffer but don't execute immediately.

**Implications:**

1. **No sync after every dispatch:** Models call `be.sync()` only when CPU needs GPU data
2. **Host pointers must stay valid:** Buffers referenced in the command buffer must not be freed until the buffer commits
3. **Stale reads without sync:** CPU reads of GPU-written data see old values until `sync()` flushes

### Example: Argmax After Logits

```text
be.gemv(x, lm_head, logits, vocab_size, n_embd)   # compute logits on GPU (deferred)

# WRONG: read logits on CPU immediately (stale data!)
token = argmax(logits)     # reads old logits, not the new ones

# CORRECT: sync first to flush GPU writes
be.sync()                  # commit command buffer, wait for completion
token = argmax(logits)     # now reads the correct logits
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`sync`)

### Sync Points in Forward Pass

For a typical model forward pass:

1. **Embedding lookup:** `be.sync()` before CPU reads embedding table (GPU may have written it)
2. **... (all layers run on GPU, no sync)**
3. **Final argmax:** `be.sync()` before CPU reads logits

**Total syncs per token:** 1-2 (depending on whether embedding is GPU-resident).

**Optimization:** Qwen3.5 eliminated 16 syncs/token by moving Q/gate split from CPU (memcpy) to GPU (kernel) → 15% throughput gain.

### Deferred Dispatch Timeline

```mermaid
sequenceDiagram
    participant CPU as CPU Thread
    participant CMB as Command Buffer<br/>(GPU queue)
    participant GPU as GPU Hardware

    Note over CPU,GPU: Forward pass begins — GPU ops encoded, not yet executing

    CPU->>CMB: encode gemv(W_q) [deferred]
    CPU->>CMB: encode gemv(W_k) [deferred]
    CPU->>CMB: encode gemv(W_v) [deferred]
    CPU->>CMB: encode sdpa(q,k,v) [deferred]

    Note over CPU: CPU tries to read logits here WITHOUT sync
    CPU->>CPU: argmax(logits) — READS STALE DATA (write hazard)

    Note over CPU,GPU: Correct pattern: sync before CPU read

    CPU->>CMB: encode gemv(lm_head) [deferred]
    CPU->>CMB: commit() — submit all at once
    CMB->>GPU: execute all encoded ops
    GPU-->>CPU: waitUntilCompleted() — CPU blocks here
    Note over CPU: GPU writes are now visible in UMA shared memory
    CPU->>CPU: argmax(logits) — CORRECT: reads fresh logits

    Note over CPU,GPU: Total syncs per token: 1-2 (not 800+)
```

## Threadgroup Memory Limits

Metal has a **per-threadgroup memory limit** of 32 KB on Apple Silicon. If your kernel allocates more, pipeline creation **fails silently** (returns `nil` without logging an error).

### Debugging Pipeline Creation

```text
makePipeline(self, name):
    fn_name = NSString.stringWithUTF8String(name) or return error.StringFailed

    function = self.library.newFunctionWithName(fn_name)
               or { log.err("Metal kernel not found: %s", name); return error.KernelNotFound }

    pipeline, err = self.device.newComputePipelineStateWithFunction(function)
    if pipeline is None:
        if err is Some(e):
            log.err("Metal pipeline creation error: %s", e.localizedDescription)  # the only place
        return error.PipelineFailed                                                # this error surfaces

    return pipeline
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`makePipeline`)

**Key:** Check the error object and log `localizedDescription` to see the actual Metal error (often "threadgroup memory exceeded").

### SDPA Threadgroup Memory Budget

Example from `src/backend/kernels/metal/sdpa.metal`:

```metal
kernel void sdpa(
    // ...
) {
    threadgroup float q_local[256];              // 1 KB (sdpa_max_head_dim)
    threadgroup float kv_block[16 * 256];        // 16 KB (sdpa_block_size × sdpa_max_head_dim)
    threadgroup float scores[16];                // 64 bytes (sdpa_block_size)
    threadgroup float out_acc[256];              // 1 KB (sdpa_max_head_dim)
    threadgroup float shared[8];                 // 32 bytes (SIMD group reduction scratch)

    // Total: 1024 + 16384 + 64 + 1024 + 32 = 18.5 KB (fits under 32 KB)
}
```

**Constraints:**

- `kv_block` size determines `max_seq_len` and `max_head_dim`
- 16 positions × 256 head_dim = 16 KB (half the budget)
- Increasing to 32 positions → 32 KB (no room for other vars)
- Increasing `max_head_dim` to 512 → 32 KB (also maxed out)

**Trade-off:** Agave caps SDPA at 65536 seq_len and 256 head_dim to fit tile buffers in 32 KB threadgroup memory. Inputs that exceed those limits panic rather than silently falling back to another attention strategy.

### Threadgroup Memory Budget Breakdown

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Budget["Apple Silicon threadgroup\nmemory limit: 32 KB total"]:::setup
    KV["kv_block\nfloat[16 × 256]\n= 16 KB\n(50% of budget)"]:::danger
    Q["q_local\nfloat[256]\n= 1 KB\n(3% of budget)"]:::sync
    Out["out_acc\nfloat[256]\n= 1 KB\n(3% of budget)"]:::sync
    Scores["scores\nfloat[16]\n= 64 bytes\n(0.2% of budget)"]:::sync
    Shared["shared\nfloat[8]\n= 32 bytes\n(0.1% of budget)"]:::sync
    Free["headroom\n~13.4 KB remaining\n(42% of budget)"]:::success
    Constraint["Constraint: 16 positions × 256 head_dim\nIncreasing to 32 pos → 32 KB (maxed)\nIncreasing head_dim to 512 → 32 KB (maxed)"]:::migration
    Fallback["seq_len > 65536 or head_dim > 256\npanics (no silent fallback)"]:::danger

    Budget --> KV
    Budget --> Q
    Budget --> Out
    Budget --> Scores
    Budget --> Shared
    Budget --> Free

    KV --> Constraint
    Free --> Fallback
```

## Profiling Counters

The Metal backend tracks dispatch/barrier/sync counts when `profile_counters` is enabled (via `--profile` flag).

```text
type MetalBackend:
    dispatch_count:    u32 = 0
    barrier_count:     u32 = 0
    sync_count:        u32 = 0
    profile_counters:  bool = false
    ...

encode(...):
    ... dispatch kernel ...
    if self.profile_counters: self.dispatch_count += 1

flush(...):
    ... commit and wait ...

sync(...):
    self.flush()
    if self.profile_counters: self.sync_count += 1
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`dispatch_count`, `barrier_count`, `sync_count`, `profile_counters`)

**Usage:**

```bash
./zig-out/bin/agave model.gguf --profile "Test prompt"
# Output (first decode token when --profile is on):
# Metal stats: <dispatches>, <barriers>, <syncs>
```

**Optimization insights:**

- High sync count → CPU/GPU ping-pong (move work to GPU or batch better)
- High barrier count → opportunity for batch mode (if operations are independent)
- Dispatch count × kernel overhead → lower bound on latency

**Example:** Moving Q/gate split to the GPU can collapse many per-layer syncs to a single end-of-token sync. Exact dispatch/barrier/sync counters come from `--profile` on your model and quant; they are not fixed numbers in `BENCHMARKS.md`.

## BF16 GEMM Kernel

The Metal backend includes a `gemm_bf16` kernel (in `gemm.metal`) for batched matrix multiplications with bf16 or f16 weights. It is used in two contexts:

1. **Vision encoder:** The vision encoder's linear projections (Q, K, V, FFN) use batched GEMM since they process all patches simultaneously (e.g., 256 patches for a 768x768 image). The `gemm()` function dispatches to `gemm_bf16` when weights are bf16 or f16.

2. **Batched prefill:** During chunked prefill, multiple prompt tokens are processed together. Each chunk's linear projections use GEMM instead of per-token GEMV.

```text
gemm(self, x, w, y, n_tok, n_out, n_in):
    if n_tok <= 1:
        self.gemv(x, w, y, n_out, n_in)      # single token -> GEMV
        return

    pipeline = match w.dtype:
        f32          => pipe_gemm_f32
        bf16 | f16   => pipe_gemm_bf16
        q8_0         => pipe_gemm_q8_0
        q4_0         => pipe_gemm_q4_0
        q4_k         => pipe_gemm_q4_k
        q5_k         => pipe_gemm_q5_k
        q6_k         => pipe_gemm_q6_k
        other        => panic("Metal GEMM: unsupported dtype, add GPU kernel")

    # ... encode dispatch with one threadgroup per output row
```

**Implementation:** [`src/backend/metal.zig`](../../src/backend/metal.zig) (`gemm`)

Each threadgroup handles one output row. The bf16 variant processes tokens sequentially without token tiling; token tiling is used in the quantized GEMM kernels (Q8_0 uses `TILE_T=8`, Q4_K uses `TILE_T=4`) to amortize weight loads across multiple input vectors.

### GEMM vs GEMV Dispatch Decision Tree

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Entry["be.gemm(x, W, y,\nn_tok, n_out, n_in)"]:::setup
    GEMV["GEMV path\ngemv(x, W, y)\n1 vector × matrix\nlow latency, no tiling"]:::success
    GemmF32["pipe_gemm_f32\n(f32 weights)\nno tiling"]:::sync
    GemmBF16["pipe_gemm_bf16\n(half-precision weights)\nsequential token loop"]:::sync
    GemmQ8["pipe_gemm_q8_0\n(8-bit quant)\nTILE_T=8 token tiling\namortize weight decode"]:::sync
    GemmQ4K["pipe_gemm_q4_k / q5_k / q6_k\n(4/5/6-bit K-quant)\nTILE_T=4 token tiling"]:::sync
    GemmQ4["pipe_gemm_q4_0\n(4-bit quant)\nTILE_T=4 token tiling"]:::sync
    Panic["@panic\nno CPU fallback\nadd GPU kernel first"]:::danger
    Threadgroup["1 threadgroup per output row\nencode → barrier → next kernel"]:::setup

    Entry --> TokCheck{"n_tok == 1?\n(token generation)"}

    TokCheck -->|"yes — single token"| GEMV
    TokCheck -->|"no — batch / prefill"| DtypeCheck{"Weight dtype?"}

    DtypeCheck -->|"f32"| GemmF32
    DtypeCheck -->|"bf16 / f16"| GemmBF16
    DtypeCheck -->|"q8_0"| GemmQ8
    DtypeCheck -->|"q4_k / q5_k / q6_k"| GemmQ4K
    DtypeCheck -->|"q4_0"| GemmQ4
    DtypeCheck -->|"other"| Panic

    subgraph Output["All paths"]
        Threadgroup
    end

    GemmF32 & GemmBF16 & GemmQ8 & GemmQ4K & GemmQ4 --> Threadgroup
    GEMV --> Threadgroup
```

## Vision Encoder GPU Acceleration

When a vision encoder (mmproj) is loaded, its transformer blocks run on the GPU via the standard `gemm()` dispatch. The vision encoder calls `be.gemm()` for all linear projections (Q/K/V/O, FFN up/gate/down, output projection), which dispatches to the appropriate Metal kernel based on weight dtype — f32, bf16, q8_0, q4_0, q4_k, q5_k, or q6_k.

The key synchronization pattern: `be.sync()` is required between GPU GEMM operations and CPU operations (like softmax or activation functions that run on the CPU thread pool). Without it, the CPU reads stale data from shared UMA memory.

```mermaid
sequenceDiagram
    participant GPU as GPU (Metal)<br/>gemm projections
    participant UMA as Shared DRAM<br/>(UMA buffer)
    participant CPU as CPU Thread Pool<br/>attention + softmax

    Note over GPU,CPU: Vision encoder — one transformer block (n_patches tokens)

    GPU->>UMA: gemm(hidden, W_q) → q_buf [encoded, deferred]
    GPU->>UMA: gemm(hidden, W_k) → k_buf [encoded, deferred]
    GPU->>UMA: gemm(hidden, W_v) → v_buf [encoded, deferred]
    GPU->>GPU: commit + waitUntilCompleted (be.sync #1)
    Note over UMA: Q/K/V writes now visible to CPU

    UMA->>CPU: read q_buf, k_buf, v_buf
    CPU->>CPU: compute n_patches × n_patches attention scores
    CPU->>CPU: softmax(scores) — full non-causal attention
    CPU->>CPU: weighted sum → attn_out
    CPU->>UMA: write attn_out

    GPU->>UMA: gemm(attn_out, W_o) → hidden [encoded, deferred]
    GPU->>GPU: commit + waitUntilCompleted (be.sync #2)
    Note over UMA: Output projection visible to CPU

    UMA->>CPU: read hidden (for residual add)
    CPU->>UMA: write hidden + residual

    Note over GPU,CPU: FFN layers follow same GPU/CPU interleave pattern
    Note over GPU,CPU: GEMM projections (bulk compute) run on GPU<br/>Softmax + score matrix run on CPU thread pool
```

This interleaving is necessary because the vision encoder uses full (non-causal) attention with `n_patches x n_patches` score matrices, which currently runs on the CPU. The GEMM projections — the bulk of the compute — run on the GPU.

## Best Practices

### Resource Management

1. **Cache stable buffers:** mmap'd weights, activation buffers
2. **Don't cache temps:** Scratch allocations with varying lifetimes
3. **Pre-allocate at init:** Command queue, pipelines, scratch buffers

### Dispatch Efficiency

1. **Batch independent ops:** Use `beginBatch()` / `endBatch()` to suppress intermediate barriers
2. **Minimize syncs:** Only sync when CPU needs GPU data
3. **Fuse kernels:** Combine sequential ops (e.g., `addRmsNorm`) to reduce dispatches
4. **Megakernel pipelines:** The `--megakernel` flag enables a three-tier fusion system. **Tier 1** (fused FFN) combines gate GEMV + up GEMV + activation into a single dispatch (3->1 per FFN layer) via 11 kernels in `megakernel.metal` (SiLU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, MLX_Q4} + GELU x {Q8_0, Q4_K, Q5_K, Q6_K, Q4_0}). **Tier 2** (true megakernels) executes entire transformer layers in a single dispatch using 18 composable building blocks in `mega_common.metal` with atomic counter grid sync (`mega_grid_sync`). **Tier 3** (composed megakernels) auto-generates model-specific MSL at runtime via `mega_compose.zig`: the `composeMSL()` function produces MSL source from a `ModelDesc` struct, then `compileComposedMegakernel()` compiles it via `newLibraryWithSource`. This enables megakernel support for new models without writing any shader code -- just a `ModelDesc` definition. The Metal backend compiles **71 MSL pipelines** total (standard ops + fused FFN + 5 true megakernels + 1 runtime-composed). See [Chapter 13](13-batched-dispatch-and-fusion.md) for details.

```mermaid
flowchart TD
    classDef setup     fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef sync      fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef migration fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef success   fill:#bbf7d0,stroke:#16a34a,color:#14532d
    classDef danger    fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef optional  fill:#f3e8ff,stroke:#9333ea,color:#581c87

    Flag["--megakernel flag\nenables fusion system"]:::setup
    T1Desc["3 dispatches → 1 dispatch per FFN layer\ngate GEMV + up GEMV + activation fused\n11 kernels: SiLU×6 dtypes + GELU×5 dtypes\nDtypes: Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, MLX_Q4"]:::sync
    T2Desc["Entire transformer layer = 1 dispatch\n18 composable building blocks\nAtomic counter grid sync (mega_grid_sync)\n5 pre-compiled megakernel pipelines"]:::setup
    T3Desc["MSL source generated at runtime\nfrom ModelDesc struct definition\ncomposeMSL() → compileComposedMegakernel()\nnewLibraryWithSource() JIT compilation\nNew models: zero shader code needed"]:::optional
    Total["71 total MSL pipelines compiled\n(standard ops + Tier 1 + Tier 2 + Tier 3)"]:::migration
    Benefit["Reduced dispatch overhead\nImproved GPU utilization\nLower CPU encoding cost per token"]:::success

    Flag --> T1
    Flag --> T2
    Flag --> T3

    subgraph T1["Tier 1 — Fused FFN\n(megakernel.metal)"]
        T1Desc
    end

    subgraph T2["Tier 2 — True Megakernels\n(mega_common.metal)"]
        T2Desc
    end

    subgraph T3["Tier 3 — Composed Megakernels\n(mega_compose.zig)"]
        T3Desc
    end

    T1Desc --> Total
    T2Desc --> Total
    T3Desc --> Total

    Total --> Benefit
```

### Debugging

1. **Log pipeline errors:** Check `localizedDescription` on creation failure
2. **Profile counters:** Enable `--profile` to see dispatch/barrier/sync counts
3. **Validate threadgroup memory:** Sum all `threadgroup` vars, ensure < 32 KB

### UMA-Specific

1. **Wrap, don't copy:** Use `newBufferWithBytesNoCopy` for CPU allocations
2. **Shared storage mode:** Always use `MTLResourceStorageModeShared` on Apple Silicon
3. **Page alignment:** Handle mid-page pointers with base+offset pattern

## Performance (from BENCHMARKS.md)

Measured 2026-03-24 on Apple M4 Pro (14-core CPU, 20-core GPU), full methodology in [BENCHMARKS.md](../BENCHMARKS.md).

| Claim | Source |
|-------|--------|
| Agave is 1.2-1.7x faster than llama.cpp on Metal for decode on supported quant formats | BENCHMARKS Decode Throughput, Notes |
| Qwen3.5 9B Q8_0 Metal decode: 41.7 tok/s vs llama.cpp Metal 25.0 tok/s (1.67x) | BENCHMARKS Decode Throughput, M4 Pro |
| Gemma 3 12B Q8_0 Metal decode: 22.3 tok/s vs llama.cpp Metal 18.7 tok/s (1.19x) | BENCHMARKS Decode Throughput, M4 Pro |
| Qwen3.5 0.8B Q8_0 Metal decode: 125 tok/s (sparse GEMV + Accelerate.framework, updated 2026-05-26) | BENCHMARKS Decode Throughput, M4 Pro |

## Gotchas

**Threadgroup memory failures are silent.** Metal's `newComputePipelineStateWithFunction:error:` returns `nil` when a kernel's `threadgroup` variables exceed the 32 KB per-threadgroup limit on Apple Silicon, but nothing about that call raises an exception or prints a message on its own. `makePipeline()` (shown above) only surfaces the real cause because it explicitly reads the returned `NSError` and logs `localizedDescription` before returning `error.PipelineFailed`. Skip that check, or catch the error generically without inspecting it, and pipeline creation looks like it "just failed" with no clue that a `threadgroup` array was the reason. The SDPA kernel lives right at this edge: `q_local + kv_block + out_acc + scores + shared` must stay under 32 KB, and `kv_block` (16 KB for `sdpa_block_size x sdpa_max_head_dim`) is both the largest contributor and the first one to blow the budget if `max_seq_len` or `head_dim` grow.

---

**In the code:** [src/backend/metal.zig](../../src/backend/metal.zig) (full implementation), [src/backend/kernels/metal/*.metal](../../src/backend/kernels/metal/) (MSL kernels), [src/backend/objc.zig](../../src/backend/objc.zig) (Objective-C bindings)

**Related:** [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), [Metal Programming Guide (archived)](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/), [Metal Documentation](https://developer.apple.com/documentation/metal)

**Next:** [Chapter 12: CPU Parallelism →](12-cpu-parallelism.md) | **Back:** [Chapter 10: Memory Safety ←](10-memory-safety.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)

---

## Glossary

**batch mode** — A Metal backend mode (`beginBatch`/`endBatch`) that suppresses intermediate memory barriers between independent operations.

**BufRef** — A struct containing a Metal buffer object and a byte offset, referencing sub-regions within a page-aligned buffer.

**buffer caching** — Storing MTLBuffer wrappers keyed by host pointer to avoid repeated ObjC allocation overhead.

**command buffer batching** — Maintaining a persistent command buffer and encoder across dispatches, committing all work at once.

**memory barrier** — A GPU synchronization primitive ensuring write visibility between kernel dispatches.

**ModelDesc** — A struct describing model architecture used by `mega_compose.zig` to auto-generate megakernel MSL.

**page alignment** — The requirement that `newBufferWithBytesNoCopy` pointers be aligned to 16 KB page boundaries on Apple Silicon.

**profiling counters** — Runtime counters (dispatch_count, barrier_count, sync_count) tracked when `--profile` is enabled.

**threadgroup memory** — Fast on-chip shared memory accessible by all threads in a threadgroup; limited to 32 KB on Apple Silicon.

**token tiling** — GEMM optimization where multiple input tokens share a single weight load (e.g., TILE_T=8 for Q8_0).

**zero-copy buffer wrapping** — Creating a Metal GPU buffer that references existing CPU memory without copying data, via `newBufferWithBytesNoCopy`.
