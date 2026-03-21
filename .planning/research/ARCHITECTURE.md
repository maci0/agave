# Architecture Patterns for Production LLM Inference

**Domain:** Production LLM inference engine with continuous batching and prefix caching
**Researched:** 2026-03-21
**Context:** Integration patterns for adding production serving to an existing Zig-based inference engine

## Recommended Architecture

Production LLM inference systems separate concerns into distinct layers with clear boundaries. This architecture is based on proven patterns from vLLM, SGLang, and other 2026 production systems.

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (HTTP/gRPC)                    │
│  - OpenAI-compatible endpoints                               │
│  - SSE streaming                                             │
│  - Request parsing/validation                                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Request Manager                            │
│  - Tokenization/Detokenization                               │
│  - Request queue management                                  │
│  - Async stream handling                                     │
│  - Timeout/cancellation                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                      Scheduler                               │
│  - Continuous batching logic                                 │
│  - Cache-aware scheduling (prefix hit optimization)          │
│  - Resource allocation (KV blocks, memory)                   │
│  - Preemption/eviction policy                                │
└─────┬──────────────┴──────────────┬────────────────────────┘
      │                             │
      │                    ┌────────▼────────┐
      │                    │  KV Cache Mgr   │
      │                    │  - Block alloc   │
      │                    │  - RadixTree     │
      │                    │  - LRU eviction  │
      │                    └────────┬────────┘
      │                             │
┌─────▼─────────────────────────────▼────────────────────────┐
│                       Executor                               │
│  - Model forward passes                                      │
│  - GPU kernel dispatch                                       │
│  - Batch assembly (ragged batching)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Backend (GPU/CPU)                           │
│  - Fused kernels (GEMV, SDPA, norms, activations)           │
│  - Memory management (activation cache, weight buffers)     │
│  - Device-specific optimizations                            │
└─────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Owns |
|-----------|---------------|-------------------|------|
| **API Layer** | HTTP/gRPC serving, protocol translation, streaming responses | Request Manager | Connection state, HTTP buffers |
| **Request Manager** | Tokenization, request lifecycle, async streaming, timeout enforcement | API Layer, Scheduler | Token buffers, request queues, AsyncStreams |
| **Scheduler** | Continuous batching, cache-aware request ordering, resource allocation, preemption | Request Manager, KV Cache Manager, Executor | Active batch state, scheduling policy, priority queues |
| **KV Cache Manager** | Block allocation/deallocation, RadixTree maintenance, LRU eviction, prefix matching | Scheduler, Executor | KV block pool, RadixTree nodes, block tables |
| **Executor** | Model execution, batch assembly, GPU dispatch coordination | Scheduler, KV Cache Manager, Backend | Model weights, activation buffers, batch metadata |
| **Backend** | GPU kernels, memory allocation, device-specific optimization | Executor | Device memory, kernel state, BufCache |

### Data Flow

**Request Ingress (Hot Path):**
```
HTTP Request
  → API Layer (parse, validate)
  → Request Manager (tokenize, enqueue)
  → Scheduler (batch assignment, block allocation)
  → Executor (forward pass)
  → Backend (GPU kernels)
  → Executor (output tokens)
  → Request Manager (detokenize, stream)
  → API Layer (SSE chunk)
  → HTTP Response
```

**Continuous Batching Loop (Background Thread):**
```
loop:
  1. Scheduler.step():
     - Check pending requests
     - Evaluate cache hits (RadixTree prefix match)
     - Allocate KV blocks for new tokens
     - Preempt low-priority requests if OOM
     - Assemble batch (ragged, no padding)

  2. Executor.execute_model(batch):
     - Dispatch fused kernels
     - Generate next tokens
     - Update KV cache (via block tables)

  3. Complete finished requests:
     - Return outputs to AsyncStreams
     - Free KV blocks (or retain in RadixTree)
     - Pull new requests from queue
```

**Cache-Aware Scheduling:**
```
For each pending request:
  1. RadixTree.match_prefix(tokens) → hit_length
  2. Priority = f(hit_length, wait_time, request_priority)
  3. Sort batch by priority (longest prefix first)
  4. Allocate only new KV blocks (reuse cached prefix)
```

## Patterns to Follow

### Pattern 1: Iteration-Level Scheduling (Continuous Batching)

**What:** Process one token step at a time across all active requests. Insert new requests as soon as any request completes, rather than waiting for entire batch to finish.

**When:** Always — continuous batching is the production standard for 2026.

**Why:** Eliminates GPU idle time from variable output lengths. Achieves 2-3× throughput vs static batching, up to 23× vs naive implementations.

**Implementation:**
```zig
// Scheduler maintains dynamic batch
const Batch = struct {
    requests: std.ArrayList(*Request),
    max_batch_size: usize,

    pub fn step(self: *Batch, pending: *RequestQueue) !void {
        // Remove finished requests
        var i: usize = 0;
        while (i < self.requests.items.len) {
            if (self.requests.items[i].is_finished) {
                _ = self.requests.swapRemove(i);
            } else {
                i += 1;
            }
        }

        // Fill batch with new requests (cache-aware priority)
        while (self.requests.items.len < self.max_batch_size) {
            const req = pending.pop_highest_priority() orelse break;
            try self.requests.append(req);
        }
    }
};
```

**References:** [Hugging Face: Continuous Batching](https://huggingface.co/blog/continuous_batching), [Anyscale: 23x Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

### Pattern 2: RadixAttention (Automatic Prefix Caching)

**What:** Maintain KV cache for all requests in a radix tree (prefix trie). Automatically detect longest common prefix across requests, reuse cached KV blocks, and evict via LRU.

**When:** Multi-turn conversations, few-shot prompting, shared system prompts, agent workflows.

**Why:** Cache hit rates of 50-99% translate directly to lower TTFT and higher throughput. SGLang achieves up to 5× throughput vs non-caching baselines.

**Implementation:**
```zig
// RadixTree node represents consecutive token span
const RadixNode = struct {
    tokens: []const u32,           // Token span this node represents
    kv_blocks: []KVBlock,          // Physical KV cache blocks
    children: std.AutoHashMap(u32, *RadixNode),
    parent: ?*RadixNode,
    last_access: i64,              // LRU timestamp
    ref_count: usize,              // Active requests using this path

    pub fn matchPrefix(self: *RadixNode, query: []const u32) struct { node: *RadixNode, depth: usize } {
        var current = self;
        var matched: usize = 0;

        while (matched < query.len) {
            const next_token = query[matched];
            const child = current.children.get(next_token) orelse break;

            // Match child's token span
            const match_len = std.mem.indexOfDiff(u32, child.tokens, query[matched..]) orelse child.tokens.len;
            matched += match_len;

            if (match_len < child.tokens.len) break; // Partial match
            current = child;
        }

        return .{ .node = current, .depth = matched };
    }
};
```

**Key Properties:**
- Each node = consecutive token span + KV blocks
- Path from root to leaf = full request prefix
- Shared prefixes share memory (single KV copy)
- Cache-aware scheduler prioritizes high hit-rate requests

**References:** [LMSYS: RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/), [SGLang Paper](https://arxiv.org/pdf/2312.07104)

---

### Pattern 3: PagedAttention (Block-Based KV Cache)

**What:** Partition KV cache into fixed-size blocks (e.g., 16 tokens/block). Map logical sequence positions to physical blocks via block tables (like OS page tables).

**When:** Always — enables continuous batching by eliminating fragmentation. Mandatory for memory-bounded serving.

**Why:** Reduces KV waste from 60-80% (contiguous allocation) to <4% (paged). Enables 2-4× more concurrent requests per GPU.

**Implementation:**
```zig
const KVCacheManager = struct {
    block_size: usize = 16,        // Tokens per block
    free_blocks: std.ArrayList(KVBlock),
    block_tables: std.AutoHashMap(u32, []KVBlock), // request_id → blocks

    pub fn allocate(self: *KVCacheManager, request_id: u32, num_tokens: usize) ![]KVBlock {
        const num_blocks = (num_tokens + self.block_size - 1) / self.block_size;
        var blocks = try self.allocator.alloc(KVBlock, num_blocks);

        for (blocks) |*block| {
            block.* = self.free_blocks.popOrNull() orelse return error.OutOfMemory;
        }

        try self.block_tables.put(request_id, blocks);
        return blocks;
    }

    pub fn free(self: *KVCacheManager, request_id: u32) void {
        const blocks = self.block_tables.get(request_id) orelse return;
        self.free_blocks.appendSlice(blocks) catch {};
        _ = self.block_tables.remove(request_id);
    }
};
```

**Integration with RadixAttention:**
- RadixTree nodes own KV blocks
- Multiple requests can reference same node (shared prefix)
- LRU eviction frees blocks from least-recently-used leaves

**References:** [vLLM PagedAttention](https://docs.vllm.ai/en/latest/), [RunPod: vLLM & PagedAttention](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention)

---

### Pattern 4: Fused GPU Kernels

**What:** Combine multiple operations (GEMV + norm, SDPA components, MLP + activation) into single GPU kernels to minimize memory traffic.

**When:** Hot path operations that would otherwise require multiple kernel launches with intermediate memory writes.

**Why:** Reduces memory bandwidth bottleneck by 2-4×. Achieves 80-90% GPU utilization vs 30% unfused. FlashAttention cuts memory by 80%, speeds inference 2-5×.

**Key Fusion Patterns:**

| Fused Operation | Components | Benefit |
|----------------|------------|---------|
| **FlashAttention** | QK^T + softmax + V matmul | 80% memory reduction, 2-5× speedup |
| **Fused RMSNorm** | Reduction + scale + residual update | Eliminates 2 memory round-trips/layer |
| **Fused SwiGLU/GeGLU** | W_gate @ x, W_up @ x, SiLU/GELU, multiply | Single kernel vs 5 separate ops |
| **Fused RoPE + Attention** | RoPE rotation + SDPA | 1.6-3.7× bandwidth utilization |
| **Fused Quantization + GEMV** | Dequant + matrix multiply | In-register dequant, zero intermediate storage |

**Example - Fused addRmsNorm (already in Agave):**
```metal
// Metal kernel: read input, add residual, normalize, write output
kernel void addRmsNorm(
    device const float* input,
    device const float* residual,
    device const float* weight,
    device float* output,
    constant uint& n,
    constant float& eps,
    uint tid [[thread_position_in_grid]]
) {
    // Single pass: read both inputs, compute norm, write result
    // No intermediate buffer needed
}
```

**Implementation Priority:**
1. **SDPA fusion** (highest impact — bandwidth-bound decode phase)
2. **Norm fusion** (addRmsNorm, fusedRmsNorm — eliminate 2 syncs/layer)
3. **MLP fusion** (SwiGLU gate + activation)
4. **Quantization fusion** (in-kernel dequant for GEMV — already done in Agave)

**References:** [FlashInfer Paper](https://arxiv.org/pdf/2501.01005), [Medium: Fused Kernels](https://medium.com/the-synaptic-stack/how-fused-kernels-are-powering-the-llm-revolution-and-why-you-should-care-1e232fa1ae70)

---

### Pattern 5: Async Request Handling with Streaming

**What:** Use async I/O for request ingress/egress. Stream tokens as generated via SSE. Decouple API layer from inference loop via queues and AsyncStreams.

**When:** All production HTTP servers — non-negotiable for responsive UX.

**Why:** Reduces Time-to-First-Token perception. Keeps API threads responsive during GPU compute. Enables concurrent request handling.

**Implementation:**
```zig
// Request Manager creates AsyncStream per request
const AsyncStream = struct {
    request_id: u32,
    tokens: std.ArrayList(u32),
    event: std.Thread.ResetEvent,  // Signal new tokens
    done: std.atomic.Value(bool),

    pub fn push(self: *AsyncStream, token: u32) void {
        self.tokens.append(token) catch {};
        self.event.set();  // Wake waiting API thread
    }

    pub fn wait(self: *AsyncStream) ?u32 {
        while (true) {
            if (self.tokens.popOrNull()) |token| return token;
            if (self.done.load(.Acquire)) return null;
            self.event.wait();
        }
    }
};

// API handler (async FastAPI-style)
fn handleRequest(request: Request) !void {
    const stream = try request_manager.enqueue(request);

    // Stream tokens as they arrive
    while (stream.wait()) |token| {
        const text = detokenize(&[_]u32{token});
        try response.writeChunk(text);  // SSE
    }
}
```

**Background Loop (Engine Core):**
```zig
fn engineLoop(scheduler: *Scheduler, executor: *Executor) void {
    while (true) {
        const batch = scheduler.step();  // Assemble batch
        const outputs = executor.execute(batch);  // GPU forward pass

        for (outputs) |output| {
            output.stream.push(output.token);  // Notify waiting clients
            if (output.is_eos) output.stream.done.store(true, .Release);
        }
    }
}
```

**References:** [vLLM Async Streaming](https://docs.vllm.ai/en/latest/examples/offline_inference/async_llm_streaming/), [Optimizing LLM Latency](https://www.techfrontier.blog/2026/02/optimizing-llm-api-latency-async.html)

---

### Pattern 6: Memory Pool Allocation

**What:** Pre-allocate GPU memory into fixed-size pools (KV blocks, activation buffers). Use arena/slab allocators instead of per-request malloc.

**When:** All GPU memory management — eliminates fragmentation and allocation overhead.

**Why:** Deep learning allocators typically waste 10-15% to fragmentation. Pooling + BFC (Best-Fit with Coalescing) achieves near-optimal utilization.

**Implementation:**
```zig
const MemoryPool = struct {
    blocks: []Block,
    free_list: std.ArrayList(*Block),
    block_size: usize,

    pub fn init(allocator: Allocator, total_memory: usize, block_size: usize) !MemoryPool {
        const num_blocks = total_memory / block_size;
        var blocks = try allocator.alloc(Block, num_blocks);
        var free_list = std.ArrayList(*Block).init(allocator);

        for (blocks) |*block| {
            try free_list.append(block);
        }

        return MemoryPool{
            .blocks = blocks,
            .free_list = free_list,
            .block_size = block_size,
        };
    }

    pub fn alloc(self: *MemoryPool, size: usize) ?*Block {
        const num_blocks = (size + self.block_size - 1) / self.block_size;
        if (num_blocks > self.free_list.items.len) return null;
        return self.free_list.pop();
    }
};
```

**Agave Integration:**
- KV cache already uses block-based allocation (PagedKvCache, RadixTree)
- Add activation buffer pooling (reuse across requests)
- Weight buffers via BufCache (already implemented for Vulkan/CUDA)

**References:** [NVIDIA: KV Cache Offload](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/), [MemServe: Elastic Memory Pool](https://arxiv.org/html/2406.17565v2)

## Anti-Patterns to Avoid

### Anti-Pattern 1: Static Batching
**What:** Wait for full batch to complete before processing new requests.
**Why bad:** GPU sits idle while fast requests wait for slow ones. Head-of-line blocking kills latency.
**Instead:** Use iteration-level continuous batching (Pattern 1).

---

### Anti-Pattern 2: Contiguous KV Allocation
**What:** Allocate max_seq_len × hidden_dim contiguous memory per request upfront.
**Why bad:** Wastes 60-80% of memory on padding. Limits concurrent requests.
**Instead:** Use PagedAttention with block tables (Pattern 3).

---

### Anti-Pattern 3: FIFO-Only Scheduling
**What:** Schedule requests in arrival order, ignoring cache state.
**Why bad:** Misses 50-99% potential cache hits. Cold requests block hot ones.
**Instead:** Cache-aware scheduling — prioritize requests with longest prefix match (Pattern 2).

---

### Anti-Pattern 4: Per-Op GPU Sync
**What:** Synchronize CPU/GPU after every kernel launch.
**Why bad:** Serializes pipeline, prevents batching, kills throughput.
**Instead:** Deferred sync — batch multiple ops, sync only when CPU reads GPU data (Agave already does this for Metal/CUDA UMA).

---

### Anti-Pattern 5: Unfused Kernels on Hot Path
**What:** Separate kernel launches for norm, activation, residual add.
**Why bad:** Memory bandwidth bottleneck — intermediate results written to slow HBM.
**Instead:** Fuse operations into single kernels (Pattern 4).

## Scalability Considerations

| Concern | At 1 req/s | At 10 req/s | At 100 req/s |
|---------|------------|-------------|--------------|
| **Batching** | Static batch OK | Continuous batching mandatory | + Preemption policy |
| **KV Cache** | Flat/Paged sufficient | PagedAttention + eviction | RadixAttention + LRU |
| **Scheduling** | FIFO acceptable | Cache-aware priority | + Multi-level queues |
| **Memory** | Pre-allocate max | Block pool + dynamic alloc | Arena allocator + compaction |
| **Streaming** | Sync response OK | Async + SSE | + Backpressure handling |
| **Prefill/Decode** | Unified executor | Separate tuning per phase | Disaggregated architecture |

### Disaggregated Prefill/Decode (Future — Level 2)

**When to consider:** >50 req/s sustained load, or multi-turn chatbot workload.

**What:** Separate compute clusters for prefill (compute-bound, batched) vs decode (memory-bound, latency-sensitive).

**Benefits:**
- Independent scaling (add decode GPUs without prefill GPUs)
- Optimized hardware (high-FLOPS GPUs for prefill, high-bandwidth for decode)
- 15-40% infrastructure cost reduction

**Requirements:**
- High-speed interconnect (RDMA, NVLink, InfiniBand)
- KV transfer protocol between clusters
- More complex routing logic

**Current status:** Supported by vLLM, SGLang, TensorRT-LLM. Production at Meta, DeepSeek.

**Agave roadmap:** Defer until Level 1 complete (single-node continuous batching working).

**References:** [Jarvislabs: Disaggregated Prefill-Decode](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode), [vLLM: Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

## Build Order & Dependencies

### Phase 1: Request Management Foundation
**Goal:** Async request handling, token streaming, basic queuing.

**Components to build:**
1. `AsyncStream` (per-request output queue)
2. `RequestQueue` (FIFO pending requests)
3. `RequestManager` (tokenize, enqueue, detokenize)
4. HTTP server integration (SSE streaming, timeout)

**Dependencies:**
- Existing: Tokenizer, HTTP server (already in Agave)
- New: Async primitives (ResetEvent, atomic done flag)

**Validation:** Single request → tokenize → enqueue → stream tokens → HTTP SSE response.

**Build time:** 1-2 weeks

---

### Phase 2: Continuous Batching Scheduler
**Goal:** Iteration-level scheduling, dynamic batch assembly.

**Components to build:**
1. `Scheduler` (step function, batch management)
2. `Batch` (dynamic request list)
3. Engine background loop (scheduler.step() → executor.execute())
4. Request completion handling (pop finished, pull new)

**Dependencies:**
- Phase 1: RequestManager, AsyncStream
- Existing: Executor (model forward), Backend

**Validation:** Multiple requests → scheduler assembles batch → executor processes → outputs routed to streams.

**Build time:** 1-2 weeks

---

### Phase 3: PagedAttention Integration
**Goal:** Block-based KV cache, memory-bounded serving.

**Components to build:**
1. `KVCacheManager` refactor (block allocation, block tables)
2. `BlockTable` per request (logical → physical mapping)
3. Executor integration (pass block tables to attention kernels)
4. Preemption (evict low-priority requests on OOM)

**Dependencies:**
- Phase 2: Scheduler (coordinates allocation)
- Existing: KV cache (refactor from flat/paged to block-based)

**Validation:** Concurrent requests exceed total KV memory → no OOM, requests preempted and resumed.

**Build time:** 2-3 weeks

**Critical:** Attention kernels must support indirect block access (already true for Agave's `pagedAttention` in attention.zig).

---

### Phase 4: RadixAttention Prefix Caching
**Goal:** Automatic prefix detection, cache reuse, LRU eviction.

**Components to build:**
1. `RadixTree` (prefix trie, node storage)
2. Cache-aware scheduler (prioritize high hit-rate requests)
3. KV block sharing (multiple requests reference same node)
4. LRU eviction (free least-recently-used leaves)

**Dependencies:**
- Phase 3: KVCacheManager (owns block pool)
- Phase 2: Scheduler (uses prefix match for priority)

**Validation:** Shared prefix requests → KV blocks reused → cache hit rate >50% → TTFT reduction.

**Build time:** 2-3 weeks

**Note:** Can be built in parallel with Phase 3 if block-based KV API is stable.

---

### Phase 5: Fused Kernel Optimization
**Goal:** Minimize memory traffic, maximize GPU utilization.

**Components to optimize:**
1. Fused SDPA (already partial — Metal has GPU kernel, needs debug)
2. Fused RMSNorm (already done — `fusedRmsNorm`, `addRmsNorm`)
3. Fused MLP (SwiGLU gate + activation in single kernel)
4. Fused RoPE + Attention (rotate + SDPA)

**Dependencies:**
- Existing: Backend kernels (extend with fusion)
- All prior phases (measure end-to-end impact)

**Validation:** Benchmark throughput, memory bandwidth. Target 80%+ GPU utilization.

**Build time:** 3-4 weeks (iterative — measure, fuse, measure)

**Note:** Can be incremental — fuse one op at a time, validate improvement.

---

### Dependency Graph

```
Phase 1 (Request Management)
  ↓
Phase 2 (Continuous Batching) ←──┐
  ↓                               │
Phase 3 (PagedAttention) ─────────┤
  ↓                               │
Phase 4 (RadixAttention) ←────────┘
  ↓
Phase 5 (Fused Kernels) — can start earlier, runs in parallel
```

**Critical path:** 1 → 2 → 3 → 4 (8-10 weeks sequential)

**Parallelization opportunities:**
- Phase 4 can start once Phase 3 block API is stable (save 2 weeks)
- Phase 5 can start anytime after Phase 2 (runs concurrently, measures impact)

**Total calendar time (with parallelization):** 10-12 weeks

## Integration with Agave's Existing Architecture

### Leverage Existing Components

| Agave Component | Production Role | Required Changes |
|-----------------|----------------|------------------|
| **Backend dispatcher** | Fused kernel execution | Add fused SDPA, fused MLP kernels |
| **KV cache (PagedKvCache, RadixTree)** | Block management | Integrate with Scheduler allocation |
| **HTTP server** | API layer | Add SSE streaming, AsyncStream wiring |
| **ThreadPool** | Background engine loop | Spawn scheduler loop thread |
| **Tokenizer** | Request Manager | Move to async context (no blocking) |

### New Components Needed

1. **Scheduler** (new file: `src/scheduler.zig`)
   - Continuous batching logic
   - Cache-aware priority queue
   - Resource allocation coordination

2. **RequestManager** (new file: `src/request_manager.zig`)
   - AsyncStream per request
   - Request queuing
   - Tokenize/detokenize in background

3. **AsyncStream** (new struct in `request_manager.zig`)
   - Per-request output queue
   - Event-based wake/sleep
   - Integration with HTTP SSE

4. **KVCacheManager refactor** (extend `src/kvcache/manager.zig`)
   - Unify PagedKvCache + RadixTree under single allocator
   - Block table management
   - Preemption/eviction policy

### Data Flow Integration

**Current (single request):**
```
main.zig → model.forward() → backend.gemv() → output token
```

**Production (multi-request continuous batching):**
```
HTTP → RequestManager.enqueue() → Scheduler.step() → Executor.execute_batch() → backend.gemv() → AsyncStream.push() → SSE
                ↑                                              ↓
                └───────────── Background loop ────────────────┘
```

### File Structure (Proposed)

```
src/
├── scheduler.zig          # NEW: Continuous batching scheduler
├── request_manager.zig    # NEW: Async request handling
├── server.zig             # MODIFY: Add SSE streaming, AsyncStream wiring
├── kvcache/
│   └── manager.zig        # MODIFY: Unify Paged + Radix under Scheduler API
├── backend/
│   ├── backend.zig        # MODIFY: Add fused kernel APIs
│   ├── metal.zig          # MODIFY: Fused SDPA debug, fused MLP
│   ├── cuda.zig           # MODIFY: Fused kernels (SDPA, MLP)
│   └── cpu.zig            # MODIFY: Fused fallbacks
└── main.zig               # MODIFY: Spawn scheduler loop, wire RequestManager
```

## Sources

### Continuous Batching
- [Hugging Face: Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching)
- [Anyscale: Achieve 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Baseten: Continuous vs Dynamic Batching](https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/)
- [Machine Learning at Scale: LLM Serving (1) - Continuous Batching](https://machinelearningatscale.substack.com/p/llm-serving-1-continuous-batching)
- [Medium: Minimal LLM Inference - Continuous Batching](https://sudhirpol522.medium.com/minimal-llm-inference-continuous-batching-df37779b1c18)

### RadixAttention & SGLang
- [LMSYS: Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
- [Medium: SGLang Learning Series - RadixAttention](https://medium.com/@dharamendra1314.kumar/sglang-learning-series-part-1-shared-prefix-kv-cache-and-radixattention-d7a847d20b1f)
- [SGLang Paper (arXiv)](https://arxiv.org/pdf/2312.07104)
- [GitHub: sgl-project/sglang](https://github.com/sgl-project/sglang)
- [SGLang Documentation](http://docs.sglang.io/)

### vLLM & PagedAttention
- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [GitHub: vllm-project/vllm](https://github.com/vllm-project/vllm)
- [RunPod: Introduction to vLLM and PagedAttention](https://www.runpod.io/blog/introduction-to-vllm-and-pagedattention)
- [Hamza's Blog: Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)
- [Data Science Dojo: Understanding Paged Attention](https://datasciencedojo.com/blog/understanding-paged-attention/)
- [Machine Learning at Scale: LLM Serving (2) - Paged Attention](https://machinelearningatscale.substack.com/p/llm-serving-2-paged-attention)

### Architecture & Scheduler
- [vLLM: Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [Aleksa Gordić: Inside vLLM - Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [vLLM Blog: Inside vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [DeepWiki: vLLM Scheduler and Resource Allocation](https://deepwiki.com/vllm-project/vllm/3.3-multimodal-models)
- [DeepWiki: Worker and Executor Architecture](https://deepwiki.com/vllm-project/vllm/4.2-worker-and-executor-architecture)
- [Hugging Face: Efficient Request Queueing](https://huggingface.co/blog/tngtech/llm-performance-request-queueing)
- [Ubicloud: Life of an Inference Request (vLLM V1)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1)

### GPU Kernel Fusion
- [FlashInfer Paper (arXiv)](https://arxiv.org/pdf/2501.01005)
- [GitHub: flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- [NVIDIA: Run High-Performance LLM Inference Kernels Using FlashInfer](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)
- [Medium: How Fused Kernels Are Powering the LLM Revolution](https://medium.com/the-synaptic-stack/how-fused-kernels-are-powering-the-llm-revolution-and-why-you-should-care-1e232fa1ae70)
- [Medium: FlashAttention-3: The Engine Powering Next-Gen LLMs](https://medium.com/the-synaptic-stack/flashattention-3-the-engine-powering-next-gen-llms-30b2843bb182)
- [yadnyesh's blog: Dissecting FlashInfer](https://ydnyshhh.github.io/posts/flash_infer/)

### Disaggregated Architecture
- [Jarvislabs: Disaggregated Prefill-Decode Architecture](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode)
- [vLLM: Disaggregated Prefilling (experimental)](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [arXiv: DistServe - Disaggregating Prefill and Decoding](https://arxiv.org/pdf/2509.17542)
- [InfoQ: Disaggregation in LLMs](https://www.infoq.com/articles/llms-evolution-ai-infrastructure/)

### Memory Management
- [NVIDIA: Accelerate LLM Inference with KV Cache Offload](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)
- [Introl: KV Cache Optimization](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide)
- [Sankalp's Blog: How Prompt Caching Works](https://sankalp.bearblog.dev/how-prompt-caching-works/)
- [arXiv: MemServe - Elastic Memory Pool](https://arxiv.org/html/2406.17565v2)

### Async & Streaming
- [vLLM: Async LLM Streaming](https://docs.vllm.ai/en/latest/examples/offline_inference/async_llm_streaming/)
- [Tech Frontier Blog: Optimizing LLM API Latency - Async, Streaming, and Pydantic](https://www.techfrontier.blog/2026/02/optimizing-llm-api-latency-async.html)
- [Medium: Streaming LLM Responses in Real Time](https://medium.com/@ansilproabl/streaming-llm-responses-in-real-time-705b8784fae5)

### Distributed Inference
- [Medium: Tensor Parallel LLM Inferencing](https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7)
- [Red Hat: Distributed Inference with vLLM](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm)
- [BentoML: Data, Tensor, Pipeline, Expert and Hybrid Parallelism](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism)
- [vLLM: Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

### Weight Loading
- [Hugging Face: Loading Big Models into Memory](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- [Analytics Vidhya: Memory-Efficient Model Weight Loading in PyTorch](https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/)
