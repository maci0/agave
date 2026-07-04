# Glossary & Unexplained-Terms Audit — Batch 3

Chapters 15–20, Appendices: Atomics, Compile-Time, Math, Profiling.

---

## Chapter 15: Chat Templates

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **BPE** | Used in diagram label "BPE encode" — never expanded or defined. |
| **EOG** | Used dozens of times (e.g. "EOG token names") before any expansion. Eventually the section title says "End-of-Generation Token Detection," but the acronym appears much earlier without expansion. |
| **GGUF** | Referenced ("from GGUF metadata") — never expanded or defined in this chapter. |
| **ChatML** | Used as a style name ("ChatML, used by Qwen3.5") — never explicitly defined as "Chat Markup Language." |
| **BOS** | Not used directly here, but `[gMASK]<sop>` for GLM-4 assumes familiarity with BOS/SOP concepts. |
| **SigLIP-2** | Mentioned in passing ("image tokens already supported via SigLIP-2") — never explained. |
| **Jinja2** | Mentioned under "Future Extensions" — never defined. |
| **CLI** | Used without expansion (common enough to be acceptable, but never expanded). |
| **MoE** | Not used in this chapter. |
| **SSE** | Not used in this chapter. |

### Glossary of terms introduced

- **chat template** — a data-driven configuration that maps conversation roles (system, user, assistant) to special-token-delimited prefix/suffix strings, replacing hardcoded prompt formatting.
- **tight coupling** — a software design problem where prompt format details are embedded directly in model code, making changes fragile and non-portable.
- **EOG token (end-of-generation token)** — a special token whose presence in the model's output signals that generation should stop (e.g., `<|im_end|>`, `<end_of_turn>`).
- **role marker** — a special token or string that identifies who is speaking in a multi-turn conversation (system, user, assistant, tool, developer).
- **ChatML** — a chat formatting convention using `<|im_start|>` / `<|im_end|>` markers, adopted by Qwen, Nemotron, and other model families.
- **generation_prefix** — a string appended after the final assistant prefix before generation begins; used to suppress or enable model reasoning (e.g., an empty `<think>` block).
- **system_role_override** — a template field that re-routes user-provided system messages through a different role prefix (e.g., "developer" in GPT-OSS).
- **default_system** — a fixed system message baked into the template, used automatically when the user supplies none.
- **image token injection** — the process of splicing visual placeholder tokens (start, pad, end) into a tokenized prompt so the model can replace them with vision-encoder embeddings during forward().
- **findImageInsertPos()** — a function that scans a token array for the last occurrence of the user-prefix token sequence and returns the position just after it, for image token insertion.
- **injectImageTokens()** — a function that splices a sequence of image placeholder tokens (start + N×pad + end) at a computed insertion point in the token array.
- **pad token (image)** — a placeholder token repeated N times (one per visual patch) whose embedding is replaced at runtime by the corresponding vision-encoder output vector.
- **formatConversation()** — the main template function that renders a multi-turn conversation (system message + messages array + final assistant prefix) into a flat prompt string.

---

## Chapter 16: Recipe System

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **OOM** | Used ("to avoid OOM") — never expanded as "Out Of Memory." |
| **CLI** | Used throughout without expansion. |
| **MoE** | Used ("MoE models use more VRAM") — never expanded as "Mixture of Experts" in this chapter. |
| **VRAM** | Used ("MoE uses more VRAM") — never expanded as "Video RAM." |
| **Q4, Q4_K_M, Q4_0** | Quantization format names used without definition in this chapter. |
| **BF16** | Referenced in recipe naming ("Gemma 27B BF16 CUDA") — never expanded. |
| **L3 cache** | Referenced ("fit in L3 cache") — not defined. |
| **ctx_size** | Used as a parameter without explaining it stands for "context size" (the maximum token window). |

### Glossary of terms introduced

- **recipe** — a named set of optional inference-parameter defaults matched by model architecture, backend, and quantization type.
- **configuration sprawl** — the problem of scattered, duplicated magic numbers across model files for settings like temperature and context size.
- **Preset** — a struct pairing a match pattern (arch_prefix, backend, quant) with a Recipe, stored in a priority-ordered array.
- **Recipe.default** — the fallback recipe with all fields set to null, used when no preset matches.
- **Overrides** — a struct of boolean flags tracking which parameters the user explicitly set via CLI, so recipe defaults don't override user intent.
- **Applied** — the fully-resolved struct with concrete (non-optional) values after merging CLI flags, recipe defaults, and CLI baseline defaults.
- **arch_prefix** — the leading substring of a model's architecture name used for prefix matching (e.g., `"qwen3"` matches `"qwen35"`).
- **wildcard match** — an empty-string field in a preset that matches any value for that criterion (arch, backend, or quant).
- **first-match-wins** — the matching rule that returns the first preset whose criteria are satisfied, making array order determine priority.
- **three-level priority chain** — the resolution order for each parameter: (1) user CLI flag, (2) recipe default, (3) CLI baseline default.

---

## Chapter 17: Speculative Decoding & DDTree

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **KV cache** | Used extensively — never defined in this chapter (assumed from prior chapters). |
| **SDPA** | Used ("tree-masked SDPA kernel") — never expanded here as "Scaled Dot-Product Attention." |
| **SSE** | Used ("SSE streaming") — refers to Server-Sent Events, never expanded. |
| **MoE** | Used in performance section — not expanded. |
| **GGUF** | Referenced ("Medusa GGUF") — not expanded. |
| **LM head** | Used ("draft model's LM head") — never expanded as "Language Model head." |
| **BOS** | Not explicitly used but implied. |
| **FlashAttention-2** | Mentioned ("GPU kernels use FlashAttention-2") — not defined or explained. |
| **ROCm** | Listed as a backend — never expanded as "Radeon Open Compute." |
| **WebGPU** | Listed as a backend — never explained. |
| **ACL** | Referenced in citation ("ACL 2025") — never expanded as the conference name. |
| **MTP** | First used in the table before the dedicated section — only defined in detail later in Ch.18. |
| **EAGLE** | The acronym is eventually expanded in its section header, but used in the modes table first without expansion. |
| **argmax** | Used without definition ("target model's argmax matches"). |
| **logits** | Used throughout without definition in this chapter. |
| **TTFT** | Used ("PFlash targets time-to-first-token (TTFT)") — expanded inline on first use. ✅ |
| **RAG** | Used ("RAG pipelines") — never expanded as "Retrieval-Augmented Generation." |

### Glossary of terms introduced

- **speculative decoding** — an inference acceleration technique where a cheap draft model proposes multiple candidate tokens that a larger target model verifies, accepting correct ones for free.
- **draft model** — a small, fast model that generates candidate tokens during speculation; can be a separate model, the target itself (self-spec), or n-gram history.
- **target model** — the full, accurate model that verifies draft tokens and produces the final output.
- **acceptance rate** — the fraction of draft tokens the target model agrees with, determining the speedup factor.
- **DDTree (Draft Distribution Tree)** — a tree-structured speculative decoding method that builds an optimal tree of candidate continuations from draft distributions using a best-first heap, maximizing expected acceptance length.
- **tree budget (B)** — the maximum number of nodes in a DDTree, controlling how wide the candidate tree can grow.
- **spec-tokens (K)** — the draft depth: how many forward passes the draft model runs to produce per-position logit distributions.
- **ancestor bitmask** — a per-node bitmask (`[8]u64`, 512 bits) encoding which tree nodes are ancestors of a given node, used in tree-masked attention to restrict which positions a node can attend to.
- **tree attention / tree-masked SDPA** — a modified attention kernel where each tree node attends to all shared-prefix KV entries plus only its ancestor nodes within the draft tree.
- **self-speculative decoding** — using the target model as its own draft by skipping a subset of transformer layers during the draft phase, trading accuracy for speed without a separate model.
- **draft-layers** — the number of transformer layers skipped during self-speculative drafting (default: 50% of layers, skipping the middle).
- **n-gram mode** — a draft-free speculation mode that searches the last 2048 generated tokens for n-gram matches and proposes the tokens that followed the match as drafts.
- **suffix decoding** — like n-gram but uses exact suffix matching over a larger cross-request cache (10,000 tokens) with dynamic speculation depth.
- **lookahead decoding** — Jacobi-style parallel decoding that maintains W branches of N candidate tokens, advancing them independently and searching for n-gram matches.
- **EAGLE (Efficient Acceleration via Greedily-Embedded Token Entropy)** — a speculative method where the draft model is conditioned on the target model's hidden state rather than just the previous token, yielding higher acceptance rates.
- **EAGLE-3** — a refinement of EAGLE that conditions on the pre-output-norm hidden state instead of the post-norm one, preserving scale information.
- **MLP Speculator** — a single-step speculation mode where all K draft steps use the frozen target hidden state (no autoregressive chain), cheaper than EAGLE but with slightly lower acceptance.
- **Medusa** — multiple parallel MLP prediction heads trained on top of the base model, each predicting the token at a different future offset simultaneously.
- **FR-Spec (Frequency-Ranked Speculative Decoding)** — restricts the draft model's vocabulary to high-frequency tokens during drafting, improving acceptance rates by focusing on tokens the target is also likely to pick.
- **rejection sampling (speculative)** — the mechanism that preserves the target model's output distribution during sampling (temperature > 0): each draft token is accepted with probability min(1, p_target/p_draft); on rejection, a correction is sampled from the residual distribution.
- **residual distribution** — the distribution max(0, p_target - p_draft) normalized, used to sample a correction token when a draft token is rejected.
- **bonus token** — an extra token sampled from p_target at position K+1 after all K draft tokens are accepted.
- **adaptive K** — runtime auto-tuning of the draft depth K based on per-K acceptance statistics accumulated during generation.
- **KV cache rollback** — resetting the KV cache sequence length to the accepted prefix length after a draft rejection, discarding speculated entries.
- **SharedNgramPool** — a thread-safe pool (~32 KB) that accumulates tokens from all concurrent server requests, allowing n-gram speculation to benefit from cross-request history.
- **DFlash (Block Diffusion Flash)** — a speculative decoding baseline that uses a block diffusion model as a drafter to produce an entire block of draft tokens in a single forward pass.
- **PFlash (speculative prefill)** — a technique that uses a cheap scorer to identify which KV blocks in a long prompt matter, then prefills only those blocks through the target model, dramatically reducing time-to-first-token.
- **forwardTree()** — a model method that verifies the entire draft tree in a single forward pass using tree-masked SDPA, reducing verification cost from O(K) sequential forwards to O(1).
- **batch tree verification** — running the entire DDTree through the target model in a single forward pass rather than verifying tokens sequentially.

---

## Chapter 18: Multi-Token Prediction (MTP)

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **GEMV** | Expanded on first use as "General Matrix-Vector multiply." ✅ |
| **Q/K/V** | Expanded as "Query, Key, Value." ✅ |
| **RoPE** | Expanded as "Rotary Position Embedding." ✅ |
| **SDPA** | Expanded as "Scaled Dot-Product Attention." ✅ |
| **SwiGLU** | Named but not fully expanded as an acronym (described functionally). |
| **SiLU** | Named without expansion — described as `x * sigmoid(x)`. |
| **RMSNorm** | Expanded as "Root Mean Square Normalization." ✅ |
| **FFN** | Used without expansion as "feed-forward network" — described in context but the acronym itself is not expanded on first use. |
| **SSM** | Used ("DeltaNet SSM layers") — never expanded as "State Space Model" in this chapter. |
| **DeltaNet** | Used ("DeltaNet SSM layers") — never defined beyond being named. |
| **KV cache** | Used without definition (assumed known). |
| **GGUF** | Used ("GGUF files with nextn tensors") — not expanded. |
| **DeepSeek V3** | Referenced as the source of offset RMSNorm — not described beyond the name. |
| **BF16** | Not used in this chapter. |
| **MoE** | Not used in this chapter. |

### Glossary of terms introduced

- **multi-token prediction (MTP)** — a technique that adds lightweight draft heads to a model, trained jointly with the main model, to predict multiple future tokens from the model's internal state.
- **MTP head** — a single transformer layer with fusion plumbing (enorm, hnorm, eh_proj) that takes the main model's pre-norm hidden state and the current token embedding to produce a draft token at ~5% of a full forward pass cost.
- **pre-norm hidden state** — the residual stream after the last attention block but before the final FFN residual and output norm, containing the model's complete context understanding; saved and passed to MTP heads.
- **offset RMSNorm (+1 norm)** — a variant of RMSNorm where the weight is applied as `(1 + w) * x_norm` instead of `w * x_norm`, ensuring the normalized input passes through even if w decays to zero; introduced by DeepSeek V3.
- **enorm** — the offset RMSNorm weight tensor applied to the token embedding branch in an MTP head.
- **hnorm** — the offset RMSNorm weight tensor applied to the hidden state branch in an MTP head.
- **eh_proj** — the GEMV projection that maps the concatenated `[embed; hidden]` vector from 2×n_embd back to n_embd dimensions.
- **shared_head_norm** — the RMSNorm weight tensor applied before the MTP head's output projection (not shared with the main model's output layer).
- **shared_head_head** — the output GEMV weight matrix in an MTP head that maps n_embd → vocab_size to produce logits.
- **nextn tensors** — GGUF tensor names prefixed with `blk.N.nextn.*` (where N ≥ block_count) that store MTP head weights.
- **nextn_predict_layers** — a GGUF metadata field indicating how many MTP depths are present in the checkpoint (typically 1).
- **SSM state checkpoint/restore** — the process of copying the full SSM recurrent state before speculation and restoring it on rejection, required for hybrid models (like Qwen 3.5) that mutate recurrent state in-place during forward passes.
- **embedding table** — a matrix of shape `[vocab_size, n_embd]` that maps each token ID to a dense vector (the hidden state).
- **hidden state** — the dense vector of n_embd floats that is the model's internal representation of a token, passed through all transformer layers.
- **residual connection** — adding the output of a sub-block to its input (`hidden = hidden + block_output`), preventing the vanishing gradient problem.
- **vanishing gradient problem** — the issue where gradients shrink to near-zero through many layers, preventing deep networks from learning; solved by residual connections.
- **attention gate** — a learned sigmoid signal that controls how much attention output flows through (used in Qwen3.5).
- **head_dim** — the dimensionality of each attention head's Q/K/V vectors (n_embd / n_heads).
- **logits** — the raw, un-normalized scores output by the model's final projection, one per vocabulary token.

---

## Chapter 19: PFlash and Block Sparse Attention

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **KV cache** | Used without definition — assumed known from prerequisites. |
| **SDPA** | Used without expansion in this chapter. |
| **DDTree** | Used without re-definition — assumes Ch.17 was read. |
| **BigBird** | Referenced ("BigBird-style block sparsity") — never explained beyond the name. |
| **TTFT** | Used without expansion until later in the chapter, then expanded. |
| **RAG** | Used ("RAG pipelines with many retrieved chunks") — never expanded. |
| **SRAM** | Not used in this chapter. |
| **BOS** | Used ("typically cover BOS") — never expanded as "Beginning of Sequence." |

### Glossary of terms introduced

- **block sparse attention** — an approximation of full attention that groups tokens into fixed-size blocks and restricts which block pairs interact, reducing complexity from O(n²) to O(n).
- **global block** — one of the first G token blocks that every query block attends to (and vice versa), typically covering the system prompt and task prefix.
- **sliding window (block sparse)** — a pattern where each query block attends to ±W neighboring blocks, preserving local sequential context.
- **sparsity pattern** — the combination of global blocks and sliding window that determines which block pairs compute attention scores; everything outside is skipped entirely.
- **PFlash** — a speculative prefill technique that uses a cheap scorer model with block sparse attention to identify which KV blocks matter, then prefills only those blocks through the full target model.
- **scorer model** — a small model (or the draft model) that runs a block sparse attention pass over the full prompt to produce per-block importance scores.
- **block importance score** — a per-block value indicating how relevant that block is to the model's output, used by PFlash to decide which blocks to keep.
- **adaptive threshold (alpha)** — the selection criterion `alpha × mean(block_scores)` that determines which blocks survive PFlash compression; adapts to prompt density unlike fixed top-K.
- **pflash-alpha** — the tunable parameter controlling PFlash compression aggressiveness (default 0.85); lower alpha = more aggressive compression.
- **pflash-block-size** — the granularity of block selection in PFlash; smaller blocks (16–32 tokens) allow finer selection, larger blocks (64–128) reduce overhead.
- **compressed prefill** — running the target model's standard prefill over only the PFlash-selected tokens instead of the full prompt.
- **max_kept_ratio** — a cap (default 0.20) on the fraction of blocks PFlash can retain, preventing over-selection in dense prompts.
- **pflash-scorer** — an optional separate model (smaller than the draft model) used solely for block importance ranking during PFlash scoring.
- **scoreFromLastQ** — a KV-dot-product scoring function defined in PFlash but not yet integrated into the main prefill pipeline.
- **prompt compressibility** — how much of a prompt's content is irrelevant and can be safely dropped by PFlash; prompts with high repetition or boilerplate compress more aggressively.

---

## Chapter 20: Diffusion Language Models

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **dLLM** | Expanded on first use as "diffusion language model." ✅ |
| **MoE** | Used ("128 experts MoE with top-8 routing") — not expanded. |
| **BF16** | Used ("BF16 SafeTensors only") — not expanded. |
| **SafeTensors** | Used without definition. |
| **GGUF** | Used ("no GGUF yet") — not expanded. |
| **KV cache** | Used without definition. |
| **VLM** | Used ("VLM image tokens") — never expanded as "Vision Language Model." |
| **MTP** | Used in comparison table — not expanded in this chapter. |
| **DDTree** | Used in comparison table — not defined here. |
| **H200** | Referenced ("~1,288 tokens/sec on H200") — GPU model name, not explained. |
| **FP8** | Used ("H200 (FP8)") — not expanded as "8-bit floating point." |

### Glossary of terms introduced

- **diffusion language model (dLLM)** — a language model that generates text by iteratively denoising random tokens in parallel, rather than producing tokens one at a time autoregressively.
- **DiffusionGemma** — Google's first publicly released diffusion language model, built on the Gemma 4 26B A4B backbone.
- **canvas** — a fixed-size buffer (default 256 tokens) initialized with random vocabulary tokens that is iteratively denoised to produce output text.
- **uniform state diffusion** — the canvas initialization strategy of filling positions with arbitrary vocabulary entries rather than special [MASK] tokens.
- **bidirectional attention** — attention where each canvas position can see all other canvas positions simultaneously (no causal mask), enabling internal consistency during denoising.
- **confidence-based acceptance** — the rule that locks a canvas position when the model's softmax probability for its top token exceeds a threshold, making it an anchor for future steps.
- **anchor token** — a canvas token that has been accepted (locked) during denoising; it provides stable context for resolving remaining positions.
- **re-noising** — replacing a rejected canvas token with a fresh random token (not the rejected guess) to give the model a clean slate.
- **denoising step** — one forward pass over the canvas followed by acceptance/re-noising decisions at each position.
- **diffusion-steps** — a CLI parameter controlling the maximum number of denoising iterations per canvas block (default 16).
- **diffusion-confidence** — a CLI parameter setting the probability threshold for locking a canvas position (higher = harder to lock, more refinement).
- **diffusion-canvas** — a CLI parameter setting the canvas size (default 256 tokens).
- **block autoregressive chaining** — the process of appending a fully denoised canvas to the KV cache and starting a new canvas conditioned on all prior context, enabling outputs longer than one canvas block.
- **self-correction** — the property unique to diffusion: if an early token becomes inconsistent with later context, re-noising lets the model fix it (impossible in autoregressive models).
- **canvas attention kernel** — `scaledDotProductAttentionCanvas()` in `attention.zig`: scores canvas queries against both prompt KV cache (causal) and all canvas tokens (bidirectional).
- **forwardCanvas()** — the canvas denoiser forward mode: takes a canvas, runs bidirectional attention against the prompt KV cache, returns logits for all canvas positions without adding canvas tokens to the KV cache.
- **fused expert weights** — a weight storage format where `experts.gate_up_proj` stores gate+up concatenated in a single 3D tensor instead of separate tensors.
- **per-layer scalar** — a learned scalar multiplied against the attention output in DiffusionGemma, controlling per-layer contribution magnitude.

---

## Appendix: Atomic Operations and Memory Ordering

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **CAS** | Used in code ("CAS on active") before being formally introduced later in the appendix under its own heading. |
| **ARM** | Used as a CPU architecture name — never expanded as "Advanced RISC Machines." |
| **x86** | Used as a CPU architecture name — not expanded. |
| **TSan / ThreadSanitizer** | Expanded inline. ✅ |
| **CI** | Used ("Use TSan in CI") — not expanded as "Continuous Integration." |
| **futex** | Expanded inline as "fast userspace mutex." ✅ |
| **SIMD** | Used ("use SIMD instead") — not expanded. |
| **hyperthreading** | Used without definition. |

### Glossary of terms introduced

- **atomic operation** — a CPU instruction that performs a read-modify-write on memory as one indivisible step, preventing race conditions between threads.
- **race condition** — a bug where two threads read and write the same memory without coordination, causing interleaved operations that corrupt data.
- **std.atomic.Value(T)** — Zig's atomic wrapper type providing atomic load, store, fetchAdd, fetchSub, and compare-and-swap operations with configurable memory ordering.
- **memory ordering** — the guarantee about when writes by one thread become visible to other threads; controls reordering of loads and stores around atomic operations.
- **.monotonic** — the weakest memory ordering: guarantees atomicity but no ordering relative to other operations; cheapest option.
- **.acquire** — a load ordering that ensures all writes that happened before a paired .release store are visible after this load completes.
- **.release** — a store ordering that ensures all writes before this store become visible to other threads before the store itself.
- **.acq_rel** — combined acquire+release ordering on a single atomic operation.
- **.seq_cst (sequential consistency)** — the strongest ordering: all threads see all operations in the same global order; slowest, rarely needed.
- **compare-and-swap (CAS)** — an atomic operation that updates a value only if it currently matches an expected value; foundational for lock-free data structures.
- **cmpxchgWeak** — a CAS variant that may spuriously fail (return failure even if values match), faster on ARM; best used in retry loops.
- **cmpxchgStrong** — a CAS variant that only fails if the current value genuinely differs from the expected one; used for one-shot CAS.
- **spinLoopHint** — a CPU hint (`pause` on x86, `yield` on ARM) that reduces power consumption during busy-wait loops and allows hyperthreading to switch cores.
- **fence** — an explicit memory barrier that orders non-atomic writes relative to atomic operations; rarely needed when acquire/release is used directly.
- **torn read/write** — reading a partially-updated value when a multi-byte write is split across two CPU operations (e.g., 64-bit write on 32-bit platform).
- **generation counter** — an atomic integer bumped by the main thread to signal new work; workers sleep on it via futex and wake when it changes.
- **futex (fast userspace mutex)** — a Linux/macOS primitive that lets threads sleep cheaply until a memory location changes, avoiding busy-waiting.
- **ThreadSanitizer (TSan)** — a runtime tool that detects data races by instrumenting memory accesses; enabled via `-Dsanitize-thread`.

---

## Appendix: Compile-Time Optimization

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **LUT** | Used ("LUT (comptime table, runtime lookup)") — expanded inline via diagram label. ✅ |
| **FP8 E4M3** | Used ("FP8 E4M3 Dequantization Table") — partially expanded; "FP8" = 8-bit floating point is not spelled out, "E4M3" (4-bit exponent, 3-bit mantissa) is explained later in the code. |
| **IQ4_NL** | Used as a quantization format name — explained as "non-linear quantization" but the "IQ" prefix is never defined. |
| **MXFP4** | Used ("MXFP4 Lookup Table") — explained as "E2M1 format" but the "MX" prefix is never defined. |
| **NVFP4** | Used ("nvfp4Dequant") — never expanded. |
| **MSL** | Used ("MSL source") — never expanded as "Metal Shading Language." |
| **SPIR-V** | Used ("SPIR-V binary") — never expanded. |
| **CUDA** | Used in build options — not expanded. |
| **AVX2** | Used ("gemvAVX2") — not expanded as "Advanced Vector Extensions 2." |
| **SSE2** | Used ("gemvSSE2") — not expanded as "Streaming SIMD Extensions 2." |
| **.rodata** | Used (".rodata section") — not explained as the read-only data segment of an executable. |
| **monomorphized** | Used in a diagram ("monomorphized copy") — not defined. |

### Glossary of terms introduced

- **comptime** — Zig's compile-time execution feature: expressions evaluated during compilation whose results are baked into the binary as constants.
- **lookup table (LUT)** — a pre-computed array where a runtime input (e.g., an 8-bit value) indexes directly into the result, replacing expensive arithmetic with a single array load.
- **comptime block** — a labeled Zig block (`blk: { ... break :blk result; }`) where the entire body runs at compile time and produces a constant.
- **dead code elimination** — the compiler's removal of code branches that can never execute (e.g., Linux-only code when compiling for macOS), reducing binary size.
- **@embedFile** — a Zig builtin that reads a file's contents at compile time and embeds them as a byte-string constant in the binary's .rodata section.
- **inline else** — a Zig switch pattern (`inline else => |be| ...`) that expands to separate cases per tagged-union variant at compile time, enabling inlining and avoiding vtable dispatch.
- **@compileError** — a Zig builtin that halts compilation with a custom error message; used to prevent instantiation of unsupported type specializations.
- **build_options** — compile-time configuration values set in `build.zig` and imported in source via `@import("build_options")`, controlling which backends are compiled.
- **conditional compilation** — using comptime feature detection (OS, CPU arch, build flags) to select code paths at compile time, ensuring only relevant code is included.
- **comptime assertion** — a `std.debug.assert()` evaluated at compile time; a failing assertion halts compilation before any binary is produced.
- **type-specialized function** — a generic function parameterized by `comptime T: type` that generates a separate, optimized code path for each type it is instantiated with.
- **format string validation** — Zig's compile-time verification that format string specifiers match the number and types of arguments, preventing printf-style runtime bugs.
- **FP8 E4M3** — an 8-bit floating-point format with 4 exponent bits and 3 mantissa bits, dequantized to f32 via a 256-entry comptime lookup table.
- **IQ4_NL** — a 4-bit non-linear quantization format using a fixed table of 16 non-uniformly-spaced dequantization values, giving better accuracy than linear Q4 for small values.
- **MXFP4 (E2M1)** — a 4-bit floating-point format with 2 exponent bits and 1 mantissa bit, yielding 16 possible values looked up from a literal constant table.

---

## Appendix: Mathematical Operations Reference

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **GEMV** | Expanded on first use. ✅ |
| **Q/K/V** | Expanded on first use. ✅ |
| **RoPE** | Not used in this appendix. |
| **SDPA** | Not used by acronym in this appendix. |
| **RMSNorm** | Expanded as "RMS Normalization" on first use. ✅ |
| **SiLU** | Named but never explicitly expanded as "Sigmoid Linear Unit." |
| **GELU** | Expanded as "Gaussian Error Linear Unit." ✅ |
| **SwiGLU** | Named and described functionally but the acronym is not expanded (SiLU + GLU fusion). |
| **MoE** | Used ("MoE routing") — not expanded. |
| **DeltaNet** | Referenced without definition. |
| **Mamba-2** | Referenced without definition. |
| **FFN** | Used without explicit first-use expansion (described as "feed-forward network" in context). |
| **CDF** | Used ("approximates Gaussian CDF") — not expanded as "Cumulative Distribution Function." |
| **LayerNorm** | Referenced ("simpler than LayerNorm") — not defined. |
| **FLOP** | Used ("0.25 FLOP/byte") — not expanded as "Floating-Point Operation." |
| **DRAM** | Used ("DRAM bandwidth") — not expanded. |
| **TFLOPS** | Used ("GPU TFLOPS") — not expanded. |
| **GPU** | Used throughout — not expanded. |

### Glossary of terms introduced

- **dot product** — multiply corresponding elements of two vectors and sum: `sum_i(a[i] * b[i])`, producing a scalar measuring similarity.
- **GEMV (General Matrix-Vector multiply)** — multiplying a weight matrix by an input vector, where each output element is a dot product of a matrix row with the input; the dominant operation in LLM decode (~95% of time).
- **outer product** — forms a matrix from two vectors: `A[i][j] = a[i] * b[j]`; used in DeltaNet state updates.
- **Q/K/V projections** — three independent linear transformations (GEMVs) of the same hidden state, producing Query ("what am I looking for"), Key ("what do I contain"), and Value ("what information do I carry") vectors.
- **scaled dot-product attention** — the attention mechanism `softmax(Q·Kᵀ / √d) · V` where division by √head_dim prevents scores from growing too large.
- **softmax** — converts raw scores to a probability distribution summing to 1.0: `softmax(x)[i] = exp(x[i]) / sum(exp(x))`.
- **numerical stability trick (softmax)** — subtracting the maximum value before exponentiating to prevent float overflow: `exp(x - max(x))`.
- **RMSNorm (Root Mean Square Normalization)** — normalizes a vector to unit RMS then scales by learned weights: `output = w * x / sqrt(mean(x²) + ε)`.
- **L2 normalization** — scaling a vector to unit magnitude with no learnable weights: `x / sqrt(sum(x²) + ε)`.
- **1D causal convolution** — a sliding-window operation that combines nearby values using learned weights, looking only backward in time: `y[t] = sum_i(w[i] * x[t-i])`.
- **SiLU (Sigmoid Linear Unit / Swish)** — the activation function `x * sigmoid(x)`, smooth and differentiable, used in SwiGLU FFN layers.
- **GELU (Gaussian Error Linear Unit)** — a smooth activation approximating the Gaussian CDF, used in Gemma3 FFN layers.
- **sigmoid** — the function `1 / (1 + exp(-x))` mapping any value to (0, 1), used for gating.
- **softplus** — the function `log(1 + exp(x))`, a smooth always-positive activation used for SSM timestep computation.
- **tanh (hyperbolic tangent)** — maps any value to (-1, 1): `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`; used for logit softcapping.
- **logit softcapping** — `tanh(x / cap) * cap`: smoothly clamps logits to ±cap, preventing extreme values.
- **argmax** — returns the index of the maximum value in an array; used for greedy decoding.
- **temperature scaling** — dividing logits by a temperature parameter before softmax; lower temperature → more deterministic, higher → more random.
- **top-K selection** — keeping only the K highest-scoring tokens and masking the rest to −∞ before sampling.
- **top-P / nucleus sampling** — keeping the smallest set of tokens whose cumulative probability reaches P, then masking the rest.
- **bandwidth-bound** — operations where time is dominated by memory reads/writes rather than arithmetic (GEMV, normalization, activations); quantization helps enormously.
- **compute-bound** — operations where arithmetic dominates over memory access (attention for long sequences, GEMM during prefill); GPU TFLOPS matters most.
- **arithmetic intensity** — the ratio of floating-point operations to bytes transferred (FLOP/byte); low values indicate bandwidth-bound operations.

---

## Appendix: Profiling and Debugging

### Unexplained acronyms / technical terms on first use

| Term | Status |
|------|--------|
| **GPU** | Used throughout — not expanded. |
| **CI** | Used ("Fail CI if throughput drops") — not expanded. |
| **SSE** | Not used in this appendix. |
| **CUDA** | Listed as a backend — not expanded. |
| **ROCm** | Listed as a backend — not expanded. |
| **MSL** | Not used in this appendix. |
| **TTFT** | Used ("Log tokens/sec, TTFT") — not expanded in this appendix. |
| **Tracy** | Named as an external profiling tool — not described beyond naming. |
| **TSan** | Not used in this appendix. |
| **A/B test** | Used ("A/B test optimizations") — not defined. |
| **megakernel** | Used extensively — defined implicitly by context and diagram but not given a one-line definition on first use. |

### Glossary of terms introduced

- **--profile flag** — a CLI flag that enables per-operation timing instrumentation and backend counter collection, printed after each token; incurs ~50% throughput loss due to forced GPU syncs.
- **PerfCounters** — the profiling struct (`perf.zig`) that accumulates per-operation call counts and microsecond durations, gated by an `enabled` boolean.
- **Op enum** — an enumeration of profiled operation types (emb_lookup, rms_norm, gemv_qkv, rope, sdpa, etc.) indexing into the counters arrays.
- **dispatch count** — the number of GPU kernel invocations per token; optimal range 300–600, >1000 indicates dispatch overhead dominance.
- **barrier count** — the number of GPU memory barriers per token serializing consecutive dispatches; high counts indicate serialized execution.
- **sync count** — the number of CPU/GPU round-trip flushes per token; should be ≤3, more indicates excessive CPU/GPU synchronization.
- **resetCounters()** — a function that zeroes all dispatch/barrier/sync counters at the start of each token's profiling window.
- **missing kernel policy** — the rule that GPU backends must @panic with a clear error message when a kernel is not implemented, never silently falling back to CPU.
- **CPU fallback exceptions** — the only two permitted cases where a GPU backend may silently use CPU: (1) embLookup (single-row dequant is faster on CPU than GPU dispatch overhead) and (2) tiny softmax (n < 128, where CPU SIMD beats GPU dispatch cost).
- **softmax_cpu_threshold** — the minimum vector size (128 elements) below which the GPU softmax kernel is slower than CPU SIMD, triggering a permitted CPU fallback.
- **cpuFallback()** — a Metal backend method that flushes pending GPU work and returns a CPU backend reference, ensuring the CPU reads current data for permitted fallback operations.
- **Tracy** — an external real-time profiling tool providing visual timelines, GPU queue visualization, and memory allocation tracking; not currently integrated into Agave.
- **megakernel** — a fused GPU kernel that combines multiple operations (e.g., entire FFN or entire layer) into a single dispatch, dramatically reducing dispatch and barrier counts.
- **Tier 1 megakernel (fused FFN)** — fusing gate + up + siluMul into one dispatch, saving ~48 dispatches per token for a 32-layer model.
- **Tier 2 megakernel (true megakernel)** — fusing an entire transformer layer (attention + FFN + norms) into a single dispatch with internal atomic barriers, reducing total dispatches to ~n_layers.
- **mega_grid_sync** — atomic barriers used inside a true megakernel to synchronize threadgroups without Metal memory barriers.
- **batch_mode** — a Metal backend flag that suppresses per-dispatch memory barriers, deferring synchronization to explicit points for better throughput.
- **performance regression** — a silent slowdown where the model still runs correctly but at lower throughput, detectable only through profiling.
- **perf.report()** — a function that prints a table of per-operation call counts, total time, average time, and percentage breakdown after generation completes.

---

## Coverage Status

| Chapter | Read | Unexplained terms | Glossary | Status |
|---------|------|-------------------|----------|--------|
| 15 – Chat Templates | ✅ | ✅ | ✅ | done |
| 16 – Recipe System | ✅ | ✅ | ✅ | done |
| 17 – Speculative Decoding | ✅ | ✅ | ✅ | done |
| 18 – Multi-Token Prediction | ✅ | ✅ | ✅ | done |
| 19 – PFlash & Block Sparse | ✅ | ✅ | ✅ | done |
| 20 – Diffusion LM | ✅ | ✅ | ✅ | done |
| Appendix: Atomics | ✅ | ✅ | ✅ | done |
| Appendix: Compile-Time | ✅ | ✅ | ✅ | done |
| Appendix: Math | ✅ | ✅ | ✅ | done |
| Appendix: Profiling | ✅ | ✅ | ✅ | done |

### Cross-cutting observations

1. **Recurring unexpanded acronyms across chapters**: GGUF, MoE, KV cache, CLI, BF16, RAG, VRAM, SSM, GPU, CUDA, ROCm — these are assumed known from earlier tutorial chapters but never re-introduced.
2. **Chapter 17 is the densest**: introduces ~30+ terms and 13 speculative decoding modes. Many acronyms (EAGLE, MTP, Medusa) appear in summary tables before their sections define them.
3. **Chapter 18 is the most self-contained**: carefully expands most acronyms on first use (GEMV, RoPE, SDPA, RMSNorm), serving as a good standalone reference.
4. **Appendix: Math is intentionally a reference**: terms like DeltaNet, Mamba-2, MoE are used as cross-references without re-definition, appropriate for an appendix.
5. **"SwiGLU" is never fully expanded anywhere** in these chapters — it is described functionally (SiLU gating + linear unit) but the acronym itself is not decomposed.
