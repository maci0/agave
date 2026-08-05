# Chapter 24: Advanced Features

**Prerequisites:** [Chapter 3: Feed-Forward Networks](03-feed-forward-networks.md) (MoE routing), [Chapter 5: Memory and Caching](05-memory-and-caching.md) (KV cache)

**Time:** ~20 min

> After this chapter you can explain directional steering (CLI), NLL quality scoring (library), expert profiling (library), KV checkpoint headers (library), mixed-quant splicing (tooling), and SSD expert streaming (library).

This chapter covers six related capabilities. Only **directional steering** is wired to CLI flags today. The others ship as library modules and/or Python tools; call them from code or tooling until CLI wiring lands.

---

## 1. Directional Steering (CLI)

**Directional steering** is a runtime activation edit that controls model behavior without fine-tuning. A steering file contains one normalized direction vector per layer (`n_layers × n_embd` floats). During inference, the direction is projected out of (or into) the activation after each layer:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

- **Positive scale** removes the direction (suppresses the concept/style)
- **Negative scale** amplifies the direction
- **Zero scale** = no-op

The FFN output is the recommended hook point: it carries style, behavior, and topic signals. Attention steering is available but more fragile.

```bash
# Suppress verbosity (FFN steering, scale 1.0 default)
agave model.gguf --dir-steering-file verbosity.f32 "Explain databases"

# Amplify verbosity (negative scale)
agave model.gguf --dir-steering-file verbosity.f32 --dir-steering-ffn -1 "Explain databases"

# Stronger suppression
agave model.gguf --dir-steering-file verbosity.f32 --dir-steering-ffn 2 "Explain databases"

# Also steer attention outputs
agave model.gguf --dir-steering-file verbosity.f32 --dir-steering-attn 0.5 "Explain databases"
```

### Building Direction Vectors

Direction vectors are built from paired prompt sets:

1. **Target** prompts (e.g. succinct answers) → `good-file`
2. **Contrast** prompts (e.g. verbose answers) → `bad-file`
3. Direction = mean(target activations) - mean(contrast activations), normalized per layer

```bash
python3 tools/dir-steering/build_direction.py \
    --agave ./zig-out/bin/agave \
    --model model.gguf \
    --good-file prompts_succinct.txt \
    --bad-file prompts_verbose.txt \
    --out verbosity.f32 \
    --component ffn_out \
    --n-layers 64 --n-embd 2048
```

**Implementation:** [`src/steering.zig`](../../src/steering.zig) (`DirectionalSteering`, zero-alloc `apply()` in hot path), [`src/models/qwen35.zig`](../../src/models/qwen35.zig) (hook points), [`tools/dir-steering/`](../../tools/dir-steering/) (extraction pipeline).

---

## 2. Quality Testing (NLL Scoring) — library + tooling

**Token-by-token NLL** (negative log-likelihood) measures how much probability a local model assigns to each ground-truth token from a reference model. Lower NLL = the local model more closely matches the reference.

```text
NLL = -mean(log P(correct_token_i)) across all continuation tokens
```

This is more rigorous than "does the output look right": it quantifies quality loss from quantization without a full benchmark suite.

There is **no `--eval` CLI flag** yet. Collect continuations with the Python tool, then call `eval.scoreCase` from Zig (tests or a thin harness):

```bash
# Collect greedy continuations from a reference API
python3 tools/quality-testing/collect_continuations.py \
    --endpoint https://api.deepseek.com/chat/completions \
    --model deepseek-v4-flash \
    --prompts prompts.txt \
    --out continuations.jsonl
```

```zig
// Library API in src/eval.zig
const result = eval.scoreCase(model, prompt_ids, continuation_ids) orelse return error.EvalFailed;
// result.mean_nll, result.n_correct_argmax / result.n_tokens
```

The metric also reports **argmax accuracy**: the fraction of positions where the local model's greedy prediction matches the reference token.

**Implementation:** [`src/eval.zig`](../../src/eval.zig) (`scoreCase()`, `EvalResult`), [`tools/quality-testing/`](../../tools/quality-testing/) (collection scripts).

---

## 3. Expert Hotlist Profiling — library

For MoE models, **expert profiling** tracks which routed experts are activated during inference. The resulting frequency data tells you:

- Which experts are "hot" (frequently routed) and should stay in fast memory
- Whether expert load is balanced (training quality signal)
- Which experts to pin in the SSD streaming cache (section 6)

There is **no `--expert-profile` CLI flag** yet. Use the library from MoE forward paths or tests:

```zig
var profile = try ExpertProfile.init(allocator, n_layers, n_experts);
defer profile.deinit(allocator);
profile.record(layer, expert_id); // per routed expert
profile.recordToken();
try profile.writeJson(allocator, "profile.json");
```

The profile records per-layer, per-expert activation counts. `topExperts()` extracts the top-K most active experts per layer for hotlist generation.

**Implementation:** [`src/expert_profile.zig`](../../src/expert_profile.zig) (`ExpertProfile`, zero-alloc `record()` in hot path).

---

## 4. KV Cache Disk Checkpointing — header only

**KV checkpointing** is intended to serialize KV cache state to disk so long system prompts need not be re-prefilled after a restart.

Today [`src/kvcache/checkpoint.zig`](../../src/kvcache/checkpoint.zig) implements the **versioned 28-byte header** only (`writeHeader` / `readHeader` / `validateHeader`). Full payload `save` / `load` and CLI flags are **not wired yet**.

Planned file format:

```text
[4 bytes] magic: "KVC\x01"
[4 bytes] version: u32
[4 bytes] payload_abi: u32 (bumped when KV layout changes)
[4 bytes] n_layers, [4 bytes] kv_dim, [4 bytes] n_tokens
[4 bytes] reserved
[payload] K data, then V data
```

The `payload_abi` field is separate from the file version: the outer envelope stays stable while internal KV layout (quantization type, dimension order) can change between releases without silent corruption.

**Implementation:** [`src/kvcache/checkpoint.zig`](../../src/kvcache/checkpoint.zig) (header format and validation).

---

## 5. Mixed-Quant Expert Splicing — tooling

**Mixed-quant splicing** creates a GGUF where most routed experts use an aggressive quantization (e.g. IQ2_XXS) but selected layers' experts are replaced with a higher-quality quantization (e.g. Q4_K) from a donor file.

```bash
python3 tools/mixed-quant/splice_mixed_experts.py \
    --base model-iq2.gguf \
    --donor model-q4.gguf \
    --layers 37-42 \
    --out model-mixed.gguf \
    --dry-run  # preview first
```

The result is nearly as small as the aggressive quant but with higher quality in the final layers (where quantization loss hurts output most). Only routed expert tensors are replaced; shared experts, projections, and routing weights stay from the base file.

**Implementation:** [`tools/mixed-quant/`](../../tools/mixed-quant/) (GGUF splicer).

---

## 6. SSD Expert Streaming — library

**SSD streaming** keeps a cache of hot MoE experts resident and streams cold experts from the mmap'd model file on demand.

The cache uses LRU eviction with `madvise(WILLNEED)` prefetching:

1. Router selects experts for the current token
2. For each selected expert: check if it's in the cache
3. **Cache hit**: use the resident weights directly (zero cost)
4. **Cache miss**: evict the LRU expert, `madvise(WILLNEED)` the new expert's byte range, then use the weights once faulted in

There is **no `--ssd-streaming` / `--ssd-cache-*` CLI** yet. Tiered KV SSD (`--kv-tiers vram+ram+ssd`) is a different feature. Wire `ExpertCache` from MoE paths:

```zig
var cache = try ExpertCache.init(allocator, n_layers, n_experts, 32);
defer cache.deinit(allocator);
_ = cache.touch(layer, expert_id);
cache.prefetch(layer, expert_id, weight_ptr, expert_bytes);
```

The expert hotlist (section 3) can pin profiled hot experts so they are never evicted.

**Implementation:** [`src/expert_cache.zig`](../../src/expert_cache.zig) (`ExpertCache`, LRU eviction, `madvise` prefetch).

---

## Gotchas

- **Steering direction quality depends on prompt diversity.** A direction built from 5 prompt pairs will be noisy and may cause repetition or nonsense at strong scales. Use 50-100 pairs for reliable results. Start with FFN scales between `-1` and `2`; if the model degrades, reduce the scale.
- **NLL scoring requires greedy (temperature=0) reference continuations.** If the reference was sampled with temperature > 0, the NLL metric becomes a noisy measure of sampling luck rather than model quality.
- **KV checkpoint payload ABI must match exactly** once payload I/O exists. A checkpoint saved with one KV quantization type cannot load into a session using a different type. The `payload_abi` field is meant to catch that.
- **SSD streaming adds latency variance** when wired. Cache hits are free; misses incur SSD read latency. Expert profiling + pinning reduces but does not eliminate variance.
- **Mixed-quant splicing requires compatible GGUFs.** The base and donor files must share architecture, layer count, expert count, and tensor naming. Splicing incompatible files produces silent corruption, not an error.

---

**In the code:** [`src/steering.zig`](../../src/steering.zig) (directional steering, CLI), [`src/eval.zig`](../../src/eval.zig) (NLL `scoreCase`), [`src/expert_profile.zig`](../../src/expert_profile.zig) (expert profiling), [`src/kvcache/checkpoint.zig`](../../src/kvcache/checkpoint.zig) (checkpoint header), [`src/expert_cache.zig`](../../src/expert_cache.zig) (SSD expert streaming), [`tools/`](../../tools/) (Python tooling)

**Next:** [Appendix: Troubleshooting →](appendix-troubleshooting.md) | **Back:** [Chapter 23: Server / HTTP API ←](23-server-http-api.md)

---

## Glossary

**argmax accuracy** — The fraction of positions where the local model's greedy prediction matches the reference model's ground-truth token.

**directional steering** — Runtime activation editing that projects a learned direction vector out of (or into) layer activations to control model behavior without fine-tuning.

**expert cache** — A fixed-size LRU cache of MoE routed-expert weight slabs, enabling SSD streaming for models that don't fit in RAM.

**expert hotlist** — The set of most-frequently-routed experts per layer, identified by profiling and optionally pinned in the expert cache to guarantee fast-path access.

**KV checkpoint** — A versioned binary envelope for serialized KV cache state; header encode/validate ships today, full save/load is not wired yet.

**mixed-quant splicing** — Creating a GGUF where selected layers' routed experts use a higher quantization from a donor file while other layers keep the base file's aggressive quantization.

**NLL (negative log-likelihood)** — The mean negative log probability assigned by a model to ground-truth continuation tokens; lower = better quality match.

**SSD streaming** — Demand-paged expert weight loading where cold experts are faulted in from a memory-mapped model file via `madvise(WILLNEED)`, backed by SSD.

**steering scale** — The coefficient controlling how strongly a directional steering vector is projected out of (positive) or into (negative) the activation.
