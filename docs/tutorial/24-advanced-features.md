# Chapter 24: Advanced Features

**Prerequisites:** [Chapter 3: Feed-Forward Networks](03-feed-forward-networks.md) (MoE routing), [Chapter 5: Memory and Caching](05-memory-and-caching.md) (KV cache)

**Time:** ~20 min

> After this chapter you can explain directional steering, NLL quality testing, expert profiling, KV checkpointing, mixed-quant splicing, and SSD expert streaming.

This chapter covers six features that extend Agave beyond basic inference: runtime behavior control, quality measurement, MoE optimization, session persistence, model compression tooling, and memory-constrained expert loading.

---

## 1. Directional Steering

**Directional steering** is a runtime activation edit that controls model behavior without fine-tuning. A steering file contains one normalized direction vector per layer (`n_layers × n_embd` floats). During inference, the direction is projected out of (or into) the activation after each layer:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

- **Positive scale** removes the direction (suppresses the concept/style)
- **Negative scale** amplifies the direction
- **Zero scale** = no-op

The FFN output is the recommended hook point — it carries style, behavior, and topic signals. Attention steering is available but more fragile.

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

## 2. Quality Testing (NLL Scoring)

**Token-by-token NLL** (negative log-likelihood) measures how much probability a local model assigns to each ground-truth token from a reference model. Lower NLL = the local model more closely matches the reference.

```text
NLL = -mean(log P(correct_token_i)) across all continuation tokens
```

This is more rigorous than "does the output look right" — it directly quantifies quality loss from quantization without requiring full benchmark suites.

### Workflow

```bash
# 1. Collect greedy continuations from official API
python3 tools/quality-testing/collect_continuations.py \
    --endpoint https://api.deepseek.com/chat/completions \
    --model deepseek-v4-flash \
    --prompts prompts.txt \
    --out continuations.jsonl

# 2. Score local model against those continuations
agave model-q4.gguf --eval continuations.jsonl    # NLL: 1.234
agave model-q8.gguf --eval continuations.jsonl    # NLL: 0.987
agave model-f16.gguf --eval continuations.jsonl   # NLL: 0.954
```

The metric also reports **argmax accuracy** — the fraction of positions where the local model's greedy prediction matches the reference token.

**Implementation:** [`src/eval.zig`](../../src/eval.zig) (`scoreCase()`, `EvalResult`), [`tools/quality-testing/`](../../tools/quality-testing/) (collection scripts).

---

## 3. Expert Hotlist Profiling

For MoE models, **expert profiling** tracks which routed experts are activated during inference. The resulting frequency data tells you:

- Which experts are "hot" (frequently routed) and should stay in fast memory
- Whether expert load is balanced (training quality signal)
- Which experts to pin in the SSD streaming cache (section 6)

```bash
# Run inference with profiling enabled
agave model.gguf --expert-profile profile.json "long prompt..."
```

The profile records per-layer, per-expert activation counts. The `topExperts()` function extracts the top-K most active experts per layer for hotlist generation.

**Implementation:** [`src/expert_profile.zig`](../../src/expert_profile.zig) (`ExpertProfile`, zero-alloc `record()` in hot path).

---

## 4. KV Cache Disk Checkpointing

**KV checkpointing** serializes the KV cache state to disk so that long system prompts and conversation prefixes don't need re-prefilling after a server restart.

The file format is versioned:

```text
[4 bytes] magic: "KVC\x01"
[4 bytes] version: u32
[4 bytes] payload_abi: u32 (bumped when KV layout changes)
[4 bytes] n_layers, [4 bytes] kv_dim, [4 bytes] n_tokens
[payload] K data, then V data
```

The `payload_abi` field is separate from the file version — the outer envelope stays stable while internal KV layout (quantization type, dimension order) can change between releases without silent corruption.

**Implementation:** [`src/kvcache/checkpoint.zig`](../../src/kvcache/checkpoint.zig) (header format, validation).

---

## 5. Mixed-Quant Expert Splicing

**Mixed-quant splicing** creates a GGUF where most routed experts use an aggressive quantization (e.g. IQ2_XXS) but selected layers' experts are replaced with a higher-quality quantization (e.g. Q4_K) from a donor file.

```bash
python3 tools/mixed-quant/splice_mixed_experts.py \
    --base model-iq2.gguf \
    --donor model-q4.gguf \
    --layers 37-42 \
    --out model-mixed.gguf \
    --dry-run  # preview first
```

The result is a model that's nearly as small as the aggressive quant but with higher quality in the final layers (where quantization loss has the most impact on output quality). Only routed expert tensors are replaced — shared experts, projections, and routing weights stay from the base file.

**Implementation:** [`tools/mixed-quant/`](../../tools/mixed-quant/) (GGUF splicer).

---

## 6. SSD Expert Streaming

**SSD streaming** enables running MoE models that don't fit in RAM by keeping only a cache of hot experts resident and streaming cold experts from the mmap'd model file on demand.

The cache uses LRU eviction with `madvise(WILLNEED)` prefetching:

1. Router selects experts for the current token
2. For each selected expert: check if it's in the cache
3. **Cache hit**: use the resident weights directly (zero cost)
4. **Cache miss**: evict the LRU expert, `madvise(WILLNEED)` the new expert's byte range to trigger background SSD page-in, then use the weights once they're faulted in

```bash
# Enable SSD streaming with 32 expert slots cached
agave model.gguf --ssd-streaming --ssd-cache-experts 32 "prompt"

# Set explicit byte budget
agave model.gguf --ssd-streaming --ssd-cache-bytes 4G "prompt"
```

The expert hotlist (section 3) integrates with the cache: profiled hot experts can be pinned so they're never evicted, ensuring the most-routed experts always hit the fast path.

**Implementation:** [`src/expert_cache.zig`](../../src/expert_cache.zig) (`ExpertCache`, LRU eviction, `madvise` prefetch).

---

## Gotchas

- **Steering direction quality depends on prompt diversity.** A direction built from 5 prompt pairs will be noisy and may cause repetition or nonsense at strong scales. Use 50-100 pairs for reliable results. Start with FFN scales between `-1` and `2`; if the model degrades, reduce the scale.
- **NLL scoring requires greedy (temperature=0) reference continuations.** If the reference was sampled with temperature > 0, the NLL metric becomes a noisy measure of sampling luck rather than model quality.
- **KV checkpoint payload ABI must match exactly.** A checkpoint saved with one KV quantization type (e.g. TurboQuant) cannot be loaded into a session using a different type (e.g. f16). The `payload_abi` field catches this, but the mismatch error doesn't tell you *which* setting differs.
- **SSD streaming adds latency variance.** Cache hits are free; cache misses incur SSD read latency (tens of microseconds on NVMe, milliseconds on SATA). Token generation speed will be bimodal — fast for common expert paths, slower for rare ones. Expert profiling + pinning reduces but doesn't eliminate this variance.
- **Mixed-quant splicing requires compatible GGUFs.** The base and donor files must have the same architecture, layer count, expert count, and tensor naming. Splicing between incompatible files produces silent corruption, not an error.

---

**In the code:** [`src/steering.zig`](../../src/steering.zig) (directional steering), [`src/eval.zig`](../../src/eval.zig) (NLL evaluation), [`src/expert_profile.zig`](../../src/expert_profile.zig) (expert profiling), [`src/kvcache/checkpoint.zig`](../../src/kvcache/checkpoint.zig) (KV checkpointing), [`src/expert_cache.zig`](../../src/expert_cache.zig) (SSD expert streaming), [`tools/`](../../tools/) (Python tooling)

**Next:** [Appendix: Troubleshooting →](appendix-troubleshooting.md) | **Back:** [Chapter 23: Server / HTTP API ←](23-server-http-api.md)

---

## Glossary

**argmax accuracy** — The fraction of positions where the local model's greedy prediction matches the reference model's ground-truth token.

**directional steering** — Runtime activation editing that projects a learned direction vector out of (or into) layer activations to control model behavior without fine-tuning.

**expert cache** — A fixed-size LRU cache of MoE routed-expert weight slabs, enabling SSD streaming for models that don't fit in RAM.

**expert hotlist** — The set of most-frequently-routed experts per layer, identified by profiling and optionally pinned in the expert cache to guarantee fast-path access.

**KV checkpoint** — A versioned binary file storing serialized KV cache state (keys and values for all layers at all cached positions), enabling session persistence across restarts.

**mixed-quant splicing** — Creating a GGUF where selected layers' routed experts use a higher quantization from a donor file while other layers keep the base file's aggressive quantization.

**NLL (negative log-likelihood)** — The mean negative log probability assigned by a model to ground-truth continuation tokens; lower = better quality match.

**SSD streaming** — Demand-paged expert weight loading where cold experts are faulted in from a memory-mapped model file via `madvise(WILLNEED)`, backed by SSD.

**steering scale** — The coefficient controlling how strongly a directional steering vector is projected out of (positive) or into (negative) the activation.
