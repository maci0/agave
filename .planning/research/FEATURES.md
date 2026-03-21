# Feature Landscape

**Domain:** LLM Inference Engine (Production Serving)
**Researched:** 2026-03-21

## Table Stakes

Features users expect. Missing = product feels incomplete or unusable in production.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Continuous Batching** | Standard in all production engines (vLLM, SGLang, TensorRT-LLM). Dynamically replaces completed sequences with new ones at each iteration instead of waiting for entire batch. 3-5× throughput vs static batching. | High | Agave has basic request handling but no continuous batching scheduler. Requires request queue + dynamic slot management + preemption logic. |
| **PagedAttention** | Near-zero memory waste (4% vs 60-80% waste in naive systems). Allows significantly larger batch sizes. Industry standard since vLLM (2023). | Medium-High | Agave has PagedAttention implementation (`src/kvcache/manager.zig`) but not integrated into model inference loop or server. Requires block tables + allocator + indirect KV access in attention kernels. |
| **OpenAI-Compatible API** | De facto standard. All major engines (vLLM, SGLang, llama.cpp, TensorRT-LLM, TGI) expose `/v1/chat/completions`. Users expect drop-in compatibility. | Low | Agave has HTTP server with basic endpoints. Needs full OpenAI schema compatibility: messages format, tools/function calling, response_format, logprobs, etc. |
| **SSE Streaming** | Real-time token delivery. Every production LLM API (OpenAI, Anthropic, all inference engines) uses Server-Sent Events for streaming. Non-negotiable for chat UX. | Low | Agave HTTP server has SSE streaming (`src/server.zig`). Verify OpenAI event format (`data: [DONE]`, etc.). |
| **Request Timeouts** | Prevent hung requests from holding GPU resources forever. Production systems always set 30-120s timeouts. Critical for multi-tenant stability. | Low | Missing. Needs per-request timeout + cancellation signal + cleanup on timeout. |
| **Rate Limiting** | Prevent abuse, manage resource contention. Expected in any multi-user API. Token-aware (requests/min + tokens/min) is standard. | Medium | Missing. Needs sliding window counters per API key/user + queue backpressure when limits exceeded. |
| **GPU GEMV for All Quant Formats** | Metal/Vulkan support Q4_K/Q5_K/Q6_K. CUDA only has f32/bf16/f16/q8_0/q4_0. Missing Q4_K/Q5_K/Q6_K/FP8 on CUDA = forces CPU fallback on most quantized models = unacceptable performance. | Medium | Agave gaps: CUDA Q4_K/Q5_K/Q6_K/FP8, Metal/Vulkan NVFP4/MXFP4 (when those formats ship). Kernel implementation + testing against golden outputs. |
| **Metrics Endpoint** | Prometheus `/metrics` endpoint. Standard observability. llama.cpp, vLLM, TGI all expose this. Essential for production monitoring (queue depth, latency percentiles, throughput, GPU util, KV cache usage). | Low | Missing. Needs Prometheus text format exporter with key metrics: `requests_total`, `request_duration_seconds{p50,p95,p99}`, `tokens_generated_total`, `queue_length`, `kv_cache_usage_ratio`. |
| **Authentication** | API key validation. Production APIs don't run unauthenticated. llama.cpp has `--api-key`, all cloud engines have auth. | Low | Missing. Needs `--api-key` CLI flag + header validation (`Authorization: Bearer <key>`) + 401 responses. |
| **Multi-Model Support** | Load multiple models, select via API parameter. Users expect `/v1/models` list + `model` field in requests. Reduces deployment complexity (one server, many models). | Medium | Missing. Needs model registry + lazy loading + per-request model selection. Memory management critical (can't keep all in VRAM). |
| **Graceful Degradation** | Handle OOM, model load failures, device resets without crashing server. Production systems log errors + return 503 + stay alive. | Low-Medium | Partially present (error handling exists). Needs hardening: catch GPU OOM, return 503 with retry-after, drain in-flight requests before shutdown. |
| **Concurrent Requests** | Handle multiple in-flight requests. Single-request engines (like Ollama) are toys, not production tools. Minimum: queue + serial processing. Ideal: continuous batching. | Medium (queue), High (batching) | Agave server is single-request serial. Needs request queue + worker threads or async event loop. Continuous batching is separate (harder). |

## Differentiators

Features that set products apart. Not expected, but valued when present. Competitive advantage.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **RadixAttention (Prefix Caching)** | Automatic KV cache reuse via radix tree. SGLang reports 5× throughput on shared-prefix workloads. Eliminates redundant computation for repeated prompts (RAG, multi-turn chat, agents). | High | Agave has RadixTree implementation (`src/kvcache/manager.zig`) but not integrated into server or scheduler. Requires LRU eviction + request routing + cache hit metrics. Key differentiator if shipped. |
| **Speculative Decoding** | 1.3-1.7× speedup via draft model. Trending in 2026 (vLLM P-EAGLE, SGLang). Significant latency win for decode-bound workloads. | High | Not present. Requires draft model loading + parallel verification + acceptance sampling. High complexity, high value. Good post-v1 feature. |
| **Chunked Prefill** | Break large prefills into chunks, batch with decode. Maximizes resource efficiency (prefill = compute-bound, decode = memory-bound). Reduces head-of-line blocking for long prompts. | High | Not present. Requires scheduler rewrite to split prefill into chunks + interleave with decode batches. Meta/vLLM technique. Complex but valuable for mixed workloads. |
| **Structured Output (Constrained Decoding)** | Guarantee valid JSON/regex output via logit filtering. XGrammar, llguidance, Outlines. OpenAI Structured Outputs credited llguidance. Expected in 2026 for production agents/tools. | Medium-High | Not present. Needs grammar compiler + logit processor + schema validation. Medium complexity if using existing library (XGrammar). High value for agentic use cases. |
| **Multi-LoRA Serving** | Serve hundreds of LoRA adapters from one base model. S-LoRA, LoRAX, NVIDIA NIM. Massive cost savings for multi-tenant fine-tuned deployments. Swap adapters per-request without reloading base. | High | Not present. Requires adapter registry + dynamic loading + batching heterogeneous ranks + memory management. Niche but valuable for fine-tuning shops. |
| **Prefill-Decode Disaggregation** | Separate prefill and decode onto different GPU pools. Prefill = compute-intensive (use A100/H100), decode = memory-bandwidth (use cheaper GPUs). Meta shows 3× TTFT improvement at high concurrency. | Very High | Not present. Requires distributed architecture + KV cache transfer between clusters + scheduler rewrite. Advanced optimization, likely out of scope for now. |
| **Tensor/Pipeline Parallelism** | Scale beyond single GPU. Tensor parallel = shard layers across GPUs (low latency, needs NVLink). Pipeline parallel = split layers across nodes (less interconnect, higher latency). Required for 70B+ models on consumer GPUs. | Very High | Not present. Requires distributed runtime + all-reduce ops + pipeline scheduler + NCCL/RCCL integration. Essential for large models but massive scope increase. |
| **Quantization During Serving** | AWQ, GPTQ, FP8 calibration at load time. Serve any HF model without pre-quantizing. TensorRT-LLM AutoDeploy does this. Reduces storage (one checkpoint = many quant levels). | Medium-High | Not present. Agave loads pre-quantized GGUF/SafeTensors. Adding runtime quantization = calibration pass + weight repack. Medium value (pre-quant is fine for most users). |
| **Vision/Multimodal Support** | Image + text inputs. Qwen3-VL, GLM-4.6V, InternVL3 are standard in 2026. Expands use cases to document QA, image understanding, VLAs. | Medium (single image), High (video, interleaved) | Agave is text-only. Adding vision = image encoder + cross-attention or early fusion + format support (base64 images in API). Medium complexity for basic support. Valuable for 2026+ relevance. |
| **Function/Tool Calling** | Structured function schema + execution loop. Part of OpenAI API spec. Required for agents. All major engines support this. | Medium | Not present. Needs `tools` field parsing + structured output for function calls + stop on tool invocation. Overlaps with structured output feature. |
| **Adaptive Routing** | Route requests to different models based on complexity. 60-80% of queries can use smaller/faster models. Reduces cost 3-10× with minimal accuracy loss. Kthena, SGLang smart routing. | Medium-High | Not present. Requires model size ladder + complexity classifier + fallback logic. Differentiator for cost-conscious deployments. |
| **Native NVFP4/MXFP4 Support** | 4-bit microscaled floating-point (Blackwell Tensor Cores). NVIDIA's newest format (2025+). TensorRT-LLM ships this. Agave has ops in `quant.zig` but not GPU kernels. Early adopter advantage. | Medium | Partially present (CPU dequant helpers exist). Needs CUDA/ROCm kernels for native Tensor Core dispatch. Medium complexity, differentiation for Blackwell+ users. |
| **Dynamic Batching + Priority Queues** | Prioritize low-latency requests, batch background requests. Fairness policies for multi-tenant. Advanced scheduler feature. | Medium | Not present. Requires priority queue + preemption + fairness policies (e.g., max-min fairness). Differentiates in enterprise multi-tenant scenarios. |
| **Edge/Mobile Deployment** | Run on Jetson, iOS, Android. TensorRT Edge-LLM SDK, llama.cpp mobile. Expanding use case beyond servers. | Medium-High | Out of scope per PROJECT.md. Worth noting as conscious decision not to pursue (focus on desktop/server). |

## Anti-Features

Features to explicitly NOT build. Either low value, high cost, or misaligned with project goals.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Training/Fine-tuning** | Completely different problem domain. Requires backward pass, optimizer state, gradient checkpointing, distributed training (FSDP/DeepSpeed). Massive scope creep. Many specialized tools exist (Axolotl, torchtune, etc.). | Stay inference-only. Users who need fine-tuning use HF Trainer → export to GGUF/SafeTensors → load in Agave. Clear boundary. |
| **Custom Prompt Templating DSL** | Chat templates are data, not code. Every model has its own format. Don't invent a templating language — use simple string substitution or vendor formats (Jinja2 subset if needed). | Keep `src/chat_template.zig` as simple data structures (prefix/suffix per role + EOG tokens). Use recipe system for defaults. No Turing-complete templating. |
| **Built-in RAG/Vector Search** | Orthogonal concern. Users building RAG systems use dedicated vector DBs (Qdrant, Weaviate, pgvector). Agave is the inference engine, not the application framework. | Provide OpenAI-compatible API. Users integrate Agave into their RAG stack via API calls. Don't bundle embeddings/chunking/retrieval. |
| **Model Zoo/Auto-Download** | Maintenance nightmare. HuggingFace Hub already does this well. Model URLs break, formats change, licensing is complex. Don't become a model registry. | Users download models themselves (HF CLI, wget, etc.). Agave loads local files. Provide example commands in docs but don't auto-fetch. |
| **Web UI for Inference** | Scope creep into UI development. llama.cpp has a basic UI, Ollama has one, LM Studio exists. Agave is a library/engine, not an app. | Provide OpenAI-compatible API. Users choose their own UI (Open WebUI, LibreChat, etc.) or build custom. Focus on API quality, not UI. |
| **Windows Support** | Cross-compilation complexity (CUDA on Windows, Metal doesn't exist, Vulkan Windows driver issues). Small market for production LLM servers on Windows (Linux dominates cloud/edge). | Linux + macOS only per PROJECT.md. Users needing Windows can use WSL2 (works well, documented by NVIDIA for RTX 50 series). |
| **Prompt Caching at Application Layer** | Confuses responsibility boundaries. KV cache (RadixAttention) handles this at engine level. Don't build a separate prompt cache in the HTTP server — that's redundant and error-prone. | Use RadixAttention for automatic prefix reuse. If users want semantic caching (embedding similarity), they build it outside Agave using vector DBs. |
| **Proprietary Binary Format** | Creates lock-in, breaks interoperability. GGUF and SafeTensors are open standards with broad ecosystem support. Inventing a new format = forcing users to convert = friction. | Support GGUF (ggml ecosystem), SafeTensors (HF ecosystem), and potentially ONNX in future. Never create `.agave` or custom binary format. |
| **Built-in Observability Platform** | Prometheus + Grafana is the standard. Don't build a monitoring UI or time-series DB. Exposing `/metrics` in Prometheus format is table stakes. Anything beyond that is scope creep. | Expose `/metrics` endpoint with key metrics (latency, throughput, queue depth, KV cache usage, error rates). Users send to their own Prometheus/Datadog/etc. Provide example Grafana dashboard JSON but don't host it. |
| **Model Conversion Tools** | HuggingFace Transformers → GGUF/SafeTensors conversion is well-solved (llama.cpp scripts, mlx tools, HF exporters). Don't duplicate this effort. | Document how to convert models using existing tools. Link to llama.cpp conversion scripts, MLX exporters, etc. Don't write converters. |
| **Multi-Cloud Abstraction** | Kubernetes, AWS/GCP/Azure deployment, container orchestration. These are deployment concerns, not inference engine concerns. Adding cloud-specific code = vendor lock-in. | Provide Docker container (done). Document deployment patterns (Kubernetes YAML examples, Terraform modules). Don't build cloud-specific integrations. |
| **Python Bindings** | Adds language binding maintenance burden. Python users want vLLM/SGLang/llama.cpp (mature, feature-complete). Agave's value is Zig-native cross-platform portability, not Python ecosystem integration. | Focus on C API (zero-copy FFI) + OpenAI HTTP API. If users need Python, they use `requests` library against HTTP API or wrap C API themselves. Don't maintain official Python package. |
| **Automatic Model Selection** | "Give me the best model for this task" requires task classification, model benchmarking, cost/latency tradeoffs. This is an orchestration layer problem (LangChain, Semantic Kernel), not inference engine. | Users specify model explicitly via API `model` field. Provide documentation on model selection criteria but don't auto-select. |
| **Blockchain/Crypto Integration** | Zero technical value for inference engine. Adds complexity, regulatory risk, reputational risk. No production LLM inference engine does this. | Never. If users want decentralized inference marketplace, they build orchestration layer on top of Agave's HTTP API. |

## Feature Dependencies

```
Continuous Batching → PagedAttention (memory efficiency at scale)
RadixAttention → PagedAttention (radix tree uses paged blocks)
Multi-LoRA Serving → Continuous Batching (batch heterogeneous adapters)
Structured Output → OpenAI API (tools/response_format fields)
Function Calling → Structured Output (functions are constrained schema)
Speculative Decoding → Continuous Batching (verify draft in batch)
Prefill-Decode Disaggregation → Continuous Batching + PagedAttention (KV transfer between clusters)
Tensor/Pipeline Parallelism → Continuous Batching (distribute batch across GPUs/nodes)
Priority Queues → Continuous Batching (preempt low-priority requests)
Adaptive Routing → Multi-Model Support (route to different models)
```

## MVP Recommendation

**Prioritize (Production Readiness v1.0):**

1. **Continuous Batching** — Single biggest throughput multiplier. Table stakes for any multi-user deployment. Without this, Agave is a toy, not a production engine.
2. **PagedAttention Integration** — Already implemented, needs integration into model loop + server. Unlocks larger batches + multi-tenant serving.
3. **OpenAI API Compatibility** — HTTP server exists but needs full schema compliance. Low effort, high compatibility value.
4. **Request Timeouts + Rate Limiting** — Basic production hygiene. Prevents runaway requests + abuse.
5. **Metrics Endpoint** — Observability is non-negotiable. Prometheus `/metrics` is standard, low effort.
6. **Authentication** — API key validation. Low effort, required for any public deployment.
7. **GPU GEMV Completion** — CUDA Q4_K/Q5_K/Q6_K/FP8. Can't ship with CPU fallbacks on common formats.
8. **Concurrent Request Handling** — Request queue + worker pool. Foundation for continuous batching.

**Defer (Post-v1.0):**

- **RadixAttention** — High value but complex. Needs LRU eviction + routing logic + metrics. Ship PagedAttention first, add radix tree optimization in v1.1+.
- **Multi-Model Support** — Nice-to-have but not critical for single-model deployments (most common case). Add after core batching works.
- **Structured Output** — Trending in 2026 but requires grammar compiler integration (XGrammar, Outlines). Medium complexity, add in v1.2+ when agentic use cases mature.
- **Speculative Decoding** — High complexity, significant engineering effort. Research shows 1.3-1.7× gains. Good v2.0 feature after batching is solid.
- **Chunked Prefill** — Advanced scheduler optimization. Needs batching to exist first. v1.3+.
- **Multi-LoRA Serving** — Niche use case (fine-tuning shops). High complexity. Only build if user demand is strong.
- **Vision/Multimodal** — Expands use cases but adds significant complexity (image encoders, format handling). v2.0+ feature.
- **Tensor/Pipeline Parallelism** — Massive scope increase. Only needed for 70B+ models on consumer GPUs. Out of scope for v1.0 (focus on single-GPU performance first).

**Never Build (Anti-Features):**

- Training/fine-tuning, custom templating DSL, built-in RAG/vector search, model zoo/auto-download, web UI, Windows support (use WSL2), prompt caching at app layer, proprietary format, built-in observability platform, model conversion tools, multi-cloud abstraction, Python bindings, automatic model selection, blockchain integration.

## Complexity vs Value Matrix

**High Value, Low-Medium Complexity (Do First):**
- Continuous batching (critical path)
- PagedAttention integration (already implemented)
- OpenAI API compatibility (minor additions)
- Metrics endpoint (standard Prometheus format)
- Request timeouts + rate limiting (production hygiene)
- Authentication (API key check)
- GPU GEMV completion (kernel work, well-scoped)

**High Value, High Complexity (Do Later):**
- RadixAttention with LRU eviction
- Structured output / constrained decoding
- Speculative decoding
- Chunked prefill
- Vision/multimodal support

**Medium Value, High Complexity (Evaluate Demand):**
- Multi-LoRA serving (niche but valuable for specific users)
- Tensor/pipeline parallelism (required for 70B+ on consumer GPUs)
- Prefill-decode disaggregation (advanced optimization)

**Low Value (Don't Build):**
- All items in Anti-Features table

## Production Readiness Checklist

Based on 2026 industry standards (vLLM, SGLang, TensorRT-LLM, llama.cpp):

**Core Serving:**
- [ ] Continuous batching
- [ ] PagedAttention integrated (already exists in codebase)
- [ ] Request queue + concurrent handling
- [ ] Request timeouts (30-120s configurable)
- [ ] Graceful shutdown (drain in-flight requests)

**API:**
- [ ] OpenAI `/v1/chat/completions` full compatibility
- [ ] SSE streaming (already exists, verify format)
- [ ] `/v1/models` endpoint (list available models)
- [ ] Authentication (API key validation)
- [ ] Rate limiting (requests/min + tokens/min)
- [ ] Error handling (proper HTTP status codes, retry-after headers)

**Performance:**
- [ ] GPU GEMV for all quantization formats on all backends (CUDA gaps: Q4_K/Q5_K/Q6_K/FP8)
- [ ] Zero allocations/locks/syscalls in hot path (already achieved per CLAUDE.md)
- [ ] Metrics: TTFT p50/p95/p99, throughput (tok/s), queue depth, KV cache utilization

**Observability:**
- [ ] Prometheus `/metrics` endpoint
- [ ] Structured logging (JSON logs with request_id, model, latency, tokens)
- [ ] Health check endpoint (`/health`, `/ready`)

**Deployment:**
- [ ] Docker container (already exists)
- [ ] Example Kubernetes YAML (deployment + service + HPA)
- [ ] Example Prometheus scrape config + Grafana dashboard

**Documentation:**
- [ ] API reference (OpenAI compatibility map)
- [ ] Deployment guide (Docker, K8s, bare metal)
- [ ] Performance tuning guide (batch size, KV cache size, GPU selection)
- [ ] Model compatibility matrix (which models work, known issues)

## Notes

**2026 Trends:**
- Continuous batching is universal (every engine has it)
- PagedAttention is table stakes (vLLM proved the value in 2023)
- RadixAttention is the next frontier (SGLang deployment on 400K GPUs shows adoption)
- Structured output is rapidly becoming expected (OpenAI, Anthropic, Google all ship it natively)
- Multi-LoRA serving is maturing (NVIDIA NIM, S-LoRA, LoRAX)
- Prefill-decode disaggregation is cutting-edge (Meta research, not yet widespread)

**Agave's Position:**
- Strong foundation: pure Zig, multi-backend, extensive quantization, zero-copy mmap, correct output on tested models
- Key gaps preventing production use: no continuous batching, PagedAttention not integrated, single-request server, missing CUDA quant kernels, no metrics/auth/rate limiting
- Differentiator opportunity: RadixAttention (already implemented but not integrated), NVFP4/MXFP4 (early support for Blackwell), pure Zig cross-platform (unique in the ecosystem)

**Strategic Recommendation:**
Focus on **production readiness v1.0** (continuous batching + PagedAttention integration + full OpenAI API + observability + missing GPU kernels). This moves Agave from "research prototype" to "production-viable engine." After v1.0, add **RadixAttention + structured output** (high-value differentiators). Leave **speculative decoding + chunked prefill + multi-LoRA + parallelism** for v2.0+ when core serving is rock-solid.

## Sources

- [vLLM GitHub](https://github.com/vllm-project/vllm) — Continuous batching, PagedAttention reference
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System | vLLM Blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [How to Speed up AI Inference with vLLM Continuous Batching - Voice.ai](https://voice.ai/hub/tts/vllm-continuous-batching/)
- [Paged Attention from First Principles | Hamza's Blog](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/)
- [SGLang GitHub](https://github.com/sgl-project/sglang) — RadixAttention, structured output
- [Fast and Expressive LLM Inference with RadixAttention and SGLang | LMSYS Org](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM vs SGLang vs LMDeploy: Fastest LLM Inference Engine in 2026?](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Welcome to TensorRT LLM's Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Self-host LLMs in production with llama.cpp llama-server](https://docs.servicestack.net/ai-server/llama-server)
- [Using llama.cpp to self-host Large Language Models in Production - ServiceStack](https://servicestack.net/posts/hosting-llama-server)
- [Disaggregated Prefill and Decode - Perplexity](https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode)
- [Prefill-decode disaggregation | LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)
- [P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM | AWS](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)
- [Scaling LLM Inference: Innovations in Tensor Parallelism, Context Parallelism, and Expert Parallelism - Engineering at Meta](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [Parallelism and Scaling - vLLM](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Data, tensor, pipeline, expert and hybrid parallelisms | LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism)
- [Observability for LLM Systems: Metrics, Traces, Logs - Rost Glukhov](https://www.glukhov.org/observability/observability-for-llm-systems)
- [Monitor LLM Inference in Production (2026): Prometheus & Grafana](https://dev.to/rosgluk/monitor-llm-inference-in-production-2026-prometheus-grafana-for-vllm-tgi-llamacpp-1o1h)
- [The complete guide to LLM observability for 2026](https://portkey.ai/blog/the-complete-guide-to-llm-observability/)
- [LLM quantization | LLM Inference Handbook](https://bentoml.com/llm/getting-started/llm-quantization)
- [LLM Quantization Guide: GGUF vs AWQ vs GPTQ vs bitsandbytes Compared (2026)](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)
- [Building a Production LLM API Server: FastAPI + vLLM Complete Guide (2026)](https://blog.premai.io/building-a-production-llm-api-server-fastapi-vllm-complete-guide-2026/)
- [OpenAI-Compatible Server - vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [How Structured Outputs and Constrained Decoding Work | Let's Data Science](https://www.letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json)
- [LLM Structured Output in 2026: Stop Parsing JSON with Regex](https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk)
- [Structured outputs | LLM Inference Handbook](https://bentoml.com/llm/getting-started/tool-integration/structured-outputs)
- [LoRA Adapters - vLLM](https://docs.vllm.ai/en/latest/features/lora/)
- [Deploy multi-LoRA adapters on LLMs | Anyscale Docs](https://docs.anyscale.com/llm/serving/multi-lora)
- [Recipe for Serving Thousands of Concurrent LoRA Adapters | LMSYS Org](https://lmsys.org/blog/2023-11-15-slora/)
- [Seamlessly Deploying a Swarm of LoRA Adapters with NVIDIA NIM | NVIDIA Technical Blog](https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/)
- [Rate Limiting and Backpressure for LLM APIs](https://dasroot.net/posts/2026/02/rate-limiting-backpressure-llm-apis/)
- [Rate Limiting in AI Gateway : The Ultimate Guide](https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway)
- [How to Implement LLM Rate Limiting](https://oneuptime.com/blog/post/2026-01-30-llm-rate-limiting/view)
- [Multimodal AI: The Best Open-Source Vision Language Models in 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [Top 10 Vision Language Models in 2026 | Benchmark, Use Cases](https://dextralabs.com/blog/top-10-vision-language-models/)
- [7 Common Pitfalls in Enterprise LLM Deployment | Dextralabs](https://dextralabs.com/blog/llm-deployment-pitfalls-enterprise-ai/)
- [Deploying LLMs in Production: Lessons from the Trenches | Adnan Masood](https://medium.com/@adnanmasood/deploying-llms-in-production-lessons-from-the-trenches-a742767be721)
- [LLM Docker Deployment: Complete Production Guide (2026)](https://blog.premai.io/llm-docker-deployment-complete-production-guide-2026/)
- [Choosing the right inference framework | LLM Inference Handbook](https://bentoml.com/llm/getting-started/choosing-the-right-inference-framework)
- [10 Best vLLM Alternatives for LLM Inference in Production (2026)](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
