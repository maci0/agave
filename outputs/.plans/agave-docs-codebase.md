# Audit Plan: agave-docs-codebase

## Scope
Cross-reference all 11 top-level Agave documentation files (`docs/*.md`) against the actual Zig source code. The previous review covered `docs/tutorial/` — this audit covers the **product-facing** docs: API reference, architecture overview, model parameters, kernel status, megakernel system, contributing guide, parallelism, benchmarks, and test matrix.

## Target files (docs)
1. **API.md** — HTTP API endpoints, request/response schemas, sampling params
2. **ARCHITECTURE.md** — System overview, module boundaries, data flow
3. **MODELS.md** — Model parameter tables (n_embd, n_heads, n_kv, etc.)
4. **KERNELS.md** — Kernel implementation status per backend
5. **MEGAKERNEL.md** — Megakernel tier system, building blocks, composition
6. **CONTRIBUTING.md** — How to add backends, models, quants
7. **PARALLELISM.md** — TP/PP/disagg, transports, NCCL integration
8. **BENCHMARKS.md** — Performance claims, throughput numbers
9. **TEST_MATRIX.md** — Test coverage, model × backend matrix
10. **DOCUMENTATION.md** — Tutorial index (lightweight)
11. **TODO.md** — Planned features (check for already-done items)

## Claims to check
- **API.md**: endpoint paths, JSON field names, default values, supported params vs `src/server/server.zig` and `src/server/json.zig`
- **MODELS.md**: every dimension (n_embd, n_heads, n_kv, head_dim, n_ff, n_layers, vocab_size, rope_theta) vs actual struct defaults in `src/models/*.zig`
- **KERNELS.md**: kernel status (implemented/missing) per backend — spot-check against actual `@hasDecl` and backend files
- **MEGAKERNEL.md**: tier counts, building block names, composeMSL API vs `src/backend/mega_compose.zig` and `src/backend/megakernel.zig`
- **ARCHITECTURE.md**: module names, dispatcher pattern, file paths
- **CONTRIBUTING.md**: step-by-step instructions accuracy, file references
- **PARALLELISM.md**: transport enum, CLI flags, NCCL details vs `src/parallel/transport.zig`
- **BENCHMARKS.md**: flag as unverifiable but check for internal consistency
- **TEST_MATRIX.md**: model × backend claims vs actual test code

## Method
1. Dispatch parallel `researcher` subagents to batch-read docs + source files
2. Cross-reference every factual claim (numbers, names, APIs, defaults)
3. Consolidate into single audit artifact
4. Use `verifier` subagent for inline source citations on disputed claims

## Output
- `outputs/agave-docs-codebase-audit.md`
