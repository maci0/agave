# Roadmap: Agave Production-Ready LLM Inference Engine

**Created:** 2026-03-21

## Milestones

- ✅ **v1.0 Production-Ready** — Phases 1-3, 5 (shipped 2026-03-22)

## Phases

<details>
<summary>✅ v1.0 Production-Ready (Phases 1-5) — SHIPPED 2026-03-22</summary>

- [x] Phase 1: Correctness Foundation (7/7 plans) — GPU kernel parity, all models verified
- [x] Phase 2: Production Serving (3/3 plans) — continuous batching, PagedAttention, rate limiting, auth, metrics
- [x] Phase 3: Memory Optimization (4/4 plans) — RadixAttention, TieredKvCache, prefetching, zero-copy
- [x] Phase 5: Integration Wiring (2/2 plans) — scheduler activation, model TieredKvCache swap, CLI flag wiring
- ~~Phase 4: Multi-GPU Parallelism~~ — moved to backlog (PARA-01 through PARA-12)

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

## Backlog

- [ ] **Multi-GPU Parallelism** — DeviceGroup, tensor/pipeline/expert parallelism (PARA-01 through PARA-12)

---
*For current project status, see .planning/PROJECT.md*
