---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
last_updated: "2026-03-21T17:16:38.652Z"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 7
  completed_plans: 7
---

# Project State: Agave Production-Ready LLM Inference Engine

**Updated:** 2026-03-22
**Status:** Ready to plan

---

## Project Reference

**Core Value**: Every supported model must produce correct output on every backend at full GPU speed

**Current Focus**: Establishing correctness foundation — eliminate CPU fallbacks, verify all models on all backends

**Mode**: YOLO (rapid iteration with continuous verification)

---

## Current Position

Phase: 2
Plan: Not started

## Performance Metrics

### Quality

- **Requirements coverage**: 50/50 v1 requirements mapped (100%)
- **Roadmap coherence**: 4 phases derived from natural requirement boundaries
- **Success criteria**: 2-5 observable behaviors per phase (goal-backward validated)

### Velocity

- **Estimated timeline**: 14-18 weeks total with parallelization
- **Critical path**: Phase 1 → Phase 2 → Phase 3 (Phase 4 can run concurrently)
- **Current blockers**: None (roadmap complete, ready for Phase 1 planning)

### Health

- **Coverage**: ✓ All requirements mapped to exactly one phase (no orphans, no duplicates)
- **Dependencies**: ✓ Clear phase boundaries (Phase 1 foundation, Phase 2-3 serving, Phase 4 parallelism)
- **Risks**: Documented in research/SUMMARY.md (Metal encoder switching, CUDA sm_121 hangs, MoE overflow)

---

## Accumulated Context

### Key Decisions

| Decision | Rationale | Phase Impact |
|----------|-----------|--------------|
| Combine GPU kernels + model verification into Phase 1 | Both are correctness foundation; can't verify models without working kernels | Phase 1 scope: 19 requirements (4-5 weeks) |
| Keep all production serving features together in Phase 2 | Continuous batching, PagedAttention, API, auth, metrics are tightly coupled | Phase 2 scope: 10 requirements (2-3 weeks) |
| Separate RadixAttention into Phase 3 | Depends on PagedAttention (Phase 2), but adds complexity beyond table stakes | Enables parallelization: Phase 3 + Phase 4 concurrent |
| Make Phase 4 independent of Phases 2-3 | Parallelism only needs working backends/models, not serving infrastructure | Critical path optimization (saves 2-3 weeks calendar time) |
| Phase 01 P04 | 117 | 3 tasks | 3 files |
| Phase 01 P01 | 292 | 3 tasks | 3 files |
| Phase 01 P03 | 439 | 3 tasks | 5 files |
| Phase 01 P05 | 467 | 3 tasks | 11 files |
| Phase 01 P02 | 8 | 3 tasks | 7 files |
| Phase 01 P06 | 45 | 1 tasks | 1 files |
| Phase 01 P07 | 2434 | 4 tasks | 9 files |

### Active TODOs

- [ ] Review roadmap with project stakeholders
- [ ] Confirm Phase 1 priorities: Metal SDPA vs CUDA quantized GEMV order
- [ ] Decide Metal SDPA approach: debug existing kernel vs rewrite with FlashAttention-2
- [ ] Validate CUDA sm_121 as target (vs falling back to sm_120 compilation)
- [ ] Proceed to `/gsd:plan-phase 1` when ready

### Known Blockers

**None at roadmap level.** Phase-specific blockers will surface during planning:

- Phase 1: Metal GPU SDPA wrong output (likely race condition or barrier placement)
- Phase 1: CUDA Blackwell sm_121 blockReduce hangs (compiler bug, warp-only workaround)
- Phase 1: Nemotron Nano MoE router overflow (-2.3e26 scores, needs per-block quantization)

### Research Insights

- **FlashAttention-2** (not FA-3): FA-3 requires H100+, beta-only. FA-2 is production-ready, cross-platform.
- **Online softmax mandatory**: Single-pass numerically stable softmax (running max+sum) reduces memory traffic 1.33×.
- **Metal hot path rule**: NEVER use blit encoders in token generation loop (150ms/layer stalls from encoder switching).
- **CUDA sm_121 workaround**: Warp-only reductions until CUDA toolkit fixes shared memory codegen bug.
- **RadixAttention eviction**: Frequency × cost metric (NOT simple LRU) — shared prefixes prioritized, last block evicted first.

---

## Session Continuity

### Last Session

- Initialized project with `/gsd:new-project`
- Created PROJECT.md, REQUIREMENTS.md, research/SUMMARY.md
- Defined 50 v1 requirements across 5 categories
- Completed comprehensive research synthesis (GPU kernels, serving, caching, parallelism)
- Created ROADMAP.md with 4 phases, 100% requirement coverage
- All files written to `.planning/`

### Next Session

**Expected action**: `/gsd:plan-phase 1`

**Context needed**: None (all context captured in PROJECT.md, REQUIREMENTS.md, research/SUMMARY.md, ROADMAP.md)

**Files to review**:

- `.planning/ROADMAP.md` — Phase 1 goal, requirements, success criteria
- `.planning/research/SUMMARY.md` — Metal SDPA pitfalls, CUDA quantized GEMV patterns, numerical testing
- `CLAUDE.md` — Zig standards, backend patterns, quantization rules

---

## Phase Completion History

**No phases completed yet.**

---

*State initialized: 2026-03-21*
*Last updated: 2026-03-21 after roadmap creation*
