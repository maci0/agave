# Milestones

## v1.0 Production-Ready Inference Engine (Shipped: 2026-03-22)

**Phases completed:** 4 phases, 16 plans, 16 tasks

**Key accomplishments:**

- One-liner:
- One-liner:
- One-liner:
- One-liner:
- One-liner
- One-liner:
- Automated golden test framework verifies all 7 models on all 5 backends against llama.cpp and HuggingFace references with prefix-matching tolerance
- Task 1: Token Bucket Rate Limiter
- One-liner:
- One-liner:
- One-liner:
- Three-tier state machine:
- One-liner:
- One-liner:
- HTTP server routes requests through continuous batching scheduler with RadixTree populated by actual physical block IDs from model's PagedKvCache
- All 6 models accept optional TieredKvCache with CLI-driven VRAM+RAM+SSD tier configuration threaded through to server scheduler

---
