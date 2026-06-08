# Review Batch 5: Tutorials 17, 18, 19

## Issues Found

### [ERROR] docs/tutorial/17-speculative-decoding.md: "Cooldown" subsection (lines ~479-489)

  Tutorial claims:
  > "When the acceptance rate over the last `adaptive_window` (8) drafted tokens drops below 25%, speculative decoding is bypassed for the next 8 steps."
  > "The cooldown counter decrements each step and re-enables speculation when it expires."

  Source says: No cooldown mechanism exists anywhere in `src/spec/spec_decode.zig` or any other file under `src/spec/`. There is no `adaptive_window` constant, no 25% threshold check, no cooldown counter, and no bypass logic. The only adaptive mechanism is `optimalK()` which selects the best draft length based on per-K acceptance statistics — it never disables speculation entirely.

  Fix: Remove the entire "### Cooldown" subsection and the `LowAccept` / `Cooldown` / `SingleDecode` nodes from the preceding Mermaid flowchart. The adaptive K section is correct on its own; the cooldown is fabricated.

---

### [ERROR] docs/tutorial/19-pflash-and-block-sparse.md: Complexity table (line ~107-115)

  Tutorial claims (in the table):
  > "| 8K tokens (128 blocks) | 64M dot products / layer |"
  > "| 32K tokens (512 blocks) | 1B dot products / layer |"
  > "| 128K tokens (2048 blocks) | 16B dot products / layer |"

  And derived claim:
  > "At 128K tokens, block sparse attention is roughly 200x cheaper per layer than full attention."

  Source says: The tutorial's own intro text correctly states "128K × 128K / 2 = 8 billion dot products" (causal masking halves the work). The table uses n² instead of n²/2, making every "Full attention" entry ~2× too high:
  - 8K: should be ~33M (not 64M)
  - 32K: should be ~537M (not 1B)
  - 128K: should be ~8.6B (not 16B)

  The "200x" ratio (16B / 80M) should be ~100x (8.6B / 80M) to be consistent with the intro text's correct formula.

  Fix: Halve all "Full attention" column values to reflect causal masking (n²/2). Change "200x" to "~100x".

---

### [WARNING] docs/tutorial/17-speculative-decoding.md: N-gram "ring buffer" (line ~177, 186)

  Tutorial claims:
  > "The ring buffer uses 8 KB."
  > "ngram.zig — N-gram: history ring buffer, n-gram matching, proposal"

  Source says: `NgramState` in `src/spec/ngram.zig` is **not** a ring buffer. It uses a linear array with shift-by-half compaction when full (copies last 1024 entries to the front and resets the write cursor). A ring buffer uses modular head/tail pointers with no data movement. The 8 KB size (2048 × 4 bytes) is correct.

  Fix: Change "ring buffer" to "history buffer" in both locations to accurately describe the data structure.

---

### [WARNING] docs/tutorial/18-multi-token-prediction.md: Transformer layer counts (line ~17)

  Tutorial claims:
  > "The hidden state passes through N **transformer layers** (e.g., 64 layers for a 0.8B model, or 32 layers for a 3B model)."

  Source says: Qwen3.5-0.8B (the only 0.8B model supported in `src/models/qwen35.zig`, default `n_layers: u32 = 32`) has 24 layers. No 0.8B model has 64 layers — that would imply extremely narrow hidden dimensions (~200) for 0.8B parameters. The example is misleading as a general illustrative statement.

  Fix: Change to realistic example values, e.g., "24 layers for a 0.8B model, or 36 layers for a 3B model" or use a more generic phrasing.

---

## Items Verified Correct

- DDTree `max_budget = 512`, default `budget = 64` — matches `src/spec/ddtree.zig`
- Ancestor mask `[8]u64` (512 bits) — matches `CompiledTree.ancestor_masks`
- N-gram range `n=3..10` — matches `min_ngram=3`, `max_ngram=10`
- PFlash default `alpha=0.85`, `block_size=64` — matches `PFlashConfig` defaults
- Block sparse default `window=1`, `n_global=2` — matches `BlockSparsePattern` defaults
- `max_kept_ratio` default 0.20 — matches code
- Adaptive K minimum rounds = 10 — matches `adaptive_k_min_rounds`
- `forwardTree` support currently only on Gemma3 — matches `src/models/gemma3.zig`
- `sdpaTree` kernel on all 6 backends (CPU, Metal, CUDA, Vulkan, ROCm, WebGPU) — verified
- GPU sdpaTree dispatches `n_nodes * nh` threadgroups (one per node-head pair) — matches Metal backend
- MTP concat order: embedding first, hidden second — matches `qwen35.zig:1356-1358`
- MTP +1 offset RMSNorm (`rmsNormPlusOne`) — matches code
- MTP GGUF tensor naming (`blk.{n_layers}.nextn.*`) — matches code
- MTP `shared_head_norm` and `shared_head_head` are MTP-specific, not shared with main output — matches
- SSM state checkpoint/restore exists for speculative decoding — confirmed via `saveSsmState`/`restoreSsmState`
- `optimalK()` expected value formula — consistent with code
- N-gram history capacity 2048, 8 KB memory — matches `history_capacity = 2048` × 4 bytes

## Coverage Status

- **Checked directly:** All three tutorial files cross-referenced against `ddtree.zig`, `spec_decode.zig`, `ngram.zig`, `pflash.zig`, `sparse_attn.zig`, `qwen35.zig`, and backend files for `sdpaTree`
- **Verified externally:** Qwen3.5-0.8B layer count (24 layers per official HuggingFace model card)
- **Not checked:** CLI flag names (`--spec-mode`, `--pflash-alpha`, etc.) were not verified against the argument parser
- **Not checked:** Performance numbers in examples (tok/s, latencies) — these are illustrative estimates
