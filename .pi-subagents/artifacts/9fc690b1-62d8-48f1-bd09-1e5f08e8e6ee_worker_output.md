All defaults verified against the actual code:
- `temperature`: `parseF32(res.option("temperature"), ...) orelse 0.0` → **0** ✓
- `top-k`: `parseU32(res.option("top-k"), ...) orelse 0` → **0** ✓
- `top-p`: `parseF32(res.option("top-p"), ...) orelse 1.0` → **1.0** ✓
- `min-p`: `parseF32(res.option("min-p"), ...) orelse 0.0` → **0** ✓
- `repeat-penalty`: `parseF32(res.option("repeat-penalty"), ...) orelse 1.0` → **1.0** ✓
- `dry-multiplier`: `parseF32(res.option("dry-multiplier"), ...) orelse 0` → **0** ✓
- `dry-length`: `parseU32(res.option("dry-length"), ...) orelse 2` → **2** ✓
- `xtc-probability`: `parseF32(res.option("xtc-probability"), ...) orelse 0` → **0** ✓
- `xtc-threshold`: `parseF32(res.option("xtc-threshold"), ...) orelse 0.1` → **0.1** ✓
- `mirostat-mode`: `parseU32(res.option("mirostat-mode"), ...) orelse 0` → **0** ✓
- `mirostat-tau`: `parseF32(res.option("mirostat-tau"), ...) orelse 5.0` → **5.0** ✓
- `mirostat-eta`: `parseF32(res.option("mirostat-eta"), ...) orelse 0.1` → **0.1** ✓

---

Implemented all 7 tutorial coverage gap changes.

**Changed files:**
1. `docs/tutorial/07-sampling.md` — Added CLI Quick Reference table with 17 sampling flags and defaults
2. `docs/tutorial/22-distributed-inference.md` — Added CLI Invocation section with flag table and 7 usage examples
3. `docs/tutorial/23-server-http-api.md` — Added CLI Quick Reference with 7 server flags and 5 usage examples
4. `docs/tutorial/02-the-transformer.md` — Added Llama 4 iRoPE + Chunked Attention to Attention Variants table with explanatory paragraph
5. `docs/tutorial/06-state-space-models.md` — Added paragraph about Nemotron-H Mamba-2 vs DeltaNet distinction with layer pattern details
6. `docs/tutorial/08-backends.md` — Added Browser/WASM note referencing `src/wasm_entry.zig`
7. `docs/tutorial/00-getting-started.md` — Added `agave pull` model download note with examples

**Validation:**
- Every flag name verified against `src/main.zig` CLI option definitions (lines 340-395)
- Every default verified against `parseF32`/`parseU32` `orelse` values (lines 676-1024)
- Llama 4 iRoPE pattern verified against `src/models/llama4.zig` (`isNopeLayer`, `default_nope_interval`)
- Nemotron-H layer counts verified against `src/models/nemotron_h.zig` (test at line 775: 21 SSM, 4 attn, 17 FFN-only)
- WASM entry point verified against `src/wasm_entry.zig` (exports, CPU-only, codegen bug note)
- Pull subcommand verified against `src/pull.zig` and `src/main.zig` line 545
- No staged files; all changes are unstaged working-tree modifications
- File sizes reasonable (165–772 lines each)

**Open risks:** The `--max-batch-size` default shows `8` in the code (`orelse 8`) but `1` in the help text display (line 1663). I used the runtime default of `8`, which is what the user actually gets. This is an existing inconsistency in main.zig, not introduced by this change.