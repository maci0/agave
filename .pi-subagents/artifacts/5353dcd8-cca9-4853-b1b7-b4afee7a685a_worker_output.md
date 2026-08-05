Fix 8 confirmed: pull block is right after the "inspecting the path" paragraph and before section 2.

Implemented all 9 fixes to tutorial files.

Changed files:
- `docs/tutorial/00-getting-started.md` — Fix 8: moved `agave pull` block to after format detection paragraph
- `docs/tutorial/02-the-transformer.md` — Fix 7: added Llama 4 MoE mention after iRoPE paragraph; Fix 9: added 3 glossary entries (chunked attention, iRoPE, NoPE) in alphabetical order
- `docs/tutorial/08-backends.md` — Fix 1: added `agave_alloc` and `agave_dealloc` to WASM export list; Fix 2: added `(wasm_entry.zig)` to bold title, removed inline link
- `docs/tutorial/22-distributed-inference.md` — Fix 3: removed duplicate Note blockquote; Fix 4: added Short column to CLI table; Fix 5: changed `--peers HOST` to `--peers ADDR` with port example; Fix 6: removed `### Examples` heading

Validation: All 9 fixes verified against source code. `agave_alloc` and `agave_dealloc` confirmed in `src/wasm_entry.zig`. MoE routing confirmed in `src/models/llama4.zig`. `--peers` format confirmed against `src/main.zig` `parsePeerAddr`.

Open risks: None. All edits are documentation-only.

Recommended next step: Review the changes, then commit.