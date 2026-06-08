# Review Batch 4: Tutorials 13–16

Reviewed tutorials 13-16 against source code in `src/backend/mega_compose.zig`, `src/backend/megakernel.zig`, `src/format/gguf.zig`, `src/chat_template.zig`, `src/recipe.zig`, `src/arch.zig`, `src/backend/backend.zig`, `src/format/format.zig`, and `src/main.zig`.

## Issues Found

### Chapter 13: Batched Dispatch and Fusion

**No issues found.** All claims about:
- GemvOp struct fields ✓ (matches `src/backend/backend.zig:37`)
- 11 Metal MSL kernels in megakernel.metal ✓ (verified: 6 SiLU + 5 GELU)
- 4 CUDA kernels ✓ (verified: `fused_ffn_{q8_0,q4_k,q5_k,q6_k}.zig`)
- mega_common.metal 732 lines, 18 primitives ✓ (verified both)
- ~4,334 lines across 12 files (hand-written) ✓ (3539 mega_* + 795 megakernel.* = 4334)
- ~1,036 lines in mega_compose.zig ✓
- `addRmsNorm` signature ✓
- `siluMul` signature ✓
- Megakernel Tier 2 implementation table ✓
- `memory_order_relaxed` in grid sync ✓
- `composeMSL` API ✓

---

### Chapter 14: Format Conventions

**No issues found.** All claims verified against source:
- Format struct with `is_safetensors` field ✓ (matches `src/format/format.zig`)
- VTable function pointers ✓ (all 7 match)
- GGUF dimension reversal ✓ (line 632: `dims[d] = raw_dims[n_dims - 1 - d]`)
- Q4_K: 256 elements, 144 bytes/block ✓ (`src/format/gguf.zig`)
- Q8_0: 32 elements, 34 bytes/block ✓
- Q4_0: 32 elements, 18 bytes/block ✓

---

### Chapter 15: Chat Templates

[ERROR] 15-chat-templates.md: "Template Selection" section
  Tutorial claims: `"const arch = Arch.detect(fmt) orelse return error.UnknownArch;"`
  Source says: `Arch.detect` takes a `[]const u8` string, not a `Format`. In `src/main.zig:1731`: `var arch = Arch.detect(arch_str)` where `arch_str = fmt.getMetaStr("general.architecture")`. Also, `detect` returns `?Arch` (no error), so `try` would not compile.
  Fix: Replace with:
  ```zig
  const arch_str = fmt.getMetaStr("general.architecture") orelse return error.UnknownArch;
  const arch = Arch.detect(arch_str) orelse return error.UnknownArch;
  const template = arch.chatTemplate();
  ```

All other claims verified correct:
- All 7 template definitions match `src/chat_template.zig` exactly ✓
- `chatTemplate()` switch statement matches `src/arch.zig` exactly ✓
- Image token IDs (Gemma 4: 258880, Gemma 3: 219, Qwen: 248053/248054/248056) ✓
- `findImageInsertPos` and `injectImageTokens` logic ✓
- `formatConversation` implementation ✓
- Message struct with `tool_call_id` field ✓
- EOG token resolution flow ✓

---

### Chapter 16: Recipe System

[ERROR] 16-recipe-system.md: "Usage Flow / In main.zig" section
  Tutorial claims: `"const recipe = Recipe.match(arch.displayName(), backend_name, quant) orelse Recipe.default;"`
  Source says: In `src/main.zig:1811`: `Recipe.match(arch_str, be_name, quant)` where `arch_str` is the raw GGUF architecture string (e.g. `"qwen35"`), NOT `arch.displayName()` (which returns `"Qwen 3.5"`). Using `displayName()` would break matching since preset `arch_prefix = "qwen3"` checks `startsWith` and `"Qwen 3.5"` starts with uppercase `Q`.
  Fix: Replace `arch.displayName()` with `arch_str` (the raw architecture string from format metadata):
  ```zig
  const arch_str = fmt.getMetaStr("general.architecture") orelse "unknown";
  const recipe = Recipe.match(arch_str, backend_name, quant) orelse Recipe.default;
  ```

All other claims verified correct:
- Recipe struct fields ✓
- All 5 presets match `src/recipe.zig` exactly ✓
- `applyDefaults` implementation ✓
- `Overrides` struct ✓
- `match` function and `Preset.matches` logic ✓
- Priority semantics (first match wins, user CLI > recipe > CLI default) ✓
- All test cases shown in tutorial match actual tests in source ✓

---

## Summary

| File | Errors | Warnings |
|------|--------|----------|
| 13-batched-dispatch-and-fusion.md | 0 | 0 |
| 14-format-conventions.md | 0 | 0 |
| 15-chat-templates.md | 1 | 0 |
| 16-recipe-system.md | 1 | 0 |
| **Total** | **2** | **0** |

Both errors involve incorrect API usage in pseudocode examples: `Arch.detect(fmt)` should take a string not a Format, and `Recipe.match` receives the raw arch string not `displayName()`.
