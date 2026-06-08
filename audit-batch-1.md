# Documentation Audit — Batch 1

Audited files: `docs/API.md`, `docs/ARCHITECTURE.md`
Cross-referenced against source: `src/server/server.zig`, `src/server/json.zig`, `src/cli.zig`, `src/main.zig`, `src/backend/backend.zig`, `src/models/model.zig`, `src/models/vision.zig`, `src/format/gguf.zig`, `src/kvcache/manager.zig`, `src/spec/spec_decode.zig`, `src/ops/attention.zig`, `src/chat_template.zig`, `src/recipe.zig`, `src/image.zig`

---

## Issues Found

### Issue 1

[ERROR] docs/ARCHITECTURE.md: Chat Templates table, GLM-4 row
  Doc claims: "`[gMASK]<sop>` prefix, `</think>` generation prefix"
  Source says: `src/chat_template.zig:224` — `.generation_prefix = ""`  (empty string, not `</think>`)
  Fix: Change to "`[gMASK]<sop>` prefix" — remove the `, \`</think>\` generation prefix` claim, or state "empty generation prefix".

---

### Issue 2

[WARNING] docs/API.md: Vision section, supported image formats
  Doc claims: "Supported image formats: PNG only (JPEG is not yet supported — convert to PNG first)."
  Source says: `src/image.zig:1-9` and `src/server/server.zig:2153-2195` — The server's `processVisionImage` function handles `.png` and explicitly rejects `.jpeg`, but PPM P6 is supported at the image library level (`image.zig:53`, `image.zig:283`). However, the server's switch statement at `server.zig:2188-2195` has a catch-all `else` branch that rejects non-PNG, non-JPEG formats. PPM is effectively not supported via the API (only via CLI `--image` flag). The doc is functionally accurate for API usage, but the `extractJsonImage` test in `json.zig:1149-1155` shows JPEG data URIs are extracted successfully before being rejected at the decode step — JPEG images are silently dropped, not rejected with an error to the client (returns `false` → no error response, just no image).
  Fix: No text change needed for the API doc. This is a minor behavioral note: JPEG data is parsed but silently ignored rather than returning an error to the API client.

---

### Issue 3

[WARNING] docs/ARCHITECTURE.md: Chat Templates table, Gemma 4 EOG tokens
  Doc claims: "`<turn|>`, `<eos>`, `<channel|>`, `<|endoftext|>`, `<|end|>`"
  Source says: `src/chat_template.zig:204` — `.eog_tokens = &.{ "<turn|>", "<eos>", "<channel|>", "<|endoftext|>", "<|end|>" }`
  Fix: No fix needed — the doc matches the source. (Verified correct.)

---

### Issue 4

[WARNING] docs/ARCHITECTURE.md: Gemma Q4 Metal recipe row
  Doc claims: "temp=0.7, top_p=0.95" (only two key defaults listed)
  Source says: `src/recipe.zig:130-137` — The recipe also sets `repeat_penalty = 1.05` and `max_tokens = 1024`.
  Fix: Consider adding `repeat=1.05` to the "Key Defaults" column for completeness. Not strictly an error since the column header says "Key Defaults" (not "All Defaults"), but the Qwen3.5 and GLM-4 rows include `repeat=1.1`, creating an inconsistency in what's considered "key."

---

### Issue 5

[WARNING] docs/ARCHITECTURE.md: GPT-OSS Metal recipe row
  Doc claims: "temp=0.5, ctx=2048"
  Source says: `src/recipe.zig:142-149` — The recipe also sets `top_p = 0.9` and `max_tokens = 512`.
  Fix: Consider adding `top_p=0.9, max_tokens=512` for consistency with other rows.

---

## Verified Correct (selected spot checks)

The following claims were cross-referenced and confirmed correct:

| Claim | Source location | Status |
|-------|----------------|--------|
| Default port 49453 | `src/main.zig:108` (`default_port: u16 = 49453`) | ✅ |
| max_tokens default 512 | `src/server/server.zig:77` (`default_max_gen_tokens: usize = 512`) | ✅ |
| max_tokens capped at 4096 | `src/server/server.zig:76` (`gen_ids_buf_size: usize = 4096`) | ✅ |
| Request body limit 1 MB | `src/server/server.zig:87,99` (`http_buf_size = 1024 * 1024`, `max_request_body_size = http_buf_size`) | ✅ |
| Max 100 conversations, 1000 msgs | `src/server/server.zig:100-101` | ✅ |
| temperature default 0 | `src/server/json.zig:62` (`temperature: f32 = 0`) | ✅ |
| top_p default 1.0 | `src/server/json.zig:64` (`top_p: f32 = 1.0`) | ✅ |
| min_p default 0 | `src/server/json.zig:65` (`min_p: f32 = 0`) | ✅ |
| repetition_penalty default 1.0 | `src/server/json.zig:68` (`repetition_penalty: f32 = 1.0`) | ✅ |
| xtc_threshold default 0.1 | `src/server/json.zig:70` (`xtc_threshold: f32 = 0.1`) | ✅ |
| dry_allowed_length default 2 | `src/server/json.zig:72` (`dry_allowed_length: u32 = 2`) | ✅ |
| mirostat_tau default 5.0 | `src/server/json.zig:74` (`mirostat_tau: f32 = 5.0`) | ✅ |
| mirostat_eta default 0.1 | `src/server/json.zig:75` (`mirostat_eta: f32 = 0.1`) | ✅ |
| logit_bias max 16 entries | `src/server/json.zig:33` (`max_logit_bias: usize = 16`) | ✅ |
| top_logprobs 0-20 | `src/server/json.zig:335` (clamped to 20) | ✅ |
| stream_include_usage default true | `src/server/json.zig:345-349` (defaults to `true` when omitted) | ✅ |
| GGUF v2/v3 parser | `src/format/gguf.zig:587` (`version < 2 or version > 3` → error) | ✅ |
| Paged KV 16-token blocks | `src/kvcache/manager.zig:14` (`default_block_size: u16 = 16`) | ✅ |
| Backend: tagged union dispatch | `src/backend/backend.zig:492` (`pub const Backend = union(enum)`) | ✅ |
| Backend variants: cpu, metal, vulkan, cuda, rocm, webgpu | `src/backend/backend.zig:493-498` | ✅ |
| Model: comptime vtable (not VTable pointer indirection for backend) | `src/models/model.zig:63-80` (VTable struct with fn pointers, from() generates vtable) | ✅ |
| system_fingerprint: "agave-v" + version | `src/server/server.zig:78` | ✅ |
| ChatML EOG: `<\|im_end\|>`, `<\|endoftext\|>` | `src/chat_template.zig:152-161` | ✅ |
| Gemma EOG: `<end_of_turn>`, `<eos>` | `src/chat_template.zig:184-192` | ✅ |
| Qwen3.5 generation_prefix: `<think>\n\n</think>\n\n` | `src/chat_template.zig:178` | ✅ |
| Gemma4 generation_prefix: `<\|channel>0\n<channel\|>` | `src/chat_template.zig:205` | ✅ |
| Llama 4 EOG: `<\|eot\|>`, `<\|end_of_text\|>` | `src/chat_template.zig:253` | ✅ |
| Llama 4 default system prompt: "You are a helpful assistant." | `src/chat_template.zig:254` | ✅ |
| GPT-OSS EOG: `<\|end\|>`, `<\|endoftext\|>` | `src/chat_template.zig:242` | ✅ |
| Gemma 4 SigLIP-2: 768×768 / 16×16 / 2304 / 3×3 / 256 | `src/models/vision.zig:73,75,80` | ✅ |
| Gemma 3 SigLIP: 896×896 / 14×14 / 4096 | `src/models/vision.zig:19-21` | ✅ |
| Gemma 4 E2B: 224×224 | `src/models/vision.zig:14-15` (`default_image_size: u32 = 224`) | ✅ |
| spec_decode.zig: draft, verify, generation loop | `src/spec/spec_decode.zig:1-16` | ✅ |
| pflash.zig exists under src/spec/ | `src/spec/pflash.zig` | ✅ |
| sdpa_tree.zig exists under backend/kernels/ | `src/backend/kernels/cpu/sdpa_tree.zig`, `cuda/`, `rocm/` | ✅ |
| sparse_attn.zig exists under src/ops/ | `src/ops/sparse_attn.zig` | ✅ |
| All endpoint paths in routing table | `src/server/server.zig:164-177` (known_endpoints) matches docs | ✅ |
| CORS disabled with --api-key | `src/server/server.zig:620` (`if (g_server.api_key != null) "" else cors_allow_headers`) | ✅ |
| /health returns 200 for ok/degraded, 503 for shutting_down | `src/server/server.zig:856-890` | ✅ |
| /ready returns 503 for degraded | `src/server/server.zig:901-940` | ✅ |
| X-API-Key header accepted | `src/server/server.zig:602-608` (checks `x-api-key` header) | ✅ |
| expertWeightStride in model.zig | `src/models/model.zig:413` | ✅ |
| frequency_penalty range [-2, 2] | `src/server/json.zig:325` (clamped to -2.0..2.0) | ✅ |
| presence_penalty range [-2, 2] | `src/server/json.zig:326` (clamped to -2.0..2.0) | ✅ |

---

## Summary

- **1 ERROR found**: GLM-4 generation_prefix documented as `</think>` but source code shows empty string `""`.
- **4 WARNINGS found**: Recipe table omissions (2 entries incomplete), plus minor notes about image format support precision.
- **All other checked claims verified correct** across both docs against >15 source files.
