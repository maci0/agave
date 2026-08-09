# Agave HTTP API Reference

**Tutorial:** [Server / HTTP API](tutorial/23-server-http-api.md)

Product version **0.1.0** (0.x SemVer: breaking HTTP/CLI changes may land without a
major bump; see [CHANGELOG](../CHANGELOG.md) and
[Versioning & Releases](CONTRIBUTING.md#versioning--releases)).
`system_fingerprint` and `/health` `version` report this string.

Start the server:
```bash
agave model.gguf --serve                    # default port 49453
agave model.gguf --serve --port 9090        # custom port
# Prefer AGAVE_API_KEY over --api-key (env wins if both set; avoids process-list exposure)
AGAVE_API_KEY=mysecret agave model.gguf --serve
agave model.gguf --serve --rate-limit-rpm 60 --rate-limit-tpm 100000  # token-bucket limits
agave model.gguf --serve --ctx-size auto      # auto-fit context to available memory
# Or: AGAVE_API_KEY=mysecret AGAVE_PORT=9090 agave model.gguf --serve
```

Cross-origin browser calls without an API key are rejected (`403`,
`cross_origin_forbidden`). `/v1/embeddings` returns `501` (not a stability promise).
See [Versioning & Releases](CONTRIBUTING.md#versioning--releases).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AGAVE_API_KEY` | API key for authentication (preferred over `--api-key` to avoid process-list exposure) |
| `AGAVE_HOST` | Bind address (default: `127.0.0.1`) |
| `AGAVE_PORT` | Listen port (default: `49453`) |
| `HF_TOKEN` | Hugging Face token for private model downloads (`agave pull`) |
| `HF_HOME` | Hugging Face cache directory (default: `~/.cache/huggingface`) |
| `XDG_CACHE_HOME` | Base cache directory when `HF_HOME` is not set |
| `AGAVE_VISION_DEBUG` | Enable vision encoder debug output |
| `NO_COLOR` | Disable colored terminal output (respects [no-color.org](https://no-color.org) convention) |

---

## Endpoints

### GET /

Serves the built-in web chat UI (single-page HTML). Requires authentication when `--api-key` is set. The UI communicates with the server via `POST /v1/chat` (streaming HTML responses).

### POST /v1/chat/completions

OpenAI-compatible chat completions.

```bash
curl http://localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100,
  "temperature": 0.7
}'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| messages | array | required | `[{"role": "user/system/assistant", "content": "..."}]` — content can be a string or an array of content parts (see [Vision](#vision)) |
| max_tokens | int | 512 | Maximum tokens to generate, capped at 4096 (also accepts `max_completion_tokens`) |
| temperature | float | 0 | 0 = greedy, >0 = sampling |
| top_k | int | 0 | Top-k filtering, 0 = disabled |
| top_p | float | 1.0 | Nucleus sampling threshold |
| min_p | float | 0 | Min-p sampling: keep tokens with prob >= min_p * max_prob [0, 1] |
| frequency_penalty | float | 0 | Penalize by token frequency in output [-2, 2] |
| presence_penalty | float | 0 | Penalize tokens that appeared at all [-2, 2] |
| repetition_penalty | float | 1.0 | Multiplicative penalty for repeated tokens (>1 = penalize) |
| seed | int | random | PRNG seed for reproducible output |
| stop | string/array | null | Stop sequence(s): `"stop": "\n"` or `"stop": ["\n", "END"]` |
| xtc_probability | float | 0 | XTC sampling: probability of excluding top tokens [0, 1] |
| xtc_threshold | float | 0.1 | XTC sampling: probability threshold for exclusion [0, 1] |
| dry_multiplier | float | 0 | DRY sampling: penalty multiplier for repeated n-grams (0=disabled) |
| dry_allowed_length | int | 2 | DRY sampling: minimum n-gram length to penalize |
| mirostat | int | 0 | Mirostat sampling mode: 0=disabled, 2=Mirostat 2.0 |
| mirostat_tau | float | 5.0 | Mirostat target entropy (surprise) |
| mirostat_eta | float | 0.1 | Mirostat learning rate |
| logit_bias | object | null | Token ID → bias mapping: `{"123": 5.0, "456": -2.0}` (max 16 entries) |
| logprobs | bool | false | Return log probabilities for output tokens (streaming only) |
| top_logprobs | int | null | Number of top token log probabilities to return per position, 0-20 (streaming only) |
| n | int | 1 | Number of completions (only n=1 supported, n>1 returns 400) |
| user | string | null | OpenAI compatibility only; accepted but ignored (not logged; often holds PII) |
| stream | bool | false | Server-Sent Events streaming |
| stream_options | object | null | `{"include_usage": true/false}` — gate usage chunk in streaming (usage included by default when omitted) |
| grammar | string | null | GBNF grammar for constrained decoding |
| json_schema | string | null | JSON schema for structured output |
| response_format | object | null | `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {...}}}` |
| tools | array | null | Tool/function definitions (see [Tool Calling](#tool-calling)) |
| tool_choice | string | "auto" | `"auto"`, `"none"`, or `"required"` |

**Response:**
```json
{
  "id": "chatcmpl-12345",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "model-name",
  "system_fingerprint": "agave-v0.1.0",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
}
```

`finish_reason` is `"stop"` (natural stop or stop sequence), `"length"` (max_tokens reached), or `"tool_calls"` (model invoked a tool — see [Tool Calling](#tool-calling)).

### POST /v1/completions

Text completions (non-chat).

```bash
curl http://localhost:49453/v1/completions -d '{
  "prompt": "The capital of France is",
  "max_tokens": 20
}'
```

Same sampling parameters as chat completions. Prompt is raw text (no chat template).

**Response:**
```json
{
  "id": "cmpl-12345",
  "object": "text_completion",
  "created": 1700000000,
  "model": "model-name",
  "system_fingerprint": "agave-v0.1.0",
  "choices": [{"text": "Paris.", "index": 0, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
}
```

### POST /v1/responses

OpenAI Responses API format.

```bash
curl http://localhost:49453/v1/responses -d '{
  "input": "Explain quantum computing",
  "max_tokens": 200
}'
```

Same sampling parameters as chat completions.

**Response:**
```json
{
  "id": "resp-12345",
  "object": "response",
  "created_at": 1700000000,
  "status": "completed",
  "model": "model-name",
  "stop_reason": "stop",
  "output": [{"type": "message", "id": "msg_0", "status": "completed", "role": "assistant",
    "content": [{"type": "output_text", "text": "..."}]}],
  "usage": {"input_tokens": 5, "output_tokens": 50, "total_tokens": 55}
}
```

`stop_reason` is `"stop"` (natural stop) or `"max_tokens"` (limit reached).

### POST /v1/messages

Anthropic Messages API format.

```bash
curl http://localhost:49453/v1/messages -d '{
  "system": "You are a helpful assistant.",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100
}'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| messages | array | required | `[{"role": "user/assistant", "content": "..."}]` |
| system | string | null | System prompt (separate from messages, per Anthropic format) |
| max_tokens | int | 512 | Maximum tokens to generate, capped at 4096 |
| stop_sequences | array | null | Stop sequence(s) |
| stream | bool | false | Server-Sent Events streaming |

All sampling parameters from `/v1/chat/completions` (temperature, top_k, top_p, min_p, penalties, seed, etc.) are also accepted.

**Response:**
```json
{
  "id": "msg_12345",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "model": "model-name",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 10, "output_tokens": 50}
}
```

`stop_reason` is `"end_turn"` (natural stop) or `"max_tokens"` (limit reached).

### POST /v1/chat

Built-in web UI chat endpoint (form-encoded). Used by the web interface at `/` when the server is running. Accepts `message`, `max_tokens`, `temperature`, `top_p`, `stream`, `system`, and `image` fields. Returns HTML fragments for the web UI.

### POST /v1/chat/regenerate

Regenerate the last assistant response in the active conversation. Rolls back the last assistant message, resets the KV cache, and generates a new response. Supports streaming via `stream=1`.

```bash
curl -X POST http://localhost:49453/v1/chat/regenerate -d 'stream=1&max_tokens=200'
```

Uses form-encoded body. Accepts `max_tokens`, `temperature`, `top_k`, `top_p`, `stream`, and `system` fields. Always operates on the currently active conversation.

### GET|POST /v1/conversations

Manage conversations.

**GET** — List active conversations:

```bash
curl http://localhost:49453/v1/conversations
# [{"id":1,"title":"Chat 1","active":true,"count":4}]
```

**POST** — Create, select, or delete conversations via form-encoded `action` field:

```bash
# Create a new conversation
curl -X POST http://localhost:49453/v1/conversations -d 'action=new'
# {"ok":true,"id":2}

# Select a conversation (returns its messages)
curl -X POST http://localhost:49453/v1/conversations -d 'action=select&id=1'
# {"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}

# Delete a conversation
curl -X POST http://localhost:49453/v1/conversations -d 'action=delete&id=1'
# {"ok":true,"cleared":false}
```

Limits: maximum 100 concurrent conversations, 1000 messages per conversation.
Conversations are process-local (in RAM only): not written to disk, wiped on
server shutdown, and message text is zeroed on delete/clear. Titles are opaque
(`Chat {id}`), never derived from user message content. The OpenAI `user`
request field is ignored (often an email or username).

### POST /v1/embeddings

Not implemented. Returns `501` (`code: not_implemented`). Experimental stub:
not part of the supported HTTP contract until a changelog entry ships a real
implementation.

### POST /v1/tokenize

Count tokens for a text string or messages array. Accepts `text`, `content`, or `messages` (applies the model's chat template before counting).

```bash
curl http://localhost:49453/v1/tokenize -d '{"text": "Hello world"}'
# {"count": 2, "model": "model-name"}

curl http://localhost:49453/v1/tokenize -d '{"messages": [{"role": "user", "content": "Hello"}]}'
# {"count": 8, "model": "model-name"}
```

### POST /v1/detokenize

Convert token IDs back to text.

```bash
curl http://localhost:49453/v1/detokenize -d '{"tokens": [9906, 1917]}'
# {"text": "Hello world", "model": "model-name"}
```

### GET /v1/models

List available models.

```bash
curl http://localhost:49453/v1/models
```

**Response:**
```json
{"object":"list","data":[{"id":"model-name","object":"model","created":1700000000,"owned_by":"agave",
  "backend":"metal","kv_seq_len":0,"ctx_size":4096,"n_layers":64,"n_embd":4096,
  "vocab_size":248320,"vision":false,"mtp_depth":0}]}
```

Additional fields beyond OpenAI spec: `backend` (compute backend), `kv_seq_len` (current KV cache position), `ctx_size` (max context), `n_layers`/`n_embd`/`vocab_size` (model dimensions), `vision` (multimodal support), `mtp_depth` (MTP prediction depth, 0=none).

### GET /health

Liveness probe (no auth required). Returns HTTP 200 for `"ok"` and `"degraded"` states, HTTP 503 only when `"shutting_down"`. Use `/ready` instead if your load balancer should stop routing traffic on degraded state.

Returns status, uptime, active connections, KV cache utilization, and request counters. Status is `"ok"`, `"degraded"` (KV pressure or high error rate), or `"shutting_down"`. When `--api-key` is configured and no valid auth header is provided, returns only `{"status":"...", "reason":"..."}` (no model/version/backend details) to prevent fingerprinting.

The `sleeping` field is `true` when the server has been idle longer than `--sleep-after`; it auto-clears on the next request.

```json
{"status":"ok","reason":"none","version":"0.1.0","model":"model-name","backend":"metal",
 "uptime_s":120,"active_connections":1,"requests_total":5,"requests_completed":5,
 "requests_failed":0,"requests_cancelled":0,"queue_depth":0,
 "kv_cache_used":100,"kv_cache_total":8192,"kv_seq_len":42,"ctx_size":4096,
 "scheduler_errors":0,"preemptions":0,"sleeping":false}
```

### GET /ready

Readiness probe (no auth required). Returns 200 with `"status":"ready"` when healthy. Returns 503 with `"status":"degraded"` (KV cache pressure or high error rate) or `"status":"shutting_down"` during shutdown.

```json
{"status":"ready","queue_depth":0,"kv_cache_used":100,"kv_cache_total":8192}
```

Degraded response (503):
```json
{"status":"degraded","reason":"kv_pressure","queue_depth":0,"kv_cache_used":7500,"kv_cache_total":8192}
```

Shutdown response (503):
```json
{"status":"shutting_down","queue_depth":2,"kv_cache_used":100,"kv_cache_total":8192}
```

### GET /metrics

Prometheus-format metrics: request count, latency, throughput, TTFT, token counts.

Requires authentication when `--api-key` or `AGAVE_API_KEY` is set (returns 401 otherwise). No auth when neither is configured.

---

## Structured Output

Three ways to constrain output:

**1. JSON mode** — forces valid JSON object:
```bash
curl localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Generate a user profile"}],
  "response_format": {"type": "json_object"}
}'
```

**2. JSON schema** — constrains to specific structure:
```bash
curl localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "User info for Alice"}],
  "json_schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}}}"
}'
```

Or via OpenAI response_format:
```bash
curl localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "User info"}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}}
}'
```

**3. GBNF grammar** — arbitrary format constraints:
```bash
curl localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Is the sky blue?"}],
  "grammar": "root ::= \"yes\" | \"no\""
}'
```

---

## Vision

Send images to multimodal models via base64 data URIs in the OpenAI content array format. Requires a model with vision support (Gemma 3/4, Qwen VL) loaded with `--mmproj` or a model that includes a built-in vision encoder.

```bash
curl http://localhost:49453/v1/chat/completions -d '{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
    ]
  }],
  "max_tokens": 200
}'
```

The `content` field can be either a string (text only) or an array of content parts. Text parts (`"type": "text"`) provide the prompt; image parts (`"type": "image_url"`) provide the image as a base64 data URI. Only one image per request is supported. The image is processed by the vision encoder (SigLIP-2) and injected as visual tokens at the appropriate position in the prompt.

Supported image formats over HTTP: PNG only (JPEG is rejected; convert to PNG first). The CLI `--image` path also accepts PPM P6. Maximum resolution depends on the model (Gemma 4 E2B/E4B: 224×224, Gemma 4 26B: 768×768, Gemma 3: 896×896, Qwen VL: model metadata / native).

---

## Tool Calling

OpenAI-compatible function/tool calling. Tools are injected into the system prompt; the model decides when to call them.

**Request with tools:**
```bash
curl http://localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
  }],
  "max_tokens": 200
}'
```

**Tool call response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_123_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Sending tool results back:**
```bash
curl http://localhost:49453/v1/chat/completions -d '{
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_123_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_123_0", "content": "{\"temp\": 18, \"condition\": \"cloudy\"}"}
  ],
  "max_tokens": 200
}'
```

**tool_choice values:**

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Model decides whether to call tools |
| `"none"` | Tools stripped from prompt, no tool calls |
| `"required"` | Model instructed to call at least one tool |

Streaming with tools is supported — tool calls are emitted as delta chunks with `tool_calls` array, followed by `finish_reason: "tool_calls"`.

---

## Prompt Prefix Caching

Consecutive API requests that share a common token prefix automatically reuse the KV cache from the previous request. When a chat application sends the full conversation history with each request (the standard OpenAI API pattern), only the new messages are prefilled — the system prompt and earlier messages remain in cache.

This is automatic and transparent — no API changes needed. The server logs prefix cache hits:
```
Prefix cache hit: 1847/2103 tokens reused
```

Cache is invalidated when the prompt prefix changes (e.g., switching conversations or modifying the system prompt). Works with both streaming and non-streaming requests.

### Cross-Instance KV Cache Sharing

For deployments with multiple agave instances serving the same model, KV cache prefixes can be transferred between instances:

**Wire format** (unversioned; not the disk `checkpoint.KVC` header):
`layer₀_K | layer₀_V | layer₁_K | layer₁_V | …` as little-endian f32.
Per-layer K/V length is `n_tokens × kvd_layer × 4` bytes (`kvd` may differ across layers on dual-attention / MLA models).
Only architectures that implement `exportKvPrefix` / `importKvPrefix` support this (currently Gemma 4); others return `501`.
The blob does **not** include prompt token IDs, so a following OpenAI-style request with `reset` still re-prefills unless the server already holds matching prefix-cache IDs from a prior local generation.

**Export** — serialize `N` tokens of KV cache as a binary blob (`n_tokens` is a required query parameter):
```bash
GET /v1/kv_cache?n_tokens=512
→ 200 OK  Content-Type: application/octet-stream
   <binary KV data>
```

**Import** — restore KV cache from a blob (sets `kv_seq_len = N`, clears prefix-cache token IDs, sets `kv_valid`):
```bash
POST /v1/kv_cache?n_tokens=512
Content-Type: application/octet-stream
<binary KV data>
→ 200 OK  {"imported":512}
```

Missing or non-positive `n_tokens` returns `400` with `invalid_request_error`
(same `type` string as other OpenAI-style 400s; was briefly `invalid_request` on
this route only).

`/v1/kv_cache` and `/v1/kv_cache/info` require authentication if `--api-key` or
`AGAVE_API_KEY` is configured. Use case: compute system-prompt KV on one instance,
distribute to a fleet for warm-start generation without redundant prefill.
Warm-start that skips prefill on the API path needs matching prompt token IDs in
the blob (not yet in the wire format); today import is coherent for chat
continuation / `kv_valid` and for orchestrators that manage prefill themselves.

**Metadata** — lightweight KV state query for external orchestrators (`GET` only; not shadowed by `/v1/kv_cache`):
```bash
GET /v1/kv_cache/info
→ 200 OK
{
  "seq_len": 42,
  "cached_prefix_len": 42,
  "prefix_hash": "a3f1b29c7d8e4f50",
  "kv_used": 100,
  "kv_total": 8192
}
```

`prefix_hash` is the FNV-1a hash of the cached prefix token IDs — use it for fast remote matching to route requests to the most cache-warm instance without full KV export.

---

## Thinking Budget

For reasoning models (DeepSeek R1, QwQ, Gemma 4 thinking variants), limit the number of thinking tokens:

```bash
# Anthropic API format
curl http://localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Solve this step by step: ..."}],
  "thinking": {"type": "enabled", "budget_tokens": 2000},
  "max_tokens": 4000
}'

# Flat field format (also accepted)
{
  "thinking_budget_tokens": 2000,
  "max_tokens": 4000
}
```

When the model generates more than `budget_tokens` tokens inside `<think>...</think>`, it is nudged out of the thinking phase by applying a strong positive logit bias to the `</think>` token. This caps reasoning time without hard-stopping mid-thought.

---

## Streaming

Set `"stream": true` for Server-Sent Events:

```bash
curl -N http://localhost:49453/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "max_tokens": 100
}'
```

**Chat completions** (`/v1/chat/completions`): `data: {"choices": [{"delta": {"content": "..."}}]}`.
**Text completions** (`/v1/completions`): `data: {"choices": [{"text": "...", "index": 0, "finish_reason": null}]}`.
Final event: `data: [DONE]`. Usage chunk sent before `[DONE]`.

**Anthropic endpoint** (`/v1/messages`): SSE events: `message_start` → `content_block_start` → `content_block_delta`* → `content_block_stop` → `message_delta` → `message_stop`.

**Responses endpoint** (`/v1/responses`): SSE events: `response.created` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta`* → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.completed`.

---

## Error Responses

All endpoints return JSON error bodies on failure.

**OpenAI format** (all endpoints except `/v1/messages`):
```json
{"error": {"message": "Missing or empty messages array", "type": "invalid_request_error", "param": "messages", "code": "missing_required_parameter"}}
```

`param` names the offending field or query key when known; otherwise `null`.
`code` is a stable machine-readable string when known (for example `missing_required_parameter`, `n_not_supported`, `invalid_api_key`, `method_not_allowed`, `unknown_endpoint`, `conversation_not_found`, `request_too_large`, `malformed_request`, `rate_limit_exceeded`, `not_implemented`, `cross_origin_forbidden`, `message_too_long`, `image_decode_failed`, `kv_import_failed`, `unknown_conversation_action`, `no_active_conversation`, `no_user_message`, `conversation_limit_reached`, `conversation_message_limit`, `server_overloaded`); otherwise `null`.

**Anthropic format** (`/v1/messages` only):
```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "Missing or empty messages array"}}
```

| Status | When |
|--------|------|
| `400 Bad Request` | Malformed JSON, missing required fields, invalid parameter values |
| `401 Unauthorized` | Missing or invalid `Authorization: Bearer <key>` or `X-API-Key` when `--api-key` is set |
| `403 Forbidden` | Cross-origin browser request rejected when no `--api-key` is configured (CSRF protection) |
| `404 Not Found` | Unknown endpoint or conversation not found |
| `405 Method Not Allowed` | Known endpoint with wrong HTTP method (includes `Allow` header) |
| `413 Payload Too Large` | Request body exceeds 1 MB server limit |
| `429 Too Many Requests` | Token-bucket rate limiting via `--rate-limit-rpm` / `--rate-limit-tpm` (includes `Retry-After`) |
| `500 Internal Server Error` | Model forward error or unexpected server failure |
| `501 Not Implemented` | Endpoint exists but is not yet implemented (e.g., `/v1/embeddings`) |
| `503 Service Unavailable` | Conversation limit reached, shutting down, connection capacity / spawn failure / request-buffer OOM (capacity responses include `Retry-After`), or degraded (`/ready` only — inference endpoints do not return 503 for degraded state) |

---

## Authentication

```bash
# Prefer AGAVE_API_KEY over --api-key (CLI args appear in process listings)
AGAVE_API_KEY=mysecret agave model.gguf --serve
curl -H "Authorization: Bearer mysecret" http://localhost:49453/v1/chat/completions -d '...'
```

Also accepts Anthropic-style `X-API-Key` header:
```bash
curl -H "X-API-Key: mysecret" http://localhost:49453/v1/messages -d '...'
```

Returns 401 if key missing or wrong. No auth required when neither `--api-key`
nor `AGAVE_API_KEY` is set. If both are set, `AGAVE_API_KEY` is used and the
CLI value is ignored.

---

## Response Headers

All responses include these headers:

| Header | Description |
|--------|-------------|
| `X-Request-Id` | Monotonic request counter for log correlation (matches server-side `req=N` logs) |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `no-referrer` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `Permissions-Policy` | Disables geolocation, microphone, camera, accelerometer, gyroscope |
| `Content-Security-Policy` | Restrictive CSP: `default-src 'none'`, allows inline scripts/styles and CDN resources for the web UI |
| `Cache-Control` | `no-store` |
| `Connection` | `close` (non-streaming) or `keep-alive` (SSE streaming) |

Rate-limited responses (429) and connection-capacity responses (503 with
`code` `server_overloaded`) include `Retry-After` with seconds until the next
request is allowed.

When no `--api-key` is configured, cross-origin browser requests (mismatched `Origin` vs `Host`) are rejected with 403 to prevent CSRF against a local `--serve`. Same-origin use of the embedded UI is unchanged. CORS `Access-Control-Allow-Origin` is not emitted; use a reverse proxy if a separate web origin must call the API.
