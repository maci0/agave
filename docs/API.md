# Agave HTTP API Reference

Start the server:
```bash
agave model.gguf --serve                    # default port 49453
agave model.gguf --serve --port 9090        # custom port
agave model.gguf --serve --api-key mysecret  # bearer token auth
```

---

## Endpoints

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
| logprobs | bool | false | Return log probabilities for output tokens |
| top_logprobs | int | null | Number of top token log probabilities to return per position, 0-20 |
| n | int | 1 | Number of completions (only n=1 supported, n>1 returns 400) |
| user | string | null | User identifier for request tracking (logged server-side) |
| stream | bool | false | Server-Sent Events streaming |
| stream_options | object | null | `{"include_usage": true/false}` — gate usage chunk in streaming |
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
  "system_fingerprint": "agave-v0.1",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
}
```

`finish_reason` is `"stop"` (natural stop or stop sequence) or `"length"` (max_tokens reached).

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
  "choices": [{"text": "Paris.", "index": 0, "finish_reason": "stop"}],
  "usage": {"completion_tokens": 2, "prompt_tokens": 7, "total_tokens": 9}
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
# [{"id":1,"title":"Hello world","active":true,"count":4}]
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

### POST /v1/embeddings

Not implemented. Returns 501.

### POST /v1/tokenize

Count tokens for a text string. Accepts `text` or `content` as the field name.

```bash
curl http://localhost:49453/v1/tokenize -d '{"text": "Hello world"}'
# {"count": 2, "model": "model-name"}
```

### POST /v1/detokenize

Convert token IDs back to text.

```bash
curl http://localhost:49453/v1/detokenize -d '{"tokens": [9906, 1917]}'
# {"text": "Hello world"}
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

Health check (no auth required). Returns status, uptime, active connections, KV cache utilization, and request counters. Status is `"ok"`, `"degraded"` (KV pressure or high error rate), or `"shutting_down"`.

```json
{"status":"ok","reason":"none","version":"0.1.0","model":"model-name","backend":"metal",
 "uptime_s":120,"active_connections":1,"requests_total":5,"requests_completed":5,
 "requests_failed":0,"requests_cancelled":0,"queue_depth":0,
 "kv_cache_used":100,"kv_cache_total":8192,"kv_seq_len":42,"ctx_size":4096,
 "scheduler_errors":0,"preemptions":0}
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

Supported image formats: PNG, JPEG. Maximum resolution depends on the model (Gemma 4 E2B: 224×224, Gemma 4 26B: 768×768, Qwen VL: 448×448).

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

**OpenAI endpoints** (`/v1/chat/completions`, `/v1/completions`): `data: {"choices": [{"delta": {"content": "..."}}]}`.
Final event: `data: [DONE]`. Usage chunk sent before `[DONE]`.

**Anthropic endpoint** (`/v1/messages`): SSE events with `event: content_block_delta` / `event: message_delta` / `event: message_stop` following the Anthropic Messages API streaming format.

**Responses endpoint** (`/v1/responses`): SSE events with `event: response.output_text.delta` / `event: response.completed` following the OpenAI Responses API streaming format.

---

## Error Responses

All endpoints return JSON error bodies on failure:

```json
{"error": {"message": "Invalid request: missing 'messages' field", "type": "invalid_request_error"}}
```

| Status | When |
|--------|------|
| `400 Bad Request` | Malformed JSON, missing required fields, invalid parameter values |
| `401 Unauthorized` | Missing or invalid `Authorization: Bearer <key>` when `--api-key` is set |
| `404 Not Found` | Unknown endpoint |
| `405 Method Not Allowed` | Known endpoint with wrong HTTP method (includes `Allow` header) |
| `413 Payload Too Large` | Request body exceeds server limit |
| `429 Too Many Requests` | Rate limit exceeded (when rate limiter is active) |
| `500 Internal Server Error` | Model forward error, OOM, or unexpected server failure |
| `501 Not Implemented` | Endpoint exists but is not yet implemented (e.g., `/v1/embeddings`) |
| `503 Service Unavailable` | Model not loaded yet (server still initializing) |

---

## Authentication

```bash
agave model.gguf --serve --api-key mysecret
curl -H "Authorization: Bearer mysecret" http://localhost:49453/v1/chat/completions -d '...'
```

Also accepts Anthropic-style `X-API-Key` header:
```bash
curl -H "X-API-Key: mysecret" http://localhost:49453/v1/messages -d '...'
```

Returns 401 if key missing or wrong. No auth required when `--api-key` not set.
