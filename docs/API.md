# Agave HTTP API Reference

Start the server:
```bash
agave model.gguf --serve                    # default port 8080
agave model.gguf --serve --port 9090        # custom port
agave model.gguf --serve --api-key mysecret  # bearer token auth
```

---

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions.

```bash
curl http://localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100,
  "temperature": 0.7
}'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| messages | array | required | `[{"role": "user/system/assistant", "content": "..."}]` |
| max_tokens | int | 1024 | Maximum tokens to generate |
| temperature | float | 0 | 0 = greedy, >0 = sampling |
| top_k | int | 0 | Top-k filtering, 0 = disabled |
| top_p | float | 1.0 | Nucleus sampling threshold |
| min_p | float | 0 | Min-p sampling: keep tokens with prob >= min_p * max_prob [0, 1] |
| frequency_penalty | float | 0 | Penalize by token frequency in output [-2, 2] |
| presence_penalty | float | 0 | Penalize tokens that appeared at all [-2, 2] |
| repetition_penalty | float | 1.0 | Multiplicative penalty for repeated tokens (>1 = penalize) |
| seed | int | random | PRNG seed for reproducible output |
| stop | string/array | null | Stop sequence(s): `"stop": "\n"` or `"stop": ["\n", "END"]` |
| logprobs | bool | false | Return log probabilities for output tokens |
| top_logprobs | int | null | Number of top token log probabilities to return per position, 0-20 |
| stream | bool | false | Server-Sent Events streaming |
| grammar | string | null | GBNF grammar for constrained decoding |
| json_schema | string | null | JSON schema for structured output |
| response_format | object | null | `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {...}}}` |

**Response:**
```json
{
  "id": "chatcmpl-12345",
  "object": "chat.completion",
  "model": "model-name",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
}
```

### POST /v1/completions

Text completions (non-chat).

```bash
curl http://localhost:8080/v1/completions -d '{
  "prompt": "The capital of France is",
  "max_tokens": 20
}'
```

Same sampling parameters as chat completions. Prompt is raw text (no chat template).

### POST /v1/responses

OpenAI Responses API format.

```bash
curl http://localhost:8080/v1/responses -d '{
  "input": "Explain quantum computing",
  "max_tokens": 200
}'
```

### POST /v1/messages

Anthropic Messages API format.

```bash
curl http://localhost:8080/v1/messages -d '{
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
| max_tokens | int | 1024 | Maximum tokens to generate |
| temperature | float | 0 | 0 = greedy, >0 = sampling |
| top_k | int | 0 | Top-k filtering, 0 = disabled |
| top_p | float | 1.0 | Nucleus sampling threshold |
| stop_sequences | array | null | Stop sequence(s) |
| stream | bool | false | Server-Sent Events streaming |

**Response:**
```json
{
  "id": "msg-12345",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "model": "model-name",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 50}
}
```

### POST /v1/chat/regenerate

Regenerate the last assistant response in an active conversation.

```bash
curl -X POST http://localhost:8080/v1/chat/regenerate -d '{
  "conversation_id": "conv-12345"
}'
```

Rolls back the last assistant message and generates a new response using the same conversation context. Useful for "retry" functionality in chat UIs.

### POST /v1/conversations

Manage conversations.

**GET** — List active conversations:

```bash
curl http://localhost:8080/v1/conversations
# [{"id": "conv-12345", "created": 1715000000, "message_count": 4}]
```

**POST** — Create a new conversation:

```bash
curl -X POST http://localhost:8080/v1/conversations -d '{
  "system": "You are a helpful assistant."
}'
# {"id": "conv-67890", "created": 1715000001}
```

### POST /v1/tokenize

Count tokens for a text string.

```bash
curl http://localhost:8080/v1/tokenize -d '{"text": "Hello world"}'
# {"count": 2, "model": "model-name"}
```

### POST /v1/detokenize

Convert token IDs back to text.

```bash
curl http://localhost:8080/v1/detokenize -d '{"tokens": [9906, 1917]}'
# {"text": "Hello world"}
```

### GET /v1/models

List available models.

```bash
curl http://localhost:8080/v1/models
```

Returns model name, backend, context size, KV cache position.

### GET /health

Health check. Returns `{"status": "ok"}`.

### GET /ready

Readiness check. Returns 200 when model is loaded and ready.

### GET /metrics

Prometheus-format metrics: request count, latency, throughput, TTFT, token counts.

---

## Structured Output

Three ways to constrain output:

**1. JSON mode** — forces valid JSON object:
```bash
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Generate a user profile"}],
  "response_format": {"type": "json_object"}
}'
```

**2. JSON schema** — constrains to specific structure:
```bash
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "User info for Alice"}],
  "json_schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}}}"
}'
```

Or via OpenAI response_format:
```bash
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "User info"}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}}
}'
```

**3. GBNF grammar** — arbitrary format constraints:
```bash
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Is the sky blue?"}],
  "grammar": "root ::= \"yes\" | \"no\""
}'
```

---

## Streaming

Set `"stream": true` for Server-Sent Events:

```bash
curl -N http://localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "max_tokens": 100
}'
```

Events follow OpenAI format: `data: {"choices": [{"delta": {"content": "..."}}]}`.
Final event: `data: [DONE]`. Usage chunk sent before `[DONE]`.

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
| `413 Payload Too Large` | Request body exceeds server limit |
| `429 Too Many Requests` | Rate limit exceeded (when rate limiter is active) |
| `500 Internal Server Error` | Model forward error, OOM, or unexpected server failure |
| `503 Service Unavailable` | Model not loaded yet (server still initializing) |

---

## Authentication

```bash
agave model.gguf --serve --api-key mysecret
curl -H "Authorization: Bearer mysecret" http://localhost:8080/v1/chat/completions -d '...'
```

Returns 401 if key missing or wrong. No auth required when `--api-key` not set.
