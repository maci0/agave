# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; this file does not prescribe code fixes.

- **Last reviewed:** 2026-09-27 (working tree; every reference below re-verified against source this pass)
- **Owner / review cadence:** organizational fields, to be assigned; not defined in-repo
- **Scope:** inference CLI (`src/main.zig`), HTTP server (`src/server/`), model loaders (`src/format/`), Hub downloads (`src/pull.zig`), distributed transports (`src/parallel/`), WASM/browser demo (`web/`, `src/wasm_entry.zig`), container artifacts (`Dockerfile`, `docker-compose.yml`)
- **Out of scope:** backend kernel internals beyond their input parsing; GPU driver attack surface

Disclosure and supported-version policy: [SECURITY.md](../SECURITY.md). HTTP contract: [API.md](API.md).

## Risk-ranked summary

| # | Threat | Boundary | Impact | Status |
|---|--------|----------|--------|--------|
| T1 | Any LAN host can join an inference cluster or inject tensors: TP/PP/disagg/discovery have no authentication | Peer node -> node (TCP 49454/49455/49456, UDP 49460/49461) | Wrong outputs accepted as correct; full prompt transcript theft via disagg KV stream | **No mitigation.** No peer credential check anywhere in `src/parallel/` |
| T2 | Downloaded models are not content-integrity-checked: `resolve/main`, magic/size only | HF Hub -> loader | Poisoned weights steer outputs; malicious GGUF/SafeTensors exercises parser bugs | Partial: TLS + parser caps + GGUF magic |
| T3 | Single-key deployments have no tenant separation: `/v1/kv_cache` export and the global prompt-prefix / radix cache cross request owners | Client -> client (same server) | One API-key holder reads KV state derived from another user's prompts | Documented limitation, unmitigated |
| T4 | Rate limiting is one global bucket, default off; grammar/json_mode bypass the batch scheduler | Client -> compute | One client exhausts GPU time / latency for all | Partial: caps exist, identity does not |
| T5 | Predictable shared-memory names let any same-uid local process read/inject tensors | Local process -> shm | Local tensor injection during `--tp 2` same-host runs | Mode 0600 + `O_EXCL` after `shm_unlink` only |
| T6 | Conversation store persists prompts to disk (`~/.cache/agave/conversations.json`) | Process -> filesystem | Prompt transcript survives process exit; compose volume `agave-cache` holds it | Bounded file, durable replace, no encryption; no backup tier, so the volume is the only copy (`docs/DURABILITY.md`) |
| T7 | `/v1/tokenize` and `/v1/detokenize` run the tokenizer on attacker-chosen text with no per-endpoint quota | Client -> compute | Tokenizer CPU burn on loopback deployments; detokenize is a decode oracle over vocabulary | Bounded by 1 MiB body and 128-message scan cap; no endpoint-level rate limit |

Highest-value correction for operators: **the API key protects only TCP 49453**. The TP/PP/disagg data ports and UDP discovery are separate listeners that never see it.

## Assets

- Model weights on disk and in VRAM: expensive to obtain, exfiltration target.
- Prompt and conversation content: in memory; in the bounded conversation store (`src/server/server.zig` `max_conversations` 100 at `:146` / `max_messages_per_conv` 1000 at `:147`); on disk via `src/server/conv_store.zig` (default `$HOME/.cache/agave/conversations.json`, `defaultPath` `:70-74`); latent in the KV cache and radix prefix cache (`src/server/scheduler.zig` `radix_tree` `:295`, `matchPrefix` `:415`, `insert` `:666`).
- Hidden states: `/v1/kv_cache` export returns raw per-layer KV blocks, capped at 64 MiB (`kv_export_max_bytes` `src/server/server.zig:143`).
- `HF_TOKEN` (`src/pull.zig:336`) and `AGAVE_API_KEY` (`src/main.zig:1283-1307`): credentials in process env.
- GPU compute and availability: generation is the costly resource; DoS converts directly to cost.
- Output integrity: poisoned weights or a poisoned allReduce silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | endpoint table `src/server/server.zig:404-423` (`KnownEndpoint` `:403`), dispatcher `handleRequest` `:2001` | OpenAI/Anthropic-compatible endpoints; embedded web UI via `@embedFile` (`:1121-1124`, head + style + body + app.js); no filesystem serving |
| Generation endpoints | `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`, `/v1/chat` (table `:405-411`), `/v1/chat/regenerate` dispatch `:3257` | Streaming SSE and batch decode; auth-gated when a key is set |
| `/v1/embeddings` | dispatch `src/server/server.zig:2800` | Auth-gated when a key is set |
| Tokenizer endpoints | `/v1/tokenize` `:2506`, `/v1/detokenize` `:2584` | Auth-gated when a key is set; no separate quota (T7) |
| Health/readiness probes | `/health` `:2071`, `/ready` `:2098` | Unauthenticated by design; reduced bodies without auth |
| Metrics | `/metrics` `:2142` | Auth-required when a key is set; includes `agave_build_info` |
| KV cache export/import | `/v1/kv_cache` GET+POST `:2688`, `/v1/kv_cache/info` GET `:2644` | Raw hidden states in/out, cap 64 MiB (`kv_export_max_bytes` `:143`) |
| Conversations API | `/v1/conversations` `:3079` | Auth-gated; backed by in-memory store + optional disk persist |
| Static UI assets | `/` and `/favicon.ico` `:2175` | No directory listing, no path join from request data |
| CLI arguments | `src/cli.zig`, consumed in `src/main.zig` | Model path, prompts, `--lora`, `--mmproj`, `--image`/`--video`, steering files, draft models: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig` `max_stdin_prompt_size` `:141` (1 MiB) | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (mmap `:311`) | Also LoRA adapters, which delegate to the same mmap path (`src/lora.zig:73` `applyLoraGguf` -> `gguf.GGUFFile.open` `:78`) and draft/mmproj models |
| SafeTensors dirs | `src/format/safetensors.zig` | Multi-shard + `index.json`; shard names filtered (`isSafeShardName` `:2185`, `:2648`) |
| PNG images (CLI + HTTP base64) | `src/image.zig` (64 MiB file, 4096 dim, 50 MiB inflate `:19-26`); HTTP via `json.extractJsonImage` (`src/server/json.zig:857`) | JPEG rejected; zip-bomb caps present |
| Video frames | `src/main.zig:3639-3725` | `ffmpeg` spawned with argv (not a shell) at `:3660-3663`; local CLI only |
| Hub download channel | `src/pull.zig` (`hf_api_base` `:31`) | Repo ids and filenames validated (`isSafeFilename` `:64-76`, `isValidRepoName` `:79-95`); TLS via `std.http.Client` system CA bundle; blob URL is `resolve/main` (`:1049`) |
| Distributed TCP data plane | rank-0 listen `src/main.zig:1664-1675`, ports consts `:143-149` | Binds `addr = 0` (all interfaces, `:1665`); raw f32 frames (`src/parallel/transport.zig` `tcpSend` `:588` / `tcpRecv` `:628`) |
| UDP peer discovery | `src/parallel/peer_discovery.zig` (broadcast sockaddr `:94-97`, `AGAVE-JOIN:` prefix `:25`, first responder wins `:117-121`) | Global broadcast 255.255.255.255 |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`src/parallel/transport.zig` `setupShm` `:219-268`, names `:221-222`) | Same-host, mode 0600 |
| NCCL | dlopen `libnccl.so.2` (`src/parallel/transport.zig:308-310`) | 128-byte NCCL ID exchanged over the unauthenticated TCP link (`:347-365`) |
| Disagg prefill->decode KV transfer | listen `src/main.zig:3956-3967`, KV stream `src/models/qwen35.zig` `sendKvCache` `:2264-2282` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT` (`src/main.zig:1057-1061`), `AGAVE_HOST` (`:1272`), `AGAVE_API_KEY` (`:1284-1301`), `HF_TOKEN` (`src/pull.zig:336`), `NCCL_*` logging (`src/parallel/transport.zig:372-383`), `HOME`/`TMPDIR`/`XDG_CACHE_HOME` | Empty/whitespace env is unset (`pull.nonemptyEnv`). Debug-only: `AGAVE_VISION_DEBUG` (`src/models/vision.zig:326`), `AGAVE_DF2_DEBUG` (`src/main.zig:1379`), both inert unless set to 1 and both writing debug buffers to stdout. No runtime config files; recipes and chat templates are compile-time (`src/chat_template.zig` is string concatenation, not Jinja eval) |
| Browser WASM demo | `web/agave.ts` `loadModel` `:307-345`; `src/wasm_entry.zig` | Fetches a user-typed model URL into WASM memory; contained to the browser sandbox |
| Container | `Dockerfile:210` (`USER agave`), `Dockerfile:212` (`EXPOSE 49453`), entrypoint binds 0.0.0.0; `docker-compose.yml:35` (127.0.0.1 mapping), `:39` (`AGAVE_API_KEY` required), `:52` (`agave-cache` volume), `:61-62` (`cap_drop: ALL`), `:63` (`read_only: true`) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below |
| Process-level tool registry | `src/server/tools.zig` (`max_tools` 16, `Registry.register`) | Not attacker-reachable: slots are filled in-process by embedders, not from request JSON. Request-scoped tools are a separate, capped array (`max_tools` 8, `max_api_messages` 128, `src/server/json.zig:19,39`); they only reach a system prompt (`buildToolSystemPrompt` `src/server/server.zig:1518`) and a constrained decode allowlist |

No entry point in the previous revision has been removed. Added since the 2026-09-02 pass: `/v1/embeddings`, `/v1/responses`, `/v1/chat`, `/v1/chat/regenerate`, `/v1/tokenize`, `/v1/detokenize`, `/v1/kv_cache/info`, `/favicon.ico`.

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: `validateAuth` constant-time compare (`src/server/server.zig:1387-1416`, helper `constantTimeEql` `:1418`). Policy: non-loopback binds refuse to start without a key (`src/main.zig:1290-1296`); loopback binds are open by design. Unauthenticated mode also rejects non-loopback `Host` (DNS rebind, `isLoopbackHttpHost` `src/server/server.zig:992-1019`, call site `:2024-2032`) and mismatched `Origin` vs `Host` (CSRF, `originMatchesHost` `:956-966`, call site `:2034-2045`).
2. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj, PNG) crosses into mmap/decode native code. Validation lives inside the parsers (see mitigations).
3. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with `O_NOFOLLOW` writes (`src/pull.zig:1212-1232`). Commit SHA is used for snapshot directory naming (`:1476`, `:1576`), not as a pin on the download URL (`resolve/main` `:1049`).
4. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. Any host that connects is accepted into a rank slot subject only to the `max_peers` cap (`src/parallel/transport.zig` `acceptPeer` `:205-213`, called at `src/main.zig:1673`); first UDP `AGAVE-JOIN` responder wins discovery (`peer_discovery.zig:117-121`). Largest unauthenticated boundary.
5. **Same-host processes -> shm segments.** Only uid/file-mode checks; names are fixed.
6. **Secrets -> process.** Env vars enter once at startup; nonempty `AGAVE_API_KEY` wins over CLI to avoid `ps` exposure (`src/main.zig:1288`, process-list warning `:1137-1139`); empty env is unset. Rotation: process restart. Storage: env only; HTTP request buffers holding secrets are zeroed (`src/server/server.zig:6988-6992`); Hub `Authorization` buffers zeroed (`src/pull.zig:761,1111`).
7. **Process -> conversation file.** Prompts written to `$HOME/.cache/agave/conversations.json` unless `--no-conv-store` (`src/server/server.zig:7135-7145`, `src/server/conv_store.zig:70-74`). Compose maps this under `agave-cache`.
8. **Embedded UI -> jsDelivr.** `src/web/head.html:12-14` loads marked / DOMPurify from `cdn.jsdelivr.net` with SRI. CSP allowlists that origin (`src/server/server.zig:1372-1378`). Compromise of the CDN without a matching hash is blocked; a rebuild that changes both script and hash is a build-time event.

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:210`).

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement (`src/server/server.zig:1387-1416`, `src/main.zig:1284-1296`).
- Information disclosure: `/health`, `/ready` reachable unauthenticated (reduced bodies, `docs/API.md` health/ready sections match code at `src/server/server.zig:2071-2141`). Residual: build info on `/metrics` requires auth (`:2142-2147`).
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`http_buf_size` `:124` / `max_request_body_size` `:141`), duplicate `Content-Length` rejection (`parseContentLength` `:1298-1311`, reject `:1350`), `Transfer-Encoding` rejected to avoid request smuggling (`:1345-1348`), scan-based JSON with message/tool caps (`src/server/json.zig:19,39`), connection cap 64 (`max_concurrent_connections` `:149`), 30 s read/write timeouts (`connection_read_timeout_sec` `:426`, applied `:6957-6966`).
- CSRF / DNS rebind on no-key loopback: mitigated by Origin vs Host (`:956-966`, `:2034-2045`) and loopback-only Host (`:992-1019`, `:2024-2032`). Residual: any local process can still call the no-key loopback API (curl, scripts).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`src/server/rate_limiter.zig:55-59`). CLI default is 0 = limiter disabled (`src/main.zig:635,637`, parsed `:1412-1413`, wired `:3925-3926`). When only one of rpm/tpm is set, the other bucket uses 1M RPM / 100M TPM (`src/server/server.zig:156-159`) so the configured side is the constraint. Grammar and `json_mode` bypass the scheduler and serialize under the model mutex (`src/server/server.zig:3950-3955`).
- Elevation: none known; single-process, no privileged helpers, no request-scoped tool execution.

**Artifact -> loader (GGUF/SafeTensors/LoRA/PNG)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array caps (`src/format/gguf.zig:20-28`), saturating size math (`tensorBytes` `:153-159`), all tensor offsets validated against file size (`:855-871` in `parseHeader`, overflow saturation rejected at `:867`); SafeTensors header capped at 100 MB and checked against file size (`src/format/safetensors.zig:20,236-237`); shard-name traversal blocked (`:2648`); PNG dimension/inflate caps (`src/image.zig:19-26`).
- Residual: unknown GGUF type codes fall back conservatively (`src/format/gguf.zig:109` `else => 1`; non-string array skip `:757-772`). Fuzz coverage exists: GGUF `src/fuzz_tests.zig:1532`, SafeTensors header `src/format/safetensors.zig:4373+`, PNG `src/image.zig:728`. This is no longer an "unfuzzed parser" gap.

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between listing and blob GET; branch `main` is fetched, not the listed SHA (`src/pull.zig:1049`; SHA used for snapshot dir `:1476,1576`). Verification ends at GGUF magic bytes + Content-Length match (`verifyGgufBlob` `:1441-1455`, `advertisedSizeAgares` `:1000-1003`): T2.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce (`src/parallel/transport.zig` `allReduceAdd` `:416-465`, `tcpAllReduce` `:467`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`src/models/qwen35.zig:2264-2282`), DoS via connection race, elevation by becoming rank 0 through spoofed beacons (`src/parallel/peer_discovery.zig:117-121`). `tcpRecv` fails on short reads rather than zero-filling; that does not authenticate the peer: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`src/parallel/transport.zig:221-222`). `shm_unlink` then `O_EXCL` create discards a pre-planted send segment, then fails if a racer recreates it; it does not randomize the name: T5. Send-size and recv-size guards are ReleaseFast-stripped asserts (`:272`, `:290`).

**Process -> conversation file**
- Disclosure: the JSON store is plaintext prompts (`src/server/conv_store.zig`). Load refuses files > 64 MiB (`max_store_bytes` `:21`, refusal `:355`). Compose persists it in `agave-cache` (`docker-compose.yml:52-58`): T6. That volume is the only copy; backup and restore are in `docs/DURABILITY.md`.

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time | Client spoofing on 49453 | `src/server/server.zig:1387-1416`, `src/main.zig:1283-1307` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `src/main.zig:1271-1282,1290-1296` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `src/server/server.zig:956-966,2034-2045` |
| Loopback-only Host when no key | DNS rebinding (CWE-350) | `src/server/server.zig:992-1019,2024-2032` |
| Empty CORS (`corsHeaders` returns `""`) | Cross-site read of a local server | `src/server/server.zig:880-882` |
| Body/header/connection/timeout caps; reject duplicate `Content-Length` and any `Transfer-Encoding` | Request DoS, HTTP smuggling | `src/server/server.zig:124-149,426,1298-1311,1345-1350,6957-6966` |
| Token-bucket rate limits (opt-in, global) | Compute DoS when flags set | `src/server/rate_limiter.zig`; defaults off `src/main.zig:635,637` |
| Parser bounds/caps (GGUF, SafeTensors, PNG) + fuzz tests | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id / filename allowlists, `O_NOFOLLOW` blob writes, redirect-safe token handling | Download-path abuse | `src/pull.zig:64-95,761,1111,1212-1232` |
| Secret zeroization, env-over-CLI key | Credential leakage via ps/buffers | `src/main.zig:1137-1139,1288`, `src/server/server.zig:6988-6992`, `src/pull.zig:761,1111` |
| Container hardening | Container escape blast radius | `Dockerfile:210`, `docker-compose.yml:35,39,61-63` |
| Bounded grammar/schema recursion; fail-closed generate | Grammar DoS / unconstrained fallback | `src/grammar.zig:22-26`, `src/server/server.zig:4171` |
| SRI on jsDelivr scripts | UI CDN swap | `src/web/head.html:12-14` |
| Response security headers | Clickjacking, MIME sniff, cache | `src/server/server.zig:1372-1381`; claims match `docs/API.md` Response Headers |

Single points of failure: the API key alone carries all client-side authn on 49453; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports.

Docs-vs-code check (2026-09-27): `docs/API.md` auth / CORS / Host-rebind / rate-limit / security-header / health / ready claims match `src/server/server.zig`. No user-facing doc claims a mitigation the code lacks. The one stale comment was the `src/server/rate_limiter.zig:1` header, which said "per-API-key" while the struct doc at `:56-59` and the single-instance implementation say otherwise; that header is now corrected.

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests. With limits unset, nothing throttles GPU time. With limits set, the single global TPM/RPM bucket starves every other client (`src/server/rate_limiter.zig:56-59,125-145`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`src/server/server.zig:2688`; radix prefix cache is likewise global, `src/server/scheduler.zig:295,415,666`).
3. **Latency gaming:** repeated user-supplied GBNF grammars or `json_mode` force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`src/server/server.zig:3950-3955`).
4. **Tokenizer abuse:** `/v1/tokenize` and `/v1/detokenize` accept arbitrary attacker text and run the vocabulary scan under the same single global bucket as generation, so a cheap endpoint can occupy the tokenizer's share of request time (`src/server/server.zig:2506,2584`; caps `src/server/json.zig:19`). T7.
5. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds arbitrary f32 tensors (`src/parallel/transport.zig:205-213`, `src/parallel/peer_discovery.zig:117-121`).
6. **Prompt harvest from disk:** on a shared Unix user or a leaked compose volume, read `conversations.json` (`src/server/conv_store.zig:70-74`).
7. **Client-side trust note:** the `--serve` web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types (`web/agave.ts` `loadModel` `:307-345`), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols. Do not treat the HTTP API key as covering those ports.
2. T2: pin downloads to commit SHAs; verify checksums/signatures. Magic-byte + size is not integrity.
3. T3: document single-trust-domain status explicitly, or namespace caches/stores per key.
4. T4: per-key rate buckets; route grammar / `json_mode` through the scheduler. Limiter stays off unless flags are set, so operators who bind non-loopback with a key and no rpm/tpm have no compute quota.
5. T7: give `/v1/tokenize` and `/v1/detokenize` their own cheap quota, or document them as sharing the generation bucket.
6. T5: randomized shm names; keep a runtime (non-assert) send-size check in ReleaseFast.
7. T6: treat the conversation file as sensitive data (permissions already follow umask; no at-rest encryption).
8. Response path: [SECURITY.md](../SECURITY.md) records that no dedicated disclosure contact or fix-shipped SLA is defined in-repo (organizational; not invented here).
9. Observability: auth failures log `authentication failed` and increment a metric (`src/server/server.zig:1787-1788`, second call site `:2913-2914`). Logs are process stdout (compose `json-file` 10m×3). There is still no durable audit trail an incident investigation can replay independently of the container log driver.

## 7. Response readiness (note only)

- Security-relevant events that do exist in logs: request start/done with `req=` / `xid=` (`src/server/server.zig:1089,1104`), 401s (`:1788,2914`), Host/Origin rejects (`:2024-2045`).
- `X-Request-Id` copied into access logs is length-capped at 64 bytes (`max_client_request_id_len` `:130`, capture `:948`), so a hostile client cannot flood the log with an unbounded correlation string.
- No in-repo path from "vulnerability reported" to "fix shipped" beyond public GitHub issues. See [SECURITY.md](../SECURITY.md).
