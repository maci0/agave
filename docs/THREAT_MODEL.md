# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; this file does not prescribe code fixes.

- **Last reviewed:** 2026-09-27 (every line reference below re-anchored against source in this pass)
- **Owner / review cadence:** organizational fields, to be assigned; not defined in-repo
- **Scope:** inference CLI (`src/main.zig`), HTTP server (`src/server/`), model loaders (`src/format/`), Hub downloads (`src/pull.zig`), distributed transports (`src/parallel/`), WASM/browser demo (`web/`, `src/wasm_entry.zig`), container artifacts (`Dockerfile`, `docker-compose.yml`)
- **Out of scope:** backend kernel internals beyond their input parsing; GPU driver attack surface

Disclosure and supported-version policy: [SECURITY.md](../SECURITY.md). HTTP contract: [API.md](API.md).

Every reference below is `path:line` plus the symbol or literal at that line, so a later drift is cheap to detect without a full re-read.

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
- Prompt and conversation content: in memory; in the bounded conversation store (`src/server/server.zig` `max_conversations` = 100 at `:147`, `max_messages_per_conv` = 1000 at `:148`); on disk via `src/server/conv_store.zig` (`defaultPath` `:70`, `$XDG_CACHE_HOME/agave` or `$HOME/.cache/agave`); latent in the KV cache and radix prefix cache (`src/server/scheduler.zig` `radix_tree: RadixTree` field `:309`, `matchPrefix` call `:429`, `insert` call `:698`).
- Hidden states: `/v1/kv_cache` export returns raw per-layer KV blocks, capped at 64 MiB (`kv_export_max_bytes` `src/server/server.zig:144`).
- `HF_TOKEN` (`src/pull.zig:339` `getenv("HF_TOKEN")`) and `AGAVE_API_KEY` (`src/main.zig:1326` `preferredSecret`): credentials in process env.
- GPU compute and availability: generation is the costly resource; DoS converts directly to cost.
- Output integrity: poisoned weights or a poisoned allReduce silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | endpoint table `src/server/server.zig:413-432` (`KnownEndpoint` `:412`), dispatcher `handleRequest` `:2114` | OpenAI/Anthropic-compatible endpoints; embedded web UI via `@embedFile` (`:1180-1182`); no filesystem serving |
| Generation endpoints | `/v1/chat/completions` (dispatch `:2335`), `/v1/completions` (`:2528`), `/v1/messages` (`:3041`), `/v1/responses` (`:2947`), `/v1/chat` (`:3551`), `/v1/chat/regenerate` (`:3398`); allow-list rows `:414-420` | Streaming SSE and batch decode; `validateAuth` gate at each dispatch, e.g. `:2340`, `:2532`, `:3045` |
| `/v1/models` | dispatch `:2307` | Auth-gated at `:2310`; discloses model id, backend, layer/embedding/vocab counts, ctx size, KV position, MTP depth |
| `/v1/embeddings` | dispatch `:2933` | Auth-gated at `:2935` |
| Tokenizer endpoints | `/v1/tokenize` `:2619`, `/v1/detokenize` `:2708` | Auth-gated at `:2621`, `:2710`; no separate quota (T7) |
| Health/readiness probes | `/health` `:2184`, `/ready` `:2211` | Unauthenticated by design; reduced bodies without auth (`authed` at `:2215`) |
| Metrics | `/metrics` `:2255` | Auth-required at `:2257` when a key is set; includes `agave_build_info` |
| KV cache export/import | `/v1/kv_cache` GET+POST `:2818`, `/v1/kv_cache/info` GET `:2770` | Raw hidden states in/out, cap 64 MiB (`:144`) |
| Conversations API | `/v1/conversations` `:3220` | Auth-gated at `:3223`; backed by in-memory store + optional disk persist |
| Static UI assets | `/` `:2295` and `/favicon.ico` `:2288` | No directory listing, no path join from request data |
| CLI arguments | `src/cli.zig`, consumed in `src/main.zig` (option table `:466-580`) | Model path, prompts, `--lora`, `--mmproj`, `--image`/`--video`, steering files, draft models: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig` `max_stdin_prompt_size` `:141` (1 MiB) | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (mmap `:311`) | Also LoRA adapters, which delegate to the same open path (`src/lora.zig:73` `applyLoraGguf` -> `gguf.GGUFFile.open` `:78`) and draft/mmproj models |
| SafeTensors dirs | `src/format/safetensors.zig` | Multi-shard + `index.json`; shard names filtered (`isSafeShardName` `:2166`, `:2648`) |
| PNG images (CLI + HTTP base64) | `src/image.zig` (64 MiB file `:19`, 4096 dim and 50 MiB inflate `:20-26`); HTTP via `json.extractJsonImage` (`src/server/json.zig:861`) | JPEG rejected; zip-bomb caps present |
| Video frames | `src/main.zig:3471-3500` | `ffmpeg` spawned with argv (not a shell) at `:3492`; local CLI only |
| Hub download channel | `src/pull.zig` (`hf_api_base` `:31`) | Repo ids and filenames validated (`isSafeFilename` `:64`, `isValidRepoName` `:79`); TLS via `std.http.Client` system CA bundle; blob URL is `resolve/main` (`:1055`) |
| Distributed TCP data plane | rank-0 listener `src/main.zig:3063-3095`, port defaults `:145`, `:147`, `:149` | Raw f32 frames (`src/parallel/transport.zig` `tcpSend` `:650` / `tcpRecv` `:690`); `max_peers` 8 (`:11`), enforced in `acceptPeer` `:268` |
| UDP peer discovery | `src/parallel/peer_discovery.zig` (`discovery_port` 49460 `:21`, broadcast to `discovery_port + 1` `:95`, `AGAVE-DISCOVER:` beacon `:24`, `AGAVE-JOIN:` reply `:25`, first responder wins `:117`) | Broadcast socket to 255.255.255.255 |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`src/parallel/transport.zig` `setupShm` `:286`, names `:288-289`) | Same-host, mode 0600; size guard is a `std.debug.assert` (`:339`) |
| NCCL | dlopen `libnccl.so.2` (`src/parallel/transport.zig:365-366`) | NCCL unique ID exchanged over the unauthenticated TCP link (`:388-392`) |
| Disagg prefill->decode KV transfer | listen `src/main.zig:3846-3903`, KV stream `src/models/qwen35.zig` `sendKvCache` `:2283` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT` (`src/main.zig:1095-1100`), `AGAVE_HOST` (`:1308-1310`), `AGAVE_API_KEY` (`:1326`, warn-if-unused `:1154`), `HF_TOKEN` (`src/pull.zig:339`), `NCCL_*` logging (`src/parallel/transport.zig:417-427`), `HOME`/`XDG_CACHE_HOME` (`src/server/conv_store.zig:70-75`), `TMPDIR` | Empty/whitespace env is unset (`pull.nonemptyEnv`). Debug-only: `AGAVE_VISION_DEBUG` (`src/models/vision.zig:360-363`), `AGAVE_DF2_DEBUG` (`src/main.zig:1417` via `pull.envFlagIsOne`), both inert unless exactly `1` and both writing debug buffers to stdout. No runtime config files; chat templates are compile-time string concatenation, not a Jinja evaluator (`src/chat_template.zig`, no `eval`) |
| Browser WASM demo | `web/agave.ts` `loadModel` `:307`; `src/wasm_entry.zig` | Fetches a user-typed model URL into WASM memory; contained to the browser sandbox |
| Container | `Dockerfile:231` (`USER agave`), `Dockerfile:233` (`EXPOSE 49453`), entrypoint binds 0.0.0.0 (`Dockerfile:253`); `docker-compose.yml:35` (127.0.0.1 mapping), `:39` (`AGAVE_API_KEY` required), `:58` (`agave-cache` volume), `:66-70` (`no-new-privileges`, `cap_drop: ALL`, `read_only: true`) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below. `EXPOSE` publishes only 49453; the distributed ports exist in the image but are not published by compose |
| Process-level tool registry | `src/server/tools.zig` (`max_tools` 16 `:10`, `Registry.register`) | Not attacker-reachable: slots are filled in-process by embedders, not from request JSON. Request-scoped tools are a separate, capped array (`max_tools` 8, `max_api_messages` 128, `src/server/json.zig:19,39`); they only reach a system prompt (`buildToolSystemPrompt` `src/server/server.zig:1631`) and a constrained decode allowlist |

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: `validateAuth` constant-time compare (`src/server/server.zig:1461-1483`, helper `constantTimeEql` `:1489`), called per endpoint (`:2194`, `:2257`, `:2310`, `:2340`, ...). Policy: non-loopback binds refuse to start without a key (`src/main.zig:1327-1333`); loopback binds are open by design. Unauthenticated mode also rejects non-loopback `Host` (DNS rebind, `isRebindHostUnauthenticated` `src/server/server.zig:1082`, backed by `isLoopbackHttpHost` `:1051`, call site `:2139-2144`) and mismatched `Origin` vs `Host` (CSRF, `isCrossOriginUnauthenticated` `:1091` over `originMatchesHost` `:1015`, call site `:2150-2158`).
2. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj, PNG) crosses into mmap/decode native code. Validation lives inside the parsers (see mitigations).
3. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with `O_NOFOLLOW` writes (`src/pull.zig:1218`). Commit SHA is used for snapshot directory naming, not as a pin on the download URL (`resolve/main` `:1055`).
4. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. Any host that connects is accepted into a rank slot subject only to the `max_peers` cap (`src/parallel/transport.zig` `acceptPeer` `:267-269`, reached from `setupTransport` `src/main.zig:3063-3095`); first UDP `AGAVE-JOIN` responder wins discovery (`peer_discovery.zig:117`). Largest unauthenticated boundary.
5. **Same-host processes -> shm segments.** Only uid/file-mode checks; names are fixed.
6. **Secrets -> process.** Env vars enter once at startup; nonempty `AGAVE_API_KEY` wins over CLI to avoid `ps` exposure (`src/main.zig:1175-1179`, `preferredSecret` `:1326`); empty env is unset (`:1322-1325`). Rotation: process restart. Storage: env only. Prompt-derived buffers are wiped before free (`wipeFree` / `wipeFreeTokens` `src/server/server.zig:1117-1126`, call sites `:2376-2404`, `:2643-2690`, `:2976-3123`), the per-connection read buffer that carries `Authorization` / `x-api-key` is zeroed before it is freed (`:7176`), and Hub `Authorization` buffers are zeroed (`src/pull.zig:764`, `:1114`).
7. **Process -> conversation file.** Prompts written to the cache-path conversation store unless `--no-conv-store` (`src/server/conv_store.zig:70`). Compose maps this under `agave-cache` (`docker-compose.yml:58`).
8. **Embedded UI -> jsDelivr.** `src/web/app.ts` fetches marked / DOMPurify / highlight.js from `cdn.jsdelivr.net` with SRI hashes (`:706-709`, `:753-756`) on the first response and the first code block rather than on page load (`loadMarkdown` `:743`, `loadHighlightJs` `:761`), and a stalled CDN resolves false after `cdn_script_timeout_ms` so the plain-text fallback settles. CSP allowlists that origin (`src/server/server.zig:1439-1445`). Compromise of the CDN without a matching hash is blocked; a rebuild that changes both script and hash is a build-time event.

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:231`), and compose adds `no-new-privileges` (`docker-compose.yml:66-67`).

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement (`src/server/server.zig:1461-1483`, `src/main.zig:1322-1343`).
- Information disclosure: `/health` and `/ready` reachable unauthenticated (reduced bodies; `docs/API.md` health/ready sections match code at `src/server/server.zig:2184-2253`). Residual: build info on `/metrics` requires auth (`:2257`); `/v1/models` (`:2307-2332`) discloses model geometry to any key holder.
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`http_buf_size` `:125` / `max_request_body_size` `:142`), duplicate `Content-Length` rejection (`parseContentLength` `:1365`, reject `:1373`), `Transfer-Encoding` rejected to avoid request smuggling (`:1412-1413`), scan-based JSON with message/tool caps (`src/server/json.zig:19,39`), connection cap 64 (`max_concurrent_connections` `:150`), 30 s read timeout (`connection_read_timeout_sec` `:435`, applied `:7146`).
- CSRF / DNS rebind on no-key loopback: mitigated by Origin vs Host (`:1091`, `:2150`) and loopback-only Host (`:1082`, `:2139`). Residual: any local process can still call the no-key loopback API (curl, scripts).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`src/server/rate_limiter.zig:1-2`, struct doc `:60-64`). CLI default is 0 = limiter disabled (`src/main.zig:657,659`, parsed `:1450-1451`, wired `:3821-3822`). When only one of rpm/tpm is set, the other bucket uses 1M RPM / 100M TPM (`src/server/server.zig:159-160`) so the configured side is the constraint. Grammar and `json_mode` bypass the scheduler and serialize under the model mutex (`src/server/server.zig:4136-4140`, same comment at `:6608`).
- Repudiation: request identity is a client-supplied `X-Request-Id`, length-capped (`max_client_request_id_len` `:131`), so a log line cannot be tied to a client beyond a self-declared id.
- Elevation: none known; single-process, no privileged helpers, no request-scoped tool execution.

**Artifact -> loader (GGUF/SafeTensors/LoRA/PNG)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array caps (`src/format/gguf.zig:20-28`), saturating size math (`tensorBytes` `:162-167`), all tensor offsets validated against file size (`:855-871` in `parseHeader`, overflow saturation rejected at `:867`); SafeTensors header capped at 100 MB and checked against file size (`src/format/safetensors.zig:20,236-237`); shard-name traversal blocked (`:2648`); PNG dimension/inflate caps (`src/image.zig:19-26`).
- Residual: unknown GGUF type codes fall back conservatively (`src/format/gguf.zig:109` `else => 1`; non-string array skip `:757-772`). Fuzz coverage exists: GGUF `src/fuzz_tests.zig`, SafeTensors header `src/format/safetensors.zig:4373+`, PNG `src/image.zig:728`. This is no longer an "unfuzzed parser" gap.

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between listing and blob GET; branch `main` is fetched, not the listed SHA (`src/pull.zig:1055`). Verification ends at GGUF magic bytes + Content-Length match (`verifyGgufBlob` `:1447`): T2.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce (`src/parallel/transport.zig` `allReduceAdd` `:473`, `tcpAllReduce` `:524`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`src/models/qwen35.zig:2283`), DoS via connection race against `max_peers` (`:268`), elevation by becoming rank 0 through spoofed beacons (`src/parallel/peer_discovery.zig:117`). `tcpRecv` fails on short reads rather than zero-filling (`:690`); that does not authenticate the peer: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`src/parallel/transport.zig:288-289`). `shm_unlink` then `O_EXCL` create discards a pre-planted send segment, then fails if a racer recreates it; it does not randomize the name: T5. The send-size guard is a ReleaseFast-stripped `std.debug.assert` (`:339`).

**Process -> conversation file**
- Disclosure: the JSON store is plaintext prompts (`src/server/conv_store.zig`). Load refuses files > 64 MiB (`max_store_bytes` `:21`, refusal `:361`). Compose persists it in `agave-cache` (`docker-compose.yml:50-58`): T6. That volume is the only copy; backup and restore are in `docs/DURABILITY.md`.

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time | Client spoofing on 49453 | `src/server/server.zig:1461-1483`, `src/main.zig:1322-1343` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `src/main.zig:1327-1333` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `src/server/server.zig:1091,2150-2158` |
| Loopback-only Host when no key | DNS rebinding (CWE-350) | `src/server/server.zig:1051,1082,2139-2144` |
| Empty CORS (`corsHeaders` returns `""`) | Cross-site read of a local server | `src/server/server.zig:939` |
| Body/header/connection/timeout caps; reject duplicate `Content-Length` and any `Transfer-Encoding` | Request DoS, HTTP smuggling | `src/server/server.zig:125,131,142,150,435,1365-1373,1412-1413,7146` |
| Token-bucket rate limits (opt-in, global) | Compute DoS when flags set | `src/server/rate_limiter.zig`; defaults off `src/main.zig:657,659` |
| Parser bounds/caps (GGUF, SafeTensors, PNG) + fuzz tests | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id / filename allowlists, `O_NOFOLLOW` blob writes, redirect-safe token handling | Download-path abuse | `src/pull.zig:64,79,764,1114,1218` |
| Secret and prompt buffer zeroization, env-over-CLI key | Credential leakage via ps / freed heap | `src/main.zig:1175-1179,1326`, `src/server/server.zig:1117-1126,7176`, `src/pull.zig:764,1114` |
| Container hardening | Container escape blast radius | `Dockerfile:231`, `docker-compose.yml:66-70` |
| Bounded grammar/schema parsing (input size, rule count, JSON schema depth/property count), each over-cap case a typed `error` | Grammar DoS from a hostile grammar or JSON schema | `src/grammar.zig:22-25` (caps), enforcement `:97,109,597,728,951` |
| SRI on jsDelivr scripts | UI CDN swap | `src/web/app.ts:706-709,753-756` |
| Response security headers (nosniff, DENY, no-referrer, HSTS, CSP, no-store) | Clickjacking, MIME sniff, cache | `src/server/server.zig:1439-1448`; claims match `docs/API.md` Response Headers |

Single points of failure: the API key alone carries all client-side authn on 49453; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports.

Docs-vs-code check (2026-09-27): `docs/API.md` auth / CORS / Host-rebind / rate-limit / security-header / health / ready claims match `src/server/server.zig`. `SECURITY.md` version and support claims match `build.zig.zon:4` (`.version = "0.3.0"`). No user-facing doc claims a mitigation the code lacks. Notes for the next pass:

- The previous revision's `src/server/server.zig` references pointed 13 to 24 lines above their current location, and most other files drifted by 3 to 200 lines. Every reference is re-anchored in this pass, and each now carries the symbol or literal at that line so drift is visible without a full re-read.
- The zeroization claim itself still holds, and is now precise: prompt-derived buffers, the per-connection request buffer, and the Hub `Authorization` buffers are all wiped before free (`src/server/server.zig:1117-1126,7176`, `src/pull.zig:764,1114`).
- `src/server/rate_limiter.zig:1` previously said "per-API-key"; the file header and the struct doc now both state that one instance is shared regardless of API key.

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests. With limits unset, nothing throttles GPU time. With limits set, the single global TPM/RPM bucket starves every other client (`src/server/rate_limiter.zig:1-2,125-145`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`src/server/server.zig:2818`; the radix prefix cache is likewise global, `src/server/scheduler.zig:309,429,698`).
3. **Latency gaming:** repeated user-supplied GBNF grammars or `json_mode` force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`src/server/server.zig:4136-4140`).
4. **Tokenizer abuse:** `/v1/tokenize` and `/v1/detokenize` accept arbitrary attacker text and run the vocabulary scan under the same single global bucket as generation, so a cheap endpoint can occupy the tokenizer's share of request time (`src/server/server.zig:2619,2708`; caps `src/server/json.zig:19`). T7.
5. **Model fingerprinting:** `/v1/models` names the loaded model, backend, layer/embedding/vocab counts, context size, and MTP depth to any holder of one key, which narrows the set of suitable attacks against that deployment (`src/server/server.zig:2307-2332`).
6. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds arbitrary f32 tensors (`src/parallel/transport.zig:267-269`, `src/parallel/peer_discovery.zig:117`).
7. **Prompt harvest from disk:** on a shared Unix user or a leaked compose volume, read `conversations.json` (`src/server/conv_store.zig:70`).
8. **Client-side trust note:** the `--serve` web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types (`web/agave.ts` `loadModel` `:307`), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols. Do not treat the HTTP API key as covering those ports.
2. T2: pin downloads to commit SHAs; verify checksums/signatures. Magic-byte + size is not integrity.
3. T3: document single-trust-domain status explicitly, or namespace caches/stores per key.
4. T4: per-key rate buckets; route grammar / `json_mode` through the scheduler. The limiter stays off unless flags are set, so operators who bind non-loopback with a key and no rpm/tpm have no compute quota.
5. T7: give `/v1/tokenize` and `/v1/detokenize` their own cheap quota, or document them as sharing the generation bucket.
6. T5: randomized shm names; keep a runtime (non-assert) send-size check in ReleaseFast.
7. T6: treat the conversation file as sensitive data (permissions already follow umask; no at-rest encryption).
8. Response path: [SECURITY.md](../SECURITY.md) records that no dedicated disclosure contact or fix-shipped SLA is defined in-repo (organizational; not invented here).
9. Observability: auth failures log `authentication failed` and increment a metric (`src/server/server.zig:1901`, second call site `:3049`). Logs are process stdout (compose `json-file` 10m x 3). There is still no durable audit trail an incident investigation can replay independently of the container log driver.

## 7. Response readiness (note only)

- Security-relevant events that do exist in logs: request start/done with `req=` / `xid=` (`logRequest` `src/server/server.zig:1140`, `logRequestDone` `:1155`), 401s (`:1901`, `:3049`), Host/Origin rejects (`:2142`, `:2154`).
- `X-Request-Id` copied into access logs is length-capped at 64 bytes (`max_client_request_id_len` `:131`, threadlocal slot `:887`, capture `:1008`, called from `handleRequest` `:2116`), so a hostile client cannot flood the log with an unbounded correlation string.
- No in-repo path from "vulnerability reported" to "fix shipped" beyond public GitHub issues. See [SECURITY.md](../SECURITY.md).
