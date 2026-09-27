# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; this file does not prescribe code fixes.

- **Last reviewed:** 2026-09-27 (every line reference re-verified against source in this pass)
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
| T8 | The idempotency replay ledger is global and keyed only on the client-supplied `X-Request-Id` | Client -> client (same server) | One key holder replays another holder's stored `/v1/chat` reply (up to 64 KiB, 1 h retention), or suppresses their retry | Unmitigated; the key carries no principal |

Highest-value correction for operators: **the API key protects only TCP 49453**. The TP/PP/disagg data ports and UDP discovery are separate listeners that never see it. Within 49453 the key authenticates but does not partition: everything a key holder sends shares one KV cache, one radix prefix cache, one conversation store, and one replay ledger.

## Assets

- Model weights on disk and in VRAM: expensive to obtain, exfiltration target.
- Prompt and conversation content: in memory; in the bounded conversation store (`max_conversations` = 100 `src/server/server.zig:150`, `max_messages_per_conv` = 1000 `:151`); on disk via `src/server/conv_store.zig` (`defaultPath` `:72`, `$XDG_CACHE_HOME/agave` or `$HOME/.cache/agave`); latent in the KV cache and radix prefix cache (`src/server/scheduler.zig` `radix_tree: RadixTree` field `:314`, `matchPrefix` call `:434`, `insert` call `:721`).
- Generated response bodies retained for replay (`src/server/idempotency.zig` `body: []u8` `:52`, `max_body_len` 64 KiB `:34`, `retention_ms` 1 h `:39`).
- Hidden states: `/v1/kv_cache` export returns raw per-layer KV blocks, capped at 64 MiB (`kv_export_max_bytes` `src/server/server.zig:147`, size math `:118-123`).
- `HF_TOKEN` (`src/pull.zig:311` `config.getenv("HF_TOKEN")`) and `AGAVE_API_KEY` (`src/main.zig:1333` `preferredSecret`, definition `:1621`): credentials in process env.
- GPU compute and availability: generation is the costly resource; DoS converts directly to cost.
- Output integrity: poisoned weights or a poisoned allReduce silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | endpoint table `src/server/server.zig:435-456` (`KnownEndpoint` `:428`, auth policy per entry), dispatcher `handleRequest` `:2263` | OpenAI/Anthropic-compatible endpoints; embedded web UI via `@embedFile` (`:1310-1313`); no filesystem serving |
| TCP listen call | `src/server/server.zig:7541` `net.IpAddress.listen` | The only listener in `src/server/`; every other socket in the tree is in `src/parallel/` |
| Auth chokepoint | `authorizedForPath` `src/server/server.zig:1625`, called once per request at `:2336`; table lookup `authPolicyFor` `:459`; credential check `validateAuth` `:1598` | One call site. Handlers do not re-check the key, so a route cannot become reachable by omitting a check, and an unknown path is indistinguishable from a known one to an unauthenticated caller (401 precedes the 405/404 handlers) |
| Generation endpoints | `/v1/chat/completions` (dispatch `:2487`), `/v1/completions` (`:2675`), `/v1/messages` (`:3158`), `/v1/responses` (`:3069`), `/v1/chat` (`:3662`), `/v1/chat/regenerate` (`:3507`); allow-list rows `:435-456` | Streaming SSE and batch decode. All inherit the chokepoint; none has a per-route auth call |
| `/v1/models` | dispatch `:2464` (body `:2464-2490`) | Discloses model id, backend, layer/embedding/vocab counts, ctx size, KV position, MTP depth |
| `/v1/embeddings` | dispatch `:3060` | Auth-gated by the chokepoint |
| Tokenizer endpoints | `/v1/tokenize` `:2761`, `/v1/detokenize` `:2847` | Auth-gated by the chokepoint; no separate quota (T7) |
| Health/readiness probes | `/health` `:2352`, `/ready` `:2379` | Unauthenticated by design; reduced bodies without auth (`validateAuth` at `:2362`, `authed` at `:2383`) |
| Metrics | `/metrics` `:2423` | Auth-required by the chokepoint (`agave_build_info` `:2435`) |
| KV cache export/import | `/v1/kv_cache` GET+POST `:2950`, `/v1/kv_cache/info` GET `:2904` | Raw hidden states in/out, cap 64 MiB (`:147`) |
| Conversations API | `/v1/conversations` `:3329` | Auth-gated by the chokepoint; backed by in-memory store + optional disk persist |
| Replay ledger on mutating routes | `claimIdempotencyKey` `src/server/server.zig:781`, ledger `src/server/idempotency.zig` `claim` `:100`, `complete` `:135`, `sendIdempotentReplay` `src/server/server.zig:1713` | Keyed on the sanitized `X-Request-Id` alone, one ledger per server: T8 |
| Static UI assets | `/` `:2457` and `/favicon.ico` `:2450` | No directory listing, no path join from request data |
| CLI arguments | `src/main.zig` (option table `:461-560`) | Model path, prompts, `--lora` `:525`, `--mmproj` `:527`, `--image`/ PPM `--video` `:528-529`, `--draft-model` `:532`, `--mtp-model` `:533`, `--spec-token-map` `:537`, `--dir-steering-file` `:553`, `--grammar` file `:481`, `--conv-store` write path `:522`, `--kv-ssd-path` `:509`: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig` `max_stdin_prompt_size` `:142` (1 MiB), read `:2721` | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (`open` `:302`) | Also LoRA adapters, which delegate to the same open path (`src/lora.zig` `applyLoraGguf`), and draft/mmproj models |
| SafeTensors dirs | `src/format/safetensors.zig` (`open` `:120`) | Multi-shard + `index.json`; shard names filtered (`isSafeShardName` `:2166`, call site `:2629`) |
| Images (CLI) | `src/main.zig` `loadImage` `:2785` (PNG `src/image.zig:109`, PPM P6 `src/image.zig:235`); caps `src/image.zig:20,23,26` | JPEG rejected |
| PNG images (HTTP base64) | `src/server/server.zig:4031` `image_mod.decodePng`; extraction `extractJsonImage` `src/server/json.zig:928` | HTTP path is PNG-only; the PPM branch is CLI-only, so no PPM caps apply to a remote caller |
| Video frames | `src/main.zig:3498` | `ffmpeg` spawned with argv (not a shell); local CLI only |
| Hub download channel | `src/pull.zig` (`hf_api_base` `:32`) | Repo ids and filenames validated (`isSafeFilename` `:65`, `isValidRepoName` `:80`); TLS via `std.http.Client` system CA bundle; blob URL is `resolve/main` (`:1027`) |
| Distributed TCP data plane | `setupTransport` `src/main.zig:1598`, rank-0 listener reached from `:3099` (TP) and `:3122` (PP), port defaults `:146`, `:148` | Raw f32 frames (`src/parallel/transport.zig` `tcpSend` `:687` / `tcpRecv` `:727`); `max_peers` 8 (`:11`), enforced in `acceptPeer` `:300-301` |
| Disagg prefill listener | `src/main.zig:3891-3907` (socket/bind/listen, `SO_REUSEADDR`), port `:150` | Plaintext; `acceptPeer` on the listener takes the first connector |
| UDP peer discovery | `src/parallel/peer_discovery.zig` (`discovery_port` 49460 `:21`, broadcast to `discovery_port + 1` `:95`, `AGAVE-DISCOVER:` beacon `:24`, `AGAVE-JOIN:` reply `:25`, first responder wins, `discoverPeer` `:54`) | Broadcast socket |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`src/parallel/transport.zig` `setupShm` `:319`, names `:321-322`) | Same-host, mode 0600; size guard is a `std.debug.assert` |
| NCCL | dlopen `libnccl.so.2` (`src/parallel/transport.zig:398-399`) | NCCL unique ID exchanged over the unauthenticated TCP link (`:396`) |
| Disagg prefill->decode KV transfer | `src/models/qwen35.zig` `sendKvCache` `:2283` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT` (`src/main.zig:1104`), `AGAVE_HOST` (`:1317`), `AGAVE_API_KEY` (`:1333`, warn-if-unused `:1161`), `HF_TOKEN` (`src/pull.zig:311`), `HOME`/`XDG_CACHE_HOME` (`src/server/conv_store.zig:72`), `TMPDIR` | Empty/whitespace env is unset (`config.nonemptyEnv` `src/config.zig:18`). Debug-only: `AGAVE_VISION_DEBUG` and `AGAVE_DF2_DEBUG` (`src/main.zig:1424` via `config.envFlagIsOne` `src/config.zig:26`), both inert unless exactly `1` and both writing debug buffers to stdout. No runtime config file is read: `src/config.zig` exposes env accessors only, and chat templates are compile-time string concatenation, not a Jinja evaluator (`src/chat_template.zig`, no `eval`) |
| Browser WASM demo | `web/agave.ts` `loadModel` `:403`; `src/wasm_entry.zig` | Fetches a user-typed model URL into WASM memory; contained to the browser sandbox |
| Container | `Dockerfile:231` (`USER agave`), `Dockerfile:233` (`EXPOSE 49453`), entrypoint binds 0.0.0.0 (`Dockerfile:257`); `docker-compose.yml:35` (127.0.0.1 mapping), `:39` (`AGAVE_API_KEY` required), `:58` (`agave-cache` volume), `:66-70` (`no-new-privileges`, `cap_drop: ALL`, `read_only: true`) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below. `EXPOSE` publishes only 49453; the distributed ports exist in the image but are not published by compose |
| Process-level tool registry | `src/server/tools.zig` (`max_tools` 16 `:10`, `slots` `:32`) | Not attacker-reachable: slots are filled in-process by embedders, not from request JSON. Request-scoped tools are a separate, capped array (`max_tools` 8 `src/server/json.zig:39`, `max_api_messages` 128 `:19`); they only reach a system prompt and a constrained decode allowlist |

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: one dispatcher chokepoint, `authorizedForPath` (`src/server/server.zig:1625`, call site `:2336`, table lookup `authPolicyFor` `:459`), which calls `validateAuth` (`:1598`, constant-time compare helper `constantTimeEql` `:1636`) for every route whose `known_endpoints` entry is `AuthPolicy.required`. The default for a new or unlisted path is `required`, so a route cannot become reachable without the key by omitting a check; only `/health` and `/ready` (`.optional`) and `/favicon.ico` (`.public`) opt out, and the first two still trim their body when `validateAuth` fails. Deny side pinned by `test "auth chokepoint denies every protected route"` (`:8797`), which also asserts an unknown path (`/v1/nope`) and a case-variant path (`/V1/MODELS`) are denied. Policy: non-loopback binds refuse to start without a key (`src/main.zig:1337-1346`); loopback binds are open by design. Unauthenticated mode also rejects non-loopback `Host` (DNS rebind, `isRebindHostUnauthenticated` `src/server/server.zig:1167`, backed by `isLoopbackHttpHost` `:1136`) and mismatched `Origin` vs `Host` (CSRF, `isCrossOriginUnauthenticated` `:1176` over `originMatchesHost` `:1100`).
2. **Client -> client (same server).** No principal is derived from the key. The KV cache, radix prefix cache, conversation store, and idempotency ledger are all per-server singletons shared by every request: T3, T8.
3. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj, PNG/PPM) crosses into mmap/decode native code. Validation lives inside the parsers (see mitigations).
4. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with `O_NOFOLLOW` writes (`src/pull.zig:1190`). Commit SHA is used for snapshot directory naming, not as a pin on the download URL (`resolve/main` `:1027`).
5. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. Any host that connects is accepted into a rank slot subject only to the `max_peers` cap (`src/parallel/transport.zig` `acceptPeer` `:300-301`); first UDP `AGAVE-JOIN` responder wins discovery (`peer_discovery.zig:54`). Largest unauthenticated boundary.
6. **Same-host processes -> shm segments.** Only uid/file-mode checks; names are fixed.
7. **Secrets -> process.** Env vars enter once at startup; nonempty `AGAVE_API_KEY` wins over CLI to avoid `ps` exposure (`src/main.zig:1183-1186`, `preferredSecret` `:1621`); empty env is unset (`:1331`). Rotation: process restart. Storage: env only. Prompt-derived buffers are wiped before free (`wipeFree` / `wipeFreeTokens` `src/server/server.zig:1213,1219`), the per-connection read buffer that carries `Authorization` / `x-api-key` is zeroed before it is freed (`:7309`), and Hub `Authorization` buffers are zeroed (`src/pull.zig:739,1089`).
8. **Process -> conversation file.** Prompts written to the cache-path conversation store unless `--no-conv-store` (`src/server/conv_store.zig:72`). Compose maps this under `agave-cache` (`docker-compose.yml:58`).
9. **Embedded UI -> jsDelivr.** `src/web/app.ts` fetches marked / DOMPurify / highlight.js from `cdn.jsdelivr.net` with SRI hashes (`:728-731`) on the first response and the first code block rather than on page load (`loadMarkdown` `:765`, `loadHighlightJs` `:783`). A stalled CDN resolves false after `cdn_script_timeout_ms` (`:736`, `loadCdnScript` `:741`) so the plain-text fallback settles. CSP allowlists that origin (`src/server/server.zig:1582`). Compromise of the CDN without a matching hash is blocked; a rebuild that changes both script and hash is a build-time event.

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:231`), and compose adds `no-new-privileges` (`docker-compose.yml:66-67`).

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement (`src/server/server.zig:1598,1636`, `src/main.zig:1333-1346`).
- Information disclosure: `/health` and `/ready` reachable unauthenticated (reduced bodies; `docs/API.md` health/ready sections match code at `src/server/server.zig:2352-2422`). Residual: build info on `/metrics` requires auth (`:2423`); `/v1/models` (`:2464-2490`) discloses model geometry to any key holder.
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`http_buf_size` `:128` / `max_request_body_size` `:145`), duplicate `Content-Length` rejection (`parseContentLength` `:1500`), `Transfer-Encoding` rejected to avoid request smuggling (`:1552`), scan-based JSON with message/tool caps (`src/server/json.zig:19,39`), connection cap 64 (`max_concurrent_connections` `:153`), 30 s read timeout (`connection_read_timeout_sec` `:478`, applied `:7279`).
- CSRF / DNS rebind on no-key loopback: mitigated by Origin vs Host (`:1176`) and loopback-only Host (`:1136`). Residual: any local process can still call the no-key loopback API (curl, scripts).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`src/server/rate_limiter.zig:1-2`, struct doc `:57-59`). CLI default is 0 = limiter disabled (`src/main.zig:660,662`, parsed `:1457-1458`, wired `:3856-3857`; unset side falls back to `rate_limit_unlimited_rpm` / `_tpm` `src/server/server.zig:162-163`). Grammar and `json_mode` bypass the scheduler and serialize under the model mutex (`src/server/server.zig:4256-4260`, repeated for the other routes `:4877-4879`, `:5617-5619`, `:6071-6073`, comment `:6741-6744`).
- Repudiation: request identity is a client-supplied `X-Request-Id`, length-capped (`max_client_request_id_len` `:134`, sanitize `:1082`), so a log line cannot be tied to a client beyond a self-declared id.
- Elevation: none known; single-process, no privileged helpers, no request-scoped tool execution.

**Client -> client (shared server state)**
- Information disclosure: `/v1/kv_cache` export (`:2950`) returns blocks derived from other holders' prompts; the radix prefix cache matches across requests (`src/server/scheduler.zig:434`).
- Information disclosure via replay: the ledger stores the response body for 1 h under the caller's `X-Request-Id` (`src/server/idempotency.zig:34,39,52`, claim `:100`) and answers a matching key with those bytes (`src/server/server.zig:1702`). A second key holder who learns or guesses that id receives the first holder's completion without supplying the prompt: T8. The ring is 64 keys (`capacity` `:30`), so guessing is not even necessary for a client that can observe its own ids in the same deployment.
- Tampering/DoS: presenting a key that is `in_flight` yields a duplicate rejection, so a hostile holder can deny another holder's retry (`src/server/idempotency.zig:100-127`, in-flight TTL 5 min `:36`).

**Artifact -> loader (GGUF/SafeTensors/LoRA/PNG/PPM)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array caps (`src/format/gguf.zig:20-28`), saturating size math (`tensorBytes` `:162`), tensor offsets validated against file size in `parseHeader`; SafeTensors header capped at 100 MB and checked against file size (`src/format/safetensors.zig:20,245`); shard-name traversal blocked (`:2166`, enforced `:2629`); PNG dimension/inflate caps (`src/image.zig:20,23,26`).
- Residual: unknown GGUF type codes fall back conservatively (`src/format/gguf.zig` `else => 1` on the type switch; non-string arrays are skipped). Fuzz coverage exists: GGUF `src/format/gguf.zig:1677,1783` and `src/fuzz_tests.zig:1551`, SafeTensors `src/format/safetensors.zig:4277`, PNG/PPM `src/image.zig:733+`.

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between listing and blob GET; branch `main` is fetched, not the listed SHA (`src/pull.zig:1027`). Verification ends at GGUF magic bytes + Content-Length match (`verifyGgufBlob` `:1419`): T2.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce (`src/parallel/transport.zig` `allReduceAdd` `:506`, `tcpAllReduce` `:561`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`src/models/qwen35.zig:2283`), DoS via connection race against `max_peers` (`:300-301`), elevation by becoming rank 0 through spoofed beacons (`peer_discovery.zig:24-25`). `tcpRecv` fails on short reads rather than zero-filling (`:727`); that does not authenticate the peer: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`src/parallel/transport.zig:321-322`). `shm_unlink` then `O_EXCL` create discards a pre-planted send segment, then fails if a racer recreates it; it does not randomize the name: T5. The send-size guard is a ReleaseFast-stripped `std.debug.assert`.

**Process -> conversation file**
- Disclosure: the JSON store is plaintext prompts (`src/server/conv_store.zig`). Load refuses files > 64 MiB (`max_store_bytes` `:25`, refusal `:403`). Compose persists it in `agave-cache` (`docker-compose.yml:48-58`): T6. That volume is the only copy; backup and restore are in `docs/DURABILITY.md`.

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time, one dispatcher chokepoint | Client spoofing on 49453; routes that forget a check | `src/server/server.zig:1598,1625,1636,2336`, `src/main.zig:1333-1346`, test `:8797` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `src/main.zig:1337-1346` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `src/server/server.zig:1176` |
| Loopback-only Host when no key | DNS rebinding (CWE-350) | `src/server/server.zig:1136,1167` |
| Empty CORS (`corsHeaders` returns `""`) | Cross-site read of a local server | `src/server/server.zig:1024` |
| Body/header/connection/timeout caps; reject duplicate `Content-Length` and any `Transfer-Encoding` | Request DoS, HTTP smuggling | `src/server/server.zig:128,134,145,153,478,1500,1552,7279` |
| Token-bucket rate limits (opt-in, global) | Compute DoS when flags set | `src/server/rate_limiter.zig`; defaults off `src/main.zig:660,662` |
| Parser bounds/caps (GGUF, SafeTensors, PNG) + fuzz tests | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id / filename allowlists, `O_NOFOLLOW` blob writes, redirect-safe token handling | Download-path abuse | `src/pull.zig:65,80,739,1089,1190` |
| Secret and prompt buffer zeroization, env-over-CLI key | Credential leakage via ps / freed heap | `src/main.zig:1183-1186,1621`, `src/server/server.zig:1213,1219,7309`, `src/pull.zig:739,1089` |
| Bounded idempotency ledger (fixed ring, byte cap, TTL) | Retry storms re-running mutating routes | `src/server/idempotency.zig:28,30,34,36,39`. Does not bind a key to a principal: T8 |
| Container hardening | Container escape blast radius | `Dockerfile:231`, `docker-compose.yml:66-70` |
| Bounded grammar/schema parsing (input size, rule count, JSON schema depth/property count), each over-cap case a typed `error` | Grammar DoS from a hostile grammar or JSON schema | `src/grammar.zig:22-31` (caps) |
| SRI on jsDelivr scripts | UI CDN swap | `src/web/app.ts:728-731,741` |
| Response security headers (nosniff, DENY, no-referrer, HSTS, CSP, no-store) | Clickjacking, MIME sniff, cache | `src/server/server.zig:1576-1585`; claims match `docs/API.md` Response Headers |

Single points of failure: the API key alone carries all client-side authn on 49453 and, because no principal is derived from it, also all of the shared-state isolation that does not exist; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports.

Docs-vs-code check (2026-09-27): `docs/API.md` auth / CORS / Host-rebind / rate-limit / security-header / health / ready claims match `src/server/server.zig`. `SECURITY.md` version and support claims match `build.zig.zon:4` (`.version = "0.4.0"`), and its distributed-port references match `src/main.zig:146,148,150` and `src/parallel/peer_discovery.zig:21`. No user-facing doc claims a mitigation the code lacks. Notes for the next pass:

- Auth is enforced at exactly one call site, `authorizedForPath` at `src/server/server.zig:2336`. Any second `validateAuth` call inside a handler is drift and a code smell, not defence in depth; grep before trusting a per-route claim.
- The zeroization claim still holds: prompt-derived buffers, the per-connection request buffer, and the Hub `Authorization` buffers are all wiped before free (`src/server/server.zig:1213,1219,7309`, `src/pull.zig:739,1089`).
- `src/server/rate_limiter.zig:1` and the struct doc at `:57-59` both state that one instance is shared regardless of API key.
- `src/server/idempotency.zig` is reachable from `/v1/chat` and `/v1/chat/regenerate` and was not modeled in the previous revision. Recorded as T8.

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests. With limits unset, nothing throttles GPU time. With limits set, the single global TPM/RPM bucket starves every other client (`src/server/rate_limiter.zig:1-2`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`src/server/server.zig:2950`; the radix prefix cache is likewise global, `src/server/scheduler.zig:314,434,721`).
3. **Replay theft:** a key holder re-sends another holder's `X-Request-Id` to `/v1/chat` and receives their stored completion, up to 64 KiB, for an hour after the original request (`src/server/idempotency.zig:100`, replay path `src/server/server.zig:1702,1713`). T8.
4. **Retry suppression:** presenting a key that is still `in_flight` collapses into a duplicate rejection, so a hostile holder can deny a victim's in-progress retry (`src/server/idempotency.zig:36,100-127`).
5. **Latency gaming:** repeated user-supplied GBNF grammars or `json_mode` force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`src/server/server.zig:4256-4260`).
6. **Tokenizer abuse:** `/v1/tokenize` and `/v1/detokenize` accept arbitrary attacker text and run the vocabulary scan under the same single global bucket as generation, so a cheap endpoint can occupy the tokenizer's share of request time (`src/server/server.zig:2761,2847`; caps `src/server/json.zig:19`). T7.
7. **Model fingerprinting:** `/v1/models` names the loaded model, backend, layer/embedding/vocab counts, context size, and MTP depth to any holder of one key, which narrows the set of suitable attacks against that deployment (`src/server/server.zig:2464-2490`).
8. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds arbitrary f32 tensors (`src/parallel/transport.zig:300-301`, `src/parallel/peer_discovery.zig:24-25`).
9. **Prompt harvest from disk:** on a shared Unix user or a leaked compose volume, read `conversations.json` (`src/server/conv_store.zig:72`).
10. **Client-side trust note:** the `--serve` web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types (`web/agave.ts` `loadModel` `:403`), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols. Do not treat the HTTP API key as covering those ports.
2. T2: pin downloads to commit SHAs; verify checksums/signatures. Magic-byte + size is not integrity.
3. T8: namespace the idempotency ledger by a derived principal (key hash plus client identity), or refuse a `X-Request-Id` that is already claimed by a different requester.
4. T3: document single-trust-domain status explicitly, or namespace caches/stores per key.
5. T4: per-key rate buckets; route grammar / `json_mode` through the scheduler. The limiter stays off unless flags are set, so operators who bind non-loopback with a key and no rpm/tpm have no compute quota.
6. T7: give `/v1/tokenize` and `/v1/detokenize` their own cheap quota, or document them as sharing the generation bucket.
7. T5: randomized shm names; keep a runtime (non-assert) send-size check in ReleaseFast.
8. T6: treat the conversation file as sensitive data (permissions already follow umask; no at-rest encryption).
9. Response path: [SECURITY.md](../SECURITY.md) records that no dedicated disclosure contact or fix-shipped SLA is defined in-repo (organizational; not invented here).
10. Observability: auth failures log `authentication failed` and increment a metric. Logs are process stdout (compose `json-file` 10m x 3). There is still no durable audit trail an incident investigation can replay independently of the container log driver.

## 7. Response readiness (note only)

- Security-relevant events that do exist in logs: request start/done with `req=` / `xid=` (`logRequest` `src/server/server.zig:1270`, `logRequestDone` `:1285`), Host/Origin rejects, 401s, oversized-body 413 (`:7318`) and pre-routing 400 (`:467`).
- `X-Request-Id` copied into access logs is length-capped at 64 bytes (`max_client_request_id_len` `:134`, threadlocal slot `:953`, capture `:1082,1094`), so a hostile client cannot flood the log with an unbounded correlation string. The same cap bounds the ledger key (`src/server/idempotency.zig:28`).
- No in-repo path from "vulnerability reported" to "fix shipped" beyond public GitHub issues. See [SECURITY.md](../SECURITY.md).
