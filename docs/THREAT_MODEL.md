# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; this file does not prescribe code fixes.

- **Last reviewed:** 2026-09-30 (every `path:line` below re-resolved against source in this pass)
- **Owner / review cadence:** organizational fields, to be assigned; not defined in-repo
- **Scope:** inference CLI (`src/main.zig`), HTTP server (`src/server/`), model loaders (`src/format/`), Hub downloads (`src/pull.zig`), the self-updater (`src/update.zig`), distributed transports (`src/parallel/`), WASM/browser demo (`web/`, `src/wasm_entry.zig`), container artifacts (`Dockerfile`, `docker-compose.yml`)
- **Out of scope:** backend kernel internals beyond their input parsing; GPU driver attack surface

Disclosure and supported-version policy: [SECURITY.md](../SECURITY.md). HTTP contract: [API.md](API.md).

Every reference below is `path:line` plus the symbol or literal at that line, so a later drift is cheap to detect without a full re-read. Two revisions of this file carried a discovery port that exists in no source file; the correct numbers are in section 1. Line numbers here were resolved by symbol, not by copying the previous revision: `rg -n 'fn <symbol>' <file>` first, then confirm the literal on the returned line. A reference that no longer resolves at the named symbol is stale, not approximate.

## Risk-ranked summary

| # | Threat | Boundary | Impact | Status |
|---|--------|----------|--------|--------|
| T1 | Any LAN host can join an inference cluster or inject tensors: TP/PP/disagg/discovery have no authentication | Peer node -> node (TCP 49454/49455/49456; UDP discovery reuses the same numbers) | Wrong outputs accepted as correct; full prompt transcript theft via disagg KV stream | **No mitigation.** No peer credential check anywhere in `src/parallel/` |
| T2 | `agave update` overwrites the running executable with bytes from a third-party host; the SHA-256 sidecar it checks against is fetched from the same host in the same run | GitHub -> local filesystem | Arbitrary code execution as the invoking user, at the next start of the binary | Partial: HTTPS-only, `trustedGithubUrl` host allowlist, SHA-256 sidecar, size caps. No signature, no pinned digest: T9 |
| T3 | Downloaded models are not content-integrity-checked: `resolve/main`, magic/size only | HF Hub -> loader | Poisoned weights steer outputs; malicious GGUF/SafeTensors exercises parser bugs | Partial: TLS + parser caps + GGUF magic |
| T4 | Single-key deployments have no tenant separation: `/v1/kv_cache` export and the global prompt-prefix / radix cache cross request owners | Client -> client (same server) | One API-key holder reads KV state derived from another user's prompts | Documented limitation, unmitigated |
| T5 | Rate limiting is one global bucket, default off; grammar/json_mode bypass the batch scheduler | Client -> compute | One client exhausts GPU time / latency for all | Partial: caps exist, identity does not |
| T6 | Predictable shared-memory names let any same-uid local process read/inject tensors | Local process -> shm | Local tensor injection during `--tp 2` same-host runs | Mode 0600 + `O_EXCL` after `shm_unlink` only |
| T7 | Conversation store persists prompts to disk (`$XDG_CACHE_HOME/agave/conversations.json`) | Process -> filesystem | Prompt transcript survives process exit; compose volume `agave-cache` holds it | Bounded file, durable replace, no encryption; no backup tier, so the volume is the only copy (`docs/DURABILITY.md`) |
| T8 | `/v1/tokenize` and `/v1/detokenize` run the tokenizer on attacker-chosen text with no per-endpoint quota | Client -> compute | Tokenizer CPU burn on loopback deployments; detokenize is a decode oracle over vocabulary | Bounded by 1 MiB body and 128-message scan cap; no endpoint-level rate limit |
| T9 | The idempotency replay ledger is global and keyed only on the client-supplied `X-Request-Id` | Client -> client (same server) | One key holder replays another holder's stored `/v1/chat` reply (up to 64 KiB, 1 h retention), or suppresses their retry | Unmitigated; the key carries no principal |

Highest-value correction for operators: **the API key protects only TCP 49453**. The TP/PP/disagg data ports and UDP discovery are separate listeners that never see it. Within 49453 the key authenticates but does not partition: everything a key holder sends shares one KV cache, one radix prefix cache, one conversation store, and one replay ledger.

## Assets

- The installed `agave` binary: `agave update` replaces it in place (`replaceVerified` `src/update.zig:193`, caller `replaceExecutable` `src/update.zig:336`, write target resolved through a symlink `:202-210`), so compromising the download path is code execution with the operator's privileges. T2.
- Model weights on disk and in VRAM: expensive to obtain, exfiltration target.
- Prompt and conversation content: in memory; in the bounded conversation store (`max_conversations` = 100 `src/server/conv_store.zig:36`, `max_messages_per_conv` = 1000 `:37`, enforcement `src/server/server.zig:699`); on disk via `src/server/conv_store.zig` (`defaultPath` `:84`, `$XDG_CACHE_HOME` else `$HOME` `:85`); latent in the KV cache and radix prefix cache (`src/server/scheduler.zig` `radix_tree: RadixTree` field `:314`, `matchPrefix` call `:434`, `insert` call `:719`).
- Generated response bodies retained for replay (`src/server/idempotency.zig` `body: []u8` `src/server/idempotency.zig:99`, `max_body_len` 64 KiB `:48`, `retention_ms` 1 h `:53`).
- Hidden states: `/v1/kv_cache` export returns raw per-layer KV blocks, capped at 64 MiB (`kv_export_max_bytes` `src/server/server.zig:112`, clamp `:88`).
- Credentials in process env: `HF_TOKEN` (`src/pull.zig:334` `config.getenv("HF_TOKEN")`), `GITHUB_TOKEN` (`src/update.zig:427`), `AGAVE_API_KEY` (`src/main.zig:1381` `preferredSecret`, definition `:1669`).
- GPU compute and availability: generation is the costly resource; DoS converts directly to cost.
- Output integrity: poisoned weights, a poisoned allReduce, or an update-channel substitution silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | endpoint table `src/server/server.zig:402-421` (`KnownEndpoint` `:395`, auth policy per entry), dispatcher `handleRequest` `:2177` | OpenAI/Anthropic-compatible endpoints; embedded web UI via `@embedFile`; no filesystem serving |
| TCP listen call | `src/server/server.zig:7695` `net.IpAddress.listen` (`.reuse_address = true`) | The only listener in `src/server/`; every other socket in the tree is in `src/parallel/` or the disagg prefill block in `src/main.zig` |
| Auth chokepoint | `authorizedForPath` `src/server/server.zig:1383`, called once per request at `:2252`; table lookup `authPolicyFor` `:426`; credential check `validateAuth` `:1356` | One routing call site. Two further `validateAuth` calls exist at `:2278` and `:2299`, both inside `/health` and `/ready` to trim the body when auth is absent, not to gate a route. Handlers do not re-check the key, and an unknown path is indistinguishable from a known one to an unauthenticated caller (401 precedes the 405/404 handlers) |
| Generation endpoints | `/v1/chat/completions` (dispatch `:2403`), `/v1/completions` (`:2596`), `/v1/messages` (`:3103`), `/v1/responses` (`:2990`), `/v1/chat` (`:3612`), `/v1/chat/regenerate` (`:3457`); allow-list rows `:403-409` | Streaming SSE and batch decode. All inherit the chokepoint; none has a per-route auth call |
| `/v1/models` | dispatch `:2380` | Discloses model id, backend, layer/embedding/vocab counts, ctx size, KV position, MTP depth |
| `/v1/embeddings` | dispatch `:2981` | Auth-gated by the chokepoint |
| Tokenizer endpoints | `/v1/tokenize` `:2682`, `/v1/detokenize` `:2768` | Auth-gated by the chokepoint; no separate quota (T8) |
| Health/readiness probes | `/health` `:2268`, `/ready` `:2295` | Unauthenticated by design (`.auth = .optional`, table `:416-417`); reduced bodies without auth |
| Metrics | `/metrics` `:2339` | Auth-required by the chokepoint (default policy; `agave_build_info` under auth) |
| KV cache export/import | `/v1/kv_cache` GET+POST `:2871`, `/v1/kv_cache/info` GET `:2825` | Raw hidden states in/out, cap 64 MiB (`:112`) |
| Conversations API | `/v1/conversations` `:3275` | Auth-gated by the chokepoint; backed by in-memory store + optional disk persist |
| Replay ledger on mutating routes | `claimIdempotencyKey` `src/server/server.zig:772`, ledger `src/server/idempotency.zig` `claim` `:139`, `complete` `:184`, `sendIdempotentReplay` `src/server/server.zig:1474` | Keyed on the sanitized `X-Request-Id` alone, one ledger per server: T9 |
| Static UI assets | `/` and `/favicon.ico` `:2366` | No directory listing, no path join from request data |
| `agave update` self-updater | `src/update.zig` `run` `:371`; dispatch from `src/main.zig:780`; docs `src/main.zig:2334` | Replaces the running executable after a SHA-256 sidecar comparison (`:159`, `decide` `:177`). Repo is operator-controlled (`--repo`, `validRepo` `:97` rejects URLs and second slashes); download URLs are pinned to GitHub hosts (`trustedGithubUrl` `:127`) |
| CLI arguments | `src/main.zig` (option table `cli_specs` `:480`, serve flags `:535-543`) | Model path, prompts, `--lora`, `--mmproj`, `--image` / `--video` (`:549,550`), `--draft-model`, `--mtp-model`, `--spec-token-map`, `--dir-steering-file`, `--grammar` file, `--conv-store` write path, `--kv-ssd-path`: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig` `max_stdin_prompt_size` `:163` (1 MiB), read `readStdinAll` `:120`, called `:2797` | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (`open` `:304`) | Also LoRA adapters, which delegate to the same open path (`src/lora.zig` `applyLoraGguf`), and draft/mmproj models |
| SafeTensors dirs | `src/format/safetensors.zig` (`open` `:120`) | Multi-shard + `index.json`; shard names filtered (`isSafeShardName` `:2169`, call site `:2632`) |
| Images (CLI) | `src/main.zig` `loadImage` `:2861` (PNG `src/image.zig:110`, PPM P6 `src/image.zig:236`); caps `src/image.zig:24,27` | JPEG rejected |
| PNG images (HTTP base64) | `src/server/json.zig:930` `extractJsonImage`; decoder `src/image.zig:110` `decodePng` | HTTP path is PNG-only; the PPM branch is CLI-only, so no PPM caps apply to a remote caller |
| Video frames | `src/main.zig:3624` | `ffmpeg` spawned with argv (not a shell); local CLI only |
| Hub download channel | `src/pull.zig` (`hf_api_base_default` `:33`) | Repo ids and filenames validated (`isSafeFilename` `:88`, `isValidRepoName` `:103`); TLS via `std.http.Client` system CA bundle; blob URL is `resolve/main` (`:1111`); `HF_ENDPOINT` may retarget the base and must be an http(s) URL (`:448,456`) |
| GitHub release channel | `src/update.zig` `releaseApiUrl` `:108`, `assetUrl` `:272`, `fetchBody` `:294` | `GITHUB_TOKEN` bearer `:427`; size caps `max_api_bytes` 10 MiB `:25`, `max_sidecar_bytes` 64 KiB `:26`, `max_asset_bytes` 256 MiB `:27`. T2 |
| Distributed TCP data plane | `setupTransport` `src/main.zig:1646` (delegates to `peer_link.setup` `src/parallel/peer_link.zig:100`), rank-0 listener reached from `src/main.zig:3175` (TP) and `:3198` (PP), port defaults `:167`, `:169` | Raw f32 frames (`src/parallel/transport.zig` `tcpSend` `:711` / `tcpRecv` `:751`); `max_peers` 8 (`:11`), enforced in `acceptPeer` `:270,313` |
| Disagg prefill listener | `src/main.zig:3985-4001` (socket/bind/listen, `SO_REUSEADDR`, `listen(1)`), port `:171` | Plaintext; `acceptPeer` on the listener takes the first connector |
| UDP peer discovery | `src/parallel/peer_discovery.zig` (`discoverPeer` `:62`, called with the parallel group's TCP data-port base from `src/main.zig:3166,3192`; rank 0 binds UDP `port` `:99` and broadcasts the beacon to UDP `port + 1` `:111`, workers bind `port + 1`; `AGAVE-DISCOVER:` beacon `:28`, `AGAVE-JOIN:` reply `:29`, first responder wins) | Broadcast socket. **There is no dedicated discovery port.** TP uses UDP 49454/49455, PP uses UDP 49455/49456, taken from `tp_discovery_port` / `pp_discovery_port` (`src/main.zig:167,169`). An earlier revision of this document, and of `SECURITY.md`, `docs/PARALLELISM.md`, and `docs/CONTRIBUTING.md`, named 49460/49461; no such constant exists in the tree. |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`src/parallel/transport.zig` `setupShm` `:331`, names `:333-334`) | Same-host, mode 0600; size guard is a `std.debug.assert` |
| NCCL | dlopen `libnccl.so.2` (`src/parallel/transport.zig:422-423`) | NCCL unique ID exchanged over the unauthenticated TCP link |
| Disagg prefill->decode KV transfer | `src/models/qwen35.zig` `sendKvCache` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT` (`src/main.zig:1132`), `AGAVE_HOST` (`:1196`, fallback `:1365`), `AGAVE_API_KEY` (`:1381`, warn-if-unused `:1201`, both-set warning `:1223`), `HF_TOKEN` (`src/pull.zig:334`), `GITHUB_TOKEN` (`src/update.zig:427`), `HF_ENDPOINT` (`src/pull.zig:449`), `HOME`/`XDG_CACHE_HOME` (`src/server/conv_store.zig:85`), `TMPDIR` (`src/main.zig:92`) | Empty/whitespace env is unset (`config.nonemptyEnv` `src/config.zig:19`). Debug-only: `AGAVE_VISION_DEBUG` (`src/models/vision.zig:357`) and `AGAVE_DF2_DEBUG` (`src/main.zig:1148,1472` via `config.envFlagIsOne` `src/config.zig:27`), both inert unless exactly `1`, both writing debug buffers to stdout, both validated at startup (`src/main.zig:1151-1152`). No runtime config file is read: `src/config.zig` exposes env accessors only, and chat templates are compile-time string concatenation, not a Jinja evaluator (`src/chat_template.zig`, no `eval`) |
| Browser WASM demo | `web/agave.ts` `loadModel` `:402`; `src/wasm_entry.zig` | Fetches a user-typed model URL into WASM memory; contained to the browser sandbox |
| Container | `Dockerfile:246` (`USER agave`), `Dockerfile:248` (`EXPOSE 49453`), entrypoint binds 0.0.0.0 (`Dockerfile:272`); `docker-compose.yml:36` (127.0.0.1 mapping), `:40` (`AGAVE_API_KEY` required), `:55` (models `:ro`), `:65` (`agave-cache` volume), `:75-78` (`no-new-privileges`, `cap_drop: ALL`, `read_only: true`) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below. `EXPOSE` publishes only 49453; the distributed ports exist in the image but are not published by compose |
| Process-level tool registry | `src/server/tools.zig` (`max_tools` 16 `:10`, `slots` `:32`) | Not attacker-reachable: slots are filled in-process by embedders, not from request JSON. Request-scoped tools are a separate, capped array (`max_tools` 8 `src/server/json.zig:39`, `max_api_messages` 128 `:19`); they only reach a system prompt and a constrained decode allowlist |

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: one dispatcher chokepoint, `authorizedForPath` (`src/server/server.zig:1383`, call site `:2252`, table lookup `authPolicyFor` `:426`), which calls `validateAuth` (`:1356`, constant-time compare helper `constantTimeEql` `:1394`) for every route whose `known_endpoints` entry is `AuthPolicy.required`. The default for a new or unlisted path is `required` (`src/server/server.zig:423-425`), so a route cannot become reachable without the key by omitting a check; only `/health` and `/ready` (`.optional`) and `/favicon.ico` (`.public`) opt out, and the first two still trim their body when `validateAuth` fails (`:2278`, `:2299`). Deny side pinned by `test "auth chokepoint denies every protected route"` (`:9148`, unknown-path assert `:9182-9183`), which also covers a case-variant path. Policy: non-loopback binds refuse to start without a key (`src/main.zig:1385-1386`), and an empty key is rejected rather than accepted as "present" (`:1394`); loopback binds are open by design. Unauthenticated mode also rejects non-loopback `Host` (DNS rebind, `isRebindHostUnauthenticated` `src/server/server.zig:1105`, applied `:2202`, backed by `isLoopbackHttpHost` `src/server/http.zig:164`) and mismatched `Origin` vs `Host` (CSRF, `isCrossOriginUnauthenticated` `src/server/server.zig:1114` applied `:2215`, over `originMatchesHost` `src/server/http.zig:128`).
2. **Client -> client (same server).** No principal is derived from the key. The KV cache, radix prefix cache, conversation store, and idempotency ledger are all per-server singletons shared by every request: T4, T9.
3. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj, PNG/PPM) crosses into mmap/decode native code. Validation lives inside the parsers (see mitigations).
4. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with `O_NOFOLLOW` writes (`src/pull.zig:1300`). Commit SHA is used for snapshot directory naming, not as a pin on the download URL (`resolve/main` `:1111`).
5. **GitHub releases -> local filesystem.** `agave update` is the only code path that overwrites a file the user executes. The channel is HTTPS with a GitHub host allowlist (`trustedGithubUrl` `src/update.zig:127`, userinfo and `github.com.evil.com` refused `:134`, tests `:585-586`), and the payload must match a `.sha256` sidecar (`checksumMatches` `:159`, sidecar name `writeSidecarName` `:83`). Both the asset and the sidecar come from the same release in the same run, so the digest proves transfer integrity, not publisher identity: T2. No code signature, no pinned digest, and the sidecar is not itself signed.
6. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. Any host that connects is accepted into a rank slot subject only to the `max_peers` cap (`src/parallel/transport.zig` `acceptPeer` `:270,313`); first UDP `AGAVE-JOIN` responder wins discovery (`src/parallel/peer_discovery.zig:62`). Largest unauthenticated boundary.
7. **Same-host processes -> shm segments.** Only uid/file-mode checks; names are fixed.
8. **Secrets -> process.** Env vars enter once at startup; nonempty `AGAVE_API_KEY` wins over CLI to avoid `ps` exposure (`src/main.zig:1222-1226`, `preferredSecret` `:1669`); empty env is unset. `GITHUB_TOKEN` is read by `agave update` and is sent as a bearer to `api.github.com` only (`src/update.zig:427,429`). Rotation: process restart. Storage: env only. Prompt-derived buffers are wiped before free (`wipeFree` / `wipeFreeTokens` `src/server/server.zig:1151,1157`), the per-connection read buffer that carries `Authorization` / `x-api-key` is zeroed before it is freed (`:7462-7464`), and Hub `Authorization` buffers are zeroed (`src/pull.zig:805,1199`).
9. **Process -> conversation file.** Prompts written to the cache-path conversation store unless `--no-conv-store` (`src/server/conv_store.zig:84`). Compose maps this under `agave-cache` (`docker-compose.yml:65`).
10. **Embedded UI -> jsDelivr.** `src/web/chat/markdown.ts` fetches marked / DOMPurify / highlight.js from `cdn.jsdelivr.net` (URLs `:13,15,17`, pinned SRI hashes `:14,16,18`) with the hashes checked in `loadCdnScript` (`:54`, `script.integrity` `:57`) on the first response and the first code block rather than on page load. A stalled CDN resolves false after the timeout so the plain-text fallback settles. CSP allowlists that origin for scripts only (`src/server/http.zig:374`); no stylesheet is fetched, so `style-src` stays `'unsafe-inline'` alone. Compromise of the CDN without a matching hash is blocked; a rebuild that changes both script and hash is a build-time event.

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:246`), and compose adds `no-new-privileges` (`docker-compose.yml:75`). `agave update` is the one deliberate write to an executable the user runs (`src/update.zig:336`), gated on the verdict of `decide` (`:177`) and confined to the release asset's bytes.

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement (`src/server/server.zig:1356,1394`, `src/main.zig:1385-1394`).
- Information disclosure: `/health` and `/ready` reachable unauthenticated (reduced bodies; `docs/API.md` health/ready sections match code at `src/server/server.zig:2268-2338`). Residual: build info on `/metrics` requires auth (`:2339`); `/v1/models` (`:2380`) discloses model geometry to any key holder.
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`http_buf_size` `:93` / `max_request_body_size` `:110`), duplicate `Content-Length` rejection (`parseContentLength` `src/server/http.zig:292`), `Transfer-Encoding` rejected to avoid request smuggling (`src/server/http.zig:341`), scan-based JSON with message/tool caps (`src/server/json.zig:19,39`), connection cap 64 (`max_concurrent_connections` `:120`, enforced `:7762`), 30 s read timeout (`connection_read_timeout_sec` `:446`, applied `:7433`).
- CSRF / DNS rebind on no-key loopback: mitigated by Origin vs Host (`:1114`) and loopback-only Host (`:1105`). Residual: any local process can still call the no-key loopback API (curl, scripts).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`src/server/rate_limiter.zig:1-2`, struct doc `:57-58`). CLI default is 0 = limiter disabled (`src/main.zig:681,683`, wired `:3952-3953`; unset side falls back to `rate_limit_unlimited_rpm` / `_tpm` `src/server/server.zig:129-130`, install site `:7633-7637`). Grammar and `json_mode` bypass the batch scheduler and run inline instead (comment at `src/server/server.zig:4226`, the condition that skips the scheduler `:4229-4230`, repeated per endpoint at `:4869-4871`, `:5676-5677`, `:6180-6181`; the inline decode under `if (sampling.json_mode)` at `:4472,4494`).
- Repudiation: request identity is a client-supplied `X-Request-Id`, length-capped (`max_client_request_id_len` `:99`, sanitize `sanitizeClientRequestId` `src/server/http.zig:117`, call site `src/server/server.zig:1100`), so a log line cannot be tied to a client beyond a self-declared id.
- Elevation: none known; single-process, no privileged helpers, no request-scoped tool execution.

**Client -> client (shared server state)**
- Information disclosure: `/v1/kv_cache` export (`:2871`) returns blocks derived from other holders' prompts; the radix prefix cache matches across requests (`src/server/scheduler.zig:434`, insert `:719`).
- Information disclosure via replay: the ledger stores the response body for 1 h under the caller's `X-Request-Id` (`src/server/idempotency.zig:48,53`, claim `:139`) and answers a matching key with those bytes (`src/server/server.zig:1474`). A second key holder who learns or guesses that id receives the first holder's completion without supplying the prompt: T9. The ring is 64 keys (`capacity` `:44`, ring `slots: [capacity]Slot` `:122`), so guessing is not even necessary for a client that can observe its own ids in the same deployment.
- Tampering/DoS: presenting a key that is `in_flight` yields a duplicate rejection (`src/server/server.zig:1446-1447`), so a hostile holder can deny another holder's retry (in-flight TTL 5 min `src/server/idempotency.zig:50`).

**Artifact -> loader (GGUF/SafeTensors/LoRA/PNG/PPM)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array/alignment caps (`src/format/gguf.zig:20-28`), saturating size math (`tensorBytes` `:162`), tensor offsets validated against file size in `parseHeader` `:832`; SafeTensors header capped at 100 MB and checked against file size (`src/format/safetensors.zig:20,248`); shard-name traversal blocked (`:2169`, enforced `:2632`); PNG dimension/inflate caps (`src/image.zig:24,27`).
- Residual: unknown GGUF type codes fall back conservatively (`src/format/gguf.zig` default branch on the type switch; non-string arrays are skipped). Fuzz coverage exists: GGUF (`src/format/gguf.zig` tests, `src/fuzz_tests.zig:1563`), SafeTensors (`:4439`), PNG/PPM (`src/fuzz_tests.zig:3138`, `src/image.zig:729`).

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between listing and blob GET; branch `main` is fetched, not the listed SHA (`src/pull.zig:1111`). Verification ends at GGUF magic bytes + Content-Length match (`verifyGgufBlob` `:1535`): T3.
- Retargeting: `HF_ENDPOINT` changes the base for every subsequent request and is validated only as an http(s) URL (`src/pull.zig:448,456`); a hostile env value redirects the download to an operator's own host. Env is a trusted input, so this is listed for completeness, not as an external threat.

**GitHub releases -> local filesystem (`agave update`)**
- Tampering / elevation: whoever controls the release host, the repo contents, or a redirect in the asset chain chooses the bytes that become the installed binary. Controls present: HTTPS only, `trustedGithubUrl` on both the asset and the sidecar (`src/update.zig:473`), SHA-256 match against the sidecar (`checksumMatches` `:159`), exact-tag asset naming (`writeAssetName` `:79`, `sameRelease` `:60`), size caps `:25-27`, and a verdict switch that refuses on every non-`.replaced` outcome (`:494-500`). Absent: any signature, a digest pinned outside the release, or a maintainer key: T2.
- Spoofing: `--repo` is operator-supplied and `validRepo` `:97` rejects URLs, extra slashes, and `.`/`..`, so the release host cannot be redirected by a crafted repo argument; the download URL must still pass `trustedGithubUrl`.
- DoS: `fetchBody` size caps keep a hostile response from exhausting memory; a 256 MiB asset download is a local CLI cost, not a service risk.
- Repudiation: `GITHUB_TOKEN` is optional; without it GitHub rate-limits the API call and `update` fails rather than proceeding. The install prints the version it installed (`:500-504`), which is the only durable record.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce (`src/parallel/transport.zig` `allReduceAdd` `:530` / `tcpAllReduce` `:585`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`src/models/qwen35.zig` `sendKvCache`), DoS via connection race against `max_peers` (`src/parallel/transport.zig:270,313`), elevation by becoming rank 0 through spoofed beacons (`src/parallel/peer_discovery.zig:28-29`). `tcpRecv` fails on short reads rather than zero-filling; that does not authenticate the peer: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`src/parallel/transport.zig:333-334`). `shm_unlink` then `O_EXCL` create discards a pre-planted send segment, then fails if a racer recreates it (`:342-343`); it does not randomize the name: T6. The send-size guard is a ReleaseFast-stripped `std.debug.assert`.

**Process -> conversation file**
- Disclosure: the JSON store is plaintext prompts (`src/server/conv_store.zig`). Load refuses files > 64 MiB (`max_store_bytes` `src/server/conv_store.zig:32`). Compose persists it in `agave-cache` (`docker-compose.yml:65`): T7. That volume is the only copy; backup and restore are in `docs/DURABILITY.md`.

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time, one dispatcher chokepoint | Client spoofing on 49453; routes that forget a check | `src/server/server.zig:1356,1383,1394,2252`, `src/main.zig:1385-1394`, test `:9148` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `src/main.zig:1385-1386` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `src/server/server.zig:1114,2215`, `src/server/http.zig:128` |
| Loopback-only Host when no key | DNS rebinding (CWE-350) | `src/server/server.zig:1105,2202`, `src/server/http.zig:164` |
| Empty CORS (`corsHeaders` returns `""`) | Cross-site read of a local server (CWE-942) | `src/server/http.zig:59` |
| Body/header/connection/timeout caps; reject duplicate `Content-Length` and any `Transfer-Encoding` | Request DoS, HTTP smuggling | `src/server/server.zig:93,99,110,120,446`, `src/server/http.zig:292,341` |
| Token-bucket rate limits (opt-in, global) | Compute DoS when flags set | `src/server/rate_limiter.zig`; defaults off `src/main.zig:681,683` |
| Parser bounds/caps (GGUF, SafeTensors, PNG) + fuzz tests | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id / filename allowlists, `O_NOFOLLOW` blob writes, redirect-safe token handling | Download-path abuse | `src/pull.zig:88,103,805,1199,1300` |
| Update-channel verification: HTTPS-only, GitHub host allowlist, exact-tag asset name, SHA-256 sidecar, size caps, refuse-on-anything-but-`replaced` | Corrupt or truncated download of the binary. Does not cover a hostile publisher: T2 | `src/update.zig:79,83,127,159,177,25-27,473,494-500`, tests `:585-586,595-599` |
| Secret and prompt buffer zeroization, env-over-CLI key | Credential leakage via ps / freed heap | `src/main.zig:1222-1226,1669`, `src/server/server.zig:1151,1157,7462`, `src/pull.zig:805,1199` |
| Bounded idempotency ledger (fixed ring, byte cap, TTL) | Retry storms re-running mutating routes | `src/server/idempotency.zig:44,48,50,53`. Does not bind a key to a principal: T9 |
| Container hardening | Container escape blast radius | `Dockerfile:246`, `docker-compose.yml:75-78` |
| Bounded grammar/schema parsing (input size, rule count, JSON schema depth/property count), each over-cap case a typed `error` | Grammar DoS from a hostile grammar or JSON schema | `src/grammar.zig:22-31` (caps) |
| SRI on jsDelivr scripts | UI CDN swap | `src/web/chat/markdown.ts:13-18,54-57` |
| Response security headers (nosniff, DENY, no-referrer, HSTS, CSP, no-store) | Clickjacking, MIME sniff, cache | `src/server/http.zig:368-377`, appended at each response writer (`:1416,1476,1972`); claims match `docs/API.md` Response Headers |

Single points of failure: the API key alone carries all client-side authn on 49453 and, because no principal is derived from it, also all of the shared-state isolation that does not exist; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports. On the update path, the single SHA-256 sidecar carries the entire publisher-authenticity guarantee, because the digest and the payload share a source.

Docs-vs-code check (2026-09-30): `docs/API.md` auth / CORS / Host-rebind / rate-limit / security-header / health / ready claims match `src/server/http.zig` and `src/server/server.zig`. `SECURITY.md` version and support claims match `build.zig.zon:4` (`.version = "0.10.2"`). Peer-discovery ports in `SECURITY.md`, `docs/PARALLELISM.md`, and `docs/CONTRIBUTING.md` all name the real 49454/49455/49456 base. **Corrections made in this pass:**

- Every `src/server/server.zig` and `src/main.zig` line number in the previous revision predated a large edit to both files, and the HTTP parsing and header helpers moved to `src/server/http.zig` (`corsHeaders`, `parseContentLength`, `sanitizeClientRequestId`, `security_headers`). This revision re-resolved all references by symbol. A cheap guard for the next pass: `rg -n` the symbol and confirm the literal, rather than trusting the number recorded here.
- `src/parallel/peer_discovery.zig` was cited bare in the previous revision; it is now always cited with its directory, and the three prose instances that omitted it were fixed.
- `src/update.zig` and `GITHUB_TOKEN` were absent from the inventory, the assets list, and the boundary list. They are now T2, an entry point, an asset, and boundary 5. A self-updater that overwrites the running executable is the highest-privilege write in the tree; it belongs in the model whether or not it is enabled by default.
- The previous revision's docs-vs-code note asserted `build.zig.zon:4` carried `0.8.0`. It carries `0.10.2`; `SECURITY.md` already said `0.10.2`. Recorded here so the next pass does not "correct" a correct claim.

Notes for the next pass:

- Auth is gated at exactly one routing call site, `authorizedForPath` at `src/server/server.zig:2252`. Two further `validateAuth` calls exist, at `:2278` and `:2299`; both are inside `/health` and `/ready` and only decide whether to omit detail from the body, not whether the request proceeds. A `validateAuth` call in any other handler is drift and a code smell; grep before trusting a per-route claim.
- The zeroization claim still holds: prompt-derived buffers, the per-connection request buffer, and the Hub `Authorization` buffers are all wiped before free (`src/server/server.zig:1151,1157,7462`, `src/pull.zig:805,1199`).
- `src/server/rate_limiter.zig:1` and the struct doc at `:57-58` both state that one instance is shared regardless of API key.
- `src/server/idempotency.zig` is reachable from `/v1/chat` and `/v1/chat/regenerate`. Recorded as T9.

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests. With limits unset, nothing throttles GPU time. With limits set, the single global TPM/RPM bucket starves every other client (`src/server/rate_limiter.zig:1-2`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`src/server/server.zig:2871`; the radix prefix cache is likewise global, `src/server/scheduler.zig:314,434,719`).
3. **Replay theft:** a key holder re-sends another holder's `X-Request-Id` to `/v1/chat` and receives their stored completion, up to 64 KiB, for an hour after the original request (`src/server/idempotency.zig:139`, replay path `src/server/server.zig:1474`). T9.
4. **Retry suppression:** presenting a key that is still `in_flight` collapses into a duplicate rejection, so a hostile holder can deny a victim's in-progress retry (`src/server/idempotency.zig:50,139`).
5. **Latency gaming:** repeated user-supplied GBNF grammars or `json_mode` force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`src/server/server.zig:4226,4229-4230,4472,4869-4871`; grammar size bounded by `src/grammar.zig:22-31`).
6. **Tokenizer abuse:** `/v1/tokenize` and `/v1/detokenize` accept arbitrary attacker text and run the vocabulary scan under the same single global bucket as generation, so a cheap endpoint can occupy the tokenizer's share of request time (`src/server/server.zig:2682,2768`; caps `src/server/json.zig:19`). T8.
7. **Model fingerprinting:** `/v1/models` names the loaded model, backend, layer/embedding/vocab counts, context size, and MTP depth to any holder of one key, which narrows the set of suitable attacks against that deployment (`src/server/server.zig:2380`).
8. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds arbitrary f32 tensors (`src/parallel/transport.zig:270,313`, `src/parallel/peer_discovery.zig:28-29`).
9. **Prompt harvest from disk:** on a shared Unix user or a leaked compose volume, read `conversations.json` (`src/server/conv_store.zig:84`).
10. **Forced update to a hostile build (no auth needed, needs a run):** an operator (or anything that can set `GITHUB_TOKEN`, `HF_ENDPOINT`, or a proxy's trust store) who runs `agave update` against a repo or a network position they control gets bytes written over the installed binary. The command's own defenses are real but all rooted in the same trust domain as the download: `src/update.zig:127,159,177`. T2.
11. **Client-side trust note:** the `--serve` web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types (`web/agave.ts:402`), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T2: sign the release, or pin a digest in-repo, so the sidecar is not the only trust anchor. Until then `agave update` is as strong as TLS to GitHub.
2. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols. Do not treat the HTTP API key as covering those ports.
3. T3: pin downloads to commit SHAs; verify checksums/signatures. Magic-byte + size is not integrity.
4. T9: namespace the idempotency ledger by a derived principal (key hash plus client identity), or refuse a `X-Request-Id` that is already claimed by a different requester.
5. T4: document single-trust-domain status explicitly, or namespace caches/stores per key.
6. T5: per-key rate buckets; route grammar / `json_mode` through the scheduler. The limiter stays off unless flags are set, so operators who bind non-loopback with a key and no rpm/tpm have no compute quota.
7. T8: give `/v1/tokenize` and `/v1/detokenize` their own cheap quota, or document them as sharing the generation bucket.
8. T6: randomized shm names; keep a runtime (non-assert) send-size check in ReleaseFast.
9. T7: treat the conversation file as sensitive data (permissions already follow umask; no at-rest encryption).
10. Response path: [SECURITY.md](../SECURITY.md) records that no dedicated disclosure contact or fix-shipped SLA is defined in-repo (organizational; not invented here).
11. Observability: auth failures log `authentication failed` and increment a metric. Logs are process stdout (compose `json-file` 10m x 3). There is still no durable audit trail an incident investigation can replay independently of the container log driver. `agave update` additionally writes no audit record beyond one stdout line, so a replaced binary is not distinguishable from a replaced binary later.

## 7. Response readiness (note only)

- Security-relevant events that do exist in logs: request start/done with `req=` / `xid=` (`logRequest` `src/server/server.zig:1208`, `logRequestDone` `:1223`), Host/Origin rejects (`:2202,2215`), 401s, oversized-body 413 (`:7472-7476`) and pre-routing 400.
- `X-Request-Id` copied into access logs is length-capped at 64 bytes (`max_client_request_id_len` `:99`), so a hostile client cannot flood the log with an unbounded correlation string. The same cap bounds the ledger key.
- `agave update` failures are reported on stderr with the reason and exit non-zero (`src/update.zig:494-509`); a successful install prints the new version. There is no log file, signature, or system-level record of the replacement.
- No in-repo path from "vulnerability reported" to "fix shipped" beyond public GitHub issues. See [SECURITY.md](../SECURITY.md).
