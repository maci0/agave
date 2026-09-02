# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; this file does not prescribe code fixes.

- **Last reviewed:** 2026-09-02 (working tree; re-verify file:line refs on each pass)
- **Owner / review cadence:** organizational fields, to be assigned; not defined in-repo
- **Scope:** inference CLI (`src/main.zig`), HTTP server (`src/server/`), model loaders (`src/format/`), Hub downloads (`src/pull.zig`), distributed transports (`src/parallel/`), WASM/browser demo (`web/`, `src/wasm_entry.zig`), container artifacts (`Dockerfile`, `docker-compose.yml`)
- **Out of scope:** backend kernel internals beyond their input parsing; GPU driver attack surface

Disclosure and supported-version policy: [SECURITY.md](../SECURITY.md). HTTP contract: [API.md](API.md).

## Risk-ranked summary

| # | Threat | Boundary | Impact | Status |
|---|--------|----------|--------|--------|
| T1 | Any LAN host can join an inference cluster or inject tensors: TP/PP/disagg/discovery have no authentication | Peer node -> node (TCP 49454/49455/49456, UDP 49460/49461) | Wrong outputs accepted as correct; full prompt transcript theft via disagg KV stream | **No mitigation.** Former wyhash transcript check is absent from `src/parallel/transport.zig` |
| T2 | Downloaded models are not content-integrity-checked: `resolve/main`, magic/size only | HF Hub -> loader | Poisoned weights steer outputs; malicious GGUF/SafeTensors exercises parser bugs | Partial: TLS + parser caps + GGUF magic |
| T3 | Single-key deployments have no tenant separation: `/v1/kv_cache` export and the global prompt-prefix / radix cache cross request owners | Client -> client (same server) | One API-key holder reads KV state derived from another user's prompts | Documented limitation, unmitigated |
| T4 | Rate limiting is one global bucket, default off; grammar/json_mode bypass the batch scheduler | Client -> compute | One client exhausts GPU time / latency for all | Partial: caps exist, identity does not |
| T5 | Predictable shared-memory names let any same-uid local process read/inject tensors | Local process -> shm | Local tensor injection during `--tp 2` same-host runs | Mode 0600 + `O_EXCL` after `shm_unlink` only |
| T6 | Conversation store persists prompts to disk (`~/.cache/agave/conversations.json`) | Process -> filesystem | Prompt transcript survives process exit; compose volume `agave-cache` holds it | Bounded file, durable replace; no encryption |

Highest-value correction for operators: **the API key protects only TCP 49453**. The TP/PP/disagg data ports and UDP discovery are separate listeners that never see it.

## Assets

- Model weights on disk and in VRAM: expensive to obtain, exfiltration target.
- Prompt and conversation content: in memory; in the bounded conversation store (`src/server/server.zig` `max_conversations` 100 / `max_messages_per_conv` 1000 at :137-138); on disk via `src/server/conv_store.zig` (default `$HOME/.cache/agave/conversations.json`, :68-73); latent in the KV cache and radix prefix cache (`src/server/scheduler.zig:277,397`).
- `HF_TOKEN` (`src/pull.zig:306`) and `AGAVE_API_KEY` (`src/main.zig:1232-1254`): credentials in process env.
- GPU compute and availability: generation is the costly resource; DoS converts directly to cost.
- Output integrity: poisoned weights or a poisoned allReduce silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | routes `src/server/server.zig:362-377`, dispatcher `handleRequest` :1758 | OpenAI/Anthropic-compatible endpoints, embedded web UI via `@embedFile` (:1013-1016); no filesystem serving |
| Health/readiness probes | `/health` :1828, `/ready` :1855 | Unauthenticated by design; reduced bodies without auth |
| Metrics | `/metrics` :1899 | Auth-required when a key is set; includes `agave_build_info` |
| KV cache export/import | `/v1/kv_cache` GET+POST :2441 | Raw hidden states in/out, cap 64 MiB (`kv_export_max_bytes` :134) |
| Conversations API | `/v1/conversations` :2828 | Auth-gated; backed by in-memory store + optional disk persist |
| CLI arguments | `src/cli.zig`, consumed in `src/main.zig` | Model path, prompts, `--lora`, `--mmproj`, `--image`/`--video`, steering files, draft models: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig` `max_stdin_prompt_size` :137 (1 MiB) | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (mmap :309) | Also LoRA adapters (`src/lora.zig:77`) and draft/mmproj models: same parser |
| SafeTensors dirs | `src/format/safetensors.zig` | Multi-shard + `index.json`; shard names filtered (`isSafeShardName` :2185, :2648) |
| PNG images (CLI + HTTP base64) | `src/image.zig` (64 MiB file, 4096 dim, 50 MiB inflate :19-26); HTTP via `json.extractJsonImage` (`src/server/json.zig:857`) | JPEG rejected; zip-bomb caps present |
| Video frames | `src/main.zig:3517-3543` | `ffmpeg` spawned with argv (not a shell); local CLI only |
| Hub download channel | `src/pull.zig` (`huggingface.co` :30) | Repo ids validated (:71-88); TLS via `std.http.Client` system CA bundle; blob URL is `resolve/main` (:1025) |
| Distributed TCP data plane | rank-0 listen `src/main.zig:1610-1620`, ports :139-145 | Binds `addr = 0` (all interfaces, :1611); raw f32 frames (`src/parallel/transport.zig` `tcpSend`/`tcpRecv` :608-660) |
| UDP peer discovery | `src/parallel/peer_discovery.zig:21,80-127,283-293` | Global broadcast 255.255.255.255; first `AGAVE-JOIN` wins (:117-121) |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`transport.zig:227-241`) | Same-host, mode 0600 |
| NCCL | dlopen `libnccl.so.2` (`transport.zig:318-321`) | 128-byte NCCL ID exchanged over the unauthenticated TCP link (:339-356) |
| Disagg prefill->decode KV transfer | listen `src/main.zig:3832-3843`, KV stream `src/models/qwen35.zig:2264-2282` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT/HOST/API_KEY` (`src/main.zig:1226-1262,2252-2266`), `HF_TOKEN` (`pull.zig:328`), `NCCL_*` logging (`transport.zig:372-383`), `HOME`/`TMPDIR`/`XDG_CACHE_HOME` | Empty/whitespace env is unset (`pull.nonemptyEnv`). No runtime config files; recipes and chat templates are compile-time (`src/chat_template.zig` is string concatenation, not Jinja eval) |
| Browser WASM demo | `web/agave.ts` `loadModel` :238-252; `src/wasm_entry.zig` | Fetches a user-typed model URL into WASM memory; contained to the browser sandbox |
| Container | `Dockerfile:199-222` (non-root `USER agave`, EXPOSE 49453, entrypoint binds 0.0.0.0), `docker-compose.yml:31-65` (127.0.0.1 mapping, `AGAVE_API_KEY` required, cap_drop ALL, read_only rootfs) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below. Compose volume persists conversations + Hub blobs |

No listed entry point in the previous revision has been removed. Added since last review: on-disk conversation store, PNG/HTTP image path, DNS-rebind Host check, `Transfer-Encoding` rejection.

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: `validateAuth` constant-time compare (`src/server/server.zig:1248-1269`). Policy: non-loopback binds refuse to start without a key (`src/main.zig:1247-1253`); loopback binds are open by design. Unauthenticated mode also rejects non-loopback `Host` (DNS rebind, :915-918, :1781-1788) and mismatched `Origin` vs `Host` (CSRF, :924-928, :1791-1801).
2. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj, PNG) crosses into mmap/decode native code. Validation lives inside the parsers (see mitigations).
3. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with `O_NOFOLLOW` writes (`src/pull.zig:1190-1197`). Commit SHA is used for snapshot directory naming (`pull.zig:1454`), not as a pin on the download URL (`resolve/main` :1025).
4. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. First TCP `accept` wins a rank slot (`transport.zig:215-222`, `main.zig:1620`); first UDP `AGAVE-JOIN` wins discovery (`peer_discovery.zig:117-121`). Largest unauthenticated boundary.
5. **Same-host processes -> shm segments.** Only uid/file-mode checks; names are fixed.
6. **Secrets -> process.** Env vars enter once at startup; nonempty `AGAVE_API_KEY` wins over CLI to avoid `ps` exposure (`src/main.zig:1241-1246`); empty env is unset. Rotation: process restart. Storage: env only; HTTP request buffers holding secrets are zeroed (`server.zig:6711-6714`); Hub `Authorization` buffers zeroed (`pull.zig:737,1089`).
7. **Process -> conversation file.** Prompts written to `$HOME/.cache/agave/conversations.json` unless `--no-conv-store` (`server.zig:6859-6864`, `conv_store.zig:68-73`). Compose maps this under `agave-cache`.
8. **Embedded UI -> jsDelivr.** `src/web/head.html:12-16` loads marked / DOMPurify / highlight.js from `cdn.jsdelivr.net` with SRI. CSP allowlists that origin (`server.zig:1238`). Compromise of the CDN without a matching hash is blocked; a rebuild that changes both script and hash is a build-time event.

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:199`).

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement (`server.zig:1245-1286`, `main.zig:1244-1251`).
- Information disclosure: `/health`, `/ready` reachable unauthenticated (reduced bodies, `docs/API.md` health/ready sections match code at `server.zig:1836-1893`). Residual: build info on `/metrics` requires auth (:1899-1904).
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`http_buf_size` / `max_request_body_size` :115, :132), duplicate `Content-Length` rejection (:1168, :1204-1212), `Transfer-Encoding` rejected to avoid request smuggling (:1207-1210), scan-based JSON with message/tool caps (`json.zig:19,39`), connection cap 64 (:140), 30 s read/write timeouts (`connection_read_timeout_sec` :383, applied :6683-6688).
- CSRF / DNS rebind on no-key loopback: mitigated by Origin vs Host (:924-928) and loopback-only Host (:915-918). Residual: any local process can still call the no-key loopback API (curl, scripts).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`rate_limiter.zig:56-59`). CLI default is 0 = limiter disabled (`src/main.zig:603-606,1358-1359`). When only one of rpm/tpm is set, the other bucket uses 1M RPM / 100M TPM (`server.zig:147-150`) so the configured side is the constraint. Grammar and `json_mode` bypass the scheduler and serialize under the model mutex (`server.zig:3701-3705,3809`).
- Elevation: none known; single-process, no privileged helpers.

**Artifact -> loader (GGUF/SafeTensors/LoRA/PNG)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array caps (`gguf.zig:20-28`), saturating size math (:145-148), all tensor offsets validated against file size (:860-873); SafeTensors header capped at 100 MB and checked against file size (`safetensors.zig:20,237`); shard-name traversal blocked (:2648); PNG dimension/inflate caps (`image.zig:19-26`).
- Residual: unknown GGUF type codes fall back conservatively (`gguf.zig:139` `else => 1`; non-string array skip `:775`). Fuzz coverage exists: GGUF `src/fuzz_tests.zig:1532`, SafeTensors header `safetensors.zig` fuzz block ~4379+, PNG `image.zig:728`. This is no longer an "unfuzzed parser" gap.

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between listing and blob GET; branch `main` is fetched, not the listed SHA (`pull.zig:1025`; SHA used for snapshot directory :1454). Verification ends at GGUF magic bytes + Content-Length match (`verifyGgufBlob` :1419-1433, `advertisedSizeAgrees` :976-978): T2.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce (`transport.zig:426-471`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`qwen35.zig:2264-2282`), DoS via connection race, elevation by becoming rank 0 through spoofed beacons (`peer_discovery.zig:117-121`). TCP `recvBuf` now fails on short reads (`transport.zig:648-660`) rather than zero-filling; that does not authenticate the peer: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`transport.zig:227-241`). `shm_unlink` then `O_EXCL` create discards a pre-planted send segment, then fails if a racer recreates it; it does not randomize the name: T5. Send-size guard is a ReleaseFast-stripped assert (`transport.zig:282`).

**Process -> conversation file**
- Disclosure: the JSON store is plaintext prompts (`conv_store.zig`). Load refuses files > 64 MiB (:21). Compose persists it in `agave-cache` (`docker-compose.yml:44-48`): T6.

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time | Client spoofing on 49453 | `server.zig:1248-1286`, `main.zig:1232-1254` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `main.zig:1236-1242` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `server.zig:924-928,1791-1801` |
| Loopback-only Host when no key | DNS rebinding (CWE-350) | `server.zig:915-918,1781-1788` |
| Empty CORS (`corsHeaders` returns `""`) | Cross-site read of a local server | `server.zig:805-810` |
| Body/header/connection/timeout caps; reject duplicate `Content-Length` and any `Transfer-Encoding` | Request DoS, HTTP smuggling | `server.zig:115-140,383,1204-1212,6683-6688` |
| Token-bucket rate limits (opt-in, global) | Compute DoS when flags set | `rate_limiter.zig`; defaults off `main.zig:603-606` |
| Parser bounds/caps (GGUF, SafeTensors, PNG) + fuzz tests | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id / filename allowlists, `O_NOFOLLOW` blob writes, redirect-safe token handling | Download-path abuse | `pull.zig:56-88,737,1089,1190-1197` |
| Secret zeroization, env-over-CLI key | Credential leakage via ps/buffers | `main.zig:1232-1235`, `server.zig:6711-6714`, `pull.zig:737,1089` |
| Container hardening | Container escape blast radius | `Dockerfile:199`, `docker-compose.yml:55-65` |
| Bounded grammar/schema recursion; fail-closed generate | Grammar DoS / unconstrained fallback | `grammar.zig:22-26`, `server.zig:3919-3921` |
| SRI on jsDelivr scripts | UI CDN swap | `src/web/head.html:12-16` |
| Response security headers | Clickjacking, MIME sniff, cache | `server.zig:1231-1238`; claims match `docs/API.md` Response Headers |

Single points of failure: the API key alone carries all client-side authn on 49453; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports.

Docs-vs-code check (2026-09-02): `docs/API.md` auth / CORS / Host-rebind / rate-limit / security-header claims match `src/server/server.zig`. No user-facing doc claims a mitigation the code lacks. `rate_limiter.zig` file comment says "per-API-key"; the implementation is one global instance (`RateLimiter` :56-59) — that comment is not an operator-facing claim.

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests. With limits unset, nothing throttles GPU time. With limits set, the single global TPM/RPM bucket starves every other client (`rate_limiter.zig:56-59,125-144`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`server.zig:2441`; radix prefix cache is likewise global, `scheduler.zig:277,397`).
3. **Latency gaming:** repeated user-supplied GBNF grammars or `json_mode` force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`server.zig:3701-3705`).
4. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds arbitrary f32 tensors (`transport.zig:215-222`, `peer_discovery.zig:117-121`).
5. **Prompt harvest from disk:** on a shared Unix user or a leaked compose volume, read `conversations.json` (`conv_store.zig:68-73`).
6. **Client-side trust note:** the `--serve` web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types (`web/agave.ts` `loadModel` :238-252), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols. Do not treat the HTTP API key as covering those ports.
2. T2: pin downloads to commit SHAs; verify checksums/signatures. Magic-byte + size is not integrity.
3. T3: document single-trust-domain status explicitly, or namespace caches/stores per key.
4. T4: per-key rate buckets; route grammar / `json_mode` through the scheduler. Limiter stays off unless flags are set — operators who bind non-loopback with a key and no rpm/tpm have no compute quota.
5. T5: randomized shm names; keep a runtime (non-assert) send-size check in ReleaseFast.
6. T6: treat the conversation file as sensitive data (permissions already follow umask; no at-rest encryption).
7. Response path: [SECURITY.md](../SECURITY.md) records that no dedicated disclosure contact or fix-shipped SLA is defined in-repo (organizational; not invented here).
8. Observability: auth failures log `authentication failed` (`server.zig:1599`) and increment a metric (`recordAuthFailure` :1598). Logs are process stdout (compose `json-file` 10m×3). There is still no durable audit trail an incident investigation can replay independently of the container log driver.

## 7. Response readiness (note only)

- Security-relevant events that do exist in logs: request start/done with `req=` / `xid=` (`server.zig:973,988,840-845`), 401s, Host/Origin rejects (`:1786,1798`).
- No in-repo path from "vulnerability reported" to "fix shipped" beyond public GitHub issues. See [SECURITY.md](../SECURITY.md).
