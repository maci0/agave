# Agave Threat Model

Living model of what this codebase exposes to attack, what it costs when attacked, and which controls stand in the way. Findings feed sec-review; fixes do not happen here.

- **Last reviewed:** 2026-08-23 (against working tree at main; re-verify file:line refs on each pass)
- **Owner / review cadence:** organizational fields, to be assigned; not defined in-repo
- **Scope:** inference CLI (`src/main.zig`), HTTP server (`src/server/`), model loaders (`src/format/`), Hub downloads (`src/pull.zig`), distributed transports (`src/parallel/`), WASM/browser demo (`web/`, `src/wasm_entry.zig`), container artifacts (`Dockerfile`, `docker-compose.yml`)
- **Out of scope:** backend kernel internals beyond their input parsing, GPU driver attack surface

## Risk-ranked summary

| # | Threat | Boundary | Impact | Status |
|---|--------|----------|--------|--------|
| T1 | Any LAN host can join an inference cluster or poison/inject tensors: distributed channels have no authentication | Peer node -> node (TCP 49454/49455/49456, UDP 49460/49461) | Wrong outputs accepted as correct; full prompt transcript theft via disagg KV stream | **No mitigation** |
| T2 | Downloaded models are not integrity-checked: mutable `resolve/main`, no checksums or signatures | HF Hub -> loader | Poisoned weights steer outputs; malicious GGUF exercises parser bugs | Partial: TLS + parser caps only |
| T3 | Model-file parsers are hardened but not fuzzed (`format/gguf.zig`, `format/safetensors.zig` absent from `src/fuzz_tests.zig`) | Artifact -> loader | Memory-safety bugs in native code from attacker-controlled bytes | Gap: fuzz coverage only |
| T4 | Single-key deployments have no tenant separation: `/v1/kv_cache` export and the global prompt-prefix cache cross request owners | Client -> client (same server) | One API key holder reads KV state derived from another user's prompts | Documented limitation, unmitigated |
| T5 | Rate limiting is one global bucket with effectively unlimited defaults; grammar requests bypass the scheduler | Client -> compute | One client exhausts budget/latency for all; GPU time is the costly asset | Partial: caps exist, identity does not |
| T6 | Predictable shared-memory segment names let any same-uid local process read/inject tensors | Local process -> shm | Local tensor injection during `--tp 2 --peers localhost` runs | None (mode 0600 only) |

Highest-value correction for operators: **the API key protects only port 49453**. The TP/PP/disagg data ports and UDP discovery are separate listeners that never see it.

## Assets

- Model weights on disk and in VRAM: expensive to obtain, exfiltration target.
- Prompt and conversation content: in memory, in the bounded conversation store (`src/server/server.zig:128-130`), and latent in the KV cache.
- `HF_TOKEN` (`src/pull.zig:295`) and `AGAVE_API_KEY` (`src/main.zig:1170`): credentials in process env.
- GPU compute and availability: generation is the billable resource; DoS converts directly to cost.
- Output integrity: poisoned weights or a poisoned allReduce silently corrupt every answer.

## 1. Attack surface inventory

Entry points found in code:

| Entry point | Location | Notes |
|---|---|---|
| HTTP API, default 49453 | routes `src/server/server.zig:334-349`, dispatcher :1393 | OpenAI/Anthropic-compatible endpoints, embedded web UI via `@embedFile` (:857-861); no filesystem serving |
| Health/readiness probes | `/health`, `/ready` (`server.zig:1450,1488`) | Unauthenticated by design; reduced bodies without auth |
| Metrics | `/metrics` (`server.zig:1536`) | Auth-required when a key is set |
| KV cache export/import | `/v1/kv_cache` GET+POST (`server.zig:2078`) | Raw hidden states in/out, up to 64 MiB (:125) |
| CLI arguments | `src/cli.zig`, consumed in `src/main.zig` | Model path, prompts, `--lora`, `--mmproj`, image/video files, steering files, draft models: all become parsed inputs |
| Stdin prompt pipe | `src/main.zig:108` (1 MiB cap) | Piped prompt mode |
| GGUF model/adapter files | `src/format/gguf.zig` (mmap, :308) | Also LoRA adapters (`src/lora.zig:77`) and draft/mmproj models: same parser |
| SafeTensors dirs | `src/format/safetensors.zig` | Multi-shard + `index.json`; shard names filtered (:1654) |
| Hub download channel | `src/pull.zig` (huggingface.co hardcoded, :29) | Repo ids validated (:70-88); TLS via `std.http.Client` system CA bundle |
| Distributed TCP data plane | rank-0 listen `src/main.zig:1518-1534`, ports `src/main.zig:110-116` | Binds `addr = 0` (all interfaces, main.zig:1519); raw f32 frames (`src/parallel/transport.zig:571-628`) |
| UDP peer discovery | `src/parallel/peer_discovery.zig:20,68-95,253-256` | Global broadcast 255.255.255.255; first responder wins (:89-95, :275-277) |
| Shared memory transport | `/agave_0to1`, `/agave_1to0` (`transport.zig:196-209`) | Same-host, mode 0600 |
| NCCL | dlopen `libnccl.so.2` (`transport.zig:288-292`) | 128-byte NCCL ID exchanged over the unauthenticated TCP link (:308-326) |
| Disagg prefill->decode KV transfer | `src/main.zig:3523-3564`, KV stream `src/models/qwen35.zig:2207-2259` | Plaintext TCP 49456; carries the entire prompt as KV |
| Environment variables | `AGAVE_PORT/HOST/API_KEY` (`main.zig:956,1157,1170`), `HF_TOKEN` (`pull.zig:295`), `NCCL_*` logging (`transport.zig:343-351`) | No runtime config files; recipes and chat templates are compile-time (`src/chat_template.zig` is string concatenation, not Jinja eval) |
| Browser demo fetches | `web/agave.ts` (`loadModel`) | Fetches arbitrary user-typed model URL into WASM memory; contained to browser sandbox |
| Container | `Dockerfile:199-222` (non-root `USER agave`, EXPOSE 49453, entrypoint binds 0.0.0.0), `docker-compose.yml:31-57` (127.0.0.1 mapping, `AGAVE_API_KEY` required, cap_drop ALL, read_only rootfs) | Compose is hardened; raw Dockerfile entrypoint relies on the API-key enforcement below |

No stale entries exist yet: this is the first version of the model.

## 2. Trust boundaries and data flow

1. **Client -> HTTP API.** Authn point: `validateAuth` constant-time compare (`server.zig:1047-1085`). Policy: non-loopback binds refuse to start without a key (`main.zig:1171-1178`); loopback binds are open by design. Cross-origin browser requests rejected when no key configured (CSRF control, `server.zig:783-790,1418-1446`).
2. **Artifact -> loader.** Whoever supplies the file (local user, Hub download, LoRA adapter, mmproj) crosses into mmap-parsing native code. Validation lives inside the parsers (see mitigations).
3. **HF Hub -> local cache.** Transport-authenticated (TLS) but content-unverified; blobs land under `$HF_HOME`-derived paths with symlink-safe writes (`pull.zig:1124-1141`).
4. **Peer node -> this node (TP/PP/disagg).** No authentication point exists anywhere on this boundary. First TCP connect wins a rank slot (`transport.zig:182-190`); first UDP reply wins discovery (`peer_discovery.zig:89-95`). This is the model's largest unnamed-in-practice boundary.
5. **Same-host processes -> shm segments.** Only uid/file-mode checks; any same-uid process can attach.
6. **Secrets -> process.** Env vars enter once at startup; `AGAVE_API_KEY` env wins over CLI to avoid `ps` exposure (`main.zig:1167-1189`). Rotation points: process restart. Storage: env only; request buffers holding secrets are zeroed (`server.zig:5818-5822`, `pull.zig:713-714`).

Privilege transitions: none at runtime. The process starts and stays at its launching privilege; the Dockerfile drops to `agave` before exec (`Dockerfile:199`).

## 3. Threats per boundary

**Client -> HTTP API**
- Spoofing: key guessing. Mitigated: constant-time compare, non-empty key enforcement.
- Information disclosure: `/health`, `/ready` reachable unauthenticated (reduced bodies, `API.md:308` matches code). Residual: build info on `/metrics` requires auth.
- Tampering/DoS: oversized or hostile JSON. Mitigated: 1 MiB body cap (`server.zig:111,123`), duplicate Content-Length rejection (:957-1011), scan-based JSON with depth/array caps (`json.zig:19-34,247-253`), connection cap 64 (:131), 30 s timeouts (:353-355).
- DoS: budget exhaustion. Partially mitigated: rate limiter exists but is one global bucket (`rate_limiter.zig:55`), defaults 1M RPM / 100M TPM (:140-141): see T5. Grammar requests bypass the scheduler and serialize under the inference mutex (`server.zig:3244-3247`).
- Elevation: none known; single-process, no privileged helpers.

**Artifact -> loader (GGUF/SafeTensors/LoRA)**
- Tampering/DoS: crafted headers driving huge allocations or OOB. Mitigated: GGUF metadata/tensor/array caps (`gguf.zig:20-28,774,777,701`), saturating size math (:145-151), all tensor offsets validated against file size (:842-856); SafeTensors header capped at 100 MB and checked against file size (`safetensors.zig:228-232`), offsets validated at lookup (:445-459), shard-name traversal blocked (:1654).
- Residual: these two parsers are absent from `src/fuzz_tests.zig` while json/grammar/CLI are covered: T3. Unknown type codes fall back conservatively but silently skip (`gguf.zig:139,757`).

**HF Hub -> loader (supply chain)**
- Tampering: repo contents change between pulls; branch `main` is fetched, not the pinned SHA (`pull.zig:970`; sha used only for directory naming :1340,1433). Verification ends at GGUF magic bytes + Content-Length match (:1236-1239,:1408-1417): T2.

**Peer node <-> peer node**
- All six STRIDE classes apply with no control present: spoofed rank joins, tensor tampering via allReduce poisoning (`transport.zig:447-471`), repudiation impossible (no identity), disclosure via disagg KV stream = full prompt transcript (`qwen35.zig:2207-2225`), DoS via connection race, elevation by becoming rank 0 through spoofed beacons (`peer_discovery.zig:275-277`). The wyhash transcript check is integrity-only and unauthenticated (`transport.zig:635-648`): detects divergence, does not attribute or prevent: T1.

**Local processes -> shm**
- Tampering/disclosure by same-uid processes on predictable names (`transport.zig:196-209`); `shm_unlink`-then-create discards a pre-planted segment rather than failing: T6. Send-size guard is a ReleaseFast-stripped assert (`transport.zig:249`).

## 4. Mitigations map

| Control | Covers | Reference |
|---|---|---|
| API key authn, constant-time | Client spoofing on 49453 | `server.zig:1047-1085`, `main.zig:1171-1189` |
| Bind policy: non-loopback requires key | Accidental internet exposure | `main.zig:1156-1178` |
| Origin/CSRF check when no key | Drive-by browser attacks on loopback servers | `server.zig:783-790,1418-1446` |
| Body/header/connection/timeout caps | Request DoS | `server.zig:111-131,353-355,6033-6052` |
| Token-bucket rate limits | Compute DoS (global only) | `rate_limiter.zig` |
| Parser bounds/caps (GGUF, SafeTensors) | Malicious artifact DoS/OOB | refs in section 3 |
| Repo-id validation, filename screening, O_NOFOLLOW blob writes, redirect-safe token handling | Download-path abuse | `pull.zig:54-65,70-88,720-725,1124-1141` |
| Secret zeroization, env-over-CLI key | Credential leakage via ps/buffers | `main.zig:1167-1189`, `server.zig:5818-5822`, `pull.zig:713-714` |
| Container hardening | Container escape blast radius | `Dockerfile:199`, `docker-compose.yml:53-57` |
| Bounded grammar/schema recursion | Grammar DoS | `grammar.zig:23-27`, fail-closed `server.zig:3459-3461` |

Single points of failure: the API key alone carries all client-side authn on 49453; the loopback-bind default carries all safety for no-key users; neither extends to the distributed ports.

Docs-vs-code check: `docs/API.md` auth/CORS/rate-limit/security-header claims were verified against code and match. No doc claims mitigations the code lacks. No `SECURITY.md` exists at all (see section 6).

## 5. Abuse cases (authenticated-hostile-user scenarios)

1. **Budget denial:** one key holder streams maximal requests to drain the single global TPM/RPM bucket, starving all other clients (`rate_limiter.zig:55,123-143`).
2. **Cross-request state reach:** a key holder exports `/v1/kv_cache` after other users' traffic and receives hidden-state blocks derived from their prompts on a shared single-key deployment (`server.zig:2078`; the radix prefix cache is likewise global, `src/server/scheduler.zig:353`).
3. **Latency gaming:** repeated user-supplied GBNF grammars force inline parse-and-constrain outside the batch scheduler, degrading concurrent clients (`server.zig:3244-3247`).
4. **Cluster hijack (no auth needed):** a LAN host answers the UDP beacon first or wins the TCP connect race and becomes a trusted rank, then feeds garbage tensors that pass silently because truncation zero-fills (`transport.zig:614-628`).
5. **Client-side trust note:** the web UI enforces nothing itself; all checks are server-side (correct posture). The standalone browser demo will load any model URL a visitor types/pastes (`web/agave.ts` `loadModel`), so a linked model can serve attacker-chosen completions locally, inside the sandbox.

## 6. Gaps requiring sec-review follow-up (ranked)

1. T1: add authentication (preshared secret at minimum) and identity handshake to TP/PP/disagg/discovery protocols.
2. T2/T3: pin downloads to commit SHAs, verify checksums/signatures; fuzz `gguf.zig`/`safetensors.zig`.
3. T4: document single-trust-domain status explicitly, or namespace caches/stores per key.
4. T5: per-key rate buckets; route grammar requests through the scheduler.
5. T6: randomized shm names with permission verification after open.
6. No `SECURITY.md` exists: there is no documented disclosure contact, supported-version list, or path from "vulnerability reported" to "fix shipped" (organizational decision required; not invented here).
7. No durable audit trail: requests carry `X-Request-Id` (`server.zig`) but auth failures and admin-ish calls (`/v1/kv_cache` POST) leave only ephemeral stdout; incident investigation currently has nothing to replay.
