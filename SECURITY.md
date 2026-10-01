# Security

Claims in this file are checked against source by the threat-model pass. If one
disagrees with the code, the code wins; see the docs-vs-code check at the end of
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#4-mitigations-map).

- **Last reviewed:** 2026-10-07

## Supported versions

Until **1.0.0** there is no multi-version support matrix and no promised LTS.
Fixes, including security fixes, land on current `main` / the latest product
tag that matches `build.zig.zon` `.version`. Older tags are not maintained.
Backports are not the default. See
[Support and lifecycle (0.x)](docs/CONTRIBUTING.md#support-and-lifecycle-0x).

Product version is **0.10.2** (0.x SemVer: breaking HTTP/CLI changes may land
without a major bump; they must appear in [CHANGELOG.md](CHANGELOG.md)).

## Reporting a vulnerability

This repository does not currently publish a dedicated disclosure mailbox,
private-reporting workflow, or fix-shipped SLA. Those are organizational
fields; they are not invented here.

The only in-repo path that exists is the project's public GitHub issue tracker.
Do not attach exploit payloads, poisoned model files, or live credentials to a
public issue.

## Model

The living attack-surface document is [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
Operator-facing HTTP auth, CORS, Host-rebind, rate-limit, and header behavior
is specified in [docs/API.md](docs/API.md) and implemented in
`src/server/server.zig`.

The API key covers the HTTP listener only (default TCP 49453). The
tensor-parallel, pipeline-parallel, and disaggregated data ports (TCP
49454/49455/49456, `src/main.zig:167,169,171`) and UDP peer discovery are
separate listeners with no authentication, so a deployment that exposes
them must rely on network-level isolation. See T1 in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).

Peer discovery has **no port of its own**: it reuses the parallel group's
TCP data-port base. `discoverPeer` takes that same base from
`src/main.zig` (`src/parallel/peer_discovery.zig:65`), and rank 0 binds
UDP `port` while broadcasting the beacon to UDP `port + 1`, where workers
bind (`src/parallel/peer_discovery.zig:104,127`). For tensor parallelism
that is UDP 49454/49455; for pipeline parallelism, UDP 49455/49456. A
firewall rule that allows 49454-49456 for TCP but denies UDP leaves
discovery closed; a rule written against any other port number, including
one quoted in older revisions of this file, does not.

The key authenticates on 49453 but does not partition: it identifies no
principal, so the KV cache, the prompt-prefix cache, the conversation
store, and the `X-Request-Id` replay ledger are shared by every request
the server accepts. On a single-key deployment, every holder is inside one
trust domain. See T4, T9 in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).

`docker-compose.yml` publishes the API port on 127.0.0.1 only
(`docker-compose.yml:36`) and requires `AGAVE_API_KEY` (`:40`). The
distributed ports are not published by compose; a raw `docker run` of the
image does not get that isolation.

## Operator notes

Two controls an operator is likely to assume are on are not, and a third
writes over the binary:

- **Rate limiting is off unless asked for.** `--rate-limit-rpm` and
  `--rate-limit-tpm` both default to `0` (`src/main.zig:681,683`), which
  substitutes the effectively unlimited values `src/server/server.zig:133-134`.
  One global bucket, not per client (`src/server/rate_limiter.zig:58-61`). A
  server bound to a non-loopback address with a key and no rate-limit flags has
  no compute quota. See T5.
- **`POST /v1/kv_cache` writes model state.** Any API-key holder can post a
  right-sized f32 blob and have it installed as the live KV cache
  (`src/server/server.zig:3047`; per-layer length bounds in
  `src/models/gemma4.zig:1654`). Lengths are checked, provenance is not, so
  injected hidden state is indistinguishable from a legitimate warm-start
  blob and every later answer on that slot is computed over it. Give
  import-only callers a separate key, or leave the route off on a
  multi-holder deployment. See T10 in
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).
- **`--kv-tiers vram+ram+ssd` writes KV to a world-readable file.** Demoted
  blocks are written to the `--kv-ssd-path` file, created mode 0644
  (`src/kvcache/tiered.zig:245,549`), while the conversation store is
  deliberately owner-only (`src/durable_file.zig:152`). Any local user can
  read prompt-derived hidden state from that file, and an unclean shutdown
  leaves it in place (the delete is a teardown step, `src/kvcache/tiered.zig:344`).
  Point the tier at a private path, or leave it off. See T11.
- **Same-host multi-rank runs share fixed shm names.** `/agave_0to1` and
  `/agave_1to0` (`src/parallel/transport.zig:333-334`) are mode 0600, so any
  other process running as the same uid can read or inject tensors. See T6.
- **`agave update` overwrites the installed binary.** It is the only code
  path that writes a file the user then executes (`src/update.zig:193,364`).
  The download must be HTTPS on a GitHub host (`trustedGithubUrl`
  `src/update.zig:127`) and must match a `.sha256` sidecar
  (`checksumMatches` `src/update.zig:159`), but the sidecar is fetched from
  the same release in the same run and is not signed. That is transfer
  integrity, not publisher identity: anything that controls the release, the
  repo, or the network position between you and GitHub controls the bytes
  that become the next binary. See T2 in
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).
