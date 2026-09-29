# Security

Claims in this file are checked against source by the threat-model pass. If one
disagrees with the code, the code wins; see the docs-vs-code check at the end of
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#4-mitigations-map).

- **Last reviewed:** 2026-09-28

## Supported versions

Until **1.0.0** there is no multi-version support matrix and no promised LTS.
Fixes, including security fixes, land on current `main` / the latest product
tag that matches `build.zig.zon` `.version`. Older tags are not maintained.
Backports are not the default. See
[Support and lifecycle (0.x)](docs/CONTRIBUTING.md#support-and-lifecycle-0x).

Product version is **0.9.0** (0.x SemVer: breaking HTTP/CLI changes may land
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
49454/49455/49456, `src/main.zig:146,148,150`) and UDP peer discovery are
separate listeners with no authentication, so a deployment that exposes
them must rely on network-level isolation. See T1 in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).

Peer discovery has **no port of its own**: it reuses the parallel group's
TCP data-port base. `discoverPeer` takes that same base from
`src/main.zig` (`src/parallel/peer_discovery.zig:62`), and rank 0 binds
UDP `port` while broadcasting the beacon to UDP `port + 1`, where workers
bind (`src/parallel/peer_discovery.zig:99,111`). For tensor parallelism
that is UDP 49454/49455; for pipeline parallelism, UDP 49455/49456. A
firewall rule that allows 49454-49456 for TCP but denies UDP leaves
discovery closed; a rule written against any other port number, including
one quoted in older revisions of this file, does not.

The key authenticates on 49453 but does not partition: it identifies no
principal, so the KV cache, the prompt-prefix cache, the conversation
store, and the `X-Request-Id` replay ledger are shared by every request
the server accepts. On a single-key deployment, every holder is inside one
trust domain. See T3, T8 in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).

`docker-compose.yml` publishes the API port on 127.0.0.1 only
(`docker-compose.yml:36`) and requires `AGAVE_API_KEY` (`:40`). The
distributed ports are not published by compose; a raw `docker run` of the
image does not get that isolation.

## Operator notes

Two controls an operator is likely to assume are on are not:

- **Rate limiting is off unless asked for.** `--rate-limit-rpm` and
  `--rate-limit-tpm` both default to `0` (`src/main.zig:660,662`), which
  substitutes the effectively unlimited values `src/server/server.zig:162-163`.
  One global bucket, not per client (`src/server/rate_limiter.zig:58`). A
  server bound to a non-loopback address with a key and no rate-limit flags has
  no compute quota. See T4.
- **Same-host multi-rank runs share fixed shm names.** `/agave_0to1` and
  `/agave_1to0` (`src/parallel/transport.zig:321-322`) are mode 0600, so any
  other process running as the same uid can read or inject tensors. See T5.
