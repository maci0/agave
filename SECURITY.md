# Security

## Supported versions

Until **1.0.0** there is no multi-version support matrix and no promised LTS.
Fixes, including security fixes, land on current `main` / the latest product
tag that matches `build.zig.zon` `.version`. Older tags are not maintained.
Backports are not the default. See
[Support and lifecycle (0.x)](docs/CONTRIBUTING.md#support-and-lifecycle-0x).

Product version is **0.3.0** (0.x SemVer: breaking HTTP/CLI changes may land
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
49454/49455/49456) and UDP peer discovery (49460/49461) are separate
listeners with no authentication, so a deployment that exposes them must
rely on network-level isolation. See T1 in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#risk-ranked-summary).
