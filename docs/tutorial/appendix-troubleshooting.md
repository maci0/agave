# Appendix: Troubleshooting

**Time:** ~10 min

> After this appendix you can diagnose common inference failures from symptoms to likely causes.

<!-- TODO: Task 6 — full prose -->

### Code Flow

```mermaid
flowchart TD
  classDef setup fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
  classDef sync fill:#dcfce7,stroke:#22c55e,color:#14532d
  classDef success fill:#bbf7d0,stroke:#16a34a,color:#14532d

  Symptom["observe symptom"]:::setup --> Classify["classify failure domain"]:::sync
  Classify --> Trace["trace to subsystem"]:::sync
  Trace --> Link["link chapter or product doc"]:::sync
  Link --> Fix["apply fix or workaround"]:::success
```

## Gotchas

- Placeholder — filled in Task 6.

## How This Relates to the Code

**In the code:** [`main` load/generate path](../../src/main.zig), [`backend` dispatcher](../../src/backend/backend.zig), [`server` request handling](../../src/server/server.zig)

```text
observe symptom → classify (OOM / garbage / backend / format / distributed / server)
→ trace to subsystem (KV, sync, dispatcher, quant, transport, HTTP parse)
→ link to tutorial chapter or product doc → verify fix
```

**Next:** [Appendix: Mathematical Operations Reference →](appendix-math.md) | **Back:** [Chapter 23: Server / HTTP API ←](23-server-http-api.md)

---

## Glossary

(Terms added in Task 6.)
