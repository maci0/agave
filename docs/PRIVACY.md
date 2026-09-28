# Privacy and data handling

What agave stores, where it lives, who can read it, and how to remove it.
Every claim here maps to code; the file references are the authority.

## What counts as personal data here

agave has no accounts, no user table, and no analytics. The personal data it
can touch is whatever a person types or pastes:

- prompt text and attached image bytes, including anything personal in them
- the model's replies to those prompts
- the system prompt typed in the REPL or the web UI
- REPL line history, in memory for the life of the process
- sampling settings, persisted per browser

Everything else (model files, KV cache, calibration files, metrics counters)
is either downloaded content or derived numbers with no free-text field.

## Where it is stored

| Data | Location | Protection |
| --- | --- | --- |
| Server conversations | `$XDG_CACHE_HOME/agave/conversations.json`, else `$HOME/.cache/agave/conversations.json` (override with `--conv-store <path>`) | owner-only mode 0600, written atomically (`src/server/conv_store.zig`, `src/durable_file.zig:160`) |
| Store sidecars | `<path>.corrupt` (a corrupt store moved aside), `<path>.overflow` (the part past the load caps) | same 0600 handling; a store an older build left world-readable is narrowed to 0600 on load (`src/server/conv_store.zig:205`) |
| Browser UI state | `localStorage`: temperature, top_p, max_tokens, stats toggle. `sessionStorage`: the system prompt | origin-scoped by the browser, never sent anywhere. The system prompt was moved out of `localStorage` and the legacy key is deleted on first read (`src/web/chat/storage.ts:29`) |
| REPL history | process memory only, up to 256 lines, wiped on free and on `/clear` (`src/readline.zig:56`) | never written to disk |
| Request logs | stderr | method, sanitized path, request id, status, duration. No prompt, reply, header value, or key (`src/server/server.zig:1280`) |
| Prometheus metrics | `/metrics`, in memory | counters and histograms only, no free-text labels |

Nothing is written outside these locations. There is no telemetry, no crash
reporting, no third-party script, and no analytics in the web UI: every
`fetch()` it makes is same-origin against the local server
(`src/web/chat/api.ts`).

## Network exposure

`--serve` binds `127.0.0.1` by default. Every endpoint except `/health` and
`/ready` requires the API key set with `--api-key` or `AGAVE_API_KEY`, and an
unauthenticated server additionally rejects non-loopback `Host` headers and
cross-origin requests, so a page in a browser cannot drive inference or read
conversation state (`src/server/server.zig:2296`).

`agave pull` is the only component that talks to a third party: it requests
model metadata and weights from `huggingface.co`. It sends the repository id
and, when set, the Hugging Face token. It never sends prompts.

## Retention

Conversations are kept until they are deleted. There is no automatic expiry,
so a store left in place keeps its messages until the user removes them. The
store caps are 100 conversations and 1000 messages per conversation; anything
past a cap is preserved in `<path>.overflow` rather than silently dropped
(`src/server/conv_store.zig:31`).

## Removing the data

- Delete one conversation: the trash control in the web UI sidebar, or
  `POST /v1/conversations` with `action=delete` and its `id`.
- Clear the active conversation: `/clear` in the REPL, or `/clear` as a chat
  message. In the REPL this also drops the line history that would otherwise
  recall the same prompts.
- Erase everything: stop the server and delete the store file and its
  `.corrupt` and `.overflow` sidecars.
- Keep nothing on disk in the first place: start the server with
  `--no-conv-store`. Conversations then live in memory and are gone on exit.
- Clear the browser side: the system prompt disappears with the tab, and the
  sampling keys are removable from `localStorage`.

Deletion wipes the freed buffers rather than leaving prompt text in the
allocator freelist, in memory (`src/main.zig:4042`,
`src/server/server.zig:549`) and on disk (the next store save rewrites the
file without the deleted conversations).

## Third parties and sub-processors

None. agave does not embed an SDK, does not load a remote script, and does
not forward a request to another service. `huggingface.co`, for `agave pull`
only, is the sole external destination.

## Reporting a privacy problem

Vulnerability reports, including accidental disclosure of conversation data,
go through the process in [SECURITY.md](../SECURITY.md).
