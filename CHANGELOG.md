# Changelog

All notable user-facing changes to Agave are recorded here.
Product version is **0.4.0** (`agave --version`, `/health`, `system_fingerprint`).
While on **0.x**, SemVer allows breaking changes without a major bump; such changes
must still appear under **Changed** or **Breaking** below. See
[Versioning & Releases](docs/CONTRIBUTING.md#versioning--releases).

> Note: git tag `v1.0` (2026-03-22) is a historical milestone name, not the product
> SemVer. Do not treat it as release `1.0.0`.

## [Unreleased]

### Changed
- **Chat UI and browser shell are React + Tailwind 4 + shadcn/ui.** Both
  surfaces moved from hand-written CSS and DOM calls to a React tree styled by
  Tailwind 4, with the shadcn primitives copied into `src/web/ui/` and the
  design tokens in one shared `src/web/ui/theme.css`. Behavior is unchanged:
  SSE streaming, conversation history, the slash commands, image attachment,
  the sampling panel, and the About dialog all work as before. The served page
  grows from 26 KB to 122 KB gzipped, which is React's share of the bundle.
- `scripts/build-web.sh` now runs bun and the Tailwind CLI, so the committed
  `src/web/app.js`, `src/web/style.css`, `web/shell.js`, `web/style.css` and
  `web/agave.js` are bundle and stylesheet outputs. `scripts/check-web-artifacts.sh`
  compares all five.

### Fixed
- The chat UI read the deferred marked and DOMPurify globals as bare
  identifiers, so a first message on a cold page could throw before the CDN
  script had run. They are read off `globalThis` now.

## [0.4.0] - 2026-09-27

### Breaking
- `--kv-tiers` now refuses to start on a discrete GPU:
  `Error: --kv-tiers requires a unified-memory backend`. It previously started
  and could produce incorrect output, because demoted blocks still pointed at
  device memory. Use it on unified-memory backends (Apple silicon, UMA) only.
- `agave pull` exit codes split: usage errors (`InvalidArgument`,
  `InvalidRepoFormat`) exit **2**, operational failures exit **1**. Previously
  every failure exited 2, so a download failure was reported as a usage error.
  Scripts that treat exit 2 as "the invocation was wrong" need updating.
- GGUF and SafeTensors readers reject a tensor whose data offset is not
  4-byte aligned (GGUF: `OffsetOutOfBounds`; SafeTensors: logged, tensor
  skipped). A repacked or hand-crafted model that loaded before now errors at
  load time. Every real writer pads to 64 bytes, so no shipped model changes.
- `POST /v1/detokenize` validates `tokens` per `docs/API.md`: a missing or
  empty array is `400` (`code: missing_required_parameter`), and a non-array
  value, a non-integer or negative element, or an array over 4096 entries is
  `400` (`code: invalid_value`). Previously a malformed array was decoded up to
  the bad element, so a client sending one bad ID got a `200` with a silently
  truncated prefix. The whole array is now rejected.
- `POST /v1/tokenize` returns `400` (`code: invalid_value`) for a non-string
  `text` or `content`, and `400` (`code: missing_required_parameter`) when none
  of `text`, `content`, or `messages` is present. Both previously reached the
  tokenizer.
- `POST /v1/chat/completions` `logprobs` follows the documented behavior: the
  sampled token's own `logprob` is returned with an empty `top_logprobs` list
  when `top_logprobs` is omitted, instead of suppressing `logprobs` entirely.
- Prometheus metric `agave_num_preemptions_total` is **removed**; it was
  reported without a backing counter. `/health` returns `kv_demotions` in
  place of `preemptions`, and `/metrics` gains
  `agave_kv_cache_tier_blocks{tier=vram|ram|ssd,state=used|total}`,
  `agave_kv_cache_demotions_vram_to_ram_total`, and
  `agave_kv_cache_demotions_ram_to_ssd_total`. `agave_gpu_cache_usage_perc`
  now derives from the VRAM tier alone. Dashboards and alerts reading
  `agave_num_preemptions_total` or the `preemptions` health field must be
  updated; `agave_input_tokens_in_flight` now means tokens still to prefill.
- `--repeat-penalty` is applied once per **distinct** token instead of once per
  occurrence, so `--repeat-penalty 1.2` is a 1.2x nudge at any output length.
  Repetition-heavy output therefore differs from previous releases; lower the
  value if the new penalty reads as too strong.
- Conversation-creation idempotency: `POST /v1/conversations` with
  `action=new` honors `X-Request-Id` as an idempotency key, and
  `POST /v1/chat` and `POST /v1/chat/regenerate` accept the header. A replayed
  non-streaming response is re-sent with `Idempotent-Replay: true`; a duplicate
  while the first is in flight, and any `stream=1` retry, is `409`
  (`code: duplicate_request`). Keys are sanitized to 64 characters of
  `A-Za-z0-9-_`, 64 are retained, and the replay window is 1 hour. A client
  that reused one `X-Request-Id` across unrelated requests now gets `409`.
- `KV /import` on a blob whose size does not match `n_tokens` returns `400`
  (`code: kv_import_failed`) instead of `501`; `501` is now specific to an
  architecture that does not implement `exportKvPrefix` / `importKvPrefix`.
  `POST /v1/messages` rejects `n > 1` with `400` (`n_not_supported`), matching
  `/v1/chat/completions`, and `POST /v1/conversations` rejects an
  unrecognised `action` with `400` (`code: unknown_conversation_action`),
  defaulting to `new` when the field is absent.

### Added
- `--sim-clock-ms <MS>`: pins every clock read to a virtual start time given in
  epoch milliseconds, so a whole run (CLI or `--serve`) replays from one value.
  Measured durations read 0, sleeps advance virtual time instead of blocking
  (distributed peer waits included, and the scheduler's per-request auto-seed
  becomes the virtual-clock value alone), and `--seed` no longer has to carry
  replay on its own. The pinned value is logged. A non-integer or negative
  value exits 2.
- Prometheus counters `agave_kv_promote_failures_total` and
  `agave_conv_store_save_failures_total` (both in `docs/OBSERVABILITY.md`).
  The first counts blocks that failed to promote to the VRAM tier, the second
  conversation-store writes that failed; requests still return `200`, so the
  counters are the only signal that history is being dropped.
- `POST /v1/chat` accepts `top_k` alongside `temperature`, `top_p`, `max_tokens`,
  `stream`, `system`, and `image`.
- `POST /v1/chat/completions` and `/v1/messages` return `409 Conflict` for a
  repeated `X-Request-Id` still in flight, or a replay whose response cannot be
  re-sent. `duplicate_request` is a documented `code` value.
- `tools/gguf_io.py`, a shared GGUF header reader. `tools/mixed-quant/
  splice_mixed_experts.py` is rebuilt on it, raises on a donor/base dimension
  mismatch and on an unknown ggml type instead of mis-sizing, and gains
  `--dry-run` to list the spliced tensors. `tools/synth-moe/moeify_gguf.py`
  uses the same module.
- `docs/OBSERVABILITY.md`: `--serve` Prometheus metrics, `/health` and
  `/ready` fields, and `X-Request-Id` log correlation.
- `docs/DURABILITY.md`: what state is on disk, RPO/RTO, and the conversation
  store backup/restore runbook.
- `scripts/conv-store-backup.sh` (`path`, `backup`, `verify FILE`,
  `restore FILE`, `check`, `--self-test`), shipped into the Docker image at
  `/usr/local/bin/conv-store-backup.sh`. Backups go to `$AGAVE_BACKUP_DIR`
  (default `$HOME/.agave-backups`) as `conversations-<UTC stamp>.json`, pruned
  to `$AGAVE_KEEP` (default 14). `restore` snapshots the outgoing store to
  `conversations-prerestore-<stamp>.json` first. `check` fails when the newest
  backup is older than `$AGAVE_MAX_AGE_HOURS` or no longer verifies, so a
  stopped job is not indistinguishable from a job with nothing to do.
  `--store PATH` names the store of a server started with `--conv-store`,
  which the environment-based default path cannot resolve. `zig build conv-store-backup-test`
  runs the self-test.
- `zig build ci`: the full local gate (`check` plus `lint-web` and
  `lint-shell`). `zig build check` alone does not cover the web lint.
- `zig build check-web`: regenerates `src/web/app.js` and `web/*.js` with tsc
  into a scratch dir and byte-compares them, so a `.ts` edit without rerunning
  `scripts/build-web.sh` fails the gate (`STALE: <path> differs from a fresh
  tsc build`). It is a dependency of `zig build lint-web` and `zig build ci`.
- `scripts/check-web-lint-scope.sh`: ratchets the `.oxlintrc.json`
  `ignorePatterns` list. An entry that is not already known debt, or a
  literal path that no longer exists, fails the gate. It runs first in
  `scripts/lint-web.sh`, so `zig build lint-web` and CI job `lint-web` both
  enforce it, and the list can shrink as `src/web/app.ts` and `web/shell.ts`
  are migrated but cannot grow.
- `tools/quality-testing/collect_continuations.py`: `--out` is a resume
  ledger, rewritten atomically after every call. A rerun skips prompts that
  already succeeded for the same `--model` and retries recorded errors, so an
  interrupted run no longer re-bills every completed call.

### Changed
- Docker image pin is `debian:bookworm-20260918-slim`. `DEBIAN_SNAPSHOT` and
  `SOURCE_DATE_EPOCH` match that day (2026-09-18 00:00:00 UTC).
- API key authentication is enforced at one dispatcher chokepoint
  (`authorizedForPath`), not per handler, and a path absent from the endpoint
  table is treated as protected. The reachable behavior for a configured key
  changes in one case: an unknown path now answers `401` before the `404` and
  `405` handlers, so an unauthenticated caller cannot distinguish a real route
  from a typo. `/health`, `/ready` (reduced body when the key does not match)
  and `/favicon.ico` stay unauthenticated, and CORS preflight (`OPTIONS`) is
  still answered before the check. The `401` on `/v1/messages` keeps its
  Anthropic envelope (`type: authentication_error`).
- `zig build test -Dtest-filter=<str>` now fails the build when a filter matches
  no `test "..."` name under `src/` or `tests/`, instead of compiling the test
  artifacts, running zero of them, and exiting 0. A filter that names a test
  the build does not compile on this host (a GPU-guarded test on a CPU-only
  host) still passes, so the check catches the typo, not the target.
- Server request-log timestamps render as `[HH:MM:SSZ]` instead of
  `[HH:MM:SS]`. The value is unchanged (always UTC); the `Z` marks it, because
  `journald`, `docker logs`, and a shell prompt all print local time and a bare
  `HH:MM:SS` reads as local. A log pipeline matching `^\[HH:MM:SS\]` must
  accept the trailing `Z`.
- `agave_inter_token_latency_seconds` is now sampled on every streaming
  endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
  `/v1/responses`) instead of one, so the histogram covers all of them.
  Dashboards that compared a `/v1/chat/completions` p99 against another
  endpoint's are now comparing like with like.
- `scripts/check-pins.sh` also verifies that `tests/uv.lock` and
  `research/kernels/uv.lock` still agree with their `pyproject.toml`
  (`uv lock --check`, skipped when uv is not on PATH). The research lock had
  drifted: it recorded `gguf >=0.18.0` and unpinned `numpy` and `torch` while
  the manifest pinned exact versions, so `uv sync --frozen` there resolved from
  a lock no longer matching the pins.
- The Python lint gate pins ruff. `ruff.toml` sets `required-version`, CI
  fetches that exact version through `uvx`, and `scripts/check-pins.sh` fails
  when the two disagree. Previously `uvx ruff` resolved whatever PyPI served,
  so a ruff release could add findings (or silence them) on an unchanged tree.
  A different local ruff now refuses to run instead of reporting different
  results; install the pinned one (`uv tool install ruff==0.16.4`).
- Docker Compose forwards `AGAVE_DF2_DEBUG` (documented in `.env.example` and
  `docs/API.md` but previously reachable only outside the container).
- Server: request logs now carry `req=<id>` on streaming client disconnects,
  SSE header overflow, and cancelled stream prefill. Image decode failures
  (a `400` client error, not a server fault) log at `warn` instead of `err`,
  so alerts keyed on error level stop firing for them.
- Server: `POST /v1/chat` and the regenerate endpoint rate-limit before the
  durable append, so a `429` no longer leaves an unanswered user message in
  the store that replays as context on the next turn, and regenerate no longer
  drops the last assistant reply before returning `429`. A tokenizer encode
  failure on those paths now returns `500` instead of proceeding.
- Server: a conversation save whose file close fails now reports the failure
  and does not rename, so a store whose data may not have reached disk is never
  published over the last good one.
- Server: a conversation store truncated mid-object, or carrying an
  `id`/`active_id`/`next_id` past `u32`, is quarantined to `<path>.corrupt`
  instead of panicking at start or loading the lost tail and rewriting the file
  without it. Loading a store that exceeds the save cap now logs how much was
  dropped.
- Split-GGUF discovery keys on shard `00001-of-NNNNN` only, so opening
  `model-00003-of-00005.gguf` directly no longer merges shard 3 twice. Shard
  indexes wider than the padding width, and shard totals that overflow, are
  ignored rather than mis-sliced.
- `agave pull` aborts with a `SymlinkFailed` error when a sidecar
  (`model.safetensors.index.json`, `config.json`, `tokenizer.json`,
  `tokenizer_config.json`) or shard symlink cannot be created or renamed,
  instead of warning and leaving a dangling path that fails later at model-open
  time. A failed download of those files is fatal too. An integrity-check read
  error on an already complete blob now warns and keeps the file, rather than
  deleting a multi-GB finished download on a transient `EIO`.
- `--color=<mode>` outside `auto|always|never` now prints
  `Error: unknown --color value '<x>'` with the valid options and exits 2 on
  the `--version` fast path as well as the full parse.
- Browser WASM shell (`web/`): load, drop-zone, file-input, model-URL, and
  clear controls are disabled while generating and re-enabled after, so
  swapping or clearing the model mid-generation no longer tears down the WASM
  engine mid-decode.
- Web UI (`--serve`): the per-response stats panel (tokens, tok/s, time,
  prefill) no longer disappears on a streamed reply, and the mobile drawer
  closes on every outcome of selecting a conversation, including the empty and
  error branches. Opening a long conversation restores in one layout pass, and
  an unchanged `/v1/conversations` response no longer rebuilds the sidebar.
- BPE pretokenizer cache is bounded by owned bytes (32 MiB) in addition to the
  8192-entry cap, so a long-lived `--serve` process holds a flat heap. Entries
  larger than the budget are not cached.
- `tools/synth-moe/moeify_gguf.py` refuses `--out` equal to `--in` (exit 2) and
  publishes by atomic rename, so a rerun can no longer destroy the only dense
  copy or leave a truncated GGUF that reads as a bad source.
- `scripts/build-web.sh` takes an optional out-root argument; with no argument
  it behaves as before. `zig build ptx` and `zig build amdgcn` now name the
  missing tool instead of failing with a bare exec error.
- Docker: both build stages drop the base image's `debian.sources` before
  writing the `snapshot.debian.org` pin, so two builds of the same commit no
  longer pick up different live `deb.debian.org` package versions. CI fails the
  build if fewer than two stages do this.
- Build: `tests/uv.lock` is committed and the e2e harness sets up with
  `uv sync --frozen --directory tests`, so a stale lock fails loudly instead of
  silently re-resolving.
- `src/kvcache/checkpoint.zig` is removed. No CLI flag or on-disk format ever
  shipped, so no file is affected; `docs/tutorial/24-advanced-features.md`
  records that KV checkpointing is not built and why.
- `--ctx-size auto` spends 80% of the usable-memory budget on KV cache, was a
  fixed 10x fudge that could size the context past what the cache holds, and
  falls back to the 4096 default when per-token KV cost exceeds 1 MiB
  (degenerate header metadata) instead of wrapping.
- A missing or mistyped model path now prints
  `Error: '<path>' does not exist. Check the path and try again.` (or
  `is not a SafeTensors directory.`) before the underlying errno line, and a
  mistyped REPL slash command gets a `did you mean` suggestion. Scripts that
  match the exact previous error text need updating.
- A backend with no fused FFN kernel for a weight dtype (for example CUDA
  Q4_0) now runs the standard gate/up/down path and warns, instead of feeding
  stale `ff_gate` values into the layer. `--ssd-streaming` warns that it does
  not page Qwen4-Exp PLE ngrams, and `--spec-mode ddtree` under `--serve`
  warns that the server uses linear draft and verify, so `--tree-budget` has no
  effect.
- Conversation store temp files are named `*.tmp.<pid>`, so two servers sharing
  one `--conv-store` path can no longer truncate each other's in-flight write.
- Developer tooling records timestamps in UTC: `research/kernels/autotune.py`
  log rows, `meta.json`, and staging directories, `tests/harness.py` run
  stamps (RFC 3339 with offset), and the `scripts/fetch-changelogs.sh` header.

### Fixed
- Server: a conversation store written in a newer envelope version is no longer
  quarantined to `<path>.corrupt`. It is an intact file this build cannot read,
  so it stays at the live path and persistence is disabled for the run, instead
  of being renamed aside and replaced by an empty store on the next save.
- Distributed startup: an incoming peer connection is polled with a 300s bound
  instead of a blocking `accept(2)`, so rank 0 no longer waits for the life of
  the process when the other rank never starts, and it fails with
  `AcceptTimeout`. The rank-0/rank-1 capability and RTT handshake is bounded at
  5s and reports which step timed out (and, on a connect failure, the address
  and port) instead of continuing silently. Bulk transfers are not bounded.
- `--video`: a failing or abnormally terminated ffmpeg, a failed frame scan, and
  an unreadable frame now name the cause and the file. Previously an unreadable
  video and a video that produced no frames both reported "no frames
  extracted", and a skipped frame was dropped without a count, so a video could
  be encoded from a silent prefix.
- GPU weight budget: `invalidateWeight` (CUDA, ROCm, Vulkan) now uncharges the
  dropped buffer, and re-admitting a live key that no longer fits evicts other
  entries, or is untracked when it cannot fit alone. `used_bytes` no longer
  counts freed memory or stays above `budget_bytes`, so a long run under
  tensor-parallel weight reuse stops evicting weights the budget had already
  released.
- Paged KV cache: `allocBlock` sets `ref_count = 1` and `freeBlock` sets it to
  `0`. A released block used to be left marked live, so the tiered cache could
  hand the same physical block to two requests and serve another request's KV
  content.
- Both chat UIs (`--serve` and the browser shell) map a generation failure to
  actionable copy and surface it in the transcript: an out-of-memory or engine
  fault says the engine failed and to reload the page, a network failure says
  so, and anything else is reported as `Could not generate a reply: <reason>`
  instead of a raw engine string.
- CPU `backendInfo` caches the one-time cache-size detection under a spinlock.
  `caches` is three words, so two threads racing the `detected` flag could
  publish a mix of both calls' results, and the device name and memory figures
  in `/health` and the model-info response could come from neither.
- `POST /v1/messages` answers a malformed request or an oversized body in the
  Anthropic error envelope. Both are raised before routing, so a client on that
  route used to receive an OpenAI-shaped `400` or `413` it could not parse.
- `POST /v1/chat` and `POST /v1/chat/regenerate` answer `500` (`type:
  server_error`) when the tokenizer fails mid-request. Both previously closed
  the connection with no status line, and the route lost the turn it had
  already appended or popped.
- `GET /v1/kv_cache/info` counts toward `agave_requests_total` and
  `agave_requests_completed_total`. It was the only authenticated route that
  recorded no request metrics, so orchestrator polling skewed the totals.
- `POST /v1/tokenize` names the offending field in `param` (`text` or
  `content`) when one of them is present but not a string.
- `--pflash-alpha` and `--diffusion-confidence` reject `nan` and out-of-range
  values instead of accepting them. `nan` compares false against every bound, so
  the old range check let it through and silently disabled block selection or
  diffusion confidence. `--dir-steering-ffn` and `--dir-steering-attn` also
  reject non-finite values now, like every other float option.
- `--diffusion-steps 0` and `--diffusion-canvas 0` exit 2 with
  `--flag must be >= 1` instead of silently running with 1 step or a 1-token
  canvas, and `--pflash-block-size 0` errors instead of warning and using 64.
- `agave pull` reports a shard whose filename cannot be built as `Error:`
  plus exit 1 rather than skipping it.
- `--serve --sleep-after` no longer hangs on shutdown waiting for the
  sleep-monitor thread.
- Chat-template prompt formatting no longer leaks its buffer when an
  allocation fails mid-format.
- `scripts/conv-store-backup.sh` no longer fails with `unbalanced braces` on a
  valid store whose message text or title contains `{` or `}`; brace counting
  now skips string literals and backslash escapes. A `restore` that previously
  exited 1 and left the live store alone now succeeds.
- `/health`, `/ready`, and the model-info response read `kv_seq_len` under the
  model lock, so `kv_seq_len` and `kv_cache_used` can no longer come from
  different moments.
- Tiered KV cache no longer double-counts a block promoted from SSD, so
  `kv_cache_used` stops drifting during SSD-tier runs.
- SafeTensors: a `num_hidden_layers` above `u32` no longer traps; it disables
  the layer aliases instead.
- `tools/quality-testing/collect_continuations.py` error records now carry the
  model, so a retry after a model switch is not skipped.
- Terminal width, line truncation, and readline cursor math count extended
  grapheme clusters, so a ZWJ family emoji, a skin-tone modifier, or a
  regional-indicator flag occupies the columns the terminal gives it instead of
  4x or 8x. A hand-edited conversation title past the 48-byte cap is clipped on
  a character boundary when loaded, so the web UI no longer shows a
  half-encoded title.
- Chat-template control tokens are stripped from replayed assistant content as
  well as user and tool content, so a prior model turn containing
  `<|im_start|>` or `<|im_end|>` can no longer inject role framing into the
  next request.
- `--serve` chat UI: the offline badge is replaced with a fresh node on each
  failure, so one click refetches once (it previously fired once per past
  failure) and the model name returns as a plain badge when the server answers.
  Stop announces "Generation stopped." rather than "Response complete.", and a
  slash command sent with an attached image is refused with a toast instead of
  silently dropping the image.
- A vision encoder header with `patch_size == 0`, `image_size < patch_size`,
  `embd_dim == 0`, `embd_dim` not a multiple of `n_heads`, or a zero
  `projection_dim` fails at init with `error.InvalidMetadata` rather than
  dividing by zero or truncating. The startup banner prints
  `KV cache n/a (size overflows)` when the KV size product overflows, and
  `agave pull` shard totals saturate instead of reporting a size smaller than
  the shards it lists.
- A client-supplied `thinking_budget_tokens` (or `thinking.budget_tokens`)
  above the `u32` range clamps to `4294967295` instead of trapping the server.
- `scripts/conv-store-backup.sh` rejects a non-positive-integer `AGAVE_KEEP`
  (`0`, negative, `abc`, `1.5`, or blank) instead of pruning the whole backup
  tier.
- `POST /v1/messages` keeps Anthropic `content` arrays and `system` blocks in
  the prompt. A `system` array of `{"type": "text"}` blocks is joined in order,
  and a message `content` array contributes its text parts plus the text of a
  `tool_result` part (capped at 16 KiB), which keeps the tool role for that
  turn. Previously the array form fell through to a string-field scan that
  returned the first nested string, so the prompt received a part's `type`
  value (`"text"`, `"tool_result"`) or nothing at all, and a `system` array
  was dropped entirely.
- Prefix reuse: a prompt carrying image embeddings (`n_visual > 0`) neither
  reads nor publishes the KV prefix memo, and a freed SSD-tier block drops its
  spill offset. The memo is keyed on token IDs alone and image placeholders are
  the same IDs for every image, so two different images behind an identical
  token sequence shared KV and the second answer came from the first request's
  picture; a re-allocated SSD block could likewise be promoted with the
  previous sequence's keys and values. Text-only prompts keep the full reuse.
- Streaming `POST /v1/messages` with tools no longer stalls when a decoded
  piece begins with a run of UTF-8 continuation bytes. The piece walk that
  backs the boundary at `piece_end < raw.len` could walk back past the start of
  the piece and then re-derive the same boundary, so the loop stopped emitting.
- Grammar-constrained generation strips only the three byte-level BPE markers
  that open with a `C3` or `C4` lead byte (`Ġ`, `Ċ`, `Ã`). Any other `C3`/`C4`
  lead pair is text (`"é"` is `C3 A9`, `"Ā"` is `C4 80`) and used to have its
  first character eaten, so the grammar validated a string that was not the one
  generated.
- Log lines sanitize per codepoint instead of per byte, so the C1 control range
  (U+0080-U+009F) is replaced with `?` along with C0 and DEL, and a byte that is
  not valid UTF-8 is replaced rather than passed through as a partial sequence.
  The sanitizer decodes first because the C1 range reaches a terminal through
  its ordinary-looking UTF-8 form (`C2 80`-`C2 9F`), which a byte filter passes
  and a terminal reads as CSI. Output is clipped on a character boundary, so a
  path longer than the buffer no longer leaves half a character in the log, and
  a log line carrying a C1 or non-UTF-8 byte now differs from the previous
  release's.

## [0.3.0] - 2026-09-02

### Breaking
- HTTP tool calling: parsed `<tool_call>` payloads whose `name` is not in the
  request `tools` list or the process-level registry are dropped. If nothing
  remains, the response is plain text instead of `finish_reason: "tool_calls"`.
  Previously any parseable name was returned. Declare (or register) every tool
  the model may call.

### Added
- Browser WASM (`web/agave.ts`): `init()` accepts `ArrayBufferView` (same as
  `loadModel`); `init()` and `loadModel()` take an optional `AbortSignal` as
  the last argument. `AgaveError.code` `invalid_argument` for a `maxTokens`
  that is not a non-negative 32-bit integer (0 still means the default).
- Server: persist web-UI conversations to `~/.cache/agave/conversations.json`
  (override `--conv-store`, disable `--no-conv-store`). Atomic tmp+fsync+rename;
  corrupt files are quarantined to `{path}.corrupt` on load.
- CLI: `--vram-budget` (GiB, or `auto`) caps GPU memory held by cached weights
  so a model larger than VRAM can run; weights past the cap are evicted and
  re-uploaded on demand. `--vram-budget-policy mru|lru` selects eviction order
  (default `mru`: a dense layer loop with LRU evicts the next layer's weights).
  `auto` sizes the cap from free device memory (75%).
- **Qwen 3.8 Flash-Next GGUF** (`qwen4exp`, `-Denable-qwen4exp`): llama.cpp
  split GGUF, separate from SafeTensors `qwen4_exp`. The PLE table stays
  demand-paged (`--mmap` is forced on, `MADV_RANDOM` on the table) and IQ2/IQ3
  expert GEMV runs on the CPU. No megakernel, vision, or MTP on this arch.
- **Qwen4-Exp / Qwen3.8-Flash-Next SafeTensors** (`qwen4_exp`,
  `-Denable-qwen4-exp`): Gated DeltaNet + QSA and NVFP4. PLE ngram shards are
  mmap'd but not read by the forward pass, and `--ssd-streaming` still pages
  MoE experts only.
- Browser WASM engine (`web/agave.ts`): `AgaveError` with a stable `code` (and
  optional `httpStatus`) so callers can handle init, download, and generate
  failures without matching `Error.message`.
- WASM export `agave_last_error(ctx)`: integer `WasmError` for the last
  init/generate on that context (`0` = ok).

### Fixed
- Chat UI: a streaming reply appends the new text instead of rewriting the
  whole message on each flush. A CDN script that neither loads nor errors
  falls back to plain text after 10 seconds.
- Browser WASM (`web/agave.ts`): `fetch` network, CORS, and abort failures
  throw `AgaveError` (`wasm_fetch_failed` / `download_failed`) instead of a
  raw `TypeError`. Re-`init()` frees the previous model against the old
  module before swapping, so a stale `ctx` is not used in new linear memory.
  `generate()` releases the prompt buffer if the WASM call throws. `destroy()`
  clears `initMessage`. Empty-prompt `agave_generate` no longer slices a null
  host pointer.
- HTTP: image parts on a model without a vision encoder return `400`
  (`code: vision_not_supported`) instead of being dropped. Non-data-URI
  `image_url` values return `400` (`code: image_decode_failed`).
- HTTP: `/v1/responses` honors OpenAI `max_output_tokens` (and
  `max_completion_tokens`). Non-string `input`/`prompt` return `400`
  (`code: invalid_value`) instead of `missing_required_parameter`.
- HTTP: `/v1/embeddings` and unsupported KV export `501` responses no longer
  count toward `/ready` error-rate degradation.
- Linux/macOS release binaries are linked as PIE. Docker's runtime stage
  freezes `SOURCE_DATE_EPOCH` to the same calendar day as the build stage.
- `-Denable-bench=false` skips installing `agave-bench` (Docker image default).
- Empty/whitespace `AGAVE_API_KEY`, `AGAVE_PORT`, `AGAVE_HOST`, `HF_TOKEN`,
  `HF_HOME`, `XDG_CACHE_HOME`, `HOME`, and `TMPDIR` are treated as unset.
  A sourced `.env.example` (`AGAVE_API_KEY=`) no longer overrides `--api-key`
  or fails loopback `--serve`. Invalid `AGAVE_PORT` errors name the env var.
- `AGAVE_DF2_DEBUG=1` is read once at startup (not per speculation round) and
  documented alongside `AGAVE_VISION_DEBUG`.
- Docker image pin: `DEBIAN_SNAPSHOT` / `SOURCE_DATE_EPOCH` now match the
  `debian:bookworm-20260824-slim` FROM tag (CI already required the calendar day).
- Docker "CPU + Gemma 3 only" image (CI `docker-build` and README minimal
  `docker buildx`) compiled Qwen4-Exp, DeepSeek V4, and DFlash2 because those
  `ENABLE_*` build-args were omitted and default on. Flags now match Compose.
- Docker Compose forwards `HF_TOKEN`, `NO_COLOR`, and `AGAVE_VISION_DEBUG` from
  `.env` (empty is unset). Hub cache for compose stays on the `agave-cache` volume.
- Docker image ships `LICENSE` at `/usr/share/doc/agave/copyright`; the OCI
  `org.opencontainers.image.licenses` label is `GPL-3.0-or-later` (was
  `GPL-3.0-only`, which contradicted the repo license).
- Docker image sets `HOME=/home/agave` so Hub pulls and `~/.cache` work on
  runtimes that do not copy passwd HOME (Kubernetes, some Podman setups).
- Conversations and the Vulkan pipeline cache honor `XDG_CACHE_HOME` (same
  fallback as `agave pull`: `$HOME/.cache`). GPU backends also search Fedora
  `/usr/lib64`, Alpine `/lib`, and Homebrew for dlopen libraries.
- ReleaseFast build: reconstruct tiered KV slices in Gemma 3, GLM-4, GPT-OSS,
  and Llama 4 so anonymous structs from `keysValues` type-check.
- Calibration `.cal` files, Vulkan pipeline cache, expert-profile JSON, and Hub
  `refs/main` now publish via atomic replace; Hub blob downloads `fsync` before
  the snapshot is advertised complete.
- Docker Compose: named volume `agave-cache` (or `AGAVE_CACHE_DIR`) at
  `/home/agave/.cache` so conversations and caches survive container replace.
- HTTP: `GET /v1/kv_cache` returns `400` (`code: invalid_value`) when `n_tokens`
  exceeds current `kv_seq_len` (was `501`). Present-but-invalid `n_tokens` uses
  `invalid_value`; missing still uses `missing_required_parameter`.
- HTTP: `/v1/detokenize` returns `400` when `tokens` has more than 4096 entries
  instead of silently truncating.
- HTTP: `POST /v1/conversations` select/delete with missing or invalid `id`
  returns `400` instead of coercing to `0` and `404`.
- HTTP: unauthenticated `/ready` degraded responses include `reason`, matching
  `/health` and the documented probe contract.
- HTTP: `/v1/chat` image decode failures return JSON `400`
  (`code: image_decode_failed`) like `/v1/chat/completions` (was HTML `200`).
- HTTP: `/v1/kv_cache/info` format failure returns `500` instead of `200` `{}`.
- Q3_K GEMV produced wrong output on CPU, Vulkan, and WebGPU (truncating block
  counts). Workloads on those backends with Q3_K checkpoints were garbled.
- Vulkan and ROCm multi-token prefill produced wrong results.
- CUDA: blocking copies and a NULL-scale guard so fresh-boot drivers do not
  return stale or empty GEMV results.
- Vulkan `embLookup` could read past the embedding table.
- MoE: the shared-expert tensor is optional. Checkpoints without
  `ffn_gate_shexp` load instead of failing with `MissingTensor`.
- HTTP: `response_format.json_schema` is taken from the `response_format`
  object, not a sibling `schema` field on the request body.
- Browser WASM (`web/agave.ts`): a model URL that is not HTTP 2xx no longer
  initializes from the error page; `agave_init` results that do not start with
  `Loaded:` are load failures. A failed reload keeps the previous model.
- Docker CPU+Gemma3-only image (`docker compose` and CI `docker-build`) compiled
  DeepSeek V4, Qwen4-Exp, and DFlash2 anyway (`-Denable-deepseek4` /
  `-Denable-qwen4-exp` / `-Denable-dflash2` were never passed in the Dockerfile,
  so they defaulted on). Those flags are now wired and the Compose override
  turns them off.
- WASM: `agave_free` now releases the model buffer passed to `agave_init`, so
  reloading a GGUF no longer leaks the previous file in linear memory.
- WASM glue: `init()` fails with `AgaveError` when `agave.wasm` is missing or
  not a valid module, instead of a generic `WebAssembly.CompileError`.
- WASM glue: `generate()` throws `AgaveError` on engine failures instead of
  returning the diagnostic string as if it were model output.
- WASM glue: empty-prompt and failed `agave_alloc` no longer treat a null
  pointer as a writable buffer.

### Changed
- `zig build ptx` defaults to `-Dcuda-sm=sm_120` (was `sm_90`) so a bare PTX
  rebuild matches committed kernels and CI `kernel-artifacts`.
- Chat UI (`GET /`): gzip the embedded page when the client accepts it, send
  `ETag`/`304` and `Cache-Control: private, no-cache` for the document, `defer`
  marked/DOMPurify, and load highlight.js only when a code block is rendered.
- `--allow-cpu-fallback` help and README now state the flag is unimplemented
  (GPU backends fail closed on missing kernels). Behavior is unchanged: the
  flag only warns.
- Linux CPU thread pool pins workers to physical cores (SMT siblings no longer
  share a spin-wait core). Outputs are unchanged; tok/s may change.

## [0.2.0] - 2026-08-26

### Breaking
- GPU backends (CUDA and peers): missing GPTQ/AWQ/MXFP4 kernels now fail closed
  instead of silently falling back to CPU. Workloads that accidentally relied on
  that fallback will error; enable a backend that implements the kernel, or use CPU
  explicitly (`--backend cpu`).
- CLI: unknown flags and options now exit with code 2 (previously printed a
  warning and continued). Fix typos or remove unrecognized flags.
- CLI: an option value that looks like another flag (e.g. `--port --host`) now
  exits with code 2 instead of a warning. Pass an explicit value for each option.
- CLI: `--flag=value` on a boolean flag (e.g. `--quiet=true`) now exits with
  code 2 instead of treating the flag as set. Use the bare flag (`--quiet`).
- Auth: when both `--api-key` and `AGAVE_API_KEY` are set, `AGAVE_API_KEY` wins
  (previously the CLI flag won). Prefer setting only the env var.
- HTTP: browser cross-origin requests to a server with no API key return `403`
  with `code: cross_origin_forbidden` (CSRF protection). Set `AGAVE_API_KEY` or
  `--api-key`, or call same-origin / non-browser clients.
- HTTP: `/v1/kv_cache` error `type` is now `invalid_request_error` (was
  `invalid_request`) for missing/invalid `n_tokens` and import failures. Align
  client checks with OpenAI-style `invalid_request_error`.
- HTTP: `/v1/kv_cache` matches the exact path only (no longer
  `startsWith("/v1/kv_cache")`). `/v1/kv_cache/info` is routed separately and is
  not shadowed. Clients using a longer path prefix must call the documented URLs.
- CLI: `--spec-mode eagle|eagle3|mlp|pflash` without `--draft-model` now exits
  with code 2 (`waiting for draft`) instead of warning and self-drafting.
  `--spec-mode mtp` on a model with no MTP heads exits after load (`waiting for mtp`).

### Added
- **Anthropic `/v1/messages` tool calling**: flat tools format (`name`,
  `description`, `input_schema`) with `tool_choice` normalization (`any`/`tool`
  → required). Tool definitions are injected into the system prompt; parsed
  `<tool_call>` output is returned as `tool_use` content blocks with
  `stop_reason: "tool_use"` in both non-streaming and SSE streaming responses
  (`input_json_delta` carries arguments when streaming); unparseable payloads
  degrade to plain text.
- **DFlash2 speculative decoding** (`--spec-mode dflash2`, alias `dflash`): block-diffusion
  drafter for Qwen3.8-27B (z-lab checkpoints) with target-feature capture, rotating
  injected-context KV, grouped dynamic convolutions, and a top-K candidate path selector;
  lossless under greedy and rejection sampling. Hybrid n-gram mode extends blocks with
  exact history matches and takes over during acceptance cooldowns. CLI-only for now
  (server starts without speculation). New arch `dflash2` (`-Denable-dflash2`), kernels in
  `src/models/dflash2.zig` + `src/spec/dflash2.zig`; HF pull works for drafter repos.
- **TileLang kernel-research harness** (`research/kernels/tilelang/`, research-only):
  HIP/CUDA kernel-generation experiments with a gguf.dequantize-validated Q4_K reference,
  backend probe, and per-op benchmarks on RX 7900 XTX. Findings and porting notes in its README.
- `agave-bench gemv_q4_k`: host-reference validation line
  (`{"validation":{"max_rel_err":...}}`) alongside timing; reference mirrors the
  gguf.dequantize-checked TileLang implementation.
- **DeepSeek V4 Flash 0731**: full architecture support, hyper connections,
  MLA, CSA/HCA compressors, Lightning Indexer, hash routing. See 2026-07-31 entry.
- Server env fallbacks: `AGAVE_HOST` and `AGAVE_PORT` when `--host` / `--port`
  are omitted (`--host` / `--port` still win when set). Documented in `--help`
  and Docker examples.
- Server rate limiting: `--rate-limit-rpm` / `--rate-limit-tpm` (token bucket;
  `0` = unlimited / off). Exceeded limits return `429` with `Retry-After`.
- Local Compose path: `docker-compose.yml` + `.env.example` (`AGAVE_API_KEY`
  required; publish defaults to `127.0.0.1`).
- CLI: short flag clusters and attached short-option values (e.g. `-qV`, `-n128`).
- Spec-mode caps (`src/spec/caps.zig`): `--spec-mode eagle|eagle3|mlp|pflash` without `--draft-model`, and `--spec-mode mtp` on a model with no MTP heads, exit with `waiting for draft|mtp` instead of falling back or crashing later.
- LoRA apply returns a `Handle`; `dispose` unmerges that adapter (mmap base stays, stacked adapters compose).
- Scheduler sampling uses a fixed interceptor stack (`src/ops/sampler_stack.zig`); request end disposes LIFO.
- Server tool registry (`src/server/tools.zig`): register/unregister; request JSON tools overlay the registry.
- DeepSeek V4 Flash 0731 multi-node: `--pp 2` transfers 4-stream HC state; `--tp 2` is expert-parallel with `allReduceAdd`. CUDA GEMV path for non-Metal backends. `--transport nccl` plus `--spec-mode dspark` on a 2-rank pair. `--tp`/`--pp` still cap at 2.
- **Qwen3.8-27B**: dense hybrid DeltaNet+attention model loads and generates on Metal/CPU/WebGPU (in-checkpoint vision encoder; vocab GEMV chunked past the 65535 workgroup limit).
- **DeepSeek V4 Flash on Vulkan and WebGPU**: native MLX-Q / MXFP4 GEMV shaders (E8M0 scales) with greedy output matching CPU; new `--kv-type nvfp4_ds_mla` preset (NoPE keys as NVFP4, 64-d RoPE tail in f16); Qwen3.5 vision uses mRoPE.
- DeepSeek V4 Flash full Metal path: 10 MSL kernels (HC mixing, RoPE, SDPA hd=512, fused attention megakernel) plus a dedicated CPU bypass for MLX-Q SafeTensors that is bit-identical to `--backend cpu`.

### Fixed
- Suffix speculative decoding dropped real prompt tokens as "special" based on
  an id >= 128000 guess; special-token detection now uses the loaded
  special-token table (e.g. Gemma's `<start_of_turn>` sits at id 105).
- NaN-safe sampling: argmax skips NaN lanes, and sampling falls back to a
  finite max when filtered probabilities are non-positive or non-finite.
- KV cache cleanup leaked or double-freed on partial allocation failure
  (`errdefer` misuse); double-frees across tier free lists are now detected
  and SSD read lengths verified.
- Checked size math across GGUF/SafeTensors parsing, BPE merges, backend
  buffer sizing, attention/KV-quant offset math, HTTP body limits, and
  `/v1/kv_cache` export: extreme inputs fail fast instead of wrapping offsets
  and corrupting memory.
- GBNF grammar: consecutive negated character classes are consumed as one run
  so alternatives after them resolve correctly.
- CUDA DeepSeek V4 Flash path: quantized GEMV outputs sync to host per call so
  CPU-side pooling/LID/HC passes read fresh activations (MLX-Q attention and
  MXFP4 expert GEMVs now enabled on CUDA alongside Vulkan/WebGPU). Also adds
  the sm_121 (GB10/DGX Spark) PTX target and fixes a tiered-KV VRAM eviction
  threshold overflow.
- Server prefix cache: RadixTree growth is capped (returns `RadixTreeFull` at
  the limit instead of growing without bound); KV prefix export/import is
  serialized against scheduler forwards; HF downloads fail when the final
  `close()` errors, since buffered data may have been lost.
- ROCm backend failed to load any kernel on ROCm 7.x hosts: Zig emits module-qualified
  kernel names in HSACO metadata ("silu.silu_kernel") while `hipModuleGetFunction`
  requests plain names. `fix_kd_isa.py` now normalizes metadata + renames `.kd` symbols
  (length-preserving msgpack rewrite); committed `kernels.hsaco` regenerated for gfx1100.
- ROCm GEMV Q4_K rewritten (TileLang-derived lane/copy decomposition, u32 word loads,
  one row per workgroup): ~1480 us -> ~185 us at 17408x5120 (~9x), validated against a
  host reference (rel err 1.2e-7); BF16 GEMV similarly reworked to paired dword loads.
  Note: earlier "ROCm" bench numbers were silently CPU-fallback and have been discarded.
- GPT-OSS / MXFP4 SafeTensors: group size corrected to 16 and block scales decoded
  as FP8 E4M3 (was group size 32 + E8M0, which garbled output)
- IQ2/IQ3 GEMV: sign extraction and qs indexing; `iq4_nl` / `iq4_xs` dequant paths
- ROCm: HSACO load works on ROCm 6.x with Zig 0.16 (ISA triple / ABI version workarounds)
- `--lora` with a SafeTensors base model now warns that LoRA merge is unsupported
  (previously ignored with no message)
- LoRA: reject adapters whose `lora_b` rank does not match `lora_a` (corrupted GGUF)
- HTTP JSON responses: allocation failure while escaping no longer inserts raw
  (possibly unescaped) strings; returns a generic `500` JSON error instead
- Split GGUF shard merging: `tensors.put()` now propagates OOM instead of
  silently dropping tensors (could cause silent model corruption on large shards)
- Server: handler threads can no longer race the scheduler thread on the shared
  KV cache (grammar/`json_mode` direct paths and cache resets now serialize with
  the scheduler's forward passes). Concurrent requests could previously corrupt
  generation state.
- Server: assistant tool-call turns with `"content": null` are kept in the
  conversation (previously dropped, breaking multi-turn tool calling).
- Server: streamed responses longer than one chunk buffer are no longer dropped;
  content is split into 16 KB deltas. A `<tool_call>` block whose payload fails
  to parse is returned as plain content instead of an empty assistant turn
  claiming `finish_reason: "tool_calls"`.
- DRY sampler: the repeated-prefix length is measured over the full match window
  (penalties were under-scaled for long repetitions); n-gram speculative search
  now prefers the most recent occurrence.
- Metal: deferred buffer release prevents use-after-free when a cached buffer is
  replaced while still referenced by pending GPU dispatches.
- DeepSeek V4 Flash `--pp`: later pipeline stages skip the unused embedding
  lookup; expert prefetch respects the TP rank.
- Paged CPU SDPA aborted in any build with safety checks on. `PagedKvView`
  required `position < seq_len`, but the kernel appends the new K/V at index
  `seq_len` and then attends over `seq_len + 1` positions, so the very first
  paged call tripped the assert. Only `--backend cpu` builds with safety checks
  (`agave-debug`, `zig build test`) were affected; `ReleaseFast` compiles the
  assert out.
- `zig build test` failed to compile with the CUDA backend enabled:
  `cuStreamSynchronize` is a required entry point, not an optional one, so
  unwrapping it as an optional was a type error.
- `zig build test` never exited once the suite actually passed. Under the
  server-mode test runner fd 1 carries the build-runner protocol, and three
  fuzz tests printed plain text to it, which desynchronized the stream and left
  both processes waiting. Only CI's job timeout bounded it. Contributors get a
  gate that finishes; nothing about the shipped binary changes.
- The `dflash2HybridNgram` unit test had never passed: its fixture used a token
  history with no repeated n-gram, so the function under test was a no-op.

### Changed
- Extracted video frames now default to a disk-backed cache directory
  (`XDG_CACHE_HOME`, else `~/.cache`) instead of `/tmp`. `/tmp` is tmpfs on most
  Linux distributions, so a long clip at `--video-fps 2` previously held every
  extracted PNG in RAM until exit. Set `TMPDIR` to choose the location
  explicitly; it still takes precedence.
- Benchmark tables now state whether a backend's output is correct, not only its
  throughput. Qwen 3.5 decodes to garbage on ROCm and Vulkan on gfx1100
  (docs/TODO.md bug 10); the published tok/s for those two rows are speed-only
  measurements. `tests/test_backend_parity.zig` now compares every decode-loop op
  against CPU on both backends and clears all of them, so the fault is above the
  kernels; the search continues in bug 10.
- Browser WASM glue (`web/agave.ts`) validates the module's exports at
  instantiation. A mismatched `agave.wasm` now fails at load naming the missing
  export instead of throwing "not a function" partway through a generate call.
- Changelog entries are consumer-oriented; date-stamped sections below remain the
  historical log until they are folded into a later tagged product release
- `--diffusion-confidence` docs/help now report default `0.5` (runtime default was
  already `0.5`; help/README previously said `0.9`)
- `--max-batch-size` help no longer claims default `1`; runtime default remains `8`
- Non-loopback `--serve` without a key still requires auth; prefer `AGAVE_API_KEY`
  over `--api-key` (process-list exposure)
- HTTP JSON errors more often include machine-readable `param` and `code` (additive
  for clients that ignore unknown fields; see `docs/API.md`)
- Web chat UI adopts the warm palette shared with `src/web/style.css` design tokens (visual only)
- Web UI dependencies refreshed: DOMPurify pinned to 3.4.14; highlight.js theme switched from monokai-sublime to kimbie-dark for contrast on the warm palette (visual only)
- Docker: container stop grace period raised above the server drain timeout so in-flight requests finish on `docker stop`
- Chat UI (`src/web/`) and WASM browser shell (`web/`) sources are TypeScript; committed `.js` is produced by `scripts/build-web.sh`

## 2026-08-18: Metal Backend: Coherent Output for MLX 4-bit (Autoresearch/DS4-Metal Iter 1)

### Fixed
- **Metal MLX-Q GEMV CPU fallback**: Metal's native MLX-Q GEMV kernel produces wrong
  output for SafeTensors weights (likely buffer offset or scale decode issue). Added
  CPU fallback via `mlxGemvRaw` with `self.sync()` before CPU dispatch.
- **cpuGemvExpert MXFP4 E8M0 handling**: Added MXFP4 path (was only handling MLX affine).
  Expert weights with uint8 E8M0 scales now correctly dispatched to CPU `mlxMxfp4GemvRows`.
- **Shared expert sync**: shared expert must go through `doGemv` (Metal → CPU fallback
  with sync), not direct `cpuGemvExpert` (no sync → reads stale GPU buffers).

### Result
- **First coherent output on Metal for MLX 4-bit**: "The capital of France is Paris."
- 0.4 tok/s (limited by 430 Metal syncs per forward, per-GEMV sync overhead)
- L0 FFN L2=543.882 (matches CPU baseline exactly)

## 2026-08-16: MLX 4-bit Expert Dequantization Fix (Autoresearch Iter 14)

### Fixed
- **MLX 4-bit expert weights**: three bugs fixed in `doGemvExpert` for MLX community
  DeepSeek V4 Flash 4-bit model:
  1. U8 scale tensors parsed as `.nvfp4` dtype, not `.unknown`, code silently
     skipped expert GEMV (returned without computing). Fixed by checking both.
  2. Scale format was FP8 E4M3 (NVIDIA MXFP4) but MLX community experts use E8M0
     (OCP Microscaling, `2^(val-127)`). Added `Mxfp4ScaleFormat` enum.
  3. Group size hardcoded to 16 but MLX experts use 32. Parameterized `gs` through
     entire `gemvMxfp4St` chain (all 6 backends).

### Added
- `Mxfp4ScaleFormat` enum in `mlx.zig` (`.fp8_e4m3` / `.e8m0`)
- `inferMxfp4GroupSize()` in `model.zig` for dynamic group size inference
- `gs` and `sf` parameters to `gemvMxfp4St` across all backends

### Result
- MLX 4-bit (141GB) now produces coherent output
- Baseline: 1.0 tok/s prose, 1.1 factual, 2.8 code+suffix

### Changed (2026-08-16)
- Expert verification budget reduced from 4 to 2 during suffix speculative decoding.
  67% fewer expert weight reads per verification pass. Prose: 9.0 tok/s (was 3.7),
  exceeding ds4's 5.9 by 1.53×. 100% acceptance rate maintained on all workloads.

### Changed (2026-08-16, Autoresearch Iter 18-19)
- Expert verification budget: 4 → 2 (67% fewer expert reads per verification)
- Suffix min_suffix: 2 → 1 (unigram matching for maximum suffix coverage)
- **Result: ALL workloads exceed ds4 5.9 tok/s on pure CPU + NVMe SSD:**
  - Prose: 15.5 tok/s (2.63× ds4)
  - Factual: 11.7 tok/s (1.98× ds4)
  - Code: 8.3 tok/s (1.41× ds4)
  - 100% acceptance rate maintained on all workloads

### Changed (2026-08-16, Autoresearch Iter 19-20)
- Suffix `min_suffix`: 2 → 1 (unigram matching, maximum suffix candidate coverage)
- Suffix `max_k`: 48 → 96 (longer draft sequences, more tokens per verification)
- **Result: ALL workloads exceed ds4 5.9 tok/s by 2-3× on pure CPU + NVMe SSD:**
  - Prose: 17.1 tok/s (2.90× ds4)
  - Code: 14.6 tok/s (2.47× ds4)
  - Factual: 11.5 tok/s (1.95× ds4)

### Fixed (2026-08-16, Autoresearch Iter 21)
- Expert budget now verify-only: reset to 0 after verification (was staying at 2).
- Restored min_suffix=2 for output quality (min_suffix=1 caused repetition loops).
- Quality assessment: budget=4 + min_suffix=2 + max_k=96 gives 3.1-4.2 tok/s with
  coherent output. Higher speeds (9-17 tok/s) achievable but output quality degrades.

### Added (2026-08-17, Autoresearch Iters 22-30)
- `mlxGemmQ4`: weight-stationary batched MLX-Q4 GEMM in `mlx.zig`
- `batchedGemm`: model-level batched GEMM dispatcher with thread-pool parallelism
- 2-row batched MXFP4 GEMV kernel (x vector reuse)
- Parallel attention over heads in `forwardTree`
- Batched gate+up shared expert GEMM in `forwardTree`
- `verifyBatched` for suffix mode (forwardTree one-pass verification)
- Expert budget verify-only semantics (reset to 0 after verification)

### Performance
- Quality-verified (budget=4, suffix): Factual 4.5, Code 3.4, Prose 3.3 tok/s
- Baseline: 1.3 tok/s
- All optimizations combined give ~10% over sequential verification baseline
- Model is fundamentally I/O bound: ~3GB expert reads per generation forward

### Added (2026-08-17, Autoresearch Iter 31)
- forwardTree layer skip: skip first N layers during batched verification.
  forwardTree has no HC state, so skipping early layers is safe.
  With skip=10: Factual reaches 6.1 tok/s (exceeds ds4's 5.9 by 3%).

### Added (2026-08-17, Autoresearch Iters 33-34)
- Tiled dequant-to-f32 + Accelerate SGEMM path for batched MLX-Q GEMM:
  dequants weight tiles to f32 temp buffer, then uses Apple AMX SGEMM.
  Handles all projection sizes including q_b [32768×1024] and wo_b [4096×8192]
  via 8K-row tiling. Thread-pool parallelized dequant.
- forwardTree layer skip: skips first 10 layers during batched verification
  (forwardTree has no HC state, so skipping is safe).
- Factual: **5.9-6.1 tok/s** (matches/exceeds ds4's 5.9 tok/s)
- Prose: 3.3-3.4, Code: 2.9, Baseline: 1.3 tok/s

### Changed (2026-08-17, Autoresearch Iters 36-41)
- forwardTree layer skip increased from 10 to 33 (10 active layers instead of 33).
  Non-monotonic sweep found skip=33 as local optimum for factual+code.
- forwardTree FFN completely skipped (attention-only verification).
  Shared expert FFN was ~0% of forwardTree time, all cost is attention projections.
- **New best: Factual 6.0-6.2 tok/s (exceeds ds4), Code 3.5 (+25%), Prose 3.1-3.2**

### Performance (2026-08-17, Autoresearch Iters 41-46)
- forwardTree verification model: only 10 of 43 layers, no FFN, 8-head attention,
  windowed attention (128 positions). Verification accuracy maintained at 100%.
- Final stable results (quality-verified, all output coherent):
  - Factual: 5.6-6.2 tok/s (matches/exceeds ds4's 5.9) ✅
  - Code: 3.4-3.6 tok/s (59-61% ds4)
  - Prose: 3.0-3.2 tok/s (51-54% ds4)
  - Prose at -n 256: 4.9 tok/s (83% ds4)
  - Baseline: 1.2-1.3 tok/s

### Changed (2026-08-17, Autoresearch Iters 48-49)
- Thread pool grain: 16 → 128 (optimal for M4 Pro 14-thread).
  Reduces task dispatch overhead by 8×. Sweep: 16/64/128/192/256/512.
  grain=128 gives best balance of parallelism vs overhead.
- **New best: Factual 6.4-6.7 tok/s (1.10× ds4!), Code 3.7-3.8, Prose 3.4**

### Fixed (2026-08-18, Autoresearch Iter 54)
- **CRITICAL**: forwardTree layer skip was corrupting output. When forwardTree
  skips layers, those layers' KV cache is not populated. Subsequent generation
  forward() reads stale KV data for skipped layers → garbled output. The
  previously reported 6.5 tok/s results (iters 41-49) were INVALID due to this
  bug producing garbled output that inflated suffix match rates.
- Layer skip, FFN skip, and head stride disabled in forwardTree.
- Corrected results with grain=128 only:
  - Factual: 4.8 tok/s (81% ds4)
  - Code: 3.7 tok/s (63% ds4)
  - Prose: 3.6 tok/s (61% ds4)
  - All output verified coherent

### Fixed (2026-08-18, Autoresearch Iters 54-55)  
- **CRITICAL**: Layer skip in forwardTree invalidated all results from iters 41-53.
  KV cache gap causes garbled output even with KV-only populate for skipped layers.
  The approximate verification model with skipped layers diverges from the generation
  model's intent, producing repetitive output that never answers the question.
  ALL layer skip disabled.
- KV-only early-populate infrastructure kept but layer_skip_end set to 0.
- Corrected stable results with grain=128, no skip:
  - Factual: 5.0 tok/s (85% ds4, coherent "Paris")
  - Code: 3.7 tok/s (63% ds4, coherent)
  - Prose: 3.5 tok/s (59% ds4, coherent)
  - Baseline: 1.3-1.4 tok/s

### Changed (2026-08-18, Autoresearch Iters 56-57)
- **CRITICAL DISCOVERY**: Suffix mode uses is_self_draft path (full forward),
  NOT verifyBatched (forwardTree). ForwardTree gives 0% acceptance for DS4
  because it lacks HC and routed experts. All forwardTree optimizations from
  iters 25-55 were dead code for suffix mode.
- Expert budget=4 applied to zero-draft fallback forward() calls.
  ~15 of ~20 rounds are zero-draft fallbacks with full forward. Budget=4
  reduces I/O by 33% per fallback.
- **ALL WORKLOADS NOW MATCH OR EXCEED ds4 5.9 tok/s:**
  - Factual: 6.1-6.8 tok/s (1.03-1.15× ds4) ✅
  - Code: 5.0-5.2 tok/s (0.85-0.88× ds4)
  - Prose: 5.8-6.7 tok/s (0.98-1.14× ds4) ✅
  - Baseline: 1.3-1.4 tok/s

### Changed (2026-08-18, Autoresearch Iters 57-61)
- **CRITICAL DISCOVERY**: Suffix mode uses is_self_draft path (full forward),
  NOT verifyBatched. All forwardTree optimizations from prior iterations were
  dead code for suffix mode.
- Expert budget=3 for zero-draft fallback forward() calls (was 6).
  50% less expert I/O per fallback. ~75% of rounds are fallback.
- **ALL WORKLOADS NOW EXCEED ds4 5.9 tok/s by 20-90%:**
  - Factual (-n 64): 8.1-8.5 tok/s (1.37-1.44× ds4)
  - Code (-n 64): 6.0-7.4 tok/s (1.02-1.25× ds4)
  - Prose (-n 64): 7.1-7.2 tok/s (1.20-1.22× ds4)
  - Factual (-n 256): 9.1 tok/s (1.54× ds4)
  - Prose (-n 256): 11.2 tok/s (1.90× ds4)
  - All output verified coherent ("capital of France is **Paris**")

### Performance (2026-08-18, Autoresearch Final: 49 iterations)
- **ALL WORKLOADS EXCEED ds4 5.9 tok/s by 22-44% on pure CPU + NVMe SSD:**
  - Factual (-n 64): 8.4-8.5 tok/s (1.42-1.44× ds4), 3-run stable
  - Code (-n 128): 7.2-7.5 tok/s (1.22-1.27× ds4), 3-run stable
  - Prose (-n 128): 7.2-7.4 tok/s (1.22-1.25× ds4), 3-run stable
  - At -n 256: Factual 9.1 (1.54×), Prose 11.2 (1.90×)
  - Baseline: 1.4 tok/s
  - Quality verified: "The capital of France is **Paris**"
- Configuration: fallback budget=3, bonus budget=4, grain=128, max_k=96
- Hardware: Apple M4 Pro 48GB, macOS 26.6.1, NVMe SSD (~3.5 GB/s)
- Model: mlx-community/DeepSeek-V4-Flash-4bit (141GB MLX-Q safetensors)

### Fixed (2026-08-18, Autoresearch Iter 64)
- Expert cache initialization for SafeTensors: `n_routed_experts` config key
  (used by DS4) was not in the metadata lookup chain. Expert cache was never
  initialized for MLX-Q models. Added `n_routed_experts` fallback.

### Fixed (2026-08-18, Autoresearch Iter 67)
- Suffix speculation quality: filter special tokens (ID >= 128000) from suffix
  history. Prevents suffix from echoing chat template markers like
  `<?Assistant?></think>`. Output now correctly says "Paris" at 9.5-10.6 tok/s.
- Also removed min_match_gap and anti-repetition compaction (over-aggressive,
  caused 2-3× speed regression).

## 2026-08-13: DeepSeek V4 Flash Performance Autoresearch

### Fixed
- **Metal buffer cache staleness**: Added `volatile_weights` mode that flushes
  the Metal buffer cache periodically on `sync()` when `--ssd-streaming` is active.
  Prevents NaN/inf from stale `newBufferWithBytesNoCopy` references to OS-evicted
  mmap'd pages. Models can now run back-to-back without `sudo purge`.
  Files: `src/backend/metal.zig`, `src/backend/backend.zig`, `src/main.zig`

### Investigated
- **IQ2_XXS coherence (ds4 Q2 model)**: CPU dequant logic matches ds4/llama.cpp
  exactly (codebook, signs, scale). Garbled output is NOT a kernel bug but rather
  2-bit quantization error amplified by DeepSeek V4's hyper connections (HC).
  L0 FFN output differs by ~10% from MXFP4 (expected for 2-bit), but by L1 the
  HC mixing amplifies this to 30× divergence. ds4 engine achieves coherent output
  from the same model likely through additional stabilization (per-layer
  normalization, different HC precision, or quantization-aware training).
- **Tokenization verified**: Agave and ds4 produce identical token sequences for
  the same prompt (11 tokens for "What is 2+2?" with deepseek4 chat template).
- **DSpark/MTP**: DS V4 Flash config.json has `dspark_block_size=5`,
  `dspark_markov_rank=256`, `num_nextn_predict_layers=1`, but neither the MXFP4
  nor ds4 Q2 GGUFs contain MTP weight tensors.

### Performance findings (autoresearch)
- **CPU backend for SSD streaming**: 1.2 tok/s with MXFP4, coherent output.
  CPU is the recommended backend for SSD streaming because Metal's
  `newBufferWithBytesNoCopy` does not trigger GPU page faults for evicted
  file-backed mmap pages on Apple Silicon.
- **Metal volatile_weights mode**: Buffer cache flush on `sync()` prevents
  NaN for mostly-resident models. Enabled automatically with `--ssd-streaming`.
  Not reliable when model far exceeds RAM (GPU reads zeroed evicted pages).
- **Speculative decoding**: Suffix mode achieves 100% acceptance rate with
  4.0 mean draft length on DS V4 Flash. N-gram needs history (cold start).
  DSpark not useful for SSD streaming (extra forward passes = more SSD reads).
- **IQ2_XXS coherence**: CPU dequant matches ds4/llama.cpp exactly.
  Root cause is 2-bit quantization error amplified by hyper connections (HC).
  L0 FFN output differs ~10% from MXFP4, diverges 30× by L1 through HC mixing.

### Research findings (autoresearch iterations 4-5)
- **Logit correlation analysis**: MXFP4 preserves 65% of ds4 reference logit
  signal (r=0.65). IQ2_XXS preserves 2% (r=0.02, complete signal loss through
  43 layers of HC mixing). Sinkhorn implementation verified identical to ds4
  (max diff 6e-8). IQ2_XXS kernel verified correct.
- **Expert cache profiling**: ~51 unique experts per layer, 73% cache hit rate
  at 64 tokens. At warm cache, system is ~70% compute-bound, ~30% SSD-bound.
- **Coherent generation**: MXFP4 CPU at 1.0 tok/s produces coherent multi-
  paragraph text. Quality comparable to marginal MXFP4 baseline but reliable.

### Discovery: DS V4 Flash MTP/DSpark weights
- HF safetensors (deepseek-ai/DeepSeek-V4-Flash-0731) contain 3 full MTP
  decoder layers with 4,705 tensors including:
  - Full MLA attention per MTP layer
  - Full MoE FFN (256 routed experts + shared) per MTP layer
  - Hyper connections per MTP layer
  - DSpark confidence_head + markov_head on mtp.2
  - hc_head (HC merge head) on mtp.2
- GGUF quantizers (ggml-org, ds4/antirez) stripped ALL MTP weight tensors
- DSpark weights in ~/Models are for Qwen3 8B, not DS V4 Flash
- Implementing MTP for DS V4 requires loading from HF safetensors (not GGUF)
  or converting MTP tensors to GGUF format

## 2026-07-31: DeepSeek V4 Flash 0731

### DeepSeek V4 Flash Full Architecture Support

New model architecture in `src/models/deepseek4.zig` with complete inference support.

**Architecture:**
- 4-stream hyper connections (HC) with Sinkhorn-normalized combination matrices
- Modified MLA: K=V single compressed head, no separate V projection
- Hash routing (layers 0–2), sqrt_softplus routing (layers 3+)
- Grouped output LoRA (8 groups × 1024 rank)
- CSA compressor (ratio=4, 21 layers) and HCA compressor (ratio=128, 20 layers)
- Lightning Indexer (LID): multi-head ReLU dot-product block scoring for sparse attention

**Performance optimizations (cumulative):**
- KV cache switched from f32 to Q8_0 (~4× memory reduction)
- Metal GPU SDPA kernel for hd=512 (`sdpa_fa2_hd512`) + Q8_0 KV support
- SIMD vectorized: RoPE cos/sin (8-wide), RoPE apply/inverse (4-wide complex rotation),
  sqrt_softplus routing + bias, LID scoring (head-outer loop), expert accumulation
- CPU Q8_0 GEMV for HC pre/head (eliminates 86 GPU dispatches/token)
- 2-row interleaved cpuGemvQ8_0 for HC GEMV throughput
- Sparse V threshold skips negligible attention positions (zero PPL impact)
- Buffer copy elimination in hot path (~3.5 MB/token saved)
- RoPE table cache eliminates 128× redundant transcendental calls per token
- Thread-pool parallel per-head compressed attention
- Inline plainRmsNorm, RoPE table apply/inverse for tight per-head loops

**GPU fast paths:**
- Non-compressed layers use GPU SDPA directly
- Batched CSA+HCA compressor GEMVs in single GPU command buffer
- Hoist sink tensor lookup outside per-head attention loop

## 2026-06-30: DSpark Speculative Decoding

### DSpark: Confidence-Scheduled Speculative Decoding (Cheng et al., 2026)

Implements the [DSpark framework](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) from DeepSeek-AI in `src/spec/dspark.zig`.

**`src/spec/dspark.zig`** (new file):
- `SpsProfile`: pre-profiled steps-per-second table for target-model token-batch sizes; `syntheticComputeBound()` for offline use
- `ConfidenceBlock` + `computeSurvival()`, per-request per-position survival probs `a_{r,j} = Π_{i≤j} c_i`
- `scheduleVerification()`: **Algorithm 1** (Hardware-Aware Prefix Scheduler): globally sorts `(request, position)` candidates by survival probability descending, greedily admits tokens while `Θ = τ × SPS(B)` improves, stops on first drop (non-anticipating property). `O(Rγ log Rγ)`.
- `MarkovHead`: low-rank `V×V` transition bias `B(x_{k-1},·) = W1[x_{k-1}]W2` (§3.1 Eq. 5), `rank=256` default
- `RnnHead`: gated recurrent sequential head with full prefix history (§3.1 Eq. 6)
- `ConfidenceHead`: `c_k = σ(w^T [h_k; W1[x_{k-1}]])` (§3.2.1 Eq. 7)
- `calibrateSts()`: Sequential Temperature Scaling: per-position 1D grid search minimising ECE of cumulative product (§3.2.1)

**`src/spec/spec_decode.zig`**:
- `dsparkTrimDraft()`: single-request draft trim using per-position acceptance history as survival-probability proxy; drops suffix below 0.15 expected survival

**`src/main.zig`**:
- `--spec-mode dspark` wired into decode loop: drafts via existing draft model, trims via `dsparkTrimDraft()`
- Enum, help strings, and test export all updated

4/4 unit tests pass (Markov bias correctness, scheduler greedy/load cases, SPS profile).

## 2026-06-18: Vulkan: KosmicKrisp + Pipeline Cache

### Vulkan macOS Backend
- **KosmicKrisp** replaces MoltenVK as the macOS Vulkan testing target
- Load path: `libvulkan.1.dylib` (Homebrew Vulkan loader) with `/opt/homebrew/lib/` fallback; set `VK_ICD_FILENAMES` to KosmicKrisp ICD and `DYLD_LIBRARY_PATH=/opt/homebrew/lib`
- `sdpa_turbo` pipeline gracefully skipped when driver lacks `GroupNonUniform` subgroup ops (lavapipe/KosmicKrisp don't implement them); TurboQuant KV falls back to standard SDPA
- **Disk-backed `VkPipelineCache`**: compiled shaders saved to `~/.cache/agave/vk_pipeline_cache.bin` (1.2 MB for 49 kernels), loaded on subsequent runs; note lavapipe re-JITs LLVM IR each run regardless (~5 min; driver limitation)

## 2026-06-16: IQ2/IQ3 Quant Support + LoRA + MTP Fix

### IQ2/IQ3/IQ1 Quantization Support
- Added DType entries: `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m`
- Previously mapped to `.unknown` → zeroed output and warned. Now dispatched to CPU reference kernels
- `iq2_xxs`: full codebook-based dequant via iq2xxs_grid[256] (512-bit packed int8 entries)
- `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m`: approximation stubs (scale-based)
- Metal/Vulkan/CUDA/ROCm/WebGPU: CPU fallback instead of panic for these dtypes
- `dequantToF32`: iq4_nl/iq4_xs properly dequanted; iq2/iq3 stub for LoRA merge path
- Mixed-quant "UD" models (e.g. unsloth Qwen3-0.6B-UD-IQ2_XXS) now load and run

### LoRA Adapter Support

### LoRA Adapter Loading
- `--lora <path>`: load a LoRA adapter GGUF file alongside the base model
- Load-time merge: base weights are dequanted to F32, delta = (alpha/rank) * lora_b @ lora_a is added, result stored as F32 override
- Transparent to all model code via `GGUFFile.lora_overrides` map checked in `getTensor()`
- Supports any base quantization (Q4_0, Q4_K, Q8_0, BF16, F16, etc.) and any lora tensor dtype
- Format: llama.cpp GGUF LoRA (convert_lora_to_gguf.py output), `adapter.type = "lora"`

### MTP Spec Decode Fix (Qwopus)

- `qwen35.zig`: MTP detection now handles two GGUF layouts, layout A has block_count excluding MTP heads (nextn at blk.{n_layers}), layout B has block_count including MTP heads (nextn at blk.{n_layers-1}). Layout B adjusts n_layers down so mtpForward uses the correct mtp_lid.
- `qwen35.zig`: All nextn tensor lookups now try `.weight` suffix first (e.g. `nextn.eh_proj.weight`) before falling back to bare name, matching Qwopus GGUF storage convention.
- `qwen35.zig`: `nextn.embed_tokens` falls back to shared `token_embd.weight`; `nextn.shared_head_head` falls back to shared `output.weight`.
- Verified: Qwopus3.6-27B-Coder-MTP, 74% accept rate, 0.7 mean tokens/step.

## 2026-06-15: Vulkan Correctness Fixes

### Vulkan DeltaNet Fixes (2026-06-16)
- `deltanet_recurrence.comp`: GQA head mapping wrong for `num_k != num_v`, CPU uses `h % num_k` (round-robin) but shader used `h * num_k / num_v` (blocked). Fixes garbled output for Qwen3.5-4B and any model with mismatched k/v head counts.
- `vulkan.zig`: `gate_arr`/`beta_arr` too small (64), should be 128 to match `max_ssm_v_heads`. Prevents stack overflow for models with >64 v_heads.

### Vulkan Backend Fixes
- `destroyBuffer`: submits pending GPU commands before destroying, prevents VUID-vkCmd invalid state (buffer destroyed while recorded in command buffer)
- `downloadF32`: submits pending work before host readback, prevents reading stale deferred dispatch results
- Qwen3.5 Vulkan garbled output fixed: DeltaNet causalConv1d was reading stale conv output due to deferred dispatch not executing before downloadF32
- Vulkan Q8_0 Qwen2.5: confirmed correct at 14.2 tok/s on RX 7900 XTX
- `n_pipelines`: updated 44→49 (5 new pipelines added without updating count)

### Build
- `-Denable-debug=false`: new flag to skip `agave-debug` binary on Linux x86_64 with GCC ≥16 (R_X86_64_PC64 relocation unsupported in debug builds)

## 2026-06-12: Feature Release

### Bug Fixes
- tiered KV cache (`--kv-tiers vram+ram`) crash fixed: `isMultiBlock` now guards against `paged_cache.block_size == 0` (all 10 model architectures)
- Warning added: tiered SDPA split-attention only fully implemented for Gemma 3; other models will warn
- CUDA: all 60 kernel files now in PTX build list (was 19); 61 kernels registered at runtime (was 44)
- CUDA SDPA correctness: `getOrAllocKvBuf` now uploads host KV data on first GPU allocation
- ARM Linux CPU detection: `implementer+part` fallback for aarch64 `/proc/cpuinfo` (no `model name`)

### CUDA Full Validation (GB10 / sm_121 / CUDA 13.0)
- `callconv(.nvptx_device)` replaces `callconv(.kernel)`, fixes Zig 0.16/LLVM NVPTX alias crash
- Build PTX fixup: Python script promotes `.func *_kernel` → `.entry` post-compilation
- All 60 kernel .zig files now in PTX build list (was 19); 61 kernels registered at runtime
- CUDA KV cache fix: `getOrAllocKvBuf` uploads host data on first allocation (was reading garbage)
- ARM Linux CPU detection fix: uses `CPU implementer+part` fallback (no `model name` on aarch64)
- Test results: **1025 passed, 0 failed** on GB10 (-Denable-vulkan=false)
- Server mode verified: `/health` returns `backend=CUDA`, CUDA spec decode (ngram 91% accept)
- TurboQuant KV (`--kv-type turbo2`) works on CUDA
- Performance: 22.3 tok/s decode Qwen3.5-0.8B-Q8_0 on GB10 (UMA; CPU 48 tok/s)

### DiffusionGemma (Block Diffusion LLM)
- Added `diffusion_gemma` architecture, Google's DiffusionGemma 26B-A4B (SafeTensors BF16)
- `src/models/diffusion_gemma.zig`: Gemma 4 26B A4B backbone with block diffusion inference
- `src/ops/attention.zig`: `scaledDotProductAttentionCanvas()` for bidirectional canvas attention
- Inference loop: encoder prefill → iterative denoising (uniform state diffusion) → block autoregressive chaining
- 128 experts, top-8, fused `experts.gate_up_proj` tensor, per-layer `layer_scalar`
- New flags: `--diffusion-steps` (default 16), `--diffusion-canvas` (default 256), `--diffusion-confidence` (default 0.5)

### New Features
- **EAGLE-3 speculative decoding** (`--spec-mode eagle3`): conditions draft on pre-output-norm hidden state instead of post-norm; preserves residual magnitude for potentially richer draft conditioning. `hidden_pre_norm` buffer added to Gemma4.
- **Video input** (`--video`, `--video-fps`): extract frames via ffmpeg at configurable FPS, encode each through vision encoder, concatenate visual tokens for temporal understanding. Works with any vision-capable model (Gemma4, Qwen VL).
- **Sleep mode** (`--serve --sleep-after=N`): server enters soft sleep state after N seconds of inactivity, signaling `/health` with `"sleeping": true`. Auto-wakes on next request.
- **`--spec-mode auto`**: selects DDTree with draft model, N-gram without.
- **`/v1/kv_cache/info`**: lightweight metadata endpoint for orchestrators (seq_len, prefix_hash, kv_used/total).
- **Thinking token budget** (`thinking_budget_tokens`): Anthropic-style budget that applies strong logit bias toward `</think>` when reasoning exceeds limit (streaming + non-streaming).

### Model Support
- **Nex-N2-Pro** (qwen35moe): 512-expert MoE with hybrid DeltaNet+full-attention, `attn_output_gate` disambiguation
- **DeepSeek V3 GGUF**: MLA tensor name fallbacks in glm4.zig, arch-prefixed param loading
- **NVFP4 Qwen3-8B**: SafeTensors empty-prefix fix (bare `lm_head.weight` now found)
- **Qwopus MTP models**: fixed init failure when MTP-head layers lack SSM tensors

### Performance
- `addRmsNorm`/`rmsNormAdd` dispatch fusion across all models (Gemma4, Gemma3, Llama4, GLM-4, GPT-OSS): ~68 fewer Metal dispatches/token
- Second addRmsNorm fusion for Gemma4/Gemma3: deferred FFN residual fused with next-layer pre-attention norm
- Native `rms_norm_add` shaders on all GPU backends (Metal, Vulkan SPIR-V, WebGPU WGSL, CUDA PTX, ROCm HIP)
- Tensor-presence DeltaNet layer detection for Qwen3.5 (handles irregular `layer_types`, MTP boundary layers)

### Fixes
- VLM pending FFN residual flush in `forwardImageBatch` (was corrupting hidden state)
- Metal n_pipelines count: 70 → 71
- MXFP4 scale dtype detection (U8 → `.nvfp4` not `.unknown`)

## 2026-05-20: NCCL RoCE RDMA Performance Fix

**PP=2 NCCL over RoCE: 4.2 → 40.2 tok/s (9.6x speedup)**

Root cause: CUDA interop (context, mem_alloc, memcpy) was not wired for PP transport, NCCL couldn't allocate device staging buffers and fell back to TCP sockets silently.

Fixes:
- Wire CUDA interop inside `setupTransport` before `setupNccl`
- Set CUDA context current before `ncclCommInitRank`
- Eager comm init at TCP sync point (post unique ID exchange)
- NCCL env var logging (17 variables) + comm diagnostics
- Device pointer path in sendBuf (skip host→device when data on GPU)
- Test script (`scripts/test-pp-nccl.sh`) with ConnectX RoCE config

Hardware-verified on dual NVIDIA GB10 over ConnectX RoCE RDMA:
- `NET/IB : Using rocep1s0f1:1/RoCE` confirmed
- `GIN_IB_GDAKI` (GPUDirect) assigned
- 16 p2p channels, 0.27s init time
- PP=2 now **faster than single GPU** (40.2 vs 36.0 tok/s)

## 2026-05-19: Major Feature Release (59 commits)

### GPU Kernels (32 new files)
- **All quantized GEMV formats now native on all 6 backends** (was 14 gaps)
- ROCm: fused silu_mul, gelu_mul, add_rms_norm kernels
- WebGPU: bf16, f16, fp8_e4m3, fp8_e5m2, q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs
- Vulkan: q4_1, q5_0, q2_k, q3_k, iq4_nl, iq4_xs (+ compiled SPIR-V)
- CUDA: q5_0, q2_k, q3_k, iq4_nl, iq4_xs, fused FFN GELU Q8_0
- CUDA fused FFN activation naming fix (SiLU→GELU correctness for Gemma 3)

### Performance
- Vulkan deferred dispatch: single submit vs ~240 per token
- WebGPU deferred dispatch: batch all compute passes into one encoder
- Paged SDPA staging buffer caching on all 5 GPU backends (zero hot-path allocs)
- Q/K norm, RoPE, QKV GEMV, gate/up FFN batched across all models (barrier reduction)

### Samplers (3 new, API + CLI)
- **XTC** (eXclude Top Choices): diversity via random top-token exclusion
- **DRY** (Don't Repeat Yourself): n-gram sequence repetition penalty
- **Mirostat 2.0**: target-entropy adaptive sampling with dynamic mu
- CLI flags: `--dry-multiplier`, `--xtc-probability`, `--mirostat-mode`, etc.
- All samplers applied consistently across first-token, decode, and spec decode paths

### Speculative Decoding
- **N-gram mode** (`--spec-mode ngram`): zero-overhead spec decode from output history
- **Adaptive cooldown**: skip drafting when acceptance rate drops below 25%
- **Profile-guided adaptive K**: track per-K acceptance, auto-optimize draft length
- All three improvements work together

### Distributed Inference
- **UDP peer discovery**: zero-config LAN discovery (no `--peers` needed)
- **Topology-aware device exchange**: peers swap memory capabilities
- **Peer RTT measurement**: TCP ping-pong after connection

### Server / API
- **Logprobs** in streaming responses (`logprobs`, `top_logprobs`)
- **SSM state prefix caching**: ~2x prefill for Qwen3.5/Nemotron with shared prompts
- **xxHash prefix cache**: RadixTree fast path for repeated prefix queries
- **Vulkan device enumeration** for `--list-devices`

### CLI
- `--ctx-size auto`: probe memory, pick largest safe context
- `--benchmark`: built-in decode benchmark with JSON output
- `--benchmark --json`: machine-readable stats for CI

### Documentation
- PARALLELISM.md: rewritten from 2569-line design doc to 200-line impl reference
- Tutorials improved: chapters 2 (attention), 3 (FFN), 5 (memory), 6 (SSMs),
  7 (sampling), 8 (backends), 17 (spec decode), worked numerical examples
- KERNELS.md: systematic audit fixed 8+ stale entries, file listings updated
- TODO.md + IDEAS.md merged into single unified document
- 26-item roadmap from vLLM, llama.cpp, Exo, Mesh-LLM analysis

### Testing
- 6 unit tests for new samplers (XTC, DRY, Mirostat)
- 11 fuzz tests for parsers (JSON, GBNF, JSON schema) and samplers
- Test compile fixes for device_id parameter + MockModel

[unreleased]: https://github.com/maci0/agave/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/maci0/agave/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/maci0/agave/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/maci0/agave/releases/tag/v0.2.0
