# Durability and Recovery

What Agave keeps on disk, what survives losing it, and how to get it back.
Written against the code and config in this tree; where a guarantee depends on
something outside the repository, that is said plainly.

## State inventory

Everything durable lives under the cache directory: `$XDG_CACHE_HOME/agave/`,
else `$HOME/.cache/agave/`. In the compose image that is
`/home/agave/.cache/agave/`, inside the `agave-cache` volume mounted at
`/home/agave/.cache`.

| State | Path | Regenerable | Consequence of loss |
|---|---|---|---|
| Conversation store | `<cache>/agave/conversations.json` | no | Web-UI conversations gone for good |
| Quarantined store | `<cache>/agave/conversations.json.corrupt` | no | Only remaining copy of a store the server could not parse |
| Hub model blobs | `<cache>/huggingface/` | yes, `agave pull` re-downloads | Bandwidth and time only |
| Vulkan pipeline cache | `<cache>/agave/vk_pipeline_cache.bin` | yes, rebuilt on first run | One slower startup |
| Expert profile | caller-supplied path | yes | Profile re-recorded |

Only the first two cannot be rebuilt. Everything else is a cache with a
rebuild path, so this document is about the conversation store.

Writes for all of them go through `src/durable_file.zig`: write a sibling
`*.tmp`, `fsync`, `rename` over the live path, `fsync` the parent directory. A
crash mid-write leaves the previous file intact, and a file at the live path is
always complete. That property is what makes a plain file copy a consistent
backup, and it is pinned by the tests in `src/durable_file.zig`.

Other verified properties, so a future pass leaves them alone:

- Hub downloads `fsync` the blob before advertising it complete
  (`src/pull.zig`), verify the byte count, and delete a blob that fails the
  GGUF magic check. A killed download resumes rather than advertising a
  truncated model.
- A conversation store that fails to parse is quarantined by rename (or, if
  the rename fails, by writing a copy) to `{path}.corrupt` before the live path
  is reused, so a bad parse never destroys the only copy
  (`src/server/conv_store.zig`).
- Any load failure other than corrupt or unsupported-version disables
  persistence for that run instead of overwriting the file with an empty list
  (`src/server/server.zig`, `loadConversationsLocked`).
- `--conv-store PATH` points the store anywhere; `--no-conv-store` keeps
  conversations in memory only.

## RPO and RTO

| Disaster | RPO | RTO | Notes |
|---|---|---|---|
| Host or instance loss, no backup | the whole store | n/a | Total loss of conversations |
| Host or instance loss, backups current | last server save | seconds to restore a small file | Store is capped at 64 MiB on load |
| `docker compose down -v` | the whole store | n/a | Deletes the volume |
| Malicious or accidental deletion | last backup taken | same | Only if backups were taken |
| Bad deploy | none expected | n/a | The on-disk envelope is version 1 and validated on load; an unreadable version is quarantined, not silently reinterpreted |

RPO is "last server save", not "last token": the server persists on conversation
mutations, so a crash mid-generation loses at most the tokens of the turn in
flight. Schedule the backup at whatever interval matches how much an in-flight
turn costs.

## Back up

```bash
scripts/conv-store-backup.sh path      # where the store is
scripts/conv-store-backup.sh backup    # copy + verify + prune old
```

`backup` resolves the path the same way the server does, copies through a
temporary file and renames (so a killed backup never leaves a partial file
that a later restore would install), verifies the copy, also copies a
`.corrupt` store when one exists, and prunes to `AGAVE_KEEP` (default 14)
backups, newest first, never below one. It exits nonzero on every failure; it
has no quiet failure mode.

Destination is `AGAVE_BACKUP_DIR`, default `$HOME/.agave-backups`. **Set it to
a different filesystem than the cache directory.** A backup on the same disk
protects against a bad save, not against losing the disk, which is the case
that matters.

Put it on a schedule (cron, systemd timer, whatever the host runs). The script
is idempotent and safe to run while the server is serving.

For the compose deployment the store is inside the `agave-cache` volume. The
image ships this script at `/usr/local/bin/conv-store-backup.sh`, so a
throwaway container can reach it. Point the backup directory somewhere that
outlives the container: `/home/agave` is a 64 MiB tmpfs in
`docker-compose.yml`, so the default `$HOME/.agave-backups` would evaporate on
the next `docker compose up`.

```bash
docker run --rm --entrypoint conv-store-backup.sh \
  -v agave-cache:/home/agave/.cache:ro \
  -v "$PWD/backups:/backups" \
  -e AGAVE_BACKUP_DIR=/backups \
  agave:local backup
```

Drop the `:ro` for a restore, which also needs to write the pre-restore
snapshot and the live path. The image runs as uid 10001, so the host backup
directory has to be writable by that uid.

Alternatively bind the cache to a host path (`AGAVE_CACHE_DIR=./.agave-cache`,
which is gitignored) and run the script directly on the host.

The volume outlives a replaced container, so losing the container is not the
disaster. What loses the store is `docker compose down -v` and a pruned Docker
data root, both of which a host-path or separate-filesystem backup survives.

## Restore

```bash
scripts/conv-store-backup.sh verify ~/.agave-backups/conversations-20260927T120000Z.json
scripts/conv-store-backup.sh restore ~/.agave-backups/conversations-20260927T120000Z.json
docker compose restart agave
```

`restore` verifies the backup before touching anything, refuses a file that is
truncated, unbalanced, or written in a different envelope version, snapshots
the current live store to `{backup dir}/conversations-prerestore-<stamp>.json`
so a wrong restore is undoable, installs through a temporary file and renames,
then verifies what it installed. Restart the server to load it; a store the
current build still cannot parse is quarantined to `.corrupt`, not dropped, so
a failed restore leaves the data recoverable.

## Verify the restore path

A backup that has never been restored is a hypothesis. This repo runs the
whole path in CI and locally:

```bash
zig build conv-store-backup-test     # backup, verify, reject-truncated,
                                    # restore, pre-restore snapshot, retention
scripts/conv-store-backup.sh --self-test   # same, standalone
```

`zig build check` depends on it, so a broken backup or restore fails the gate.

## Out of scope

Encryption at rest for the conversation store: it holds plaintext prompts by
design (`docs/THREAT_MODEL.md`, T6). Backups inherit that; store the backup
directory on an encrypted filesystem.
