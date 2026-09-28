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
| Overflow store | `<cache>/agave/conversations.json.overflow` | no | The part of a store past the load caps, which the next save overwrites |
| Hub model blobs | `<cache>/huggingface/` | yes, `agave pull` re-downloads | Bandwidth and time only |
| Hub model symlinks | `<cache>/agave/models/{org}/{repo}` | yes, `agave pull` recreates them | A convenience path, nothing else |
| Vulkan pipeline cache | `<cache>/agave/vk_pipeline_cache.bin` | yes, rebuilt on first run | One slower startup |
| Expert profile | caller-supplied path | yes | Profile re-recorded |
| TriAttention calibration | `<model>.cal`, next to the model | yes, `agave calibrate <model.gguf>` | Re-measured at full cost, and it is not in the cache dir, so the backup tier never covered it |

Only the first three cannot be rebuilt. Everything else is a cache with a
rebuild path, so this document is about the conversation store.

Writes for all of them go through `src/durable_file.zig`: write a sibling
`*.tmp.<pid>`, `fsync`, `rename` over the live path, `fsync` the parent
directory. A crash mid-write leaves the previous file intact, and a file at the
live path is always complete. The pid in the tmp name keeps two processes
replacing the same path from truncating or renaming each other's bytes. That
property is what makes a plain file copy a consistent backup, and it is pinned
by the tests in `src/durable_file.zig`.

Other verified properties, so a future pass leaves them alone:

- Hub downloads `fsync` the blob before advertising it complete
  (`src/pull.zig`), verify the byte count, and delete a blob that fails the
  GGUF magic check. A killed download resumes rather than advertising a
  truncated model.
- A conversation store that fails to parse is quarantined by rename (or, if
  the rename fails, by writing a copy) to `{path}.corrupt` before the live path
  is reused, so a bad parse never destroys the only copy
  (`src/server/conv_store.zig`). A second quarantine takes `{path}.corrupt.1`
  rather than replacing the first: each holds a different store, and each is the
  only copy of it. Up to 8 slots per kind are kept, and a full set is reported
  with persistence disabled rather than overwriting one of them.
- Any load failure other than corruption disables persistence for that run
  instead of overwriting the file with an empty list (`src/server/server.zig`,
  `loadConversationsLocked`). A store whose envelope version this build does
  not read is intact, not corrupt: it is left at the live path and persistence
  is disabled, so a downgrade to an older agave does not rename the newer
  build's store aside and replace it with an empty one.
- A store over the load caps (100 conversations, 1000 messages each) is capped
  rather than rejected, and the next save writes back only what loaded. So the
  load writes the whole original to `{path}.overflow` first: without it, the
  part past the caps is destroyed by the first save and the backup taken after
  that save never saw it (`src/server/conv_store.zig`, `preserveOverflow`). A
  second capped store takes `{path}.overflow.1` instead of overwriting the
  first, by the same slot rule as the quarantine copy.
- A conversation id is the store's primary key, so a store carrying the same id
  twice keeps the first record and drops the later one, which no lookup could
  reach behind it. The drop is treated like a cap overflow: the whole original
  goes to `{path}.overflow` before the next save overwrites the live path.
- `--conv-store PATH` points the store anywhere; `--no-conv-store` keeps
  conversations in memory only. The backup script follows a relocated store
  with `--store PATH`.
- The backup copy uses the same discipline as the server: a sibling tmp,
  `sync` on the file, rename, then `sync` on the destination directory, so a
  power loss cannot leave a backup directory that verifies while holding no
  backup. A missing `sync` command fails the backup rather than passing it
  (`copy_atomic` in `scripts/conv-store-backup.sh`).

## Permissions

Message content is user data, so every copy of the store is owner-only:

- The server writes the live store, the quarantine copy, and the overflow
  sidecar with mode `0600` (`durable_file.replacePrivate`), and tightens a
  store written by an older agave to `0600` on load, so an upgrade does not
  leave a `0644` store behind.
- The backup script locks its own backup directory to `0700` and restricts
  each copy to `0600` before the rename publishes it, whatever umask the
  operator runs with. The self-test fails if a backup is not `0600`.

## RPO and RTO

| Disaster | RPO | RTO | Notes |
|---|---|---|---|
| Host or instance loss, no backup | the whole store | n/a | Total loss of conversations |
| Host or instance loss, backups current | last server save | seconds to restore a small file | Store is capped at 64 MiB on load |
| `docker compose down -v` | the whole store | n/a | Deletes the volume |
| Malicious or accidental deletion | last backup taken | same | Only if backups were taken |
| Logical corruption (a bad build writing a wrong but well-formed store) | the interval between backups | seconds to restore | `verify` checks structure, not meaning; see below |
| Bad deploy | none expected | n/a | The on-disk envelope is version 1 and validated on load; an unreadable version is left in place, not silently reinterpreted and not quarantined, so a downgrade keeps the newer build's store for the build that wrote it |

RPO is "last server save", not "last token": the server persists on conversation
mutations, so a crash mid-generation loses at most the tokens of the turn in
flight. Schedule the backup at whatever interval matches how much an in-flight
turn costs.

There is no point-in-time recovery beyond the backup tier. Each save replaces
the live file, so the tier is the only history. A store that is valid JSON,
carries envelope version 1, and has balanced braces but is *wrong* (a title
mangled, a message attributed to the wrong conversation) passes `verify` and
will be copied into the next backup, so backup frequency is the RPO for that
class of corruption. Keep the frequency low enough for that to be tolerable.

## Back up

```bash
scripts/conv-store-backup.sh path      # where the store is
scripts/conv-store-backup.sh backup    # copy + verify + prune old
scripts/conv-store-backup.sh --store /srv/agave/conversations.json backup
```

`backup` resolves the path the same way the server does, copies through a
temporary file and renames (so a killed backup never leaves a partial file
that a later restore would install), verifies the copy, also copies a
`.corrupt` or `.overflow` store when one exists, and prunes. It exits nonzero
on every failure; it has no quiet failure mode.

A server started with `--conv-store PATH` writes its store wherever it was
told, which the environment-based resolution above cannot see. Pass
`--store PATH` to `backup`, `check`, `restore`, or `path` for that
deployment; the flag goes before the command and wins over
`XDG_CACHE_HOME`. Without it the job either backs up a path nothing writes or
fails on a missing file, and a store the operator moved is the one kind of
state here with no protection at all.

Retention has two tiers, because the kinds of file in the backup
directory are not interchangeable:

| Kind | Name | Retained by | Default |
|---|---|---|---|
| Dated backup | `conversations-<stamp>.json` | `AGAVE_KEEP` | 14 |
| Quarantined store, overflow store, pre-restore snapshot | `conversations-corrupt-<stamp>.json`, `conversations-overflow-<stamp>.json`, `conversations-prerestore-<stamp>.json` (numbered sidecars end in `-<slot>.json`) | `AGAVE_KEEP_SNAPSHOT` | 5 |

Rotation of dated backups never touches the snapshot tier: a quarantined store
is the only remaining copy of a file the server could not parse, an overflow
store is the only copy of the part a capped load dropped, and a pre-restore
snapshot is the only undo for a restore installed by mistake. The snapshot
tier is a single count, so set `AGAVE_KEEP_SNAPSHOT` above the number of
snapshot kinds you expect to keep at once (three, if quarantined and overflow
stores are both present).

Pruning matches the exact names above, so it never deletes a file this script
did not create.

Destination is `AGAVE_BACKUP_DIR`, default `$HOME/.agave-backups`. **Set it to
a different filesystem than the cache directory.** A backup on the same disk
protects against a bad save, not against losing the disk, which is the case
that matters, so `backup` refuses to run when both resolve to the same
filesystem (`df -P` device) and names the device. `AGAVE_ALLOW_SAME_FS=1`
overrides the refusal when that tradeoff is deliberate, for instance while
testing. The check is filesystem-level, not disk-level: two directories on
different partitions of one disk pass it and do not survive losing the disk.

Put it on a schedule (cron, systemd timer, whatever the host runs). The script
is idempotent and safe to run while the server is serving.

`scripts/systemd/agave-conv-store-backup.service` and
`agave-conv-store-backup.timer` are that schedule for a systemd host: an
hourly run that also runs `check` and retries a failed run, with
`Persistent=true` so a host that was off at :17 backs up at the next boot
instead of waiting for the next hour. Edit `User=` and `AGAVE_BACKUP_DIR` in
the service before enabling it, and create the account and the directory it
names (`useradd --system` for the user, `install -d -o <user> -g <user> -m 0700`
for the destination). A missing one fails the unit at start, not at the first
backup. Nothing in this repository installs or starts them.

## Know whether the backup ran

A backup job that stopped running, or that cannot write its destination, is
indistinguishable from a job with nothing to do. `check` asks:

```bash
scripts/conv-store-backup.sh check      # exit 0 = the tier is a recovery path
```

It fails when the backup directory does not exist, holds no dated backup, holds
a newest backup older than `AGAVE_MAX_AGE_HOURS` (default 26), or when the
newest backup no longer verifies. Schedule it next to the backup and alert on a
nonzero exit; a monthly `check` against a daily backup is the minimum, since
`AGAVE_MAX_AGE_HOURS` has to exceed the real backup interval to be meaningful.

```cron
# Hourly backup, freshness checked every run: a nonzero exit is the alert.
17 * * * * AGAVE_BACKUP_DIR=/mnt/backup/agave $HOME/agave/scripts/conv-store-backup.sh backup
23 * * * * AGAVE_BACKUP_DIR=/mnt/backup/agave $HOME/agave/scripts/conv-store-backup.sh check
```

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

`backups/` at the repo root is gitignored: the tier is plaintext prompt
history, and a `git add -A` must not reach it.

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

Add `--store PATH` to `restore` when the server was started with
`--conv-store PATH`; without it the restore installs the backup at the
default path and leaves the relocated store untouched.

`restore` verifies the backup before touching anything, refuses a file that is
truncated, unbalanced, or written in a different envelope version, snapshots
the current live store to `{backup dir}/conversations-prerestore-<stamp>.json`
so a wrong restore is undoable, installs through a temporary file and renames,
then verifies what it installed. Restart the server to load it; a store the
current build still cannot parse is left in place, not dropped, so a failed
restore leaves the data recoverable.

## Verify the restore path

A backup that has never been restored is a hypothesis. This repo runs the
whole path in CI and locally:

```bash
zig build conv-store-backup-test     # backup, verify, reject-truncated,
                                    # reject-other-envelope-version,
                                    # format-version drift guard, restore,
                                    # pre-restore snapshot, retention,
                                    # retention scope, same-filesystem refusal,
                                    # check fresh/missing/stale,
                                    # --store override, whole help text
scripts/conv-store-backup.sh --self-test   # same, standalone
```

`zig build check` depends on it, so a broken backup or restore fails the gate.

The self-test also compares the script's `STORE_FORMAT_VERSION` against
`format_version` in `src/server/conv_store.zig`. A store format bump that never
reaches the script would otherwise make every `backup` fail after it has
already copied the store, and every `restore` refuse a file the server writes
daily, with the first sign of it being a broken deployment.

What the self-test does not cover: a restore into a store the current build
rejects at load. The load paths above mean that copy stays at the live path
rather than being dropped, but proving it needs a running server.

## Configuration and secrets

`AGAVE_API_KEY` and `HF_TOKEN` exist only in the deployment's `.env`
(gitignored, never in the image) and in the process environment
(`docs/THREAT_MODEL.md`). Nothing in the cache volume or the backup tier holds
them, so a lost `.env` means a new key, not a recovered one: every client has
to be updated with the replacement. Rotating is a `.env` edit plus
`docker compose up -d`, which restarts the server. Keep a copy in whatever
password manager the deployment already uses; the repo cannot supply one.

## Out of scope

Encryption at rest for the conversation store: it holds plaintext prompts by
design (`docs/THREAT_MODEL.md`, T6). Backups inherit that; store the backup
directory on an encrypted filesystem.
