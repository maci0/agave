#!/usr/bin/env bash
# conv-store-backup.sh, back up, verify, and restore the web-UI conversation
# store.
#
# The conversation store is the only state Agave owns that cannot be
# regenerated: model blobs re-download from the Hub, the Vulkan pipeline cache
# and expert profiles rebuild themselves, but a deleted conversation is gone.
# The server writes it with tmp+fsync+rename (src/durable_file.zig), so a copy
# taken at any moment is a complete file, never a half-written one. That makes
# a plain file copy a consistent backup.
#
# Resolve the live path the same way src/server/conv_store.zig defaultPath does:
#   $XDG_CACHE_HOME/agave/conversations.json   (if XDG_CACHE_HOME is non-empty)
#   $HOME/.cache/agave/conversations.json      (otherwise)
# In the compose image that is /home/agave/.cache/agave/conversations.json,
# inside the `agave-cache` volume.
#
# --store PATH names the store explicitly, for a server started with
# `--conv-store PATH`. Without it the store is resolved from the environment
# only, and a store the operator moved is silently not the one backed up.
# Options go before the command; the command follows.
#
# Usage:
#   scripts/conv-store-backup.sh path                 # print the live path
#   scripts/conv-store-backup.sh backup               # copy + verify, prune old
#   scripts/conv-store-backup.sh verify FILE          # check a backup is loadable
#   scripts/conv-store-backup.sh restore FILE         # verify, snapshot, install
#   scripts/conv-store-backup.sh check                # backup tier is fresh and loadable
#   scripts/conv-store-backup.sh --store PATH backup  # store is not at the default path
#   scripts/conv-store-backup.sh --self-test          # exercise all of the above
#
# Environment:
#   AGAVE_BACKUP_DIR   backup destination (default: $HOME/.agave-backups)
#   AGAVE_KEEP         dated backups to keep, oldest pruned first (default: 14)
#   AGAVE_KEEP_SNAPSHOT quarantined-store and pre-restore copies to keep
#                      (default: 5); ordinary rotation never touches them
#   AGAVE_MAX_AGE_HOURS newest backup older than this fails `check` (default: 26)
#   AGAVE_ALLOW_SAME_FS=1 permit a backup dir on the store's own filesystem
#
# Exit 0 on success, 1 on any failure. Nothing is silent: every step prints
# what it did, and a failed copy, a malformed store, or a missing file is a
# nonzero exit, not a warning.
#
# Needs bash, coreutils-style stat, awk, grep, and cp. Runs on any host agave
# runs on, including macOS: no GNU-only find(1) or stat(1) syntax.
#
# Runbook: docs/DURABILITY.md
set -euo pipefail
export LC_ALL=C TZ=UTC

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Envelope version this script understands. src/server/conv_store.zig
# format_version must stay in step; verify fails loudly on anything else
# rather than installing a file the current build cannot load.
STORE_FORMAT_VERSION=1
KEEP="${AGAVE_KEEP:-14}"
KEEP_SNAPSHOT="${AGAVE_KEEP_SNAPSHOT:-5}"
MAX_AGE_HOURS="${AGAVE_MAX_AGE_HOURS:-26}"
HOUR_SECONDS=3600

# Set by --store PATH. Empty means "resolve from the environment", which is
# what conv_store.defaultPath does. A server run with `--conv-store PATH` puts
# its store somewhere this resolution cannot see, so without the override the
# script would back up (or find nothing at) a path the operator never writes.
STORE_OVERRIDE=""

# The header comment, minus the shebang. Bounded by the first non-comment line
# rather than a line number, so adding a line to the header cannot silently
# truncate the help.
usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'
}

die() {
    echo "conv-store-backup: $*" >&2
    exit 1
}

note() {
    echo "conv-store-backup: $*"
}

# KEEP is a loop bound and an array index, so a non-numeric or out-of-range
# value either aborts in the arithmetic context or, at 0 and below, prunes the
# whole tier including the copy this run just made. Reject anything that is not
# a plain positive integer.
if [[ ! "$KEEP" =~ ^[1-9][0-9]*$ ]]; then
    die "AGAVE_KEEP must be a positive integer, got '${KEEP}'"
fi
if [[ ! "$KEEP_SNAPSHOT" =~ ^[1-9][0-9]*$ ]]; then
    die "AGAVE_KEEP_SNAPSHOT must be a positive integer, got '${KEEP_SNAPSHOT}'"
fi
if [[ ! "$MAX_AGE_HOURS" =~ ^[0-9]+$ ]]; then
    die "AGAVE_MAX_AGE_HOURS must be a non-negative integer, got '${MAX_AGE_HOURS}'"
fi

# Same precedence as conv_store.defaultPath: --store wins (it is what
# `--conv-store` told the server), then XDG when non-empty, then HOME, and
# neither set means the store has no path at all.
live_store_path() {
    if [[ -n "$STORE_OVERRIDE" ]]; then
        printf '%s\n' "$STORE_OVERRIDE"
        return 0
    fi
    local xdg="${XDG_CACHE_HOME:-}"
    local home="${HOME:-}"
    if [[ -n "$xdg" ]]; then
        printf '%s\n' "$xdg/agave/conversations.json"
    elif [[ -n "$home" ]]; then
        printf '%s\n' "$home/.cache/agave/conversations.json"
    else
        return 1
    fi
}

backup_dir() {
    printf '%s\n' "${AGAVE_BACKUP_DIR:-${HOME:?HOME is unset and AGAVE_BACKUP_DIR is unset}/.agave-backups}"
}

stamp() {
    date -u +%Y%m%dT%H%M%SZ
}

# The file kinds have separate retention because they are not
# interchangeable: a dated backup is one point in time, while the quarantined
# and overflow stores and a pre-restore snapshot are the only copies of
# something the server could not parse, could not keep whole, or an operator
# replaced by mistake. Rotation of ordinary backups must never decide their
# fate.
# EREs matched against the basename by bash's own regex engine, not find(1):
# -regextype and -printf are GNU extensions that BSD/macOS find rejects, and
# the runbook schedules this script with cron on any host, macOS included.
readonly DATED_RE='^conversations-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'
readonly SNAPSHOT_RE='^conversations-(corrupt|overflow|prerestore)-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'

# mtime as seconds.fraction. GNU stat and BSD/macOS stat take the same field
# under different syntax, so the flavor is probed once instead of guessed from
# uname. The fraction is not optional: a `backup` run writes the dated copy and
# the sidecar copies in one second, and pruning the wrong one of two files with
# the same whole-second mtime is a deletion with no recovery window. Fixed
# width on each platform, so `sort -r` below orders by mtime.
if stat -c %Y . >/dev/null 2>&1; then
    mtime_of() { stat -c %.9Y "$1"; }
else
    mtime_of() { stat -f %Fm "$1"; }
fi

# Newest first, one path per line. $1 is the directory, $2 a basename ERE.
list_backups() {
    local dir="$1" re="$2" f
    (
        shopt -s nullglob
        for f in "$dir"/*; do
            [[ -f "$f" && "${f##*/}" =~ $re ]] || continue
            printf '%s %s\n' "$(mtime_of "$f")" "$f"
        done
    ) | sort -r | cut -d' ' -f2-
}

# A copy on the store's own filesystem survives a bad save and nothing else:
# not a lost disk, not `down -v`, not a pruned Docker root. Refuse that
# configuration rather than let a green backup script imply coverage.
# AGAVE_ALLOW_SAME_FS=1 opts out (it is a caller decision, not a default).
assert_separate_fs() {
    local live_dir="$1" backup="$2"
    [[ "${AGAVE_ALLOW_SAME_FS:-0}" == "1" ]] && return 0
    local live_dev backup_dev
    live_dev="$(device_of "$live_dir")"
    backup_dev="$(device_of "$backup")"
    [[ -n "$live_dev" && -n "$backup_dev" ]] ||
        die "cannot determine the filesystem of '$live_dir' or '$backup' (df -P); set AGAVE_ALLOW_SAME_FS=1 to skip the check"
    [[ "$live_dev" != "$backup_dev" ]] && return 0
    die "backup dir $backup and the store's directory $live_dir are on the same filesystem ($live_dev): the backup would not survive loss of that filesystem. Point AGAVE_BACKUP_DIR elsewhere, or set AGAVE_ALLOW_SAME_FS=1 if you accept that"
}

# `df -P` device of the filesystem holding $1. Two directories under one
# filesystem, not one disk: a partition per directory is a different device and
# is not detected, so a shared host disk with two mount points still slips
# through. docs/DURABILITY.md records the limit.
device_of() {
    df -P -- "$1" 2>/dev/null | awk 'NR == 2 { print $1 }'
}

# Structural check: the server only needs a non-empty object carrying the
# expected envelope version. Full parse validation is the server's job on
# restart (src/server/conv_store.zig load), and a file that fails it is left
# in place rather than dropped, so nothing is destroyed by a false negative
# here.
# Count `{` and `}` outside JSON string literals, honoring backslash escapes,
# so a literal brace in message content cannot look like truncation.
brace_balance() {
    awk '
    {
        n = length($0)
        for (i = 1; i <= n; i++) {
            c = substr($0, i, 1)
            if (esc) { esc = 0; continue }
            if (c == "\\") { if (in_string) esc = 1; continue }
            if (c == "\"") { in_string = !in_string; continue }
            if (in_string) continue
            if (c == "{") ob++
            else if (c == "}") cb++
        }
    }
    END { print ob + 0, cb + 0 }
    ' "$1"
}

# The envelope version src/server/conv_store.zig writes. `verify_store` matches
# STORE_FORMAT_VERSION, so a format bump that never reaches this script makes
# every backup fail after it has already copied the store, and every restore
# refuse a file the server writes daily. The check runs in the self-test rather
# than at load time because the container image ships this script without the
# source tree, and it fails loudly instead of inferring a version it cannot read.
assert_format_version_agrees() {
    local src="$REPO_ROOT/src/server/conv_store.zig" found
    [[ -f "$src" ]] || return 0
    found="$(grep -Eo 'pub const format_version: u32 = [0-9]+' "$src" | grep -Eo '[0-9]+$' | head -1)"
    [[ -n "$found" ]] || die "could not read format_version from $src; keep this script's STORE_FORMAT_VERSION in step with it"
    [[ "$found" == "$STORE_FORMAT_VERSION" ]] ||
        die "STORE_FORMAT_VERSION=$STORE_FORMAT_VERSION but src/server/conv_store.zig writes envelope version $found: update this script, or backups and restores fail against every store the server writes"
}

verify_store() {
    local file="$1"
    [[ -f "$file" ]] || die "not a file: $file"
    [[ -s "$file" ]] || die "empty file: $file"
    [[ "$(head -c 1 "$file")" == "{" ]] || die "not a JSON object: $file"
    grep -Eq '"version":[[:space:]]*'"$STORE_FORMAT_VERSION"'([[:space:]]*[,}])' "$file" ||
        die "no \"version\":$STORE_FORMAT_VERSION envelope in $file (written by a different format version; inspect before restoring)"
    # Balanced braces catches the truncation that a single missing byte causes.
    # Only braces outside string literals count: message content is arbitrary
    # user text, so a store whose conversation mentions `fn f() {` is loadable
    # by the server and must verify here too.
    local open close
    read -r open close < <(brace_balance "$file")
    [[ "$open" == "$close" ]] || die "unbalanced braces in $file (open=$open close=$close): truncated or corrupt"
    note "verified $file"
}

# Copy through a sibling tmp so a killed backup never leaves a partial file
# that a later restore would happily install. The rename is followed by a sync
# of the destination directory, matching the tmp+fsync+rename+fsync-parent
# sequence the server itself uses (src/durable_file.zig): without it a power
# loss can drop the renamed entry and leave a backup directory that verifies
# while holding no backup at all.
#
# Every copy of the store is chat history, so the tmp is owner-only before it
# is published. `cp` would otherwise leave it at the caller's umask, and a
# 0644 backup hands every local user the full conversation history.
copy_atomic() {
    local src="$1" dest="$2"
    cp -- "$src" "$dest.tmp" || die "copy $src -> $dest.tmp failed"
    chmod 600 -- "$dest.tmp" || die "could not restrict permissions on $dest.tmp"
    if ! command -v sync >/dev/null 2>&1; then
        die "no sync command available; cannot flush $dest.tmp to disk (install coreutils or busybox)"
    fi
    sync "$dest.tmp" || die "could not flush $dest.tmp to disk"
    mv -- "$dest.tmp" "$dest" || die "rename $dest.tmp -> $dest failed"
    sync -- "$(dirname -- "$dest")" || die "could not flush the directory holding $dest"
}

do_backup() {
    local live dest dir
    live="$(live_store_path)" || die "neither XDG_CACHE_HOME nor HOME is set; pass --store PATH"
    dir="$(backup_dir)"
    mkdir -p -- "$dir" || die "cannot create $dir"
    # The backup dir holds nothing but copies of the conversation store, so it
    # is owner-only even when the operator's umask is not.
    chmod 700 -- "$dir" || die "cannot restrict permissions on $dir"
    if [[ ! -f "$live" ]]; then
        die "no conversation store at $live (nothing to back up; the server writes one on its first conversation)"
    fi
    assert_separate_fs "$(dirname -- "$live")" "$dir"
    dest="$dir/conversations-$(stamp).json"
    [[ -e "$dest" ]] && dest="${dest%.json}-$$.json"
    copy_atomic "$live" "$dest"
    verify_store "$dest"
    # The sidecars are the only remaining trace of state the server did not
    # keep at the live path: a store it could not parse, and a store it
    # loaded only in part. Losing either loses recoverable data. The name is
    # stamped once: stamping per use straddles a second boundary and verifies
    # a file that was never written.
    local suffix
    for suffix in corrupt overflow; do
        local sidecar="$live.$suffix"
        [[ -f "$sidecar" ]] || continue
        local sidecar_copy
        sidecar_copy="$dir/conversations-$suffix-$(stamp).json"
        [[ -e "$sidecar_copy" ]] && sidecar_copy="${sidecar_copy%.json}-$$.json"
        copy_atomic "$sidecar" "$sidecar_copy"
        verify_store "$sidecar_copy" ||
            note "kept $sidecar_copy even though it does not verify; it is the only copy"
        note "backed up $suffix store $sidecar"
    done
    prune "$dir"
    note "backed up $live -> $dest"
}

# Prune oldest first, never below one file in a tier, and only files this
# script names: a retention job that can reach zero backups, or that reaches a
# quarantined store or pre-restore snapshot by counting them as ordinary
# backups, is a deletion path with no recovery window.
prune() {
    local dir="$1"
    prune_tier "$dir" "$DATED_RE" "$KEEP"
    prune_tier "$dir" "$SNAPSHOT_RE" "$KEEP_SNAPSHOT"
}

prune_tier() {
    local dir="$1" re="$2" keep="$3"
    local -a files
    mapfile -t files < <(list_backups "$dir" "$re")
    (( ${#files[@]} <= keep )) && return 0
    local i
    for ((i = keep; i < ${#files[@]}; i++)); do
        rm -f -- "${files[i]}" || die "prune failed: ${files[i]}"
        note "pruned old backup ${files[i]}"
    done
}

do_restore() {
    local file="${1:-}"
    [[ -n "$file" ]] || die "restore needs a backup file: scripts/conv-store-backup.sh restore FILE"
    verify_store "$file"
    local live
    live="$(live_store_path)" || die "neither XDG_CACHE_HOME nor HOME is set; pass --store PATH"
    # Snapshot the outgoing store before overwriting it: a restore of the wrong
    # file must be undoable without going back to the backup tier.
    if [[ -f "$live" ]]; then
        local dir snap
        dir="$(backup_dir)"
        mkdir -p -- "$dir" || die "cannot create $dir"
        chmod 700 -- "$dir" || die "cannot restrict permissions on $dir"
        snap="$dir/conversations-prerestore-$(stamp).json"
        [[ -e "$snap" ]] && snap="${snap%.json}-$$.json"
        copy_atomic "$live" "$snap"
        note "snapshotted the live store to $snap"
    fi
    mkdir -p -- "$(dirname -- "$live")" || die "cannot create $(dirname -- "$live")"
    copy_atomic "$file" "$live"
    verify_store "$live" || die "restored copy at $live does not verify; the backup file is untouched"
    note "restored $file -> $live"
    note "restart the server (docker compose restart agave) to load it; a store it cannot parse is quarantined, not dropped"
}

# Is the backup tier still a recovery path? A backup job that stopped running
# looks exactly like a backup job that has nothing to do, so freshness and
# loadability of the newest copy are checked rather than assumed. Run this on
# a schedule alongside the backup and alert on a nonzero exit.
do_check() {
    local dir
    dir="$(backup_dir)"
    [[ -d "$dir" ]] || die "no backup dir at $dir (the backup job has never run, or AGAVE_BACKUP_DIR moved)"
    local newest
    newest="$(list_backups "$dir" "$DATED_RE" | head -1)"
    [[ -n "$newest" ]] || die "no dated backup in $dir (the backup job has never produced one)"
    local mtime
    mtime="$(mtime_of "$newest")"
    local age_seconds max_age_seconds
    age_seconds=$(( $(date -u +%s) - ${mtime%%.*} ))
    (( age_seconds >= 0 )) || die "newest backup $newest has a timestamp in the future; the host clock is wrong"
    max_age_seconds=$(( MAX_AGE_HOURS * HOUR_SECONDS ))
    if (( age_seconds > max_age_seconds )); then
        die "newest backup $newest is $(( age_seconds / HOUR_SECONDS ))h old, over AGAVE_MAX_AGE_HOURS=$MAX_AGE_HOURS; the backup job is not running or cannot write $dir"
    fi
    verify_store "$newest"
    local snapshots
    snapshots="$(list_backups "$dir" "$SNAPSHOT_RE" | wc -l)"
    note "newest backup $newest is $(( age_seconds / HOUR_SECONDS ))h old; $(( age_seconds % HOUR_SECONDS / 60 ))m; $snapshots quarantined/overflow/pre-restore copies on file"
}

do_self_test() {
    local tmp status=0
    tmp="$(mktemp -d)"
    # mktemp puts the store and the backup tier on one filesystem, which is the
    # case do_backup refuses. Opt in for the run; the rejection case below turns
    # it back off explicitly.
    export AGAVE_ALLOW_SAME_FS=1
    # Global, because the EXIT trap below runs after do_self_test's locals are
    # out of scope.
    SELF_TEST_DIR="$tmp"
    trap 'rm -rf -- "$SELF_TEST_DIR"' EXIT
    local store="$tmp/cache/agave/conversations.json"
    mkdir -p "$(dirname "$store")"
    printf '%s' '{"version":1,"active_id":2,"next_id":3,"conversations":[{"id":2,"title":"self test","messages":[{"role":"user","content":"hello"}]}]}' >"$store"

    AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup >/dev/null
    local backup
    backup="$(find "$tmp/backups" -name 'conversations-2*.json' | head -1)"
    [[ -f "$backup" ]] || die "self-test: backup produced no file"
    # A backup is the whole chat history, so it must not be readable by any
    # other user of the host whatever umask the operator runs with.
    if [[ "$(stat -c '%a' -- "$backup")" != "600" ]]; then
        echo "conv-store-backup: self-test FAILED: backup mode is $(stat -c '%a' -- "$backup"), expected 600" >&2
        status=1
    fi

    # A restore must be rejected before it can clobber the live store.
    printf '%s' '{"version":1,' >"$tmp/truncated.json"
    if (AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_restore "$tmp/truncated.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: truncated file was accepted" >&2
        status=1
    fi
    cmp -s "$store" "$tmp/truncated.json" && {
        echo "conv-store-backup: self-test FAILED: rejected restore still wrote the live store" >&2
        status=1
    }

    # A store this build does not read is not restorable into, whatever else is
    # wrong with it, and the cases above only cover truncation. verify_store
    # ends in `exit`, so the call is a subshell: a direct call would take the
    # self-test down with it instead of recording the failure.
    printf '%s' '{"version":99,"conversations":[]}' >"$tmp/future.json"
    if (verify_store "$tmp/future.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store in another envelope version was accepted" >&2
        status=1
    fi

    # The version this script verifies has to be the version the server writes,
    # or every backup of a good store dies at the verify step.
    assert_format_version_agrees

    # Braces inside message content are text, not structure.
    printf '%s' '{"version":1,"active_id":2,"next_id":3,"conversations":[{"id":2,"title":"brace { title","messages":[{"role":"user","content":"see fn f() { } \"quoted\""}]}]}' >"$tmp/braces.json"
    if ! verify_store "$tmp/braces.json" >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store with braces in message content was rejected" >&2
        status=1
    fi

    # The real path: verify, snapshot the outgoing store, install, verify again.
    AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_restore "$backup" >/dev/null
    cmp -s "$store" "$backup" || {
        echo "conv-store-backup: self-test FAILED: restored bytes differ from the backup" >&2
        status=1
    }
    find "$tmp/backups" -name 'conversations-prerestore-*.json' | grep -q . || {
        echo "conv-store-backup: self-test FAILED: no pre-restore snapshot" >&2
        status=1
    }

    # Retention must leave at least one backup and prune the rest, and it must
    # reach only dated backups: the pre-restore snapshot is the only undo for
    # the restore above, so counting it as an ordinary backup would let the
    # next run delete it.
    #
    # Retention values are read into globals at load time, so they are set in
    # the subshell rather than as an env prefix on the call: `AGAVE_KEEP=1
    # do_backup` leaves KEEP at its default and the case below would pass
    # without ever pruning anything.
    ( KEEP=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    local remaining
    remaining=$(find "$tmp/backups" -name 'conversations-2*.json' | wc -l)
    [[ "$remaining" -eq 1 ]] || {
        echo "conv-store-backup: self-test FAILED: AGAVE_KEEP=1 left $remaining dated backups, expected 1" >&2
        status=1
    }
    find "$tmp/backups" -name 'conversations-prerestore-*.json' | grep -q . || {
        echo "conv-store-backup: self-test FAILED: retention deleted the pre-restore snapshot" >&2
        status=1
    }

    # A quarantined store copied by a later backup is the only remaining trace
    # of a file the server could not parse, so its tier rotates on its own
    # retention rather than on the dated-backup count. The overflow sidecar
    # holds a store the server loaded only in part, and the next save
    # destroys the rest, so it travels the same way.
    cp -- "$store" "$tmp/cache/agave/conversations.json.corrupt"
    cp -- "$store" "$tmp/cache/agave/conversations.json.overflow"
    ( KEEP_SNAPSHOT=2; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    rm -f -- "$tmp/cache/agave/conversations.json.corrupt" "$tmp/cache/agave/conversations.json.overflow"
    [[ "$(find "$tmp/backups" -name 'conversations-corrupt-*.json' | wc -l)" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: dated-backup rotation deleted the quarantined-store copy" >&2
        status=1
    }
    [[ "$(find "$tmp/backups" -name 'conversations-overflow-*.json' | wc -l)" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: the overflow sidecar was not backed up" >&2
        status=1
    }

    # Retention must not reach files this script did not name, so a store
    # dropped into the backup dir by hand survives.
    printf '%s' '{"version":1}' >"$tmp/backups/conversations-manual.json"
    ( KEEP=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    [[ -f "$tmp/backups/conversations-manual.json" ]] || {
        echo "conv-store-backup: self-test FAILED: retention deleted a file it did not create" >&2
        status=1
    }

    # A backup on the store's own filesystem covers a bad save and nothing
    # else. The temp dir here is one filesystem, so this is that case.
    if (AGAVE_ALLOW_SAME_FS=0 AGAVE_BACKUP_DIR="$tmp/cache/backups" XDG_CACHE_HOME="$tmp/cache" do_backup) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: accepted a backup dir on the store's filesystem" >&2
        status=1
    fi

    # A server started with `--conv-store PATH` writes outside the resolved
    # default, so the override is the only way that store gets protected. It
    # must win over XDG_CACHE_HOME, and a store it names must back up that
    # file's bytes rather than the default one.
    local self_path="${BASH_SOURCE[0]}"
    mkdir -p "$tmp/moved"
    printf '%s' '{"version":1,"active_id":7,"next_id":8,"conversations":[{"id":7,"title":"moved","messages":[{"role":"user","content":"elsewhere"}]}]}' >"$tmp/moved/store.json"
    local moved_backup
    AGAVE_BACKUP_DIR="$tmp/moved-backups" "$self_path" --store "$tmp/moved/store.json" backup >/dev/null
    moved_backup="$(find "$tmp/moved-backups" -name 'conversations-2*.json' | head -1)"
    if [[ -z "$moved_backup" ]] || ! cmp -s "$moved_backup" "$tmp/moved/store.json"; then
        echo "conv-store-backup: self-test FAILED: --store did not back up the store it names" >&2
        status=1
    fi
    [[ "$("$self_path" --store "$tmp/moved/store.json" path)" == "$tmp/moved/store.json" ]] || {
        echo "conv-store-backup: self-test FAILED: --store= or --store PATH did not set the store path" >&2
        status=1
    }
    [[ "$("$self_path" --store="$tmp/moved/store.json" path)" == "$tmp/moved/store.json" ]] || {
        echo "conv-store-backup: self-test FAILED: --store=PATH was not accepted" >&2
        status=1
    }
    # Without the override the default path wins, so the flag cannot be ignored.
    [[ "$("$self_path" path)" != "$tmp/moved/store.json" ]] || {
        echo "conv-store-backup: self-test FAILED: a store path survived without --store" >&2
        status=1
    }
    if "$self_path" --store backup >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: --store with no path was accepted" >&2
        status=1
    fi

    # Freshness: a backup job that stopped running is only visible if `check`
    # is asked, so exercise the passing case, a missing tier, and a stale copy.
    AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check >/dev/null || {
        echo "conv-store-backup: self-test FAILED: check rejected a fresh, loadable backup" >&2
        status=1
    }
    if (AGAVE_BACKUP_DIR="$tmp/nowhere" XDG_CACHE_HOME="$tmp/cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check passed with no backup dir" >&2
        status=1
    fi
    touch -d '2 hours ago' -- "$tmp/backups"/conversations-2*.json
    if (MAX_AGE_HOURS=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check passed on a 2h-old backup with AGAVE_MAX_AGE_HOURS=1" >&2
        status=1
    fi

    # A retention value of 0 or below prunes the whole tier, so the guard
    # rejects it before do_backup can run. It lives at load time, so exercise
    # it by re-entering the script rather than calling do_backup here.
    local bad_keep
    for bad_keep in 0 -3 abc '1.5' ' '; do
        if AGAVE_KEEP="$bad_keep" "$self_path" path >/dev/null 2>&1; then
            echo "conv-store-backup: self-test FAILED: accepted AGAVE_KEEP='$bad_keep'" >&2
            status=1
        fi
    done
    for bad_keep in 0 abc '1.5'; do
        if AGAVE_KEEP_SNAPSHOT="$bad_keep" "$self_path" path >/dev/null 2>&1; then
            echo "conv-store-backup: self-test FAILED: accepted AGAVE_KEEP_SNAPSHOT='$bad_keep'" >&2
            status=1
        fi
    done
    for bad_age in -1 abc '26h'; do
        if AGAVE_MAX_AGE_HOURS="$bad_age" "$self_path" path >/dev/null 2>&1; then
            echo "conv-store-backup: self-test FAILED: accepted AGAVE_MAX_AGE_HOURS='$bad_age'" >&2
            status=1
        fi
    done

    # Help must be the whole header, not a fragment of it: a truncated
    # `--help` hides the flags an operator needs to find.
    local help
    help="$("$self_path" --help)"
    if [[ "$help" != "conv-store-backup.sh, back up, verify, and restore the web-UI conversation"* ||
        "$help" != *"Runbook: docs/DURABILITY.md"* ]]; then
        echo "conv-store-backup: self-test FAILED: --help does not print the whole header" >&2
        status=1
    fi

    if (( status == 0 )); then
        note "self-test passed: backup, verify, reject-truncated, reject-other-version, format-version-agrees, braces-in-content, restore, pre-restore snapshot, retention, snapshot-tier-retention, sidecars, retention-scope, reject-same-filesystem, check-fresh, check-missing, check-stale, reject-bad-retention, store-override, whole-help"
    fi
    return "$status"
}

main() {
    # Options come before the command, so --store is consumed here and the
    # command is whatever remains. An unknown option is a die rather than a
    # silent no-op: a mistyped --store would otherwise back up the default
    # path and report success.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --store)
                [[ $# -ge 2 ]] || die "--store needs a path: scripts/conv-store-backup.sh --store PATH backup"
                # `--store backup` is a missing value, not a store literally
                # named "backup" that silently swallows the command.
                case "$2" in
                    path | backup | verify | restore | check | --self-test)
                        die "--store needs a path, got the command '$2': scripts/conv-store-backup.sh --store PATH backup"
                        ;;
                esac
                STORE_OVERRIDE="$2"
                shift 2
                ;;
            --store=*)
                STORE_OVERRIDE="${1#--store=}"
                [[ -n "$STORE_OVERRIDE" ]] || die "--store needs a non-empty path"
                shift
                ;;
            *) break ;;
        esac
    done

    case "${1:-}" in
        path) live_store_path || die "neither XDG_CACHE_HOME nor HOME is set" ;;
        backup) do_backup ;;
        verify)
            [[ -n "${2:-}" ]] || die "verify needs a file: scripts/conv-store-backup.sh verify FILE"
            verify_store "$2"
            ;;
        restore) do_restore "${2:-}" ;;
        check) do_check ;;
        --self-test) do_self_test ;;
        -h | --help | '') usage ;;
        *)
            die "unknown command '$1' (path, backup, verify, restore, check, --self-test)"
            ;;
    esac
}

main "$@"
