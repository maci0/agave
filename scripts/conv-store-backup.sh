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
# Usage:
#   scripts/conv-store-backup.sh path                 # print the live path
#   scripts/conv-store-backup.sh backup               # copy + verify, prune old
#   scripts/conv-store-backup.sh verify FILE          # check a backup is loadable
#   scripts/conv-store-backup.sh restore FILE         # verify, snapshot, install
#   scripts/conv-store-backup.sh check                # backup tier is fresh and loadable
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

usage() {
    sed -n '3,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# Same precedence as conv_store.defaultPath: XDG wins when non-empty, HOME is
# the fallback, and neither set means the store has no path at all.
live_store_path() {
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

# The three file kinds have separate retention because they are not
# interchangeable: a dated backup is one point in time, while a quarantined
# store and a pre-restore snapshot are the only copies of something the server
# could not parse or a store an operator replaced by mistake. Rotation of
# ordinary backups must never decide their fate.
# find -regex matches the whole path, so each pattern anchors on the basename.
readonly DATED_RE='.*/conversations-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'
readonly SNAPSHOT_RE='.*/conversations-(corrupt|prerestore)-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'

# Newest first, "<mtime> <path>". $1 is a find -regextype pattern.
list_backups() {
    local dir="$1" re="$2"
    find "$dir" -maxdepth 1 -type f -regextype posix-extended -regex "$re" -printf '%T@ %p\n' |
        sort -rn | cut -d' ' -f2-
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
# restart (src/server/conv_store.zig load), and a file that fails it is
# quarantined to {path}.corrupt rather than dropped, so nothing is destroyed
# by a false negative here.
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
# that a later restore would happily install.
copy_atomic() {
    local src="$1" dest="$2"
    cp -- "$src" "$dest.tmp" || die "copy $src -> $dest.tmp failed"
    if command -v sync >/dev/null 2>&1; then
        sync "$dest.tmp" 2>/dev/null || true
    fi
    mv -- "$dest.tmp" "$dest" || die "rename $dest.tmp -> $dest failed"
}

do_backup() {
    local live dest dir
    live="$(live_store_path)" || die "neither XDG_CACHE_HOME nor HOME is set; pass --store PATH"
    dir="$(backup_dir)"
    mkdir -p -- "$dir" || die "cannot create $dir"
    if [[ ! -f "$live" ]]; then
        die "no conversation store at $live (nothing to back up; the server writes one on its first conversation)"
    fi
    assert_separate_fs "$(dirname -- "$live")" "$dir"
    dest="$dir/conversations-$(stamp).json"
    [[ -e "$dest" ]] && dest="${dest%.json}-$$.json"
    copy_atomic "$live" "$dest"
    verify_store "$dest"
    # The quarantine copy is the only remaining trace of a store the server
    # could not parse. Losing it loses the recoverable data. The name is
    # stamped once: stamping per use straddles a second boundary and verifies
    # a file that was never written.
    local corrupt="$live.corrupt"
    if [[ -f "$corrupt" ]]; then
        local corrupt_copy
        corrupt_copy="$dir/conversations-corrupt-$(stamp).json"
        [[ -e "$corrupt_copy" ]] && corrupt_copy="${corrupt_copy%.json}-$$.json"
        copy_atomic "$corrupt" "$corrupt_copy"
        verify_store "$corrupt_copy" ||
            note "kept $corrupt_copy even though it does not verify; it is the only copy"
        note "backed up quarantined store $corrupt"
    fi
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
    local entry newest
    entry="$(find "$dir" -maxdepth 1 -type f -regextype posix-extended -regex "$DATED_RE" -printf '%T@ %p\n' | sort -rn | head -1)"
    [[ -n "$entry" ]] || die "no dated backup in $dir (the backup job has never produced one)"
    newest="${entry#* }"
    local age_seconds max_age_seconds
    age_seconds=$(( $(date -u +%s) - ${entry%%.*} ))
    (( age_seconds >= 0 )) || die "newest backup $newest has a timestamp in the future; the host clock is wrong"
    max_age_seconds=$(( MAX_AGE_HOURS * HOUR_SECONDS ))
    if (( age_seconds > max_age_seconds )); then
        die "newest backup $newest is $(( age_seconds / HOUR_SECONDS ))h old, over AGAVE_MAX_AGE_HOURS=$MAX_AGE_HOURS; the backup job is not running or cannot write $dir"
    fi
    verify_store "$newest"
    local snapshots
    snapshots="$(list_backups "$dir" "$SNAPSHOT_RE" | wc -l)"
    note "newest backup $newest is $(( age_seconds / HOUR_SECONDS ))h old; $(( age_seconds % HOUR_SECONDS / 60 ))m; $snapshots quarantined/pre-restore copies on file"
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
    # retention rather than on the dated-backup count.
    cp -- "$store" "$tmp/cache/agave/conversations.json.corrupt"
    ( KEEP_SNAPSHOT=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    rm -f -- "$tmp/cache/agave/conversations.json.corrupt"
    [[ "$(find "$tmp/backups" -name 'conversations-corrupt-*.json' | wc -l)" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: dated-backup rotation deleted the quarantined-store copy" >&2
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
    local self_path="${BASH_SOURCE[0]}"
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

    if (( status == 0 )); then
        note "self-test passed: backup, verify, reject-truncated, braces-in-content, restore, pre-restore snapshot, retention, snapshot-tier-retention, retention-scope, reject-same-filesystem, check-fresh, check-missing, check-stale, reject-bad-retention"
    fi
    return "$status"
}

main() {
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
        *) die "unknown command '$1' (path, backup, verify, restore, check, --self-test)" ;;
    esac
}

main "$@"
