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
#   scripts/conv-store-backup.sh check                # tier is fresh, loadable,
#                                                    # and on its own filesystem
#   scripts/conv-store-backup.sh --store PATH backup  # store is not at the default path
#   scripts/conv-store-backup.sh --self-test          # exercise all of the above
#
# Environment:
#   AGAVE_BACKUP_DIR   backup destination (default: $HOME/.agave-backups)
#   AGAVE_KEEP         dated backups to keep, oldest pruned first (default: 14)
#   AGAVE_KEEP_SNAPSHOT quarantined, overflow, pre-deletion, and pre-restore
#                      copies to keep
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
# Load cap the server refuses to read past, mirroring max_store_bytes in
# src/server/conv_store.zig. A store over it is left at the live path with
# persistence disabled, so `backup` would keep copying it and `restore` would
# keep installing it, and both report success: the operator walks away from a
# restore the server never loaded. assert_max_store_bytes_agrees keeps this in
# step with the server the way STORE_FORMAT_VERSION does.
MAX_STORE_BYTES=67108864
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
# interchangeable: a dated backup is one point in time, while the quarantined,
# overflow, and pre-deletion stores and a pre-restore snapshot are the only
# copies of something the server could not parse, could not keep whole, was
# about to erase, or an operator replaced by mistake. Rotation of ordinary
# backups must never decide their fate.
# EREs matched against the basename by bash's own regex engine, not find(1):
# -regextype and -printf are GNU extensions that BSD/macOS find rejects, and
# the runbook schedules this script with cron on any host, macOS included.
readonly DATED_RE='^conversations-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'
readonly SNAPSHOT_RE='^conversations-(corrupt|overflow|deleted|prerestore)-[0-9]{8}T[0-9]{6}Z(-[0-9]+)?\.json$'

# Undated rotation record. Matches no tier name, so ordinary rotation and
# retention never reach it, and it is not a copy of any conversation.
readonly ROTATION_LOG_NAME='.conversations-rotated.log'

# mtime as seconds.fraction. GNU stat and BSD/macOS stat take the same field
# under different syntax, so the flavor is probed once instead of guessed from
# uname. The fraction is not optional: a `backup` run writes the dated copy and
# the sidecar copies in one second, and pruning the wrong one of two files with
# the same whole-second mtime is a deletion with no recovery window. Fixed
# width on each platform, so `sort -r` below orders by mtime.
if stat -c %Y . >/dev/null 2>&1; then
    mtime_of() { stat -c %.9Y "$1"; }
    mode_of() { stat -c %a "$1"; }
    # Caller passes a whole number of hours; only the spelling differs.
    backdated_stamp() { date -u -d "$1 hours ago" +%Y%m%d%H%M; }
else
    mtime_of() { stat -f %Fm "$1"; }
    mode_of() { stat -f %Lp "$1"; }
    # BSD date takes a relative shift as -v-2H and has no -d at all, so the
    # same hours argument drives both.
    backdated_stamp() { date -u -v-"${1}"H +%Y%m%d%H%M; }
fi

# Size in bytes. The same platform probe as mtime_of, for the same reason: GNU
# and BSD stat share no flag for it.
file_size() {
    if stat -c %Y . >/dev/null 2>&1; then
        stat -c %s -- "$1"
    else
        stat -f %z -- "$1"
    fi
}

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

# The envelope version of a store, on stdout, or nothing and a nonzero exit
# when there is no readable one.
#
# The top-level `"version"` is the only thing that says which schema wrote the
# file, and matching the string anywhere in the file does not find it: a
# conversation that quotes `"version":1` satisfies such a grep, so a store in
# another envelope version (or one truncated before its version reached the
# top level) verifies and then gets restored. A restore installs that file at
# the live path, so the server leaves it there with persistence disabled
# instead of loading it: the backup looks fine and the conversation history is
# gone.
#
# Read the way the server reads it: src/server/json.zig extractIntField takes
# the first `"version":` whose value parses as a number, without caring how
# deep the key sits, and conv_store.parse rejects any value but
# format_version. A quoted `"version":"1"` is text the server never reads as a
# version, so it is rejected here too.
#
# The scan is a byte loop over the whole file because the string and escape
# state has to carry across lines: a store is one JSON line, but awk's record
# handling is not something to depend on for correctness here. Byte offsets
# inside a line, so a store larger than one awk line still reads right.
envelope_version() {
    awk -v want="\"version\":" '
    function emit(v) { print v; found = 1; exit }
    {
        n = length($0)
        for (i = 1; i <= n; i++) {
            c = substr($0, i, 1)
            if (esc) { esc = 0; continue }
            if (c == "\\") { if (in_string) esc = 1; continue }
            if (c == "\"") {
                if (!in_string && substr($0, i, length(want)) == want) {
                    # want is the whole key and its colon, so the value starts
                    # after whatever whitespace the writer put there.
                    j = i + length(want)
                    while (j <= n && substr($0, j, 1) ~ /[ \t]/) j++
                    neg = ""
                    if (j <= n && substr($0, j, 1) == "-") { neg = "-"; j++ }
                    start = j
                    while (j <= n && substr($0, j, 1) ~ /[0-9]/) j++
                    if (j > start) emit(neg substr($0, start, j - start))
                }
                in_string = !in_string
                continue
            }
            if (in_string) continue
            if (c == "{") depth++
            else if (c == "}") { if (depth == 0) exit; depth-- }
        }
    }
    END { if (!found) exit 1 }
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

# The load cap, the same drift guard for max_store_bytes. The Zig side is
# spelled as an expression rather than a decimal, so it is evaluated instead of
# text-matched: a cap that moves to 128 MiB has to be visible here, and one this
# cannot read is a failure, not a reason to guess the old number.
assert_max_store_bytes_agrees() {
    local src="$REPO_ROOT/src/server/conv_store.zig" expr found
    [[ -f "$src" ]] || return 0
    expr="$(grep -Eo 'const max_store_bytes: usize = [^;]+;' "$src" | head -1 | sed -e 's/^const max_store_bytes: usize = //' -e 's/;$//')"
    [[ -n "$expr" ]] || die "could not read max_store_bytes from $src; keep this script's MAX_STORE_BYTES in step with it"
    # bash arithmetic, not awk's: eval() is a gawk extension and this script
    # runs on macOS and busybox hosts too. A cap spelled with anything but
    # digits and arithmetic operators is refused rather than guessed at, so a
    # cap that stops being a literal fails the self-test instead of leaving
    # MAX_STORE_BYTES stale.
    [[ "$expr" =~ ^[0-9]+([[:space:]]*[*+][[:space:]]*[0-9]+)*$ ]] ||
        die "max_store_bytes in $src is '$expr', not a literal byte count this script can read; keep this script's MAX_STORE_BYTES in step with it"
    found=$((expr))
    [[ "$found" == "$MAX_STORE_BYTES" ]] ||
        die "MAX_STORE_BYTES=$MAX_STORE_BYTES but src/server/conv_store.zig refuses to load a store over $found bytes: update this script, or verify accepts stores the server cannot load"
}

verify_store() {
    local file="$1"
    [[ -f "$file" ]] || die "not a file: $file"
    [[ -s "$file" ]] || die "empty file: $file"
    # Loadable is checked before structure, and off the file's size rather
    # than off a scan of it. The server refuses to read a store past
    # max_store_bytes and leaves it at the live path with persistence
    # disabled, so a file that passes every structural check below can still
    # be one the server never loads. Catching it here means `backup` says so at
    # the copy rather than reporting success, and `restore` refuses it before
    # it snapshots and overwrites a good live store to install a dead one.
    # brace_balance walks the file a character at a time, so an oversize file
    # has to be rejected before that, not after it.
    local size
    size="$(file_size "$file")"
    (( size <= MAX_STORE_BYTES )) ||
        die "$file is $size bytes, over the $MAX_STORE_BYTES-byte limit the server loads within: the server would not read it back. Nothing has been changed."
    [[ "$(head -c 1 "$file")" == "{" ]] || die "not a JSON object: $file"
    envelope_version "$file" | grep -qx "$STORE_FORMAT_VERSION" ||
        die "no top-level \"version\":$STORE_FORMAT_VERSION envelope in $file (written by a different format version, or too damaged for the envelope to be read; inspect before restoring)"
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
    # keep at the live path: a store it could not parse, a store it loaded only
    # in part, and the store as it stood just before a conversation was deleted
    # or cleared. Losing any of them loses recoverable data. The server
    # numbers a second sidecar of the same kind (`.corrupt.1`, `.overflow.1`,
    # `.deleted.1`) instead of overwriting the first, so every slot is copied.
    # The name is stamped once: stamping per use straddles a second boundary
    # and verifies a file that was never written.
    local suffix sidecar sidecar_copy index
    for suffix in corrupt overflow deleted; do
        for sidecar in "$live.$suffix" "$live.$suffix".[0-9]*; do
            [[ -f "$sidecar" ]] || continue
            index="${sidecar##*.}"
            [[ "$index" == "$suffix" ]] && index=""
            sidecar_copy="$dir/conversations-$suffix-$(stamp).json"
            # The slot number goes last, where the same-second collision
            # suffix already lives, so both stay inside SNAPSHOT_RE.
            [[ -n "$index" ]] && sidecar_copy="${sidecar_copy%.json}-$index.json"
            [[ -e "$sidecar_copy" ]] && sidecar_copy="${sidecar_copy%.json}-$$.json"
            copy_atomic "$sidecar" "$sidecar_copy"
            verify_store "$sidecar_copy" ||
                note "kept $sidecar_copy even though it does not verify; it is the only copy"
            note "backed up $suffix store $sidecar"
        done
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
    prune_dated "$dir"
    prune_tier "$dir" "$SNAPSHOT_RE" "$KEEP_SNAPSHOT"
}

# Prune the dated tier, recording what went. Rotation is a deletion path, and
# this is the record that keeps it from being a silent one: a store deleted
# from the live path looks exactly like a store the server never had, so
# without the record the dated copies of the deleted conversations are rotated
# away on schedule and the loss becomes permanent with nothing left that names
# what was there. The record is written before the files go, so a crash
# mid-rotation leaves the trail rather than the hole.
#
# The record is dated by nothing, so it is outside both tiers and outside
# DATED_RE and SNAPSHOT_RE: ordinary rotation must never reach it. It carries
# no conversation text (file name, byte size, envelope version), so it is not
# another copy of the history to protect.
prune_dated() {
    local dir="$1"
    # `=()` not just `-a`: under `set -u` an array that was declared but never
    # assigned reads as unbound when the tier is empty.
    local -a files=()
    local line f
    # A `while read` loop, not mapfile: macOS still ships bash 3.2, and the
    # runbook schedules this script with cron on any host, macOS included.
    while IFS= read -r line; do
        files+=("$line")
    done < <(list_backups "$dir" "$DATED_RE")
    (( ${#files[@]} > KEEP )) || return 0

    local log="$dir/$ROTATION_LOG_NAME" when version
    when="$(stamp)"
    local i
    for ((i = KEEP; i < ${#files[@]}; i++)); do
        f="${files[i]}"
        version="$(envelope_version "$f" 2>/dev/null || printf 'unreadable')"
        printf '%s pruned %s bytes=%s version=%s\n' \
            "$when" "${f##*/}" "$(file_size "$f")" "$version" >>"$log" ||
            die "cannot append to the rotation log $log; refusing to prune so the loss is recorded"
        rm -f -- "$f" || die "prune failed: $f"
        note "pruned old backup $f (recorded in $log)"
    done
}

# Rotate a snapshot tier, recording what went before it goes. This tier holds
# state the live path does not: a quarantined store is the only remaining copy
# of a file the server could not parse, an overflow store the only copy of the
# part a capped load dropped, a pre-deletion store the only copy of a
# conversation the API erased, and a pre-restore snapshot the only undo for a
# restore installed by mistake. prune_dated records for exactly this reason, and
# a deletion here is strictly worse than one there: a dated copy going is one
# point in time fewer, while one of these going is the last copy of something,
# with nothing to say what it held afterwards.
#
# The record carries the kind and size, no conversation text, so the log is not
# another copy of the history to protect. It goes to the same undated log
# prune_dated writes, which no tier's rotation can reach, so one ordered record
# of everything this script has removed lives in one place.
prune_tier() {
    local dir="$1" re="$2" keep="$3"
    # `=()` not just `-a`: under `set -u` an array that was declared but never
    # assigned reads as unbound when the tier is empty.
    local -a files=()
    local line
    # A `while read` loop, not mapfile: macOS still ships bash 3.2, and the
    # runbook schedules this script with cron on any host, macOS included.
    while IFS= read -r line; do
        files+=("$line")
    done < <(list_backups "$dir" "$re")
    (( ${#files[@]} <= keep )) && return 0
    local log="$dir/$ROTATION_LOG_NAME" when version i f
    when="$(stamp)"
    for ((i = keep; i < ${#files[@]}; i++)); do
        f="${files[i]}"
        # A quarantined store the server could not parse is exactly the case
        # where the envelope version is unreadable, so `unreadable` is a normal
        # value here rather than the exception it is for a dated backup.
        version="$(envelope_version "$f" 2>/dev/null || printf 'unreadable')"
        printf '%s pruned snapshot %s bytes=%s version=%s\n' \
            "$when" "${f##*/}" "$(file_size "$f")" "$version" >>"$log" ||
            die "cannot append to the rotation log $log; refusing to prune so the loss is recorded"
        rm -f -- "$f" || die "prune failed: $f"
        note "pruned old snapshot $f (recorded in $log)"
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
    # The same refusal `backup` makes, and for the same reason. A tier moved
    # onto the cache volume (the container's default $HOME/.agave-backups is a
    # 64 MiB tmpfs, so the fix an operator reaches for first is to point
    # AGAVE_BACKUP_DIR at a mount) stops every backup run at once, and the old
    # copies stay fresh for a retention window after that. Without this,
    # `check` reports a healthy recovery path for exactly the deployment whose
    # backups are not being taken, which is the one case the check exists for.
    # A store path that cannot be resolved, or whose directory does not exist
    # (the tier is being checked on a recovery host before the store is back),
    # has no filesystem to compare against, so freshness and loadability are
    # the whole answer there.
    local live_dir
    if live_dir="$(live_store_path 2>/dev/null)"; then
        live_dir="$(dirname -- "$live_dir")"
        [[ -d "$live_dir" ]] && assert_separate_fs "$live_dir" "$dir"
    fi
    local newest
    newest="$(list_backups "$dir" "$DATED_RE" | head -1)"
    if [[ -z "$newest" ]]; then
        # The rotation log is the record of what this script pruned. If it is
        # there and the dated tier is not, the copies were removed by something
        # else, and `check` reporting an empty tier as merely unprimed would
        # read as "the job has never run" rather than "the copies are gone".
        if [[ -s "$dir/$ROTATION_LOG_NAME" ]]; then
            die "no dated backup in $dir, but $ROTATION_LOG_NAME records pruned copies: the dated tier was emptied outside this script; read the log and the backup destination's own history before trusting the tier"
        fi
        die "no dated backup in $dir (the backup job has never produced one)"
    fi
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
    # The snapshot tier holds state the live path does not: a quarantined store
    # is the only copy of a file the server could not parse, an overflow store
    # the only copy of the part a capped load dropped, a pre-deletion store the
    # only copy of a conversation the API erased, and a pre-restore snapshot the
    # only undo for a restore installed by mistake. Counting them
    # without reading them makes `check` report a recovery path for a tier
    # whose only remaining copy of something is unreadable, which is the same
    # failure `check` exists to catch for the dated tier. Each one that fails
    # is named and does not fail the run, for the reason `backup` keeps a
    # non-verifying sidecar: the copy is still the only one there is, and
    # refusing to say so would hide it. An alert can grep for these lines.
    #
    # Each verify runs in a subshell because verify_store ends in `die`, which
    # exits: called directly it would take the whole check down on the first
    # unreadable sidecar instead of counting it.
    local snapshots snap snap_bad=0
    while IFS= read -r snap; do
        [[ -n "$snap" ]] || continue
        if ! (verify_store "$snap") >/dev/null 2>&1; then
            note "WARNING: $snap does not verify; it is the only copy of the state it holds, so keep it and inspect by hand"
            snap_bad=$((snap_bad + 1))
        fi
    done < <(list_backups "$dir" "$SNAPSHOT_RE")
    snapshots="$(list_backups "$dir" "$SNAPSHOT_RE" | wc -l)"
    note "newest backup $newest is $(( age_seconds / HOUR_SECONDS ))h old; $(( age_seconds % HOUR_SECONDS / 60 ))m; $snapshots quarantined/overflow/pre-deletion/pre-restore copies on file, $snap_bad of them not verifying"
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
    if [[ "$(mode_of "$backup")" != "600" ]]; then
        echo "conv-store-backup: self-test FAILED: backup mode is $(mode_of "$backup"), expected 600" >&2
        status=1
    fi

    # A second quarantine of the same kind lands at `.corrupt.1` beside the
    # first, and both are the only copy of what they hold: every slot is
    # backed up, and the copies stay distinct.
    printf '%s' '{"version":1,"active_id":0,"next_id":1,"conversations":[]}' >"$tmp/cache/agave/conversations.json.corrupt"
    printf '%s' '{"version":1,"active_id":0,"next_id":2,"conversations":[]}' >"$tmp/cache/agave/conversations.json.corrupt.1"
    AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup >/dev/null
    local kept_corrupt
    kept_corrupt="$(find "$tmp/backups" -name 'conversations-corrupt-*.json' | wc -l | tr -d ' ')"
    if [[ "$kept_corrupt" -lt 2 ]]; then
        echo "conv-store-backup: self-test FAILED: 2 quarantined stores, $kept_corrupt copies in the tier" >&2
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

    # The envelope version is only meaningful as the value of the top-level
    # key, and message content is arbitrary user text: a conversation quoting
    # `"version":1` used to satisfy a whole-file grep, so a store the server
    # refuses to load verified and then got installed over a good live store,
    # leaving persistence disabled with nothing to restore from. The version
    # has to be read the way the server reads it.
    printf '%s' '{"version":99,"conversations":[{"id":1,"title":"t","messages":[{"role":"user","content":"see \"version\":1, in a chat"}]}]}' >"$tmp/future_quote.json"
    if (verify_store "$tmp/future_quote.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: an envelope version quoted in message content was accepted" >&2
        status=1
    fi
    if (AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_restore "$tmp/future_quote.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store whose version appears only in message content was restored" >&2
        status=1
    fi
    cmp -s "$store" "$backup" || {
        echo "conv-store-backup: self-test FAILED: the rejected version-quote restore changed the live store" >&2
        status=1
    }
    # A quoted value is text the server never reads as a version either, so it
    # does not stand in for one. The brace scan is satisfied by the whole file,
    # which is what makes this case a false accept if the version check ever
    # loosens again.
    printf '%s' '{"version":"1","conversations":[]}' >"$tmp/quoted_version.json"
    if (verify_store "$tmp/quoted_version.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a quoted envelope version was accepted" >&2
        status=1
    fi
    # A store with no version key at all: the check fails on absence, not only
    # on a disagreeing number.
    printf '%s' '{"conversations":[]}' >"$tmp/no_version.json"
    if (verify_store "$tmp/no_version.json") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store with no envelope version was accepted" >&2
        status=1
    fi
    # A good store still verifies, and the reader agrees with the script about
    # which version it carries.
    [[ "$(envelope_version "$store")" == "$STORE_FORMAT_VERSION" ]] || {
        echo "conv-store-backup: self-test FAILED: envelope_version did not read the live store's version" >&2
        status=1
    }

    # The version this script verifies has to be the version the server writes,
    # or every backup of a good store dies at the verify step.
    assert_format_version_agrees
    assert_max_store_bytes_agrees

    # A store the server refuses to load is not a restore candidate, however
    # well-formed it is, so verify has to reject it and restore has to leave
    # the live store alone when it does.
    local oversize="$tmp/oversize.json"
    printf '{"version":1,"conversations":[]}' >"$oversize"
    truncate -s "$((MAX_STORE_BYTES + 1))" "$oversize" 2>/dev/null ||
        dd if=/dev/zero bs=1 count=0 seek="$((MAX_STORE_BYTES + 1))" of="$oversize" 2>/dev/null
    if (verify_store "$oversize") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store over the server's load cap was accepted" >&2
        status=1
    fi
    if (AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_restore "$oversize") >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: a store over the server's load cap was restored" >&2
        status=1
    fi
    cmp -s "$store" "$backup" || {
        echo "conv-store-backup: self-test FAILED: the rejected oversize restore changed the live store" >&2
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
    # retention rather than on the dated-backup count. The overflow sidecar
    # holds a store the server loaded only in part, and the next save
    # destroys the rest, so it travels the same way. The pre-deletion sidecar
    # holds the store as it stood before the API erased a conversation: the
    # live path is rewritten without those messages at once, so until this
    # copy reaches the tier the next scheduled backup is the only other
    # evidence they ever existed.
    cp -- "$store" "$tmp/cache/agave/conversations.json.corrupt"
    cp -- "$store" "$tmp/cache/agave/conversations.json.overflow"
    cp -- "$store" "$tmp/cache/agave/conversations.json.deleted"
    ( KEEP_SNAPSHOT=2; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    rm -f -- "$tmp/cache/agave/conversations.json.corrupt" "$tmp/cache/agave/conversations.json.overflow" "$tmp/cache/agave/conversations.json.deleted"
    [[ "$(find "$tmp/backups" -name 'conversations-corrupt-*.json' | wc -l)" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: dated-backup rotation deleted the quarantined-store copy" >&2
        status=1
    }
    [[ "$(find "$tmp/backups" -name 'conversations-overflow-*.json' | wc -l)" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: the overflow sidecar was not backed up" >&2
        status=1
    }
    # The numbered slot too: the server keeps the first deletion at
    # `.deleted` and the next at `.deleted.1`, and each holds different
    # history, so both have to travel.
    cp -- "$store" "$tmp/cache/agave/conversations.json.deleted.1"
    AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup >/dev/null
    rm -f -- "$tmp/cache/agave/conversations.json.deleted.1"
    [[ "$(find "$tmp/backups" -name 'conversations-deleted-*.json' | wc -l)" -ge 2 ]] || {
        echo "conv-store-backup: self-test FAILED: the pre-deletion sidecars were not both backed up" >&2
        status=1
    }

    # The snapshot tier rotates too, and every copy it drops is state the live
    # path does not hold: a quarantined store is the only remaining copy of a
    # file the server could not parse, an overflow store the only copy of the
    # part a capped load dropped, a pre-restore snapshot the only undo for a
    # restore installed by mistake. Deleting one silently is the worst kind of
    # rotation this script does, so the record has to cover the tier, not only
    # the dated backups where it already does. A dated copy going is one point
    # in time fewer; one of these going is the last of something.
    mkdir -p "$tmp/snapshot-tier"
    local snap_n
    for snap_n in 1 2 3; do
        printf '{"version":1,"active_id":0,"next_id":%d,"conversations":[]}\n' "$snap_n" \
            >"$tmp/snapshot-tier/conversations-corrupt-2026010${snap_n}T000000Z.json"
    done
    local snapshot_log="$tmp/snapshot-tier/.conversations-rotated.log"
    if ! prune_tier "$tmp/snapshot-tier" "$SNAPSHOT_RE" 1 >/dev/null; then
        echo "conv-store-backup: self-test FAILED: prune_tier failed on a tier it should have rotated" >&2
        status=1
    fi
    [[ "$(find "$tmp/snapshot-tier" -name 'conversations-corrupt-*.json' | wc -l)" -eq 1 ]] || {
        echo "conv-store-backup: self-test FAILED: keep=1 left more than one quarantined copy" >&2
        status=1
    }
    if ! grep -q 'pruned snapshot conversations-corrupt-' "$snapshot_log" 2>/dev/null; then
        echo "conv-store-backup: self-test FAILED: snapshot-tier rotation deleted the only copy of a quarantined store without recording it" >&2
        status=1
    fi
    # The record has to survive the rotation that wrote it, or it is no record.
    [[ -f "$snapshot_log" ]] || {
        echo "conv-store-backup: self-test FAILED: snapshot rotation deleted its own record" >&2
        status=1
    }

    # Rotation has to leave a record. A store deleted from the live path is
    # indistinguishable from one the server never had, so without it the dated
    # copies of the deleted conversations are rotated away on schedule and the
    # loss is permanent with nothing left naming what was there. The record is
    # written before the files go, names each removed file, and is never itself
    # rotated.
    ( KEEP=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    local rotation_log="$tmp/backups/.conversations-rotated.log"
    if ! grep -q 'conversations-2[0-9]*T[0-9]*Z\.json' "$rotation_log" 2>/dev/null; then
        echo "conv-store-backup: self-test FAILED: rotation pruned files without recording them" >&2
        status=1
    fi
    # The record is metadata, not another copy of the history, and it must
    # survive the rotation that writes it.
    ( KEEP=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    [[ -f "$rotation_log" ]] || {
        echo "conv-store-backup: self-test FAILED: rotation deleted its own record" >&2
        status=1
    }
    # And a tier emptied by hand is not the same as one that was never primed:
    # `check` has to say so rather than read it as an unstarted job.
    mkdir -p "$tmp/emptied"
    cp -- "$rotation_log" "$tmp/emptied/.conversations-rotated.log"
    if (AGAVE_BACKUP_DIR="$tmp/emptied" XDG_CACHE_HOME="$tmp/cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check passed on a tier emptied by hand" >&2
        status=1
    fi

    # Retention must not reach files this script did not name, so a store
    # dropped into the backup dir by hand survives.
    printf '%s' '{"version":1}' >"$tmp/backups/conversations-manual.json"
    ( KEEP=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup ) >/dev/null
    [[ -f "$tmp/backups/conversations-manual.json" ]] || {
        echo "conv-store-backup: self-test FAILED: retention deleted a file it did not create" >&2
        status=1
    }

    # `check` has to read the snapshot tier, not just count it. A quarantined
    # store the server could not parse is the only copy of that file, so a tier
    # holding one that does not verify is not a recovery path for it. The run
    # stays green and names the file: refusing to say so would hide the only
    # copy there is, which is the same reason `backup` keeps a non-verifying
    # sidecar instead of dropping it.
    printf '%s' '{"version":1,' >"$tmp/backups/conversations-corrupt-20260101T000000Z.json"
    local check_out
    check_out="$(AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check 2>&1)" || {
        echo "conv-store-backup: self-test FAILED: check failed on an unreadable quarantined store" >&2
        status=1
    }
    if [[ "$check_out" != *"conversations-corrupt-20260101T000000Z.json"* ||
        "$check_out" != *"1 of them not verifying"* ]]; then
        echo "conv-store-backup: self-test FAILED: check did not report the unreadable quarantined store" >&2
        status=1
    fi
    # A quarantined store that verifies is counted, not warned about.
    printf '%s' '{"version":1,"active_id":0,"next_id":1,"conversations":[]}' >"$tmp/backups/conversations-corrupt-20260101T000000Z.json"
    check_out="$(AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check 2>&1)"
    if [[ "$check_out" != *"0 of them not verifying"* ]]; then
        echo "conv-store-backup: self-test FAILED: check warned about a quarantined store that verifies" >&2
        status=1
    fi

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
    touch -t "$(backdated_stamp 2)" -- "$tmp/backups"/conversations-2*.json
    if (MAX_AGE_HOURS=1; AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check passed on a 2h-old backup with AGAVE_MAX_AGE_HOURS=1" >&2
        status=1
    fi

    # `check` has to reach the same conclusion `backup` does about a tier on
    # the store's own filesystem. The temp dir is one filesystem, so a tier
    # under the store's directory is that case: the copies are fresh and
    # loadable, and reporting a recovery path here hides a backup job that
    # cannot run at all.
    if (AGAVE_ALLOW_SAME_FS=0 AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check passed with the tier on the store's filesystem" >&2
        status=1
    fi
    # The recovery host has the tier and not the store, so there is no
    # filesystem to compare and a fresh loadable copy is the whole answer.
    if ! (AGAVE_ALLOW_SAME_FS=0 AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/empty-cache" do_check) >/dev/null 2>&1; then
        echo "conv-store-backup: self-test FAILED: check failed with no store present to compare against" >&2
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
        note "self-test passed: backup, verify, reject-truncated, reject-other-version, reject-version-quoted-in-content, reject-quoted-version, reject-missing-version, format-version-agrees, load-cap-agrees, reject-oversize, braces-in-content, restore, pre-restore snapshot, retention, snapshot-tier-retention, snapshot-rotation-record, sidecars, pre-deletion-sidecars, retention-scope, reject-same-filesystem, check-fresh, check-missing, check-stale, check-rejects-shared-filesystem, check-without-a-store, reject-bad-retention, store-override, whole-help"
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
