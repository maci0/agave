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
#   scripts/conv-store-backup.sh --self-test          # exercise all of the above
#
# Environment:
#   AGAVE_BACKUP_DIR   backup destination (default: $HOME/.agave-backups)
#   AGAVE_KEEP         backups to keep, oldest pruned first (default: 14)
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

# Structural check: the server only needs a non-empty object carrying the
# expected envelope version. Full parse validation is the server's job on
# restart (src/server/conv_store.zig load), and a file that fails it is
# quarantined to {path}.corrupt rather than dropped, so nothing is destroyed
# by a false negative here.
verify_store() {
    local file="$1"
    [[ -f "$file" ]] || die "not a file: $file"
    [[ -s "$file" ]] || die "empty file: $file"
    [[ "$(head -c 1 "$file")" == "{" ]] || die "not a JSON object: $file"
    grep -Eq '"version":[[:space:]]*'"$STORE_FORMAT_VERSION"'([[:space:]]*[,}])' "$file" ||
        die "no \"version\":$STORE_FORMAT_VERSION envelope in $file (written by a different format version; inspect before restoring)"
    # Balanced braces catches the truncation that a single missing byte causes.
    local open close
    open=$(tr -cd '{' <"$file" | wc -c)
    close=$(tr -cd '}' <"$file" | wc -c)
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
    dest="$dir/conversations-$(stamp).json"
    [[ -e "$dest" ]] && dest="$dir/conversations-$(stamp)-$$.json"
    copy_atomic "$live" "$dest"
    verify_store "$dest"
    # The quarantine copy is the only remaining trace of a store the server
    # could not parse. Losing it loses the recoverable data.
    local corrupt="$live.corrupt"
    if [[ -f "$corrupt" ]]; then
        copy_atomic "$corrupt" "$dir/conversations-corrupt-$(stamp).json"
        verify_store "$dir/conversations-corrupt-$(stamp).json" ||
            note "kept $dir/conversations-corrupt-$(stamp).json even though it does not verify; it is the only copy"
        note "backed up quarantined store $corrupt"
    fi
    prune "$dir"
    note "backed up $live -> $dest"
}

# Prune oldest first, never below one backup. A retention job that can reach
# zero backups is a deletion path with no recovery window.
prune() {
    local dir="$1"
    local -a keep_files
    mapfile -t keep_files < <(find "$dir" -maxdepth 1 -type f -name 'conversations-*.json' -printf '%T@ %p\n' |
        sort -rn | cut -d' ' -f2-)
    (( ${#keep_files[@]} <= KEEP )) && return 0
    local i
    for ((i = KEEP; i < ${#keep_files[@]}; i++)); do
        rm -f -- "${keep_files[i]}" || die "prune failed: ${keep_files[i]}"
        note "pruned old backup ${keep_files[i]}"
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
        copy_atomic "$live" "$snap"
        note "snapshotted the live store to $snap"
    fi
    mkdir -p -- "$(dirname -- "$live")" || die "cannot create $(dirname -- "$live")"
    copy_atomic "$file" "$live"
    verify_store "$live" || die "restored copy at $live does not verify; the backup file is untouched"
    note "restored $file -> $live"
    note "restart the server (docker compose restart agave) to load it; a store it cannot parse is quarantined, not dropped"
}

do_self_test() {
    local tmp status=0
    tmp="$(mktemp -d)"
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

    # Retention must leave at least one backup and prune the rest.
    AGAVE_KEEP=1 AGAVE_BACKUP_DIR="$tmp/backups" XDG_CACHE_HOME="$tmp/cache" do_backup >/dev/null
    local remaining
    remaining=$(find "$tmp/backups" -name 'conversations-2*.json' | wc -l)
    [[ "$remaining" -ge 1 ]] || {
        echo "conv-store-backup: self-test FAILED: retention left no backup" >&2
        status=1
    }

    if (( status == 0 )); then
        note "self-test passed: backup, verify, reject-truncated, restore, pre-restore snapshot, retention"
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
        --self-test) do_self_test ;;
        -h | --help | '') usage ;;
        *) die "unknown command '$1' (path, backup, verify, restore, --self-test)" ;;
    esac
}

main "$@"
