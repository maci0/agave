#!/usr/bin/env bash
# Run the tests of one source file, for the edit-test loop.
#
# `zig build test -Dtest-filter=<name>` filters the whole suite by test *name*,
# which compiles src/main.zig and every other test artifact and then runs the
# matching tests. It is the only single-test loop this build had, and it is the
# wrong shape for the common case: the file you are editing does not tell you
# which tests to run, so the answer comes back as a substring you have to grep
# for first, and every run of the suite is paid for no matter how narrow the
# match.
#
# `zig test src/ops/kv_quant.zig`, the obvious thing to try instead, fails for
# most of src/:
#
#     src/ops/quant.zig:7:23: error: import of file outside module path
#     const DType = @import("../format/dtype.zig").DType;
#
# A Zig module reaches only the files under its root source file's directory,
# and src/ is written as a tree of relative `@import("../sibling.zig")` calls,
# so a module rooted at src/ops/ cannot see src/format/. `src/test_exports.zig`
# is the repo's existing answer to that, a bridge module rooted directly in
# src/ so tests/ can reach the backend types.
#
# This writes the same kind of bridge: one throwaway root in src/ that imports
# the file asked for, which pulls in that file's tests and everything it
# imports, runs `zig test` on it, and removes it. The bridge never outlives the
# command, including on a failure or a signal, so an interrupted run cannot
# leave a file that the next `zig build` would compile into the agave binary.
#
# Why no --test-filter passthrough: Zig 0.16's `zig test --test-filter` matches
# only the root module's own test declarations, and the tests of an imported
# file do not match it (verified against Zig 0.16.0: with the bridge in place,
# every filter returns "All 0 tests passed."). The tests here are therefore the
# file's own plus the ones in the files it imports, all of them run. A name
# filter that matches nothing is rejected up front instead of being passed on
# to report a green run that tested nothing, which is the same trap
# `rejectEmptyTestFilters` guards in build.zig for -Dtest-filter.
#
# Usage:
#   scripts/test-file.sh src/ops/kv_quant.zig              # every test in the file
#   scripts/test-file.sh src/ops/kv_quant.zig wht32        # refuse unless a test matches
#
# Ceiling: this roots a module at src/, so it works for a file under src/ (every
# file that has inline tests today). A file outside src/ is already its own
# root and needs no bridge: `zig test tests/models/test_gemma3.zig --test-filter CPU`.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    echo "usage: scripts/test-file.sh <src/file.zig> [expected-test-name-substring ...]" >&2
    echo "  e.g. scripts/test-file.sh src/ops/kv_quant.zig" >&2
}

if (($# < 1)); then
    usage
    exit 2
fi

target="$1"
shift

if ! command -v zig >/dev/null 2>&1; then
    echo "test-file: zig not on PATH. Install Zig $(tr -d '[:space:]' < .zigversion) (https://ziglang.org/download/)." >&2
    exit 1
fi
zig_pin="$(tr -d '[:space:]' < .zigversion)"
zig_got="$(zig version)"
if [[ "$zig_got" != "$zig_pin" ]]; then
    echo "test-file: zig $zig_got on PATH, this build needs $zig_pin (pin: .zigversion)." >&2
    exit 1
fi

case "$target" in
    src/*.zig) ;;
    *)
        echo "test-file: '$target' is not a file directly under src/." >&2
        echo "  A Zig module only reaches files under its root's directory, so the bridge" >&2
        echo "  has to sit in src/. Tests outside src/ are their own root already:" >&2
        echo "    zig test tests/models/test_gemma3.zig --test-filter CPU" >&2
        usage
        exit 2
        ;;
esac

if [[ ! -f "$target" ]]; then
    echo "test-file: no such file: $target" >&2
    echo "  Files that carry inline tests: rg -l '^\s*test \"' src/" >&2
    exit 2
fi

# A file with no tests is almost always the wrong path, and running the bridge
# anyway reports the tests of whatever it imports, which reads as a pass.
if ! grep -qE '^[[:space:]]*test[[:space:]]+"' "$target"; then
    echo "test-file: $target declares no tests of its own." >&2
    echo "  The bridge runs the tests of everything this file imports, so a file" >&2
    echo "  with none would report another file's results under this name." >&2
    exit 2
fi

# Substrings the caller expects to match. Checked here because `zig test
# --test-filter` cannot see an imported file's tests, so passing them on would
# report "All 0 tests passed." and exit 0.
for filter in "$@"; do
    if ! grep -qE "^[[:space:]]*test[[:space:]]+\"[^\"]*${filter}" "$target"; then
        echo "test-file: no test in $target matches '$filter'." >&2
        echo "  A filter that matches nothing exits 0, so the run would be green" >&2
        echo "  without testing anything. The names in this file:" >&2
        grep -nE '^[[:space:]]*test[[:space:]]+"' "$target" >&2
        exit 1
    fi
done

# src/relative/import/path.zig -> "relative/import/path.zig", the form the
# bridge writes. src/ has no @import of a file above it, so one directory level
# down is the deepest a target can be.
relative="${target#src/}"
parent="${relative%/*}"
if [[ "$parent" == "$relative" ]]; then
    import_path="$relative"
else
    import_path="$parent/${relative##*/}"
fi

# A fixed name, so two runs cannot collide and so a stale file from an
# interrupted run is overwritten rather than silently shadowing the real root.
bridge="src/zz_test_file_root.zig"
if [[ -e "$bridge" ]]; then
    echo "test-file: $bridge already exists; refusing to overwrite it." >&2
    echo "  Only this script writes that path. Delete it if no run is in flight." >&2
    exit 1
fi

# 13 files in src/ `@import("build_options")`: the compile-time enable-* flags
# and version the build system generates, so a standalone `zig test` has no such
# module and fails with "no module named 'build_options' available". The build
# writes exactly one options.zig under .zig-cache/c/, and it is what the suite
# itself compiles against, so reuse it rather than inventing a second copy that
# could disagree with build.zig. A fresh clone with no build yet has none; say
# so instead of failing on a name the contributor did not write.
options_module=""
for candidate in .zig-cache/c/*/options.zig; do
    [[ -f "$candidate" ]] || continue
    options_module="$candidate"
    break
done
if [[ -z "$options_module" ]]; then
    echo "test-file: this file needs the build_options module, and .zig-cache has none yet." >&2
    echo "  Run \`zig build\` once (it writes .zig-cache/c/*/options.zig), then rerun." >&2
    echo "  Or reach the same tests through the build, which wires it for you:" >&2
    echo "    zig build test -Dtest-filter=<name>" >&2
    exit 1
fi

# The import has to be inside a `test` body or a `comptime` block: at file
# scope a bare `const _ = @import(...)` is an unused declaration, and Zig only
# includes the tests of files referenced from *test* or comptime context. With
# the import in a comptime block, the file's 28 tests come along (verified on
# Zig 0.16.0 against src/ops/kv_quant.zig: 52 tests, the file's own plus the
# ones in src/ops/quant.zig and src/format/dtype.zig it imports).
printf 'comptime {\n    _ = @import("%s");\n}\n' "$import_path" > "$bridge"

# Remove on every exit, including an interrupt, so the tree is never left with
# a file `zig build` would compile into the binary.
cleanup() { rm -f "$bridge"; }
trap cleanup EXIT INT TERM

if (($# > 0)); then
    echo "test-file: $target (expected to match: $*)"
else
    echo "test-file: $target"
fi
# ReleaseSafe for the same reason `zig build test` uses it: Debug optimize mode
# breaks linking with GCC 16 .sframe, and ReleaseFast would no-op every
# std.debug.assert in the file. libc is linked because the main suite links it
# and anything reaching std.Thread or std.posix needs it.
#
# `zig test` leaves stderr as it is: the test runner talks over it, so the
# per-test lines and the "All N tests passed." summary land there, not on
# stdout. Nothing is piped through a filter here for the same reason - a pipe
# would swallow the exit status the caller checks, and a failing run would
# read as a pass.
zig test -OReleaseSafe -lc \
    --dep build_options -Mroot="$bridge" -Mbuild_options="$options_module"