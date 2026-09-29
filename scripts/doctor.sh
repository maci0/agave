#!/usr/bin/env bash
# Report whether this machine can run the local gate, and name every tool it
# is missing, before `zig build ci` fails halfway through on the first gap.
# Canonical: zig build doctor  (or this script from the repo root).
#
# A probe, not a gate: it always exits 0 and prints its verdict on the last
# line, so it can be run on any machine, including one that is deliberately
# only part-provisioned. Version pins are read from the files the gates read
# (.zigversion, package.json, ruff.toml) rather than repeated here, so a pin
# bump cannot leave this script reporting a stale requirement.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

missing=0

# ok <tool> <detail> / gap <tool> <how to get it>
ok() { printf 'ok      %-10s %s\n' "$1" "$2"; }
gap() {
    printf 'MISSING %-10s %s\n' "$1" "$2"
    missing=$((missing + 1))
}
optional() { printf 'n/a     %-10s %s\n' "$1" "$2"; }

# Zig, at the exact pin build.zig refuses to build against.
zig_pin="$(tr -d '[:space:]' < .zigversion)"
zig_got="$(zig version 2>/dev/null || true)"
if [[ "$zig_got" == "$zig_pin" ]]; then
    ok zig "$zig_got (pin $zig_pin)"
else
    gap zig "found '${zig_got:-none}', need $zig_pin (https://ziglang.org/download/)"
fi

# Python, for scripts/check-docs.py, check-third-party-notices.py and the
# unit suites. The same 3.11 floor check-docs.py enforces on itself.
if command -v python3 >/dev/null 2>&1; then
    py_ok="$(python3 -c 'import sys; print(sys.version_info >= (3, 11))' 2>/dev/null || echo False)"
    if [[ "$py_ok" == "True" ]]; then
        ok python3 "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') (need 3.11+)"
    else
        gap python3 "$(python3 -c 'import sys; print(sys.version.split()[0])') is older than 3.11"
    fi
else
    gap python3 "not on PATH (zig build check needs 3.11+)"
fi

# bun, at the pin lint-web.sh refuses to run against, plus the installed deps
# it then requires.
if command -v bun >/dev/null 2>&1; then
    bun_pin="$(sed -n 's/.*"packageManager": "bun@\([^"]*\)".*/\1/p' package.json | head -n1)"
    bun_got="$(bun --version 2>/dev/null || true)"
    if [[ -z "$bun_pin" ]]; then
        gap bun "could not parse packageManager bun@X.Y.Z from package.json"
    elif [[ "$bun_got" != "$bun_pin" ]]; then
        gap bun "found $bun_got, need $bun_pin (package.json packageManager)"
    elif [[ -x node_modules/.bin/oxlint && -x node_modules/.bin/tsc ]]; then
        ok bun "$bun_got (node_modules installed)"
    else
        gap bun "$bun_got installed but node_modules is missing: bun install --frozen-lockfile"
    fi
else
    gap bun "not on PATH (zig build lint-web needs it; pin: package.json packageManager)"
fi

# Java runs vnu (the vnu-jar dev dependency) for scripts/check-w3c.sh.
if command -v java >/dev/null 2>&1; then
    ok java "$(java -version 2>&1 | head -n1) (vnu for check-w3c)"
else
    gap java "not on PATH (zig build lint-web runs vnu, which needs Java 11+)"
fi

# Static shell analysis. lint-shell.sh owns the 0.9.0 floor and fails on a
# mismatch, so this only reports what is installed.
if command -v shellcheck >/dev/null 2>&1; then
    ok shellcheck "$(shellcheck --version | sed -n '2s/^version: //p') (floor enforced by lint-shell)"
else
    gap shellcheck "not on PATH (zig build lint-shell needs it)"
fi

# ruff, resolved the way lint-python.sh resolves it: a PATH ruff already at
# the pin, else uvx, else a documented install.
ruff_pin="$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"==\([^"]*\)".*/\1/p' ruff.toml | head -n1)"
if [[ -z "$ruff_pin" ]]; then
    gap ruff "could not parse required-version \"==<version>\" from ruff.toml"
elif command -v ruff >/dev/null 2>&1 && [[ "$(ruff --version 2>/dev/null)" == "ruff ${ruff_pin}" ]]; then
    ok ruff "$ruff_pin (on PATH)"
elif command -v uvx >/dev/null 2>&1; then
    ok ruff "$ruff_pin via uvx (no network install needed at lint time)"
else
    gap ruff "no ruff $ruff_pin on PATH and no uvx: uv tool install ruff==${ruff_pin}"
fi

# Tools only the CI-only jobs need. Listed so the list is complete, never
# counted against the verdict.
for tool in docker glslangValidator jq gh; do
    if command -v "$tool" >/dev/null 2>&1; then
        optional "$tool" "present (only some CI jobs need it)"
    else
        optional "$tool" "absent (only some CI jobs need it)"
    fi
done

if ((missing == 0)); then
    echo "RESULT: ready for \`zig build ci\`"
else
    echo "RESULT: ${missing} tool(s) missing; \`zig build check\` alone needs only Zig and Python 3.11+"
fi
