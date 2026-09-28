#!/usr/bin/env bash
# Reproducibility pins that CI's fmt-check job and `zig build check` both run.
#
# A green `zig build check` must not disagree with CI: these pins
# (Zig toolchain, Zig download checksums, Debian snapshot day,
# SOURCE_DATE_EPOCH, apt source isolation, listen port, ruff version, bun
# version) are what make a build or a gate reproducible, and a mismatch only
# surfaced in CI, so a contributor learned about it after pushing.
#
# Exit 0 when every pin agrees, 1 on a mismatch or an unparseable file.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Midnight UTC of a calendar date, as a Unix epoch, in POSIX shell arithmetic.
#
# `date -u -d <str> +%s` parses the string but is GNU-only: BSD date (macOS, the
# platform the Metal backend and the macOS test job target) has no -d, so it
# exited non-zero and took `zig build check` down with it on a developer Mac.
# Days-from-civil needs no date(1), no locale and no timezone database, so both
# platforms compute the same number. The known-epoch table below is what pins
# the arithmetic; it was cross-checked against GNU date(1) when it was written.
midnight_epoch() {
    # 10# keeps a zero-padded "08" out of bash's octal-integer parsing.
    local year=$((10#$1)) month=$((10#$2)) day=$((10#$3))
    local y era yoe doy doe days
    if ((month <= 2)); then
        y=$((year - 1))
    else
        y=$year
    fi
    era=$(((y >= 0 ? y : y - 399) / 400))
    yoe=$((y - era * 400))
    if ((month > 2)); then
        doy=$(((153 * (month - 3) + 2) / 5 + day - 1))
    else
        doy=$(((153 * (month + 9) + 2) / 5 + day - 1))
    fi
    doe=$((yoe * 365 + yoe / 4 - yoe / 100 + doy))
    days=$((era * 146097 + doe - 719468))
    echo $((days * 86400))
}

# Known epochs, so a typo in the arithmetic above fails here instead of turning
# a correct SOURCE_DATE_EPOCH into a spurious mismatch (or, worse, accepting a
# wrong one). 2026-08-24 is the Debian snapshot day this repo pins.
while read -r want y m d; do
    got="$(midnight_epoch "$y" "$m" "$d")"
    if [[ "$got" != "$want" ]]; then
        echo "check-pins: midnight_epoch $y-$m-$d = $got, expected $want" >&2
        exit 1
    fi
done <<'EOF'
0 1970 01 01
1009843200 2002 01 01
1787529600 2026 08 24
951782400 2000 02 29
EOF
echo "midnight_epoch arithmetic OK"

# Read the pin from .zigversion rather than from a CI step output, so the
# local and CI paths run the identical check.
pin="$(tr -d '[:space:]' < .zigversion)"

zon="$(sed -n 's/^[[:space:]]*\.minimum_zig_version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' build.zig.zon | head -n1)"
if [[ -z "$zon" ]]; then
    echo "check-pins: could not parse minimum_zig_version from build.zig.zon" >&2
    exit 1
fi
if [[ "$pin" != "$zon" ]]; then
    echo "check-pins: .zigversion ($pin) != build.zig.zon minimum_zig_version ($zon)" >&2
    exit 1
fi
echo "Zig pin OK: $pin"

# The Dockerfile verifies the Zig tarball by SHA256, but nothing tied those
# hashes to a Zig release: bumping .zigversion and build.zig.zon left the old
# hashes in place, `zig build check` stayed green, and only the docker-build job
# failed, at the download, with a checksum warning. ZIG_CHECKSUMS_FOR names the
# release the hashes were copied from, so the bump is caught here instead.
checksums_for="$(sed -n 's/^ARG ZIG_CHECKSUMS_FOR=\([^[:space:]]*\).*/\1/p' Dockerfile | head -n1)"
if [[ -z "$checksums_for" ]]; then
    echo "check-pins: could not parse ARG ZIG_CHECKSUMS_FOR from Dockerfile" >&2
    exit 1
fi
if [[ "$checksums_for" != "$pin" ]]; then
    echo "check-pins: Dockerfile ZIG_SHA256_* are for Zig $checksums_for != .zigversion pin $pin" >&2
    echo "check-pins: replace ZIG_CHECKSUMS_FOR and both ZIG_SHA256_* with the values from https://ziglang.org/download/$pin/shasum.txt" >&2
    exit 1
fi
for arch in X86_64 AARCH64; do
    sha="$(sed -n "s/^ARG ZIG_SHA256_${arch}=\([0-9a-f]*\).*/\1/p" Dockerfile | head -n1)"
    if [[ ! "$sha" =~ ^[0-9a-f]{64}$ ]]; then
        echo "check-pins: Dockerfile ZIG_SHA256_${arch} is not a 64-char lowercase sha256: '$sha'" >&2
        exit 1
    fi
done
echo "Zig download checksums OK: $checksums_for (ZIG_SHA256_X86_64, ZIG_SHA256_AARCH64)"

# Dated debian:bookworm-YYYYMMDD-slim must match DEBIAN_SNAPSHOT=YYYYMMDDT...
from_day="$(sed -n 's/.*debian:bookworm-\([0-9]\{8\}\)-slim.*/\1/p' Dockerfile | head -n1)"
snap_day="$(sed -n 's/.*DEBIAN_SNAPSHOT=\([0-9]\{8\}\)T.*/\1/p' Dockerfile | head -n1)"
if [[ -z "$from_day" || -z "$snap_day" ]]; then
    echo "check-pins: could not parse debian FROM tag day and DEBIAN_SNAPSHOT from Dockerfile" >&2
    exit 1
fi
if [[ "$from_day" != "$snap_day" ]]; then
    echo "check-pins: Dockerfile debian day ($from_day) != DEBIAN_SNAPSHOT day ($snap_day)" >&2
    exit 1
fi
epoch="$(sed -n 's/^ARG SOURCE_DATE_EPOCH=\([0-9][0-9]*\).*/\1/p' Dockerfile | head -n1)"
expected_epoch="$(midnight_epoch "${from_day:0:4}" "${from_day:4:2}" "${from_day:6:2}")"
if [[ -z "$epoch" ]]; then
    echo "check-pins: could not parse ARG SOURCE_DATE_EPOCH from Dockerfile" >&2
    exit 1
fi
if [[ "$epoch" != "$expected_epoch" ]]; then
    echo "check-pins: Dockerfile SOURCE_DATE_EPOCH ($epoch) != midnight UTC of debian day $from_day ($expected_epoch)" >&2
    exit 1
fi
echo "Debian snapshot day OK: $from_day (SOURCE_DATE_EPOCH=$epoch)"

# Both stages must drop the base image's deb822 debian.sources, or apt
# keeps resolving against live deb.debian.org and the pin does nothing.
dropped="$(grep -c 'rm -f /etc/apt/sources.list.d/\*\.list /etc/apt/sources.list.d/\*\.sources' Dockerfile || true)"
if [[ "$dropped" != "2" ]]; then
    echo "check-pins: Dockerfile must remove pre-existing apt sources in both stages (found $dropped of 2)" >&2
    exit 1
fi
echo "Apt source isolation OK"

# The image declares its listen port three times (ENV, EXPOSE, HEALTHCHECK) and
# docker-compose.yml a fourth time in its own healthcheck. They must agree: an
# unset AGAVE_PORT with an unbraced $AGAVE_PORT collapses the probe URL to
# http://localhost/ready, so the probe silently checks port 80 and reports the
# container healthy while the server listens elsewhere.
env_port="$(sed -n 's/^ENV AGAVE_PORT=\([0-9][0-9]*\).*/\1/p' Dockerfile | head -n1)"
expose_port="$(sed -n 's/^EXPOSE \([0-9][0-9]*\).*/\1/p' Dockerfile | head -n1)"
# shellcheck disable=SC2016  # the ${...} below is sed's literal, not a shell expansion
probe_port="$(sed -n 's/.*http:\/\/localhost:\$\${AGAVE_PORT:-\([0-9][0-9]*\)}\/ready.*/\1/p' Dockerfile | head -n1)"
# shellcheck disable=SC2016
compose_probe_port="$(sed -n 's/.*AGAVE_PORT:-\([0-9][0-9]*\)}\/ready.*/\1/p' docker-compose.yml | head -n1)"
if [[ -z "$env_port" || -z "$expose_port" || -z "$probe_port" || -z "$compose_probe_port" ]]; then
    echo "check-pins: could not parse the AGAVE_PORT default from Dockerfile (ENV/EXPOSE/HEALTHCHECK) or docker-compose.yml" >&2
    exit 1
fi
if [[ "$env_port$expose_port$probe_port$compose_probe_port" != "$env_port$env_port$env_port$env_port" ]]; then
    echo "check-pins: port mismatch: ENV=$env_port EXPOSE=$expose_port Dockerfile HEALTHCHECK=$probe_port compose healthcheck=$compose_probe_port" >&2
    exit 1
fi
echo "Listen port OK: $env_port (ENV, EXPOSE, both healthchecks)"

# The runtime user is spelled out in four places: the Dockerfile creates it,
# docker-compose.yml runs as it and mounts the tmpfs it owns, and the CI smoke
# test asserts it. They agree today because someone edited all four. Change the
# UID in the Dockerfile alone and CI still goes green (the smoke test reads the
# id out of the image, which is now the new one, against its own literal, which
# fails loudly there) while compose keeps running the server as an account the
# image does not have and its tmpfs mounts land root-owned. That surfaces as an
# EACCES inside a running container, hours later, not as a red check. All four
# must carry the Dockerfile's numbers.
df_uid="$(sed -n 's/^[[:space:]]*.*useradd -r -u \([0-9][0-9]*\).*/\1/p' Dockerfile | head -n1)"
df_gid="$(sed -n 's/^[[:space:]]*.*groupadd -r -g \([0-9][0-9]*\).*/\1/p' Dockerfile | head -n1)"
compose_user="$(sed -n 's/^[[:space:]]*user: "\([0-9][0-9]*\):\([0-9][0-9]*\)".*/\1:\2/p' docker-compose.yml | head -n1)"
compose_tmpfs="$(sed -n 's/.*uid=\([0-9][0-9]*\),gid=\([0-9][0-9]*\).*/\1:\2/p' docker-compose.yml | head -n1)"
ci_uid="$(sed -n 's/.*expected USER agave (uid \([0-9][0-9]*\)).*/\1/p' scripts/check-docker-image.sh | head -n1)"
if [[ -z "$df_uid" || -z "$df_gid" || -z "$compose_user" || -z "$compose_tmpfs" || -z "$ci_uid" ]]; then
    echo "check-pins: could not parse the runtime uid/gid from Dockerfile, docker-compose.yml or scripts/check-docker-image.sh" >&2
    exit 1
fi
if [[ "$compose_user" != "$df_uid:$df_gid" || "$compose_tmpfs" != "$df_uid:$df_gid" || "$ci_uid" != "$df_uid" ]]; then
    echo "check-pins: runtime uid mismatch: Dockerfile useradd=$df_uid groupadd=$df_gid, compose user:=$compose_user, compose tmpfs uid/gid=$compose_tmpfs, check-docker-image.sh=$ci_uid" >&2
    echo "check-pins: the Dockerfile owns the value; update docker-compose.yml (user:, tmpfs) and the smoke test in scripts/check-docker-image.sh in the same change" >&2
    exit 1
fi
echo "Runtime user OK: $df_uid:$df_gid (Dockerfile, compose user:, compose tmpfs, check-docker-image.sh)"

# `zig fmt --check` and `zig build fmt-check` must cover the same files. A path
# added to build.zig's fmt_paths but not to the ci.yml step is formatted locally
# and shipped unformatted; the reverse is a file CI rejects that no local gate
# can reproduce. Neither side can see the other, so compare the two lists here.
# A ci.yml step that stops spelling the command out fails this check rather than
# quietly dropping the comparison.
ci_fmt="$(sed -n 's/^[[:space:]]*run: zig fmt --check \(.*\)$/\1/p' .github/workflows/ci.yml | head -n1)"
build_fmt_paths="$(sed -n 's/^.*const fmt_paths = \[_\]\[\]const u8{ \(.*\) };.*/\1/p' build.zig | head -n1)"
# Directories are spelled with a trailing slash on both sides ("src/" here,
# "src/" on the command line); bare file names must not gain one.
build_fmt="$(printf '%s\n' "$build_fmt_paths" |
    grep -oE '"[^"]+"' | tr -d '"' |
    awk '{ if ($0 ~ /\/$/) print; else print $0 }' |
    tr '\n' ' ' | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//')"
ci_fmt_norm="$(printf '%s\n' "$ci_fmt" | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//')"
if [[ -z "$ci_fmt_norm" || -z "$build_fmt" ]]; then
    echo "check-pins: could not parse the zig fmt path list from ci.yml fmt-check or build.zig fmt_paths" >&2
    echo "check-pins: ci_fmt='$ci_fmt_norm' build_fmt='$build_fmt'" >&2
    exit 1
fi
if [[ "$ci_fmt_norm" != "$build_fmt" ]]; then
    echo "check-pins: zig fmt path mismatch: ci.yml fmt-check checks '$ci_fmt_norm', build.zig fmt_paths checks '$build_fmt'" >&2
    exit 1
fi
echo "Zig fmt paths OK: $ci_fmt_norm (ci.yml fmt-check, build.zig fmt_paths)"

# ruff.toml owns the ruff version: required-version gates every run, and
# scripts/lint-python.sh resolves its uvx fetch from the same file. The CI job
# must call that script rather than spelling out its own invocation, otherwise a
# check added to the script passes locally and never runs remotely.
ruff_pin="$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"==\([^"]*\)".*/\1/p' ruff.toml | head -n1)"
if [[ -z "$ruff_pin" ]]; then
    echo "check-pins: could not parse required-version from ruff.toml" >&2
    exit 1
fi
# shellcheck disable=SC2016  # the ${PIN} below is the literal text being matched, not an expansion
if ! grep -qF 'ruff@${PIN}' scripts/lint-python.sh; then
    echo "check-pins: scripts/lint-python.sh must fetch the pinned ruff from ruff.toml (uvx ruff@\${PIN})" >&2
    exit 1
fi
if ! grep -q 'run: bash scripts/lint-python.sh' .github/workflows/ci.yml; then
    echo "check-pins: ci.yml lint-python must run 'bash scripts/lint-python.sh' (the single entry point)" >&2
    exit 1
fi
echo "Ruff pin OK: $ruff_pin (ruff.toml, scripts/lint-python.sh, ci.yml lint-python)"

# package.json packageManager is what scripts/lint-web.sh gates a local run on;
# ci.yml names the same version for setup-bun. If the two drift, CI lints with a
# different bun than the one a developer's `zig build lint-web` verified.
bun_pin="$(sed -n 's/.*"packageManager": "bun@\([^"]*\)".*/\1/p' package.json | head -n1)"
ci_bun_pin="$(sed -n 's/.*bun-version: "\([^"]*\)".*/\1/p' .github/workflows/ci.yml | head -n1)"
if [[ -z "$bun_pin" || -z "$ci_bun_pin" ]]; then
    echo "check-pins: could not parse packageManager bun@<version> from package.json and bun-version from ci.yml" >&2
    exit 1
fi
if [[ "$bun_pin" != "$ci_bun_pin" ]]; then
    echo "check-pins: package.json packageManager (bun@$bun_pin) != ci.yml setup-bun bun-version ($ci_bun_pin)" >&2
    exit 1
fi
engines_bun="$(sed -n 's/^[[:space:]]*"bun": "\([^"]*\)".*/\1/p' package.json | head -n1)"
if [[ "$engines_bun" != "$bun_pin" ]]; then
    echo "check-pins: package.json engines.bun ($engines_bun) != packageManager (bun@$bun_pin)" >&2
    exit 1
fi
echo "Bun pin OK: $bun_pin (packageManager, engines.bun, ci.yml setup-bun)"

# Vendored third-party source has to be traceable too. tools/oxlint/anti-slop is
# a copy of dmmulroy/anti-slop, and the upstream commit it was copied from was
# never recorded, so the sha256 manifest is the only provenance anchor the tree
# has: a rule that no longer hashes to its recorded value is a local edit wearing
# third-party clothing, and the lint gate would load it as if it were upstream.
vendored_dir="tools/oxlint/anti-slop"
vendored_manifest="$vendored_dir/VENDORED.sha256"
if command -v sha256sum >/dev/null 2>&1; then
    sha_cmd=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    # macOS ships no coreutils sha256sum; shasum is the same tool by another name.
    sha_cmd=(shasum -a 256)
else
    sha_cmd=()
fi
if [[ ${#sha_cmd[@]} -eq 0 ]]; then
    echo "check-pins: no sha256sum or shasum on PATH, skipping the vendored source manifest check"
elif [[ ! -f "$vendored_manifest" ]]; then
    echo "check-pins: $vendored_manifest is missing; the vendored anti-slop copy must stay hash-anchored" >&2
    exit 1
else
    vendored_fail=0
    while read -r want rel; do
        [[ -n "$want" && -n "$rel" ]] || continue
        if [[ ! -f "$vendored_dir/$rel" ]]; then
            echo "check-pins: $vendored_manifest lists $rel, which is no longer in the tree" >&2
            vendored_fail=1
            continue
        fi
        got="$("${sha_cmd[@]}" "$vendored_dir/$rel" | cut -d' ' -f1)"
        if [[ "$got" != "$want" ]]; then
            echo "check-pins: $rel is not the vendored copy $vendored_manifest records ($got != $want)" >&2
            vendored_fail=1
        fi
    done <"$vendored_manifest"
    # The other direction: an unlisted .ts file loads as a rule nobody anchored.
    while read -r rel; do
        if ! grep -qF -- "  $rel" "$vendored_manifest"; then
            echo "check-pins: $rel is not in $vendored_manifest; re-vendor and regenerate it" >&2
            vendored_fail=1
        fi
    done < <(cd "$vendored_dir" && find . -type f -name '*.ts' | sed 's|^\./||' | LC_ALL=C sort)
    if [[ "$vendored_fail" -ne 0 ]]; then
        exit 1
    fi
    echo "Vendored source OK: $vendored_dir matches $vendored_manifest"
fi

# Every pyproject.toml that ships a uv.lock must have that lock agree with its
# pins. A lock written before a pin tightened (research/kernels/ once recorded
# "numpy" and "torch" with no specifier) resolves to versions the manifest no
# longer allows, and `uv sync --frozen` then installs what the lock says rather
# than what the manifest pins. `uv lock --check` re-resolves from the lock, so
# it needs no network. Dirs with no lock (research/kernels/tilelang installs
# torch from a hardware-specific index) are out of scope: nothing to compare.
if ! command -v uv >/dev/null 2>&1; then
    echo "check-pins: uv not on PATH, skipping the uv.lock freshness check"
else
    for dir in tests research/kernels; do
        [[ -f "$dir/pyproject.toml" && -f "$dir/uv.lock" ]] || continue
        if ! (cd "$dir" && uv lock --check >/dev/null 2>&1); then
            echo "check-pins: $dir/uv.lock disagrees with $dir/pyproject.toml; run 'uv lock' in $dir" >&2
            exit 1
        fi
    done
    echo "uv lock OK: tests/uv.lock, research/kernels/uv.lock match their pyproject.toml"
fi

# Every third-party requirement names one version. A range ("torch>=2.10",
# "oxlint@^1.57.0", "~1.2") resolves to a different tree on every resolve, so
# the lock stops describing what anyone reviewed, and a compromised or merely
# surprising release lands without a manifest diff. Exact pins move by a
# deliberate commit (Dependabot opens one), which is the record the audit trail
# is made of. Both manifest dialects express that: npm/bun write a bare version
# ("oxlint": "1.57.0"), PEP 508 writes "==". Extras and environment markers are
# compared on the specifier text, not resolved.
#
# A `file:`/`link:` spec is exempt: it names a directory in this repository
# (`vendor/react` and `vendor/react-dom`, the preact/compat shims), so it is
# pinned by the commit, not by a version. Everything from a registry is checked.
exact_pin_fail=0
check_exact_pins() {
    local file=$1
    local spec
    while IFS= read -r spec; do
        [[ -n "$spec" ]] || continue
        if [[ "$spec" == file:* || "$spec" == link:* ]]; then
            continue
        fi
        if [[ ! "$spec" =~ ^[A-Za-z0-9._-]+(==)?[0-9][A-Za-z0-9._+-]*(\[[^]]+\])?(\;.*)?$ ]]; then
            echo "check-pins: $file: '$spec' does not pin one exact version" >&2
            exact_pin_fail=1
        fi
    done
}

# package.json dependencies and devDependencies (npm/bun semver) and
# pyproject dependencies (PEP 508) sit behind different syntaxes, so each is
# listed with its own parser rather than a regex that would have to cover both.
# Both package.json blocks are checked: a range in the production block reaches
# the shipped bundle. The range end is a bare closing brace, which no entry line
# ("name": "version") can contain.
for block in dependencies devDependencies; do
    check_exact_pins "package.json ${block}" < <(
        sed -n "/\"${block}\"/,/^[[:space:]]*}/p" package.json |
            sed -n 's/^[[:space:]]*"[^"]*"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
    )
done

# Every quoted requirement token inside a dependency array, optional extras
# and markers included. Only the arrays are read, so the quoted values of other
# fields (name, description, requires-python) cannot be mistaken for a pin.
# shellcheck disable=SC2016  # awk's $0 and /"[^"]+"/ below are the awk language, not the shell
check_exact_pins "pyproject dependencies" < <(
    # -exec ... + rather than -print0 | xargs -0: `-0` is a GNU xargs option
    # that BSD/macOS xargs rejects, and this runs on a macOS workstation too.
    find tests research -name pyproject.toml -not -path '*/.venv/*' -exec awk '
            /^[[:space:]]*(dependencies|\[project\.optional-dependencies\])/ { in_deps = 1 }
            in_deps {
                while (match($0, /"[^"]+"/)) {
                    print substr($0, RSTART + 1, RLENGTH - 2)
                    $0 = substr($0, RSTART + RLENGTH)
                }
                if ($0 ~ /\]/) { in_deps = 0 }
            }
        ' {} +
)

if ((exact_pin_fail)); then
    echo "check-pins: every third-party requirement must pin one exact version" >&2
    exit 1
fi
echo "Dependency pins OK: package.json dependencies, devDependencies and pyproject requirements are exact"
