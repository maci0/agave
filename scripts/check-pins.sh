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

# ruff.toml owns the ruff version (required-version gates every local and CI
# run); the CI job names the same version so the uvx fetch cannot drift.
ruff_pin="$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"==\([^"]*\)".*/\1/p' ruff.toml | head -n1)"
ci_ruff_pin="$(sed -n 's/.*uvx ruff@\([^ ]*\) check.*/\1/p' .github/workflows/ci.yml | head -n1)"
if [[ -z "$ruff_pin" || -z "$ci_ruff_pin" ]]; then
    echo "check-pins: could not parse required-version from ruff.toml and uvx ruff@<version> from ci.yml" >&2
    exit 1
fi
if [[ "$ruff_pin" != "$ci_ruff_pin" ]]; then
    echo "check-pins: ruff.toml required-version (==$ruff_pin) != ci.yml uvx ruff@$ci_ruff_pin" >&2
    exit 1
fi
echo "Ruff pin OK: $ruff_pin"

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
