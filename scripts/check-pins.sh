#!/usr/bin/env bash
# Reproducibility pins that CI's fmt-check job and `zig build check` both run.
#
# A green `zig build check` must not disagree with CI: these pins
# (Zig toolchain, Debian snapshot day, SOURCE_DATE_EPOCH, apt source
# isolation, listen port, ruff version, bun version) are what make a build or a
# gate reproducible, and a mismatch only surfaced in CI, so a contributor
# learned about it after pushing.
#
# Exit 0 when every pin agrees, 1 on a mismatch or an unparseable file.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
expected_epoch="$(date -u -d "${from_day:0:4}-${from_day:4:2}-${from_day:6:2} 00:00:00 UTC" +%s)"
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
