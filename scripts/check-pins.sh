#!/usr/bin/env bash
# Reproducibility pins that CI's fmt-check job and `zig build check` both run.
#
# A green `zig build check` must not disagree with CI: these four pins
# (Zig toolchain, Debian snapshot day, SOURCE_DATE_EPOCH, apt source
# isolation) are what make a Docker build reproducible, and a mismatch only
# surfaced in CI, so a contributor learned about it after pushing.
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
