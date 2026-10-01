#!/usr/bin/env bash
# Print the product SemVer from build.zig.zon.
#
# `LABEL` cannot read a file, so the Dockerfile falls back to the string "dev"
# for org.opencontainers.image.version unless it is passed AGAVE_VERSION as a
# build arg, and build.zig.zon .version is the only authoritative record of what
# a checkout is. CI needs that value before the image build and inside a YAML
# expression, which neither `$(cat ...)` nor a `run:` block may carry without
# putting workflow logic in the shell body; a script is what the rest of this
# tree does for the same reason.
#
# Usage: bash scripts/product-version.sh          # prints the version on stdout
#        eval "AGAVE_VERSION=$(bash scripts/product-version.sh)"
#
# Exit 0 and print the version, or exit 1 naming what could not be read.
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

version="$(sed -n 's/^[[:space:]]*\.version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$ROOT/build.zig.zon" | head -n1)"
if [[ -z "$version" ]]; then
    echo "product-version: could not parse .version from build.zig.zon" >&2
    exit 1
fi
echo "$version"