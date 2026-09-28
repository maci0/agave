#!/usr/bin/env bash
# Smoke test the runtime image: the CLI runs, the container is not root, the
# healthcheck probes the ready endpoint, the GPL license ships, HOME is set,
# the OCI license label is right, and the binary is static musl.
# Canonical: bash scripts/check-docker-image.sh [image] (CI docker-build job).
#
# Kept in scripts/ rather than inline in the workflow so the blocking
# `zig build lint-shell` analyses the assertions; each of them is an `if` whose
# failure is the only thing standing between a broken image and a release.
set -euo pipefail
export LC_ALL=C TZ=UTC

image="${1:-agave:ci}"
copyright_path="/usr/share/doc/agave/copyright"

if ! command -v docker >/dev/null 2>&1; then
    echo "check-docker-image: docker not found. Install Docker, then rerun this script." >&2
    exit 1
fi

docker run --rm --entrypoint agave "$image" --help

# Runtime image must not run as root (Dockerfile USER agave, uid 10001).
uid="$(docker run --rm --entrypoint id "$image" -u)"
if [[ "$uid" != "10001" ]]; then
    echo "::error::Container uid=$uid; expected USER agave (uid 10001)"
    exit 1
fi
echo "Container uid=$uid (non-root ok)"

# HEALTHCHECK must probe /ready (not /health) so orchestrators do not route to
# degraded instances. One-shot CLI runs should pass --no-healthcheck.
hc="$(docker inspect --format='{{json .Config.Healthcheck.Test}}' "$image")"
if [[ "$hc" != *"/ready"* ]]; then
    echo "::error::Image HEALTHCHECK missing /ready probe: $hc"
    exit 1
fi
echo "HEALTHCHECK ok: $hc"

# GPL requires the license to accompany the binary.
if ! docker run --rm --entrypoint cat "$image" "$copyright_path" | grep -q "GNU GENERAL PUBLIC LICENSE"; then
    echo "::error::Runtime image missing GPL license at $copyright_path"
    exit 1
fi
echo "LICENSE ok: $copyright_path"

# The embedded web UI is a minified bundle, so its upstream license headers
# exist nowhere in the image. The notices file is the only record of the grant
# for the Preact and Radix code inside it.
notices_path="/usr/share/doc/agave/third-party-notices.md"
if ! docker run --rm --entrypoint cat "$image" "$notices_path" | grep -q 'lucide-react@'; then
    echo "::error::Runtime image missing third-party notices at $notices_path"
    exit 1
fi
echo "Third-party notices ok: $notices_path"

# The man page ships with the binary and has to match `agave --help`.
if ! docker run --rm --entrypoint cat "$image" /usr/share/man/man1/agave.1 | grep -q '^\.TH AGAVE 1'; then
    echo "::error::Runtime image missing the man page at /usr/share/man/man1/agave.1"
    exit 1
fi
echo "Man page ok: /usr/share/man/man1/agave.1"

# HOME must be set in the image config (k8s/podman do not copy passwd HOME).
home="$(docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' "$image" | sed -n 's/^HOME=//p')"
if [[ "$home" != "/home/agave" ]]; then
    echo "::error::Image HOME='$home'; expected /home/agave"
    exit 1
fi
echo "HOME ok: $home"

lic="$(docker inspect --format='{{index .Config.Labels "org.opencontainers.image.licenses"}}' "$image")"
if [[ "$lic" != "GPL-3.0-or-later" ]]; then
    echo "::error::OCI licenses label is '$lic'; expected GPL-3.0-or-later"
    exit 1
fi
echo "OCI licenses ok: $lic"

# All dlopen backends disabled in the build above, so the Dockerfile must
# select a static musl binary. ldd exits nonzero on a static binary, which is
# the expected path, not a failure.
ldd_out="$(docker run --rm --entrypoint ldd "$image" /usr/local/bin/agave 2>&1 || true)"
echo "$ldd_out"
if echo "$ldd_out" | grep -qiE '=>|libc\.so'; then
    echo "::error::Expected static musl binary when CUDA/Vulkan/ROCm/WebGPU are disabled"
    exit 1
fi
if ! echo "$ldd_out" | grep -qiE 'not a dynamic|statically linked'; then
    echo "::error::Unexpected ldd output (expected static binary): $ldd_out"
    exit 1
fi
echo "Static musl binary ok"
