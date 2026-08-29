# syntax=docker/dockerfile:1
# Pin to a dated Debian tag for reproducible builds (bump with dependabot/docker).
FROM --platform=$BUILDPLATFORM debian:bookworm-20260824-slim AS build

# Empty default: install the version pinned in .zigversion (single source of truth).
# Override with --build-arg ZIG_VERSION=x.y.z and matching ZIG_SHA256_* args.
ARG ZIG_VERSION=
ARG TARGETARCH

# Freeze apt to the same calendar day as the FROM tag. A dated image alone is not
# enough: `apt-get update` against deb.debian.org still floats package versions.
# Bump this when bumping debian:bookworm-YYYYMMDD-slim (CI checks they match).
ARG DEBIAN_SNAPSHOT=20260713T000000Z
# 2026-07-13 00:00:00 UTC; keep aligned with DEBIAN_SNAPSHOT / FROM tag day.
ARG SOURCE_DATE_EPOCH=1783900800

# Backend enable flags. Metal disabled by default (macOS-only, not usable in Docker).
ARG ENABLE_CPU=true
ARG ENABLE_METAL=false
ARG ENABLE_VULKAN=true
ARG ENABLE_CUDA=true
ARG ENABLE_ROCM=true
ARG ENABLE_WEBGPU=true
# Debug binary is not copied into the runtime image; skip compiling it.
ARG ENABLE_DEBUG=false

# Model enable flags (all enabled by default). Disable to reduce binary size.
ARG ENABLE_GEMMA3=true
ARG ENABLE_QWEN35=true
ARG ENABLE_GPT_OSS=true
ARG ENABLE_NEMOTRON_H=true
ARG ENABLE_NEMOTRON_NANO=true
ARG ENABLE_GLM4=true
ARG ENABLE_GEMMA4=true
ARG ENABLE_DIFFUSION_GEMMA=true
ARG ENABLE_LLAMA4=true

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C \
    TZ=UTC \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bookworm main\n' "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list; \
    printf 'deb http://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/security.list; \
    printf 'Acquire::Check-Valid-Until "false";\nAcquire::Retries "3";\n' > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl xz-utils ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Zig toolchain checksums (SHA256). Update when bumping .zigversion / ZIG_VERSION.
ARG ZIG_SHA256_X86_64=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
ARG ZIG_SHA256_AARCH64=ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17

# Product SemVer for the OCI version label. Empty: derived from build.zig.zon
# (.version, single source of truth); an explicit override must match it or the
# build fails below. README shows how to stamp it from build.zig.zon.
ARG AGAVE_VERSION=

# Read pin before full COPY so the toolchain layer stays cacheable.
COPY --link .zigversion /tmp/agave.zigversion

RUN ZIG_VER="${ZIG_VERSION:-$(tr -d '[:space:]' </tmp/agave.zigversion)}" && \
    ARCH=$(uname -m) && \
    ZIG_URL="https://ziglang.org/download/${ZIG_VER}/zig-${ARCH}-linux-${ZIG_VER}.tar.xz" && \
    if [ "$ARCH" = "x86_64" ]; then EXPECTED_SHA256="$ZIG_SHA256_X86_64"; \
    elif [ "$ARCH" = "aarch64" ]; then EXPECTED_SHA256="$ZIG_SHA256_AARCH64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    echo "Installing Zig ${ZIG_VER} (${ARCH})" && \
    curl -fsSL "$ZIG_URL" -o /tmp/zig.tar.xz && \
    echo "${EXPECTED_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /usr/local/zig && \
    tar -xJf /tmp/zig.tar.xz -C /usr/local/zig --strip-components=1 && \
    ln -s /usr/local/zig/zig /usr/local/bin/zig && \
    rm /tmp/zig.tar.xz

WORKDIR /src
COPY --link . .

# Default builds must match .zigversion and build.zig.zon. Explicit ZIG_VERSION
# overrides are for experiments only (also bump ZIG_SHA256_* when overriding).
RUN PIN=$(tr -d '[:space:]' < .zigversion) && \
    ZON=$(sed -n 's/^[[:space:]]*\.minimum_zig_version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' build.zig.zon | head -n1) && \
    INSTALLED=$(zig version) && \
    if [ -z "$ZON" ]; then \
      echo "error: could not parse minimum_zig_version from build.zig.zon" >&2; exit 1; \
    fi && \
    if [ "$PIN" != "$ZON" ]; then \
      echo "error: .zigversion ($PIN) != build.zig.zon minimum_zig_version ($ZON)" >&2; exit 1; \
    fi && \
    if [ -z "$ZIG_VERSION" ]; then \
      if [ "$INSTALLED" != "$PIN" ]; then \
        echo "error: zig version $INSTALLED does not match .zigversion=$PIN" >&2; exit 1; \
      fi; \
    elif [ "$INSTALLED" != "$ZIG_VERSION" ]; then \
      echo "error: zig version $INSTALLED does not match ZIG_VERSION=$ZIG_VERSION" >&2; exit 1; \
    fi && \
    PRODUCT_VER=$(sed -n 's/^[[:space:]]*\.version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' build.zig.zon | head -n1) && \
    if [ -z "$PRODUCT_VER" ]; then \
      echo "error: could not parse .version from build.zig.zon" >&2; exit 1; \
    fi && \
    if [ -n "$AGAVE_VERSION" ] && [ "$AGAVE_VERSION" != "$PRODUCT_VER" ]; then \
      echo "error: AGAVE_VERSION ($AGAVE_VERSION) != build.zig.zon .version ($PRODUCT_VER)" >&2; exit 1; \
    fi && \
    printf '%s\n' "$PRODUCT_VER" > /agave-version && \
    echo "Zig ok: $INSTALLED (pin=$PIN zon=$ZON${ZIG_VERSION:+ override=$ZIG_VERSION}) product=$PRODUCT_VER${AGAVE_VERSION:+ label=$AGAVE_VERSION}"

# Cross-compile for the target platform.
# Use glibc (-gnu) when any dlopen backend is enabled (CUDA/Vulkan/ROCm/WebGPU).
# Use musl when only CPU/Metal backends are active (smaller static binary).
RUN --mount=type=cache,target=/src/.zig-cache \
    --mount=type=cache,target=/root/.cache/zig \
    ZIG_TARGET=$(case "$TARGETARCH" in \
        amd64) echo "x86_64-linux" ;; \
        arm64) echo "aarch64-linux" ;; \
        *) echo "error: unsupported TARGETARCH=$TARGETARCH (expected amd64 or arm64)" >&2; exit 1 ;; \
    esac) && \
    if [ "$ENABLE_CUDA" = "true" ] || [ "$ENABLE_VULKAN" = "true" ] || [ "$ENABLE_ROCM" = "true" ] || [ "$ENABLE_WEBGPU" = "true" ]; then \
        ZIG_TARGET="${ZIG_TARGET}-gnu"; \
    else \
        ZIG_TARGET="${ZIG_TARGET}-musl"; \
    fi && \
    zig build \
        -Dtarget="$ZIG_TARGET" \
        -Denable-cpu="$ENABLE_CPU" \
        -Denable-metal="$ENABLE_METAL" \
        -Denable-vulkan="$ENABLE_VULKAN" \
        -Denable-cuda="$ENABLE_CUDA" \
        -Denable-rocm="$ENABLE_ROCM" \
        -Denable-webgpu="$ENABLE_WEBGPU" \
        -Denable-debug="$ENABLE_DEBUG" \
        -Denable-gemma3="$ENABLE_GEMMA3" \
        -Denable-qwen35="$ENABLE_QWEN35" \
        -Denable-gpt-oss="$ENABLE_GPT_OSS" \
        -Denable-nemotron-h="$ENABLE_NEMOTRON_H" \
        -Denable-nemotron-nano="$ENABLE_NEMOTRON_NANO" \
        -Denable-glm4="$ENABLE_GLM4" \
        -Denable-gemma4="$ENABLE_GEMMA4" \
        -Denable-diffusion-gemma="$ENABLE_DIFFUSION_GEMMA" \
        -Denable-llama4="$ENABLE_LLAMA4" \
        --prefix /out

# Runtime image: Debian for glibc dlopen compatibility.
# Musl static binaries also run fine on Debian.
# No --platform needed: under BuildKit each stage defaults to its own
# TARGETPLATFORM, regardless of the build stage's BUILDPLATFORM pin above.
FROM debian:bookworm-20260824-slim

# Keep in sync with the build stage (same FROM day / snapshot).
ARG DEBIAN_SNAPSHOT=20260713T000000Z

# Version label: build-arg validated against build.zig.zon in the build stage.
# LABEL cannot read files, so plain builds fall back to "dev"; the authoritative
# product version is always shipped at /usr/share/agave/version (parsed from
# build.zig.zon). Pass --build-arg AGAVE_VERSION=<semver> for release images.
ARG AGAVE_VERSION=dev

LABEL org.opencontainers.image.title="agave" \
      org.opencontainers.image.description="High-performance LLM inference engine" \
      org.opencontainers.image.source="https://github.com/maci0/agave" \
      org.opencontainers.image.version="${AGAVE_VERSION}" \
      org.opencontainers.image.licenses="GPL-3.0-only"

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C \
    TZ=UTC

# curl is only used by HEALTHCHECK (not on the inference hot path).
# Pin UID/GID so compose tmpfs mounts (read_only root) can match ownership.
# Apt packages come from snapshot.debian.org (not live bookworm) for hermeticity.
RUN set -eux; \
    printf 'deb http://snapshot.debian.org/archive/debian/%s bookworm main\n' "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list; \
    printf 'deb http://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/security.list; \
    printf 'Acquire::Check-Valid-Until "false";\nAcquire::Retries "3";\n' > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd -r -g 10001 agave; \
    useradd -r -u 10001 -g agave -d /home/agave -m -s /sbin/nologin agave

COPY --link --from=build /out/bin/agave /usr/local/bin/agave
# Authoritative product version parsed from build.zig.zon (see build-stage check).
COPY --link --from=build /agave-version /usr/share/agave/version

# Writable workdir for non-root runtime (logs, temp files, bind-mount targets).
WORKDIR /home/agave
USER agave

EXPOSE 49453
STOPSIGNAL SIGTERM

# Keep in sync with the process listen port (CLI --port or AGAVE_PORT).
ENV AGAVE_PORT=49453

# Shell form + $$ so AGAVE_PORT expands at container runtime (not image build).
# Use /ready (not /health): Docker HEALTHCHECK gates routing/depends_on, and
# /health returns 200 while degraded (KV pressure / high error rate).
# Probe assumes --serve. One-shot inference should pass --no-healthcheck.
# start-period covers slow model load before probes count as failures.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:$$AGAVE_PORT/ready || exit 1

# Binds all interfaces for container networking. --serve requires AGAVE_API_KEY
# (or --api-key) because non-loopback binds are rejected without auth.
# Override listen port with -e AGAVE_PORT=<port> (and publish the same host port).
# Prefer docker compose (see docker-compose.yml) over ad-hoc runs for local serve.
ENTRYPOINT ["agave", "--host", "0.0.0.0"]
