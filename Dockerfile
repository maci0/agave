# syntax=docker/dockerfile:1
# Pin to a dated Debian tag for reproducible builds (bump with dependabot/docker).
FROM --platform=$BUILDPLATFORM debian:bookworm-20260713-slim AS build

ARG ZIG_VERSION=0.16.0
ARG TARGETARCH

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

RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates && rm -rf /var/lib/apt/lists/*

# Zig toolchain checksums (SHA256). Update when bumping ZIG_VERSION.
ARG ZIG_SHA256_X86_64=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
ARG ZIG_SHA256_AARCH64=ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17

RUN ARCH=$(uname -m) && \
    ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/zig-${ARCH}-linux-${ZIG_VERSION}.tar.xz" && \
    if [ "$ARCH" = "x86_64" ]; then EXPECTED_SHA256="$ZIG_SHA256_X86_64"; \
    elif [ "$ARCH" = "aarch64" ]; then EXPECTED_SHA256="$ZIG_SHA256_AARCH64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    curl -fsSL "$ZIG_URL" -o /tmp/zig.tar.xz && \
    echo "${EXPECTED_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /usr/local/zig && \
    tar -xJf /tmp/zig.tar.xz -C /usr/local/zig --strip-components=1 && \
    ln -s /usr/local/zig/zig /usr/local/bin/zig && \
    rm /tmp/zig.tar.xz

WORKDIR /src
COPY --link . .

# Cross-compile for the target platform.
# Use glibc (-gnu) when any dlopen backend is enabled (CUDA/Vulkan/ROCm need glibc).
# Use musl when only CPU/Metal backends are active (smaller static binary).
RUN --mount=type=cache,target=/src/.zig-cache \
    --mount=type=cache,target=/root/.cache/zig \
    ZIG_TARGET=$(case "$TARGETARCH" in \
        amd64) echo "x86_64-linux" ;; \
        arm64) echo "aarch64-linux" ;; \
    esac) && \
    if [ "$ENABLE_CUDA" = "true" ] || [ "$ENABLE_VULKAN" = "true" ] || [ "$ENABLE_ROCM" = "true" ]; then \
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
FROM debian:bookworm-20260713-slim

LABEL org.opencontainers.image.title="agave" \
      org.opencontainers.image.description="High-performance LLM inference engine" \
      org.opencontainers.image.source="https://github.com/maci0/agave"

# curl is only used by HEALTHCHECK (not on the inference hot path).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/* && \
    groupadd -r agave && \
    useradd -r -g agave -d /home/agave -m -s /sbin/nologin agave

COPY --link --from=build /out/bin/agave /usr/local/bin/agave

# Writable workdir for non-root runtime (logs, temp files, bind-mount targets).
WORKDIR /home/agave
USER agave

EXPOSE 49453
STOPSIGNAL SIGTERM

# Keep in sync with the process listen port (CLI --port or AGAVE_PORT).
ENV AGAVE_PORT=49453

# Shell form + $$ so AGAVE_PORT expands at container runtime (not image build).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:$$AGAVE_PORT/health || exit 1

# Binds all interfaces for container networking. --serve requires AGAVE_API_KEY
# (or --api-key) because non-loopback binds are rejected without auth.
# Override listen port with -e AGAVE_PORT=<port> (and publish the same host port).
ENTRYPOINT ["agave", "--host", "0.0.0.0"]
