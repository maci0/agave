FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS build

ARG ZIG_VERSION=0.15.2
ARG TARGETARCH

# Backend enable flags — Metal disabled by default (macOS-only, not usable in Docker).
ARG ENABLE_CPU=true
ARG ENABLE_METAL=false
ARG ENABLE_VULKAN=true
ARG ENABLE_CUDA=true
ARG ENABLE_ROCM=true

# Model enable flags — all enabled by default. Disable to reduce binary size.
ARG ENABLE_GEMMA3=true
ARG ENABLE_QWEN35=true
ARG ENABLE_GPT_OSS=true
ARG ENABLE_NEMOTRON_H=true
ARG ENABLE_NEMOTRON_NANO=true
ARG ENABLE_GLM4=true

RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates && rm -rf /var/lib/apt/lists/*

RUN ARCH=$(uname -m) && \
    curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-linux-${ARCH}-${ZIG_VERSION}.tar.xz" \
    | tar -xJ -C /usr/local --strip-components=1

WORKDIR /src
COPY . .

# Cross-compile for the target platform.
# Use glibc (-gnu) when any dlopen backend is enabled (CUDA/Vulkan/ROCm need glibc).
# Use musl when only CPU/Metal backends are active (smaller static binary).
RUN ZIG_TARGET=$(case "$TARGETARCH" in \
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
        -Denable-gemma3="$ENABLE_GEMMA3" \
        -Denable-qwen35="$ENABLE_QWEN35" \
        -Denable-gpt-oss="$ENABLE_GPT_OSS" \
        -Denable-nemotron-h="$ENABLE_NEMOTRON_H" \
        -Denable-nemotron-nano="$ENABLE_NEMOTRON_NANO" \
        -Denable-glm4="$ENABLE_GLM4" \
        --prefix /out

# Runtime image — Debian for glibc dlopen compatibility.
# Musl static binaries also run fine on Debian.
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="agave" \
      org.opencontainers.image.description="High-performance LLM inference engine" \
      org.opencontainers.image.source="https://github.com/anthropics/agave"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* && \
    groupadd -r agave && useradd -r -g agave -s /sbin/nologin agave

COPY --from=build /out/bin/ /usr/local/bin/

USER agave

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD agave --version || exit 1

ENTRYPOINT ["agave"]
