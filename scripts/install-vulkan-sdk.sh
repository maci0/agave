#!/usr/bin/env bash
# Install the LunarG Vulkan SDK on a GitHub-hosted runner.
# Canonical: bash scripts/install-vulkan-sdk.sh (CI golden-tests job, Vulkan leg).
#
# GitHub-hosted runners have no Vulkan loader. A self-hosted runner supplies
# its own driver and SDK, so the workflow skips this step there. Kept in
# scripts/ rather than inline in the workflow so the blocking
# `zig build lint-shell` (scripts/*.sh) analyses it.
set -euo pipefail
export LC_ALL=C TZ=UTC

sudo install -dm755 /etc/apt/keyrings
# Pin LunarG archive key by fingerprint (avoid trusting a bare wget of the key URL alone).
KEY_URL="https://packages.lunarg.com/lunarg-signing-key-pub.asc"
EXPECTED_FP="03F5D621A3710FA95A93D89BAA8452080E383F7E"
curl -fsSL "$KEY_URL" -o /tmp/lunarg-archive-keyring.asc
FP="$(gpg --show-keys --with-colons /tmp/lunarg-archive-keyring.asc | awk -F: '/^fpr:/ {print $10; exit}')"
if [[ "$FP" != "$EXPECTED_FP" ]]; then
    echo "::error::LunarG apt key fingerprint mismatch: got $FP expected $EXPECTED_FP"
    exit 1
fi
sudo install -m644 /tmp/lunarg-archive-keyring.asc /etc/apt/keyrings/lunarg-archive-keyring.asc
CODENAME=$(lsb_release -cs)
echo "deb [signed-by=/etc/apt/keyrings/lunarg-archive-keyring.asc] https://packages.lunarg.com/vulkan/ ${CODENAME} main" | sudo tee /etc/apt/sources.list.d/lunarg-vulkan.list > /dev/null
sudo apt-get update
sudo apt-get install -y --no-install-recommends vulkan-sdk
