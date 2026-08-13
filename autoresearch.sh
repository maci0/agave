#!/bin/bash
# Autoresearch benchmark runner — DeepSeek V4 Flash on Agave
# Usage: ./autoresearch.sh [mxfp4|q2|both]

set -euo pipefail

MXFP4_GGUF="$HOME/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf"
Q2_GGUF="/tmp/ds4/ds4flash.gguf"
AGAVE="./zig-out/bin/agave"
PROMPT="What is the capital of France?"

run_bench() {
    local model="$1"
    local label="$2"
    echo "=== $label ==="
    timeout 180 "$AGAVE" "$model" --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "$PROMPT" 2>&1 | tail -20
    echo ""
}

MODE="${1:-both}"

case "$MODE" in
    mxfp4) run_bench "$MXFP4_GGUF" "MXFP4" ;;
    q2)    run_bench "$Q2_GGUF" "ds4 Q2 imatrix" ;;
    both)
        run_bench "$MXFP4_GGUF" "MXFP4"
        run_bench "$Q2_GGUF" "ds4 Q2 imatrix"
        ;;
esac
