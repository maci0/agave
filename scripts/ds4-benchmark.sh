#!/bin/bash
# DS4 Benchmark Suite, speed + coherence across quants and KV types
set -euo pipefail

AGAVE="./zig-out/bin/agave"
BLOB_DIR="/Users/mwysocki/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs"
PROMPT="Explain the theory of relativity step by step."

echo "============================================"
echo "DS4 Benchmark Suite, $(date)"
echo "============================================"
echo ""

benchmark_model() {
    local name="$1"
    local model="$2"
    local extra_args="${3:-}"
    
    echo "--- $name ---"
    
    # Check model exists
    if [ ! -f "$model" ]; then
        echo "  SKIP: model not found"
        echo ""
        return
    fi
    
    # Warmup
    $AGAVE "$model" --ssd-streaming $extra_args --max-tokens 8 --ctx-size 512 -t 0.0 "Hi" > /dev/null 2>&1 || true
    sleep 1
    
    # Speed benchmark (3 runs, report all)
    echo "  Speed (128 tok, t=0.0):"
    for run in 1 2 3; do
        result=$($AGAVE "$model" --ssd-streaming $extra_args --max-tokens 128 --ctx-size 512 -t 0.0 "Hello" 2>&1 | grep "tok/s" || echo "FAIL")
        echo "    Run $run: $result"
    done
    
    # Coherence check (t=0.7 for more natural output)
    echo "  Coherence (t=0.7):"
    output=$($AGAVE "$model" --ssd-streaming $extra_args --max-tokens 64 --ctx-size 512 -t 0.7 "$PROMPT" 2>&1 | grep -v "^info:\|^agave\|^system:\|^loading\|^recipe:\|^context:\|^loaded:\|^ssd-\|^error")
    echo "    $output"
    echo ""
}

# Q2_K (baseline)
benchmark_model "Q2_K" "$BLOB_DIR/DeepSeek-V4-Flash-0731-Q2_K-00001-of-00002.gguf"

# Q2_K_S
benchmark_model "Q2_K_S" "$BLOB_DIR/DeepSeek-V4-Flash-0731-Q2_K_S-00001-of-00002.gguf"

# MXFP4
benchmark_model "MXFP4" "$BLOB_DIR/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf"

echo "============================================"
echo "Done, $(date)"
echo "============================================"
