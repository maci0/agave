#!/usr/bin/env bash
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
    # Third argument is a flag string ("--kq-type q8 -ctk q8"); split it once
    # so each flag stays a separate argv entry.
    local -a extra_args=()
    if [[ -n "${3:-}" ]]; then
        read -r -a extra_args <<< "$3"
    fi
    # bash 3.2 (the macOS default) treats "${arr[@]}" on an empty array as
    # unset under set -u, so the extras are spliced in only when present.
    local -a base=("$AGAVE" "$model" --ssd-streaming)
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        base+=("${extra_args[@]}")
    fi
    
    echo "--- $name ---"
    
    # Check model exists
    if [ ! -f "$model" ]; then
        echo "  SKIP: model not found"
        echo ""
        return
    fi
    
    # Warmup
    "${base[@]}" --max-tokens 8 --ctx-size 512 -t 0.0 "Hi" > /dev/null 2>&1 || true
    sleep 1
    
    # Speed benchmark (3 runs, report all)
    echo "  Speed (128 tok, t=0.0):"
    for run in 1 2 3; do
        result=$("${base[@]}" --max-tokens 128 --ctx-size 512 -t 0.0 "Hello" 2>&1 | grep "tok/s" || echo "FAIL")
        echo "    Run $run: $result"
    done
    
    # Coherence check (t=0.7 for more natural output)
    echo "  Coherence (t=0.7):"
    output=$("${base[@]}" --max-tokens 64 --ctx-size 512 -t 0.7 "$PROMPT" 2>&1 | grep -v "^info:\|^agave\|^system:\|^loading\|^recipe:\|^context:\|^loaded:\|^ssd-\|^error")
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
