#!/bin/bash
GGUF="$HOME/.cache/huggingface/hub/models--ggml-org--DeepSeek-V4-Flash-0731-GGUF/blobs/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00002.gguf"
AGAVE="./zig-out/bin/agave"

echo "=== Prose ==="
timeout 180 $AGAVE "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 64 --spec-mode suffix -t 0.0 "Explain the theory of general relativity in simple terms." 2>&1 | grep "tok/s"

echo "=== Code ==="
timeout 120 $AGAVE "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 128 --spec-mode suffix -t 0.0 "Write a Python function to sort a list." 2>&1 | grep "tok/s"

echo "=== Baseline (no spec) ==="
timeout 120 $AGAVE "$GGUF" --backend cpu --ssd-streaming --ctx-size 512 -n 32 -t 0.0 "What is the capital of France?" 2>&1 | grep "tok/s"
