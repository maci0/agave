#!/bin/bash
# Autoresearch benchmark script for DS4 Metal performance
set -e

MLX=~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/38c0bd20a6fba70f22c5ee2940ec0092b36ab936/

cd /Users/mwysocki/Code/Experiments/ai-inference/agave

# Build
zig build 2>&1 | head -5
if [ $? -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi

# Warmup run
timeout 60 ./zig-out/bin/agave "$MLX" --backend metal --ssd-streaming --ctx-size 512 --kv-type f32 --spec-mode suffix -n 64 -t 0.0 "Explain quicksort" 2>&1 | grep "tok/s" || true

# 3 benchmark runs
echo "=== BENCHMARK ==="
for i in 1 2 3; do
    timeout 60 ./zig-out/bin/agave "$MLX" --backend metal --ssd-streaming --ctx-size 512 --kv-type f32 --spec-mode suffix -n 64 -t 0.0 "Explain quicksort" 2>&1 | grep "tok/s"
done

echo "=== QUALITY CHECK ==="
timeout 30 ./zig-out/bin/agave "$MLX" --backend metal --ssd-streaming --ctx-size 128 --kv-type f32 -n 16 -t 0.0 "What is the capital of France?" 2>&1 | grep -v "^info:\|^warning:\|^ssd-\|^agave\|^system\|^recipe\|^context\|^loaded" | tail -3
