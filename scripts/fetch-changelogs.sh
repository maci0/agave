#!/usr/bin/env bash
# fetch-changelogs.sh, Download latest changelogs from major LLM inference engines
# Usage: ./scripts/fetch-changelogs.sh [output_dir]
# Output: One file per engine in output_dir (default: docs/changelogs/)

set -euo pipefail

OUT="${1:-docs/changelogs}"
mkdir -p "$OUT"
DATE=$(date +%Y-%m-%d)

echo "Fetching changelogs → $OUT (as of $DATE)"

# ── Helpers ──────────────────────────────────────────────────────────────────

fetch_github_releases() {
    local name="$1" repo="$2" pages="${3:-3}"
    local out="$OUT/${name}.md"
    echo "  $name (github releases: $repo)"
    {
        echo "# $name, GitHub Releases (fetched $DATE)"
        echo "Source: https://github.com/$repo/releases"
        echo
        for page in $(seq 1 "$pages"); do
            gh api "repos/$repo/releases?per_page=30&page=$page" \
                --jq '.[] | "## " + .tag_name + " (" + (.published_at // "unknown") + ")\n" + (.body // "(no body)") + "\n\n---\n"' \
                2>/dev/null || break
        done
    } > "$out"
    echo "    → $out"
}

fetch_url() {
    local name="$1" url="$2"
    local out="$OUT/${name}.md"
    echo "  $name ($url)"
    {
        echo "# $name, Changelog (fetched $DATE)"
        echo "Source: $url"
        echo
        curl -fsSL "$url" 2>/dev/null || echo "(fetch failed)"
    } > "$out"
    echo "    → $out"
}

# ── Engines ──────────────────────────────────────────────────────────────────

# vLLM
fetch_github_releases "vllm" "vllm-project/vllm" 4

# SGLang
fetch_github_releases "sglang" "sgl-project/sglang" 4

# llama.cpp
fetch_github_releases "llamacpp" "ggml-org/llama.cpp" 4

# TensorRT-LLM
fetch_github_releases "tensorrt-llm" "NVIDIA/TensorRT-LLM" 4

# HuggingFace TGI
fetch_github_releases "tgi" "huggingface/text-generation-inference" 4

# Ollama
fetch_github_releases "ollama" "ollama/ollama" 4

# MLX
fetch_github_releases "mlx" "ml-explore/mlx" 4

# MLX-LM (language model layer on top of MLX)
fetch_github_releases "mlx-lm" "ml-explore/mlx-lm" 4

# LM Studio, uses a public changelog page (no GitHub releases)
fetch_url "lmstudio" "https://lmstudio.ai/changelog"

# Modular MAX, docs changelog
fetch_url "modular-max" "https://docs.modular.com/max/changelog/"

# ── Summary index ─────────────────────────────────────────────────────────────

INDEX="$OUT/INDEX.md"
{
    echo "# LLM Inference Engine Changelogs"
    echo "Fetched: $DATE"
    echo
    echo "| Engine | File | Source |"
    echo "|--------|------|--------|"
    echo "| vLLM | [vllm.md](vllm.md) | github.com/vllm-project/vllm/releases |"
    echo "| SGLang | [sglang.md](sglang.md) | github.com/sgl-project/sglang/releases |"
    echo "| llama.cpp | [llamacpp.md](llamacpp.md) | github.com/ggml-org/llama.cpp/releases |"
    echo "| TensorRT-LLM | [tensorrt-llm.md](tensorrt-llm.md) | github.com/NVIDIA/TensorRT-LLM/releases |"
    echo "| HuggingFace TGI | [tgi.md](tgi.md) | github.com/huggingface/text-generation-inference/releases |"
    echo "| Ollama | [ollama.md](ollama.md) | github.com/ollama/ollama/releases |"
    echo "| MLX | [mlx.md](mlx.md) | github.com/ml-explore/mlx/releases |"
    echo "| MLX-LM | [mlx-lm.md](mlx-lm.md) | github.com/ml-explore/mlx-lm/releases |"
    echo "| LM Studio | [lmstudio.md](lmstudio.md) | lmstudio.ai/changelog |"
    echo "| Modular MAX | [modular-max.md](modular-max.md) | docs.modular.com/max/changelog/ |"
    echo
    echo "Run \`./scripts/fetch-changelogs.sh\` to refresh."
} > "$INDEX"

echo
echo "Done. Index: $INDEX"
echo "Files written: $(ls -1 "$OUT"/*.md | wc -l | tr -d ' ') changelogs"
