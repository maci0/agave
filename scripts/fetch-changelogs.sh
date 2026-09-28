#!/usr/bin/env bash
# fetch-changelogs.sh, Download latest changelogs from major LLM inference engines
# Usage: ./scripts/fetch-changelogs.sh [output_dir]
# Output: One file per engine in output_dir (default: docs/changelogs/)

set -euo pipefail
export LC_ALL=C TZ=UTC

OUT="${1:-docs/changelogs}"
mkdir -p "$OUT"
# UTC, not the host's local calendar date: the GitHub release timestamps
# written into the same files are UTC, and a local date recorded next to them
# reads as a day off depending on where the doc was generated.
DATE=$(date -u +%Y-%m-%d)

echo "Fetching changelogs → $OUT (as of $DATE)"

# A failed fetch must not replace a good committed file with an error string:
# these land in docs/changelogs/, where "(fetch failed)" reads as content and a
# partial GitHub page looks like an engine with no releases. Stage each file in
# a temp dir, require a body, and only then move it into place. The script fails
# at the end with the list of engines that did not produce output, so a rate
# limit or a missing gh login is a red run rather than a silent data loss.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
failed=()
total=0
fetch() { total=$((total + 1)); "$@"; }

publish() {
    local name="$1" header
    local body="$STAGE/${name}.body"
    IFS= read -r header
    IFS= read -r header2
    header="$header"$'\n'"$header2"
    if [[ ! -s "$body" ]]; then
        echo "    x $name: fetch produced no content" >&2
        failed+=("$name")
        return
    fi
    {
        printf '%s\n\n' "$header"
        cat "$body"
    } >"$STAGE/${name}.md"
    mv "$STAGE/${name}.md" "$OUT/${name}.md"
    echo "    → $OUT/${name}.md"
}

# ── Helpers ──────────────────────────────────────────────────────────────────

fetch_github_releases() {
    local name="$1" repo="$2" pages="${3:-3}"
    echo "  $name (github releases: $repo)"
    for page in $(seq 1 "$pages"); do
        # A page failure ends pagination; page 1 failing is the whole fetch.
        if ! gh api "repos/$repo/releases?per_page=30&page=$page" \
            --jq '.[] | "## " + .tag_name + " (" + (.published_at // "unknown") + ")\n" + (.body // "(no body)") + "\n\n---\n"' \
            >>"$STAGE/${name}.body" 2>>"$STAGE/${name}.err"; then
            echo "    ! $repo page $page: $(head -n1 "$STAGE/${name}.err" 2>/dev/null)" >&2
            break
        fi
    done
    publish "$name" <<<"# $name, GitHub Releases (fetched $DATE)
Source: https://github.com/$repo/releases"
}

fetch_url() {
    local name="$1" url="$2"
    echo "  $name ($url)"
    if ! curl -fsSL "$url" -o "$STAGE/${name}.body" 2>>"$STAGE/${name}.err"; then
        echo "    ! $url: $(head -n1 "$STAGE/${name}.err" 2>/dev/null)" >&2
    fi
    publish "$name" <<<"# $name, Changelog (fetched $DATE)
Source: $url"
}

# ── Engines ──────────────────────────────────────────────────────────────────

fetch fetch_github_releases "vllm" "vllm-project/vllm" 4

fetch fetch_github_releases "sglang" "sgl-project/sglang" 4

fetch fetch_github_releases "llamacpp" "ggml-org/llama.cpp" 4

fetch fetch_github_releases "tensorrt-llm" "NVIDIA/TensorRT-LLM" 4

fetch fetch_github_releases "tgi" "huggingface/text-generation-inference" 4

fetch fetch_github_releases "ollama" "ollama/ollama" 4

fetch fetch_github_releases "mlx" "ml-explore/mlx" 4

# MLX-LM (language model layer on top of MLX)
fetch fetch_github_releases "mlx-lm" "ml-explore/mlx-lm" 4

# LM Studio, uses a public changelog page (no GitHub releases)
fetch fetch_url "lmstudio" "https://lmstudio.ai/changelog"

# Modular MAX, docs changelog
fetch fetch_url "modular-max" "https://docs.modular.com/max/changelog/"

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
if ((${#failed[@]})); then
    # A partial refresh is a failure, not a shorter run: the index below lists
    # every engine, and a file left at its previous content reads as current
    # while silently describing an older release set.
    echo "FAILED: ${#failed[@]} of $((total)) engine(s) produced no content: ${failed[*]}" >&2
    echo "The previous copy of each was left in place. Fix the cause above and rerun." >&2
    exit 1
fi
echo "Done. Index: $INDEX"
echo "Files written: $(find "$OUT" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ') changelogs"
