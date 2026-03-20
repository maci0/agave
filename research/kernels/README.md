# Agave Kernel Research

Python tools for testing, benchmarking, and tuning Agave's compute kernels.

This directory is **not part of the main build** — nothing here ships in the
final binary (per CLAUDE.md section 13). For background on kernels, backends,
quantization, and other concepts, see [DOCUMENTATION.md](../../docs/DOCUMENTATION.md).

## Setup

```bash
cd research/kernels
uv sync                    # Install PyTorch + numpy (CPU only)
uv sync --extra cuda       # With Triton (Linux + NVIDIA only)
uv sync --extra bayesian   # With Optuna for parameter search
```

---

## CLI Overview

All commands go through `run.py`:

```bash
uv run run.py <command> [args...]
```

| Command    | Description                                              |
|------------|----------------------------------------------------------|
| `list`     | List all kernels grouped by category                     |
| `info`     | Show kernel details (source files, test data, etc.)      |
| `diff`     | Show which kernels have changed source files (git)       |
| `golden`   | Generate golden test data (known-correct reference outputs) |
| `bench`    | Run benchmarks (per-kernel by default, `--e2e` for full model) |
| `tune`     | Single optimization cycle (build, benchmark, log result) |
| `grid`     | Try every value in a list for a parameter, pick the best |
| `auto`     | Automated optimization loop over a search space          |
| `staged`   | Manage staged improvements (list/apply/diff/drop)        |
| `coverage` | Report which kernels have reference impls, golden data, Zig tests |
| `status`   | Show optimization history                                |

---

## Kernel Registry

The registry (`registry.py`) maps kernel names to their source files across
backends. Each entry tracks:

- **Source files** per backend (which `.zig`, `.metal`, `.comp` files implement it)
- **Golden prefix** — filename prefix for known-correct test data in `golden/`
- **Bench quants** — which weight formats to use when benchmarking
- **Reference function** — the PyTorch equivalent in `reference.py`

```bash
uv run run.py list          # See all kernels
uv run run.py info sdpa     # Details for one kernel
uv run run.py info gemv     # Details for a whole group
```

### Kernel Groups

| Group       | Kernels |
|-------------|---------|
| elementwise | silu, gelu, add, mul |
| norm        | rms_norm, softmax, l2_norm |
| rope        | rope |
| sdpa        | sdpa, paged_sdpa |
| gemv        | gemv_f32, gemv_q8_0, gemv_q4_0, gemv_q4_k, ... |
| fused       | swiglu |
| math        | argmax, sigmoid, softplus |
| embedding   | embedding |
| ssm         | conv1d, deltanet |
| routing     | moe_routing_topk, moe_routing_sigmoid |

---

## Workflows

### 1. Add a new kernel

Write a Python reference, generate test data, port to Zig, benchmark.

```bash
# 1. Add reference implementation to reference.py (pure Python/PyTorch)
# 2. Add golden data generator to generate_golden.py
uv run run.py golden sdpa          # Generate test data for one kernel
uv run run.py golden               # Generate all

# 3. Port to Zig — your Zig tests load the golden data to verify correctness:
#    const expected = @embedFile("../../research/kernels/golden/rms_norm_n1152_out.bin");

# 4. Benchmark your implementation
uv run run.py bench sdpa --save-baseline  # Save initial performance
# ... optimize the Zig code ...
uv run run.py bench sdpa --compare        # Compare against saved baseline
```

### 2. Optimize an existing kernel

```bash
# See what you're working with
uv run run.py info gemv_q4_0

# Edit the source, then measure the effect
uv run run.py tune gemv_q4_0 -b metal -d "vectorized inner loop"

# Or let the tool auto-detect what you changed
uv run run.py diff                    # Which kernels have changed files?
uv run run.py bench --changed         # Benchmark just those

# Grid search — try every value and pick the fastest
#   --pattern: the exact line in the source file to replace
#   --template: what to replace it with ({value} gets substituted)
uv run run.py grid gemv_q4_0 \
    --param BLOCK_SIZE -b metal \
    --values "32,64,128,256" \
    --pattern "const block_size: usize = 64;" \
    --template "const block_size: usize = {value};"

# Automated search — cycles through parameters from search_spaces.toml
uv run run.py auto softmax --backend cpu --patience 3 --max-iters 10
```

### 3. AI-agent optimization loop

For use by Claude Code or similar AI coding agents:

```
1. Read kernel details:     uv run run.py info <kernel>
2. Edit the source file
3. Measure the change:      uv run run.py tune <kernel> -b <backend> -d "<change>" --auto-revert
4. If faster → keep. If slower → auto-reverted. Try next idea.
5. Or let it search:        uv run run.py auto <kernel> --backend <backend>
```

---

## Tool Details

### Benchmarks

Two modes:

- **Micro-benchmark** (default) — runs one kernel in isolation, measures raw
  speed in nanoseconds. No model files needed.
- **End-to-end** (`--e2e`) — runs full model inference, measures tokens per
  second. Requires `.gguf` model files in `weights/`.

```bash
# Micro-benchmark (default)
uv run run.py bench softmax                      # Single kernel
uv run run.py bench gemv -b metal                # All gemv_* on Metal GPU
uv run run.py bench gemv_f32 --dim=2048 --k=2048 # Custom matrix dimensions
uv run run.py bench silu --iters=500             # More iterations for stability

# End-to-end (needs model files in weights/)
uv run run.py bench --e2e                        # All weight formats
uv run run.py bench gemv_q4_0 --e2e              # Just the relevant format

# Change detection
uv run run.py bench --changed                    # Benchmark only changed kernels

# Baseline comparison
uv run run.py bench softmax --save-baseline      # Save current numbers
# ... make changes ...
uv run run.py bench softmax --compare            # Show delta vs baseline
```

### Optimization Commands

**`tune`** — Build, benchmark, and log the result. Use after manual edits.

```bash
uv run run.py tune gemv_q4_0 -b metal -d "description of change"
uv run run.py tune gemv_q4_0 -b cpu -d "V8 SIMD" --auto-revert  # revert on failure
```

**`grid`** — Try every value for a parameter, pick the best. The source file is
temporarily modified for each value, then restored.

```bash
uv run run.py grid sdpa -b metal \
    --param SDPA_BLOCK --values "32,64,128,256" \
    --pattern "const sdpa_block: usize = 64;" \
    --template "const sdpa_block: usize = {value};"
```

**`auto`** — Automated search over parameters defined in `search_spaces.toml`.
Two strategies:
- **hill-climb** (default) — try each value, keep improvements, stop after
  `--patience` consecutive non-improvements
- **bayesian** — uses Optuna to intelligently sample the search space
  (install with `uv sync --extra bayesian`)

```bash
uv run run.py auto softmax --backend cpu --patience 3
uv run run.py auto softmax --strategy bayesian --max-iters 20
```

**`status`** — View the log of all tune/grid/auto runs.

```bash
uv run run.py status
uv run run.py status gemv   # Filter by kernel name
```

### Staging Area

`grid` and `auto` save their best results to `staging/` instead of immediately
overwriting source files. This lets you review before applying.

```bash
uv run run.py staged                     # List staged improvements
uv run run.py staged diff <name>         # Show what would change
uv run run.py staged apply <name>        # Apply to source tree
uv run run.py staged drop <name>         # Discard one entry
uv run run.py staged clean               # Discard all entries
```

### Golden Test Data

Golden data = known-correct outputs generated by the Python reference
implementations. The Zig tests load these files to verify the engine produces
matching results.

```bash
uv run run.py golden              # Generate all
uv run run.py golden sdpa rope    # Generate specific kernels
uv run run.py golden --list       # See available generators
```

Using golden data in a Zig test:
```zig
test "RMSNorm matches reference" {
    const x = std.mem.bytesAsSlice(f32,
        @embedFile("../../research/kernels/golden/rms_norm_n1152_x.bin"));
    const w = std.mem.bytesAsSlice(f32,
        @embedFile("../../research/kernels/golden/rms_norm_n1152_w.bin"));
    const expected = std.mem.bytesAsSlice(f32,
        @embedFile("../../research/kernels/golden/rms_norm_n1152_out.bin"));

    var out: [1152]f32 = undefined;
    rmsNormImpl(x, w, &out, 1152, 1e-5);

    for (out, expected) |got, exp| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-5);
    }
}
```

### Reference Implementations (`reference.py`)

Pure Python/PyTorch implementations of every kernel, used to generate golden
data. These are the "ground truth" that Zig implementations are tested against.

| Function | Group | Zig equivalent |
|----------|-------|----------------|
| `rms_norm(x, w, eps)` | norm | `cpu.zig:rmsNormImpl` |
| `softmax(x)` | norm | `cpu.zig:softmaxSimd` |
| `l2_norm(x)` | norm | `cpu.zig:l2NormImpl` |
| `silu(x)` | elementwise | `math.zig:silu` |
| `gelu(x)` | elementwise | `math.zig:gelu` |
| `add(a, b)` | elementwise | `cpu.zig:add` |
| `mul(a, b)` | elementwise | `cpu.zig:mul` |
| `sigmoid(x)` | math | `math.zig:sigmoid` |
| `softplus(x)` | math | `math.zig:softplus` |
| `argmax(x)` | math | `math.zig:argmax` |
| `rope(q, k, pos, hd, rd, theta)` | rope | model `rope()` methods |
| `sdpa(q, keys, values, nh, nkv, hd)` | sdpa | `attention.zig:sdpa` |
| `paged_sdpa(...)` | sdpa | `attention.zig:pagedSdpa` |
| `embedding_lookup(table, id)` | embedding | `cpu.zig:embLookup` |
| `gemv_f32(W, x)` | gemv | `cpu.zig:gemv` |
| `conv1d_causal(...)` | ssm | `ssm.zig:conv1d` |
| `deltanet_recurrence(...)` | ssm | `qwen35.zig:deltaNet` |
| `moe_routing_topk(logits, k)` | routing | `gpt_oss.zig:moeRouting` |
| `moe_routing_sigmoid(logits)` | routing | `glm4.zig:moeRouting` |
| `swiglu_fused(x, w_gate, w_up)` | fused | (not yet in Zig) |

---

## Current Baselines

End-to-end inference speed on the 1B parameter Gemma3 model:

| Model | Backend | tok/s |
|-------|---------|-------|
| Gemma3 1B Q4_0 | CPU | 5.6 |
| Gemma3 1B Q4_0 | Metal | 21.7 |
| Gemma3 1B Q8_0 | CPU | 4.4 |
| Gemma3 1B Q8_0 | Metal | 34.1 |
| Gemma3 1B BF16 | CPU | 16.8 |
| Gemma3 1B BF16 | Metal | 34.5 |

---

## File Organization

```
research/kernels/
├── README.md                 This file
├── pyproject.toml            Python dependencies (uv)
├── uv.lock                   Locked dependency versions
├── run.py                    CLI entry point — routes to other modules
├── registry.py               Kernel → source files, golden data, bench config
├── reference.py              PyTorch reference implementations (ground truth)
├── generate_golden.py        Generates golden .bin test data per kernel
├── autotune.py               Bench/tune/grid/auto/staged orchestrator
├── search_spaces.toml        Parameter search spaces for auto mode
├── golden/                   Generated .bin files (loaded by Zig tests via @embedFile)
├── staging/                  Staged improvements from grid/auto (not committed)
├── results.tsv               Benchmark history (appended by bench)
├── baseline.json             Saved baseline for --compare
└── optimization_log.tsv      Optimization experiment log (appended by tune/grid/auto)
```

## Permitted Prototyping Tools

The following may be used **only in this directory** for rapid prototyping:
- **PyTorch** — reference implementations, golden test generation
- **Triton** — GPU kernel prototyping (Linux + NVIDIA only)
- **CUDA C++** / **HIP** — GPU kernel development
- **CUTLASS / TVM / MLIR / TileLang** — advanced kernel frameworks

### Mandatory Porting Rule (CLAUDE.md section 13)

Every prototype **must** be manually re-implemented in native Zig before
merging into `src/`. Python/PyTorch code never ships.
