# Performance baselines

`baselines.json` holds the decode throughput `zig build gate` enforces, plus
the entries it checks for greedy determinism.

It ships empty. A baseline is only meaningful next to the machine that
recorded it, so there is nothing useful to commit up front: numbers copied
from `docs/BENCHMARKS.md` were measured on other hardware with other build
flags, and gating against them would compare two different setups.

## Recording baselines

Build the release binary, put the models where the entries point, then let
the gate measure:

```bash
zig build
python3 scripts/perf-gate.py --update-baselines
```

## Adding an entry

Each entry names a model, a backend, and optionally extra CLI arguments. Add
it with no `decode_tps`, then run `--update-baselines` to fill it in:

```json
{
  "name": "qwen3.5-0.8b-q8_0-metal",
  "model": "models/Qwen3.5-0.8B-Q8_0.gguf",
  "backend": "metal",
  "determinism": true
}
```

| Field | Meaning |
|-------|---------|
| `name` | Identifier in gate output. Convention: `model-quant-backend`. |
| `model` | Path to the model, absolute or relative to the repo root. |
| `backend` | `--backend` value: `cpu`, `metal`, `vulkan`, `cuda`, `rocm`, `webgpu`. |
| `args` | Extra CLI arguments, e.g. `["--kv-type", "turbo4"]`. |
| `decode_tps` | Baseline decode throughput. Written by `--update-baselines`. |
| `determinism` | Check two greedy runs at one seed match byte for byte. Default true. |

## Running the gate

```bash
zig build gate                            # 5% tolerance
python3 scripts/perf-gate.py --tolerance 10
```

Exit codes: `0` every runnable entry held, `1` a regression or a determinism
mismatch, `2` the gate could not run (no binary, malformed baselines).

An entry whose model file is absent is reported as skipped, never as a pass.
A run where every entry skipped prints that nothing was verified and exits
`0`: it has no evidence either way, and a machine without the models should
not block a build. Read the skip list before trusting a green gate.

## Comparing fairly

A baseline and the run measured against it have to match in workload, build
flags, backend, quantization and machine. Re-record after a hardware change,
a Zig upgrade, or a change to the build flags, and say so in the commit;
`--update-baselines` on a slower machine silently lowers the floor for
everyone.
