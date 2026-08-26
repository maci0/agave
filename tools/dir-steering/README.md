# Directional Steering

Runtime activation editing for Agave. A steering file is a flat `f32` matrix
with one normalized direction vector per layer (`n_layers × n_embd` floats).
During inference, Agave projects the direction out of (or into) activations:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the direction. Negative scale amplifies it.

## Runtime Flags

```bash
agave model.gguf --dir-steering-file direction.f32 "prompt"                    # FFN steering (default scale 1.0)
agave model.gguf --dir-steering-file direction.f32 --dir-steering-ffn -1 "..."  # amplify direction
agave model.gguf --dir-steering-file direction.f32 --dir-steering-ffn 2 "..."   # stronger suppression
agave model.gguf --dir-steering-file direction.f32 --dir-steering-attn 0.5 "..."  # also steer attention
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir-steering-file` | | Path to flat f32 direction file |
| `--dir-steering-ffn` | `1.0` (when file provided) | Scale for FFN output steering |
| `--dir-steering-attn` | `0` | Scale for attention output steering |

## Building Direction Vectors

```bash
python3 tools/dir-steering/build_direction.py \
    --agave ./zig-out/bin/agave \
    --model model.gguf \
    --good-file prompts_succinct.txt \
    --bad-file prompts_verbose.txt \
    --out direction.f32 \
    --component ffn_out \
    --n-layers 64 --n-embd 2048
```

Each prompt file has one prompt per line. Pairs are matched by line number.

**Note:** The extraction script currently requires `--export-activations` support
in agave (not yet implemented). The pipeline structure is in place; activations
are placeholder zeros until the export flag is added.

## How It Works

1. Collect per-layer activations for target prompts (e.g. succinct answers)
2. Collect per-layer activations for contrast prompts (e.g. verbose answers)
3. Direction = mean(target) - mean(contrast), normalized per layer
4. At runtime: `y -= scale * dot(direction, y) * direction` after each layer

The FFN output is usually the best target because it carries style and behavior
signals. Attention steering is available but more fragile.

## Implementation

- `src/steering.zig`: `DirectionalSteering` struct, zero-alloc `apply()` in hot path
- `src/models/qwen35.zig`: Hook points after attention and FFN outputs
- `src/models/model.zig`: `setSteering()` dispatcher (field-presence comptime check)
- `src/main.zig`: CLI flags, file loading, model attachment

Based on the technique from [antirez/ds4](https://github.com/antirez/ds4).
