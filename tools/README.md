# tools/

Offline utilities that operate on model files or on the repo itself. Nothing
here is compiled or linked into `agave`; `zig build` never reads this tree.

If a tool has to run as part of a build, a test, or a release, it belongs in
`scripts/` instead. The split is by ownership: `scripts/` is wired into
`build.zig` and CI, `tools/` is run by hand.

| Path | What it does |
|------|--------------|
| `gguf_io.py` | Shared GGUF header parsing and writing. Metadata block, tensor table, and per-tensor byte length; tensor payloads are passed through as bytes. Every other script here imports it. |
| `dir-steering/` | Builds the flat `f32` steering matrix consumed by `agave --dir-steering-file` (one normalized direction per layer). See its README. |
| `mixed-quant/` | Splices higher-precision routed experts into selected layers of an existing GGUF. See its README. |
| `synth-moe/` | `moeify_gguf.py` turns a dense GGUF into a routed-MoE one, so the MoE forward path can be exercised without a multi-GB MoE checkpoint. |
| `quality-testing/` | Token-by-token NLL scoring of a local GGUF against official outputs. See its README. |
| `oxlint/` | Vendored third-party oxlint plugin (`anti-slop`). Not agave code; update it through the plugin's own install path, not by hand. |

## Conventions

Python here follows the same rules as `scripts/` and `research/`: `uv` only,
every parameter annotated, and `ruff` clean. `zig build lint-python` covers
`tools/`.
