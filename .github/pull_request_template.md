## Change

<!-- What changed and why. User-facing behavior goes in CHANGELOG.md [Unreleased]. -->

## Test plan

- [ ] `zig build ci` (or `zig build check` + `zig build lint-web` if bun is not installed)
- [ ] `scripts/check-shader-artifacts.sh --ptx-only` if CUDA kernel sources changed
- [ ] `CHANGELOG.md` `[Unreleased]` entry for user-facing changes
