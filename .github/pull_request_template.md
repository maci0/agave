## Change

<!-- What changed and why. User-facing behavior goes in CHANGELOG.md [Unreleased]. -->

## Test plan

- [ ] `zig build check`
- [ ] `zig build lint-web` if `src/web/` or `web/` changed
- [ ] `scripts/check-shader-artifacts.sh --ptx-only` if CUDA kernel sources changed
- [ ] `CHANGELOG.md` `[Unreleased]` entry for user-facing changes
