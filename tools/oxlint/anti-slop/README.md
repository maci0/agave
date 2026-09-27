# anti-slop (vendored)

Custom oxlint rules, loaded by `.oxlintrc.json` via
`jsPlugins[].specifier: ./tools/oxlint/anti-slop/index.ts`.

- Upstream: https://github.com/dmmulroy/anti-slop
- Vendored because oxlint resolves JS plugins from a path, and the repo keeps
  zero runtime dependencies outside `package.json` devDependencies.

## Local patches

None. The tree is upstream as copied.

## Pinning

`VENDORED.sha256` records the sha256 of every vendored `.ts` file, and
`zig build check-pins` fails when the tree stops matching it. The upstream
commit is still unrecorded, so the manifest is the provenance anchor until
someone re-vendors from a named commit; that change also copies upstream's
LICENSE file here. See `NOTICE.md`.

`tools/oxlint/anti-slop/**` is in `.oxlintrc.json` `ignorePatterns`: the rules
are third-party source and are not held to this repo's lint config.
