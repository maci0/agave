# anti-slop (vendored)

Custom oxlint rules, loaded by `.oxlintrc.json` via
`jsPlugins[].specifier: ./tools/oxlint/anti-slop/index.ts`.

- Upstream: https://github.com/dmmulroy/anti-slop
- Vendored because oxlint resolves JS plugins from a path, and the repo keeps
  zero runtime dependencies outside `package.json` devDependencies.

## Local patches

None. The tree is upstream as copied.

## Pinning

The upstream commit this was copied from is **not recorded**: it was vendored
before this manifest existed and the source revision is not recoverable from the
files. Re-vendor from a known commit and write the hash here the next time these
rules are updated, so a future diff against upstream is meaningful.

`tools/oxlint/anti-slop/**` is in `.oxlintrc.json` `ignorePatterns`: the rules
are third-party source and are not held to this repo's lint config.
