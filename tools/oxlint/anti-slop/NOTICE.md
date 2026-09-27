# Third-party notice

`index.ts`, `rules/`, and `shared/` are third-party source, copied from:

- Upstream: <https://github.com/dmmulroy/anti-slop>
- Vendored by: this repository, first in commit `d5ed1f6` ("chore: add the
  oxlint gate, green on what it covers")

The license text upstream publishes is **not recorded here**: the copy was made
before this notice existed and no commit hash, LICENSE file, or package metadata
for the upstream revision was kept. Treat the grant as unknown until someone
re-vendors from a named upstream commit and copies that revision's LICENSE file
into this directory. Until then, redistribution of this directory is not
traceable to a license grant.

## Provenance anchor

`VENDORED.sha256` records the sha256 of every vendored `.ts` file. The upstream
commit hash is unrecoverable, so the manifest is the anchor: a file that no
longer hashes to its recorded value is a local edit, and `zig build check-pins`
fails on one rather than letting it ride as "third-party source".

The manifest also makes a later re-vendor diffable. Copy upstream's LICENSE
file here, note the commit in `README.md`, and regenerate the manifest in the
same change:

```sh
find . -name '*.ts' -type f | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > VENDORED.sha256
```

## Why vendored at all

oxlint resolves JavaScript plugins from a path (`jsPlugins[].specifier` in
`.oxlintrc.json`), and the repository keeps zero runtime dependencies outside
`package.json` devDependencies. `index.ts` imports `@oxlint/plugins`, which is
already a devDependency; only the rule sources are vendored.
