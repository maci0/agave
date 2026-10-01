# Third-party notice

`index.ts`, `rules/`, and `shared/` are third-party source, copied from:

- Upstream: <https://github.com/dmmulroy/anti-slop>
- Vendored by: this repository, first in commit `14396ab` ("chore: add the
  oxlint gate, green on what it covers")

The upstream commit hash for the copy was not recorded, and upstream publishes
no LICENSE file that travels with the source, so the grant is recorded here
rather than beside the code: **MIT**, from the upstream repository's
`package.json` and its README license line, the same two places every other
entry in `THIRD_PARTY_NOTICES.md` is read from. Redistribution of this
directory is therefore traceable to an MIT grant, which is what the GPL-3.0
distribution of `agave` needs.

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
`.oxlintrc.json`), so the rules load from this directory and never enter the
resolved dependency tree. `index.ts` imports `@oxlint/plugins`, which is
already a devDependency; only the rule sources are vendored.
