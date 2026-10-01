# vendor/

Two shim packages that make `react` and `react-dom` resolve to Preact, and
`patches/`, the local patches `bun install` applies to pinned dependencies.

## Why they exist

agave's two chat surfaces (`src/web/`, `web/`) run Preact. The shadcn/ui
components in `src/web/ui/` and the Radix primitives they build on import from
`react` and `react-dom`, because that is the API those packages are published
against, and the sources keep those canonical imports.

The usual way to bridge that is a bundler alias (`react` → `preact/compat`).
The pinned bundler here is bun 1.4.2, which has no module aliasing: no `--alias`
flag in `bun build`, and bunfig's `[alias]` table is not consulted by
`bun build`, `bun install` or `bun run`. `bun.lock` alias entries are rejected by
`scripts/check-third-party-notices.py` by design.

So the substitution is done with the one mechanism every resolver honours:
`package.json` declares `react` and `react-dom` as `file:` dependencies, and
these packages answer to those names. `preact/compat` is the real
implementation; each file here re-exports it. The bundler, the test run
(`bun test src/web`) and the type checker all resolve the same graph.

## What each file is

| Package | Specifier | Target |
| --- | --- | --- |
| `react` | `react` | `preact/compat` |
| `react` | `react/jsx-runtime` | `preact/jsx-runtime` |
| `react` | `react/jsx-dev-runtime` | `preact/jsx-dev-runtime` |
| `react-dom` | `react-dom` | `preact/compat` |
| `react-dom` | `react-dom/client` | `preact/compat/client` |

These files are configuration, not application code: every one of them is a
one-line re-export, which is why `vendor/**` is in `.oxlintrc.json`
`ignorePatterns` (they are barrels by construction and there is nothing in them
to lint).

## Types come from React, not from these files

`tsconfig.json` maps the `react`/`react-dom` specifiers to `@types/react` and
`@types/react-dom` rather than letting them resolve to `preact/compat`'s
declarations. `preact/compat` implements the React API, so React's declarations
are the accurate description of the calls the source makes, and the Radix
declarations the components consume are written against them. Resolving types
through these shims instead made every hook and prop access `any` and turned
the tree's `no-unsafe-*` lint rules red.

`@types/react` and `@types/react-dom` are build-only (devDependencies): nothing
from them reaches a bundle.

## Removing them

If the bundler gains module aliasing, delete this directory, drop the two
`file:` dependencies from `package.json`, and add the alias to the build
command. Nothing else changes: no source file imports these paths directly.

## patches/

`bun install` applies each file here to the exact package version named in
`package.json` `patchedDependencies`; a version bump without a matching patch
fails the install instead of quietly running the pristine package.

| Patch | What it changes | Why |
| --- | --- | --- |
| `@rikalabs%2Foxlint-standards@0.8.1.patch` | Drops `oxc/no-map-object-keys`, `unicorn/prefer-logical-operator-over-short-circuit`, `import/no-extraneous-dependencies`, `import/no-unresolved` and `import/no-reexport` from the presets; renames `oxc/no-new-buffer` to `unicorn/no-new-buffer` | oxlint 1.86 implements none of those names and refuses a config that enables an unknown rule. Every other rule in the strict chain stays on. Remove the patch once a preset release matches oxlint's rule set. |

`VENDORED.sha256` here records the sha256 of every patch in this directory, and
`zig build check-pins` fails when a file stops matching its recorded value, when
an unrecorded `.patch` appears, or when `patchedDependencies` points at a patch
outside this directory or at one the manifest does not list. The reason is the
same as for `tools/oxlint/anti-slop`, and stronger here: `bun install` applies
these patches to `node_modules` on every install, before any script runs, and
`.oxlintrc.json` extends the preset they edit, so an unreviewed byte here
silently decides which rules lint-web enforces over `src/web` while
`package.json`, `bun.lock` and `VENDORED.sha256` all stay byte-identical.
Regenerate the manifest in the same change as the patch:

```sh
find . -name '*.patch' -type f | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > VENDORED.sha256
```

Rebuild a patch with `bun patch @rikalabs/oxlint-standards`, edit the files under
`node_modules/`, then
`bun patch --commit node_modules/@rikalabs/oxlint-standards --patches-dir vendor/patches`.
