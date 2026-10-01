# Third-party notices

Agave is GPL-3.0-or-later. It ships third-party code, so this file records
where that code came from and under what grant.

## Shipped in the committed bundles

`src/web/app.js` (embedded by the server into the chat page) and
`web/shell.js` (shipped beside `agave.wasm`) are minified bun bundles. The
bundler drops every upstream license header, so the notices those packages
carry in their own source trees exist nowhere in the release. The list below
is what travels instead.

Versions are the exact pins in `package.json` and `bun.lock`; the closure is
recomputed from those two files by `scripts/check-third-party-notices.py`
(`zig build check-third-party`), which fails when a package ships without an
entry here or an entry names a package that no longer ships.

`react` and `react-dom` are not listed: they are repo-local shims
(`vendor/react`, `vendor/react-dom`) that re-export `preact/compat` under the
names the shadcn/ui components and Radix primitives import. See
[vendor/README.md](vendor/README.md). The `preact` entry below is what those
shims ship.

Each package is listed with its SPDX license identifier and upstream URL. The
full license text is not inlined here: it ships inside the package itself
under `node_modules/<name>/LICENSE`, and upstream publishes it at the tagged
revision linked in the table.

| Package (name@version) | License | Upstream |
| --- | --- | --- |
| `@radix-ui/primitive@1.1.7` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-compose-refs@1.1.5` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-context@1.2.2` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-dialog@1.1.23` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-dismissable-layer@1.1.19` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-focus-guards@1.1.6` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-focus-scope@1.1.16` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-id@1.1.4` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-label@2.1.15` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-portal@1.1.17` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-presence@1.1.10` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-primitive@2.1.10` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-slot@1.3.3` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-use-callback-ref@1.1.4` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-use-controllable-state@1.2.6` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-use-effect-event@0.0.5` | MIT | <https://github.com/radix-ui/primitives> |
| `@radix-ui/react-use-layout-effect@1.1.4` | MIT | <https://github.com/radix-ui/primitives> |
| `aria-hidden@1.2.6` | MIT | <https://github.com/theKashey/aria-hidden> |
| `class-variance-authority@0.7.1` | Apache-2.0 | <https://github.com/joe-bell/cva> |
| `clsx@2.1.1` | MIT | <https://github.com/lukeed/clsx> |
| `detect-node-es@1.1.0` | MIT | <https://github.com/thekashey/detect-node> |
| `get-nonce@1.0.1` | MIT | <ssh://git@github.com/theKashey/get-nonce> |
| `lucide-react@1.48.0` | ISC | <https://github.com/lucide-icons/lucide> |
| `preact@10.29.8` | MIT | <https://github.com/preactjs/preact> |
| `react-remove-scroll@2.7.2` | MIT | <https://github.com/theKashey/react-remove-scroll> |
| `react-remove-scroll-bar@2.3.8` | MIT | <https://github.com/theKashey/react-remove-scroll-bar> |
| `react-style-singleton@2.2.3` | MIT | <https://github.com/theKashey/react-style-singleton> |
| `tailwind-merge@3.7.0` | MIT | <https://github.com/dcastil/tailwind-merge> |
| `tslib@2.8.1` | 0BSD | <https://github.com/Microsoft/tslib> |
| `use-callback-ref@1.3.3` | MIT | <https://github.com/theKashey/use-callback-ref> |
| `use-sidecar@1.1.3` | MIT | <https://github.com/theKashey/use-sidecar> |

`class-variance-authority` is the only Apache-2.0 package in the bundle.
Apache-2.0 is one-way compatible with GPL-3.0, so its code is usable in a
GPL-3.0-or-later work, and its patent grant and NOTICE obligations travel with
it (upstream publishes no NOTICE file, so there is nothing further to carry).
Every other entry is MIT, ISC or 0BSD: permissive, and each requires only that
its copyright and permission notice travel with the code, which is what this
file and the packages themselves provide.

## Not shipped

Development and build tooling is installed on a contributor's machine and
never reaches a release artifact: `oxlint`, `oxlint-tsgolint`, `@oxlint/plugins`,
`@rikalabs/oxlint-standards`, `@shadcn/lint` (and its `cn` and
`@eslint/core` dependencies), `vnu-jar` (the W3C Nu validator), `typescript`, `tailwindcss`, `@tailwindcss/cli`,
`happy-dom` (via `@happy-dom/global-registrator`), `@types/*`, and
`@parcel/watcher` (pulled in by the Tailwind CLI). Their licenses are MIT or
Apache-2.0 and are recorded in their own packages.

The Python trees are harnesses, not shipped code: `rich` for
`tests/harness.py`, and `torch`, `numpy`, `gguf`, `optuna` under
`research/kernels/`. They run in a developer or CI environment and are not part
of any distributed artifact.

Two dependency sets live in the tree but in no manifest, so neither
`package.json` nor any `uv.lock` records them. Both are build-time only:

| Requirement | License | Used by | Why it is not a lockfile entry |
| --- | --- | --- | --- |
| `fonttools==4.66.0` | MIT | `scripts/brand-glyphs.py` | PEP 723 header: `uv run` on the file builds its own environment from it. Nothing else references it, so a project-level entry would be an orphan. |

`docs/render-diagrams.mjs` is the other one: it pins `beautiful-mermaid` (MIT)
and `@resvg/resvg-js` (MPL-2.0) in its own header and installs them with
`bun add -g`, deliberately outside `package.json` so a layout engine and a
native rasterizer stay out of every contributor's and CI runner's
`bun install`. Neither ships: the rendered PNG/SVG copies carry no third-party
code. `scripts/check-pins.sh` holds the exact version of everything named
here, and `scripts/check-third-party-notices.py` fails when the `fonttools`
pin drifts from the header that carries it.

The Zig engine itself has no third-party dependencies: `build.zig.zon` declares
none, so there is nothing to attribute there.

`tools/oxlint/anti-slop/` is vendored third-party source with its own notice at
[tools/oxlint/anti-slop/NOTICE.md](tools/oxlint/anti-slop/NOTICE.md). It is
loaded from a path by oxlint, not bundled, and its upstream license text is
still unrecorded there.

## Updating this file

After a dependency bump, run `zig build check-third-party`. It names every
package that ships without an entry; add each with its upstream license and URL
in the same change. The reverse check also fires, so an entry for a package
that has left the bundle is removed rather than left to drift.
