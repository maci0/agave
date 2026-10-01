# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Every third-party package that ships in a committed bundle is listed in THIRD_PARTY_NOTICES.md.

`src/web/app.js` and `web/shell.js` are committed minified bundles of Preact,
Radix, lucide and friends. The bundler drops every upstream license header, and
the project is GPL-3.0-or-later, which requires the copyright and permission
notices of the bundled code to travel with it. A notices file nobody regenerates
is a notices file that goes stale on the first dependency bump, so the closure
is recomputed here from package.json and bun.lock and compared both ways.

Two other dependency sets exist in-tree, and neither is covered by
package.json, bun.lock or any uv.lock, so nothing else can see them:

- PEP 723 headers (`# /// script`) make a Python file its own environment.
  `uv run scripts/brand-glyphs.py` resolves the requirements in that header
  against PyPI with no lockfile in between.
- docs/render-diagrams.mjs pins its two renderer packages in a comment and
  installs them globally, so no manifest holds them.

Neither ships in a release artifact, so the closure above cannot name them and
the versioned `name@version` table is the wrong shape for them. They are
recorded in the notices file in PEP 508 form (`name==version`), which the
table's PACKAGE_REF does not match, and the check below compares that set both
ways so neither direction drifts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# bun.lock stores every resolved package as
#   "name": ["name@version", "", { dependencies: {...} }, "sha512-..."],
# with one line per package and no nesting beyond the dependency object. The
# manifest is JSONC (trailing commas), so it is not json.loads-able as is.
PACKAGE_ENTRY = re.compile(
    r'^\s{4}"?(?P<name>[^"\n]+?)"?:\s*\["(?P<name2>[^"]+?)@(?P<version>\d[^"]*)",'
    r'\s*"",\s*(?P<deps>\{.*?\}),\s*"sha',
    re.MULTILINE | re.DOTALL,
)


def find_root() -> Path:
    """Walk up from this file to the directory holding package.json."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "package.json").is_file():
            return candidate
    sys.exit("check-third-party-notices: could not find the project root (no package.json above this file)")


def strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments so json.loads accepts package.json."""
    without_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", without_block, flags=re.MULTILINE)


def read_lock(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Map package name -> version -> its dependency names, from bun.lock."""
    body = (root / "bun.lock").read_text()
    packages: dict[str, dict[str, dict[str, str]]] = {}
    for match in PACKAGE_ENTRY.finditer(body):
        key = match.group("name").strip('"')
        name = match.group("name2")
        if key != name:
            if not key.endswith("/" + name):
                # A bun alias entry (["alias@npm:name@version", ...]) resolves
                # to a package the notices file would have to name differently.
                # The tree has none today, so fail rather than skip one.
                sys.exit(
                    f"check-third-party-notices: bun.lock entry {key} is an alias, which this script does not resolve"
                )
            # "<parent>/<name>" with "bundled": true is a dependency npm ships
            # inside the parent tarball. The same name@version has its own
            # top-level entry, so walking the top-level map already covers it.
            continue
        versions = packages.setdefault(name, {})
        versions[match.group("version")] = json.loads(match.group("deps")).get("dependencies", {})
    if not packages:
        sys.exit("check-third-party-notices: no packages parsed from bun.lock")
    return packages


def local_package(root: Path, spec: str) -> Path | None:
    """Resolve a `file:`/`link:` dependency spec to its directory.

    `vendor/react` and `vendor/react-dom` are repo-local shims that re-export
    preact/compat under the names the shadcn/ui components and Radix import. They
    are this project's own source, so they are not third-party notices entries;
    what they re-export is, so the walk continues into their manifests.
    """
    for prefix in ("file:", "link:"):
        if spec.startswith(prefix):
            return (root / spec[len(prefix) :]).resolve()
    return None


def production_closure(root: Path) -> set[str]:
    """Every third-party name@version reachable from package.json dependencies."""
    manifest = json.loads(strip_jsonc((root / "package.json").read_text()))
    lock = read_lock(root)

    deps: list[tuple[str, str]] = list(manifest["dependencies"].items())
    missing = sorted(name for name, spec in deps if name not in lock and local_package(root, spec) is None)
    if missing:
        sys.exit(f"check-third-party-notices: package.json dependencies absent from bun.lock: {', '.join(missing)}")

    closure: set[str] = set()
    queue: list[tuple[str, str]] = deps
    while queue:
        name, spec = queue.pop()
        shim = local_package(root, spec)
        if shim is not None:
            inner = json.loads(strip_jsonc((shim / "package.json").read_text()))
            queue.extend(inner.get("dependencies", {}).items())
            continue
        versions = lock.get(name)
        if versions is None:
            sys.exit(f"check-third-party-notices: transitive dependency {name} is absent from bun.lock")
        for version, deps_of in versions.items():
            closure.add(f"{name}@{version}")
            queue.extend(deps_of.items())
    return closure


# A package reference: an optional @scope, a name, an exact version. Only
# these are compared, so prose that mentions a package without pinning it
# (an upgrade note, a version range) is not mistaken for a listed entry.
PACKAGE_REF = re.compile(r"`(?P<name>@?[\w.-]+(?:/[\w.-]+)*)@(?P<version>\d[^\s`]*)`")


def listed_entries(notices: Path) -> set[str]:
    """name@version pairs listed in the notices file."""
    return {match.group(0).strip("`") for match in PACKAGE_REF.finditer(notices.read_text())}


# A PEP 508 reference (`name==version`), the form an inline requirement takes in
# a PEP 723 header. Distinct from PACKAGE_REF, so a `name==version` in the
# notices file is not read as a bundle entry and vice versa.
PEP508_REF = re.compile(r"`(?P<name>@?[\w.-]+)==(?P<version>[^\s`]+)`")


def listed_pep508(notices: Path) -> set[str]:
    """name==version pairs listed in the notices file."""
    return {match.group(0).strip("`") for match in PEP508_REF.finditer(notices.read_text())}


# `# dependencies = ["fonttools==4.66.0", ...]` in a PEP 723 header. The list
# is read as a block from the opening bracket to the line that closes it, so a
# header that wraps across lines is read the same as an inline one: uv reads
# both identically, so this must too.
PEP723_DEPENDENCIES = re.compile(r"^# dependencies = \[(?P<body>.*?)\]", re.MULTILINE | re.DOTALL)
PEP723_MARKER = re.compile(r"^# /// script$", re.MULTILINE)


def pep723_requirements(root: Path) -> set[str]:
    """Every inline requirement a PEP 723 script in the tree carries."""
    found: set[str] = set()
    for base in ("scripts", "tools", "tests"):
        for path in sorted((root / base).rglob("*.py")):
            if ".venv" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not PEP723_MARKER.search(text):
                continue
            for body in PEP723_DEPENDENCIES.findall(text):
                found.update(re.findall(r'"([^"]+)"', body))
    return found


def main() -> int:
    root = find_root()
    notices = root / "THIRD_PARTY_NOTICES.md"
    if not notices.is_file():
        sys.exit(
            "check-third-party-notices: THIRD_PARTY_NOTICES.md is missing; the committed "
            "web bundles carry no license headers of their own"
        )

    closure = production_closure(root)
    listed = listed_entries(notices)

    unlisted = sorted(closure - listed)
    if unlisted:
        sys.exit(
            "check-third-party-notices: these bundled packages are missing from THIRD_PARTY_NOTICES.md:\n"
            + "\n".join(f"  {entry}" for entry in unlisted)
            + "\nadd each with its upstream license and version, then rerun"
        )

    stale = sorted(listed - closure)
    if stale:
        sys.exit(
            "check-third-party-notices: THIRD_PARTY_NOTICES.md lists packages that no longer ship in a bundle:\n"
            + "\n".join(f"  {entry}" for entry in stale)
            + "\nremove them, or move them to the not-shipped section if they are build-only"
        )

    inline = pep723_requirements(root)
    listed_inline = listed_pep508(notices)
    unlisted_inline = sorted(inline - listed_inline)
    if unlisted_inline:
        sys.exit(
            "check-third-party-notices: these PEP 723 script requirements are missing from "
            "THIRD_PARTY_NOTICES.md:\n"
            + "\n".join(f"  {entry}" for entry in unlisted_inline)
            + "\nrecord each with its license in the not-shipped section, then rerun"
        )
    stale_inline = sorted(listed_inline - inline)
    if stale_inline:
        sys.exit(
            "check-third-party-notices: THIRD_PARTY_NOTICES.md lists PEP 508 requirements that "
            "no PEP 723 header in the tree carries:\n"
            + "\n".join(f"  {entry}" for entry in stale_inline)
            + "\nremove them, or record the package that needs them"
        )

    print(
        f"Third-party notices OK: {len(closure)} bundled packages and "
        f"{len(inline)} PEP 723 requirement(s) listed in THIRD_PARTY_NOTICES.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
