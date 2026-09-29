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

Only the production closure is checked: devDependencies are installed but never
bundled, and the Python trees are harnesses that no release artifact carries.
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

    print(f"Third-party notices OK: {len(closure)} bundled packages listed in THIRD_PARTY_NOTICES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
