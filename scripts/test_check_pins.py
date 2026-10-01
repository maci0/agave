#!/usr/bin/env python3
"""Unit tests for the dependency-integrity checks in check-pins.sh.

`check-pins.sh` decides whether a contributor's tree can install what CI will
install: it reconciles every spelling of each pin, hashes the vendored oxlint
rules, and now hashes the patches `bun install` applies. A check that fails to
fire is the failure mode that matters here, because each one guards a hole
nothing else in the tree can see:

- a patch file is read by `bun install` before any script runs and rewritten
  into `node_modules/@rikalabs/oxlint-standards`, so an edit decides which
  rules lint-web enforces while package.json, bun.lock and the manifests all
  stay byte-identical;
- a `patchedDependencies` entry can name a patch that is not in the tree, or
  one outside vendor/patches, and the install still succeeds;
- `@types/bun` and the bun pin are one version by construction, so a bump that
  edits only packageManager's sibling leaves `tsc` typing the UI against an API
  the pinned runner no longer has.

The tests drive the real script end to end against a staged fixture tree, which
is what `check-pins.sh` reads: it resolves its own root from `BASH_SOURCE`, so
copying it beside the fixture files is enough and the logic under test is the
logic that runs in CI. Re-implementing the parses here would test a copy that
can drift from the script; running the script cannot.

Fixtures stage every file the script reads (see FIXTURE_FILES). uv is stubbed
out with a directory holding a fake `uv` on PATH, so the suite stays
network-free and does not resolve anything: `uv lock --check` is not what these
cases are about, and the real uv would need a cache directory to answer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check-pins.sh"

# Every path check-pins.sh reads, relative to the root it derives from
# BASH_SOURCE. The suite copies these into the fixture so the script runs its
# real parses over real file shapes instead of a reconstruction of them.
FIXTURE_FILES = (
    ".zigversion",
    "build.zig",
    "build.zig.zon",
    "Dockerfile",
    "docker-compose.yml",
    "package.json",
    "ruff.toml",
    "scripts/lint-python.sh",
    "scripts/product-version.sh",
    "scripts/check-reproducible.sh",
    "scripts/check-docker-image.sh",
    ".github/workflows/ci.yml",
    "tests/pyproject.toml",
    "tests/uv.lock",
    "research/kernels/pyproject.toml",
    "research/kernels/uv.lock",
    "vendor/patches/@rikalabs%2Foxlint-standards@0.8.1.patch",
)

# The vendored oxlint rules the script hashes, with the manifest that anchors
# them. Copied whole so the anti-slop check runs its own logic on real bytes.
VENDORED_ANTI_SLOP = REPO / "tools" / "oxlint" / "anti-slop"

# The patch bun install applies, and the manifest that anchors it.
PATCH_FILE = "@rikalabs%2Foxlint-standards@0.8.1.patch"


def sha256_of(path: Path) -> str:
    """The digest the manifests are written in: `sha256sum`, first field."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckPinsFixture:
    """A staged tree the real check-pins.sh can run against.

    The script derives its root from its own location, so the fixture is built
    by copying scripts/check-pins.sh in beside the files it reads. Mutating
    methods return self so a case reads as the sequence of edits it makes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        # The script under test. It resolves its root from BASH_SOURCE, so a
        # copy here reads the fixture files beside it and nothing else.
        scripts = root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CHECKER, scripts / "check-pins.sh")
        for rel in FIXTURE_FILES:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dest)

        # The vendored oxlint rules and their manifest.
        anti = root / "tools" / "oxlint" / "anti-slop"
        shutil.copytree(VENDORED_ANTI_SLOP, anti)

        # The patches and their manifest, regenerated from the staged bytes so
        # the fixture starts in the state the tree is in when the check passes.
        self.rewrite_patch_manifest()

        # uv stands in as a stub: `uv lock --check` needs a writable cache and
        # the network, and none of these cases assert on it. An empty stub that
        # answers 0 keeps the script's own "is uv on PATH" branch taken and the
        # rest of the script reachable.
        self._stub_uv()

    def _stub_uv(self) -> None:
        stub_dir = self.root / "stub-bin"
        stub_dir.mkdir(exist_ok=True)
        uv = stub_dir / "uv"
        uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        uv.chmod(0o755)

    # -- patch manifest ----------------------------------------------------

    @property
    def patch_path(self) -> Path:
        return self.root / "vendor" / "patches" / PATCH_FILE

    @property
    def patch_manifest(self) -> Path:
        return self.root / "vendor" / "patches" / "VENDORED.sha256"

    def rewrite_patch_manifest(self) -> None:
        """Regenerate vendor/patches/VENDORED.sha256 from the staged bytes.

        Public because a deliberate re-patch is a supported way through the
        check: the manifest moves in the same change as the patch.
        """
        lines = []
        for path in sorted(self.root.glob("vendor/patches/*.patch")):
            lines.append(f"{sha256_of(path)}  {path.name}\n")
        self.patch_manifest.write_text("".join(lines), encoding="utf-8")

    def edit_patch(self, extra: str = "\n# drift\n") -> None:
        """Append bytes to the patch: the shape an unreviewed edit takes."""
        with self.patch_path.open("a", encoding="utf-8") as handle:
            handle.write(extra)

    def add_stray_patch(self) -> Path:
        """Add a .patch nothing anchors, so the reverse direction has a case."""
        stray = self.root / "vendor" / "patches" / "stray.patch"
        stray.write_text("--- a\n+++ b\n", encoding="utf-8")
        return stray

    def set_patch_target(self, target: str) -> None:
        """Point package.json patchedDependencies at `target`."""
        pkg = self.root / "package.json"
        pkg.write_text(
            pkg.read_text(encoding="utf-8").replace(f'"vendor/patches/{PATCH_FILE}"', f'"{target}"'),
            encoding="utf-8",
        )

    # -- manifest pins -----------------------------------------------------

    def set_types_bun(self, version: str) -> None:
        pkg = self.root / "package.json"
        text = pkg.read_text(encoding="utf-8")
        start = text.index('"@types/bun": "')
        end = text.index('"', start + len('"@types/bun": "'))
        pkg.write_text(
            text[:start] + f'"@types/bun": "{version}"' + text[end:],
            encoding="utf-8",
        )

    def drop_types_bun(self) -> None:
        pkg = self.root / "package.json"
        pkg.write_text(
            pkg.read_text(encoding="utf-8").replace('"@types/bun": "1.4.2",', ""),
            encoding="utf-8",
        )

    # -- run ---------------------------------------------------------------

    def run(self) -> subprocess.CompletedProcess:
        """Run the real script inside the fixture, with the uv stub on PATH."""
        env = dict(os.environ)
        env["PATH"] = f"{self.root / 'stub-bin'}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            ["bash", str(self.root / "scripts" / "check-pins.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


class CheckPinsPatchManifestTest(unittest.TestCase):
    """The patch manifest: what must fail, and what must stay green."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "tree"
        self.root.mkdir()
        self.fixture = CheckPinsFixture(self.root)

    def test_untouched_fixture_passes(self) -> None:
        """The state the repository is in is green, so the assertions below
        are about the edits and not about a fixture that never passed."""
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Patch source OK", result.stdout)

    def test_edited_patch_fails(self) -> None:
        """A byte added to the patch is the hole this closes: bun install
        applies it to node_modules and nothing else in the tree records it."""
        self.fixture.edit_patch()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not the patch", result.stderr)
        self.assertIn(PATCH_FILE, result.stderr)

    def test_unlisted_patch_fails(self) -> None:
        """The reverse direction: a .patch nothing hashes can be pointed at by
        a patchedDependencies entry without any manifest change."""
        self.fixture.add_stray_patch()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("stray.patch", result.stderr)
        self.assertIn("not in", result.stderr)

    def test_patch_target_outside_vendor_fails(self) -> None:
        """package.json naming a patch outside vendor/patches is the same hole
        one level up, and must not resolve."""
        self.fixture.set_patch_target("scripts/somewhere-else.patch")
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("outside vendor/patches", result.stderr)

    def test_missing_patch_file_fails(self) -> None:
        """A patchedDependencies entry whose file is gone must fail, not pass
        because the manifest no longer lists it either."""
        self.fixture.patch_path.unlink()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("no longer in the tree", result.stderr)

    def test_manifest_rewrite_makes_the_edit_legitimate(self) -> None:
        """The other direction of the same gate: a deliberate re-patch, with the
        manifest regenerated in the same change, is accepted. Without this the
        check would be a lock the tree cannot legitimately leave."""
        self.fixture.edit_patch()
        self.fixture.rewrite_patch_manifest()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Patch source OK", result.stdout)


class CheckPinsTypesBunTest(unittest.TestCase):
    """@types/bun and the bun pin are one version by construction."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "tree"
        self.root.mkdir()
        self.fixture = CheckPinsFixture(self.root)

    def test_matching_pin_passes(self) -> None:
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Bun pin OK", result.stdout)

    def test_drifted_types_bun_fails(self) -> None:
        """A @types/bun that trails the bun pin typechecks the UI against an
        API the pinned runner no longer has, with nothing reporting it."""
        self.fixture.set_types_bun("1.4.1")
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("@types/bun", result.stderr)

    def test_unparseable_types_bun_fails(self) -> None:
        """A missing pin is not a passing pin: without the entry the parse
        returns empty and the check must say so rather than compare nothing."""
        self.fixture.drop_types_bun()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("could not parse", result.stderr)


if __name__ == "__main__":
    unittest.main()
