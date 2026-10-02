#!/usr/bin/env python3
"""Unit tests for scripts/test-file.sh, the one-file edit-test loop.

`test-file.sh` exists because `zig test src/<file>.zig` does not work for most
of src/: a Zig module reaches only the files under its root source file's
directory, so a file that imports `../format/dtype.zig` cannot be a root. The
script writes a throwaway bridge root into src/, runs `zig test` against it, and
removes it again.

That is the shape of thing that fails silently and expensively, so the cases
below pin the properties a contributor depends on:

- the bridge is removed on every exit, so `zig build` never compiles a stray
  `src/zz_test_file_root.zig` into the agave binary, and an interrupted run
  leaves nothing behind;
- a path that cannot work is refused *before* a file is written, with a
  message naming the reason rather than a compiler error against a bridge the
  contributor never wrote;
- a file with no tests of its own is refused, because the bridge also runs the
  tests of everything the target imports, and reporting those under the target's
  name reads as a pass;
- an expected-name substring that matches nothing fails the run, because
  `zig test --test-filter` cannot see an imported file's tests: it reports
  "All 0 tests passed." and exits 0 (verified on Zig 0.16.0), so passing the
  filter through would be a green run that tested nothing. build.zig's
  rejectEmptyTestFilters guards the same trap for -Dtest-filter;
- a `zig` that is not the pinned version is refused up front, the same rule
  build.zig enforces for `zig build`.

The script is driven as a subprocess against a staged fixture tree (it derives
its root from BASH_SOURCE), and `zig` is a stub that records its argv and
exits 0, so the suite exercises the script's own logic and argument assembly
without compiling anything. The one case that needs the real compiler is
guarded by ZIG and skipped when it is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "test-file.sh"
BRIDGE = "src/zz_test_file_root.zig"
ZIG_PIN = (REPO / ".zigversion").read_text(encoding="utf-8").strip()

# The helper every staged fixture file shares. Kept as a constant so the
# fixture files themselves stay one short line each and ruff's line length does
# not have to be worked around with an escaped quote soup.
ZIG_SOURCE = "pub fn helper() u32 {\n    return 1;\n}\n\n"


class Fixture:
    """A staged tree the real test-file.sh can run against.

    The script derives its root from BASH_SOURCE and reads only `.zigversion`,
    the target file and `.zig-cache/c/*/options.zig`, so staging those three is
    enough for the script to run its real validation over real file shapes.
    """

    def __init__(self, root: Path, zig_stub: bool = True) -> None:
        self.root = root
        scripts = root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT, scripts / "test-file.sh")
        shutil.copy2(REPO / ".zigversion", root / ".zigversion")

        # A file with tests, in a subdirectory, to exercise the import path the
        # bridge writes and the nested-directory branch of the case block.
        (root / "src" / "ops").mkdir(parents=True, exist_ok=True)
        (root / "src" / "ops" / "widget.zig").write_text(
            ZIG_SOURCE
            + 'test "widget handles the basic case" {\n    _ = helper();\n}\n\n'
            + 'test "widget handles the edge case" {\n    _ = helper();\n}\n',
            encoding="utf-8",
        )

        # A file directly in src/, for the "no parent directory" branch.
        (root / "src" / "top.zig").write_text(
            ZIG_SOURCE + 'test "top level helper" {\n    _ = helper();\n}\n', encoding="utf-8"
        )

        # A file with no tests of its own.
        (root / "src" / "ops" / "notests.zig").write_text(ZIG_SOURCE, encoding="utf-8")

        self.log = root / "zig-argv.txt"
        self.stub_dir = root / "stub-bin"
        if zig_stub:
            self.stub_zig()

    def options(self) -> Path:
        """The build-generated build_options module the real build writes."""
        cache = self.root / ".zig-cache" / "c" / "deadbeefdeadbeefdeadbeefdeadbeef"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "options.zig"
        path.write_text(
            'pub const enable_cuda = false;\npub const version = "test";\n',
            encoding="utf-8",
        )
        return path

    def stub_zig(self, version: str | None = None, exit_code: int = 0) -> None:
        """A `zig` that snapshots the bridge, records its argv, and exits.

        The import path the script derives is the *content* of the bridge file,
        which the script deletes before returning, so the stub copies it aside
        while it still exists: reading the sidecar afterwards is how a case
        checks the derived path without depending on the cleanup order.

        `zig version` answers the pinned version so the script's version check
        passes; every other invocation records and exits with `exit_code`.
        """
        self.stub_dir.mkdir(exist_ok=True)
        stub = self.stub_dir / "zig"
        answer = version if version is not None else ZIG_PIN
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'if [[ "$1" == "version" ]]; then echo "{answer}"; exit 0; fi\n'
            'printf "%s\\n" "$*" >> "$ZIG_LOG"\n'
            "if [[ -f src/zz_test_file_root.zig ]]; then"
            ' cp src/zz_test_file_root.zig "$ZIG_LOG.bridge"; fi\n'
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PATH"] = f"{self.stub_dir}{os.pathsep}{env.get('PATH', '')}"
        env["ZIG_LOG"] = str(self.log)
        return subprocess.run(
            ["bash", "scripts/test-file.sh", *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def argv(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""

    def bridge_source(self) -> str:
        """The bridge file's content, as the stub saw it before cleanup."""
        sidecar = Path(str(self.log) + ".bridge")
        return sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""


class TestRejections(unittest.TestCase):
    """Every shape that cannot work is refused before anything is written."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))
        self.fixture.options()

    def assert_rejected(self, *args: str, needle: str) -> subprocess.CompletedProcess[str]:
        proc = self.fixture.run(*args)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(needle, proc.stderr, proc.stderr)
        self.assertFalse(
            (self.fixture.root / BRIDGE).exists(),
            "a refused run must not leave a bridge behind",
        )
        return proc

    def test_no_arguments_prints_usage(self) -> None:
        self.assert_rejected(needle="usage: scripts/test-file.sh")

    def test_path_outside_src_names_the_alternative(self) -> None:
        self.assert_rejected("tests/models/test_gemma3.zig", needle="are their own root already")

    def test_path_outside_src_rejects_traversal(self) -> None:
        self.assert_rejected("../src/ops/widget.zig", needle="not a file directly under src/")

    def test_missing_file_is_named(self) -> None:
        self.assert_rejected("src/ops/nope.zig", needle="no such file")

    def test_file_without_tests_is_refused(self) -> None:
        proc = self.assert_rejected("src/ops/notests.zig", needle="declares no tests of its own")
        # The reason matters: the bridge also runs the tests of everything the
        # target imports, so accepting the file would report another file's
        # results under this one's name.
        self.assertIn("everything this file imports", proc.stderr)

    def test_unmatched_substring_fails_the_run(self) -> None:
        proc = self.assert_rejected(
            "src/ops/widget.zig",
            "no-such-test-name",
            needle="no test in src/ops/widget.zig matches",
        )
        # The trap being guarded: `zig test --test-filter` would report
        # "All 0 tests passed." and exit 0 here.
        self.assertIn("exits 0", proc.stderr)
        self.assertEqual("", self.fixture.argv(), "zig must not run")

    def test_missing_options_module_names_the_fix(self) -> None:
        cache = self.fixture.root / ".zig-cache"
        if cache.exists():
            shutil.rmtree(cache)
        self.assert_rejected("src/ops/widget.zig", needle="Run `zig build` once")

    def test_wrong_zig_version_is_refused(self) -> None:
        self.fixture.stub_zig(version="0.15.1")
        proc = self.fixture.run("src/ops/widget.zig")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(f"this build needs {ZIG_PIN}", proc.stderr)
        self.assertEqual("", self.fixture.argv(), "zig must not compile")

    def test_existing_bridge_is_not_overwritten(self) -> None:
        bridge = self.fixture.root / BRIDGE
        bridge.write_text("// left behind by an interrupted run\n", encoding="utf-8")
        proc = self.fixture.run("src/ops/widget.zig")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing to overwrite", proc.stderr)
        # The pre-existing content is the contributor's, so it must survive.
        self.assertIn("interrupted run", bridge.read_text(encoding="utf-8"))


class TestSuccessfulRun(unittest.TestCase):
    """The happy path assembles the invocation the manual version needs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Fixture(Path(self._tmp.name))
        self.fixture.options()

    def test_bridge_import_path_is_src_relative(self) -> None:
        proc = self.fixture.run("src/ops/widget.zig")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # A file-scope `const _ = @import(...)` is an unused declaration and
        # carries no tests; the comptime block is what makes Zig include the
        # imported file's tests.
        source = self.fixture.bridge_source()
        self.assertIn('@import("ops/widget.zig")', source)
        self.assertIn("comptime", source)
        self.assertFalse(
            (self.fixture.root / BRIDGE).exists(),
            "the bridge must be gone once the run returns",
        )

    def test_top_level_file_has_no_parent_in_the_import(self) -> None:
        self.assertEqual(self.fixture.run("src/top.zig").returncode, 0)
        self.assertIn('@import("top.zig")', self.fixture.bridge_source())

    def test_release_safe_and_libc_and_build_options(self) -> None:
        self.assertEqual(self.fixture.run("src/ops/widget.zig").returncode, 0)
        argv = self.fixture.argv()
        # ReleaseSafe: Debug breaks linking with GCC 16 .sframe, and ReleaseFast
        # would no-op every std.debug.assert in the file under test.
        self.assertIn("-OReleaseSafe", argv)
        self.assertIn("-lc", argv)
        self.assertIn("build_options", argv)

    def test_matching_substring_is_accepted(self) -> None:
        proc = self.fixture.run("src/ops/widget.zig", "edge case")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("expected to match", proc.stdout)

    def test_bridge_is_removed_after_a_failing_run(self) -> None:
        self.fixture.stub_zig(exit_code=1)
        proc = self.fixture.run("src/ops/widget.zig")
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(
            (self.fixture.root / BRIDGE).exists(),
            "a failing run must not leave a bridge behind",
        )


# Opt in rather than default: the case compiles the whole ops/kv_quant.zig test
# graph, which is a minute of work that the stub cases above do not need, and
# CI's python-tests job has no Zig at all. Run it when the compiler is what you
# are testing:
#     TEST_FILE_E2E=1 python3 scripts/test_test_file.py
@unittest.skipUnless(
    os.environ.get("TEST_FILE_E2E") == "1" and shutil.which("zig"),
    "the end-to-end case needs a real compile; set TEST_FILE_E2E=1",
)
class TestEndToEnd(unittest.TestCase):
    """One run against the real compiler, proving the bridge really works.

    The stub above proves the script's logic; this proves the thing the script
    exists for, that a file that `zig test` refuses to compile as a root runs
    through the bridge and reports its tests. It needs `zig build` to have run
    once, for the build_options module the script reuses.
    """

    def test_import_outside_module_path_runs_through_the_bridge(self) -> None:
        if not list((REPO / ".zig-cache" / "c").glob("*/options.zig")):
            self.skipTest("no build_options module; run `zig build` once")

        direct = subprocess.run(
            ["zig", "test", "src/ops/kv_quant.zig"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if direct.returncode == 0:
            self.skipTest("this Zig build can root a module on a subdirectory")

        bridged = subprocess.run(
            ["bash", "scripts/test-file.sh", "src/ops/kv_quant.zig"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        self.assertEqual(bridged.returncode, 0, bridged.stdout + bridged.stderr)
        # The point of the whole script: the tests the direct invocation could
        # not reach, reported by name. `zig test` prints its progress and its
        # summary on stderr (the test runner speaks a protocol over it), so
        # that is where a pass has to show up.
        self.assertIn("ops.kv_quant.test.", bridged.stderr)
        self.assertNotIn("0 tests passed", bridged.stderr)
        self.assertFalse((REPO / BRIDGE).exists(), "the bridge must not survive the run")


if __name__ == "__main__":
    unittest.main()
