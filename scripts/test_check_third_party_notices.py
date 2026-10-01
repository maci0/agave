#!/usr/bin/env python3
"""Unit tests for the inventory in check-third-party-notices.py.

The bundle closure is recomputed from package.json and bun.lock and compared
against THIRD_PARTY_NOTICES.md, so a package that ships without an entry fails
the gate. Two dependency sets sit outside both files: the inline requirements
of a PEP 723 header, and the globally installed diagram tools. These tests
cover the PEP 723 half, which had no check at all before it: a range or a
missing notices entry passed silently, so the parse is pinned here.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parent / "check-third-party-notices.py"
_spec = importlib.util.spec_from_file_location("check_third_party_notices", CHECKER)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot import {CHECKER}")
notices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notices)

# A PEP 723 header, exactly the shape uv writes.
SCRIPT_WITH_DEP = '''# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools==4.66.0"]
# ///
"""Docstring after the header."""
'''

# The same script with no requirement: the stdlib-only scripts in the tree.
SCRIPT_STDLIB_ONLY = '''# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Docstring after the header."""
'''

SCRIPT_WITHOUT_HEADER = '"""No PEP 723 header, so no inline environment."""\n'


class Pep723RequirementsTest(unittest.TestCase):
    """pep723_requirements() finds the header and ignores everything else."""

    def _requirements(self, files: dict[str, str]) -> set[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return notices.pep723_requirements(root)

    def test_finds_an_inline_requirement(self) -> None:
        found = self._requirements({"scripts/brand-glyphs.py": SCRIPT_WITH_DEP})
        self.assertEqual(found, {"fonttools==4.66.0"})

    def test_empty_requirement_list_yields_nothing(self) -> None:
        found = self._requirements({"scripts/check-docs.py": SCRIPT_STDLIB_ONLY})
        self.assertEqual(found, set())

    def test_file_without_a_header_is_ignored(self) -> None:
        found = self._requirements({"scripts/plain.py": SCRIPT_WITHOUT_HEADER})
        self.assertEqual(found, set())

    def test_a_range_is_reported_rather_than_accepted(self) -> None:
        # Not a pin check, which lives in check-pins.sh: this asserts the
        # requirement is visible to the inventory at all, so a range written
        # there has to be noticed by somebody.
        ranged = SCRIPT_WITH_DEP.replace("fonttools==4.66.0", "fonttools>=4.66")
        found = self._requirements({"scripts/brand-glyphs.py": ranged})
        self.assertEqual(found, {"fonttools>=4.66"})

    def test_a_header_wrapped_across_lines_is_read(self) -> None:
        # uv reads a multi-line requirement block the same as an inline one,
        # so this must too: otherwise a reformat silently drops the
        # requirement out of the inventory.
        wrapped = '# /// script\n# dependencies = [\n#     "fonttools==4.66.0",\n#     "pillow>=11.0",\n# ]\n# ///\n'
        found = self._requirements({"scripts/brand-glyphs.py": wrapped})
        self.assertEqual(found, {"fonttools==4.66.0", "pillow>=11.0"})

    def test_the_parse_stops_at_the_end_of_the_list(self) -> None:
        # `requires-python` and the docstring follow the block; a read that ran
        # past the closing bracket would pick up a token out of one of them.
        found = self._requirements({"scripts/brand-glyphs.py": SCRIPT_WITH_DEP})
        self.assertEqual(found, {"fonttools==4.66.0"})

    def test_virtualenv_copies_are_skipped(self) -> None:
        files = {
            "scripts/brand-glyphs.py": SCRIPT_WITH_DEP,
            "tests/.venv/lib/site-packages/_copy.py": SCRIPT_WITH_DEP,
        }
        found = self._requirements(files)
        self.assertEqual(found, {"fonttools==4.66.0"})


class ListedPep508Test(unittest.TestCase):
    """listed_pep508() reads notices entries without mixing in bundle entries."""

    def test_reads_a_pep508_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "THIRD_PARTY_NOTICES.md"
            path.write_text("| `fonttools==4.66.0` | MIT | x | y |\n", encoding="utf-8")
            self.assertEqual(notices.listed_pep508(path), {"fonttools==4.66.0"})

    def test_a_bare_version_entry_is_not_a_pep508_entry(self) -> None:
        # The bundle table is `name@version`; it must not satisfy the inline
        # half, or a package that ships would stand in for one that does not.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "THIRD_PARTY_NOTICES.md"
            path.write_text("| `preact@10.29.8` | MIT | x | y |\n", encoding="utf-8")
            self.assertEqual(notices.listed_pep508(path), set())
            self.assertEqual(notices.listed_entries(path), {"preact@10.29.8"})


class RealTreeTest(unittest.TestCase):
    """The tree as committed: what the header requires is what the file records."""

    def test_repo_header_matches_the_notices_entry(self) -> None:
        root = notices.find_root()
        required = notices.pep723_requirements(root)
        listed = notices.listed_pep508(root / "THIRD_PARTY_NOTICES.md")
        self.assertEqual(required - listed, set(), "PEP 723 requirement missing from THIRD_PARTY_NOTICES.md")
        self.assertEqual(
            listed - required,
            set(),
            "THIRD_PARTY_NOTICES.md lists an inline requirement nothing declares",
        )

    def test_the_one_inline_requirement_is_fonttools(self) -> None:
        # Pins the inventory to the set it is expected to hold, so a new
        # inline dependency is a manifest diff here and not a silent arrival.
        self.assertEqual(notices.pep723_requirements(notices.find_root()), {"fonttools==4.66.0"})


if __name__ == "__main__":
    unittest.main()
