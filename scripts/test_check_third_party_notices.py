#!/usr/bin/env python3
"""Unit tests for the inventory in check-third-party-notices.py.

The bundle closure is recomputed from package.json and bun.lock and compared
against THIRD_PARTY_NOTICES.md, so a package that ships without an entry fails
the gate. Two dependency sets sit outside both files: the inline requirements
of a PEP 723 header, and the packages a header installs with `bun add -g`.
These tests cover both, which had no check at all before: a range or a
missing notices entry passed silently, so the parses are pinned here.
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

# The renderer header that installs the diagram tools outside any manifest: the
# pins live on the `bun add -g` line, which is also the command a re-run has to
# type.
SCRIPT_WITH_GLOBAL_ADD = """#!/usr/bin/env bun
/**
 * Usage: bun run docs/render-diagrams.mjs
 *
 * Requires (installed globally via bun, at the versions pinned below):
 *   bun add -g beautiful-mermaid@1.1.3 @resvg/resvg-js@2.6.2
 */

import { renderMermaidSVG } from 'beautiful-mermaid';
import { Resvg } from '@resvg/resvg-js';
"""

SCRIPT_WITHOUT_GLOBAL_ADD = "// no installs here\n"


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


class GlobalBunRequirementsTest(unittest.TestCase):
    """global_bun_requirements() reads the pins a `bun add -g` line carries."""

    def _requirements(self, files: dict[str, str]) -> set[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return notices.global_bun_requirements(root)

    def test_reads_both_packages_off_the_line(self) -> None:
        found = self._requirements({"docs/render-diagrams.mjs": SCRIPT_WITH_GLOBAL_ADD})
        self.assertEqual(found, {"beautiful-mermaid==1.1.3", "@resvg/resvg-js==2.6.2"})

    def test_a_script_without_an_install_line_yields_nothing(self) -> None:
        found = self._requirements({"docs/render-diagrams.mjs": SCRIPT_WITHOUT_GLOBAL_ADD})
        self.assertEqual(found, set())

    def test_a_scoped_name_keeps_its_scope(self) -> None:
        # Folding `@scope/name@1.2.3` to PEP 508 has to split at the version's
        # `@`, not the scope's: `resvg-js==1.2.3` would name a package that
        # does not exist, and the notices row for it would never compare.
        scoped = " *   bun add -g @resvg/resvg-js@2.6.2\n"
        found = self._requirements({"docs/render-diagrams.mjs": scoped})
        self.assertEqual(found, {"@resvg/resvg-js==2.6.2"})

    def test_the_line_stops_at_the_comment_terminator(self) -> None:
        # The install command sits in a block comment; the prose after it is
        # not part of the command, so it must not be read as a requirement.
        prose = SCRIPT_WITH_GLOBAL_ADD + " * \n * licenses: MIT and MPL-2.0.\n */\n"
        found = self._requirements({"docs/render-diagrams.mjs": prose})
        self.assertEqual(found, {"beautiful-mermaid==1.1.3", "@resvg/resvg-js==2.6.2"})


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

    def test_a_scoped_notices_row_is_read(self) -> None:
        # Same name grammar as the bundle table: a scoped package whose row the
        # pinned list could not read would be silently unaccounted for, since
        # the check compares the row against a requirement nothing produces.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "THIRD_PARTY_NOTICES.md"
            path.write_text("| `@resvg/resvg-js==2.6.2` | MPL-2.0 | x | y |\n", encoding="utf-8")
            self.assertEqual(notices.listed_pep508(path), {"@resvg/resvg-js==2.6.2"})


class RealTreeTest(unittest.TestCase):
    """The tree as committed: what the headers require is what the file records."""

    def test_repo_headers_match_the_notices_entries(self) -> None:
        root = notices.find_root()
        required = notices.pep723_requirements(root) | notices.global_bun_requirements(root)
        listed = notices.listed_pep508(root / "THIRD_PARTY_NOTICES.md")
        self.assertEqual(required - listed, set(), "inline requirement missing from THIRD_PARTY_NOTICES.md")
        self.assertEqual(
            listed - required,
            set(),
            "THIRD_PARTY_NOTICES.md lists an inline requirement nothing declares",
        )

    def test_the_inline_inventory_is_the_set_expected(self) -> None:
        # Pins the inventory to what the tree is expected to hold, so a new
        # inline dependency is a manifest diff here and not a silent arrival.
        # Both header forms are in here: the Python one carries fonttools, and
        # the diagram renderer installs its two packages with `bun add -g`.
        root = notices.find_root()
        self.assertEqual(notices.pep723_requirements(root), {"fonttools==4.66.0"})
        self.assertEqual(
            notices.global_bun_requirements(root),
            {"beautiful-mermaid==1.1.3", "@resvg/resvg-js==2.6.2"},
        )


if __name__ == "__main__":
    unittest.main()
