#!/usr/bin/env python3
"""Unit tests for the release SemVer guard in check-docs.py.

Builds a throwaway release tree, points the checker at it, and asserts a stale
product version in any guarded file is reported.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CHECKER = Path(__file__).resolve().parent / "check-docs.py"
_spec = importlib.util.spec_from_file_location("check_docs", CHECKER)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot import {CHECKER}")
check_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docs)

PRODUCT = "9.9.9"

# path -> text, with {v} standing in for the product version
FILES: dict[str, str] = {
    ".zigversion": "0.16.0\n",
    "build.zig.zon": (
        '.{\n    .name = .agave,\n    .version = "' + PRODUCT + '",\n'
        '    .minimum_zig_version = "0.16.0",\n}\n'
    ),
    "CHANGELOG.md": (
        f"Product version is **{PRODUCT}**\n\n## [Unreleased]\n\n"
        f"## [{PRODUCT}] - 2026-01-01\n\n"
        f"[unreleased]: https://example.invalid/compare/v{PRODUCT}...HEAD\n"
        f"[{PRODUCT}]: https://example.invalid/compare/v0.0.1...v{PRODUCT}\n"
    ),
    "docs/API.md": f'Product version **{PRODUCT}**\n\n"system_fingerprint": "agave-v{PRODUCT}"\n',
    "docs/CONTRIBUTING.md": f"Product version: **{PRODUCT}**\n",
    "README.md": f"product version `{PRODUCT}`\n",
    "SECURITY.md": f"Product version is **{PRODUCT}**\n",
    "docs/DOCUMENTATION.md": f"product version {PRODUCT}\n",
}


class VersionConsistencyTest(unittest.TestCase):
    def _errors(self, overrides: dict[str, str] | None = None) -> list[str]:
        files = dict(FILES)
        if overrides:
            files.update(overrides)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            with patch.object(check_docs, "ROOT", root):
                return check_docs.check_version_consistency()

    def test_aligned_tree_reports_nothing(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_stale_readme_is_reported(self) -> None:
        errors = self._errors({"README.md": "product version `0.1.0`\n"})
        self.assertEqual(len(errors), 1)
        self.assertIn("README.md", errors[0])

    def test_stale_security_is_reported(self) -> None:
        errors = self._errors({"SECURITY.md": "Product version is **0.1.0**\n"})
        self.assertEqual(len(errors), 1)
        self.assertIn("SECURITY.md", errors[0])

    def test_stale_documentation_index_is_reported(self) -> None:
        errors = self._errors({"docs/DOCUMENTATION.md": "product version 0.1.0\n"})
        self.assertEqual(len(errors), 1)
        self.assertIn("docs/DOCUMENTATION.md", errors[0])

    def test_missing_guarded_file_is_reported(self) -> None:
        files = dict(FILES)
        del files["SECURITY.md"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, text in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            with patch.object(check_docs, "ROOT", root):
                errors = check_docs.check_version_consistency()
        self.assertTrue(any("SECURITY.md" in e for e in errors), errors)

    def test_zig_pin_mismatch_still_reported(self) -> None:
        errors = self._errors({".zigversion": "0.15.0\n"})
        self.assertTrue(any(".zigversion" in e for e in errors), errors)

    def test_released_section_without_link_definition_is_reported(self) -> None:
        changelog = FILES["CHANGELOG.md"].replace(
            f"\n[{PRODUCT}]: https://example.invalid/compare/v0.0.1...v{PRODUCT}\n", ""
        )
        errors = self._errors({"CHANGELOG.md": changelog})
        self.assertTrue(any("has no" in e and "link definition" in e for e in errors), errors)

    def test_unreleased_compare_against_stale_tag_is_reported(self) -> None:
        changelog = FILES["CHANGELOG.md"].replace(
            f"[unreleased]: https://example.invalid/compare/v{PRODUCT}...HEAD",
            "[unreleased]: https://example.invalid/compare/v0.1.0...HEAD",
        )
        errors = self._errors({"CHANGELOG.md": changelog})
        self.assertTrue(any("[unreleased]" in e for e in errors), errors)

    def test_release_tag_disagreeing_with_manifest_is_reported(self) -> None:
        with patch.object(check_docs, "_tags_at_head", return_value=["v0.4.0"]):
            errors = self._errors()
        self.assertTrue(any("tagged v0.4.0" in e for e in errors), errors)

    def test_milestone_tag_is_not_a_release_tag(self) -> None:
        with patch.object(check_docs, "_tags_at_head", return_value=["v1.0"]):
            errors = self._errors()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())
