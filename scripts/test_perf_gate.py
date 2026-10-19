#!/usr/bin/env python3
"""Unit tests for the throughput and determinism gate in perf-gate.py.

Covers the parts that decide pass, fail and skip. The subprocess runs
themselves need a model and a built binary, so they are stubbed: what matters
here is that a missing model reports as skipped rather than as a pass, and
that a regression past the tolerance fails.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

GATE = Path(__file__).resolve().parent / "perf-gate.py"
_spec = importlib.util.spec_from_file_location("perf_gate", GATE)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot import {GATE}")
perf_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf_gate)


class LoadBaselines(unittest.TestCase):
    def test_absent_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(perf_gate.load_baselines(Path(tmp) / "nope.json"), [])

    def test_malformed_json_is_a_gate_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            path.write_text("{not json")
            with self.assertRaises(perf_gate.GateError):
                perf_gate.load_baselines(path)

    def test_entries_must_be_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            path.write_text(json.dumps({"entries": {}}))
            with self.assertRaises(perf_gate.GateError):
                perf_gate.load_baselines(path)

    def test_missing_required_field_names_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            path.write_text(json.dumps({"entries": [{"name": "x", "model": "m.gguf"}]}))
            with self.assertRaises(perf_gate.GateError) as caught:
                perf_gate.load_baselines(path)
            self.assertIn("backend", str(caught.exception))

    def test_committed_baselines_parse(self) -> None:
        root = Path(__file__).resolve().parent.parent
        perf_gate.load_baselines(root / perf_gate.BASELINES)


class CheckEntry(unittest.TestCase):
    ENTRY: ClassVar[dict[str, Any]] = {
        "name": "e",
        "model": "m.gguf",
        "backend": "cpu",
        "decode_tps": 100.0,
        "determinism": False,
    }

    def test_within_tolerance_passes(self) -> None:
        with patch.object(perf_gate, "run_benchmark", return_value={"decode_tps": 96.0}):
            _, tps, failures = perf_gate.check_entry(Path("agave"), self.ENTRY, Path("."), 5.0)
        self.assertEqual(tps, 96.0)
        self.assertEqual(failures, [])

    def test_regression_past_tolerance_fails(self) -> None:
        with patch.object(perf_gate, "run_benchmark", return_value={"decode_tps": 90.0}):
            _, _, failures = perf_gate.check_entry(Path("agave"), self.ENTRY, Path("."), 5.0)
        self.assertEqual(len(failures), 1)
        self.assertIn("10.0% under", failures[0])

    def test_entry_without_a_baseline_only_reports(self) -> None:
        entry = dict(self.ENTRY)
        del entry["decode_tps"]
        with patch.object(perf_gate, "run_benchmark", return_value={"decode_tps": 1.0}):
            _, tps, failures = perf_gate.check_entry(Path("agave"), entry, Path("."), 5.0)
        self.assertEqual(tps, 1.0)
        self.assertEqual(failures, [])

    def test_nondeterministic_generation_fails(self) -> None:
        entry = dict(self.ENTRY, determinism=True)
        with (
            patch.object(perf_gate, "run_benchmark", return_value={"decode_tps": 100.0}),
            patch.object(perf_gate, "run_greedy", side_effect=["one", "two"]),
        ):
            _, _, failures = perf_gate.check_entry(Path("agave"), entry, Path("."), 5.0)
        self.assertEqual(len(failures), 1)
        self.assertIn("differs between two runs", failures[0])

    def test_identical_generation_passes(self) -> None:
        entry = dict(self.ENTRY, determinism=True)
        with (
            patch.object(perf_gate, "run_benchmark", return_value={"decode_tps": 100.0}),
            patch.object(perf_gate, "run_greedy", side_effect=["same", "same"]),
        ):
            _, _, failures = perf_gate.check_entry(Path("agave"), entry, Path("."), 5.0)
        self.assertEqual(failures, [])


class MainSkips(unittest.TestCase):
    def test_missing_model_is_skipped_not_passed(self) -> None:
        # A gate that counts an unrunnable entry as a pass reads as a healthy
        # gate while verifying nothing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build.zig.zon").write_text("")
            (root / "benchmarks").mkdir()
            (root / "benchmarks" / "baselines.json").write_text(
                json.dumps({"entries": [{"name": "e", "model": "absent.gguf", "backend": "cpu"}]})
            )
            binary = root / perf_gate.BINARY
            binary.parent.mkdir(parents=True)
            binary.write_text("")
            with (
                patch.object(perf_gate, "find_root", return_value=root),
                patch.object(perf_gate, "run_benchmark", side_effect=AssertionError("must not run")),
                patch("sys.argv", ["perf-gate.py"]),
            ):
                self.assertEqual(perf_gate.main(), 0)


if __name__ == "__main__":
    unittest.main()
