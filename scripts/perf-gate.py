# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Hold decode throughput and greedy determinism against committed baselines.

`docs/CONTRIBUTING.md` asks every change under `src/backend/`, `src/models/`
or `src/kvcache/` to ship before/after numbers, and treats a regression past
5% as needing a written justification. Nothing enforced either, so both
depended on whoever wrote the change remembering.

This runs the baselines in `benchmarks/baselines.json` that this machine can
actually run and fails on:

  * decode throughput more than `--tolerance` percent under its baseline, and
  * greedy generation that is not byte-identical across two runs at the same
    seed and virtual clock.

Entries whose model file is missing are reported as skipped, never as passes:
a gate that silently measures nothing reads like a healthy gate. Exit code 2
means the gate could not run at all (no binary, malformed baselines), 1 means
a real regression, 0 means every runnable entry held.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# A release build is the only thing worth measuring: `agave-debug` is
# ReleaseSafe and its profile does not predict the shipped binary's.
BINARY = "zig-out/bin/agave"
BASELINES = "benchmarks/baselines.json"
# Matches the "> 5% regression needs a written justification" rule in
# CLAUDE.md, so the gate and the policy cannot drift apart.
DEFAULT_TOLERANCE_PCT = 5.0
# Determinism runs are short: the check is whether two runs agree, not how
# fast they are.
DETERMINISM_TOKENS = 32
DETERMINISM_SEED = 42
# Pinning the clock keeps timing out of the sampled path, so a difference
# between the two runs is the model and not the scheduler.
DETERMINISM_CLOCK_MS = 0
RUN_TIMEOUT_S = 900


def find_root() -> Path:
    """Walk up from this file to the directory holding build.zig.zon."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "build.zig.zon").is_file():
            return candidate
    sys.exit("perf-gate: no build.zig.zon found above this script")


class GateError(Exception):
    """The gate could not run, as opposed to running and finding a regression."""


def load_baselines(path: Path) -> list[dict[str, Any]]:
    """Read the committed baseline entries, or an empty list if absent."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise GateError(f"{path}: malformed JSON: {err}") from err
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise GateError(f"{path}: expected an 'entries' array")
    for entry in entries:
        missing = {"name", "model", "backend"} - set(entry)
        if missing:
            raise GateError(f"{path}: entry {entry!r} is missing {sorted(missing)}")
    return entries


def run_benchmark(binary: Path, entry: dict[str, Any], root: Path) -> dict[str, float]:
    """Run one `--benchmark --json` pass and return its parsed numbers."""
    model = Path(entry["model"])
    if not model.is_absolute():
        model = root / model
    cmd = [
        str(binary),
        str(model),
        "--benchmark",
        "--json",
        "--backend",
        str(entry["backend"]),
    ]
    cmd += [str(a) for a in entry.get("args", [])]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        raise GateError(f"{entry['name']}: benchmark exited {proc.returncode}\n{proc.stderr.strip()}")
    for raw in reversed(proc.stdout.splitlines()):
        line = raw.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise GateError(f"{entry['name']}: no JSON object in benchmark output")


def run_greedy(binary: Path, entry: dict[str, Any], root: Path) -> str:
    """Generate once at a fixed seed and virtual clock, returning the text."""
    model = Path(entry["model"])
    if not model.is_absolute():
        model = root / model
    cmd = [
        str(binary),
        str(model),
        entry.get("prompt", "Count to five."),
        "--quiet",
        "--backend",
        str(entry["backend"]),
        "--seed",
        str(DETERMINISM_SEED),
        "--sim-clock-ms",
        str(DETERMINISM_CLOCK_MS),
        "-t",
        "0",
        "-n",
        str(DETERMINISM_TOKENS),
    ]
    cmd += [str(a) for a in entry.get("args", [])]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        raise GateError(f"{entry['name']}: generation exited {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout


def check_entry(binary: Path, entry: dict[str, Any], root: Path, tolerance: float) -> tuple[str, float, list[str]]:
    """Measure one entry. Returns its name, decode tok/s and any failures."""
    failures: list[str] = []
    measured = run_benchmark(binary, entry, root)
    decode_tps = float(measured.get("decode_tps", 0.0))

    baseline = entry.get("decode_tps")
    if baseline is not None:
        floor = float(baseline) * (1.0 - tolerance / 100.0)
        if decode_tps < floor:
            drop = (1.0 - decode_tps / float(baseline)) * 100.0
            failures.append(
                f"{entry['name']}: decode {decode_tps:.1f} tok/s is {drop:.1f}% under "
                f"the {float(baseline):.1f} tok/s baseline (tolerance {tolerance:.0f}%)"
            )

    if entry.get("determinism", True):
        first = run_greedy(binary, entry, root)
        second = run_greedy(binary, entry, root)
        if first != second:
            failures.append(f"{entry['name']}: greedy generation differs between two runs at seed {DETERMINISM_SEED}")
    return entry["name"], decode_tps, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="rewrite benchmarks/baselines.json with the measured numbers",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PCT,
        help=f"percent below baseline that still passes (default {DEFAULT_TOLERANCE_PCT})",
    )
    args = parser.parse_args()

    root = find_root()
    binary = root / BINARY
    baselines_path = root / BASELINES

    try:
        entries = load_baselines(baselines_path)
    except GateError as err:
        print(f"perf-gate: {err}", file=sys.stderr)
        return 2

    if not entries:
        print(f"perf-gate: no entries in {BASELINES}, nothing to gate")
        return 0
    if not binary.is_file():
        print(f"perf-gate: {BINARY} not built; run `zig build` first", file=sys.stderr)
        return 2

    failures: list[str] = []
    skipped: list[str] = []
    measured: dict[str, float] = {}

    for entry in entries:
        model = Path(entry["model"])
        if not model.is_absolute():
            model = root / model
        if not model.exists():
            skipped.append(f"{entry['name']} (no {entry['model']})")
            continue
        try:
            name, decode_tps, entry_failures = check_entry(binary, entry, root, args.tolerance)
        except (GateError, subprocess.TimeoutExpired, json.JSONDecodeError) as err:
            failures.append(f"{entry['name']}: {err}")
            continue
        measured[name] = decode_tps
        failures += entry_failures
        baseline = entry.get("decode_tps")
        shown = f" (baseline {float(baseline):.1f})" if baseline is not None else " (no baseline)"
        print(f"perf-gate: {name}: {decode_tps:.1f} tok/s{shown}")

    if args.update_baselines:
        for entry in entries:
            if entry["name"] in measured:
                entry["decode_tps"] = round(measured[entry["name"]], 1)
        baselines_path.write_text(json.dumps({"entries": entries}, indent=2) + "\n")
        print(f"perf-gate: wrote {len(measured)} measured baselines to {BASELINES}")
        return 0

    for note in skipped:
        print(f"perf-gate: skipped {note}")
    if skipped and not measured:
        # Every entry skipped. Saying "passed" here would be a faked zero.
        print("perf-gate: no entry could run on this machine; nothing was verified")
        return 0
    for failure in failures:
        print(f"perf-gate: FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"perf-gate: {len(measured)} entries held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
