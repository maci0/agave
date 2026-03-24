#!/usr/bin/env python3
"""
Agave Inference Engine - Test Harness

Comprehensive test framework for validating all backends, models, and
quantization formats. Supports correctness validation via golden references,
performance benchmarking with repeat runs, regression detection, and
structured output (JSON, Markdown, CSV).

Requirements: pip install rich (or use tests/.venv)

Usage:
    # Run all tests with auto-detected models and backends
    tests/.venv/bin/python tests/harness.py

    # Filter by backend, model arch, or quant
    tests/.venv/bin/python tests/harness.py --backend metal cpu --arch gemma3 qwen35
    tests/.venv/bin/python tests/harness.py --quant Q4_0 Q8_0

    # Run only benchmarks (skip correctness)
    tests/.venv/bin/python tests/harness.py --bench-only

    # Run only correctness tests (skip benchmarks)
    tests/.venv/bin/python tests/harness.py --correctness-only

    # Validate model metadata loads correctly (fast, no inference)
    tests/.venv/bin/python tests/harness.py --model-info-only

    # Custom model directory and prompt
    tests/.venv/bin/python tests/harness.py --model-dir ./weights --prompt "Hello, world!"

    # Multiple repeat runs for performance stability
    tests/.venv/bin/python tests/harness.py --repeat 3

    # Output formats
    tests/.venv/bin/python tests/harness.py --output json      # Machine-readable JSON
    tests/.venv/bin/python tests/harness.py --output markdown   # Markdown table
    tests/.venv/bin/python tests/harness.py --output csv        # CSV spreadsheet

    # Golden reference management
    tests/.venv/bin/python tests/harness.py --generate-golden   # Save outputs as golden refs
    tests/.venv/bin/python tests/harness.py --check-golden      # Validate against golden refs

    # Compare against a baseline
    tests/.venv/bin/python tests/harness.py --baseline results/baseline.json

    # Profiling with Instruments (macOS)
    tests/.venv/bin/python tests/harness.py --profile instruments --arch gemma3

    # Save results
    tests/.venv/bin/python tests/harness.py --save results/run-$(date +%Y%m%d).json

    # List what would run without running it
    tests/.venv/bin/python tests/harness.py --dry-run

    # Increase token count for throughput measurement
    tests/.venv/bin/python tests/harness.py --max-tokens 256

    # Set timeout per run (seconds)
    tests/.venv/bin/python tests/harness.py --timeout 120
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGAVE_ROOT = Path(__file__).resolve().parent.parent
AGAVE_BIN = AGAVE_ROOT / "zig-out" / "bin" / "agave"
DEFAULT_WEIGHTS_DIR = AGAVE_ROOT / "models"
GOLDEN_DIR = AGAVE_ROOT / "tests" / "golden_outputs"
DEFAULT_PROMPT = "What is 2+2? Answer briefly."
DEFAULT_MAX_TOKENS = 64
DEFAULT_TIMEOUT = 120  # seconds

# Architecture detection from filename patterns
ARCH_PATTERNS: dict[str, list[str]] = {
    "gemma3": ["gemma-3", "gemma3"],
    "qwen35": ["qwen3.5", "qwen35"],
    "gpt_oss": ["gpt-oss", "gptoss"],
    "nemotron_h": ["nemotron-h", "nemotron_h", "nemotron-3-nano-4b"],
    "nemotron_nano": ["nemotron-nano", "nemotron_nano", "nemotron-3-nano-30b"],
    "glm4": ["glm-4", "glm4"],
}

# Quant detection from filename (order matters - check longer patterns first)
QUANT_PATTERNS = [
    # MLX/QAT SafeTensors formats
    "qat-4bit", "qat-6bit",
    "MLX-4bit", "MLX-5bit", "MLX-6bit",
    # NVFP4/MXFP4
    "NVFP4", "MXFP4",
    # Standard float
    "BF16", "F16", "F32",
    # GGUF integer quants (longer patterns first)
    "IQ4_XS", "IQ4_NL",
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K_L", "Q4_K",
    "Q4_0", "Q4_1", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
    "Q2_K", "Q5_0",
    # FP8
    "FP8",
]

# Files to skip during GGUF discovery (not inference models)
SKIP_GGUF_PREFIXES = ["mmproj-", "mmproj_"]

# Backends to try per platform
PLATFORM_BACKENDS: dict[str, list[str]] = {
    "Darwin": ["metal", "cpu", "vulkan"],
    "Linux": ["cuda", "vulkan", "cpu"],
}

console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    path: Path
    arch: str
    quant: str
    size_mb: float
    format: str  # "gguf" or "safetensors"

    @property
    def name(self) -> str:
        if self.format == "safetensors":
            return self.path.name
        return self.path.stem

    @property
    def short_name(self) -> str:
        return f"{self.arch}/{self.quant}"

    @property
    def golden_key(self) -> str:
        """Unique key for golden reference storage."""
        return f"{self.arch}_{self.quant}_{self.format}"


@dataclass
class ModelInfoResult:
    """Result from --model-info --json validation."""
    model: str
    arch: str
    quant: str
    status: str  # "pass", "fail", "error"
    metadata: dict = field(default_factory=dict)
    error_message: str = ""
    load_ms: float = 0.0


@dataclass
class RunResult:
    model: str
    arch: str
    quant: str
    backend: str
    status: str  # "pass", "fail", "skip", "timeout", "error"
    tokens_generated: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    load_time_ms: float = 0.0
    output_text: str = ""
    error_message: str = ""
    model_size_mb: float = 0.0
    exit_code: int = 0
    command: str = ""
    correctness: str = ""  # "match", "mismatch", "no_golden", ""


@dataclass
class RepeatStats:
    """Aggregate stats from multiple runs."""
    runs: list[RunResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def passing(self) -> list[RunResult]:
        return [r for r in self.runs if r.status == "pass"]

    @property
    def tps_values(self) -> list[float]:
        return [r.tokens_per_sec for r in self.passing if r.tokens_per_sec > 0]

    @property
    def tps_avg(self) -> float:
        vals = self.tps_values
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def tps_min(self) -> float:
        vals = self.tps_values
        return min(vals) if vals else 0.0

    @property
    def tps_max(self) -> float:
        vals = self.tps_values
        return max(vals) if vals else 0.0

    @property
    def tps_stddev(self) -> float:
        vals = self.tps_values
        if len(vals) < 2:
            return 0.0
        avg = self.tps_avg
        return math.sqrt(sum((v - avg) ** 2 for v in vals) / (len(vals) - 1))

    @property
    def best(self) -> Optional[RunResult]:
        """Best (or first) passing run."""
        passing = self.passing
        if not passing:
            return self.runs[0] if self.runs else None
        return max(passing, key=lambda r: r.tokens_per_sec)


@dataclass
class BenchResult:
    backend: str
    status: str
    raw_output: str = ""
    metrics: dict = field(default_factory=dict)
    error_message: str = ""


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def detect_arch(name: str) -> Optional[str]:
    """Detect model architecture from file/directory name."""
    lower = name.lower()
    # Check nemotron variants carefully (nano-30b vs nano-4b)
    if "nemotron" in lower:
        if "30b" in lower or "nano-30b" in lower:
            return "nemotron_nano"
        if "nano" in lower:
            return "nemotron_h"  # Nemotron-H uses the Nano-4B GGUF
        if "nemotron-h" in lower or "nemotron_h" in lower:
            return "nemotron_h"
    for arch, patterns in ARCH_PATTERNS.items():
        if arch.startswith("nemotron"):
            continue  # Already handled above
        for pat in patterns:
            if pat.lower() in lower:
                return arch
    return None


def detect_quant(name: str) -> str:
    """Detect quantization format from filename."""
    # Normalize for matching
    check = name.replace("-", "_").replace(" ", "_")
    for q in QUANT_PATTERNS:
        q_norm = q.replace("-", "_")
        if q_norm.lower() in check.lower():
            return q
    # Check for bf16 in SafeTensors dir names
    if "bf16" in name.lower():
        return "BF16"
    # MLX models with non-standard naming (e.g., "-mlx" suffix without explicit bit width)
    if "-mlx" in name.lower() or "_mlx" in name.lower():
        return "MLX-4bit"
    return "unknown"


def detect_format(path: Path) -> str:
    """Detect weight format (gguf or safetensors)."""
    if path.is_dir():
        if any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").exists():
            return "safetensors"
    if path.suffix == ".gguf":
        return "gguf"
    return "unknown"


def _is_skip_gguf(name: str) -> bool:
    """Return True if this GGUF file should be skipped (e.g. mmproj)."""
    lower = name.lower()
    return any(lower.startswith(p.lower()) for p in SKIP_GGUF_PREFIXES)


def discover_models(search_dirs: list[Path]) -> list[ModelInfo]:
    """Discover models in search directories, recursing into org/repo structure."""
    models = []
    seen = set()

    for d in search_dirs:
        if not d.exists():
            continue

        # Find GGUF files at any depth
        for p in sorted(d.glob("**/*.gguf")):
            if p.resolve() in seen or _is_skip_gguf(p.name):
                continue
            seen.add(p.resolve())
            arch = detect_arch(p.name) or detect_arch(p.parent.name)
            if arch is None:
                continue
            quant = detect_quant(p.name)
            size_mb = p.stat().st_size / (1024 * 1024)
            models.append(ModelInfo(path=p, arch=arch, quant=quant, size_mb=size_mb, format="gguf"))

        # Find SafeTensors directories at any depth (look for config.json as marker)
        for config in sorted(d.glob("**/config.json")):
            p = config.parent
            if p.resolve() in seen:
                continue
            fmt = detect_format(p)
            if fmt != "safetensors":
                continue
            seen.add(p.resolve())
            arch = detect_arch(p.name)
            if arch is None:
                continue
            quant = detect_quant(p.name)
            size_mb = sum(f.stat().st_size for f in p.rglob("*.safetensors")) / (1024 * 1024)
            models.append(ModelInfo(path=p, arch=arch, quant=quant, size_mb=size_mb, format=fmt))

    return sorted(models, key=lambda m: (m.arch, m.quant))


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_available_backends() -> list[str]:
    system = platform.system()
    candidates = PLATFORM_BACKENDS.get(system, ["cpu"])
    available = []

    for be in candidates:
        if be == "cpu":
            available.append(be)
        elif be == "metal" and system == "Darwin":
            available.append(be)
        elif be == "vulkan":
            if shutil.which("vulkaninfo") or system == "Darwin":
                available.append(be)
        elif be == "cuda":
            if shutil.which("nvidia-smi"):
                available.append(be)

    return available if available else ["cpu"]


# ---------------------------------------------------------------------------
# Running agave
# ---------------------------------------------------------------------------

def run_model_info(
    model: ModelInfo,
    backend: str,
    timeout: int,
) -> ModelInfoResult:
    """Run --model-info --json to validate model metadata loads correctly."""
    cmd = [
        str(AGAVE_BIN),
        str(model.path),
        "--backend", backend,
        "--json",
        "--model-info",
    ]

    result = ModelInfoResult(
        model=model.name, arch=model.arch, quant=model.quant, status="error",
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            result.status = "fail"
            result.error_message = _truncate(proc.stdout + proc.stderr, 500)
            return result

        # Parse JSON output
        stdout = proc.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                result.metadata = data
                result.load_ms = data.get("load_ms", 0)
                # Validate essential fields
                required = ["name", "arch", "layers", "embed", "heads", "vocab_size"]
                missing = [f for f in required if f not in data or data[f] in (None, 0, "")]
                if missing:
                    result.status = "fail"
                    result.error_message = f"Missing/zero metadata fields: {', '.join(missing)}"
                else:
                    result.status = "pass"
            except json.JSONDecodeError as e:
                result.status = "fail"
                result.error_message = f"Invalid JSON: {e}"
        else:
            result.status = "fail"
            result.error_message = "Empty output"

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error_message = f"Timed out after {timeout}s"
    except FileNotFoundError:
        result.status = "error"
        result.error_message = f"Binary not found: {AGAVE_BIN}"
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def run_inference(
    model: ModelInfo,
    backend: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    extra_flags: list[str] | None = None,
) -> RunResult:
    """Run inference with --json for structured output parsing."""
    cmd = [
        str(AGAVE_BIN),
        str(model.path),
        "--backend", backend,
        "--max-tokens", str(max_tokens),
        "--json",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.append(prompt)

    result = RunResult(
        model=model.name,
        arch=model.arch,
        quant=model.quant,
        backend=backend,
        status="error",
        model_size_mb=model.size_mb,
        command=" ".join(cmd),
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result.exit_code = proc.returncode

        if proc.returncode != 0:
            combined = proc.stdout + proc.stderr
            result.status = "fail"
            result.error_message = _truncate(combined, 500)
            return result

        # Parse JSON output from --json mode
        stdout = proc.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                result.status = "pass"
                result.output_text = data.get("output", "")
                result.tokens_generated = data.get("tokens", 0)
                result.tokens_per_sec = data.get("tok_per_sec", 0.0)
                result.time_to_first_token_ms = data.get("prefill_ms", 0.0)
                result.total_time_ms = data.get("gen_ms", 0.0)
                result.load_time_ms = data.get("load_ms", 0.0)

                # Basic correctness: output should be non-empty
                if not result.output_text.strip():
                    result.status = "fail"
                    result.error_message = "Empty output text"

            except json.JSONDecodeError:
                # Fallback: try parsing human-readable output
                result.status = "pass"
                _parse_stats_fallback(proc.stdout, result)
        else:
            result.status = "fail"
            result.error_message = "Empty stdout"

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error_message = f"Timed out after {timeout}s"
    except FileNotFoundError:
        result.status = "error"
        result.error_message = f"Binary not found: {AGAVE_BIN}"
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def _parse_stats_fallback(output: str, result: RunResult) -> None:
    """Fallback: parse performance statistics from human-readable output."""
    # Stats line: "8 tok · 26.8 tok/s · prefill 9 tok in 391ms (23 tok/s)"
    m = re.search(r"(\d+)\s+tok\s+·\s+([\d.]+)\s+tok/s", output)
    if m:
        result.tokens_generated = int(m.group(1))
        result.tokens_per_sec = float(m.group(2))

    m = re.search(r"prefill\s+\d+\s+tok\s+in\s+(\d+)ms", output)
    if m:
        result.time_to_first_token_ms = float(m.group(1))

    m = re.search(r"agave\s+.+\((\d+)ms\)", output)
    if m:
        result.load_time_ms = float(m.group(1))

    # Extract text between "ready" and stats line
    lines = output.strip().splitlines()
    text_lines = []
    in_text = False
    for line in lines:
        if line.startswith("ready "):
            in_text = True
            continue
        if in_text and re.match(r"\d+ tok · ", line):
            break
        if in_text:
            text_lines.append(line)
    result.output_text = "\n".join(text_lines).strip()

    if result.tokens_per_sec > 0 and result.tokens_generated > 0:
        decode_ms = (result.tokens_generated / result.tokens_per_sec) * 1000
        result.total_time_ms = result.time_to_first_token_ms + decode_ms


def run_bench(backend: str, timeout: int) -> BenchResult:
    cmd = [str(AGAVE_BIN), "--bench", "--backend", backend]
    result = BenchResult(backend=backend, status="error")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result.raw_output = proc.stdout + proc.stderr
        result.status = "pass" if proc.returncode == 0 else "fail"
        if proc.returncode != 0:
            result.error_message = _truncate(result.raw_output, 500)
        else:
            result.metrics = _parse_bench_output(result.raw_output)
    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error_message = f"Timed out after {timeout}s"
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def _parse_bench_output(output: str) -> dict:
    metrics = {}
    current_section = "general"

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"---\s*(.+?)\s*---", line)
        if m:
            current_section = m.group(1).lower().replace(" ", "_")
            continue

        m = re.match(r"(.+?):\s+([\d.]+)\s*(ms|μs|us|tok/s|GB/s|MB)", line)
        if m:
            key = f"{current_section}/{m.group(1).strip()}"
            metrics[key] = {"value": float(m.group(2)), "unit": m.group(3)}
            bw = re.search(r"\(([\d.]+)\s*GB/s\)", line)
            if bw:
                metrics[f"{key}/bandwidth"] = {"value": float(bw.group(1)), "unit": "GB/s"}

    return metrics


@dataclass
class SmokeResult:
    """Result from a quick 1-token smoke test."""
    model: str
    arch: str
    quant: str
    backend: str
    status: str  # "pass", "fail", "crash", "timeout", "error"
    tokens_generated: int = 0
    error_message: str = ""
    exit_code: int = 0
    signal: int = 0  # non-zero if killed by signal (e.g. SIGSEGV=11)


def run_smoke(
    model: ModelInfo,
    backend: str,
    timeout: int,
) -> SmokeResult:
    """Run a quick 1-token inference test to catch crashes and 0-token failures.

    This is faster than full inference but catches:
    - Segfaults (signal 11)
    - Panics (missing kernels, unsupported dtypes)
    - 0-token output (broken dequant, missing tensor mappings)
    - Immediate EOG (wrong logits)
    """
    cmd = [
        str(AGAVE_BIN),
        str(model.path),
        "--backend", backend,
        "--max-tokens", "4",
        "--json",
        "-q",
        "Hello",
    ]

    result = SmokeResult(
        model=model.name, arch=model.arch, quant=model.quant, backend=backend,
        status="error",
    )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result.exit_code = proc.returncode

        if proc.returncode < 0:
            # Killed by signal (e.g. -11 = SIGSEGV)
            result.signal = -proc.returncode
            result.status = "crash"
            sig_name = {11: "SIGSEGV", 6: "SIGABRT", 134: "SIGABRT"}.get(result.signal, f"signal {result.signal}")
            result.error_message = f"Crashed with {sig_name}"
            return result

        if proc.returncode != 0:
            combined = proc.stdout + proc.stderr
            # Check for panic
            if "panic:" in combined:
                panic_line = next((l for l in combined.splitlines() if "panic:" in l), "")
                result.status = "crash"
                result.error_message = f"Panic: {_truncate(panic_line, 120)}"
            else:
                result.status = "fail"
                result.error_message = _truncate(combined, 200)
            return result

        # Parse JSON output
        stdout = proc.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                result.tokens_generated = data.get("tokens", 0)
                output_text = data.get("output", "")
                if result.tokens_generated == 0 and not output_text.strip():
                    result.status = "fail"
                    result.error_message = "0 tokens generated (immediate EOG)"
                else:
                    result.status = "pass"
            except json.JSONDecodeError:
                # Non-JSON output — check if any text was produced
                if proc.stdout.strip():
                    result.status = "pass"
                    result.tokens_generated = 1  # approximate
                else:
                    result.status = "fail"
                    result.error_message = "Empty output"
        else:
            result.status = "fail"
            result.error_message = "Empty stdout"

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error_message = f"Timed out after {timeout}s"
    except FileNotFoundError:
        result.status = "error"
        result.error_message = f"Binary not found: {AGAVE_BIN}"
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def run_zig_tests(timeout: int) -> tuple[str, int]:
    cmd = ["zig", "build", "test"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(AGAVE_ROOT))
        return proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s", 1
    except Exception as e:
        return str(e), 1


# ---------------------------------------------------------------------------
# Golden reference system
# ---------------------------------------------------------------------------

def golden_path(model: ModelInfo, backend: str) -> Path:
    """Path for a golden reference file."""
    return GOLDEN_DIR / f"{model.golden_key}_{backend}.json"


def save_golden(result: RunResult, model: ModelInfo) -> None:
    """Save a passing result as a golden reference."""
    if result.status != "pass" or not result.output_text.strip():
        return
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    gp = golden_path(model, result.backend)
    data = {
        "model": result.model,
        "arch": result.arch,
        "quant": result.quant,
        "backend": result.backend,
        "output_text": result.output_text,
        "tokens_generated": result.tokens_generated,
        "tokens_per_sec": result.tokens_per_sec,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(gp, "w") as f:
        json.dump(data, f, indent=2)


def check_golden(result: RunResult, model: ModelInfo) -> str:
    """Check a result against golden reference. Returns 'match', 'mismatch', or 'no_golden'."""
    gp = golden_path(model, result.backend)
    if not gp.exists():
        return "no_golden"
    try:
        with open(gp) as f:
            golden = json.load(f)
        golden_text = golden.get("output_text", "")
        # Fuzzy match: first N chars should match (models may vary slightly due to sampling)
        # Use first 20 chars as a fingerprint
        n = min(20, len(golden_text), len(result.output_text))
        if n == 0:
            return "mismatch"
        if golden_text[:n] == result.output_text[:n]:
            return "match"
        return "mismatch"
    except Exception:
        return "no_golden"


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_with_instruments(
    model: ModelInfo, backend: str, prompt: str,
    max_tokens: int, timeout: int,
) -> Path | None:
    if platform.system() != "Darwin" or not shutil.which("xctrace"):
        console.print("  [yellow]Instruments profiling only available on macOS with Xcode[/]")
        return None

    trace_dir = AGAVE_ROOT / "tests" / "traces"
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"{model.arch}_{model.quant}_{backend}_{int(time.time())}.trace"

    cmd = [
        "xctrace", "record",
        "--template", "Time Profiler",
        "--output", str(trace_path),
        "--launch", "--",
        str(AGAVE_BIN), str(model.path),
        "--backend", backend,
        "--max-tokens", str(max_tokens),
        "-q", prompt,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        if proc.returncode == 0 and trace_path.exists():
            return trace_path
        else:
            console.print(f"  [red]Instruments failed: {_truncate(proc.stderr, 200)}[/]")
    except Exception as e:
        console.print(f"  [red]Instruments error: {e}[/]")

    return None


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    results: list[RunResult], baseline_path: Path,
    regression_threshold: float = 0.05,
) -> list[dict]:
    if not baseline_path.exists():
        return []

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_map = {}
    for entry in baseline_data.get("inference_results", []):
        key = (entry["arch"], entry["quant"], entry["backend"])
        baseline_map[key] = entry

    regressions = []
    for r in results:
        if r.status != "pass":
            continue
        key = (r.arch, r.quant, r.backend)
        base = baseline_map.get(key)
        if not base or base.get("tokens_per_sec", 0) == 0:
            continue

        base_tps = base["tokens_per_sec"]
        delta = (r.tokens_per_sec - base_tps) / base_tps

        if delta < -regression_threshold:
            regressions.append({
                "arch": r.arch, "quant": r.quant, "backend": r.backend,
                "baseline_tps": base_tps, "current_tps": r.tokens_per_sec,
                "delta_pct": delta * 100,
            })

    return regressions


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def status_markup(status: str) -> str:
    return {
        "pass": "[bold green]PASS[/]",
        "fail": "[bold red]FAIL[/]",
        "skip": "[yellow]SKIP[/]",
        "timeout": "[yellow]TIMEOUT[/]",
        "error": "[bold red]ERROR[/]",
    }.get(status, status)


def correctness_markup(c: str) -> str:
    return {
        "match": "[green]match[/]",
        "mismatch": "[bold red]MISMATCH[/]",
        "no_golden": "[dim]--[/]",
        "": "[dim]-[/]",
    }.get(c, c)


def fmt_ms(v: float) -> str:
    if v == 0:
        return "[dim]-[/]"
    if v < 1:
        return f"{v:.2f} ms"
    return f"{v:.0f} ms"


def fmt_tps(v: float) -> str:
    if v == 0:
        return "[dim]-[/]"
    if v >= 100:
        return f"[bold green]{v:.0f}[/]"
    if v >= 30:
        return f"[green]{v:.1f}[/]"
    if v >= 10:
        return f"[yellow]{v:.1f}[/]"
    return f"[red]{v:.1f}[/]"


def print_model_info_table(results: list[ModelInfoResult]) -> None:
    if not results:
        return

    table = Table(
        title="Model Info Validation",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold",
    )
    table.add_column("Status", justify="center", width=8)
    table.add_column("Architecture", width=14)
    table.add_column("Quant", width=12)
    table.add_column("Load", justify="right", width=8)
    table.add_column("Layers", justify="right", width=7)
    table.add_column("Embed", justify="right", width=7)
    table.add_column("Heads", justify="right", width=7)
    table.add_column("Vocab", justify="right", width=8)
    table.add_column("Params", justify="right", width=10)

    for r in results:
        if r.status == "pass":
            md = r.metadata
            n_params = md.get("n_params", 0)
            params_str = _format_params(n_params) if n_params else "[dim]-[/]"
            table.add_row(
                status_markup(r.status),
                md.get("arch", r.arch), r.quant,
                fmt_ms(r.load_ms),
                str(md.get("layers", "")),
                str(md.get("embed", "")),
                str(md.get("heads", "")),
                str(md.get("vocab_size", "")),
                params_str,
            )
        else:
            err = _truncate(r.error_message, 40) if r.error_message else ""
            table.add_row(
                status_markup(r.status),
                r.arch, r.quant,
                "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]",
                "[dim]-[/]",
                f"[dim]{err}[/]" if err else "[dim]-[/]",
            )

    console.print()
    console.print(table)


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)


def print_inference_table(results: list[RunResult], show_golden: bool = False) -> None:
    if not results:
        return

    table = Table(
        title="Inference Results",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
        title_style="bold",
    )
    table.add_column("Status", justify="center", width=8)
    table.add_column("Architecture", width=14)
    table.add_column("Quant", width=12)
    table.add_column("Backend", width=8)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Load", justify="right", width=8)
    table.add_column("TTFT", justify="right", width=9)
    table.add_column("tok/s", justify="right", width=8)
    table.add_column("Tokens", justify="right", width=6)
    table.add_column("Total", justify="right", width=9)
    if show_golden:
        table.add_column("Golden", justify="center", width=10)

    for r in results:
        row = []
        if r.status == "pass":
            row = [
                status_markup(r.status),
                r.arch, r.quant, r.backend,
                f"{r.model_size_mb:.0f} MB",
                fmt_ms(r.load_time_ms),
                fmt_ms(r.time_to_first_token_ms),
                fmt_tps(r.tokens_per_sec),
                str(r.tokens_generated),
                fmt_ms(r.total_time_ms),
            ]
        else:
            err = _truncate(r.error_message, 40) if r.error_message else ""
            row = [
                status_markup(r.status),
                r.arch, r.quant, r.backend,
                f"{r.model_size_mb:.0f} MB",
                "[dim]-[/]", "[dim]-[/]", "[dim]-[/]", "[dim]-[/]",
                f"[dim]{err}[/]" if err else "[dim]-[/]",
            ]
        if show_golden:
            row.append(correctness_markup(r.correctness))
        table.add_row(*row)

    console.print()
    console.print(table)


def print_repeat_table(repeat_results: dict[tuple, RepeatStats]) -> None:
    """Print aggregate performance table for repeat runs."""
    if not repeat_results:
        return

    table = Table(
        title="Performance Summary (Repeat Runs)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold",
    )
    table.add_column("Architecture", width=14)
    table.add_column("Quant", width=12)
    table.add_column("Backend", width=8)
    table.add_column("Runs", justify="right", width=5)
    table.add_column("Pass", justify="right", width=5)
    table.add_column("Avg tok/s", justify="right", width=10)
    table.add_column("Min", justify="right", width=8)
    table.add_column("Max", justify="right", width=8)
    table.add_column("Stddev", justify="right", width=8)
    table.add_column("CV%", justify="right", width=6)

    for (arch, quant, backend), stats in sorted(repeat_results.items()):
        n_pass = len(stats.passing)
        avg = stats.tps_avg
        cv = (stats.tps_stddev / avg * 100) if avg > 0 else 0
        table.add_row(
            arch, quant, backend,
            str(stats.n), str(n_pass),
            fmt_tps(avg),
            fmt_tps(stats.tps_min),
            fmt_tps(stats.tps_max),
            f"{stats.tps_stddev:.1f}" if stats.tps_stddev > 0 else "[dim]-[/]",
            f"{cv:.1f}" if cv > 0 else "[dim]-[/]",
        )

    console.print()
    console.print(table)


def print_bench_table(results: list[BenchResult]) -> None:
    if not results:
        return

    for br in results:
        console.print()
        if br.status != "pass":
            console.print(f"[bold]Synthetic Benchmarks: {br.backend}[/]  {status_markup(br.status)}")
            if br.error_message:
                console.print(f"  [red]{_truncate(br.error_message, 100)}[/]")
            continue

        table = Table(
            title=f"Synthetic Benchmarks: {br.backend}",
            box=box.SIMPLE_HEAVY,
            header_style="bold",
            title_style="bold",
        )
        table.add_column("Kernel", width=35)
        table.add_column("Value", justify="right", width=10)
        table.add_column("Unit", width=8)
        table.add_column("Bandwidth", justify="right", width=12)

        if br.metrics:
            for key, m in sorted(br.metrics.items()):
                if "/bandwidth" in key:
                    continue
                bw_key = f"{key}/bandwidth"
                bw_str = f"{br.metrics[bw_key]['value']:.1f} GB/s" if bw_key in br.metrics else ""
                table.add_row(key, f"{m['value']:.2f}", m["unit"], bw_str)
            console.print(table)
        else:
            for line in br.raw_output.splitlines():
                line = line.strip()
                if line:
                    console.print(f"  {line}")


def print_summary(
    results: list[RunResult],
    bench_results: list[BenchResult],
    model_info_results: list[ModelInfoResult],
    regressions: list[dict],
    golden_stats: dict,
    elapsed: float,
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errors = sum(1 for r in results if r.status in ("error", "timeout"))
    skipped = sum(1 for r in results if r.status == "skip")
    bench_total = len(bench_results)
    bench_passed = sum(1 for b in bench_results if b.status == "pass")

    summary_lines = []

    # Model info results
    if model_info_results:
        mi_passed = sum(1 for r in model_info_results if r.status == "pass")
        mi_total = len(model_info_results)
        summary_lines.append(f"Model info:   [bold green]{mi_passed}[/]/{mi_total} validated")

    # Inference results
    if total > 0:
        parts = [f"[bold green]{passed}[/]/{total} passed"]
        if failed:
            parts.append(f"[bold red]{failed} failed[/]")
        if errors:
            parts.append(f"[bold red]{errors} errors[/]")
        if skipped:
            parts.append(f"[yellow]{skipped} skipped[/]")
        summary_lines.append(f"Inference:    {', '.join(parts)}")

    # Golden stats
    if golden_stats:
        gs = golden_stats
        if gs.get("checked", 0) > 0:
            match_count = gs.get("match", 0)
            mismatch_count = gs.get("mismatch", 0)
            no_golden = gs.get("no_golden", 0)
            parts = [f"[green]{match_count}[/] match"]
            if mismatch_count:
                parts.append(f"[bold red]{mismatch_count} mismatch[/]")
            if no_golden:
                parts.append(f"[dim]{no_golden} no ref[/]")
            summary_lines.append(f"Golden:       {', '.join(parts)}")
        if gs.get("saved", 0) > 0:
            summary_lines.append(f"Golden saved: [bold]{gs['saved']}[/] references")

    if bench_total:
        summary_lines.append(f"Benchmarks:   {bench_passed}/{bench_total} passed")

    summary_lines.append(f"Elapsed:      {elapsed:.1f}s")

    # Top performers
    passing = [r for r in results if r.status == "pass" and r.tokens_per_sec > 0]
    if passing:
        best = max(passing, key=lambda r: r.tokens_per_sec)
        summary_lines.append(
            f"Fastest:      [bold cyan]{best.arch}/{best.quant}[/] on [cyan]{best.backend}[/] "
            f"@ [bold]{best.tokens_per_sec:.1f}[/] tok/s"
        )
        fastest_ttft = min(passing, key=lambda r: r.time_to_first_token_ms if r.time_to_first_token_ms > 0 else float("inf"))
        if fastest_ttft.time_to_first_token_ms > 0:
            summary_lines.append(
                f"Best TTFT:    [bold cyan]{fastest_ttft.arch}/{fastest_ttft.quant}[/] on "
                f"[cyan]{fastest_ttft.backend}[/] @ [bold]{fastest_ttft.time_to_first_token_ms:.0f}[/] ms"
            )

    console.print()
    console.print(Panel(
        "\n".join(summary_lines),
        title="Summary",
        border_style="bold",
        padding=(0, 2),
    ))

    if regressions:
        reg_lines = []
        for reg in regressions:
            reg_lines.append(
                f"[red]{reg['arch']}/{reg['quant']}[/] on {reg['backend']}: "
                f"{reg['baseline_tps']:.1f} -> {reg['current_tps']:.1f} tok/s "
                f"([bold red]{reg['delta_pct']:+.1f}%[/])"
            )
        console.print(Panel(
            "\n".join(reg_lines),
            title=f"Regressions Detected ({len(regressions)})",
            border_style="bold red",
            padding=(0, 2),
        ))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def results_to_json(
    results: list[RunResult],
    bench_results: list[BenchResult],
    model_info_results: list[ModelInfoResult],
    repeat_results: dict[tuple, RepeatStats],
    meta: dict,
) -> dict:
    data = {
        "meta": meta,
        "inference_results": [dataclasses.asdict(r) for r in results],
        "bench_results": [
            {"backend": b.backend, "status": b.status, "metrics": b.metrics, "error_message": b.error_message}
            for b in bench_results
        ],
    }
    if model_info_results:
        data["model_info_results"] = [
            {"model": r.model, "arch": r.arch, "quant": r.quant,
             "status": r.status, "load_ms": r.load_ms,
             "error_message": r.error_message, "metadata": r.metadata}
            for r in model_info_results
        ]
    if repeat_results:
        data["repeat_stats"] = {
            f"{k[0]}_{k[1]}_{k[2]}": {
                "n": v.n, "pass": len(v.passing),
                "tps_avg": v.tps_avg, "tps_min": v.tps_min,
                "tps_max": v.tps_max, "tps_stddev": v.tps_stddev,
            }
            for k, v in repeat_results.items()
        }
    return data


def results_to_csv(results: list[RunResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "model", "arch", "quant", "backend", "status",
        "load_ms", "ttft_ms", "tok_per_s", "tokens", "total_ms",
        "size_mb", "correctness", "error",
    ])
    for r in results:
        writer.writerow([
            r.model, r.arch, r.quant, r.backend, r.status,
            f"{r.load_time_ms:.1f}", f"{r.time_to_first_token_ms:.1f}",
            f"{r.tokens_per_sec:.1f}", r.tokens_generated,
            f"{r.total_time_ms:.1f}", f"{r.model_size_mb:.1f}",
            r.correctness, r.error_message,
        ])
    return buf.getvalue()


def results_to_markdown(results: list[RunResult]) -> str:
    lines = [
        "| Status | Arch | Quant | Backend | Size | Load | TTFT | tok/s | Tokens | Total | Golden |",
        "|--------|------|-------|---------|------|------|------|-------|--------|-------|--------|",
    ]
    for r in results:
        golden = r.correctness or "-"
        if r.status == "pass":
            lines.append(
                f"| PASS | {r.arch} | {r.quant} | {r.backend} | "
                f"{r.model_size_mb:.0f} MB | {r.load_time_ms:.0f} ms | "
                f"{r.time_to_first_token_ms:.0f} ms | {r.tokens_per_sec:.1f} | "
                f"{r.tokens_generated} | {r.total_time_ms:.0f} ms | {golden} |"
            )
        else:
            lines.append(
                f"| {r.status.upper()} | {r.arch} | {r.quant} | {r.backend} | "
                f"{r.model_size_mb:.0f} MB | - | - | - | - | - | {golden} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agave Inference Engine - Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    filt = p.add_argument_group("Filtering")
    filt.add_argument("--backend", nargs="+", metavar="BE",
                      help="Backends to test (e.g., metal cpu vulkan cuda)")
    filt.add_argument("--arch", nargs="+", metavar="ARCH",
                      help="Model architectures (e.g., gemma3 qwen35)")
    filt.add_argument("--quant", nargs="+", metavar="Q",
                      help="Quantization formats (e.g., Q4_0 Q8_0 BF16)")
    filt.add_argument("--model", nargs="+", metavar="PATH",
                      help="Specific model files/dirs (overrides discovery)")

    sel = p.add_argument_group("Test selection")
    sel.add_argument("--bench-only", action="store_true",
                     help="Only synthetic benchmarks, skip inference")
    sel.add_argument("--correctness-only", action="store_true",
                     help="Only inference, skip benchmarks")
    sel.add_argument("--model-info-only", action="store_true",
                     help="Only validate model metadata (fast, no inference)")
    sel.add_argument("--smoke", action="store_true", default=True,
                     help="Run 1-token smoke tests per model/backend (default: on)")
    sel.add_argument("--no-smoke", action="store_true",
                     help="Skip smoke tests")
    sel.add_argument("--zig-test", action="store_true",
                     help="Also run zig build test (unit tests)")
    sel.add_argument("--no-bench", action="store_true",
                     help="Skip synthetic benchmarks")

    cfg = p.add_argument_group("Configuration")
    cfg.add_argument("--model-dir", type=Path, default=DEFAULT_WEIGHTS_DIR,
                     help=f"Model search directory (default: {DEFAULT_WEIGHTS_DIR})")
    cfg.add_argument("--prompt", default=DEFAULT_PROMPT,
                     help="Prompt text for inference tests")
    cfg.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                     help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    cfg.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                     help=f"Timeout per run in seconds (default: {DEFAULT_TIMEOUT})")
    cfg.add_argument("--binary", type=Path, default=AGAVE_BIN,
                     help=f"Path to agave binary (default: {AGAVE_BIN})")
    cfg.add_argument("--repeat", type=int, default=1, metavar="N",
                     help="Number of repeat runs per model/backend (default: 1)")

    gold = p.add_argument_group("Golden references")
    gold.add_argument("--generate-golden", action="store_true",
                      help="Save passing outputs as golden references")
    gold.add_argument("--check-golden", action="store_true",
                      help="Validate outputs against golden references")

    out = p.add_argument_group("Output")
    out.add_argument("--output", choices=["table", "json", "csv", "markdown"],
                     default="table", help="Output format (default: table)")
    out.add_argument("--save", type=Path, metavar="PATH",
                     help="Save results to JSON file")
    out.add_argument("--baseline", type=Path, metavar="PATH",
                     help="Compare against baseline JSON for regressions")
    out.add_argument("--regression-threshold", type=float, default=0.05,
                     help="Regression threshold fraction (default: 0.05 = 5%%)")

    prof = p.add_argument_group("Profiling")
    prof.add_argument("--profile", choices=["instruments"],
                      help="Enable profiling (instruments = macOS Time Profiler)")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--dry-run", action="store_true",
                      help="Show what would run without running it")
    misc.add_argument("--verbose", action="store_true",
                      help="Show generated text from each run")
    misc.add_argument("--fail-fast", action="store_true",
                      help="Stop on first failure")

    return p


def main() -> int:
    global AGAVE_BIN

    parser = build_parser()
    args = parser.parse_args()

    if args.binary:
        AGAVE_BIN = args.binary

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold]Platform:[/] {platform.system()} {platform.machine()}\n"
        f"[bold]Binary:[/]   {AGAVE_BIN}",
        title="[bold cyan]Agave Test Harness[/]",
        border_style="cyan",
    ))

    if not AGAVE_BIN.exists():
        console.print(f"\n[bold red]Binary not found:[/] {AGAVE_BIN}")
        console.print("[dim]Run 'zig build' first.[/]")
        return 1

    # Detect backends
    available_backends = detect_available_backends()
    if args.backend:
        backends = [b for b in args.backend if b in available_backends or b == "cpu"]
        missing = set(args.backend) - set(backends)
        if missing:
            console.print(f"[yellow]Backends not available: {', '.join(missing)}[/]")
    else:
        backends = available_backends

    # Discover models
    if args.model:
        models = []
        for mp in args.model:
            p = Path(mp)
            if not p.exists():
                console.print(f"[yellow]Model not found: {mp}[/]")
                continue
            arch = detect_arch(p.name) or "unknown"
            quant = detect_quant(p.name)
            fmt = detect_format(p) if p.is_dir() else ("gguf" if p.suffix == ".gguf" else "unknown")
            size_mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0
            models.append(ModelInfo(path=p, arch=arch, quant=quant, size_mb=size_mb, format=fmt))
    else:
        search_dirs = [args.model_dir]
        # Also check old default if different
        weights_dir = AGAVE_ROOT / "weights"
        if weights_dir.exists() and weights_dir != args.model_dir:
            search_dirs.append(weights_dir)
        models = discover_models(search_dirs)

    if args.arch:
        arch_set = set(a.lower() for a in args.arch)
        models = [m for m in models if m.arch.lower() in arch_set]
    if args.quant:
        quant_set = set(q.upper().replace("-", "_") for q in args.quant)
        models = [m for m in models if m.quant.upper().replace("-", "_") in quant_set]

    # Model table
    model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    model_table.add_column("Name", style="bold")
    model_table.add_column("Size", justify="right")
    model_table.add_column("Format", style="dim")
    model_table.add_column("File", style="dim")
    for m in models:
        model_table.add_row(m.short_name, f"{m.size_mb:.0f} MB", m.format, m.path.name)
    console.print()
    console.print(f"[bold]Backends:[/] {', '.join(backends)}")
    console.print(f"[bold]Models:[/]   {len(models)} found")
    if models:
        console.print(model_table)

    if not models:
        console.print(f"\n[yellow]No models found in {args.model_dir}[/]")
        console.print("[dim]Download models or specify --model-dir / --model[/]")
        return 1

    # Test matrix
    test_matrix: list[tuple[ModelInfo, str]] = [(m, be) for m in models for be in backends]
    total_runs = len(test_matrix) * args.repeat
    do_bench = not args.correctness_only and not args.no_bench and not args.model_info_only
    do_inference = not args.bench_only and not args.model_info_only
    do_smoke = args.smoke and not args.no_smoke and not args.bench_only
    do_model_info = args.model_info_only or True  # Always run model-info as fast validation

    console.print()
    if do_model_info:
        console.print(f"Model info:      [bold]{len(models)}[/] models (metadata validation)")
    if do_smoke:
        console.print(f"Smoke tests:     [bold]{len(test_matrix)}[/] ({len(models)} models x {len(backends)} backends)")
    if do_inference:
        run_label = f"{total_runs}" if args.repeat > 1 else f"{len(test_matrix)}"
        console.print(f"Inference tests: [bold]{run_label}[/] ({len(models)} models x {len(backends)} backends" +
                      (f" x {args.repeat} repeats)" if args.repeat > 1 else ")"))
    if do_bench:
        console.print(f"Synthetic benchmarks: [bold]{len(backends)}[/] backends")
    if args.zig_test:
        console.print("Zig unit tests: [bold]enabled[/]")
    if args.check_golden:
        console.print(f"Golden check:   [bold]enabled[/] ({GOLDEN_DIR})")
    if args.generate_golden:
        console.print(f"Golden save:    [bold]enabled[/] ({GOLDEN_DIR})")

    # Dry run
    if args.dry_run:
        console.print()
        console.print("[bold]DRY RUN - would execute:[/]")
        if do_model_info:
            for m in models:
                console.print(f"  [dim]agave[/] {m.path.name} [dim]--backend {backends[0]} --json --model-info[/]")
        if do_inference:
            for m, be in test_matrix:
                repeat_str = f" [dim](x{args.repeat})[/]" if args.repeat > 1 else ""
                console.print(f"  [dim]agave[/] {m.path.name} [dim]--backend[/] {be} [dim]--json -n[/] {args.max_tokens}{repeat_str}")
        if do_bench:
            for be in backends:
                console.print(f"  [dim]agave[/] --bench [dim]--backend[/] {be}")
        if args.zig_test:
            console.print("  [dim]zig build test[/]")
        return 0

    start_time = time.monotonic()
    inference_results: list[RunResult] = []
    bench_results: list[BenchResult] = []
    model_info_results: list[ModelInfoResult] = []
    repeat_stats: dict[tuple, RepeatStats] = {}
    golden_stats: dict = {"saved": 0, "checked": 0, "match": 0, "mismatch": 0, "no_golden": 0}

    # --- Zig unit tests ---
    if args.zig_test:
        console.print()
        with console.status("[bold]Running zig unit tests...[/]"):
            test_output, test_rc = run_zig_tests(args.timeout * 2)
        if test_rc == 0:
            console.print(f"  Zig unit tests: [bold green]PASS[/]")
        else:
            console.print(f"  Zig unit tests: [bold red]FAIL[/]")
            if args.verbose:
                console.print(f"[dim]{_truncate(test_output, 500)}[/]")

    # --- Model info validation ---
    if do_model_info:
        console.print()
        console.print("[bold]Validating model metadata...[/]")
        for model in models:
            be = backends[0]  # Only need one backend for metadata
            with console.status(f"  [bold]{model.short_name}...[/]"):
                mi = run_model_info(model, be, args.timeout)
            model_info_results.append(mi)
            if mi.status == "pass":
                md = mi.metadata
                params = _format_params(md.get("n_params", 0)) if md.get("n_params") else ""
                console.print(
                    f"  [green]PASS[/] {model.short_name} "
                    f"[dim]({md.get('arch', '?')} {md.get('layers', '?')}L "
                    f"{md.get('embed', '?')}E {params} {mi.load_ms:.0f}ms)[/]"
                )
            else:
                console.print(f"  {status_markup(mi.status)} {model.short_name}")
                if mi.error_message:
                    console.print(f"    [dim]{_truncate(mi.error_message, 120)}[/]")

            if args.fail_fast and mi.status in ("fail", "error"):
                console.print("[bold red]Stopping (--fail-fast)[/]")
                break

        if args.model_info_only:
            elapsed = time.monotonic() - start_time
            if args.output == "table":
                print_model_info_table(model_info_results)
                print_summary([], [], model_info_results, [], golden_stats, elapsed)
            elif args.output == "json":
                data = results_to_json([], [], model_info_results, {}, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "platform": platform.system(), "machine": platform.machine(),
                    "elapsed_s": round(elapsed, 1),
                })
                console.print_json(json.dumps(data, indent=2))
            any_fail = any(r.status in ("fail", "error") for r in model_info_results)
            return 1 if any_fail else 0

    # --- Smoke tests (1-token inference to catch crashes/0-token) ---
    smoke_results: list[SmokeResult] = []
    if do_smoke and not args.model_info_only:
        console.print()
        console.print("[bold]Running smoke tests (1-token inference)...[/]")
        smoke_timeout = min(args.timeout, 60)  # Smoke tests should be fast
        for model, backend in test_matrix:
            label = f"{model.short_name} on {backend}"
            with console.status(f"  [bold]{label}...[/]"):
                sr = run_smoke(model, backend, smoke_timeout)
            smoke_results.append(sr)

            if sr.status == "pass":
                console.print(f"  [green]PASS[/] {label} [dim]({sr.tokens_generated} tok)[/]")
            elif sr.status == "crash":
                console.print(f"  [bold red]CRASH[/] {label}")
                console.print(f"    [red]{sr.error_message}[/]")
            elif sr.status == "fail":
                console.print(f"  [bold red]FAIL[/] {label}")
                console.print(f"    [red]{sr.error_message}[/]")
            else:
                console.print(f"  {status_markup(sr.status)} {label}")
                if sr.error_message:
                    console.print(f"    [dim]{_truncate(sr.error_message, 120)}[/]")

            if args.fail_fast and sr.status in ("crash", "fail", "error"):
                console.print("[bold red]Stopping (--fail-fast)[/]")
                break

        # Print smoke summary
        smoke_pass = sum(1 for s in smoke_results if s.status == "pass")
        smoke_crash = sum(1 for s in smoke_results if s.status == "crash")
        smoke_fail = sum(1 for s in smoke_results if s.status == "fail")
        if smoke_crash > 0 or smoke_fail > 0:
            console.print()
            console.print(f"  [bold]Smoke: {smoke_pass}/{len(smoke_results)} pass", end="")
            if smoke_crash: console.print(f", [bold red]{smoke_crash} CRASH[/]", end="")
            if smoke_fail: console.print(f", [bold red]{smoke_fail} FAIL[/]", end="")
            console.print("[/]")

    # --- Synthetic benchmarks ---
    if do_bench:
        console.print()
        for be in backends:
            with console.status(f"[bold]Benchmarking {be}...[/]"):
                br = run_bench(be, args.timeout)
            bench_results.append(br)
            icon = "[green]PASS[/]" if br.status == "pass" else status_markup(br.status)
            console.print(f"  Benchmark {be}: {icon}")

    # --- Inference tests ---
    stop_early = False
    if do_inference and test_matrix:
        console.print()
        console.print("[bold]Running inference tests...[/]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Inference", total=total_runs)

            for model, backend in test_matrix:
                key = (model.arch, model.quant, backend)
                if key not in repeat_stats:
                    repeat_stats[key] = RepeatStats()

                for run_idx in range(args.repeat):
                    label = f"{model.short_name} on {backend}"
                    if args.repeat > 1:
                        label += f" [{run_idx + 1}/{args.repeat}]"
                    progress.update(task, description=f"[bold]{label}[/]")

                    result = run_inference(
                        model, backend, args.prompt, args.max_tokens, args.timeout,
                    )

                    # Golden checks
                    if args.check_golden and result.status == "pass":
                        result.correctness = check_golden(result, model)
                        golden_stats["checked"] += 1
                        golden_stats[result.correctness] += 1
                    if args.generate_golden and result.status == "pass" and run_idx == 0:
                        save_golden(result, model)
                        golden_stats["saved"] += 1

                    inference_results.append(result)
                    repeat_stats[key].runs.append(result)
                    progress.advance(task)

                    # Print per-run output (only for first run if repeating)
                    if run_idx == 0 or args.verbose:
                        if result.status == "pass":
                            perf_parts = []
                            if result.tokens_per_sec > 0:
                                perf_parts.append(f"{result.tokens_per_sec:.1f} tok/s")
                            if result.time_to_first_token_ms > 0:
                                perf_parts.append(f"TTFT={result.time_to_first_token_ms:.0f}ms")
                            perf = f" [dim]({', '.join(perf_parts)})[/]" if perf_parts else ""
                            golden_str = ""
                            if result.correctness:
                                golden_str = f" {correctness_markup(result.correctness)}"
                            console.print(f"  [green]PASS[/] {label}{perf}{golden_str}")
                        else:
                            console.print(f"  {status_markup(result.status)} {label}")
                            if result.error_message:
                                console.print(f"    [dim]{_truncate(result.error_message, 120)}[/]")

                    if args.verbose and result.output_text:
                        console.print(f"    [dim]Output: {_truncate(result.output_text, 100)}[/]")

                    if args.profile == "instruments" and result.status == "pass" and run_idx == 0:
                        with console.status("  Profiling with Instruments..."):
                            trace = profile_with_instruments(
                                model, backend, args.prompt, args.max_tokens, args.timeout,
                            )
                        if trace:
                            console.print(f"    [dim]Trace: {trace}[/]")

                    if args.fail_fast and result.status in ("fail", "error"):
                        console.print("[bold red]Stopping (--fail-fast)[/]")
                        stop_early = True
                        break

                if stop_early:
                    break

    elapsed = time.monotonic() - start_time

    # --- Regression check ---
    regressions = []
    if args.baseline:
        # Use best run from repeats for regression comparison
        best_results = []
        for stats in repeat_stats.values():
            best = stats.best
            if best:
                best_results.append(best)
        regressions = compare_to_baseline(
            best_results if best_results else inference_results,
            args.baseline, args.regression_threshold,
        )

    # --- Output ---
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "binary": str(AGAVE_BIN),
        "backends": backends,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "elapsed_s": round(elapsed, 1),
    }

    if args.output == "json":
        data = results_to_json(inference_results, bench_results, model_info_results, repeat_stats, meta)
        console.print_json(json.dumps(data, indent=2))
    elif args.output == "csv":
        print(results_to_csv(inference_results), end="")
    elif args.output == "markdown":
        print(results_to_markdown(inference_results))
    else:
        if model_info_results:
            print_model_info_table(model_info_results)
        if bench_results:
            print_bench_table(bench_results)
        if inference_results:
            show_golden = args.check_golden or args.generate_golden
            print_inference_table(inference_results, show_golden=show_golden)
        if args.repeat > 1 and repeat_stats:
            print_repeat_table(repeat_stats)
        print_summary(inference_results, bench_results, model_info_results, regressions, golden_stats, elapsed)

    # --- Save ---
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        data = results_to_json(inference_results, bench_results, model_info_results, repeat_stats, meta)
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[dim]Results saved to {args.save}[/]")

    any_fail = any(r.status in ("fail", "error") for r in inference_results)
    any_mi_fail = any(r.status in ("fail", "error") for r in model_info_results)
    any_smoke_fail = any(r.status in ("fail", "crash", "error") for r in smoke_results)
    any_mismatch = golden_stats.get("mismatch", 0) > 0
    return 1 if (any_fail or any_mi_fail or any_smoke_fail or regressions or any_mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
