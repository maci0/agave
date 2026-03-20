#!/usr/bin/env python3
"""
Unified autotune orchestrator for Agave kernel research.

Modes:
    bench   - Run benchmarks (micro by default, --e2e for end-to-end)
    tune    - Single optimization cycle
    grid    - Grid search over a parameter
    auto    - Autonomous optimization loop
    staged  - Manage staged improvements
    status  - Show optimization history
"""

import subprocess
import sys
import time
import json
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

from registry import AGAVE_ROOT, KERNELS, Kernel, find_kernels, changed_kernels

BENCH_BINARY = AGAVE_ROOT / "zig-out" / "bin" / "agave-bench"
DATA_DIR = Path(__file__).parent
RESULTS_FILE = DATA_DIR / "results.tsv"
BASELINE_FILE = DATA_DIR / "baseline.json"
LOG_FILE = DATA_DIR / "optimization_log.tsv"
SEARCH_SPACES_FILE = DATA_DIR / "search_spaces.toml"
STAGING_DIR = DATA_DIR / "staging"

BENCH_MODELS = {
    "q4_0": "weights/gemma-3-1b-it-q4_0.gguf",
    "q8_0": "weights/gemma-3-1b-it-q8_0.gguf",
    "bf16": "weights/gemma-3-1b-it-bf16.gguf",
}


# ── Helpers ───────────────────────────────────────────────────────

def die(msg: str):
    """Print error to stderr and exit."""
    print(msg, file=sys.stderr)
    sys.exit(1)


def resolve_kernel_args(parsed) -> list[Kernel]:
    """Resolve kernels from --changed or positional arg. Exits on failure."""
    if getattr(parsed, "changed", False):
        kernels = changed_kernels()
        if not kernels:
            die("No kernel source files changed.")
        return kernels
    name = getattr(parsed, "kernel", None)
    if not name:
        die("Specify a kernel name or use --changed. Use 'run.py list' to see available.")
    kernels = find_kernels(name)
    if not kernels:
        die(f"No kernels matching {name!r}. Use 'run.py list' to see available.")
    return kernels


def confirm_modify(file_path: str, kernel_name: str) -> bool:
    """Ask for confirmation before modifying source files."""
    print(f"\nThis will temporarily modify {file_path} to test parameter values")
    print(f"for kernel '{kernel_name}'. Improvements will be saved to staging/.")
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def stage_improvement(kernel_name: str, file_path: str, patched_content: str,
                      result_info: dict) -> Path:
    """Save an improved source file to the staging area for later review."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    desc = result_info.get("description", "improvement")
    safe_desc = "".join(c if c.isalnum() or c in "-_" else "_" for c in desc)
    entry_dir = STAGING_DIR / f"{kernel_name}_{ts}_{safe_desc}"
    entry_dir.mkdir(parents=True, exist_ok=True)

    dest = entry_dir / Path(file_path).name
    dest.write_text(patched_content)

    meta = {
        "kernel": kernel_name,
        "source_file": str(file_path),
        "staged_file": str(dest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **result_info,
    }
    (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return entry_dir


def find_bench_quant(kernel: Kernel) -> str | None:
    """Find the first available bench quant key for a kernel."""
    for q in kernel.bench_quants:
        mp = BENCH_MODELS.get(q)
        if mp and (AGAVE_ROOT / mp).exists():
            return q
    for q, mp in BENCH_MODELS.items():
        if (AGAVE_ROOT / mp).exists():
            return q
    return None


# ── Build / run helpers ───────────────────────────────────────────

def load_search_spaces() -> dict:
    """Load search space definitions from TOML."""
    if not SEARCH_SPACES_FILE.exists():
        return {}
    with open(SEARCH_SPACES_FILE, "rb") as f:
        return tomllib.load(f)


def ensure_built(e2e: bool = False):
    """Build the required binary if it doesn't exist yet (skip if already built)."""
    target = "" if e2e else "bench"
    binary = (AGAVE_ROOT / "zig-out" / "bin" / "agave") if e2e else BENCH_BINARY
    if not binary.exists():
        print(f"Building {'agave' if e2e else 'agave-bench'}...")
        cmd = ["zig", "build"] + ([target] if target else [])
        subprocess.run(cmd, cwd=AGAVE_ROOT, check=True)


def _run_json_cmd(cmd: list[str]) -> dict:
    """Run a command that outputs JSON, returning parsed dict or error status."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(AGAVE_ROOT)
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    if result.returncode != 0:
        return {"status": "FAIL", "error": result.stderr[:200]}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"status": "FAIL", "error": f"Bad JSON: {result.stdout[:100]}"}


def run_micro_bench(kernel: str, backend: str = "cpu",
                    n: int = 4096, k: int = 4096, iters: int = 100) -> dict:
    """Run micro-benchmark for a single kernel."""
    return _run_json_cmd([
        str(BENCH_BINARY), kernel,
        f"--n={n}", f"--k={k}", f"--iters={iters}",
        f"--backend={backend}",
    ])


def run_e2e(model_path: str, backend: str = "cpu", n_tokens: int = 10) -> dict:
    """Run end-to-end inference benchmark."""
    return _run_json_cmd([
        str(BENCH_BINARY), "e2e",
        f"--model={AGAVE_ROOT / model_path}",
        f"--backend={backend}", f"--n={n_tokens}",
    ])


def resolve_quants(kernels: list[Kernel] | None) -> set[str]:
    """Determine which quant formats to benchmark."""
    if kernels is None:
        return set(BENCH_MODELS.keys())
    quants = set()
    for k in kernels:
        quants.update(k.bench_quants)
    return quants if quants else set(BENCH_MODELS.keys())


def rebuild() -> bool:
    """Unconditionally rebuild Agave. Returns False on failure."""
    result = subprocess.run(
        ["zig", "build"], cwd=AGAVE_ROOT, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr[:500]}", file=sys.stderr)
        return False
    return True


def revert_sources(kernel: Kernel):
    """Revert source files for a kernel via git checkout."""
    paths = []
    for p in kernel.sources.values():
        if isinstance(p, list):
            paths.extend(p)
        else:
            paths.append(p)
    if paths:
        subprocess.run(["git", "checkout", "--"] + paths, cwd=AGAVE_ROOT)


def log_result(kernel_name: str, backend: str, metric_value: float,
               metric_name: str, status: str, description: str):
    """Append to optimization log."""
    header = "timestamp\tkernel\tbackend\tmetric\tvalue\tstatus\tdescription\n"
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w") as f:
            f.write(header)
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts}\t{kernel_name}\t{backend}\t{metric_name}\t"
                f"{metric_value:.2f}\t{status}\t{description}\n")


def measure(kernel: Kernel, backend: str, e2e: bool, metric: str) -> float | None:
    """Run a measurement and return the requested metric."""
    if e2e:
        quant = find_bench_quant(kernel)
        if not quant:
            return None
        stats = run_e2e(BENCH_MODELS[quant], backend=backend)
    else:
        stats = run_micro_bench(kernel.name, backend=backend)

    # Return metric even from non-PASS results if the key exists
    if stats.get("status") not in ("PASS", None) and metric not in stats:
        return None
    return stats.get(metric)


# ── Commands ──────────────────────────────────────────────────────

def cmd_bench(args):
    """Run benchmarks (micro by default)."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py bench")
    parser.add_argument("kernel", nargs="?", default=None)
    parser.add_argument("--backend", "-b", nargs="+", default=["cpu"])
    parser.add_argument("--dim", type=int, default=4096,
                        help="Output dimension / vector length for micro mode (default: 4096)")
    parser.add_argument("--k", type=int, default=4096,
                        help="Input dimension for GEMV micro mode (default: 4096)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of timed iterations for micro mode (default: 100)")
    parser.add_argument("-n", "--n-tokens", type=int, default=10,
                        help="Tokens to generate in e2e mode (default: 10)")
    parser.add_argument("--e2e", action="store_true",
                        help="Use end-to-end mode instead of micro-benchmark")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--changed", action="store_true")
    parsed = parser.parse_args(args)

    selected = None
    if parsed.changed:
        selected = changed_kernels()
        if not selected:
            die("No kernel source files changed.")
    elif parsed.kernel:
        selected = find_kernels(parsed.kernel)
        if not selected:
            die(f"No kernels matching {parsed.kernel!r}. Use 'run.py list' to see available.")

    if not parsed.e2e and not selected:
        die("Specify a kernel name, use --changed, or use --e2e for end-to-end mode.")

    ensure_built(e2e=parsed.e2e)

    if not parsed.e2e:
        # Micro-benchmark mode (default): run each kernel in isolation
        results = []
        for kernel in selected:
            for backend in parsed.backend:
                label = f"{kernel.name}/{backend}"
                print(f"  {label:<22}", end="", flush=True)
                r = run_micro_bench(kernel.name, backend=backend,
                                    n=parsed.dim, k=parsed.k, iters=parsed.iters)
                r["backend"] = backend
                r["kernels"] = [kernel.name]
                results.append(r)
                if "ns_median" in r:
                    print(f"{r['ns_median']:>8} ns  ({r.get('gb_s', 0):.1f} GB/s)")
                else:
                    print(f"{r.get('status', 'FAIL')}: {r.get('error', '')[:60]}")
    else:
        # E2E mode
        quants = resolve_quants(selected)
        kernel_names = [k.name for k in selected] if selected else ["all"]
        results = []
        for quant in sorted(quants):
            model_path = BENCH_MODELS.get(quant)
            if not model_path or not (AGAVE_ROOT / model_path).exists():
                continue
            for backend in parsed.backend:
                label = f"{quant}/{backend}"
                print(f"  {label:<16}", end="", flush=True)
                r = run_e2e(model_path, backend=backend, n_tokens=parsed.n_tokens)
                r["quant"] = quant
                r["backend"] = backend
                r["kernels"] = kernel_names
                results.append(r)
                tps = r.get("tok_per_sec", 0)
                if tps:
                    print(f"{tps:>6.1f} tok/s")
                else:
                    print(f"{r.get('status', 'FAIL')}: {r.get('error', '')[:60]}")

    if parsed.save_baseline:
        save_baseline(results)
    if parsed.compare:
        compare_with_baseline(results)

    tag = parsed.tag or (
        ",".join(k.name for k in selected) if selected else ""
    )
    append_tsv(results, tag=tag)


def save_baseline(results):
    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {BASELINE_FILE}")


def compare_with_baseline(results):
    if not BASELINE_FILE.exists():
        print("No baseline found. Run with --save-baseline first.", file=sys.stderr)
        return
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    # Detect mode: micro results have ns_median, e2e results have tok_per_sec
    is_micro = any(r.get("ns_median") for r in results)

    matched = False
    if is_micro:
        # Micro-benchmark: key by kernel name(s) + backend
        baseline_map = {}
        for r in baseline:
            knames = ",".join(r.get("kernels", []))
            baseline_map[(knames, r.get("backend", ""))] = r

        print(f"\n  {'Kernel':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        for r in results:
            knames = ",".join(r.get("kernels", []))
            key = (knames, r.get("backend", ""))
            if key not in baseline_map:
                continue
            base = baseline_map[key]
            base_ns = base.get("ns_median", 0)
            curr_ns = r.get("ns_median", 0)
            if base_ns > 0:
                matched = True
                change = (curr_ns - base_ns) / base_ns * 100
                marker = "+" if change > 0 else ""
                # For ns, positive change = slower = regression
                flag = " <-- REGRESSION" if change > 5 else ""
                label = f"{knames}/{r.get('backend', '')}"
                print(f"  {label:<20} {base_ns:>8} ns {curr_ns:>8} ns {marker}{change:>8.1f}%{flag}")
    else:
        # E2E: key by quant + backend
        baseline_map = {(r.get("quant", ""), r.get("backend", "")): r for r in baseline}
        print(f"\n  {'Config':<16} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        for r in results:
            key = (r.get("quant", ""), r.get("backend", ""))
            if key not in baseline_map:
                continue
            base = baseline_map[key]
            base_tps = base.get("tok_per_sec", 0)
            curr_tps = r.get("tok_per_sec", 0)
            if base_tps > 0:
                matched = True
                change = (curr_tps - base_tps) / base_tps * 100
                marker = "+" if change > 0 else ""
                flag = " <-- REGRESSION" if change < -5 else ""
                print(f"  {r.get('quant','')}/{r.get('backend',''):<10} "
                      f"{base_tps:>9.1f} {curr_tps:>9.1f} {marker}{change:>8.1f}%{flag}")

    if not matched:
        print("  (no matching baseline entries — was baseline saved in the same mode?)",
              file=sys.stderr)


def append_tsv(results, tag=""):
    header = "timestamp\ttag\tkernels\tquant\tbackend\ttok_per_sec\tprefill_ms\tgen_ms\tns_median\tgb_s\tstatus\n"
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(RESULTS_FILE, "a") as f:
        for r in results:
            knames = ",".join(r.get("kernels", ["all"]))
            f.write(f"{ts}\t{tag}\t{knames}\t{r.get('quant', '')}\t{r.get('backend', '')}\t"
                    f"{r.get('tok_per_sec', 0):.1f}\t{r.get('prefill_ms', 0):.0f}\t"
                    f"{r.get('gen_ms', 0):.0f}\t{r.get('ns_median', 0)}\t"
                    f"{r.get('gb_s', 0):.1f}\t{r.get('status', '')}\n")


def cmd_tune(args):
    """Single optimization cycle: build -> benchmark -> log."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py tune")
    parser.add_argument("kernel", nargs="?")
    parser.add_argument("--backend", "-b", nargs="+", default=["cpu"])
    parser.add_argument("--description", "-d", default="manual optimization")
    parser.add_argument("--auto-revert", action="store_true",
                        help="Revert source files on build failure")
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--e2e", action="store_true",
                        help="Use end-to-end mode instead of micro-benchmark")
    parsed = parser.parse_args(args)

    kernels = resolve_kernel_args(parsed)

    for kernel in kernels:
        for backend in parsed.backend:
            print(f"\n{'=' * 60}")
            print(f"Kernel:  {kernel.name} ({kernel.description})")
            print(f"Backend: {backend}")
            print(f"Change:  {parsed.description}")
            print(f"{'=' * 60}")

            print("Building...", end=" ", flush=True)
            if not rebuild():
                if parsed.auto_revert:
                    print("Reverting...")
                    revert_sources(kernel)
                sys.exit(1)

            print("OK")
            print(f"Benchmarking ({backend})...", end=" ", flush=True)

            if parsed.e2e:
                quant = find_bench_quant(kernel)
                if not quant:
                    print("SKIP: No model files found", file=sys.stderr)
                    continue
                stats = run_e2e(BENCH_MODELS[quant], backend=backend)
                metric_name = "tok_per_sec"
                metric_val = stats.get("tok_per_sec", 0)
                print(f"{metric_val:.1f} tok/s (model: {quant})")
            else:
                stats = run_micro_bench(kernel.name, backend=backend)
                metric_name = "ns_median"
                metric_val = stats.get("ns_median", 0)
                print(f"{metric_val} ns ({stats.get('gb_s', 0):.1f} GB/s)")

            log_result(kernel.name, backend, metric_val,
                       metric_name, stats.get("status", "PASS"), parsed.description)
            print(f"Logged to {LOG_FILE}")


def cmd_grid(args):
    """Exhaustive grid search over a parameter."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py grid")
    parser.add_argument("kernel", nargs="?")
    parser.add_argument("--backend", "-b", default="cpu",
                        help="Single backend — grid modifies source files (default: cpu)")
    parser.add_argument("--param", required=True)
    parser.add_argument("--values", required=True, help="Comma-separated values")
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--file", default=None)
    parser.add_argument("--e2e", action="store_true",
                        help="Use end-to-end mode instead of micro-benchmark")
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parsed = parser.parse_args(args)

    kernels = resolve_kernel_args(parsed)
    kernel = kernels[0]

    file_path = parsed.file
    if not file_path:
        src = kernel.sources.get(parsed.backend)
        if not src:
            die(f"Kernel {kernel.name} has no {parsed.backend} source.")
        file_path = src if isinstance(src, str) else src[0]

    full_path = AGAVE_ROOT / file_path
    if not full_path.exists():
        die(f"Source file not found: {file_path}")

    original = full_path.read_text()
    if parsed.pattern not in original:
        die(f"Pattern not found in {file_path}: '{parsed.pattern}'")

    if not parsed.yes and not confirm_modify(file_path, kernel.name):
        print("Aborted.")
        return

    values = []
    for v in parsed.values.split(","):
        v = v.strip()
        try:
            values.append(int(v))
        except ValueError:
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)

    results = []
    unit = "ns"
    print(f"\nGrid search: {kernel.name} / {parsed.param}")
    print(f"File: {file_path}")
    print(f"{'=' * 60}")

    try:
        for v in values:
            replacement = parsed.template.format(value=v)
            patched = original.replace(parsed.pattern, replacement, 1)
            full_path.write_text(patched)

            print(f"\n  {parsed.param}={v}:", end=" ", flush=True)

            if not rebuild():
                print("BUILD FAILED")
                log_result(kernel.name, parsed.backend, 0, parsed.param,
                           "FAIL", f"grid:{parsed.param}={v}")
                results.append((v, 0, "FAIL", None))
                continue

            if parsed.e2e:
                quant = find_bench_quant(kernel)
                stats = run_e2e(BENCH_MODELS[quant], backend=parsed.backend) if quant else {}
                metric = stats.get("tok_per_sec", 0)
                unit = "tok/s"
            else:
                stats = run_micro_bench(kernel.name, backend=parsed.backend)
                metric = stats.get("ns_median", 0)
                unit = "ns"

            status = stats.get("status", "PASS")
            print(f"{metric:.1f} {unit}" if status == "PASS" else status)
            results.append((v, metric, status, patched))
            log_result(kernel.name, parsed.backend, metric, parsed.param,
                       status, f"grid:{parsed.param}={v}")
    finally:
        # Always restore original source
        full_path.write_text(original)

    print(f"\n{'=' * 60}")
    best_v, best_metric, best_patched = None, (0 if unit == "tok/s" else float("inf")), None
    for v, m, s, patched in results:
        is_better = (m > best_metric) if unit == "tok/s" else (m < best_metric and m > 0)
        if s == "PASS" and is_better:
            best_metric = m
            best_v = v
            best_patched = patched
        print(f"  {str(v):>10}  {m:>10.1f} {unit}  {s}")

    if best_v is not None and best_patched is not None:
        print(f"\n  Best: {parsed.param}={best_v} -> {best_metric:.1f} {unit}")
        entry_dir = stage_improvement(kernel.name, file_path, best_patched, {
            "description": f"grid:{parsed.param}={best_v}",
            "param": parsed.param,
            "value": best_v,
            "metric_name": "tok_per_sec" if unit == "tok/s" else "ns_median",
            "metric_value": best_metric,
            "backend": parsed.backend,
        })
        print(f"  Staged to {entry_dir.relative_to(DATA_DIR)}/")
        print(f"  Apply with: run.py staged apply {entry_dir.name}")


def cmd_auto(args):
    """Autonomous optimization loop."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py auto")
    parser.add_argument("kernel", nargs="?")
    parser.add_argument("--backend", "-b", default="cpu",
                        help="Single backend — auto modifies source files (default: cpu)")
    parser.add_argument("--metric", default="ns_median",
                        help="Metric to optimize: ns_median (lower=better) or tok_per_sec (higher=better)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--target-ns", type=int, default=None)
    parser.add_argument("--target-tps", type=float, default=None)
    parser.add_argument("--strategy", choices=["hill-climb", "bayesian"], default="hill-climb")
    parser.add_argument("--e2e", action="store_true",
                        help="Use end-to-end mode instead of micro-benchmark")
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parsed = parser.parse_args(args)

    kernels = resolve_kernel_args(parsed)
    kernel = kernels[0]

    spaces = load_search_spaces()
    space = spaces.get(kernel.name)
    if not space:
        die(f"No search space defined for {kernel.name} in search_spaces.toml")

    # Resolve file and params for this backend
    backend_override = space.get("backends", {}).get(parsed.backend)
    if backend_override:
        file_path = backend_override["file"]
        dimensions = backend_override["dimensions"]
        params = backend_override.get("params", {})
    else:
        file_path = space["file"]
        dimensions = space["dimensions"]
        params = space.get("params", {})

    full_path = AGAVE_ROOT / file_path
    if not full_path.exists():
        die(f"Source file not found: {file_path}")

    if not parsed.yes and not confirm_modify(file_path, kernel.name):
        print("Aborted.")
        return

    lower_is_better = parsed.metric in ("ns_median",)

    # Measure baseline
    print(f"{'=' * 60}")
    print(f"Autonomous optimization: {kernel.name}")
    print(f"Backend: {parsed.backend}, Strategy: {parsed.strategy}")
    print(f"Patience: {parsed.patience}, Max iters: {parsed.max_iters}")
    print(f"{'=' * 60}")

    ensure_built(e2e=parsed.e2e)
    baseline = measure(kernel, parsed.backend, parsed.e2e, parsed.metric)
    if baseline is None:
        die("Failed to measure baseline.")
    print(f"Baseline: {baseline:.2f} {parsed.metric}")

    best = baseline
    original_source = full_path.read_text()

    if parsed.strategy == "bayesian":
        best_patched = run_bayesian(kernel, parsed, params, dimensions, full_path,
                                    original_source, baseline, lower_is_better)
        if best_patched:
            entry_dir = stage_improvement(kernel.name, file_path, best_patched["source"], {
                "description": "bayesian:best",
                "metric_name": parsed.metric,
                "metric_value": best_patched["value"],
                "params": best_patched.get("params", {}),
                "baseline": baseline,
                "backend": parsed.backend,
            })
            print(f"\n  Staged to {entry_dir.relative_to(DATA_DIR)}/")
            print(f"  Apply with: run.py staged apply {entry_dir.name}")
        return

    # Hill-climb
    patience_remaining = parsed.patience
    consecutive_build_failures = 0
    best_patched_source = None

    # Track which values have been tried per dimension
    tried: dict[str, set] = {d: set() for d in dimensions}
    exhausted: set[str] = set()

    try:
        for iteration in range(1, parsed.max_iters + 1):
            if patience_remaining <= 0:
                print(f"\nPatience exhausted after {iteration - 1} iterations.")
                break

            # Check if all dimensions exhausted
            if len(exhausted) == len(dimensions):
                print(f"\nAll dimensions exhausted after {iteration - 1} iterations.")
                break

            # Check target
            if parsed.target_ns and best <= parsed.target_ns:
                print(f"\nTarget reached: {best} <= {parsed.target_ns}")
                break
            if parsed.target_tps and best >= parsed.target_tps:
                print(f"\nTarget reached: {best} >= {parsed.target_tps}")
                break

            # Pick next hypothesis: cycle through non-exhausted dimensions
            active_dims = [d for d in dimensions if d not in exhausted]
            if not active_dims:
                break
            dim_name = active_dims[(iteration - 1) % len(active_dims)]
            param = params.get(dim_name, {})
            values = param.get("values", [])
            untried = [v for v in values if v not in tried[dim_name]]
            if not untried:
                exhausted.add(dim_name)
                continue
            value = untried[0]
            tried[dim_name].add(value)

            pattern = param.get("pattern", "")
            template = param.get("template", "")
            if not pattern or not template:
                continue

            current_source = full_path.read_text()
            if pattern not in current_source:
                print(f"\n  [{iteration}] Pattern not found: '{pattern[:60]}...'")
                print("  ABORT: search space configuration error.", file=sys.stderr)
                break

            replacement = template.format(value=value)
            patched = current_source.replace(pattern, replacement, 1)
            full_path.write_text(patched)

            print(f"\n  [{iteration}] {dim_name}={value}", end=" ", flush=True)

            if not rebuild():
                print("BUILD FAILED")
                full_path.write_text(current_source)
                consecutive_build_failures += 1
                if consecutive_build_failures >= 3:
                    print("  ABORT: 3 consecutive build failures.", file=sys.stderr)
                    break
                log_result(kernel.name, parsed.backend, 0, parsed.metric,
                           "FAIL", f"auto:{dim_name}={value}")
                continue

            consecutive_build_failures = 0
            result = measure(kernel, parsed.backend, parsed.e2e, parsed.metric)

            if result is None:
                print("CRASH/TIMEOUT")
                full_path.write_text(current_source)
                log_result(kernel.name, parsed.backend, 0, parsed.metric,
                           "CRASH", f"auto:{dim_name}={value}")
                continue

            improved = (result < best) if lower_is_better else (result > best)

            if improved:
                print(f"-> {result:.2f} (KEEP, was {best:.2f})")
                best = result
                best_patched_source = patched
                patience_remaining = parsed.patience
                # Update pattern for next iteration
                param["pattern"] = replacement
                log_result(kernel.name, parsed.backend, result, parsed.metric,
                           "KEEP", f"auto:{dim_name}={value}")
            else:
                print(f"-> {result:.2f} (REVERT, best={best:.2f})")
                full_path.write_text(current_source)
                patience_remaining -= 1
                log_result(kernel.name, parsed.backend, result, parsed.metric,
                           "REVERT", f"auto:{dim_name}={value}")
    finally:
        # Always restore original source
        full_path.write_text(original_source)

    # Final report
    print(f"\n{'=' * 60}")
    print(f"Result: {baseline:.2f} -> {best:.2f} {parsed.metric}")
    improvement = ((baseline - best) / baseline * 100) if lower_is_better else (
        (best - baseline) / baseline * 100) if baseline > 0 else 0
    print(f"Improvement: {improvement:+.1f}%")
    print(f"{'=' * 60}")

    if best_patched_source is not None and best != baseline:
        entry_dir = stage_improvement(kernel.name, file_path, best_patched_source, {
            "description": "auto:hill-climb",
            "metric_name": parsed.metric,
            "metric_value": best,
            "baseline": baseline,
            "improvement_pct": round(improvement, 1),
            "backend": parsed.backend,
        })
        print(f"\nBest result staged to {entry_dir.relative_to(DATA_DIR)}/")
        print(f"Apply with: run.py staged apply {entry_dir.name}")


def run_bayesian(kernel, parsed, params, dimensions, full_path,
                 original_source, baseline, lower_is_better):
    """Bayesian optimization using Optuna. Returns best patched source dict or None."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Install with: uv pip install optuna", file=sys.stderr)
        print("Or use --strategy hill-climb", file=sys.stderr)
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if lower_is_better else "maximize"
    study = optuna.create_study(direction=direction)

    best_source = [None]  # mutable container for closure

    def objective(trial):
        current_source = original_source  # Always start from original
        for dim_name in dimensions:
            param = params.get(dim_name, {})
            ptype = param.get("type", "categorical")
            pattern = param.get("pattern", "")
            template = param.get("template", "")

            if ptype == "categorical":
                value = trial.suggest_categorical(dim_name, param.get("values", []))
            elif ptype == "int_range":
                value = trial.suggest_int(dim_name, param["low"], param["high"],
                                          step=param.get("step", 1))
            elif ptype == "float_range":
                value = trial.suggest_float(dim_name, param["low"], param["high"],
                                            step=param.get("step"))
            elif ptype == "boolean":
                value = trial.suggest_categorical(dim_name, [True, False])
                if value:
                    template = param.get("template_true", template)
                else:
                    template = param.get("template_false", pattern)
            else:
                continue

            if pattern and pattern in current_source:
                replacement = template.format(value=value)
                current_source = current_source.replace(pattern, replacement, 1)

        full_path.write_text(current_source)

        if not rebuild():
            full_path.write_text(original_source)
            return float("inf") if lower_is_better else 0.0

        result = measure(kernel, parsed.backend, parsed.e2e, parsed.metric)
        full_path.write_text(original_source)

        if result is None:
            return float("inf") if lower_is_better else 0.0

        # Track best source
        if best_source[0] is None:
            best_source[0] = {"source": current_source, "value": result}
        else:
            is_better = (result < best_source[0]["value"]) if lower_is_better else (
                result > best_source[0]["value"])
            if is_better:
                best_source[0] = {"source": current_source, "value": result,
                                  "params": trial.params}

        log_result(kernel.name, parsed.backend, result, parsed.metric,
                   "TRIAL", f"bayesian:trial-{trial.number}")
        return result

    try:
        # n_jobs=1: objective mutates full_path on disk — parallel trials would corrupt
        study.optimize(objective, n_trials=parsed.max_iters, n_jobs=1)
    finally:
        full_path.write_text(original_source)

    print(f"\nBest trial: {study.best_value:.2f} {parsed.metric}")
    print(f"Best params: {study.best_params}")
    print(f"Baseline: {baseline:.2f} -> Best: {study.best_value:.2f}")

    if best_source[0] is not None:
        best_source[0]["params"] = study.best_params
    return best_source[0]


def cmd_status(args):
    """Show optimization history."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py status")
    parser.add_argument("kernel", nargs="?", help="Filter by kernel name")
    parsed = parser.parse_args(args)

    if not LOG_FILE.exists():
        print("No optimization history yet.")
        return

    print("\nOptimization History:")
    print(f"  {'Timestamp':<18} {'Kernel':<18} {'Backend':<8} {'Value':>14} {'Status':<8} Description")
    print(f"  {'-' * 92}")
    with open(LOG_FILE) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 7:
                continue
            if parsed.kernel and parsed.kernel not in parts[1]:
                continue
            ts, kernel, backend, metric, value, status, desc = parts[:7]
            unit = "ns" if "ns" in metric else "tok/s" if "tok" in metric else ""
            val_str = f"{value} {unit}" if unit else value
            print(f"  {ts:<18} {kernel:<18} {backend:<8} {val_str:>14} {status:<8} {desc}")


def _load_staged_entry(action: str, name: str | None) -> tuple[Path, dict, list[Path]]:
    """Resolve and validate a staged entry. Exits on failure."""
    if not name:
        die(f"Usage: run.py staged {action} <name>")
    entry = STAGING_DIR / name
    if not entry.exists() or not (entry / "meta.json").exists():
        die(f"Staged entry not found: {name}")
    meta = json.loads((entry / "meta.json").read_text())
    staged_files = [f for f in entry.iterdir() if f.name != "meta.json"]
    if action != "drop" and not staged_files:
        die("No source file in staged entry.")
    return entry, meta, staged_files


def cmd_staged(args):
    """Manage staged improvements."""
    import argparse
    parser = argparse.ArgumentParser(prog="run.py staged")
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "apply", "diff", "drop", "clean"],
                        help="Action: list (default), apply, diff, drop, clean")
    parser.add_argument("name", nargs="?", help="Staged entry name (for apply/diff/drop)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parsed = parser.parse_args(args)

    if parsed.action == "list":
        if not STAGING_DIR.exists():
            print("No staged improvements.")
            return
        entries = sorted(e for e in STAGING_DIR.iterdir()
                         if e.is_dir() and (e / "meta.json").exists())
        if not entries:
            print("No staged improvements.")
            return
        print(f"\nStaged improvements ({len(entries)}):\n")
        print(f"  {'Name':<50} {'Result':>14} {'Backend':<8} Timestamp")
        print(f"  {'-' * 90}")
        for entry in entries:
            meta = json.loads((entry / "meta.json").read_text())
            metric_str = "?"
            if isinstance(meta.get("metric_value"), (int, float)):
                mname = meta.get("metric_name", "")
                metric_str = f"{meta['metric_value']:.1f} {mname}"
            print(f"  {entry.name:<50} {metric_str:>14} "
                  f"{meta.get('backend', '?'):<8} {meta.get('timestamp', '?')}")
        print(f"\nApply with: run.py staged apply <name>")
        print(f"Diff with:  run.py staged diff <name>")

    elif parsed.action in ("apply", "diff", "drop"):
        entry, meta, staged_files = _load_staged_entry(parsed.action, parsed.name)

    if parsed.action == "apply":
        source_file = meta["source_file"]
        if not parsed.yes:
            try:
                answer = input(f"Apply {parsed.name} -> {source_file}? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if answer not in ("y", "yes"):
                print("Aborted.")
                return
        staged_source = staged_files[0].read_text()
        dest = AGAVE_ROOT / source_file
        dest.write_text(staged_source)
        print(f"Applied {parsed.name} -> {source_file}")

    elif parsed.action == "diff":
        source_file = meta["source_file"]
        current = AGAVE_ROOT / source_file
        result = subprocess.run(
            ["diff", "-u", str(current), str(staged_files[0])],
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        else:
            print("No differences (staged version matches current source).")

    elif parsed.action == "drop":
        import shutil
        shutil.rmtree(entry)
        print(f"Removed {parsed.name}")

    elif parsed.action == "clean":
        if not STAGING_DIR.exists():
            print("Nothing to clean.")
            return
        entries = [e for e in STAGING_DIR.iterdir() if e.is_dir()]
        if not entries:
            print("Nothing to clean.")
            return
        count = len(entries)
        try:
            answer = input(f"Remove all {count} staged entries? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer in ("y", "yes"):
            import shutil
            for entry in entries:
                shutil.rmtree(entry)
            print(f"Removed {count} staged entries.")


# Entry point — called from run.py
def main(command: str, args: list[str]):
    if command == "bench":
        cmd_bench(args)
    elif command == "tune":
        cmd_tune(args)
    elif command == "grid":
        cmd_grid(args)
    elif command == "auto":
        cmd_auto(args)
    elif command == "staged":
        cmd_staged(args)
    elif command == "status":
        cmd_status(args)
    else:
        die(f"Unknown autotune command: {command}")
