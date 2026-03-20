#!/usr/bin/env python3
"""
Unified CLI for Agave kernel research tooling.

Examples:
    uv run run.py info sdpa                          # Kernel details
    uv run run.py diff                               # What changed?
    uv run run.py golden sdpa rope                   # Generate golden data
    uv run run.py bench gemv_q4_0 -b metal           # Micro-benchmark one kernel
    uv run run.py bench gemv_f32 --dim=2048 --k=2048 # Custom dimensions
    uv run run.py bench --changed                    # Benchmark changed kernels
    uv run run.py bench --e2e --compare              # End-to-end benchmark vs baseline
    uv run run.py tune gemv_q4_0 -b metal -d "vectorized inner loop"
    uv run run.py tune --changed                     # Tune changed kernels
    uv run run.py grid sdpa --param BLOCK --values "32,64,128" \\
        --pattern "const BLOCK: usize = 64;" --template "const BLOCK: usize = {value};"
    uv run run.py auto softmax -b cpu --strategy hill-climb
    uv run run.py staged                             # List staged improvements
    uv run run.py staged apply <name>                # Apply a staged improvement
"""

import sys
import subprocess
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent


def cmd_info(args: list[str]):
    """Show kernel details."""
    from registry import KERNELS, find_kernels, golden_files, groups

    if not args or args[0] in ("-h", "--help"):
        print("Usage: run.py info [kernel|group]")
        print(f"\nGroups: {', '.join(sorted(groups().keys()))}")
        print(f"Kernels: {', '.join(sorted(KERNELS.keys()))}")
        return

    for query in args:
        kernels = find_kernels(query)
        if not kernels:
            print(f"No kernels matching {query!r}")
            continue

        for k in kernels:
            print(f"\n  {k.name}")
            print(f"  {'=' * len(k.name)}")
            print(f"  Group:       {k.group}")
            print(f"  Description: {k.description}")
            print(f"  Reference:   reference.py:{k.reference_fn}()" if k.reference_fn else "  Reference:   (none)")
            print(f"  Bench quants: {', '.join(k.bench_quants) if k.bench_quants else '(no bench model)'}")
            print(f"  Sources:")
            for backend, path in k.sources.items():
                paths = [path] if isinstance(path, str) else path
                for p in paths:
                    print(f"    {backend:<8} {p}")
            gf = golden_files(k.golden_prefix)
            if gf:
                print(f"  Golden files: {len(gf)} files ({k.golden_prefix}_*.bin)")
            else:
                print(f"  Golden files: (none)")


def cmd_diff(args: list[str]):
    """Show which kernels have changed source files."""
    from registry import changed_kernels

    changed = changed_kernels()
    if not changed:
        print("No kernel source files changed (vs HEAD).")
        return

    print(f"Changed kernels ({len(changed)}):\n")
    for k in changed:
        backends = ", ".join(k.sources.keys())
        print(f"  {k.name:<20} [{backends}]")

    print(f"\nRecommended:")
    print(f"  uv run run.py bench --changed")


def cmd_list(args: list[str]):
    """List all kernels grouped."""
    from registry import groups

    for group_name, kerns in sorted(groups().items()):
        print(f"\n  {group_name}:")
        for k in kerns:
            quants = ", ".join(k.bench_quants) if k.bench_quants else "-"
            backends = ", ".join(k.sources.keys())
            print(f"    {k.name:<20} {k.description:<45} [{backends}]")


def cmd_coverage(args: list[str]):
    """Report reference/golden/Zig test coverage per kernel."""
    import ast
    from registry import KERNELS, golden_files, AGAVE_ROOT

    # Scan reference.py for function names
    ref_path = Path(__file__).parent / "reference.py"
    ref_source = ref_path.read_text()
    ref_tree = ast.parse(ref_source)
    ref_fns = {node.name for node in ast.walk(ref_tree)
               if isinstance(node, ast.FunctionDef)}

    # Scan Zig files for @embedFile containing golden/
    zig_golden_kernels: set[str] = set()
    src_dir = AGAVE_ROOT / "src"
    if src_dir.exists():
        for zig_file in src_dir.rglob("*.zig"):
            content = zig_file.read_text()
            for kernel in KERNELS.values():
                if f'golden/{kernel.golden_prefix}_' in content:
                    zig_golden_kernels.add(kernel.name)

    # Report
    print(f"\n  {'Kernel':<22} {'Reference':<12} {'Golden':<10} {'Zig Test':<10}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 10} {'─' * 10}")

    ref_count = golden_count = zig_count = 0
    total = len(KERNELS)

    for name, kernel in sorted(KERNELS.items()):
        has_ref = kernel.reference_fn in ref_fns if kernel.reference_fn else False
        has_golden = len(golden_files(kernel.golden_prefix)) > 0
        has_zig = name in zig_golden_kernels

        ref_count += has_ref
        golden_count += has_golden
        zig_count += has_zig

        print(f"  {name:<22} {'yes' if has_ref else 'no':<12} "
              f"{'yes' if has_golden else 'no':<10} {'yes' if has_zig else 'no':<10}")

    print(f"\n  Coverage: {ref_count}/{total} reference, "
          f"{golden_count}/{total} golden, {zig_count}/{total} zig tests")


def delegate(script: str, args: list[str]):
    """Delegate to a sub-script, forwarding args."""
    cmd = [sys.executable, str(RESEARCH_DIR / script)] + args
    sys.exit(subprocess.run(cmd, cwd=str(RESEARCH_DIR)).returncode)


COMMANDS = {
    "info":     "Show kernel details",
    "list":     "List all kernels",
    "diff":     "Show changed kernels",
    "golden":   "Generate golden test data",
    "bench":    "Run benchmarks (micro by default, --e2e for end-to-end)",
    "tune":     "Single optimization cycle (build + benchmark + log)",
    "grid":     "Grid search over a parameter",
    "auto":     "Autonomous optimization loop (hill-climb or bayesian)",
    "staged":   "Manage staged kernel improvements",
    "coverage": "Report reference/golden/Zig test coverage",
    "status":   "Show optimization history",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        for name, desc in COMMANDS.items():
            print(f"  {name:<10} {desc}")
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "info":
        cmd_info(rest)
    elif cmd == "list":
        cmd_list(rest)
    elif cmd == "diff":
        cmd_diff(rest)
    elif cmd == "golden":
        delegate("generate_golden.py", rest)
    elif cmd in ("bench", "tune", "grid", "auto", "staged", "status"):
        from autotune import main as autotune_main
        autotune_main(cmd, rest)
    elif cmd == "coverage":
        cmd_coverage(rest)
    elif cmd == "optimize":
        print("The 'optimize' command has been replaced:")
        print("  Single cycle:  run.py tune <kernel> ...")
        print("  Grid search:   run.py grid <kernel> ...")
        print("  Autonomous:    run.py auto <kernel> ...")
        sys.exit(1)
    else:
        # Maybe it's a kernel name — show info
        from registry import find_kernels
        if find_kernels(cmd):
            cmd_info([cmd] + rest)
        else:
            print(f"Unknown command: {cmd!r}", file=sys.stderr)
            print("Run with --help for usage.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
