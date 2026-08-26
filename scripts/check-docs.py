# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lightweight docs hygiene checks for Agave.

Validates relative links, mermaid vs diagram asset counts, backend kernel
count claims against source constants, and product SemVer / Zig version
alignment across build.zig.zon, CHANGELOG, API docs, and .zigversion.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def find_root() -> Path:
    """Walk up from this file to the directory holding build.zig.zon."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "build.zig.zon").is_file():
            return candidate
    sys.exit("check-docs: no build.zig.zon found above this script")


ROOT = find_root()


def check_links() -> list[str]:
    errors: list[str] = []
    md_files = list((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in link_re.finditer(text):
            url = m.group(2).split()[0]
            if url.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = url.split("#")[0]
            if not path_part:
                continue
            target = (f.parent / path_part).resolve()
            if not target.exists():
                line = text[: m.start()].count("\n") + 1
                errors.append(f"{f.relative_to(ROOT)}:{line}: broken link -> {url}")
    return errors


def check_diagram_counts() -> list[str]:
    """Warn-only: mermaid in Markdown is the source of truth; PNGs are optional renders."""
    warnings: list[str] = []
    tutorial = ROOT / "docs" / "tutorial"
    diagrams = ROOT / "docs" / "diagrams"
    for md in sorted(tutorial.glob("*.md")):
        if md.name == "README.md":
            continue
        n_m = len(re.findall(r"```mermaid", md.read_text(encoding="utf-8", errors="replace")))
        if n_m == 0:
            continue
        ddir = diagrams / md.stem
        n_png = len(list(ddir.glob("*.png"))) if ddir.exists() else 0
        if n_png and n_png != n_m:
            warnings.append(
                f"{md.relative_to(ROOT)}: mermaid={n_m} png={n_png} (optional: re-run docs/render-diagrams.mjs)"
            )
    for w in warnings:
        print(f"  warn: {w}")
    return []


def check_kernel_constants() -> list[str]:
    errors: list[str] = []
    kernels_md = (ROOT / "docs" / "KERNELS.md").read_text(encoding="utf-8", errors="replace")
    checks = [
        ("metal.zig", r"n_pipelines:\s*u32\s*=\s*(\d+)", "Metal", r"Metal `n_pipelines = (\d+)`"),
        ("cuda.zig", r"n_kernels:\s*u32\s*=\s*(\d+)", "CUDA", r"CUDA `n_kernels = (\d+)`"),
        ("rocm.zig", r"n_kernels:\s*u32\s*=\s*(\d+)", "ROCm", r"ROCm `n_kernels = (\d+)`"),
        ("vulkan.zig", r"n_pipelines:\s*u32\s*=\s*(\d+)", "Vulkan", r"Vulkan `n_pipelines = (\d+)`"),
    ]
    for fname, src_pat, name, doc_pat in checks:
        src = (ROOT / "src" / "backend" / fname).read_text(encoding="utf-8", errors="replace")
        sm = re.search(src_pat, src)
        dm = re.search(doc_pat, kernels_md)
        if not sm:
            errors.append(f"src/backend/{fname}: missing {name} count constant")
            continue
        code_n = sm.group(1)
        if not dm:
            errors.append(f"docs/KERNELS.md: missing documented {name} count (code has {code_n})")
            continue
        if dm.group(1) != code_n:
            errors.append(
                f"docs/KERNELS.md: {name} count {dm.group(1)} != code {code_n} in {fname}"
            )
    return errors


def check_version_consistency() -> list[str]:
    """Keep product SemVer and minimum Zig aligned across release SSOT files."""
    errors: list[str] = []
    zon = (ROOT / "build.zig.zon").read_text(encoding="utf-8", errors="replace")
    ver_m = re.search(r'\.version\s*=\s*"([^"]+)"', zon)
    zig_m = re.search(r'\.minimum_zig_version\s*=\s*"([^"]+)"', zon)
    if not ver_m:
        return ["build.zig.zon: missing .version string"]
    if not zig_m:
        return ["build.zig.zon: missing .minimum_zig_version string"]
    product = ver_m.group(1)
    min_zig = zig_m.group(1)

    zigversion_path = ROOT / ".zigversion"
    if zigversion_path.exists():
        file_zig = zigversion_path.read_text(encoding="utf-8", errors="replace").strip()
        if file_zig != min_zig:
            errors.append(
                f".zigversion: {file_zig!r} != build.zig.zon minimum_zig_version {min_zig!r}"
            )
    else:
        errors.append(".zigversion: missing (must match build.zig.zon .minimum_zig_version)")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version is **{product}**" not in changelog:
        errors.append(
            f"CHANGELOG.md: must state Product version is **{product}** "
            "(match build.zig.zon .version)"
        )
    if "## [Unreleased]" not in changelog:
        errors.append("CHANGELOG.md: missing ## [Unreleased] section")

    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version **{product}**" not in api:
        errors.append(
            f"docs/API.md: must state Product version **{product}** "
            "(match build.zig.zon .version)"
        )
    if f'"agave-v{product}"' not in api and f"agave-v{product}" not in api:
        errors.append(
            f"docs/API.md: system_fingerprint examples should use agave-v{product}"
        )

    contrib = (ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version: **{product}**" not in contrib:
        errors.append(
            f"docs/CONTRIBUTING.md: must state Product version: **{product}** "
            "(match build.zig.zon .version)"
        )

    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_links())
    errors.extend(check_diagram_counts())
    errors.extend(check_kernel_constants())
    errors.extend(check_version_consistency())
    if errors:
        print(f"check-docs: {len(errors)} issue(s)")
        for e in errors:
            print(f"  {e}")
        return 1
    print("check-docs: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
