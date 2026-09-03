# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lightweight docs hygiene checks for Agave.

Validates relative links, mermaid vs diagram asset counts, backend kernel
count claims against source constants, product SemVer / Zig version
alignment across build.zig.zon, CHANGELOG, API docs, and .zigversion,
and Docker image packaging (Debian pin, OCI license, LICENSE shipment).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
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

    # Leftover "until the next tagged product release bumps `0.1.0`" after a 0.2.0 cut.
    for bump in re.findall(r"bumps `([0-9]+\.[0-9]+\.[0-9]+)`", changelog):
        if bump != product:
            errors.append(
                f"CHANGELOG.md: 'bumps `{bump}`' is stale (product version is {product})"
            )

    return errors


_BACKEND_ENABLE = frozenset(
    {
        "enable-cpu",
        "enable-metal",
        "enable-cuda",
        "enable-rocm",
        "enable-vulkan",
        "enable-webgpu",
        "enable-debug",
        "enable-bench",
    }
)


def check_cli_flags_in_readme() -> list[str]:
    """Every cli_specs long flag must appear in the README CLI Options block."""
    main = (ROOT / "src" / "main.zig").read_text(encoding="utf-8", errors="replace")
    specs = re.search(
        r"const cli_specs = \[_\]cli_mod\.ArgSpec\{(.*?)^};",
        main,
        re.S | re.M,
    )
    if not specs:
        return ["src/main.zig: could not find cli_specs array"]
    flags = re.findall(r'\.long = "([^"]+)"', specs.group(1))
    if not flags:
        return ["src/main.zig: cli_specs has no .long flags"]

    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    cli_block = re.search(r"## CLI Options\n\n```(?:[a-z]*)\n(.*?)```", readme, re.S)
    if not cli_block:
        return ["README.md: missing ## CLI Options fenced block"]
    block = cli_block.group(1)
    errors: list[str] = []
    for flag in flags:
        if f"--{flag}" not in block:
            errors.append(
                f"README.md CLI Options: missing --{flag} (declared in src/main.zig cli_specs)"
            )
    return errors


def check_model_enable_flags() -> list[str]:
    """Model -Denable-* flags must be documented and passable through Docker."""
    build = (ROOT / "build.zig").read_text(encoding="utf-8", errors="replace")
    all_enable = re.findall(r'b\.option\(bool, "(enable-[^"]+)"', build)
    models = [name for name in all_enable if name not in _BACKEND_ENABLE]
    if not models:
        return ["build.zig: no model enable-* options found"]

    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    for name in models:
        if f"`{name}`" not in readme:
            errors.append(f"README.md: missing build option `{name}`")
        zig_flag = f"-D{name}="
        if zig_flag not in dockerfile:
            errors.append(
                f"Dockerfile: missing {zig_flag} (model defaults on; image/compose cannot disable it)"
            )
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8", errors="replace")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8", errors="replace"
    )
    # Local Compose / CI docker-build / README "minimal" claim "CPU + Gemma3
    # only"; any new model ENABLE_* must be turned off there or the image
    # silently compiles it in (Dockerfile ARG defaults are true).
    skip_compose = {
        "ENABLE_CPU",
        "ENABLE_METAL",
        "ENABLE_VULKAN",
        "ENABLE_CUDA",
        "ENABLE_ROCM",
        "ENABLE_WEBGPU",
        "ENABLE_DEBUG",
        "ENABLE_BENCH",
        "ENABLE_GEMMA3",
    }
    for arg in re.findall(r"^ARG (ENABLE_[A-Z0-9_]+)=", dockerfile, re.M):
        if arg in skip_compose:
            continue
        if f"{arg}:" not in compose:
            errors.append(
                f"docker-compose.yml: missing {arg} (Dockerfile ARG; "
                "Gemma3-only image would compile it in at the Zig default)"
            )
        if f"{arg}=false" not in ci:
            errors.append(
                f".github/workflows/ci.yml: missing {arg}=false "
                "(docker-build is 'single model'; Dockerfile ARG defaults on)"
            )
        if f"{arg}=false" not in readme:
            errors.append(
                f"README.md: missing --build-arg {arg}=false "
                "(Minimal build: single model + CPU only)"
            )
    return errors


def check_debian_snapshot_pin() -> list[str]:
    """FROM bookworm-YYYYMMDD-slim, DEBIAN_SNAPSHOT, and SOURCE_DATE_EPOCH share a day."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8", errors="replace")
    days = re.findall(r"debian:bookworm-(\d{8})-slim", text)
    snaps = re.findall(r"DEBIAN_SNAPSHOT=(\d{8})T", text)
    epoch_m = re.search(r"^ARG SOURCE_DATE_EPOCH=(\d+)", text, re.M)
    if not days:
        return ["Dockerfile: missing debian:bookworm-YYYYMMDD-slim FROM tag"]
    if not snaps:
        return ["Dockerfile: missing DEBIAN_SNAPSHOT=YYYYMMDDT... ARG"]
    if not epoch_m:
        return ["Dockerfile: missing ARG SOURCE_DATE_EPOCH"]
    from_day = days[0]
    errors: list[str] = []
    if any(d != from_day for d in days) or any(s != from_day for s in snaps):
        errors.append(
            f"Dockerfile: debian FROM days {days} and DEBIAN_SNAPSHOT days {snaps} "
            f"must all equal {from_day}"
        )
    expected = int(
        datetime.strptime(from_day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    )
    got = int(epoch_m.group(1))
    if got != expected:
        errors.append(
            f"Dockerfile: SOURCE_DATE_EPOCH {got} != midnight UTC of {from_day} ({expected})"
        )
    epochs = re.findall(r"SOURCE_DATE_EPOCH=(\d+)", text)
    if epochs and any(e != str(got) for e in epochs):
        errors.append(
            f"Dockerfile: SOURCE_DATE_EPOCH values disagree: {epochs} (expected {got})"
        )
    return errors


def check_docker_packaging() -> list[str]:
    """Debian snapshot pin, OCI license, LICENSE shipment, and HOME in the image."""
    errors: list[str] = []
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8", errors="replace")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8", errors="replace")

    from_days = re.findall(r"debian:bookworm-(\d{8})-slim", dockerfile)
    snap_days = re.findall(r"DEBIAN_SNAPSHOT=(\d{8})T", dockerfile)
    if not from_days:
        errors.append("Dockerfile: missing debian:bookworm-YYYYMMDD-slim FROM tag")
    if not snap_days:
        errors.append("Dockerfile: missing DEBIAN_SNAPSHOT=YYYYMMDDT... pin")
    if from_days and len(set(from_days)) != 1:
        errors.append(f"Dockerfile: FROM tag days disagree: {from_days}")
    if snap_days and len(set(snap_days)) != 1:
        errors.append(f"Dockerfile: DEBIAN_SNAPSHOT days disagree: {snap_days}")
    if from_days and snap_days and from_days[0] != snap_days[0]:
        errors.append(
            f"Dockerfile: debian FROM day {from_days[0]} != DEBIAN_SNAPSHOT day {snap_days[0]}"
        )
    elif from_days:
        day = from_days[0]
        expected_epoch = int(
            datetime(
                int(day[:4]), int(day[4:6]), int(day[6:8]), tzinfo=timezone.utc
            ).timestamp()
        )
        epoch_m = re.search(r"SOURCE_DATE_EPOCH=(\d+)", dockerfile)
        if not epoch_m:
            errors.append("Dockerfile: missing SOURCE_DATE_EPOCH")
        elif int(epoch_m.group(1)) != expected_epoch:
            errors.append(
                f"Dockerfile: SOURCE_DATE_EPOCH={epoch_m.group(1)} != {expected_epoch} "
                f"(00:00:00 UTC on FROM/snapshot day {day})"
            )

    if 'org.opencontainers.image.licenses="GPL-3.0-or-later"' not in dockerfile:
        errors.append(
            'Dockerfile: OCI licenses label must be "GPL-3.0-or-later" '
            "(LICENSE is GPLv3 or later)"
        )
    if "LICENSE /usr/share/doc/agave/copyright" not in dockerfile:
        errors.append(
            "Dockerfile: must COPY LICENSE to /usr/share/doc/agave/copyright"
        )
    if "HOME=/home/agave" not in dockerfile:
        errors.append(
            "Dockerfile: must set ENV HOME=/home/agave for non-Docker OCI runtimes"
        )
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8", errors="replace")
    if "HOME: /home/agave" not in compose:
        errors.append(
            "docker-compose.yml: must set HOME: /home/agave (same as the image ENV)"
        )

    for i, raw in enumerate(dockerignore.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line in {"LICENSE", "/LICENSE", "**/LICENSE"}:
            errors.append(
                f".dockerignore:{i}: excludes LICENSE "
                "(runtime image must ship the GPL notice)"
            )

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "(at your option) any later version" not in license_text:
        errors.append(
            "LICENSE: expected GPL-3.0-or-later wording "
            "('(at your option) any later version')"
        )

    return errors


def check_bun_pin() -> list[str]:
    """package.json packageManager, engines.bun, and CI lint-web must agree."""
    errors: list[str] = []
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    pm = pkg.get("packageManager", "")
    m = re.fullmatch(r"bun@([0-9]+\.[0-9]+\.[0-9]+)", str(pm))
    if not m:
        return ['package.json: packageManager must be bun@X.Y.Z']
    ver = m.group(1)
    engines = (pkg.get("engines") or {}).get("bun")
    if engines != ver:
        errors.append(
            f"package.json: engines.bun ({engines!r}) must equal packageManager bun@{ver}"
        )
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8", errors="replace"
    )
    if f'bun-version: "{ver}"' not in ci:
        errors.append(
            f'.github/workflows/ci.yml: bun-version must be "{ver}" '
            "(package.json packageManager)"
        )
    return errors


def check_cuda_sm_default() -> list[str]:
    """Bare `zig build ptx` must match CI kernel-artifacts (committed PTX SM)."""
    build = (ROOT / "build.zig").read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'b\.option\(CudaSm, "cuda-sm", "CUDA SM target \(default: (sm_\d+)\)"\) orelse \.(sm_\d+)',
        build,
    )
    if not m:
        return ['build.zig: could not parse cuda-sm default']
    if m.group(1) != m.group(2):
        return [
            f"build.zig: cuda-sm option text default {m.group(1)} != orelse .{m.group(2)}"
        ]
    default = m.group(1)
    errors: list[str] = []
    script = (ROOT / "scripts" / "check-shader-artifacts.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    if f"zig build ptx -Dcuda-sm={default}" not in script:
        errors.append(
            f"scripts/check-shader-artifacts.sh: PTX rebuild must use -Dcuda-sm={default} "
            "(same as build.zig default so `zig build ptx` matches CI)"
        )
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    if not re.search(
        rf"`cuda-sm`\s*\|\s*enum\s*\|\s*{re.escape(default)}\s*\|",
        readme,
    ):
        errors.append(f"README.md: cuda-sm default column must be {default}")
    if '@embedFile(".zigversion")' not in build:
        errors.append(
            "build.zig: must embed .zigversion and refuse a mismatched compiler"
        )
    return errors


def check_ci_runner_pins() -> list[str]:
    """GitHub Actions must not use floating *-latest runner tags."""
    errors: list[str] = []
    workflows = ROOT / ".github" / "workflows"
    if not workflows.is_dir():
        return [".github/workflows: missing"]
    for path in sorted(workflows.glob("*.yml")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.split("#", 1)[0]
            if re.search(r"\b(ubuntu|macos|windows)-latest\b", stripped):
                errors.append(f"{rel}:{i}: floating runner tag (pin ubuntu-24.04 / macos-15)")
    return errors


def main() -> int:
    if sys.version_info < (3, 11):
        sys.exit(
            f"check-docs: Python {sys.version.split()[0]} is too old; need 3.11+ "
            "(PEP 723 requires-python in this file)"
        )
    errors: list[str] = []
    errors.extend(check_links())
    errors.extend(check_diagram_counts())
    errors.extend(check_kernel_constants())
    errors.extend(check_version_consistency())
    errors.extend(check_cli_flags_in_readme())
    errors.extend(check_model_enable_flags())
    errors.extend(check_debian_snapshot_pin())
    errors.extend(check_docker_packaging())
    errors.extend(check_ci_runner_pins())
    errors.extend(check_bun_pin())
    errors.extend(check_cuda_sm_default())
    if errors:
        print(f"check-docs: {len(errors)} issue(s)")
        for e in errors:
            print(f"  {e}")
        return 1
    print("check-docs: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
