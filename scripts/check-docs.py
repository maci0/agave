# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lightweight docs hygiene checks for Agave.

Validates relative links, `path:line` claims in the security docs, mermaid vs
diagram asset counts, backend kernel count claims against source constants,
product SemVer / Zig version alignment across build.zig.zon, CHANGELOG, API
docs, README, SECURITY, and .zigversion, and Docker image packaging (Debian
pin, OCI license, LICENSE shipment).
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

# `zig build check` runs this file with a bare `python3`, so the
# `requires-python` metadata above and ruff.toml's `target-version` are enforced
# by nothing on that path. The 3.11-only `datetime.UTC` below reports as an
# ImportError from the middle of the import block, naming neither this script
# nor the version it needs, so the requirement is stated here instead. UP036
# reads the guard as dead code under that target and is the point of it.
if sys.version_info < (3, 11):  # noqa: UP036
    sys.exit(
        f"check-docs: needs Python 3.11+ (running {sys.version.split()[0]}). "
        "Install a newer Python, then rerun `zig build check`."
    )

from datetime import UTC, datetime


def find_root() -> Path:
    """Walk up from this file to the directory holding build.zig.zon."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "build.zig.zon").is_file():
            return candidate
    sys.exit("check-docs: no build.zig.zon found above this script")


ROOT = find_root()


def check_links() -> list[str]:
    errors: list[str] = []
    md_files = [*(ROOT / "docs").rglob("*.md"), ROOT / "README.md"]
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


LINE_REF_DOCS = ("docs/THREAT_MODEL.md", "SECURITY.md")
LINE_REF_RE = re.compile(
    r"(?:\.\./)?((?:src|web|scripts|docs|tests|tools|research)/[A-Za-z0-9_./-]+\.(?:zig|ts|js|py|sh|md|html)"
    r"|Dockerfile|docker-compose\.yml|build\.zig\.zon):(\d+)(?:-(\d+))?"
)


def check_doc_line_refs() -> list[str]:
    """Every `path:line` claim in the security docs must land inside the file.

    Catches the gross drift mode: a renamed file, a deleted file, or a section
    that shrank past a cited line. Line shifts within a surviving file are not
    detectable here and need a re-anchoring pass.
    """
    errors: list[str] = []
    for rel in LINE_REF_DOCS:
        doc = ROOT / rel
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        for m in LINE_REF_RE.finditer(text):
            ref = m.group(1)
            target = (ROOT / ref).resolve()
            line = text[: m.start()].count("\n") + 1
            if not target.is_file():
                errors.append(f"{rel}:{line}: line ref to missing file -> {ref}")
                continue
            count = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            for num in filter(None, (m.group(2), m.group(3))):
                if int(num) > count:
                    errors.append(f"{rel}:{line}: line ref past EOF -> {ref}:{num} (file has {count} lines)")
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
            errors.append(f"docs/KERNELS.md: {name} count {dm.group(1)} != code {code_n} in {fname}")
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
            errors.append(f".zigversion: {file_zig!r} != build.zig.zon minimum_zig_version {min_zig!r}")
    else:
        errors.append(".zigversion: missing (must match build.zig.zon .minimum_zig_version)")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version is **{product}**" not in changelog:
        errors.append(f"CHANGELOG.md: must state Product version is **{product}** (match build.zig.zon .version)")
    if "## [Unreleased]" not in changelog:
        errors.append("CHANGELOG.md: missing ## [Unreleased] section")

    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version **{product}**" not in api:
        errors.append(f"docs/API.md: must state Product version **{product}** (match build.zig.zon .version)")
    if f'"agave-v{product}"' not in api and f"agave-v{product}" not in api:
        errors.append(f"docs/API.md: system_fingerprint examples should use agave-v{product}")

    contrib = (ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8", errors="replace")
    if f"Product version: **{product}**" not in contrib:
        errors.append(f"docs/CONTRIBUTING.md: must state Product version: **{product}** (match build.zig.zon .version)")

    # Reader-facing pages that restate the product version. Nothing in the build
    # or the binary reads them, so they drift silently at the next release bump.
    secondary = [
        ("README.md", f"product version `{product}`"),
        ("SECURITY.md", f"Product version is **{product}**"),
        ("docs/DOCUMENTATION.md", f"product version {product}"),
    ]
    for rel, needle in secondary:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"{rel}: missing (states the product version)")
        elif needle not in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(f"{rel}: must state {needle!r} (match build.zig.zon .version)")

    # Leftover "until the next tagged product release bumps `0.1.0`" after a 0.2.0 cut.
    for bump in re.findall(r"bumps `([0-9]+\.[0-9]+\.[0-9]+)`", changelog):
        if bump != product:
            errors.append(f"CHANGELOG.md: 'bumps `{bump}`' is stale (product version is {product})")

    errors.extend(_check_release_tags(changelog, product))
    errors.extend(_check_release_section_matches_manifest(changelog, product))
    errors.extend(_check_bump_matches_breaking(changelog))
    return errors


def _released_sections(changelog: str) -> list[tuple[str, str]]:
    """(version, body) for every dated release section, oldest first.

    Sorted by parsed version rather than file position: a section appended out
    of order would otherwise compare against the wrong predecessor, and a hand
    reordered changelog must not decide whether a break is reported.
    """
    pat = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.M)
    matches = list(pat.finditer(changelog))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(changelog)
        version = m.group(1)
        out.append((version, changelog[m.end() : end]))
    out.sort(key=lambda pair: [int(p) for p in pair[0].split(".")])
    return out


def _check_release_section_matches_manifest(changelog: str, product: str) -> list[str]:
    """`build.zig.zon` `.version` must name a section the changelog actually has.

    Every other guard compares a string against the manifest, and a link
    definition against a heading, so a version with no section at all slips
    between them: bump `.version` to 0.11.0, repoint `[unreleased]` at
    `v0.11.0...HEAD`, and every check passes with the notes still sitting under
    `## [Unreleased]`. The release then ships a version whose entry describes no
    release, and a `### Breaking` heading under `[Unreleased]` never reaches
    `_check_bump_matches_breaking`, which reads only dated sections.
    """
    errors: list[str] = []
    sections = {version for version, _ in _released_sections(changelog)}
    if product not in sections:
        errors.append(
            f"CHANGELOG.md: build.zig.zon .version is {product} but no `## [{product}] - YYYY-MM-DD` "
            f"section exists; move the Unreleased notes into one in the same commit that bumps the version"
        )
    product_parts = [int(p) for p in product.split(".")]
    ahead = sorted(v for v in sections if [int(p) for p in v.split(".")] > product_parts)
    if ahead:
        errors.append(
            f"CHANGELOG.md: {', '.join('[' + v + ']' for v in ahead)} is dated above the product version "
            f"{product}; those notes belong under [Unreleased] until the version is cut"
        )
    return errors


def _check_bump_matches_breaking(changelog: str) -> list[str]:
    """A `### Breaking` entry may not ride a patch release.

    The 0.x rule in docs/CONTRIBUTING.md lets a breaking change land without a
    major digit but still requires at least a minor bump, so a consumer who
    tracks `-Dversion_patch` upgrades without reading notes keeps the breaking
    one. This is the only check that ties a section's content to its bump;
    every other guard compares strings against the manifest.
    """
    errors: list[str] = []
    sections = _released_sections(changelog)
    prev: str | None = None
    for version, body in sections:
        if prev is not None and re.search(r"^### Breaking", body, re.M):
            prev_parts = prev.split(".")
            parts = version.split(".")
            is_patch = prev_parts[:2] == parts[:2]
            if is_patch:
                errors.append(
                    f"CHANGELOG.md: [{version}] has a Breaking entry but bumps only "
                    f"the patch digit from [{prev}]; cut it as a minor release "
                    f"({prev_parts[0]}.{int(prev_parts[1]) + 1}.0) or drop the Breaking heading"
                )
        prev = version
    return errors


def _tags_at_head() -> list[str] | None:
    """Tags pointing at HEAD, or None when git is unavailable or HEAD is untagged."""
    if not (ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [t for t in out.stdout.split() if t]


def _check_release_tags(changelog: str, product: str) -> list[str]:
    """A release must be taggable and already-tagged commits must match the manifest.

    Covers the two ways a release goes out inconsistent: shipping a tag whose
    name disagrees with `build.zig.zon` `.version`, and cutting a version with a
    changelog section no link definition resolves.
    """
    errors: list[str] = []

    for version in re.findall(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - \d{4}-\d{2}-\d{2}$", changelog, re.M):
        if f"\n[{version}]:" not in changelog:
            errors.append(f"CHANGELOG.md: released section [{version}] has no [{version}]: link definition")

    # The Unreleased section diffs against the last cut, so its base tag has to
    # be the current product version.
    m = re.search(r"^\[unreleased\]: (\S+)$", changelog, re.M)
    if not m:
        errors.append("CHANGELOG.md: missing [unreleased] link definition")
    elif f"v{product}" not in m.group(1):
        errors.append(
            f"CHANGELOG.md: [unreleased] compares against {m.group(1)}, "
            f"which is not tag v{product} (match build.zig.zon .version)"
        )

    tags = _tags_at_head()
    if tags:
        expected = f"v{product}"
        for tag in tags:
            if tag != expected and re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+.*", tag):
                errors.append(
                    f"HEAD is tagged {tag} but build.zig.zon .version is {product} (a release tag must be v{expected})"
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
            errors.append(f"README.md CLI Options: missing --{flag} (declared in src/main.zig cli_specs)")
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
            errors.append(f"Dockerfile: missing {zig_flag} (model defaults on; image/compose cannot disable it)")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8", errors="replace")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8", errors="replace")
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
            errors.append(f"README.md: missing --build-arg {arg}=false (Minimal build: single model + CPU only)")
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
        errors.append(f"Dockerfile: debian FROM days {days} and DEBIAN_SNAPSHOT days {snaps} must all equal {from_day}")
    expected = int(datetime.strptime(from_day, "%Y%m%d").replace(tzinfo=UTC).timestamp())
    got = int(epoch_m.group(1))
    if got != expected:
        errors.append(f"Dockerfile: SOURCE_DATE_EPOCH {got} != midnight UTC of {from_day} ({expected})")
    epochs = re.findall(r"SOURCE_DATE_EPOCH=(\d+)", text)
    if epochs and any(e != str(got) for e in epochs):
        errors.append(f"Dockerfile: SOURCE_DATE_EPOCH values disagree: {epochs} (expected {got})")
    return errors


def check_docker_packaging() -> list[str]:
    """OCI license, LICENSE shipment, and HOME in the image.

    The Debian snapshot pin is checked by check_debian_snapshot_pin().
    """
    errors: list[str] = []
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8", errors="replace")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8", errors="replace")

    if 'org.opencontainers.image.licenses="GPL-3.0-or-later"' not in dockerfile:
        errors.append('Dockerfile: OCI licenses label must be "GPL-3.0-or-later" (LICENSE is GPLv3 or later)')
    if "LICENSE /usr/share/doc/agave/copyright" not in dockerfile:
        errors.append("Dockerfile: must COPY LICENSE to /usr/share/doc/agave/copyright")
    if "HOME=/home/agave" not in dockerfile:
        errors.append("Dockerfile: must set ENV HOME=/home/agave for non-Docker OCI runtimes")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8", errors="replace")
    if "HOME: /home/agave" not in compose:
        errors.append("docker-compose.yml: must set HOME: /home/agave (same as the image ENV)")

    for i, raw in enumerate(dockerignore.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line in {"LICENSE", "/LICENSE", "**/LICENSE"}:
            errors.append(f".dockerignore:{i}: excludes LICENSE (runtime image must ship the GPL notice)")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "(at your option) any later version" not in license_text:
        errors.append("LICENSE: expected GPL-3.0-or-later wording ('(at your option) any later version')")

    return errors


def check_bun_pin() -> list[str]:
    """package.json packageManager, engines.bun, and CI lint-web must agree."""
    errors: list[str] = []
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    pm = pkg.get("packageManager", "")
    m = re.fullmatch(r"bun@([0-9]+\.[0-9]+\.[0-9]+)", str(pm))
    if not m:
        return ["package.json: packageManager must be bun@X.Y.Z"]
    ver = m.group(1)
    engines = (pkg.get("engines") or {}).get("bun")
    if engines != ver:
        errors.append(f"package.json: engines.bun ({engines!r}) must equal packageManager bun@{ver}")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8", errors="replace")
    if f'bun-version: "{ver}"' not in ci:
        errors.append(f'.github/workflows/ci.yml: bun-version must be "{ver}" (package.json packageManager)')
    return errors


def check_cuda_sm_default() -> list[str]:
    """Bare `zig build ptx` must match CI kernel-artifacts (committed PTX SM)."""
    build = (ROOT / "build.zig").read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'b\.option\(CudaSm, "cuda-sm", "CUDA SM target \(default: (sm_\d+)\)"\) orelse \.(sm_\d+)',
        build,
    )
    if not m:
        return ["build.zig: could not parse cuda-sm default"]
    if m.group(1) != m.group(2):
        return [f"build.zig: cuda-sm option text default {m.group(1)} != orelse .{m.group(2)}"]
    default = m.group(1)
    errors: list[str] = []
    script = (ROOT / "scripts" / "check-shader-artifacts.sh").read_text(encoding="utf-8", errors="replace")
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
        errors.append("build.zig: must embed .zigversion and refuse a mismatched compiler")
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


# Repo-relative inputs the docs-check workflow must trigger on: everything
# below is read by a check in this file, so a path filter that omits one means
# those checks never run on the PR that changed the file.
DOCS_CHECK_PATH_INPUTS = (
    "README.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "AGENTS.md",
    "LICENSE",
    "build.zig",
    "build.zig.zon",
    ".zigversion",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "package.json",
    "src/main.zig",
    "src/backend/metal.zig",
    "src/backend/cuda.zig",
    "src/backend/rocm.zig",
    "src/backend/vulkan.zig",
    "scripts/check-docs.py",
    "scripts/test_check_docs.py",
    "scripts/check-third-party-notices.py",
    "scripts/test_check_third_party_notices.py",
    "scripts/brand-glyphs.py",
    "THIRD_PARTY_NOTICES.md",
    "scripts/check-shader-artifacts.sh",
    ".github/workflows/ci.yml",
)


def check_docs_workflow_paths() -> list[str]:
    """docs-check.yml must trigger on every file this script reads.

    The workflow filters on `paths:`, so a check whose input is missing from
    that list never runs: package.json can change the bun pin or
    scripts/check-shader-artifacts.sh the -Dcuda-sm default, and docs-check
    stays silent on the PR that made the change.
    """
    workflow = ROOT / ".github" / "workflows" / "docs-check.yml"
    if not workflow.is_file():
        return [".github/workflows/docs-check.yml: missing"]
    text = workflow.read_text(encoding="utf-8", errors="replace")
    patterns = re.findall(r"^\s+- '([^']+)'$", text, re.M)
    if not patterns:
        return [".github/workflows/docs-check.yml: no path filters found"]

    def covered(rel: str) -> bool:
        for pat in patterns:
            if pat.endswith("/**"):
                if rel.startswith(pat[:-2]):
                    return True
            elif pat.endswith("/*"):
                # GitHub path filters: `*` matches within one path segment and
                # never crosses `/`, which fnmatch's `*` does. Compare the
                # directory part with the pattern's, and the last segment on
                # its own, so a `*` cannot swallow a `/`.
                dir_pattern = pat[:-2]
                dir_actual, _, name = rel.rpartition("/")
                if (
                    (dir_pattern == dir_actual or fnmatch.fnmatch(dir_actual, dir_pattern))
                    and name
                    and fnmatch.fnmatch(name, "*")
                ):
                    return True
            elif pat == rel:
                return True
        return False

    return [
        f".github/workflows/docs-check.yml: paths filter does not cover {rel} "
        "(this script reads it, so its checks would not run on that PR)"
        for rel in DOCS_CHECK_PATH_INPUTS
        if not covered(rel)
    ]


def main() -> int:
    errors: list[str] = []
    errors.extend(check_links())
    errors.extend(check_doc_line_refs())
    errors.extend(check_diagram_counts())
    errors.extend(check_kernel_constants())
    errors.extend(check_version_consistency())
    errors.extend(check_cli_flags_in_readme())
    errors.extend(check_model_enable_flags())
    errors.extend(check_debian_snapshot_pin())
    errors.extend(check_docker_packaging())
    errors.extend(check_ci_runner_pins())
    errors.extend(check_docs_workflow_paths())
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
