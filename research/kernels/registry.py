"""
Kernel registry — maps kernel names to source files, golden data, and benchmark config.

This is the central piece connecting the split kernel files in src/backend/kernels/
to the research tooling. Each entry knows:
  - Which source files implement the kernel (per backend)
  - Which golden test files verify correctness
  - Which model quant formats exercise the kernel during benchmarks
"""

from dataclasses import dataclass
from pathlib import Path

AGAVE_ROOT = Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class Kernel:
    name: str
    group: str
    description: str
    sources: dict[str, str | list[str]]  # backend → relative path(s)
    golden_prefix: str  # prefix in golden/ dir (e.g., "silu" matches silu_*.bin)
    bench_quants: tuple[str, ...] = ("q4_0", "q8_0", "bf16")
    reference_fn: str = ""  # function name in reference.py


# ── Kernel definitions ────────────────────────────────────────────

KERNELS: dict[str, Kernel] = {}


def _reg(k: Kernel):
    KERNELS[k.name] = k


# ── Elementwise ───────────────────────────────────────────────────

_reg(Kernel(
    name="silu",
    group="elementwise",
    description="SiLU activation: x * sigmoid(x)",
    sources={
        "metal": "src/backend/kernels/metal/elementwise.metal",
        "vulkan": "src/backend/kernels/vulkan/silu.comp",
        "cuda": "src/backend/kernels/cuda/silu.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="silu",
    reference_fn="silu",
))

_reg(Kernel(
    name="gelu",
    group="elementwise",
    description="GELU activation (tanh approximation)",
    sources={
        "metal": "src/backend/kernels/metal/elementwise.metal",
        "vulkan": "src/backend/kernels/vulkan/gelu.comp",
        "cuda": "src/backend/kernels/cuda/gelu.zig",
        "cpu": "src/backend/cpu.zig",
        "shared": "src/ops/math.zig",
    },
    golden_prefix="gelu",
    reference_fn="gelu",
))

_reg(Kernel(
    name="add",
    group="elementwise",
    description="Vector add: out = a + b",
    sources={
        "metal": "src/backend/kernels/metal/elementwise.metal",
        "vulkan": "src/backend/kernels/vulkan/add.comp",
        "cuda": "src/backend/kernels/cuda/add.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="add",
    reference_fn="add",
))

_reg(Kernel(
    name="mul",
    group="elementwise",
    description="Vector multiply: out = a * b",
    sources={
        "metal": "src/backend/kernels/metal/elementwise.metal",
        "vulkan": "src/backend/kernels/vulkan/mul.comp",
        "cuda": "src/backend/kernels/cuda/mul.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="mul",
    reference_fn="mul",
))

# ── Math ops ──────────────────────────────────────────────────────

_reg(Kernel(
    name="sigmoid",
    group="math",
    description="Sigmoid activation: 1 / (1 + exp(-x))",
    sources={"shared": "src/ops/math.zig"},
    golden_prefix="sigmoid",
    reference_fn="sigmoid",
))

_reg(Kernel(
    name="softplus",
    group="math",
    description="Softplus: log(1 + exp(x))",
    sources={"shared": "src/ops/math.zig"},
    golden_prefix="softplus",
    reference_fn="softplus",
))

# ── Normalization ─────────────────────────────────────────────────

_reg(Kernel(
    name="rms_norm",
    group="norm",
    description="RMS normalization: x / sqrt(mean(x^2) + eps) * weight",
    sources={
        "metal": "src/backend/kernels/metal/norm.metal",
        "vulkan": "src/backend/kernels/vulkan/rms_norm.comp",
        "cuda": "src/backend/kernels/cuda/rms_norm.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="rms_norm",
    reference_fn="rms_norm",
))

_reg(Kernel(
    name="softmax",
    group="norm",
    description="Softmax: exp(x - max) / sum(exp(x - max))",
    sources={
        "metal": "src/backend/kernels/metal/norm.metal",
        "vulkan": "src/backend/kernels/vulkan/softmax.comp",
        "cuda": "src/backend/kernels/cuda/softmax.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="softmax",
    reference_fn="softmax",
))

_reg(Kernel(
    name="l2_norm",
    group="norm",
    description="L2 normalization: x / sqrt(sum(x^2) + eps)",
    sources={
        "metal": "src/backend/kernels/metal/norm.metal",
        "vulkan": "src/backend/kernels/vulkan/l2_norm.comp",
        "cuda": "src/backend/kernels/cuda/l2_norm.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="l2_norm",
    reference_fn="l2_norm",
))

# ── RoPE ──────────────────────────────────────────────────────────

_reg(Kernel(
    name="rope",
    group="rope",
    description="Rotary Position Embedding applied to Q and K",
    sources={
        "metal": "src/backend/kernels/metal/rope.metal",
        "vulkan": "src/backend/kernels/vulkan/rope.comp",
        "cuda": "src/backend/kernels/cuda/rope.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="rope",
    reference_fn="rope",
))

# ── Embedding ────────────────────────────────────────────────────

_reg(Kernel(
    name="embedding",
    group="embedding",
    description="Embedding table lookup (+ Gemma sqrt scaling)",
    sources={
        "cpu": "src/backend/cpu.zig",
        "metal": "src/backend/metal.zig",
    },
    golden_prefix="embedding",
    bench_quants=(),
    reference_fn="embedding_lookup",
))

# ── SDPA ──────────────────────────────────────────────────────────

_reg(Kernel(
    name="sdpa",
    group="sdpa",
    description="Scaled dot-product attention (fused kernel)",
    sources={
        "metal": "src/backend/kernels/metal/sdpa.metal",
        "vulkan": "src/backend/kernels/vulkan/sdpa.comp",
        "cpu": "src/backend/cpu.zig",
        "shared": "src/ops/attention.zig",
    },
    golden_prefix="sdpa",
    reference_fn="sdpa",
))

_reg(Kernel(
    name="paged_sdpa",
    group="sdpa",
    description="Paged SDPA with block-table indirection into KV cache",
    sources={"shared": "src/ops/attention.zig"},
    golden_prefix="paged_sdpa",
    bench_quants=(),
    reference_fn="paged_sdpa",
))

# ── GEMV (one entry per quant format) ─────────────────────────────

_reg(Kernel(
    name="gemv_f32",
    group="gemv",
    description="GEMV: y = W @ x (f32 weights)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_f32.comp",
        "cuda": "src/backend/kernels/cuda/gemv_f32.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=("bf16",),  # f32 weights only in bf16 model
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_q8_0",
    group="gemv",
    description="GEMV with Q8_0 dequantization (32 vals/block, f16 scale + 32 int8)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_q8_0.comp",
        "cuda": "src/backend/kernels/cuda/gemv_q8_0.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=("q8_0",),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_q4_0",
    group="gemv",
    description="GEMV with Q4_0 dequantization (32 vals/block, nibble-packed)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_q4_0.comp",
        "cuda": "src/backend/kernels/cuda/gemv_q4_0.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=("q4_0",),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_q4_k",
    group="gemv",
    description="GEMV with Q4_K dequantization (256 vals/super-block, 144 bytes)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_q4_k.comp",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=("q4_0",),  # Q4_K uses same bench model
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_q5_k",
    group="gemv",
    description="GEMV with Q5_K dequantization (256 vals/super-block, 176 bytes)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_q5_k.comp",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),  # No Q5_K model in bench set
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_q6_k",
    group="gemv",
    description="GEMV with Q6_K dequantization (256 vals/super-block, 210 bytes)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_q6_k.comp",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),  # No Q6_K model in bench set
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_bf16",
    group="gemv",
    description="GEMV with BF16 weights (byte-pair reads for alignment safety)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_bf16.comp",
        "cuda": "src/backend/kernels/cuda/gemv_bf16.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=("bf16",),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_f16",
    group="gemv",
    description="GEMV with F16 weights (byte-pair reads for alignment safety)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_f16.comp",
        "cuda": "src/backend/kernels/cuda/gemv_f16.zig",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_fp8_e4m3",
    group="gemv",
    description="GEMV with FP8 E4M3 weights (256-entry LUT dequant)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_fp8_e4m3.comp",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_fp8_e5m2",
    group="gemv",
    description="GEMV with FP8 E5M2 weights (runtime bit manipulation)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "vulkan": "src/backend/kernels/vulkan/gemv_fp8_e5m2.comp",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),
    reference_fn="gemv_f32",
))

_reg(Kernel(
    name="gemv_nvfp4_st",
    group="gemv",
    description="GEMV with NVFP4 SafeTensors (E2M1 nibbles + FP8 E4M3 scales)",
    sources={
        "metal": "src/backend/kernels/metal/gemv.metal",
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="gemv",
    bench_quants=(),
    reference_fn="gemv_f32",
))

# ── Fused ops ─────────────────────────────────────────────────────

_reg(Kernel(
    name="swiglu",
    group="fused",
    description="Fused SwiGLU: silu(W_gate @ x) * (W_up @ x)",
    sources={
        "cpu": "src/backend/cpu.zig",
    },
    golden_prefix="swiglu",
    reference_fn="swiglu_fused",
))

# ── SSM ops ──────────────────────────────────────────────────────

_reg(Kernel(
    name="conv1d",
    group="ssm",
    description="Causal 1D convolution for SSM layers (ring buffer, single step)",
    sources={"shared": "src/ops/ssm.zig"},
    golden_prefix="conv1d",
    bench_quants=(),
    reference_fn="conv1d_causal",
))

_reg(Kernel(
    name="deltanet",
    group="ssm",
    description="DeltaNet linear attention recurrence (single step)",
    sources={"cpu": "src/backend/cpu.zig"},
    golden_prefix="deltanet",
    bench_quants=(),
    reference_fn="deltanet_recurrence",
))

# ── MoE routing ──────────────────────────────────────────────────

_reg(Kernel(
    name="moe_routing_topk",
    group="routing",
    description="Top-k expert routing with softmax + renormalization",
    sources={"cpu": "src/backend/cpu.zig"},
    golden_prefix="moe_topk",
    bench_quants=(),
    reference_fn="moe_routing_topk",
))

_reg(Kernel(
    name="moe_routing_sigmoid",
    group="routing",
    description="Sigmoid expert routing (GLM4-style, independent gates)",
    sources={"cpu": "src/backend/cpu.zig"},
    golden_prefix="moe_sigmoid",
    bench_quants=(),
    reference_fn="moe_routing_sigmoid",
))

# ── Math ops ──────────────────────────────────────────────────────

_reg(Kernel(
    name="argmax",
    group="math",
    description="Index of maximum element",
    sources={
        "shared": "src/ops/math.zig",
    },
    golden_prefix="argmax",
    reference_fn="argmax",
))


# ── Helpers ───────────────────────────────────────────────────────

def find_kernels(query: str) -> list[Kernel]:
    """Find kernels matching a query (exact name, group name, or prefix)."""
    # Exact match
    if query in KERNELS:
        return [KERNELS[query]]
    # Group match
    group_matches = [k for k in KERNELS.values() if k.group == query]
    if group_matches:
        return group_matches
    # Prefix match
    prefix_matches = [k for k in KERNELS.values() if k.name.startswith(query)]
    if prefix_matches:
        return prefix_matches
    return []


def groups() -> dict[str, list[Kernel]]:
    """Return kernels grouped by their group field."""
    out: dict[str, list[Kernel]] = {}
    for k in KERNELS.values():
        out.setdefault(k.group, []).append(k)
    return out


def changed_kernels() -> list[Kernel]:
    """Detect which kernels have changed source files since HEAD (git diff)."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=str(AGAVE_ROOT),
            capture_output=True,
            text=True,
        )
        changed_files = set(result.stdout.strip().split("\n"))
    except Exception:
        return []

    # Also check staged changes
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=str(AGAVE_ROOT),
            capture_output=True,
            text=True,
        )
        changed_files.update(result.stdout.strip().split("\n"))
    except Exception:
        pass

    changed_files.discard("")

    matched = []
    for kernel in KERNELS.values():
        for path in kernel.sources.values():
            paths = [path] if isinstance(path, str) else path
            if any(p in changed_files for p in paths):
                matched.append(kernel)
                break

    return matched


def golden_files(prefix: str) -> list[Path]:
    """List golden .bin files matching a prefix."""
    golden_dir = Path(__file__).parent / "golden"
    if not golden_dir.exists():
        return []
    return sorted(golden_dir.glob(f"{prefix}_*.bin"))
