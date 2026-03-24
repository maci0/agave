//! Backend abstraction for compute operations.
//! Uses a tagged union with `inline else` dispatch for zero-overhead
//! backend selection — no VTable indirection in the hot path.
//!
//! ## UMA (Unified Memory Architecture) contract
//!
//! On UMA platforms (Apple Silicon, NVIDIA Grace/GB10, AMD APU/Ryzen AI Max+),
//! the CPU and GPU share the same physical DRAM. GPU backends on these platforms
//! can wrap existing CPU allocations as GPU buffers with zero copies:
//!
//!   - **Metal**: `newBufferWithBytesNoCopy` + `storageModeShared`
//!   - **Vulkan**: `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` memory type
//!   - **CUDA**: `cudaMallocManaged` or `cudaHostAlloc(cudaHostAllocMapped)`
//!   - **ROCm/HIP**: `hipMallocManaged` or `hipHostMalloc(hipHostMallocMapped)`
//!
//! All GPU backends use deferred dispatch — operations are encoded into command
//! buffers without blocking. Models call `be.sync()` only at points where CPU
//! code reads GPU-produced data. On discrete GPUs, `sync()` must also copy
//! results back; on UMA, results are already visible in CPU address space.
//!
//! Implementations: cpu.zig, metal.zig, vulkan.zig, cuda.zig, rocm.zig

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// Tensor data with type info for backend dispatch
pub const TensorData = struct {
    data: [*]const u8,
    dtype: DType,
};

/// A single GEMV operation within a batched dispatch: y[n] = W[n,k] @ x.
/// All ops in a batch share the same input vector x and input dimension k.
pub const GemvOp = struct {
    w: TensorData,
    y: [*]f32,
    n: usize,
};

/// Supported tensor data types — canonical definition in format/format.zig,
/// re-exported here for backend consumers.
pub const DType = @import("../format/format.zig").DType;

/// KV cache quantization type — re-exported for backend consumers.
pub const KvQuantType = @import("../ops/kv_quant.zig").KvQuantType;

/// Parameters for DeltaNet SSM recurrence (Qwen3.5 hybrid model).
/// Passed to `Backend.deltaNet()` to keep the function signature manageable.
pub const DeltaNetParams = struct {
    conv_ch: u32,
    d_conv: u32,
    d_inner: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    q_scale: f32,
    rms_eps: f32,
};

/// Backend startup information — filled by each backend during init.
/// Displayed in the system info line after the model banner.
pub const BackendInfo = struct {
    /// Backend display name (e.g., "Metal", "CUDA", "Vulkan", "ROCm", "CPU").
    name: []const u8 = "CPU",

    /// GPU device name (e.g., "Apple M4 Pro", "NVIDIA GB10").
    device_name: []const u8 = "",

    /// Dynamic library loaded via dlopen (e.g., "libcuda.so.1", "libMoltenVK.dylib").
    lib_name: []const u8 = "",

    /// Number of GPU kernels/pipelines loaded at init.
    n_gpu_kernels: u32 = 0,

    /// Kernel type label (e.g., "MSL", "PTX", "SPIR-V", "HSACO").
    kernel_type: []const u8 = "",

    /// Total device memory in bytes (VRAM or unified memory).
    total_mem: usize = 0,

    /// Available (free) device memory in bytes at query time.
    avail_mem: usize = 0,

    /// Whether the GPU shares memory with the CPU (unified memory architecture).
    is_uma: bool = false,

    /// Compute capability string (e.g., "sm_121" for CUDA, "gfx1100" for ROCm).
    compute_cap: []const u8 = "",

    /// Driver/API version string (e.g., "CUDA 13.0", "Vulkan 1.3").
    driver_version: []const u8 = "",

    // ── System-level info (populated by caller, not backend) ──

    /// OS version string (e.g., "macOS 14.2.1", "Linux 6.5.0"). Populated by main.
    os_version: []const u8 = "",

    /// CPU thread count (populated by main).
    n_threads: u32 = 0,

    /// Total system physical memory in bytes (populated by main).
    system_mem: usize = 0,

    /// Available system memory in bytes at query time (populated by main).
    system_avail: usize = 0,

    /// CPU cache sizes in bytes (0 = not detected). Populated by main.
    l1_cache: usize = 0,
    l2_cache: usize = 0,
    l3_cache: usize = 0,

    /// Host architecture string (e.g., "aarch64", "x86_64").
    arch: []const u8 = @tagName(builtin.cpu.arch),

    /// Host OS string (e.g., "macos", "linux").
    os: []const u8 = @tagName(builtin.os.tag),
};

/// CPU cache sizes returned by detectCacheSizes().
pub const CacheSizes = struct { l1: usize = 0, l2: usize = 0, l3: usize = 0 };

/// Detect total system physical memory in bytes. Re-exported from cpu.zig.
pub const detectSystemMem = @import("cpu.zig").detectSystemMem;

/// Detect CPU cache sizes (L1d, L2, L3). Re-exported from cpu.zig.
pub const detectCacheSizes = @import("cpu.zig").detectCacheSizes;

/// Detect available (free) system memory in bytes. Re-exported from cpu.zig.
pub const detectAvailMem = @import("cpu.zig").detectAvailMem;

/// Detect OS version string (e.g., "macOS 14.2.1", "Linux 6.5.0"). Re-exported from cpu.zig.
pub const detectOsVersion = @import("cpu.zig").detectOsVersion;

/// Pre-allocated capacity for GPU buffer caches (weights + activations + KV).
/// Used by Metal, CUDA, Vulkan, and ROCm backends to avoid OOM during hot-path cache puts.
pub const buf_cache_initial_capacity: usize = 512;

/// Elements per small quantization block (Q4_0, Q8_0, etc.).
const quant_block_elems: usize = 32;
/// Elements per large quantization super-block (Q4_K, Q5_K, Q6_K, etc.).
const quant_super_block_elems: usize = 256;

/// Compute raw byte size of a weight matrix [n, k] for a given dtype.
/// Used by GPU backends to determine upload buffer sizes. Accounts for
/// quantization block structure (e.g., Q4_0 = 18 bytes per 32-element block).
pub fn weightBytes(dtype: DType, n: usize, k: usize) usize {
    const nb = (k + quant_block_elems - 1) / quant_block_elems;
    const nsb = (k + quant_super_block_elems - 1) / quant_super_block_elems;
    return switch (dtype) {
        .f32 => n * k * 4,
        .f16, .bf16 => n * k * 2,
        .fp8_e4m3, .fp8_e5m2 => n * k,
        .q8_0 => n * nb * 34,
        .q4_0 => n * nb * 18,
        .q4_1 => n * nb * 20,
        .q5_0 => n * nb * 22,
        .q4_k => n * nsb * 144,
        .q5_k => n * nsb * 176,
        .q6_k => n * nsb * 210,
        .mxfp4 => n * nb * 17,
        .q2_k => n * nsb * 84,
        .q3_k => n * nsb * 110,
        // Conservative fallback for unsupported dtypes (overestimates at f32).
        .tq1_0, .iq4_xs, .iq4_nl, .nvfp4, .mlx_q, .unknown => n * k * 4,
    };
}

/// Placeholder for backends disabled at build time.
/// The tagged union variant exists but can never be instantiated.
/// init() is a @compileError; methods are unreachable stubs for inline else.
pub const NullBackend = struct {
    allow_cpu_fallback: bool = false,

    /// Compile error — this backend was disabled at build time.
    pub fn init(_: std.mem.Allocator) error{BackendDisabled}!NullBackend {
        @compileError("this backend was disabled at build time");
    }

    // Stub methods — unreachable because the variant is never constructed.

    /// Stub KV slice alloc — unreachable (backend disabled).
    pub fn allocKvSlice(_: *NullBackend, _: std.mem.Allocator, _: usize) error{OutOfMemory}![]u8 {
        unreachable;
    }

    /// Stub KV slice free — unreachable (backend disabled).
    pub fn freeKvSlice(_: *NullBackend, _: std.mem.Allocator, _: []u8) void {
        unreachable;
    }

    /// Stub GEMV — unreachable (backend disabled).
    pub fn gemv(_: *NullBackend, _: [*]const f32, _: TensorData, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub RMS normalization — unreachable (backend disabled).
    pub fn rmsNorm(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub SiLU activation — unreachable (backend disabled).
    pub fn silu(_: *NullBackend, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub GELU activation — unreachable (backend disabled).
    pub fn gelu(_: *NullBackend, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub element-wise add — unreachable (backend disabled).
    pub fn add(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub gemvT — unreachable (backend disabled).
    pub fn gemvT(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub addScaled — unreachable (backend disabled).
    pub fn addScaled(_: *NullBackend, _: [*]const f32, _: [*]f32, _: f32, _: usize) void {
        unreachable;
    }

    /// Stub fused add + rmsNorm — unreachable (backend disabled).
    pub fn addRmsNorm(_: *NullBackend, _: [*]f32, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub element-wise mul — unreachable (backend disabled).
    pub fn mul(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub softmax — unreachable (backend disabled).
    pub fn softmax(_: *NullBackend, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub RoPE — unreachable (backend disabled).
    pub fn rope(_: *NullBackend, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub embedding lookup — unreachable (backend disabled).
    pub fn embLookup(_: *NullBackend, _: TensorData, _: u32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub L2 norm — unreachable (backend disabled).
    pub fn l2Norm(_: *NullBackend, _: [*]f32, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub sigmoid mul — unreachable (backend disabled).
    pub fn sigmoidMul(_: *NullBackend, _: [*]f32, _: [*]const f32, _: usize) void {
        unreachable;
    }

    /// Stub fused silu*mul — unreachable (backend disabled).
    pub fn siluMul(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        unreachable;
    }

    /// Stub per-head rmsNorm — unreachable (backend disabled).
    pub fn rmsNormMulti(_: *NullBackend, _: [*]f32, _: [*]const f32, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub deinterleave — unreachable (backend disabled).
    pub fn deinterleave(_: *NullBackend, _: [*]const f32, _: [*]f32, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub sync — unreachable (backend disabled).
    pub fn sync(_: *NullBackend) void {
        unreachable;
    }

    /// Stub SDPA — unreachable (backend disabled).
    pub fn sdpa(_: *NullBackend, _: [*]const f32, _: []u8, _: []u8, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: f32, _: KvQuantType) void {
        unreachable;
    }

    /// Stub NVFP4 SafeTensors GEMV — unreachable (backend disabled).
    pub fn gemvNvfp4St(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub MLX-Q GEMV — unreachable (backend disabled).
    pub fn gemvMlxQ(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: u32) void {
        unreachable;
    }

    /// Stub MXFP4 SafeTensors GEMV — unreachable (backend disabled).
    pub fn gemvMxfp4St(_: *NullBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub batched GEMV — unreachable (backend disabled).
    pub fn gemvMulti(_: *NullBackend, _: [*]const f32, _: []const GemvOp, _: usize) void {
        unreachable;
    }

    /// Stub GEMM — unreachable (backend disabled).
    pub fn gemm(_: *NullBackend, _: [*]const f32, _: TensorData, _: [*]f32, _: usize, _: usize, _: usize) void {
        unreachable;
    }

    /// Stub batched rmsNorm — unreachable (backend disabled).
    pub fn rmsNormBatched(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub batched RoPE — unreachable (backend disabled).
    pub fn ropeBatched(_: *NullBackend, _: [*]f32, _: [*]const u32, _: usize, _: usize, _: usize, _: usize, _: f32) void {
        unreachable;
    }

    /// Stub prefill SDPA — unreachable (backend disabled).
    pub fn sdpaPrefill(_: *NullBackend, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: []u8, _: []u8, _: [*]f32, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32, _: KvQuantType) void {
        unreachable;
    }

    /// Stub DeltaNet recurrence — unreachable (backend disabled).
    pub fn deltaNet(_: *NullBackend, _: [*]const f32, _: [*]f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]f32, _: [*]f32, _: []f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: DeltaNetParams) void {
        unreachable;
    }

    /// Stub beginBatch — unreachable (backend disabled).
    pub fn beginBatch(_: *NullBackend) void {
        unreachable;
    }

    /// Stub endBatch — unreachable (backend disabled).
    pub fn endBatch(_: *NullBackend) void {
        unreachable;
    }

    /// Stub backendInfo — unreachable (backend disabled).
    pub fn backendInfo(_: *const NullBackend) BackendInfo {
        unreachable;
    }
};

/// CPU backend implementation — re-exported so callers use backend.zig as the single import.
pub const CpuBackend = if (build_options.enable_cpu)
    @import("cpu.zig").CpuBackend
else
    NullBackend;

/// Metal GPU backend — re-exported so callers use backend.zig as the single import.
/// On non-macOS platforms, MetalBackend aliases NullBackend so the tagged union
/// remains valid; the .metal variant is simply never constructed.
pub const MetalBackend = if (build_options.enable_metal and builtin.os.tag == .macos)
    @import("metal.zig").MetalBackend
else
    NullBackend;

/// Vulkan GPU backend — re-exported so callers use backend.zig as the single import.
/// Disabled when cross-compiling without Vulkan headers/libs available.
pub const VulkanBackend = if (build_options.enable_vulkan)
    @import("vulkan.zig").VulkanBackend
else
    NullBackend;

/// CUDA GPU backend — re-exported so callers use backend.zig as the single import.
pub const CudaBackend = if (build_options.enable_cuda)
    @import("cuda.zig").CudaBackend
else
    NullBackend;

/// ROCm GPU backend — re-exported so callers use backend.zig as the single import.
pub const RocmBackend = if (build_options.enable_rocm)
    @import("rocm.zig").RocmBackend
else
    NullBackend;

/// Backend interface — all compute goes through this tagged union.
/// Dispatch is resolved via `inline else`, giving the compiler full visibility
/// into each backend's implementation for inlining and optimization.
/// No VTable pointer indirection, no `*anyopaque` casts.
pub const Backend = union(enum) {
    cpu: *CpuBackend,
    metal: *MetalBackend,
    vulkan: *VulkanBackend,
    cuda: *CudaBackend,
    rocm: *RocmBackend,

    /// Allocate a KV cache slice using backend-optimal memory.
    /// `n` is the byte count (caller computes via kvSliceBytes).
    /// On UMA (Metal/Apple Silicon): page-aligned for zero-copy GPU access.
    /// On CUDA UMA: cudaMallocManaged for direct GPU access.
    /// On CUDA discrete: pinned host memory for fast transfers.
    /// On CPU/Vulkan/ROCm: plain allocator.
    /// The allocator is used as fallback; GPU backends may ignore it.
    pub inline fn allocKvSlice(self: Backend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        switch (self) {
            inline else => |be| return be.allocKvSlice(allocator, n),
        }
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    /// Must use the same backend and allocator as allocation.
    pub inline fn freeKvSlice(self: Backend, allocator: std.mem.Allocator, slice: []u8) void {
        switch (self) {
            inline else => |be| be.freeKvSlice(allocator, slice),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] with automatic dequantization.
    pub inline fn gemv(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemv(x, w, y, n, k),
        }
    }

    /// Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    pub inline fn gemm(self: Backend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        switch (self) {
            inline else => |be| be.gemm(x, w, y, n_tok, n_out, n_in),
        }
    }

    /// Normalize each of n_tok rows independently.
    pub inline fn rmsNormBatched(self: Backend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNormBatched(input, weight, output, n_tok, dim, eps),
        }
    }

    /// Apply RoPE to n_tok vectors at positions[0..n_tok].
    pub inline fn ropeBatched(self: Backend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        switch (self) {
            inline else => |be| be.ropeBatched(x, positions, n_tok, n_heads, head_dim, rope_dim, theta),
        }
    }

    /// Prefill attention: causal self-attention for n_tok new tokens.
    pub inline fn sdpaPrefill(self: Backend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpaPrefill(q, k, v, kv_keys, kv_values, output, nh, nkv, hd, prev_len, n_tok, scale, kv_type),
        }
    }

    /// Compute RMS normalization: output = input * weight / rms(input).
    pub inline fn rmsNorm(self: Backend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNorm(input, weight, output, n, eps),
        }
    }

    /// Apply SiLU activation: output = input * sigmoid(input).
    pub inline fn silu(self: Backend, input: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.silu(input, output, n),
        }
    }

    /// GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³))).
    pub inline fn gelu(self: Backend, input: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.gelu(input, output, n),
        }
    }

    /// Element-wise addition: output = a + b.
    pub inline fn add(self: Backend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.add(a, b, output, n),
        }
    }

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D weights.
    /// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
    pub inline fn gemvT(self: Backend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        switch (self) {
            inline else => |be| be.gemvT(x, w, y, out_dim, in_dim),
        }
    }

    /// Scaled accumulate: dst[i] += src[i] * scale.
    /// Used for MoE expert output accumulation to avoid per-expert GPU sync.
    pub inline fn addScaled(self: Backend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        switch (self) {
            inline else => |be| be.addScaled(src, dst, scale, n),
        }
    }

    /// Fused add + rms_norm: a[i] = a[i] + b[i], output = rms_norm(a+b, weight, eps).
    /// Replaces separate add + rmsNorm with a single dispatch.
    pub inline fn addRmsNorm(self: Backend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.addRmsNorm(a, b, weight, output, n, eps),
        }
    }

    /// Element-wise multiplication: output = a * b.
    pub inline fn mul(self: Backend, a: [*]const f32, b: [*]const f32, output: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.mul(a, b, output, n),
        }
    }

    /// Apply softmax normalization in-place.
    pub inline fn softmax(self: Backend, data: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.softmax(data, n),
        }
    }

    /// Apply Rotary Position Embedding (RoPE) in-place.
    pub inline fn rope(self: Backend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        switch (self) {
            inline else => |be| be.rope(x, pos, n_heads, head_dim, rope_dim, theta),
        }
    }

    /// Look up a token embedding with automatic dequantization.
    pub inline fn embLookup(self: Backend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        switch (self) {
            inline else => |be| be.embLookup(table, token_id, output, dim),
        }
    }

    /// L2 normalize a vector in-place.
    pub inline fn l2Norm(self: Backend, x: [*]f32, n: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.l2Norm(x, n, eps),
        }
    }

    /// In-place sigmoid-gated multiply: data[i] *= sigmoid(gate[i]).
    /// Used to apply attention gates on GPU without a CPU sync.
    pub inline fn sigmoidMul(self: Backend, data: [*]f32, gate: [*]const f32, n: usize) void {
        switch (self) {
            inline else => |be| be.sigmoidMul(data, gate, n),
        }
    }

    /// De-interleave paired blocks: input[n_pairs * 2 * stride] → out_a + out_b.
    /// For each pair h: out_a[h*stride .. (h+1)*stride] = input[(2*h)*stride .. (2*h+1)*stride],
    ///                  out_b[h*stride .. (h+1)*stride] = input[(2*h+1)*stride .. (2*h+2)*stride].
    pub inline fn deinterleave(self: Backend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        switch (self) {
            inline else => |be| be.deinterleave(input, out_a, out_b, stride, n_pairs),
        }
    }

    /// Fused SiLU activation + multiply: out[i] = silu(a[i]) * b[i].
    /// Used in SwiGLU FFN to replace separate silu + mul dispatches.
    pub inline fn siluMul(self: Backend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        switch (self) {
            inline else => |be| be.siluMul(a, b, out, n),
        }
    }

    /// In-place rmsNorm applied to n_heads independent heads (each head_dim
    /// elements, contiguous), sharing the same weight vector.
    /// Replaces N separate rmsNorm calls with a single batched dispatch.
    pub inline fn rmsNormMulti(self: Backend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        switch (self) {
            inline else => |be| be.rmsNormMulti(data, weight, n_heads, head_dim, eps),
        }
    }

    /// Commit pending GPU work and wait for completion.
    /// Call before CPU code reads buffers written by deferred GPU ops.
    /// No-op on CPU and Vulkan (each dispatch already synchronizes).
    /// On Metal: commits the active command buffer. On CUDA/ROCm:
    /// synchronizes the context, downloads dirty activations, and marks
    /// all activation cache entries as stale (so subsequent CPU writes
    /// are re-uploaded on next GPU use).
    pub inline fn sync(self: Backend) void {
        switch (self) {
            inline else => |be| be.sync(),
        }
    }

    /// Scaled dot-product attention with KV cache append.
    /// Appends `k_new`/`v_new` at position `seq_len` in the KV cache, then
    /// computes softmax(Q @ K^T * scale) @ V over `seq_len + 1` positions.
    /// KV cache is stored in `kv_type` format (f32, f16, q8_0, etc.).
    /// Keys/values are byte slices; backends quantize on append and dequantize on read.
    /// Each backend handles sync internally. No caller sync needed.
    pub inline fn sdpa(self: Backend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type: KvQuantType) void {
        switch (self) {
            inline else => |be| be.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type),
        }
    }

    /// Batched GEMV: dispatch multiple y[n] = W[n,k] @ x[k] ops sharing the same
    /// input vector x and dimension k. GPU backends may fuse into a single kernel
    /// launch to reduce dispatch overhead. CPU/fallback backends run sequentially.
    pub inline fn gemvMulti(self: Backend, x: [*]const f32, ops: []const GemvOp, k: usize) void {
        switch (self) {
            inline else => |be| be.gemvMulti(x, ops, k),
        }
    }

    /// DeltaNet SSM recurrence: conv1d + L2 norm + recurrence + gated output.
    /// On GPU backends, runs entirely on the GPU (no CPU sync needed).
    /// On CPU, runs the same computation inline.
    /// Parameters (in order): conv_in, conv_out, z_buf, alpha_buf, beta_buf,
    /// output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, params.
    pub inline fn deltaNet(self: Backend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: DeltaNetParams) void {
        switch (self) {
            inline else => |be| be.deltaNet(conv_in, conv_out, z_buf, alpha_buf, beta_buf, output, conv_state, ssm_state, ssm_a, dt_bias, conv_w, ssm_norm_w, p),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] for NVFP4 SafeTensors layout (separated weight + scale).
    ///
    /// NVFP4 SafeTensors stores weights as packed nibble pairs and FP8 E4M3
    /// scales in separate tensors, with group_size=16 elements per scale.
    /// GPU backends that don't yet have a native kernel will error unless
    /// `--allow-cpu-fallback` was passed, in which case they delegate to CPU.
    ///
    /// Parameters:
    ///   - x: Input vector [k].
    ///   - weight: Packed nibble pairs [n * k/2] bytes.
    ///   - scale: FP8 E4M3 block scales [n * k/16] bytes.
    ///   - y: Output vector [n].
    ///   - n: Number of output rows.
    ///   - k: Number of input columns (must be divisible by 16).
    pub inline fn gemvNvfp4St(self: Backend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemvNvfp4St(x, weight, scale, y, n, k),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] for MLX affine quantized layout.
    ///
    /// MLX quantization stores weights as packed integer nibbles (4-bit or 6-bit)
    /// in u32 words, with per-group bf16 scales and biases (group_size=64).
    /// Dequant: float_val = scale * int_val + bias.
    ///
    /// Parameters:
    ///   - x: Input vector [k].
    ///   - weight: Packed integer values [n * groups_per_row * words_per_group] as bytes.
    ///   - scales: BF16 per-group scales [n * groups_per_row * 2] bytes.
    ///   - biases: BF16 per-group biases [n * groups_per_row * 2] bytes.
    ///   - y: Output vector [n].
    ///   - n: Number of output rows.
    ///   - k: Number of input columns.
    ///   - bits: Quantization bit width (4, 6, or 8).
    pub inline fn gemvMlxQ(self: Backend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        switch (self) {
            inline else => |be| be.gemvMlxQ(x, weight, scales, biases, y, n, k, bits),
        }
    }

    /// Compute y[n] = W[n,k] @ x[k] for MXFP4 SafeTensors layout (MLX-style packing).
    ///
    /// MXFP4 stores weights as U32-packed 4-bit nibbles (8 per word) with
    /// FP8 E4M3 per-group scales (group_size=32). No quantization bias.
    /// Dequant: float_val = mxfp4_lut[nibble] * fp8_scale.
    pub inline fn gemvMxfp4St(self: Backend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        switch (self) {
            inline else => |be| be.gemvMxfp4St(x, weight, scale, y, n, k),
        }
    }

    /// Begin a batch of independent GPU dispatches. While active, memory barriers
    /// between dispatches are suppressed so the GPU can overlap execution.
    /// Call endBatch() after the last independent op to insert a single barrier.
    /// No-op on CPU.
    pub inline fn beginBatch(self: Backend) void {
        switch (self) {
            inline else => |be| be.beginBatch(),
        }
    }

    /// End a batch of independent GPU dispatches and insert a memory barrier.
    /// No-op on CPU.
    pub inline fn endBatch(self: Backend) void {
        switch (self) {
            inline else => |be| be.endBatch(),
        }
    }

    /// Returns backend startup information (device name, lib, VRAM, etc.).
    pub inline fn backendInfo(self: Backend) BackendInfo {
        switch (self) {
            inline else => |be| return be.backendInfo(),
        }
    }
};
