//! Metal GPU backend for Apple Silicon.
//! Compiles MSL kernels at init, dispatches compute via command buffers.
//! On Apple Silicon the CPU and GPU share unified memory, so Metal buffers
//! wrapping existing allocations require no copies on the GPU side.
//!
//! Command buffer batching: multiple kernel dispatches share a single command
//! buffer. The buffer is committed only when `flush()` is called, which happens
//! just before the CPU needs to read results. Multi-pass ops (softmax = 3 passes,
//! l2Norm = 2 passes) batch in a single commit, reducing GPU round-trips.
//! rmsNorm uses a fused single-dispatch kernel.

const std = @import("std");
const objc = @import("objc.zig");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;
const DType = backend_mod.DType;
const CpuBackend = @import("cpu.zig").CpuBackend;
const KvQuantType = backend_mod.KvQuantType;
const ThreadPool = @import("../thread_pool.zig").ThreadPool;
const kv_quant = @import("../ops/kv_quant.zig");
const mlx_ops = @import("../ops/mlx.zig");

const msl_source = @embedFile("kernels/metal/common.metal") ++
    @embedFile("kernels/metal/elementwise.metal") ++
    @embedFile("kernels/metal/norm.metal") ++
    @embedFile("kernels/metal/rope.metal") ++
    @embedFile("kernels/metal/gemv.metal") ++
    @embedFile("kernels/metal/gemm.metal") ++
    @embedFile("kernels/metal/sdpa.metal") ++
    @embedFile("kernels/metal/sdpa_tree.metal") ++
    @embedFile("kernels/metal/deltanet.metal") ++
    @embedFile("kernels/metal/gemv_tiled.metal") ++
    @embedFile("kernels/metal/megakernel.metal") ++
    @embedFile("kernels/metal/mega_common.metal") ++
    @embedFile("kernels/metal/mega_qwen35_q8.metal") ++
    @embedFile("kernels/metal/mega_gemma_q4k.metal") ++
    @embedFile("kernels/metal/mega_gemma_q8.metal") ++
    @embedFile("kernels/metal/mega_qwen35_q4k.metal") ++
    @embedFile("kernels/metal/mega_nemotron_h_q8.metal");

const page_size = std.heap.page_size_min;

// ── Tuning constants ────────────────────────────────────────────
// These thresholds control when the Metal backend falls back to CPU
// or caps kernel dispatch sizes. Tuned for Apple Silicon M-series.

/// Threadgroup size for reduction/GEMV kernels (matches MSL `[[ threads_per_threadgroup ]]`).
const threadgroup_size: usize = 256;
/// Softmax inputs smaller than this threshold run on CPU (GPU dispatch overhead dominates).
const softmax_cpu_threshold: usize = 128;
/// Maximum sequence length for the fused SDPA kernel (limited by threadgroup memory).
const sdpa_max_seq_len: usize = 4096;
/// Maximum per-head dimension for the fused SDPA kernel.
const sdpa_max_head_dim: usize = 256;
/// Number of output rows processed per threadgroup in Q4_0 GEMV (must match q4_0_nr in gemv.metal).
const q4_0_nr: usize = 4;
/// Number of output rows processed per threadgroup in Q8_0 GEMV (must match q8_0_nr in gemv.metal).
const q8_0_nr: usize = 4;
/// Number of output rows processed per threadgroup in Q4_K/Q5_K/Q6_K GEMV (must match q4_k_nr in gemv.metal).
const q4_k_nr: usize = 2;
/// Number of output rows processed per threadgroup in Q2_K GEMV (must match q2_k_nr in gemv.metal).
const q2_k_nr: usize = 2;
/// Number of output rows processed per threadgroup in Q3_K GEMV (must match q3_k_nr in gemv.metal).
const q3_k_nr: usize = 2;
/// Number of output rows processed per threadgroup in BF16 GEMV (must match bf16_nr in gemv.metal).
const bf16_nr: usize = 2;
/// Number of output rows processed per threadgroup in F16 GEMV (must match f16_nr in gemv.metal).
const f16_nr: usize = 2;
/// Elements per small quantization block (Q4_0, Q8_0, etc.).
const quant_block_elems: usize = backend_mod.quant_block_elems;
/// Bytes per Q8_0 block: 2 bytes f16 scale + 32 bytes i8 data.
const q8_0_block_bytes: usize = backend_mod.q8_0_block_bytes;
/// Elements per large quantization super-block (Q4_K, Q5_K, Q6_K, etc.).
const quant_super_block_elems: usize = backend_mod.quant_super_block_elems;
/// Elements per MLX group (64-element groups for MLX 4-bit quantization).
const mlx_group_size: usize = mlx_ops.mlx_group_size;
/// Elements per MXFP4 group (32-element groups for microscaled FP4 quantization).
const mxfp4_group_size: usize = mlx_ops.mxfp4_group_size;
/// SIMD group width on Apple Silicon GPUs (threads per SIMD group).
const simd_width: usize = 32;
/// GPU SDPA threadgroup size (threads per query head, smaller than general threadgroup_size
/// to fit per-head accumulation state within threadgroup memory).
const sdpa_threadgroup_size: usize = 128;
/// Words per group for MLX 4-bit quantization (64 elems / 8 nibbles per word).
const mlx_words_per_group_q4: usize = 8;
/// MLX 6-bit: 64 elements × 6 bits / 32 bits per word = 12 words per group.
const mlx_words_per_group_q6: usize = 12;
/// Words per group for MLX 8-bit quantization (64 elems / 4 values per word).
const mlx_words_per_group_q8: usize = 16;
/// Words per group for MXFP4 quantization (32 nibbles / 8 per word).
const mxfp4_words_per_group: usize = 4;

/// Reference to a cached Metal buffer with a byte offset.
/// Allows sub-region access (e.g. per-head slices) without creating separate
/// Metal buffer objects — the parent buffer is reused with an offset.
const BufRef = struct {
    buf: objc.id,
    offset: usize,
};

/// Cached Metal buffer info, keyed by page-aligned base address.
const BufferInfo = struct {
    metal_buf: objc.id,
    len: usize,
};

/// Metal GPU backend state.
pub const MetalBackend = struct {
    device: objc.id,
    queue: objc.id,
    library: objc.id,
    // Pipeline states for each kernel
    pipe_silu: objc.id,
    pipe_add: objc.id,
    pipe_mul: objc.id,
    pipe_rms_ss: objc.id,
    pipe_rms_apply: objc.id,
    pipe_softmax_max: objc.id,
    pipe_softmax_exp_sum: objc.id,
    pipe_softmax_div: objc.id,
    pipe_rope: objc.id,
    pipe_l2_apply: objc.id,
    pipe_gemv_f32: objc.id,
    pipe_gemv_q8_0: objc.id,
    pipe_gemv_q4_0: objc.id,
    pipe_gemv_q4_1: objc.id,
    pipe_gemv_bf16: objc.id,
    pipe_gemv_f16: objc.id,
    pipe_gemv_nvfp4_st: objc.id,
    pipe_split_qgate: objc.id,
    pipe_gemv_q4_k: objc.id,
    pipe_gemv_tiled_q4_k: objc.id,
    pipe_gemv_tiled_q8_0: objc.id,
    pipe_gemv_q6_k: objc.id,
    pipe_gemv_q2_k: objc.id,
    pipe_gemv_q3_k: objc.id,
    pipe_gemv_q5_0: objc.id,
    pipe_gemv_iq4_nl: objc.id,
    pipe_gemv_iq4_xs: objc.id,
    pipe_gemv_q5_k: objc.id,
    pipe_gemv_mlx_q4: objc.id,
    pipe_gemv_mlx_q6: objc.id,
    pipe_gemv_mlx_q8: objc.id,
    pipe_gemv_mxfp4: objc.id,
    pipe_gemv_mxfp4_st: objc.id,
    pipe_gemv_fp8_e4m3: objc.id,
    pipe_gemv_fp8_e5m2: objc.id,
    pipe_gemm_f32: objc.id,
    pipe_gemm_bf16: objc.id,
    pipe_gemm_q8_0: objc.id,
    pipe_gemm_q4_0: objc.id,
    pipe_gemm_q4_k: objc.id,
    pipe_gemm_q6_k: objc.id,
    pipe_gemm_q5_k: objc.id,
    pipe_rope_batched: objc.id,
    pipe_sdpa_prefill: objc.id,
    pipe_copy_f32: objc.id,
    pipe_gelu: objc.id,
    pipe_sigmoid_mul: objc.id,
    pipe_add_scaled: objc.id,
    pipe_gemv_t_q8_0: objc.id,
    pipe_deinterleave: objc.id,
    pipe_silu_mul: objc.id,
    pipe_gelu_mul: objc.id,
    pipe_rms_norm_fused: objc.id,
    pipe_add_rms_norm_fused: objc.id,
    pipe_kv_append: objc.id,
    pipe_sdpa: objc.id,
    pipe_sdpa_tree: objc.id,
    pipe_sdpa_tree_turbo: objc.id,
    pipe_sdpa_turbo: objc.id,
    pipe_dn_gate_beta: objc.id,
    pipe_dn_conv1d: objc.id,
    pipe_dn_l2_norm: objc.id,
    pipe_dn_recurrence: objc.id,
    pipe_fused_ffn_q8: objc.id,
    pipe_fused_ffn_q4_k: objc.id,
    pipe_fused_ffn_q4_0: objc.id,
    pipe_mega_qwen35_q8: objc.id,
    pipe_mega_gemma_q4k: objc.id,
    pipe_mega_gemma_q8: objc.id,
    pipe_mega_qwen35_q4k: objc.id,
    pipe_mega_nemotron_h_q8: objc.id,
    /// Auto-composed megakernel pipeline (null until compileComposedMegakernel called).
    pipe_mega_auto: ?objc.id = null,
    pipe_fused_ffn_silu_mlx_q4: objc.id,
    pipe_fused_ffn_gelu_q8: objc.id,
    pipe_fused_ffn_gelu_q4_k: objc.id,
    pipe_fused_ffn_gelu_q4_0: objc.id,
    pipe_fused_ffn_q6_k: objc.id,
    pipe_fused_ffn_gelu_q6_k: objc.id,
    pipe_fused_ffn_q5_k: objc.id,
    pipe_fused_ffn_gelu_q5_k: objc.id,
    /// Scratch buffer for multi-pass reductions: 8 bytes = 2 × f32.
    /// Used by softmax (3-pass: max at offset 0, sum at offset 4)
    /// and l2Norm (2-pass: sum-of-squares at offset 0).
    scratch_buf: objc.id,
    /// Persistent command buffer for batching multiple dispatches.
    /// Created lazily on first encode, committed and cleared on flush().
    active_cmd: ?objc.id = null,
    /// Persistent compute encoder reused across dispatches within one command buffer.
    /// Memory barriers between dispatches ensure write visibility.
    /// Ended and cleared on flush(). Eliminates per-op encoder overhead.
    active_enc: ?objc.id = null,
    /// Cache of MTLBuffer objects keyed by data pointer (usize).
    /// Avoids recreating wrapBuffer objects for stable pointers (mmap'd weights,
    /// model activation buffers) on every GEMV / norm / elementwise call.
    /// Cached buffers live for the model's lifetime and are released in deinit().
    buf_cache: std.AutoHashMap(usize, BufferInfo),
    /// When true, memory barriers between dispatches are suppressed to allow
    /// the GPU to overlap independent operations. Set by beginBatch(), cleared
    /// by endBatch() which inserts a single barrier.
    batch_mode: bool = false,
    /// Optional thread pool for parallelizing CPU fallback work (e.g. unsupported dtypes).
    /// Set by the caller after init — Metal doesn't own the pool.
    pool: ?*ThreadPool = null,

    /// Dispatch/barrier/sync counters — active only when profiling is enabled.
    dispatch_count: u32 = 0,
    barrier_count: u32 = 0,
    sync_count: u32 = 0,
    profile_counters: bool = false,

    // ── Init / deinit ─────────────────────────────────────────

    /// Initialize the Metal backend: get device, compile shaders, create pipelines.
    pub fn init(allocator: std.mem.Allocator) !MetalBackend {
        const device = objc.MTLCreateSystemDefaultDevice() orelse return error.NoMetalDevice;
        errdefer release(device);

        const queue = objc.msgSend(?objc.id, device, objc.sel("newCommandQueue"), .{}) orelse
            return error.CommandQueueFailed;
        errdefer release(queue);

        // Compile MSL source via NSString + newLibraryWithSource:options:error:
        const NSString = objc.getClass("NSString") orelse return error.NoFoundation;
        const source_ns = objc.msgSend(
            ?objc.id,
            NSString,
            objc.sel("stringWithUTF8String:"),
            .{@as([*:0]const u8, @ptrCast(msl_source.ptr))},
        ) orelse return error.StringFailed;

        var compile_err: ?objc.id = null;
        const library = objc.msgSend(
            ?objc.id,
            device,
            objc.sel("newLibraryWithSource:options:error:"),
            .{ source_ns, @as(?objc.id, null), @as(*?objc.id, &compile_err) },
        ) orelse {
            if (compile_err) |err_obj| {
                const desc_ns = objc.msgSend(?objc.id, err_obj, objc.sel("localizedDescription"), .{});
                if (desc_ns) |ns| {
                    const cstr = objc.msgSend(?[*:0]const u8, ns, objc.sel("UTF8String"), .{});
                    if (cstr) |d| std.log.err("Metal shader compile error: {s}", .{d});
                }
            }
            return error.ShaderCompileFailed;
        };
        errdefer release(library);

        // Scratch buffer — 8 bytes, MTLResourceStorageModeShared (0)
        const scratch_buf = objc.msgSend(
            objc.id,
            device,
            objc.sel("newBufferWithLength:options:"),
            .{ @as(objc.NSUInteger, 8), @as(objc.NSUInteger, 0) },
        );
        errdefer release(scratch_buf);

        var self = MetalBackend{
            .device = device,
            .queue = queue,
            .library = library,
            .pipe_silu = undefined,
            .pipe_add = undefined,
            .pipe_mul = undefined,
            .pipe_rms_ss = undefined,
            .pipe_rms_apply = undefined,
            .pipe_softmax_max = undefined,
            .pipe_softmax_exp_sum = undefined,
            .pipe_softmax_div = undefined,
            .pipe_rope = undefined,
            .pipe_l2_apply = undefined,
            .pipe_gemv_f32 = undefined,
            .pipe_gemv_q8_0 = undefined,
            .pipe_gemv_q4_0 = undefined,
            .pipe_gemv_q4_1 = undefined,
            .pipe_gemv_bf16 = undefined,
            .pipe_gemv_f16 = undefined,
            .pipe_gemv_mlx_q4 = undefined,
            .pipe_gemv_mlx_q6 = undefined,
            .pipe_gemv_mlx_q8 = undefined,
            .pipe_gemv_mxfp4 = undefined,
            .pipe_gemv_mxfp4_st = undefined,
            .pipe_gemv_nvfp4_st = undefined,
            .pipe_split_qgate = undefined,
            .pipe_gemv_q4_k = undefined,
            .pipe_gemv_tiled_q4_k = undefined,
            .pipe_gemv_tiled_q8_0 = undefined,
            .pipe_gemv_q6_k = undefined,
            .pipe_gemv_q2_k = undefined,
            .pipe_gemv_q3_k = undefined,
            .pipe_gemv_q5_0 = undefined,
            .pipe_gemv_iq4_nl = undefined,
            .pipe_gemv_iq4_xs = undefined,
            .pipe_gemv_q5_k = undefined,
            .pipe_gemv_fp8_e4m3 = undefined,
            .pipe_gemv_fp8_e5m2 = undefined,
            .pipe_gemm_f32 = undefined,
            .pipe_gemm_bf16 = undefined,
            .pipe_gemm_q8_0 = undefined,
            .pipe_gemm_q4_0 = undefined,
            .pipe_gemm_q4_k = undefined,
            .pipe_gemm_q6_k = undefined,
            .pipe_gemm_q5_k = undefined,
            .pipe_rope_batched = undefined,
            .pipe_sdpa_prefill = undefined,
            .pipe_copy_f32 = undefined,
            .pipe_gelu = undefined,
            .pipe_sigmoid_mul = undefined,
            .pipe_add_scaled = undefined,
            .pipe_gemv_t_q8_0 = undefined,
            .pipe_deinterleave = undefined,
            .pipe_silu_mul = undefined,
            .pipe_gelu_mul = undefined,
            .pipe_rms_norm_fused = undefined,
            .pipe_add_rms_norm_fused = undefined,
            .pipe_kv_append = undefined,
            .pipe_sdpa = undefined,
            .pipe_sdpa_tree = undefined,
            .pipe_sdpa_tree_turbo = undefined,
            .pipe_sdpa_turbo = undefined,
            .pipe_dn_gate_beta = undefined,
            .pipe_dn_conv1d = undefined,
            .pipe_dn_l2_norm = undefined,
            .pipe_dn_recurrence = undefined,
            .pipe_fused_ffn_q8 = undefined,
            .pipe_fused_ffn_q4_k = undefined,
            .pipe_fused_ffn_q4_0 = undefined,
            .pipe_mega_qwen35_q8 = undefined,
            .pipe_mega_gemma_q4k = undefined,
            .pipe_mega_gemma_q8 = undefined,
            .pipe_mega_qwen35_q4k = undefined,
            .pipe_mega_nemotron_h_q8 = undefined,
            .pipe_fused_ffn_silu_mlx_q4 = undefined,
            .pipe_fused_ffn_gelu_q8 = undefined,
            .pipe_fused_ffn_gelu_q4_k = undefined,
            .pipe_fused_ffn_gelu_q4_0 = undefined,
            .pipe_fused_ffn_q6_k = undefined,
            .pipe_fused_ffn_gelu_q6_k = undefined,
            .pipe_fused_ffn_q5_k = undefined,
            .pipe_fused_ffn_gelu_q5_k = undefined,
            .scratch_buf = scratch_buf,
            .active_cmd = null,
            .buf_cache = std.AutoHashMap(usize, BufferInfo).init(allocator),
        };
        // Pre-allocate capacity so hot-path buf_cache.put() calls won't need to grow.
        try self.buf_cache.ensureTotalCapacity(backend_mod.buf_cache_initial_capacity);
        errdefer self.buf_cache.deinit();

        self.pipe_silu = try self.makePipeline("silu_f32");
        self.pipe_add = try self.makePipeline("add_f32");
        self.pipe_mul = try self.makePipeline("mul_f32");
        self.pipe_rms_ss = try self.makePipeline("rms_norm_ss");
        self.pipe_rms_apply = try self.makePipeline("rms_norm_apply");
        self.pipe_softmax_max = try self.makePipeline("softmax_max");
        self.pipe_softmax_exp_sum = try self.makePipeline("softmax_exp_sum");
        self.pipe_softmax_div = try self.makePipeline("softmax_div");
        self.pipe_rope = try self.makePipeline("rope_f32");
        self.pipe_l2_apply = try self.makePipeline("l2_norm_apply");
        self.pipe_gemv_f32 = try self.makePipeline("gemv_f32");
        self.pipe_gemv_q8_0 = try self.makePipeline("gemv_q8_0");
        self.pipe_gemv_q4_0 = try self.makePipeline("gemv_q4_0");
        self.pipe_gemv_q4_1 = try self.makePipeline("gemv_q4_1");
        self.pipe_gemv_bf16 = try self.makePipeline("gemv_bf16");
        self.pipe_gemv_f16 = try self.makePipeline("gemv_f16");
        self.pipe_gemv_mlx_q4 = try self.makePipeline("gemv_mlx_q4");
        self.pipe_gemv_mlx_q6 = try self.makePipeline("gemv_mlx_q6");
        self.pipe_gemv_mlx_q8 = try self.makePipeline("gemv_mlx_q8");
        self.pipe_gemv_mxfp4 = try self.makePipeline("gemv_mxfp4");
        self.pipe_gemv_mxfp4_st = try self.makePipeline("gemv_mxfp4_st");
        self.pipe_gemv_nvfp4_st = try self.makePipeline("gemv_nvfp4_st");
        self.pipe_split_qgate = try self.makePipeline("split_qgate");
        self.pipe_gemv_q4_k = try self.makePipeline("gemv_q4_k");
        self.pipe_gemv_tiled_q4_k = try self.makePipeline("gemv_tiled_q4_k");
        self.pipe_gemv_tiled_q8_0 = try self.makePipeline("gemv_tiled_q8_0");
        self.pipe_gemv_q6_k = try self.makePipeline("gemv_q6_k");
        self.pipe_gemv_q2_k = try self.makePipeline("gemv_q2_k");
        self.pipe_gemv_q3_k = try self.makePipeline("gemv_q3_k");
        self.pipe_gemv_q5_0 = try self.makePipeline("gemv_q5_0");
        self.pipe_gemv_iq4_nl = try self.makePipeline("gemv_iq4_nl");
        self.pipe_gemv_iq4_xs = try self.makePipeline("gemv_iq4_xs");
        self.pipe_gemv_q5_k = try self.makePipeline("gemv_q5_k");
        self.pipe_gemv_fp8_e4m3 = try self.makePipeline("gemv_fp8_e4m3");
        self.pipe_gemv_fp8_e5m2 = try self.makePipeline("gemv_fp8_e5m2");
        self.pipe_gemm_f32 = try self.makePipeline("gemm_f32");
        self.pipe_gemm_bf16 = try self.makePipeline("gemm_bf16");
        self.pipe_gemm_q8_0 = try self.makePipeline("gemm_q8_0");
        self.pipe_gemm_q4_0 = try self.makePipeline("gemm_q4_0");
        self.pipe_gemm_q4_k = try self.makePipeline("gemm_q4_k");
        self.pipe_gemm_q6_k = try self.makePipeline("gemm_q6_k");
        self.pipe_gemm_q5_k = try self.makePipeline("gemm_q5_k");
        self.pipe_rope_batched = try self.makePipeline("rope_batched_f32");
        self.pipe_sdpa_prefill = try self.makePipeline("sdpa_prefill_fa2");
        self.pipe_copy_f32 = try self.makePipeline("copy_f32");
        self.pipe_gelu = try self.makePipeline("gelu_f32");
        self.pipe_sigmoid_mul = try self.makePipeline("sigmoid_mul_f32");
        self.pipe_add_scaled = try self.makePipeline("add_scaled_f32");
        self.pipe_gemv_t_q8_0 = try self.makePipeline("gemv_t_q8_0");
        self.pipe_deinterleave = try self.makePipeline("deinterleave_f32");
        self.pipe_silu_mul = try self.makePipeline("silu_mul_f32");
        self.pipe_gelu_mul = try self.makePipeline("gelu_mul_f32");
        self.pipe_rms_norm_fused = try self.makePipeline("rms_norm_fused_f32");
        self.pipe_add_rms_norm_fused = try self.makePipeline("add_rms_norm_fused_f32");
        self.pipe_kv_append = try self.makePipeline("kv_append");
        self.pipe_sdpa = try self.makePipeline("sdpa_fa2");
        self.pipe_sdpa_tree = try self.makePipeline("sdpa_tree_fa2");
        self.pipe_sdpa_tree_turbo = try self.makePipeline("sdpa_tree_fa2_turbo");
        self.pipe_sdpa_turbo = try self.makePipeline("sdpa_fa2_turbo");
        self.pipe_dn_gate_beta = try self.makePipeline("deltanet_gate_beta");
        self.pipe_dn_conv1d = try self.makePipeline("deltanet_conv1d");
        self.pipe_dn_l2_norm = try self.makePipeline("deltanet_l2_norm");
        self.pipe_dn_recurrence = try self.makePipeline("deltanet_recurrence");
        self.pipe_fused_ffn_q8 = try self.makePipeline("fused_ffn_gate_up_silu_q8");
        self.pipe_fused_ffn_q4_k = try self.makePipeline("fused_ffn_gate_up_silu_q4_k");
        self.pipe_fused_ffn_q4_0 = try self.makePipeline("fused_ffn_gate_up_silu_q4_0");
        self.pipe_mega_qwen35_q8 = try self.makePipeline("megakernel_qwen35_q8");
        self.pipe_mega_gemma_q4k = try self.makePipeline("megakernel_gemma_q4k");
        self.pipe_mega_gemma_q8 = try self.makePipeline("megakernel_gemma_q8");
        self.pipe_mega_qwen35_q4k = try self.makePipeline("megakernel_qwen35_q4k");
        self.pipe_mega_nemotron_h_q8 = try self.makePipeline("megakernel_nemotron_h_q8");
        self.pipe_fused_ffn_silu_mlx_q4 = try self.makePipeline("fused_ffn_gate_up_silu_mlx_q4");
        self.pipe_fused_ffn_gelu_q8 = try self.makePipeline("fused_ffn_gate_up_gelu_q8");
        self.pipe_fused_ffn_gelu_q4_k = try self.makePipeline("fused_ffn_gate_up_gelu_q4_k");
        self.pipe_fused_ffn_gelu_q4_0 = try self.makePipeline("fused_ffn_gate_up_gelu_q4_0");
        self.pipe_fused_ffn_q6_k = try self.makePipeline("fused_ffn_gate_up_silu_q6_k");
        self.pipe_fused_ffn_gelu_q6_k = try self.makePipeline("fused_ffn_gate_up_gelu_q6_k");
        self.pipe_fused_ffn_q5_k = try self.makePipeline("fused_ffn_gate_up_silu_q5_k");
        self.pipe_fused_ffn_gelu_q5_k = try self.makePipeline("fused_ffn_gate_up_gelu_q5_k");

        return self;
    }

    /// Number of MSL compute pipelines compiled at init.
    pub const n_pipelines: u32 = 70;

    /// Returns the Metal device name (e.g., "Apple M4 Pro").
    pub fn deviceName(self: *const MetalBackend) []const u8 {
        const ns_name = objc.msgSend(objc.id, self.device, objc.sel("name"), .{});
        return std.mem.span(objc.msgSend([*:0]const u8, ns_name, objc.sel("UTF8String"), .{}));
    }

    /// Returns backend startup information for display.
    pub fn backendInfo(self: *const MetalBackend) backend_mod.BackendInfo {
        const total: u64 = objc.msgSend(u64, self.device, objc.sel("recommendedMaxWorkingSetSize"), .{});
        const allocated: u64 = objc.msgSend(u64, self.device, objc.sel("currentAllocatedSize"), .{});
        return .{
            .name = "Metal",
            .device_name = self.deviceName(),
            .lib_name = "Metal.framework",
            .n_gpu_kernels = n_pipelines,
            .kernel_type = "MSL",
            .total_mem = total,
            .avail_mem = if (total > allocated) total - allocated else 0,
            .is_uma = true,
            .driver_version = self.detectMetalFamily(),
        };
    }

    /// Detect the highest supported Metal GPU family.
    fn detectMetalFamily(self: *const MetalBackend) []const u8 {
        // MTLGPUFamily enum values (Apple-defined)
        const families = [_]struct { val: c_long, name: []const u8 }{
            .{ .val = 5001, .name = "Metal 3" },
            .{ .val = 1009, .name = "Apple Family 9" },
            .{ .val = 1008, .name = "Apple Family 8" },
            .{ .val = 1007, .name = "Apple Family 7" },
        };
        for (families) |fam| {
            if (objc.msgSend(bool, self.device, objc.sel("supportsFamily:"), .{fam.val}))
                return fam.name;
        }
        return "";
    }

    /// Compile a named MSL kernel into a compute pipeline state.
    fn makePipeline(self: *MetalBackend, name: [*:0]const u8) !objc.id {
        const NSString = objc.getClass("NSString").?;
        const ns_name = objc.msgSend(
            ?objc.id,
            NSString,
            objc.sel("stringWithUTF8String:"),
            .{name},
        ) orelse return error.StringFailed;

        const func = objc.msgSend(
            ?objc.id,
            self.library,
            objc.sel("newFunctionWithName:"),
            .{ns_name},
        ) orelse {
            std.log.err("Metal kernel not found: {s}", .{name});
            return error.KernelNotFound;
        };

        var err: ?objc.id = null;
        return objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ func, @as(*?objc.id, &err) },
        ) orelse {
            std.log.err("Metal pipeline failed for kernel: {s}", .{name});
            if (err) |err_obj| {
                const desc_ns = objc.msgSend(?objc.id, err_obj, objc.sel("localizedDescription"), .{});
                if (desc_ns) |ns| {
                    const cstr = objc.msgSend(?[*:0]const u8, ns, objc.sel("UTF8String"), .{});
                    if (cstr) |d| std.log.err("  Detail: {s}", .{d});
                }
            }
            return error.PipelineFailed;
        };
    }

    // ── Buffer helpers ────────────────────────────────────────

    /// Create a Metal buffer by copying `len` bytes from `ptr`.
    /// MTLResourceStorageModeShared = 0.
    fn makeBuffer(self: *MetalBackend, ptr: *const anyopaque, len: usize) objc.id {
        return objc.msgSend(
            objc.id,
            self.device,
            objc.sel("newBufferWithBytes:length:options:"),
            .{ ptr, @as(objc.NSUInteger, len), @as(objc.NSUInteger, 0) },
        );
    }

    /// Wrap existing memory as a Metal buffer with zero copy (Apple Silicon unified memory).
    /// The caller retains ownership of the memory — Metal will not free it.
    /// `ptr` MUST be page-aligned; returns null if Metal rejects the pointer.
    fn wrapBuffer(self: *MetalBackend, ptr: *const anyopaque, len: usize) ?objc.id {
        return objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
            .{ ptr, @as(objc.NSUInteger, len), @as(objc.NSUInteger, 0), @as(?objc.id, null) },
        );
    }

    /// Release all cached buffers, the scratch buffer, and free the cache map.
    /// Call this when the MetalBackend is no longer needed.
    pub fn deinit(self: *MetalBackend) void {
        release(self.scratch_buf);
        var it = self.buf_cache.valueIterator();
        while (it.next()) |info| release(info.metal_buf);
        self.buf_cache.deinit();
    }

    /// Return a BufRef (Metal buffer + byte offset) for `ptr`.
    ///
    /// Page-aligned pointers are wrapped zero-copy via newBufferWithBytesNoCopy.
    /// Non-page-aligned pointers are handled by wrapping the enclosing page-aligned
    /// region and returning an offset, so sub-regions (e.g. per-head slices) reuse
    /// the parent buffer instead of creating isolated copies.
    ///
    /// Cached by page-aligned base address — subsequent calls for the same page
    /// skip ObjC allocation entirely.
    fn getBufRef(self: *MetalBackend, ptr: *const anyopaque, len: usize) BufRef {
        const addr = @intFromPtr(ptr);
        const aligned_base = addr & ~(@as(usize, page_size - 1));
        const offset = addr - aligned_base;
        const needed = offset + len;

        // Check cache for this page-aligned base
        if (self.buf_cache.get(aligned_base)) |cached| {
            if (cached.len >= needed) return .{ .buf = cached.metal_buf, .offset = offset };
            // Buffer too small (e.g. KV cache grew) — release old, recreate below
            release(cached.metal_buf);
            _ = self.buf_cache.remove(aligned_base);
        }

        const aligned_len = (needed + page_size - 1) & ~(@as(usize, page_size - 1));
        const aligned_ptr: *const anyopaque = @ptrFromInt(aligned_base);
        if (self.wrapBuffer(aligned_ptr, aligned_len)) |buf| {
            self.buf_cache.put(aligned_base, .{ .metal_buf = buf, .len = aligned_len }) catch |err| {
                // Cache full — release wrap buffer to avoid leak, fall through to copy path
                std.log.warn("Metal buf_cache put failed: {}", .{err});
                release(buf);
                return .{ .buf = self.makeBuffer(ptr, len), .offset = 0 };
            };
            return .{ .buf = buf, .offset = offset };
        }
        // Fallback: copy the data into a Metal-managed buffer (extremely rare)
        const copy_buf = self.makeBuffer(ptr, len);
        self.buf_cache.put(addr, .{ .metal_buf = copy_buf, .len = len }) catch |err| {
            std.log.warn("Metal buf_cache put failed: {}", .{err});
        };
        return .{ .buf = copy_buf, .offset = 0 };
    }

    /// Bind a BufRef (buffer + offset) at the given argument index.
    fn setBuf(enc: objc.id, ref: BufRef, index: u32) void {
        objc.msgSend(void, enc, objc.sel("setBuffer:offset:atIndex:"), .{
            ref.buf, @as(objc.NSUInteger, ref.offset), @as(objc.NSUInteger, index),
        });
    }

    /// Read `count` f32 values back from a Metal buffer's contents pointer.
    fn readBuffer(buf: objc.id, dst: [*]f32, count: usize) void {
        const contents: [*]const f32 = @ptrCast(@alignCast(
            objc.msgSend(*anyopaque, buf, objc.sel("contents"), .{}),
        ));
        @memcpy(dst[0..count], contents[0..count]);
    }

    /// Release an ObjC object (decrements retain count).
    fn release(obj: objc.id) void {
        objc.msgSend(void, obj, objc.sel("release"), .{});
    }

    // ── Dispatch helpers ──────────────────────────────────────

    /// Get or reuse the active compute command encoder.
    /// Creates command buffer and encoder lazily. Reuses the same encoder
    /// across dispatches — pipeline state is swapped via setComputePipelineState.
    /// Memory barriers between dispatches ensure write visibility.
    fn getEncoder(self: *MetalBackend, pipeline: objc.id) objc.id {
        if (self.active_cmd == null) {
            self.active_cmd = objc.msgSend(objc.id, self.queue, objc.sel("commandBuffer"), .{});
        }
        if (self.active_enc == null) {
            self.active_enc = objc.msgSend(objc.id, self.active_cmd.?, objc.sel("computeCommandEncoder"), .{});
        }
        objc.msgSend(void, self.active_enc.?, objc.sel("setComputePipelineState:"), .{pipeline});
        return self.active_enc.?;
    }

    /// Bind a Metal buffer at the given slot with zero offset.
    fn setBuffer(enc: objc.id, buf: objc.id, index: u32) void {
        objc.msgSend(void, enc, objc.sel("setBuffer:offset:atIndex:"), .{
            buf, @as(objc.NSUInteger, 0), @as(objc.NSUInteger, index),
        });
    }

    /// Bind a Metal buffer at the given slot with a byte offset.
    fn setBufferOffset(enc: objc.id, buf: objc.id, offset: usize, index: u32) void {
        objc.msgSend(void, enc, objc.sel("setBuffer:offset:atIndex:"), .{
            buf, @as(objc.NSUInteger, offset), @as(objc.NSUInteger, index),
        });
    }

    /// Push a small constant directly into the argument table (avoids a buffer alloc).
    fn setBytes(enc: objc.id, ptr: *const anyopaque, len: usize, index: u32) void {
        objc.msgSend(void, enc, objc.sel("setBytes:length:atIndex:"), .{
            ptr, @as(objc.NSUInteger, len), @as(objc.NSUInteger, index),
        });
    }

    /// MTLBarrierScope.Buffers — ensures buffer writes from one dispatch
    /// are visible to subsequent dispatches within the same encoder.
    const barrier_scope_buffers: objc.NSUInteger = 1;

    /// Create a 1-D MTLSize (height=1, depth=1).
    inline fn mtlSize1D(width: usize) objc.MTLSize {
        return .{ .width = width, .height = 1, .depth = 1 };
    }

    /// Insert a buffer memory barrier (unless batch_mode defers it) and update profiling counters.
    fn insertBarrier(self: *MetalBackend, enc: objc.id) void {
        if (self.profile_counters) self.dispatch_count += 1;
        if (!self.batch_mode) {
            objc.msgSend(void, enc, objc.sel("memoryBarrierWithScope:"), .{barrier_scope_buffers});
            if (self.profile_counters) self.barrier_count += 1;
        }
    }

    /// Dispatch threadgroups with a memory barrier (unless batch_mode is active,
    /// in which case the barrier is deferred until endBatch).
    fn endEncodeThreadgroups(self: *MetalBackend, enc: objc.id, n_groups: usize, tg_size: usize) void {
        objc.msgSend(void, enc, objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"), .{
            mtlSize1D(n_groups), mtlSize1D(tg_size),
        });
        self.insertBarrier(enc);
    }

    /// Dispatch grid_size threads (1-D) with conditional barrier.
    fn endEncode1D(self: *MetalBackend, enc: objc.id, pipeline: objc.id, grid_size: usize) void {
        const max_tpg = objc.msgSend(
            objc.NSUInteger,
            pipeline,
            objc.sel("maxTotalThreadsPerThreadgroup"),
            .{},
        );
        const tg = @min(max_tpg, grid_size);
        objc.msgSend(void, enc, objc.sel("dispatchThreads:threadsPerThreadgroup:"), .{
            mtlSize1D(grid_size), mtlSize1D(tg),
        });
        self.insertBarrier(enc);
    }

    /// Dispatch exactly one threadgroup with conditional barrier.
    fn endEncodeOneThreadgroup(self: *MetalBackend, enc: objc.id, tg_size: usize) void {
        objc.msgSend(void, enc, objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"), .{
            mtlSize1D(1), mtlSize1D(tg_size),
        });
        self.insertBarrier(enc);
    }

    /// Commit the active command buffer and block until the GPU finishes.
    /// After this returns, all GPU writes are visible to the CPU.
    /// Sets active_cmd to null so the next getEncoder() starts a fresh buffer.
    fn flush(self: *MetalBackend) void {
        if (self.active_enc) |enc| {
            objc.msgSend(void, enc, objc.sel("endEncoding"), .{});
            self.active_enc = null;
        }
        if (self.active_cmd) |cmd| {
            objc.msgSend(void, cmd, objc.sel("commit"), .{});
            objc.msgSend(void, cmd, objc.sel("waitUntilCompleted"), .{});
            self.active_cmd = null;
        }
    }

    /// Flush pending GPU work and return a CPU backend for fallback.
    /// Call this instead of constructing CpuBackend directly so that
    /// GPU writes (e.g. rmsNorm output) are visible before the CPU reads.
    fn cpuFallback(self: *MetalBackend) CpuBackend {
        self.flush();
        return CpuBackend{ .pool = self.pool };
    }

    // ── KV cache allocation ────────────────────────────────────

    /// Allocate a KV cache slice using page-aligned memory for zero-copy GPU access.
    /// On Apple Silicon's UMA, page-aligned buffers can be wrapped via
    /// newBufferWithBytesNoCopy without any data copies. Pre-registers the
    /// allocation in buf_cache so GPU ops use the buffer without lazy creation.
    pub fn allocKvSlice(self: *MetalBackend, _: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        const byte_len = n;
        // Round up to page boundary for newBufferWithBytesNoCopy compatibility
        const aligned_bytes = std.mem.alignForward(usize, byte_len, page_size);
        const raw = std.heap.page_allocator.alloc(u8, aligned_bytes) catch return error.OutOfMemory;
        errdefer std.heap.page_allocator.free(raw);
        @memset(raw, 0);
        // Pre-register in buf_cache — wraps as Metal buffer for zero-copy GPU access.
        const addr = @intFromPtr(raw.ptr);
        const metal_buf = objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
            .{
                @as(*anyopaque, @ptrCast(raw.ptr)),
                @as(objc.NSUInteger, aligned_bytes),
                @as(objc.NSUInteger, 0), // MTLResourceStorageModeShared
                @as(?objc.id, null), // no deallocator — we manage lifetime
            },
        ) orelse return error.OutOfMemory;
        self.buf_cache.put(addr, .{ .metal_buf = metal_buf, .len = aligned_bytes }) catch {
            release(metal_buf);
            return error.OutOfMemory;
        };
        return raw[0..n];
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    /// Removes the Metal buffer from buf_cache and releases page-aligned memory.
    pub fn freeKvSlice(self: *MetalBackend, _: std.mem.Allocator, slice: []u8) void {
        if (slice.len == 0) return;
        const addr = @intFromPtr(slice.ptr);
        if (self.buf_cache.fetchRemove(addr)) |entry| {
            release(entry.value.metal_buf);
        }
        const byte_len = slice.len;
        const aligned_bytes = std.mem.alignForward(usize, byte_len, page_size);
        std.heap.page_allocator.free(@as([*]align(std.heap.pageSize()) u8, @ptrCast(@alignCast(slice.ptr)))[0..aligned_bytes]);
    }

    // ── Weight size helper ────────────────────────────────────

    const weightBytes = backend_mod.weightBytes;

    // ── GEMV ──────────────────────────────────────────────────

    /// y[n] = W[n,k] @ x[k].  Dispatches a Metal kernel per supported dtype;
    /// panics for unsupported dtypes (no silent CPU fallback).
    pub fn gemv(self: *MetalBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const pipeline: objc.id = switch (w.dtype) {
            .f32 => self.pipe_gemv_f32,
            .q8_0 => self.pipe_gemv_q8_0,
            .q4_0 => self.pipe_gemv_q4_0,
            .q4_1 => self.pipe_gemv_q4_1,
            .q4_k => self.pipe_gemv_q4_k,
            .q5_k => self.pipe_gemv_q5_k,
            .q6_k => self.pipe_gemv_q6_k,
            .q2_k => self.pipe_gemv_q2_k,
            .q3_k => self.pipe_gemv_q3_k,
            .q5_0 => self.pipe_gemv_q5_0,
            .iq4_nl => self.pipe_gemv_iq4_nl,
            .iq4_xs => self.pipe_gemv_iq4_xs,
            .bf16 => self.pipe_gemv_bf16,
            .f16 => self.pipe_gemv_f16,
            .fp8_e4m3 => self.pipe_gemv_fp8_e4m3,
            .fp8_e5m2 => self.pipe_gemv_fp8_e5m2,
            .mxfp4 => self.pipe_gemv_mxfp4,
            else => @panic("Metal GEMV: unsupported dtype — add a GPU kernel"),
        };

        const w_bytes = weightBytes(w.dtype, n, k);
        const x_ref = self.getBufRef(@ptrCast(x), k * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(w.data), w_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var k_val: u32 = @intCast(k);

        const enc = self.getEncoder(pipeline);
        setBuf(enc, x_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, y_ref, 2);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&k_val), @sizeOf(u32), 4);

        // Adaptive threadgroup size: match block count to avoid idle threads.
        // Block-based formats have nb blocks/row; threads beyond nb are wasted.
        const tg = gemvThreadgroupSize(w.dtype, k);
        const n_groups = gemvThreadgroups(w.dtype, n);
        self.endEncodeThreadgroups(enc, n_groups, tg);
    }

    /// Compute optimal threadgroup size for GEMV based on dtype and k.
    /// For block-based quantization, threads beyond block count are idle.
    /// Returns a multiple of 32 (SIMD group width) capped at threadgroup_size.
    fn gemvThreadgroupSize(dtype: DType, k: usize) usize {
        const nb: usize = switch (dtype) {
            // 32-element block formats
            .q4_0, .q4_1, .q5_0, .q8_0, .iq4_nl => (k + quant_block_elems - 1) / quant_block_elems,
            // 256-element superblock formats
            .q4_k, .q5_k, .q6_k, .q2_k, .q3_k, .iq4_xs => (k + quant_super_block_elems - 1) / quant_super_block_elems,
            // MLX 4-bit: 64-element groups, each thread processes one group
            .mlx_q => (k + mlx_group_size - 1) / mlx_group_size,
            // Element-level formats — always fully utilized at 256 threads
            else => threadgroup_size,
        };
        // Round up to SIMD group boundary, clamp to [simd_width, threadgroup_size]
        const rounded = (nb + simd_width - 1) & ~(simd_width - 1);
        return @min(threadgroup_size, @max(simd_width, rounded));
    }

    /// Number of threadgroups to dispatch for GEMV. Multi-row kernels
    /// (e.g. Q4_0 with NR=4) process multiple output rows per threadgroup.
    fn gemvThreadgroups(dtype: DType, n: usize) usize {
        return switch (dtype) {
            .q4_0 => (n + q4_0_nr - 1) / q4_0_nr,
            .q8_0 => (n + q8_0_nr - 1) / q8_0_nr,
            .q4_k, .q5_k, .q6_k => (n + q4_k_nr - 1) / q4_k_nr,
            .q2_k => (n + q2_k_nr - 1) / q2_k_nr,
            .q3_k => (n + q3_k_nr - 1) / q3_k_nr,
            .bf16 => (n + bf16_nr - 1) / bf16_nr,
            .f16 => (n + f16_nr - 1) / f16_nr,
            else => n,
        };
    }

    // ── RMS Norm ──────────────────────────────────────────────

    /// output = rms_norm(input, weight, n, eps).
    /// Fused single-dispatch: sum-of-squares + normalize in one threadgroup.
    /// Data stays in threadgroup memory between phases, avoiding extra bandwidth.
    pub fn rmsNorm(self: *MetalBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const in_ref = self.getBufRef(@ptrCast(input), n * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), n * @sizeOf(f32));
        const out_ref = self.getBufRef(@ptrCast(output), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var eps_val: f32 = eps;

        const tg = @min(threadgroup_size, n);
        const enc = self.getEncoder(self.pipe_rms_norm_fused);
        setBuf(enc, in_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, out_ref, 2);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 4);
        self.endEncodeThreadgroups(enc, 1, tg);
    }

    /// Fused add + rms_norm: a[i] = a[i] + b[i], output[i] = rms_norm(a+b, weight, eps).
    /// Single dispatch replaces separate add + rmsNorm (saves one dispatch + barrier).
    pub fn addRmsNorm(self: *MetalBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const a_ref = self.getBufRef(@ptrCast(a), n * @sizeOf(f32));
        const b_ref = self.getBufRef(@ptrCast(b), n * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), n * @sizeOf(f32));
        const out_ref = self.getBufRef(@ptrCast(output), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var eps_val: f32 = eps;

        const tg = @min(threadgroup_size, n);
        const enc = self.getEncoder(self.pipe_add_rms_norm_fused);
        setBuf(enc, a_ref, 0);
        setBuf(enc, b_ref, 1);
        setBuf(enc, w_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 5);
        self.endEncodeThreadgroups(enc, 1, tg);
    }

    // ── Element-wise dispatch helpers ─────────────────────────

    /// Dispatch a unary element-wise kernel: out[i] = f(a[i]).
    fn dispatchUnaryOp(self: *MetalBackend, pipeline: objc.id, a: [*]const f32, out: [*]f32, n: usize) void {
        const a_ref = self.getBufRef(@ptrCast(a), n * @sizeOf(f32));
        const o_ref = self.getBufRef(@ptrCast(out), n * @sizeOf(f32));
        var n_val: u32 = @intCast(n);
        const enc = self.getEncoder(pipeline);
        setBuf(enc, a_ref, 0);
        setBuf(enc, o_ref, 1);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 2);
        self.endEncode1D(enc, pipeline, n);
    }

    /// Dispatch a binary element-wise kernel: out[i] = f(a[i], b[i]).
    fn dispatchBinaryOp(self: *MetalBackend, pipeline: objc.id, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const a_ref = self.getBufRef(@ptrCast(a), n * @sizeOf(f32));
        const b_ref = self.getBufRef(@ptrCast(b), n * @sizeOf(f32));
        const o_ref = self.getBufRef(@ptrCast(out), n * @sizeOf(f32));
        var n_val: u32 = @intCast(n);
        const enc = self.getEncoder(pipeline);
        setBuf(enc, a_ref, 0);
        setBuf(enc, b_ref, 1);
        setBuf(enc, o_ref, 2);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 3);
        self.endEncode1D(enc, pipeline, n);
    }

    // ── SiLU ──────────────────────────────────────────────────

    /// output[i] = input[i] * sigmoid(input[i])
    pub fn silu(self: *MetalBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        self.dispatchUnaryOp(self.pipe_silu, input, output, n);
    }

    // ── GELU ─────────────────────────────────────────────────

    /// GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))
    pub fn gelu(self: *MetalBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        self.dispatchUnaryOp(self.pipe_gelu, input, output, n);
    }

    // ── SiLU Mul (fused) ──────────────────────────────────────

    /// out[i] = silu(a[i]) * b[i] — fused SwiGLU activation.
    /// Replaces separate silu + mul dispatches (2 dispatches → 1).
    pub fn siluMul(self: *MetalBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        self.dispatchBinaryOp(self.pipe_silu_mul, a, b, out, n);
    }

    /// out[i] = gelu(a[i]) * b[i] — fused GeGLU activation.
    pub fn geluMul(self: *MetalBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        self.dispatchBinaryOp(self.pipe_gelu_mul, a, b, out, n);
    }

    // ── Fused FFN Gate+Up+SiLU (megakernel) ──────────────────────

    /// Fused FFN: computes silu(W_gate @ x) * (W_up @ x) in a single dispatch.
    /// Replaces 3 dispatches (gate GEMV + up GEMV + siluMul) with 1.
    /// Only supports Q8_0 weights. x is f32[k], output is f32[n].
    pub fn fusedFfnGateUpSiluQ8(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q8_0, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));

        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);

        const enc = self.getEncoder(self.pipe_fused_ffn_q8);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);

        const tg = gemvThreadgroupSize(.q8_0, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN for Q4_K weights. Same as fusedFfnGateUpSiluQ8 but Q4_K quant.
    pub fn fusedFfnGateUpSiluQ4K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q4_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));

        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);

        const enc = self.getEncoder(self.pipe_fused_ffn_q4_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);

        const tg = gemvThreadgroupSize(.q4_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN for Q4_0 weights. Same pattern, Q4_0 quant.
    pub fn fusedFfnGateUpSiluQ40(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q4_0, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));

        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);

        const enc = self.getEncoder(self.pipe_fused_ffn_q4_0);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);

        const tg = gemvThreadgroupSize(.q4_0, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q4_K (Gemma 3/4). gelu(gate) * up in 1 dispatch.
    /// Fused FFN with SiLU for MLX Q4 weights (GLM-4, Qwen MLX).
    /// 10 buffer args: x, gate_w/s/b, up_w/s/b, out, n_ff, n_embd.
    pub fn fusedFfnGateUpSiluMlxQ4(
        self: *MetalBackend,
        x: [*]const f32,
        gate_w: [*]const u8,
        gate_s: [*]const u8,
        gate_b: [*]const u8,
        up_w: [*]const u8,
        up_s: [*]const u8,
        up_b: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const gpr = (n_embd + mlx_group_size - 1) / mlx_group_size;
        const w_bytes = n_ff * gpr * mlx_words_per_group_q4 * @sizeOf(u32);
        const sb_bytes = n_ff * gpr * @sizeOf(u16);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gw_ref = self.getBufRef(gate_w, w_bytes);
        const gs_ref = self.getBufRef(gate_s, sb_bytes);
        const gb_ref = self.getBufRef(gate_b, sb_bytes);
        const uw_ref = self.getBufRef(up_w, w_bytes);
        const us_ref = self.getBufRef(up_s, sb_bytes);
        const ub_ref = self.getBufRef(up_b, sb_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_silu_mlx_q4);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gw_ref, 1);
        setBuf(enc, gs_ref, 2);
        setBuf(enc, gb_ref, 3);
        setBuf(enc, uw_ref, 4);
        setBuf(enc, us_ref, 5);
        setBuf(enc, ub_ref, 6);
        setBuf(enc, out_ref, 7);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 8);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 9);
        const tg = gemvThreadgroupSize(.mlx_q, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q8_0 weights. gelu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpGeluQ8(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q8_0, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_gelu_q8);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q8_0, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q4_K (Gemma 3/4). gelu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpGeluQ4K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q4_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_gelu_q4_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q4_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q4_0 (Gemma 3). gelu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpGeluQ40(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q4_0, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_gelu_q4_0);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q4_0, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with SiLU for Q6_K weights. silu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpSiluQ6K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q6_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_q6_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q6_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q6_K weights. gelu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpGeluQ6K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q6_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_gelu_q6_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q6_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with SiLU for Q5_K weights. silu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpSiluQ5K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q5_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_q5_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q5_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    /// Fused FFN with GELU for Q5_K weights. gelu(gate) * up in 1 dispatch.
    pub fn fusedFfnGateUpGeluQ5K(
        self: *MetalBackend,
        x: [*]const f32,
        w_gate: [*]const u8,
        w_up: [*]const u8,
        ff_out: [*]f32,
        n_ff: usize,
        n_embd: usize,
    ) void {
        const w_bytes = weightBytes(.q5_k, n_ff, n_embd);
        const x_ref = self.getBufRef(@ptrCast(x), n_embd * @sizeOf(f32));
        const gate_ref = self.getBufRef(w_gate, w_bytes);
        const up_ref = self.getBufRef(w_up, w_bytes);
        const out_ref = self.getBufRef(@ptrCast(ff_out), n_ff * @sizeOf(f32));
        var nf: u32 = @intCast(n_ff);
        var ne: u32 = @intCast(n_embd);
        const enc = self.getEncoder(self.pipe_fused_ffn_gelu_q5_k);
        setBuf(enc, x_ref, 0);
        setBuf(enc, gate_ref, 1);
        setBuf(enc, up_ref, 2);
        setBuf(enc, out_ref, 3);
        setBytes(enc, @ptrCast(&nf), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&ne), @sizeOf(u32), 5);
        const tg = gemvThreadgroupSize(.q5_k, n_embd);
        self.endEncodeThreadgroups(enc, n_ff, tg);
    }

    // ── True Megakernel Dispatch ──────────────────────────────────

    /// Dispatch the Qwen 3.5 Q8_0 true megakernel: single launch for all layers.
    /// All 8 buffers bound directly; params passed as constant buffer.
    /// weights: mmap'd GGUF weight base pointer (buffer 0).
    /// layer_offsets: [n_layers * 160 bytes] packed LayerOffsets (buffer 1).
    /// kv_keys/kv_values: KV cache (buffers 2-3, Phase 2).
    /// hidden: [n_embd] f32 hidden state (buffer 4).
    /// scratch: intermediate buffers (buffer 5).
    /// sync_ctrs: [32] atomic_uint grid sync counters (buffer 6).
    /// params: MegaQwen35Params struct (buffer 7).
    pub fn dispatchMegakernelQwen35Q8(
        self: *MetalBackend,
        weights: [*]const u8,
        weights_size: usize,
        layer_offsets: [*]const u8,
        layer_offsets_size: usize,
        kv_keys: [*]f32,
        kv_keys_size: usize,
        kv_values: [*]f32,
        kv_values_size: usize,
        hidden: [*]f32,
        hidden_size: usize,
        scratch: [*]f32,
        scratch_size: usize,
        sync_ctrs: [*]f32,
        sync_ctrs_size: usize,
        params: *const anyopaque,
        params_size: usize,
        n_tgs: u32,
    ) void {
        const w_ref = self.getBufRef(weights, weights_size);
        const lo_ref = self.getBufRef(layer_offsets, layer_offsets_size);
        const kk_ref = self.getBufRef(@ptrCast(kv_keys), kv_keys_size);
        const kv_ref = self.getBufRef(@ptrCast(kv_values), kv_values_size);
        const h_ref = self.getBufRef(@ptrCast(hidden), hidden_size);
        const s_ref = self.getBufRef(@ptrCast(scratch), scratch_size);
        const sc_ref = self.getBufRef(@ptrCast(sync_ctrs), sync_ctrs_size);

        const enc = self.getEncoder(self.pipe_mega_qwen35_q8);
        setBuf(enc, w_ref, 0);
        setBuf(enc, lo_ref, 1);
        setBuf(enc, kk_ref, 2);
        setBuf(enc, kv_ref, 3);
        setBuf(enc, h_ref, 4);
        setBuf(enc, s_ref, 5);
        setBuf(enc, sc_ref, 6);
        setBytes(enc, params, params_size, 7);

        self.endEncodeThreadgroups(enc, n_tgs, threadgroup_size);
    }

    /// Dispatch the Gemma 3/4 Q4_K true megakernel: single launch for all layers.
    /// Same 8-buffer binding as Qwen 3.5, but with Gemma-specific params struct.
    pub fn dispatchMegakernelGemmaQ4K(
        self: *MetalBackend,
        weights: [*]const u8,
        weights_size: usize,
        layer_offsets: [*]const u8,
        layer_offsets_size: usize,
        kv_keys: [*]f32,
        kv_keys_size: usize,
        kv_values: [*]f32,
        kv_values_size: usize,
        hidden: [*]f32,
        hidden_size: usize,
        scratch: [*]f32,
        scratch_size: usize,
        sync_ctrs: [*]f32,
        sync_ctrs_size: usize,
        params: *const anyopaque,
        params_size: usize,
        n_tgs: u32,
    ) void {
        const w_ref = self.getBufRef(weights, weights_size);
        const lo_ref = self.getBufRef(layer_offsets, layer_offsets_size);
        const kk_ref = self.getBufRef(@ptrCast(kv_keys), kv_keys_size);
        const kv_ref = self.getBufRef(@ptrCast(kv_values), kv_values_size);
        const h_ref = self.getBufRef(@ptrCast(hidden), hidden_size);
        const s_ref = self.getBufRef(@ptrCast(scratch), scratch_size);
        const sc_ref = self.getBufRef(@ptrCast(sync_ctrs), sync_ctrs_size);

        const enc = self.getEncoder(self.pipe_mega_gemma_q4k);
        setBuf(enc, w_ref, 0);
        setBuf(enc, lo_ref, 1);
        setBuf(enc, kk_ref, 2);
        setBuf(enc, kv_ref, 3);
        setBuf(enc, h_ref, 4);
        setBuf(enc, s_ref, 5);
        setBuf(enc, sc_ref, 6);
        setBytes(enc, params, params_size, 7);

        self.endEncodeThreadgroups(enc, n_tgs, threadgroup_size);
    }

    /// Dispatch the Qwen 3.5 Q4_K true megakernel: single launch for all layers.
    /// Same structure as Q8_0 variant but with Q4_K dequantization in GEMV stages.
    pub fn dispatchMegakernelQwen35Q4K(
        self: *MetalBackend,
        weights: [*]const u8,
        weights_size: usize,
        layer_offsets: [*]const u8,
        layer_offsets_size: usize,
        kv_keys: [*]f32,
        kv_keys_size: usize,
        kv_values: [*]f32,
        kv_values_size: usize,
        hidden: [*]f32,
        hidden_size: usize,
        scratch: [*]f32,
        scratch_size: usize,
        sync_ctrs: [*]f32,
        sync_ctrs_size: usize,
        params: *const anyopaque,
        params_size: usize,
        n_tgs: u32,
    ) void {
        const w_ref = self.getBufRef(weights, weights_size);
        const lo_ref = self.getBufRef(layer_offsets, layer_offsets_size);
        const kk_ref = self.getBufRef(@ptrCast(kv_keys), kv_keys_size);
        const kv_ref = self.getBufRef(@ptrCast(kv_values), kv_values_size);
        const h_ref = self.getBufRef(@ptrCast(hidden), hidden_size);
        const s_ref = self.getBufRef(@ptrCast(scratch), scratch_size);
        const sc_ref = self.getBufRef(@ptrCast(sync_ctrs), sync_ctrs_size);

        const enc = self.getEncoder(self.pipe_mega_qwen35_q4k);
        setBuf(enc, w_ref, 0);
        setBuf(enc, lo_ref, 1);
        setBuf(enc, kk_ref, 2);
        setBuf(enc, kv_ref, 3);
        setBuf(enc, h_ref, 4);
        setBuf(enc, s_ref, 5);
        setBuf(enc, sc_ref, 6);
        setBytes(enc, params, params_size, 7);

        self.endEncodeThreadgroups(enc, n_tgs, threadgroup_size);
    }

    /// Dispatch the Nemotron-H Q8_0 true megakernel: single launch for
    /// attention and FFN-only layers. SSM layers break out.
    /// Buffer 8 carries the per-layer type array (u32 per layer).
    pub fn dispatchMegakernelNemotronHQ8(
        self: *MetalBackend,
        weights: [*]const u8,
        weights_size: usize,
        layer_offsets: [*]const u8,
        layer_offsets_size: usize,
        kv_keys: [*]f32,
        kv_keys_size: usize,
        kv_values: [*]f32,
        kv_values_size: usize,
        hidden: [*]f32,
        hidden_size: usize,
        scratch: [*]f32,
        scratch_size: usize,
        sync_ctrs: [*]f32,
        sync_ctrs_size: usize,
        params: *const anyopaque,
        params_size: usize,
        layer_types: [*]const u32,
        layer_types_size: usize,
        n_tgs: u32,
    ) void {
        const w_ref = self.getBufRef(weights, weights_size);
        const lo_ref = self.getBufRef(layer_offsets, layer_offsets_size);
        const kk_ref = self.getBufRef(@ptrCast(kv_keys), kv_keys_size);
        const kv_ref = self.getBufRef(@ptrCast(kv_values), kv_values_size);
        const h_ref = self.getBufRef(@ptrCast(hidden), hidden_size);
        const s_ref = self.getBufRef(@ptrCast(scratch), scratch_size);
        const sc_ref = self.getBufRef(@ptrCast(sync_ctrs), sync_ctrs_size);
        const lt_ref = self.getBufRef(@ptrCast(layer_types), layer_types_size);

        const enc = self.getEncoder(self.pipe_mega_nemotron_h_q8);
        setBuf(enc, w_ref, 0);
        setBuf(enc, lo_ref, 1);
        setBuf(enc, kk_ref, 2);
        setBuf(enc, kv_ref, 3);
        setBuf(enc, h_ref, 4);
        setBuf(enc, s_ref, 5);
        setBuf(enc, sc_ref, 6);
        setBytes(enc, params, params_size, 7);
        setBuf(enc, lt_ref, 8);

        self.endEncodeThreadgroups(enc, n_tgs, threadgroup_size);
    }

    // ── Composed Megakernel (auto-generated from model metadata) ──

    /// Compile a composed megakernel from MSL source generated by mega_compose.zig.
    /// The source is appended to the base MSL (which provides building blocks).
    /// Call this at model init when --megakernel is enabled.
    pub fn compileComposedMegakernel(self: *MetalBackend, composed_msl: []const u8) !void {
        const NSString = objc.getClass("NSString") orelse return error.NoFoundation;

        // Concatenate base MSL (building blocks) + composed kernel
        const full_source = msl_source ++ composed_msl;

        const source_ns = objc.msgSend(
            ?objc.id,
            NSString,
            objc.sel("stringWithUTF8String:"),
            .{@as([*:0]const u8, @ptrCast(full_source.ptr))},
        ) orelse return error.StringFailed;

        var compile_err: ?objc.id = null;
        const lib = objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newLibraryWithSource:options:error:"),
            .{ source_ns, @as(?objc.id, null), @as(*?objc.id, &compile_err) },
        ) orelse {
            if (compile_err) |err_obj| {
                const desc_ns = objc.msgSend(?objc.id, err_obj, objc.sel("localizedDescription"), .{});
                if (desc_ns) |ns| {
                    const cstr = objc.msgSend(?[*:0]const u8, ns, objc.sel("UTF8String"), .{});
                    if (cstr) |d| std.log.err("Composed megakernel compile error: {s}", .{d});
                }
            }
            return error.ShaderCompileFailed;
        };
        defer release(lib);

        // Look up the auto-generated kernel function
        const fn_name_ns = objc.msgSend(
            ?objc.id,
            NSString,
            objc.sel("stringWithUTF8String:"),
            .{@as([*:0]const u8, "megakernel_auto")},
        ) orelse return error.StringFailed;

        const func = objc.msgSend(
            ?objc.id,
            lib,
            objc.sel("newFunctionWithName:"),
            .{fn_name_ns},
        ) orelse return error.FunctionNotFound;
        defer release(func);

        var pipe_err: ?objc.id = null;
        self.pipe_mega_auto = objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ func, @as(*?objc.id, &pipe_err) },
        ) orelse {
            if (pipe_err) |err_obj| {
                const desc_ns = objc.msgSend(?objc.id, err_obj, objc.sel("localizedDescription"), .{});
                if (desc_ns) |ns| {
                    const cstr = objc.msgSend(?[*:0]const u8, ns, objc.sel("UTF8String"), .{});
                    if (cstr) |d| std.log.err("Composed megakernel pipeline error: {s}", .{d});
                }
            }
            return error.PipelineCreationFailed;
        };
    }

    /// Dispatch the auto-composed megakernel. Same 8-buffer binding as the
    /// hand-written megakernels.
    pub fn dispatchMegakernelAuto(
        self: *MetalBackend,
        weights: [*]const u8,
        weights_size: usize,
        layer_offsets: [*]const u8,
        layer_offsets_size: usize,
        kv_keys: [*]f32,
        kv_keys_size: usize,
        kv_values: [*]f32,
        kv_values_size: usize,
        hidden: [*]f32,
        hidden_size: usize,
        scratch: [*]f32,
        scratch_size: usize,
        sync_ctrs: [*]f32,
        sync_ctrs_size: usize,
        params: *const anyopaque,
        params_size: usize,
        n_tgs: u32,
    ) void {
        const pipe = self.pipe_mega_auto orelse @panic("composed megakernel not compiled — call compileComposedMegakernel first");
        const w_ref = self.getBufRef(weights, weights_size);
        const lo_ref = self.getBufRef(layer_offsets, layer_offsets_size);
        const kk_ref = self.getBufRef(@ptrCast(kv_keys), kv_keys_size);
        const kv_ref = self.getBufRef(@ptrCast(kv_values), kv_values_size);
        const h_ref = self.getBufRef(@ptrCast(hidden), hidden_size);
        const s_ref = self.getBufRef(@ptrCast(scratch), scratch_size);
        const sc_ref = self.getBufRef(@ptrCast(sync_ctrs), sync_ctrs_size);

        const enc = self.getEncoder(pipe);
        setBuf(enc, w_ref, 0);
        setBuf(enc, lo_ref, 1);
        setBuf(enc, kk_ref, 2);
        setBuf(enc, kv_ref, 3);
        setBuf(enc, h_ref, 4);
        setBuf(enc, s_ref, 5);
        setBuf(enc, sc_ref, 6);
        setBytes(enc, params, params_size, 7);

        self.endEncodeThreadgroups(enc, n_tgs, threadgroup_size);
    }

    // ── Per-Head RMS Norm ──────────────────────────────────────

    /// In-place rmsNorm applied to n_heads independent heads (each of head_dim
    /// elements), using the same weight vector. Single dispatch:
    /// n_heads threadgroups × head_dim threads (capped at 256).
    pub fn rmsNormMulti(self: *MetalBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        const total = n_heads * head_dim;
        const d_ref = self.getBufRef(@ptrCast(data), total * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), head_dim * @sizeOf(f32));

        var hd_val: u32 = @intCast(head_dim);
        var eps_val: f32 = eps;

        const tg = @min(threadgroup_size, head_dim);
        const enc = self.getEncoder(self.pipe_rms_norm_fused);
        setBuf(enc, d_ref, 0); // input
        setBuf(enc, w_ref, 1); // weight
        setBuf(enc, d_ref, 2); // output (in-place)
        setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 4);
        self.endEncodeThreadgroups(enc, n_heads, tg);
    }

    // ── Sigmoid Mul ──────────────────────────────────────────

    /// data[i] *= sigmoid(gate[i]) — in-place sigmoid-gated multiply.
    pub fn sigmoidMul(self: *MetalBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        self.dispatchBinaryOp(self.pipe_sigmoid_mul, @ptrCast(data), gate, data, n);
    }

    // ── Deinterleave ─────────────────────────────────────────

    /// Split interleaved pairs: input[n_pairs * 2 * stride] → out_a[n_pairs * stride] + out_b[n_pairs * stride].
    /// For each pair h: out_a[h*stride..+stride] = input[h*2*stride..+stride],
    ///                  out_b[h*stride..+stride] = input[h*2*stride+stride..+stride].
    pub fn deinterleave(self: *MetalBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        const total = n_pairs * stride;
        const in_ref = self.getBufRef(@ptrCast(input), n_pairs * 2 * stride * @sizeOf(f32));
        const a_ref = self.getBufRef(@ptrCast(out_a), total * @sizeOf(f32));
        const b_ref = self.getBufRef(@ptrCast(out_b), total * @sizeOf(f32));

        var stride_val: u32 = @intCast(stride);
        var pairs_val: u32 = @intCast(n_pairs);

        const enc = self.getEncoder(self.pipe_deinterleave);
        setBuf(enc, in_ref, 0);
        setBuf(enc, a_ref, 1);
        setBuf(enc, b_ref, 2);
        setBytes(enc, @ptrCast(&stride_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&pairs_val), @sizeOf(u32), 4);
        self.endEncode1D(enc, self.pipe_deinterleave, total);
    }

    /// Split concatenated Q+gate per head into separate Q and gate arrays.
    /// Input layout: [Q0..Q_{hd-1}, G0..G_{hd-1}] × nh heads.
    /// Output: q_out[nh*hd], g_out[nh*hd].
    pub fn splitQGate(self: *MetalBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, hd: usize, nh: usize) void {
        const total = nh * hd;
        const in_ref = self.getBufRef(@ptrCast(qg), nh * hd * 2 * @sizeOf(f32));
        const q_ref = self.getBufRef(@ptrCast(q_out), total * @sizeOf(f32));
        const g_ref = self.getBufRef(@ptrCast(g_out), total * @sizeOf(f32));
        var hd_val: u32 = @intCast(hd);
        var nh_val: u32 = @intCast(nh);
        const enc = self.getEncoder(self.pipe_split_qgate);
        setBuf(enc, in_ref, 0);
        setBuf(enc, q_ref, 1);
        setBuf(enc, g_ref, 2);
        setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 4);
        self.endEncode1D(enc, self.pipe_split_qgate, total);
    }

    // ── Add ───────────────────────────────────────────────────

    /// out[i] = a[i] + b[i]
    pub fn add(self: *MetalBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        self.dispatchBinaryOp(self.pipe_add, a, b, out, n);
    }

    // ── Transposed GEMV (Q8_0) ─────────────────────────────────

    /// Transposed GEMV: y[out_dim] = W^T @ x[in_dim] for Q8_0 3D multi-head weights.
    /// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
    pub fn gemvT(self: *MetalBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const x_ref = self.getBufRef(@ptrCast(x), in_dim * @sizeOf(f32));
        const blocks_per_row = (out_dim + simd_width - 1) / simd_width;
        const w_bytes = in_dim * blocks_per_row * q8_0_block_bytes;
        const w_ref = self.getBufRef(w, w_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), out_dim * @sizeOf(f32));
        var od: u32 = @intCast(out_dim);
        var id: u32 = @intCast(in_dim);
        const enc = self.getEncoder(self.pipe_gemv_t_q8_0);
        setBuf(enc, x_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, y_ref, 2);
        setBytes(enc, @ptrCast(&od), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&id), @sizeOf(u32), 4);
        // One threadgroup per output element
        self.endEncode1D(enc, self.pipe_gemv_t_q8_0, out_dim);
    }

    // ── Add Scaled ────────────────────────────────────────────

    /// dst[i] += src[i] * scale
    pub fn addScaled(self: *MetalBackend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        const src_ref = self.getBufRef(@ptrCast(src), n * @sizeOf(f32));
        const dst_ref = self.getBufRef(@ptrCast(dst), n * @sizeOf(f32));
        var s = scale;
        var n_val: u32 = @intCast(n);
        const enc = self.getEncoder(self.pipe_add_scaled);
        setBuf(enc, src_ref, 0);
        setBuf(enc, dst_ref, 1);
        setBytes(enc, @ptrCast(&s), @sizeOf(f32), 2);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 3);
        self.endEncode1D(enc, self.pipe_add_scaled, n);
    }

    // ── Mul ───────────────────────────────────────────────────

    /// out[i] = a[i] * b[i]
    pub fn mul(self: *MetalBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        self.dispatchBinaryOp(self.pipe_mul, a, b, out, n);
    }

    // ── Softmax ───────────────────────────────────────────────

    /// In-place softmax.
    /// Three passes: (1) find max, (2) exp(x - max) and sum, (3) divide by sum.
    /// All three passes share one command buffer — single commit instead of three.
    /// Uses scratch_buf as two adjacent f32 slots: [0..4]=max, [4..8]=sum.
    pub fn softmax(self: *MetalBackend, data: [*]f32, n: usize) void {
        // For very small n the GPU dispatch overhead isn't worth it.
        if (n < softmax_cpu_threshold) {
            var cpu = self.cpuFallback();
            cpu.softmax(data, n);
            return;
        }

        const d_ref = self.getBufRef(@ptrCast(data), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);

        // Pass 1: max reduction → scratch_buf offset 0
        const enc1 = self.getEncoder(self.pipe_softmax_max);
        setBuf(enc1, d_ref, 0);
        setBufferOffset(enc1, self.scratch_buf, 0, 1); // max_out
        setBytes(enc1, @ptrCast(&n_val), @sizeOf(u32), 2);
        self.endEncodeOneThreadgroup(enc1, threadgroup_size);

        // Pass 2: exp(x - max) in-place, sum → scratch_buf offset 4
        // (same command buffer — no commit between passes)
        const enc2 = self.getEncoder(self.pipe_softmax_exp_sum);
        setBuf(enc2, d_ref, 0);
        setBufferOffset(enc2, self.scratch_buf, 0, 1); // max_buf  (read)
        setBufferOffset(enc2, self.scratch_buf, 4, 2); // sum_out  (write)
        setBytes(enc2, @ptrCast(&n_val), @sizeOf(u32), 3);
        self.endEncodeOneThreadgroup(enc2, threadgroup_size);

        // Pass 3: divide by sum
        // (same command buffer — no commit between passes)
        const enc3 = self.getEncoder(self.pipe_softmax_div);
        setBuf(enc3, d_ref, 0);
        setBufferOffset(enc3, self.scratch_buf, 4, 1); // sum_buf
        setBytes(enc3, @ptrCast(&n_val), @sizeOf(u32), 2);
        self.endEncode1D(enc3, self.pipe_softmax_div, n);
    }

    // ── RoPE ──────────────────────────────────────────────────

    /// Apply rotary position embedding in-place.
    /// Grid = n_heads × rope_dim / 2 threads; each thread rotates one (re, im) pair.
    pub fn rope(self: *MetalBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const total = n_heads * head_dim;
        const x_ref = self.getBufRef(@ptrCast(x), total * @sizeOf(f32));

        var pos_val: u32 = @intCast(pos);
        var nh_val: u32 = @intCast(n_heads);
        var hd_val: u32 = @intCast(head_dim);
        var rd_val: u32 = @intCast(rope_dim);
        var theta_val: f32 = theta;

        const grid = n_heads * rope_dim / 2;
        const enc = self.getEncoder(self.pipe_rope);
        setBuf(enc, x_ref, 0);
        setBytes(enc, @ptrCast(&pos_val), @sizeOf(u32), 1);
        setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 2);
        setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&rd_val), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&theta_val), @sizeOf(f32), 5);
        self.endEncode1D(enc, self.pipe_rope, grid);
    }

    // ── Embedding lookup — CPU fallback ───────────────────────

    /// Embedding lookup is a single-row read; CPU is faster than GPU dispatch overhead.
    pub fn embLookup(self: *MetalBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        var cpu = self.cpuFallback();
        cpu.embLookup(table, token_id, output, dim);
    }

    // ── L2 Norm ───────────────────────────────────────────────

    /// L2 normalize in-place.
    /// Two-pass: (1) sum-of-squares via rms_norm_ss kernel, (2) l2_norm_apply.
    /// Both passes share one command buffer — single commit instead of two.
    pub fn l2Norm(self: *MetalBackend, x: [*]f32, n: usize, eps: f32) void {
        const x_ref = self.getBufRef(@ptrCast(x), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var eps_val: f32 = eps;

        // Pass 1: sum of squares → scratch_buf[0]
        const enc1 = self.getEncoder(self.pipe_rms_ss);
        setBuf(enc1, x_ref, 0);
        setBuffer(enc1, self.scratch_buf, 1);
        setBytes(enc1, @ptrCast(&n_val), @sizeOf(u32), 2);
        self.endEncodeOneThreadgroup(enc1, threadgroup_size);

        // Pass 2: normalize in-place
        // (same command buffer — no commit between passes)
        const enc2 = self.getEncoder(self.pipe_l2_apply);
        setBuf(enc2, x_ref, 0);
        setBuffer(enc2, self.scratch_buf, 1);
        setBytes(enc2, @ptrCast(&n_val), @sizeOf(u32), 2);
        setBytes(enc2, @ptrCast(&eps_val), @sizeOf(f32), 3);
        self.endEncode1D(enc2, self.pipe_l2_apply, n);
    }

    // ── NVFP4 SafeTensors GEMV ─────────────────────────────────

    /// NVFP4 SafeTensors GEMV with GPU acceleration.
    /// Weight nibbles and FP8 E4M3 scales are in separate buffers.
    /// One threadgroup per output row, 256 threads per threadgroup.
    pub fn gemvNvfp4St(self: *MetalBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        const w_bytes = n * (k / 2);
        const s_bytes = n * (k / 16);
        const x_ref = self.getBufRef(@ptrCast(x), k * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), w_bytes);
        const s_ref = self.getBufRef(@ptrCast(scale), s_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var k_val: u32 = @intCast(k);

        const enc = self.getEncoder(self.pipe_gemv_nvfp4_st);
        setBuf(enc, x_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, s_ref, 2);
        setBuf(enc, y_ref, 3);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&k_val), @sizeOf(u32), 5);
        self.endEncodeThreadgroups(enc, n, threadgroup_size);
    }

    /// MLX affine 4-bit quantized GEMV on GPU.
    /// Dispatches to a native Metal kernel for the 3-buffer MLX-Q layout
    /// (packed u32 weights + bf16 scales + bf16 biases, group_size=64).
    pub fn gemvMlxQ(self: *MetalBackend, x: [*]const f32, weight: [*]const u8, scales: [*]const u8, biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        if (bits != 4 and bits != 6 and bits != 8) @panic("Metal MLX GEMV: unsupported bit width");
        const gpr = (k + mlx_group_size - 1) / mlx_group_size;
        const wpg: usize = switch (bits) {
            8 => mlx_words_per_group_q8,
            6 => mlx_words_per_group_q6,
            else => mlx_words_per_group_q4,
        };
        const w_bytes = n * gpr * wpg * @sizeOf(u32);
        const sb_bytes = n * gpr * @sizeOf(u16); // bf16 scales/biases

        const x_ref = self.getBufRef(@ptrCast(x), k * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), w_bytes);
        const s_ref = self.getBufRef(@ptrCast(scales), sb_bytes);
        const b_ref = self.getBufRef(@ptrCast(biases), sb_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var k_val: u32 = @intCast(k);

        const pipe = switch (bits) {
            8 => self.pipe_gemv_mlx_q8,
            6 => self.pipe_gemv_mlx_q6,
            else => self.pipe_gemv_mlx_q4,
        };
        const enc = self.getEncoder(pipe);
        setBuf(enc, x_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, s_ref, 2);
        setBuf(enc, b_ref, 3);
        setBuf(enc, y_ref, 4);
        setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 5);
        setBytes(enc, @ptrCast(&k_val), @sizeOf(u32), 6);
        self.endEncodeThreadgroups(enc, n, gemvThreadgroupSize(.mlx_q, k));
    }

    /// MXFP4 SafeTensors GEMV on GPU.
    /// U32-packed nibbles with FP8 E4M3 per-group scales (no bias).
    /// group_size=32, 4 words per group (8 nibbles per word × 4 = 32 values).
    pub fn gemvMxfp4St(self: *MetalBackend, x: [*]const f32, weight: [*]const u8, scale: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        const gpr = (k + mxfp4_group_size - 1) / mxfp4_group_size;
        const wpg: usize = mxfp4_words_per_group;
        const w_bytes = n * gpr * wpg * @sizeOf(u32);
        const s_bytes = n * gpr; // U8 FP8 E4M3 scales

        const x_ref = self.getBufRef(@ptrCast(x), k * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), w_bytes);
        const s_ref = self.getBufRef(@ptrCast(scale), s_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), n * @sizeOf(f32));

        var n_val: u32 = @intCast(n);
        var k_val: u32 = @intCast(k);

        const enc_m = self.getEncoder(self.pipe_gemv_mxfp4_st);
        setBuf(enc_m, x_ref, 0);
        setBuf(enc_m, w_ref, 1);
        setBuf(enc_m, s_ref, 2);
        setBuf(enc_m, y_ref, 3);
        setBytes(enc_m, @ptrCast(&n_val), @sizeOf(u32), 4);
        setBytes(enc_m, @ptrCast(&k_val), @sizeOf(u32), 5);
        self.endEncodeThreadgroups(enc_m, n, gemvThreadgroupSize(.mxfp4, k));
    }

    /// Batched GEMV: dispatches all ops sharing input x without inter-dispatch
    /// memory barriers. Only a single barrier is placed after the last dispatch.
    /// This allows the GPU to overlap independent GEMV dispatches.
    pub fn gemvMulti(self: *MetalBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        if (ops.len == 0) return;
        const x_ref = self.getBufRef(@ptrCast(x), k * @sizeOf(f32));
        var k_val: u32 = @intCast(k);

        for (ops, 0..) |op, idx| {
            const is_last = (idx == ops.len - 1);

            // MLX quantized weights — dispatch MLX-Q kernel with companion buffers
            if (op.mlx_scales != null) {
                const bits = op.mlx_bits;
                const gpr = (k + mlx_group_size - 1) / mlx_group_size;
                const wpg: usize = switch (bits) {
                    8 => mlx_words_per_group_q8,
                    6 => mlx_words_per_group_q6,
                    else => mlx_words_per_group_q4,
                };
                const w_bytes_mlx = op.n * gpr * wpg * @sizeOf(u32);
                const sb_bytes = op.n * gpr * @sizeOf(u16); // bf16 scales/biases

                const w_ref = self.getBufRef(@ptrCast(op.w.data), w_bytes_mlx);
                const y_ref = self.getBufRef(@ptrCast(op.y), op.n * @sizeOf(f32));
                const s_ref = self.getBufRef(@ptrCast(op.mlx_scales.?), sb_bytes);
                const b_ref = self.getBufRef(@ptrCast(op.mlx_biases.?), sb_bytes);
                var n_val: u32 = @intCast(op.n);

                const pipe = switch (bits) {
                    8 => self.pipe_gemv_mlx_q8,
                    6 => self.pipe_gemv_mlx_q6,
                    else => self.pipe_gemv_mlx_q4,
                };
                const enc = self.getEncoder(pipe);
                setBuf(enc, x_ref, 0);
                setBuf(enc, w_ref, 1);
                setBuf(enc, s_ref, 2);
                setBuf(enc, b_ref, 3);
                setBuf(enc, y_ref, 4);
                setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 5);
                setBytes(enc, @ptrCast(&k_val), @sizeOf(u32), 6);

                const tg = gemvThreadgroupSize(.mlx_q, k);
                if (is_last) {
                    self.endEncodeThreadgroups(enc, op.n, tg);
                } else {
                    objc.msgSend(void, enc, objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"), .{
                        mtlSize1D(op.n), mtlSize1D(tg),
                    });
                    if (self.profile_counters) self.dispatch_count += 1;
                }
                continue;
            }

            const pipeline: objc.id = switch (op.w.dtype) {
                .f32 => self.pipe_gemv_f32,
                .q8_0 => self.pipe_gemv_q8_0,
                .q4_0 => self.pipe_gemv_q4_0,
                .q4_1 => self.pipe_gemv_q4_1,
                .q4_k => self.pipe_gemv_q4_k,
                .q5_k => self.pipe_gemv_q5_k,
                .q6_k => self.pipe_gemv_q6_k,
                .q2_k => self.pipe_gemv_q2_k,
                .q3_k => self.pipe_gemv_q3_k,
                .q5_0 => self.pipe_gemv_q5_0,
                .iq4_nl => self.pipe_gemv_iq4_nl,
                .iq4_xs => self.pipe_gemv_iq4_xs,
                .bf16 => self.pipe_gemv_bf16,
                .f16 => self.pipe_gemv_f16,
                .fp8_e4m3 => self.pipe_gemv_fp8_e4m3,
                .fp8_e5m2 => self.pipe_gemv_fp8_e5m2,
                .mxfp4 => self.pipe_gemv_mxfp4,
                else => {
                    self.gemv(x, op.w, op.y, op.n, k);
                    continue;
                },
            };

            const w_bytes = weightBytes(op.w.dtype, op.n, k);
            const w_ref = self.getBufRef(@ptrCast(op.w.data), w_bytes);
            const y_ref = self.getBufRef(@ptrCast(op.y), op.n * @sizeOf(f32));

            var n_val: u32 = @intCast(op.n);
            const enc = self.getEncoder(pipeline);
            setBuf(enc, x_ref, 0);
            setBuf(enc, w_ref, 1);
            setBuf(enc, y_ref, 2);
            setBytes(enc, @ptrCast(&n_val), @sizeOf(u32), 3);
            setBytes(enc, @ptrCast(&k_val), @sizeOf(u32), 4);

            const tg = gemvThreadgroupSize(op.w.dtype, k);
            const n_groups = gemvThreadgroups(op.w.dtype, op.n);
            if (is_last) {
                self.endEncodeThreadgroups(enc, n_groups, tg);
            } else {
                objc.msgSend(void, enc, objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"), .{
                    mtlSize1D(n_groups), mtlSize1D(tg),
                });
                if (self.profile_counters) self.dispatch_count += 1;
            }
        }
    }

    // ── Sync ─────────────────────────────────────────────────

    /// Begin a batch of independent GPU dispatches. Suppresses per-dispatch
    /// memory barriers so the GPU can overlap execution of independent ops.
    pub fn beginBatch(self: *MetalBackend) void {
        self.batch_mode = true;
    }

    /// End a batch and insert a single memory barrier for all preceding dispatches.
    pub fn endBatch(self: *MetalBackend) void {
        self.batch_mode = false;
        if (self.active_enc) |enc|
            objc.msgSend(void, enc, objc.sel("memoryBarrierWithScope:"), .{barrier_scope_buffers});
    }

    /// Commit all pending GPU commands and wait for completion.
    /// Call before CPU code reads from buffers written by GPU ops.
    /// On Apple Silicon's unified memory, this is the only synchronization
    /// needed — zero-copy buffers mean no data transfer, just a fence.
    pub fn sync(self: *MetalBackend) void {
        self.flush();
        if (self.profile_counters) self.sync_count += 1;
    }

    /// Reset dispatch counters and enable counting (call at start of profiled token).
    pub fn resetCounters(self: *MetalBackend) void {
        self.dispatch_count = 0;
        self.barrier_count = 0;
        self.sync_count = 0;
        self.profile_counters = true;
    }

    // ── SDPA ─────────────────────────────────────────────────

    /// Scaled dot-product attention with KV cache append.
    /// Appends k_new/v_new to KV cache, then runs FlashAttention-2 on GPU.
    /// Supports f32 KV cache (existing fast path) and TurboQuant 2/3/4-bit
    /// KV cache (native GPU dequant — no CPU fallback for SDPA compute).
    /// KV append for turbo types uses CPU quantization (once per token per layer,
    /// not the SDPA hot path). Panics on non-f32, non-turbo KV types.
    /// Panics for sequences > 4096 or head dims > 256.
    pub fn sdpa(self: *MetalBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: backend_mod.KvQuantType, kv_type_v: backend_mod.KvQuantType) void {
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Non-turbo, non-f32 quantized KV: not supported — add GPU kernel or use --kv-type f32
        if ((!is_f32_k and !is_turbo_k) or (!is_f32_v and !is_turbo_v))
            @panic("Metal SDPA: unsupported KV type — use --kv-type f32 or turbo2/3/4");

        const kvd = nkv * hd;
        const sl = seq_len + 1;

        if (sl > sdpa_max_seq_len) @panic("Metal SDPA: sequence length exceeds GPU limit (" ++ std.fmt.comptimePrint("{d}", .{sdpa_max_seq_len}) ++ ") — reduce --ctx-size");
        if (hd > sdpa_max_head_dim) @panic("Metal SDPA: head_dim exceeds GPU limit (" ++ std.fmt.comptimePrint("{d}", .{sdpa_max_head_dim}) ++ ")");

        // ── KV append ──
        if (is_f32_k and is_f32_v) {
            // f32 path: GPU KV append via compute kernel
            const f32_keys: [*]f32 = @ptrCast(@alignCast(keys.ptr));
            const f32_values: [*]f32 = @ptrCast(@alignCast(values.ptr));
            const k_ref = self.getBufRef(@ptrCast(k_new), kvd * @sizeOf(f32));
            const v_ref = self.getBufRef(@ptrCast(v_new), kvd * @sizeOf(f32));
            const keys_ref = self.getBufRef(@ptrCast(f32_keys), sl * kvd * @sizeOf(f32));
            const vals_ref = self.getBufRef(@ptrCast(f32_values), sl * kvd * @sizeOf(f32));
            const enc = self.getEncoder(self.pipe_kv_append);
            setBuf(enc, k_ref, 0);
            setBuf(enc, v_ref, 1);
            setBuf(enc, keys_ref, 2);
            setBuf(enc, vals_ref, 3);
            var kvd_u: u32 = @intCast(kvd);
            var kv_off_u: u32 = @intCast(seq_len * kvd);
            setBytes(enc, @ptrCast(&kvd_u), @sizeOf(u32), 4);
            setBytes(enc, @ptrCast(&kv_off_u), @sizeOf(u32), 5);
            self.endEncode1D(enc, self.pipe_kv_append, kvd);
        } else {
            // Turbo/mixed path: CPU quantization for KV append (once per token,
            // not the SDPA hot path — acceptable per AGENTS.md).
            self.sync();
            const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
            const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
            kv_quant.kvStore(keys.ptr + k_off, k_new, kvd, kv_type_k);
            kv_quant.kvStore(values.ptr + v_off, v_new, kvd, kv_type_v);
        }

        // ── GPU SDPA: FlashAttention-2 ──
        if (is_f32_k and is_f32_v) {
            // Pure f32 path: use existing sdpa_fa2 kernel
            const f32_keys: [*]f32 = @ptrCast(@alignCast(keys.ptr));
            const f32_values: [*]f32 = @ptrCast(@alignCast(values.ptr));
            const q_ref = self.getBufRef(@ptrCast(q), nh * hd * @sizeOf(f32));
            const keys_ref = self.getBufRef(@ptrCast(f32_keys), sl * kvd * @sizeOf(f32));
            const vals_ref = self.getBufRef(@ptrCast(f32_values), sl * kvd * @sizeOf(f32));
            const out_ref = self.getBufRef(@ptrCast(output), nh * hd * @sizeOf(f32));
            const enc = self.getEncoder(self.pipe_sdpa);
            setBuf(enc, q_ref, 0);
            setBuf(enc, keys_ref, 1);
            setBuf(enc, vals_ref, 2);
            setBuf(enc, out_ref, 3);
            var nh_u: u32 = @intCast(nh);
            var nkv_u: u32 = @intCast(nkv);
            var hd_u: u32 = @intCast(hd);
            var sl_u: u32 = @intCast(sl);
            setBytes(enc, @ptrCast(&nh_u), @sizeOf(u32), 4);
            setBytes(enc, @ptrCast(&nkv_u), @sizeOf(u32), 5);
            setBytes(enc, @ptrCast(&hd_u), @sizeOf(u32), 6);
            setBytes(enc, @ptrCast(&sl_u), @sizeOf(u32), 7);
            setBytes(enc, @ptrCast(&scale), @sizeOf(f32), 8);
            self.endEncodeThreadgroups(enc, nh, sdpa_threadgroup_size);
        } else {
            // Turbo or mixed path: use sdpa_fa2_turbo kernel with in-GPU dequant
            const k_cache_bytes = kv_quant.kvSliceBytes(kv_type_k, sl * kvd);
            const v_cache_bytes = kv_quant.kvSliceBytes(kv_type_v, sl * kvd);
            const q_ref = self.getBufRef(@ptrCast(q), nh * hd * @sizeOf(f32));
            const keys_ref = self.getBufRef(@ptrCast(keys.ptr), k_cache_bytes);
            const vals_ref = self.getBufRef(@ptrCast(values.ptr), v_cache_bytes);
            const out_ref = self.getBufRef(@ptrCast(output), nh * hd * @sizeOf(f32));
            const enc = self.getEncoder(self.pipe_sdpa_turbo);
            setBuf(enc, q_ref, 0);
            setBuf(enc, keys_ref, 1);
            setBuf(enc, vals_ref, 2);
            setBuf(enc, out_ref, 3);
            var nh_u: u32 = @intCast(nh);
            var nkv_u: u32 = @intCast(nkv);
            var hd_u: u32 = @intCast(hd);
            var sl_u: u32 = @intCast(sl);
            var bits_k_u: u32 = kv_type_k.turboBits();
            var bits_v_u: u32 = kv_type_v.turboBits();
            var bb_k_u: u32 = kv_type_k.turboBlockByteSize();
            var bb_v_u: u32 = kv_type_v.turboBlockByteSize();
            setBytes(enc, @ptrCast(&nh_u), @sizeOf(u32), 4);
            setBytes(enc, @ptrCast(&nkv_u), @sizeOf(u32), 5);
            setBytes(enc, @ptrCast(&hd_u), @sizeOf(u32), 6);
            setBytes(enc, @ptrCast(&sl_u), @sizeOf(u32), 7);
            setBytes(enc, @ptrCast(&scale), @sizeOf(f32), 8);
            setBytes(enc, @ptrCast(&bits_k_u), @sizeOf(u32), 9);
            setBytes(enc, @ptrCast(&bits_v_u), @sizeOf(u32), 10);
            setBytes(enc, @ptrCast(&bb_k_u), @sizeOf(u32), 11);
            setBytes(enc, @ptrCast(&bb_v_u), @sizeOf(u32), 12);
            self.endEncodeThreadgroups(enc, nh, sdpa_threadgroup_size);
        }
    }

    /// SDPA with per-head softmax stats for split-attention merge.
    /// GPU stats export not yet implemented — syncs GPU, then runs CPU-side
    /// sdpaQuantHeadsWithStats as fallback. Native GPU stats is future work.
    pub fn sdpaWithStats(self: *MetalBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const sdpa_cpu = @import("kernels/cpu/sdpa.zig");
        self.sync();
        const kvd = nkv * hd;
        const k_off = kv_quant.kvByteOffset(kv_type_k, seq_len * kvd);
        const v_off = kv_quant.kvByteOffset(kv_type_v, seq_len * kvd);
        kv_quant.kvStore(keys.ptr + k_off, k_new, kvd, kv_type_k);
        kv_quant.kvStore(values.ptr + v_off, v_new, kvd, kv_type_v);
        sdpa_cpu.sdpaQuantHeadsWithStats(q, keys.ptr, values.ptr, output, nh, nkv, hd, seq_len + 1, scale, kv_type_k, kv_type_v, head_max, head_sum);
    }

    // ── Batched prefill ops ────────────────────────────────────

    /// GEMM: Y[n_tok × n_out] = X[n_tok × n_in] @ W[n_out × n_in]^T.
    /// One threadgroup per output row. Weight reused across TILE_T=8 tokens.
    /// Threadgroup size matched to block count for full thread utilization.
    pub fn gemm(self: *MetalBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        if (n_tok <= 1) {
            self.gemv(x, w, y, n_out, n_in);
            return;
        }

        const pipeline: objc.id = switch (w.dtype) {
            .f32 => self.pipe_gemm_f32,
            .bf16, .f16 => self.pipe_gemm_bf16,
            .q8_0 => self.pipe_gemm_q8_0,
            .q4_0 => self.pipe_gemm_q4_0,
            .q4_k => self.pipe_gemm_q4_k,
            .q6_k => self.pipe_gemm_q6_k,
            .q5_k => self.pipe_gemm_q5_k,
            else => @panic("Metal GEMM: unsupported dtype — add GPU kernel"),
        };

        const w_bytes = weightBytes(w.dtype, n_out, n_in);
        const x_ref = self.getBufRef(@ptrCast(x), n_tok * n_in * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(w.data), w_bytes);
        const y_ref = self.getBufRef(@ptrCast(y), n_tok * n_out * @sizeOf(f32));

        var n_out_val: u32 = @intCast(n_out);
        var n_in_val: u32 = @intCast(n_in);
        var n_tok_val: u32 = @intCast(n_tok);

        const enc = self.getEncoder(pipeline);
        setBuf(enc, x_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, y_ref, 2);
        setBytes(enc, @ptrCast(&n_out_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&n_in_val), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&n_tok_val), @sizeOf(u32), 5);

        const tg = gemvThreadgroupSize(w.dtype, n_in);
        self.endEncodeThreadgroups(enc, n_out, tg);
    }

    /// Batched RMSNorm: GPU dispatch using rms_norm_fused_f32 with n_tok threadgroups.
    /// Each threadgroup normalizes one row of dim elements, sharing the same weight.
    pub fn rmsNormBatched(self: *MetalBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        const total = n_tok * dim;
        const in_ref = self.getBufRef(@ptrCast(input), total * @sizeOf(f32));
        const w_ref = self.getBufRef(@ptrCast(weight), dim * @sizeOf(f32));
        const out_ref = self.getBufRef(@ptrCast(output), total * @sizeOf(f32));

        var dim_val: u32 = @intCast(dim);
        var eps_val: f32 = eps;

        const tg = @min(threadgroup_size, dim);
        const enc = self.getEncoder(self.pipe_rms_norm_fused);
        setBuf(enc, in_ref, 0);
        setBuf(enc, w_ref, 1);
        setBuf(enc, out_ref, 2);
        setBytes(enc, @ptrCast(&dim_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 4);
        self.endEncodeThreadgroups(enc, n_tok, tg);
    }

    /// Batched RoPE: single GPU dispatch for all n_tok tokens at different positions.
    pub fn ropeBatched(self: *MetalBackend, x: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const half_rope = rope_dim / 2;
        const stride = n_heads * head_dim;
        const x_ref = self.getBufRef(@ptrCast(x), n_tok * stride * @sizeOf(f32));
        const pos_ref = self.getBufRef(@ptrCast(positions), n_tok * @sizeOf(u32));

        var n_tok_val: u32 = @intCast(n_tok);
        var nh_val: u32 = @intCast(n_heads);
        var hd_val: u32 = @intCast(head_dim);
        var rd_val: u32 = @intCast(rope_dim);
        var theta_val: f32 = theta;

        const enc = self.getEncoder(self.pipe_rope_batched);
        setBuf(enc, x_ref, 0);
        setBuf(enc, pos_ref, 1);
        setBytes(enc, @ptrCast(&n_tok_val), @sizeOf(u32), 2);
        setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 3);
        setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 4);
        setBytes(enc, @ptrCast(&rd_val), @sizeOf(u32), 5);
        setBytes(enc, @ptrCast(&theta_val), @sizeOf(f32), 6);

        const grid_size = n_tok * n_heads * half_rope;
        self.endEncode1D(enc, self.pipe_rope_batched, grid_size);
    }

    /// Prefill SDPA: zero-flush GPU pipeline.
    /// FA2 reads old K/V from cache, new K/V from pf_k/pf_v (no copy needed
    /// for attention). Then GPU copy kernel writes pf_k/pf_v into the KV cache
    /// Tree-masked SDPA for DDTree verification. GPU-accelerated for f32 and TurboQuant KV.
    pub fn sdpaTree(self: *MetalBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: backend_mod.KvQuantType, kv_type_v: backend_mod.KvQuantType) void {
        if (n_nodes == 0) return;
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        if ((is_f32_k or is_turbo_k) and (is_f32_v or is_turbo_v)) {
            const kvd = nkv * hd;
            const q_ref = self.getBufRef(@ptrCast(q_all), n_nodes * nh * hd * @sizeOf(f32));
            const pk_sz = if (is_f32_k) prefix_len * kvd * @sizeOf(f32) else kv_quant.kvSliceBytes(kv_type_k, prefix_len * kvd);
            const pv_sz = if (is_f32_v) prefix_len * kvd * @sizeOf(f32) else kv_quant.kvSliceBytes(kv_type_v, prefix_len * kvd);
            const pk_ref = self.getBufRef(@ptrCast(prefix_keys), pk_sz);
            const pv_ref = self.getBufRef(@ptrCast(prefix_values), pv_sz);
            const tk_ref = self.getBufRef(@ptrCast(tree_keys), n_nodes * kvd * @sizeOf(f32));
            const tv_ref = self.getBufRef(@ptrCast(tree_values), n_nodes * kvd * @sizeOf(f32));
            const out_ref = self.getBufRef(@ptrCast(output), n_nodes * nh * hd * @sizeOf(f32));
            const mask_ref = self.getBufRef(@ptrCast(ancestor_masks), n_nodes * 8 * @sizeOf(u64));

            var nh_val: u32 = @intCast(nh);
            var nkv_val: u32 = @intCast(nkv);
            var hd_val: u32 = @intCast(hd);
            var pl_val: u32 = @intCast(prefix_len);
            var nn_val: u32 = n_nodes;
            var sc_val: f32 = scale;

            if (is_f32_k and is_f32_v) {
                const enc = self.getEncoder(self.pipe_sdpa_tree);
                setBuf(enc, q_ref, 0);
                setBuf(enc, pk_ref, 1);
                setBuf(enc, pv_ref, 2);
                setBuf(enc, tk_ref, 3);
                setBuf(enc, tv_ref, 4);
                setBuf(enc, out_ref, 5);
                setBuf(enc, mask_ref, 6);
                setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 7);
                setBytes(enc, @ptrCast(&nkv_val), @sizeOf(u32), 8);
                setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 9);
                setBytes(enc, @ptrCast(&pl_val), @sizeOf(u32), 10);
                setBytes(enc, @ptrCast(&nn_val), @sizeOf(u32), 11);
                setBytes(enc, @ptrCast(&sc_val), @sizeOf(f32), 12);
                self.endEncodeThreadgroups(enc, n_nodes * nh, threadgroup_size);
            } else {
                var bits_k_val: u32 = if (is_turbo_k) kv_type_k.turboBits() else 0;
                var bits_v_val: u32 = if (is_turbo_v) kv_type_v.turboBits() else 0;
                var bbk_val: u32 = if (is_turbo_k) kv_type_k.turboBlockByteSize() else 0;
                var bbv_val: u32 = if (is_turbo_v) kv_type_v.turboBlockByteSize() else 0;
                const enc = self.getEncoder(self.pipe_sdpa_tree_turbo);
                setBuf(enc, q_ref, 0);
                setBuf(enc, pk_ref, 1);
                setBuf(enc, pv_ref, 2);
                setBuf(enc, tk_ref, 3);
                setBuf(enc, tv_ref, 4);
                setBuf(enc, out_ref, 5);
                setBuf(enc, mask_ref, 6);
                setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 7);
                setBytes(enc, @ptrCast(&nkv_val), @sizeOf(u32), 8);
                setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 9);
                setBytes(enc, @ptrCast(&pl_val), @sizeOf(u32), 10);
                setBytes(enc, @ptrCast(&nn_val), @sizeOf(u32), 11);
                setBytes(enc, @ptrCast(&sc_val), @sizeOf(f32), 12);
                setBytes(enc, @ptrCast(&bits_k_val), @sizeOf(u32), 13);
                setBytes(enc, @ptrCast(&bits_v_val), @sizeOf(u32), 14);
                setBytes(enc, @ptrCast(&bbk_val), @sizeOf(u32), 15);
                setBytes(enc, @ptrCast(&bbv_val), @sizeOf(u32), 16);
                self.endEncodeThreadgroups(enc, n_nodes * nh, threadgroup_size);
            }
            return;
        }
        // CPU fallback for non-turbo quantized prefix KV
        @import("kernels/cpu/sdpa_tree.zig").sdpaTree(q_all, prefix_keys, prefix_values, tree_keys, tree_values, output, ancestor_masks, nh, nkv, hd, prefix_len, n_nodes, scale, kv_type_k, kv_type_v);
    }

    /// for future chunks/decode. All dispatches in one command buffer.
    ///
    /// For turbo KV types: CPU-side KV append + sequential GPU turbo SDPA per token.
    /// For non-turbo quantized types (q8_0, f16, etc.): full CPU fallback.
    pub fn sdpaPrefill(self: *MetalBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const is_turbo_k = kv_type_k.isTurbo();
        const is_turbo_v = kv_type_v.isTurbo();
        const is_f32_k = (kv_type_k == .f32);
        const is_f32_v = (kv_type_v == .f32);

        // Turbo/mixed prefill: CPU KV append + sequential GPU turbo SDPA per token.
        // The GPU turbo kernel handles the SDPA compute; CPU does the quantization
        // for KV append (once per position, not the SDPA hot path).
        if ((is_turbo_k or is_f32_k) and (is_turbo_v or is_f32_v) and (is_turbo_k or is_turbo_v)) {
            self.sync();
            const kvd = nkv * hd;
            // Append all n_tok keys/values to KV cache on CPU
            for (0..n_tok) |t| {
                const src_off = t * kvd;
                const dst_elem = (prev_len + t) * kvd;
                const dst_byte_k = kv_quant.kvByteOffset(kv_type_k, dst_elem);
                const dst_byte_v = kv_quant.kvByteOffset(kv_type_v, dst_elem);
                kv_quant.kvStore(kv_keys.ptr + dst_byte_k, k + src_off, kvd, kv_type_k);
                kv_quant.kvStore(kv_values.ptr + dst_byte_v, v + src_off, kvd, kv_type_v);
            }
            // Sequential GPU SDPA per token (each uses turbo kernel over full history)
            const k_cache_bytes = kv_quant.kvSliceBytes(kv_type_k, (prev_len + n_tok) * kvd);
            const v_cache_bytes = kv_quant.kvSliceBytes(kv_type_v, (prev_len + n_tok) * kvd);
            for (0..n_tok) |t| {
                const sl = prev_len + t + 1;
                const q_off = t * nh * hd;
                const out_off = t * nh * hd;
                const q_ref = self.getBufRef(@ptrCast(q + q_off), nh * hd * @sizeOf(f32));
                const keys_ref = self.getBufRef(@ptrCast(kv_keys.ptr), k_cache_bytes);
                const vals_ref = self.getBufRef(@ptrCast(kv_values.ptr), v_cache_bytes);
                const out_ref = self.getBufRef(@ptrCast(output + out_off), nh * hd * @sizeOf(f32));
                const enc = self.getEncoder(self.pipe_sdpa_turbo);
                setBuf(enc, q_ref, 0);
                setBuf(enc, keys_ref, 1);
                setBuf(enc, vals_ref, 2);
                setBuf(enc, out_ref, 3);
                var nh_u: u32 = @intCast(nh);
                var nkv_u: u32 = @intCast(nkv);
                var hd_u: u32 = @intCast(hd);
                var sl_u: u32 = @intCast(sl);
                var scale_val: f32 = scale;
                var bits_k_u: u32 = kv_type_k.turboBits();
                var bits_v_u: u32 = kv_type_v.turboBits();
                var bb_k_u: u32 = kv_type_k.turboBlockByteSize();
                var bb_v_u: u32 = kv_type_v.turboBlockByteSize();
                setBytes(enc, @ptrCast(&nh_u), @sizeOf(u32), 4);
                setBytes(enc, @ptrCast(&nkv_u), @sizeOf(u32), 5);
                setBytes(enc, @ptrCast(&hd_u), @sizeOf(u32), 6);
                setBytes(enc, @ptrCast(&sl_u), @sizeOf(u32), 7);
                setBytes(enc, @ptrCast(&scale_val), @sizeOf(f32), 8);
                setBytes(enc, @ptrCast(&bits_k_u), @sizeOf(u32), 9);
                setBytes(enc, @ptrCast(&bits_v_u), @sizeOf(u32), 10);
                setBytes(enc, @ptrCast(&bb_k_u), @sizeOf(u32), 11);
                setBytes(enc, @ptrCast(&bb_v_u), @sizeOf(u32), 12);
                self.endEncodeThreadgroups(enc, nh, sdpa_threadgroup_size);
            }
            return;
        }

        // Non-turbo, non-f32 quantized KV: not supported
        if (kv_type_k != .f32 or kv_type_v != .f32)
            @panic("Metal SDPA prefill: unsupported KV type — use --kv-type f32 or turbo2/3/4");

        const kvd = nkv * hd;
        const f32_keys: [*]f32 = @ptrCast(@alignCast(kv_keys.ptr));
        const f32_values: [*]f32 = @ptrCast(@alignCast(kv_values.ptr));
        const copy_elems = n_tok * kvd;

        // GPU dispatch 1: FA2 with dual K/V sources (cache + new)
        const q_ref = self.getBufRef(@ptrCast(q), n_tok * nh * hd * @sizeOf(f32));
        const kc_ref = self.getBufRef(@ptrCast(f32_keys), (prev_len + n_tok) * kvd * @sizeOf(f32));
        const vc_ref = self.getBufRef(@ptrCast(f32_values), (prev_len + n_tok) * kvd * @sizeOf(f32));
        const kn_ref = self.getBufRef(@ptrCast(k), copy_elems * @sizeOf(f32));
        const vn_ref = self.getBufRef(@ptrCast(v), copy_elems * @sizeOf(f32));
        const o_ref = self.getBufRef(@ptrCast(output), n_tok * nh * hd * @sizeOf(f32));

        var nh_val: u32 = @intCast(nh);
        var nkv_val: u32 = @intCast(nkv);
        var hd_val: u32 = @intCast(hd);
        var prev_val: u32 = @intCast(prev_len);
        var ntok_val: u32 = @intCast(n_tok);
        var scale_val: f32 = scale;

        {
            const enc = self.getEncoder(self.pipe_sdpa_prefill);
            setBuf(enc, q_ref, 0);
            setBuf(enc, kc_ref, 1); // K_cache (old positions)
            setBuf(enc, vc_ref, 2); // V_cache (old positions)
            setBuf(enc, kn_ref, 3); // K_new (this chunk)
            setBuf(enc, vn_ref, 4); // V_new (this chunk)
            setBuf(enc, o_ref, 5);
            setBytes(enc, @ptrCast(&nh_val), @sizeOf(u32), 6);
            setBytes(enc, @ptrCast(&nkv_val), @sizeOf(u32), 7);
            setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 8);
            setBytes(enc, @ptrCast(&prev_val), @sizeOf(u32), 9);
            setBytes(enc, @ptrCast(&ntok_val), @sizeOf(u32), 10);
            setBytes(enc, @ptrCast(&scale_val), @sizeOf(f32), 11);
            self.endEncodeThreadgroups(enc, n_tok * nh, sdpa_threadgroup_size);
        }

        // GPU dispatch 2: copy pf_k/pf_v → KV cache at prev_len offset
        // Contiguous: pf_k[0..n_tok*kvd] → kv_keys[prev_len*kvd..]
        {
            const dst_k = self.getBufRef(@ptrCast(f32_keys + prev_len * kvd), copy_elems * @sizeOf(f32));
            const dst_v = self.getBufRef(@ptrCast(f32_values + prev_len * kvd), copy_elems * @sizeOf(f32));
            // Copy keys
            const enc_k = self.getEncoder(self.pipe_copy_f32);
            setBuf(enc_k, kn_ref, 0);
            setBuf(enc_k, dst_k, 1);
            self.endEncode1D(enc_k, self.pipe_copy_f32, copy_elems);
            // Copy values
            const enc_v = self.getEncoder(self.pipe_copy_f32);
            setBuf(enc_v, vn_ref, 0);
            setBuf(enc_v, dst_v, 1);
            self.endEncode1D(enc_v, self.pipe_copy_f32, copy_elems);
        }
    }

    /// DeltaNet SSM — all 4 kernels on GPU, no CPU sync per layer.
    /// GPU handles gate/beta, conv1d, L2 norm, recurrence+gated output.
    pub fn deltaNet(self: *MetalBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        // GPU DeltaNet: all 4 kernels run on GPU without CPU sync.

        const conv_ch: usize = p.conv_ch;
        const num_v_heads: usize = p.num_v_heads;
        const num_k_heads: usize = p.num_k_heads;
        const head_k_dim: usize = p.head_k_dim;
        const head_v_dim: usize = p.head_v_dim;

        // ── GPU Kernel 1: gate & beta ──
        // gate_ref/beta_out_ref alias alpha_ref/beta_ref (in-place overwrite).
        const alpha_ref = self.getBufRef(@ptrCast(alpha_buf), num_v_heads * @sizeOf(f32));
        const beta_ref = self.getBufRef(@ptrCast(beta_buf), num_v_heads * @sizeOf(f32));
        const ssm_a_ref = self.getBufRef(@ptrCast(ssm_a), num_v_heads * @sizeOf(f32));
        const dt_bias_ref = self.getBufRef(@ptrCast(dt_bias), num_v_heads * @sizeOf(f32));
        const gate_ref = alpha_ref;
        const beta_out_ref = beta_ref;
        {
            var n_heads_val: u32 = @intCast(num_v_heads);
            const enc = self.getEncoder(self.pipe_dn_gate_beta);
            setBuf(enc, alpha_ref, 0);
            setBuf(enc, beta_ref, 1);
            setBuf(enc, ssm_a_ref, 2);
            setBuf(enc, dt_bias_ref, 3);
            setBuf(enc, gate_ref, 4);
            setBuf(enc, beta_out_ref, 5);
            setBytes(enc, @ptrCast(&n_heads_val), @sizeOf(u32), 6);
            self.endEncodeOneThreadgroup(enc, num_v_heads);
        }

        // ── GPU Kernel 2: conv1d + SiLU ──
        const conv_in_ref = self.getBufRef(@ptrCast(conv_in), conv_ch * @sizeOf(f32));
        const cs_ref = self.getBufRef(@ptrCast(conv_state), (p.d_conv - 1) * conv_ch * @sizeOf(f32));
        const cw_ref = self.getBufRef(@ptrCast(conv_w), conv_ch * p.d_conv * @sizeOf(f32));
        const co_ref = self.getBufRef(@ptrCast(conv_out), conv_ch * @sizeOf(f32));
        {
            var conv_ch_val: u32 = @intCast(conv_ch);
            var d_conv_val: u32 = p.d_conv;
            const enc = self.getEncoder(self.pipe_dn_conv1d);
            setBuf(enc, conv_in_ref, 0);
            setBuf(enc, cs_ref, 1);
            setBuf(enc, cw_ref, 2);
            setBuf(enc, co_ref, 3);
            setBytes(enc, @ptrCast(&conv_ch_val), @sizeOf(u32), 4);
            setBytes(enc, @ptrCast(&d_conv_val), @sizeOf(u32), 5);
            const n_groups = (conv_ch + threadgroup_size - 1) / threadgroup_size;
            self.endEncodeThreadgroups(enc, n_groups, threadgroup_size);
        }

        // ── GPU Kernel 3: L2 norm Q & K ──
        {
            var hd_val: u32 = @intCast(head_k_dim);
            var nk_val: u32 = @intCast(num_k_heads);
            var eps_val: f32 = p.rms_eps;
            var q_off_val: u32 = if (p.kqv_order) @intCast(num_k_heads * head_k_dim) else 0;
            var k_off_val: u32 = if (p.kqv_order) 0 else @intCast(num_k_heads * head_k_dim);
            const enc = self.getEncoder(self.pipe_dn_l2_norm);
            setBuf(enc, co_ref, 0);
            setBytes(enc, @ptrCast(&hd_val), @sizeOf(u32), 1);
            setBytes(enc, @ptrCast(&nk_val), @sizeOf(u32), 2);
            setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 3);
            setBytes(enc, @ptrCast(&q_off_val), @sizeOf(u32), 4);
            setBytes(enc, @ptrCast(&k_off_val), @sizeOf(u32), 5);
            const l2_tg_size: usize = @min(threadgroup_size, @max(simd_width, (head_k_dim + simd_width - 1) & ~(simd_width - 1)));
            self.endEncodeThreadgroups(enc, 2 * num_k_heads, l2_tg_size);
        }

        // ── GPU Kernel 4: recurrence + gated output ──
        // Q/K/V are sub-regions of conv_out — reuse co_ref.buf with byte offsets
        // so GPU writes from conv1d + L2 norm are visible (same Metal buffer).
        {
            const qk_dim = num_k_heads * head_k_dim * @sizeOf(f32);
            const v_byte_off = 2 * num_k_heads * head_k_dim * @sizeOf(f32);
            const q_byte_off: usize = if (p.kqv_order) qk_dim else 0;
            const k_byte_off: usize = if (p.kqv_order) 0 else qk_dim;
            const q_ref = BufRef{ .buf = co_ref.buf, .offset = co_ref.offset + q_byte_off };
            const k_ref = BufRef{ .buf = co_ref.buf, .offset = co_ref.offset + k_byte_off };
            const v_ref = BufRef{ .buf = co_ref.buf, .offset = co_ref.offset + v_byte_off };
            const state_ref = self.getBufRef(@ptrCast(ssm_state.ptr), ssm_state.len * @sizeOf(f32));
            const z_ref = self.getBufRef(@ptrCast(z_buf), num_v_heads * head_v_dim * @sizeOf(f32));
            const nw_ref = self.getBufRef(@ptrCast(ssm_norm_w), head_v_dim * @sizeOf(f32));
            const out_ref = self.getBufRef(@ptrCast(output), num_v_heads * head_v_dim * @sizeOf(f32));

            var hvd_val: u32 = @intCast(head_v_dim);
            var hkd_val: u32 = @intCast(head_k_dim);
            var nk_val: u32 = @intCast(num_k_heads);
            var nv_val: u32 = @intCast(num_v_heads);
            var qs_val: f32 = p.q_scale;
            var eps_val: f32 = p.rms_eps;

            const enc = self.getEncoder(self.pipe_dn_recurrence);
            setBuf(enc, q_ref, 0);
            setBuf(enc, k_ref, 1);
            setBuf(enc, v_ref, 2);
            setBuf(enc, state_ref, 3);
            setBuf(enc, gate_ref, 4);
            setBuf(enc, beta_out_ref, 5);
            setBuf(enc, z_ref, 6);
            setBuf(enc, nw_ref, 7);
            setBuf(enc, out_ref, 8);
            setBytes(enc, @ptrCast(&hvd_val), @sizeOf(u32), 9);
            setBytes(enc, @ptrCast(&hkd_val), @sizeOf(u32), 10);
            setBytes(enc, @ptrCast(&nk_val), @sizeOf(u32), 11);
            setBytes(enc, @ptrCast(&nv_val), @sizeOf(u32), 12);
            setBytes(enc, @ptrCast(&qs_val), @sizeOf(f32), 13);
            setBytes(enc, @ptrCast(&eps_val), @sizeOf(f32), 14);
            var use_grouped: u32 = if (p.kqv_order) 1 else 0;
            setBytes(enc, @ptrCast(&use_grouped), @sizeOf(u32), 15);
            const rec_tg_size: usize = @min(threadgroup_size, @max(simd_width, (head_v_dim + simd_width - 1) & ~(simd_width - 1)));
            self.endEncodeThreadgroups(enc, num_v_heads, rec_tg_size);
        }
    }
};

// ── Tests ─────────────────────────────────────────────────────────

const builtin = @import("builtin");

test "Metal backend init and silu" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    metal.silu(&input, &output, 4);
    metal.sync();

    // SiLU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.001);
    // SiLU(1) ≈ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), output[1], 0.01);
    // SiLU(-1) ≈ -0.2689
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), output[2], 0.01);
    // SiLU(2) ≈ 1.7616
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), output[3], 0.01);
}

test "Metal backend rmsNorm" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var weight = [_]f32{ 1.0, 2.0, 0.5, 3.0 };
    var output: [4]f32 = undefined;
    metal.rmsNorm(&input, &weight, &output, 4, 1e-6);
    metal.sync();

    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // output[i] = input[i] * weight[i] / RMS
    try std.testing.expectApproxEqAbs(@as(f32, 0.3651), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4606), output[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5477), output[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4.3818), output[3], 0.01);
}

test "Metal backend add" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var output: [4]f32 = undefined;
    metal.add(&a, &b, &output, 4);
    metal.sync();

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), output[3], 0.001);
}

test "Metal backend gemvNvfp4St basic" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    // 1x16 GEMV: one output row, 16 input elements (one group).
    // x = [1.0, 0, 0, ..., 0]
    var x = [16]f32{ 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // nibble 0x2 = e2m1(2) = 1.0. byte[0] low nibble = elem[0] = 1.0.
    var weight = [8]u8{ 0x02, 0, 0, 0, 0, 0, 0, 0 };
    // scale = 0x38 = FP8 E4M3 for 1.0
    var scale = [1]u8{0x38};
    var y = [1]f32{0};
    metal.gemvNvfp4St(&x, &weight, &scale, &y, 1, 16);
    metal.sync();
    // x[0]=1.0 * e2m1(2)=1.0 * fp8(0x38)=1.0 = 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), y[0], 1e-6);
}

test "Metal backend gemvNvfp4St multi-row" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    // 2x16 GEMV: two output rows.
    var x = [16]f32{ 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // Row 0: byte[0] = 0x42 → low=2 (1.0), high=4 (2.0)
    // Row 1: byte[0] = 0x24 → low=4 (2.0), high=2 (1.0)
    var weight = [16]u8{ 0x42, 0, 0, 0, 0, 0, 0, 0, 0x24, 0, 0, 0, 0, 0, 0, 0 };
    var scale = [2]u8{ 0x38, 0x40 }; // row 0 scale=1.0, row 1 scale=2.0
    var y = [2]f32{ 0, 0 };
    metal.gemvNvfp4St(&x, &weight, &scale, &y, 2, 16);
    metal.sync();
    // Row 0: (1.0*1.0 + 1.0*2.0) * scale(1.0) = 3.0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), y[0], 1e-6);
    // Row 1: (1.0*2.0 + 1.0*1.0) * scale(2.0) = 6.0
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), y[1], 1e-6);
}

test "Metal backend gemvMlxQ4 basic" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    // 1x64 GEMV: one output row, 64 input elements (one group).
    // x = [1.0, 1.0, 0, 0, ...] — first two elements set
    var x: [64]f32 = [_]f32{0} ** 64;
    x[0] = 1.0;
    x[1] = 1.0;
    // Weight: u32 word[0] nibbles: elem[0]=3, elem[1]=5, rest=0.
    // word[0] = 0x00000053 (nibble[0]=3, nibble[1]=5)
    var weight: [8]u32 = [_]u32{0} ** 8;
    weight[0] = 0x53; // nibble[0]=3, nibble[1]=5
    // Scale = bf16 1.0 = 0x3F80 (lo=0x80, hi=0x3F)
    var sc = [2]u8{ 0x80, 0x3F };
    // Bias = bf16 0.5 = 0x3F00 (lo=0x00, hi=0x3F)
    var bi = [2]u8{ 0x00, 0x3F };
    var y = [1]f32{0};
    // y = scale*sum(x*q) + bias*sum(x)
    //   = 1.0*(1.0*3 + 1.0*5) + 0.5*(1.0+1.0)
    //   = 8.0 + 1.0 = 9.0
    metal.gemvMlxQ(&x, @ptrCast(&weight), @ptrCast(&sc), @ptrCast(&bi), &y, 1, 64, 4);
    metal.sync();
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), y[0], 1e-4);
}

test "Metal backend gemvMlxQ4 matches CPU" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var metal = MetalBackend.init(allocator) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    // Test with k=128 (2 groups), n=4 rows
    const k = 128;
    const n = 4;
    const gpr = (k + mlx_group_size - 1) / mlx_group_size; // = 2
    const wpg = 8;

    // Fill x with a pattern
    var x: [k]f32 = undefined;
    for (0..k) |i| x[i] = @as(f32, @floatFromInt(i % 7)) * 0.3 - 1.0;

    // Fill weight with a pattern (u32 words, 4 nibbles give values 0-15)
    var weight: [n * gpr * wpg]u32 = undefined;
    for (0..weight.len) |i| weight[i] = @truncate((i * 0x37) ^ 0x1234);

    // Fill scales and biases (bf16 as u16)
    var sc16: [n * gpr]u16 = undefined;
    var bi16: [n * gpr]u16 = undefined;
    for (0..sc16.len) |i| {
        sc16[i] = 0x3C00 + @as(u16, @truncate(i * 0x80)); // small positive bf16
        bi16[i] = 0x3800 + @as(u16, @truncate(i * 0x40));
    }

    // CPU reference
    var y_cpu: [n]f32 = undefined;
    mlx_ops.mlxGemvRaw(&x, &weight, &sc16, &bi16, &y_cpu, n, k, 4);

    // Metal
    var y_metal: [n]f32 = undefined;
    metal.gemvMlxQ(&x, @ptrCast(&weight), @ptrCast(&sc16), @ptrCast(&bi16), &y_metal, n, k, 4);
    metal.sync();

    for (0..n) |i| {
        try std.testing.expectApproxEqRel(y_cpu[i], y_metal[i], 1e-3);
    }
}

test "Metal backend gemvMlxQ4 large matrix" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const al = std.testing.allocator;
    var metal = MetalBackend.init(al) catch |err| {
        if (err == error.NoMetalDevice) return error.SkipZigTest;
        return err;
    };
    defer metal.deinit();

    const k = 2560;
    const n = 64;
    const gpr = (k + mlx_group_size - 1) / mlx_group_size; // 40
    const wpg = 8;

    const x = try al.alloc(f32, k);
    defer al.free(x);
    const weight = try al.alloc(u32, n * gpr * wpg);
    defer al.free(weight);
    const sc16 = try al.alloc(u16, n * gpr);
    defer al.free(sc16);
    const bi16 = try al.alloc(u16, n * gpr);
    defer al.free(bi16);

    // Fill with deterministic pattern — scales near 1.0, biases near 0 to avoid non-finite outputs
    for (0..k) |i| x[i] = @as(f32, @floatFromInt(i % 13)) * 0.2 - 1.2;
    for (0..weight.len) |i| weight[i] = @truncate((i *% 0xDEAD) ^ 0x1234ABCD);
    for (0..sc16.len) |i| sc16[i] = @bitCast(@as(f16, @floatCast(0.5 + @as(f32, @floatFromInt(i % 7)) * 0.2)));
    for (0..bi16.len) |i| bi16[i] = @bitCast(@as(f16, @floatCast(@as(f32, @floatFromInt(i % 5)) * 0.1 - 0.2)));

    const y_cpu = try al.alloc(f32, n);
    defer al.free(y_cpu);
    const y_metal = try al.alloc(f32, n);
    defer al.free(y_metal);

    mlx_ops.mlxGemvRaw(x.ptr, weight.ptr, sc16.ptr, bi16.ptr, y_cpu.ptr, n, k, 4);
    metal.gemvMlxQ(x.ptr, @ptrCast(weight.ptr), @ptrCast(sc16.ptr), @ptrCast(bi16.ptr), y_metal.ptr, n, k, 4);
    metal.sync();

    for (0..n) |i| {
        try std.testing.expectApproxEqRel(y_cpu[i], y_metal[i], 1e-3);
    }
}
