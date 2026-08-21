//! WebGPU compute backend via wgpu-native.
//!
//! Dynamically loads the wgpu-native C library at runtime. Uses WGSL compute
//! shaders embedded at compile time for GPU kernel dispatch. Buffer management
//! follows the Vulkan backend pattern: weight cache + activation pool.

const std = @import("std");
const backend_mod = @import("backend.zig");
const kv_quant = @import("../ops/kv_quant.zig");

const TensorData = backend_mod.TensorData;
const KvQuantType = backend_mod.KvQuantType;
const PagedKvView = backend_mod.PagedKvView;

// ── WGSL shader sources (embedded at compile time) ──────────────────

const wgsl_silu = @embedFile("kernels/webgpu/silu.wgsl");
const wgsl_gelu = @embedFile("kernels/webgpu/gelu.wgsl");
const wgsl_add = @embedFile("kernels/webgpu/add.wgsl");
const wgsl_mul = @embedFile("kernels/webgpu/mul.wgsl");
const wgsl_silu_mul = @embedFile("kernels/webgpu/silu_mul.wgsl");
const wgsl_gelu_mul = @embedFile("kernels/webgpu/gelu_mul.wgsl");
const wgsl_rms_norm = @embedFile("kernels/webgpu/rms_norm.wgsl");
const wgsl_softmax = @embedFile("kernels/webgpu/softmax.wgsl");
const wgsl_rope = @embedFile("kernels/webgpu/rope.wgsl");
const wgsl_embedding = @embedFile("kernels/webgpu/embedding.wgsl");
const wgsl_gemv_f32 = @embedFile("kernels/webgpu/gemv_f32.wgsl");
const wgsl_gemv_bf16 = @embedFile("kernels/webgpu/gemv_bf16.wgsl");
const wgsl_gemv_f16 = @embedFile("kernels/webgpu/gemv_f16.wgsl");
const wgsl_gemv_fp8_e4m3 = @embedFile("kernels/webgpu/gemv_fp8_e4m3.wgsl");
const wgsl_gemv_fp8_e5m2 = @embedFile("kernels/webgpu/gemv_fp8_e5m2.wgsl");
const wgsl_gemv_q4_1 = @embedFile("kernels/webgpu/gemv_q4_1.wgsl");
const wgsl_gemv_q5_0 = @embedFile("kernels/webgpu/gemv_q5_0.wgsl");
const wgsl_gemv_q2_k = @embedFile("kernels/webgpu/gemv_q2_k.wgsl");
const wgsl_gemv_q3_k = @embedFile("kernels/webgpu/gemv_q3_k.wgsl");
const wgsl_gemv_iq4_nl = @embedFile("kernels/webgpu/gemv_iq4_nl.wgsl");
const wgsl_gemv_iq4_xs = @embedFile("kernels/webgpu/gemv_iq4_xs.wgsl");
const wgsl_gemv_q8_0 = @embedFile("kernels/webgpu/gemv_q8_0.wgsl");
const wgsl_gemv_q4_0 = @embedFile("kernels/webgpu/gemv_q4_0.wgsl");
const wgsl_gemv_q4_k = @embedFile("kernels/webgpu/gemv_q4_k.wgsl");
const wgsl_gemv_q5_k = @embedFile("kernels/webgpu/gemv_q5_k.wgsl");
const wgsl_gemv_q6_k = @embedFile("kernels/webgpu/gemv_q6_k.wgsl");
const wgsl_sigmoid_mul = @embedFile("kernels/webgpu/sigmoid_mul.wgsl");
const wgsl_l2_norm = @embedFile("kernels/webgpu/l2_norm.wgsl");
const wgsl_rms_norm_multi = @embedFile("kernels/webgpu/rms_norm_multi.wgsl");
const wgsl_deinterleave = @embedFile("kernels/webgpu/deinterleave.wgsl");
const wgsl_split_qgate = @embedFile("kernels/webgpu/split_qgate.wgsl");
const wgsl_add_rms_norm = @embedFile("kernels/webgpu/add_rms_norm.wgsl");
const wgsl_rms_norm_add = @embedFile("kernels/webgpu/rms_norm_add.wgsl");
const wgsl_add_scaled = @embedFile("kernels/webgpu/add_scaled.wgsl");
const wgsl_gemv_t_q8_0 = @embedFile("kernels/webgpu/gemv_t_q8_0.wgsl");
const wgsl_sdpa = @embedFile("kernels/webgpu/sdpa.wgsl");
const wgsl_conv1d = @embedFile("kernels/webgpu/conv1d.wgsl");
const wgsl_deltanet = @embedFile("kernels/webgpu/deltanet_recurrence.wgsl");
const wgsl_sdpa_tree = @embedFile("kernels/webgpu/sdpa_tree.wgsl");
const wgsl_sdpa_paged = @embedFile("kernels/webgpu/sdpa_paged.wgsl");
const wgsl_gemv_nvfp4_st = @embedFile("kernels/webgpu/gemv_nvfp4_st.wgsl");
const wgsl_gemv_mlx_q4 = @embedFile("kernels/webgpu/gemv_mlx_q4.wgsl");
const wgsl_gemv_mxfp4_st = @embedFile("kernels/webgpu/gemv_mxfp4_st.wgsl");
const wgsl_gemv_gptq = @embedFile("kernels/webgpu/gemv_gptq.wgsl");
const wgsl_gemv_awq = @embedFile("kernels/webgpu/gemv_awq.wgsl");
const wgsl_gemv_hqq = @embedFile("kernels/webgpu/gemv_hqq.wgsl");
const wgsl_gemv_tq1_0 = @embedFile("kernels/webgpu/gemv_tq1_0.wgsl");
const wgsl_gemv_tq2_0 = @embedFile("kernels/webgpu/gemv_tq2_0.wgsl");

// ── WebGPU C API types ──────────────────────────────────────────────

const WGPUInstance = ?*anyopaque;
const WGPUAdapter = ?*anyopaque;
const WGPUDevice = ?*anyopaque;
const WGPUQueue = ?*anyopaque;
const WGPUBuffer = ?*anyopaque;
const WGPUShaderModule = ?*anyopaque;
const WGPUComputePipeline = ?*anyopaque;
const WGPUBindGroupLayout = ?*anyopaque;
const WGPUBindGroup = ?*anyopaque;
const WGPUCommandEncoder = ?*anyopaque;
const WGPUComputePassEncoder = ?*anyopaque;
const WGPUCommandBuffer = ?*anyopaque;
const WGPUPipelineLayout = ?*anyopaque;

const WGPUBufferUsage = u64;
const wgpu_buffer_usage_storage: WGPUBufferUsage = 0x0080;
const wgpu_buffer_usage_copy_src: WGPUBufferUsage = 0x0004;
const wgpu_buffer_usage_copy_dst: WGPUBufferUsage = 0x0008;
const wgpu_buffer_usage_map_read: WGPUBufferUsage = 0x0001;
const wgpu_buffer_usage_uniform: WGPUBufferUsage = 0x0040;

const wgpu_buffer_binding_storage = 7;
const wgpu_buffer_binding_read_only_storage = 6;
const wgpu_buffer_binding_uniform = 1;

const WGPUMapMode = u64;
const wgpu_map_mode_read: WGPUMapMode = 1;

const WGPURequestAdapterStatus = u32;
const WGPURequestDeviceStatus = u32;
const WGPUBufferMapAsyncStatus = u32;
const WGPUCallbackMode = u32;
const wgpu_callback_mode_allow_process_events: WGPUCallbackMode = 2;

const WGPUStringView = extern struct {
    data: ?[*]const u8 = null,
    length: usize = 0,

    fn fromSlice(s: [:0]const u8) WGPUStringView {
        return .{ .data = s.ptr, .length = s.len };
    }
};

const WGPUChainedStruct = extern struct {
    next: ?*anyopaque = null,
    s_type: u32 = 0,
};

const WGPUFuture = extern struct {
    id: u64 = 0,
};

const WGPUInstanceDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    required_feature_count: usize = 0,
    required_features: ?*anyopaque = null,
    required_limits: ?*anyopaque = null,
};

const WGPURequestAdapterOptions = extern struct {
    next_in_chain: ?*anyopaque = null,
    feature_level: u32 = 0,
    power_preference: u32 = 0,
    force_fallback_adapter: u32 = 0,
    backend_type: u32 = 0,
};

const WGPURequestAdapterCallbackInfo = extern struct {
    next_in_chain: ?*anyopaque = null,
    mode: WGPUCallbackMode = wgpu_callback_mode_allow_process_events,
    callback: ?*const fn (WGPURequestAdapterStatus, WGPUAdapter, WGPUStringView, ?*anyopaque, ?*anyopaque) callconv(.c) void = null,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const WGPURequestDeviceCallbackInfo = extern struct {
    next_in_chain: ?*anyopaque = null,
    mode: WGPUCallbackMode = wgpu_callback_mode_allow_process_events,
    callback: ?*const fn (WGPURequestDeviceStatus, WGPUDevice, WGPUStringView, ?*anyopaque, ?*anyopaque) callconv(.c) void = null,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const WGPUBufferMapCallbackInfo = extern struct {
    next_in_chain: ?*anyopaque = null,
    mode: WGPUCallbackMode = wgpu_callback_mode_allow_process_events,
    callback: ?*const fn (WGPUBufferMapAsyncStatus, WGPUStringView, ?*anyopaque, ?*anyopaque) callconv(.c) void = null,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const WGPUQueueDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
};

const WGPUDeviceLostCallbackInfo = extern struct {
    next_in_chain: ?*anyopaque = null,
    mode: WGPUCallbackMode = 0,
    callback: ?*anyopaque = null,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const WGPUUncapturedErrorCallbackInfo = extern struct {
    next_in_chain: ?*anyopaque = null,
    callback: ?*anyopaque = null,
    userdata1: ?*anyopaque = null,
    userdata2: ?*anyopaque = null,
};

const WGPUDeviceDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
    required_feature_count: usize = 0,
    required_features: ?*anyopaque = null,
    required_limits: ?*anyopaque = null,
    default_queue: WGPUQueueDescriptor = .{},
    device_lost_callback_info: WGPUDeviceLostCallbackInfo = .{},
    uncaptured_error_callback_info: WGPUUncapturedErrorCallbackInfo = .{},
};

const WGPULimits = extern struct {
    nextInChain: ?*anyopaque = null,
    maxTextureDimension1D: u32 = std.math.maxInt(u32),
    maxTextureDimension2D: u32 = std.math.maxInt(u32),
    maxTextureDimension3D: u32 = std.math.maxInt(u32),
    maxTextureArrayLayers: u32 = std.math.maxInt(u32),
    maxBindGroups: u32 = std.math.maxInt(u32),
    maxBindGroupsPlusVertexBuffers: u32 = std.math.maxInt(u32),
    maxBindingsPerBindGroup: u32 = std.math.maxInt(u32),
    maxDynamicUniformBuffersPerPipelineLayout: u32 = std.math.maxInt(u32),
    maxDynamicStorageBuffersPerPipelineLayout: u32 = std.math.maxInt(u32),
    maxSampledTexturesPerShaderStage: u32 = std.math.maxInt(u32),
    maxSamplersPerShaderStage: u32 = std.math.maxInt(u32),
    maxStorageBuffersPerShaderStage: u32 = std.math.maxInt(u32),
    maxStorageTexturesPerShaderStage: u32 = std.math.maxInt(u32),
    maxUniformBuffersPerShaderStage: u32 = std.math.maxInt(u32),
    maxUniformBufferBindingSize: u64 = std.math.maxInt(u64),
    maxStorageBufferBindingSize: u64 = std.math.maxInt(u64),
    minUniformBufferOffsetAlignment: u32 = std.math.maxInt(u32),
    minStorageBufferOffsetAlignment: u32 = std.math.maxInt(u32),
    maxVertexBuffers: u32 = std.math.maxInt(u32),
    maxBufferSize: u64 = std.math.maxInt(u64),
    maxVertexAttributes: u32 = std.math.maxInt(u32),
    maxVertexBufferArrayStride: u32 = std.math.maxInt(u32),
    maxInterStageShaderVariables: u32 = std.math.maxInt(u32),
    maxColorAttachments: u32 = std.math.maxInt(u32),
    maxColorAttachmentBytesPerSample: u32 = std.math.maxInt(u32),
    maxComputeWorkgroupStorageSize: u32 = std.math.maxInt(u32),
    maxComputeInvocationsPerWorkgroup: u32 = std.math.maxInt(u32),
    maxComputeWorkgroupSizeX: u32 = std.math.maxInt(u32),
    maxComputeWorkgroupSizeY: u32 = std.math.maxInt(u32),
    maxComputeWorkgroupSizeZ: u32 = std.math.maxInt(u32),
    maxComputeWorkgroupsPerDimension: u32 = std.math.maxInt(u32),
    maxImmediateSize: u32 = std.math.maxInt(u32),
};

const WGPUBufferDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
    usage: WGPUBufferUsage = 0,
    size: u64 = 0,
    mapped_at_creation: u32 = 0,
};

const WGPUShaderModuleDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
};

const WGPUShaderSourceWGSL = extern struct {
    chain: WGPUChainedStruct = .{ .s_type = 2 }, // WGPUSType_ShaderSourceWGSL
    code: WGPUStringView = .{},
};

const WGPUComputePipelineDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
    layout: WGPUPipelineLayout = null,
    compute: extern struct {
        next_in_chain: ?*anyopaque = null,
        module: WGPUShaderModule = null,
        entry_point: WGPUStringView = .{},
        constant_count: usize = 0,
        constants: ?*anyopaque = null,
    } = .{},
};

const WGPUBindGroupEntry = extern struct {
    next_in_chain: ?*anyopaque = null,
    binding: u32 = 0,
    buffer: WGPUBuffer = null,
    offset: u64 = 0,
    size: u64 = 0,
    sampler: ?*anyopaque = null,
    texture_view: ?*anyopaque = null,
};

const WGPUBindGroupDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: WGPUStringView = .{},
    layout: WGPUBindGroupLayout = null,
    entry_count: usize = 0,
    entries: ?[*]const WGPUBindGroupEntry = null,
};

// ── Tuning constants ────────────────────────────────────────────────

const workgroup_size: u32 = 256;
const act_pool_capacity: u32 = 32;
/// Max params buffers to defer-destroy per sync cycle.
const max_deferred_destroy: u32 = 256;
/// Max vocab rows uploaded to GPU for f32 embedding lookup.
const emb_max_vocab: usize = 256000;
/// Max heads for deltaNet stack arrays (gate, beta, gate_beta).
const delta_net_max_heads: usize = 256;
/// Max Q+K elements for deltaNet combined buffer (num_k_heads * head_k_dim * 2).
const delta_net_max_qk_elems: usize = 32768;
const max_bindings: u32 = 8;
/// Maximum device poll iterations for buffer map completion.
const max_map_poll_iterations: u32 = 10000;

// ── Pipeline info ───────────────────────────────────────────────────

const PipelineInfo = struct {
    pipeline: WGPUComputePipeline = null,
    bind_group_layout: WGPUBindGroupLayout = null,
};

// ── Buffer pool entry ───────────────────────────────────────────────

const PoolEntry = struct {
    buffer: WGPUBuffer = null,
    size: usize = 0,
    in_use: bool = false,
};

// ── Cached buffer ───────────────────────────────────────────────────

const CachedBuf = struct {
    buffer: WGPUBuffer,
    size: usize,
    generation: u32 = 0,
};

// ── WebGPU Backend ──────────────────────────────────────────────────

/// WebGPU compute backend for cross-platform GPU inference via WGSL shaders.
pub const WebGpuBackend = struct {
    const max_dirty_entries: usize = 512;
    const DirtyEntry = struct { buf: WGPUBuffer, ptr: [*]f32, count: usize };

    allocator: std.mem.Allocator,
    lib: std.DynLib,

    // WebGPU objects
    instance: WGPUInstance = null,
    adapter: WGPUAdapter = null,
    device: WGPUDevice = null,
    queue: WGPUQueue = null,

    // Pipelines
    pipe_silu: PipelineInfo = .{},
    pipe_gelu: PipelineInfo = .{},
    pipe_add: PipelineInfo = .{},
    pipe_mul: PipelineInfo = .{},
    pipe_silu_mul: PipelineInfo = .{},
    pipe_gelu_mul: PipelineInfo = .{},
    pipe_rms_norm: PipelineInfo = .{},
    pipe_softmax: PipelineInfo = .{},
    pipe_rope: PipelineInfo = .{},
    pipe_embedding: PipelineInfo = .{},
    pipe_gemv_f32: PipelineInfo = .{},
    pipe_gemv_bf16: PipelineInfo = .{},
    pipe_gemv_f16: PipelineInfo = .{},
    pipe_gemv_fp8_e4m3: PipelineInfo = .{},
    pipe_gemv_fp8_e5m2: PipelineInfo = .{},
    pipe_gemv_q4_1: PipelineInfo = .{},
    pipe_gemv_q5_0: PipelineInfo = .{},
    pipe_gemv_q2_k: PipelineInfo = .{},
    pipe_gemv_q3_k: PipelineInfo = .{},
    pipe_gemv_iq4_nl: PipelineInfo = .{},
    pipe_gemv_iq4_xs: PipelineInfo = .{},
    pipe_gemv_q8_0: PipelineInfo = .{},
    pipe_gemv_q4_0: PipelineInfo = .{},
    pipe_gemv_q4_k: PipelineInfo = .{},
    pipe_gemv_q5_k: PipelineInfo = .{},
    pipe_gemv_q6_k: PipelineInfo = .{},
    pipe_sigmoid_mul: PipelineInfo = .{},
    pipe_l2_norm: PipelineInfo = .{},
    pipe_rms_norm_multi: PipelineInfo = .{},
    pipe_deinterleave: PipelineInfo = .{},
    pipe_split_qgate: PipelineInfo = .{},
    pipe_add_rms_norm: PipelineInfo = .{},
    pipe_rms_norm_add: PipelineInfo = .{},
    pipe_add_scaled: PipelineInfo = .{},
    pipe_gemv_t_q8_0: PipelineInfo = .{},
    pipe_gemv_nvfp4_st: PipelineInfo = .{},
    pipe_gemv_mlx_q4: PipelineInfo = .{},
    pipe_gemv_mxfp4_st: PipelineInfo = .{},
    pipe_gemv_gptq: PipelineInfo = .{},
    pipe_gemv_awq: PipelineInfo = .{},
    pipe_gemv_hqq: PipelineInfo = .{},
    pipe_gemv_tq1_0: PipelineInfo = .{},
    pipe_gemv_tq2_0: PipelineInfo = .{},
    pipe_sdpa: PipelineInfo = .{},
    pipe_conv1d: PipelineInfo = .{},
    pipe_deltanet: PipelineInfo = .{},
    pipe_sdpa_tree: PipelineInfo = .{},
    pipe_sdpa_paged: PipelineInfo = .{},

    // Buffer management
    buf_cache: std.AutoHashMap(usize, CachedBuf) = undefined,
    act_pool: [act_pool_capacity]PoolEntry = [_]PoolEntry{.{}} ** act_pool_capacity,
    act_pool_count: u32 = 0,

    // Deferred buffer destruction: params buffers can't be destroyed until
    // the command buffer that references them has been submitted + completed.
    deferred_destroy: [max_deferred_destroy]WGPUBuffer = .{null} ** max_deferred_destroy,
    deferred_count: u32 = 0,

    // Staging buffer for readbacks
    staging_buf: WGPUBuffer = null,
    staging_size: usize = 0,

    // Cached staging buffers for paged SDPA (avoid hot-path allocation)
    sdpa_flat_keys: ?[]f32 = null,
    sdpa_flat_vals: ?[]f32 = null,
    upload_generation: u32 = 0,

    // Batched command encoding
    batch_encoder: WGPUCommandEncoder = null,
    batch_pass: WGPUComputePassEncoder = null,
    in_batch: bool = false,

    // Lazy readback: GPU results cached by CPU pointer, downloaded on sync()
    dirty_bufs: [max_dirty_entries]DirtyEntry = undefined,
    dirty_count: u32 = 0,

    // WebGPU C function pointers
    fn_create_instance: *const fn (?*const WGPUInstanceDescriptor) callconv(.c) WGPUInstance = undefined,
    fn_instance_request_adapter: *const fn (WGPUInstance, ?*const WGPURequestAdapterOptions, WGPURequestAdapterCallbackInfo) callconv(.c) WGPUFuture = undefined,
    fn_adapter_request_device: *const fn (WGPUAdapter, ?*const WGPUDeviceDescriptor, WGPURequestDeviceCallbackInfo) callconv(.c) WGPUFuture = undefined,
    fn_instance_process_events: *const fn (WGPUInstance) callconv(.c) void = undefined,
    fn_device_get_queue: *const fn (WGPUDevice) callconv(.c) WGPUQueue = undefined,
    fn_device_create_buffer: *const fn (WGPUDevice, *const WGPUBufferDescriptor) callconv(.c) WGPUBuffer = undefined,
    fn_device_create_shader_module: *const fn (WGPUDevice, *const WGPUShaderModuleDescriptor) callconv(.c) WGPUShaderModule = undefined,
    fn_device_create_compute_pipeline: *const fn (WGPUDevice, *const WGPUComputePipelineDescriptor) callconv(.c) WGPUComputePipeline = undefined,
    fn_device_create_command_encoder: *const fn (WGPUDevice, ?*anyopaque) callconv(.c) WGPUCommandEncoder = undefined,
    fn_device_create_bind_group: *const fn (WGPUDevice, *const WGPUBindGroupDescriptor) callconv(.c) WGPUBindGroup = undefined,
    fn_device_poll: *const fn (WGPUDevice, u32, ?*anyopaque) callconv(.c) u32 = undefined,
    fn_compute_pipeline_get_bind_group_layout: *const fn (WGPUComputePipeline, u32) callconv(.c) WGPUBindGroupLayout = undefined,
    fn_command_encoder_begin_compute_pass: *const fn (WGPUCommandEncoder, ?*anyopaque) callconv(.c) WGPUComputePassEncoder = undefined,
    fn_command_encoder_copy_buffer_to_buffer: *const fn (WGPUCommandEncoder, WGPUBuffer, u64, WGPUBuffer, u64, u64) callconv(.c) void = undefined,
    fn_command_encoder_finish: *const fn (WGPUCommandEncoder, ?*anyopaque) callconv(.c) WGPUCommandBuffer = undefined,
    fn_compute_pass_set_pipeline: *const fn (WGPUComputePassEncoder, WGPUComputePipeline) callconv(.c) void = undefined,
    fn_compute_pass_set_bind_group: *const fn (WGPUComputePassEncoder, u32, WGPUBindGroup, usize, ?*const u32) callconv(.c) void = undefined,
    fn_compute_pass_dispatch: *const fn (WGPUComputePassEncoder, u32, u32, u32) callconv(.c) void = undefined,
    fn_compute_pass_end: *const fn (WGPUComputePassEncoder) callconv(.c) void = undefined,
    fn_queue_submit: *const fn (WGPUQueue, usize, *const WGPUCommandBuffer) callconv(.c) void = undefined,
    fn_queue_write_buffer: *const fn (WGPUQueue, WGPUBuffer, u64, ?*const anyopaque, usize) callconv(.c) void = undefined,
    fn_buffer_map_async: *const fn (WGPUBuffer, WGPUMapMode, usize, usize, WGPUBufferMapCallbackInfo) callconv(.c) WGPUFuture = undefined,
    fn_buffer_get_mapped_range: *const fn (WGPUBuffer, usize, usize) callconv(.c) ?*anyopaque = undefined,
    fn_buffer_unmap: *const fn (WGPUBuffer) callconv(.c) void = undefined,
    fn_buffer_destroy: *const fn (WGPUBuffer) callconv(.c) void = undefined,
    fn_buffer_release: *const fn (WGPUBuffer) callconv(.c) void = undefined,
    fn_instance_release: *const fn (WGPUInstance) callconv(.c) void = undefined,
    fn_adapter_release: *const fn (WGPUAdapter) callconv(.c) void = undefined,
    fn_device_release: *const fn (WGPUDevice) callconv(.c) void = undefined,
    fn_shader_module_release: *const fn (WGPUShaderModule) callconv(.c) void = undefined,
    fn_pipeline_release: *const fn (WGPUComputePipeline) callconv(.c) void = undefined,
    fn_bind_group_release: *const fn (WGPUBindGroup) callconv(.c) void = undefined,
    fn_bind_group_layout_release: *const fn (WGPUBindGroupLayout) callconv(.c) void = undefined,
    fn_command_encoder_release: *const fn (WGPUCommandEncoder) callconv(.c) void = undefined,
    fn_command_buffer_release: *const fn (WGPUCommandBuffer) callconv(.c) void = undefined,

    // ── Initialization ──────────────────────────────────────────

    /// Initialize the WebGPU backend: load the wgpu-native dynamic library,
    /// resolve function pointers, create the GPU instance/adapter/device, and
    /// compile all compute shader pipelines.
    pub fn init(allocator: std.mem.Allocator) !WebGpuBackend {
        var self = WebGpuBackend{ .allocator = allocator, .lib = undefined };
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);

        const lib_names = switch (@import("builtin").os.tag) {
            .macos => [_][:0]const u8{ "libwgpu_native.dylib", "/opt/homebrew/lib/libwgpu_native.dylib", "/usr/local/lib/libwgpu_native.dylib" },
            .linux => [_][:0]const u8{ "libwgpu_native.so", "/usr/lib/libwgpu_native.so", "/usr/local/lib/libwgpu_native.so" },
            .windows => [_][:0]const u8{ "wgpu_native.dll", "wgpu_native.dll", "wgpu_native.dll" },
            else => return error.WebGpuNotAvailable,
        };
        self.lib = for (lib_names) |name| {
            break std.DynLib.open(name) catch continue;
        } else return error.WebGpuNotAvailable;
        errdefer self.lib.close();
        std.log.info("WebGPU: library loaded", .{});

        self.loadFunctions() catch |err| {
            std.log.warn("WebGPU: loadFunctions failed: {s}", .{@errorName(err)});
            return error.WebGpuNotAvailable;
        };
        std.log.warn("WebGPU: functions loaded", .{});

        const desc = WGPUInstanceDescriptor{};
        self.instance = self.fn_create_instance(&desc);
        if (self.instance == null) {
            std.log.warn("WebGPU: wgpuCreateInstance returned null", .{});
            return error.WebGpuNotAvailable;
        }
        errdefer self.fn_instance_release(self.instance);

        self.requestAdapter() catch |err| {
            std.log.warn("WebGPU: requestAdapter failed: {s}", .{@errorName(err)});
            return err;
        };
        errdefer self.fn_adapter_release(self.adapter);
        self.requestDevice() catch |err| {
            std.log.warn("WebGPU: requestDevice failed: {s}", .{@errorName(err)});
            return err;
        };
        errdefer self.fn_device_release(self.device);
        self.queue = self.fn_device_get_queue(self.device);

        try self.createPipelines();

        return self;
    }

    fn loadFunctions(self: *WebGpuBackend) !void {
        self.fn_create_instance = self.lib.lookup(@TypeOf(self.fn_create_instance), "wgpuCreateInstance") orelse return error.WebGpuNotAvailable;
        self.fn_instance_request_adapter = self.lib.lookup(@TypeOf(self.fn_instance_request_adapter), "wgpuInstanceRequestAdapter") orelse return error.WebGpuNotAvailable;
        self.fn_adapter_request_device = self.lib.lookup(@TypeOf(self.fn_adapter_request_device), "wgpuAdapterRequestDevice") orelse return error.WebGpuNotAvailable;
        self.fn_instance_process_events = self.lib.lookup(@TypeOf(self.fn_instance_process_events), "wgpuInstanceProcessEvents") orelse return error.WebGpuNotAvailable;
        self.fn_device_get_queue = self.lib.lookup(@TypeOf(self.fn_device_get_queue), "wgpuDeviceGetQueue") orelse return error.WebGpuNotAvailable;
        self.fn_device_create_buffer = self.lib.lookup(@TypeOf(self.fn_device_create_buffer), "wgpuDeviceCreateBuffer") orelse return error.WebGpuNotAvailable;
        self.fn_device_create_shader_module = self.lib.lookup(@TypeOf(self.fn_device_create_shader_module), "wgpuDeviceCreateShaderModule") orelse return error.WebGpuNotAvailable;
        self.fn_device_create_compute_pipeline = self.lib.lookup(@TypeOf(self.fn_device_create_compute_pipeline), "wgpuDeviceCreateComputePipeline") orelse return error.WebGpuNotAvailable;
        self.fn_device_create_command_encoder = self.lib.lookup(@TypeOf(self.fn_device_create_command_encoder), "wgpuDeviceCreateCommandEncoder") orelse return error.WebGpuNotAvailable;
        self.fn_device_create_bind_group = self.lib.lookup(@TypeOf(self.fn_device_create_bind_group), "wgpuDeviceCreateBindGroup") orelse return error.WebGpuNotAvailable;
        self.fn_device_poll = self.lib.lookup(@TypeOf(self.fn_device_poll), "wgpuDevicePoll") orelse return error.WebGpuNotAvailable;
        self.fn_compute_pipeline_get_bind_group_layout = self.lib.lookup(@TypeOf(self.fn_compute_pipeline_get_bind_group_layout), "wgpuComputePipelineGetBindGroupLayout") orelse return error.WebGpuNotAvailable;
        self.fn_command_encoder_begin_compute_pass = self.lib.lookup(@TypeOf(self.fn_command_encoder_begin_compute_pass), "wgpuCommandEncoderBeginComputePass") orelse return error.WebGpuNotAvailable;
        self.fn_command_encoder_copy_buffer_to_buffer = self.lib.lookup(@TypeOf(self.fn_command_encoder_copy_buffer_to_buffer), "wgpuCommandEncoderCopyBufferToBuffer") orelse return error.WebGpuNotAvailable;
        self.fn_command_encoder_finish = self.lib.lookup(@TypeOf(self.fn_command_encoder_finish), "wgpuCommandEncoderFinish") orelse return error.WebGpuNotAvailable;
        self.fn_compute_pass_set_pipeline = self.lib.lookup(@TypeOf(self.fn_compute_pass_set_pipeline), "wgpuComputePassEncoderSetPipeline") orelse return error.WebGpuNotAvailable;
        self.fn_compute_pass_set_bind_group = self.lib.lookup(@TypeOf(self.fn_compute_pass_set_bind_group), "wgpuComputePassEncoderSetBindGroup") orelse return error.WebGpuNotAvailable;
        self.fn_compute_pass_dispatch = self.lib.lookup(@TypeOf(self.fn_compute_pass_dispatch), "wgpuComputePassEncoderDispatchWorkgroups") orelse return error.WebGpuNotAvailable;
        self.fn_compute_pass_end = self.lib.lookup(@TypeOf(self.fn_compute_pass_end), "wgpuComputePassEncoderEnd") orelse return error.WebGpuNotAvailable;
        self.fn_queue_submit = self.lib.lookup(@TypeOf(self.fn_queue_submit), "wgpuQueueSubmit") orelse return error.WebGpuNotAvailable;
        self.fn_queue_write_buffer = self.lib.lookup(@TypeOf(self.fn_queue_write_buffer), "wgpuQueueWriteBuffer") orelse return error.WebGpuNotAvailable;
        self.fn_buffer_map_async = self.lib.lookup(@TypeOf(self.fn_buffer_map_async), "wgpuBufferMapAsync") orelse return error.WebGpuNotAvailable;
        self.fn_buffer_get_mapped_range = self.lib.lookup(@TypeOf(self.fn_buffer_get_mapped_range), "wgpuBufferGetMappedRange") orelse return error.WebGpuNotAvailable;
        self.fn_buffer_unmap = self.lib.lookup(@TypeOf(self.fn_buffer_unmap), "wgpuBufferUnmap") orelse return error.WebGpuNotAvailable;
        self.fn_buffer_destroy = self.lib.lookup(@TypeOf(self.fn_buffer_destroy), "wgpuBufferDestroy") orelse return error.WebGpuNotAvailable;
        self.fn_buffer_release = self.lib.lookup(@TypeOf(self.fn_buffer_release), "wgpuBufferRelease") orelse return error.WebGpuNotAvailable;
        self.fn_instance_release = self.lib.lookup(@TypeOf(self.fn_instance_release), "wgpuInstanceRelease") orelse return error.WebGpuNotAvailable;
        self.fn_adapter_release = self.lib.lookup(@TypeOf(self.fn_adapter_release), "wgpuAdapterRelease") orelse return error.WebGpuNotAvailable;
        self.fn_device_release = self.lib.lookup(@TypeOf(self.fn_device_release), "wgpuDeviceRelease") orelse return error.WebGpuNotAvailable;
        self.fn_shader_module_release = self.lib.lookup(@TypeOf(self.fn_shader_module_release), "wgpuShaderModuleRelease") orelse return error.WebGpuNotAvailable;
        self.fn_pipeline_release = self.lib.lookup(@TypeOf(self.fn_pipeline_release), "wgpuComputePipelineRelease") orelse return error.WebGpuNotAvailable;
        self.fn_bind_group_release = self.lib.lookup(@TypeOf(self.fn_bind_group_release), "wgpuBindGroupRelease") orelse return error.WebGpuNotAvailable;
        self.fn_bind_group_layout_release = self.lib.lookup(@TypeOf(self.fn_bind_group_layout_release), "wgpuBindGroupLayoutRelease") orelse return error.WebGpuNotAvailable;
        self.fn_command_encoder_release = self.lib.lookup(@TypeOf(self.fn_command_encoder_release), "wgpuCommandEncoderRelease") orelse return error.WebGpuNotAvailable;
        self.fn_command_buffer_release = self.lib.lookup(@TypeOf(self.fn_command_buffer_release), "wgpuCommandBufferRelease") orelse return error.WebGpuNotAvailable;
    }

    fn requestAdapter(self: *WebGpuBackend) !void {
        const AdapterCtx = struct { adapter: WGPUAdapter = null, ready: bool = false };
        var ctx = AdapterCtx{};
        const opts = WGPURequestAdapterOptions{};
        const cb_info = WGPURequestAdapterCallbackInfo{
            .mode = wgpu_callback_mode_allow_process_events,
            .callback = struct {
                fn cb(_: WGPURequestAdapterStatus, adapter: WGPUAdapter, _: WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
                    const c: *AdapterCtx = @ptrCast(@alignCast(userdata1));
                    c.adapter = adapter;
                    c.ready = true;
                }
            }.cb,
            .userdata1 = @ptrCast(&ctx),
        };
        const future = self.fn_instance_request_adapter(self.instance, &opts, cb_info);
        _ = future;
        self.fn_instance_process_events(self.instance);
        if (!ctx.ready or ctx.adapter == null) return error.WebGpuNotAvailable;
        self.adapter = ctx.adapter;
    }

    fn requestDevice(self: *WebGpuBackend) !void {
        const DeviceCtx = struct { device: WGPUDevice = null, ready: bool = false };
        var ctx = DeviceCtx{};
        var limits = WGPULimits{
            .maxBufferSize = 1024 * 1024 * 1024, // 1GB
            .maxStorageBufferBindingSize = 1024 * 1024 * 1024,
            .maxStorageBuffersPerShaderStage = 10,
        };
        var desc = WGPUDeviceDescriptor{
            .required_limits = @ptrCast(&limits),
        };
        const cb_info = WGPURequestDeviceCallbackInfo{
            .mode = wgpu_callback_mode_allow_process_events,
            .callback = struct {
                fn cb(_: WGPURequestDeviceStatus, device: WGPUDevice, _: WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
                    const c: *DeviceCtx = @ptrCast(@alignCast(userdata1));
                    c.device = device;
                    c.ready = true;
                }
            }.cb,
            .userdata1 = @ptrCast(&ctx),
        };
        const future = self.fn_adapter_request_device(self.adapter, &desc, cb_info);
        _ = future;
        self.fn_instance_process_events(self.instance);
        if (!ctx.ready or ctx.device == null) return error.WebGpuNotAvailable;
        self.device = ctx.device;
    }

    fn createPipelines(self: *WebGpuBackend) !void {
        self.pipe_silu = try self.createPipeline(wgsl_silu);
        self.pipe_gelu = try self.createPipeline(wgsl_gelu);
        self.pipe_add = try self.createPipeline(wgsl_add);
        self.pipe_mul = try self.createPipeline(wgsl_mul);
        self.pipe_silu_mul = try self.createPipeline(wgsl_silu_mul);
        self.pipe_gelu_mul = try self.createPipeline(wgsl_gelu_mul);
        self.pipe_rms_norm = try self.createPipeline(wgsl_rms_norm);
        self.pipe_softmax = try self.createPipeline(wgsl_softmax);
        self.pipe_rope = try self.createPipeline(wgsl_rope);
        self.pipe_embedding = try self.createPipeline(wgsl_embedding);
        self.pipe_gemv_f32 = try self.createPipeline(wgsl_gemv_f32);
        self.pipe_gemv_bf16 = try self.createPipeline(wgsl_gemv_bf16);
        self.pipe_gemv_f16 = try self.createPipeline(wgsl_gemv_f16);
        self.pipe_gemv_fp8_e4m3 = try self.createPipeline(wgsl_gemv_fp8_e4m3);
        self.pipe_gemv_fp8_e5m2 = try self.createPipeline(wgsl_gemv_fp8_e5m2);
        self.pipe_gemv_q4_1 = try self.createPipeline(wgsl_gemv_q4_1);
        self.pipe_gemv_q5_0 = try self.createPipeline(wgsl_gemv_q5_0);
        self.pipe_gemv_q2_k = try self.createPipeline(wgsl_gemv_q2_k);
        self.pipe_gemv_q3_k = try self.createPipeline(wgsl_gemv_q3_k);
        self.pipe_gemv_iq4_nl = try self.createPipeline(wgsl_gemv_iq4_nl);
        self.pipe_gemv_iq4_xs = try self.createPipeline(wgsl_gemv_iq4_xs);
        self.pipe_gemv_q8_0 = try self.createPipeline(wgsl_gemv_q8_0);
        self.pipe_gemv_q4_0 = try self.createPipeline(wgsl_gemv_q4_0);
        self.pipe_gemv_q4_k = try self.createPipeline(wgsl_gemv_q4_k);
        self.pipe_gemv_q5_k = try self.createPipeline(wgsl_gemv_q5_k);
        self.pipe_gemv_q6_k = try self.createPipeline(wgsl_gemv_q6_k);
        self.pipe_sigmoid_mul = try self.createPipeline(wgsl_sigmoid_mul);
        self.pipe_l2_norm = try self.createPipeline(wgsl_l2_norm);
        self.pipe_rms_norm_multi = try self.createPipeline(wgsl_rms_norm_multi);
        self.pipe_deinterleave = try self.createPipeline(wgsl_deinterleave);
        self.pipe_split_qgate = try self.createPipeline(wgsl_split_qgate);
        self.pipe_add_rms_norm = try self.createPipeline(wgsl_add_rms_norm);
        self.pipe_rms_norm_add = try self.createPipeline(wgsl_rms_norm_add);
        self.pipe_add_scaled = try self.createPipeline(wgsl_add_scaled);
        self.pipe_gemv_t_q8_0 = try self.createPipeline(wgsl_gemv_t_q8_0);
        self.pipe_gemv_nvfp4_st = try self.createPipeline(wgsl_gemv_nvfp4_st);
        self.pipe_gemv_mlx_q4 = try self.createPipeline(wgsl_gemv_mlx_q4);
        self.pipe_gemv_mxfp4_st = try self.createPipeline(wgsl_gemv_mxfp4_st);
        self.pipe_gemv_gptq = try self.createPipeline(wgsl_gemv_gptq);
        self.pipe_gemv_awq = try self.createPipeline(wgsl_gemv_awq);
        self.pipe_gemv_hqq = try self.createPipeline(wgsl_gemv_hqq);
        self.pipe_gemv_tq1_0 = try self.createPipeline(wgsl_gemv_tq1_0);
        self.pipe_gemv_tq2_0 = try self.createPipeline(wgsl_gemv_tq2_0);
        self.pipe_sdpa = try self.createPipeline(wgsl_sdpa);
        self.pipe_conv1d = try self.createPipeline(wgsl_conv1d);
        self.pipe_deltanet = try self.createPipeline(wgsl_deltanet);
        self.pipe_sdpa_tree = try self.createPipeline(wgsl_sdpa_tree);
        self.pipe_sdpa_paged = try self.createPipeline(wgsl_sdpa_paged);
    }

    fn createPipeline(self: *WebGpuBackend, wgsl_source: [:0]const u8) !PipelineInfo {
        var wgsl_src = WGPUShaderSourceWGSL{
            .chain = .{ .s_type = 2 }, // WGPUSType_ShaderSourceWGSL
            .code = .{ .data = wgsl_source.ptr, .length = wgsl_source.len },
        };
        var shader_desc = WGPUShaderModuleDescriptor{ .next_in_chain = @ptrCast(&wgsl_src) };
        const shader = self.fn_device_create_shader_module(self.device, &shader_desc);
        if (shader == null) return error.WebGpuShaderCompilationFailed;
        defer self.fn_shader_module_release(shader);

        var pipeline_desc = WGPUComputePipelineDescriptor{};
        pipeline_desc.compute.module = shader;
        pipeline_desc.compute.entry_point = WGPUStringView.fromSlice("main");
        const pipeline = self.fn_device_create_compute_pipeline(self.device, &pipeline_desc);
        if (pipeline == null) return error.WebGpuPipelineCreationFailed;

        const bgl = self.fn_compute_pipeline_get_bind_group_layout(pipeline, 0);
        return PipelineInfo{ .pipeline = pipeline, .bind_group_layout = bgl };
    }

    /// Release all GPU resources, pipelines, and cached buffers.
    pub fn deinit(self: *WebGpuBackend) void {
        // Flush any pending command buffer so deferred buffers are safe to destroy.
        if (self.batch_encoder != null) self.submitPending();

        // Destroy deferred buffers that haven't been flushed yet.
        for (self.deferred_destroy[0..self.deferred_count]) |buf| {
            if (buf != null) self.fn_buffer_destroy(buf);
        }
        self.deferred_count = 0;

        // Release all compute pipelines and their bind group layouts.
        inline for (@typeInfo(WebGpuBackend).@"struct".fields) |field| {
            if (comptime std.mem.startsWith(u8, field.name, "pipe_")) {
                if (field.type == PipelineInfo) {
                    const info = @field(self, field.name);
                    if (info.pipeline != null) self.fn_pipeline_release(info.pipeline);
                    if (info.bind_group_layout != null) self.fn_bind_group_layout_release(info.bind_group_layout);
                }
            }
        }

        // Release pool buffers
        for (&self.act_pool) |*entry| {
            if (entry.buffer != null) self.fn_buffer_destroy(entry.buffer);
        }
        // Release cached buffers
        var it = self.buf_cache.valueIterator();
        while (it.next()) |v| self.fn_buffer_destroy(v.buffer);
        self.buf_cache.deinit();

        if (self.staging_buf != null) self.fn_buffer_destroy(self.staging_buf);
        if (self.sdpa_flat_keys) |buf| self.allocator.free(buf);
        if (self.sdpa_flat_vals) |buf| self.allocator.free(buf);
        if (self.device != null) self.fn_device_release(self.device);
        if (self.adapter != null) self.fn_adapter_release(self.adapter);
        if (self.instance != null) self.fn_instance_release(self.instance);
        self.lib.close();
    }

    // ── Buffer Management ───────────────────────────────────────

    fn createBuffer(self: *WebGpuBackend, size: usize, usage: WGPUBufferUsage) WGPUBuffer {
        const desc = WGPUBufferDescriptor{
            .size = @intCast(size),
            .usage = usage,
        };
        return self.fn_device_create_buffer(self.device, &desc);
    }

    fn uploadToBuffer(self: *WebGpuBackend, buf: WGPUBuffer, data: *const anyopaque, size: usize) void {
        self.fn_queue_write_buffer(self.queue, buf, 0, data, size);
    }

    const max_buffer_size: usize = 1024 * 1024 * 1024; // 1GB — requested via device limits

    fn getOrUpload(self: *WebGpuBackend, ptr: *const anyopaque, size: usize) WGPUBuffer {
        const key = @intFromPtr(ptr);
        if (self.buf_cache.getPtr(key)) |cached| {
            if (cached.generation == self.upload_generation and cached.size >= size) {
                return cached.buffer;
            }
            if (cached.size < size) {
                self.fn_buffer_release(cached.buffer);
                const new_buf = self.createBuffer(size, wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src | wgpu_buffer_usage_copy_dst);
                self.uploadToBuffer(new_buf, ptr, size);
                cached.buffer = new_buf;
                cached.size = size;
                cached.generation = self.upload_generation;
                return new_buf;
            }
            self.uploadToBuffer(cached.buffer, ptr, size);
            cached.generation = self.upload_generation;
            return cached.buffer;
        }
        const buf = self.createBuffer(size, wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src | wgpu_buffer_usage_copy_dst);
        self.uploadToBuffer(buf, ptr, size);
        self.buf_cache.put(key, .{ .buffer = buf, .size = size, .generation = self.upload_generation }) catch |err| {
            std.log.warn("webgpu: buf_cache.put failed (key={d}): {s}", .{ key, @errorName(err) });
        };
        return buf;
    }

    fn getPooledBuf(self: *WebGpuBackend, size: usize) struct { buf: WGPUBuffer, idx: u32 } {
        // Find free buffer of sufficient size
        for (&self.act_pool, 0..) |*entry, i| {
            if (!entry.in_use and entry.buffer != null and entry.size >= size) {
                entry.in_use = true;
                return .{ .buf = entry.buffer, .idx = @intCast(i) };
            }
        }
        // Allocate new
        const buf = self.createBuffer(size, wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src | wgpu_buffer_usage_copy_dst);
        if (self.act_pool_count < act_pool_capacity) {
            const idx = self.act_pool_count;
            self.act_pool[idx] = .{ .buffer = buf, .size = size, .in_use = true };
            self.act_pool_count += 1;
            return .{ .buf = buf, .idx = idx };
        }
        // Pool is full — return buffer without a valid pool index.
        // Use act_pool_capacity as a sentinel so releasePooledBuf is a no-op.
        std.log.warn("webgpu: activation pool full (capacity={d}), buffer not tracked", .{act_pool_capacity});
        return .{ .buf = buf, .idx = act_pool_capacity };
    }

    fn releasePooledBuf(self: *WebGpuBackend, idx: u32) void {
        if (idx < act_pool_capacity) {
            self.act_pool[idx].in_use = false;
        }
        // idx == act_pool_capacity is the sentinel for untracked buffers — no-op.
    }

    fn downloadF32(self: *WebGpuBackend, src: WGPUBuffer, dst: [*]f32, count: usize) void {
        self.submitPending();
        const size = count * @sizeOf(f32);
        // Ensure staging buffer is large enough
        if (self.staging_buf == null or self.staging_size < size) {
            if (self.staging_buf != null) self.fn_buffer_destroy(self.staging_buf);
            self.staging_buf = self.createBuffer(size, wgpu_buffer_usage_copy_dst | wgpu_buffer_usage_map_read);
            self.staging_size = size;
        }
        // Copy GPU buffer → staging
        const encoder = self.fn_device_create_command_encoder(self.device, null);
        self.fn_command_encoder_copy_buffer_to_buffer(encoder, src, 0, self.staging_buf, 0, @intCast(size));
        const cmd = self.fn_command_encoder_finish(encoder, null);
        self.fn_queue_submit(self.queue, 1, &cmd);
        self.fn_command_buffer_release(cmd);
        self.fn_command_encoder_release(encoder);

        // Map staging → CPU
        var mapped = false;
        const map_cb_info = WGPUBufferMapCallbackInfo{
            .mode = wgpu_callback_mode_allow_process_events,
            .callback = struct {
                fn cb(_: WGPUBufferMapAsyncStatus, _: WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
                    const m: *bool = @ptrCast(@alignCast(userdata1));
                    m.* = true;
                }
            }.cb,
            .userdata1 = @ptrCast(&mapped),
        };
        _ = self.fn_buffer_map_async(self.staging_buf, wgpu_map_mode_read, 0, size, map_cb_info);
        // Poll device until map completes
        var polls: u32 = 0;
        while (!mapped and polls < max_map_poll_iterations) : (polls += 1) {
            _ = self.fn_device_poll(self.device, 1, null);
            self.fn_instance_process_events(self.instance);
        }
        const mapped_ptr = self.fn_buffer_get_mapped_range(self.staging_buf, 0, size);
        if (mapped_ptr) |p| {
            const src_slice: [*]const f32 = @ptrCast(@alignCast(p));
            @memcpy(dst[0..count], src_slice[0..count]);
        }
        self.fn_buffer_unmap(self.staging_buf);
        self.upload_generation +%= 1;
    }

    /// Register a GPU buffer as the cached result for a CPU pointer.
    /// Defers download to sync() — eliminates per-op CPU↔GPU round-trips.
    fn cacheGpuResult(self: *WebGpuBackend, ptr: [*]f32, buf: WGPUBuffer, size: usize) void {
        const key = @intFromPtr(ptr);
        if (self.buf_cache.getPtr(key)) |cached| {
            if (cached.buffer != buf) self.deferDestroy(cached.buffer);
            cached.buffer = buf;
            cached.size = size;
            cached.generation = self.upload_generation;
        } else {
            self.buf_cache.put(key, .{ .buffer = buf, .size = size, .generation = self.upload_generation }) catch {
                std.log.warn("WebGPU buffer cache insert failed (OOM), falling back to uncached path", .{});
                self.downloadF32(buf, ptr, size / @sizeOf(f32));
                self.deferDestroy(buf);
                return;
            };
        }
        if (self.dirty_count < max_dirty_entries) {
            self.dirty_bufs[self.dirty_count] = .{ .buf = buf, .ptr = ptr, .count = size / @sizeOf(f32) };
            self.dirty_count += 1;
        } else {
            self.downloadF32(buf, ptr, size / @sizeOf(f32));
        }
    }

    /// Create a GPU buffer for output (not pooled — stays alive in cache).
    fn createOutputBuf(self: *WebGpuBackend, size: usize) WGPUBuffer {
        return self.createBuffer(size, wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src | wgpu_buffer_usage_copy_dst);
    }

    // ── Dispatch Helpers ────────────────────────────────────────

    /// Align size up to 16 bytes (WebGPU minimum uniform buffer alignment).
    fn alignUniform(size: usize) usize {
        return (size + 15) & ~@as(usize, 15);
    }

    /// Create a uniform buffer, upload params data, return GPU buffer handle.
    fn createUniformBuf(self: *WebGpuBackend, comptime T: type, params: T) WGPUBuffer {
        const aligned_size = alignUniform(@sizeOf(T));
        const buf = self.createBuffer(aligned_size, wgpu_buffer_usage_uniform | wgpu_buffer_usage_copy_dst);
        const p = params;
        self.fn_queue_write_buffer(self.queue, buf, 0, @ptrCast(&p), @sizeOf(T));
        return buf;
    }

    /// Central dispatch: create bind group, encode compute pass into deferred encoder.
    /// Accumulates dispatches — submit happens at submitPending()/sync().
    fn dispatchCompute(self: *WebGpuBackend, pipe: PipelineInfo, entries: []const WGPUBindGroupEntry, workgroups_x: u32) void {
        var bg_desc = WGPUBindGroupDescriptor{
            .layout = pipe.bind_group_layout,
            .entry_count = entries.len,
            .entries = entries.ptr,
        };
        const bind_group = self.fn_device_create_bind_group(self.device, &bg_desc);
        if (bind_group == null) @panic("WebGPU: bind group creation failed — layout/entry mismatch");
        // Don't release bind group here — defer until sync() so referenced
        // buffers stay alive until the command buffer is submitted + completed.

        if (self.batch_encoder == null)
            self.batch_encoder = self.fn_device_create_command_encoder(self.device, null);

        const pass = self.fn_command_encoder_begin_compute_pass(self.batch_encoder, null);
        self.fn_compute_pass_set_pipeline(pass, pipe.pipeline);
        self.fn_compute_pass_set_bind_group(pass, 0, bind_group, 0, null);
        self.fn_compute_pass_dispatch(pass, @min(workgroups_x, 65535), 1, 1);
        self.fn_compute_pass_end(pass);
    }

    /// Submit all pending compute dispatches.
    fn submitPending(self: *WebGpuBackend) void {
        if (self.batch_encoder == null) return;
        const cmd = self.fn_command_encoder_finish(self.batch_encoder, null);
        self.fn_queue_submit(self.queue, 1, &cmd);
        self.fn_command_buffer_release(cmd);
        self.fn_command_encoder_release(self.batch_encoder);
        self.batch_encoder = null;
    }

    /// Create a WGPUBindGroupEntry for a storage buffer at the given binding slot.
    fn storageEntry(binding: u32, buf: WGPUBuffer, size: u64) WGPUBindGroupEntry {
        return .{ .binding = binding, .buffer = buf, .size = size };
    }

    /// Create a WGPUBindGroupEntry for a uniform buffer at the given binding slot.
    fn uniformEntry(binding: u32, buf: WGPUBuffer, comptime T: type) WGPUBindGroupEntry {
        return .{ .binding = binding, .buffer = buf, .size = alignUniform(@sizeOf(T)) };
    }

    // ── Core Compute Operations ─────────────────────────────────

    /// Element-wise SiLU activation: output[i] = input[i] * sigmoid(input[i]).
    pub fn silu(self: *WebGpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, out_buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_silu, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(output, out_buf, size);
    }

    /// Element-wise GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
    pub fn gelu(self: *WebGpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, out_buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gelu, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(output, out_buf, size);
    }

    /// Element-wise addition: out[i] = a[i] + b[i].
    pub fn add(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_add, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(out, out_buf, size);
    }

    /// Element-wise multiplication: out[i] = a[i] * b[i].
    pub fn mul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(out, out_buf, size);
    }

    /// Fused SiLU×mul (SwiGLU): out[i] = silu(a[i]) * b[i].
    pub fn siluMul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_silu_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(out, out_buf, size);
    }

    /// SwiGLU with clamped gate/up values to [-10, 10] (prevents exp overflow in SiLU).
    pub fn clampedSiluMul(self: *WebGpuBackend, gate: [*]const f32, up: [*]const f32, out: [*]f32, n: usize) void {
        // CPU fallback — sync first so GPU-written data is visible to CPU reads.
        self.sync();
        for (0..n) |idx| {
            const g = @min(@as(f32, 10.0), @max(@as(f32, -10.0), gate[idx]));
            const u = @min(@as(f32, 10.0), @max(@as(f32, -10.0), up[idx]));
            out[idx] = (g / (1.0 + @exp(-g))) * u;
        }
    }

    /// Fused GELU×mul (GeGLU): out[i] = gelu(a[i]) * b[i].
    pub fn geluMul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params_buf = self.createUniformBuf(Params, .{ .n = @intCast(n) });
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gelu_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(out, out_buf, size);
    }

    /// RMS normalization with learned scale.
    pub fn rmsNorm(self: *WebGpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, eps: f32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, w_buf, size),
            storageEntry(2, out_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rms_norm, &entries, 1);
        self.cacheGpuResult(output, out_buf, size);
    }

    /// Softmax over n elements in-place.
    pub fn softmax(self: *WebGpuBackend, data: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const data_buf = self.getOrUpload(@ptrCast(data), size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_softmax, &entries, 1);
        self.cacheGpuResult(data, data_buf, size);
    }

    /// Rotary position embedding (RoPE) applied in-place.
    pub fn rope(self: *WebGpuBackend, data: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const total_elems = n_heads * head_dim;
        const size = total_elems * @sizeOf(f32);
        const data_buf = self.getOrUpload(@ptrCast(data), size);

        const Params = extern struct {
            pos: u32,
            n_heads: u32,
            head_dim: u32,
            rope_dim: u32,
            theta: f32,
            _pad: [12]u8 = .{0} ** 12,
        };
        const params = Params{
            .pos = @intCast(pos),
            .n_heads = @intCast(n_heads),
            .head_dim = @intCast(head_dim),
            .rope_dim = @intCast(rope_dim),
            .theta = theta,
        };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const half_rope = n_heads * rope_dim / 2;
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rope, &entries, @intCast((half_rope + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(data, data_buf, size);
    }

    /// Embedding table lookup: copy row `token_id` from table into output.
    pub fn embLookup(self: *WebGpuBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        // For non-f32 tables, dequant the single row on host and upload f32.
        // This avoids uploading the entire multi-hundred-MB vocab table to GPU.
        if (table.dtype != .f32) {
            const quant = @import("../ops/quant.zig");
            const DType = @import("../format/format.zig").DType;
            const row_dtype: DType = switch (table.dtype) {
                .q8_0 => .q8_0,
                .bf16 => .bf16,
                .f16 => .f16,
                else => @panic("WebGPU embLookup: unsupported dtype"),
            };
            const bpe = backend_mod.weightBytes(row_dtype, 1, dim);
            const row_offset = @as(usize, token_id) * bpe;
            quant.dequantToF32(output[0..dim], table.data + row_offset, row_dtype, dim);
            return;
        }
        std.debug.assert(token_id < emb_max_vocab); // vocab exceeds emb_max_vocab
        const table_size = dim * @sizeOf(f32) * emb_max_vocab;
        const table_buf = self.getOrUpload(table.data, table_size);
        const out_size = dim * @sizeOf(f32);
        const out_buf = self.createOutputBuf(out_size);

        const Params = extern struct { n_embd: u32, dtype: u32, token_id_v: u32, _pad: u32 };
        const params = Params{ .n_embd = @intCast(dim), .dtype = 0, .token_id_v = token_id, ._pad = 0 };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, table_buf, table_size),
            storageEntry(1, out_buf, out_size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_embedding, &entries, @intCast((dim + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(output, out_buf, out_size);
    }

    /// General matrix-vector multiply: y[n] = W[n,k] @ x[k]. Dispatches by quant type.
    pub fn gemv(self: *WebGpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const x_size = k * @sizeOf(f32);
        const y_size = n * @sizeOf(f32);
        const x_buf = self.getOrUpload(@ptrCast(x), x_size);
        const out_buf = self.createOutputBuf(y_size);

        const Params = extern struct { n: u32, k: u32, row_offset: u32, _pad: u32 };

        const pipe = switch (w.dtype) {
            .f32 => self.pipe_gemv_f32,
            .bf16 => self.pipe_gemv_bf16,
            .f16 => self.pipe_gemv_f16,
            .fp8_e4m3 => self.pipe_gemv_fp8_e4m3,
            .fp8_e5m2 => self.pipe_gemv_fp8_e5m2,
            .q4_1 => self.pipe_gemv_q4_1,
            .q5_0 => self.pipe_gemv_q5_0,
            .q2_k => self.pipe_gemv_q2_k,
            .q3_k => self.pipe_gemv_q3_k,
            .iq4_nl => self.pipe_gemv_iq4_nl,
            .iq4_xs => self.pipe_gemv_iq4_xs,
            .q8_0 => self.pipe_gemv_q8_0,
            .q4_0 => self.pipe_gemv_q4_0,
            .q4_k => self.pipe_gemv_q4_k,
            .q5_k => self.pipe_gemv_q5_k,
            .q6_k => self.pipe_gemv_q6_k,
            .tq1_0 => self.pipe_gemv_tq1_0,
            .tq2_0 => self.pipe_gemv_tq2_0,
            // IQ2/IQ3/IQ1: no WebGPU shader — fail closed (no silent CPU fallback).
            .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m => @panic("WebGPU gemv: IQ2/IQ3/IQ1 kernels not implemented"),
            else => std.debug.panic("WebGPU gemv: unsupported weight dtype {s}", .{@tagName(w.dtype)}),
        };
        const nb32 = (k + 31) / 32;
        const nb256 = (k + 255) / 256;
        const w_size = switch (w.dtype) {
            .f32 => n * k * @sizeOf(f32),
            .bf16 => n * k * 2,
            .f16 => n * k * 2,
            .fp8_e4m3 => n * k,
            .fp8_e5m2 => n * k,
            .q8_0 => n * nb32 * 34,
            .q4_0 => n * nb32 * 18,
            .q4_1 => n * nb32 * 20,
            .q5_0 => n * nb32 * 22,
            .iq4_nl => n * nb32 * 18,
            .q2_k => n * nb256 * 84,
            .q3_k => n * nb256 * 110,
            .q4_k => n * nb256 * 144,
            .q5_k => n * nb256 * 176,
            .q6_k => n * nb256 * 210,
            .iq4_xs => n * nb256 * 136,
            .tq1_0 => n * nb256 * 54,
            .tq2_0 => n * nb256 * 66,
            else => unreachable,
        };
        const w_buf = self.getOrUpload(w.data, w_size);

        const max_dispatch: u32 = 65535;
        var row_offset: u32 = 0;
        while (row_offset < n) {
            const chunk = @min(@as(u32, @intCast(n)) - row_offset, max_dispatch);
            const params = Params{ .n = @intCast(n), .k = @intCast(k), .row_offset = row_offset, ._pad = 0 };
            const params_buf = self.createUniformBuf(Params, params);
            self.deferDestroy(params_buf);
            const entries = [_]WGPUBindGroupEntry{
                storageEntry(0, x_buf, x_size),
                storageEntry(1, w_buf, w_size),
                storageEntry(2, out_buf, y_size),
                uniformEntry(3, params_buf, Params),
            };
            self.dispatchCompute(pipe, &entries, chunk);
            row_offset += chunk;
        }
        self.cacheGpuResult(y, out_buf, y_size);
    }

    /// Batched GEMM via per-row GEMV: y[tok,n] = W[n,k] @ x[tok,k].
    pub fn gemm(self: *WebGpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        for (0..n_tok) |i| {
            self.gemv(x + i * n_in, w, y + i * n_out, n_out, n_in);
        }
    }

    /// L2 normalization in-place: data[i] /= sqrt(sum(data^2) + eps).
    pub fn l2Norm(self: *WebGpuBackend, data: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        const data_buf = self.getOrUpload(@ptrCast(data), size);

        const Params = extern struct { n: u32, eps: f32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_l2_norm, &entries, 1);
        self.cacheGpuResult(data, data_buf, size);
    }

    /// Fused residual add + RMS norm: data[i] += residual[i], then out = rmsNorm(data, weight, eps).
    pub fn addRmsNorm(self: *WebGpuBackend, data: [*]f32, residual: [*]const f32, weight: [*]const f32, out: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        // data is read_write: add residual in-place, then normalize into out
        const data_buf = self.getOrUpload(@ptrCast(data), size);
        const res_buf = self.getOrUpload(@ptrCast(residual), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), size);
        const out_buf = self.createOutputBuf(size);

        const Params = extern struct { n: u32, eps: f32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            storageEntry(1, res_buf, size),
            storageEntry(2, w_buf, size),
            storageEntry(3, out_buf, size),
            uniformEntry(4, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_add_rms_norm, &entries, 1);
        // Cache both data (modified in-place with residual add) and normalized output
        self.cacheGpuResult(data, data_buf, size);
        self.cacheGpuResult(out, out_buf, size);
    }

    /// Fused rmsNorm + accumulate: b[i] += rmsNorm(a, weight, eps)[i].
    pub fn rmsNormAdd(self: *WebGpuBackend, a: [*]const f32, weight: [*]const f32, b: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        const a_buf = self.getOrUpload(@ptrCast(a), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), size);
        const b_buf = self.getOrUpload(@ptrCast(b), size);
        const Params = extern struct { n: u32, eps: f32, _pad: [8]u8 = .{0} ** 8 };
        const p = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, a_buf, size),
            storageEntry(1, w_buf, size),
            storageEntry(2, b_buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rms_norm_add, &entries, 1);
        self.cacheGpuResult(b, b_buf, size);
    }

    /// Scaled addition: dst[i] += src[i] * scale.
    pub fn addScaled(self: *WebGpuBackend, src: [*]const f32, dst: [*]f32, scale: f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const src_buf = self.getOrUpload(@ptrCast(src), size);
        const dst_buf = self.getOrUpload(@ptrCast(dst), size);
        const Params = extern struct { n: u32, scale: f32 };
        const params = Params{ .n = @intCast(n), .scale = scale };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, src_buf, size),
            storageEntry(1, dst_buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_add_scaled, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(dst, dst_buf, size);
    }

    /// Element-wise sigmoid gating: data[i] *= sigmoid(gate[i]).
    pub fn sigmoidMul(self: *WebGpuBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        const size = n * @sizeOf(f32);
        // data is read_write (result written back in place)
        const data_buf = self.getOrUpload(@ptrCast(data), size);
        const gate_buf = self.getOrUpload(@ptrCast(gate), size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        const params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            storageEntry(1, gate_buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_sigmoid_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(data, data_buf, size);
    }

    pub fn deinterleave(self: *WebGpuBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        const in_size = n_pairs * stride * 2 * @sizeOf(f32);
        const out_size = n_pairs * stride * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), in_size);
        const out_a_buf = self.createOutputBuf(out_size);
        const out_b_buf = self.createOutputBuf(out_size);

        const Params = extern struct { stride: u32, n_pairs: u32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .stride = @intCast(stride), .n_pairs = @intCast(n_pairs) };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const total_elems = n_pairs * stride;
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, in_size),
            storageEntry(1, out_a_buf, out_size),
            storageEntry(2, out_b_buf, out_size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_deinterleave, &entries, @intCast((total_elems + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(out_a, out_a_buf, out_size);
        self.cacheGpuResult(out_b, out_b_buf, out_size);
    }

    pub fn splitQGate(self: *WebGpuBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, head_dim: usize, n_heads: usize) void {
        const total_elems = n_heads * head_dim;
        const in_size = total_elems * 2 * @sizeOf(f32);
        const out_size = total_elems * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(qg), in_size);
        const q_buf = self.createOutputBuf(out_size);
        const g_buf = self.createOutputBuf(out_size);

        const Params = extern struct { hd: u32, nh: u32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .hd = @intCast(head_dim), .nh = @intCast(n_heads) };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, in_size),
            storageEntry(1, q_buf, out_size),
            storageEntry(2, g_buf, out_size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_split_qgate, &entries, @intCast((total_elems + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(q_out, q_buf, out_size);
        self.cacheGpuResult(g_out, g_buf, out_size);
    }

    pub fn rmsNormMulti(self: *WebGpuBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        const total_elems = n_heads * head_dim;
        const size = total_elems * @sizeOf(f32);
        const w_size = head_dim * @sizeOf(f32);
        // data is read_write (normalized in-place)
        const data_buf = self.getOrUpload(@ptrCast(data), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), w_size);

        const Params = extern struct { n_heads: u32, head_dim: u32, eps: f32, _pad: u32 = 0 };
        const params = Params{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .eps = eps };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_buf, size),
            storageEntry(1, w_buf, w_size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rms_norm_multi, &entries, @intCast(n_heads));
        self.cacheGpuResult(data, data_buf, size);
    }

    pub fn rmsNormBatched(self: *WebGpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        for (0..n_tok) |t| {
            self.rmsNorm(input + t * dim, weight, output + t * dim, dim, eps);
        }
    }

    pub fn ropeBatched(self: *WebGpuBackend, data: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const stride = n_heads * head_dim;
        for (0..n_tok) |t| {
            self.rope(data + t * stride, positions[t], n_heads, head_dim, rope_dim, theta);
        }
    }

    /// Scaled dot-product attention (single-token decode). Appends k/v, computes QK^T, softmax, V lookup.
    pub fn sdpa(self: *WebGpuBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        if (kv_type_k != .f32 or kv_type_v != .f32)
            @panic("WebGPU sdpa: only f32 KV supported — use --kv-type f32");

        const kvd = nkv * hd;
        const sl = seq_len + 1;

        // Append new K/V to cache (CPU-side, then upload everything)
        self.sync();
        const f32_keys: [*]f32 = @ptrCast(@alignCast(keys.ptr));
        const f32_values: [*]f32 = @ptrCast(@alignCast(values.ptr));
        @memcpy(f32_keys[seq_len * kvd ..][0..kvd], k_new[0..kvd]);
        @memcpy(f32_values[seq_len * kvd ..][0..kvd], v_new[0..kvd]);

        const q_sz = nh * hd * @sizeOf(f32);
        const k_sz = sl * kvd * @sizeOf(f32);
        const v_sz = k_sz;
        const o_sz = q_sz;

        const q_buf = self.getOrUpload(@ptrCast(q), q_sz);
        // KV cache was just memcpy'd on CPU — upload fresh each time
        const k_buf = self.createOutputBuf(k_sz);
        self.uploadToBuffer(k_buf, @ptrCast(f32_keys), k_sz);
        const v_buf = self.createOutputBuf(v_sz);
        self.uploadToBuffer(v_buf, @ptrCast(f32_values), v_sz);
        const o_buf = self.createOutputBuf(o_sz);

        const Params = extern struct { nh_v: u32, nkv_v: u32, hd_v: u32, sl_v: u32, scale_v: f32 };
        const p = Params{
            .nh_v = @intCast(nh),
            .nkv_v = @intCast(nkv),
            .hd_v = @intCast(hd),
            .sl_v = @intCast(sl),
            .scale_v = scale,
        };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, q_buf, q_sz),
            storageEntry(1, k_buf, k_sz),
            storageEntry(2, v_buf, v_sz),
            storageEntry(3, o_buf, o_sz),
            uniformEntry(4, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_sdpa, &entries, @intCast(nh));
        self.cacheGpuResult(output, o_buf, o_sz);
    }

    /// SDPA with per-head max/sum statistics (for disaggregated attention merging).
    pub fn sdpaWithStats(self: *WebGpuBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, head_max: [*]f32, head_sum: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        self.sdpa(q, keys, values, k_new, v_new, output, nh, nkv, hd, seq_len, scale, kv_type_k, kv_type_v);
        // Identity stats (max=0, sum=1) — GPU SDPA output is already normalized
        for (0..nh) |h| {
            head_max[h] = 0.0;
            head_sum[h] = 1.0;
        }
    }

    /// Paged SDPA: block-table-indexed attention for non-contiguous KV cache.
    /// Appends k_new/v_new to paged cache, gathers blocks into flat contiguous
    /// buffers, uploads to GPU, then dispatches the paged SDPA WGSL kernel.
    /// f32 KV only (paged cache stores f32).
    pub fn sdpaPaged(self: *WebGpuBackend, q: [*]const f32, kv_view: PagedKvView, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, scale: f32, _: KvQuantType, _: KvQuantType) void {
        const kvd = nkv * hd;
        const sl = kv_view.seq_len + 1;

        // Flush pending GPU work so CPU memcpy is safe
        self.sync();

        // Append k_new/v_new at current seq_len position
        @memcpy(kv_view.keyPtrMut(kv_view.seq_len)[0..kvd], k_new[0..kvd]);
        @memcpy(kv_view.valuePtrMut(kv_view.seq_len)[0..kvd], v_new[0..kvd]);

        // Gather scattered blocks into flat contiguous buffers.
        const n_logical_blocks = (sl + @as(usize, kv_view.block_size) - 1) / @as(usize, kv_view.block_size);
        var max_phys: u32 = 0;
        for (kv_view.block_table[0..n_logical_blocks]) |phys_id| max_phys = @max(max_phys, phys_id);
        const n_phys_blocks: usize = @as(usize, max_phys) + 1;
        const block_stride = @as(usize, kv_view.block_size) * kvd;
        const flat_elems = n_phys_blocks * block_stride;
        const flat_bytes = flat_elems * @sizeOf(f32);

        // Grow with 2x capacity so decode rarely re-allocates as block count rises.
        if (self.sdpa_flat_keys == null or self.sdpa_flat_keys.?.len < flat_elems) {
            const new_cap = @max(flat_elems, if (self.sdpa_flat_keys) |old| old.len * 2 else flat_elems);
            if (self.sdpa_flat_keys) |old| self.allocator.free(old);
            self.sdpa_flat_keys = self.allocator.alloc(f32, new_cap) catch
                @panic("WebGPU sdpaPaged: out of memory for flat key staging buffer");
        }
        if (self.sdpa_flat_vals == null or self.sdpa_flat_vals.?.len < flat_elems) {
            const new_cap_v = @max(flat_elems, if (self.sdpa_flat_vals) |old| old.len * 2 else flat_elems);
            if (self.sdpa_flat_vals) |old| self.allocator.free(old);
            self.sdpa_flat_vals = self.allocator.alloc(f32, new_cap_v) catch
                @panic("WebGPU sdpaPaged: out of memory for flat value staging buffer");
        }
        const flat_keys = self.sdpa_flat_keys.?;
        const flat_vals = self.sdpa_flat_vals.?;

        for (kv_view.block_table[0..n_logical_blocks]) |phys_id| {
            const dst_off = @as(usize, phys_id) * block_stride;
            const blk = kv_view.blocks[phys_id];
            @memcpy(flat_keys[dst_off..][0..block_stride], blk.keys[0..block_stride]);
            @memcpy(flat_vals[dst_off..][0..block_stride], blk.values[0..block_stride]);
        }

        // GPU dispatch
        const q_sz = nh * hd * @sizeOf(f32);
        const bt_sz = n_logical_blocks * @sizeOf(u32);
        const o_sz = q_sz;

        const q_buf = self.getOrUpload(@ptrCast(q), q_sz);
        const k_buf = self.createOutputBuf(flat_bytes);
        self.uploadToBuffer(k_buf, @ptrCast(flat_keys.ptr), flat_bytes);
        const v_buf = self.createOutputBuf(flat_bytes);
        self.uploadToBuffer(v_buf, @ptrCast(flat_vals.ptr), flat_bytes);
        const o_buf = self.createOutputBuf(o_sz);
        const bt_buf = self.createOutputBuf(bt_sz);
        self.uploadToBuffer(bt_buf, @ptrCast(kv_view.block_table.ptr), bt_sz);

        const Params = extern struct { nh_v: u32, nkv_v: u32, hd_v: u32, sl_v: u32, scale_v: f32, paged_bs_v: u32 };
        const p = Params{
            .nh_v = @intCast(nh),
            .nkv_v = @intCast(nkv),
            .hd_v = @intCast(hd),
            .sl_v = @intCast(sl),
            .scale_v = scale,
            .paged_bs_v = kv_view.block_size,
        };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, q_buf, q_sz),
            storageEntry(1, k_buf, flat_bytes),
            storageEntry(2, v_buf, flat_bytes),
            storageEntry(3, o_buf, o_sz),
            storageEntry(4, bt_buf, bt_sz),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_sdpa_paged, &entries, @intCast(nh));
        self.cacheGpuResult(output, o_buf, o_sz);
    }

    /// Tree-structured SDPA for speculative decoding with ancestor masks.
    pub fn sdpaTree(self: *WebGpuBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        if (kv_type_k == .f32 and kv_type_v == .f32 and n_nodes > 0) {
            const kvd = nkv * hd;
            const q_sz = n_nodes * nh * hd * @sizeOf(f32);
            const kv_sz = prefix_len * kvd * @sizeOf(f32);
            const tk_sz = n_nodes * kvd * @sizeOf(f32);
            const mask_sz = n_nodes * 8 * @sizeOf(u64);

            const q_buf = self.getOrUpload(@ptrCast(q_all), q_sz);
            const pk_buf = self.getOrUpload(@ptrCast(prefix_keys), kv_sz);
            const pv_buf = self.getOrUpload(@ptrCast(prefix_values), kv_sz);
            const tk_buf = self.getOrUpload(@ptrCast(tree_keys), tk_sz);
            const tv_buf = self.getOrUpload(@ptrCast(tree_values), tk_sz);
            const o_buf = self.createOutputBuf(q_sz);
            const m_buf = self.getOrUpload(@ptrCast(ancestor_masks), mask_sz);

            const Params = extern struct { nh_v: u32, nkv_v: u32, hd_v: u32, prefix_len_v: u32, n_nodes_v: u32, scale_v: f32, _pad: [8]u8 = .{0} ** 8 };
            const p = Params{ .nh_v = @intCast(nh), .nkv_v = @intCast(nkv), .hd_v = @intCast(hd), .prefix_len_v = @intCast(prefix_len), .n_nodes_v = n_nodes, .scale_v = scale };
            const params_buf = self.createUniformBuf(Params, p);
            self.deferDestroy(params_buf);

            const entries = [_]WGPUBindGroupEntry{
                storageEntry(0, q_buf, q_sz),
                storageEntry(1, pk_buf, kv_sz),
                storageEntry(2, pv_buf, kv_sz),
                storageEntry(3, tk_buf, tk_sz),
                storageEntry(4, tv_buf, tk_sz),
                storageEntry(5, o_buf, q_sz),
                storageEntry(6, m_buf, mask_sz),
                uniformEntry(7, params_buf, Params),
            };
            self.dispatchCompute(self.pipe_sdpa_tree, &entries, n_nodes * @as(u32, @intCast(nh)));
            self.cacheGpuResult(output, o_buf, q_sz);
            return;
        }
        @panic("WebGPU sdpaTree: unsupported KV type (need f32); use --kv-type f32 or --backend cpu");
    }

    /// Multi-token prefill SDPA: processes n_tok query tokens against the full KV cache.
    pub fn sdpaPrefill(self: *WebGpuBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        const kvd = nkv * hd;
        const qkv_dim = nh * hd;
        for (0..n_tok) |t| {
            const seq_pos = prev_len + t;
            self.sdpa(q + t * qkv_dim, kv_keys, kv_values, k + t * kvd, v + t * kvd, output + t * qkv_dim, nh, nkv, hd, seq_pos, scale, kv_type_k, kv_type_v);
        }
    }

    /// Transposed GEMV for Q8_0 3D weights: y[out_dim] = W^T @ x[in_dim].
    /// W is stored as [in_dim rows, out_dim cols] in Q8_0 blocks.
    /// One workgroup per output element, threads stride over input rows.
    pub fn gemvT(self: *WebGpuBackend, x: [*]const f32, w: [*]const u8, y: [*]f32, out_dim: usize, in_dim: usize) void {
        const blocks_per_row = (out_dim + 31) / 32;
        const row_bytes = blocks_per_row * 34; // 34 bytes per Q8_0 block
        const x_size = in_dim * @sizeOf(f32);
        const w_size = in_dim * row_bytes;
        const y_size = out_dim * @sizeOf(f32);

        const x_buf = self.getOrUpload(@ptrCast(x), x_size);
        const w_buf = self.getOrUpload(@ptrCast(w), w_size);
        const out_buf = self.createOutputBuf(y_size);

        const Params = extern struct { out_dim_val: u32, in_dim_val: u32, _pad: [8]u8 = .{0} ** 8 };
        const params = Params{ .out_dim_val = @intCast(out_dim), .in_dim_val = @intCast(in_dim) };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_size),
            storageEntry(1, w_buf, w_size),
            storageEntry(2, out_buf, y_size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_t_q8_0, &entries, @intCast(out_dim));
        self.cacheGpuResult(y, out_buf, y_size);
    }

    pub fn gemvNvfp4St(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        const x_sz = k * @sizeOf(f32);
        const w_sz = n * k / 2;
        const s_sz = n * k / 16;
        const y_sz = n * @sizeOf(f32);
        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const w_buf = self.getOrUpload(@ptrCast(w_packed), w_sz);
        const s_buf = self.getOrUpload(@ptrCast(w_scales), s_sz);
        const y_buf = self.createOutputBuf(y_sz);
        const Params = extern struct { n: u32, k: u32 };
        const p = Params{ .n = @intCast(n), .k = @intCast(k) };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, w_buf, w_sz),
            storageEntry(2, s_buf, s_sz),
            storageEntry(3, y_buf, y_sz),
            uniformEntry(4, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_nvfp4_st, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }

    pub fn gemvMlxQ(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, w_biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32, _: u32) void {
        std.debug.assert(bits == 4);
        const x_sz = k * @sizeOf(f32);
        const gpr = (k + 63) / 64;
        const wpr = gpr * 8;
        const w_sz = n * wpr * @sizeOf(u32);
        const s_sz = n * gpr * @sizeOf(u16);
        const b_sz = s_sz;
        const y_sz = n * @sizeOf(f32);
        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const w_buf = self.getOrUpload(@ptrCast(w_packed), w_sz);
        const s_buf = self.getOrUpload(@ptrCast(w_scales), s_sz);
        const b_buf = self.getOrUpload(@ptrCast(w_biases), b_sz);
        const y_buf = self.createOutputBuf(y_sz);
        const Params = extern struct { n: u32, k: u32 };
        const p = Params{ .n = @intCast(n), .k = @intCast(k) };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, w_buf, w_sz),
            storageEntry(2, s_buf, s_sz),
            storageEntry(3, b_buf, b_sz),
            storageEntry(4, y_buf, y_sz),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_mlx_q4, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }
    pub fn gemvMlxQGpu(self: *WebGpuBackend, x: [*]const f32, w: [*]const u8, s: [*]const u8, b: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32, gs: u32) void { self.gemvMlxQ(x, w, s, b, y, n, k, bits, gs); }

    pub fn gemvMxfp4St(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, y: [*]f32, n: usize, k: usize, _: usize, _: @import("../ops/mlx.zig").Mxfp4ScaleFormat) void {
        const x_sz = k * @sizeOf(f32);
        const w_sz = n * k / 2;
        const mxfp4_gs: usize = 16; // NVIDIA MXFP4 group size (must match gemv_mxfp4_st.wgsl)
        const s_sz = n * ((k + mxfp4_gs - 1) / mxfp4_gs);
        const y_sz = n * @sizeOf(f32);
        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const w_buf = self.getOrUpload(@ptrCast(w_packed), w_sz);
        const s_buf = self.getOrUpload(@ptrCast(w_scales), s_sz);
        const y_buf = self.createOutputBuf(y_sz);
        const Params = extern struct { n: u32, k: u32 };
        const p = Params{ .n = @intCast(n), .k = @intCast(k) };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, w_buf, w_sz),
            storageEntry(2, s_buf, s_sz),
            storageEntry(3, y_buf, y_sz),
            uniformEntry(4, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_mxfp4_st, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }
    pub fn gemvMxfp4StGpu(self: *WebGpuBackend, x: [*]const f32, w: [*]const u8, s: [*]const u8, y: [*]f32, n: usize, k: usize, gs: usize, sf: @import("../ops/mlx.zig").Mxfp4ScaleFormat) void { self.gemvMxfp4St(x, w, s, y, n, k, gs, sf); }

    pub fn gemvGptq(self: *WebGpuBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        const words_per_row = k / 8;
        const n_groups = (k + group_size - 1) / group_size;
        const x_sz = k * @sizeOf(f32);
        const w_sz = n * words_per_row * @sizeOf(u32);
        const s_sz = n * n_groups * @sizeOf(u16);
        const z_sz = n_groups * ((n + 7) / 8) * @sizeOf(u32);
        const y_sz = n * @sizeOf(f32);

        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const w_buf = self.getOrUpload(@ptrCast(qweight), w_sz);
        const s_buf = self.getOrUpload(@ptrCast(scales), s_sz);
        const z_buf = self.getOrUpload(@ptrCast(qzeros), z_sz);
        const y_buf = self.createOutputBuf(y_sz);

        const Params = extern struct { n_v: u32, k_v: u32, gs: u32, _pad: u32 = 0 };
        const p = Params{ .n_v = @intCast(n), .k_v = @intCast(k), .gs = group_size };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, w_buf, w_sz),
            storageEntry(2, s_buf, s_sz),
            storageEntry(3, z_buf, z_sz),
            storageEntry(4, y_buf, y_sz),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_gptq, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }

    pub fn gemvAwq(self: *WebGpuBackend, x: [*]const f32, qweight: [*]const u32, scales: [*]const u16, qzeros: [*]const u32, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        const words_per_col = k / 8;
        const n_groups = (k + group_size - 1) / group_size;
        const x_sz = k * @sizeOf(f32);
        const w_sz = k * (n / 8) * @sizeOf(u32);
        const s_sz = n_groups * n * @sizeOf(u16);
        const z_sz = n_groups * (n / 8) * @sizeOf(u32);
        const y_sz = n * @sizeOf(f32);
        _ = words_per_col;

        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const w_buf = self.getOrUpload(@ptrCast(qweight), w_sz);
        const s_buf = self.getOrUpload(@ptrCast(scales), s_sz);
        const z_buf = self.getOrUpload(@ptrCast(qzeros), z_sz);
        const y_buf = self.createOutputBuf(y_sz);

        const Params = extern struct { n_v: u32, k_v: u32, gs: u32, _pad: u32 = 0 };
        const p = Params{ .n_v = @intCast(n), .k_v = @intCast(k), .gs = group_size };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, w_buf, w_sz),
            storageEntry(2, s_buf, s_sz),
            storageEntry(3, z_buf, z_sz),
            storageEntry(4, y_buf, y_sz),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_awq, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }

    /// HQQ INT4 GEMV on WebGPU.
    /// w_q: packed uint8 nibbles [n * k/2], scale/zero: bf16 packed two-per-u32 [n * k/group_size].
    pub fn gemvHqq(self: *WebGpuBackend, x: [*]const f32, w_q: [*]const u8, scale: [*]const u8, zero: [*]const u8, y: [*]f32, n: usize, k: usize, group_size: u32) void {
        const n_groups = (k + group_size - 1) / group_size;
        const x_sz = k * @sizeOf(f32);
        const wq_sz = n * (k / 2); // packed nibbles: 2 per byte
        const sq_sz = n * n_groups * @sizeOf(u16); // bf16 scale
        const zr_sz = n * n_groups * @sizeOf(u16); // bf16 zero
        const y_sz = n * @sizeOf(f32);

        const x_buf = self.getOrUpload(@ptrCast(x), x_sz);
        const wq_buf = self.getOrUpload(@ptrCast(w_q), wq_sz);
        const sc_buf = self.getOrUpload(@ptrCast(scale), sq_sz);
        const zr_buf = self.getOrUpload(@ptrCast(zero), zr_sz);
        const y_buf = self.createOutputBuf(y_sz);

        const Params = extern struct { n_v: u32, k_v: u32, gs: u32, _pad: u32 = 0 };
        const p = Params{ .n_v = @intCast(n), .k_v = @intCast(k), .gs = group_size };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, x_sz),
            storageEntry(1, wq_buf, wq_sz),
            storageEntry(2, sc_buf, sq_sz),
            storageEntry(3, zr_buf, zr_sz),
            storageEntry(4, y_buf, y_sz),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gemv_hqq, &entries, @intCast(n));
        self.cacheGpuResult(y, y_buf, y_sz);
    }

    pub fn gemvMulti(self: *WebGpuBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    pub fn causalConv1dSilu(self: *WebGpuBackend, x: [*]const f32, state: [*]f32, conv_w: [*]const f32, conv_bias: ?[*]const f32, output: [*]f32, conv_ch: usize, d_conv: usize) void {
        const ch_sz = conv_ch * @sizeOf(f32);
        const state_sz = (d_conv - 1) * conv_ch * @sizeOf(f32);
        const w_sz = d_conv * conv_ch * @sizeOf(f32);

        const x_buf = self.getOrUpload(@ptrCast(x), ch_sz);
        const s_buf = self.createOutputBuf(state_sz);
        self.uploadToBuffer(s_buf, @ptrCast(state), state_sz);
        const w_buf = self.getOrUpload(@ptrCast(conv_w), w_sz);
        const o_buf = self.createOutputBuf(ch_sz);

        var zero: f32 = 0.0;
        const b_buf = if (conv_bias) |b| self.getOrUpload(@ptrCast(b), ch_sz) else self.getOrUpload(@ptrCast(&zero), @sizeOf(f32));

        const Params = extern struct { conv_ch_v: u32, d_conv_v: u32, has_bias: u32, _pad: u32 };
        const p = Params{
            .conv_ch_v = @intCast(conv_ch),
            .d_conv_v = @intCast(d_conv),
            .has_bias = if (conv_bias != null) 1 else 0,
            ._pad = 0,
        };
        const params_buf = self.createUniformBuf(Params, p);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, x_buf, ch_sz),
            storageEntry(1, s_buf, state_sz),
            storageEntry(2, w_buf, w_sz),
            storageEntry(3, o_buf, ch_sz),
            storageEntry(4, b_buf, if (conv_bias != null) ch_sz else @sizeOf(f32)),
            uniformEntry(5, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_conv1d, &entries, @intCast((conv_ch + workgroup_size - 1) / workgroup_size));
        self.cacheGpuResult(output, o_buf, ch_sz);

        // Update conv state ring buffer (shift left, append new input)
        self.sync();
        const hist = d_conv - 1;
        if (hist > 1) {
            for (0..hist - 1) |i| {
                @memcpy(state[i * conv_ch ..][0..conv_ch], state[(i + 1) * conv_ch ..][0..conv_ch]);
            }
        }
        @memcpy(state[(hist - 1) * conv_ch ..][0..conv_ch], x[0..conv_ch]);
    }

    pub fn deltaNet(self: *WebGpuBackend, conv_in: [*]const f32, conv_out: [*]f32, z_buf: [*]const f32, alpha_buf: [*]const f32, beta_buf: [*]const f32, output: [*]f32, conv_state: [*]f32, ssm_state: []f32, ssm_a: [*]const f32, dt_bias: [*]const f32, conv_w: [*]const f32, ssm_norm_w: [*]const f32, p: backend_mod.DeltaNetParams) void {
        const math_ops = @import("../ops/math.zig");
        const num_v_heads = p.num_v_heads;
        const num_k_heads = p.num_k_heads;
        const head_k_dim = p.head_k_dim;
        const head_v_dim = p.head_v_dim;

        // 1. Gate & beta computation (small, on CPU)
        self.sync();
        std.debug.assert(num_v_heads <= delta_net_max_heads); // num_v_heads exceeds delta_net_max_heads
        var gate_arr: [delta_net_max_heads]f32 = undefined;
        var beta_arr: [delta_net_max_heads]f32 = undefined;
        for (0..num_v_heads) |h| {
            const alpha_biased = alpha_buf[h] + dt_bias[h];
            gate_arr[h] = ssm_a[h] * math_ops.softplus(alpha_biased);
            beta_arr[h] = math_ops.sigmoid(beta_buf[h]);
        }

        // 2. Conv1d + SiLU (GPU dispatch via existing kernel)
        self.causalConv1dSilu(conv_in, conv_state, conv_w, null, conv_out, p.conv_ch, p.d_conv);
        self.sync();

        // 3. L2 normalize Q and K per head (GPU dispatch)
        const q_off: usize = if (p.kqv_order) num_k_heads * head_k_dim else 0;
        const k_off: usize = if (p.kqv_order) 0 else num_k_heads * head_k_dim;
        for (0..num_k_heads) |h| {
            self.l2Norm(conv_out + q_off + h * head_k_dim, head_k_dim, p.rms_eps);
            self.l2Norm(conv_out + k_off + h * head_k_dim, head_k_dim, p.rms_eps);
        }
        self.sync();

        // 4. Recurrence + gated output (GPU kernel — 1 thread per head)
        const q_ptr = conv_out + q_off;
        const k_ptr = conv_out + k_off;
        const v_off: usize = 2 * num_k_heads * head_k_dim;
        const v_ptr = conv_out + v_off;

        const v_sz = num_v_heads * head_v_dim * @sizeOf(f32);
        const state_sz = ssm_state.len * @sizeOf(f32);
        const norm_sz = head_v_dim * @sizeOf(f32);
        const z_sz = v_sz;

        // Combine Q+K into one buffer to stay within 8 storage buffer limit
        std.debug.assert(num_k_heads * head_k_dim * 2 <= delta_net_max_qk_elems); // Q+K exceeds delta_net_max_qk_elems
        var qk_combined: [delta_net_max_qk_elems]f32 = undefined;
        const qk_elems = num_k_heads * head_k_dim;
        @memcpy(qk_combined[0..qk_elems], q_ptr[0..qk_elems]);
        @memcpy(qk_combined[qk_elems..][0..qk_elems], k_ptr[0..qk_elems]);
        const qk_sz = qk_elems * 2 * @sizeOf(f32);
        const qk_pool = self.getPooledBuf(qk_sz);
        defer self.releasePooledBuf(qk_pool.idx);
        self.uploadToBuffer(qk_pool.buf, @ptrCast(&qk_combined), qk_sz);
        const v_buf = self.getOrUpload(@ptrCast(v_ptr), v_sz);
        // Merge gate+beta into one buffer: [gates..., betas...]
        var gate_beta_arr: [delta_net_max_heads * 2]f32 = undefined;
        @memcpy(gate_beta_arr[0..num_v_heads], gate_arr[0..num_v_heads]);
        @memcpy(gate_beta_arr[num_v_heads..][0..num_v_heads], beta_arr[0..num_v_heads]);
        const gb_sz = num_v_heads * 2 * @sizeOf(f32);
        const gb_pool = self.getPooledBuf(gb_sz);
        defer self.releasePooledBuf(gb_pool.idx);
        self.uploadToBuffer(gb_pool.buf, @ptrCast(&gate_beta_arr), gb_sz);
        const z_buf_gpu = self.getOrUpload(@ptrCast(z_buf), z_sz);
        const norm_buf = self.getOrUpload(@ptrCast(ssm_norm_w), norm_sz);
        const state_pool = self.getPooledBuf(state_sz);
        defer self.releasePooledBuf(state_pool.idx);
        self.uploadToBuffer(state_pool.buf, @ptrCast(ssm_state.ptr), state_sz);
        const out_buf = self.createOutputBuf(v_sz);

        const Params = extern struct { num_v_heads_v: u32, num_k_heads_v: u32, head_k_dim_v: u32, head_v_dim_v: u32, q_scale_v: f32, rms_eps_v: f32 };
        const params = Params{
            .num_v_heads_v = @intCast(num_v_heads),
            .num_k_heads_v = @intCast(num_k_heads),
            .head_k_dim_v = @intCast(head_k_dim),
            .head_v_dim_v = @intCast(head_v_dim),
            .q_scale_v = p.q_scale,
            .rms_eps_v = p.rms_eps,
        };
        const params_buf = self.createUniformBuf(Params, params);
        self.deferDestroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, qk_pool.buf, qk_sz),
            storageEntry(1, v_buf, v_sz),
            storageEntry(2, gb_pool.buf, gb_sz),
            storageEntry(3, z_buf_gpu, z_sz),
            storageEntry(4, norm_buf, norm_sz),
            storageEntry(5, state_pool.buf, state_sz),
            storageEntry(6, out_buf, v_sz),
            uniformEntry(7, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_deltanet, &entries, @intCast(num_v_heads));
        self.cacheGpuResult(output, out_buf, v_sz);
        // ssm_state is persistent recurrent state — must download immediately
        self.downloadF32(state_pool.buf, @ptrCast(ssm_state.ptr), ssm_state.len);
    }

    // ── Sync + Batch + Memory ───────────────────────────────────

    /// Queue a buffer for destruction after the next sync() completes.
    /// Params buffers can't be destroyed immediately because the command
    /// buffer that references them hasn't been submitted yet.
    fn deferDestroy(self: *WebGpuBackend, buf: WGPUBuffer) void {
        if (self.deferred_count < max_deferred_destroy) {
            self.deferred_destroy[self.deferred_count] = buf;
            self.deferred_count += 1;
        } else {
            // Array is full — flush pending GPU work so all referenced buffers
            // are consumed, then clear the deferred array and add this buffer.
            // sync() calls submitPending() which sets batch_encoder = null,
            // and then iterates deferred_destroy[0..deferred_count] — it does
            // not call deferDestroy, so there is no re-entrancy risk.
            self.sync();
            // sync() resets deferred_count to 0, so we can safely add now.
            self.deferred_destroy[self.deferred_count] = buf;
            self.deferred_count += 1;
        }
    }

    /// Submit pending GPU work, download dirty buffers to CPU, and destroy deferred params.
    pub fn sync(self: *WebGpuBackend) void {
        self.submitPending();
        // Flush dirty buffers: download GPU results to CPU.
        // Deduplicate by pointer — only download the latest buffer for each CPU address.
        var di: u32 = self.dirty_count;
        while (di > 0) {
            di -= 1;
            const entry = self.dirty_bufs[di];
            // Check if a later entry overwrote this pointer
            var superseded = false;
            for (self.dirty_bufs[di + 1 .. self.dirty_count]) |later| {
                if (later.ptr == entry.ptr) {
                    superseded = true;
                    break;
                }
            }
            if (!superseded) {
                self.downloadF32(entry.buf, entry.ptr, entry.count);
            }
        }
        self.dirty_count = 0;
        _ = self.fn_device_poll(self.device, 1, null);
        // Destroy deferred params buffers now that GPU has consumed them
        for (self.deferred_destroy[0..self.deferred_count]) |buf| {
            if (buf != null) self.fn_buffer_destroy(buf);
        }
        self.deferred_count = 0;
        self.upload_generation +%= 1;
    }

    pub fn beginBatch(_: *WebGpuBackend) void {}

    pub fn endBatch(_: *WebGpuBackend) void {}

    pub fn allocKvSlice(_: *WebGpuBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        return allocator.alloc(u8, n);
    }

    pub fn freeKvSlice(_: *WebGpuBackend, allocator: std.mem.Allocator, slice: []u8) void {
        allocator.free(slice);
    }

    pub fn backendInfo(_: *WebGpuBackend) backend_mod.BackendInfo {
        return .{
            .name = "WebGPU",
            .kernel_type = "WGSL",
        };
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "WebGPU tuning constants are valid" {
    const testing = std.testing;

    try testing.expect(workgroup_size > 0);
    try testing.expect(workgroup_size <= 1024);
    try testing.expect(workgroup_size & (workgroup_size - 1) == 0); // power of 2

    try testing.expect(act_pool_capacity > 0);
    try testing.expect(act_pool_capacity <= 256);

    try testing.expect(max_deferred_destroy > 0);
    try testing.expect(max_bindings > 0);
    try testing.expect(max_bindings <= 16);

    try testing.expect(max_map_poll_iterations > 0);
}

test "WebGPU alignUniform rounds up to 16" {
    const testing = std.testing;

    // Already aligned
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(16));
    try testing.expectEqual(@as(usize, 32), WebGpuBackend.alignUniform(32));

    // Needs rounding
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(1));
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(4));
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(8));
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(12));
    try testing.expectEqual(@as(usize, 16), WebGpuBackend.alignUniform(15));
    try testing.expectEqual(@as(usize, 32), WebGpuBackend.alignUniform(17));
    try testing.expectEqual(@as(usize, 32), WebGpuBackend.alignUniform(20));

    // Zero edge case
    try testing.expectEqual(@as(usize, 0), WebGpuBackend.alignUniform(0));
}

test "WebGPU buffer usage flags are distinct" {
    const testing = std.testing;

    // Each usage flag must be a distinct power of 2
    try testing.expect(wgpu_buffer_usage_storage != wgpu_buffer_usage_copy_src);
    try testing.expect(wgpu_buffer_usage_copy_src != wgpu_buffer_usage_copy_dst);
    try testing.expect(wgpu_buffer_usage_copy_dst != wgpu_buffer_usage_map_read);
    try testing.expect(wgpu_buffer_usage_map_read != wgpu_buffer_usage_uniform);

    // Flags must be non-zero
    try testing.expect(wgpu_buffer_usage_storage > 0);
    try testing.expect(wgpu_buffer_usage_copy_src > 0);
    try testing.expect(wgpu_buffer_usage_copy_dst > 0);
    try testing.expect(wgpu_buffer_usage_map_read > 0);
    try testing.expect(wgpu_buffer_usage_uniform > 0);

    // Combination of storage | copy_src | copy_dst must not collide with map_read
    const combined = wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src | wgpu_buffer_usage_copy_dst;
    try testing.expect(combined & wgpu_buffer_usage_map_read == 0);
}

test "WebGPU binding type constants are distinct" {
    const testing = std.testing;

    try testing.expect(wgpu_buffer_binding_storage != wgpu_buffer_binding_read_only_storage);
    try testing.expect(wgpu_buffer_binding_read_only_storage != wgpu_buffer_binding_uniform);
    try testing.expect(wgpu_buffer_binding_storage != wgpu_buffer_binding_uniform);
}

test "WebGPU storageEntry and uniformEntry helpers" {
    const testing = std.testing;

    // storageEntry: binding, buffer, size
    const se = WebGpuBackend.storageEntry(3, null, 1024);
    try testing.expectEqual(@as(u32, 3), se.binding);
    try testing.expectEqual(@as(u64, 1024), se.size);

    // uniformEntry: binding, buffer, aligned size
    const ue = WebGpuBackend.uniformEntry(1, null, extern struct { n: u32, eps: f32 });
    try testing.expectEqual(@as(u32, 1), ue.binding);
    try testing.expectEqual(@as(u64, 16), ue.size); // alignUniform(8) = 16
}

test "WebGPU DirtyEntry and PoolEntry struct sizes" {
    const testing = std.testing;

    // DirtyEntry must hold a buffer handle, pointer, and count
    try testing.expect(@sizeOf(WebGpuBackend.DirtyEntry) >= @sizeOf(usize) * 2);

    // PoolEntry must hold a buffer handle, size, and in_use flag
    try testing.expect(@sizeOf(PoolEntry) >= @sizeOf(usize) + @sizeOf(bool));
}

test "WebGPU max_buffer_size is 1GB" {
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), WebGpuBackend.max_buffer_size);
}

test "WebGPU backend public function signatures compile" {
    comptime {
        // Core ops
        _ = @TypeOf(WebGpuBackend.silu);
        _ = @TypeOf(WebGpuBackend.gelu);
        _ = @TypeOf(WebGpuBackend.add);
        _ = @TypeOf(WebGpuBackend.mul);
        _ = @TypeOf(WebGpuBackend.siluMul);
        _ = @TypeOf(WebGpuBackend.geluMul);
        _ = @TypeOf(WebGpuBackend.rmsNorm);
        _ = @TypeOf(WebGpuBackend.softmax);
        _ = @TypeOf(WebGpuBackend.rope);
        _ = @TypeOf(WebGpuBackend.embLookup);
        _ = @TypeOf(WebGpuBackend.gemv);
        _ = @TypeOf(WebGpuBackend.gemm);
        _ = @TypeOf(WebGpuBackend.gemvMulti);
        _ = @TypeOf(WebGpuBackend.l2Norm);
        _ = @TypeOf(WebGpuBackend.addRmsNorm);
        _ = @TypeOf(WebGpuBackend.addScaled);
        _ = @TypeOf(WebGpuBackend.sigmoidMul);
        _ = @TypeOf(WebGpuBackend.deinterleave);
        _ = @TypeOf(WebGpuBackend.splitQGate);
        _ = @TypeOf(WebGpuBackend.rmsNormMulti);
        _ = @TypeOf(WebGpuBackend.gemvT);
        _ = @TypeOf(WebGpuBackend.gemvNvfp4St);
        _ = @TypeOf(WebGpuBackend.gemvMlxQ);
        _ = @TypeOf(WebGpuBackend.gemvMxfp4St);
        _ = @TypeOf(WebGpuBackend.gemvGptq);
        _ = @TypeOf(WebGpuBackend.gemvAwq);

        // SDPA
        _ = @TypeOf(WebGpuBackend.sdpa);
        _ = @TypeOf(WebGpuBackend.sdpaWithStats);
        _ = @TypeOf(WebGpuBackend.sdpaPaged);
        _ = @TypeOf(WebGpuBackend.sdpaPrefill);
        _ = @TypeOf(WebGpuBackend.sdpaTree);

        // Batched prefill
        _ = @TypeOf(WebGpuBackend.rmsNormBatched);
        _ = @TypeOf(WebGpuBackend.ropeBatched);

        // SSM
        _ = @TypeOf(WebGpuBackend.causalConv1dSilu);
        _ = @TypeOf(WebGpuBackend.deltaNet);

        // Sync / lifecycle
        _ = @TypeOf(WebGpuBackend.sync);
        _ = @TypeOf(WebGpuBackend.beginBatch);
        _ = @TypeOf(WebGpuBackend.endBatch);
        _ = @TypeOf(WebGpuBackend.init);
        _ = @TypeOf(WebGpuBackend.deinit);
        _ = @TypeOf(WebGpuBackend.backendInfo);

        // KV cache
        _ = @TypeOf(WebGpuBackend.allocKvSlice);
        _ = @TypeOf(WebGpuBackend.freeKvSlice);
    }
}

test "WebGPU WGPUStringView fromSlice" {
    const sv = WGPUStringView.fromSlice("main");
    try std.testing.expectEqual(@as(usize, 4), sv.length);
    try std.testing.expect(sv.data != null);
}

test "WebGPU deferred_destroy capacity matches max_deferred_destroy" {
    const testing = std.testing;

    // max_deferred_destroy must be a positive power of 2 for efficient modular arithmetic
    try testing.expect(max_deferred_destroy > 0);
    try testing.expect(max_deferred_destroy & (max_deferred_destroy - 1) == 0);

    // The deferred_destroy array in WebGpuBackend must have exactly max_deferred_destroy slots
    try testing.expectEqual(max_deferred_destroy, @as(u32, @intCast(@typeInfo(@TypeOf(@as(WebGpuBackend, undefined).deferred_destroy)).array.len)));

    // deferred_count is u32, so max_deferred_destroy must fit in u32
    try testing.expect(max_deferred_destroy <= std.math.maxInt(u32));

    // Verify the dirty_bufs array matches max_dirty_entries
    try testing.expectEqual(WebGpuBackend.max_dirty_entries, @typeInfo(@TypeOf(@as(WebGpuBackend, undefined).dirty_bufs)).array.len);
}

test "WebGPU comptime struct layout verification" {
    comptime {
        // CachedBuf must contain buffer handle, size, and generation counter
        const cached_fields = @typeInfo(CachedBuf).@"struct".fields;
        var has_buffer = false;
        var has_size = false;
        var has_generation = false;
        for (cached_fields) |f| {
            if (std.mem.eql(u8, f.name, "buffer")) has_buffer = true;
            if (std.mem.eql(u8, f.name, "size")) has_size = true;
            if (std.mem.eql(u8, f.name, "generation")) has_generation = true;
        }
        if (!has_buffer) @compileError("CachedBuf missing 'buffer' field");
        if (!has_size) @compileError("CachedBuf missing 'size' field");
        if (!has_generation) @compileError("CachedBuf missing 'generation' field");

        // PipelineInfo must contain pipeline and bind_group_layout
        const pipe_fields = @typeInfo(PipelineInfo).@"struct".fields;
        var has_pipeline = false;
        var has_bgl = false;
        for (pipe_fields) |f| {
            if (std.mem.eql(u8, f.name, "pipeline")) has_pipeline = true;
            if (std.mem.eql(u8, f.name, "bind_group_layout")) has_bgl = true;
        }
        if (!has_pipeline) @compileError("PipelineInfo missing 'pipeline' field");
        if (!has_bgl) @compileError("PipelineInfo missing 'bind_group_layout' field");

        // PoolEntry must have buffer, size, and in_use fields
        const pool_fields = @typeInfo(PoolEntry).@"struct".fields;
        var has_in_use = false;
        for (pool_fields) |f| {
            if (std.mem.eql(u8, f.name, "in_use")) has_in_use = true;
        }
        if (!has_in_use) @compileError("PoolEntry missing 'in_use' field");

        // Verify act_pool array element type is PoolEntry
        const pool_info = @typeInfo(@TypeOf(@as(WebGpuBackend, undefined).act_pool));
        if (pool_info.array.child != PoolEntry) @compileError("act_pool element type is not PoolEntry");
    }
}

// ── Per-function comptime signature tests ────────────────────────

test "WebGpuBackend.init" {
    comptime {
        _ = &WebGpuBackend.init;
    }
}
test "WebGpuBackend.deinit" {
    comptime {
        _ = &WebGpuBackend.deinit;
    }
}
test "WebGpuBackend.silu" {
    comptime {
        _ = &WebGpuBackend.silu;
    }
}
test "WebGpuBackend.gelu" {
    comptime {
        _ = &WebGpuBackend.gelu;
    }
}
test "WebGpuBackend.add" {
    comptime {
        _ = &WebGpuBackend.add;
    }
}
test "WebGpuBackend.mul" {
    comptime {
        _ = &WebGpuBackend.mul;
    }
}
test "WebGpuBackend.siluMul" {
    comptime {
        _ = &WebGpuBackend.siluMul;
    }
}
test "WebGpuBackend.geluMul" {
    comptime {
        _ = &WebGpuBackend.geluMul;
    }
}
test "WebGpuBackend.rmsNorm" {
    comptime {
        _ = &WebGpuBackend.rmsNorm;
    }
}
test "WebGpuBackend.softmax" {
    comptime {
        _ = &WebGpuBackend.softmax;
    }
}
test "WebGpuBackend.rope" {
    comptime {
        _ = &WebGpuBackend.rope;
    }
}
test "WebGpuBackend.embLookup" {
    comptime {
        _ = &WebGpuBackend.embLookup;
    }
}
test "WebGpuBackend.gemv" {
    comptime {
        _ = &WebGpuBackend.gemv;
    }
}
test "WebGpuBackend.gemm" {
    comptime {
        _ = &WebGpuBackend.gemm;
    }
}
test "WebGpuBackend.l2Norm" {
    comptime {
        _ = &WebGpuBackend.l2Norm;
    }
}
test "WebGpuBackend.addRmsNorm" {
    comptime {
        _ = &WebGpuBackend.addRmsNorm;
    }
}
test "WebGpuBackend.addScaled" {
    comptime {
        _ = &WebGpuBackend.addScaled;
    }
}
test "WebGpuBackend.sigmoidMul" {
    comptime {
        _ = &WebGpuBackend.sigmoidMul;
    }
}
test "WebGpuBackend.deinterleave" {
    comptime {
        _ = &WebGpuBackend.deinterleave;
    }
}
test "WebGpuBackend.splitQGate" {
    comptime {
        _ = &WebGpuBackend.splitQGate;
    }
}
test "WebGpuBackend.rmsNormMulti" {
    comptime {
        _ = &WebGpuBackend.rmsNormMulti;
    }
}
test "WebGpuBackend.rmsNormBatched" {
    comptime {
        _ = &WebGpuBackend.rmsNormBatched;
    }
}
test "WebGpuBackend.ropeBatched" {
    comptime {
        _ = &WebGpuBackend.ropeBatched;
    }
}
test "WebGpuBackend.sdpa" {
    comptime {
        _ = &WebGpuBackend.sdpa;
    }
}
test "WebGpuBackend.sdpaWithStats" {
    comptime {
        _ = &WebGpuBackend.sdpaWithStats;
    }
}
test "WebGpuBackend.sdpaPaged" {
    comptime {
        _ = &WebGpuBackend.sdpaPaged;
    }
}
test "WebGpuBackend.sdpaTree" {
    comptime {
        _ = &WebGpuBackend.sdpaTree;
    }
}
test "WebGpuBackend.sdpaPrefill" {
    comptime {
        _ = &WebGpuBackend.sdpaPrefill;
    }
}
test "WebGpuBackend.gemvT" {
    comptime {
        _ = &WebGpuBackend.gemvT;
    }
}
test "WebGpuBackend.gemvNvfp4St" {
    comptime {
        _ = &WebGpuBackend.gemvNvfp4St;
    }
}
test "WebGpuBackend.gemvMlxQ" {
    comptime {
        _ = &WebGpuBackend.gemvMlxQ;
    }
}
test "WebGpuBackend.gemvMxfp4St" {
    comptime {
        _ = &WebGpuBackend.gemvMxfp4St;
    }
}
test "WebGpuBackend.gemvGptq" {
    comptime {
        _ = &WebGpuBackend.gemvGptq;
    }
}
test "WebGpuBackend.gemvAwq" {
    comptime {
        _ = &WebGpuBackend.gemvAwq;
    }
}
test "WebGpuBackend.gemvMulti" {
    comptime {
        _ = &WebGpuBackend.gemvMulti;
    }
}
test "WebGpuBackend.causalConv1dSilu" {
    comptime {
        _ = &WebGpuBackend.causalConv1dSilu;
    }
}
test "WebGpuBackend.deltaNet" {
    comptime {
        _ = &WebGpuBackend.deltaNet;
    }
}
test "WebGpuBackend.sync" {
    comptime {
        _ = &WebGpuBackend.sync;
    }
}
test "WebGpuBackend.beginBatch" {
    comptime {
        _ = &WebGpuBackend.beginBatch;
    }
}
test "WebGpuBackend.endBatch" {
    comptime {
        _ = &WebGpuBackend.endBatch;
    }
}
test "WebGpuBackend.allocKvSlice" {
    comptime {
        _ = &WebGpuBackend.allocKvSlice;
    }
}
test "WebGpuBackend.freeKvSlice" {
    comptime {
        _ = &WebGpuBackend.freeKvSlice;
    }
}
test "WebGpuBackend.backendInfo" {
    comptime {
        _ = &WebGpuBackend.backendInfo;
    }
}

test "fuzz: all webgpu functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            _ = smith;
            comptime {
                _ = &WebGpuBackend.init;
                _ = &WebGpuBackend.deinit;
                _ = &WebGpuBackend.silu;
                _ = &WebGpuBackend.gelu;
                _ = &WebGpuBackend.add;
                _ = &WebGpuBackend.mul;
                _ = &WebGpuBackend.siluMul;
                _ = &WebGpuBackend.geluMul;
                _ = &WebGpuBackend.rmsNorm;
                _ = &WebGpuBackend.softmax;
                _ = &WebGpuBackend.rope;
                _ = &WebGpuBackend.embLookup;
                _ = &WebGpuBackend.gemv;
                _ = &WebGpuBackend.gemm;
                _ = &WebGpuBackend.l2Norm;
                _ = &WebGpuBackend.addRmsNorm;
                _ = &WebGpuBackend.addScaled;
                _ = &WebGpuBackend.sigmoidMul;
                _ = &WebGpuBackend.deinterleave;
                _ = &WebGpuBackend.splitQGate;
                _ = &WebGpuBackend.rmsNormMulti;
                _ = &WebGpuBackend.rmsNormBatched;
                _ = &WebGpuBackend.ropeBatched;
                _ = &WebGpuBackend.sdpa;
                _ = &WebGpuBackend.sdpaWithStats;
                _ = &WebGpuBackend.sdpaPaged;
                _ = &WebGpuBackend.sdpaTree;
                _ = &WebGpuBackend.sdpaPrefill;
                _ = &WebGpuBackend.gemvT;
                _ = &WebGpuBackend.gemvNvfp4St;
                _ = &WebGpuBackend.gemvMlxQ;
                _ = &WebGpuBackend.gemvMxfp4St;
                _ = &WebGpuBackend.gemvGptq;
                _ = &WebGpuBackend.gemvAwq;
                _ = &WebGpuBackend.gemvHqq;
                _ = &WebGpuBackend.gemvMulti;
                _ = &WebGpuBackend.causalConv1dSilu;
                _ = &WebGpuBackend.deltaNet;
                _ = &WebGpuBackend.sync;
                _ = &WebGpuBackend.beginBatch;
                _ = &WebGpuBackend.endBatch;
                _ = &WebGpuBackend.allocKvSlice;
                _ = &WebGpuBackend.freeKvSlice;
                _ = &WebGpuBackend.backendInfo;
            }
        }
    }.f, .{});
}
