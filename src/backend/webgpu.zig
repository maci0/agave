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
const wgsl_gemv_q8_0 = @embedFile("kernels/webgpu/gemv_q8_0.wgsl");
const wgsl_sigmoid_mul = @embedFile("kernels/webgpu/sigmoid_mul.wgsl");
const wgsl_l2_norm = @embedFile("kernels/webgpu/l2_norm.wgsl");
const wgsl_rms_norm_multi = @embedFile("kernels/webgpu/rms_norm_multi.wgsl");
const wgsl_deinterleave = @embedFile("kernels/webgpu/deinterleave.wgsl");
const wgsl_split_qgate = @embedFile("kernels/webgpu/split_qgate.wgsl");
const wgsl_add_rms_norm = @embedFile("kernels/webgpu/add_rms_norm.wgsl");

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

const WGPUBufferUsage = u32;
const wgpu_buffer_usage_storage = 0x0080;
const wgpu_buffer_usage_copy_src = 0x0004;
const wgpu_buffer_usage_copy_dst = 0x0008;
const wgpu_buffer_usage_map_read = 0x0001;
const wgpu_buffer_usage_uniform = 0x0040;

const WGPUBufferBindingType = u32;
const wgpu_buffer_binding_storage = 7;
const wgpu_buffer_binding_read_only_storage = 6;
const wgpu_buffer_binding_uniform = 1;

const WGPUMapMode = u32;
const wgpu_map_mode_read = 1;

const WGPURequestAdapterStatus = u32;
const WGPURequestDeviceStatus = u32;
const WGPUBufferMapAsyncStatus = u32;

const WGPUInstanceDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
};

const WGPURequestAdapterOptions = extern struct {
    next_in_chain: ?*anyopaque = null,
    compatible_surface: ?*anyopaque = null,
    power_preference: u32 = 2, // high-performance
    backend_type: u32 = 0, // undefined = auto
    force_fallback_adapter: u32 = 0,
};

const WGPUDeviceDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: ?[*:0]const u8 = null,
    required_feature_count: usize = 0,
    required_features: ?*anyopaque = null,
    required_limits: ?*anyopaque = null,
    default_queue: extern struct {
        next_in_chain: ?*anyopaque = null,
        label: ?[*:0]const u8 = null,
    } = .{},
    device_lost_callback: ?*anyopaque = null,
    device_lost_userdata: ?*anyopaque = null,
};

const WGPUBufferDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: ?[*:0]const u8 = null,
    usage: WGPUBufferUsage = 0,
    size: u64 = 0,
    mapped_at_creation: u32 = 0,
};

const WGPUShaderModuleDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: ?[*:0]const u8 = null,
};

const WGPUShaderModuleWGSLDescriptor = extern struct {
    chain: extern struct {
        next: ?*anyopaque = null,
        s_type: u32 = 6, // SType_ShaderModuleWGSLDescriptor
    } = .{},
    code: [*:0]const u8,
};

const WGPUComputePipelineDescriptor = extern struct {
    next_in_chain: ?*anyopaque = null,
    label: ?[*:0]const u8 = null,
    layout: WGPUPipelineLayout = null,
    compute: extern struct {
        next_in_chain: ?*anyopaque = null,
        module: WGPUShaderModule = null,
        entry_point: [*:0]const u8 = "main",
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
    label: ?[*:0]const u8 = null,
    layout: WGPUBindGroupLayout = null,
    entry_count: usize = 0,
    entries: ?[*]const WGPUBindGroupEntry = null,
};

// ── Tuning constants ────────────────────────────────────────────────

const workgroup_size: u32 = 256;
const act_pool_capacity: u32 = 32;
const max_bindings: u32 = 8;

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
};

// ── WebGPU Backend ──────────────────────────────────────────────────

pub const WebGpuBackend = struct {
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
    pipe_gemv_q8_0: PipelineInfo = .{},
    pipe_sigmoid_mul: PipelineInfo = .{},
    pipe_l2_norm: PipelineInfo = .{},
    pipe_rms_norm_multi: PipelineInfo = .{},
    pipe_deinterleave: PipelineInfo = .{},
    pipe_split_qgate: PipelineInfo = .{},
    pipe_add_rms_norm: PipelineInfo = .{},

    // Buffer management
    buf_cache: std.AutoHashMap(usize, CachedBuf) = undefined,
    act_pool: [act_pool_capacity]PoolEntry = [_]PoolEntry{.{}} ** act_pool_capacity,
    act_pool_count: u32 = 0,

    // Staging buffer for readbacks
    staging_buf: WGPUBuffer = null,
    staging_size: usize = 0,

    // WebGPU C function pointers
    fn_create_instance: *const fn (?*const WGPUInstanceDescriptor) callconv(.c) WGPUInstance = undefined,
    fn_instance_request_adapter: *const fn (WGPUInstance, ?*const WGPURequestAdapterOptions, *const fn (WGPURequestAdapterStatus, WGPUAdapter, ?[*:0]const u8, ?*anyopaque) callconv(.c) void, ?*anyopaque) callconv(.c) void = undefined,
    fn_adapter_request_device: *const fn (WGPUAdapter, ?*const WGPUDeviceDescriptor, *const fn (WGPURequestDeviceStatus, WGPUDevice, ?[*:0]const u8, ?*anyopaque) callconv(.c) void, ?*anyopaque) callconv(.c) void = undefined,
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
    fn_buffer_map_async: *const fn (WGPUBuffer, WGPUMapMode, usize, usize, *const fn (WGPUBufferMapAsyncStatus, ?*anyopaque) callconv(.c) void, ?*anyopaque) callconv(.c) void = undefined,
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

    pub fn init(allocator: std.mem.Allocator) !*WebGpuBackend {
        var self = try allocator.create(WebGpuBackend);
        self.* = .{ .allocator = allocator, .lib = undefined };
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);

        const lib_name = switch (@import("builtin").os.tag) {
            .macos => "libwgpu_native.dylib",
            .linux => "libwgpu_native.so",
            .windows => "wgpu_native.dll",
            else => return error.WebGpuNotAvailable,
        };
        self.lib = std.DynLib.open(lib_name) catch return error.WebGpuNotAvailable;

        self.loadFunctions() catch return error.WebGpuNotAvailable;

        const desc = WGPUInstanceDescriptor{};
        self.instance = self.fn_create_instance(&desc);
        if (self.instance == null) return error.WebGpuNotAvailable;

        try self.requestAdapter();
        try self.requestDevice();
        self.queue = self.fn_device_get_queue(self.device);

        try self.createPipelines();

        return self;
    }

    fn loadFunctions(self: *WebGpuBackend) !void {
        inline for (@typeInfo(WebGpuBackend).@"struct".fields) |field| {
            if (comptime std.mem.startsWith(u8, field.name, "fn_")) {
                const c_name = comptime cFunctionName(field.name);
                const ptr = self.lib.lookup(@TypeOf(@field(self.*, field.name)), c_name) orelse return error.WebGpuNotAvailable;
                @field(self.*, field.name) = ptr;
            }
        }
    }

    fn cFunctionName(comptime field_name: []const u8) [:0]const u8 {
        comptime {
            const name = field_name["fn_".len..];
            var result: [128]u8 = undefined;
            var i: usize = 0;
            // fn_create_instance → wgpuCreateInstance
            result[i] = 'w';
            i += 1;
            result[i] = 'g';
            i += 1;
            result[i] = 'p';
            i += 1;
            result[i] = 'u';
            i += 1;
            var capitalize_next = true;
            for (name) |c| {
                if (c == '_') {
                    capitalize_next = true;
                    continue;
                }
                if (capitalize_next) {
                    result[i] = std.ascii.toUpper(c);
                    capitalize_next = false;
                } else {
                    result[i] = c;
                }
                i += 1;
            }
            result[i] = 0;
            return result[0..i :0];
        }
    }

    fn requestAdapter(self: *WebGpuBackend) !void {
        const AdapterCtx = struct {
            adapter: WGPUAdapter = null,
            ready: bool = false,
        };
        var ctx = AdapterCtx{};
        const opts = WGPURequestAdapterOptions{};
        self.fn_instance_request_adapter(self.instance, &opts, struct {
            fn cb(_: WGPURequestAdapterStatus, adapter: WGPUAdapter, _: ?[*:0]const u8, userdata: ?*anyopaque) callconv(.c) void {
                const c: *AdapterCtx = @ptrCast(@alignCast(userdata));
                c.adapter = adapter;
                c.ready = true;
            }
        }.cb, @ptrCast(&ctx));
        self.fn_device_poll(self.device orelse self.instance, 1, null);
        if (!ctx.ready or ctx.adapter == null) return error.WebGpuNotAvailable;
        self.adapter = ctx.adapter;
    }

    fn requestDevice(self: *WebGpuBackend) !void {
        const DeviceCtx = struct {
            device: WGPUDevice = null,
            ready: bool = false,
        };
        var ctx = DeviceCtx{};
        const desc = WGPUDeviceDescriptor{};
        self.fn_adapter_request_device(self.adapter, &desc, struct {
            fn cb(_: WGPURequestDeviceStatus, device: WGPUDevice, _: ?[*:0]const u8, userdata: ?*anyopaque) callconv(.c) void {
                const c: *DeviceCtx = @ptrCast(@alignCast(userdata));
                c.device = device;
                c.ready = true;
            }
        }.cb, @ptrCast(&ctx));
        // Poll instance to drive adapter request callback
        if (self.device == null) {
            // No device yet — poll instance (wgpu-native processes callbacks during poll)
            _ = self.fn_device_poll(self.instance, 1, null);
        }
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
        self.pipe_gemv_q8_0 = try self.createPipeline(wgsl_gemv_q8_0);
        self.pipe_sigmoid_mul = try self.createPipeline(wgsl_sigmoid_mul);
        self.pipe_l2_norm = try self.createPipeline(wgsl_l2_norm);
        self.pipe_rms_norm_multi = try self.createPipeline(wgsl_rms_norm_multi);
        self.pipe_deinterleave = try self.createPipeline(wgsl_deinterleave);
        self.pipe_split_qgate = try self.createPipeline(wgsl_split_qgate);
        self.pipe_add_rms_norm = try self.createPipeline(wgsl_add_rms_norm);
    }

    fn createPipeline(self: *WebGpuBackend, wgsl_source: [:0]const u8) !PipelineInfo {
        var wgsl_desc = WGPUShaderModuleWGSLDescriptor{ .code = wgsl_source };
        var shader_desc = WGPUShaderModuleDescriptor{ .next_in_chain = @ptrCast(&wgsl_desc) };
        const shader = self.fn_device_create_shader_module(self.device, &shader_desc);
        if (shader == null) return error.WebGpuShaderCompilationFailed;
        defer self.fn_shader_module_release(shader);

        var pipeline_desc = WGPUComputePipelineDescriptor{};
        pipeline_desc.compute.module = shader;
        pipeline_desc.compute.entry_point = "main";
        const pipeline = self.fn_device_create_compute_pipeline(self.device, &pipeline_desc);
        if (pipeline == null) return error.WebGpuPipelineCreationFailed;

        const bgl = self.fn_compute_pipeline_get_bind_group_layout(pipeline, 0);
        return PipelineInfo{ .pipeline = pipeline, .bind_group_layout = bgl };
    }

    pub fn deinit(self: *WebGpuBackend) void {
        // Release pool buffers
        for (&self.act_pool) |*entry| {
            if (entry.buffer != null) self.fn_buffer_destroy(entry.buffer);
        }
        // Release cached buffers
        var it = self.buf_cache.valueIterator();
        while (it.next()) |v| self.fn_buffer_destroy(v.buffer);
        self.buf_cache.deinit();

        if (self.staging_buf != null) self.fn_buffer_destroy(self.staging_buf);
        if (self.device != null) self.fn_device_release(self.device);
        if (self.adapter != null) self.fn_adapter_release(self.adapter);
        if (self.instance != null) self.fn_instance_release(self.instance);
        self.lib.close();
        self.allocator.destroy(self);
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

    fn getOrUpload(self: *WebGpuBackend, ptr: *const anyopaque, size: usize) WGPUBuffer {
        const key = @intFromPtr(ptr);
        if (self.buf_cache.get(key)) |cached| return cached.buffer;
        const buf = self.createBuffer(size, wgpu_buffer_usage_storage | wgpu_buffer_usage_copy_src);
        self.uploadToBuffer(buf, ptr, size);
        self.buf_cache.put(key, .{ .buffer = buf, .size = size }) catch {};
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
        return .{ .buf = buf, .idx = 0 };
    }

    fn releasePooledBuf(self: *WebGpuBackend, idx: u32) void {
        if (idx < act_pool_capacity) {
            self.act_pool[idx].in_use = false;
        }
    }

    fn downloadF32(self: *WebGpuBackend, src: WGPUBuffer, dst: [*]f32, count: usize) void {
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
        self.fn_buffer_map_async(self.staging_buf, wgpu_map_mode_read, 0, size, struct {
            fn cb(_: WGPUBufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.c) void {
                const m: *bool = @ptrCast(@alignCast(userdata));
                m.* = true;
            }
        }.cb, @ptrCast(&mapped));
        while (!mapped) {
            _ = self.fn_device_poll(self.device, 1, null);
        }
        const mapped_ptr = self.fn_buffer_get_mapped_range(self.staging_buf, 0, size);
        if (mapped_ptr) |p| {
            const src_slice: [*]const f32 = @ptrCast(@alignCast(p));
            @memcpy(dst[0..count], src_slice[0..count]);
        }
        self.fn_buffer_unmap(self.staging_buf);
    }

    // ── Dispatch Helpers ────────────────────────────────────────

    /// Align size up to 16 bytes (WebGPU minimum uniform buffer alignment).
    fn alignUniform(size: usize) usize {
        return (size + 15) & ~@as(usize, 15);
    }

    /// Create a uniform buffer, upload params data, return GPU buffer handle.
    fn createUniformBuf(self: *WebGpuBackend, comptime T: type, params: *const T) WGPUBuffer {
        const aligned_size = alignUniform(@sizeOf(T));
        const buf = self.createBuffer(aligned_size, wgpu_buffer_usage_uniform | wgpu_buffer_usage_copy_dst);
        self.fn_queue_write_buffer(self.queue, buf, 0, @ptrCast(params), @sizeOf(T));
        return buf;
    }

    /// Central dispatch: create bind group from entries, encode + submit compute pass.
    fn dispatchCompute(self: *WebGpuBackend, pipe: PipelineInfo, entries: []const WGPUBindGroupEntry, workgroups_x: u32) void {
        var bg_desc = WGPUBindGroupDescriptor{
            .layout = pipe.bind_group_layout,
            .entry_count = entries.len,
            .entries = entries.ptr,
        };
        const bind_group = self.fn_device_create_bind_group(self.device, &bg_desc);
        defer self.fn_bind_group_release(bind_group);

        const encoder = self.fn_device_create_command_encoder(self.device, null);
        const pass = self.fn_command_encoder_begin_compute_pass(encoder, null);
        self.fn_compute_pass_set_pipeline(pass, pipe.pipeline);
        self.fn_compute_pass_set_bind_group(pass, 0, bind_group, 0, null);
        self.fn_compute_pass_dispatch(pass, workgroups_x, 1, 1);
        self.fn_compute_pass_end(pass);
        const cmd = self.fn_command_encoder_finish(encoder, null);
        self.fn_queue_submit(self.queue, 1, &cmd);
        self.fn_command_buffer_release(cmd);
        self.fn_command_encoder_release(encoder);
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

    pub fn silu(self: *WebGpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const out = self.getPooledBuf(size);
        defer self.releasePooledBuf(out.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, out.buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_silu, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out.buf, output, n);
    }

    pub fn gelu(self: *WebGpuBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const out = self.getPooledBuf(size);
        defer self.releasePooledBuf(out.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, out.buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gelu, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out.buf, output, n);
    }

    pub fn add(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(out_pool.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_pool.buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_add, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out_pool.buf, out, n);
    }

    pub fn mul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(out_pool.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_pool.buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out_pool.buf, out, n);
    }

    pub fn siluMul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(out_pool.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_pool.buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_silu_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out_pool.buf, out, n);
    }

    pub fn geluMul(self: *WebGpuBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const buf_a = self.getOrUpload(@ptrCast(a), size);
        const buf_b = self.getOrUpload(@ptrCast(b), size);
        const out_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(out_pool.idx);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, buf_a, size),
            storageEntry(1, buf_b, size),
            storageEntry(2, out_pool.buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_gelu_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out_pool.buf, out, n);
    }

    pub fn rmsNorm(self: *WebGpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), size);
        const out = self.getPooledBuf(size);
        defer self.releasePooledBuf(out.idx);

        const Params = extern struct { n: u32, _pad0: u32 = 0, eps: f32, _pad1: u32 = 0 };
        var params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, size),
            storageEntry(1, w_buf, size),
            storageEntry(2, out.buf, size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rms_norm, &entries, 1);
        self.downloadF32(out.buf, output, n);
    }

    pub fn softmax(self: *WebGpuBackend, data: [*]f32, n: usize) void {
        const size = n * @sizeOf(f32);
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_softmax, &entries, 1);
        self.downloadF32(data_pool.buf, data, n);
    }

    pub fn rope(self: *WebGpuBackend, data: [*]f32, pos: u32, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const total_elems = n_heads * head_dim;
        const size = total_elems * @sizeOf(f32);
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);

        const Params = extern struct {
            pos: u32,
            n_heads: u32,
            head_dim: u32,
            rope_dim: u32,
            theta: f32,
            _pad: [12]u8 = .{0} ** 12,
        };
        var params = Params{
            .pos = pos,
            .n_heads = @intCast(n_heads),
            .head_dim = @intCast(head_dim),
            .rope_dim = @intCast(rope_dim),
            .theta = theta,
        };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const half_rope = n_heads * rope_dim / 2;
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rope, &entries, @intCast((half_rope + workgroup_size - 1) / workgroup_size));
        self.downloadF32(data_pool.buf, data, total_elems);
    }

    pub fn embLookup(self: *WebGpuBackend, table: TensorData, output: [*]f32, n_embd: usize, token_id: u32) void {
        if (table.dtype != .f32) @panic("WebGPU embLookup: only f32 embedding tables supported");
        const table_size = table.n_elements * @sizeOf(f32);
        const table_buf = self.getOrUpload(table.data_ptr, table_size);
        const out_size = n_embd * @sizeOf(f32);
        const out = self.getPooledBuf(out_size);
        defer self.releasePooledBuf(out.idx);

        const Params = extern struct { token_id: u32, n_embd: u32, _pad: [8]u8 = .{0} ** 8 };
        var params = Params{ .token_id = token_id, .n_embd = @intCast(n_embd) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, table_buf, table_size),
            storageEntry(1, out.buf, out_size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_embedding, &entries, @intCast((n_embd + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out.buf, output, n_embd);
    }

    pub fn gemv(self: *WebGpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const x_size = k * @sizeOf(f32);
        const y_size = n * @sizeOf(f32);
        const x_buf = self.getOrUpload(@ptrCast(x), x_size);
        const out = self.getPooledBuf(y_size);
        defer self.releasePooledBuf(out.idx);

        const Params = extern struct { n: u32, k: u32, _pad: [8]u8 = .{0} ** 8 };
        var params = Params{ .n = @intCast(n), .k = @intCast(k) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        switch (w.dtype) {
            .f32 => {
                const w_size = n * k * @sizeOf(f32);
                const w_buf = self.getOrUpload(w.data_ptr, w_size);
                const entries = [_]WGPUBindGroupEntry{
                    storageEntry(0, x_buf, x_size),
                    storageEntry(1, w_buf, w_size),
                    storageEntry(2, out.buf, y_size),
                    uniformEntry(3, params_buf, Params),
                };
                self.dispatchCompute(self.pipe_gemv_f32, &entries, @intCast(n));
            },
            .q8_0 => {
                const block_size: usize = 34; // 32 quants (i8) + 1 scale (f16) = 34 bytes per block
                const blocks_per_row = (k + 31) / 32;
                const w_size = n * blocks_per_row * block_size;
                const w_buf = self.getOrUpload(w.data_ptr, w_size);
                const entries = [_]WGPUBindGroupEntry{
                    storageEntry(0, x_buf, x_size),
                    storageEntry(1, w_buf, w_size),
                    storageEntry(2, out.buf, y_size),
                    uniformEntry(3, params_buf, Params),
                };
                self.dispatchCompute(self.pipe_gemv_q8_0, &entries, @intCast(n));
            },
            else => @panic("WebGPU gemv: unsupported weight dtype"),
        }
        self.downloadF32(out.buf, y, n);
    }

    pub fn gemm(self: *WebGpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n_tok: usize, n_out: usize, n_in: usize) void {
        _ = self;
        _ = x;
        _ = w;
        _ = y;
        _ = n_tok;
        _ = n_out;
        _ = n_in;
        @panic("WebGPU gemm: not yet implemented");
    }

    pub fn l2Norm(self: *WebGpuBackend, data: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);

        const Params = extern struct { n: u32, _pad0: u32 = 0, eps: f32, _pad1: u32 = 0 };
        var params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            uniformEntry(1, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_l2_norm, &entries, 1);
        self.downloadF32(data_pool.buf, data, n);
    }

    pub fn addRmsNorm(self: *WebGpuBackend, data: [*]f32, residual: [*]const f32, weight: [*]const f32, out: [*]f32, n: usize, eps: f32) void {
        const size = n * @sizeOf(f32);
        // data is read_write: add residual in-place, then normalize into out
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);
        const res_buf = self.getOrUpload(@ptrCast(residual), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), size);
        const out_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(out_pool.idx);

        const Params = extern struct { n: u32, _pad0: u32 = 0, eps: f32, _pad1: u32 = 0 };
        var params = Params{ .n = @intCast(n), .eps = eps };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            storageEntry(1, res_buf, size),
            storageEntry(2, w_buf, size),
            storageEntry(3, out_pool.buf, size),
            uniformEntry(4, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_add_rms_norm, &entries, 1);
        // Download both data (modified in-place with residual add) and normalized output
        self.downloadF32(data_pool.buf, data, n);
        self.downloadF32(out_pool.buf, out, n);
    }

    pub fn addScaled(self: *WebGpuBackend, a: [*]f32, b: [*]const f32, out: [*]f32, n: usize) void {
        _ = self;
        _ = a;
        _ = b;
        _ = out;
        _ = n;
        @panic("WebGPU addScaled: not yet implemented");
    }

    pub fn sigmoidMul(self: *WebGpuBackend, data: [*]f32, gate: [*]const f32, n: usize) void {
        const size = n * @sizeOf(f32);
        // data is read_write (result written back in place)
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);
        const gate_buf = self.getOrUpload(@ptrCast(gate), size);

        const Params = extern struct { n: u32, _pad: [12]u8 = .{0} ** 12 };
        var params = Params{ .n = @intCast(n) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            storageEntry(1, gate_buf, size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_sigmoid_mul, &entries, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(data_pool.buf, data, n);
    }

    pub fn deinterleave(self: *WebGpuBackend, input: [*]const f32, out_a: [*]f32, out_b: [*]f32, stride: usize, n_pairs: usize) void {
        const in_size = n_pairs * stride * 2 * @sizeOf(f32);
        const out_size = n_pairs * stride * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(input), in_size);
        const out_a_pool = self.getPooledBuf(out_size);
        defer self.releasePooledBuf(out_a_pool.idx);
        const out_b_pool = self.getPooledBuf(out_size);
        defer self.releasePooledBuf(out_b_pool.idx);

        const Params = extern struct { stride: u32, n_pairs: u32, _pad: [8]u8 = .{0} ** 8 };
        var params = Params{ .stride = @intCast(stride), .n_pairs = @intCast(n_pairs) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const total_elems = n_pairs * stride;
        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, in_size),
            storageEntry(1, out_a_pool.buf, out_size),
            storageEntry(2, out_b_pool.buf, out_size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_deinterleave, &entries, @intCast((total_elems + workgroup_size - 1) / workgroup_size));
        self.downloadF32(out_a_pool.buf, out_a, total_elems);
        self.downloadF32(out_b_pool.buf, out_b, total_elems);
    }

    pub fn splitQGate(self: *WebGpuBackend, qg: [*]const f32, q_out: [*]f32, g_out: [*]f32, head_dim: usize, n_heads: usize) void {
        const total_elems = n_heads * head_dim;
        const in_size = total_elems * 2 * @sizeOf(f32);
        const out_size = total_elems * @sizeOf(f32);
        const in_buf = self.getOrUpload(@ptrCast(qg), in_size);
        const q_pool = self.getPooledBuf(out_size);
        defer self.releasePooledBuf(q_pool.idx);
        const g_pool = self.getPooledBuf(out_size);
        defer self.releasePooledBuf(g_pool.idx);

        const Params = extern struct { hd: u32, nh: u32, _pad: [8]u8 = .{0} ** 8 };
        var params = Params{ .hd = @intCast(head_dim), .nh = @intCast(n_heads) };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, in_buf, in_size),
            storageEntry(1, q_pool.buf, out_size),
            storageEntry(2, g_pool.buf, out_size),
            uniformEntry(3, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_split_qgate, &entries, @intCast((total_elems + workgroup_size - 1) / workgroup_size));
        self.downloadF32(q_pool.buf, q_out, total_elems);
        self.downloadF32(g_pool.buf, g_out, total_elems);
    }

    pub fn rmsNormMulti(self: *WebGpuBackend, data: [*]f32, weight: [*]const f32, n_heads: usize, head_dim: usize, eps: f32) void {
        const total_elems = n_heads * head_dim;
        const size = total_elems * @sizeOf(f32);
        const w_size = head_dim * @sizeOf(f32);
        // data is read_write (normalized in-place)
        const data_pool = self.getPooledBuf(size);
        defer self.releasePooledBuf(data_pool.idx);
        self.uploadToBuffer(data_pool.buf, @ptrCast(data), size);
        const w_buf = self.getOrUpload(@ptrCast(weight), w_size);

        const Params = extern struct { n_heads: u32, head_dim: u32, eps: f32, _pad: u32 = 0 };
        var params = Params{ .n_heads = @intCast(n_heads), .head_dim = @intCast(head_dim), .eps = eps };
        const params_buf = self.createUniformBuf(Params, &params);
        defer self.fn_buffer_destroy(params_buf);

        const entries = [_]WGPUBindGroupEntry{
            storageEntry(0, data_pool.buf, size),
            storageEntry(1, w_buf, w_size),
            uniformEntry(2, params_buf, Params),
        };
        self.dispatchCompute(self.pipe_rms_norm_multi, &entries, @intCast(n_heads));
        self.downloadF32(data_pool.buf, data, total_elems);
    }

    pub fn rmsNormBatched(self: *WebGpuBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n_tok: usize, dim: usize, eps: f32) void {
        _ = self;
        _ = input;
        _ = weight;
        _ = output;
        _ = n_tok;
        _ = dim;
        _ = eps;
        @panic("WebGPU rmsNormBatched: not yet implemented");
    }

    pub fn ropeBatched(self: *WebGpuBackend, data: [*]f32, positions: [*]const u32, n_tok: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        _ = self;
        _ = data;
        _ = positions;
        _ = n_tok;
        _ = n_heads;
        _ = head_dim;
        _ = rope_dim;
        _ = theta;
        @panic("WebGPU ropeBatched: not yet implemented");
    }

    pub fn sdpa(self: *WebGpuBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: []const f32, v_new: []const f32, output: [*]f32, scores: [*]f32, nh: usize, nkv: usize, hd: usize, pos: u32, scale: f32, be_unused: anytype, window: anytype, n_pad: usize, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        _ = self;
        _ = q;
        _ = keys;
        _ = values;
        _ = k_new;
        _ = v_new;
        _ = output;
        _ = scores;
        _ = nh;
        _ = nkv;
        _ = hd;
        _ = pos;
        _ = scale;
        _ = be_unused;
        _ = window;
        _ = n_pad;
        _ = kv_type_k;
        _ = kv_type_v;
        @panic("WebGPU sdpa: not yet implemented");
    }

    pub fn sdpaWithStats(self: *WebGpuBackend, q: [*]const f32, keys: [*]const u8, values: [*]const u8, k_new: []const f32, v_new: []const f32, output: [*]f32, max_out: [*]f32, sum_out: [*]f32, nh: usize, nkv: usize, hd: usize, pos: u32, scale: f32, window: anytype, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        _ = self;
        _ = q;
        _ = keys;
        _ = values;
        _ = k_new;
        _ = v_new;
        _ = output;
        _ = max_out;
        _ = sum_out;
        _ = nh;
        _ = nkv;
        _ = hd;
        _ = pos;
        _ = scale;
        _ = window;
        _ = kv_type_k;
        _ = kv_type_v;
        @panic("WebGPU sdpaWithStats: not yet implemented");
    }

    pub fn sdpaTree(_: *WebGpuBackend, q_all: [*]const f32, prefix_keys: [*]const u8, prefix_values: [*]const u8, tree_keys: [*]const f32, tree_values: [*]const f32, output: [*]f32, ancestor_masks: [*]const [8]u64, nh: usize, nkv: usize, hd: usize, prefix_len: usize, n_nodes: u32, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        @import("kernels/cpu/sdpa_tree.zig").sdpaTree(q_all, prefix_keys, prefix_values, tree_keys, tree_values, output, ancestor_masks, nh, nkv, hd, prefix_len, n_nodes, scale, kv_type_k, kv_type_v);
    }

    pub fn sdpaPrefill(self: *WebGpuBackend, q: [*]const f32, k: [*]const f32, v: [*]const f32, kv_keys: []u8, kv_values: []u8, output: [*]f32, nh: usize, nkv: usize, hd: usize, prev_len: usize, n_tok: usize, scale: f32, kv_type_k: KvQuantType, kv_type_v: KvQuantType) void {
        _ = self;
        _ = q;
        _ = k;
        _ = v;
        _ = kv_keys;
        _ = kv_values;
        _ = output;
        _ = nh;
        _ = nkv;
        _ = hd;
        _ = prev_len;
        _ = n_tok;
        _ = scale;
        _ = kv_type_k;
        _ = kv_type_v;
        @panic("WebGPU sdpaPrefill: not yet implemented");
    }

    pub fn gemvT(self: *WebGpuBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        _ = self;
        _ = x;
        _ = w;
        _ = y;
        _ = n;
        _ = k;
        @panic("WebGPU gemvT: not yet implemented");
    }

    pub fn gemvNvfp4St(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, global_scale: f32, y: [*]f32, n: usize, k: usize) void {
        _ = self;
        _ = x;
        _ = w_packed;
        _ = w_scales;
        _ = global_scale;
        _ = y;
        _ = n;
        _ = k;
        @panic("WebGPU gemvNvfp4St: not yet implemented");
    }

    pub fn gemvMlxQ(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, w_biases: [*]const u8, y: [*]f32, n: usize, k: usize, bits: u32) void {
        _ = self;
        _ = x;
        _ = w_packed;
        _ = w_scales;
        _ = w_biases;
        _ = y;
        _ = n;
        _ = k;
        _ = bits;
        @panic("WebGPU gemvMlxQ: not yet implemented");
    }

    pub fn gemvMxfp4St(self: *WebGpuBackend, x: [*]const f32, w_packed: [*]const u8, w_scales: [*]const u8, y: [*]f32, n: usize, k: usize) void {
        _ = self;
        _ = x;
        _ = w_packed;
        _ = w_scales;
        _ = y;
        _ = n;
        _ = k;
        @panic("WebGPU gemvMxfp4St: not yet implemented");
    }

    pub fn gemvMulti(self: *WebGpuBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    pub fn causalConv1dSilu(self: *WebGpuBackend, x: [*]const f32, state: [*]f32, conv_w: [*]const f32, conv_bias: ?[*]const f32, output: [*]f32, conv_ch: usize, d_conv: usize) void {
        _ = self;
        _ = x;
        _ = state;
        _ = conv_w;
        _ = conv_bias;
        _ = output;
        _ = conv_ch;
        _ = d_conv;
        @panic("WebGPU causalConv1dSilu: not yet implemented");
    }

    pub fn deltaNet(self: *WebGpuBackend, q: [*]const f32, k_in: [*]const f32, v: [*]const f32, beta: [*]const f32, state: []f32, output: [*]f32, nh: usize, hd: usize, kv_dim: usize) void {
        _ = self;
        _ = q;
        _ = k_in;
        _ = v;
        _ = beta;
        _ = state;
        _ = output;
        _ = nh;
        _ = hd;
        _ = kv_dim;
        @panic("WebGPU deltaNet: not yet implemented");
    }

    // ── Sync + Batch + Memory ───────────────────────────────────

    pub fn sync(self: *WebGpuBackend) void {
        _ = self.fn_device_poll(self.device, 1, null);
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
            .is_gpu = true,
            .is_uma = false,
        };
    }
};
