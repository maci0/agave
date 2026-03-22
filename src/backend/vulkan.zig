//! Vulkan compute backend via MoltenVK (macOS) or native Vulkan (Linux).
//! Uses SPIR-V compute shaders for GPU dispatch with subgroup reduction.
//! Falls back to CPU for unsupported quantization types.
//!
//! The Vulkan library (libvulkan.so / libMoltenVK.dylib) is loaded at runtime
//! via std.DynLib — no link-time dependency. If the library is not available,
//! init() returns error.VulkanNotAvailable and the caller falls back to
//! another backend.

const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("backend.zig");
const TensorData = backend_mod.TensorData;
const DType = backend_mod.DType;
const CpuBackend = backend_mod.CpuBackend;

// ── Vulkan types (native Zig definitions) ───────────────────────

// Base types
const VkResult = c_int;
const VkBool32 = u32;
const VkDeviceSize = u64;
const VkFlags = u32;

const VK_SUCCESS: VkResult = 0;

// Handles (opaque pointers)
const VkInstance = ?*anyopaque;
const VkPhysicalDevice = ?*anyopaque;
const VkDevice = ?*anyopaque;
const VkQueue = ?*anyopaque;
const VkCommandPool = ?*anyopaque;
const VkCommandBuffer = ?*anyopaque;
const VkFence = ?*anyopaque;
const VkDescriptorPool = ?*anyopaque;
const VkDescriptorSet = ?*anyopaque;
const VkDescriptorSetLayout = ?*anyopaque;
const VkPipeline = ?*anyopaque;
const VkPipelineLayout = ?*anyopaque;
const VkPipelineCache = ?*anyopaque;
const VkShaderModule = ?*anyopaque;
const VkBuffer = ?*anyopaque;
const VkDeviceMemory = ?*anyopaque;
const VkSampler = ?*anyopaque;
const VkBufferView = ?*anyopaque;
const VkImageView = ?*anyopaque;

// Structure type constants (VkStructureType values from Vulkan spec)
const VK_STRUCTURE_TYPE_APPLICATION_INFO: c_int = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: c_int = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: c_int = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: c_int = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: c_int = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: c_int = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: c_int = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: c_int = 12;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: c_int = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: c_int = 18;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: c_int = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: c_int = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: c_int = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: c_int = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: c_int = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: c_int = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: c_int = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: c_int = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: c_int = 42;

// Flag constants
const VK_QUEUE_COMPUTE_BIT: VkFlags = 2;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkFlags = 1;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags = 2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 4;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags = 0x20;
const VK_SHARING_MODE_EXCLUSIVE: c_int = 0;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: c_int = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: VkFlags = 0x20;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: c_int = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 1;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkFlags = 2;
const VK_PIPELINE_BIND_POINT_COMPUTE: c_int = 1;
const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: VkFlags = 1;
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkFlags = 1;
const VK_TRUE: VkBool32 = 1;

// VK_API_VERSION_1_1 = VK_MAKE_API_VERSION(0, 1, 1, 0)
const VK_API_VERSION_1_1: u32 = (0 << 29) | (1 << 22) | (1 << 12) | 0;

// ── Vulkan struct definitions (extern struct for C ABI) ─────────

const VkApplicationInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

const VkInstanceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

const VkDeviceQueueCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 0,
    pQueuePriorities: ?*const f32 = null,
};

/// VkPhysicalDeviceFeatures has 55 VkBool32 fields. We define it as a
/// fixed-size blob matching the C struct layout (55 × 4 = 220 bytes).
const VkPhysicalDeviceFeatures = extern struct {
    robustBufferAccess: VkBool32 = 0,
    fullDrawIndexUint32: VkBool32 = 0,
    imageCubeArray: VkBool32 = 0,
    independentBlend: VkBool32 = 0,
    geometryShader: VkBool32 = 0,
    tessellationShader: VkBool32 = 0,
    sampleRateShading: VkBool32 = 0,
    dualSrcBlend: VkBool32 = 0,
    logicOp: VkBool32 = 0,
    multiDrawIndirect: VkBool32 = 0,
    drawIndirectFirstInstance: VkBool32 = 0,
    depthClamp: VkBool32 = 0,
    depthBiasClamp: VkBool32 = 0,
    fillModeNonSolid: VkBool32 = 0,
    depthBounds: VkBool32 = 0,
    wideLines: VkBool32 = 0,
    largePoints: VkBool32 = 0,
    alphaToOne: VkBool32 = 0,
    multiViewport: VkBool32 = 0,
    samplerAnisotropy: VkBool32 = 0,
    textureCompressionETC2: VkBool32 = 0,
    textureCompressionASTC_LDR: VkBool32 = 0,
    textureCompressionBC: VkBool32 = 0,
    occlusionQueryPrecise: VkBool32 = 0,
    pipelineStatisticsQuery: VkBool32 = 0,
    vertexPipelineStoresAndAtomics: VkBool32 = 0,
    fragmentStoresAndAtomics: VkBool32 = 0,
    shaderTessellationAndGeometryPointSize: VkBool32 = 0,
    shaderImageGatherExtended: VkBool32 = 0,
    shaderStorageImageExtendedFormats: VkBool32 = 0,
    shaderStorageImageMultisample: VkBool32 = 0,
    shaderStorageImageReadWithoutFormat: VkBool32 = 0,
    shaderStorageImageWriteWithoutFormat: VkBool32 = 0,
    shaderUniformBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderSampledImageArrayDynamicIndexing: VkBool32 = 0,
    shaderStorageBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderStorageImageArrayDynamicIndexing: VkBool32 = 0,
    shaderClipDistance: VkBool32 = 0,
    shaderCullDistance: VkBool32 = 0,
    shaderFloat64: VkBool32 = 0,
    shaderInt64: VkBool32 = 0,
    shaderInt16: VkBool32 = 0,
    shaderResourceResidency: VkBool32 = 0,
    shaderResourceMinLod: VkBool32 = 0,
    sparseBinding: VkBool32 = 0,
    sparseResidencyBuffer: VkBool32 = 0,
    sparseResidencyImage2D: VkBool32 = 0,
    sparseResidencyImage3D: VkBool32 = 0,
    sparseResidency2Samples: VkBool32 = 0,
    sparseResidency4Samples: VkBool32 = 0,
    sparseResidency8Samples: VkBool32 = 0,
    sparseResidency16Samples: VkBool32 = 0,
    sparseResidencyAliased: VkBool32 = 0,
    variableMultisampleRate: VkBool32 = 0,
    inheritedQueries: VkBool32 = 0,
};

const VkDeviceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?*const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const VkPhysicalDeviceFeatures = null,
};

const VkCommandPoolCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    queueFamilyIndex: u32 = 0,
};

const VkCommandBufferAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = null,
    level: c_int = 0,
    commandBufferCount: u32 = 0,
};

const VkCommandBufferBeginInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

const VkFenceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
};

const VkDescriptorPoolSize = extern struct {
    type: c_int = 0,
    descriptorCount: u32 = 0,
};

const VkDescriptorPoolCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};

const VkDescriptorSetAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool = null,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?*const VkDescriptorSetLayout = null,
};

const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: c_int = 0,
    descriptorCount: u32 = 0,
    stageFlags: VkFlags = 0,
    pImmutableSamplers: ?*const VkSampler = null,
};

const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

const VkPushConstantRange = extern struct {
    stageFlags: VkFlags = 0,
    offset: u32 = 0,
    size: u32 = 0,
};

const VkPipelineLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?*const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?*const VkPushConstantRange = null,
};

const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = 0,
    pMapEntries: ?*const anyopaque = null,
    dataSize: usize = 0,
    pData: ?*const anyopaque = null,
};

const VkPipelineShaderStageCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    stage: VkFlags = 0,
    module: VkShaderModule = null,
    pName: ?[*:0]const u8 = null,
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

const VkComputePipelineCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    stage: VkPipelineShaderStageCreateInfo = .{},
    layout: VkPipelineLayout = null,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = 0,
};

const VkShaderModuleCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    codeSize: usize = 0,
    pCode: ?*const u32 = null,
};

const VkBufferCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFlags = 0,
    size: VkDeviceSize = 0,
    usage: VkFlags = 0,
    sharingMode: c_int = 0,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*const u32 = null,
};

const VkMemoryRequirements = extern struct {
    size: VkDeviceSize = 0,
    alignment: VkDeviceSize = 0,
    memoryTypeBits: u32 = 0,
};

const VkMemoryAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

const VkMemoryType = extern struct {
    propertyFlags: VkFlags = 0,
    heapIndex: u32 = 0,
};

const VkMemoryHeap = extern struct {
    size: VkDeviceSize = 0,
    flags: VkFlags = 0,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32 = 0,
    memoryTypes: [32]VkMemoryType = [_]VkMemoryType{.{}} ** 32,
    memoryHeapCount: u32 = 0,
    memoryHeaps: [16]VkMemoryHeap = [_]VkMemoryHeap{.{}} ** 16,
};

const VkQueueFamilyProperties = extern struct {
    queueFlags: VkFlags = 0,
    queueCount: u32 = 0,
    timestampValidBits: u32 = 0,
    minImageTransferGranularity: extern struct {
        width: u32 = 0,
        height: u32 = 0,
        depth: u32 = 0,
    } = .{},
};

const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer = null,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = 0,
};

const VkWriteDescriptorSet = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet = null,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 0,
    descriptorType: c_int = 0,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const VkBufferView = null,
};

const VkSubmitInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const VkFlags = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?*const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

/// VkPhysicalDeviceProperties — device name, type, API/driver version, limits.
/// VkPhysicalDeviceLimits (504 bytes) and VkPhysicalDeviceSparseProperties (20 bytes)
/// are defined as opaque blobs since we only need the header fields.
const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: c_int,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: [504]u8, // VkPhysicalDeviceLimits
    sparseProperties: [20]u8, // VkPhysicalDeviceSparseProperties
};

// ── Vulkan function pointer types ───────────────────────────────

const FnCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(.c) VkResult;
const FnDestroyInstance = *const fn (VkInstance, ?*const anyopaque) callconv(.c) void;
const FnEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
const FnGetPhysicalDeviceQueueFamilyProperties = *const fn (VkPhysicalDevice, *u32, ?[*]VkQueueFamilyProperties) callconv(.c) void;
const FnGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
const FnGetPhysicalDeviceProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties) callconv(.c) void;
const FnCreateDevice = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*const anyopaque, *VkDevice) callconv(.c) VkResult;
const FnDestroyDevice = *const fn (VkDevice, ?*const anyopaque) callconv(.c) void;
const FnGetDeviceQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
const FnDeviceWaitIdle = *const fn (VkDevice) callconv(.c) VkResult;

const FnCreateCommandPool = *const fn (VkDevice, *const VkCommandPoolCreateInfo, ?*const anyopaque, *VkCommandPool) callconv(.c) VkResult;
const FnDestroyCommandPool = *const fn (VkDevice, VkCommandPool, ?*const anyopaque) callconv(.c) void;
const FnAllocateCommandBuffers = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, *VkCommandBuffer) callconv(.c) VkResult;
const FnResetCommandBuffer = *const fn (VkCommandBuffer, VkFlags) callconv(.c) VkResult;
const FnBeginCommandBuffer = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
const FnEndCommandBuffer = *const fn (VkCommandBuffer) callconv(.c) VkResult;

const FnCreateFence = *const fn (VkDevice, *const VkFenceCreateInfo, ?*const anyopaque, *VkFence) callconv(.c) VkResult;
const FnDestroyFence = *const fn (VkDevice, VkFence, ?*const anyopaque) callconv(.c) void;
const FnResetFences = *const fn (VkDevice, u32, *const VkFence) callconv(.c) VkResult;
const FnWaitForFences = *const fn (VkDevice, u32, *const VkFence, VkBool32, u64) callconv(.c) VkResult;

const FnQueueSubmit = *const fn (VkQueue, u32, *const VkSubmitInfo, VkFence) callconv(.c) VkResult;

const FnCreateDescriptorPool = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*const anyopaque, *VkDescriptorPool) callconv(.c) VkResult;
const FnDestroyDescriptorPool = *const fn (VkDevice, VkDescriptorPool, ?*const anyopaque) callconv(.c) void;
const FnAllocateDescriptorSets = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, *VkDescriptorSet) callconv(.c) VkResult;
const FnFreeDescriptorSets = *const fn (VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) callconv(.c) VkResult;
const FnUpdateDescriptorSets = *const fn (VkDevice, u32, [*]const VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.c) void;

const FnCreateDescriptorSetLayout = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
const FnDestroyDescriptorSetLayout = *const fn (VkDevice, VkDescriptorSetLayout, ?*const anyopaque) callconv(.c) void;

const FnCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
const FnDestroyPipelineLayout = *const fn (VkDevice, VkPipelineLayout, ?*const anyopaque) callconv(.c) void;

const FnCreateComputePipelines = *const fn (VkDevice, VkPipelineCache, u32, *const VkComputePipelineCreateInfo, ?*const anyopaque, *VkPipeline) callconv(.c) VkResult;
const FnDestroyPipeline = *const fn (VkDevice, VkPipeline, ?*const anyopaque) callconv(.c) void;

const FnCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(.c) VkResult;
const FnDestroyShaderModule = *const fn (VkDevice, VkShaderModule, ?*const anyopaque) callconv(.c) void;

const FnCreateBuffer = *const fn (VkDevice, *const VkBufferCreateInfo, ?*const anyopaque, *VkBuffer) callconv(.c) VkResult;
const FnDestroyBuffer = *const fn (VkDevice, VkBuffer, ?*const anyopaque) callconv(.c) void;
const FnGetBufferMemoryRequirements = *const fn (VkDevice, VkBuffer, *VkMemoryRequirements) callconv(.c) void;

const FnAllocateMemory = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*const anyopaque, *VkDeviceMemory) callconv(.c) VkResult;
const FnFreeMemory = *const fn (VkDevice, VkDeviceMemory, ?*const anyopaque) callconv(.c) void;
const FnBindBufferMemory = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.c) VkResult;
const FnMapMemory = *const fn (VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, *?*anyopaque) callconv(.c) VkResult;
const FnUnmapMemory = *const fn (VkDevice, VkDeviceMemory) callconv(.c) void;

const FnCmdBindPipeline = *const fn (VkCommandBuffer, c_int, VkPipeline) callconv(.c) void;
const FnCmdBindDescriptorSets = *const fn (VkCommandBuffer, c_int, VkPipelineLayout, u32, u32, *const VkDescriptorSet, u32, ?*const u32) callconv(.c) void;
const FnCmdPushConstants = *const fn (VkCommandBuffer, VkPipelineLayout, VkFlags, u32, u32, [*]const u8) callconv(.c) void;
const FnCmdDispatch = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;

// ── Library name ────────────────────────────────────────────────

const vk_lib_name = switch (builtin.os.tag) {
    .macos => "libMoltenVK.dylib",
    .linux => "libvulkan.so.1",
    .windows => "vulkan-1.dll",
    else => "libvulkan.so",
};

// ── Embedded SPIR-V shaders ─────────────────────────────────────

// Elementwise
const spv_silu = @embedFile("kernels/vulkan/silu.spv");
const spv_gelu = @embedFile("kernels/vulkan/gelu.spv");
const spv_add = @embedFile("kernels/vulkan/add.spv");
const spv_mul = @embedFile("kernels/vulkan/mul.spv");

// Normalization
const spv_rms_norm = @embedFile("kernels/vulkan/rms_norm.spv");
const spv_softmax = @embedFile("kernels/vulkan/softmax.spv");
const spv_l2_norm = @embedFile("kernels/vulkan/l2_norm.spv");

// Position
const spv_rope = @embedFile("kernels/vulkan/rope.spv");

// GEMV
const spv_gemv_f32 = @embedFile("kernels/vulkan/gemv_f32.spv");
const spv_gemv_q8_0 = @embedFile("kernels/vulkan/gemv_q8_0.spv");
const spv_gemv_q4_0 = @embedFile("kernels/vulkan/gemv_q4_0.spv");
const spv_gemv_bf16 = @embedFile("kernels/vulkan/gemv_bf16.spv");
const spv_gemv_f16 = @embedFile("kernels/vulkan/gemv_f16.spv");
const spv_gemv_q4_k = @embedFile("kernels/vulkan/gemv_q4_k.spv");
const spv_gemv_q5_k = @embedFile("kernels/vulkan/gemv_q5_k.spv");
const spv_gemv_q6_k = @embedFile("kernels/vulkan/gemv_q6_k.spv");
const spv_gemv_fp8_e4m3 = @embedFile("kernels/vulkan/gemv_fp8_e4m3.spv");
const spv_gemv_fp8_e5m2 = @embedFile("kernels/vulkan/gemv_fp8_e5m2.spv");

// Attention
const spv_sdpa = @embedFile("kernels/vulkan/sdpa.spv");

// Embedding
const spv_embedding = @embedFile("kernels/vulkan/embedding.spv");

// SSM (State Space Model)
const spv_conv1d = @embedFile("kernels/vulkan/conv1d.spv");

// ── Tuning constants ─────────────────────────────────────────────

/// Maximum number of physical devices to enumerate.
const max_physical_devices: u32 = 8;

/// Maximum number of queue families to query.
const max_queue_families: u32 = 16;

/// Maximum descriptor sets / descriptor count for the descriptor pool.
const max_descriptor_sets: u32 = 128;

/// Workgroup size for all compute shaders.
const workgroup_size: u32 = 256;

/// Maximum sequence length for the fused SDPA kernel (limited by shared memory).
const sdpa_max_seq_len: usize = 4096;

/// Maximum per-head dimension for the fused SDPA kernel.
const sdpa_max_head_dim: usize = 256;

// ── Backend struct ───────────────────────────────────────────────

/// Vulkan GPU backend — SPIR-V compute shaders with dynamic library loading.
pub const VulkanBackend = struct {
    instance: VkInstance = null,
    phys_device: VkPhysicalDevice = null,
    device: VkDevice = null,
    queue: VkQueue = null,
    queue_family: u32 = 0,
    cmd_pool: VkCommandPool = null,
    cmd_buf: VkCommandBuffer = null,
    fence: VkFence = null,
    desc_pool: VkDescriptorPool = null,

    // Elementwise pipelines
    pipe_silu: PipelineInfo = .{},
    pipe_gelu: PipelineInfo = .{},
    pipe_add: PipelineInfo = .{},
    pipe_mul: PipelineInfo = .{},

    // Normalization pipelines
    pipe_rms_norm: PipelineInfo = .{},
    pipe_softmax: PipelineInfo = .{},
    pipe_l2_norm: PipelineInfo = .{},

    // Position pipelines
    pipe_rope: PipelineInfo = .{},

    // GEMV pipelines
    pipe_gemv_f32: PipelineInfo = .{},
    pipe_gemv_q8_0: PipelineInfo = .{},
    pipe_gemv_q4_0: PipelineInfo = .{},
    pipe_gemv_bf16: PipelineInfo = .{},
    pipe_gemv_f16: PipelineInfo = .{},
    pipe_gemv_q4_k: PipelineInfo = .{},
    pipe_gemv_q5_k: PipelineInfo = .{},
    pipe_gemv_q6_k: PipelineInfo = .{},
    pipe_gemv_fp8_e4m3: PipelineInfo = .{},
    pipe_gemv_fp8_e5m2: PipelineInfo = .{},

    // Attention pipelines
    pipe_sdpa: PipelineInfo = .{},

    // Embedding pipeline
    pipe_embedding: PipelineInfo = .{},

    // SSM (State Space Model) pipelines
    pipe_conv1d: PipelineInfo = .{},

    /// Device name retrieved from VkPhysicalDeviceProperties (e.g., "Apple M4 Pro").
    device_name: [256]u8 = undefined,
    device_name_len: usize = 0,

    /// Total device-local VRAM in bytes (sum of all device-local heaps).
    total_vram: usize = 0,

    /// Vulkan API version reported by the device (e.g., VK_API_VERSION_1_3).
    api_version: u32 = 0,

    /// Pre-formatted Vulkan version string (e.g., "Vulkan 1.3").
    vk_ver_str: [24]u8 = .{0} ** 24,

    // Memory type index for host-visible coherent memory
    host_mem_type: u32 = 0,

    /// Allocator for internal data structures (buffer cache).
    allocator: std.mem.Allocator = undefined,

    /// Cache of GPU buffers keyed by host pointer address.
    /// Used for immutable data (mmap'd weights, norm weights). Buffer is
    /// created and uploaded once on first use, then reused on subsequent calls.
    /// This eliminates re-uploading gigabytes of weight data per token.
    buf_cache: std.AutoHashMap(usize, CachedBuf) = undefined,

    // Dynamic library handle
    lib: std.DynLib = undefined,

    // Vulkan function pointers (loaded from libvulkan / libMoltenVK)
    vkDestroyInstance: FnDestroyInstance = undefined,
    vkEnumeratePhysicalDevices: FnEnumeratePhysicalDevices = undefined,
    vkGetPhysicalDeviceQueueFamilyProperties: FnGetPhysicalDeviceQueueFamilyProperties = undefined,
    vkGetPhysicalDeviceMemoryProperties: FnGetPhysicalDeviceMemoryProperties = undefined,
    vkCreateDevice: FnCreateDevice = undefined,
    vkDestroyDevice: FnDestroyDevice = undefined,
    vkGetDeviceQueue: FnGetDeviceQueue = undefined,
    vkDeviceWaitIdle: FnDeviceWaitIdle = undefined,
    vkCreateCommandPool: FnCreateCommandPool = undefined,
    vkDestroyCommandPool: FnDestroyCommandPool = undefined,
    vkAllocateCommandBuffers: FnAllocateCommandBuffers = undefined,
    vkResetCommandBuffer: FnResetCommandBuffer = undefined,
    vkBeginCommandBuffer: FnBeginCommandBuffer = undefined,
    vkEndCommandBuffer: FnEndCommandBuffer = undefined,
    vkCreateFence: FnCreateFence = undefined,
    vkDestroyFence: FnDestroyFence = undefined,
    vkResetFences: FnResetFences = undefined,
    vkWaitForFences: FnWaitForFences = undefined,
    vkQueueSubmit: FnQueueSubmit = undefined,
    vkCreateDescriptorPool: FnCreateDescriptorPool = undefined,
    vkDestroyDescriptorPool: FnDestroyDescriptorPool = undefined,
    vkAllocateDescriptorSets: FnAllocateDescriptorSets = undefined,
    vkFreeDescriptorSets: FnFreeDescriptorSets = undefined,
    vkUpdateDescriptorSets: FnUpdateDescriptorSets = undefined,
    vkCreateDescriptorSetLayout: FnCreateDescriptorSetLayout = undefined,
    vkDestroyDescriptorSetLayout: FnDestroyDescriptorSetLayout = undefined,
    vkCreatePipelineLayout: FnCreatePipelineLayout = undefined,
    vkDestroyPipelineLayout: FnDestroyPipelineLayout = undefined,
    vkCreateComputePipelines: FnCreateComputePipelines = undefined,
    vkDestroyPipeline: FnDestroyPipeline = undefined,
    vkCreateShaderModule: FnCreateShaderModule = undefined,
    vkDestroyShaderModule: FnDestroyShaderModule = undefined,
    vkCreateBuffer: FnCreateBuffer = undefined,
    vkDestroyBuffer: FnDestroyBuffer = undefined,
    vkGetBufferMemoryRequirements: FnGetBufferMemoryRequirements = undefined,
    vkAllocateMemory: FnAllocateMemory = undefined,
    vkFreeMemory: FnFreeMemory = undefined,
    vkBindBufferMemory: FnBindBufferMemory = undefined,
    vkMapMemory: FnMapMemory = undefined,
    vkUnmapMemory: FnUnmapMemory = undefined,
    vkCmdBindPipeline: FnCmdBindPipeline = undefined,
    vkCmdBindDescriptorSets: FnCmdBindDescriptorSets = undefined,
    vkCmdPushConstants: FnCmdPushConstants = undefined,
    vkCmdDispatch: FnCmdDispatch = undefined,

    // Activation buffer pool
    act_pool: [act_pool_capacity]PoolEntry = [_]PoolEntry{.{}} ** act_pool_capacity,
    act_pool_count: u32 = 0,

    /// Number of SPIR-V compute pipelines compiled at init.
    pub const n_pipelines: u32 = 19;

    /// Library name loaded via dlopen at init.
    pub const lib_name = vk_lib_name;

    const CachedBuf = struct {
        vk_buf: VkBuf,
        size: usize,
    };

    /// Pool of reusable activation buffers to avoid per-op vkAllocateMemory/vkFreeMemory.
    const act_pool_capacity = 32;

    const PoolEntry = struct {
        buf: VkBuf = .{ .buf = null, .mem = null },
        size: usize = 0,
        in_use: bool = false,
    };

    /// Get a pooled buffer of at least `size` bytes.
    /// Returns a buffer from the pool if available, otherwise creates a new one.
    fn getPooledBuf(self: *VulkanBackend, size: usize) VkBuf {
        // Try to find a free buffer with sufficient size
        var best_idx: ?usize = null;
        var best_size: usize = std.math.maxInt(usize);
        for (0..self.act_pool_count) |i| {
            if (!self.act_pool[i].in_use and self.act_pool[i].size >= size and self.act_pool[i].size < best_size) {
                best_idx = i;
                best_size = self.act_pool[i].size;
            }
        }
        if (best_idx) |idx| {
            self.act_pool[idx].in_use = true;
            return self.act_pool[idx].buf;
        }
        // Create new buffer and add to pool
        const buf = self.createBuffer(size);
        if (self.act_pool_count < act_pool_capacity) {
            self.act_pool[self.act_pool_count] = .{ .buf = buf, .size = size, .in_use = true };
            self.act_pool_count += 1;
        }
        return buf;
    }

    /// Return a buffer to the pool for reuse.
    fn releasePooledBuf(self: *VulkanBackend, buf: VkBuf) void {
        for (0..self.act_pool_count) |i| {
            if (self.act_pool[i].buf.buf == buf.buf) {
                self.act_pool[i].in_use = false;
                return;
            }
        }
        // Not in pool (pool was full when created) — destroy it
        self.destroyBuffer(buf);
    }

    const PipelineInfo = struct {
        pipeline: VkPipeline = null,
        layout: VkPipelineLayout = null,
        desc_layout: VkDescriptorSetLayout = null,
        desc_set: VkDescriptorSet = null,
    };

    /// Look up a function pointer from the dynamically loaded Vulkan library.
    fn lookup(self: *VulkanBackend, comptime T: type, name: [:0]const u8) ?T {
        return self.lib.lookup(T, name);
    }

    // ── Init ─────────────────────────────────────────────────────

    /// Initialize the Vulkan backend: load library, create instance/device, compile SPIR-V pipelines.
    pub fn init(allocator: std.mem.Allocator) !VulkanBackend {
        var self = VulkanBackend{};
        self.allocator = allocator;
        self.buf_cache = std.AutoHashMap(usize, CachedBuf).init(allocator);
        self.buf_cache.ensureTotalCapacity(backend_mod.buf_cache_initial_capacity) catch {};
        errdefer self.buf_cache.deinit();
        errdefer self.deinitCachedBuffers();

        // Dynamically load Vulkan library
        self.lib = std.DynLib.open(vk_lib_name) catch return error.VulkanNotAvailable;
        errdefer self.lib.close();

        // Resolve all function pointers
        const vkCreateInstance = self.lookup(FnCreateInstance, "vkCreateInstance") orelse return error.VulkanNotAvailable;
        self.vkDestroyInstance = self.lookup(FnDestroyInstance, "vkDestroyInstance") orelse return error.VulkanNotAvailable;
        self.vkEnumeratePhysicalDevices = self.lookup(FnEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices") orelse return error.VulkanNotAvailable;
        self.vkGetPhysicalDeviceQueueFamilyProperties = self.lookup(FnGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return error.VulkanNotAvailable;
        self.vkGetPhysicalDeviceMemoryProperties = self.lookup(FnGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties") orelse return error.VulkanNotAvailable;
        self.vkCreateDevice = self.lookup(FnCreateDevice, "vkCreateDevice") orelse return error.VulkanNotAvailable;
        self.vkDestroyDevice = self.lookup(FnDestroyDevice, "vkDestroyDevice") orelse return error.VulkanNotAvailable;
        self.vkGetDeviceQueue = self.lookup(FnGetDeviceQueue, "vkGetDeviceQueue") orelse return error.VulkanNotAvailable;
        self.vkDeviceWaitIdle = self.lookup(FnDeviceWaitIdle, "vkDeviceWaitIdle") orelse return error.VulkanNotAvailable;
        self.vkCreateCommandPool = self.lookup(FnCreateCommandPool, "vkCreateCommandPool") orelse return error.VulkanNotAvailable;
        self.vkDestroyCommandPool = self.lookup(FnDestroyCommandPool, "vkDestroyCommandPool") orelse return error.VulkanNotAvailable;
        self.vkAllocateCommandBuffers = self.lookup(FnAllocateCommandBuffers, "vkAllocateCommandBuffers") orelse return error.VulkanNotAvailable;
        self.vkResetCommandBuffer = self.lookup(FnResetCommandBuffer, "vkResetCommandBuffer") orelse return error.VulkanNotAvailable;
        self.vkBeginCommandBuffer = self.lookup(FnBeginCommandBuffer, "vkBeginCommandBuffer") orelse return error.VulkanNotAvailable;
        self.vkEndCommandBuffer = self.lookup(FnEndCommandBuffer, "vkEndCommandBuffer") orelse return error.VulkanNotAvailable;
        self.vkCreateFence = self.lookup(FnCreateFence, "vkCreateFence") orelse return error.VulkanNotAvailable;
        self.vkDestroyFence = self.lookup(FnDestroyFence, "vkDestroyFence") orelse return error.VulkanNotAvailable;
        self.vkResetFences = self.lookup(FnResetFences, "vkResetFences") orelse return error.VulkanNotAvailable;
        self.vkWaitForFences = self.lookup(FnWaitForFences, "vkWaitForFences") orelse return error.VulkanNotAvailable;
        self.vkQueueSubmit = self.lookup(FnQueueSubmit, "vkQueueSubmit") orelse return error.VulkanNotAvailable;
        self.vkCreateDescriptorPool = self.lookup(FnCreateDescriptorPool, "vkCreateDescriptorPool") orelse return error.VulkanNotAvailable;
        self.vkDestroyDescriptorPool = self.lookup(FnDestroyDescriptorPool, "vkDestroyDescriptorPool") orelse return error.VulkanNotAvailable;
        self.vkAllocateDescriptorSets = self.lookup(FnAllocateDescriptorSets, "vkAllocateDescriptorSets") orelse return error.VulkanNotAvailable;
        self.vkFreeDescriptorSets = self.lookup(FnFreeDescriptorSets, "vkFreeDescriptorSets") orelse return error.VulkanNotAvailable;
        self.vkUpdateDescriptorSets = self.lookup(FnUpdateDescriptorSets, "vkUpdateDescriptorSets") orelse return error.VulkanNotAvailable;
        self.vkCreateDescriptorSetLayout = self.lookup(FnCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout") orelse return error.VulkanNotAvailable;
        self.vkDestroyDescriptorSetLayout = self.lookup(FnDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout") orelse return error.VulkanNotAvailable;
        self.vkCreatePipelineLayout = self.lookup(FnCreatePipelineLayout, "vkCreatePipelineLayout") orelse return error.VulkanNotAvailable;
        self.vkDestroyPipelineLayout = self.lookup(FnDestroyPipelineLayout, "vkDestroyPipelineLayout") orelse return error.VulkanNotAvailable;
        self.vkCreateComputePipelines = self.lookup(FnCreateComputePipelines, "vkCreateComputePipelines") orelse return error.VulkanNotAvailable;
        self.vkDestroyPipeline = self.lookup(FnDestroyPipeline, "vkDestroyPipeline") orelse return error.VulkanNotAvailable;
        self.vkCreateShaderModule = self.lookup(FnCreateShaderModule, "vkCreateShaderModule") orelse return error.VulkanNotAvailable;
        self.vkDestroyShaderModule = self.lookup(FnDestroyShaderModule, "vkDestroyShaderModule") orelse return error.VulkanNotAvailable;
        self.vkCreateBuffer = self.lookup(FnCreateBuffer, "vkCreateBuffer") orelse return error.VulkanNotAvailable;
        self.vkDestroyBuffer = self.lookup(FnDestroyBuffer, "vkDestroyBuffer") orelse return error.VulkanNotAvailable;
        self.vkGetBufferMemoryRequirements = self.lookup(FnGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements") orelse return error.VulkanNotAvailable;
        self.vkAllocateMemory = self.lookup(FnAllocateMemory, "vkAllocateMemory") orelse return error.VulkanNotAvailable;
        self.vkFreeMemory = self.lookup(FnFreeMemory, "vkFreeMemory") orelse return error.VulkanNotAvailable;
        self.vkBindBufferMemory = self.lookup(FnBindBufferMemory, "vkBindBufferMemory") orelse return error.VulkanNotAvailable;
        self.vkMapMemory = self.lookup(FnMapMemory, "vkMapMemory") orelse return error.VulkanNotAvailable;
        self.vkUnmapMemory = self.lookup(FnUnmapMemory, "vkUnmapMemory") orelse return error.VulkanNotAvailable;
        self.vkCmdBindPipeline = self.lookup(FnCmdBindPipeline, "vkCmdBindPipeline") orelse return error.VulkanNotAvailable;
        self.vkCmdBindDescriptorSets = self.lookup(FnCmdBindDescriptorSets, "vkCmdBindDescriptorSets") orelse return error.VulkanNotAvailable;
        self.vkCmdPushConstants = self.lookup(FnCmdPushConstants, "vkCmdPushConstants") orelse return error.VulkanNotAvailable;
        self.vkCmdDispatch = self.lookup(FnCmdDispatch, "vkCmdDispatch") orelse return error.VulkanNotAvailable;

        // Create instance
        const app_info = VkApplicationInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "agave",
            .applicationVersion = 1,
            .pEngineName = "agave",
            .engineVersion = 1,
            .apiVersion = VK_API_VERSION_1_1,
        };
        const inst_info = VkInstanceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        };
        if (vkCreateInstance(&inst_info, null, &self.instance) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Pick first physical device
        var dev_count: u32 = 0;
        _ = self.vkEnumeratePhysicalDevices(self.instance, &dev_count, null);
        if (dev_count == 0) return error.NoVulkanDevice;
        var devs: [max_physical_devices]VkPhysicalDevice = undefined;
        _ = self.vkEnumeratePhysicalDevices(self.instance, &dev_count, &devs);
        self.phys_device = devs[0];

        // Query device properties (name, API version)
        if (self.lookup(FnGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties")) |getProps| {
            var props: VkPhysicalDeviceProperties = undefined;
            getProps(self.phys_device, &props);
            self.api_version = props.apiVersion;
            const name_len = std.mem.indexOfScalar(u8, &props.deviceName, 0) orelse props.deviceName.len;
            @memcpy(self.device_name[0..name_len], props.deviceName[0..name_len]);
            self.device_name_len = name_len;
            // Format "Vulkan X.Y" from VK_API_VERSION
            const vk_major = (props.apiVersion >> 22) & 0x7F;
            const vk_minor = (props.apiVersion >> 12) & 0x3FF;
            _ = std.fmt.bufPrint(&self.vk_ver_str, "Vulkan {d}.{d}", .{ vk_major, vk_minor }) catch {};
        }

        // Query total device-local VRAM
        {
            var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
            self.vkGetPhysicalDeviceMemoryProperties(self.phys_device, &mem_props);
            var total: usize = 0;
            for (0..mem_props.memoryHeapCount) |i| {
                if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                    total += mem_props.memoryHeaps[i].size;
                }
            }
            self.total_vram = total;
        }

        // Find compute queue family
        var qf_count: u32 = 0;
        self.vkGetPhysicalDeviceQueueFamilyProperties(self.phys_device, &qf_count, null);
        var qf_props: [max_queue_families]VkQueueFamilyProperties = undefined;
        self.vkGetPhysicalDeviceQueueFamilyProperties(self.phys_device, &qf_count, &qf_props);
        for (0..qf_count) |i| {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT != 0) {
                self.queue_family = @intCast(i);
                break;
            }
        }

        // Create logical device
        const queue_priority: f32 = 1.0;
        const queue_ci = VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = self.queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        const dev_ci = VkDeviceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_ci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,
        };
        if (self.vkCreateDevice(self.phys_device, &dev_ci, null, &self.device) != VK_SUCCESS)
            return error.VulkanInitFailed;

        self.vkGetDeviceQueue(self.device, self.queue_family, 0, &self.queue);

        // Command pool + buffer
        const pool_ci = VkCommandPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.queue_family,
        };
        if (self.vkCreateCommandPool(self.device, &pool_ci, null, &self.cmd_pool) != VK_SUCCESS)
            return error.VulkanInitFailed;

        const alloc_ci = VkCommandBufferAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (self.vkAllocateCommandBuffers(self.device, &alloc_ci, &self.cmd_buf) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Fence
        const fence_ci = VkFenceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };
        if (self.vkCreateFence(self.device, &fence_ci, null, &self.fence) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Descriptor pool (increased for more shader pipelines)
        const pool_sizes = [_]VkDescriptorPoolSize{
            .{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_descriptor_sets },
        };
        const dp_ci = VkDescriptorPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = max_descriptor_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_sizes,
        };
        if (self.vkCreateDescriptorPool(self.device, &dp_ci, null, &self.desc_pool) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Find best memory type: prefer DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (ReBAR/SAM)
        // for VRAM-speed access with direct mapping. Fall back to HOST_VISIBLE + HOST_COHERENT.
        var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
        self.vkGetPhysicalDeviceMemoryProperties(self.phys_device, &mem_props);
        const wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const ideal = wanted | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        var found_fallback: bool = false;
        for (0..mem_props.memoryTypeCount) |i| {
            const flags = mem_props.memoryTypes[i].propertyFlags;
            if (flags & ideal == ideal) {
                self.host_mem_type = @intCast(i);
                break;
            }
            if (!found_fallback and flags & wanted == wanted) {
                self.host_mem_type = @intCast(i);
                found_fallback = true;
            }
        }

        // Create pipelines — bindings, push_size
        // Elementwise: 2 bufs (in, out), 4 bytes push (n)
        self.pipe_silu = try self.createPipeline(spv_silu, 2, 4);
        self.pipe_gelu = try self.createPipeline(spv_gelu, 2, 4);
        // Binary: 3 bufs (a, b, out), 4 bytes push (n)
        self.pipe_add = try self.createPipeline(spv_add, 3, 4);
        self.pipe_mul = try self.createPipeline(spv_mul, 3, 4);
        // Normalization: rmsNorm 3 bufs, 8 bytes push (n, eps)
        self.pipe_rms_norm = try self.createPipeline(spv_rms_norm, 3, 8);
        // Softmax: 1 buf (in-place), 4 bytes push (n)
        self.pipe_softmax = try self.createPipeline(spv_softmax, 1, 4);
        // L2 norm: 1 buf (in-place), 8 bytes push (n, eps)
        self.pipe_l2_norm = try self.createPipeline(spv_l2_norm, 1, 8);
        // RoPE: 1 buf (in-place), 20 bytes push (pos, n_heads, head_dim, rope_dim, theta)
        self.pipe_rope = try self.createPipeline(spv_rope, 1, 20);
        // GEMV: 3 bufs (x, W, y), 8 bytes push (n, k)
        self.pipe_gemv_f32 = try self.createPipeline(spv_gemv_f32, 3, 8);
        self.pipe_gemv_q8_0 = try self.createPipeline(spv_gemv_q8_0, 3, 8);
        self.pipe_gemv_q4_0 = try self.createPipeline(spv_gemv_q4_0, 3, 8);
        self.pipe_gemv_bf16 = try self.createPipeline(spv_gemv_bf16, 3, 8);
        self.pipe_gemv_f16 = try self.createPipeline(spv_gemv_f16, 3, 8);
        self.pipe_gemv_q4_k = try self.createPipeline(spv_gemv_q4_k, 3, 8);
        self.pipe_gemv_q5_k = try self.createPipeline(spv_gemv_q5_k, 3, 8);
        self.pipe_gemv_q6_k = try self.createPipeline(spv_gemv_q6_k, 3, 8);
        self.pipe_gemv_fp8_e4m3 = try self.createPipeline(spv_gemv_fp8_e4m3, 3, 8);
        self.pipe_gemv_fp8_e5m2 = try self.createPipeline(spv_gemv_fp8_e5m2, 3, 8);
        // SDPA: 4 bufs (Q, K, V, out), 20 bytes push (nh, nkv, hd, sl, scale)
        self.pipe_sdpa = try self.createPipeline(spv_sdpa, 4, 20);
        // Embedding: 3 bufs (token_id, emb_table, output), 8 bytes push (vocab_size, n_embd)
        self.pipe_embedding = try self.createPipeline(spv_embedding, 3, 8);
        // Conv1d: 4 bufs (input, state, conv_w, output), 8 bytes push (conv_ch, d_conv)
        self.pipe_conv1d = try self.createPipeline(spv_conv1d, 4, 8);

        return self;
    }

    /// Release all cached GPU buffers (used by deinit and errdefer).
    fn deinitCachedBuffers(self: *VulkanBackend) void {
        var it = self.buf_cache.valueIterator();
        while (it.next()) |cached| {
            self.vkDestroyBuffer(self.device, cached.vk_buf.buf, null);
            self.vkFreeMemory(self.device, cached.vk_buf.mem, null);
        }
        self.buf_cache.clearRetainingCapacity();
    }

    /// Release all Vulkan resources: cached buffers, pipelines, descriptor pool, device, and instance.
    pub fn deinit(self: *VulkanBackend) void {
        if (self.device == null) return;
        _ = self.vkDeviceWaitIdle(self.device);

        // Release pooled activation buffers
        for (0..self.act_pool_count) |i| {
            self.destroyBuffer(self.act_pool[i].buf);
        }

        // Release all cached weight buffers
        self.deinitCachedBuffers();
        self.buf_cache.deinit();

        const pipelines = [_]*PipelineInfo{
            // Elementwise
            &self.pipe_silu,          &self.pipe_gelu,
            &self.pipe_add,           &self.pipe_mul,
            // Normalization
            &self.pipe_rms_norm,      &self.pipe_softmax,
            &self.pipe_l2_norm,
            // Position
                  &self.pipe_rope,
            // GEMV
            &self.pipe_gemv_f32,      &self.pipe_gemv_q8_0,
            &self.pipe_gemv_q4_0,     &self.pipe_gemv_bf16,
            &self.pipe_gemv_f16,      &self.pipe_gemv_q4_k,
            &self.pipe_gemv_q5_k,     &self.pipe_gemv_q6_k,
            &self.pipe_gemv_fp8_e4m3, &self.pipe_gemv_fp8_e5m2,
            // Attention
            &self.pipe_sdpa,
            // Embedding
                     &self.pipe_embedding,
            // SSM
            &self.pipe_conv1d,
        };
        for (pipelines) |p| {
            if (p.pipeline != null) self.vkDestroyPipeline(self.device, p.pipeline, null);
            if (p.layout != null) self.vkDestroyPipelineLayout(self.device, p.layout, null);
            if (p.desc_layout != null) self.vkDestroyDescriptorSetLayout(self.device, p.desc_layout, null);
        }
        if (self.desc_pool != null) self.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        if (self.fence != null) self.vkDestroyFence(self.device, self.fence, null);
        if (self.cmd_pool != null) self.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        self.vkDestroyDevice(self.device, null);
        if (self.instance != null) self.vkDestroyInstance(self.instance, null);
        self.lib.close();
    }

    fn createPipeline(self: *VulkanBackend, spv: []const u8, n_bindings: u32, push_size: u32) !PipelineInfo {
        // Shader module
        const mod_ci = VkShaderModuleCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spv.len,
            .pCode = @ptrCast(@alignCast(spv.ptr)),
        };
        var shader_mod: VkShaderModule = null;
        if (self.vkCreateShaderModule(self.device, &mod_ci, null, &shader_mod) != VK_SUCCESS)
            return error.ShaderCompileFailed;

        // Descriptor set layout
        var bindings: [4]VkDescriptorSetLayoutBinding = undefined;
        for (0..n_bindings) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            };
        }
        const dsl_ci = VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .bindingCount = n_bindings,
            .pBindings = @ptrCast(&bindings),
        };
        var desc_layout: VkDescriptorSetLayout = null;
        if (self.vkCreateDescriptorSetLayout(self.device, &dsl_ci, null, &desc_layout) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Pipeline layout with push constants
        const push_range = VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_size,
        };
        const pl_ci = VkPipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        var pipe_layout: VkPipelineLayout = null;
        if (self.vkCreatePipelineLayout(self.device, &pl_ci, null, &pipe_layout) != VK_SUCCESS)
            return error.VulkanInitFailed;

        // Compute pipeline
        const stage = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod,
            .pName = "main",
            .pSpecializationInfo = null,
        };
        const pipe_ci = VkComputePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage,
            .layout = pipe_layout,
            .basePipelineHandle = null,
            .basePipelineIndex = 0,
        };
        var pipeline: VkPipeline = null;
        if (self.vkCreateComputePipelines(self.device, null, 1, &pipe_ci, null, &pipeline) != VK_SUCCESS)
            return error.VulkanInitFailed;

        self.vkDestroyShaderModule(self.device, shader_mod, null);

        // Pre-allocate descriptor set — reused across dispatches (synchronous execution).
        var desc_set: VkDescriptorSet = null;
        const ds_ai = VkDescriptorSetAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &desc_layout,
        };
        _ = self.vkAllocateDescriptorSets(self.device, &ds_ai, &desc_set);

        return .{
            .pipeline = pipeline,
            .layout = pipe_layout,
            .desc_layout = desc_layout,
            .desc_set = desc_set,
        };
    }

    // ── Buffer helpers ───────────────────────────────────────────

    const VkBuf = struct { buf: VkBuffer, mem: VkDeviceMemory };

    fn createBuffer(self: *VulkanBackend, size: usize) VkBuf {
        const buf_ci = VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = @max(size, 4),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };
        var buf: VkBuffer = null;
        _ = self.vkCreateBuffer(self.device, &buf_ci, null, &buf);

        var mem_req: VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements(self.device, buf, &mem_req);

        const alloc_info = VkMemoryAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = mem_req.size,
            .memoryTypeIndex = self.host_mem_type,
        };
        var mem: VkDeviceMemory = null;
        _ = self.vkAllocateMemory(self.device, &alloc_info, null, &mem);
        _ = self.vkBindBufferMemory(self.device, buf, mem, 0);

        return .{ .buf = buf, .mem = mem };
    }

    fn destroyBuffer(self: *VulkanBackend, b: VkBuf) void {
        self.vkDestroyBuffer(self.device, b.buf, null);
        self.vkFreeMemory(self.device, b.mem, null);
    }

    /// Get or create a GPU buffer for the given host pointer.
    /// If the pointer was seen before (e.g. mmap'd weights), returns the
    /// cached buffer without re-uploading. Otherwise creates a new buffer,
    /// uploads the data, and caches it for future calls.
    fn getOrUpload(self: *VulkanBackend, ptr: [*]const u8, size: usize) VkBuf {
        const addr = @intFromPtr(ptr);
        if (self.buf_cache.get(addr)) |cached| {
            if (cached.size >= size) return cached.vk_buf;
            // Size grew — release old buffer, recreate below
            self.vkDestroyBuffer(self.device, cached.vk_buf.buf, null);
            self.vkFreeMemory(self.device, cached.vk_buf.mem, null);
            _ = self.buf_cache.remove(addr);
        }
        const buf = self.createBuffer(size);
        self.uploadBuffer(buf.mem, ptr, size);
        self.buf_cache.put(addr, .{ .vk_buf = buf, .size = size }) catch |err| {
            std.log.warn("cache put failed: {}", .{err});
        };
        return buf;
    }

    fn uploadBuffer(self: *VulkanBackend, mem: VkDeviceMemory, data: [*]const u8, size: usize) void {
        var mapped: ?*anyopaque = null;
        _ = self.vkMapMemory(self.device, mem, 0, size, 0, &mapped);
        if (mapped) |ptr| {
            @memcpy(@as([*]u8, @ptrCast(ptr))[0..size], data[0..size]);
            self.vkUnmapMemory(self.device, mem);
        }
    }

    fn downloadF32(self: *VulkanBackend, mem: VkDeviceMemory, data: [*]f32, count: usize) void {
        var mapped: ?*anyopaque = null;
        _ = self.vkMapMemory(self.device, mem, 0, count * @sizeOf(f32), 0, &mapped);
        if (mapped) |ptr| {
            const src: [*]const f32 = @ptrCast(@alignCast(ptr));
            @memcpy(data[0..count], src[0..count]);
            self.vkUnmapMemory(self.device, mem);
        }
    }

    // ── Dispatch helper ──────────────────────────────────────────

    fn dispatch(self: *VulkanBackend, pipe: PipelineInfo, bufs: []const VkBuffer, buf_sizes: []const usize, push_data: [*]const u8, push_size: u32, n_groups: u32) void {
        const desc_set = pipe.desc_set;

        // Update descriptor set with buffer bindings
        var buf_infos: [4]VkDescriptorBufferInfo = undefined;
        var writes: [4]VkWriteDescriptorSet = undefined;
        for (0..bufs.len) |i| {
            buf_infos[i] = .{
                .buffer = bufs[i],
                .offset = 0,
                .range = @max(buf_sizes[i], 4),
            };
            writes[i] = .{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = desc_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buf_infos[i],
                .pTexelBufferView = null,
            };
        }
        self.vkUpdateDescriptorSets(self.device, @intCast(bufs.len), &writes, 0, null);

        // Record command buffer
        const begin_info = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = self.vkResetCommandBuffer(self.cmd_buf, 0);
        _ = self.vkBeginCommandBuffer(self.cmd_buf, &begin_info);

        self.vkCmdBindPipeline(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        self.vkCmdBindDescriptorSets(self.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &desc_set, 0, null);
        self.vkCmdPushConstants(self.cmd_buf, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_data);
        self.vkCmdDispatch(self.cmd_buf, n_groups, 1, 1);

        _ = self.vkEndCommandBuffer(self.cmd_buf);

        // Submit and wait
        const submit_info = VkSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.cmd_buf,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };
        _ = self.vkResetFences(self.device, 1, &self.fence);
        _ = self.vkQueueSubmit(self.queue, 1, &submit_info, self.fence);
        _ = self.vkWaitForFences(self.device, 1, &self.fence, VK_TRUE, ~@as(u64, 0));

        // Descriptor set is pre-allocated per pipeline — no alloc/free per dispatch.
    }

    // ── Weight size helper ──────────────────────────────────────

    const weightBytes = @import("backend.zig").weightBytes;

    // ── Backend interface ────────────────────────────────────────

    /// Return this backend wrapped in the Backend tagged union for dispatch.
    pub fn backend(self: *VulkanBackend) backend_mod.Backend {
        return .{ .vulkan = self };
    }

    /// y[n] = W[n,k] @ x[k]. Native GPU dispatch for supported dtypes.
    pub fn gemv(self: *VulkanBackend, x: [*]const f32, w: TensorData, y: [*]f32, n: usize, k: usize) void {
        const pipe = switch (w.dtype) {
            .f32 => self.pipe_gemv_f32,
            .q8_0 => self.pipe_gemv_q8_0,
            .q4_0 => self.pipe_gemv_q4_0,
            .bf16 => self.pipe_gemv_bf16,
            .f16 => self.pipe_gemv_f16,
            .q4_k => self.pipe_gemv_q4_k,
            .q5_k => self.pipe_gemv_q5_k,
            .q6_k => self.pipe_gemv_q6_k,
            .fp8_e4m3 => self.pipe_gemv_fp8_e4m3,
            .fp8_e5m2 => self.pipe_gemv_fp8_e5m2,
            else => @panic("Vulkan GEMV: unsupported dtype — add a GPU shader"),
        };

        const x_sz = k * @sizeOf(f32);
        const w_sz = weightBytes(w.dtype, n, k);
        const y_sz = n * @sizeOf(f32);

        // Activation buffers: pooled to avoid per-op alloc/free overhead
        const x_buf = self.getPooledBuf(x_sz);
        defer self.releasePooledBuf(x_buf);
        const y_buf = self.getPooledBuf(y_sz);
        defer self.releasePooledBuf(y_buf);

        // Weight buffer: cached (large, immutable mmap'd data)
        const w_vk = self.getOrUpload(w.data, w_sz);

        self.uploadBuffer(x_buf.mem, @ptrCast(x), x_sz);

        const params = [2]u32{ @intCast(n), @intCast(k) };
        const bufs = [_]VkBuffer{ x_buf.buf, w_vk.buf, y_buf.buf };
        const sizes = [_]usize{ x_sz, w_sz, y_sz };
        self.dispatch(pipe, &bufs, &sizes, @ptrCast(&params), 8, @intCast(n));
        self.downloadF32(y_buf.mem, y, n);
    }

    /// output[i] = input[i] * weight[i] * rsqrt(sum_sq / n + eps)
    pub fn rmsNorm(self: *VulkanBackend, input: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);

        const in_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(in_buf);
        const o_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(o_buf);

        // Norm weights are stable pointers — cache them
        const w_vk = self.getOrUpload(@ptrCast(weight), sz);

        self.uploadBuffer(in_buf.mem, @ptrCast(input), sz);

        const params = extern struct { n_val: u32, eps_val: f32 }{ .n_val = @intCast(n), .eps_val = eps };
        const bufs = [_]VkBuffer{ in_buf.buf, w_vk.buf, o_buf.buf };
        const sizes = [_]usize{ sz, sz, sz };
        self.dispatch(self.pipe_rms_norm, &bufs, &sizes, @ptrCast(&params), 8, 1);
        self.downloadF32(o_buf.mem, output, n);
    }

    /// SiLU activation: output[i] = input[i] * sigmoid(input[i])
    pub fn silu(self: *VulkanBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        const a_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(a_buf);
        const o_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(o_buf);

        self.uploadBuffer(a_buf.mem, @ptrCast(input), sz);
        const params = [1]u32{@intCast(n)};
        const bufs = [_]VkBuffer{ a_buf.buf, o_buf.buf };
        const sizes = [_]usize{ sz, sz };
        self.dispatch(self.pipe_silu, &bufs, &sizes, @ptrCast(&params), 4, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(o_buf.mem, output, n);
    }

    /// GELU activation: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(self: *VulkanBackend, input: [*]const f32, output: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        const a_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(a_buf);
        const o_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(o_buf);

        self.uploadBuffer(a_buf.mem, @ptrCast(input), sz);
        const params = [1]u32{@intCast(n)};
        const bufs = [_]VkBuffer{ a_buf.buf, o_buf.buf };
        const sizes = [_]usize{ sz, sz };
        self.dispatch(self.pipe_gelu, &bufs, &sizes, @ptrCast(&params), 4, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(o_buf.mem, output, n);
    }

    /// out[i] = a[i] + b[i]
    pub fn add(self: *VulkanBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        const a_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(a_buf);
        const b_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(b_buf);
        const o_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(o_buf);

        self.uploadBuffer(a_buf.mem, @ptrCast(a), sz);
        self.uploadBuffer(b_buf.mem, @ptrCast(b), sz);
        const params = [1]u32{@intCast(n)};
        const bufs = [_]VkBuffer{ a_buf.buf, b_buf.buf, o_buf.buf };
        const sizes = [_]usize{ sz, sz, sz };
        self.dispatch(self.pipe_add, &bufs, &sizes, @ptrCast(&params), 4, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(o_buf.mem, out, n);
    }

    /// Fused add + rmsNorm (sequential fallback — no fused Vulkan kernel yet).
    pub fn addRmsNorm(self: *VulkanBackend, a: [*]f32, b: [*]const f32, weight: [*]const f32, output: [*]f32, n: usize, eps: f32) void {
        self.add(a, b, a, n);
        self.rmsNorm(a, weight, output, n, eps);
    }

    /// out[i] = a[i] * b[i]
    pub fn mul(self: *VulkanBackend, a: [*]const f32, b: [*]const f32, out: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        const a_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(a_buf);
        const b_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(b_buf);
        const o_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(o_buf);

        self.uploadBuffer(a_buf.mem, @ptrCast(a), sz);
        self.uploadBuffer(b_buf.mem, @ptrCast(b), sz);
        const params = [1]u32{@intCast(n)};
        const bufs = [_]VkBuffer{ a_buf.buf, b_buf.buf, o_buf.buf };
        const sizes = [_]usize{ sz, sz, sz };
        self.dispatch(self.pipe_mul, &bufs, &sizes, @ptrCast(&params), 4, @intCast((n + workgroup_size - 1) / workgroup_size));
        self.downloadF32(o_buf.mem, out, n);
    }

    /// In-place softmax. Single-dispatch fused kernel with subgroup reduction.
    pub fn softmax(self: *VulkanBackend, data: [*]f32, n: usize) void {
        const sz = n * @sizeOf(f32);
        const d_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(d_buf);

        self.uploadBuffer(d_buf.mem, @ptrCast(data), sz);
        const params = [1]u32{@intCast(n)};
        const bufs = [_]VkBuffer{d_buf.buf};
        const sizes = [_]usize{sz};
        self.dispatch(self.pipe_softmax, &bufs, &sizes, @ptrCast(&params), 4, 1);
        self.downloadF32(d_buf.mem, data, n);
    }

    /// Rotary Position Embedding (in-place).
    pub fn rope(self: *VulkanBackend, x: [*]f32, pos: usize, n_heads: usize, head_dim: usize, rope_dim: usize, theta: f32) void {
        const total = n_heads * head_dim;
        const sz = total * @sizeOf(f32);
        const x_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(x_buf);

        self.uploadBuffer(x_buf.mem, @ptrCast(x), sz);
        const params = extern struct { pos: u32, n_heads: u32, head_dim: u32, rope_dim: u32, theta: f32 }{
            .pos = @intCast(pos),
            .n_heads = @intCast(n_heads),
            .head_dim = @intCast(head_dim),
            .rope_dim = @intCast(rope_dim),
            .theta = theta,
        };
        const grid = (n_heads * rope_dim / 2 + workgroup_size - 1) / workgroup_size;
        const bufs = [_]VkBuffer{x_buf.buf};
        const sizes = [_]usize{sz};
        self.dispatch(self.pipe_rope, &bufs, &sizes, @ptrCast(&params), 20, @intCast(grid));
        self.downloadF32(x_buf.mem, x, total);
    }

    /// Embedding lookup via GPU shader (eliminates CPU fallback).
    pub fn embLookup(self: *VulkanBackend, table: TensorData, token_id: u32, output: [*]f32, dim: usize) void {
        // Only f32 embeddings supported on GPU.
        if (table.dtype != .f32) @panic("Vulkan embLookup: non-f32 dtype not supported — add GPU shader");

        // For f32 embeddings, we use the buffer cache to avoid re-uploading
        // the entire embedding table every token. We only know one row size (dim),
        // so estimate a reasonable table size for caching.
        const table_sz = dim * @sizeOf(f32) * 65536; // Assume max 65k vocab

        // Get or upload cached buffer for embedding table
        const table_buf = self.getOrUpload(table.data, table_sz);

        // Create token_id buffer (single u32)
        const token_id_sz = @sizeOf(u32);
        const token_id_buf = self.getPooledBuf(token_id_sz);
        defer self.releasePooledBuf(token_id_buf);

        // Create output buffer
        const output_sz = dim * @sizeOf(f32);
        const output_buf = self.getPooledBuf(output_sz);
        defer self.releasePooledBuf(output_buf);

        // Upload token_id
        self.uploadBuffer(token_id_buf.mem, @ptrCast(&token_id), token_id_sz);

        // Push constants: vocab_size (unused, set to 0), n_embd
        // vocab_size is not actually needed by the shader since the offset
        // is computed directly as token_id * n_embd
        const params = extern struct { vocab_size_val: u32, n_embd_val: u32 }{
            .vocab_size_val = 0,
            .n_embd_val = @intCast(dim),
        };

        const bufs = [_]VkBuffer{ token_id_buf.buf, table_buf.buf, output_buf.buf };
        const sizes = [_]usize{ token_id_sz, table_sz, output_sz };
        const n_groups = (dim + workgroup_size - 1) / workgroup_size;

        self.dispatch(self.pipe_embedding, &bufs, &sizes, @ptrCast(&params), 8, @intCast(n_groups));

        // Download result
        self.downloadF32(output_buf.mem, output, dim);
    }

    /// Causal 1D convolution with SiLU activation for SSM models (Qwen3.5, Nemotron).
    /// Uses GPU shader to eliminate CPU fallback.
    pub fn causalConv1dSilu(
        self: *VulkanBackend,
        conv_out: [*]f32,
        conv_state: [*]f32,
        conv_in: [*]const f32,
        conv_w: [*]const f32,
        conv_b: ?[*]const f32,
        conv_ch: usize,
        d_conv: usize,
    ) void {
        // Note: This kernel does NOT handle conv_b (bias). The CPU reference
        // causalConv1dSilu in ops/ssm.zig supports optional bias, but the GPU
        // shader currently does not. For models that need bias, we fall back to CPU.
        if (conv_b != null) @panic("Vulkan conv1d: bias not supported — add GPU shader support for conv bias");

        // Buffer sizes
        const conv_ch_sz = conv_ch * @sizeOf(f32);
        const state_sz = (d_conv - 1) * conv_ch * @sizeOf(f32);
        const conv_w_sz = d_conv * conv_ch * @sizeOf(f32);

        // Create buffers for input, state, weights, output
        const input_buf = self.getPooledBuf(conv_ch_sz);
        defer self.releasePooledBuf(input_buf);
        const state_buf = self.getPooledBuf(state_sz);
        defer self.releasePooledBuf(state_buf);
        const output_buf = self.getPooledBuf(conv_ch_sz);
        defer self.releasePooledBuf(output_buf);

        // Get or upload cached buffer for conv_w (immutable weights)
        const conv_w_buf = self.getOrUpload(@ptrCast(conv_w), conv_w_sz);

        // Upload input and state
        self.uploadBuffer(input_buf.mem, @ptrCast(conv_in), conv_ch_sz);
        self.uploadBuffer(state_buf.mem, @ptrCast(conv_state), state_sz);

        // Push constants: conv_ch, d_conv
        const params = extern struct { conv_ch_val: u32, d_conv_val: u32 }{
            .conv_ch_val = @intCast(conv_ch),
            .d_conv_val = @intCast(d_conv),
        };

        const bufs = [_]VkBuffer{ input_buf.buf, state_buf.buf, conv_w_buf.buf, output_buf.buf };
        const sizes = [_]usize{ conv_ch_sz, state_sz, conv_w_sz, conv_ch_sz };
        const n_groups = (conv_ch + workgroup_size - 1) / workgroup_size;

        self.dispatch(self.pipe_conv1d, &bufs, &sizes, @ptrCast(&params), 8, @intCast(n_groups));

        // Download result
        self.downloadF32(output_buf.mem, conv_out, conv_ch);

        // Update ring buffer state on CPU (shift rows left, append new input)
        // This is a small memcpy operation and doesn't significantly impact performance
        const hist = d_conv - 1;
        if (hist > 1) {
            for (0..hist - 1) |p| {
                @memcpy(conv_state[p * conv_ch ..][0..conv_ch], conv_state[(p + 1) * conv_ch ..][0..conv_ch]);
            }
        }
        @memcpy(conv_state[(hist - 1) * conv_ch ..][0..conv_ch], conv_in[0..conv_ch]);
    }

    /// L2 normalize in-place. Single-dispatch fused kernel.
    pub fn l2Norm(self: *VulkanBackend, x: [*]f32, n: usize, eps: f32) void {
        const sz = n * @sizeOf(f32);
        const x_buf = self.getPooledBuf(sz);
        defer self.releasePooledBuf(x_buf);

        self.uploadBuffer(x_buf.mem, @ptrCast(x), sz);
        const params = extern struct { n_val: u32, eps_val: f32 }{ .n_val = @intCast(n), .eps_val = eps };
        const bufs = [_]VkBuffer{x_buf.buf};
        const sizes = [_]usize{sz};
        self.dispatch(self.pipe_l2_norm, &bufs, &sizes, @ptrCast(&params), 8, 1);
        self.downloadF32(x_buf.mem, x, n);
    }

    /// NVFP4 SafeTensors GEMV.
    pub fn gemvNvfp4St(_: *VulkanBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize) void {
        @panic("Vulkan NVFP4 SafeTensors GEMV: no GPU shader — add a Vulkan compute shader");
    }

    /// MLX affine quantized GEMV.
    pub fn gemvMlxQ(_: *VulkanBackend, _: [*]const f32, _: [*]const u8, _: [*]const u8, _: [*]const u8, _: [*]f32, _: usize, _: usize, _: u32) void {
        @panic("Vulkan MLX GEMV: no GPU shader — add a Vulkan compute shader");
    }

    /// In-place sigmoid-gated multiply.
    pub fn sigmoidMul(_: *VulkanBackend, _: [*]f32, _: [*]const f32, _: usize) void {
        @panic("Vulkan sigmoidMul: no GPU shader — add a Vulkan compute shader");
    }

    /// Fused SiLU + multiply.
    pub fn siluMul(_: *VulkanBackend, _: [*]const f32, _: [*]const f32, _: [*]f32, _: usize) void {
        @panic("Vulkan siluMul: no GPU shader — add a Vulkan compute shader");
    }

    /// In-place per-head rmsNorm.
    pub fn rmsNormMulti(_: *VulkanBackend, _: [*]f32, _: [*]const f32, _: usize, _: usize, _: f32) void {
        @panic("Vulkan rmsNormMulti: no GPU shader — add a Vulkan compute shader");
    }

    /// Deinterleave paired data into two separate output buffers.
    pub fn deinterleave(_: *VulkanBackend, _: [*]const f32, _: [*]f32, _: [*]f32, _: usize, _: usize) void {
        @panic("Vulkan deinterleave: no GPU shader — add a Vulkan compute shader");
    }

    /// Batched GEMV — sequential dispatch on Vulkan.
    pub fn gemvMulti(self: *VulkanBackend, x: [*]const f32, ops: []const backend_mod.GemvOp, k: usize) void {
        for (ops) |op| self.gemv(x, op.w, op.y, op.n, k);
    }

    /// Allocate a KV cache slice — plain allocator on Vulkan.
    pub fn allocKvSlice(_: *VulkanBackend, allocator: std.mem.Allocator, n: usize) error{OutOfMemory}![]u8 {
        return allocator.alloc(u8, n);
    }

    /// Free a KV cache slice allocated via allocKvSlice.
    pub fn freeKvSlice(_: *VulkanBackend, allocator: std.mem.Allocator, slice: []u8) void {
        allocator.free(slice);
    }

    /// Create Vulkan buffer wrapping RAM-tier KV block with zero copy.
    /// On UMA platforms, HOST_VISIBLE | DEVICE_LOCAL memory allows GPU to
    /// access RAM-tier blocks directly without separate device allocation.
    /// On discrete GPUs, performs one-time upload to cached device buffer.
    ///
    /// Returns cached buffer if already created for this pointer.
    pub fn createKvBuffer(self: *VulkanBackend, host_ptr: [*]u8, size: usize) !VkBuffer {
        // Check cache
        const addr = @intFromPtr(host_ptr);
        if (self.buf_cache.get(addr)) |cached| return cached.vk_buf.buf;

        // Create buffer info
        const buffer_info = VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        var buffer: VkBuffer = null;
        const result = self.vkCreateBuffer(self.device, &buffer_info, null, &buffer);
        if (result != VK_SUCCESS) return error.VulkanBufferCreationFailed;

        // Get memory requirements
        var mem_reqs: VkMemoryRequirements = undefined;
        self.vkGetBufferMemoryRequirements(self.device, buffer, &mem_reqs);

        // Allocate HOST_VISIBLE | DEVICE_LOCAL memory (UMA-friendly)
        const mem_type_index = self.findMemoryType(
            mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );

        const alloc_info = VkMemoryAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type_index,
        };

        var memory: VkDeviceMemory = null;
        const alloc_result = self.vkAllocateMemory(self.device, &alloc_info, null, &memory);
        if (alloc_result != VK_SUCCESS) return error.VulkanMemoryAllocationFailed;

        // Bind buffer to memory
        _ = self.vkBindBufferMemory(self.device, buffer, memory, 0);

        // Map and copy host data
        var mapped_ptr: ?*anyopaque = null;
        _ = self.vkMapMemory(self.device, memory, 0, size, 0, &mapped_ptr);
        @memcpy(@as([*]u8, @ptrCast(mapped_ptr.?))[0..size], host_ptr[0..size]);
        self.vkUnmapMemory(self.device, memory);

        // Cache buffer
        try self.buf_cache.put(addr, .{ .vk_buf = .{ .buf = buffer, .mem = memory }, .size = size });

        std.log.debug("Created Vulkan KV buffer for RAM-tier block at {x}", .{addr});
        return buffer.?;
    }

    /// DeltaNet SSM recurrence.
    pub fn deltaNet(_: *VulkanBackend, _: [*]const f32, _: [*]f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]f32, _: [*]f32, _: []f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: [*]const f32, _: backend_mod.DeltaNetParams) void {
        @panic("Vulkan DeltaNet: no GPU shader — add a Vulkan compute shader");
    }

    /// No-op — each Vulkan dispatch already submits and waits on a fence.
    /// Exists for API consistency with other backends.
    pub fn sync(_: *VulkanBackend) void {}
    /// No-op — Vulkan dispatches are not batched.
    pub fn beginBatch(_: *VulkanBackend) void {}
    /// No-op — Vulkan dispatches are not batched.
    pub fn endBatch(_: *VulkanBackend) void {}

    /// Returns backend startup information for display.
    pub fn backendInfo(self: *const VulkanBackend) @import("backend.zig").BackendInfo {
        return .{
            .name = "Vulkan",
            .device_name = self.device_name[0..self.device_name_len],
            .lib_name = vk_lib_name,
            .n_gpu_kernels = n_pipelines,
            .kernel_type = "SPIR-V",
            .total_mem = self.total_vram,
            .driver_version = std.mem.sliceTo(&self.vk_ver_str, 0),
        };
    }

    /// Fused Scaled Dot-Product Attention.
    /// One workgroup per query head. Falls back to CPU for large seq_len / head_dim.
    pub fn sdpa(self: *VulkanBackend, q: [*]const f32, keys: []u8, values: []u8, k_new: [*]const f32, v_new: [*]const f32, output: [*]f32, nh: usize, nkv: usize, hd: usize, seq_len: usize, scale: f32, kv_type: @import("backend.zig").KvQuantType) void {
        if (kv_type != .f32) @panic("Vulkan SDPA: quantized KV not supported — add GPU shader or use --kv-type f32");

        const f32_keys: []f32 = @as([*]f32, @ptrCast(@alignCast(keys.ptr)))[0 .. keys.len / 4];
        const f32_values: []f32 = @as([*]f32, @ptrCast(@alignCast(values.ptr)))[0 .. values.len / 4];

        const kvd = nkv * hd;
        const sl = seq_len + 1;

        if (sl > sdpa_max_seq_len or hd > sdpa_max_head_dim) @panic("Vulkan SDPA: sequence or head dim exceeds GPU limit — reduce --ctx-size");

        // Flush pending GPU work so k_new/v_new are readable, then append to KV cache
        self.sync();
        @memcpy(f32_keys[seq_len * kvd ..][0..kvd], k_new[0..kvd]);
        @memcpy(f32_values[seq_len * kvd ..][0..kvd], v_new[0..kvd]);

        const q_sz = nh * hd * @sizeOf(f32);
        const k_sz = f32_keys.len * @sizeOf(f32);
        const v_sz = f32_values.len * @sizeOf(f32);
        const o_sz = nh * hd * @sizeOf(f32);

        const q_buf = self.getPooledBuf(q_sz);
        defer self.releasePooledBuf(q_buf);
        const k_buf = self.getPooledBuf(k_sz);
        defer self.releasePooledBuf(k_buf);
        const v_buf = self.getPooledBuf(v_sz);
        defer self.releasePooledBuf(v_buf);
        const o_buf = self.getPooledBuf(o_sz);
        defer self.releasePooledBuf(o_buf);

        self.uploadBuffer(q_buf.mem, @ptrCast(q), q_sz);
        self.uploadBuffer(k_buf.mem, @ptrCast(f32_keys.ptr), k_sz);
        self.uploadBuffer(v_buf.mem, @ptrCast(f32_values.ptr), v_sz);

        const params = extern struct { nh: u32, nkv: u32, hd: u32, sl: u32, scale: f32 }{
            .nh = @intCast(nh),
            .nkv = @intCast(nkv),
            .hd = @intCast(hd),
            .sl = @intCast(sl),
            .scale = scale,
        };
        const bufs = [_]VkBuffer{ q_buf.buf, k_buf.buf, v_buf.buf, o_buf.buf };
        const sizes = [_]usize{ q_sz, k_sz, v_sz, o_sz };
        self.dispatch(self.pipe_sdpa, &bufs, &sizes, @ptrCast(&params), 20, @intCast(nh));
        self.downloadF32(o_buf.mem, output, nh * hd);
    }
};

// ── Tests ─────────────────────────────────────────────────────────

test "VulkanBackend init and silu" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    vk_be.silu(&input, &output, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.731), output[1], 0.02);
}

test "VulkanBackend gelu" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    var input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    vk_be.gelu(&input, &output, 4);
    // GELU(0) = 0, GELU(1) ≈ 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), output[1], 0.02);
}

test "VulkanBackend rmsNorm" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output: [4]f32 = undefined;
    vk_be.rmsNorm(&input, &weight, &output, 4, 1e-6);
    // RMS = sqrt((1+4+9+16)/4) ≈ 2.7386, output[0] ≈ 0.365
    try std.testing.expectApproxEqAbs(@as(f32, 0.365), output[0], 0.05);
}

test "VulkanBackend softmax" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    vk_be.softmax(&data, 4);
    // softmax([1,2,3,4]) ≈ [0.0321, 0.0871, 0.2369, 0.6439]
    var sum: f32 = 0;
    for (&data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);
    try std.testing.expect(data[3] > data[2]);
    try std.testing.expect(data[2] > data[1]);
}

test "VulkanBackend l2Norm" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    var data = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    vk_be.l2Norm(&data, 4, 1e-6);
    // L2 = 5, normalized: [0.6, 0.8, 0, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), data[1], 0.01);
}

test "VulkanBackend rope" {
    var vk_be = VulkanBackend.init(std.testing.allocator) catch {
        std.log.warn("Vulkan not available — skipping test", .{});
        return;
    };
    defer vk_be.deinit();

    // 1 head, head_dim=4, rope_dim=4, pos=1
    // Input: pair0=(1.0, 0.0), pair1=(0.0, 1.0) in split-complex layout
    var x = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    vk_be.rope(&x, 1, 1, 4, 4, 10000.0);
    // RoPE is a rotation — magnitude of each pair must be preserved
    const mag0 = @sqrt(x[0] * x[0] + x[2] * x[2]);
    const mag1 = @sqrt(x[1] * x[1] + x[3] * x[3]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mag0, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mag1, 0.01);
}
