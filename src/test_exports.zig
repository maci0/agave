//! Re-exports for test modules outside src/.
//! Test files in tests/ cannot import src/ files directly (Zig 0.16 module
//! boundary rules). This bridge module, rooted inside src/, provides named
//! access to the types that SDPA correctness tests need.

const backend = @import("backend/backend.zig");

/// Re-exported Backend tagged union for cross-module test imports.
pub const Backend = backend.Backend;
/// Re-exported CpuBackend for test access.
pub const CpuBackend = backend.CpuBackend;
/// Re-exported CudaBackend for test access.
pub const CudaBackend = backend.CudaBackend;
/// Re-exported MetalBackend for test access.
pub const MetalBackend = backend.MetalBackend;
/// Re-exported VulkanBackend for test access.
pub const VulkanBackend = backend.VulkanBackend;
/// Re-exported RocmBackend for test access.
pub const RocmBackend = backend.RocmBackend;
/// Re-exported WebGpuBackend for test access.
pub const WebGpuBackend = backend.WebGpuBackend;
