//! Re-exports for test modules outside src/.
//! Test files in tests/ cannot import src/ files directly (Zig 0.15 module
//! boundary rules). This bridge module, rooted inside src/, provides named
//! access to the types that SDPA correctness tests need.

const backend = @import("backend/backend.zig");

pub const Backend = backend.Backend;
pub const CpuBackend = backend.CpuBackend;
pub const CudaBackend = backend.CudaBackend;
pub const MetalBackend = backend.MetalBackend;
pub const VulkanBackend = backend.VulkanBackend;
pub const RocmBackend = backend.RocmBackend;
