//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q5_K weights.
//! Placeholder — Q5_K block dot is complex (5-bit + high-bit array).
//! Cross-file import causes LLVM aliasee error on nvptx64.
//! TODO: inline q5kBlockDot here when LLVM issue is resolved.

// Empty file — no exported kernel. CUDA backend gracefully falls back
// to CPU for Q5_K fused FFN via `catch null` on function lookup.
