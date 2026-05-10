//! Fused FFN kernel: gate GEMV + up GEMV + SiLU*mul for Q6_K weights.
//! Placeholder — Q6_K block dot is complex (6-bit with split ql/qh arrays).
//! Cross-file import causes LLVM aliasee error on nvptx64.
//! TODO: inline q6kBlockDot here when LLVM issue is resolved.

// Empty file — no exported kernel. CUDA backend gracefully falls back
// to CPU for Q6_K fused FFN via `catch null` on function lookup.
