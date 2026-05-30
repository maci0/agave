# Progress

## Tutorial Audit: FFN/MoE (Ch3) + Quantization (Ch4) — COMPLETE

Output: `notes/tutorial-audit-ffn-quant.md`

### Summary
- **17/18 claims MATCH** the source code exactly
- **1 claim UNCERTAIN**: RotorQuant FMA count (~2,400 tutorial vs ~720 apparent in sparse implementation)
- All MoE expert counts, activation functions, quantization block sizes, and KV quant formats verified
- Factored dequantization optimization confirmed in mlx.zig with exact pattern match
