# Mixed-Quantization Expert Splicing

Create mixed-quantization GGUF files where selected layers use higher-precision
routed experts while the rest of the model stays at a smaller quantization.

## Why

MoE models (DeepSeek V4, Qwen 3.5 35B, GPT-OSS, Llama 4) route tokens to a
small subset of experts per layer. Upgrading only those experts from e.g.
IQ2_XXS to Q4_K improves output quality with minimal size increase, since
non-expert tensors (shared experts, projections, routing weights) remain at the
base quantization.

## Usage

```bash
# Splice Q4_K experts into layers 37-42 of an IQ2 base model
python3 splice_mixed_experts.py \
    --base model-iq2.gguf \
    --donor model-q4.gguf \
    --layers 37-42 \
    --out model-mixed.gguf

# Preview which tensors would be replaced (no file written)
python3 splice_mixed_experts.py \
    --base model-iq2.gguf \
    --donor model-q4.gguf \
    --layers 0-2,40-42 \
    --out model-mixed.gguf \
    --dry-run
```

## How it works

1. Parses both GGUF files to extract tensor metadata and data offsets.
2. For each tensor in the specified layers, checks if the name matches the
   routed-expert pattern (`blk.N.ffn_*.weight` where the tensor belongs to
   an expert slot).
3. Copies matching tensors from the donor file; all other tensors come from
   the base file.
4. Writes a new GGUF with merged metadata and tensor data.

## Attribution

Based on the mixed-quant splicing tool from [antirez/ds4](https://github.com/antirez/ds4).
