# Chapter 1: Tokens and Text

Language models don't see text — they see **tokens**, which are integer IDs representing subword pieces (fragments like "Hello" → "He" + "llo" that are smaller than words but larger than individual characters). Before anything else happens, we need to convert text to numbers and back.

## What is Inference?

**Training** teaches a model by adjusting billions of **weights** (learned parameters — the numbers in matrices and vectors that encode the model's knowledge) over trillions of tokens. **Inference** uses those frozen weights to generate new text. Agave only does inference — it loads pre-trained weights and runs the model forward (a single pass through the network layers to produce output).

## Tokenization

The tokenizer converts between text and token IDs:

```
"Hello, world!" → [15496, 11, 995, 0]     (encode)
[15496, 11, 995, 0] → "Hello, world!"     (decode)
```

**BPE (Byte Pair Encoding)** is the most common algorithm. It works by iteratively merging the most frequent pair of adjacent symbols:

1. Start with individual bytes: `H e l l o`
2. Most frequent pair is `l l` → merge to `ll`: `H e ll o`
3. Next most frequent is `H e` → merge to `He`: `He ll o`
4. Continue until vocabulary is built: `Hello`

The merge rules are learned during training and stored alongside the model. Agave's BPE tokenizer (`src/tokenizer/bpe.zig`) supports two modes:
- **BPE mode** — uses merge rules (Qwen, GPT)
- **SPM mode** — greedy longest-match without merges (Gemma)

## Embedding Lookup

The first operation in the forward pass converts a token ID into a vector. The model has an **embedding table** — a matrix of shape `[vocab_size × n_embd]` where `n_embd` is the embedding dimension (typically 1024–8192 floating-point numbers) and each row is the learned representation of one token.

Embedding lookup is just a table read: take row `token_id` from the matrix. It's so simple that CPU memcpy is faster than GPU dispatch overhead, which is why all backends run this on the CPU.

The table may be **quantized** (compressed to lower precision formats like Q4_0 or BF16 to save memory) — the implementation **dequantizes** (converts back to full precision) on the fly during the lookup. Gemma3 scales embeddings by `sqrt(n_embd)` after lookup, amplifying the signal for its architecture.

## Vocabulary Projection

At the end of the forward pass, we need to go back from a vector to token probabilities. This is a matrix multiply: `logits = W_output @ hidden`, where **hidden** is the output vector from the final layer and **logits** are the raw scores (unnormalized probabilities) — one score per vocabulary token.

This is the **largest single GEMV** (matrix-vector multiply — multiplying a weight matrix by a single hidden state vector) in the model — for a 128K-token vocabulary, it's 128K output rows. For models with **tied embeddings** (Gemma3), the output weight matrix is the same as the embedding table (reusing the same parameters for both input and output), saving memory.

After projection, **argmax** (the operation that finds the index of the maximum value) over the logits gives the predicted next token ID.

---

**In the code:** `src/tokenizer/bpe.zig` (tokenizer), `src/backend/kernels/cpu/embedding.zig` (embedding lookup), `src/ops/math.zig` (finalLogits, argmax)

**Next:** [Chapter 2: The Transformer →](02-the-transformer.md)
