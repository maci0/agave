# Chapter 1: Tokens and Text

Language models don't see text — they see **tokens**, which are integer IDs representing subword pieces (fragments like "Hello" → "He" + "llo" that are smaller than words but larger than individual characters). Before anything else happens, we need to convert text to numbers and back.

## What is Inference?

**Training** teaches a model by adjusting billions of **weights** (learned parameters — the numbers in matrices and vectors that encode the model's knowledge) over trillions of tokens. **Inference** uses those **frozen** (fixed, no longer changing) weights to generate new text. Agave only does inference — it loads **pre-trained** (already trained by someone else, ready to use) weights and runs the model forward (a single pass through the network layers to produce output).

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

The merge rules are learned during training and stored alongside the model. This process creates the **vocabulary** — the complete set of all possible tokens the model knows about. Each token gets a unique ID (0 to vocab_size-1). For example:

```
Token ID 0:    "<pad>" (padding — fills empty space when batching multiple sequences of different lengths)
Token ID 1:    "<s>" (start of sequence)
Token ID 15496: "Hello"
Token ID 11:    ","
Token ID 128000: (last valid token)
```

The **vocabulary size** (vocab_size) is the total number of distinct tokens. Modern models have vocabularies of 32K–256K tokens. Larger vocabularies encode text more efficiently (fewer tokens per sentence) but increase memory and compute costs.

Agave's BPE tokenizer (`src/tokenizer/bpe.zig`) supports two modes:
- **BPE mode** — uses merge rules (Qwen, GPT)
- **SPM mode** — **greedy** (always picks the best option at each step without backtracking) longest-match without merges (Gemma)

## Embedding Lookup

The first operation in the forward pass converts a token ID into a **vector** (a 1D array of numbers). The model has an **embedding table** — a **matrix** (a 2D array) of shape `[vocab_size × n_embd]` where `vocab_size` is the total number of tokens in the vocabulary (e.g., 128K) and `n_embd` is the **embedding dimension** (the size/length of each vector — how many numbers it contains, typically 1024–8192 floating-point numbers). Each row is the learned representation of one token.

**Note on terminology:** Machine learning uses the term **tensor** for multi-dimensional arrays — a **scalar** (single number, 0D), vector (1D), matrix (2D), or higher-dimensional array (3D, 4D, etc.) are all tensors. Throughout this tutorial we use the more specific terms (scalar/vector/matrix) since nearly all operations are 0D, 1D, or 2D, but you'll see "tensor" in the code and documentation referring to these same arrays.

Embedding lookup is just a table read: take row `token_id` from the matrix. It's so simple that CPU memcpy is faster than GPU **dispatch** overhead (the cost of sending work to the GPU and synchronizing), which is why all backends run this on the CPU.

The table may be **quantized** (compressed to lower **precision** — fewer bits per number, less accurate — formats like Q4_0 or BF16 to save memory) — the implementation **dequantizes** (converts back to full precision) on the fly during the lookup. Gemma3 scales embeddings by `sqrt(n_embd)` after lookup, **amplifying the signal** (making the values larger to increase their influence) for its architecture.

## Vocabulary Projection

At the end of the forward pass, we need to go back from a vector to token probabilities. This is a matrix multiply: `logits = W_output @ hidden`, where **hidden** is the output vector from the final layer and **logits** are the raw scores (unnormalized probabilities) — one score per vocabulary token.

This is the **largest single GEMV** (matrix-vector multiply — multiplying a weight matrix by a single hidden state vector) in the model — for a 128K-token vocabulary, it's 128K output rows. For models with **tied embeddings** (Gemma3), the output weight matrix is the same as the embedding table (reusing the same parameters for both input and output), saving memory.

After projection, **argmax** (the operation that finds the index of the maximum value) over the logits gives the predicted next token ID.

---

**In the code:** `src/tokenizer/bpe.zig` (tokenizer), `src/backend/kernels/cpu/embedding.zig` (embedding lookup), `src/ops/math.zig` (finalLogits, argmax)

**Next:** [Chapter 2: The Transformer →](02-the-transformer.md)
