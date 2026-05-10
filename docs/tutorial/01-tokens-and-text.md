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

### Tokenizer Strategies

| Strategy | Algorithm | Models | Merge Table |
|----------|-----------|--------|-------------|
| **BPE** | Iterative pair merging | Qwen, GPT | Required — learned merge priority list |
| **SPM** | Greedy longest-match | Gemma | Not needed — matches vocabulary entries directly |

Agave's tokenizer (`src/tokenizer/bpe.zig`) supports both:
- **BPE mode** — uses merge rules to decide which byte pairs to combine first
- **SPM mode** — greedy longest-match without merges, including a "no dummy" variant for Gemma

### Token Statistics

| Model | Vocab Size | Avg Tokens/Word | Encoding |
|-------|:----------:|:---------------:|----------|
| Qwen 3.5 | 151,936 | ~1.3 | BPE (GPT-2 style, byte-level) |
| Gemma 3/4 | 262,144 | ~1.1 | SentencePiece |
| GPT-OSS | 200,064 | ~1.3 | BPE |
| GLM-4 | 151,552 | ~1.3 | BPE |

Larger vocabularies encode text more efficiently (fewer tokens per sentence) but increase the embedding table size — a 256K vocabulary with 4096-dim embeddings takes 4 GB in FP32.

### Special Tokens

Every vocabulary includes reserved tokens with structural meaning:

| Token | Purpose | Typical ID |
|-------|---------|:----------:|
| `<bos>` | Beginning of sequence — signals the model to start generating | 1 |
| `<eos>` | End of sequence — signals generation should stop | 2 |
| `<pad>` | Padding — fills unused positions in batched inputs | 0 |
| `<|im_start|>` | Chat role marker (Qwen) — marks the beginning of a message | varies |
| `<start_of_turn>` | Chat role marker (Gemma) | varies |

The tokenizer tracks these IDs for chat template formatting and end-of-generation detection. Some models define additional **end-of-generation (EOG)** tokens beyond `<eos>` — for example, Qwen uses `<|endoftext|>` and `<|im_end|>`.

### Byte-Level BPE Encoding

Qwen/GPT-style tokenizers use **byte-level** encoding where every possible byte (0x00–0xFF) maps to a printable Unicode character. For example, a space (0x20) is represented as `Ġ` (U+0120). This means:
- Every text can be tokenized (no unknown characters)
- Token text looks odd in raw form: `"Ġhello"` = `" hello"` (space prefix)
- Grammar-constrained decoding must strip these prefixes via `getEffectiveText()`

## Embedding Lookup

The first operation in the forward pass converts a token ID into a **vector** (a 1D array of numbers). The model has an **embedding table** — a **matrix** (a 2D array) of shape `[vocab_size × n_embd]` where `vocab_size` is the total number of tokens in the vocabulary (e.g., 128K) and `n_embd` is the **embedding dimension** (the size/length of each vector — how many numbers it contains, typically 1024–8192 floating-point numbers). Each row is the learned representation of one token.

**Note on terminology:** Machine learning uses the term **tensor** for multi-dimensional arrays — a **scalar** (single number, 0D), vector (1D), matrix (2D), or higher-dimensional array (3D, 4D, etc.) are all tensors. Throughout this tutorial we use the more specific terms (scalar/vector/matrix) since nearly all operations are 0D, 1D, or 2D, but you'll see "tensor" in the code and documentation referring to these same arrays.

Embedding lookup is just a table read: take row `token_id` from the matrix.

**Why vectors, not just integer IDs?** Because vectors let the model represent *relationships*. During training, words that appear in similar contexts end up with similar vectors — "cat" and "dog" are close together, while "cat" and "database" are far apart. The distance and direction between vectors encode meaning. Integer IDs can't do this — ID 4517 and ID 4518 have no meaningful relationship. Vectors give the model a continuous space where it can generalize: if it knows about "cat" and "dog", it can infer things about "kitten" because its vector is nearby.

Embedding lookup is so simple that CPU memcpy is faster than GPU **dispatch** overhead (the cost of sending work to the GPU and synchronizing), which is why all backends run this on the CPU.

The table may be **quantized** (compressed to lower **precision** — fewer bits per number, less accurate — formats like Q4_0 or BF16 to save memory) — the implementation **dequantizes** (converts back to full precision) on the fly during the lookup. Gemma3 scales embeddings by `sqrt(n_embd)` after lookup, **amplifying the signal** (making the values larger to increase their influence) for its architecture.

## Vocabulary Projection

At the end of the forward pass, we need to go back from a vector to token probabilities. This is a matrix multiply: `logits = W_output @ hidden`, where **hidden** is the output vector from the final layer and **logits** are the raw scores (unnormalized probabilities) — one score per vocabulary token.

This is the **largest single GEMV** (matrix-vector multiply — multiplying a weight matrix by a single hidden state vector) in the model — for a 128K-token vocabulary, it's 128K output rows. For models with **tied embeddings** (Gemma3), the output weight matrix is the same as the embedding table (reusing the same parameters for both input and output), saving memory.

After projection, **argmax** (the operation that finds the index of the maximum value) over the logits gives the predicted next token ID.

## The Generation Loop

Text generation is **autoregressive** — each generated token becomes the input for the next step:

```
tokens = tokenize("The capital of France is")
for each token in tokens:
    logits = forward(token)           // process prompt tokens one at a time

while not done:
    next_token = argmax(logits)       // pick highest-scoring token
    if next_token == EOS: break       // stop at end-of-sequence
    print(decode(next_token))         // output: " Paris"
    logits = forward(next_token)      // feed output back as input
```

This is why inference speed matters — generating 100 tokens requires 100 sequential forward passes through the entire model. Each pass is dominated by GEMV (matrix-vector multiply), which is memory-bandwidth bound. The rest of this tutorial series explains every component of that forward pass and how Agave optimizes it.

---

**In the code:** [src/tokenizer/bpe.zig](../../src/tokenizer/bpe.zig) (tokenizer), [src/backend/kernels/cpu/embedding.zig](../../src/backend/kernels/cpu/embedding.zig) (embedding lookup), [src/ops/math.zig](../../src/ops/math.zig) (argmax, sampleToken)

**Next:** [Chapter 2: The Transformer →](02-the-transformer.md) | **Product docs:** [Architecture](../ARCHITECTURE.md)
