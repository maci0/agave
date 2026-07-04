# Glossary & Term Analysis — Chapters 1–7

---

## Chapter 1: Tokens and Text

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **tokens** | "they see **tokens**, which are integer IDs representing subword pieces" — opening paragraph |
| **weights** | "(learned parameters — the numbers in matrices and vectors that encode the model's knowledge)" — What is Inference section |
| **Inference** | "uses those frozen weights to generate new text" — What is Inference section |
| **frozen** | "(fixed, no longer changing)" — What is Inference section |
| **pre-trained** | "(already trained by someone else, ready to use)" — What is Inference section |
| **forward** | "(a single pass through the network layers to produce output)" — What is Inference section |
| **BPE (Byte Pair Encoding)** | "is the most common algorithm. It works by iteratively merging the most frequent pair of adjacent symbols" — Tokenization section |
| **vocabulary** | "the complete set of all possible tokens the model knows about" — after BPE explanation |
| **vocabulary size** | "the total number of distinct tokens" — after vocabulary definition |
| **SPM** | "Greedy longest-match" — Tokenizer Strategies table (brief, in table) |
| **EOG (end-of-generation)** | "end-of-generation (EOG) tokens beyond `<eos>`" — Special Tokens section |
| **vector** | "(a 1D array of numbers)" — Embedding Lookup section |
| **embedding** | "(a learned numerical representation — a fixed-size array of floats that encodes the token's meaning)" — Embedding Lookup section |
| **matrix** | "(a 2D array)" — Embedding Lookup section |
| **embedding dimension** | "(the size/length of each vector — how many numbers it contains)" — Embedding Lookup section |
| **tensor** | "for multi-dimensional arrays — a scalar (single number, 0D), vector (1D), matrix (2D), or higher-dimensional array" — Note on terminology |
| **scalar** | "(single number, 0D)" — Note on terminology |
| **dispatch** | "(the cost of sending work to the GPU and synchronizing)" — after embedding lookup |
| **quantized** | "(compressed to lower precision formats)" — embedding quantization note |
| **precision** | "(fewer bits per number, less accurate)" — with quantized |
| **dequantizes** | "(converts back to full precision)" — embedding quantization note |
| **amplifying the signal** | "(making the values larger to increase their influence)" — Gemma3 note |
| **logits** | "raw scores (unnormalized probabilities) — one score per vocabulary token" — Vocabulary Projection |
| **GEMV** | "(matrix-vector multiply — multiplying a weight matrix by a single hidden state vector)" — Vocabulary Projection |
| **tied embeddings** | "(Gemma3), the output weight matrix is the same as the embedding table" — Vocabulary Projection |
| **argmax** | "(the operation that finds the index of the maximum value)" — Vocabulary Projection |
| **autoregressive** | "each generated token becomes the input for the next step" — Generation Loop |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **FP32** | "Larger vocabularies encode text more efficiently...takes 4 GB in FP32" — Token Statistics section | Not explained; means 32-bit floating point. Related "bf16" also appears in quantization note without definition. |
| **BF16** | "formats like Q4_0 or BF16 to save memory" — quantized weights note | Not defined (brain-float 16-bit). |
| **Q4_0** | "formats like Q4_0 or BF16 to save memory" — quantized weights note | Not defined; naming convention for quantization formats not explained. |
| **n_embd** | Embedding table shape `[vocab_size × n_embd]` | Used as a variable name without explaining it's a hyperparameter name meaning "embedding dimension." Partially covered by surrounding text but not the variable name itself. |
| **GPT** | Tokenizer Strategies table | Not expanded (Generative Pre-trained Transformer). Used as a known model family name. |
| **Qwen** | Tokenizer Strategies table | Not explained; a model family. |
| **Gemma** | Tokenizer Strategies table | Not explained; a model family. |
| **GGUF** | Not in Ch1 but Q4_0 references the format implicitly | — |
| **EOS** | Code sample: `if next_token == EOS: break` | Used as bare acronym in code. `<eos>` is explained in the Special Tokens table but "EOS" as a code identifier is not explicitly tied back. Minor. |
| **GPU** | "CPU memcpy is faster than GPU dispatch overhead" | Not expanded; assumed known. |
| **CPU** | Same sentence | Not expanded; assumed known. |
| **UMA** | Not in Ch1 | — |
| **byte-level** | "byte-level encoding where every possible byte (0x00–0xFF) maps to a printable Unicode character" | The concept is explained via example but the term "byte-level" itself is not explicitly defined as a tokenizer type. Borderline — context makes it clear. |

### Glossary — Chapter 1

| Term | Definition |
|------|-----------|
| token | An integer ID representing a subword piece of text — the basic unit language models operate on. |
| tokenizer | A component that converts between text strings and sequences of token IDs. |
| BPE (Byte Pair Encoding) | A tokenization algorithm that iteratively merges the most frequent adjacent byte/character pairs to build a vocabulary. |
| SPM (SentencePiece) | A tokenization algorithm that uses greedy longest-match against a vocabulary, without requiring a merge table. |
| vocabulary | The complete set of all tokens a model can recognize, each mapped to a unique integer ID. |
| vocab_size | The total number of distinct tokens in the vocabulary (e.g., 32K–256K). |
| embedding | A learned fixed-size float vector that encodes a token's meaning; looked up from a table by token ID. |
| embedding dimension (n_embd) | The number of floats in each embedding vector (typically 1024–8192). |
| embedding table | A matrix of shape [vocab_size × n_embd] where each row is one token's embedding vector. |
| vector | A one-dimensional array of floating-point numbers. |
| matrix | A two-dimensional array of numbers. |
| tensor | A general multi-dimensional array (scalar=0D, vector=1D, matrix=2D, etc.). |
| inference | Using a trained model's frozen weights to generate new output, as opposed to training. |
| forward pass | A single pass of input data through all model layers to produce output. |
| weights | The learned numerical parameters (matrix/vector values) that encode a model's knowledge. |
| frozen weights | Model parameters that are fixed and no longer updated (inference-only). |
| pre-trained | A model whose weights have already been learned through training. |
| autoregressive | A generation mode where each output token is fed back as input to produce the next token. |
| logits | Raw, unnormalized scores output by the model — one per vocabulary token — before conversion to probabilities. |
| argmax | The operation that selects the index of the highest value in an array. |
| GEMV (General Matrix-Vector multiply) | Multiplying a weight matrix by a single input vector to produce one output vector. |
| quantized | Compressed to a lower-precision numerical format to save memory. |
| dequantize | Convert quantized (compressed) values back to full-precision floats. |
| precision | The number of bits used to represent a number; fewer bits = lower precision. |
| FP32 | 32-bit floating-point format; 4 bytes per value, full precision. |
| BF16 (bfloat16) | 16-bit brain floating-point format; same exponent range as FP32 but fewer mantissa bits. |
| tied embeddings | When input embedding and output projection share the same weight matrix, saving memory. |
| dispatch overhead | The cost of sending work to an accelerator (GPU) and synchronizing, which can exceed the compute cost for small operations. |
| special tokens | Reserved tokens with structural meaning (e.g., BOS, EOS, PAD) rather than textual content. |
| BOS (Beginning of Sequence) | A special token signaling the start of a sequence. |
| EOS (End of Sequence) | A special token signaling generation should stop. |
| byte-level BPE | A BPE variant where every possible byte maps to a printable character, ensuring any text can be tokenized with no unknown tokens. |

---

## Chapter 2: The Transformer

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **hidden state** | "(the internal vector representation flowing through each layer)" — opening |
| **transformer layer** | "has two sublayers: 1. Attention... 2. FFN" — structure description |
| **Attention** | "lets the model look at previous tokens" — sublayer list |
| **FFN (Feed-Forward Network)** | "processes each position independently" — sublayer list |
| **residual connections** | "`output = input + sublayer(input)` so information flows through unchanged" |
| **vanishing gradient problem** | "(where gradients get exponentially smaller in deep networks during training, making learning impossible)" |
| **linear projections** | "(matrix-vector multiplies)" — Q/K/V explanation |
| **softmax** | "normalization (converts raw scores into probabilities that sum to 1.0)" |
| **causal mask** | "lower-triangular attention matrix...token at position i must only attend to positions ≤ i" |
| **heads** | "(independent attention mechanisms, each focusing on different aspects of the input)" |
| **in parallel** | "(all heads compute simultaneously, not one after another)" |
| **GQA (Grouped Query Attention)** | "reduces memory by sharing K/V heads across multiple Q heads" — with paper citation |
| **MLA (Multi-head Latent Attention)** | "compresses K/V into a low-rank latent space before caching" — with paper citation |
| **low-rank latent space** | "(a smaller intermediate representation with fewer dimensions)" |
| **kernel** (compute) | "(a single computational function that runs on the CPU or GPU)" — SDPA section |
| **tiles** | "(small rectangular blocks of the attention matrix processed one at a time)" — FlashAttention |
| **online softmax** | "(incrementally updating the softmax result as new tiles arrive, avoiding the need to store all scores at once)" |
| **materializing** | "(allocating memory for and storing)" — FlashAttention |
| **SIMD-vectorized** | "(using Single Instruction Multiple Data — processing multiple values at once with one CPU instruction)" |
| **fallback** | "(alternative implementation used when the primary method isn't available)" |
| **magnitude** | "(the size/scale of the values — how large the numbers are)" — QK Norm |
| **alternation** | "(switching back and forth between limited and full attention across layers)" — Sliding Window |
| **scalar** | "(single number, not a vector)" — Attention Sinks |
| **prepended** | "(added to the beginning)" — Attention Sinks |
| **over-concentration** | "(too much attention weight)" — Attention Sinks |
| **element-wise** | "(applied independently to each element, not as a matrix operation)" — Sigmoid Gate |
| **residual stream** | "(the main path through the model where outputs accumulate via residual connections)" — Sigmoid Gate |
| **soft-clamps** | "(gently constrains via a smooth curve, unlike hard clamping which abruptly cuts off)" — Logit Softcapping |
| **preserving relative ordering** | "(keeping the same rank order)" — Logit Softcapping |
| **position-agnostic** | "(they don't know the order of tokens)" — RoPE intro |
| **rotation matrices** | "(mathematical transformations that rotate vectors by an angle without changing their length)" |
| **wavelengths** | "(cycles per distance — like how light has different wavelengths for different colors)" |
| **context** | "(context = maximum sequence length the model can process)" — theta table |
| **LayerNorm** | "(an older normalization method that also subtracts the mean)" — RMSNorm section |
| **pre-norm** | "(normalizing the input to each sublayer)" |
| **post-norms** | "(normalizing the output after the sublayer)" |
| **learnable weights** | "(parameters that the model adjusts during training)" — L2 Norm section |
| **DeltaNet** | "(a linear-complexity alternative to attention covered in Chapter 6)" |
| **decode** | "(one token at a time)" — GEMV vs GEMM |
| **GEMM** | "(General Matrix-Matrix multiply)" — GEMV vs GEMM |
| **arithmetic intensity** | "(compute-to-memory ratio)" — GEMV vs GEMM |
| **prefill** | "(processing the entire prompt)" — GEMV vs GEMM |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **RMSNorm** | Diagram label "RMSNorm" — first transformer layer diagram | Appears as a diagram label before being explained in the RMSNorm section later. The formula and explanation come later in the chapter. |
| **SDPA** | Section heading "SDPA (Scaled Dot-Product Attention)" | The acronym is expanded in the heading but not really explained before being used — however, the heading does expand it. Borderline. |
| **FlashAttention** / **FlashAttention-2** | FlashAttention section | Explained by description but the name itself is just a paper reference, which is fine. |
| **RoPE** | Section heading "RoPE (Rotary Position Encoding)" | Expanded in heading. OK. |
| **Q, K, V** | First appear in diagram labels before full explanation | The diagram shows "Q/K/V + SDPA" before the "What are Q, K, V?" paragraph explains them. Minor — immediate explanation follows. |
| **O(n²)** | "This is O(n²) in sequence length" | Big-O notation not explained; assumed known for systems programmers. Acceptable. |
| **HBM** | FlashAttention diagram: "written to HBM" | Not expanded. Means "High Bandwidth Memory" — the off-chip DRAM on GPUs. |
| **SRAM** | FlashAttention diagram: "scores never leave on-chip SRAM" | Not expanded. Means "Static Random-Access Memory" — fast on-chip cache. |
| **DRAM** | FlashAttention diagram: "2n² elements to/from DRAM" | Not expanded. Means "Dynamic Random-Access Memory." |
| **UMA** | "you read stale data on UMA platforms" — Common Pitfalls | Not expanded (Unified Memory Architecture). |
| **VRAM** | GEMV diagram: "Loaded fully from VRAM" | Not expanded (Video Random-Access Memory). |
| **MHA** | "Memory: 4× smaller KV cache vs full Multi-Head Attention (MHA)" | Expanded inline but only in a diagram caption. |
| **unit RMS** | "normalizing each vector to unit RMS (Root Mean Square...)" | Expanded inline. OK. |
| **eps / epsilon** | In formula: "eps = epsilon, a tiny constant (e.g., 1e-6) to prevent division by zero" | Explained inline. OK. |

### Glossary — Chapter 2

| Term | Definition |
|------|-----------|
| hidden state | The fixed-size internal vector representation that flows through each transformer layer, being progressively refined. |
| transformer layer | A processing unit consisting of an attention sublayer and a feed-forward network sublayer, stacked N times in a model. |
| attention | A mechanism that lets each token decide which previous tokens to focus on by computing similarity scores. |
| Q (Query) | A linear projection of the hidden state representing "what this token is looking for." |
| K (Key) | A linear projection of the hidden state representing "what this token contains." |
| V (Value) | A linear projection of the hidden state representing "what information this token carries." |
| linear projection | A matrix-vector multiply that transforms a vector into a different representation. |
| softmax | A function that converts a vector of raw scores into probabilities summing to 1.0. |
| causal mask | A constraint that prevents tokens from attending to future positions, enforced by setting future scores to −∞. |
| attention head | One independent attention computation; multiple heads run in parallel, each learning different relationships. |
| GQA (Grouped Query Attention) | An optimization that shares K/V heads across multiple Q heads to reduce KV cache memory. |
| MHA (Multi-Head Attention) | Standard attention where each Q head has its own dedicated K and V heads. |
| MLA (Multi-head Latent Attention) | An attention variant that compresses K/V into a low-rank latent space before caching. |
| SDPA (Scaled Dot-Product Attention) | The core attention formula: softmax(Q·Kᵀ/√d)·V, extracted as a reusable kernel. |
| FlashAttention | An optimization that computes attention in tiles using online softmax, avoiding materializing the full score matrix. |
| online softmax | Incrementally computing softmax as tiles arrive, without storing all scores in memory at once. |
| tile | A small rectangular block of a larger matrix, processed independently to reduce memory usage. |
| RoPE (Rotary Position Encoding) | A position encoding method that rotates Q and K vectors by position-dependent angles, encoding relative distance. |
| RMSNorm | Root Mean Square Normalization — scales a vector so its average squared value equals 1, then applies learned weights. |
| LayerNorm | An older normalization that subtracts the mean and divides by standard deviation; RMSNorm is simpler. |
| pre-norm | Applying normalization to the input before each sublayer (standard in modern transformers). |
| post-norm | Applying normalization to the output after a sublayer. |
| L2 normalization | Scaling a vector to unit length (norm = 1) without learned weights. |
| residual connection | Adding the input directly to the sublayer output (`output = input + sublayer(input)`), preserving information flow. |
| residual stream | The main data path through the model where sublayer outputs accumulate via residual connections. |
| GEMV (General Matrix-Vector multiply) | Multiplying a weight matrix by a single vector; bandwidth-bound because each weight is used once. |
| GEMM (General Matrix-Matrix multiply) | Multiplying a weight matrix by multiple vectors at once; more compute-efficient per byte loaded. |
| arithmetic intensity | The ratio of compute operations to memory bytes transferred; higher = more compute-bound. |
| prefill | Processing all prompt tokens at once through the model (GEMM, batched). |
| decode | Generating tokens one at a time in the autoregressive loop (GEMV, sequential). |
| SIMD | Single Instruction Multiple Data — processing multiple values simultaneously with one CPU instruction. |
| HBM (High Bandwidth Memory) | Off-chip DRAM on GPUs; fast but slower than on-chip SRAM. |
| SRAM | Static RAM — fast on-chip memory used for caches and registers on GPUs. |
| VRAM | Video RAM — GPU-attached memory for model weights and intermediate data. |
| UMA (Unified Memory Architecture) | A system where CPU and GPU share the same physical memory (e.g., Apple Silicon). |
| kernel (compute) | A single computational function dispatched to run on CPU or GPU hardware. |
| sliding window attention | An attention variant where each layer only attends to the most recent N tokens instead of the full sequence. |
| attention sinks | Learned per-head scalar values that absorb excess attention probability, preventing over-concentration on early positions. |

---

## Chapter 3: Feed-Forward Networks

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **FFN (Feed-Forward Network)** | "the second sublayer in each transformer layer" — opening |
| **sublayer** | "(component within a transformer layer)" — opening |
| **recurrence** | "unlike RNNs — Recurrent Neural Networks — which cycle back on themselves" — opening |
| **RNNs (Recurrent Neural Networks)** | "(Recurrent Neural Networks — which cycle back on themselves, feeding outputs back as inputs)" — opening |
| **pattern detectors** | "rows of the up-projection act as pattern detectors — each row activates strongly for specific input patterns" — knowledge explanation |
| **intermediate dimension** | "(the expanded size between projections, typically 4-8× the hidden size)" — SwiGLU section |
| **activation function** | "a nonlinear transformation applied element-wise" — SwiGLU section |
| **nonlinear** | "(output is not proportional to input — e.g., sigmoid curves, not straight lines)" — SwiGLU section |
| **GLU (Gated Linear Unit)** | "This gating pattern is called a GLU" — SwiGLU explanation |
| **SwiGLU** | "uses SiLU as the activation — hence the name: Swish + GLU = SwiGLU" — with paper citation |
| **SiLU (Sigmoid Linear Unit)** | "also called Swish" — SwiGLU section |
| **gating** | "(controlling)" — SwiGLU structure |
| **hard clamping** | "(forcing values to stay within fixed bounds)" — Clamped SwiGLU |
| **overflow** | "(values becoming too large to represent, causing errors or infinity)" — Clamped SwiGLU |
| **mixed-precision** | "(using different bit widths for different operations — e.g., 16-bit for some, 32-bit for others)" — Clamped SwiGLU |
| **router** | "(a learned selection mechanism that scores and picks which experts should process each token)" — MoE section |
| **capacity** | "(total model size/knowledge)" — MoE advantage description |
| **stack-allocated** | "(fixed-size buffers on the call stack, automatically freed when the function returns)" — expert selection |
| **heap allocation** | "(dynamic memory from the system allocator, requires explicit free)" — expert selection |
| **sigmoid routing** | "Each expert gate is independent...multiple experts can have high activation simultaneously" — GLM-4 |
| **independent** | "(evaluated separately, not competing with each other for probability mass like softmax does)" — sigmoid routing |
| **baseline** | "(consistent minimum contribution that all tokens receive, ensuring basic functionality)" — shared expert |
| **sparse activation** | "only K of N experts run per token" — MoE Sparse Activation section |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **GELU** | Activation Functions table: "GELU — 0.5x(1 + tanh(...))" | Formula given but acronym not expanded (Gaussian Error Linear Unit). Expanded only in the math reference link at the bottom. |
| **Softplus** | Activation Functions table: "Softplus — log(1 + exp(x))" | Formula given but no explanation of what it's used for beyond "SSM dt computation." |
| **ReLU²** | Activation Functions table: "ReLU² — max(0, x)²" | ReLU not expanded (Rectified Linear Unit). |
| **MoE** | Section heading "Mixture of Experts (MoE)" | Expanded in heading — OK. |
| **conv1d** | Activation table "Used by" column: "conv1d, SSM gating" | Not explained here (1D convolution). Explained in Chapter 6. |
| **SSM** | Activation table "Used by" column: "SSM gating" | Not explained here (State Space Model). Explained in Chapter 6. |
| **3D tensors** | "Expert weights are stored as 3D tensors: [n_experts, rows, cols]" | tensor was defined in Ch1, 3D usage assumed. OK. |
| **megakernel** | "fused into a single dispatch via the megakernel system" — Megakernel Fusion section | Described but the term "megakernel" itself just means a fused kernel. Explained by context. Borderline. |

### Glossary — Chapter 3

| Term | Definition |
|------|-----------|
| FFN (Feed-Forward Network) | The second sublayer in each transformer layer; processes each token independently through expansion, activation, and compression. |
| SwiGLU | A gated FFN architecture using SiLU activation on a gate projection multiplied element-wise with an up-projection. |
| GLU (Gated Linear Unit) | A neural network structure where one projection's output gates (controls) another via element-wise multiplication. |
| SiLU / Swish | Activation function: x × sigmoid(x); smooth, passes positive values, dampens negatives. |
| GELU (Gaussian Error Linear Unit) | Activation function similar to SiLU but using a Gaussian-weighted smoothing. |
| Softplus | Activation function: log(1 + exp(x)); a smooth approximation of ReLU used for ensuring positive outputs. |
| ReLU (Rectified Linear Unit) | Activation function: max(0, x); sets negatives to zero. ReLU² squares the positive values. |
| sigmoid | Activation function: 1/(1 + e^(−x)); outputs values in (0, 1), used for gates and routing. |
| activation function | A nonlinear transformation applied element-wise to introduce non-linearity into the network. |
| intermediate dimension | The expanded hidden size inside the FFN (typically 4–8× the model's hidden dimension). |
| gate projection | A linear projection whose output is passed through an activation and used to gate another projection. |
| up-projection / down-projection | Linear projections that expand (up) and compress (down) the hidden state in the FFN. |
| MoE (Mixture of Experts) | An architecture with multiple FFN "experts" where a router selects a subset to process each token. |
| router | A small learned network that scores and selects which experts should process each token. |
| top-K routing | Selecting the K highest-scoring experts for each token. |
| shared expert | An expert that is always active regardless of router output, providing a baseline contribution. |
| sparse activation | Only a small subset of total parameters is used per token; the rest remain idle. |
| expert stride | The byte offset between consecutive experts' weight data in a 3D weight tensor. |
| mixed-precision | Using different numerical bit-widths for different operations (e.g., 16-bit and 32-bit). |
| megakernel fusion | Combining multiple GPU dispatches (e.g., gate + up + down projections) into a single kernel to eliminate memory round-trips. |

---

## Chapter 4: Quantization

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **Quantization** | "maps floating-point values to lower-precision representations, trading a small amount of accuracy for massive memory and speed gains" — opening |
| **precision** | "(fewer bits per number, less accurate but smaller)" — opening |
| **7B parameter model** | "(7 billion weight values — the 'B' in model names like 'Qwen3.5-7B')" — Why Quantize |
| **memory-bandwidth bound** | "(the bottleneck is reading weights from RAM/VRAM, not arithmetic operations)" — Why Quantize |
| **scale factor** | "(a multiplier that converts small integers back to approximate float values)" — Q4_0 |
| **hierarchical scales** | "(multiple levels of scale factors — a coarse scale for the whole block, then fine-grained adjustments per sub-block)" — Super-block formats |
| **companion tensors** | "(separate tensors with matching names like `weight.scales` and `weight.biases` that store per-group quantization parameters)" — MLX section |
| **multiply-accumulates** | "(multiply two numbers and add the result to a running sum — the core operation in matrix math)" — GEMV section |
| **dtype** | "(data type — f32, bf16, q4_0, etc.)" — GEMV section |
| **cascade** | "(compound/multiply through many operations)" — FP8 E5M2 |
| **dynamic range** | "(the span from smallest to largest representable value)" — FP8 E5M2 |
| **adaptive precision** | "(more bits near zero, fewer bits for large values — precision varies based on magnitude)" — FP8 vs int8 |
| **orders of magnitude** | "(factors of 10 — e.g., from 0.001 to 1000)" — FP8 vs int8 |
| **hardware-accelerated** | "(dedicated silicon on the chip for fast execution)" — FP8 vs int8 |
| **Walsh-Hadamard Transform (WHT)** | "a deterministic rotation (like a Fourier transform but with only additions and subtractions)" — TurboQuant |
| **Lloyd-Max codebook** | "an optimal codebook for scalar quantization" — TurboQuant |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **float32** | "Model weights are trained in float32 (32 bits per value)" — opening | Partially explained by "(32 bits per value)" but the format name itself is not defined. Borderline — clear from context. |
| **Q4_0, Q8_0** | "Q4_0, Q8_0 (GGUF-style)" — Block Quantization section | Naming convention not explained. What does "Q4" mean? What does "_0" mean? (Q=quantized, 4=bits, _0=variant 0). |
| **GGUF** | "GGUF-style" — Block Quantization section | Not expanded. (GGUF = a model file format used by llama.cpp ecosystem). |
| **Q4_K, Q5_K, Q6_K** | "Super-block formats (Q4_K, Q5_K, Q6_K)" | "_K" variant naming not explained. |
| **f16** | Used throughout (e.g., "scale: f16, 2 bytes") | Not explicitly defined as 16-bit floating point; assumed known. |
| **bf16** | MLX section: "scales and biases are stored as bf16" | Not expanded here (defined in Ch1 note only tangentially). |
| **nibble** | "8 nibbles per word for 4-bit" — MLX Memory Layout | Not defined (a 4-bit value; half a byte). |
| **FMA** | "1 instruction instead of separate multiply + add" — MLX SIMD section | Not expanded (Fused Multiply-Add). |
| **NEON** | "`@mulAdd` maps to NEON `vfma`" — MLX SIMD section | Not expanded (ARM's SIMD instruction set). |
| **SafeTensors** | "stored as companion SafeTensors entries" — GPTQ/AWQ section | Not defined (a model file format). |
| **Hessian** | "Hessian-based weight updates" — GPTQ section | Not explained (second-order derivative matrix). |
| **INT4** | "Calibration-Based INT4" — section heading | Not explicitly defined (4-bit integer). |
| **GPTQ** | Section heading; no acronym expansion | Not expanded. |
| **AWQ** | Section heading; no acronym expansion | Not expanded. |
| **HQQ** | "Half-Quadratic Quantization" — section heading | Expanded in heading. OK. |
| **NVFP4, MXFP4** | "4-bit microscaled floating-point" — brief mention | Briefly described but acronyms not expanded. NVFP4 = NVIDIA FP4; MXFP4 = Microscaling FP4. |
| **E4M3, E5M2** | FP8 section headings | Expanded in subheading "(4-bit exponent, 3-bit mantissa)" — OK. |
| **mantissa** | "3-bit mantissa" — FP8 section | Not explicitly defined (the fractional part of a floating-point number). |
| **exponent** | "4-bit exponent" — FP8 section | Not explicitly defined (determines the magnitude/range of a floating-point number). |
| **subnormals** | "with subnormals" — FP8 range | Not defined (very small floating-point numbers below the normal range). |
| **PPL** | TurboQuant format table: "PPL impact" | Not expanded (Perplexity — a measure of model quality). |
| **IQ4_NL, IQ4_XS, Q2_K, Q3_K** | Choosing a Format table | Additional quantization format names without explanation. |
| **vLLM** | "not llama.cpp, vLLM, etc." | Not defined (a serving framework). |
| **llama.cpp** | Same context | Not defined (a C++ inference framework). |
| **Givens rotation** | Geometric KV section: "Givens 2D rotation" | Not defined (a type of orthogonal 2D rotation matrix). |
| **quaternion** | "Quaternion 3D rotation" | Not defined (a 4-component number system for representing 3D rotations). |
| **Clifford algebra** | "Clifford algebra Cl(3,0) rotor" | Not defined (a mathematical framework generalizing complex numbers and quaternions). |
| **bivector** | "3 bivector components" | Not defined (an oriented area element in geometric algebra). |
| **W4A16** | "AutoRound W4A16" — section heading | Not expanded (4-bit weights, 16-bit activations). |
| **QAT** | "Gemma QAT 4-bit" | Not expanded (Quantization-Aware Training). |

### Glossary — Chapter 4

| Term | Definition |
|------|-----------|
| quantization | Mapping floating-point values to lower-precision representations to reduce memory and increase speed. |
| block quantization | Grouping values (typically 32) that share a single scale factor for dequantization. |
| scale factor | A per-block multiplier that converts stored small integers back to approximate float values. |
| super-block | A larger group (typically 256 values) with hierarchical two-level scales for finer-grained quantization. |
| dequantization | Converting quantized integers back to floating-point values, typically inside the GEMV kernel. |
| Q4_0 | A GGUF quantization format: 4-bit values in blocks of 32 with one f16 scale per block. |
| Q8_0 | A GGUF quantization format: 8-bit values in blocks of 32 with one f16 scale per block. |
| Q4_K / Q5_K / Q6_K | GGUF super-block quantization formats with 4/5/6-bit values and hierarchical scales. |
| GGUF | A binary model file format (used by llama.cpp ecosystem) that embeds quantization metadata in weight blocks. |
| SafeTensors | A model file format where tensors and metadata are stored separately, used by Hugging Face ecosystem. |
| MLX quantization | Apple's affine quantization: float = scale × uint + bias, with separate companion tensors for scales and biases. |
| affine quantization | A dequantization formula using both scale and bias: float = scale × int + bias. |
| companion tensors | Separate tensors (e.g., `.scales`, `.biases`) storing per-group quantization parameters alongside packed weight tensors. |
| nibble | A 4-bit value; half a byte. Two nibbles pack into one byte. |
| FP8 E4M3 | An 8-bit floating-point format with 4-bit exponent and 3-bit mantissa; used for weights. |
| FP8 E5M2 | An 8-bit floating-point format with 5-bit exponent and 2-bit mantissa; wider range, used for activations. |
| mantissa | The fractional precision bits of a floating-point number. |
| exponent | The bits determining the magnitude range of a floating-point number. |
| subnormal | A very small floating-point number below the normal representable range. |
| ternary quantization | Encoding weights as {−1, 0, +1}, enabling multiplication-free inference. |
| TQ1_0 | Ternary format at 1.58 bits/weight using base-3 packing. |
| TQ2_0 | Ternary format at 2 bits/weight using simple 2-bit binary packing. |
| GPTQ | A calibration-based INT4 quantization method using Hessian-based weight updates to minimize quantization error. |
| AWQ | A calibration-based INT4 quantization method that finds per-channel activation scales to protect important weights. |
| HQQ (Half-Quadratic Quantization) | A calibration-free INT4 quantization method using iterative optimization. |
| Hessian | A matrix of second-order derivatives used by GPTQ for optimal weight rounding decisions. |
| NVFP4 | NVIDIA's 4-bit microscaled floating-point format with 16-element blocks and FP8 scales. |
| MXFP4 | A 4-bit microscaled floating-point format with 32-element blocks and FP8 scales. |
| W4A16 | Shorthand for 4-bit weights with 16-bit activations. |
| QAT (Quantization-Aware Training) | Training a model with quantization effects simulated, producing weights optimized for low-bit inference. |
| TurboQuant | A KV cache quantization method using Walsh-Hadamard Transform preprocessing and Lloyd-Max codebooks. |
| Walsh-Hadamard Transform (WHT) | A deterministic orthogonal rotation using only additions and subtractions that gaussianizes distributions before quantization. |
| Lloyd-Max codebook | A set of optimal quantization bins and centroid values for scalar quantization of Gaussian-distributed data. |
| PlanarQuant | A KV cache quantization method using Givens 2D rotations for decorrelation. |
| IsoQuant | A KV cache quantization method using quaternion 3D rotations for decorrelation. |
| RotorQuant | A KV cache quantization method using Clifford algebra rotors for structure-preserving 3D rotations. |
| Givens rotation | An orthogonal rotation applied to a pair of coordinates in a 2D plane. |
| quaternion | A 4-component number system used to represent 3D rotations without gimbal lock. |
| perplexity (PPL) | A measure of how well a model predicts text; lower = better quality. |
| memory-bandwidth bound | When performance is limited by the rate of reading data from memory, not by compute speed. |
| FMA (Fused Multiply-Add) | A single hardware instruction that computes a×b+c in one step, more accurate and faster than separate multiply then add. |

---

## Chapter 5: Memory and Caching

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **autoregressive generation** | "(generating text one token at a time, where each new token depends on all previous tokens)" — opening |
| **KV cache** | "stores them" (K and V vectors) — opening; structure explained immediately after |
| **internal fragmentation** | "(wasted space within allocated regions)" — PagedAttention benefits |
| **reference counting** | "(tracking how many sequences use each block)" — PagedAttention benefits |
| **copy-on-write** | "(sharing read-only data, duplicating only when modified)" — PagedAttention benefits |
| **radix tree** | "(also called a prefix trie — a tree data structure where shared prefixes are stored only once)" — RadixAttention |
| **prefix trie** | Inline with radix tree definition |
| **LRU** | "(Least Recently Used — remove the oldest unused data first)" — RadixAttention eviction |
| **timestamps** | "(recorded times when each block was last used)" — RadixAttention eviction |
| **eviction cost** | "(penalty score that makes them harder to remove)" — RadixAttention eviction |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **PagedAttention** | Section heading only; explained by description | The concept is well-explained but the term itself is just a proper noun. OK. |
| **SDPA** | "The CPU SDPA kernel supports block-table-indexed attention" | Used without expansion; defined in Ch2. Cross-chapter reference. |
| **FA2** | Chunked Prefill diagram: "GEMM + FA2" | Not expanded (FlashAttention-2). Abbreviation not used elsewhere. |
| **OOM** | "Use `--ctx-size auto` to avoid OOM at startup" | Not expanded (Out Of Memory). |
| **APEX** | "inspired by APEX" — Async Split-Attention section | The paper is cited but APEX as an acronym is not expanded. |
| **absmax** | "tracked as the running absmax across all positions" — Per-Head KV section | Not defined (absolute maximum value). |
| **LMCache** | "LMCache-style" — Cross-Instance KV Cache Sharing | Not defined (a system for sharing KV caches across instances). |
| **fleet** | "distribute to a fleet" | Jargon for a group of server instances; not technical ML terminology. |

### Glossary — Chapter 5

| Term | Definition |
|------|-----------|
| KV cache | Storage for previously computed Key and Value vectors so they don't need to be recomputed for each new token. |
| PagedAttention | A memory management technique that maps logical sequence positions to non-contiguous physical memory blocks, like OS virtual memory. |
| block (KV cache) | A fixed-size unit of KV cache storage (default 16 positions) allocated on demand. |
| block table | A per-request mapping from logical sequence positions to physical memory blocks. |
| RadixAttention | A caching strategy that uses a radix tree (prefix trie) to detect and share common prompt prefixes across requests. |
| prefix trie / radix tree | A tree data structure where shared prefixes are stored once and branched at divergence points. |
| continuous batching | Processing multiple requests simultaneously where each can grow/shrink independently. |
| chunked prefill | Splitting long prompts into fixed-size chunks (e.g., 512 tokens) to bound memory usage during batched prefill. |
| KV cache eviction | Removing low-value entries from the KV cache when context exceeds the budget, allowing generation to continue. |
| attention sinks | The first few token positions that accumulate disproportionate attention mass and should never be evicted. |
| norm-based eviction | Scoring cached positions by L2 norm of their K vector; low-norm positions are evicted first. |
| split-attention (APEX) | Running GPU and CPU SDPA concurrently on different KV cache tiers, merging results via online softmax correction. |
| tiered KV cache | Storing KV cache blocks across multiple memory tiers (VRAM, RAM, SSD) based on access recency. |
| per-head KV quantization | Using one dynamic scale per KV head (tracked as running absmax) rather than per-block scales. |
| auto context sizing | Automatically probing available memory to select the largest safe context window. |
| OOM (Out Of Memory) | An error when the system cannot allocate enough memory for the requested operation. |
| cross-instance KV sharing | Exporting and importing KV cache data between server instances to avoid redundant prefill computation. |

---

## Chapter 6: State Space Models

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **SSMs** | "a family of sequence models based on state-space theory" — opening |
| **selective** | "input-dependent parameters that give SSMs content-aware reasoning ability" — Mamba reference |
| **O(1) with respect to sequence length** | "(constant time — doesn't grow with the number of previous tokens)" — opening |
| **state matrix** | "a fixed-size state matrix that summarizes the past" — opening |
| **decay** | "controls how quickly old information fades — like a leaky bucket" — after simplified formula |
| **hybrid models** | "combine attention and SSM layers" — after state matrix explanation |
| **causal convolution / convolution** | "a sliding window operation that combines nearby values using learned weights. Causal means it only looks at past inputs" — Causal Convolution section |
| **ring buffer** | "(a fixed-size circular array where new entries overwrite the oldest, avoiding reallocation)" — Causal Convolution section |
| **linear-complexity recurrence** | "(an update loop where each step depends only on the previous step's state, not all history)" — DeltaNet |
| **outer-product** | "(forming a matrix by multiplying a column vector by a row vector)" — DeltaNet |
| **delta rule** | "the update is proportional to the error (v - S^T * k), not just the raw value" — DeltaNet |
| **discretization** | "(choosing how much time passes between updates)" — Mamba-2 |
| **dt (delta-time)** | "the dt (timestep, delta-time) is computed from the input" — Mamba-2 |
| **selectivity** | "(the model can choose what to remember based on the current input, not just a fixed decay pattern)" — Mamba-2 |
| **metadata** | "(descriptive information about the model structure — layer counts, dimensions, patterns — stored in the model file header)" — Hybrid Layer Patterns |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **Mamba** | "Mamba (Gu & Dao, 2023)" — opening | Proper name with citation; not an acronym. OK. |
| **DeltaNet** | "DeltaNet (Qwen3.5)" — section heading | Proper name. Was briefly mentioned in Ch2. |
| **d_conv** | "With d_conv=4" — Causal Convolution | Variable name not explicitly defined (convolution width). Context makes it clear. |
| **GQA** | "GQA head mapping uses tiling" — DeltaNet section | Not re-explained; defined in Ch2. Cross-chapter reference. |
| **conv1d** | "After conv1d, output splits as [Q | K | V]" — DeltaNet | Not defined here (1-dimensional convolution). Explained earlier as "causal convolution" in same chapter but without this specific abbreviation. |
| **B, C, D** | Mamba-2: "B projection (what to write), C projection (what to read), D skip" | Explained inline in the diagram. OK. |
| **V8** | "V8-vectorized, not scalar" — Hardware Considerations | Not defined (8-wide SIMD vector type in Zig). |
| **SIMD** | "CPU SIMD kernel" — Hardware Considerations | Not re-explained; defined in Ch2. |

### Glossary — Chapter 6

| Term | Definition |
|------|-----------|
| SSM (State Space Model) | A sequence model that maintains a fixed-size state matrix as a compressed summary of all past tokens, updating in O(1) per step. |
| state matrix | A fixed-size matrix (e.g., 128×128 per head) that accumulates key-value associations via outer-product updates with decay. |
| decay factor | A multiplier < 1 applied to the state matrix each step, causing older information to exponentially fade. |
| selective state space | An SSM variant (Mamba) where parameters like decay and input gating are input-dependent, not fixed. |
| causal convolution | A sliding-window operation that combines nearby values using learned weights, looking only at past positions. |
| ring buffer | A fixed-size circular array where new entries overwrite the oldest, used to store convolution history without allocation. |
| d_conv | The width of the causal convolution window (e.g., 4 = current + 3 past inputs). |
| DeltaNet | A linear-complexity recurrence that uses the delta rule (error-correcting outer-product updates) to maintain associative memory. |
| delta rule | An update rule where the state correction is proportional to the error between the desired value and what the state already encodes. |
| outer product | Forming a matrix by multiplying a column vector by a row vector; used to write key-value associations into the state matrix. |
| Mamba-2 | An SSM architecture with input-dependent discretization (dt), allowing the model to selectively remember or forget. |
| discretization (dt) | Computing a per-step timestep from the input, controlling how much the state decays and how much new information is written. |
| hybrid model | A model that interleaves attention layers (for exact recall) with SSM layers (for speed) in a single architecture. |
| recurrence | A computation where each step depends on the previous step's output, processing sequentially rather than in parallel. |
| linear-complexity recurrence | An update loop running in O(d²) per step (constant with respect to sequence length), as opposed to O(n) attention. |

---

## Chapter 7: Sampling

### Terms PROPERLY explained on first use ✅

| Term | Where explained |
|------|----------------|
| **greedy decoding** | "(pick the highest score)" — opening |
| **deterministic** | "(always the same output for the same input)" — opening |
| **temperature** | "Controls randomness by scaling logits before sampling" — Temperature section |
| **top-K** | "Restricts sampling to only the K highest-scoring tokens" — Top-K section |
| **renormalize** | "(rescale so they sum to 1.0 again)" — Top-K section |
| **nucleus sampling / top-P** | "restricts sampling to the smallest set of tokens whose cumulative probability exceeds P" — Top-P section, with paper citation |
| **cumulative probability** | "(running sum of probabilities in sorted order)" — Top-P section |
| **min-P** | "keeps tokens whose probability is at least min_p × the top token's probability" — Min-P section |
| **repeat penalty** | "Discourages repeating previously generated tokens" — Repeat Penalty section |
| **frequency penalty** | "Per-occurrence penalty — penalizes repeated tokens proportionally" — table |
| **presence penalty** | "One-time penalty — discourages any reuse of generated tokens" — table |
| **XTC (eXclude Top Choices)** | "randomly excludes high-probability tokens to increase diversity" — XTC section |
| **mode collapse** | "where the model repeatedly generates the same high-probability sequences" — XTC section |
| **DRY (Don't Repeat Yourself)** | "penalizes tokens that would continue a repeated n-gram sequence" — DRY section |
| **Mirostat** | "maintains consistent perplexity during generation by dynamically adjusting the sampling threshold" — Mirostat section |
| **perplexity** | "(unpredictability)" — Mirostat section |
| **logit bias** | "Direct per-token adjustments to logits via the API" — Logit Bias section |
| **grammar-constrained decoding** | "Forces output to match a formal grammar (GBNF format)" — Grammar section |
| **jump decoding** | "When the grammar allows exactly one valid next token...the forward pass is skipped entirely" — Grammar section |

### Terms NOT explained on first use ❌

| Term | First appearance | Issue |
|------|------------------|-------|
| **GBNF** | "Forces output to match a formal grammar (GBNF format)" | Not expanded or defined. (Generative BNF — a grammar format used by llama.cpp). |
| **n-gram** | "penalizes tokens that would continue a repeated n-gram sequence" — DRY section | Not defined (a contiguous sequence of n tokens). |
| **bigram** | "penalize repeated bigrams and longer" — DRY section | Not defined (a sequence of exactly 2 tokens). |
| **entropy** | "Target entropy — lower = more focused, higher = more creative" — Mirostat | Not formally defined (a measure of uncertainty/information content in a distribution). |
| **tau** | "targets a specific entropy level (tau)" — Mirostat | Greek letter used as parameter name without explaining why tau or its mathematical role beyond "target entropy." |
| **eta** | "adapts via learning rate (eta)" — Mirostat | Greek letter used as parameter name. |
| **mu** | "current entropy estimate, starts at 2 * tau" — Mirostat diagram | Greek letter used as variable name. |
| **learning rate** | "adapts via learning rate (eta)" — Mirostat | Not defined for this context (step size for the adaptive update). |

### Glossary — Chapter 7

| Term | Definition |
|------|-----------|
| sampling | Selecting the next token from the probability distribution produced by softmax on the logits. |
| greedy decoding | Always selecting the highest-probability token (argmax); deterministic but often repetitive. |
| temperature | A scaling factor applied to logits before softmax; lower values make the distribution peakier, higher values make it flatter. |
| top-K sampling | Restricting the candidate set to only the K highest-scoring tokens before sampling. |
| top-P / nucleus sampling | Keeping the smallest set of tokens whose cumulative probability exceeds threshold P, then renormalizing. |
| min-P | An adaptive threshold keeping only tokens whose probability is at least min_p × the top token's probability. |
| repeat penalty | A multiplicative penalty applied to logits of previously generated tokens to discourage repetition. |
| frequency penalty | An additive per-occurrence penalty proportional to how many times a token has appeared. |
| presence penalty | A one-time additive penalty applied to any token that has appeared at least once. |
| XTC (eXclude Top Choices) | A sampling method that randomly removes high-probability tokens to increase output diversity. |
| DRY (Don't Repeat Yourself) | A penalty method that detects repeated n-gram sequences and penalizes tokens that would continue them. |
| n-gram | A contiguous sequence of n tokens (bigram = 2, trigram = 3, etc.). |
| Mirostat | An adaptive sampling method that dynamically adjusts the candidate set to maintain a target entropy (perplexity) level. |
| entropy | A measure of uncertainty in a probability distribution; higher entropy = more uniform/unpredictable. |
| logit bias | Direct additive adjustments to specific token logits before sampling, used for API-level steering. |
| grammar-constrained decoding | Masking logits so only tokens consistent with a formal grammar (GBNF, JSON schema) can be selected. |
| GBNF | A grammar format (Generative BNF) used to specify valid output patterns for constrained decoding. |
| jump decoding | Skipping the forward pass when the grammar allows exactly one valid next token, emitting it directly. |
| mode collapse | When sampling repeatedly produces the same high-probability sequences due to insufficient diversity. |
| renormalize | Rescaling probabilities after filtering so they sum to 1.0 again. |

---

## Coverage Status

### Completed ✅
- All 7 chapters fully analyzed for:
  - Terms explained on first use
  - Terms NOT explained on first use (with location and issue description)
  - Per-chapter glossary of ML/inference terms a systems programmer wouldn't know

### Key patterns observed
1. **Chapters 1–3 are excellent** at defining terms inline on first use. Nearly every ML term gets a parenthetical definition.
2. **Chapter 4 (Quantization)** has the most unexplained terms, primarily: quantization format naming conventions (Q4_0, Q4_K, etc.), file format names (GGUF, SafeTensors), hardware terms (NEON, FMA), and advanced math terms (Hessian, Clifford algebra, bivector).
3. **Cross-chapter references** are the main source of unexplained terms in later chapters — terms defined in earlier chapters (SDPA, GQA, SIMD) are used without re-explanation.
4. **Hardware acronyms** (HBM, SRAM, DRAM, VRAM, UMA, NEON) are consistently not expanded across all chapters.
5. **Model family names** (Qwen, Gemma, GPT-OSS, Nemotron) are used as proper nouns without introduction, which is reasonable but a glossary entry would help newcomers.
6. **Greek letters** used as parameter names (tau, eta, mu, epsilon) are sometimes but not always explained.

### Not checked
- Chapters 8+ (not in scope for this batch)
- Cross-references to appendix-math.md for correctness
- Whether code links are valid
