# Chapter 14: Format Conventions

The same model can be stored in different file formats — **GGUF** (llama.cpp native) and **SafeTensors** (HuggingFace native). They store identical weights but use **different conventions** for tensor layout, metadata keys, and even mathematical transformations.

**Critical insight:** Using GGUF conventions on SafeTensors data (or vice versa) produces **silent correctness failures** — the model runs but outputs garbage. Agave found **6 separate bugs** when adding SafeTensors support for Qwen3.5.

## Why Formats Have Different Conventions

**GGUF** is designed by llama.cpp maintainers who optimize for:
- Mmap-friendly layout (weights in file order)
- Quantization-first design
- C++ naming conventions

**SafeTensors** follows HuggingFace/PyTorch conventions:
- Python/PyTorch tensor names
- Original research paper layouts
- JSON metadata (not binary-packed)

When llama.cpp converts a HuggingFace model to GGUF, it **transforms** the data to match llama.cpp's internal conventions. Agave must **detect the format** and apply the correct convention.

## Format Detection

```zig
// src/format/format.zig — vtable-based polymorphism
pub const Format = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// True for SafeTensors (HF conventions), false for GGUF (llama.cpp conventions).
    is_safetensors: bool = false,

    pub const VTable = struct {
        get_tensor: *const fn (self: *anyopaque, name: []const u8) ?TensorInfo,
        get_meta_str: *const fn (self: *anyopaque, key: []const u8) ?[]const u8,
        get_meta_u32: *const fn (self: *anyopaque, key: []const u8) ?u32,
        // ...
    };

    pub fn getTensor(self: Format, name: []const u8) ?TensorInfo {
        return self.vtable.get_tensor(self.ptr, name);
    }
};

// Usage in main.zig — format detection by path type:
// Directory → SafeTensors, single file → GGUF
if (is_dir) {
    st_dir = try SafeTensorsDir.open(allocator, model_path);
    fmt = st_dir.?.format(); // returns Format interface
} else {
    gguf_file = try GGUFFile.open(allocator, model_path);
    fmt = gguf_file.?.format(); // returns Format interface
}
```

**Flag:** `is_safetensors` field added to Format interface to decouple format detection from convention selection.

## Convention Differences

### 1. DeltaNet Conv Output Split Order

**Operation:** After causal conv1d, output is split into Q, K, V tensors.

**GGUF (llama.cpp):**
```zig
// Split order: Q, K, V (matches llama.cpp ggml_repeat semantics)
const q_start = 0;
const k_start = key_dim;
const v_start = key_dim + key_dim;

@memcpy(q_buf[0..key_dim], conv_out[q_start..][0..key_dim]);
@memcpy(k_buf[0..key_dim], conv_out[k_start..][0..key_dim]);
@memcpy(v_buf[0..v_dim],   conv_out[v_start..][0..v_dim]);
```

**SafeTensors (HuggingFace):**
```zig
// Split order: K, Q, V (matches original DeltaNet paper)
const k_start = 0;
const q_start = key_dim;
const v_start = key_dim + key_dim;

@memcpy(k_buf[0..key_dim], conv_out[k_start..][0..key_dim]);
@memcpy(q_buf[0..key_dim], conv_out[q_start..][0..key_dim]);
@memcpy(v_buf[0..v_dim],   conv_out[v_start..][0..v_dim]);
```

**Controlled by:** `kqv_order` flag in `DeltaNetParams`:

```zig
pub const DeltaNetParams = struct {
    // ...
    /// True when conv_out split order is K,Q,V (HuggingFace/SafeTensors).
    /// False (default) when split order is Q,K,V (GGUF/llama.cpp convention).
    kqv_order: bool = false,
};
```

**Model code:**

```zig
// src/models/qwen35.zig
const p = DeltaNetParams{
    // ... other params ...
    .kqv_order = self.fmt.is_safetensors,  // Detect format
};
```

### 2. DeltaNet GQA Head Mapping

**Problem:** GQA maps Q heads to KV heads. Two different semantics exist.

**GGUF (llama.cpp TILING):**
```zig
// V-head maps to K-head via modulo wrapping
const kh = h % num_k_heads;
```

**Example:** 8 V-heads, 2 K-heads
- V-head 0 → K-head 0 (0 % 2)
- V-head 1 → K-head 1 (1 % 2)
- V-head 2 → K-head 0 (2 % 2)
- V-head 3 → K-head 1 (3 % 2)
- Pattern: `0,1,0,1,0,1,0,1` (tiled)

**SafeTensors (INTERLEAVED GROUPING):**
```zig
// V-heads grouped by K-head
const kh = h * num_k_heads / num_v_heads;
```

**Example:** 8 V-heads, 2 K-heads
- V-heads 0-3 → K-head 0 (0×2/8 = 0)
- V-heads 4-7 → K-head 1 (4×2/8 = 1)
- Pattern: `0,0,0,0,1,1,1,1` (interleaved groups)

**Controlled by:** Same `kqv_order` flag (GQA mapping convention follows split order convention).

**Implementation:**

```zig
// src/backend/kernels/cpu/deltanet.zig
const kh = if (p.kqv_order)
    h * p.num_k_heads / p.num_v_heads  // SafeTensors: interleaved groups
else
    h % p.num_k_heads;                 // GGUF: tiling
```

### 3. SSM A_log Pre-Conversion

**Operation:** DeltaNet state decay uses `exp(A_log * dt)`.

**GGUF:**
```zig
// A_log is stored as -exp(A_log) (pre-converted by llama.cpp)
const decay = ssm_a[h] * dt;  // ssm_a already contains -exp(A_log)
```

**SafeTensors:**
```zig
// A_log is stored raw (must convert at init)
for (ssm_a) |*a| {
    a.* = -@exp(a.*);  // Convert once at model load
}
// Then use same code as GGUF
const decay = ssm_a[h] * dt;
```

**Detection:**

```zig
// src/models/qwen35.zig init()
if (self.fmt.is_safetensors) {
    // Convert A_log to -exp(A_log)
    for (0..n_layers) |layer| {
        const ssm_a = self.getLayerTensor(layer, "ssm_a");
        for (ssm_a) |*a| {
            a.* = -@exp(a.*);
        }
    }
}
```

**Why the difference?** llama.cpp pre-computes this to avoid calling `exp()` on every token. PyTorch stores the raw value for flexibility.

### 4. Q/Gate Split Layout

**Operation:** DeltaNet projects Q and gate together, then splits them.

**GGUF (interleaved per head):**
```
[Q0, G0, Q1, G1, Q2, G2, ..., Q_{hd-1}, G_{hd-1}] × nh heads
```

**SafeTensors (concatenated per head):**
```
[Q0..Q_{hd-1}, G0..G_{hd-1}] × nh heads
```

**Split code:**

```zig
if (self.fmt.is_safetensors) {
    // Concatenated: first half = Q, second half = gate
    for (0..nh) |h| {
        const src = h * hd * 2;
        const q_src = src;
        const g_src = src + hd;
        @memcpy(q_buf[h*hd..][0..hd], qg[q_src..][0..hd]);
        @memcpy(g_buf[h*hd..][0..hd], qg[g_src..][0..hd]);
    }
} else {
    // Interleaved: alternating Q and gate
    for (0..nh) |h| {
        for (0..hd) |i| {
            const src = h * hd * 2 + i * 2;
            q_buf[h*hd + i] = qg[src];
            g_buf[h*hd + i] = qg[src + 1];
        }
    }
}
```

**Impact:** Wrong layout → Q gets half of gate's values, gate gets half of Q's → attention completely broken.

### 5. Gate Detection via Tensor Dimensions

**Problem:** Detect whether a projection has a gate by checking tensor shape.

**GGUF:**
```zig
// numElements() returns actual element count
const has_gate = (tensor.numElements() == n_embd * 2);
```

**SafeTensors (MLX quantized):**
```zig
// numElements() returns U32 word count, not element count!
// Must use dims[0] (output dimension) instead
const has_gate = (tensor.dims[0] == n_embd * 2);
```

**Root cause:** MLX quantization packs weights in U32 words. GGUF `numElements()` returns the element count (after unpacking), but SafeTensors `numElements()` returns word count.

**Fix:**

```zig
pub fn hasGate(tensor: TensorInfo, n_embd: usize, is_safetensors: bool) bool {
    if (is_safetensors) {
        return tensor.dims[0] == n_embd * 2;  // Use dims[0] for MLX
    } else {
        return tensor.numElements() / n_embd == 2;  // Use element count for GGUF
    }
}
```

### 6. Norm Weight Caching (Affects Both Formats)

**Problem:** Metal `getBufRef()` caches buffer wrappers by host pointer. If you modify host memory after caching, GPU reads stale data.

**Bad pattern:**

```zig
// Dequant bf16 norm weights to f32 into scratch buffer
dequantToF32(bf16_norm, scratch, n_embd);  // Write to scratch
const buf = be.getBufRef(scratch);         // Cache scratch pointer → MTLBuffer
// ... use for this layer ...

// Next layer: reuse scratch
dequantToF32(bf16_norm_layer2, scratch, n_embd);  // Modify scratch
const buf2 = be.getBufRef(scratch);  // Returns CACHED buffer (stale!)
// GPU reads layer 1's norm weights, not layer 2's
```

**Fix:** Use **per-tensor cache** instead of reusable scratch:

```zig
// src/models/qwen35.zig
norm_cache: std.AutoHashMap(usize, []f32)  // Key = bf16 norm ptr

pub fn normAsF32(self: *Model, bf16_ptr: [*]const u8, n: usize) ![]f32 {
    const key = @intFromPtr(bf16_ptr);
    if (self.norm_cache.get(key)) |cached| return cached;

    // Allocate permanent f32 buffer
    const f32_buf = try self.allocator.alloc(f32, n);
    dequantToF32(bf16_ptr, f32_buf, n);

    self.norm_cache.put(key, f32_buf) catch {};
    return f32_buf;
}
```

**Key insight:** Each norm weight gets its own permanent f32 buffer. Metal caches the pointer → always correct data.

## Metadata Key Mapping

**GGUF and HuggingFace use different metadata key names.**

### SSM Dimension Mappings

```zig
// src/format/safetensors.zig
const gguf_hf_meta_map = std.StaticStringMap([]const u8).initComptime(.{
    .{ "full_attn_interval", "full_attention_interval" },
    .{ "ssm.conv_kernel", "linear_conv_kernel_dim" },
    .{ "ssm.state_size", "linear_key_head_dim" },
    .{ "ssm.group_count", "linear_num_key_heads" },
    .{ "ssm.time_step_rank", "linear_num_value_heads" },
    .{ "partial_rotary_factor", "partial_rotary_factor" },
});
```

**Usage:**

```zig
pub fn getMetaU32(self: *GGUFFile, key: []const u8) ?u32 {
    // Try GGUF key first
    if (self.meta.get(key)) |val| return val.asU32();

    // Fallback to HF key
    if (gguf_hf_meta_map.get(key)) |hf_key| {
        if (self.meta.get(hf_key)) |val| return val.asU32();
    }

    return null;
}
```

**Example:** Qwen3.5 reads `ssm.conv_kernel` (GGUF) or `linear_conv_kernel_dim` (HF) transparently.

## Tensor Name Mapping

**HuggingFace uses different tensor names than llama.cpp.**

### DeltaNet Tensor Names

```zig
// GGUF → HF mapping
const tensor_name_map = std.StaticStringMap([]const u8).initComptime(.{
    .{ "attn_qkv", "linear_attn.in_proj_qkv" },
    .{ "attn_gate", "in_proj_z" },
    .{ "ssm_alpha", "in_proj_a" },
    .{ "ssm_beta", "in_proj_b" },
    .{ "ssm_out", "out_proj" },
    .{ "ssm_a", "A_log" },
    .{ "ssm_conv1d", "conv1d" },
    .{ "ssm_norm", "norm" },
    .{ "ssm_dt.bias", "dt_bias" },  // Special case: HF uses underscore
});
```

### Attribute-less Tensor Names

**GGUF:** All tensors have `.weight` suffix
```
blk.0.attn_qkv.weight
blk.0.ssm_a.weight
```

**SafeTensors:** Some tensors have no suffix
```
model.layers.0.linear_attn.in_proj_qkv.weight  ← has .weight
model.layers.0.linear_attn.A_log                ← NO .weight
```

**Translation function:**

```zig
pub fn ggufToHfName(gguf_name: []const u8) []const u8 {
    // Handle attribute-less tensors
    if (std.mem.endsWith(u8, gguf_name, "ssm_a")) {
        return "linear_attn.A_log";  // No .weight suffix
    }
    if (std.mem.endsWith(u8, gguf_name, "ssm_dt.bias")) {
        return "dt_bias";  // No .bias suffix in HF
    }

    // Regular tensors: use map
    if (tensor_name_map.get(gguf_name)) |hf_name| {
        return hf_name ++ ".weight";  // Append .weight
    }

    return gguf_name;  // No mapping found
}
```

## Dimension Order Normalization

**GGUF stores dims reversed** (inner dimension first), while **SafeTensors stores dims in PyTorch order** (outer dimension first).

Agave normalizes GGUF dimensions during parsing so `dims[0]` always means output rows, regardless of format:

```zig
// src/format/gguf.zig — dims reversed at parse time
var raw_dims: [4]u64 = .{ 0, 0, 0, 0 };
for (0..n_dims) |d| {
    raw_dims[d] = try self.readU64(off);
    off += 8;
}
var dims: [4]u64 = .{ 0, 0, 0, 0 };
for (0..n_dims) |d| dims[d] = raw_dims[n_dims - 1 - d];
```

This means all model code can use `dims[0]` uniformly:

```zig
// In model code (all formats):
const out_dim = tensor.dims[0];  // Always outer dimension
```

## Testing Across Formats

**Strategy:** Load the same model in both formats, compare outputs token-by-token.

```zig
test "qwen35 GGUF vs SafeTensors equivalence" {
    const gguf_model = try loadModel(allocator, "model.gguf");
    defer gguf_model.deinit();

    const st_model = try loadModel(allocator, "model_safetensors/");
    defer st_model.deinit();

    const prompt = "Hello, world!";
    const tokens = try tokenize(prompt);

    for (tokens) |token| {
        const gguf_logits = try gguf_model.forward(token);
        const st_logits = try st_model.forward(token);

        // Compare logits (should be identical within FP precision)
        for (gguf_logits, st_logits) |g, s| {
            try std.testing.expectApproxEqAbs(g, s, 1e-4);
        }
    }
}
```

**Catches:**
- Wrong split order → different Q/K/V → different attention scores
- Wrong GQA mapping → different KV lookup → different outputs
- Missing A_log conversion → different decay → state diverges

## Common Pitfalls

### Pitfall 1: Assuming Single Convention

```zig
// BAD: Hardcoded GGUF convention
const kh = h % num_k_heads;  // Wrong for SafeTensors!
```

**Fix:** Detect format, apply correct convention.

### Pitfall 2: Format Detection via Quantization

```zig
// BAD: Conflates format (GGUF vs SafeTensors) with quantization (MLX vs GGUF-Q)
const is_mlx = tensor.dtype == .mlx_q;
if (is_mlx) {
    // Apply SafeTensors conventions  ← WRONG! BF16 SafeTensors exists
}
```

**Fix:** Use `is_safetensors` flag, not dtype.

### Pitfall 3: Cached Buffer Corruption

```zig
// BAD: Reuse scratch buffer for different norms
dequant(norm1, scratch);
gpu_buffer = getBufRef(scratch);  // Caches scratch → GPU buffer mapping
dequant(norm2, scratch);          // Overwrites scratch
// GPU buffer still points to old norm1 data!
```

**Fix:** Per-tensor cache or disable caching for scratch buffers.

### Pitfall 4: Forgetting Metadata Mapping

```zig
// BAD: Only check GGUF key
const d_conv = fmt.getMetaU32("ssm.conv_kernel") orelse return error.MissingMeta;
// Fails on SafeTensors (uses "linear_conv_kernel_dim")
```

**Fix:** Use bidirectional mapping (gguf_hf_meta_map).

## mmproj GGUF — Vision Encoder Weights

Multimodal models store vision encoder weights in a **separate GGUF file** (the "mmproj" file), distinct from the main language model GGUF. This keeps the text model self-contained — vision is an optional add-on.

### Tensor Naming

Vision encoder tensors use a different prefix scheme than the main model:

```
v.blk.0.attn_q.weight      — Vision transformer block 0, Q projection
v.blk.0.attn_k.weight      — K projection
v.blk.0.ffn_up.weight      — FFN up projection
v.patch_embd.weight         — Patch embedding convolution
v.position_embd.weight      — Positional embedding
mm.input_projection.weight  — Final projection into LLM embedding space
mm.soft_emb_norm.weight     — Soft embedding norm (Gemma 3)
mm.0.weight, mm.2.weight    — MLP projector layers (Qwen VL)
```

The `v.` prefix denotes vision encoder layers, while `mm.` denotes the multimodal projection head that maps vision features into the language model's embedding dimension.

### Auto-Detection

Agave auto-detects mmproj files by scanning the model directory for files matching `mmproj*.gguf`:

```zig
// src/main.zig — mmproj auto-detection
if (mmproj_path == null and (cli.image != null or cli.serve)) {
    // Scan model directory for mmproj*.gguf
    while (dir.next()) |entry| {
        if (std.mem.startsWith(u8, entry.name, "mmproj") and
            std.mem.endsWith(u8, entry.name, ".gguf"))
        {
            mmproj_path = entry.name;
        }
    }
}
```

You can also specify the path explicitly with `--mmproj path/to/mmproj.gguf`.

### Key Metadata

The mmproj GGUF carries its own architecture metadata under the `clip.vision` namespace:

| Metadata Key | Description | Example |
|---|---|---|
| `clip.vision.image_size` | Input image resolution (pixels) | 768 (Gemma 4), 896 (Gemma 3) |
| `clip.vision.patch_size` | Patch extraction stride (pixels) | 16 |
| `clip.vision.projection_dim` | Output embedding dimension (must match LLM) | 2816 |
| `clip.vision.embedding_length` | Internal ViT hidden dimension | 1152 |
| `clip.vision.block_count` | Number of ViT transformer blocks | 27 |
| `clip.vision.attention.head_count` | Number of attention heads | 16 |

The `projection_dim` is the critical interface parameter — it must match the language model's `n_embd` so that visual embeddings can replace token embeddings in the forward pass. The vision encoder auto-detects its architecture variant (Gemma 4 SigLIP-2, Gemma 3 SigLIP, Qwen VL) from the available tensors in the mmproj file.

## Summary: Format Checklist

When adding support for a new model architecture:

- [ ] Detect format via `is_safetensors` flag
- [ ] Check if tensor split order differs (Q/K/V, Q/gate, etc.)
- [ ] Check if GQA head mapping differs
- [ ] Check if any tensors need init-time conversion (A_log, etc.)
- [ ] Check if tensor names differ (use mapping)
- [ ] Check if metadata keys differ (use mapping)
- [ ] Check if dimension order needs normalization
- [ ] Use per-tensor norm cache (not reusable scratch)
- [ ] Write equivalence test (GGUF vs SafeTensors)

**Golden rule:** Same model, different format → **identical outputs**. Any divergence is a bug.

---

**In the code:** [src/format/gguf.zig](../../src/format/gguf.zig) (GGUF loader with HF mapping), [src/format/safetensors.zig](../../src/format/safetensors.zig) (SafeTensors loader), [src/models/qwen35.zig](../../src/models/qwen35.zig) (format-aware model), [src/backend/kernels/cpu/deltanet.zig](../../src/backend/kernels/cpu/deltanet.zig) (convention-aware kernels)

**Related:** [Chapter 4: Quantization](04-quantization.md#mlx-affine-quantization) (MLX format details)

**Next:** [Chapter 15: Chat Templates →](15-chat-templates.md) | **Back:** [Chapter 13: Batched Dispatch and Fusion ←](13-batched-dispatch-and-fusion.md) | **Product docs:** [Models](../MODELS.md)
