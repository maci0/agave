//! SafeTensors format loader with multi-shard and MLX quantization support.
//! Loads model weights from a directory containing .safetensors shards,
//! config.json for model metadata, and tokenizer.json for vocabulary.
//!
//! Binary layout of each .safetensors shard:
//!   [8 bytes: u64 LE header_json_len]
//!   [header_json_len bytes: UTF-8 JSON]
//!   [tensor data bytes starting at offset 8+header_json_len;
//!    JSON offsets are relative to this region, not file start]

const std = @import("std");
const Allocator = std.mem.Allocator;
const format_mod = @import("format.zig");
const Format = format_mod.Format;
const TensorInfo = format_mod.TensorInfo;
const DType = format_mod.DType;

/// Maximum allowed SafeTensors JSON header size (100 MB).
/// Legitimate models have headers well under 10 MB even with thousands of tensors.
const max_header_json_size: u64 = 100_000_000;
/// Buffer size for tensor name translation (GGUF↔HuggingFace).
const name_buf_size: usize = 256;

// ── Shard & tensor storage ────────────────────────────────────────────────────

/// One mmap'd shard file.
const ShardInfo = struct {
    /// Full mmap'd region (includes the 8-byte length prefix + JSON header).
    data: []align(std.heap.page_size_min) const u8,
    /// Byte offset within `data` where raw tensor bytes begin
    /// (= 8 + json_header_len).
    tensor_base: usize,
};

/// Per-tensor index entry resolved from the shard headers.
const TensorEntry = struct {
    shard_idx: usize,
    /// Byte range within the shard's tensor region.
    data_start: usize,
    data_end: usize,
    dtype: DType,
    n_dims: u32,
    dims: [4]u64,
};

/// Metadata value variants for config.json entries.
const MetaValue = union(enum) {
    string: []const u8,
    uint: u64,
    float: f64,
    bool_val: bool,
};

// ── Public struct ─────────────────────────────────────────────────────────────

/// Loader for a directory of .safetensors shard files plus config.json and
/// tokenizer.json.  Implements the `Format` interface.
pub const SafeTensorsDir = struct {
    allocator: Allocator,

    /// All tensors across all shards, keyed by their original HuggingFace name.
    tensors: std.StringHashMap(TensorEntry),

    /// One entry per mmap'd shard file.
    shard_data: []ShardInfo,

    /// Key-value metadata parsed from config.json.
    config_meta: std.StringHashMap(MetaValue),

    /// Tokenizer vocabulary: index → token string (owned).
    vocab: ?[][]u8,

    /// Tokenizer merge rules: array of "A B" strings (owned).
    merges: ?[][]u8,

    /// All heap-allocated strings we own (keys, values, vocab entries, merges).
    owned_strings: std.ArrayList([]u8),

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Open a directory containing safetensors model shards.
    /// Reads `model.safetensors.index.json`, mmaps each shard, parses headers,
    /// parses `config.json` for metadata, and optionally parses `tokenizer.json`.
    pub fn open(allocator: Allocator, dir_path: []const u8) !SafeTensorsDir {
        var tensors = std.StringHashMap(TensorEntry).init(allocator);
        errdefer tensors.deinit();

        var config_meta = std.StringHashMap(MetaValue).init(allocator);
        errdefer config_meta.deinit();

        var owned_strings: std.ArrayList([]u8) = .empty;
        errdefer {
            for (owned_strings.items) |s| allocator.free(s);
            owned_strings.deinit(allocator);
        }

        // --- 1. Read model.safetensors.index.json ---------------------------
        // The index maps tensor_name → shard_filename.
        // We collect unique shard filenames in insertion order.
        var shard_name_list: std.ArrayList([]const u8) = .empty;
        defer shard_name_list.deinit(allocator);
        var shard_name_to_idx = std.StringHashMap(usize).init(allocator);
        defer shard_name_to_idx.deinit();

        {
            const index_path = try std.fs.path.join(allocator, &.{ dir_path, "model.safetensors.index.json" });
            defer allocator.free(index_path);

            var index_valid = false;
            if (readFile(allocator, index_path)) |index_json| {
                defer allocator.free(index_json);
                // Index parsing is best-effort; fall through to shard discovery on failure.
                parseIndexJson(
                    allocator,
                    index_json,
                    &shard_name_list,
                    &shard_name_to_idx,
                    &owned_strings,
                ) catch |err| {
                    std.log.warn("index.json parse failed (will discover shards): {}", .{err});
                };
                // Verify the first shard actually exists on disk.
                if (shard_name_list.items.len > 0) {
                    const check_path = try std.fs.path.join(allocator, &.{ dir_path, shard_name_list.items[0] });
                    defer allocator.free(check_path);
                    if (std.fs.openFileAbsolute(check_path, .{})) |f| {
                        f.close();
                        index_valid = true;
                    } else |_| {}
                }
            } else |_| {}

            // Fallback: scan directory for model-*.safetensors files.
            if (!index_valid) {
                shard_name_list.clearRetainingCapacity();
                shard_name_to_idx.clearRetainingCapacity();
                try discoverShards(allocator, dir_path, &shard_name_list, &shard_name_to_idx, &owned_strings);
            }

            if (shard_name_list.items.len == 0) return error.FileNotFound;
        }

        // --- 2. mmap each shard file ----------------------------------------
        const shard_count = shard_name_list.items.len;
        const shard_data = try allocator.alloc(ShardInfo, shard_count);
        errdefer allocator.free(shard_data);

        var shards_mmapped: usize = 0;
        errdefer {
            for (shard_data[0..shards_mmapped]) |s| std.posix.munmap(s.data);
        }

        for (shard_name_list.items, 0..) |shard_name, si| {
            const shard_path = try std.fs.path.join(allocator, &.{ dir_path, shard_name });
            defer allocator.free(shard_path);

            const file = try std.fs.openFileAbsolute(shard_path, .{});
            defer file.close();

            const file_size = (try file.stat()).size;
            if (file_size < 8) return error.InvalidSafeTensors;

            const mapped = try std.posix.mmap(
                null,
                file_size,
                std.posix.PROT.READ,
                .{ .TYPE = .SHARED },
                file.handle,
                0,
            );
            const json_len = std.mem.readInt(u64, mapped[0..8], .little);
            if (json_len > max_header_json_size or 8 + json_len > file_size) {
                std.posix.munmap(mapped);
                return error.InvalidSafeTensors;
            }

            shard_data[si] = .{
                .data = mapped,
                .tensor_base = 8 + @as(usize, @intCast(json_len)),
            };
            shards_mmapped += 1;

            const json_bytes = mapped[8 .. 8 + json_len];
            try parseShardHeader(allocator, json_bytes, si, &tensors, &owned_strings);
        }

        // --- 3. Parse config.json -------------------------------------------
        {
            const cfg_path = try std.fs.path.join(allocator, &.{ dir_path, "config.json" });
            defer allocator.free(cfg_path);

            if (readFile(allocator, cfg_path)) |cfg_json| {
                defer allocator.free(cfg_json);
                // Config parsing is best-effort; model will use defaults.
                parseConfigJson(allocator, cfg_json, &config_meta, &owned_strings) catch |err| {
                    std.log.warn("config.json parse failed (using defaults): {}", .{err});
                };
            } else |_| {} // config.json is optional
        }

        // --- 4. Parse tokenizer.json ----------------------------------------
        var vocab: ?[][]u8 = null;
        var merges: ?[][]u8 = null;
        {
            const tok_path = try std.fs.path.join(allocator, &.{ dir_path, "tokenizer.json" });
            defer allocator.free(tok_path);

            if (readFile(allocator, tok_path)) |tok_json| {
                defer allocator.free(tok_json);
                parseTokenizerJson(allocator, tok_json, &vocab, &merges, &owned_strings) catch |err| {
                    std.log.warn("tokenizer.json parse failed: {}", .{err});
                };
            } else |_| {} // tokenizer.json is optional
        }

        return SafeTensorsDir{
            .allocator = allocator,
            .tensors = tensors,
            .shard_data = shard_data,
            .config_meta = config_meta,
            .vocab = vocab,
            .merges = merges,
            .owned_strings = owned_strings,
        };
    }

    /// Returns the number of tensors across all shards.
    pub fn tensorCount(self: *const SafeTensorsDir) u64 {
        return @intCast(self.tensors.count());
    }

    /// Returns the total number of parameters (sum of all tensor element counts).
    /// Uses checked arithmetic to prevent overflow on crafted metadata.
    pub fn totalParams(self: *const SafeTensorsDir) u64 {
        var total: u64 = 0;
        var it = self.tensors.valueIterator();
        while (it.next()) |entry| {
            var n: u64 = 1;
            for (0..entry.n_dims) |i| n = std.math.mul(u64, n, entry.dims[i]) catch std.math.maxInt(u64);
            total = std.math.add(u64, total, n) catch std.math.maxInt(u64);
        }
        return total;
    }

    /// Release all resources: unmap shards, free owned strings, deinit maps.
    pub fn deinit(self: *SafeTensorsDir) void {
        for (self.shard_data) |s| std.posix.munmap(s.data);
        self.allocator.free(self.shard_data);

        if (self.vocab) |v| self.allocator.free(v);
        if (self.merges) |m| self.allocator.free(m);

        for (self.owned_strings.items) |s| self.allocator.free(s);
        self.owned_strings.deinit(self.allocator);

        self.tensors.deinit();
        self.config_meta.deinit();
    }

    /// Return a `Format` interface backed by this loader.
    pub fn format(self: *SafeTensorsDir) Format {
        return .{ .ptr = self, .vtable = &vtable, .is_safetensors = true };
    }

    // ── VTable implementations ────────────────────────────────────────────────

    /// Look up a tensor by name. Tries exact GGUF-style name first, then
    /// translates to HuggingFace-style using known prefixes ("language_model.model.", "model.").
    fn getTensorImpl(ptr: *anyopaque, name: []const u8) ?TensorInfo {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        if (self.lookupStable(name)) |r| return self.entryToInfo(r.key, r.entry);
        // Fallback: translate GGUF-style name to HuggingFace-style.
        // Try both "language_model.model." and "model." prefixes.
        var buf: [name_buf_size]u8 = undefined;
        for (hf_prefixes) |pfx| {
            if (ggufToHfName(name, &buf, pfx)) |hf_name| {
                if (self.lookupStable(hf_name)) |r| return self.entryToInfo(r.key, r.entry);
            }
        }
        return null;
    }

    const StableLookup = struct { key: []const u8, entry: TensorEntry };

    /// Look up a tensor by name, returning the HashMap's stable key (heap-allocated,
    /// lifetime matches SafeTensorsDir) so TensorInfo.name never dangles.
    fn lookupStable(self: *SafeTensorsDir, name: []const u8) ?StableLookup {
        const e = self.tensors.getEntry(name) orelse return null;
        return .{ .key = e.key_ptr.*, .entry = e.value_ptr.* };
    }

    fn entryToInfo(self: *SafeTensorsDir, name: []const u8, entry: TensorEntry) TensorInfo {
        if (entry.shard_idx >= self.shard_data.len) {
            std.log.err("Invalid shard index {d} for tensor {s}", .{ entry.shard_idx, name });
            return TensorInfo{ .name = name, .n_dims = 0, .dims = .{ 0, 0, 0, 0 }, .dtype = .unknown, .data_ptr = self.shard_data[0].data.ptr };
        }
        const shard = self.shard_data[entry.shard_idx];
        const abs_start = std.math.add(usize, shard.tensor_base, entry.data_start) catch {
            std.log.err("Tensor offset overflow for {s}", .{name});
            return TensorInfo{ .name = name, .n_dims = 0, .dims = .{ 0, 0, 0, 0 }, .dtype = .unknown, .data_ptr = shard.data.ptr };
        };
        if (abs_start > shard.data.len) {
            std.log.err("Tensor offset out of bounds for {s}", .{name});
            return TensorInfo{ .name = name, .n_dims = 0, .dims = .{ 0, 0, 0, 0 }, .dtype = .unknown, .data_ptr = shard.data.ptr };
        }
        return TensorInfo{
            .name = name,
            .n_dims = entry.n_dims,
            .dims = entry.dims,
            .dtype = entry.dtype,
            .data_ptr = shard.data[abs_start..].ptr,
        };
    }

    fn getMetaStrImpl(ptr: *anyopaque, key: []const u8) ?[]const u8 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const lookup = self.config_meta.get(key) orelse
            self.config_meta.get(ggufKeyToHf(key) orelse return null) orelse return null;
        return switch (lookup) {
            .string => |s| s,
            else => null,
        };
    }

    fn getMetaU32Impl(ptr: *anyopaque, key: []const u8) ?u32 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const v = self.config_meta.get(key) orelse
            self.config_meta.get(ggufKeyToHf(key) orelse return null) orelse return null;
        return switch (v) {
            .uint => |u| if (u <= std.math.maxInt(u32)) @intCast(u) else null,
            .float => |f| if (f >= 0 and f <= std.math.maxInt(u32)) @intFromFloat(f) else null,
            else => null,
        };
    }

    fn getMetaF32Impl(ptr: *anyopaque, key: []const u8) ?f32 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const v = self.config_meta.get(key) orelse
            self.config_meta.get(ggufKeyToHf(key) orelse return null) orelse return null;
        return switch (v) {
            .float => |f| @floatCast(f),
            .uint => |u| @floatFromInt(u),
            else => null,
        };
    }

    fn getMetaU32ArrayImpl(_: *anyopaque, _: []const u8) ?[]const u32 {
        return null;
    }

    fn getVocabImpl(ptr: *anyopaque) ?[]const []const u8 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const v = self.vocab orelse return null;
        return v;
    }

    fn getMergesImpl(ptr: *anyopaque) ?[]const []const u8 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const m = self.merges orelse return null;
        return m;
    }

    const vtable = Format.VTable{
        .get_tensor = getTensorImpl,
        .get_meta_str = getMetaStrImpl,
        .get_meta_u32 = getMetaU32Impl,
        .get_meta_f32 = getMetaF32Impl,
        .get_meta_u32_array = getMetaU32ArrayImpl,
        .get_vocab = getVocabImpl,
        .get_merges = getMergesImpl,
    };
};

// ── GGUF → HuggingFace name translation ──────────────────────────────────────

/// GGUF layer component → HuggingFace layer component mapping.
const gguf_hf_layer_map = [_]struct { []const u8, []const u8 }{
    .{ "attn_norm", "input_layernorm" },
    .{ "attn_q_norm", "self_attn.q_norm" },
    .{ "attn_k_norm", "self_attn.k_norm" },
    .{ "attn_q", "self_attn.q_proj" },
    .{ "attn_k", "self_attn.k_proj" },
    .{ "attn_v", "self_attn.v_proj" },
    .{ "attn_output", "self_attn.o_proj" },
    .{ "post_attention_norm", "post_attention_layernorm" },
    .{ "ffn_norm", "pre_feedforward_layernorm" },
    .{ "ffn_gate", "mlp.gate_proj" },
    .{ "ffn_up", "mlp.up_proj" },
    .{ "ffn_down", "mlp.down_proj" },
    .{ "post_ffw_norm", "post_feedforward_layernorm" },
    // MoE expert tensors (packed [n_experts, rows, cols])
    .{ "ffn_gate_exps", "mlp.experts.gate_proj" },
    .{ "ffn_up_exps", "mlp.experts.up_proj" },
    .{ "ffn_down_exps", "mlp.experts.down_proj" },
    .{ "ffn_gate_inp", "mlp.router" },
    // MLA attention (DeepSeek2/GLM-4)
    .{ "attn_q_a", "self_attn.q_a_proj" },
    .{ "attn_q_b", "self_attn.q_b_proj" },
    .{ "attn_q_a_norm", "self_attn.q_a_layernorm" },
    .{ "attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa" },
    .{ "attn_kv_a_norm", "self_attn.kv_a_layernorm" },
    .{ "attn_k_b", "self_attn.embed_q" },
    .{ "attn_v_b", "self_attn.unembed_out" },
    // MoE shared expert + routing bias
    .{ "ffn_gate_shexp", "mlp.shared_experts.gate_proj" },
    .{ "ffn_up_shexp", "mlp.shared_experts.up_proj" },
    .{ "ffn_down_shexp", "mlp.shared_experts.down_proj" },
    .{ "exp_probs_b", "mlp.gate.e_score_correction_bias" },
    // DeltaNet SSM (Qwen3.5 linear attention layers)
    .{ "attn_qkv", "linear_attn.in_proj_qkv" },
    .{ "attn_gate", "linear_attn.in_proj_z" },
    .{ "ssm_alpha", "linear_attn.in_proj_a" },
    .{ "ssm_beta", "linear_attn.in_proj_b" },
    .{ "ssm_out", "linear_attn.out_proj" },
    .{ "ssm_a", "linear_attn.A_log" },
    .{ "ssm_conv1d", "linear_attn.conv1d" },
    .{ "ssm_norm", "linear_attn.norm" },
};

/// HuggingFace model prefixes to try (multimodal first, then plain).
const hf_prefixes = [_][]const u8{
    "language_model.model.",
    "model.",
};

/// Translate a GGUF-style tensor name to HuggingFace-style using a given prefix.
/// Returns the translated name written into `buf`, or null if no mapping exists.
fn ggufToHfName(name: []const u8, buf: *[name_buf_size]u8, prefix: []const u8) ?[]const u8 {
    // Top-level tensors
    if (std.mem.startsWith(u8, name, "token_embd.")) {
        const attr = name["token_embd.".len..];
        return std.fmt.bufPrint(buf, "{s}embed_tokens.{s}", .{ prefix, attr }) catch null;
    }
    if (std.mem.startsWith(u8, name, "output_norm.")) {
        const attr = name["output_norm.".len..];
        return std.fmt.bufPrint(buf, "{s}norm.{s}", .{ prefix, attr }) catch null;
    }
    if (std.mem.startsWith(u8, name, "output.")) {
        const attr = name["output.".len..];
        // lm_head sits one level above "model." — strip that part from prefix
        const lm_prefix = if (std.mem.endsWith(u8, prefix, "model."))
            prefix[0 .. prefix.len - "model.".len]
        else
            prefix;
        return std.fmt.bufPrint(buf, "{s}lm_head.{s}", .{ lm_prefix, attr }) catch null;
    }

    // Layer tensors: "blk.{i}.{component}.{attr}"
    if (std.mem.startsWith(u8, name, "blk.")) {
        const rest = name["blk.".len..];
        const dot1 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_str = rest[0..dot1];
        const suffix = rest[dot1 + 1 ..]; // e.g. "attn_q.weight"
        const dot2 = std.mem.indexOfScalar(u8, suffix, '.') orelse {
            // No attribute suffix (e.g., "ssm_a" → "linear_attn.A_log")
            for (gguf_hf_layer_map) |mapping| {
                if (std.mem.eql(u8, suffix, mapping[0])) {
                    return std.fmt.bufPrint(buf, "{s}layers.{s}.{s}", .{ prefix, layer_str, mapping[1] }) catch null;
                }
            }
            return null;
        };
        const component = suffix[0..dot2]; // e.g. "attn_q"
        const attr = suffix[dot2 + 1 ..]; // e.g. "weight"

        // Special case: ssm_dt.bias → linear_attn.dt_bias (HF uses underscore, not dot)
        if (std.mem.eql(u8, component, "ssm_dt") and std.mem.eql(u8, attr, "bias")) {
            return std.fmt.bufPrint(buf, "{s}layers.{s}.linear_attn.dt_bias", .{ prefix, layer_str }) catch null;
        }

        for (gguf_hf_layer_map) |mapping| {
            if (std.mem.eql(u8, component, mapping[0])) {
                return std.fmt.bufPrint(buf, "{s}layers.{s}.{s}.{s}", .{ prefix, layer_str, mapping[1], attr }) catch null;
            }
        }
    }
    return null;
}

// ── GGUF metadata key → HuggingFace config.json key translation ───────────────

/// Map from GGUF-style suffix (after stripping arch prefix) to HF config.json key.
const gguf_hf_meta_map = [_]struct { []const u8, []const u8 }{
    .{ "block_count", "num_hidden_layers" },
    .{ "embedding_length", "hidden_size" },
    .{ "attention.head_count", "num_attention_heads" },
    .{ "attention.head_count_kv", "num_key_value_heads" },
    .{ "attention.key_length", "head_dim" },
    .{ "feed_forward_length", "intermediate_size" },
    .{ "context_length", "max_position_embeddings" },
    .{ "rope.freq_base", "rope_theta" },
    .{ "attention.layer_norm_rms_epsilon", "rms_norm_eps" },
    // Qwen3.5 DeltaNet SSM
    .{ "full_attention_interval", "full_attention_interval" },
    .{ "ssm.conv_kernel", "linear_conv_kernel_dim" },
    .{ "ssm.state_size", "linear_key_head_dim" },
    .{ "ssm.group_count", "linear_num_key_heads" },
    .{ "ssm.time_step_rank", "linear_num_value_heads" },
    .{ "partial_rotary_factor", "partial_rotary_factor" },
};

/// Translate a GGUF-style metadata key to HuggingFace config.json key.
/// Handles both arch-prefixed keys ("gemma3.block_count") and bare keys
/// ("general.architecture", "tokenizer.ggml.eos_token_id").
fn ggufKeyToHf(key: []const u8) ?[]const u8 {
    // "general.architecture" → "model_type"
    if (std.mem.eql(u8, key, "general.architecture")) return "model_type";
    // "tokenizer.ggml.eos_token_id" → "eos_token_id"
    if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id")) return "eos_token_id";

    // "{arch}.{suffix}" → strip arch prefix and look up suffix.
    if (std.mem.indexOfScalar(u8, key, '.')) |dot| {
        const suffix = key[dot + 1 ..];
        for (gguf_hf_meta_map) |mapping| {
            if (std.mem.eql(u8, suffix, mapping[0])) return mapping[1];
        }
    }
    return null;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Scan a directory for `model-*.safetensors` files (or a single `model.safetensors`).
/// Populates shard list in sorted order. Used as fallback when the index JSON
/// is missing or references non-existent shards (e.g. after MLX re-quantization).
fn discoverShards(
    allocator: Allocator,
    dir_path: []const u8,
    shard_list: *std.ArrayList([]const u8),
    shard_to_idx: *std.StringHashMap(usize),
    owned: *std.ArrayList([]u8),
) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    // Collect matching filenames.
    var names: std.ArrayList([]u8) = .empty;
    defer names.deinit(allocator);

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        // Skip index files.
        if (std.mem.endsWith(u8, entry.name, ".index.json.safetensors")) continue;
        const name_copy = try dupeString(allocator, owned, entry.name);
        try names.append(allocator, name_copy);
    }

    // Sort lexicographically so shards are in order (model-00001, model-00002, ...).
    std.mem.sort([]u8, names.items, {}, struct {
        fn lessThan(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);

    for (names.items) |name| {
        const idx = shard_list.items.len;
        try shard_list.append(allocator, name);
        try shard_to_idx.put(name, idx);
    }
}

/// Read an entire file into a heap-allocated slice (caller must free).
fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    const size = (try file.stat()).size;
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const n = try file.readAll(buf);
    if (n != size) return error.UnexpectedEof;
    return buf;
}

/// Duplicate a byte slice into `owned_strings`, return the owned copy.
fn dupeString(allocator: Allocator, owned: *std.ArrayList([]u8), s: []const u8) ![]u8 {
    const copy = try allocator.dupe(u8, s);
    errdefer allocator.free(copy);
    try owned.append(allocator, copy);
    return copy;
}

/// Duplicate a JSON string value, unescaping \\, \", \n, \t, \r, \b, \f, \/, and \uXXXX.
fn dupeUnescaped(allocator: Allocator, owned: *std.ArrayList([]u8), s: []const u8) ![]u8 {
    // Fast path: no backslashes means no escapes
    if (std.mem.indexOfScalar(u8, s, '\\') == null) {
        return dupeString(allocator, owned, s);
    }
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '\\' and i + 1 < s.len) {
            i += 1;
            switch (s[i]) {
                '"' => try buf.append(allocator, '"'),
                '\\' => try buf.append(allocator, '\\'),
                '/' => try buf.append(allocator, '/'),
                'n' => try buf.append(allocator, '\n'),
                't' => try buf.append(allocator, '\t'),
                'r' => try buf.append(allocator, '\r'),
                'b' => try buf.append(allocator, 0x08),
                'f' => try buf.append(allocator, 0x0C),
                'u' => {
                    // \uXXXX — parse 4 hex digits as a UTF-16 code unit
                    if (i + 4 < s.len) {
                        const hex = s[i + 1 .. i + 5];
                        const cp = std.fmt.parseInt(u16, hex, 16) catch {
                            try buf.append(allocator, '\\');
                            try buf.append(allocator, 'u');
                            i += 1;
                            continue;
                        };
                        i += 4; // skip hex digits (the +1 below advances past last)

                        // Handle UTF-16 surrogate pairs (\uD800-\uDBFF followed by \uDC00-\uDFFF)
                        var codepoint: u21 = cp;
                        if (cp >= 0xD800 and cp <= 0xDBFF) {
                            if (i + 6 < s.len and s[i + 1] == '\\' and s[i + 2] == 'u') {
                                const low = std.fmt.parseInt(u16, s[i + 3 .. i + 7], 16) catch 0;
                                if (low >= 0xDC00 and low <= 0xDFFF) {
                                    codepoint = 0x10000 + (@as(u21, cp - 0xD800) << 10) + (low - 0xDC00);
                                    i += 6; // skip \uXXXX low surrogate
                                } else {
                                    try buf.append(allocator, '?');
                                    i += 1;
                                    continue;
                                }
                            } else {
                                try buf.append(allocator, '?');
                                i += 1;
                                continue;
                            }
                        }

                        // Encode as UTF-8
                        var utf8_buf: [4]u8 = undefined;
                        const len = std.unicode.utf8Encode(codepoint, &utf8_buf) catch {
                            try buf.append(allocator, '?');
                            i += 1;
                            continue;
                        };
                        try buf.appendSlice(allocator, utf8_buf[0..len]);
                    } else {
                        try buf.append(allocator, '\\');
                        try buf.append(allocator, 'u');
                    }
                },
                else => {
                    try buf.append(allocator, '\\');
                    try buf.append(allocator, s[i]);
                },
            }
        } else {
            try buf.append(allocator, s[i]);
        }
        i += 1;
    }
    const result = try allocator.dupe(u8, buf.items);
    errdefer allocator.free(result);
    try owned.append(allocator, result);
    return result;
}

// ── Minimal JSON scanner ──────────────────────────────────────────────────────

/// Skip ASCII whitespace; return index of first non-WS byte.
fn skipWs(json: []const u8, pos: usize) usize {
    var i = pos;
    while (i < json.len and (json[i] == ' ' or json[i] == '\t' or
        json[i] == '\n' or json[i] == '\r')) : (i += 1)
    {}
    return i;
}

/// Expect byte `ch` at the next non-WS position; return index after it.
fn expect(json: []const u8, pos: usize, ch: u8) !usize {
    const i = skipWs(json, pos);
    if (i >= json.len or json[i] != ch) return error.JsonUnexpected;
    return i + 1;
}

/// Parse a JSON string literal (including surrounding `"`).
/// Returns the content slice (no quotes) and the position after the closing `"`.
/// Simple escape sequences are passed through as raw bytes — callers that need
/// unescaped values should unescape themselves.
fn parseString(json: []const u8, pos: usize) !struct { val: []const u8, next: usize } {
    var i = skipWs(json, pos);
    if (i >= json.len or json[i] != '"') return error.JsonExpectedString;
    i += 1;
    const start = i;
    while (i < json.len and json[i] != '"') {
        if (json[i] == '\\') i += 1; // skip the escaped character
        i += 1;
    }
    if (i >= json.len) return error.JsonUnterminated;
    return .{ .val = json[start..i], .next = i + 1 };
}

/// Skip over any JSON value (string, number, object, array, literal).
/// Returns the position of the first byte after the value.
fn skipValue(json: []const u8, start_pos: usize) !usize {
    var i = skipWs(json, start_pos);
    if (i >= json.len) return error.JsonUnexpected;
    switch (json[i]) {
        '"' => {
            i += 1;
            while (i < json.len and json[i] != '"') {
                if (json[i] == '\\') i += 1;
                i += 1;
            }
            return if (i < json.len) i + 1 else error.JsonUnterminated;
        },
        '{', '[' => {
            const open = json[i];
            const close: u8 = if (open == '{') '}' else ']';
            var depth: usize = 1;
            i += 1;
            while (i < json.len and depth > 0) : (i += 1) {
                if (json[i] == open) {
                    depth += 1;
                } else if (json[i] == close) {
                    depth -= 1;
                } else if (json[i] == '"') {
                    i += 1;
                    while (i < json.len and json[i] != '"') {
                        if (json[i] == '\\') i += 1;
                        i += 1;
                    }
                }
            }
            return i;
        },
        else => {
            // number / true / false / null
            while (i < json.len and
                json[i] != ',' and json[i] != '}' and json[i] != ']' and
                json[i] != ' ' and json[i] != '\n' and json[i] != '\r' and
                json[i] != '\t') : (i += 1)
            {}
            return i;
        },
    }
}

/// Parse an unsigned integer from a raw JSON token slice.
fn parseU64Slice(s: []const u8) !u64 {
    const trimmed = std.mem.trim(u8, s, " \t\r\n,]})");
    return std.fmt.parseUnsigned(u64, trimmed, 10);
}

/// Map a safetensors dtype string to our DType enum.
fn parseDType(s: []const u8) DType {
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F16")) return .f16;
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    // SafeTensors U32 dtype indicates MLX-quantized packed weights.
    if (std.mem.eql(u8, s, "U32")) return .mlx_q;
    return .unknown; // unsupported dtypes
}

// ── Shard header parser ───────────────────────────────────────────────────────

/// Parse the JSON header of one shard and insert tensor entries into `tensors`.
///
/// Header format:
/// ```json
/// {
///   "__metadata__": {"format": "mlx"},
///   "tensor.name": {
///     "dtype": "BF16",
///     "shape": [dim0, dim1, ...],
///     "data_offsets": [start, end]
///   }, ...
/// }
/// ```
fn parseShardHeader(
    allocator: Allocator,
    json: []const u8,
    shard_idx: usize,
    tensors: *std.StringHashMap(TensorEntry),
    owned: *std.ArrayList([]u8),
) !void {
    var i: usize = try expect(json, 0, '{');

    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == '}') break;

        const name_res = try parseString(json, i);
        i = name_res.next;
        i = try expect(json, i, ':');
        i = skipWs(json, i);

        if (std.mem.eql(u8, name_res.val, "__metadata__")) {
            i = try skipValue(json, i);
        } else {
            // Parse tensor descriptor object.
            i = try expect(json, i, '{');

            var dtype: DType = .unknown;
            var n_dims: u32 = 0;
            var dims: [4]u64 = .{ 0, 0, 0, 0 };
            var data_start: u64 = 0;
            var data_end: u64 = 0;

            while (true) {
                i = skipWs(json, i);
                if (i >= json.len or json[i] == '}') break;

                const key_res = try parseString(json, i);
                i = key_res.next;
                i = try expect(json, i, ':');
                i = skipWs(json, i);

                if (std.mem.eql(u8, key_res.val, "dtype")) {
                    const dt_res = try parseString(json, i);
                    i = dt_res.next;
                    dtype = parseDType(dt_res.val);
                } else if (std.mem.eql(u8, key_res.val, "shape")) {
                    i = try expect(json, i, '[');
                    n_dims = 0;
                    while (n_dims < 4) {
                        i = skipWs(json, i);
                        if (i >= json.len or json[i] == ']') break;
                        const num_start = i;
                        while (i < json.len and json[i] != ',' and json[i] != ']') : (i += 1) {}
                        dims[n_dims] = try parseU64Slice(json[num_start..i]);
                        n_dims += 1;
                        i = skipWs(json, i);
                        if (i < json.len and json[i] == ',') i += 1;
                    }
                    // Skip remaining dimensions beyond 4 (e.g. 5-D vision tensors).
                    while (i < json.len and json[i] != ']') : (i += 1) {}
                    if (i < json.len and json[i] == ']') i += 1;
                } else if (std.mem.eql(u8, key_res.val, "data_offsets")) {
                    i = try expect(json, i, '[');
                    i = skipWs(json, i);
                    const s0 = i;
                    while (i < json.len and json[i] != ',') : (i += 1) {}
                    data_start = try parseU64Slice(json[s0..i]);
                    if (i < json.len and json[i] == ',') i += 1;
                    i = skipWs(json, i);
                    const s1 = i;
                    while (i < json.len and json[i] != ']') : (i += 1) {}
                    data_end = try parseU64Slice(json[s1..i]);
                    if (i < json.len and json[i] == ']') i += 1;
                } else {
                    i = try skipValue(json, i);
                }

                i = skipWs(json, i);
                if (i < json.len and json[i] == ',') i += 1;
            }
            if (i < json.len and json[i] == '}') i += 1;

            const owned_name = try dupeString(allocator, owned, name_res.val);
            try tensors.put(owned_name, TensorEntry{
                .shard_idx = shard_idx,
                .data_start = @intCast(data_start),
                .data_end = @intCast(data_end),
                .dtype = dtype,
                .n_dims = n_dims,
                .dims = dims,
            });
        }

        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
}

// ── Index JSON parser ─────────────────────────────────────────────────────────

/// Parse `model.safetensors.index.json`.
/// We only need the `weight_map` object (tensor_name → shard_filename) to
/// collect the ordered list of unique shard files.
fn parseIndexJson(
    allocator: Allocator,
    json: []const u8,
    shard_list: *std.ArrayList([]const u8),
    shard_to_idx: *std.StringHashMap(usize),
    owned: *std.ArrayList([]u8),
) !void {
    var i: usize = try expect(json, 0, '{');

    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == '}') break;

        const key_res = try parseString(json, i);
        i = key_res.next;
        i = try expect(json, i, ':');
        i = skipWs(json, i);

        if (std.mem.eql(u8, key_res.val, "weight_map")) {
            i = try expect(json, i, '{');
            while (true) {
                i = skipWs(json, i);
                if (i >= json.len or json[i] == '}') break;

                // Tensor name — consumed but not stored.
                const _tname = try parseString(json, i);
                i = _tname.next;
                i = try expect(json, i, ':');

                const shard_res = try parseString(json, i);
                i = shard_res.next;

                if (!shard_to_idx.contains(shard_res.val)) {
                    const owned_shard = try dupeString(allocator, owned, shard_res.val);
                    const idx = shard_list.items.len;
                    try shard_list.append(allocator, owned_shard);
                    try shard_to_idx.put(owned_shard, idx);
                }

                i = skipWs(json, i);
                if (i < json.len and json[i] == ',') i += 1;
            }
            if (i < json.len and json[i] == '}') i += 1;
        } else {
            i = try skipValue(json, i);
        }

        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
}

// ── config.json parser ────────────────────────────────────────────────────────

/// Parse `config.json` into a string → MetaValue map.
/// Top-level scalar values are stored. For multimodal models, scalar values
/// from `text_config` override top-level ones (text model params take priority).
fn parseConfigJson(
    allocator: Allocator,
    json: []const u8,
    meta: *std.StringHashMap(MetaValue),
    owned: *std.ArrayList([]u8),
) !void {
    _ = try parseConfigObject(allocator, json, 0, meta, owned, false);
}

/// Parse a JSON object's scalar values into the meta map.
/// If `is_override` is true, existing keys are overwritten (for nested text_config).
fn parseConfigObject(
    allocator: Allocator,
    json: []const u8,
    start: usize,
    meta: *std.StringHashMap(MetaValue),
    owned: *std.ArrayList([]u8),
    is_override: bool,
) !usize {
    var i: usize = try expect(json, start, '{');

    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == '}') break;

        const key_res = try parseString(json, i);
        i = key_res.next;
        i = try expect(json, i, ':');
        i = skipWs(json, i);

        if (i >= json.len) break;

        // Recurse into text_config / quantization to flatten important nested values.
        if (!is_override and (std.mem.eql(u8, key_res.val, "text_config") or
            std.mem.eql(u8, key_res.val, "quantization")) and json[i] == '{')
        {
            i = try parseConfigObject(allocator, json, i, meta, owned, true);
        } else {
            const owned_key = try dupeString(allocator, owned, key_res.val);

            switch (json[i]) {
                '"' => {
                    const val_res = try parseString(json, i);
                    i = val_res.next;
                    const owned_val = try dupeString(allocator, owned, val_res.val);
                    if (is_override or !meta.contains(owned_key))
                        try meta.put(owned_key, .{ .string = owned_val });
                },
                '{' => {
                    i = try skipValue(json, i);
                },
                '[' => {
                    // For integer arrays (e.g. eos_token_id: [154820, 154827]),
                    // store the first element as a uint value.
                    const arr_start = i;
                    i += 1; // skip '['
                    i = skipWs(json, i);
                    if (i < json.len and json[i] != ']' and json[i] != '"' and json[i] != '{' and json[i] != '[') {
                        // Looks like a number — try to parse the first element.
                        const num_start = i;
                        while (i < json.len and json[i] != ',' and json[i] != ']') : (i += 1) {}
                        if (parseU64Slice(json[num_start..i])) |u| {
                            if (is_override or !meta.contains(owned_key))
                                try meta.put(owned_key, .{ .uint = u });
                        } else |_| {}
                    }
                    // Skip the rest of the array.
                    i = try skipValue(json, arr_start);
                },
                't' => {
                    i = try skipValue(json, i);
                    if (is_override or !meta.contains(owned_key))
                        try meta.put(owned_key, .{ .bool_val = true });
                },
                'f' => {
                    i = try skipValue(json, i);
                    if (is_override or !meta.contains(owned_key))
                        try meta.put(owned_key, .{ .bool_val = false });
                },
                'n' => {
                    i = try skipValue(json, i);
                },
                else => {
                    const num_start = i;
                    while (i < json.len and json[i] != ',' and json[i] != '}' and
                        json[i] != ' ' and json[i] != '\n' and json[i] != '\r' and
                        json[i] != '\t') : (i += 1)
                    {}
                    const num_str = std.mem.trim(u8, json[num_start..i], " \t\r\n");
                    if (std.fmt.parseUnsigned(u64, num_str, 10)) |u| {
                        if (is_override or !meta.contains(owned_key))
                            try meta.put(owned_key, .{ .uint = u });
                    } else |_| {
                        if (std.fmt.parseFloat(f64, num_str)) |f| {
                            if (is_override or !meta.contains(owned_key))
                                try meta.put(owned_key, .{ .float = f });
                        } else |_| {}
                    }
                },
            }
        }

        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
    if (i < json.len and json[i] == '}') i += 1;
    return i;
}

// ── tokenizer.json parser ─────────────────────────────────────────────────────

/// Parse `tokenizer.json`, extracting `model.vocab` and `model.merges`.
///
/// `model.vocab`  — JSON object: token_string → integer id
/// `model.merges` — JSON array of "A B" merge-rule strings
fn parseTokenizerJson(
    allocator: Allocator,
    json: []const u8,
    out_vocab: *?[][]u8,
    out_merges: *?[][]u8,
    owned: *std.ArrayList([]u8),
) !void {
    var i: usize = try expect(json, 0, '{');

    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == '}') break;

        const key_res = try parseString(json, i);
        i = key_res.next;
        i = try expect(json, i, ':');
        i = skipWs(json, i);

        if (std.mem.eql(u8, key_res.val, "model")) {
            i = try expect(json, i, '{');

            while (true) {
                i = skipWs(json, i);
                if (i >= json.len or json[i] == '}') break;

                const mkey_res = try parseString(json, i);
                i = mkey_res.next;
                i = try expect(json, i, ':');
                i = skipWs(json, i);

                if (std.mem.eql(u8, mkey_res.val, "vocab")) {
                    i = try parseVocab(allocator, json, i, out_vocab, owned);
                } else if (std.mem.eql(u8, mkey_res.val, "merges")) {
                    i = try parseMerges(allocator, json, i, out_merges, owned);
                } else {
                    i = try skipValue(json, i);
                }

                i = skipWs(json, i);
                if (i < json.len and json[i] == ',') i += 1;
            }
            if (i < json.len and json[i] == '}') i += 1;
        } else if (std.mem.eql(u8, key_res.val, "added_tokens")) {
            i = try parseAddedTokens(allocator, json, i, out_vocab, owned);
        } else {
            i = try skipValue(json, i);
        }

        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
}

/// Parse the `model.vocab` object (token_string → id) into a sorted array.
/// Returns position after the closing `}`.
fn parseVocab(
    allocator: Allocator,
    json: []const u8,
    start: usize,
    out_vocab: *?[][]u8,
    owned: *std.ArrayList([]u8),
) !usize {
    // First pass: find max id to size the array.
    var max_id: usize = 0;
    {
        var vi = try expect(json, start, '{');
        while (true) {
            vi = skipWs(json, vi);
            if (vi >= json.len or json[vi] == '}') break;
            const _tok = try parseString(json, vi);
            vi = _tok.next;
            vi = try expect(json, vi, ':');
            vi = skipWs(json, vi);
            const id_s = vi;
            while (vi < json.len and json[vi] != ',' and json[vi] != '}' and
                json[vi] != ' ' and json[vi] != '\n' and json[vi] != '\t') : (vi += 1)
            {}
            if (parseU64Slice(json[id_s..vi])) |id| {
                if (id > max_id) max_id = id;
            } else |_| {}
            vi = skipWs(json, vi);
            if (vi < json.len and json[vi] == ',') vi += 1;
        }
    }

    const vocab_size = max_id + 1;
    // Reuse existing array (from added_tokens parsed earlier) if large enough,
    // otherwise allocate a new one and copy existing entries.
    const vocab_arr = if (out_vocab.*) |old| blk: {
        if (old.len >= vocab_size) break :blk old;
        const new = try allocator.alloc([]u8, vocab_size);
        @memcpy(new[0..old.len], old);
        for (new[old.len..]) |*slot| slot.* = @constCast(&[_]u8{});
        allocator.free(old);
        out_vocab.* = new;
        break :blk new;
    } else blk: {
        const new = try allocator.alloc([]u8, vocab_size);
        for (new) |*slot| slot.* = @constCast(&[_]u8{});
        out_vocab.* = new; // Set early so caller can free on error
        break :blk new;
    };

    // Second pass: populate.
    var i = try expect(json, start, '{');
    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == '}') break;
        const tok_res = try parseString(json, i);
        i = tok_res.next;
        i = try expect(json, i, ':');
        i = skipWs(json, i);
        const id_s = i;
        while (i < json.len and json[i] != ',' and json[i] != '}' and
            json[i] != ' ' and json[i] != '\n' and json[i] != '\t') : (i += 1)
        {}
        if (parseU64Slice(json[id_s..i])) |id| {
            if (id < vocab_size) {
                vocab_arr[id] = try dupeUnescaped(allocator, owned, tok_res.val);
            }
        } else |_| {}
        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
    if (i < json.len and json[i] == '}') i += 1;

    return i;
}

/// Parse the `model.merges` array of strings.
/// Returns position after the closing `]`.
fn parseMerges(
    allocator: Allocator,
    json: []const u8,
    start: usize,
    out_merges: *?[][]u8,
    owned: *std.ArrayList([]u8),
) !usize {
    var i = try expect(json, start, '[');
    var list: std.ArrayList([]u8) = .empty;
    // Elements are owned by `owned`; free only the ArrayList wrapper on error.
    errdefer list.deinit(allocator);

    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == ']') break;

        if (json[i] == '"') {
            // String format: "a b"
            const mr = try parseString(json, i);
            i = mr.next;
            const owned_m = try dupeString(allocator, owned, mr.val);
            try list.append(allocator, owned_m);
        } else if (json[i] == '[') {
            // Array format: ["a", "b"] — join with space
            i += 1; // skip '['
            i = skipWs(json, i);
            const first = try parseString(json, i);
            i = first.next;
            i = skipWs(json, i);
            if (i < json.len and json[i] == ',') i += 1;
            i = skipWs(json, i);
            const second = try parseString(json, i);
            i = second.next;
            i = skipWs(json, i);
            if (i < json.len and json[i] == ']') i += 1;
            // Join "a" + " " + "b"
            const joined = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first.val, second.val });
            try owned.append(allocator, joined);
            try list.append(allocator, joined);
        } else {
            return error.UnexpectedToken;
        }

        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
    if (i < json.len and json[i] == ']') i += 1;

    out_merges.* = try list.toOwnedSlice(allocator);
    return i;
}

/// Parse `added_tokens` array from tokenizer.json.
/// Each element is `{"id": N, "content": "...", ...}`.
/// Extends the existing vocab array to include these tokens at their IDs.
fn parseAddedTokens(
    allocator: Allocator,
    json: []const u8,
    start: usize,
    out_vocab: *?[][]u8,
    owned: *std.ArrayList([]u8),
) !usize {
    // First pass: collect (id, content) pairs and find max id.
    const Entry = struct { id: usize, content: []const u8 };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(allocator);

    var i = try expect(json, start, '[');
    while (true) {
        i = skipWs(json, i);
        if (i >= json.len or json[i] == ']') break;

        // Parse one added_token object.
        i = try expect(json, i, '{');
        var token_id: ?usize = null;
        var content: ?[]const u8 = null;
        while (true) {
            i = skipWs(json, i);
            if (i >= json.len or json[i] == '}') break;
            const fk = try parseString(json, i);
            i = fk.next;
            i = try expect(json, i, ':');
            i = skipWs(json, i);

            if (std.mem.eql(u8, fk.val, "id")) {
                const ns = i;
                while (i < json.len and json[i] != ',' and json[i] != '}') : (i += 1) {}
                token_id = @intCast(parseU64Slice(json[ns..i]) catch 0);
            } else if (std.mem.eql(u8, fk.val, "content")) {
                const cr = try parseString(json, i);
                i = cr.next;
                content = cr.val;
            } else {
                i = try skipValue(json, i);
            }
            i = skipWs(json, i);
            if (i < json.len and json[i] == ',') i += 1;
        }
        if (i < json.len and json[i] == '}') i += 1;

        if (token_id != null and content != null) {
            try entries.append(allocator, .{ .id = token_id.?, .content = content.? });
        }
        i = skipWs(json, i);
        if (i < json.len and json[i] == ',') i += 1;
    }
    if (i < json.len and json[i] == ']') i += 1;

    if (entries.items.len == 0) return i;

    // Find max id across existing vocab and added tokens.
    var max_id: usize = if (out_vocab.*) |v| v.len else 0;
    for (entries.items) |e| {
        if (e.id + 1 > max_id) max_id = e.id + 1;
    }

    // Extend or create vocab array.
    if (out_vocab.*) |old_vocab| {
        if (max_id > old_vocab.len) {
            const new_vocab = try allocator.alloc([]u8, max_id);
            @memcpy(new_vocab[0..old_vocab.len], old_vocab);
            for (new_vocab[old_vocab.len..]) |*slot| slot.* = @constCast(&[_]u8{});
            allocator.free(old_vocab);
            out_vocab.* = new_vocab;
        }
    } else {
        const new_vocab = try allocator.alloc([]u8, max_id);
        for (new_vocab) |*slot| slot.* = @constCast(&[_]u8{});
        out_vocab.* = new_vocab;
    }

    // Insert added tokens.
    const vocab = out_vocab.*.?;
    for (entries.items) |e| {
        if (e.id < vocab.len) {
            vocab[e.id] = try dupeUnescaped(allocator, owned, e.content);
        }
    }

    return i;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test "parseDType" {
    try std.testing.expectEqual(DType.f32, parseDType("F32"));
    try std.testing.expectEqual(DType.bf16, parseDType("BF16"));
    try std.testing.expectEqual(DType.f16, parseDType("F16"));
    try std.testing.expectEqual(DType.mlx_q, parseDType("U32"));
    try std.testing.expectEqual(DType.unknown, parseDType("U8"));
}

test "parseShardHeader basic" {
    const allocator = std.testing.allocator;
    const json =
        \\{"__metadata__":{"format":"mlx"},"model.embed_tokens.weight":{"dtype":"BF16","shape":[32000,4096],"data_offsets":[0,262144000]}}
    ;

    var tensors = std.StringHashMap(TensorEntry).init(allocator);
    defer tensors.deinit();
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    try parseShardHeader(allocator, json, 0, &tensors, &owned);

    try std.testing.expectEqual(@as(usize, 1), tensors.count());
    const entry = tensors.get("model.embed_tokens.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(DType.bf16, entry.dtype);
    try std.testing.expectEqual(@as(u32, 2), entry.n_dims);
    try std.testing.expectEqual(@as(u64, 32000), entry.dims[0]);
    try std.testing.expectEqual(@as(u64, 4096), entry.dims[1]);
    try std.testing.expectEqual(@as(usize, 0), entry.data_start);
    try std.testing.expectEqual(@as(usize, 262144000), entry.data_end);
}

test "parseConfigJson scalars" {
    const allocator = std.testing.allocator;
    const json =
        \\{"hidden_size":4096,"num_hidden_layers":32,"rms_norm_eps":1e-05,"model_type":"llama","tie_word_embeddings":true}
    ;
    var meta = std.StringHashMap(MetaValue).init(allocator);
    defer meta.deinit();
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    try parseConfigJson(allocator, json, &meta, &owned);

    const hs = meta.get("hidden_size") orelse return error.Missing;
    try std.testing.expectEqual(@as(u64, 4096), hs.uint);

    const mt = meta.get("model_type") orelse return error.Missing;
    try std.testing.expectEqualStrings("llama", mt.string);

    const tie = meta.get("tie_word_embeddings") orelse return error.Missing;
    try std.testing.expect(tie.bool_val);
}

test "parseIndexJson collects shards in order" {
    const allocator = std.testing.allocator;
    const json =
        \\{"metadata":{},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors","c.weight":"model-00001-of-00002.safetensors"}}
    ;
    var shard_list: std.ArrayList([]const u8) = .empty;
    defer shard_list.deinit(allocator);
    var shard_to_idx = std.StringHashMap(usize).init(allocator);
    defer shard_to_idx.deinit();
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    try parseIndexJson(allocator, json, &shard_list, &shard_to_idx, &owned);

    try std.testing.expectEqual(@as(usize, 2), shard_list.items.len);
    try std.testing.expectEqualStrings("model-00001-of-00002.safetensors", shard_list.items[0]);
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", shard_list.items[1]);
}

test "parseShardHeader 5D shape" {
    const allocator = std.testing.allocator;
    const json =
        \\{"__metadata__":{},"vision.proj.weight":{"dtype":"BF16","shape":[1152,2,16,16,3],"data_offsets":[0,1769472]}}
    ;

    var tensors = std.StringHashMap(TensorEntry).init(allocator);
    defer tensors.deinit();
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    try parseShardHeader(allocator, json, 0, &tensors, &owned);

    try std.testing.expectEqual(@as(usize, 1), tensors.count());
    const entry = tensors.get("vision.proj.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(DType.bf16, entry.dtype);
    // Only first 4 dims are stored; 5th is skipped gracefully
    try std.testing.expectEqual(@as(u32, 4), entry.n_dims);
    try std.testing.expectEqual(@as(u64, 1152), entry.dims[0]);
    try std.testing.expectEqual(@as(u64, 2), entry.dims[1]);
    try std.testing.expectEqual(@as(u64, 16), entry.dims[2]);
    try std.testing.expectEqual(@as(u64, 16), entry.dims[3]);
}

test "dupeUnescaped surrogate pair" {
    const allocator = std.testing.allocator;
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    // \uD83D\uDE00 = U+1F600 = 😀 = F0 9F 98 80 in UTF-8
    const result = try dupeUnescaped(allocator, &owned, "\\uD83D\\uDE00");
    try std.testing.expectEqualSlices(u8, "\xF0\x9F\x98\x80", result);
}

test "dupeUnescaped lone high surrogate" {
    const allocator = std.testing.allocator;
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    // Lone high surrogate with no low surrogate following → '?'
    const result = try dupeUnescaped(allocator, &owned, "\\uD83Dhello");
    try std.testing.expectEqual(@as(u8, '?'), result[0]);
}

test "dupeUnescaped basic escapes" {
    const allocator = std.testing.allocator;
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    const result_n = try dupeUnescaped(allocator, &owned, "hello\\nworld");
    try std.testing.expectEqualSlices(u8, "hello\nworld", result_n);

    const result_t = try dupeUnescaped(allocator, &owned, "a\\tb");
    try std.testing.expectEqualSlices(u8, "a\tb", result_t);

    const result_r = try dupeUnescaped(allocator, &owned, "a\\rb");
    try std.testing.expectEqualSlices(u8, "a\rb", result_r);

    const result_bs = try dupeUnescaped(allocator, &owned, "a\\\\b");
    try std.testing.expectEqualSlices(u8, "a\\b", result_bs);

    const result_q = try dupeUnescaped(allocator, &owned, "a\\\"b");
    try std.testing.expectEqualSlices(u8, "a\"b", result_q);
}
