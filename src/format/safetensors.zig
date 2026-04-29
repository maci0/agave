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
/// Maximum allowed auxiliary JSON file size (200 MB — covers large tokenizer.json).
const max_json_file_size: u64 = 200_000_000;
/// Maximum vocabulary size to prevent OOM from crafted tokenizer.json with huge IDs.
const max_vocab_size: usize = 10_000_000;
/// Maximum allowed shard count (prevents OOM from crafted index.json).
const max_shard_count: usize = 10_000;
/// Buffer size for tensor name translation (GGUF↔HuggingFace).
const name_buf_size: usize = 256;
/// Maximum JSON nesting depth to prevent stack exhaustion from crafted inputs.
const max_json_depth: usize = 128;

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

    /// Fused NVFP4 expert tensor entries (GGUF-named, separate from shard-parsed tensors).
    fused_tensors: std.StringHashMap(TensorEntry),
    /// Repacked f32 arrays (global_scale, input_scale) allocated during fusion.
    repacked_f32: std.ArrayList([]f32),

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Open a directory containing safetensors model shards.
    /// Reads `model.safetensors.index.json`, mmaps each shard, parses headers,
    /// parses `config.json` for metadata, and optionally parses `tokenizer.json`.
    pub fn open(allocator: Allocator, dir_path: []const u8) !SafeTensorsDir {
        var tensors = std.StringHashMap(TensorEntry).init(allocator);
        errdefer tensors.deinit();
        var fused_tensors = std.StringHashMap(TensorEntry).init(allocator);
        errdefer fused_tensors.deinit();
        var repacked_f32: std.ArrayList([]f32) = .empty;
        errdefer {
            for (repacked_f32.items) |s| allocator.free(s);
            repacked_f32.deinit(allocator);
        }

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
                    const check_z = try allocator.dupeZ(u8, check_path);
                    defer allocator.free(check_z);
                    const check_fd = std.posix.system.open(check_z.ptr, .{}, @as(std.posix.mode_t, 0));
                    if (check_fd >= 0) {
                        _ = std.c.close(check_fd);
                        index_valid = true;
                    }
                }
            } else |_| {}

            // Fallback: scan directory for model-*.safetensors files.
            if (!index_valid) {
                std.log.debug("[st] index invalid, discovering shards in '{s}'", .{dir_path});
                shard_name_list.clearRetainingCapacity();
                shard_name_to_idx.clearRetainingCapacity();
                try discoverShards(allocator, dir_path, &shard_name_list, &shard_name_to_idx, &owned_strings);
                std.log.debug("[st] discovered {d} shards", .{shard_name_list.items.len});
            } else {
                std.log.debug("[st] index valid with {d} shards", .{shard_name_list.items.len});
            }

            if (shard_name_list.items.len == 0) return error.FileNotFound;
        }

        // --- 2. mmap each shard file ----------------------------------------
        std.log.debug("[st] about to mmap {d} shards", .{shard_name_list.items.len});
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

            std.log.debug("[st] opening shard {d}: '{s}'", .{ si, shard_path });
            const shard_z = try allocator.dupeZ(u8, shard_path);
            defer allocator.free(shard_z);
            const fd = std.posix.system.open(shard_z.ptr, .{}, @as(std.posix.mode_t, 0));
            if (fd < 0) {
                // Skip missing optional shards (visual, MTP) — only text weights needed
                shard_data[si] = .{ .data = &.{}, .tensor_base = 0 };
                continue;
            }
            defer _ = std.c.close(fd);

            const file_size: usize = blk: {
                var s: std.posix.Stat = undefined;
                if (std.c.fstat(fd, &s) != 0) return error.FileNotFound;
                break :blk @intCast(s.size);
            };
            if (file_size < 8) return error.InvalidSafeTensors;

            const mapped = try std.posix.mmap(
                null,
                file_size,
                .{ .READ = true },
                .{ .TYPE = .SHARED },
                fd,
                0,
            );
            // Hint sequential access for weight loading — enables OS readahead and reduces page faults.
            std.posix.madvise(mapped.ptr, mapped.len, std.posix.MADV.SEQUENTIAL) catch {};
            const json_len = std.mem.readInt(u64, mapped[0..8], .little);
            const total_header_size = std.math.add(u64, 8, json_len) catch return error.InvalidSafeTensors;
            if (total_header_size > max_header_json_size or total_header_size > file_size) {
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

        // --- 5. Fuse compressed-tensors NVFP4 per-expert tensors -------
        // Detects per-expert weight_packed/weight_scale/weight_global_scale
        // patterns and creates synthetic GGUF-named fused entries.
        fuseNvfp4Experts(allocator, &tensors, &fused_tensors, &repacked_f32, shard_data, &config_meta, &owned_strings) catch |err| {
            std.log.warn("NVFP4 expert fusion failed (will use individual lookup): {}", .{err});
        };
        if (fused_tensors.count() > 0)
            std.log.info("[st] fused {d} NVFP4 entries", .{fused_tensors.count()});

        return SafeTensorsDir{
            .allocator = allocator,
            .tensors = tensors,
            .fused_tensors = fused_tensors,
            .repacked_f32 = repacked_f32,
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

    /// Returns the total mmap'd file size in bytes (sum of all shard lengths).
    pub fn totalBytes(self: *const SafeTensorsDir) usize {
        var total: usize = 0;
        for (self.shard_data) |shard| total += shard.data.len;
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
        self.fused_tensors.deinit();
        for (self.repacked_f32.items) |s| self.allocator.free(s);
        self.repacked_f32.deinit(self.allocator);
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
        // Check fused NVFP4 entries first (repacked expert weights)
        if (self.fused_tensors.getEntry(name)) |e| {
            return self.entryToInfo(e.key_ptr.*, e.value_ptr.*);
        }
        if (self.lookupStable(name)) |r| return self.entryToInfo(r.key, r.entry);
        var buf: [name_buf_size]u8 = undefined;
        for (hf_prefixes) |pfx| {
            var iter = ggufToHfNameIter(name, pfx);
            while (iter.next(&buf)) |hf_name| {
                if (self.lookupStable(hf_name)) |r| return self.entryToInfo(r.key, r.entry);
            }
        }
        // Log ALL tensor misses
        return null;
    }

    const StableLookup = struct { key: []const u8, entry: TensorEntry };

    /// Look up a tensor by name, returning the HashMap's stable key (heap-allocated,
    /// lifetime matches SafeTensorsDir) so TensorInfo.name never dangles.
    fn lookupStable(self: *SafeTensorsDir, name: []const u8) ?StableLookup {
        const e = self.tensors.getEntry(name) orelse return null;
        return .{ .key = e.key_ptr.*, .entry = e.value_ptr.* };
    }

    fn entryToInfo(self: *SafeTensorsDir, name: []const u8, entry: TensorEntry) ?TensorInfo {
        // Sentinel shard_idx = max_shard_count: data_start is an absolute pointer
        // to repacked contiguous buffer (used for non-contiguous NVFP4 experts).
        if (entry.shard_idx == max_shard_count) {
            const data_ptr: [*]const u8 = @ptrFromInt(entry.data_start);
            return TensorInfo{
                .name = name,
                .n_dims = entry.n_dims,
                .dims = .{
                    @intCast(entry.dims[0]),
                    @intCast(entry.dims[1]),
                    @intCast(entry.dims[2]),
                    @intCast(entry.dims[3]),
                },
                .dtype = entry.dtype,
                .data_ptr = data_ptr,
            };
        }
        if (entry.shard_idx >= self.shard_data.len) {
            std.log.err("Invalid shard index {d} for tensor {s} (have {d} shards)", .{ entry.shard_idx, name, self.shard_data.len });
            return null;
        }
        const shard = self.shard_data[entry.shard_idx];
        const abs_start = std.math.add(usize, shard.tensor_base, entry.data_start) catch {
            std.log.err("Tensor offset overflow for {s}", .{name});
            return null;
        };
        const data_len = std.math.sub(usize, entry.data_end, entry.data_start) catch {
            std.log.err("Tensor data_end < data_start for {s}", .{name});
            return null;
        };
        const abs_end = std.math.add(usize, abs_start, data_len) catch {
            std.log.err("Tensor end overflow for {s}", .{name});
            return null;
        };
        if (abs_end > shard.data.len) {
            std.log.err("Tensor data exceeds shard bounds for {s} (end={d}, shard_size={d})", .{ name, abs_end, shard.data.len });
            return null;
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
            lookupMetaAllTranslations(&self.config_meta, key) orelse return null;
        return switch (lookup) {
            .string => |s| s,
            else => null,
        };
    }

    fn getMetaU32Impl(ptr: *anyopaque, key: []const u8) ?u32 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const v = self.config_meta.get(key) orelse
            lookupMetaAllTranslations(&self.config_meta, key) orelse return null;
        return switch (v) {
            .uint => |u| if (u <= std.math.maxInt(u32)) @intCast(u) else null,
            .float => |f| if (f >= 0 and f <= std.math.maxInt(u32)) @intFromFloat(f) else null,
            else => null,
        };
    }

    fn getMetaF32Impl(ptr: *anyopaque, key: []const u8) ?f32 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        const v = self.config_meta.get(key) orelse
            lookupMetaAllTranslations(&self.config_meta, key) orelse return null;
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
        return self.vocab;
    }

    fn getMergesImpl(ptr: *anyopaque) ?[]const []const u8 {
        const self: *SafeTensorsDir = @ptrCast(@alignCast(ptr));
        return self.merges;
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
    .{ "ffn_gate_inp", "mlp.gate" },
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
    // Alternate shared expert naming (compressed-tensors: singular "shared_expert")
    .{ "ffn_gate_shexp", "mlp.shared_expert.gate_proj" },
    .{ "ffn_up_shexp", "mlp.shared_expert.up_proj" },
    .{ "ffn_down_shexp", "mlp.shared_expert.down_proj" },
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
    "model.language_model.",
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
        // lm_head is at top level in HF models — try without any prefix first,
        // then with stripped prefix for nested models (model.language_model. → "")
        return std.fmt.bufPrint(buf, "lm_head.{s}", .{attr}) catch null;
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

/// Iterator over all possible GGUF→HF name translations for a layer tensor.
/// Handles components with multiple HF mappings (e.g., ffn_gate_inp → mlp.router, mlp.gate).
const GgufHfNameIter = struct {
    name: []const u8,
    prefix: []const u8,
    map_idx: usize = 0,
    /// True if this is a top-level tensor (token_embd, output_norm, output).
    /// Top-level tensors have exactly one translation, so we emit it once and stop.
    is_toplevel: bool = false,
    toplevel_emitted: bool = false,

    fn next(self: *GgufHfNameIter, buf: *[name_buf_size]u8) ?[]const u8 {
        // Top-level tensors: delegate to ggufToHfName (one translation only).
        if (self.is_toplevel) {
            if (self.toplevel_emitted) return null;
            self.toplevel_emitted = true;
            return ggufToHfName(self.name, buf, self.prefix);
        }

        // Layer tensors: iterate all matching entries in gguf_hf_layer_map.
        if (!std.mem.startsWith(u8, self.name, "blk.")) return null;
        const rest = self.name["blk.".len..];
        const dot1 = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_str = rest[0..dot1];
        const suffix = rest[dot1 + 1 ..];
        const dot2 = std.mem.indexOfScalar(u8, suffix, '.') orelse {
            // No attribute suffix (e.g., "ssm_a")
            while (self.map_idx < gguf_hf_layer_map.len) {
                const mapping = gguf_hf_layer_map[self.map_idx];
                self.map_idx += 1;
                if (std.mem.eql(u8, suffix, mapping[0])) {
                    return std.fmt.bufPrint(buf, "{s}layers.{s}.{s}", .{ self.prefix, layer_str, mapping[1] }) catch null;
                }
            }
            return null;
        };
        const component = suffix[0..dot2];
        const attr = suffix[dot2 + 1 ..];

        // Special case: ssm_dt.bias
        if (self.map_idx == 0 and std.mem.eql(u8, component, "ssm_dt") and std.mem.eql(u8, attr, "bias")) {
            self.map_idx = gguf_hf_layer_map.len; // Exhaust after this
            return std.fmt.bufPrint(buf, "{s}layers.{s}.linear_attn.dt_bias", .{ self.prefix, layer_str }) catch null;
        }

        while (self.map_idx < gguf_hf_layer_map.len) {
            const mapping = gguf_hf_layer_map[self.map_idx];
            self.map_idx += 1;
            if (std.mem.eql(u8, component, mapping[0])) {
                return std.fmt.bufPrint(buf, "{s}layers.{s}.{s}.{s}", .{ self.prefix, layer_str, mapping[1], attr }) catch null;
            }
        }
        return null;
    }
};

/// Create an iterator over all possible GGUF→HF translations for a tensor name.
fn ggufToHfNameIter(name: []const u8, prefix: []const u8) GgufHfNameIter {
    const is_toplevel = std.mem.startsWith(u8, name, "token_embd.") or
        std.mem.startsWith(u8, name, "output_norm.") or
        std.mem.startsWith(u8, name, "output.");
    return .{ .name = name, .prefix = prefix, .is_toplevel = is_toplevel };
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
    // MoE configuration
    .{ "expert_count", "num_experts" },
    .{ "expert_count", "n_routed_experts" },
    .{ "expert_used_count", "num_experts_per_tok" },
    .{ "expert_feed_forward_length", "moe_intermediate_size" },
    .{ "expert_shared_feed_forward_length", "shared_expert_intermediate_size" },
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
    // Try all possible dot positions to handle arch names with dots (e.g. "qwen3_5_moe_text").
    {
        var pos: usize = 0;
        while (std.mem.indexOfScalarPos(u8, key, pos, '.')) |dot| {
            const suffix = key[dot + 1 ..];
            for (gguf_hf_meta_map) |mapping| {
                if (std.mem.eql(u8, suffix, mapping[0])) return mapping[1];
            }
            pos = dot + 1;
        }
    }
    return null;
}

/// GGUF metadata key aliases: alternative HF config.json keys for the same concept.
/// When the primary translation from gguf_hf_meta_map fails, try these alternatives.
const gguf_hf_meta_aliases = [_]struct { []const u8, []const u8 }{
    .{ "expert_count", "num_local_experts" },
    .{ "expert_count", "num_experts" },
    .{ "expert_count", "num_experts" },
};

/// Try all possible GGUF→HF translations for a metadata key, including aliases.
/// Returns the first matching MetaValue from the config map, or null.
fn lookupMetaAllTranslations(config_meta: *const std.StringHashMap(MetaValue), key: []const u8) ?MetaValue {
    // Primary translation
    if (ggufKeyToHf(key)) |hf_key| {
        if (config_meta.get(hf_key)) |v| return v;
    }
    // Try aliases: extract GGUF suffix at each dot position
    {
        var pos2: usize = 0;
        while (std.mem.indexOfScalarPos(u8, key, pos2, '.')) |dot| {
            const suffix = key[dot + 1 ..];
            for (gguf_hf_meta_aliases) |alias| {
                if (std.mem.eql(u8, suffix, alias[0])) {
                    if (config_meta.get(alias[1])) |v| return v;
                }
            }
            pos2 = dot + 1;
        }
    }
    return null;
}

// ── Compressed-tensors NVFP4 expert fusion ──────────────────────────────────

/// NVFP4 packing: 2 elements per byte (4-bit nibbles).
const nvfp4_values_per_byte: usize = 2;
/// NVFP4 group size: 1 FP8 scale per 16 elements.
const nvfp4_scale_group_size: usize = 16;

/// Maximum supported expert count for NVFP4 fusion scan.
const max_nvfp4_experts: usize = 1024;
/// Buffer size for composing per-expert tensor names during fusion scan.
const fusion_name_buf_size: usize = 256;

/// Compressed-tensors projection types to scan for per-expert NVFP4 tensors.
const ct_projections = [_]struct { hf: []const u8, gguf_weight: []const u8, gguf_scales: []const u8 }{
    .{ .hf = "gate_proj", .gguf_weight = "ffn_gate_exps.weight", .gguf_scales = "ffn_gate_exps.scales" },
    .{ .hf = "up_proj", .gguf_weight = "ffn_up_exps.weight", .gguf_scales = "ffn_up_exps.scales" },
    .{ .hf = "down_proj", .gguf_weight = "ffn_down_exps.weight", .gguf_scales = "ffn_down_exps.scales" },
};

/// Shared expert projection types — same pattern but no expert index.
const ct_shared_projections = [_]struct { hf: []const u8, gguf_weight: []const u8, gguf_scales: []const u8 }{
    .{ .hf = "gate_proj", .gguf_weight = "ffn_gate_shexp.weight", .gguf_scales = "ffn_gate_shexp.scales" },
    .{ .hf = "up_proj", .gguf_weight = "ffn_up_shexp.weight", .gguf_scales = "ffn_up_shexp.scales" },
    .{ .hf = "down_proj", .gguf_weight = "ffn_down_shexp.weight", .gguf_scales = "ffn_down_shexp.scales" },
};

/// Detect compressed-tensors NVFP4 per-expert tensor patterns and create
/// synthetic GGUF-named fused entries that the model code can look up.
///
/// The compressed-tensors format stores each expert as separate tensors:
///   layers.{l}.mlp.experts.{i}.{proj}.weight_packed   — U8 [out, in/2]
///   layers.{l}.mlp.experts.{i}.{proj}.weight_scale     — F8_E4M3 [out, in/16]
///   layers.{l}.mlp.experts.{i}.{proj}.weight_global_scale — F32 [1]
///
/// This function creates synthetic fused entries:
///   blk.{l}.ffn_{proj}_exps.weight — nvfp4 dtype, pointing to expert 0's weight_packed
///   blk.{l}.ffn_{proj}_exps.scales — fp8_e4m3, pointing to expert 0's weight_scale
///
/// Expert stride is embedded in dims as [out, in/2, n_experts] so
/// expertWeightStride() computes per-expert = dims[0]*dims[1] bytes.
///
/// Additionally handles shared expert tensors and stores global_scale values
/// as f32 tensors accessible via blk.{l}.ffn_{proj}_exps.global_scale.
fn fuseNvfp4Experts(
    allocator: Allocator,
    tensors: *std.StringHashMap(TensorEntry),
    fused: *std.StringHashMap(TensorEntry),
    repacked_f32: *std.ArrayList([]f32),
    shard_data: []ShardInfo,
    config_meta: *std.StringHashMap(MetaValue),
    owned: *std.ArrayList([]u8),
) !void {
    // Determine number of layers from config metadata.
    const n_layers: u32 = blk: {
        if (config_meta.get("num_hidden_layers")) |v| {
            switch (v) {
                .uint => |u| break :blk if (u <= std.math.maxInt(u32)) @intCast(u) else return,
                .float => |f| break :blk if (f >= 0 and f <= std.math.maxInt(u32)) @intFromFloat(f) else return,
                else => return,
            }
        }
        return; // Can't determine layer count
    };

    // Detect compressed-tensors NVFP4 by probing for weight_packed tensors.
    // Try expert (MoE) and dense (non-MoE) tensor patterns with all prefixes.
    const prefix = for (hf_prefixes) |pfx| {
        var probe_buf: [fusion_name_buf_size]u8 = undefined;
        // MoE: experts.0.gate_proj.weight_packed
        if (std.fmt.bufPrint(&probe_buf, "{s}layers.0.mlp.experts.0.gate_proj.weight_packed", .{pfx}) catch null) |name| {
            if (tensors.contains(name)) break pfx;
        }
        // Dense: gate_proj.weight_packed (no expert index)
        if (std.fmt.bufPrint(&probe_buf, "{s}layers.0.mlp.gate_proj.weight_packed", .{pfx}) catch null) |name| {
            if (tensors.contains(name)) break pfx;
        }
        // Dense attention: self_attn.q_proj.weight_packed
        if (std.fmt.bufPrint(&probe_buf, "{s}layers.0.self_attn.q_proj.weight_packed", .{pfx}) catch null) |name| {
            if (tensors.contains(name)) break pfx;
        }
    } else return;

    std.log.info("[st] detected compressed-tensors NVFP4 format, fusing tensors", .{});

    // Determine expert count from config (0 = dense model, no MoE).
    const n_experts: u32 = blk: {
        inline for (.{ "n_routed_experts", "num_local_experts", "num_experts" }) |key| {
            if (config_meta.get(key)) |v| {
                switch (v) {
                    .uint => |u| break :blk if (u <= std.math.maxInt(u32)) @intCast(u) else 0,
                    .float => |fv| break :blk if (fv >= 0 and fv <= std.math.maxInt(u32)) @intFromFloat(fv) else 0,
                    else => {},
                }
            }
        }
        break :blk 0; // Dense model (no experts)
    };

    // Process each layer.
    for (0..n_layers) |li| {
        // MoE expert projections (gate/up/down per expert)
        if (n_experts > 0 and n_experts <= max_nvfp4_experts) {
            for (ct_projections) |proj| {
                try fuseOneProjection(allocator, tensors, fused, repacked_f32, shard_data, owned, prefix, @intCast(li), n_experts, proj.hf, proj.gguf_weight, proj.gguf_scales, true);
            }
            for (ct_shared_projections) |proj| {
                try fuseSharedExpertProjection(allocator, tensors, fused, repacked_f32, shard_data, owned, prefix, @intCast(li), proj.hf, proj.gguf_weight, proj.gguf_scales);
            }
        }

        // Dense MLP projections (non-MoE gate/up/down)
        const ct_dense_mlp = [_]struct { hf: []const u8, gguf: []const u8 }{
            .{ .hf = "mlp.gate_proj", .gguf = "ffn_gate" },
            .{ .hf = "mlp.up_proj", .gguf = "ffn_up" },
            .{ .hf = "mlp.down_proj", .gguf = "ffn_down" },
        };
        for (ct_dense_mlp) |proj| {
            try fuseSingleNvfp4Tensor(allocator, tensors, fused, owned, prefix, @intCast(li), proj.hf, proj.gguf);
        }

        // Attention projections
        const ct_attn_projs = [_]struct { hf: []const u8, gguf: []const u8 }{
            .{ .hf = "self_attn.q_proj", .gguf = "attn_q" },
            .{ .hf = "self_attn.k_proj", .gguf = "attn_k" },
            .{ .hf = "self_attn.v_proj", .gguf = "attn_v" },
            .{ .hf = "self_attn.o_proj", .gguf = "attn_output" },
        };
        for (ct_attn_projs) |proj| {
            try fuseSingleNvfp4Tensor(allocator, tensors, fused, owned, prefix, @intCast(li), proj.hf, proj.gguf);
        }

        // DeltaNet projections
        const ct_deltanet_projs = [_]struct { hf: []const u8, gguf: []const u8 }{
            .{ .hf = "linear_attn.in_proj_qkv", .gguf = "attn_qkv" },
            .{ .hf = "linear_attn.in_proj_z", .gguf = "attn_gate" },
            .{ .hf = "linear_attn.in_proj_a", .gguf = "ssm_alpha" },
            .{ .hf = "linear_attn.in_proj_b", .gguf = "ssm_beta" },
            .{ .hf = "linear_attn.out_proj", .gguf = "ssm_out" },
        };
        for (ct_deltanet_projs) |proj| {
            try fuseSingleNvfp4Tensor(allocator, tensors, fused, owned, prefix, @intCast(li), proj.hf, proj.gguf);
        }
    }
}

/// Fuse a single NVFP4 compressed-tensors weight+scale into a GGUF-named entry.
/// Used for non-expert tensors (attention projections, DeltaNet projections).
fn fuseSingleNvfp4Tensor(
    allocator: Allocator,
    tensors: *std.StringHashMap(TensorEntry),
    fused: *std.StringHashMap(TensorEntry),
    owned: *std.ArrayList([]u8),
    prefix: []const u8,
    layer: u32,
    hf_name: []const u8,
    gguf_component: []const u8,
) !void {
    var name_buf: [fusion_name_buf_size]u8 = undefined;
    var s_buf: [fusion_name_buf_size]u8 = undefined;

    const w_name = std.fmt.bufPrint(&name_buf, "{s}layers.{d}.{s}.weight_packed", .{ prefix, layer, hf_name }) catch return;
    const w_entry = tensors.get(w_name) orelse return;

    const s_name = std.fmt.bufPrint(&s_buf, "{s}layers.{d}.{s}.weight_scale", .{ prefix, layer, hf_name }) catch return;
    const s_entry = tensors.get(s_name) orelse return;

    const w_rows = if (w_entry.n_dims >= 1) w_entry.dims[0] else return;
    const w_cols = if (w_entry.n_dims >= 2) w_entry.dims[1] else return;
    const s_rows = if (s_entry.n_dims >= 1) s_entry.dims[0] else return;
    const s_cols = if (s_entry.n_dims >= 2) s_entry.dims[1] else return;

    // Create GGUF weight entry: blk.{l}.{component}.weight
    var gguf_w_buf: [fusion_name_buf_size]u8 = undefined;
    const gguf_w = std.fmt.bufPrint(&gguf_w_buf, "blk.{d}.{s}.weight", .{ layer, gguf_component }) catch return;
    const owned_w = try dupeString(allocator, owned, gguf_w);
    try fused.put(owned_w, TensorEntry{
        .shard_idx = w_entry.shard_idx,
        .data_start = w_entry.data_start,
        .data_end = w_entry.data_end,
        .dtype = .nvfp4,
        .n_dims = 2,
        .dims = .{ w_rows, w_cols, 0, 0 },
    });

    // Create GGUF scale entry: blk.{l}.{component}.scales
    var gguf_s_buf: [fusion_name_buf_size]u8 = undefined;
    const gguf_s = std.fmt.bufPrint(&gguf_s_buf, "blk.{d}.{s}.scales", .{ layer, gguf_component }) catch return;
    const owned_s = try dupeString(allocator, owned, gguf_s);
    try fused.put(owned_s, TensorEntry{
        .shard_idx = s_entry.shard_idx,
        .data_start = s_entry.data_start,
        .data_end = s_entry.data_end,
        .dtype = .fp8_e4m3,
        .n_dims = 2,
        .dims = .{ s_rows, s_cols, 0, 0 },
    });

    // Create GGUF global_scale entry: blk.{l}.{component}.global_scale
    var gs_buf2: [fusion_name_buf_size]u8 = undefined;
    const gs_name = std.fmt.bufPrint(&gs_buf2, "{s}layers.{d}.{s}.weight_global_scale", .{ prefix, layer, hf_name }) catch return;
    if (tensors.get(gs_name)) |gs_entry| {
        var gguf_gs_buf2: [fusion_name_buf_size]u8 = undefined;
        const gguf_gs = std.fmt.bufPrint(&gguf_gs_buf2, "blk.{d}.{s}.global_scale", .{ layer, gguf_component }) catch return;
        const owned_gs = try dupeString(allocator, owned, gguf_gs);
        try fused.put(owned_gs, gs_entry);
    }
    // Create GGUF input_scale entry: blk.{l}.{component}.input_scale
    var is_buf: [fusion_name_buf_size]u8 = undefined;
    const is_name = std.fmt.bufPrint(&is_buf, "{s}layers.{d}.{s}.input_global_scale", .{ prefix, layer, hf_name }) catch return;
    if (tensors.get(is_name)) |is_entry| {
        var gguf_is_buf: [fusion_name_buf_size]u8 = undefined;
        const gguf_is = std.fmt.bufPrint(&gguf_is_buf, "blk.{d}.{s}.input_scale", .{ layer, gguf_component }) catch return;
        const owned_is = try dupeString(allocator, owned, gguf_is);
        try fused.put(owned_is, is_entry);
    }
}

/// Fuse one projection (e.g., gate_proj) across all experts for a single layer.
/// Creates synthetic GGUF-named weight and scale entries.
fn fuseOneProjection(
    allocator: Allocator,
    tensors: *std.StringHashMap(TensorEntry),
    fused: *std.StringHashMap(TensorEntry),
    repacked_f32: *std.ArrayList([]f32),
    shard_data: []ShardInfo,
    owned: *std.ArrayList([]u8),
    prefix: []const u8,
    layer: u32,
    n_experts: u32,
    hf_proj: []const u8,
    gguf_weight_name: []const u8,
    gguf_scales_name: []const u8,
    create_global_scale: bool,
) !void {
    var name_buf: [fusion_name_buf_size]u8 = undefined;

    // Look up expert 0 weight_packed tensor.
    const e0_weight_name = std.fmt.bufPrint(&name_buf, "{s}layers.{d}.mlp.experts.0.{s}.weight_packed", .{ prefix, layer, hf_proj }) catch return;
    const e0_w = tensors.get(e0_weight_name) orelse return;

    // Look up expert 0 weight_scale tensor.
    var s_buf: [fusion_name_buf_size]u8 = undefined;
    const e0_scale_name = std.fmt.bufPrint(&s_buf, "{s}layers.{d}.mlp.experts.0.{s}.weight_scale", .{ prefix, layer, hf_proj }) catch return;
    const e0_s = tensors.get(e0_scale_name) orelse return;

    // Compute per-expert byte sizes.
    // weight_packed: [out, in/2] U8 bytes → total = out * in/2
    const w_rows = if (e0_w.n_dims >= 1) e0_w.dims[0] else return;
    const w_cols = if (e0_w.n_dims >= 2) e0_w.dims[1] else return;
    const w_bytes: usize = @as(usize, @intCast(w_rows)) * @as(usize, @intCast(w_cols));

    // weight_scale: [out, in/16] FP8 bytes → total = out * in/16
    const s_rows = if (e0_s.n_dims >= 1) e0_s.dims[0] else return;
    const s_cols = if (e0_s.n_dims >= 2) e0_s.dims[1] else return;
    const s_bytes: usize = @as(usize, @intCast(s_rows)) * @as(usize, @intCast(s_cols));

    // Verify all experts are contiguous in memory (same shard, sequential offsets).
    var contiguous = true;
    var en_buf: [fusion_name_buf_size]u8 = undefined;
    var sn_buf: [fusion_name_buf_size]u8 = undefined;
    for (1..n_experts) |ei| {
        const ei_w_name = std.fmt.bufPrint(&en_buf, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_packed", .{ prefix, layer, ei, hf_proj }) catch {
            contiguous = false;
            break;
        };
        const ei_w = tensors.get(ei_w_name) orelse {
            contiguous = false;
            break;
        };
        // Check same shard and expected offset.
        if (ei_w.shard_idx != e0_w.shard_idx or
            ei_w.data_start != e0_w.data_start + ei * w_bytes)
        {
            contiguous = false;
            break;
        }
        // Also verify scales are contiguous.
        const ei_s_name = std.fmt.bufPrint(&sn_buf, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_scale", .{ prefix, layer, ei, hf_proj }) catch {
            contiguous = false;
            break;
        };
        const ei_s = tensors.get(ei_s_name) orelse {
            contiguous = false;
            break;
        };
        if (ei_s.shard_idx != e0_s.shard_idx or
            ei_s.data_start != e0_s.data_start + ei * s_bytes)
        {
            contiguous = false;
            break;
        }
    }

    // Experts are typically NOT contiguous in SafeTensors (interleaved with scale tensors).
    // We still create synthetic fused entries regardless — the model accesses individual
    // experts via stride, and each expert's data is at the correct shard offset.
    // For non-contiguous layouts, we store expert 0's start and the actual per-expert
    // stride as dims[0]*dims[1]. The model's expertWeightStride() returns the stride,
    // and expert_data = base + expert_idx * stride. For this to work, all experts must
    // be in the same shard with matching sizes (but not necessarily contiguous offsets).
    //
    // If non-contiguous, we allocate a repacked buffer and copy all experts contiguously.

    if (!contiguous) {
        // Experts are non-contiguous in the SafeTensors file. Allocate contiguous
        // buffers and copy each expert's data sequentially.
        const total_w = @as(usize, n_experts) * w_bytes;
        const total_s = @as(usize, n_experts) * s_bytes;
        // Use page_allocator for GPU compatibility (Metal newBufferWithBytesNoCopy
        // requires page-aligned pointers).
        const pa = std.heap.page_allocator;
        const repacked_w = try pa.alloc(u8, total_w);
        errdefer pa.free(repacked_w);
        const repacked_s = try pa.alloc(u8, total_s);
        errdefer pa.free(repacked_s);

        for (0..n_experts) |ei| {
            // Copy weight_packed
            const ei_w_name2 = std.fmt.bufPrint(&en_buf, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_packed", .{ prefix, layer, ei, hf_proj }) catch return;
            const ei_w2 = tensors.get(ei_w_name2) orelse return;
            const w_shard = shard_data[ei_w2.shard_idx];
            if (w_shard.data.len > 0) {
                const w_src_start = w_shard.tensor_base + ei_w2.data_start;
                const w_src_end = w_src_start + w_bytes;
                if (w_src_end <= w_shard.data.len) {
                    @memcpy(repacked_w[ei * w_bytes ..][0..w_bytes], w_shard.data[w_src_start..w_src_end]);
                }
            }
            // Copy weight_scale
            const ei_s_name2 = std.fmt.bufPrint(&sn_buf, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_scale", .{ prefix, layer, ei, hf_proj }) catch return;
            const ei_s2 = tensors.get(ei_s_name2) orelse return;
            const s_shard = shard_data[ei_s2.shard_idx];
            if (s_shard.data.len > 0) {
                const s_src_start = s_shard.tensor_base + ei_s2.data_start;
                const s_src_end = s_src_start + s_bytes;
                if (s_src_end <= s_shard.data.len) {
                    @memcpy(repacked_s[ei * s_bytes ..][0..s_bytes], s_shard.data[s_src_start..s_src_end]);
                }
            }
        }

        // Create synthetic GGUF entries pointing to repacked buffers
        var gguf_w_buf2: [fusion_name_buf_size]u8 = undefined;
        const gguf_w_name2 = std.fmt.bufPrint(&gguf_w_buf2, "blk.{d}.{s}", .{ layer, gguf_weight_name }) catch return;
        const owned_w_name2 = try dupeString(allocator, owned, gguf_w_name2);
        // Store as a special shard_idx that signals "use repacked data"
        // We add a new shard entry with the repacked buffer
        // Actually: we can't add shards dynamically. Instead, create a TensorEntry
        // with data_start pointing into the repacked buffer via pointer arithmetic.
        // The entryToInfo function resolves: shard.data.ptr + shard.tensor_base + entry.data_start
        // For repacked: we need shard.data = repacked_w, tensor_base = 0, data_start = 0.
        // But we can't add new ShardInfo entries (shard_data is fixed-size).
        //
        // Workaround: store the repacked pointer directly as data_start (abs ptr encoded).
        // This requires a special dtype flag or shard_idx sentinel.
        // Simplest: use shard_idx = max_shard_count as sentinel for "repacked" entries,
        // and store the repacked pointer in data_start/data_end as usize.
        try fused.put(owned_w_name2, TensorEntry{
            .shard_idx = max_shard_count, // sentinel: repacked data
            .data_start = @intFromPtr(repacked_w.ptr),
            .data_end = @intFromPtr(repacked_w.ptr) + total_w,
            .dtype = .nvfp4,
            .n_dims = 3,
            .dims = .{ w_rows, w_cols, n_experts, 0 },
        });

        var gguf_s_buf2: [fusion_name_buf_size]u8 = undefined;
        const gguf_s_name2 = std.fmt.bufPrint(&gguf_s_buf2, "blk.{d}.{s}", .{ layer, gguf_scales_name }) catch return;
        const owned_s_name2 = try dupeString(allocator, owned, gguf_s_name2);
        try fused.put(owned_s_name2, TensorEntry{
            .shard_idx = max_shard_count,
            .data_start = @intFromPtr(repacked_s.ptr),
            .data_end = @intFromPtr(repacked_s.ptr) + total_s,
            .dtype = .fp8_e4m3,
            .n_dims = 3,
            .dims = .{ s_rows, s_cols, n_experts, 0 },
        });

        // Also repack global_scale for non-contiguous experts
        if (create_global_scale) {
            const gs_array2 = try allocator.alloc(f32, n_experts);
            try repacked_f32.append(allocator, gs_array2);
            var gs_buf3: [fusion_name_buf_size]u8 = undefined;
            for (0..n_experts) |gi| {
                const gi_gs = std.fmt.bufPrint(&gs_buf3, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_global_scale", .{ prefix, layer, gi, hf_proj }) catch {
                    gs_array2[gi] = 1.0;
                    continue;
                };
                if (tensors.get(gi_gs)) |gs_e| {
                    if (gs_e.shard_idx < shard_data.len and shard_data[gs_e.shard_idx].data.len > 0) {
                        const sh = shard_data[gs_e.shard_idx];
                        const a = sh.tensor_base + gs_e.data_start;
                        if (a + 4 <= sh.data.len) {
                            gs_array2[gi] = @as(*const f32, @ptrCast(@alignCast(sh.data.ptr + a))).*;
                            continue;
                        }
                    }
                }
                gs_array2[gi] = 1.0;
            }
            var gguf_gs3: [fusion_name_buf_size]u8 = undefined;
            const bn = gguf_weight_name[0 .. gguf_weight_name.len - ".weight".len];
            const gsn = std.fmt.bufPrint(&gguf_gs3, "blk.{d}.{s}.global_scale", .{ layer, bn }) catch return;
            const owned_gsn = try dupeString(allocator, owned, gsn);
            try fused.put(owned_gsn, TensorEntry{ .shard_idx = max_shard_count, .data_start = @intFromPtr(gs_array2.ptr), .data_end = @intFromPtr(gs_array2.ptr) + n_experts * @sizeOf(f32), .dtype = .f32, .n_dims = 1, .dims = .{ n_experts, 0, 0, 0 } });

            // Also repack input_global_scale
            const is_array = try allocator.alloc(f32, n_experts);
            try repacked_f32.append(allocator, is_array);
            var is_buf3: [fusion_name_buf_size]u8 = undefined;
            for (0..n_experts) |gi| {
                const gi_is = std.fmt.bufPrint(&is_buf3, "{s}layers.{d}.mlp.experts.{d}.{s}.input_global_scale", .{ prefix, layer, gi, hf_proj }) catch {
                    is_array[gi] = 1.0;
                    continue;
                };
                if (tensors.get(gi_is)) |is_e| {
                    if (is_e.shard_idx < shard_data.len and shard_data[is_e.shard_idx].data.len > 0) {
                        const sh2 = shard_data[is_e.shard_idx];
                        const a2 = sh2.tensor_base + is_e.data_start;
                        if (a2 + 4 <= sh2.data.len) {
                            is_array[gi] = @as(*const f32, @ptrCast(@alignCast(sh2.data.ptr + a2))).*;
                            continue;
                        }
                    }
                }
                is_array[gi] = 1.0;
            }
            var gguf_is3: [fusion_name_buf_size]u8 = undefined;
            const isn = std.fmt.bufPrint(&gguf_is3, "blk.{d}.{s}.input_scale", .{ layer, bn }) catch return;
            const owned_isn = try dupeString(allocator, owned, isn);
            try fused.put(owned_isn, TensorEntry{ .shard_idx = max_shard_count, .data_start = @intFromPtr(is_array.ptr), .data_end = @intFromPtr(is_array.ptr) + n_experts * @sizeOf(f32), .dtype = .f32, .n_dims = 1, .dims = .{ n_experts, 0, 0, 0 } });
        }
        return;
    }

    // Create synthetic GGUF-named weight entry: blk.{l}.ffn_{proj}_exps.weight
    // dtype = .nvfp4, dims = [out, in/2, n_experts] (expertWeightStride uses dims[0]*dims[1])
    var gguf_w_buf: [fusion_name_buf_size]u8 = undefined;
    const gguf_w_name = std.fmt.bufPrint(&gguf_w_buf, "blk.{d}.{s}", .{ layer, gguf_weight_name }) catch return;
    const owned_w_name = try dupeString(allocator, owned, gguf_w_name);
    try fused.put(owned_w_name, TensorEntry{
        .shard_idx = e0_w.shard_idx,
        .data_start = e0_w.data_start,
        .data_end = e0_w.data_start + @as(usize, n_experts) * w_bytes,
        .dtype = .nvfp4,
        .n_dims = 3,
        .dims = .{ w_rows, w_cols, n_experts, 0 },
    });

    // Create synthetic GGUF-named scale entry: blk.{l}.ffn_{proj}_exps.scales
    // dtype = .fp8_e4m3, dims = [out, in/16, n_experts]
    var gguf_s_buf: [fusion_name_buf_size]u8 = undefined;
    const gguf_s_name = std.fmt.bufPrint(&gguf_s_buf, "blk.{d}.{s}", .{ layer, gguf_scales_name }) catch return;
    const owned_s_name = try dupeString(allocator, owned, gguf_s_name);
    try fused.put(owned_s_name, TensorEntry{
        .shard_idx = e0_s.shard_idx,
        .data_start = e0_s.data_start,
        .data_end = e0_s.data_start + @as(usize, n_experts) * s_bytes,
        .dtype = .fp8_e4m3,
        .n_dims = 3,
        .dims = .{ s_rows, s_cols, n_experts, 0 },
    });

    // Create synthetic global_scale entry: blk.{l}.ffn_{proj}_exps.global_scale
    if (create_global_scale) {
        const gs_array = try allocator.alloc(f32, n_experts);
        try repacked_f32.append(allocator, gs_array);
        var gs_name_buf2: [fusion_name_buf_size]u8 = undefined;
        for (0..n_experts) |gi| {
            const gi_gs_name = std.fmt.bufPrint(&gs_name_buf2, "{s}layers.{d}.mlp.experts.{d}.{s}.weight_global_scale", .{ prefix, layer, gi, hf_proj }) catch {
                gs_array[gi] = 1.0;
                continue;
            };
            if (tensors.get(gi_gs_name)) |gs_entry| {
                if (gs_entry.shard_idx < shard_data.len and shard_data[gs_entry.shard_idx].data.len > 0) {
                    const shard = shard_data[gs_entry.shard_idx];
                    const abs = shard.tensor_base + gs_entry.data_start;
                    if (abs + 4 <= shard.data.len) {
                        gs_array[gi] = @as(*const f32, @ptrCast(@alignCast(shard.data.ptr + abs))).*;
                        continue;
                    }
                }
            }
            gs_array[gi] = 1.0;
        }

        // Also repack input_global_scale (contiguous path)
        const is_array = try allocator.alloc(f32, n_experts);
        try repacked_f32.append(allocator, is_array);
        var is_buf_c: [fusion_name_buf_size]u8 = undefined;
        for (0..n_experts) |gi| {
            const gi_is = std.fmt.bufPrint(&is_buf_c, "{s}layers.{d}.mlp.experts.{d}.{s}.input_global_scale", .{ prefix, layer, gi, hf_proj }) catch {
                is_array[gi] = 1.0;
                continue;
            };
            if (tensors.get(gi_is)) |is_e| {
                if (is_e.shard_idx < shard_data.len and shard_data[is_e.shard_idx].data.len > 0) {
                    const sh2 = shard_data[is_e.shard_idx];
                    const a2 = sh2.tensor_base + is_e.data_start;
                    if (a2 + 4 <= sh2.data.len) {
                        is_array[gi] = @as(*const f32, @ptrCast(@alignCast(sh2.data.ptr + a2))).*;
                        continue;
                    }
                }
            }
            is_array[gi] = 1.0;
        }

        var gguf_gs_buf: [fusion_name_buf_size]u8 = undefined;
        const base_name = gguf_weight_name[0 .. gguf_weight_name.len - ".weight".len];
        const gguf_gs_name = std.fmt.bufPrint(&gguf_gs_buf, "blk.{d}.{s}.global_scale", .{ layer, base_name }) catch return;
        const owned_gs_name = try dupeString(allocator, owned, gguf_gs_name);
        try fused.put(owned_gs_name, TensorEntry{
            .shard_idx = max_shard_count, // sentinel: repacked data
            .data_start = @intFromPtr(gs_array.ptr),
            .data_end = @intFromPtr(gs_array.ptr) + n_experts * @sizeOf(f32),
            .dtype = .f32,
            .n_dims = 1,
            .dims = .{ n_experts, 0, 0, 0 },
        });

        var gguf_is_buf: [fusion_name_buf_size]u8 = undefined;
        const isn = std.fmt.bufPrint(&gguf_is_buf, "blk.{d}.{s}.input_scale", .{ layer, base_name }) catch return;
        const owned_isn = try dupeString(allocator, owned, isn);
        try fused.put(owned_isn, TensorEntry{
            .shard_idx = max_shard_count,
            .data_start = @intFromPtr(is_array.ptr),
            .data_end = @intFromPtr(is_array.ptr) + n_experts * @sizeOf(f32),
            .dtype = .f32,
            .n_dims = 1,
            .dims = .{ n_experts, 0, 0, 0 },
        });
    }
}

/// Fuse shared expert projection (no expert index).
/// Creates GGUF-named weight and scale entries for the shared expert.
fn fuseSharedExpertProjection(
    allocator: Allocator,
    tensors: *std.StringHashMap(TensorEntry),
    fused: *std.StringHashMap(TensorEntry),
    repacked_f32: *std.ArrayList([]f32),
    shard_data: []ShardInfo,
    owned: *std.ArrayList([]u8),
    prefix: []const u8,
    layer: u32,
    hf_proj: []const u8,
    gguf_weight_name: []const u8,
    gguf_scales_name: []const u8,
) !void {
    _ = shard_data;
    _ = repacked_f32;

    var name_buf: [fusion_name_buf_size]u8 = undefined;

    // Try both "shared_expert" (singular, compressed-tensors) and "shared_experts" (plural, HF default)
    const shared_prefixes = [_][]const u8{ "shared_expert", "shared_experts" };
    for (shared_prefixes) |sp| {
        const w_name = std.fmt.bufPrint(&name_buf, "{s}layers.{d}.mlp.{s}.{s}.weight_packed", .{ prefix, layer, sp, hf_proj }) catch continue;
        const w_entry = tensors.get(w_name) orelse continue;

        var s_buf: [fusion_name_buf_size]u8 = undefined;
        const s_name = std.fmt.bufPrint(&s_buf, "{s}layers.{d}.mlp.{s}.{s}.weight_scale", .{ prefix, layer, sp, hf_proj }) catch continue;
        const s_entry = tensors.get(s_name) orelse continue;

        // Create GGUF-named weight entry: blk.{l}.ffn_{proj}_shexp.weight
        // For shared expert (single tensor), use 2D dims [out, in/2] with nvfp4 dtype.
        var gguf_w_buf: [fusion_name_buf_size]u8 = undefined;
        const gguf_w = std.fmt.bufPrint(&gguf_w_buf, "blk.{d}.{s}", .{ layer, gguf_weight_name }) catch continue;
        const owned_w = try dupeString(allocator, owned, gguf_w);
        try fused.put(owned_w, TensorEntry{
            .shard_idx = w_entry.shard_idx,
            .data_start = w_entry.data_start,
            .data_end = w_entry.data_end,
            .dtype = .nvfp4,
            .n_dims = w_entry.n_dims,
            .dims = w_entry.dims,
        });

        // Create GGUF-named scale entry: blk.{l}.ffn_{proj}_shexp.scales
        var gguf_s_buf: [fusion_name_buf_size]u8 = undefined;
        const gguf_s = std.fmt.bufPrint(&gguf_s_buf, "blk.{d}.{s}", .{ layer, gguf_scales_name }) catch continue;
        const owned_s = try dupeString(allocator, owned, gguf_s);
        try fused.put(owned_s, TensorEntry{
            .shard_idx = s_entry.shard_idx,
            .data_start = s_entry.data_start,
            .data_end = s_entry.data_end,
            .dtype = .fp8_e4m3,
            .n_dims = s_entry.n_dims,
            .dims = s_entry.dims,
        });

        // Global scale for shared expert
        var gs_buf: [fusion_name_buf_size]u8 = undefined;
        const gs_name = std.fmt.bufPrint(&gs_buf, "{s}layers.{d}.mlp.{s}.{s}.weight_global_scale", .{ prefix, layer, sp, hf_proj }) catch continue;
        if (tensors.get(gs_name)) |gs_entry| {
            var gguf_gs_buf: [fusion_name_buf_size]u8 = undefined;
            const base = gguf_weight_name[0 .. gguf_weight_name.len - ".weight".len];
            const gguf_gs = std.fmt.bufPrint(&gguf_gs_buf, "blk.{d}.{s}.global_scale", .{ layer, base }) catch continue;
            const owned_gs = try dupeString(allocator, owned, gguf_gs);
            try fused.put(owned_gs, gs_entry);
        }

        break; // Found matching shared expert
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Reject shard filenames that could escape the model directory.
/// A safe name is non-empty, contains no path separators or `..` sequences,
/// and has no null bytes.
fn isSafeShardName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (std.mem.indexOf(u8, name, "..") != null) return false;
    if (std.mem.indexOfScalar(u8, name, '/') != null) return false;
    if (std.mem.indexOfScalar(u8, name, '\\') != null) return false;
    if (std.mem.indexOfScalar(u8, name, 0) != null) return false;
    return true;
}

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
    // Use C opendir/readdir to iterate without Zig Io context (0.16 compat).
    const dir_z = try allocator.dupeZ(u8, dir_path);
    defer allocator.free(dir_z);
    const dirp = std.c.opendir(dir_z.ptr) orelse return;
    defer _ = std.c.closedir(dirp);

    // Collect matching filenames.
    var names: std.ArrayList([]u8) = .empty;
    defer names.deinit(allocator);

    while (std.c.readdir(dirp)) |entry| {
        const name = std.mem.sliceTo(&entry.name, 0);
        if (!std.mem.endsWith(u8, name, ".safetensors")) continue;
        // Skip index files.
        if (std.mem.endsWith(u8, name, ".index.json.safetensors")) continue;
        if (names.items.len >= max_shard_count) return error.InvalidSafeTensors;
        const name_copy = try dupeString(allocator, owned, name);
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
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const fd = std.posix.system.open(path_z.ptr, .{}, @as(std.posix.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const size: usize = blk: {
        var s: std.posix.Stat = undefined;
        if (std.c.fstat(fd, &s) != 0) return error.FileNotFound;
        break :blk @intCast(s.size);
    };
    if (size > max_json_file_size) return error.FileTooLarge;
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    var total: usize = 0;
    while (total < size) {
        const n = std.posix.read(fd, buf[total..]) catch |e| switch (e) {
            error.WouldBlock => continue,
            else => return e,
        };
        if (n == 0) return error.UnexpectedEof;
        total += n;
    }
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
        if (json[i] == '\\') {
            i += 1;
            if (i >= json.len) return error.JsonUnterminated;
        }
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
                if (json[i] == '\\') {
                    i += 1;
                    if (i >= json.len) return error.JsonUnterminated;
                }
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
                    if (depth > max_json_depth) return error.JsonUnexpected;
                } else if (json[i] == close) {
                    depth -= 1;
                } else if (json[i] == '"') {
                    i += 1;
                    while (i < json.len and json[i] != '"') {
                        if (json[i] == '\\') {
                            i += 1;
                            if (i >= json.len) return error.JsonUnterminated;
                        }
                        i += 1;
                    }
                    if (i >= json.len) return error.JsonUnterminated;
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
/// Rejects digit sequences longer than 20 chars to prevent CPU exhaustion
/// from crafted JSON with millions of leading zeros.
fn parseU64Slice(s: []const u8) !u64 {
    const trimmed = std.mem.trim(u8, s, " \t\r\n,]})");
    if (trimmed.len > 20) return error.Overflow;
    return std.fmt.parseUnsigned(u64, trimmed, 10);
}

/// Map a safetensors dtype string to our DType enum.
fn parseDType(s: []const u8) DType {
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F16")) return .f16;
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    // SafeTensors U32 dtype indicates MLX-quantized packed weights.
    if (std.mem.eql(u8, s, "U32")) return .mlx_q;
    // FP8 E4M3 — used by compressed-tensors NVFP4 scale tensors.
    if (std.mem.eql(u8, s, "F8_E4M3")) return .fp8_e4m3;
    // U8 — packed FP4 bytes, used by compressed-tensors NVFP4 weight_packed tensors.
    if (std.mem.eql(u8, s, "U8")) return .nvfp4;
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

            if (data_end < data_start) return error.InvalidSafeTensors;
            const owned_name = try dupeString(allocator, owned, name_res.val);
            try tensors.put(owned_name, TensorEntry{
                .shard_idx = shard_idx,
                .data_start = std.math.cast(usize, data_start) orelse return error.InvalidSafeTensors,
                .data_end = std.math.cast(usize, data_end) orelse return error.InvalidSafeTensors,
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
                    // Reject shard names with path traversal sequences or separators.
                    if (!isSafeShardName(shard_res.val)) return error.InvalidSafeTensors;
                    if (shard_list.items.len >= max_shard_count) return error.InvalidSafeTensors;
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

    const vocab_size = std.math.add(usize, max_id, 1) catch return error.InvalidSafeTensors;
    if (vocab_size > max_vocab_size) return error.InvalidSafeTensors;
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
                token_id = std.math.cast(usize, parseU64Slice(json[ns..i]) catch 0);
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
        const id_end = std.math.add(usize, e.id, 1) catch return error.InvalidSafeTensors;
        if (id_end > max_id) max_id = id_end;
    }
    if (max_id > max_vocab_size) return error.InvalidSafeTensors;

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
    try std.testing.expectEqual(DType.nvfp4, parseDType("U8"));
    // Compressed-tensors NVFP4 scale dtype
    try std.testing.expectEqual(DType.fp8_e4m3, parseDType("F8_E4M3"));
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

test "ggufToHfNameIter multiple mappings" {
    // ffn_gate_inp has two mappings: mlp.router and mlp.gate
    var buf: [name_buf_size]u8 = undefined;
    var iter = ggufToHfNameIter("blk.0.ffn_gate_inp.weight", "model.");
    const first = iter.next(&buf);
    try std.testing.expect(first != null);
    try std.testing.expectEqualStrings("model.layers.0.mlp.router.weight", first.?);
    const second = iter.next(&buf);
    try std.testing.expect(second != null);
    try std.testing.expectEqualStrings("model.layers.0.mlp.gate.weight", second.?);
    // No more mappings
    try std.testing.expect(iter.next(&buf) == null);
}

test "fuseNvfp4Experts creates synthetic entries" {
    const allocator = std.testing.allocator;

    var tensors = std.StringHashMap(TensorEntry).init(allocator);
    defer tensors.deinit();
    var config_meta = std.StringHashMap(MetaValue).init(allocator);
    defer config_meta.deinit();
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |s| allocator.free(s);
        owned.deinit(allocator);
    }

    // Simulate 2 experts, 1 layer, gate_proj only.
    // weight_packed: [4, 8] U8, weight_scale: [4, 1] FP8
    // Expert 0 at offset 0, expert 1 at offset 32 (contiguous)
    const w_bytes: usize = 4 * 8; // 32 bytes per expert
    const s_bytes: usize = 4 * 1; // 4 bytes per expert

    const e0_w_name = try dupeString(allocator, &owned, "model.layers.0.mlp.experts.0.gate_proj.weight_packed");
    try tensors.put(e0_w_name, TensorEntry{
        .shard_idx = 0,
        .data_start = 0,
        .data_end = w_bytes,
        .dtype = .nvfp4,
        .n_dims = 2,
        .dims = .{ 4, 8, 0, 0 },
    });
    const e1_w_name = try dupeString(allocator, &owned, "model.layers.0.mlp.experts.1.gate_proj.weight_packed");
    try tensors.put(e1_w_name, TensorEntry{
        .shard_idx = 0,
        .data_start = w_bytes,
        .data_end = 2 * w_bytes,
        .dtype = .nvfp4,
        .n_dims = 2,
        .dims = .{ 4, 8, 0, 0 },
    });
    const e0_s_name = try dupeString(allocator, &owned, "model.layers.0.mlp.experts.0.gate_proj.weight_scale");
    try tensors.put(e0_s_name, TensorEntry{
        .shard_idx = 0,
        .data_start = 2 * w_bytes,
        .data_end = 2 * w_bytes + s_bytes,
        .dtype = .fp8_e4m3,
        .n_dims = 2,
        .dims = .{ 4, 1, 0, 0 },
    });
    const e1_s_name = try dupeString(allocator, &owned, "model.layers.0.mlp.experts.1.gate_proj.weight_scale");
    try tensors.put(e1_s_name, TensorEntry{
        .shard_idx = 0,
        .data_start = 2 * w_bytes + s_bytes,
        .data_end = 2 * w_bytes + 2 * s_bytes,
        .dtype = .fp8_e4m3,
        .n_dims = 2,
        .dims = .{ 4, 1, 0, 0 },
    });

    // Config metadata
    try config_meta.put("num_hidden_layers", .{ .uint = 1 });
    try config_meta.put("num_local_experts", .{ .uint = 2 });

    // Dummy shard_data (not used directly by fusion when contiguous)
    var shard_data = [_]ShardInfo{.{ .data = &.{}, .tensor_base = 0 }};

    var fused = std.StringHashMap(TensorEntry).init(allocator);
    defer fused.deinit();
    var repacked_f32: std.ArrayList([]f32) = .empty;
    defer {
        for (repacked_f32.items) |s| allocator.free(s);
        repacked_f32.deinit(allocator);
    }

    try fuseNvfp4Experts(allocator, &tensors, &fused, &repacked_f32, &shard_data, &config_meta, &owned);

    // Fused entries go into separate fused map, not tensors
    const fused_w = fused.get("blk.0.ffn_gate_exps.weight") orelse return error.MissingTensor;
    try std.testing.expectEqual(DType.nvfp4, fused_w.dtype);
    try std.testing.expectEqual(@as(u32, 3), fused_w.n_dims);
    try std.testing.expectEqual(@as(u64, 4), fused_w.dims[0]); // rows
    try std.testing.expectEqual(@as(u64, 8), fused_w.dims[1]); // cols/2
    try std.testing.expectEqual(@as(u64, 2), fused_w.dims[2]); // n_experts
    try std.testing.expectEqual(@as(usize, 0), fused_w.data_start);
    try std.testing.expectEqual(@as(usize, 2 * w_bytes), fused_w.data_end);

    const fused_s = fused.get("blk.0.ffn_gate_exps.scales") orelse return error.MissingTensor;
    try std.testing.expectEqual(DType.fp8_e4m3, fused_s.dtype);
    try std.testing.expectEqual(@as(u32, 3), fused_s.n_dims);
}
