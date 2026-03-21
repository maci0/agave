//! Model interface for LLM inference.
//! Provides a type-erased interface via comptime vtable generation, allowing
//! the engine to work with any model architecture through a uniform API.
//!
//! Implementations: gemma3.zig, qwen35.zig, gpt_oss.zig, nemotron_h.zig,
//! nemotron_nano.zig, glm4.zig

const std = @import("std");
const build_options = @import("build_options");
const backend_mod = @import("../backend/backend.zig");
const format_mod = @import("../format/format.zig");

/// Errors that can occur during model forward pass.
pub const ForwardError = error{
    /// A required weight tensor was not found in the model file.
    MissingTensor,
    /// The KV cache has reached its maximum capacity.
    KVCacheFull,
    /// The forward pass was cancelled by another thread.
    Cancelled,
    /// Memory allocation failed.
    OutOfMemory,
    /// No physical blocks available in PagedKvCache.
    OutOfBlocks,
};

/// Model interface — all models implement this via comptime vtable generation.
///
/// Usage: implement `forward`, `resetCache`, `cancel` methods and fields
/// `eos_token_id`, `vocab_size`, `n_layers`, `n_embd`, `n_head`, `n_head_kv`,
/// then call `Model.from(MyModel, &my_instance)`.
pub const Model = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// Function pointer table for polymorphic model dispatch.
    pub const VTable = struct {
        forward: *const fn (self: *anyopaque, token_id: u32) ForwardError!u32,
        reset_cache: *const fn (self: *anyopaque) void,
        cancel: *const fn (self: *anyopaque) void,
        get_eos_id: *const fn (self: *anyopaque) u32,
        get_vocab_size: *const fn (self: *anyopaque) u32,
        get_n_layers: *const fn (self: *anyopaque) u32,
        get_n_embd: *const fn (self: *anyopaque) u32,
        get_n_head: *const fn (self: *anyopaque) u32,
        get_n_head_kv: *const fn (self: *anyopaque) u32,
        get_logits: *const fn (self: *anyopaque) []f32,
    };

    /// Construct a Model interface from any concrete model type at comptime.
    /// The concrete type must have: forward(token_id) !u32, resetCache(), cancel(),
    /// and fields: eos_token_id, vocab_size, n_layers, n_embd, n_head, n_head_kv.
    pub fn from(comptime T: type, ptr: *T) Model {
        const vtable = comptime genVTable(T);
        return .{ .ptr = ptr, .vtable = vtable };
    }

    fn genVTable(comptime T: type) *const VTable {
        return &comptime .{
            .forward = @ptrCast(&struct {
                fn call(self: *T, token_id: u32) ForwardError!u32 {
                    return self.forward(token_id);
                }
            }.call),
            .reset_cache = @ptrCast(&struct {
                fn call(self: *T) void {
                    self.resetCache();
                }
            }.call),
            .cancel = @ptrCast(&struct {
                fn call(self: *T) void {
                    self.cancel();
                }
            }.call),
            .get_eos_id = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.eos_token_id;
                }
            }.call),
            .get_vocab_size = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.vocab_size;
                }
            }.call),
            .get_n_layers = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_layers;
                }
            }.call),
            .get_n_embd = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_embd;
                }
            }.call),
            .get_n_head = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_head;
                }
            }.call),
            .get_n_head_kv = @ptrCast(&struct {
                fn call(self: *T) u32 {
                    return self.n_head_kv;
                }
            }.call),
            .get_logits = @ptrCast(&struct {
                fn call(self: *T) []f32 {
                    return if (@hasField(T, "logits")) self.logits else self.logits_buf;
                }
            }.call),
        };
    }

    /// Run one decode step: process `token_id` through all layers,
    /// returning the predicted next-token ID (argmax of logits).
    ///
    /// Parameters:
    ///   - token_id: Input token to process.
    ///
    /// Returns: The predicted next token ID.
    /// Errors: MissingTensor, KVCacheFull, Cancelled, OutOfMemory.
    pub fn forward(self: Model, token_id: u32) ForwardError!u32 {
        return self.vtable.forward(self.ptr, token_id);
    }

    /// Return the raw logits buffer from the last forward() call.
    /// Used for temperature-based sampling instead of greedy argmax.
    pub fn getLogits(self: Model) []f32 {
        return self.vtable.get_logits(self.ptr);
    }

    /// Reset the KV cache position to zero, allowing a fresh conversation.
    pub fn resetCache(self: Model) void {
        self.vtable.reset_cache(self.ptr);
    }

    /// Signal the model to cancel the current forward pass.
    /// Checked between layers; the next forward() call returns error.Cancelled.
    pub fn cancel(self: Model) void {
        self.vtable.cancel(self.ptr);
    }

    /// Return the end-of-sequence token ID for this model.
    pub fn eosId(self: Model) u32 {
        return self.vtable.get_eos_id(self.ptr);
    }

    /// Return the vocabulary size.
    pub fn vocabSize(self: Model) u32 {
        return self.vtable.get_vocab_size(self.ptr);
    }

    /// Return the number of transformer layers.
    pub fn nLayers(self: Model) u32 {
        return self.vtable.get_n_layers(self.ptr);
    }

    /// Return the embedding dimension.
    pub fn nEmbd(self: Model) u32 {
        return self.vtable.get_n_embd(self.ptr);
    }

    /// Return the number of query attention heads.
    pub fn nHead(self: Model) u32 {
        return self.vtable.get_n_head(self.ptr);
    }

    /// Return the number of key/value attention heads (for GQA).
    pub fn nHeadKv(self: Model) u32 {
        return self.vtable.get_n_head_kv(self.ptr);
    }
};

// ── Shared helpers for model implementations ─────────────────────

/// Reset common inference state (KV cache position + cancellation flag).
/// Models with additional state (e.g. SSM conv/recurrence) should clear
/// that first, then call this.
pub inline fn resetInferenceState(kv_seq_len: *usize, cancelled: *std.atomic.Value(bool)) void {
    kv_seq_len.* = 0;
    cancelled.store(false, .release);
}

/// Compute the byte stride between consecutive experts in a packed weight tensor.
/// GGUF stores packed expert tensors with dims [inner, outer, n_experts] where
/// the expert index is the slowest-varying axis.
pub fn expertWeightStride(t: format_mod.TensorInfo) usize {
    std.debug.assert(t.n_dims >= 3);
    const elems: usize = @as(usize, @intCast(t.dims[0])) * @as(usize, @intCast(t.dims[1]));
    return backend_mod.weightBytes(t.dtype, 1, elems);
}

/// Signal cancellation of a forward pass (thread-safe).
pub inline fn signalCancel(cancelled: *std.atomic.Value(bool)) void {
    cancelled.store(true, .release);
}

/// Gemma 3 model implementation — conditionally compiled via -Denable-gemma3.
pub const Gemma3Model = if (build_options.enable_gemma3) @import("gemma3.zig").Gemma3Model else void;
/// Qwen3.5 hybrid DeltaNet + attention model — conditionally compiled via -Denable-qwen35.
pub const Qwen35Model = if (build_options.enable_qwen35) @import("qwen35.zig").Qwen35Model else void;
/// GPT-OSS Mixture-of-Experts model — conditionally compiled via -Denable-gpt-oss.
pub const GptOssModel = if (build_options.enable_gpt_oss) @import("gpt_oss.zig").GptOssModel else void;
/// Nemotron-H hybrid Mamba-2 + Attention + FFN-only model — conditionally compiled via -Denable-nemotron-h.
pub const NemotronHModel = if (build_options.enable_nemotron_h) @import("nemotron_h.zig").NemotronHModel else void;
/// GLM-4 MoE Lite model with MLA attention — conditionally compiled via -Denable-glm4.
pub const Glm4Model = if (build_options.enable_glm4) @import("glm4.zig").Glm4Model else void;
/// Nemotron Nano 30B-A3B hybrid Mamba-2 + MoE + Attention model — conditionally compiled via -Denable-nemotron-nano.
pub const NemotronNanoModel = if (build_options.enable_nemotron_nano) @import("nemotron_nano.zig").NemotronNanoModel else void;

// ── Tests ─────────────────────────────────────────────────────────

test "expertWeightStride f32 2x2 layout" {
    const t = format_mod.TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 4, 4, 2, 0 },
        .dtype = .f32,
        .data_ptr = undefined,
    };
    // 4*4 = 16 elements per expert, 4 bytes each = 64 bytes.
    try std.testing.expectEqual(@as(usize, 64), expertWeightStride(t));
}
