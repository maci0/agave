//! Chat prompt templates for different model architectures.
//! Defines the special token framing (role prefixes/suffixes) and
//! end-of-generation tokens for each supported model family.

const std = @import("std");

/// Role in a conversation message.
pub const Role = enum { user, assistant };

/// A single message in a conversation.
pub const Message = struct {
    role: Role,
    content: []const u8,
};

/// Chat template definition for a model architecture.
/// Each field pair (prefix/suffix) wraps a role's content in the prompt.
pub const ChatTemplate = struct {
    system_prefix: []const u8,
    system_suffix: []const u8,
    user_prefix: []const u8,
    user_suffix: []const u8,
    assistant_prefix: []const u8,
    /// Suffix appended after an assistant response to close the turn.
    assistant_suffix: []const u8,
    /// Well-known special token names that signal end-of-generation.
    eog_tokens: []const []const u8,
    /// Optional fixed system message prepended before the user's system prompt.
    default_system: ?[]const u8 = null,
    /// Role override for user-supplied system prompts. When set, system_msg
    /// uses this role instead of the default system_prefix/suffix.
    system_role_override: ?struct { prefix: []const u8, suffix: []const u8 } = null,
    /// Extra text appended after the final assistant_prefix when generating a
    /// response (but NOT for past assistant messages in conversation history).
    /// Used by Qwen3.5 to inject an empty `<think>` block that disables
    /// reasoning (greedy decoding makes open-ended thinking unstable).
    generation_prefix: []const u8 = "",

    /// Format a single-turn chat prompt using this template.
    pub fn format(self: ChatTemplate, allocator: std.mem.Allocator, system_msg: ?[]const u8, user_msg: []const u8) ![]u8 {
        return self.formatConversation(allocator, system_msg, &.{.{ .role = .user, .content = user_msg }});
    }

    /// Format a multi-turn conversation prompt. The last message should be
    /// from the user; the returned prompt ends with the assistant prefix
    /// so the model generates the next response.
    /// If `default_system` is set and no `system_role_override` exists,
    /// user-provided `system_msg` is ignored to avoid duplicate system prompts.
    pub fn formatConversation(self: ChatTemplate, allocator: std.mem.Allocator, system_msg: ?[]const u8, messages: []const Message) ![]u8 {
        var result = std.ArrayList(u8).empty;
        // Fixed default system message (e.g. GPT-OSS reasoning preamble)
        if (self.default_system) |ds| {
            try result.appendSlice(allocator, self.system_prefix);
            try result.appendSlice(allocator, ds);
            try result.appendSlice(allocator, self.system_suffix);
        }
        if (system_msg) |sys| {
            if (self.system_role_override) |role| {
                try result.appendSlice(allocator, role.prefix);
                try result.appendSlice(allocator, sys);
                try result.appendSlice(allocator, role.suffix);
            } else if (self.default_system == null) {
                try result.appendSlice(allocator, self.system_prefix);
                try result.appendSlice(allocator, sys);
                try result.appendSlice(allocator, self.system_suffix);
            }
        }
        for (messages) |msg| {
            switch (msg.role) {
                .user => {
                    try result.appendSlice(allocator, self.user_prefix);
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, self.user_suffix);
                },
                .assistant => {
                    try result.appendSlice(allocator, self.assistant_prefix);
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, self.assistant_suffix);
                },
            }
        }
        // End with assistant prefix so the model generates the next response
        try result.appendSlice(allocator, self.assistant_prefix);
        try result.appendSlice(allocator, self.generation_prefix);
        return result.toOwnedSlice(allocator);
    }

    /// Format a continuation prompt for KV cache reuse. Produces only the
    /// tokens needed to bridge from the end of the previous assistant
    /// response to the start of the next generation:
    /// `assistant_suffix + user_prefix + user_msg + user_suffix + assistant_prefix`
    pub fn formatContinuation(self: ChatTemplate, allocator: std.mem.Allocator, user_msg: []const u8) ![]u8 {
        var result = std.ArrayList(u8).empty;
        try result.appendSlice(allocator, self.assistant_suffix);
        try result.appendSlice(allocator, self.user_prefix);
        try result.appendSlice(allocator, user_msg);
        try result.appendSlice(allocator, self.user_suffix);
        try result.appendSlice(allocator, self.assistant_prefix);
        try result.appendSlice(allocator, self.generation_prefix);
        return result.toOwnedSlice(allocator);
    }

    // ── Preset templates ─────────────────────────────────────

    /// ChatML — Nemotron-H, Nemotron-Nano, and most open models.
    pub const chatml = ChatTemplate{
        .system_prefix = "<|im_start|>system\n",
        .system_suffix = "<|im_end|>\n",
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "",
        .assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n",
        .assistant_suffix = "<|im_end|>\n",
        .eog_tokens = &.{ "<|im_end|>", "<|endoftext|>" },
    };

    /// Qwen 3.5 — ChatML with thinking disabled (empty `<think>` block
    /// prepended to skip straight to the response). Greedy decoding without
    /// sampling makes open-ended thinking unstable, so thinking is disabled
    /// by default until sampling is implemented.
    pub const qwen35 = ChatTemplate{
        .system_prefix = "<|im_start|>system\n",
        .system_suffix = "<|im_end|>\n",
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "",
        .assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n",
        .assistant_suffix = "<|im_end|>\n",
        .eog_tokens = &.{ "<|im_end|>", "<|endoftext|>" },
        .generation_prefix = "<think>\n\n</think>\n\n",
    };

    /// Gemma 3 (and Gemma 2, which auto-detects as gemma3).
    pub const gemma = ChatTemplate{
        .system_prefix = "<start_of_turn>user\n",
        .system_suffix = "\n\n",
        .user_prefix = "<start_of_turn>user\n",
        .user_suffix = "",
        .assistant_prefix = "<end_of_turn>\n<start_of_turn>model\n",
        .assistant_suffix = "<end_of_turn>\n",
        .eog_tokens = &.{ "<end_of_turn>", "<eos>" },
    };

    /// GLM-4 — uses `[gMASK]<sop>` prefix (BOS sends `[gMASK]`, template starts
    /// with `<sop>`) and `<|user|>`/`<|assistant|>` role markers. Thinking is
    /// disabled by default via `</think>` generation prefix.
    pub const glm4 = ChatTemplate{
        .system_prefix = "<sop>",
        .system_suffix = "",
        .user_prefix = "<|user|>",
        .user_suffix = "",
        .assistant_prefix = "<|assistant|>",
        .assistant_suffix = "",
        .eog_tokens = &.{ "<|endoftext|>", "<|user|>" },
        .default_system = "",
        .system_role_override = .{
            .prefix = "<|system|>",
            .suffix = "",
        },
        .generation_prefix = "</think>",
    };

    /// GPT-OSS Harmony.
    pub const gpt_oss = ChatTemplate{
        .system_prefix = "<|start|>system<|message|>",
        .system_suffix = "<|end|>",
        .user_prefix = "<|start|>user<|message|>",
        .user_suffix = "",
        .assistant_prefix = "<|end|><|start|>assistant",
        .assistant_suffix = "<|end|>",
        .eog_tokens = &.{ "<|end|>", "<|endoftext|>" },
        .default_system = "You are a helpful assistant.\nReasoning: medium\n# Valid channels: analysis, commentary, final. Channel must be included for every message.",
        .system_role_override = .{
            .prefix = "<|start|>developer<|message|># Instructions\n",
            .suffix = "<|end|>",
        },
    };
};

// ── Tests ─────────────────────────────────────────────────────────

test "chatml format basic" {
    const result = try ChatTemplate.chatml.format(std.testing.allocator, null, "Hi");
    defer std.testing.allocator.free(result);
    // Exact output: user_prefix + content + user_suffix + assistant_prefix
    try std.testing.expectEqualStrings("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", result);
}

test "chatml format with system" {
    const result = try ChatTemplate.chatml.format(std.testing.allocator, "Be helpful", "Hello");
    defer std.testing.allocator.free(result);
    // System must appear before user
    try std.testing.expect(std.mem.startsWith(u8, result, "<|im_start|>system\nBe helpful<|im_end|>\n"));
    // Verify correct structure: system + user + assistant prefix
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}

test "gemma format basic" {
    const result = try ChatTemplate.gemma.format(std.testing.allocator, null, "Hi");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n", result);
}

test "gpt_oss format has default system" {
    const result = try ChatTemplate.gpt_oss.format(std.testing.allocator, null, "Hi");
    defer std.testing.allocator.free(result);
    // Default system must appear before user
    const sys_pos = std.mem.indexOf(u8, result, "You are a helpful assistant.") orelse return error.TestUnexpectedResult;
    const user_pos = std.mem.indexOf(u8, result, "<|start|>user<|message|>Hi") orelse return error.TestUnexpectedResult;
    try std.testing.expect(sys_pos < user_pos);
    try std.testing.expect(std.mem.endsWith(u8, result, "<|end|><|start|>assistant"));
}

test "chatml multi-turn conversation" {
    const messages = &[_]Message{
        .{ .role = .user, .content = "hello" },
        .{ .role = .assistant, .content = "Hi there!" },
        .{ .role = .user, .content = "my name is marcel" },
    };
    const result = try ChatTemplate.chatml.formatConversation(std.testing.allocator, null, messages);
    defer std.testing.allocator.free(result);
    // Verify correct ordering: user1 < assistant < user2
    const pos_u1 = std.mem.indexOf(u8, result, "<|im_start|>user\nhello") orelse return error.TestUnexpectedResult;
    const pos_a1 = std.mem.indexOf(u8, result, "<|im_start|>assistant\nHi there!<|im_end|>") orelse return error.TestUnexpectedResult;
    const pos_u2 = std.mem.indexOf(u8, result, "<|im_start|>user\nmy name is marcel") orelse return error.TestUnexpectedResult;
    try std.testing.expect(pos_u1 < pos_a1);
    try std.testing.expect(pos_a1 < pos_u2);
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n"));
}

test "gemma multi-turn conversation" {
    const messages = &[_]Message{
        .{ .role = .user, .content = "hello" },
        .{ .role = .assistant, .content = "Hi!" },
        .{ .role = .user, .content = "what is my name?" },
    };
    const result = try ChatTemplate.gemma.formatConversation(std.testing.allocator, null, messages);
    defer std.testing.allocator.free(result);
    // Verify correct ordering
    const pos_u1 = std.mem.indexOf(u8, result, "<start_of_turn>user\nhello") orelse return error.TestUnexpectedResult;
    const pos_a1 = std.mem.indexOf(u8, result, "<start_of_turn>model\nHi!<end_of_turn>") orelse return error.TestUnexpectedResult;
    const pos_u2 = std.mem.indexOf(u8, result, "<start_of_turn>user\nwhat is my name?") orelse return error.TestUnexpectedResult;
    try std.testing.expect(pos_u1 < pos_a1);
    try std.testing.expect(pos_a1 < pos_u2);
    try std.testing.expect(std.mem.endsWith(u8, result, "<start_of_turn>model\n"));
}

test "chatml continuation for KV cache reuse" {
    const result = try ChatTemplate.chatml.formatContinuation(std.testing.allocator, "what is my name?");
    defer std.testing.allocator.free(result);
    // Should start with assistant_suffix (closing previous assistant turn)
    try std.testing.expect(std.mem.startsWith(u8, result, "<|im_end|>\n"));
    // Should contain the new user message
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\nwhat is my name?") != null);
    // Should end with assistant_prefix
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n"));
}

test "gemma continuation for KV cache reuse" {
    const result = try ChatTemplate.gemma.formatContinuation(std.testing.allocator, "what is my name?");
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.startsWith(u8, result, "<end_of_turn>\n"));
    try std.testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>user\nwhat is my name?") != null);
    try std.testing.expect(std.mem.endsWith(u8, result, "<start_of_turn>model\n"));
}

test "qwen35 format includes generation prefix" {
    const result = try ChatTemplate.qwen35.format(std.testing.allocator, null, "Hi");
    defer std.testing.allocator.free(result);
    // Must end with assistant_prefix + generation_prefix
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "qwen35 continuation includes generation prefix" {
    const result = try ChatTemplate.qwen35.formatContinuation(std.testing.allocator, "what?");
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "continuation matches full format suffix" {
    // Verify that formatContinuation produces the same trailing text as
    // formatConversation — ensuring KV cache reuse sees identical tokens.
    const alloc = std.testing.allocator;
    const response = "Hi there!";
    const user2 = "what is my name?";

    const full = try ChatTemplate.chatml.formatConversation(alloc, null, &.{
        .{ .role = .user, .content = "hello" },
        .{ .role = .assistant, .content = response },
        .{ .role = .user, .content = user2 },
    });
    defer alloc.free(full);

    const cont = try ChatTemplate.chatml.formatContinuation(alloc, user2);
    defer alloc.free(cont);

    // The full format should end with exactly the continuation text
    try std.testing.expect(std.mem.endsWith(u8, full, cont));
}
