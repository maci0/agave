# Chapter 15: Chat Templates

A chat model expects prompts in a specific format with special tokens marking roles (user, assistant, system). Hardcoding these in model code creates **tight coupling** and makes the codebase fragile. **Chat templates** are data-driven: role markers and end-of-generation tokens are **configuration**, not code.

## The Problem: Hardcoded Prompt Formatting

**Bad pattern** (don't do this):

```zig
// Hardcoded in qwen35.zig
pub fn formatPrompt(user_msg: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
        .{user_msg}
    );
}

// Hardcoded end-of-generation check
pub fn isEOG(token_id: u32) bool {
    return token_id == 151643 or token_id == 151645;  // <|im_end|>, <|endoftext|>
}
```

**Problems:**

1. **Non-portable:** Different models use different markers (Llama uses `[INST]`, Gemma uses `<start_of_turn>`)
2. **Duplicate logic:** Every model file has its own prompt builder
3. **Brittle:** EOG token IDs change between model versions
4. **Unmaintainable:** Adding multi-turn chat requires editing every model

## The Solution: Data-Driven Templates

**Template structure** (from `src/chat_template.zig`):

```zig
pub const ChatTemplate = struct {
    system_prefix: []const u8,
    system_suffix: []const u8,
    user_prefix: []const u8,
    user_suffix: []const u8,
    assistant_prefix: []const u8,
    assistant_suffix: []const u8,
    eog_tokens: []const []const u8,  // Token names, not IDs
    default_system: ?[]const u8 = null,
    system_role_override: ?struct { prefix: []const u8, suffix: []const u8 } = null,
    generation_prefix: []const u8 = "",
};
```

**Example template** (Qwen3.5):

```zig
pub const qwen35_template = ChatTemplate{
    .system_prefix = "<|im_start|>system\n",
    .system_suffix = "<|im_end|>\n",
    .user_prefix = "<|im_start|>user\n",
    .user_suffix = "<|im_end|>\n",
    .assistant_prefix = "<|im_start|>assistant\n",
    .assistant_suffix = "<|im_end|>\n",
    .eog_tokens = &.{"<|im_end|>", "<|endoftext|>"},
    .generation_prefix = "\n<think>\n</think>\n",  // Suppress reasoning
};
```

## Template Usage

### Single-Turn Prompt

```zig
const template = qwen35_template;
const prompt = try template.format(
    allocator,
    "You are a helpful assistant.",  // System message
    "What is 2+2?"                   // User message
);
defer allocator.free(prompt);

// Result:
// <|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// What is 2+2?<|im_end|>
// <|im_start|>assistant
// <think>
// </think>
```

**Note:** `generation_prefix` is only appended **after the final assistant prefix** when generating a response, not for past assistant messages in conversation history.

### Multi-Turn Conversation

```zig
const messages = [_]Message{
    .{ .role = .user, .content = "Hello!" },
    .{ .role = .assistant, .content = "Hi there!" },
    .{ .role = .user, .content = "How are you?" },
};

const prompt = try template.formatConversation(
    allocator,
    null,  // No system message
    &messages
);

// Result:
// <|im_start|>user
// Hello!<|im_end|>
// <|im_start|>assistant
// Hi there!<|im_end|>
// <|im_start|>user
// How are you?<|im_end|>
// <|im_start|>assistant
// <think>
// </think>
```

## Architecture-Specific Templates

**Each model architecture has its own template** (defined in `src/chat_template.zig`):

### Gemma 3

```zig
pub const gemma3_template = ChatTemplate{
    .system_prefix = "<start_of_turn>user\n",  // No dedicated system role
    .system_suffix = "<end_of_turn>\n",
    .user_prefix = "<start_of_turn>user\n",
    .user_suffix = "<end_of_turn>\n",
    .assistant_prefix = "<start_of_turn>model\n",
    .assistant_suffix = "<end_of_turn>\n",
    .eog_tokens = &.{"<end_of_turn>", "<eos>"},
};
```

**Note:** Gemma doesn't have a separate system role — system messages use the user prefix.

### Llama 3.1 / GPT-OSS

```zig
pub const llama31_template = ChatTemplate{
    .system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n",
    .system_suffix = "<|eot_id|>",
    .user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n",
    .user_suffix = "<|eot_id|>",
    .assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    .assistant_suffix = "<|eot_id|>",
    .eog_tokens = &.{"<|eot_id|>", "<|end_of_text|>"},
};
```

### GLM-4

```zig
pub const glm4_template = ChatTemplate{
    .system_prefix = "[gMASK]<sop><|system|>\n",
    .system_suffix = "",
    .user_prefix = "<|user|>\n",
    .user_suffix = "",
    .assistant_prefix = "<|assistant|>\n",
    .assistant_suffix = "",
    .eog_tokens = &.{"<|endoftext|>", "<|user|>", "<|observation|>"},
};
```

### Nemotron-H (Llama 3.1 Compatible)

```zig
pub const nemotron_h_template = llama31_template;  // Reuse Llama 3.1 template
```

## Template Selection

**Architecture determines template** (from `src/arch.zig`):

```zig
pub const Arch = enum {
    gemma3,
    qwen35,
    gpt_oss,
    nemotron_h,
    nemotron_nano,
    glm4,

    pub fn chatTemplate(self: Arch) ChatTemplate {
        return switch (self) {
            .gemma3 => chat_template.gemma3_template,
            .qwen35 => chat_template.qwen35_template,
            .gpt_oss => chat_template.llama31_template,
            .nemotron_h => chat_template.nemotron_h_template,
            .nemotron_nano => chat_template.llama31_template,
            .glm4 => chat_template.glm4_template,
        };
    }
};
```

**Main loop uses architecture's template:**

```zig
const arch = try Arch.detect(fmt);
const template = arch.chatTemplate();

const prompt = if (args.system_msg) |sys|
    try template.format(allocator, sys, args.user_msg)
else
    try template.format(allocator, null, args.user_msg);
```

**No model-specific code needed** — the architecture enum handles it.

## End-of-Generation Token Detection

**Templates define EOG tokens by name**, not by ID. The tokenizer resolves them at runtime.

### Template Definition

```zig
.eog_tokens = &.{"<|im_end|>", "<|endoftext|>"},
```

### Tokenizer Lookup

```zig
// src/main.zig
pub fn resolveEOGTokens(
    allocator: Allocator,
    tokenizer: *Tokenizer,
    template: ChatTemplate,
) ![]u32 {
    var eog_ids = std.ArrayList(u32).empty;

    for (template.eog_tokens) |token_name| {
        // Encode the special token name
        const ids = try tokenizer.encode(allocator, token_name, .{});
        defer allocator.free(ids);

        if (ids.len == 1) {
            try eog_ids.append(allocator, ids[0]);
        } else {
            std.log.warn("EOG token '{s}' encoded to {d} tokens (expected 1), skipping",
                .{token_name, ids.len});
        }
    }

    return eog_ids.toOwnedSlice(allocator);
}
```

**Generation loop:**

```zig
const eog_ids = try resolveEOGTokens(allocator, &tokenizer, template);
defer allocator.free(eog_ids);

while (n_gen < args.n_predict) : (n_gen += 1) {
    const token_id = try model.forward(current_token);

    // Check for EOG
    for (eog_ids) |eog| {
        if (token_id == eog) break :outer;
    }

    // ... emit token ...
}
```

**Why token names?** Token IDs vary between tokenizers (e.g., same model with different vocab files). Token names are stable.

## Special Features

### Default System Message

Some models inject a **fixed system message** before the user's system prompt.

**Example:** GPT-OSS reasoning model:

```zig
pub const gpt_oss_reasoning_template = ChatTemplate{
    .system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n",
    .system_suffix = "<|eot_id|>",
    .user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n",
    .user_suffix = "<|eot_id|>",
    .assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    .assistant_suffix = "<|eot_id|>",
    .eog_tokens = &.{"<|eot_id|>"},
    .default_system = "You are a helpful assistant with advanced reasoning capabilities. " ++
        "Show your thought process step-by-step before answering.",
};
```

**Behavior:**

```zig
const prompt = try template.format(allocator, "Be concise.", "What is 2+2?");
// Result:
// <|start_header_id|>system<|end_header_id|>
//
// You are a helpful assistant with advanced reasoning capabilities. Show your thought process step-by-step before answering.<|eot_id|>
// <|start_header_id|>user<|end_header_id|>
//
// What is 2+2?<|eot_id|>
// ...
```

**Note:** User-provided system message `"Be concise."` is **ignored** when `default_system` is set and no `system_role_override` exists (prevents duplicate system prompts).

### System Role Override

Some models don't have a dedicated system role — they map system messages to the user role.

**Example:** Gemma 3:

```zig
pub const gemma3_template = ChatTemplate{
    .system_prefix = "<start_of_turn>user\n",  // System uses user prefix
    .system_suffix = "<end_of_turn>\n",
    // ...
};
```

**Alternative:** Explicitly override the system role:

```zig
pub const custom_template = ChatTemplate{
    .system_prefix = "<start_of_turn>system\n",  // Default system prefix (unused)
    .system_suffix = "<end_of_turn>\n",
    .user_prefix = "<start_of_turn>user\n",
    .user_suffix = "<end_of_turn>\n",
    .assistant_prefix = "<start_of_turn>assistant\n",
    .assistant_suffix = "<end_of_turn>\n",
    .eog_tokens = &.{"<end_of_turn>"},
    .system_role_override = .{
        .prefix = "<start_of_turn>user\n",  // Override: use user prefix for system
        .suffix = "<end_of_turn>\n",
    },
};
```

**When to use:** Template wants to use user role for system messages, but still allow user-provided system text (unlike `default_system` which ignores user input).

### Generation Prefix

**Qwen3.5 reasoning suppression:** Empty `<think>` block disables reasoning (greedy decoding makes open-ended reasoning unstable).

```zig
pub const qwen35_template = ChatTemplate{
    // ...
    .generation_prefix = "\n<think>\n</think>\n",
};
```

**Applied only to the final assistant turn:**

```zig
// Past assistant message (in conversation history)
<|im_start|>assistant
Previous response<|im_end|>

// New assistant response (generation)
<|im_start|>assistant
<think>
</think>
← generation starts here
```

**Why?** Past assistant messages are complete — they don't need reasoning suppression. Only the **new generation** needs the empty think block.

## Implementation Details

### Format Function

```zig
pub fn format(
    self: ChatTemplate,
    allocator: Allocator,
    system_msg: ?[]const u8,
    user_msg: []const u8,
) ![]u8 {
    return self.formatConversation(allocator, system_msg, &.{
        .{ .role = .user, .content = user_msg },
    });
}
```

### Multi-Turn Format Function

```zig
pub fn formatConversation(
    self: ChatTemplate,
    allocator: Allocator,
    system_msg: ?[]const u8,
    messages: []const Message,
) ![]u8 {
    var result = std.ArrayList(u8).empty;

    // 1. Fixed default system message
    if (self.default_system) |ds| {
        try result.appendSlice(allocator, self.system_prefix);
        try result.appendSlice(allocator, ds);
        try result.appendSlice(allocator, self.system_suffix);
    }

    // 2. User-provided system message (if no default or role override exists)
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

    // 3. Conversation messages
    for (messages, 0..) |msg, i| {
        const is_last = (i == messages.len - 1);

        switch (msg.role) {
            .user => {
                try result.appendSlice(allocator, self.user_prefix);
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, self.user_suffix);
            },
            .assistant => {
                try result.appendSlice(allocator, self.assistant_prefix);
                if (is_last and self.generation_prefix.len > 0) {
                    try result.appendSlice(allocator, self.generation_prefix);
                }
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, self.assistant_suffix);
            },
        }
    }

    // 4. Final assistant prefix for generation
    try result.appendSlice(allocator, self.assistant_prefix);
    if (self.generation_prefix.len > 0) {
        try result.appendSlice(allocator, self.generation_prefix);
    }

    return result.toOwnedSlice(allocator);
}
```

## Benefits of Data-Driven Templates

### Maintainability

✅ **Single source of truth:** All prompt formatting logic in `chat_template.zig`
✅ **Easy to add models:** Define a template, map it in `arch.zig`, done
✅ **No model code changes:** Adding multi-turn support doesn't touch model files

### Testability

```zig
test "qwen35 template single-turn" {
    const template = qwen35_template;
    const prompt = try template.format(allocator, "You are helpful.", "Hello!");
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello!") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>assistant\n") != null);
}
```

### Flexibility

- Different tokenizers → resolve EOG token IDs at runtime
- Different model versions → template stays the same
- Custom models → user can define their own template

## Common Patterns

### Llama-Style (Header Tags)

```zig
.system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n",
.system_suffix = "<|eot_id|>",
.user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n",
.user_suffix = "<|eot_id|>",
.assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n",
.assistant_suffix = "<|eot_id|>",
```

### ChatML-Style (Im Tags)

```zig
.system_prefix = "<|im_start|>system\n",
.system_suffix = "<|im_end|>\n",
.user_prefix = "<|im_start|>user\n",
.user_suffix = "<|im_end|>\n",
.assistant_prefix = "<|im_start|>assistant\n",
.assistant_suffix = "<|im_end|>\n",
```

### Turn-Based (Gemma)

```zig
.system_prefix = "<start_of_turn>user\n",
.system_suffix = "<end_of_turn>\n",
.user_prefix = "<start_of_turn>user\n",
.user_suffix = "<end_of_turn>\n",
.assistant_prefix = "<start_of_turn>model\n",
.assistant_suffix = "<end_of_turn>\n",
```

## Future Extensions

**Potential additions** (not yet implemented):

- **Jinja2 template support:** Parse HuggingFace's `.jinja` templates directly
- **Tool/function calling:** Special formatting for function results
- **Multi-modal:** Image/audio/video markers
- **Custom templates via CLI:** `--template path/to/template.json`

---

**In the code:** [src/chat_template.zig](../../src/chat_template.zig) (template definitions and format functions), [src/arch.zig](../../src/arch.zig) (architecture → template mapping), [src/main.zig](../../src/main.zig) (EOG token resolution)

**Related:** [Tokenization](01-tokens-and-text.md) (how tokens are encoded/decoded)

**Back:** [Chapter 14: Format Conventions ←](14-format-conventions.md) | **Product docs:** [Models](../MODELS.md)
