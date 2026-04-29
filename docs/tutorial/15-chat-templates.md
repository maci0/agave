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

1. **Non-portable:** Different models use different markers (GPT-OSS uses `<|start|>`, Gemma uses `<start_of_turn>`)
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

**Example template** (ChatML, used by Qwen3.5):

```zig
pub const qwen35 = ChatTemplate{
    .system_prefix = "<|im_start|>system\n",
    .system_suffix = "<|im_end|>\n",
    .user_prefix = "<|im_start|>user\n",
    .user_suffix = "",
    .assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n",
    .assistant_suffix = "<|im_end|>\n",
    .eog_tokens = &.{ "<|im_end|>", "<|endoftext|>" },
    .generation_prefix = "<think>\n\n</think>\n\n",  // Suppress reasoning
};
```

**Note:** `user_suffix` is empty because `assistant_prefix` already includes `<|im_end|>\n` — the end-of-user marker is baked into the transition.

## Template Usage

### Single-Turn Prompt

```zig
const template = ChatTemplate.qwen35;
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
//
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
//
// </think>
```

## Architecture-Specific Templates

**Each model architecture has its own template** (defined in `src/chat_template.zig`):

### Gemma 3

```zig
pub const gemma = ChatTemplate{
    .system_prefix = "<start_of_turn>user\n",  // No dedicated system role
    .system_suffix = "\n\n",
    .user_prefix = "<start_of_turn>user\n",
    .user_suffix = "",
    .assistant_prefix = "<end_of_turn>\n<start_of_turn>model\n",
    .assistant_suffix = "<end_of_turn>\n",
    .eog_tokens = &.{ "<end_of_turn>", "<eos>" },
};
```

**Note:** Gemma doesn't have a separate system role — system messages use the user prefix. The `assistant_prefix` includes `<end_of_turn>\n` to close the prior turn before opening the model turn.

### Gemma 4

```zig
pub const gemma4 = ChatTemplate{
    .system_prefix = "<|turn>system\n",
    .system_suffix = "<turn|>\n",
    .user_prefix = "<|turn>user\n",
    .user_suffix = "<turn|>\n",
    .assistant_prefix = "<|turn>model\n",
    .assistant_suffix = "<turn|>\n",
    .eog_tokens = &.{ "<turn|>", "<eos>", "<channel|>", "<|endoftext|>", "<|end|>" },
    .generation_prefix = "<|channel>0\n<channel|>",
};
```

**Note:** Gemma 4 uses a channel system. `generation_prefix` selects channel 0 (direct answer) and closes it immediately, preventing reasoning tokens.

### GPT-OSS

```zig
pub const gpt_oss = ChatTemplate{
    .system_prefix = "<|start|>system<|message|>",
    .system_suffix = "<|end|>",
    .user_prefix = "<|start|>user<|message|>",
    .user_suffix = "",
    .assistant_prefix = "<|end|><|start|>assistant",
    .assistant_suffix = "<|end|>",
    .eog_tokens = &.{ "<|end|>", "<|endoftext|>" },
    .default_system = "You are a helpful assistant.\n" ++
        "Reasoning: medium\n" ++
        "# Valid channels: analysis, commentary, final. " ++
        "Channel must be included for every message.",
    .system_role_override = .{
        .prefix = "<|start|>developer<|message|># Instructions\n",
        .suffix = "<|end|>",
    },
};
```

**Note:** GPT-OSS uses `<|start|>`/`<|end|>` markers (not Llama-style headers). It has both a `default_system` message and a `system_role_override` — user-provided system messages are formatted as "developer" instructions.

### GLM-4

```zig
pub const glm4 = ChatTemplate{
    .system_prefix = "<sop>",
    .system_suffix = "",
    .user_prefix = "<|user|>",
    .user_suffix = "",
    .assistant_prefix = "<|assistant|>\n",
    .assistant_suffix = "",
    .eog_tokens = &.{ "<|endoftext|>", "<|user|>", "<|observation|>" },
    .default_system = "",
    .generation_prefix = "</think>",
    .system_role_override = .{
        .prefix = "<|system|>\n",
        .suffix = "",
    },
};
```

**Note:** GLM-4 uses `<sop>` as the initial BOS marker. The `system_role_override` maps user-provided system messages to the `<|system|>` role. `generation_prefix = "</think>"` disables reasoning mode (forces direct answers).

### Nemotron-H / Nemotron-Nano (ChatML)

```zig
// Both use the default ChatML template (via the `else` fallback in arch.zig)
pub const chatml = ChatTemplate{
    .system_prefix = "<|im_start|>system\n",
    .system_suffix = "<|im_end|>\n",
    .user_prefix = "<|im_start|>user\n",
    .user_suffix = "",
    .assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n",
    .assistant_suffix = "<|im_end|>\n",
    .eog_tokens = &.{ "<|im_end|>", "<|endoftext|>" },
};
```

## Template Selection

**Architecture determines template** (from `src/arch.zig`):

```zig
pub const Arch = enum {
    gemma3,
    gemma4,
    qwen35,
    gpt_oss,
    nemotron_h,
    nemotron_nano,
    glm4,

    pub fn chatTemplate(self: Arch) ChatTemplate {
        return switch (self) {
            .gemma3 => ChatTemplate.gemma,
            .gemma4 => ChatTemplate.gemma4,
            .gpt_oss => ChatTemplate.gpt_oss,
            .qwen35 => ChatTemplate.qwen35,
            .glm4 => ChatTemplate.glm4,
            else => ChatTemplate.chatml,  // Nemotron-H, Nemotron-Nano
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
.eog_tokens = &.{ "<|im_end|>", "<|endoftext|>" },
```

### Tokenizer Lookup

At startup, the engine looks up each EOG token name in the tokenizer's special token
map (loaded from GGUF metadata or `tokenizer.json`):

```zig
// src/main.zig — EOG token resolution
const tmpl = arch.chatTemplate();
for (tmpl.eog_tokens) |eog_name| {
    if (tok.special_tokens.get(eog_name)) |id| {
        if (!isEogToken(id, eog) and eog.len < eog.ids.len) {
            eog.ids[eog.len] = id;
            eog.len += 1;
        }
    }
}
```

During generation, each produced token is checked against the resolved EOG IDs to
detect when the model signals end-of-generation.

**Why token names?** Token IDs vary between tokenizers (e.g., same model with different vocab files). Token names are stable.

## Special Features

### Default System Message

Some models inject a **fixed system message** before the user's system prompt.

**Example:** GPT-OSS includes a default system prompt with reasoning instructions:

```zig
pub const gpt_oss = ChatTemplate{
    .system_prefix = "<|start|>system<|message|>",
    .system_suffix = "<|end|>",
    // ...
    .default_system = "You are a helpful assistant.\n" ++
        "Reasoning: medium\n# Valid channels: ...",
};
```

**Behavior:** When no user-provided system message is given, `default_system` is used automatically. When the user does provide a system message AND `system_role_override` exists, the user's message is formatted using the override (as a "developer" instruction in GPT-OSS's case), while `default_system` remains.

### System Role Override

Some models route user-provided system messages through a different role.

**Example:** GPT-OSS maps user system messages to a "developer" role:

```zig
.system_role_override = .{
    .prefix = "<|start|>developer<|message|># Instructions\n",
    .suffix = "<|end|>",
},
```

**Example:** GLM-4 maps user system messages to `<|system|>`:

```zig
.system_role_override = .{
    .prefix = "<|system|>\n",
    .suffix = "",
},
```

**When to use:** The template has a default system prompt (`default_system`) but still wants to accept user-provided system text through a different role prefix.

### Generation Prefix

**Qwen3.5 reasoning suppression:** Empty `<think>` block disables reasoning (greedy decoding makes open-ended reasoning unstable).

```zig
pub const qwen35 = ChatTemplate{
    // ...
    .generation_prefix = "<think>\n\n</think>\n\n",
};
```

**Applied only to the final assistant turn:**

```
// Past assistant message (in conversation history)
<|im_end|>
<|im_start|>assistant
Previous response<|im_end|>

// New assistant response (generation)
<|im_end|>
<|im_start|>assistant
<think>

</think>

<-- generation starts here
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

    // 4. Final assistant prefix for generation
    try result.appendSlice(allocator, self.assistant_prefix);
    try result.appendSlice(allocator, self.generation_prefix);

    return result.toOwnedSlice(allocator);
}
```

## Benefits of Data-Driven Templates

### Maintainability

- **Single source of truth:** All prompt formatting logic in `chat_template.zig`
- **Easy to add models:** Define a template, map it in `arch.zig`, done
- **No model code changes:** Adding multi-turn support doesn't touch model files

### Testability

```zig
test "chatml format basic" {
    const result = try ChatTemplate.chatml.format(std.testing.allocator, null, "Hi");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        result,
    );
}
```

### Flexibility

- Different tokenizers -> resolve EOG token IDs at runtime
- Different model versions -> template stays the same
- Custom models -> user can define their own template

## Common Patterns

### ChatML-Style (Im Tags)

Used by: Nemotron-H, Nemotron-Nano (Qwen3.5 uses a variant with `generation_prefix`)

```zig
.system_prefix = "<|im_start|>system\n",
.system_suffix = "<|im_end|>\n",
.user_prefix = "<|im_start|>user\n",
.user_suffix = "",
.assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n",
.assistant_suffix = "<|im_end|>\n",
```

### Turn-Based (Gemma 3)

```zig
.system_prefix = "<start_of_turn>user\n",
.system_suffix = "\n\n",
.user_prefix = "<start_of_turn>user\n",
.user_suffix = "",
.assistant_prefix = "<end_of_turn>\n<start_of_turn>model\n",
.assistant_suffix = "<end_of_turn>\n",
```

### Marker-Based (GPT-OSS)

```zig
.system_prefix = "<|start|>system<|message|>",
.system_suffix = "<|end|>",
.user_prefix = "<|start|>user<|message|>",
.user_suffix = "",
.assistant_prefix = "<|end|><|start|>assistant",
.assistant_suffix = "<|end|>",
```

## Image Token Injection (Multimodal)

When an image is attached to a prompt, the tokenized text needs image placeholder tokens spliced in at the right position. The chat template system handles this through two functions: `findImageInsertPos()` and `injectImageTokens()`.

### Finding the Insertion Point

`findImageInsertPos()` scans the token array for the **last occurrence** of the user-turn prefix token sequence (e.g., the tokens for `<start_of_turn>user\n`), then returns the position immediately after that match. Using the last occurrence avoids false positives when individual prefix tokens (like `\n`) appear earlier in the prompt (e.g., in the system section):

```zig
// src/chat_template.zig
pub fn findImageInsertPos(tokens: []const u32, prefix_seq: []const u32) usize {
    var last_match: usize = 0;
    if (tokens.len >= prefix_seq.len) {
        var i: usize = 0;
        while (i + prefix_seq.len <= tokens.len) : (i += 1) {
            if (std.mem.eql(u32, tokens[i..][0..prefix_seq.len], prefix_seq)) {
                last_match = i + prefix_seq.len;
            }
        }
    }
    return last_match;
}
```

### Injecting the Image Sequence

`injectImageTokens()` splices a sequence of `[start, pad, pad, ..., pad, end]` tokens at the insertion point. The pad tokens are repeated `n_visual_tokens` times (determined by the vision encoder's output patch count). During `forward()`, whenever the model encounters a pad token ID, it replaces the normal embedding lookup with the corresponding visual embedding from the vision encoder output.

### Architecture-Specific Image Tokens

Different model architectures use different special tokens for image placeholders:

| Architecture | Start Token | End Token | Pad Token | Notes |
|---|---|---|---|---|
| Gemma 4 | `<\|image\|>` (258880) | `<\|image\|>` (258880) | `<\|image\|>` (258880) | Single token for all three roles |
| Gemma 3 | `<img>` (219) | `</img>` (230) | `<img>` (219) | Distinct end token |
| Qwen 3.5 | `<\|vision_start\|>` (248053) | `<\|vision_end\|>` (248054) | `<\|image_pad\|>` (248056) | Three distinct tokens |

When start equals pad (Gemma 4), `injectImageTokens()` omits the start wrapper to avoid the model consuming the start token as a visual embedding — it just injects `pad * N + end`:

```zig
// src/chat_template.zig — architecture-aware wrapping
const has_distinct_start = image_tokens.start != image_tokens.pad;
const has_distinct_end = image_tokens.end != image_tokens.pad;
const prefix_len: usize = if (has_distinct_start) 1 else 0;
const suffix_len: usize = if (has_distinct_end) 1 else 0;
```

### Embedding Replacement During Forward

The image tokens are not just markers — they trigger embedding replacement in the model's forward pass. When `forward()` encounters a pad token ID, it copies the next visual embedding vector from the vision encoder output instead of performing the normal embedding table lookup:

```zig
// src/models/gemma4.zig — forward() embedding replacement
if (self.image_embeddings) |vis_embd| {
    if (token_id == self.image_pad_token_id) {
        const idx = self.visual_token_idx;
        const offset = @as(usize, idx) * self.n_embd;
        @memcpy(self.hidden, vis_embd[offset..][0..self.n_embd]);
        self.visual_token_idx = idx + 1;
        is_image_token = true;
    }
}
if (!is_image_token) {
    self.embLookup(token_id);  // Normal text embedding
}
```

The visual embeddings are set before generation via `model.setImageEmbeddings()`, which stores the vision encoder's output buffer and the pad token ID. The `visual_token_idx` counter advances through the visual embeddings one token at a time, ensuring each pad token gets the correct patch embedding.

## Future Extensions

**Potential additions** (not yet implemented):

- **Jinja2 template support:** Parse HuggingFace's `.jinja` templates directly
- **Tool/function calling:** Special formatting for function results
- **Multi-modal:** Audio/video markers (image tokens already supported via SigLIP-2)
- **Custom templates via CLI:** `--template path/to/template.json`

---

**In the code:** [src/chat_template.zig](../../src/chat_template.zig) (template definitions and format functions), [src/arch.zig](../../src/arch.zig) (architecture -> template mapping), [src/main.zig](../../src/main.zig) (EOG token resolution)

**Related:** [Tokenization](01-tokens-and-text.md) (how tokens are encoded/decoded)

**Next:** [Chapter 16: Recipe System →](16-recipe-system.md) | **Back:** [Chapter 14: Format Conventions ←](14-format-conventions.md) | **Product docs:** [Models](../MODELS.md)
