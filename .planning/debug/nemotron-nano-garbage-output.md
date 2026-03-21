---
status: investigating
trigger: "nemotron-nano-garbage-output"
created: 2026-03-22T00:00:00Z
updated: 2026-03-22T00:00:00Z
---

## Current Focus

hypothesis: ROOT CAUSE FOUND — Prompt is not being formatted with ChatML template. Input tokens are `[1, 10, 3263]` = `<BOS>\nHello` instead of proper ChatML `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant`. Model is responding to raw "Hello" instead of a chat turn.
test: Check how prompt templating works in main.zig — likely the template is not being applied or is applied incorrectly
expecting: main.zig is passing raw prompt text without applying the ChatML template
next_action: Find where prompt templating happens and fix it to use ChatML format correctly

## Symptoms

expected: Coherent text response to "Hello" prompt
actual: Repeating pattern of newlines then 'c' characters (tokens 1010 and 1256 in a loop). Model loads fine, all 42 layers execute without errors, but logits are wrong.
errors: No errors or crashes. Model runs to completion but output is nonsensical.
reproduction: ./zig-out/bin/agave models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf --backend cpu "Hello"
started: Model has never produced correct output in this codebase. The 30B SafeTensors variant also produces garbage (MoE router scores of -2.3e26 were observed previously). This 4B GGUF variant is Q8_0 quantized.

## Eliminated

## Evidence

- timestamp: 2026-03-22T00:05:00Z
  checked: Ran model with "Hello" prompt
  found: Model outputs repeating newlines then 'c' characters (tokens 1010 and 1256 loop). Layers execute without errors but output is garbage.
  implication: Logits calculation is fundamentally wrong. Not a crash/error but wrong math somewhere in forward pass.

- timestamp: 2026-03-22T00:10:00Z
  checked: Read nemotron_h.zig implementation
  found: Model has three layer types (SSM, attention, ffn_only) detected from tensor presence. SSM uses Mamba-2 recurrence. Architecture looks reasonable but needs comparison to reference.
  implication: Need to verify layer type detection is correct and compare implementation against llama.cpp

- timestamp: 2026-03-22T00:20:00Z
  checked: llama.cpp src/models/nemotron-h.cpp and layer detection logic
  found: llama.cpp uses per-layer arrays `n_head_kv[i]` and `n_ff[i]` from GGUF metadata to determine layer types. Rule: `recurrent_layer_arr[i] = (n_head_kv(i) == 0 && n_ff(i) == 0)`. Our code checks for tensor presence which is wrong!
  implication: Layer type detection is fundamentally broken. We're checking if `blk.{l}.ssm_in.weight` exists, but should be reading per-layer metadata arrays.

- timestamp: 2026-03-22T00:30:00Z
  checked: Added diagnostic output to show layer type counts
  found: Model detects 21 SSM, 4 attention, 17 FFN layers (total 42) — this matches GGUF tensor structure.
  implication: Layer detection appears correct by tensor presence. The per-layer array approach might not be the issue. Need to check if the forward pass logic is correct or if logits are being computed wrong.

- timestamp: 2026-03-22T00:45:00Z
  checked: Added diagnostics to show input token IDs, logits, and hidden state evolution
  found: Input tokens are `[1, 10, 3263]` = `<BOS>\nHello`. Model generates reasonable logits (values 6-9 range), hidden state updates correctly. Model picks token 1045 first, then 1044 repeatedly (which decodes to 'c').
  implication: **The model inference is actually working!** The bug is that the prompt is NOT being formatted with the ChatML template. It's receiving raw text `\nHello` instead of `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant`. Model is correctly responding to a malformed prompt with garbage.

## Resolution

root_cause: HYPOTHESIS ELIMINATED. The ChatML prompt IS being formatted and encoded correctly. The full sequence is: `[BOS, <|im_start|>, user, \n, Hello, <|im_end|>, \n, <|im_start|>, assistant, <newline>]` (10 tokens total). The model processes all 10 tokens correctly. The garbage output starts from position 10 (first generated token). Need to investigate why the model's first response token is wrong — possibly wrong chat template (should there be a trailing newline after "assistant"?), or model genuinely doesn't know how to respond in Chat format.
fix: Need to verify the exact ChatML format this model expects. The format used is `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant` but maybe it should be `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n` (with trailing newline).
verification: Test with correct ChatML format variations, or try a different prompt to see if model can generate reasonable text at all.
files_changed: [src/chat_template.zig]
