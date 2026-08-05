//! Multimodal image placeholder token IDs.
//!
//! Leaf type shared by `arch.zig` (per-architecture defaults) and
//! `chat_template.zig` (`injectImageTokens`). Lives in its own module so
//! those two files do not form an import cycle.

/// Image token IDs for multimodal models.
/// These are special tokens in the vocabulary that serve as placeholders
/// for visual embeddings during forward passes.
pub const ImageTokens = struct {
    /// Start-of-image token ID (e.g. `<img>`, `<|vision_start|>`).
    start: u32,
    /// End-of-image token ID (e.g. `</img>`, `<|vision_end|>`).
    end: u32,
    /// Placeholder token ID repeated n_visual_tokens times between start/end.
    pad: u32,
};
