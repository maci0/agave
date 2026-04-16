const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf";
const test_prompt = "Explain photosynthesis in simple terms.";
const model_name = "qwen35";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "Qwen35 CPU" {
    try testBackend("cpu");
}

test "Qwen35 Metal" {
    try testBackend("metal");
}

test "Qwen35 CUDA" {
    try testBackend("cuda");
}

test "Qwen35 Vulkan" {
    try testBackend("vulkan");
}

test "Qwen35 ROCm" {
    try testBackend("rocm");
}
