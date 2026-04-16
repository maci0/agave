const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_0.gguf";
const test_prompt = "What is the capital of France?";
const model_name = "gemma3";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "Gemma3 CPU" {
    try testBackend("cpu");
}

test "Gemma3 Metal" {
    try testBackend("metal");
}

test "Gemma3 CUDA" {
    try testBackend("cuda");
}

test "Gemma3 Vulkan" {
    try testBackend("vulkan");
}

test "Gemma3 ROCm" {
    try testBackend("rocm");
}
