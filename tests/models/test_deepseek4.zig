const golden = @import("golden_harness.zig");

const model_path = "models/deepseek-v4/deepseek-v4.gguf";
const test_prompt = "What is the capital of France?";
const model_name = "deepseek4";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "DeepSeek V4 CPU" {
    try testBackend("cpu");
}

test "DeepSeek V4 Metal" {
    try testBackend("metal");
}

test "DeepSeek V4 CUDA" {
    try testBackend("cuda");
}

test "DeepSeek V4 Vulkan" {
    try testBackend("vulkan");
}

test "DeepSeek V4 ROCm" {
    try testBackend("rocm");
}
