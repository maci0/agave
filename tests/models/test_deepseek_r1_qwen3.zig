const golden = @import("golden_test.zig");

const model_path = "models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf";
const test_prompt = "Write a Python function to calculate factorial.";
const model_name = "deepseek_r1_qwen3";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "DeepSeek-R1-Qwen3 CPU" {
    try testBackend("cpu");
}

test "DeepSeek-R1-Qwen3 Metal" {
    try testBackend("metal");
}

test "DeepSeek-R1-Qwen3 CUDA" {
    try testBackend("cuda");
}

test "DeepSeek-R1-Qwen3 Vulkan" {
    try testBackend("vulkan");
}

test "DeepSeek-R1-Qwen3 ROCm" {
    try testBackend("rocm");
}
