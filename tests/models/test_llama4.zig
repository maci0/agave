const golden = @import("golden_harness.zig");

const model_path = "models/llama-4/llama-4.gguf";
const test_prompt = "What is the capital of France?";
const model_name = "llama4";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "Llama 4 CPU" {
    try testBackend("cpu");
}

test "Llama 4 Metal" {
    try testBackend("metal");
}

test "Llama 4 CUDA" {
    try testBackend("cuda");
}

test "Llama 4 Vulkan" {
    try testBackend("vulkan");
}

test "Llama 4 ROCm" {
    try testBackend("rocm");
}
