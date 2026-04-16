const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q8_0.gguf";
const test_prompt = "What is quantum computing?";
const model_name = "glm4";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "GLM4 CPU" {
    try testBackend("cpu");
}

test "GLM4 Metal" {
    try testBackend("metal");
}

test "GLM4 CUDA" {
    try testBackend("cuda");
}

test "GLM4 Vulkan" {
    try testBackend("vulkan");
}

test "GLM4 ROCm" {
    try testBackend("rocm");
}
