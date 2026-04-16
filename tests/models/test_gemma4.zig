const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf";
const test_prompt = "Explain the theory of relativity.";
const model_name = "gemma4";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "Gemma4 CPU" {
    try testBackend("cpu");
}

test "Gemma4 Metal" {
    try testBackend("metal");
}

test "Gemma4 CUDA" {
    try testBackend("cuda");
}

test "Gemma4 Vulkan" {
    try testBackend("vulkan");
}

test "Gemma4 ROCm" {
    try testBackend("rocm");
}
