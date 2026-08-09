const golden = @import("golden_harness.zig");

const model_path = "models/diffusiongemma/diffusiongemma-26B-A4B-it";
const test_prompt = "Describe the future of artificial intelligence.";
const model_name = "diffusion_gemma";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "DiffusionGemma CPU" {
    try testBackend("cpu");
}

test "DiffusionGemma Metal" {
    try testBackend("metal");
}

test "DiffusionGemma CUDA" {
    try testBackend("cuda");
}

test "DiffusionGemma Vulkan" {
    try testBackend("vulkan");
}

test "DiffusionGemma ROCm" {
    try testBackend("rocm");
}
