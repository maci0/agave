const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf";
const test_prompt = "List three benefits of exercise.";
const model_name = "nemotron_nano";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, true);
}

test "Nemotron Nano CPU" {
    try testBackend("cpu");
}

test "Nemotron Nano Metal" {
    try testBackend("metal");
}

test "Nemotron Nano CUDA" {
    try testBackend("cuda");
}

test "Nemotron Nano Vulkan" {
    try testBackend("vulkan");
}

test "Nemotron Nano ROCm" {
    try testBackend("rocm");
}
