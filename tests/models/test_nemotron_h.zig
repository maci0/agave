const golden = @import("golden_test.zig");

// Using Nemotron Nano 30B as Nemotron-H test model (SafeTensors NVFP4)
const model_path = "models/mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
const test_prompt = "Describe the water cycle.";
const model_name = "nemotron_h";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, true);
}

test "Nemotron-H CPU" {
    try testBackend("cpu");
}

test "Nemotron-H Metal" {
    try testBackend("metal");
}

test "Nemotron-H CUDA" {
    try testBackend("cuda");
}

test "Nemotron-H Vulkan" {
    try testBackend("vulkan");
}

test "Nemotron-H ROCm" {
    try testBackend("rocm");
}
