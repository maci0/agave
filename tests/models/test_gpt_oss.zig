const golden = @import("golden_harness.zig");

const model_path = "models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf";
const test_prompt = "Once upon a time in a distant galaxy,";
const model_name = "gpt_oss";

fn testBackend(backend: []const u8) !void {
    return golden.runGoldenTest(model_path, test_prompt, model_name, backend, false);
}

test "GPT-OSS CPU" {
    try testBackend("cpu");
}

test "GPT-OSS Metal" {
    try testBackend("metal");
}

test "GPT-OSS CUDA" {
    try testBackend("cuda");
}

test "GPT-OSS Vulkan" {
    try testBackend("vulkan");
}

test "GPT-OSS ROCm" {
    try testBackend("rocm");
}
