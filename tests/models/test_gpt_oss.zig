const std = @import("std");

const backends_to_test = [_][]const u8{ "cpu", "metal", "cuda", "vulkan", "rocm" };
const model_path = "models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf";
const test_prompt = "Once upon a time in a distant galaxy,";

fn testBackend(backend: []const u8) !void {
    const allocator = std.testing.allocator;

    // Skip if backend not available on this platform
    if (std.mem.eql(u8, backend, "metal") and @import("builtin").os.tag != .macos) {
        return error.SkipZigTest;
    }
    if (std.mem.eql(u8, backend, "rocm") and @import("builtin").cpu.arch != .x86_64) {
        return error.SkipZigTest; // ROCm only on x86_64 Linux
    }

    // Run Agave with JSON output
    var child = std.process.Child.init(&[_][]const u8{
        "./zig-out/bin/agave",
        model_path,
        "--backend",
        backend,
        "--prompt",
        test_prompt,
        "--json",
        "-n",
        "32", // Generate 32 tokens
        "-s",
        "42", // Deterministic seed
        "--temp",
        "0.0", // Greedy sampling
    }, allocator);

    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(stdout);

    const stderr = try child.stderr.?.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(stderr);

    const term = try child.wait();
    if (term != .Exited or term.Exited != 0) {
        std.debug.print("Agave failed:\n{s}\n", .{stderr});
        return error.TestFailed;
    }

    // Write output to temp file
    const tmp_file = try std.fmt.allocPrint(allocator, "/tmp/agave_gpt_oss_{s}.json", .{backend});
    defer allocator.free(tmp_file);

    var file = try std.fs.cwd().createFile(tmp_file, .{});
    defer file.close();
    try file.writeAll(stdout);

    // Verify against golden reference
    var verify_child = std.process.Child.init(&[_][]const u8{
        "python3",
        "tests/golden/verify_output.py",
        "gpt_oss",
        backend,
        tmp_file,
    }, allocator);

    const verify_term = try verify_child.spawnAndWait();
    if (verify_term != .Exited or verify_term.Exited != 0) {
        return error.GoldenTestFailed;
    }
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
