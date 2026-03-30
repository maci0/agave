//! Shared golden test helper for model integration tests.
//! Each model test file defines model-specific constants and delegates to
//! `runGoldenTest()` to avoid duplicating the spawn/capture/verify logic.

const std = @import("std");
const builtin = @import("builtin");

/// Maximum bytes to capture from child process stdout/stderr.
const max_child_output: usize = 1024 * 1024;

/// Number of tokens to generate in golden tests (enough to verify correctness
/// without making tests slow).
const golden_token_count = "32";

/// Fixed RNG seed for reproducible output across runs.
const golden_seed = "42";

/// Run a golden integration test: spawn Agave, capture JSON output, verify
/// against the golden reference via `tests/golden/verify_output.py`.
///
/// Parameters:
///   - model_path: Path to model file/directory (relative to repo root).
///   - test_prompt: Prompt string to pass to Agave.
///   - model_name: Short name for temp files and verify script (e.g. "gemma3").
///   - backend: Backend to test (e.g. "cpu", "metal", "cuda").
///   - skip_rocm: If true, skip ROCm unconditionally (e.g. SafeTensors-only models).
pub fn runGoldenTest(
    model_path: []const u8,
    test_prompt: []const u8,
    model_name: []const u8,
    backend: []const u8,
    skip_rocm: bool,
) !void {
    const allocator = std.testing.allocator;

    // Skip if model file not available
    std.fs.cwd().access(model_path, .{}) catch return error.SkipZigTest;

    // Skip if backend not available on this platform
    if (std.mem.eql(u8, backend, "metal") and builtin.os.tag != .macos) {
        return error.SkipZigTest;
    }
    if (std.mem.eql(u8, backend, "vulkan") and builtin.os.tag == .macos) {
        return error.SkipZigTest; // No Vulkan on macOS (MoltenVK not tested)
    }
    if (std.mem.eql(u8, backend, "cuda")) return error.SkipZigTest; // Requires NVIDIA GPU
    if (std.mem.eql(u8, backend, "rocm")) {
        if (skip_rocm or builtin.cpu.arch != .x86_64) {
            return error.SkipZigTest;
        }
    }

    // Run Agave with JSON output
    var child = std.process.Child.init(&[_][]const u8{
        "./zig-out/bin/agave",
        model_path,
        test_prompt,
        "--backend",
        backend,
        "--json",
        "-n",
        golden_token_count,
        "--seed",
        golden_seed,
        "-t",
        "0.0", // Greedy sampling
    }, allocator);

    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout = try child.stdout.?.readToEndAlloc(allocator, max_child_output);
    defer allocator.free(stdout);

    const stderr_out = try child.stderr.?.readToEndAlloc(allocator, max_child_output);
    defer allocator.free(stderr_out);

    const term = try child.wait();
    if (term != .Exited or term.Exited != 0) {
        std.debug.print("Agave failed:\n{s}\n", .{stderr_out});
        return error.TestFailed;
    }

    // Write output to temp file
    const tmp_file = try std.fmt.allocPrint(allocator, "/tmp/agave_{s}_{s}.json", .{ model_name, backend });
    defer allocator.free(tmp_file);

    var file = try std.fs.cwd().createFile(tmp_file, .{});
    defer file.close();
    defer std.fs.cwd().deleteFile(tmp_file) catch {};
    try file.writeAll(stdout);

    // Verify output is valid JSON with non-empty "output" field
    if (stdout.len == 0) {
        std.debug.print("FAIL: Agave produced empty stdout\n", .{});
        return error.TestFailed;
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, stdout, .{}) catch {
        std.debug.print("FAIL: Agave output is not valid JSON:\n{s}\n", .{stdout});
        return error.TestFailed;
    };
    defer parsed.deinit();

    const output_val = parsed.value.object.get("output") orelse {
        std.debug.print("FAIL: JSON missing 'output' field\n", .{});
        return error.TestFailed;
    };
    const output_str = switch (output_val) {
        .string => |s| s,
        else => {
            std.debug.print("FAIL: 'output' is not a string\n", .{});
            return error.TestFailed;
        },
    };
    if (output_str.len < 10) {
        std.debug.print("FAIL: output too short ({d} chars): \"{s}\"\n", .{ output_str.len, output_str });
        return error.TestFailed;
    }
    std.debug.print("PASS [{s}/{s}]: \"{s}\"\n", .{ model_name, backend, output_str[0..@min(output_str.len, 80)] });

    // If golden reference exists, verify against it
    var verify_child = std.process.Child.init(&[_][]const u8{
        "python3",
        "tests/golden/verify_output.py",
        model_name,
        backend,
        tmp_file,
    }, allocator);
    verify_child.stderr_behavior = .Pipe;
    try verify_child.spawn();
    const verify_stderr = try verify_child.stderr.?.readToEndAlloc(allocator, max_child_output);
    defer allocator.free(verify_stderr);
    const verify_term = try verify_child.wait();
    if (verify_term != .Exited or verify_term.Exited != 0) {
        // If reference file missing, that's OK — test still passes for basic sanity
        if (std.mem.indexOf(u8, verify_stderr, "Reference not found") != null) {
            std.debug.print("WARN: No golden reference for {s}/{s} — skipping comparison\n", .{ model_name, backend });
        } else {
            std.debug.print("Golden verify failed:\n{s}\n", .{verify_stderr});
            return error.GoldenTestFailed;
        }
    }
}
