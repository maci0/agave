//! Shared golden test helper for model integration tests.
//! Each model test file defines model-specific constants and delegates to
//! `runGoldenTest()` to avoid duplicating the spawn/capture/verify logic.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// Maximum bytes to capture from child process stdout/stderr.
const max_child_output: usize = 1024 * 1024;

/// Number of tokens to generate in golden tests (enough to verify correctness
/// without making tests slow).
const golden_token_count = "32";

/// Fixed RNG seed for reproducible output across runs.
const golden_seed = "42";

/// Built binary the golden tests spawn. Run `zig build` first.
const agave_bin = "zig-out/bin/agave";

/// Detect degenerate output: repeated characters, no alphabetic content,
/// or excessive word repetition. These indicate a broken model producing
/// garbage that would otherwise pass the length check.
fn isDegenerate(output: []const u8) bool {
    // Thresholds for degenerate output detection
    const min_letter_pct_inv = 5; // at least 1/5 (20%) of chars must be alphabetic
    const max_single_char_pct = 80; // single char can't exceed 80% of output
    const pattern_lengths = [_]usize{ 2, 3, 4 }; // short patterns to check
    const min_pattern_occurrences = 4; // need at least 4× pattern length of output
    const max_pattern_repeat_pct = 60; // pattern can't repeat more than 60%
    const min_output_for_word_check = 20; // need at least 20 chars for word analysis
    const min_words_for_check = 3; // need at least 3 words to check dominance
    const max_word_dominance_inv = 2; // single word can't exceed 1/2 (50%) of total

    if (output.len == 0) return true;

    // Check 1: at least 20% of characters should be ASCII letters
    var letter_count: usize = 0;
    for (output) |c| {
        if (std.ascii.isAlphabetic(c)) letter_count += 1;
    }
    if (letter_count * min_letter_pct_inv < output.len) return true;

    // Check 2: any single character repeated >80% of output (e.g., "aaaaaaaaaa")
    var char_counts = [_]usize{0} ** 256;
    for (output) |c| char_counts[c] += 1;
    var max_char_count: usize = 0;
    for (char_counts) |cnt| max_char_count = @max(max_char_count, cnt);
    if (max_char_count * 100 / output.len >= max_single_char_pct) return true;

    // Check 3: excessive short-pattern repetition (e.g., "abcabcabc")
    // Scan start offsets across the output, leaving enough tail for the
    // ratio check to be statistically meaningful (at least min_pattern_occurrences windows).
    for (pattern_lengths) |pat_len| {
        if (output.len < pat_len * min_pattern_occurrences) continue;
        const max_start = output.len -| (pat_len * min_pattern_occurrences);
        for (0..max_start) |start| {
            const pattern = output[start..][0..pat_len];
            var repeats: usize = 0;
            var i: usize = start;
            while (i + pat_len <= output.len) : (i += pat_len) {
                if (std.mem.eql(u8, output[i..][0..pat_len], pattern)) {
                    repeats += 1;
                }
            }
            const total_windows = (output.len - start) / pat_len;
            if (total_windows > 0 and repeats * 100 / total_windows > max_pattern_repeat_pct) return true;
        }
    }

    // Check 4: word-level repetition (e.g., "The The The The The The")
    if (output.len >= min_output_for_word_check) {
        var word_counts: [256]struct { word: []const u8, count: usize } = undefined;
        var n_unique: usize = 0;
        var total_words: usize = 0;
        var iter = std.mem.splitScalar(u8, output, ' ');
        while (iter.next()) |word| {
            if (word.len == 0) continue;
            total_words += 1;
            var found = false;
            for (word_counts[0..n_unique]) |*entry| {
                if (std.mem.eql(u8, entry.word, word)) {
                    entry.count += 1;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // If we've exhausted the tracking array, stop adding new words.
                // Having 256+ unique words implies high diversity, so the
                // dominance check on tracked words is still meaningful.
                if (n_unique < word_counts.len) {
                    word_counts[n_unique] = .{ .word = word, .count = 1 };
                    n_unique += 1;
                }
            }
        }
        if (total_words >= min_words_for_check) {
            var max_count: usize = 0;
            for (word_counts[0..n_unique]) |entry| {
                max_count = @max(max_count, entry.count);
            }
            if (max_count * max_word_dominance_inv > total_words) return true;
        }
    }

    return false;
}

test "isDegenerate detects garbage output" {
    // Single repeated character
    try std.testing.expect(isDegenerate("aaaaaaaaaaaaaaaa"));
    // No alphabetic characters
    try std.testing.expect(isDegenerate("!@#$%^&*()12345"));
    // Short repeated pattern
    try std.testing.expect(isDegenerate("abcabcabcabcabcabcabcabc"));
    // Empty
    try std.testing.expect(isDegenerate(""));
    // Word-level repetition
    try std.testing.expect(isDegenerate("The The The The The The The The"));
    // Single character dominates but is NOT the first character
    try std.testing.expect(isDegenerate("Hello aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    // Mid-output degeneration (coherent start, repeated tail)
    try std.testing.expect(isDegenerate("Hello there xyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyz"));
    // Normal text should pass
    try std.testing.expect(!isDegenerate("The capital of France is Paris and it is beautiful"));
    // Short but valid
    try std.testing.expect(!isDegenerate("Hello world this is a test"));
    // All digits, no letters, degenerate
    try std.testing.expect(isDegenerate("12345678901234567890"));
    // Mostly non-alphabetic but just enough letters to pass letter check,
    // yet single char still dominates
    try std.testing.expect(isDegenerate("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"));
    // Two-word output where one word is >50%, degenerate
    try std.testing.expect(isDegenerate("ok ok ok ok ok ok ok no"));
    // Short 5-word output with 100% word dominance, should still be caught
    // (exercises lowered min_words_for_check threshold)
    try std.testing.expect(isDegenerate("test test test test test"));
    // Exactly 3 words, one dominates (2/3 > 50%)
    try std.testing.expect(isDegenerate("hello hello world hello hello hello"));
    // Late-starting degeneration: coherent first half, repeating pattern in second half.
    // Must be caught even when pattern starts past the midpoint.
    try std.testing.expect(isDegenerate("The capital of France is Paris and it is a beautiful city abcabcabcabcabcabcabcabcabcabcabcabcabc"));
}

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
    const io = std.testing.io;

    // Skip if model file not available
    _ = Io.Dir.cwd().statFile(io, model_path, .{}) catch {
        std.debug.print("SKIP: Model not found: {s}\n", .{model_path});
        return error.SkipZigTest;
    };

    // Skip if backend not available on this platform
    if (std.mem.eql(u8, backend, "metal") and builtin.os.tag != .macos) {
        return error.SkipZigTest;
    }
    if (std.mem.eql(u8, backend, "vulkan") and builtin.os.tag == .macos) {
        return error.SkipZigTest; // No Vulkan on macOS (MoltenVK not tested)
    }
    if (std.mem.eql(u8, backend, "cuda") and builtin.os.tag != .linux) return error.SkipZigTest; // CUDA backend only tested on Linux
    if (std.mem.eql(u8, backend, "rocm")) {
        if (skip_rocm or builtin.cpu.arch != .x86_64) {
            return error.SkipZigTest;
        }
    }

    _ = Io.Dir.cwd().statFile(io, agave_bin, .{}) catch {
        std.debug.print("FAIL: {s} not found. Run zig build first.\n", .{agave_bin});
        return error.TestFailed;
    };

    const result = std.process.run(allocator, io, .{
        .argv = &.{
            agave_bin,
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
            "0.0",
        },
        .stdout_limit = .limited(max_child_output),
        .stderr_limit = .limited(max_child_output),
    }) catch |err| {
        std.debug.print("FAIL: could not run {s}: {s}\n", .{ agave_bin, @errorName(err) });
        return error.TestFailed;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("Agave failed:\n{s}\n", .{result.stderr});
            return error.TestFailed;
        },
        else => {
            std.debug.print("Agave failed:\n{s}\n", .{result.stderr});
            return error.TestFailed;
        },
    }

    const stdout = result.stdout;

    // Write output under tests/golden_outputs (gitignored) so verify_output.py can read it.
    Io.Dir.cwd().createDir(io, "tests/golden_outputs", .default_dir) catch {};
    const tmp_file = try std.fmt.allocPrint(allocator, "tests/golden_outputs/{s}_{s}.json", .{ model_name, backend });
    defer allocator.free(tmp_file);

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp_file, .data = stdout });
    defer Io.Dir.cwd().deleteFile(io, tmp_file) catch {};

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
    // 32 tokens should produce at least 64 chars (~2 chars/token is very conservative).
    if (output_str.len < 64) {
        std.debug.print("FAIL: output too short ({d} chars): \"{s}\"\n", .{ output_str.len, output_str });
        return error.TestFailed;
    }

    // Coherence check: detect degenerate output that passes length check
    // but indicates a broken model (repeated characters, no letters, etc.)
    if (isDegenerate(output_str)) {
        std.debug.print("FAIL: output is degenerate (repeated chars or no letters): \"{s}\"\n", .{output_str[0..@min(output_str.len, 80)]});
        return error.TestFailed;
    }

    std.debug.print("COHERENT [{s}/{s}]: \"{s}\"\n", .{ model_name, backend, output_str[0..@min(output_str.len, 80)] });

    const verify = std.process.run(allocator, io, .{
        .argv = &.{
            "python3",
            "tests/golden/verify_output.py",
            model_name,
            backend,
            tmp_file,
        },
        .stdout_limit = .limited(max_child_output),
        .stderr_limit = .limited(max_child_output),
    }) catch |err| {
        std.debug.print("FAIL: python3 not found; golden verify needs python3 for tests/golden/verify_output.py ({s})\n", .{@errorName(err)});
        return error.TestFailed;
    };
    defer allocator.free(verify.stdout);
    defer allocator.free(verify.stderr);

    const verify_failed = switch (verify.term) {
        .exited => |code| code != 0,
        else => true,
    };
    if (verify_failed) {
        if (std.mem.indexOf(u8, verify.stderr, "Reference not found") != null) {
            // No golden reference: skip so CI shows this test as skipped, not passed.
            // A passing-without-comparison test gives false confidence.
            std.debug.print("SKIP: No golden reference for {s}/{s}\n", .{ model_name, backend });
            return error.SkipZigTest;
        } else {
            std.debug.print("Golden verify failed:\n{s}\n", .{verify.stderr});
            return error.GoldenTestFailed;
        }
    }
}
