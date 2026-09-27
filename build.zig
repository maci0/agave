//! Build configuration for Agave, LLM inference engine.
//! Targets: ReleaseFast (agave), ReleaseSafe (agave-debug), WASM (agave.wasm),
//! CUDA PTX kernels (zig build ptx), ROCm AMDGCN kernels (zig build amdgcn),
//! micro-benchmarks (zig build bench).

const std = @import("std");
const builtin = @import("builtin");
/// Product SemVer from build.zig.zon; injected into binaries via build_options.version.
const package_version: []const u8 = @import("build.zig.zon").version;

pub fn build(b: *std.Build) void {
    // Exact pin (.zigversion), not only build.zig.zon minimum_zig_version.
    // A newer compiler passes the minimum and then fails later with a parse error.
    const pinned_zig = std.mem.trim(u8, @embedFile(".zigversion"), " \t\r\n");
    const pinned_semver = std.SemanticVersion.parse(pinned_zig) catch std.debug.panic(
        ".zigversion ({s}) is not a semantic version",
        .{pinned_zig},
    );
    if (builtin.zig_version.order(pinned_semver) != .eq) {
        std.debug.panic(
            "Zig {s} does not match .zigversion pin {s}. Install {s} from https://ziglang.org/download/",
            .{ builtin.zig_version_string, pinned_zig, pinned_zig },
        );
    }

    const target = b.standardTargetOptions(.{});

    // ── Backend enable/disable flags (all default to true) ────────
    const enable_cpu = b.option(bool, "enable-cpu", "Enable CPU backend (default: true)") orelse true;
    const enable_metal = b.option(bool, "enable-metal", "Enable Metal backend (default: true)") orelse true;
    const enable_cuda = b.option(bool, "enable-cuda", "Enable CUDA backend (default: true)") orelse true;
    const enable_rocm = b.option(bool, "enable-rocm", "Enable ROCm backend (default: true)") orelse true;
    const enable_debug_binary = b.option(bool, "enable-debug", "Build agave-debug binary (default: true)") orelse true;
    const enable_bench = b.option(bool, "enable-bench", "Install agave-bench binary (default: true)") orelse true;

    const enable_vulkan = b.option(bool, "enable-vulkan", "Enable Vulkan backend (default: true)") orelse true;
    const enable_webgpu = b.option(bool, "enable-webgpu", "Enable WebGPU backend via wgpu-native (default: true)") orelse true;

    // ── Model enable/disable flags (all default to true) ─────────
    const enable_gemma3 = b.option(bool, "enable-gemma3", "Enable Gemma3 model support (default: true)") orelse true;
    const enable_qwen35 = b.option(bool, "enable-qwen35", "Enable Qwen3.5 model support (default: true)") orelse true;
    const enable_qwen4exp = b.option(bool, "enable-qwen4exp", "Enable Qwen3.8-Flash-Next GGUF (qwen4exp) model support (default: true)") orelse true;
    const enable_qwen4_exp = b.option(bool, "enable-qwen4-exp", "Enable Qwen4-Exp SafeTensors (qwen4_exp) model support (default: true)") orelse true;
    const enable_gpt_oss = b.option(bool, "enable-gpt-oss", "Enable GPT-OSS model support (default: true)") orelse true;
    const enable_nemotron_h = b.option(bool, "enable-nemotron-h", "Enable Nemotron-H model support (default: true)") orelse true;
    const enable_nemotron_nano = b.option(bool, "enable-nemotron-nano", "Enable Nemotron-Nano model support (default: true)") orelse true;
    const enable_glm4 = b.option(bool, "enable-glm4", "Enable GLM-4 model support (default: true)") orelse true;
    const enable_gemma4 = b.option(bool, "enable-gemma4", "Enable Gemma4 model support (default: true)") orelse true;
    const enable_diffusion_gemma = b.option(bool, "enable-diffusion-gemma", "Enable DiffusionGemma model support (default: true)") orelse true;
    const enable_deepseek4 = b.option(bool, "enable-deepseek4", "Enable DeepSeek V4 model support (default: true)") orelse true;
    const enable_llama4 = b.option(bool, "enable-llama4", "Enable Llama 4 model support (default: true)") orelse true;
    const enable_dflash2 = b.option(bool, "enable-dflash2", "Enable DFlash2 block-diffusion drafter support (default: true)") orelse true;

    const link_metal = enable_metal and target.result.os.tag == .macos;

    const pin_spawned_python = struct {
        fn apply(cmd: *std.Build.Step.Run) void {
            cmd.setEnvironmentVariable("LC_ALL", "C");
            cmd.setEnvironmentVariable("TZ", "UTC");
            cmd.setEnvironmentVariable("PYTHONHASHSEED", "0");
            cmd.setEnvironmentVariable("PYTHONIOENCODING", "utf-8");
        }
    }.apply;

    // A bare `python3` in addSystemCommand surfaces as an exec failure with no
    // name in the message. Resolve it up front so a missing interpreter names
    // itself and the step that wanted it.
    const python3 = b.findProgram(&.{"python3"}, &.{}) catch null;

    // ── CUDA PTX kernels (cross-compiled via nvptx64-cuda) ─────────
    // Compiles Zig CUDA kernels to PTX assembly. The resulting .s file
    // is placed in zig-out/ and can be embedded into cuda.zig via @embedFile.
    // Build with: zig build ptx [-Dcuda-sm=sm_120]
    // Default matches committed PTX and CI (`scripts/check-shader-artifacts.sh --ptx-only`).
    const CudaSm = enum { sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120, sm_121 };
    const cuda_sm = b.option(CudaSm, "cuda-sm", "CUDA SM target (default: sm_120)") orelse .sm_120;
    const sm_model: *const std.Target.Cpu.Model = switch (cuda_sm) {
        .sm_50 => &std.Target.nvptx.cpu.sm_50,
        .sm_60 => &std.Target.nvptx.cpu.sm_60,
        .sm_70 => &std.Target.nvptx.cpu.sm_70,
        .sm_75 => &std.Target.nvptx.cpu.sm_75,
        .sm_80 => &std.Target.nvptx.cpu.sm_80,
        .sm_86 => &std.Target.nvptx.cpu.sm_86,
        .sm_89 => &std.Target.nvptx.cpu.sm_89,
        .sm_90 => &std.Target.nvptx.cpu.sm_90,
        .sm_100 => &std.Target.nvptx.cpu.sm_100,
        .sm_120 => &std.Target.nvptx.cpu.sm_120,
        // GB10 (NVIDIA DGX Spark), Blackwell Ultra. PTX for sm_121 is
        // forward-compatible with the sm_120 model; JITs at load time.
        .sm_121 => &std.Target.nvptx.cpu.sm_121,
    };

    // ── ROCm AMDGCN kernels ─────────────────────────────────────────
    const RocmArch = enum { gfx90a, gfx942, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151 };
    const rocm_arch = b.option(RocmArch, "rocm-arch", "ROCm GFX target (default: gfx1100)") orelse .gfx1100;
    const gfx_model: *const std.Target.Cpu.Model = switch (rocm_arch) {
        .gfx90a => &std.Target.amdgcn.cpu.gfx90a,
        .gfx942 => &std.Target.amdgcn.cpu.gfx942,
        .gfx1100 => &std.Target.amdgcn.cpu.gfx1100,
        .gfx1101 => &std.Target.amdgcn.cpu.gfx1101,
        .gfx1102 => &std.Target.amdgcn.cpu.gfx1102,
        .gfx1150 => &std.Target.amdgcn.cpu.gfx1150,
        .gfx1151 => &std.Target.amdgcn.cpu.gfx1151,
    };

    const ptx_step = b.step("ptx", "Compile CUDA kernels to PTX (nvptx64)");
    if (python3 == null) {
        ptx_step.dependOn(&b.addFail("python3 not found; zig build ptx runs src/backend/kernels/cuda/fix_kernel_alias.py").step);
    }
    if (python3) |py| {
        const kernel_files = [_][]const u8{
            // Core ops
            "all",             "silu",           "gelu",          "add",            "mul",
            "rms_norm",        "softmax",        "l2_norm",       "rope",           "add_scaled",
            "silu_mul",        "gelu_mul",       "add_rms_norm",  "rms_norm_add",   "rms_norm_batched",
            "rope_batched",    "sigmoid_mul",    "deinterleave",  "split_qgate",    "deltanet_recurrence",
            // SDPA
            "sdpa",            "sdpa_turbo",     "sdpa_prefill",  "sdpa_tree",
            // Dense GEMV
                 "gemv_f32",
            "gemv_bf16",       "gemv_f16",       "gemv_t_q8_0",
            // Quantized GEMV, standard GGUF formats
              "gemv_q8_0",      "gemv_q4_0",
            "gemv_q4_0_batch", "gemv_q4_1",      "gemv_q5_0",     "gemv_q4_k",      "gemv_q5_k",
            "gemv_q6_k",       "gemv_q2_k",      "gemv_q3_k",     "gemv_iq4_nl",    "gemv_iq4_xs",
            // FP8/FP4
            "gemv_fp8_e4m3",   "gemv_fp8_e5m2",  "gemv_nvfp4_st", "gemv_mxfp4_st",  "gemv_mxfp4_st_batched",
            "gemv_fp4_tc",
            // MLX / TQ
                "gemv_mlx_q4",    "gemv_mlx_q6",   "gemv_mlx_q8",    "gemv_tq1_0",
            "gemv_tq2_0",
            // Specialist formats
                 "gemv_gptq",      "gemv_awq",      "gemv_hqq",
            // GEMM
                  "gemm_q8_0",
            // Megakernels
            "mega_qwen35_q8",  "mega_gemma_q4k", "mega_gemma_q8",
            // Fused FFN
            "fused_ffn_q8_0", "fused_ffn_q4_k",
            "fused_ffn_q5_k",  "fused_ffn_q6_k",
        };

        for (kernel_files) |name| {
            const path = b.fmt("src/backend/kernels/cuda/{s}.zig", .{name});
            const ptx = b.addObject(.{
                .name = b.fmt("cuda_{s}", .{name}),
                .root_module = b.createModule(.{
                    .root_source_file = b.path(path),
                    .target = b.resolveTargetQuery(.{
                        .cpu_arch = .nvptx64,
                        .os_tag = .cuda,
                        .cpu_model = .{ .explicit = sm_model },
                    }),
                    .optimize = .ReleaseFast,
                }),
            });
            ptx.root_module.strip = true;

            // Post-process PTX: work around Zig 0.16 + LLVM aliasee bug.
            // callconv(.kernel) causes LLVM NVPTX to reject aliases to kernel functions.
            // Kernels use callconv(.nvptx_device) which generates .func (device function).
            // Post-processing: find .alias directives, promote .func → .entry, remove aliases.
            // Pass the fixup script via addFileArg so the build graph tracks it as an
            // input (rebuilds when the script changes) and avoids configure-time getPath
            // absolute host paths (same pattern as the ROCm fixup below).
            const fixup = b.addSystemCommand(&.{py});
            pin_spawned_python(fixup);
            fixup.addFileArg(b.path("src/backend/kernels/cuda/fix_kernel_alias.py"));
            fixup.addFileArg(ptx.getEmittedAsm());
            const fixed_ptx = fixup.captureStdOut(.{});
            const install = b.addInstallFile(fixed_ptx, b.fmt("ptx/{s}.ptx", .{name}));
            ptx_step.dependOn(&install.step);
        }
    }

    // ── ROCm AMDGCN kernels (cross-compiled via amdgcn-amdhsa) ───────
    // Compiles Zig ROCm kernels to AMDGCN ISA, producing an ELF object.
    // Build with: zig build amdgcn [-Drocm-arch=gfx1100]
    // After building, copy zig-out/rocm/kernels.o to
    // src/backend/kernels/rocm/kernels.hsaco and commit.
    const amdgcn_step = b.step("amdgcn", "Compile ROCm kernels to AMDGCN ISA");
    // The HSACO link needs lld (ROCm ships it at /opt/rocm/lib/llvm/bin/ld.lld).
    const ld_lld = b.findProgram(&.{"ld.lld"}, &.{}) catch null;
    if (python3 == null) {
        amdgcn_step.dependOn(&b.addFail("python3 not found; zig build amdgcn runs src/backend/kernels/rocm/fix_kd_isa.py").step);
    }
    if (ld_lld == null) {
        amdgcn_step.dependOn(&b.addFail("ld.lld not found; add ROCm's llvm bin dir (/opt/rocm/lib/llvm/bin) to PATH").step);
    }
    if (python3 != null and ld_lld != null) {
        const obj = b.addObject(.{
            .name = "rocm_kernels",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/backend/kernels/rocm/all.zig"),
                .target = b.resolveTargetQuery(.{
                    .cpu_arch = .amdgcn,
                    .os_tag = .amdhsa,
                    .cpu_model = .{ .explicit = gfx_model },
                }),
                .optimize = .ReleaseFast,
            }),
        });
        obj.root_module.strip = true;

        // Install the relocatable .o (for debugging / manual linking)
        const install_obj = b.addInstallFile(obj.getEmittedBin(), "rocm/kernels.o");
        amdgcn_step.dependOn(&install_obj.step);

        // Workaround: two Zig 0.16 bugs for AMDGCN targets (upstream Zig AMDGCN issues):
        //
        // Bug 1, Wrong ISA string in metadata:
        //   Zig emits amdhsa.target = "amdgcn-amd-amdhsa5.0.0-unknown-gfx1100"
        //   (OS semver from Target.zig appended to triple).
        //   HIP does exact-string matching; expects "amdgcn-amd-amdhsa--gfx1100".
        //   Workaround: patch the NT_AMDGPU_METADATA note section in the .o
        //   before linking (ET_REL has no VirtAddr constraints, safe to shrink).
        //
        // Bug 2, .kd symbols emitted as LOCAL:
        //   Kernel descriptor symbols (foo.kd) need GLOBAL binding so the linker
        //   exports them to .dynsym. Zig emits them as LOCAL → HIP can't find them.
        //   Workaround: llvm-objcopy --globalize-symbol for every .kd symbol.
        //
        // Run this build step on a Linux machine with ROCm installed and llvm-objcopy
        // (from /opt/rocm/lib/llvm/bin) in PATH. The resulting HSACO is committed.
        // Pass the fixup script via addFileArg so the build graph tracks it as an
        // input (rebuilds when the script changes) and avoids configure-time getPath
        // absolute host paths.
        const fix_obj = b.addSystemCommand(&.{python3.?});
        pin_spawned_python(fix_obj);
        fix_obj.addFileArg(b.path("src/backend/kernels/rocm/fix_kd_isa.py"));
        fix_obj.addFileArg(obj.getEmittedBin());
        const fixed_obj = fix_obj.addOutputFileArg("kernels_fixed.o");
        fix_obj.step.dependOn(&obj.step);

        const link = b.addSystemCommand(&.{ ld_lld.?, "-shared", "-o" });
        const hsaco_out = link.addOutputFileArg("kernels.hsaco");
        link.addFileArg(fixed_obj);
        link.step.dependOn(&fix_obj.step);

        const install_hsaco = b.addInstallFile(hsaco_out, "rocm/kernels.hsaco");
        install_hsaco.step.dependOn(&link.step);
        amdgcn_step.dependOn(&install_hsaco.step);
    }

    // ── ReleaseFast executable (default) ──────────────────────────
    const backend_options = b.addOptions();
    backend_options.addOption([]const u8, "version", package_version);
    backend_options.addOption(bool, "enable_cpu", enable_cpu);
    backend_options.addOption(bool, "enable_metal", enable_metal);
    backend_options.addOption(bool, "enable_vulkan", enable_vulkan);
    backend_options.addOption(bool, "enable_cuda", enable_cuda);
    backend_options.addOption(bool, "enable_rocm", enable_rocm);
    backend_options.addOption(bool, "enable_webgpu", enable_webgpu);
    backend_options.addOption(bool, "enable_gemma3", enable_gemma3);
    backend_options.addOption(bool, "enable_qwen35", enable_qwen35);
    backend_options.addOption(bool, "enable_qwen4exp", enable_qwen4exp);
    backend_options.addOption(bool, "enable_qwen4_exp", enable_qwen4_exp);
    backend_options.addOption(bool, "enable_gpt_oss", enable_gpt_oss);
    backend_options.addOption(bool, "enable_nemotron_h", enable_nemotron_h);
    backend_options.addOption(bool, "enable_nemotron_nano", enable_nemotron_nano);
    backend_options.addOption(bool, "enable_glm4", enable_glm4);
    backend_options.addOption(bool, "enable_gemma4", enable_gemma4);
    backend_options.addOption(bool, "enable_diffusion_gemma", enable_diffusion_gemma);
    backend_options.addOption(bool, "enable_deepseek4", enable_deepseek4);
    backend_options.addOption(bool, "enable_llama4", enable_llama4);
    backend_options.addOption(bool, "enable_dflash2", enable_dflash2);

    // Strip ReleaseFast: unstripped ELF/Mach-O embeds host absolute paths
    // (project root, zig lib, global cache) and breaks path-independent rebuilds.
    const mod_rel = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .strip = true,
    });
    mod_rel.addImport("build_options", backend_options.createModule());

    const exe_rel = b.addExecutable(.{ .name = "agave", .root_module = mod_rel });
    linkPlatform(mod_rel, exe_rel, target, link_metal);
    b.installArtifact(exe_rel);

    // ── Debug executable (also built by default) ─────────────────
    // ReleaseSafe keeps every safety check a Debug build has while linking
    // against modern system crt1.o (GCC 16 emits .sframe sections that
    // break Debug-mode linking).
    const mod_dbg = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    mod_dbg.addImport("build_options", backend_options.createModule());

    const exe_dbg = b.addExecutable(.{ .name = "agave-debug", .root_module = mod_dbg });
    linkPlatform(mod_dbg, exe_dbg, target, link_metal);
    if (enable_debug_binary) b.installArtifact(exe_dbg);

    // ── Run step (uses the optimized binary) ─────────────────────
    const run_cmd = b.addRunArtifact(exe_rel);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run agave (ReleaseFast)").dependOn(&run_cmd.step);

    // ── Test step ────────────────────────────────────────────────
    const test_step = b.step("test", "Run unit tests");

    // Substring filter applied to every test artifact:
    //   zig build test -Dtest-filter=wht32
    // Repeat the flag to AND multiple filters.
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Only run tests whose name contains this substring (repeatable)",
    ) orelse &.{};
    if (test_filters.len != 0) _ = rejectEmptyTestFilters(b, test_filters);

    // Test modules use ReleaseSafe so std.debug.assert / unreachable fire.
    // Reusing mod_rel (ReleaseFast) silently no-ops ~400 assert-based checks
    // in fuzz and unit tests (see std.debug.assert docs).
    const test_optimize: std.builtin.OptimizeMode = .ReleaseSafe;

    // Default `--listen=-` server deadlocks under parallel `addRunArtifact` on
    // this host: children block in receiveMessage and the parent never sends
    // query_test_metadata after another artifact writes stderr. Simple mode
    // uses the same runner without the server protocol; failure is exit status.
    // Exception: the main suite runs in server mode so its fuzz tests register
    // with the build runner, without the protocol, `zig build test --fuzz`
    // aborts with "no fuzz tests found" and CI's fuzz-smoke never fuzzes.
    const zig_lib = b.graph.zig_lib_directory.path orelse ".";
    const simple_test_runner: std.Build.Step.Compile.TestRunner = .{
        .path = .{ .cwd_relative = b.pathJoin(&.{ zig_lib, "compiler", "test_runner.zig" }) },
        .mode = .simple,
    };
    const fuzz_test_runner: std.Build.Step.Compile.TestRunner = .{
        .path = simple_test_runner.path,
        .mode = .server,
    };

    // Main test suite (inline tests from src/)
    {
        const mod_test = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod_test.addImport("build_options", backend_options.createModule());
        // No name filters: run the full inline suite from src/ (ReleaseSafe so asserts fire).
        const t = b.addTest(.{ .root_module = mod_test, .test_runner = fuzz_test_runner, .filters = test_filters });
        linkPlatform(mod_test, t, target, link_metal);
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // SDPA oracle self-tests (validates ground-truth reference for GPU tests)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/sdpa_oracle.zig"),
            .target = target,
            .optimize = test_optimize,
        }),
        .test_runner = simple_test_runner,
        .filters = test_filters,
    })).step);

    // Golden harness unit tests (degenerate output detection)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/models/golden_harness.zig"),
            .target = target,
            .optimize = test_optimize,
        }),
        .test_runner = simple_test_runner,
        .filters = test_filters,
    })).step);

    // Shared backend module for SDPA hardware tests (provides named "backend" import).
    // Rooted at src/test_exports.zig so transitive imports resolve within src/.
    const backend_test_mod = b.createModule(.{
        .root_source_file = b.path("src/test_exports.zig"),
        .target = target,
        .optimize = test_optimize,
    });
    backend_test_mod.addImport("build_options", backend_options.createModule());

    // Shared oracle module for SDPA hardware tests
    const oracle_mod = b.createModule(.{
        .root_source_file = b.path("tests/sdpa_oracle.zig"),
        .target = target,
        .optimize = test_optimize,
    });

    // Shared dual-delta test harness for GPU SDPA correctness tests
    const sdpa_harness_mod = b.createModule(.{
        .root_source_file = b.path("tests/sdpa_harness.zig"),
        .target = target,
        .optimize = test_optimize,
    });
    sdpa_harness_mod.addImport("backend", backend_test_mod);
    sdpa_harness_mod.addImport("sdpa_oracle", oracle_mod);

    const backend_tests: BackendTest = .{
        .b = b,
        .test_step = test_step,
        .backend_mod = backend_test_mod,
        .target = target,
        .optimize = test_optimize,
        .filters = test_filters,
        .test_runner = simple_test_runner,
        .link_metal = link_metal,
    };

    // CUDA SDPA correctness tests (skips at runtime if no CUDA hardware).
    // Skip compile when CUDA is NullBackend: init() is a compileError.
    if (enable_cuda) {
        _ = backend_tests.add("tests/test_cuda_sdpa.zig", "sdpa_harness", sdpa_harness_mod);
    }

    // Metal SDPA correctness tests (skips at runtime if not macOS).
    // Skip compile when Metal is NullBackend: init() arity does not match.
    if (enable_metal) {
        _ = backend_tests.add("tests/test_metal_sdpa.zig", "sdpa_harness", sdpa_harness_mod);
    }

    // WebGPU MLX GEMV row-chunking (vocab > 65535). Skips if wgpu-native missing.
    // Skip compile when WebGPU is NullBackend: init(allocator) vs init(allocator, device).
    if (enable_webgpu) {
        const run = backend_tests.add("tests/test_webgpu_mlx_gemv.zig", null, null);
        b.step("test-webgpu-mlx", "WebGPU MLX-Q4 GEMV chunking test").dependOn(&run.step);
    }

    // Cross-backend op parity against CPU (skips at runtime with no device).
    // Compiled only when both GPU backends are real: NullBackend.init is a
    // compileError, and this test instantiates each backend by union tag.
    if (enable_vulkan and enable_rocm) {
        _ = backend_tests.add("tests/test_backend_parity.zig", null, null);
    }

    // micro_bench pure-function tests (parseKeyValue, parseKernelName, etc.)
    {
        const mod_bench_test = b.createModule(.{
            .root_source_file = b.path("src/micro_bench.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod_bench_test.addImport("build_options", backend_options.createModule());
        const t = b.addTest(.{ .root_module = mod_bench_test, .test_runner = simple_test_runner, .filters = test_filters });
        linkPlatform(mod_bench_test, t, target, link_metal);
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // wasm_entry pure-function tests (agave_alloc, agave_dealloc, wasmLogFn)
    {
        const mod_wasm_test = b.createModule(.{
            .root_source_file = b.path("src/wasm_entry.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod_wasm_test.addImport("build_options", backend_options.createModule());
        const t = b.addTest(.{ .root_module = mod_wasm_test, .test_runner = simple_test_runner, .filters = test_filters });
        linkPlatform(mod_wasm_test, t, target, link_metal);
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // ── Benchmark binary (standalone micro-benchmark) ──────────────
    const mod_bench = b.createModule(.{
        .root_source_file = b.path("src/micro_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .strip = true,
    });
    mod_bench.addImport("build_options", backend_options.createModule());

    const exe_bench = b.addExecutable(.{ .name = "agave-bench", .root_module = mod_bench });
    linkPlatform(mod_bench, exe_bench, target, link_metal);
    if (enable_bench) b.installArtifact(exe_bench);

    const bench_run = b.addRunArtifact(exe_bench);
    bench_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench_run.addArgs(args);
    b.step("bench", "Run micro-benchmarks (ReleaseFast)").dependOn(&bench_run.step);

    // ── GPU kernel correctness sweep ────────────────────────────
    // Every kernel the harness can build, run on the chosen backend and again on
    // the CPU with byte-identical inputs; agave-bench exits non-zero past its
    // relative tolerance, so this fails the build on a wrong kernel.
    //
    // Not part of `zig build test`: it needs a working GPU, which a CI runner or
    // a cross-compile host does not have. Run it where the hardware is:
    //     zig build validate -Dvalidate-backend=rocm
    //
    // Two values of k, one a multiple of the 32- and 256-element block sizes and
    // one a multiple of neither, because a truncated block count is only visible
    // at the second (it is what made Q4_K produce garbage at k=896).
    const validate_backend = b.option([]const u8, "validate-backend", "Backend for `zig build validate` (default: rocm)") orelse "rocm";
    const validate_step = b.step("validate", "Validate every GPU kernel against the CPU backend");
    {
        const kernels = [_][]const u8{
            "gemv_f32",         "gemv_f16",      "gemv_bf16",      "gemv_q8_0",
            "gemv_q4_0",        "gemv_q4_1",     "gemv_q5_0",      "gemv_q2_k",
            "gemv_q3_k",        "gemv_q4_k",     "gemv_q5_k",      "gemv_q6_k",
            "gemv_iq4_nl",      "gemv_iq4_xs",   "gemv_tq1_0",     "gemv_tq2_0",
            "gemv_fp8_e4m3",    "gemv_fp8_e5m2", "gemm_q8_0",      "rms_norm_batched",
            "rope_batched",     "sdpa_prefill",  "rms_norm_multi", "rms_norm",
            "silu",             "gelu",          "softmax",        "l2_norm",
            "add",              "mul",           "rope",           "add_aliased",
            "silu_mul_aliased", "deinterleave",  "split_q_gate",   "add_rms_norm",
            "rms_norm_add",     "sigmoid_mul",   "gelu_mul",       "clamped_silu_mul",
            "add_scaled",       "gemv_multi",    "gemv_t",         "emb_lookup",
        };
        // Deliberately absent:
        //   rope_mrope     - CPU and Metal only; it panics on ROCm and Vulkan by
        //                    design, which is the documented contract for a
        //                    missing GPU kernel rather than something to gate on.
        //   all_reduce_add - no backend but CPU implements it and the dispatcher
        //                    skips it silently, so a comparison would only measure
        //                    that skip. Nothing calls Backend.allReduceAdd today;
        //                    the models reduce through transport.allReduceAdd.
        const k_values = [_][]const u8{ "896", "900" };
        for (k_values) |kv| {
            for (kernels) |kern| {
                const run = b.addRunArtifact(exe_bench);
                run.addArgs(&.{ kern, "--backend", validate_backend, "--n", "1024", "--k", kv, "--iters", "3", "--validate" });
                run.expectExitCode(0);
                validate_step.dependOn(&run.step);
            }
        }
    }

    // ── WASM build (browser inference) ──────────────────────────
    const wasm_step = b.step("wasm", "Build WebAssembly module for browser inference");
    const wasm_options = b.addOptions();
    wasm_options.addOption([]const u8, "version", package_version);
    wasm_options.addOption(bool, "enable_cpu", true);
    wasm_options.addOption(bool, "enable_metal", false);
    wasm_options.addOption(bool, "enable_vulkan", false);
    wasm_options.addOption(bool, "enable_cuda", false);
    wasm_options.addOption(bool, "enable_rocm", false);
    wasm_options.addOption(bool, "enable_webgpu", false);
    wasm_options.addOption(bool, "enable_gemma3", enable_gemma3);
    wasm_options.addOption(bool, "enable_qwen35", false); // disabled: Zig+LLVM wasm32 codegen bug in DeltaNet SSM
    wasm_options.addOption(bool, "enable_qwen4exp", false);
    wasm_options.addOption(bool, "enable_qwen4_exp", false);
    wasm_options.addOption(bool, "enable_gpt_oss", false);
    wasm_options.addOption(bool, "enable_nemotron_h", false);
    wasm_options.addOption(bool, "enable_nemotron_nano", false);
    wasm_options.addOption(bool, "enable_glm4", false);
    wasm_options.addOption(bool, "enable_gemma4", false); // disabled: test isolation
    wasm_options.addOption(bool, "enable_diffusion_gemma", false);
    wasm_options.addOption(bool, "enable_deepseek4", false);
    wasm_options.addOption(bool, "enable_llama4", false);
    wasm_options.addOption(bool, "enable_dflash2", false); // draft-only arch; not used in browser inference

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_entry.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
        .strip = true,
    });
    wasm_mod.addImport("build_options", wasm_options.createModule());
    const wasm_lib = b.addExecutable(.{
        .name = "agave",
        .root_module = wasm_mod,
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;
    const install_wasm = b.addInstallArtifact(wasm_lib, .{
        .dest_dir = .{ .override = .{ .custom = "web" } },
    });
    wasm_step.dependOn(&install_wasm.step);

    // Contributor gates. Paths must stay in lockstep with
    // `.github/workflows/ci.yml` fmt-check (`zig fmt --check src/ tests/ build.zig build.zig.zon`).
    const fmt_paths = [_][]const u8{ "src/", "tests/", "build.zig", "build.zig.zon" };
    {
        const fmt_apply = b.addSystemCommand(&.{ b.graph.zig_exe, "fmt" });
        fmt_apply.addArgs(&fmt_paths);
        fmt_apply.has_side_effects = true;
        b.step("fmt", "Apply zig fmt to the paths CI checks").dependOn(&fmt_apply.step);

        const fmt_check_cmd = b.addSystemCommand(&.{ b.graph.zig_exe, "fmt", "--check" });
        fmt_check_cmd.addArgs(&fmt_paths);
        fmt_check_cmd.has_side_effects = true;
        const fmt_check_step = b.step("fmt-check", "Check formatting (same paths as CI)");
        fmt_check_step.dependOn(&fmt_check_cmd.step);

        const docs_check_step = b.step("docs-check", "Docs hygiene (scripts/check-docs.py + its SemVer guard tests)");
        if (python3) |py| {
            const docs_check_cmd = b.addSystemCommand(&.{ py, "scripts/check-docs.py" });
            pin_spawned_python(docs_check_cmd);
            docs_check_cmd.has_side_effects = true;
            docs_check_step.dependOn(&docs_check_cmd.step);
            // CI's docs-check job runs the guard's own tests; keeping them in
            // `check` means a broken guard fails locally, not after a push.
            const docs_check_test_cmd = b.addSystemCommand(&.{ py, "scripts/test_check_docs.py" });
            pin_spawned_python(docs_check_test_cmd);
            docs_check_test_cmd.has_side_effects = true;
            docs_check_step.dependOn(&docs_check_test_cmd.step);
        } else {
            docs_check_step.dependOn(&b.addFail(
                "python3 not found; zig build check needs Python 3.11+ for scripts/check-docs.py",
            ).step);
        }

        const lint_web_cmd = b.addSystemCommand(&.{ "bash", "scripts/lint-web.sh" });
        lint_web_cmd.has_side_effects = true;
        const lint_web_step = b.step("lint-web", "oxlint + tsc (CI lint-web job)");
        lint_web_step.dependOn(&lint_web_cmd.step);

        const lint_shell_cmd = b.addSystemCommand(&.{ "bash", "scripts/lint-shell.sh" });
        lint_shell_cmd.has_side_effects = true;
        const lint_shell_step = b.step("lint-shell", "shellcheck (CI lint-shell job)");
        lint_shell_step.dependOn(&lint_shell_cmd.step);

        const lint_python_cmd = b.addSystemCommand(&.{ "bash", "scripts/lint-python.sh" });
        lint_python_cmd.has_side_effects = true;
        const lint_python_step = b.step("lint-python", "ruff (CI lint-python job)");
        lint_python_step.dependOn(&lint_python_cmd.step);

        // A backup nobody has restored is a hypothesis. This runs the
        // conversation-store backup, verify, and restore path end to end
        // against a scratch store, so the runbook in docs/DURABILITY.md is
        // exercised rather than described.
        const conv_backup_cmd = b.addSystemCommand(&.{ "bash", "scripts/conv-store-backup.sh", "--self-test" });
        conv_backup_cmd.has_side_effects = true;
        const conv_backup_step = b.step("conv-store-backup-test", "Conversation store backup + restore self-test (docs/DURABILITY.md)");
        conv_backup_step.dependOn(&conv_backup_cmd.step);

        // src/web/app.js, the two style.css files and web/*.js are committed
        // bun + Tailwind outputs, @embedFile'd or shipped as-is. CI's lint-web
        // job regenerates and byte-compares them.
        const web_artifacts_cmd = b.addSystemCommand(&.{ "bash", "scripts/check-web-artifacts.sh" });
        web_artifacts_cmd.has_side_effects = true;
        const web_artifacts_step = b.step("check-web", "Committed browser bundles and stylesheets match a fresh build (CI lint-web job)");
        web_artifacts_step.dependOn(&web_artifacts_cmd.step);
        lint_web_step.dependOn(web_artifacts_step);

        // CI's fmt-check job runs the same script, so the pins that make a
        // Docker build reproducible fail locally too, not only after a push.
        const check_pins_cmd = b.addSystemCommand(&.{ "bash", "scripts/check-pins.sh" });
        check_pins_cmd.has_side_effects = true;
        const check_pins_step = b.step("check-pins", "Zig / Debian / SOURCE_DATE_EPOCH pins agree (CI fmt-check job)");
        check_pins_step.dependOn(&check_pins_cmd.step);

        // Two builds of the same source, byte-compared. Not in `check`: it
        // compiles the engine twice and needs its own CI job, not every
        // contributor's gate.
        const reproducible_cmd = b.addSystemCommand(&.{ "bash", "scripts/check-reproducible.sh" });
        reproducible_cmd.has_side_effects = true;
        const reproducible_step = b.step("check-reproducible", "Build twice from different paths and byte-compare (CI reproducible-build job)");
        reproducible_step.dependOn(&reproducible_cmd.step);

        const check_step = b.step("check", "Local CI gate: format check + docs hygiene + pin consistency + unit tests");
        check_step.dependOn(fmt_check_step);
        check_step.dependOn(docs_check_step);
        check_step.dependOn(check_pins_step);
        check_step.dependOn(test_step);
        check_step.dependOn(conv_backup_step);

        // The blocking ci-pass jobs a workstation can reproduce. Docker,
        // cross-compile, wasm, fuzz and PTX freshness stay in CI (or the
        // CONTRIBUTING table): they need Docker, cross toolchains, or a
        // long fuzz budget.
        const ci_step = b.step("ci", "Full local CI gate: check + lint-web + lint-shell + lint-python (needs bun, shellcheck, ruff)");
        ci_step.dependOn(check_step);
        ci_step.dependOn(lint_web_step);
        ci_step.dependOn(lint_shell_step);
        ci_step.dependOn(lint_python_step);
    }
}

/// Wiring shared by the GPU correctness tests under `tests/`: each imports the
/// same `backend` module, compiles at the test optimize mode, and runs under the
/// platform link settings.
const BackendTest = struct {
    b: *std.Build,
    test_step: *std.Build.Step,
    backend_mod: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    filters: []const []const u8,
    test_runner: std.Build.Step.Compile.TestRunner,
    link_metal: bool,

    /// Compiles `root` with a named `backend` import, adds it to `test_step`, and
    /// returns the run step. `extra_name`/`extra_mod` add a second named import
    /// for the SDPA tests, which need the shared harness on top of `backend`.
    fn add(
        self: BackendTest,
        root: []const u8,
        extra_name: ?[]const u8,
        extra_mod: ?*std.Build.Module,
    ) *std.Build.Step.Run {
        const mod = self.b.createModule(.{
            .root_source_file = self.b.path(root),
            .target = self.target,
            .optimize = self.optimize,
        });
        mod.addImport("backend", self.backend_mod);
        if (extra_name) |name| mod.addImport(name, extra_mod.?);
        const t = self.b.addTest(.{ .root_module = mod, .test_runner = self.test_runner, .filters = self.filters });
        linkPlatform(mod, t, self.target, self.link_metal);
        const run = self.b.addRunArtifact(t);
        self.test_step.dependOn(&run.step);
        return run;
    }
};

/// Applies the platform link settings every artifact in this build shares: libc
/// linkage, PIE on the ELF/Mach-O targets, and, when `link_metal` is set, the
/// three macOS frameworks the Metal backend calls into. Vulkan
/// (libvulkan.so / libvulkan.1.dylib via the KosmicKrisp ICD) is loaded at
/// runtime through std.DynLib and needs no link-time dependency.
fn linkPlatform(
    mod: *std.Build.Module,
    compile: *std.Build.Step.Compile,
    resolved: std.Build.ResolvedTarget,
    link_metal: bool,
) void {
    mod.link_libc = true;
    // zig 0.16 ReleaseFast defaults to a non-PIE ET_EXEC on Linux.
    switch (resolved.result.os.tag) {
        .linux, .macos => compile.pie = true,
        else => {},
    }
    if (link_metal) {
        mod.linkFramework("Metal", .{});
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Accelerate", .{});
    }
}

/// A `-Dtest-filter` that matches nothing still compiles every test artifact,
/// runs zero of them, and exits 0: a green run that tested nothing. Report it
/// as a build failure instead, before the compile cost is paid.
///
/// Scans `test "..."` declaration names under `src/` and `tests/`. A filter
/// that matches a name the build does not compile (an unreferenced file, a
/// GPU-guarded test on a CPU-only host) still passes the check; the failure
/// this catches is the typo, which matches nothing at all.
///
/// Returns null when every filter matched, and otherwise prints why and aborts
/// the configure before any test artifact is compiled.
fn rejectEmptyTestFilters(b: *std.Build, filters: []const []const u8) ?void {
    const io = b.graph.io;
    const arena = b.graph.arena;
    const max_source_bytes: std.Io.Limit = .limited(16 << 20);

    const matched = arena.alloc(bool, filters.len) catch @panic("out of memory");
    @memset(matched, false);

    for ([_][]const u8{ "src", "tests" }) |root| {
        var dir = b.build_root.handle.openDir(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var walker = dir.walk(arena) catch @panic("out of memory");
        defer walker.deinit();
        while (walker.next(io) catch null) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;
            const source = dir.readFileAlloc(io, entry.path, arena, max_source_bytes) catch continue;
            var lines = std.mem.splitScalar(u8, source, '\n');
            while (lines.next()) |line| {
                const name = testDeclName(line) orelse continue;
                for (filters, 0..) |filter, i| {
                    if (!matched[i] and std.mem.indexOf(u8, name, filter) != null) matched[i] = true;
                }
            }
        }
    }

    for (filters, 0..) |filter, i| {
        if (matched[i]) continue;
        std.debug.print(
            "error: -Dtest-filter={s} matches no test name under src/ or tests/.\n" ++
                "A filter that matches nothing reports \"All 0 tests passed.\" and exits 0, so the\n" ++
                "run would be green without testing anything. Drop the flag to run everything, or\n" ++
                "list the candidates with: rg -n '^test \"' src/ tests/\n",
            .{filter},
        );
        std.process.exit(1);
    }
    return null;
}

/// Name of a `test "name" {` declaration, or null for any other line.
fn testDeclName(line: []const u8) ?[]const u8 {
    const prefix = "test \"";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const rest = line[prefix.len..];
    const end = std.mem.indexOfScalar(u8, rest, '"') orelse return null;
    return rest[0..end];
}
