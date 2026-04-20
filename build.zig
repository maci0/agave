const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    // ── Backend enable/disable flags (all default to true) ────────
    const enable_cpu = b.option(bool, "enable-cpu", "Enable CPU backend (default: true)") orelse true;
    const enable_metal = b.option(bool, "enable-metal", "Enable Metal backend (default: true)") orelse true;
    const enable_cuda = b.option(bool, "enable-cuda", "Enable CUDA backend (default: true)") orelse true;
    const enable_rocm = b.option(bool, "enable-rocm", "Enable ROCm backend (default: true)") orelse true;

    const enable_vulkan = b.option(bool, "enable-vulkan", "Enable Vulkan backend (default: true)") orelse true;

    // ── Model enable/disable flags (all default to true) ─────────
    const enable_gemma3 = b.option(bool, "enable-gemma3", "Enable Gemma3 model support (default: true)") orelse true;
    const enable_qwen35 = b.option(bool, "enable-qwen35", "Enable Qwen3.5 model support (default: true)") orelse true;
    const enable_gpt_oss = b.option(bool, "enable-gpt-oss", "Enable GPT-OSS model support (default: true)") orelse true;
    const enable_nemotron_h = b.option(bool, "enable-nemotron-h", "Enable Nemotron-H model support (default: true)") orelse true;
    const enable_nemotron_nano = b.option(bool, "enable-nemotron-nano", "Enable Nemotron-Nano model support (default: true)") orelse true;
    const enable_glm4 = b.option(bool, "enable-glm4", "Enable GLM-4 model support (default: true)") orelse true;
    const enable_gemma4 = b.option(bool, "enable-gemma4", "Enable Gemma4 model support (default: true)") orelse true;

    // ── Helper: link frameworks for macOS ─────────────────────────
    // Note: Vulkan (libvulkan.so / libMoltenVK.dylib) is loaded at runtime
    // via std.DynLib — no link-time dependency needed.
    const link_metal = enable_metal and target.result.os.tag == .macos;
    const link_platform = struct {
        fn apply(mod: *std.Build.Module, _: *std.Build.Step.Compile, _: std.Build.ResolvedTarget) void {
            mod.link_libc = true;
        }
    }.apply;

    // ── CUDA PTX kernels (cross-compiled via nvptx64-cuda) ─────────
    // Compiles Zig CUDA kernels to PTX assembly. The resulting .s file
    // is placed in zig-out/ and can be embedded into cuda.zig via @embedFile.
    // Build with: zig build ptx [-Dcuda-sm=sm_80]
    const CudaSm = enum { sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 };
    const cuda_sm = b.option(CudaSm, "cuda-sm", "CUDA SM target (default: sm_90)") orelse .sm_90;
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
    {
        const kernel_files = [_][]const u8{
            "all",       "silu",     "gelu",      "add",       "mul",
            "rms_norm",  "softmax",  "l2_norm",   "rope",      "gemv_f32",
            "gemv_bf16", "gemv_f16", "gemv_q8_0", "gemv_q4_0",
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
            const install = b.addInstallFile(ptx.getEmittedAsm(), b.fmt("ptx/{s}.ptx", .{name}));
            ptx_step.dependOn(&install.step);
        }
    }

    // ── ROCm AMDGCN kernels (cross-compiled via amdgcn-amdhsa) ───────
    // Compiles Zig ROCm kernels to AMDGCN ISA, producing an ELF object.
    // Build with: zig build amdgcn [-Drocm-arch=gfx1100]
    // After building, copy zig-out/rocm/kernels.o to
    // src/backend/kernels/rocm/kernels.hsaco and commit.
    const amdgcn_step = b.step("amdgcn", "Compile ROCm kernels to AMDGCN ISA");
    {
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

        // Link into shared ELF (HSACO) for hipModuleLoadData
        const link = b.addSystemCommand(&.{ "ld.lld", "-shared", "-o" });
        const hsaco_out = link.addOutputFileArg("kernels.hsaco");
        link.addFileArg(obj.getEmittedBin());
        const install_hsaco = b.addInstallFile(hsaco_out, "rocm/kernels.hsaco");
        amdgcn_step.dependOn(&install_hsaco.step);
    }

    // ── ReleaseFast executable (default) ──────────────────────────
    const backend_options = b.addOptions();
    backend_options.addOption(bool, "enable_cpu", enable_cpu);
    backend_options.addOption(bool, "enable_metal", enable_metal);
    backend_options.addOption(bool, "enable_vulkan", enable_vulkan);
    backend_options.addOption(bool, "enable_cuda", enable_cuda);
    backend_options.addOption(bool, "enable_rocm", enable_rocm);
    backend_options.addOption(bool, "enable_gemma3", enable_gemma3);
    backend_options.addOption(bool, "enable_qwen35", enable_qwen35);
    backend_options.addOption(bool, "enable_gpt_oss", enable_gpt_oss);
    backend_options.addOption(bool, "enable_nemotron_h", enable_nemotron_h);
    backend_options.addOption(bool, "enable_nemotron_nano", enable_nemotron_nano);
    backend_options.addOption(bool, "enable_glm4", enable_glm4);
    backend_options.addOption(bool, "enable_gemma4", enable_gemma4);

    const mod_rel = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    mod_rel.addImport("build_options", backend_options.createModule());

    const exe_rel = b.addExecutable(.{ .name = "agave", .root_module = mod_rel });
    link_platform(mod_rel, exe_rel, target);
    if (link_metal) {
        mod_rel.linkFramework("Metal", .{});
        mod_rel.linkFramework("Foundation", .{});
    }
    b.installArtifact(exe_rel);

    // ── Debug executable (also built by default) ─────────────────
    const mod_dbg = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = .Debug,
    });
    mod_dbg.addImport("build_options", backend_options.createModule());

    const exe_dbg = b.addExecutable(.{ .name = "agave-debug", .root_module = mod_dbg });
    link_platform(mod_dbg, exe_dbg, target);
    if (link_metal) {
        mod_dbg.linkFramework("Metal", .{});
        mod_dbg.linkFramework("Foundation", .{});
    }
    b.installArtifact(exe_dbg);

    // ── Run step (uses the optimized binary) ─────────────────────
    const run_cmd = b.addRunArtifact(exe_rel);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run agave (ReleaseFast)").dependOn(&run_cmd.step);

    // ── Test step ────────────────────────────────────────────────
    const test_step = b.step("test", "Run unit tests");

    // Main test suite (inline tests from src/)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod_rel })).step);

    // SDPA oracle self-tests (validates ground-truth reference for GPU tests)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/sdpa_oracle.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }) })).step);

    // Golden harness unit tests (degenerate output detection)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/models/golden_harness.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }) })).step);

    // Shared backend module for SDPA hardware tests (provides named "backend" import).
    // Rooted at src/test_exports.zig so transitive imports resolve within src/.
    const backend_test_mod = b.createModule(.{
        .root_source_file = b.path("src/test_exports.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    backend_test_mod.addImport("build_options", backend_options.createModule());

    // Shared oracle module for SDPA hardware tests
    const oracle_mod = b.createModule(.{
        .root_source_file = b.path("tests/sdpa_oracle.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    // Shared dual-delta test harness for GPU SDPA correctness tests
    const sdpa_harness_mod = b.createModule(.{
        .root_source_file = b.path("tests/sdpa_harness.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    sdpa_harness_mod.addImport("backend", backend_test_mod);
    sdpa_harness_mod.addImport("sdpa_oracle", oracle_mod);

    // CUDA SDPA correctness tests (skips at runtime if no CUDA hardware)
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cuda_sdpa.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        mod.addImport("backend", backend_test_mod);
        mod.addImport("sdpa_harness", sdpa_harness_mod);
        const t = b.addTest(.{ .root_module = mod });
        link_platform(mod, t, target);
        if (link_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("Foundation", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Metal SDPA correctness tests (skips at runtime if not macOS)
    {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_metal_sdpa.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        mod.addImport("backend", backend_test_mod);
        mod.addImport("sdpa_harness", sdpa_harness_mod);
        const t = b.addTest(.{ .root_module = mod });
        link_platform(mod, t, target);
        if (link_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("Foundation", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // ROCm kernel tests (placeholder — skips until hardware available)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("tests/test_rocm_kernel.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }) })).step);

    // ── Benchmark binary (standalone micro-benchmark) ──────────────
    const mod_bench = b.createModule(.{
        .root_source_file = b.path("src/micro_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    mod_bench.addImport("build_options", backend_options.createModule());

    const exe_bench = b.addExecutable(.{ .name = "agave-bench", .root_module = mod_bench });
    link_platform(mod_bench, exe_bench, target);
    if (link_metal) {
        mod_bench.linkFramework("Metal", .{});
        mod_bench.linkFramework("Foundation", .{});
    }
    b.installArtifact(exe_bench);

    const bench_run = b.addRunArtifact(exe_bench);
    bench_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench_run.addArgs(args);
    b.step("bench", "Run micro-benchmarks (ReleaseFast)").dependOn(&bench_run.step);
}
