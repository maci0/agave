//! Build configuration for Agave — LLM inference engine.
//! Targets: ReleaseFast (agave), Debug (agave-debug), WASM (agave.wasm),
//! CUDA PTX kernels (zig build ptx), ROCm AMDGCN kernels (zig build amdgcn),
//! micro-benchmarks (zig build bench).

const std = @import("std");
const builtin = @import("builtin");
/// Product SemVer from build.zig.zon; injected into binaries via build_options.version.
const package_version: []const u8 = @import("build.zig.zon").version;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    // ── Backend enable/disable flags (all default to true) ────────
    const enable_cpu = b.option(bool, "enable-cpu", "Enable CPU backend (default: true)") orelse true;
    const enable_metal = b.option(bool, "enable-metal", "Enable Metal backend (default: true)") orelse true;
    const enable_cuda = b.option(bool, "enable-cuda", "Enable CUDA backend (default: true)") orelse true;
    const enable_rocm = b.option(bool, "enable-rocm", "Enable ROCm backend (default: true)") orelse true;
    // Debug binary fails to link on Linux x86_64 with GCC >= 16 (R_X86_64_PC64 relocation).
    // Use -Denable-debug=false to skip it on affected systems.
    const enable_debug_binary = b.option(bool, "enable-debug", "Build agave-debug binary (default: true)") orelse true;

    const enable_vulkan = b.option(bool, "enable-vulkan", "Enable Vulkan backend (default: true)") orelse true;
    const enable_webgpu = b.option(bool, "enable-webgpu", "Enable WebGPU backend via wgpu-native (default: true)") orelse true;

    // ── Model enable/disable flags (all default to true) ─────────
    const enable_gemma3 = b.option(bool, "enable-gemma3", "Enable Gemma3 model support (default: true)") orelse true;
    const enable_qwen35 = b.option(bool, "enable-qwen35", "Enable Qwen3.5 model support (default: true)") orelse true;
    const enable_gpt_oss = b.option(bool, "enable-gpt-oss", "Enable GPT-OSS model support (default: true)") orelse true;
    const enable_nemotron_h = b.option(bool, "enable-nemotron-h", "Enable Nemotron-H model support (default: true)") orelse true;
    const enable_nemotron_nano = b.option(bool, "enable-nemotron-nano", "Enable Nemotron-Nano model support (default: true)") orelse true;
    const enable_glm4 = b.option(bool, "enable-glm4", "Enable GLM-4 model support (default: true)") orelse true;
    const enable_gemma4 = b.option(bool, "enable-gemma4", "Enable Gemma4 model support (default: true)") orelse true;
    const enable_diffusion_gemma = b.option(bool, "enable-diffusion-gemma", "Enable DiffusionGemma model support (default: true)") orelse true;
    const enable_deepseek4 = b.option(bool, "enable-deepseek4", "Enable DeepSeek V4 model support (default: true)") orelse true;
    const enable_llama4 = b.option(bool, "enable-llama4", "Enable Llama 4 model support (default: true)") orelse true;

    // ── Helper: link frameworks for macOS ─────────────────────────
    // Note: Vulkan (libvulkan.so / libvulkan.1.dylib (via KosmicKrisp ICD)) is loaded at runtime
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
            // Core ops
            "all",           "silu",           "gelu",           "add",            "mul",
            "rms_norm",      "softmax",        "l2_norm",        "rope",           "add_scaled",
            "silu_mul",      "gelu_mul",       "add_rms_norm",   "rms_norm_add",   "rms_norm_batched",
            "rope_batched",  "sigmoid_mul",    "deinterleave",   "split_qgate",
            "deltanet_recurrence",
            // SDPA
               "sdpa",
            "sdpa_turbo",    "sdpa_prefill",   "sdpa_tree",
            // Dense GEMV
                 "gemv_f32",       "gemv_bf16",
            "gemv_f16",      "gemv_t_q8_0",
            // Quantized GEMV — standard GGUF formats
               "gemv_q8_0",      "gemv_q4_0",      "gemv_q4_0_batch",
            "gemv_q4_1",     "gemv_q5_0",      "gemv_q4_k",      "gemv_q5_k",      "gemv_q6_k",
            "gemv_q2_k",     "gemv_q3_k",      "gemv_iq4_nl",    "gemv_iq4_xs",
            // FP8/FP4
               "gemv_fp8_e4m3",
            "gemv_fp8_e5m2", "gemv_nvfp4_st",  "gemv_mxfp4_st",  "gemv_fp4_tc",
            // MLX / TQ
               "gemv_mlx_q4",
            "gemv_mlx_q6",   "gemv_mlx_q8",    "gemv_tq1_0",     "gemv_tq2_0",
            // Specialist formats
                "gemv_gptq",
            "gemv_awq",      "gemv_hqq",
            // GEMM
                  "gemm_q8_0",
            // Megakernels
                 "mega_qwen35_q8", "mega_gemma_q4k",
            "mega_gemma_q8",
            // Fused FFN
            "fused_ffn_q8_0", "fused_ffn_q4_k", "fused_ffn_q5_k", "fused_ffn_q6_k",
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
            const fixup = b.addSystemCommand(&.{
                "python3", "-c",
                \\import re, sys
                \\ptx = open(sys.argv[1]).read()
                \\for clean, mangled in re.findall(r'\.alias (\w+_kernel), ([^;]+);', ptx):
                \\    ptx = ptx.replace(f'.func {mangled}(', f'.entry {clean}(')
                \\ptx = re.sub(r'\.alias \w+_kernel, [^;]+;\n', '', ptx)
                \\ptx = re.sub(r'^\.func (\w+_kernel)$', r'.entry \1', ptx, flags=re.MULTILINE)
                \\sys.stdout.write(ptx)
            });
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

        // Workaround: two Zig 0.16 bugs for AMDGCN targets (upstream Zig AMDGCN issues):
        //
        // Bug 1 — Wrong ISA string in metadata:
        //   Zig emits amdhsa.target = "amdgcn-amd-amdhsa5.0.0-unknown-gfx1100"
        //   (OS semver from Target.zig appended to triple).
        //   HIP does exact-string matching; expects "amdgcn-amd-amdhsa--gfx1100".
        //   Workaround: patch the NT_AMDGPU_METADATA note section in the .o
        //   before linking (ET_REL has no VirtAddr constraints, safe to shrink).
        //
        // Bug 2 — .kd symbols emitted as LOCAL:
        //   Kernel descriptor symbols (foo.kd) need GLOBAL binding so the linker
        //   exports them to .dynsym. Zig emits them as LOCAL → HIP can't find them.
        //   Workaround: llvm-objcopy --globalize-symbol for every .kd symbol.
        //
        // Run this build step on a Linux machine with ROCm installed and llvm-objcopy
        // (from /opt/rocm/lib/llvm/bin) in PATH. The resulting HSACO is committed.
        // Pass the fixup script via addFileArg so the build graph tracks it as an
        // input (rebuilds when the script changes) and avoids configure-time getPath
        // absolute host paths.
        const fix_obj = b.addSystemCommand(&.{"python3"});
        fix_obj.addFileArg(b.path("src/backend/kernels/rocm/fix_kd_isa.py"));
        fix_obj.addFileArg(obj.getEmittedBin());
        const fixed_obj = fix_obj.addOutputFileArg("kernels_fixed.o");
        fix_obj.step.dependOn(&obj.step);

        const link = b.addSystemCommand(&.{ "ld.lld", "-shared", "-o" });
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
    backend_options.addOption(bool, "enable_gpt_oss", enable_gpt_oss);
    backend_options.addOption(bool, "enable_nemotron_h", enable_nemotron_h);
    backend_options.addOption(bool, "enable_nemotron_nano", enable_nemotron_nano);
    backend_options.addOption(bool, "enable_glm4", enable_glm4);
    backend_options.addOption(bool, "enable_gemma4", enable_gemma4);
    backend_options.addOption(bool, "enable_diffusion_gemma", enable_diffusion_gemma);
    backend_options.addOption(bool, "enable_deepseek4", enable_deepseek4);
    backend_options.addOption(bool, "enable_llama4", enable_llama4);

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
    link_platform(mod_rel, exe_rel, target);
    if (link_metal) {
        mod_rel.linkFramework("Metal", .{});
        mod_rel.linkFramework("Foundation", .{});
        mod_rel.linkFramework("Accelerate", .{});
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
        mod_dbg.linkFramework("Accelerate", .{});
    }
    if (enable_debug_binary) b.installArtifact(exe_dbg);

    // ── Run step (uses the optimized binary) ─────────────────────
    const run_cmd = b.addRunArtifact(exe_rel);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run agave (ReleaseFast)").dependOn(&run_cmd.step);

    // ── Test step ────────────────────────────────────────────────
    const test_step = b.step("test", "Run unit tests");

    // Test modules use ReleaseSafe so std.debug.assert / unreachable fire.
    // Reusing mod_rel (ReleaseFast) silently no-ops ~400 assert-based checks
    // in fuzz and unit tests (see std.debug.assert docs).
    const test_optimize: std.builtin.OptimizeMode = .ReleaseSafe;

    // Default `--listen=-` server deadlocks under parallel `addRunArtifact` on
    // this host: children block in receiveMessage and the parent never sends
    // query_test_metadata after another artifact writes stderr. Simple mode
    // uses the same runner without the server protocol; failure is exit status.
    const zig_lib = b.graph.zig_lib_directory.path orelse ".";
    const simple_test_runner: std.Build.Step.Compile.TestRunner = .{
        .path = .{ .cwd_relative = b.pathJoin(&.{ zig_lib, "compiler", "test_runner.zig" }) },
        .mode = .simple,
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
        const t = b.addTest(.{ .root_module = mod_test, .test_runner = simple_test_runner });
        link_platform(mod_test, t, target);
        if (link_metal) {
            mod_test.linkFramework("Metal", .{});
            mod_test.linkFramework("Foundation", .{});
            mod_test.linkFramework("Accelerate", .{});
        }
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
    })).step);

    // Golden harness unit tests (degenerate output detection)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/models/golden_harness.zig"),
            .target = target,
            .optimize = test_optimize,
        }),
        .test_runner = simple_test_runner,
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

    // CUDA SDPA correctness tests (skips at runtime if no CUDA hardware).
    // Skip compile when CUDA is NullBackend: init() is a compileError.
    if (enable_cuda) {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cuda_sdpa.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod.addImport("backend", backend_test_mod);
        mod.addImport("sdpa_harness", sdpa_harness_mod);
        const t = b.addTest(.{ .root_module = mod, .test_runner = simple_test_runner });
        link_platform(mod, t, target);
        if (link_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("Foundation", .{});
            mod.linkFramework("Accelerate", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Metal SDPA correctness tests (skips at runtime if not macOS).
    // Skip compile when Metal is NullBackend: init() arity does not match.
    if (enable_metal) {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_metal_sdpa.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod.addImport("backend", backend_test_mod);
        mod.addImport("sdpa_harness", sdpa_harness_mod);
        const t = b.addTest(.{ .root_module = mod, .test_runner = simple_test_runner });
        link_platform(mod, t, target);
        if (link_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("Foundation", .{});
            mod.linkFramework("Accelerate", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // WebGPU MLX GEMV row-chunking (vocab > 65535). Skips if wgpu-native missing.
    // Skip compile when WebGPU is NullBackend: init(allocator) vs init(allocator, device).
    if (enable_webgpu) {
        const mod = b.createModule(.{
            .root_source_file = b.path("tests/test_webgpu_mlx_gemv.zig"),
            .target = target,
            .optimize = test_optimize,
        });
        mod.addImport("backend", backend_test_mod);
        const t = b.addTest(.{ .root_module = mod, .test_runner = simple_test_runner });
        link_platform(mod, t, target);
        if (link_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("Foundation", .{});
            mod.linkFramework("Accelerate", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
        const webgpu_mlx_step = b.step("test-webgpu-mlx", "WebGPU MLX-Q4 GEMV chunking test");
        webgpu_mlx_step.dependOn(&b.addRunArtifact(t).step);
    }

    // ROCm kernel tests (placeholder — skips until hardware available)
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_rocm_kernel.zig"),
            .target = target,
            .optimize = test_optimize,
        }),
        .test_runner = simple_test_runner,
    })).step);

    // micro_bench pure-function tests (parseKeyValue, parseKernelName, etc.)
    {
        const mod_bench_test = b.createModule(.{
            .root_source_file = b.path("src/micro_bench.zig"),
            .target = target,
            .optimize = .Debug,
        });
        mod_bench_test.addImport("build_options", backend_options.createModule());
        const t = b.addTest(.{ .root_module = mod_bench_test, .test_runner = simple_test_runner });
        link_platform(mod_bench_test, t, target);
        if (link_metal) {
            mod_bench_test.linkFramework("Metal", .{});
            mod_bench_test.linkFramework("Foundation", .{});
            mod_bench_test.linkFramework("Accelerate", .{});
        }
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // wasm_entry pure-function tests (agave_alloc, agave_dealloc, wasmLogFn)
    {
        const mod_wasm_test = b.createModule(.{
            .root_source_file = b.path("src/wasm_entry.zig"),
            .target = target,
            .optimize = .Debug,
        });
        mod_wasm_test.addImport("build_options", backend_options.createModule());
        const t = b.addTest(.{ .root_module = mod_wasm_test, .test_runner = simple_test_runner });
        link_platform(mod_wasm_test, t, target);
        if (link_metal) {
            mod_wasm_test.linkFramework("Metal", .{});
            mod_wasm_test.linkFramework("Foundation", .{});
            mod_wasm_test.linkFramework("Accelerate", .{});
        }
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
    link_platform(mod_bench, exe_bench, target);
    if (link_metal) {
        mod_bench.linkFramework("Metal", .{});
        mod_bench.linkFramework("Foundation", .{});
        mod_bench.linkFramework("Accelerate", .{});
    }
    b.installArtifact(exe_bench);

    const bench_run = b.addRunArtifact(exe_bench);
    bench_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench_run.addArgs(args);
    b.step("bench", "Run micro-benchmarks (ReleaseFast)").dependOn(&bench_run.step);

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
    wasm_options.addOption(bool, "enable_gpt_oss", false);
    wasm_options.addOption(bool, "enable_nemotron_h", false);
    wasm_options.addOption(bool, "enable_nemotron_nano", false);
    wasm_options.addOption(bool, "enable_glm4", false);
    wasm_options.addOption(bool, "enable_gemma4", false); // disabled: test isolation
    wasm_options.addOption(bool, "enable_diffusion_gemma", false);
    wasm_options.addOption(bool, "enable_deepseek4", false);
    wasm_options.addOption(bool, "enable_llama4", false);

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_entry.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
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
}
