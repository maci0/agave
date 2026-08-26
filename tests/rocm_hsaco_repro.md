# ROCm HSACO Zig 0.16.0 Bug Reproducer

**Zig version**: 0.16.0 (LLVM 21)
**ROCm version**: 7.2.3 (HIP 7.2)
**GPU**: AMD Radeon RX 7900 XTX (gfx1100)
**Error**: `hipModuleLoadData` returns 209 (`hipErrorNoBinaryForGpu`)

## Two Bugs

### Bug 1: Malformed `amdhsa.target` Triple

Zig's amdgcn backend generates a wrong target triple in the AMDGPU msgpack metadata:

```
# Zig generates:
amdhsa.target: amdgcn-amd-amdhsa5.0.0-unknown-gfx1100

# Should be:
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
```

The OS version `5.0.0` and environment `unknown` are incorrectly appended to `amdhsa`.

ROCm 7.x validates this field against the ISA reported by `rocminfo` (`amdgcn-amd-amdhsa--gfx1100`). Mismatch → error 209.

**Reproduce:**
```bash
zig build-obj -OReleaseFast -target amdgcn-amdhsa -mcpu=gfx1100 \
    -fstrip -Mroot=src/backend/kernels/rocm/all.zig -fno-emit-asm
llvm-readelf --notes all.o | grep amdhsa.target
# Output: amdhsa.target: amdgcn-amd-amdhsa5.0.0-unknown-gfx1100
```

**Expected**: `amdhsa.target: amdgcn-amd-amdhsa--gfx1100`

### Bug 2: Local Kernel Descriptor Symbols

Zig emits kernel descriptor (`.kd`) symbols as **local** with module prefix.
ROCm requires them as **global** in `.dynsym` without module prefix.

```
# Zig generates (in .symtab):
0000000000000000 l O .rodata 0x40 silu.silu_kernel.kd    # LOCAL, module-prefixed

# hipcc generates (in .dynsym):
0000000000000880 g O .rodata 0x40 .protected silu_kernel.kd   # GLOBAL PROTECTED
```

`hipModuleGetFunction("silu_kernel")` looks up `silu_kernel.kd` in `.dynsym`.
Since Zig's `.kd` symbols are local and not in `.dynsym`, lookup fails.

**Reproduce:**
```bash
zig build-obj -OReleaseFast -target amdgcn-amdhsa -mcpu=gfx1100 \
    -fstrip -Mroot=src/backend/kernels/rocm/all.zig -fno-emit-asm
llvm-nm all.o | grep '.kd'
# All .kd symbols are 'r' (local read-only) with module prefix
```

**Expected**: `.kd` symbols should be `R` (global) without module prefix.

## Workaround Attempts (All Failed)

| Approach | Result |
|----------|--------|
| Binary patch msgpack target string (same-size, null-padded) | ROCm matches full msgpack string including nulls → rejected |
| Binary patch msgpack target string (shorter, fix descsz) | ELF section offsets shift → linker rejects |
| `llvm-objcopy --remove-section=.note` + `--add-section` | Note replaced correctly but KD symbols still local |
| `llvm-objcopy --globalize-symbol` on .o | Doesn't work for section-relative AMDGPU symbols |
| Raw ELF symtab patch (change STB_LOCAL→STB_GLOBAL) | Breaks `sh_info` local/global ordering → linker rejects |
| `os_version_min = .none` in build.zig | Zig ignores it, still generates `5.0.0` |
| Different lld flags (`--pack-dyn-relocs=none`) | Adds PT_NOTE but still error 209 |

## Impact

- HSACO cannot be regenerated on Zig 0.16.0 for ROCm 7.x
- Old pre-committed HSACO (from earlier Zig version) loads but lacks newer kernels
- ROCm GPU inference blocked; CPU fallback works (61 tok/s on Ryzen 9950X)

## Fix Required

In Zig's `src/codegen/llvm.zig` (AMDGPU target handling):
1. Don't append OS version to `amdhsa` in the data layout / target triple
2. Emit `.kd` symbols as `STB_GLOBAL` / `STV_PROTECTED` without module prefix

## Related

- CUDA nvptx64 aliasee bug: LLVM PR #81170 (same class, Zig codegen → LLVM IR incompatibility)
- WASM Invalid cast: similar pattern (Zig LLVM codegen produces invalid IR for specific targets)
