#!/usr/bin/env python3
"""
Workaround for two Zig 0.16 bugs in AMDGCN kernel objects:

  Bug 1: ISA string in NT_AMDGPU_METADATA note includes OS semver.
    Zig emits: amdgcn-amd-amdhsa5.0.0-unknown-gfxXXXX
    HIP needs: amdgcn-amd-amdhsa--gfxXXXX
    Fix: surgical byte replace in the note section (ET_REL, no VirtAddr issue).

  Bug 2: Kernel descriptor symbols (.kd) have LOCAL binding.
    HIP looks up foo.kd in .dynsym; LOCAL symbols aren't exported by the linker.
    Fix: use llvm-objcopy --globalize-symbol for each .kd symbol.

Usage: python3 fix_kd_isa.py input.o output.o
Requires: llvm-objcopy and llvm-readelf in PATH (from /opt/rocm/lib/llvm/bin or similar).
"""
import re
import struct
import subprocess
import sys
import tempfile
import os

NT_META = 32
# Match Zig's broken ISA triple for any gfx arch: amdhsa{semver}-unknown-gfxXXXX
ISA_RE = re.compile(rb'amdgcn-amd-amdhsa[0-9.]+-unknown-(gfx[0-9a-z]+)')


def fixstr(s: bytes) -> bytes:
    if len(s) <= 31:
        return bytes([0xA0 | len(s)]) + s
    return bytes([0xD9, len(s)]) + s


def patch_isa(d: bytearray) -> bytearray:
    m = ISA_RE.search(bytes(d))
    if m is None:
        return d  # already correct or unexpected format

    wrong = m.group(0)
    right = b'amdgcn-amd-amdhsa--' + m.group(1)
    wrong_enc = fixstr(wrong)
    right_enc = fixstr(right)

    idx = bytes(d).find(wrong_enc)
    if idx == -1:
        return d

    e_shoff = struct.unpack_from('<Q', d, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', d, 0x3A)[0]
    e_shnum = struct.unpack_from('<H', d, 0x3C)[0]

    for i in range(e_shnum):
        hdr = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from('<I', d, hdr + 4)[0]
        sh_off = struct.unpack_from('<Q', d, hdr + 24)[0]
        sh_size = struct.unpack_from('<Q', d, hdr + 32)[0]
        if not (sh_off <= idx < sh_off + sh_size):
            continue

        if sh_type == 7:  # SHT_NOTE
            note = bytes(d[sh_off:sh_off + sh_size])
            pos = 0
            new_sh_size = 0
            while pos + 12 <= len(note):
                namesz, descsz, ntype = struct.unpack_from('<III', note, pos)
                np = (namesz + 3) & ~3
                dp = (descsz + 3) & ~3
                desc_start = sh_off + pos + 12 + np
                if desc_start <= idx < desc_start + descsz and ntype == NT_META:
                    new_descsz = descsz + len(right_enc) - len(wrong_enc)
                    new_dp = (new_descsz + 3) & ~3
                    struct.pack_into('<I', d, sh_off + pos + 4, new_descsz)
                    new_sh_size += 12 + np + new_dp
                else:
                    new_sh_size += 12 + np + dp
                pos += 12 + np + dp
            struct.pack_into('<Q', d, hdr + 32, new_sh_size)

        delta = len(right_enc) - len(wrong_enc)
        for j in range(e_shnum):
            h = e_shoff + j * e_shentsize
            off = struct.unpack_from('<Q', d, h + 24)[0]
            if off > idx:
                struct.pack_into('<Q', d, h + 24, off + delta)
        if e_shoff > idx:
            struct.pack_into('<Q', d, 0x28, e_shoff + delta)

        new_d = bytearray(d[:idx]) + right_enc + bytearray(d[idx + len(wrong_enc):])
        print(f'[fix_kd_isa] ISA patched: {wrong.decode()} -> {right.decode()}')
        return new_d

    return d


def require_tool(name: str) -> None:
    from shutil import which
    if which(name) is None:
        raise SystemExit(f'[fix_kd_isa] error: {name} not found in PATH (needed for HSACO patch)')


def globalize_kd(obj_path: str, out_path: str):
    """Use llvm-objcopy to globalize all .kd symbols."""
    require_tool('llvm-readelf')
    require_tool('llvm-objcopy')
    result = subprocess.run(
        ['llvm-readelf', '--syms', obj_path],
        capture_output=True, text=True, check=True
    )
    kd_syms = [
        line.split()[-1]
        for line in result.stdout.splitlines()
        if '.kd' in line and 'LOCAL' in line
    ]
    if not kd_syms:
        # No LOCAL .kd symbols — still copy, but warn if no .kd at all.
        if '.kd' not in result.stdout:
            raise SystemExit('[fix_kd_isa] error: no .kd symbols found in object')
        subprocess.run(['cp', obj_path, out_path], check=True)
        return

    args = ['llvm-objcopy']
    for sym in kd_syms:
        args += ['--globalize-symbol', sym]
    args += [obj_path, out_path]
    subprocess.run(args, check=True)
    print(f'[fix_kd_isa] globalized {len(kd_syms)} .kd symbols')


def main():
    inp = sys.argv[1]
    out = sys.argv[2]

    # Step 1: patch ISA string in memory
    d = bytearray(open(inp, 'rb').read())
    d = patch_isa(d)

    # Step 2: write patched .o to temp file, then globalize .kd symbols
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
        tmp.write(d)
        tmp_path = tmp.name

    try:
        globalize_kd(tmp_path, out)
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    main()
