#!/usr/bin/env python3
"""
Workaround for three Zig 0.16 bugs in AMDGCN kernel objects:

  Bug 1: ISA string in NT_AMDGPU_METADATA note includes OS semver.
    Zig emits: amdgcn-amd-amdhsa5.0.0-unknown-gfxXXXX
    HIP needs: amdgcn-amd-amdhsa--gfxXXXX
    Fix: surgical byte replace in the note section (ET_REL, no VirtAddr issue).

  Bug 2: Kernel descriptor symbols (.kd) have LOCAL binding.
    HIP looks up foo.kd in .dynsym; LOCAL symbols aren't exported by the linker.
    Fix: use llvm-objcopy --globalize-symbol for each .kd symbol.

  Bug 3: Kernel metadata names are module-qualified.
    Metadata emits .name = "modname.kernel_name" / .symbol = "modname.kernel_name.kd"
    while the exported function symbol is the plain "kernel_name". HIP resolves
    hipModuleGetFunction("kernel_name") against metadata .name -> no match
    ("Cannot find Symbol with name: ...").
    Fix: msgpack-rewrite the metadata stripping the module prefix from every
    kernel's .name/.symbol, and --redefine-sym the .kd dynamic symbols so
    "<plain>.kd" resolves too.

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


# ── Minimal msgpack codec (subset emitted by comgr metadata) ─────────────────

def mp_decode(buf: bytes, pos: int = 0):
    b = buf[pos]
    if b <= 0x7F:
        return b, pos + 1
    if b >= 0xE0:
        return b - 256, pos + 1
    if 0x80 <= b <= 0x8F:
        n, pos = b & 0xF, pos + 1
    elif 0xDE == b:
        n, pos = struct.unpack_from('>H', buf, pos + 1)[0], pos + 3
    elif 0xDF == b:
        n, pos = struct.unpack_from('>I', buf, pos + 1)[0], pos + 5
    else:
        n = None
    if n is not None:
        out = {}
        for _ in range(n):
            k, pos = mp_decode(buf, pos)
            v, pos = mp_decode(buf, pos)
            out[k] = v
        return out, pos
    if 0x90 <= b <= 0x9F:
        n, pos = b & 0xF, pos + 1
    elif b == 0xDC:
        n, pos = struct.unpack_from('>H', buf, pos + 1)[0], pos + 3
    elif b == 0xDD:
        n, pos = struct.unpack_from('>I', buf, pos + 1)[0], pos + 5
    else:
        n = None
    if n is not None:
        out = []
        for _ in range(n):
            v, pos = mp_decode(buf, pos)
            out.append(v)
        return out, pos
    if b <= 0xBF and b >= 0xA0:
        n = b & 0x1F
        s = buf[pos + 1:pos + 1 + n]
        return s.decode(), pos + 1 + n
    if b == 0xD9:
        n = buf[pos + 1]
        s = buf[pos + 2:pos + 2 + n]
        return s.decode(), pos + 2 + n
    if b == 0xDA:
        n = struct.unpack_from('>H', buf, pos + 1)[0]
        return buf[pos + 3:pos + 3 + n].decode(), pos + 3 + n
    if b == 0xC4:
        n = buf[pos + 1]
        return bytes(buf[pos + 2:pos + 2 + n]), pos + 2 + n
    if b == 0xC5:
        n = struct.unpack_from('>H', buf, pos + 1)[0]
        return bytes(buf[pos + 3:pos + 3 + n]), pos + 3 + n
    if b < 0x80:
        return b, pos + 1
    if b == 0xCC:
        return buf[pos + 1], pos + 2
    if b == 0xCD:
        return struct.unpack_from('>H', buf, pos + 1)[0], pos + 3
    if b == 0xCE:
        return struct.unpack_from('>I', buf, pos + 1)[0], pos + 5
    if b == 0xCF:
        return struct.unpack_from('>Q', buf, pos + 1)[0], pos + 9
    if b == 0xD0:
        return struct.unpack_from('<b', buf, pos + 1)[0], pos + 2
    if b == 0xD1:
        return struct.unpack_from('<h', buf, pos + 1)[0], pos + 3
    if b == 0xD2:
        return struct.unpack_from('<i', buf, pos + 1)[0], pos + 5
    if b == 0xD3:
        return struct.unpack_from('<q', buf, pos + 1)[0], pos + 9
    if b == 0xCA:
        return struct.unpack_from('<f', buf, pos + 1)[0], pos + 5
    if b == 0xCB:
        return struct.unpack_from('<d', buf, pos + 1)[0], pos + 9
    if b == 0xC2:
        return False, pos + 1
    if b == 0xC3:
        return True, pos + 1
    if b == 0xC0:
        return None, pos + 1
    raise ValueError(f'msgpack: unsupported byte {b:#x} at {pos}')


def mp_encode(v) -> bytes:
    if v is None:
        return b'\xc0'
    if v is True:
        return b'\xc3'
    if v is False:
        return b'\xc2'
    if isinstance(v, int):
        if 0 <= v <= 0x7F:
            return bytes([v])
        if 0 <= v <= 0xFF:
            return b'\xcc' + bytes([v])
        if 0 <= v <= 0xFFFF:
            return b'\xcd' + struct.pack('>H', v)
        if 0 <= v <= 0xFFFFFFFF:
            return b'\xce' + struct.pack('>I', v)
        return b'\xcf' + struct.pack('>Q', v)
    if isinstance(v, float):
        return b'\xcb' + struct.pack('<d', v)
    if isinstance(v, str):
        raw = v.encode()
        if len(raw) <= 31:
            return bytes([0xA0 | len(raw)]) + raw
        if len(raw) <= 255:
            return b'\xd9' + bytes([len(raw)]) + raw
        return b'\xda' + struct.pack('>H', len(raw)) + raw
    if isinstance(v, (bytes, bytearray)):
        n = len(v)
        if n <= 255:
            return b'\xc4' + bytes([n]) + bytes(v)
        return b'\xc5' + struct.pack('>H', n) + bytes(v)
    if isinstance(v, list):
        head = bytes([0x90 | len(v)]) if len(v) <= 15 else b'\xdc' + struct.pack('>H', len(v))
        return head + b''.join(mp_encode(x) for x in v)
    if isinstance(v, dict):
        head = bytes([0x80 | len(v)]) if len(v) <= 15 else b'\xde' + struct.pack('>I', len(v))
        out = head
        for k in v:
            out += mp_encode(k) + mp_encode(v[k])
        return out
    raise TypeError(f'msgpack: unsupported type {type(v)}')


def find_note(d: bytearray):
    """Locate (section_hdr_off, namesz_pos, descsz_pos, desc_slice) of NT_META."""
    e_shoff = struct.unpack_from('<Q', d, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', d, 0x3A)[0]
    e_shnum = struct.unpack_from('<H', d, 0x3C)[0]
    for i in range(e_shnum):
        hdr = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from('<I', d, hdr + 4)[0]
        if sh_type != 7:  # SHT_NOTE
            continue
        sh_off = struct.unpack_from('<Q', d, hdr + 24)[0]
        sh_size = struct.unpack_from('<Q', d, hdr + 32)[0]
        note = bytes(d[sh_off:sh_off + sh_size])
        pos = 0
        while pos + 12 <= len(note):
            namesz, descsz, ntype = struct.unpack_from('<III', note, pos)
            npad = (namesz + 3) & ~3
            if ntype == NT_META:
                desc_start = pos + 12 + npad
                return hdr, sh_off, pos, note[desc_start:desc_start + descsz], sh_size - (desc_start + descsz)
            pos += 12 + npad + ((descsz + 3) & ~3)
    return None


def replace_note_section(d: bytearray, loc, transform):
    """Decode the NT_META desc, apply transform(meta)->meta, re-encode, and
    splice the record back into its SHT_NOTE section, fixing sizes/offsets.
    Returns (new_bytearray, renames)."""
    hdr, sh_off, rec_pos, desc, _tail = loc
    namesz = struct.unpack_from('<I', d, sh_off + rec_pos)[0]
    npad = (namesz + 3) & ~3
    meta, _ = mp_decode(desc)
    renames = transform(meta)
    new_desc = mp_encode(meta)
    # Length preservation: pad the top-level map with an ignorable entry so the
    # rewritten note is byte-identical in size (no ELF layout shifts anywhere).
    if len(new_desc) < len(desc):
        # Filler entry: fixstr key + fixint value; iterate L until the encoded
        # size lands exactly on the original (covers fixmap->map16 growth).
        target = len(desc)
        meta[""] = 0
        for _ in range(8):
            cur = mp_encode(meta)
            diff = target - len(cur)
            if diff == 0:
                break
            if diff < 2:
                raise SystemExit(f'[fix_kd_isa] note pad {diff} unrepresentable')
            key_len = diff - 2
            meta.pop("")
            meta["z" * key_len] = 0
        else:
            raise SystemExit('[fix_kd_isa] note padding failed to converge')
        new_desc = mp_encode(meta)
        if len(new_desc) != len(desc):
            raise SystemExit(f'[fix_kd_isa] cannot pad note exactly: {len(desc)} -> {len(new_desc)}')
    elif len(new_desc) > len(desc):
        raise SystemExit(f'[fix_kd_isa] metadata grew: {len(desc)} -> {len(new_desc)}')
    old_dp = (len(desc) + 3) & ~3
    new_dp = (len(new_desc) + 3) & ~3

    rec_start_abs = sh_off + rec_pos
    rec_len = 12 + npad + old_dp
    d[rec_start_abs:rec_start_abs + rec_len] = \
        d[rec_start_abs:rec_start_abs + 12 + npad] + new_desc + b'\0' * (new_dp - len(new_desc))
    return d, renames


def normalize_and_fix_isa(d: bytearray):
    """Bugs 1+3 in one metadata round-trip. Returns (d, symbol_renames)."""
    loc = find_note(d)
    if loc is None:
        print('[fix_kd_isa] no NT_AMDGPU_METADATA note found')
        return d, []

    def transform(meta):
        renames = []
        # Bug 1: ISA triple
        target = meta.get('amdhsa.target') if isinstance(meta, dict) else None
        if isinstance(target, str):
            m = ISA_RE.search(target.encode())
            if m:
                fixed = 'amdgcn-amd-amdhsa--' + m.group(1).decode()
                print(f'[fix_kd_isa] ISA patched: {target} -> {fixed}')
                meta['amdhsa.target'] = fixed
        # Bug 3: kernel name prefixes
        kernels = meta.get('amdhsa.kernels') or []
        for k in kernels:
            name = k.get('.name')
            sym = k.get('.symbol')
            if isinstance(name, str) and '.' in name:
                plain = name.split('.', 1)[1]
                k['.name'] = plain
                print(f'[fix_kd_isa] kernel .name {name} -> {plain}')
            if isinstance(sym, str) and '.' in sym:
                new_sym = sym.split('.', 1)[1]
                k['.symbol'] = new_sym
                if new_sym != sym:
                    renames.append((sym, new_sym))
                    print(f'[fix_kd_isa] symbol {sym} -> {new_sym}')
        return renames

    return replace_note_section(d, loc, transform)


def require_tool(name: str) -> None:
    from shutil import which
    if which(name) is None:
        raise SystemExit(f'[fix_kd_isa] error: {name} not found in PATH (needed for HSACO patch)')


def globalize_kd(obj_path: str, out_path: str, renames=()):
    """Globalize all .kd symbols and apply module-prefix symbol renames."""
    assert isinstance(renames, list)
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
    for old, new in renames:
        args += ['--redefine-sym', f'{old}={new}']
    args += [obj_path, out_path]
    subprocess.run(args, check=True)
    print(f'[fix_kd_isa] globalized {len(kd_syms)} .kd symbols')


def main():
    inp = sys.argv[1]
    out = sys.argv[2]

    # Step 1+2: one metadata round-trip fixes the ISA triple and strips
    # module prefixes from kernel names; collect .kd renames.
    d = bytearray(open(inp, 'rb').read())
    d, renames = normalize_and_fix_isa(d)

    # Step 3: write patched .o to temp file, then fix up symbols.
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
        tmp.write(d)
        tmp_path = tmp.name

    try:
        globalize_kd(tmp_path, out, renames)
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    main()
