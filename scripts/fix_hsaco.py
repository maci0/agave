#!/usr/bin/env python3
"""Fix Zig 0.16 amdgcn HSACO metadata for ROCm 7.x compatibility.

Zig generates malformed amdhsa.target triple: 'amdhsa5.0.0-unknown-gfx1100'
instead of 'amdhsa--gfx1100'. This script patches the msgpack metadata
in the .o file to fix the target triple, then relinks to HSACO.

Usage:
    zig build amdgcn
    python3 scripts/fix_hsaco.py zig-out/rocm/kernels.o zig-out/rocm/kernels.hsaco
    cp zig-out/rocm/kernels.hsaco src/backend/kernels/rocm/kernels.hsaco
"""
import struct, sys, os

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <input.o> <output.hsaco>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]
data = bytearray(open(input_path, 'rb').read())

# Find the malformed target in msgpack (str8 format: \xd9 + len_byte + string)
bad = b'amdhsa5.0.0-unknown-'
idx = data.find(bad)
if idx < 0:
    print("Target triple already correct or not found — linking as-is")
else:
    good = b'amdhsa--'
    # The str8 length byte is at idx-1 (part of the msgpack str8 encoding)
    # Original: \xd9 \x26 (38 bytes) 'amdgcn-amd-amdhsa5.0.0-unknown-gfxNNNN'
    # Fixed:    \xd9 \x1a (26 bytes) 'amdgcn-amd-amdhsa--gfxNNNN'
    str_start = idx - len(b'amdgcn-amd-')  # start of 'amdgcn-amd-amdhsa...'
    len_byte_off = str_start - 1
    old_len = data[len_byte_off]
    new_len = old_len - (len(bad) - len(good))
    data[len_byte_off] = new_len

    # Replace the string content (changes file size)
    data = data[:idx] + good + data[idx+len(bad):]

    # Fix the ELF section header offset (e_shoff) since we shortened the file
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    new_shoff = e_shoff - (len(bad) - len(good))
    struct.pack_into('<Q', data, 40, new_shoff)

    # Fix .note descsz (4 bytes at the note header)
    # Find the AMDGPU note header
    amdgpu_idx = data.find(b'AMDGPU\x00')
    if amdgpu_idx > 12:
        for candidate in range(max(0, amdgpu_idx - 20), amdgpu_idx):
            namesz = struct.unpack_from('<I', data, candidate)[0]
            if namesz == 7:
                old_descsz = struct.unpack_from('<I', data, candidate + 4)[0]
                new_descsz = old_descsz - (len(bad) - len(good))
                struct.pack_into('<I', data, candidate + 4, new_descsz)
                print(f"Fixed target triple: len {old_len}->{new_len}, descsz {old_descsz}->{new_descsz}")
                break

    # Also fix any section headers that point past the patched region
    # The .note section's sh_size needs updating too
    e_shoff_new = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    for i in range(e_shnum):
        sh_off = e_shoff_new + i * e_shentsize
        if sh_off + e_shentsize > len(data):
            break
        sh_offset = struct.unpack_from('<Q', data, sh_off + 24)[0]
        sh_size = struct.unpack_from('<Q', data, sh_off + 32)[0]
        # If this section starts at or after the patch point, shift its offset
        if sh_offset > idx:
            struct.pack_into('<Q', data, sh_off + 24, sh_offset - (len(bad) - len(good)))

    # Write fixed .o
    fixed_o = input_path + '.fixed'
    open(fixed_o, 'wb').write(data)
    input_path = fixed_o

# Link to HSACO
import subprocess
subprocess.run(['ld.lld', '-shared', '-o', output_path, input_path], check=True)
print(f"HSACO written to {output_path}")
