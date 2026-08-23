#!/usr/bin/env python3
r"""
Post-process Zig 0.16 NVPTX assembly to work around the LLVM aliasee bug.

callconv(.kernel) makes LLVM NVPTX reject aliases to kernel functions, so
kernels use callconv(.nvptx_device), which emits `.func` (device function).
This script promotes the aliased `*_kernel` functions to `.entry` and drops
the `.alias` directives:

1. For every `.alias <clean>, <mangled>;`, rewrite `.func <mangled>(` as
   `.entry <clean>(`.
2. Remove all remaining ``.alias \w+_kernel, ...;`` lines.
3. Promote any bare `.func <name>_kernel` declarations to `.entry`.

Reads the PTX path from argv[1] and writes the fixed PTX to stdout.

Wired into `zig build ptx`; keep in sync with the ROCm equivalent
(src/backend/kernels/rocm/fix_kd_isa.py) when rebasing on new Zig releases.
"""
import re
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <input.ptx>")
    with open(sys.argv[1]) as f:
        ptx = f.read()
    for clean, mangled in re.findall(r'\.alias (\w+_kernel), ([^;]+);', ptx):
        ptx = ptx.replace(f'.func {mangled}(', f'.entry {clean}(')
    ptx = re.sub(r'\.alias \w+_kernel, [^;]+;\n', '', ptx)
    ptx = re.sub(r'^\.func (\w+_kernel)$', r'.entry \1', ptx, flags=re.MULTILINE)
    sys.stdout.write(ptx)


if __name__ == '__main__':
    main()

# cache-buster touch

# cache-buster 2
