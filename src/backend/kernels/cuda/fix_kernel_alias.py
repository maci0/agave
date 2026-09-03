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
    with open(sys.argv[1], encoding="utf-8") as f:
        ptx = f.read()
    for clean, mangled in re.findall(r'\.alias (\w+_kernel), ([^;]+);', ptx):
        ptx = ptx.replace(f'.func {mangled}(', f'.entry {clean}(')
    ptx = re.sub(r'\.alias \w+_kernel, [^;]+;\n', '', ptx)
    # Drop forward declarations of aliased kernels. Each aliased kernel
    # appears twice: a body-less declaration `.func <clean>` + param list
    # `;` plus the real mangled definition promoted to `.entry` in step 1.
    # Promoting the declaration would create an empty `.entry` stub that
    # shadows the real kernel (TP poisoned with stub-batched FFN).
    # Detect declarations: `.func <name>` followed by `(` on next line and
    # terminating with `;` not `{`.
    lines = ptx.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r'^\.func (\w+_kernel)$', lines[i])
        if m:
            j = i + 1
            while j < n and lines[j].strip() == '':
                j += 1
            if j < n and lines[j].strip().startswith('('):
                k = j
                while k < n and ')' not in lines[k]:
                    k += 1
                t = k
                while t < n and lines[t].strip() == '':
                    t += 1
                if t < n:
                    term = lines[t].strip()
                else:
                    term = ''
                if '{' in lines[k] or term.startswith('{'):
                    out.append('.entry ' + m.group(1))
                    i += 1
                else:
                    while k < n and ';' not in lines[k]:
                        k += 1
                    i = k + 1
                continue
        out.append(lines[i])
        i += 1
    ptx = '\n'.join(out)
    sys.stdout.write(ptx)


if __name__ == '__main__':
    main()

# cache-buster touch

# cache-buster 2
