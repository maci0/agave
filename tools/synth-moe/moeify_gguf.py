#!/usr/bin/env python3
"""Turn a dense GGUF into a routed-MoE GGUF, for testing the MoE code path.

There is no small MoE checkpoint to test against: the supported ones start
around 17 GB. This derives one from a dense model instead, so the MoE forward
path can be exercised, and in particular compared between CPU and GPU, without
downloading anything.

It rewrites each layer's `ffn_{gate,up,down}.weight` into
`ffn_{gate,up,down}_exps.weight` holding `--experts` identical copies, adds an
f32 `ffn_gate_inp.weight` router, and sets `expert_count` / `expert_used_count`.
Everything else, including the tokenizer, is copied through unchanged.

Identical experts make this a test with a KNOWN ANSWER rather than a smoke run.
Softmax over equal router logits gives every expert the same score, so top-k
normalises to 1/k each and the mixture is
    sum_i (1/k) * FFN(x) = FFN(x)
the dense result exactly. So the derived model must reproduce the original
token for token, on every backend. Anything else is a bug in the MoE path.

    python3 moeify_gguf.py --in model.gguf --out moe.gguf --experts 4
    agave moe.gguf --backend rocm -t 0 "..."   # must match model.gguf exactly

"""

import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF metadata value type tags.
(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64,
 T_I64, T_F64) = range(13)

_FIXED = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
          T_I32: "<i", T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q",
          T_F64: "<d"}

# ggml type -> (block size in elements, bytes per block). Only what a dense
# checkpoint's FFN tensors are likely to be; anything else is refused rather
# than guessed at, since a wrong stride would silently corrupt the copy.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
}


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.i = buf, 0

    def take(self, n: int) -> bytes:
        out = self.b[self.i:self.i + n]
        if len(out) != n:
            raise ValueError("truncated GGUF")
        self.i += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u64())

    def value(self, t: int):
        """Read one metadata value, returning it in a form `write_value` accepts."""
        if t in _FIXED:
            return struct.unpack(_FIXED[t], self.take(struct.calcsize(_FIXED[t])))[0]
        if t == T_STR:
            return self.string()
        if t == T_ARR:
            et, n = self.u32(), self.u64()
            return (et, [self.value(et) for _ in range(n)])
        raise ValueError(f"unknown metadata type {t}")


def write_string(out: bytearray, s: bytes) -> None:
    out += struct.pack("<Q", len(s)) + s


def write_value(out: bytearray, t: int, v) -> None:
    if t in _FIXED:
        out += struct.pack(_FIXED[t], v)
    elif t == T_STR:
        write_string(out, v)
    elif t == T_ARR:
        et, items = v
        out += struct.pack("<I", et) + struct.pack("<Q", len(items))
        for it in items:
            write_value(out, et, it)
    else:
        raise ValueError(f"unknown metadata type {t}")


def tensor_bytes(dims, ggml_type: int) -> int:
    name, block_elems, block_bytes = GGML_TYPES[ggml_type]
    n = 1
    for d in dims:
        n *= d
    if n % block_elems:
        raise ValueError(f"{name}: {n} elements is not a whole number of blocks")
    return n // block_elems * block_bytes


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--experts-used", type=int, default=2)
    args = ap.parse_args()

    if args.experts < 1 or args.experts_used < 1 or args.experts_used > args.experts:
        print("error: need 1 <= experts-used <= experts", file=sys.stderr)
        return 2

    raw = args.src.read_bytes()
    r = Reader(raw)
    if r.take(4) != GGUF_MAGIC:
        print("error: not a GGUF file", file=sys.stderr)
        return 2
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    kv = []
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        kv.append((key, t, r.value(t)))

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        offset = r.u64()
        if ttype not in GGML_TYPES:
            print(f"error: tensor {name.decode()} has unsupported ggml type {ttype}",
                  file=sys.stderr)
            return 2
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

    alignment = 32
    for key, t, v in kv:
        if key == b"general.alignment":
            alignment = v
    data_start = align_up(r.i, alignment)

    arch = next((v.decode() for key, t, v in kv if key == b"general.architecture"), None)
    if arch is None:
        print("error: no general.architecture", file=sys.stderr)
        return 2

    def blob(t):
        return raw[data_start + t["offset"]: data_start + t["offset"] + tensor_bytes(t["dims"], t["type"])]

    # Rewrite: ffn_{gate,up,down}.weight -> _exps with `experts` stacked copies,
    # plus an f32 router per layer. The expert dimension is appended, matching
    # llama.cpp's [in, out, n_expert] layout for routed experts.
    out_tensors, payloads = [], []
    n_embd = next((v for key, t, v in kv if key.endswith(b".embedding_length")), None)
    if n_embd is None:
        print("error: no embedding_length", file=sys.stderr)
        return 2

    layers_seen = set()
    for t in tensors:
        name = t["name"].decode()
        parts = name.split(".")
        is_ffn = (len(parts) == 4 and parts[0] == "blk"
                  and parts[2] in ("ffn_gate", "ffn_up", "ffn_down")
                  and parts[3] == "weight")
        if not is_ffn:
            out_tensors.append(dict(t))
            payloads.append(blob(t))
            continue

        layer = int(parts[1])
        layers_seen.add(layer)
        out_tensors.append({"name": f"blk.{layer}.{parts[2]}_exps.weight".encode(),
                            "dims": t["dims"] + [args.experts], "type": t["type"]})
        payloads.append(blob(t) * args.experts)

    # One router per layer that had an FFN. Uniform weights: with identical
    # experts the routing cannot change the result, which is the point.
    for layer in sorted(layers_seen):
        router = struct.pack("<f", 0.0) * (n_embd * args.experts)
        out_tensors.append({"name": f"blk.{layer}.ffn_gate_inp.weight".encode(),
                            "dims": [n_embd, args.experts], "type": 0})
        payloads.append(router)

    # expert_feed_forward_length is not optional in practice: without it the
    # loader falls back to a small default and every expert GEMV runs at the
    # wrong width.
    ff_dim = next((v for key, t, v in kv if key == f"{arch}.feed_forward_length".encode()), None)
    if ff_dim is None:
        print("error: no feed_forward_length to derive expert_feed_forward_length from",
              file=sys.stderr)
        return 2

    drop = {f"{arch}.expert_count".encode(), f"{arch}.expert_used_count".encode(),
            f"{arch}.expert_feed_forward_length".encode()}
    kv = [(k, t, v) for (k, t, v) in kv if k not in drop]
    kv.append((f"{arch}.expert_count".encode(), T_U32, args.experts))
    kv.append((f"{arch}.expert_used_count".encode(), T_U32, args.experts_used))
    kv.append((f"{arch}.expert_feed_forward_length".encode(), T_U32, ff_dim))

    head = bytearray(GGUF_MAGIC)
    head += struct.pack("<I", version)
    head += struct.pack("<Q", len(out_tensors))
    head += struct.pack("<Q", len(kv))
    for key, t, v in kv:
        write_string(head, key)
        head += struct.pack("<I", t)
        write_value(head, t, v)

    # Tensor offsets are relative to the data section, which starts after the
    # table; the table's size depends on the offsets only through their fixed
    # width, so one pass suffices.
    table = bytearray()
    off = 0
    for t, p in zip(out_tensors, payloads):
        write_string(table, t["name"])
        table += struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            table += struct.pack("<Q", d)
        table += struct.pack("<I", t["type"])
        table += struct.pack("<Q", off)
        off = align_up(off + len(p), alignment)

    body = bytearray(head + table)
    pad = align_up(len(body), alignment) - len(body)
    body += b"\0" * pad
    base = len(body)
    for p in payloads:
        body += p
        # Pad each tensor to the alignment the offsets above assumed.
        body += b"\0" * (align_up(len(body) - base, alignment) - (len(body) - base))

    args.dst.write_bytes(bytes(body))
    print(f"wrote {args.dst} ({len(body) / 2**20:.1f} MB): "
          f"{len(out_tensors)} tensors, {args.experts} experts "
          f"({args.experts_used} used), {len(layers_seen)} MoE layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
