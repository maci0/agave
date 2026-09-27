"""Shared GGUF header parsing and writing for the tools/ scripts.

Only the header is modelled: the metadata key/value block, the tensor table,
and the byte length of each tensor's data. Tensor payloads stay in the file
they came from and are handed to `write_gguf` as bytes.
"""

import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF metadata value type tags.
(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64,
 T_I64, T_F64) = range(13)

_FIXED = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
          T_I32: "<i", T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q",
          T_F64: "<d"}

# ggml type -> (block size in elements, bytes per block). Quant types whose
# block size this table gets wrong would silently mis-size every copy, so an
# unknown type raises rather than being guessed at.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    31: ("TQ1_0", 256, 54), 32: ("TQ2_0", 256, 66), 33: ("MXFP4", 32, 17),
}

DEFAULT_ALIGNMENT = 32


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


def type_name(ggml_type: int) -> str:
    if ggml_type not in GGML_TYPES:
        raise ValueError(f"unsupported ggml type {ggml_type}")
    return GGML_TYPES[ggml_type][0]


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


class Gguf:
    """A parsed GGUF file: its raw bytes plus the header split into parts."""

    def __init__(self, raw: bytes, path: Path):
        self.raw, self.path = raw, path
        r = Reader(raw)
        if r.take(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        self.version = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()

        self.kv = []
        for _ in range(n_kv):
            key = r.string()
            t = r.u32()
            self.kv.append((key, t, r.value(t)))

        self.tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            self.tensors.append({"name": name, "dims": dims, "type": ttype,
                                 "offset": r.u64()})
            if ttype not in GGML_TYPES:
                raise ValueError(
                    f"{path}: tensor {name.decode()} has unsupported ggml type {ttype}")

        self.alignment = DEFAULT_ALIGNMENT
        for key, t, v in self.kv:
            if key == b"general.alignment":
                self.alignment = v
        self.data_start = align_up(r.i, self.alignment)

    def meta(self, key: bytes):
        return next((v for k, t, v in self.kv if k == key), None)

    def by_name(self) -> dict:
        return {t["name"]: t for t in self.tensors}

    def blob(self, t) -> bytes:
        start = self.data_start + t["offset"]
        return self.raw[start:start + tensor_bytes(t["dims"], t["type"])]


def load(path: Path) -> Gguf:
    return Gguf(path.read_bytes(), path)


def write_gguf(version: int, kv, tensors, payloads, alignment: int) -> bytes:
    """Serialize a complete GGUF: header, tensor table, then padded payloads."""
    head = bytearray(GGUF_MAGIC)
    head += struct.pack("<I", version)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", len(kv))
    for key, t, v in kv:
        write_string(head, key)
        head += struct.pack("<I", t)
        write_value(head, t, v)

    # Offsets are relative to the data section, which starts after the table.
    # The table's size depends on the offsets only through their fixed width,
    # so one pass suffices.
    table = bytearray()
    off = 0
    for t, p in zip(tensors, payloads):
        write_string(table, t["name"])
        table += struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            table += struct.pack("<Q", d)
        table += struct.pack("<I", t["type"])
        table += struct.pack("<Q", off)
        off = align_up(off + len(p), alignment)

    body = bytearray(head + table)
    body += b"\0" * (align_up(len(body), alignment) - len(body))
    base = len(body)
    for p in payloads:
        body += p
        body += b"\0" * (align_up(len(body) - base, alignment) - (len(body) - base))
    return bytes(body)
