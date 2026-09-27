#!/usr/bin/env python3
"""Unit tests for mixed-quant splicing: two synthetic GGUFs, no model needed."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_io import Gguf, T_U32, load, write_gguf
from splice_mixed_experts import parse_layer_ranges, splice

# 256 elements: one block of every quant type the tests use.
BLOCKS = 256
FFN = [BLOCKS, 8]


def make_gguf(path: Path, expert_type: int, expert_byte: int,
              ffn_dims: list[int] | None = None) -> None:
    """A two-layer model: F32 everywhere except the expert tensors."""
    ffn_dims = ffn_dims or FFN
    kv = [(b"general.architecture", 8, b"qwen35"), (b"qwen35.block_count", T_U32, 2)]
    tensors = [
        {"name": b"blk.0.ffn_gate_exps.weight", "dims": ffn_dims, "type": expert_type},
        {"name": b"blk.0.attn_q.weight", "dims": [8, 8], "type": 0},
        {"name": b"blk.0.shared_expert.weight", "dims": [8, 8], "type": 0},
        {"name": b"blk.1.ffn_gate_exps.weight", "dims": ffn_dims, "type": expert_type},
        {"name": b"blk.1.attn_q.weight", "dims": [8, 8], "type": 0},
    ]
    n = ffn_dims[0] * ffn_dims[1] // BLOCKS * 72
    payloads = [
        bytes([expert_byte]) * n, b"\x01" * 256, b"\x07" * 256,
        bytes([expert_byte]) * n, b"\x02" * 256,
    ]
    path.write_bytes(write_gguf(3, kv, tensors, payloads, 32))


class SpliceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.base = self.dir / "base.gguf"
        self.donor = self.dir / "donor.gguf"
        make_gguf(self.base, 0, 0xAA)  # F32 experts
        make_gguf(self.donor, 10, 0x55)  # Q2_K experts

    def test_parse_layer_ranges(self) -> None:
        self.assertEqual(parse_layer_ranges("3"), [3])
        self.assertEqual(parse_layer_ranges("1-3"), [1, 2, 3])
        self.assertEqual(parse_layer_ranges("0-1,4"), [0, 1, 4])

    def test_only_targeted_layer_experts_change(self) -> None:
        tensors, payloads, spliced = splice(load(self.base), load(self.donor), {1})
        out = self.dir / "mixed.gguf"
        out.write_bytes(write_gguf(3, load(self.base).kv, tensors, payloads, 32))
        by_name = {t["name"].decode(): t for t in Gguf(out.read_bytes(), out).tensors}
        result = Gguf(out.read_bytes(), out)

        self.assertEqual([name for name, *_ in spliced], ["blk.1.ffn_gate_exps.weight"])
        self.assertEqual(by_name["blk.1.ffn_gate_exps.weight"]["type"], 10)
        self.assertEqual(result.blob(by_name["blk.1.ffn_gate_exps.weight"])[0], 0x55)
        self.assertEqual(by_name["blk.0.ffn_gate_exps.weight"]["type"], 0)
        self.assertEqual(result.blob(by_name["blk.0.ffn_gate_exps.weight"])[0], 0xAA)
        # Non-expert tensors keep the base bytes even in a spliced layer.
        self.assertEqual(result.blob(by_name["blk.1.attn_q.weight"]),
                         b"\x02" * 256)
        self.assertEqual(result.blob(by_name["blk.0.shared_expert.weight"]),
                         b"\x07" * 256)

    def test_dim_mismatch_is_refused(self) -> None:
        make_gguf(self.donor, 10, 0x55, ffn_dims=[BLOCKS, 4])
        with self.assertRaisesRegex(ValueError, "differ"):
            splice(load(self.base), load(self.donor), {1})


if __name__ == "__main__":
    unittest.main()
