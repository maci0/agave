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

The output is published by atomic rename and the source is never written, so
a rerun after an interrupted run converges on the same file and cannot destroy
the dense input.
"""

import argparse
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_io import T_U32, load, write_gguf


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

    # Writing the output over the input would destroy the only dense copy the
    # rerun depends on, and an interrupted write would leave a truncated GGUF
    # that a rerun reads as a bad source. Refuse in place, publish atomically.
    if args.dst.resolve() == args.src.resolve():
        print("error: --out must differ from --in", file=sys.stderr)
        return 2

    try:
        src = load(args.src)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    kv, tensors, alignment = src.kv, src.tensors, src.alignment
    blob = src.blob

    arch = src.meta(b"general.architecture")
    if arch is None:
        print("error: no general.architecture", file=sys.stderr)
        return 2
    arch = arch.decode()

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

    body = write_gguf(src.version, kv, out_tensors, payloads, alignment)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst.with_name(args.dst.name + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, args.dst)
    print(f"wrote {args.dst} ({len(body) / 2**20:.1f} MB): "
          f"{len(out_tensors)} tensors, {args.experts} experts "
          f"({args.experts_used} used), {len(layers_seen)} MoE layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
