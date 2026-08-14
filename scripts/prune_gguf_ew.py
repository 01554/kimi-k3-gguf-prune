#!/usr/bin/env python3
"""Combined expert + width pruning of a GGUF in one pass — no requantization.

  scripts/prune_gguf_ew.py prune --src <first-shard.gguf> --plan plan_ew.json \
      --out pruned.gguf

Extends prune_gguf.py's expert-slab slicing with intra-expert width cuts at
quant-superblock (256-element) granularity, the trick mmnga's width builds
demonstrated: cut boundaries never split a quant block, so everything stays a
byte copy.

plan_ew.json:
  {"num_experts": 512, "n_ff_exp": 2048, "block": 256,
   "layers": {"<blk>": {"keep": [expert ids...],
                         "wblocks": {"<expert id>": [block ids...]}}}}
keep counts and wblock counts must be uniform (GGUF has one global
expert_count / expert_feed_forward_length).

Per-tensor geometry (ne is innermost-first):
  gate/up [hidden, n_ff, n_expert]: a width block is a contiguous run of
      `block` rows inside each expert slab -> slice whole byte ranges.
  down   [n_ff, hidden, n_expert]: a width block is a `block`-element span
      inside every row -> vectorized gather of spans, per expert.
  router [hidden, n_expert] and exp_probs_b [n_expert]: expert axis only,
      renumbered to keep order (the classic trap).
KV expert_count and expert_feed_forward_length are rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prune_gguf import (  # noqa: E402
    BIAS_RE, EXPERT_RE, ROUTER_RE, _byte_shape, _copy_kv, _shards,
)
from gguf import GGUFReader, GGUFWriter  # noqa: E402

DOWN_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")
GATEUP_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up)_exps\.weight$")


def _load_plan_ew(path: Path):
    raw = json.load(open(path))
    n_ff = int(raw["n_ff_exp"])
    block = int(raw["block"])
    assert n_ff % block == 0
    layers = {}
    n_keep_set, n_wb_set = set(), set()
    for k, v in raw["layers"].items():
        keep = np.asarray(v["keep"], dtype=np.int64)
        wb = np.asarray([v["wblocks"][str(e)] for e in v["keep"]], dtype=np.int64)
        if len(np.unique(keep)) != len(keep):
            raise SystemExit(f"layer {k}: duplicate expert ids")
        layers[int(k)] = (keep, wb)
        n_keep_set.add(len(keep))
        n_wb_set.add(wb.shape[1])
    if len(n_keep_set) != 1 or len(n_wb_set) != 1:
        raise SystemExit("non-uniform keep or wblock counts across layers")
    return layers, n_keep_set.pop(), n_wb_set.pop(), n_ff, block


def cmd_prune(args):
    shards = _shards(Path(args.src))
    readers = [GGUFReader(p) for p in shards]
    r0 = readers[0]
    arch = r0.fields["general.architecture"].contents()
    n_expert = int(r0.fields[f"{arch}.expert_count"].contents())
    layers, n_keep, n_wb, n_ff, block = _load_plan_ew(Path(args.plan))
    n_blocks = n_ff // block
    n_ff_new = n_wb * block
    print(f"arch={arch}  experts {n_expert}->{n_keep}  n_ff {n_ff}->{n_ff_new}")

    writer = GGUFWriter(args.out, arch)
    _copy_kv(r0, writer, overrides={f"{arch}.expert_count": None,
                                    f"{arch}.expert_feed_forward_length": None,
                                    "general.architecture": None})
    writer.add_uint32(f"{arch}.expert_count", n_keep)
    writer.add_uint32(f"{arch}.expert_feed_forward_length", n_ff_new)

    def emit(t):
        """-> (materialize() -> array, byte_shape, out_nbytes, dtype)"""
        name = t.name
        m_gu = GATEUP_RE.match(name)
        m_dn = DOWN_RE.match(name)
        m_rt = ROUTER_RE.match(name)
        m_b = BIAS_RE.match(name)
        ne = [int(d) for d in t.shape]

        if m_gu or m_dn:
            layer = int((m_gu or m_dn).group(1))
            if layer not in layers:
                raise SystemExit(f"{name}: layer {layer} missing from plan")
            keep, wb = layers[layer]
            if ne[-1] != n_expert:
                raise SystemExit(f"{name}: expert axis mismatch ne={ne}")
            slab = int(t.n_bytes) // n_expert
            if m_gu:
                # slab = n_ff rows of row_bytes; width blocks are row runs
                if ne[1] != n_ff:
                    raise SystemExit(f"{name}: n_ff mismatch ne={ne}")
                if slab % n_blocks:
                    raise SystemExit(f"{name}: slab {slab}B not divisible by {n_blocks} blocks")
                blk_bytes = slab // n_blocks
                out_bytes_per = blk_bytes * n_wb
                shape = _byte_shape(t, [n_expert, ne[1]] + [ne[0]])
                # after cuts: [n_keep, n_ff_new, row_bytes]
                shape = [n_keep, n_ff_new, shape[2]]

                def mat(t=t, keep=keep, wb=wb, slab=slab, blk_bytes=blk_bytes):
                    d = np.asarray(t.data).reshape(-1).view(np.uint8)
                    d = d.reshape(n_expert, n_blocks, blk_bytes)[keep]
                    return np.ascontiguousarray(
                        np.take_along_axis(d, wb[:, :, None], axis=1)).reshape(-1)
                return mat, shape, out_bytes_per * n_keep, np.uint8
            else:
                # slab = hidden rows, each row = n_blocks spans of span_bytes
                if ne[0] != n_ff:
                    raise SystemExit(f"{name}: n_ff mismatch ne={ne}")
                rows = ne[1]
                if slab % (rows * n_blocks):
                    raise SystemExit(f"{name}: slab {slab}B not divisible into row spans")
                span = slab // rows // n_blocks
                shape = [n_keep, rows, span * n_wb]

                def mat(t=t, keep=keep, wb=wb, rows=rows, span=span):
                    d = np.asarray(t.data).reshape(-1).view(np.uint8)
                    d = d.reshape(n_expert, rows, n_blocks, span)[keep]
                    return np.ascontiguousarray(
                        np.take_along_axis(d, wb[:, None, :, None], axis=2)).reshape(-1)
                return mat, shape, span * n_wb * rows * n_keep, np.uint8

        if m_rt:
            keep, _ = layers[int(m_rt.group(1))]
            slab = int(t.n_bytes) // n_expert
            shape = _byte_shape(t, [n_expert, ne[0]])
            shape[0] = n_keep

            def mat(t=t, keep=keep, slab=slab):
                return np.asarray(t.data).reshape(-1).view(np.uint8) \
                         .reshape(n_expert, slab)[keep].copy()
            return mat, shape, slab * n_keep, np.uint8

        if m_b:
            keep, _ = layers[int(m_b.group(1))]
            dt = np.asarray(t.data).dtype

            def mat(t=t, keep=keep):
                return np.asarray(t.data).reshape(n_expert)[keep].copy()
            return mat, [n_keep], n_keep * dt.itemsize, dt

        def mat(t=t):
            return np.asarray(t.data).reshape(-1).view(np.uint8)
        return mat, _byte_shape(t, ne[::-1]), int(t.n_bytes), np.uint8

    entries = []
    total_in = total_out = 0
    for reader in readers:
        for t in reader.tensors:
            total_in += int(t.n_bytes)
            mat, shape, nbytes, dtype = emit(t)
            writer.add_tensor_info(t.name, shape, np.dtype(dtype), nbytes,
                                   raw_dtype=t.tensor_type)
            entries.append((t.name, mat, nbytes))
            total_out += nbytes

    print(f"writing {total_out/1e9:.1f} GB (from {total_in/1e9:.1f} GB) -> {args.out}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done = 0
    for i, (name, mat, nbytes) in enumerate(entries):
        writer.write_tensor_data(mat())
        done += nbytes
        if i % 100 == 0 or i == len(entries) - 1:
            print(f"  [{i+1:>5}/{len(entries)}] {done/1e9:>6.1f} / {total_out/1e9:.1f} GB", flush=True)
    writer.close()
    print("done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--src", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prune)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
