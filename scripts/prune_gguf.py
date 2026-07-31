#!/usr/bin/env python3
"""REAP-prune a Kimi-K3 GGUF by slicing expert slabs — no requantization.

  scripts/prune_gguf.py inspect  --src <first-shard.gguf>
  scripts/prune_gguf.py prune    --src <first-shard.gguf> --plan reap_plan.json \
      --out pruned.gguf

Why slicing is lossless here: GGUF stores dims innermost-first, so a logical
[n_expert, rows, cols] expert tensor has the expert axis outermost in memory —
one contiguous byte slab per expert, and quantization blocks (which run along
the innermost axis) never cross a slab boundary. Keeping a subset of experts is
therefore a pure byte copy of the surviving slabs; the quantized codes are
untouched and no new quantization error is introduced. This is the same reason
kimi-k3-mlx's converter can pass MXFP4 through bit-exact.

The trap (same one kimi-k3-mlx documents): pruning renumbers experts. The
router matrix (`ffn_gate_inp.weight`) and the routing bias (`exp_probs_b`) must
be sliced *in keep order* so new row i is old expert keep[i]. Get that wrong and
the model runs fluently while routing every token to the wrong expert — no
shape check can see it, so tests/test_prune.py pins the behaviour instead.

The keep plan is kimi-k3-mlx's reap_plan.json. Only uniform plans are accepted:
GGUF carries a single global expert_count, so every layer must keep the same
number of experts (order may differ per layer).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

# Tensors carrying one slab per routed expert, expert axis outermost.
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:up|gate|down)_exps\.weight$")
ROUTER_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
BIAS_RE = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")

SPLIT_KEYS = {"split.no", "split.count", "split.tensors.count"}


def _shards(first: Path) -> list[Path]:
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    stem, _, total = m.groups()
    return [first.parent / f"{stem}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]


def _load_plan(path: Path, n_layers_hint: int | None = None) -> dict[int, np.ndarray]:
    """-> {moe_layer_index: keep_ids in keep order}.

    Accepts kimi-k3-mlx's reap_plan.json (`layers[str(L)] = {"keep": [...]}`,
    layer indices matching model.layers.N == blk.N) or a bare
    `{"keep": {layer: [...]}}` mapping as the tests use.
    """
    raw = json.load(open(path))
    if "layers" in raw:
        keep = {k: v["keep"] for k, v in raw["layers"].items()}
    else:
        keep = raw.get("keep", raw)
    if isinstance(keep, list):
        plan = {i: np.asarray(v, dtype=np.int64) for i, v in enumerate(keep)}
    else:
        plan = {int(k): np.asarray(v, dtype=np.int64) for k, v in keep.items()}
    sizes = {len(v) for v in plan.values()}
    if len(sizes) != 1:
        raise SystemExit(
            f"non-uniform plan (layer keep sizes {sorted(sizes)}); GGUF needs one "
            "global expert_count — regenerate with reap_plan.py --mode uniform"
        )
    for l, ids in plan.items():
        if len(np.unique(ids)) != len(ids):
            raise SystemExit(f"layer {l}: duplicate expert ids in keep list")
    return plan


def _copy_kv(reader: GGUFReader, writer: GGUFWriter, overrides: dict) -> None:
    for field in reader.fields.values():
        if field.name in SPLIT_KEYS or field.name.startswith("GGUF."):
            continue
        if field.name in overrides:
            continue
        val = field.contents()
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_array(field.name, val)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(field.name, val)
        elif vtype == GGUFValueType.BOOL:
            writer.add_bool(field.name, val)
        elif vtype == GGUFValueType.FLOAT32:
            writer.add_float32(field.name, val)
        elif vtype == GGUFValueType.FLOAT64:
            writer.add_float64(field.name, val)
        elif vtype in (GGUFValueType.UINT8, GGUFValueType.UINT16, GGUFValueType.UINT32):
            writer.add_uint32(field.name, val)
        elif vtype == GGUFValueType.UINT64:
            writer.add_uint64(field.name, val)
        elif vtype in (GGUFValueType.INT8, GGUFValueType.INT16, GGUFValueType.INT32):
            writer.add_int32(field.name, val)
        elif vtype == GGUFValueType.INT64:
            writer.add_int64(field.name, val)
        else:
            raise SystemExit(f"unhandled KV type {vtype} for {field.name}")


def _byte_shape(t, logical: list[int]) -> list[int]:
    """gguf-py's add_tensor convention for raw (pre-quantized) data: shape is in
    logical (numpy) order and the *innermost* dimension is given in bytes; the
    writer converts it back to elements via the quant type's block geometry
    (`quant_shape_from_byte_shape`). Quantization blocks never span rows, so the
    innermost byte count is always an integer."""
    outer = 1
    for d in logical[:-1]:
        outer *= d
    nbytes = int(t.n_bytes)
    if nbytes % outer:
        raise SystemExit(f"{t.name}: {nbytes} bytes not divisible by outer dims {logical[:-1]}")
    return logical[:-1] + [nbytes // outer]


def _slab_slice(t, keep: np.ndarray, n_expert: int) -> tuple[np.ndarray, list[int]]:
    """Slice expert slabs out of a (possibly quantized) tensor's raw bytes.

    Returns (uint8 data, byte-shape) where byte-shape is the pruned logical
    shape with the innermost dim in bytes, as _byte_shape describes.
    """
    ne = [int(d) for d in t.shape]  # gguf ne order, innermost first
    if ne[-1] != n_expert:
        raise SystemExit(f"{t.name}: expected expert axis last in ne={ne}")
    data = np.asarray(t.data).reshape(-1).view(np.uint8)
    if data.nbytes % n_expert:
        raise SystemExit(f"{t.name}: {data.nbytes} bytes not divisible by {n_expert} experts")
    out = data.reshape(n_expert, data.nbytes // n_expert)[keep].copy()
    logical_full = ne[::-1]                      # [n_expert, ..., innermost]
    shape = _byte_shape(t, logical_full)          # innermost -> bytes
    shape[0] = len(keep)
    return out, shape


def cmd_inspect(args):
    # Split GGUFs keep all KV in the first shard; later shards carry only
    # tensors. Always resolve the full shard list and read KV from shard 1.
    shards = _shards(Path(args.src))
    r0 = GGUFReader(shards[0])
    arch = r0.fields["general.architecture"].contents()
    print(f"arch: {arch}   shards: {len(shards)}")
    for name in sorted(r0.fields):
        if "expert" in name or name.startswith("split."):
            print(f"  kv {name} = {r0.fields[name].contents()}")
    shown: set[str] = set()
    total = 0
    for path in shards:
        if not path.exists():
            print(f"  (missing shard: {path.name})")
            continue
        reader = r0 if path == shards[0] else GGUFReader(path)
        total += len(reader.tensors)
        for t in reader.tensors:
            for p in (EXPERT_RE, ROUTER_RE, BIAS_RE):
                if p.match(t.name) and p.pattern not in shown:
                    print(f"  tensor {t.name:<38} type={t.tensor_type.name:<8} ne={[int(d) for d in t.shape]}")
                    shown.add(p.pattern)
        if len(shown) == 3 and path != shards[0]:
            break
    print(f"tensors seen: {total}")


def cmd_prune(args):
    shards = _shards(Path(args.src))
    readers = [GGUFReader(p) for p in shards]
    r0 = readers[0]
    arch = r0.fields["general.architecture"].contents()
    n_expert = int(r0.fields[f"{arch}.expert_count"].contents())
    plan = _load_plan(Path(args.plan))
    n_keep = len(next(iter(plan.values())))
    print(f"arch={arch}  experts {n_expert} -> {n_keep}  shards={len(shards)}")

    writer = GGUFWriter(args.out, arch)
    _copy_kv(r0, writer, overrides={f"{arch}.expert_count": None,
                                    "general.architecture": None})
    writer.add_uint32(f"{arch}.expert_count", n_keep)

    total_in = total_out = 0
    for reader in readers:
        for t in reader.tensors:
            total_in += int(t.n_bytes)
            m = EXPERT_RE.match(t.name) or ROUTER_RE.match(t.name)
            b = BIAS_RE.match(t.name)
            if m:
                layer = int(m.group(1))
                keep = plan.get(layer)
                if keep is None:
                    raise SystemExit(f"{t.name}: layer {layer} missing from plan")
                out, byte_shape = _slab_slice(t, keep, n_expert)
                writer.add_tensor(t.name, out, raw_shape=byte_shape,
                                  raw_dtype=t.tensor_type)
                total_out += out.nbytes
            elif b:
                layer = int(b.group(1))
                keep = plan[layer]
                vals = np.asarray(t.data).reshape(n_expert)[keep].copy()
                # float-typed numpy data: shape stays in elements (the byte
                # conversion in gguf-py only applies to uint8 input)
                writer.add_tensor(t.name, vals, raw_shape=[len(keep)],
                                  raw_dtype=t.tensor_type)
                total_out += vals.nbytes
            else:
                raw = np.asarray(t.data).reshape(-1).view(np.uint8)
                writer.add_tensor(t.name, raw,
                                  raw_shape=_byte_shape(t, [int(d) for d in t.shape][::-1]),
                                  raw_dtype=t.tensor_type)
                total_out += int(t.n_bytes)

    print(f"writing {total_out/1e9:.1f} GB (from {total_in/1e9:.1f} GB) -> {args.out}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print("done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--src", required=True); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("prune")
    p.add_argument("--src", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prune)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
