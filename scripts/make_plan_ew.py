#!/usr/bin/env python3
"""Build a combined expert+width plan (plan_ew.json for prune_gguf_ew.py).

  scripts/make_plan_ew.py --hotness out/qwen38_hotness.csv \
      --imatrix out/qwen38_resident.imatrix --keep 304 --wblocks 6 \
      --out out/plan_ew_qwen_304x6.json

Expert selection: top-`keep` per layer by router selection counts (same
count-based proxy as hotness_to_plan.py — see its disclosure note).

Width selection: the imatrix's per-expert ffn_down_exps input stats are the
squared activations of the intermediate channels (what MUL_MAT_ID collected
per expert). For each kept expert, keep the `wblocks` 256-channel superblocks
with the largest activation-energy sums. Reported "width coverage" = share of
each expert's activation energy retained by the kept blocks.

Experts that never fired during the imatrix run (zero counts) have no width
signal; they fall back to the first `wblocks` blocks and are counted in the
summary (should be ~0 among kept experts — kept experts are the hot ones).
"""

import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np
from gguf import GGUFReader

SUM2_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight\.in_sum2$")


def load_hotness(path):
    counts = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[int(row["layer"])][int(row["expert"])] = int(row["count"])
    layers = sorted(counts)
    n_expert = max(max(c) for c in counts.values()) + 1
    mat = np.zeros((len(layers), n_expert), dtype=np.float64)
    for i, l in enumerate(layers):
        for e, c in counts[l].items():
            mat[i, e] = c
    return layers, mat


def load_width_energy(path, block):
    """-> {layer: [n_expert, n_blocks] block energy}, n_ff"""
    r = GGUFReader(path)
    out, n_ff = {}, None
    for t in r.tensors:
        m = SUM2_RE.match(t.name)
        if not m:
            continue
        d = np.asarray(t.data, dtype=np.float64)
        n_ff = int(t.shape[0])  # ne: innermost = channels
        d = d.reshape(-1, n_ff)  # (n_expert, n_ff)
        out[int(m.group(1))] = d.reshape(d.shape[0], n_ff // block, block).sum(axis=2)
    if not out:
        raise SystemExit(f"{path}: no ffn_down_exps.weight.in_sum2 tensors "
                         "(streamed imatrix? needs a resident run)")
    return out, n_ff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotness", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--wblocks", type=int, required=True)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers, hot = load_hotness(args.hotness)
    energy, n_ff = load_width_energy(args.imatrix, args.block)
    n_expert = hot.shape[1]
    n_blocks = n_ff // args.block
    # Stored-but-unused layers (e.g. an MTP/nextn block, like Qwen3.8's
    # blk.92) never run, so they have zero hotness and no imatrix entry —
    # but they must still be sliced to the uniform shape or the file won't
    # load. A hot layer missing from the imatrix is a real error.
    missing = sorted(set(layers) - set(energy))
    hot_missing = [l for l in missing if hot[layers.index(l)].sum() > 0]
    if hot_missing:
        raise SystemExit(f"imatrix missing HOT MoE layers: {hot_missing}")
    if missing:
        print(f"unused (zero-hotness) layers sliced blind: {missing}")
        n_ff_im = n_ff
        for l in missing:
            energy[l] = np.zeros((n_expert, n_ff_im // args.block))

    plan = {"num_experts": n_expert, "n_ff_exp": n_ff, "block": args.block,
            "layers": {}}
    exp_cov, w_cov, dead = [], [], 0
    drop_hist = np.zeros(n_blocks)
    for i, l in enumerate(layers):
        row = hot[i]
        keep = np.sort(np.argsort(row)[::-1][: args.keep])
        if row.sum() > 0:
            exp_cov.append(row[keep].sum() / row.sum())
        en = energy[l]
        assert en.shape == (n_expert, n_blocks), (l, en.shape)
        wblocks = {}
        for e in keep:
            if en[e].sum() == 0:
                dead += 1
                wb = list(range(args.wblocks))
            else:
                wb = sorted(np.argsort(en[e])[::-1][: args.wblocks].tolist())
                w_cov.append(en[e][wb].sum() / en[e].sum())
            for b in range(n_blocks):
                if b not in wb:
                    drop_hist[b] += 1
            wblocks[str(int(e))] = [int(b) for b in wb]
        plan["layers"][str(l)] = {"keep": [int(e) for e in keep],
                                  "wblocks": wblocks}

    json.dump(plan, open(args.out, "w"))
    print(f"wrote {args.out}")
    print(f"experts {n_expert}->{args.keep}/layer, width {n_ff}->{args.wblocks * args.block}")
    print(f"expert count-coverage: mean {np.mean(exp_cov)*100:.2f}%  "
          f"min {np.min(exp_cov)*100:.2f}%")
    print(f"width energy-coverage: mean {np.mean(w_cov)*100:.2f}%  "
          f"min {np.min(w_cov)*100:.2f}%  (over {len(w_cov)} kept experts)")
    print(f"zero-signal kept experts (fallback blocks): {dead}")
    print(f"dropped-block histogram (block: times dropped): "
          f"{dict(enumerate(drop_hist.astype(int).tolist()))}")


if __name__ == "__main__":
    main()
