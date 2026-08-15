#!/usr/bin/env python3
"""Turn a moe-stream route-hotness dump into a uniform keep plan.

  scripts/hotness_to_plan.py --csv out/qwen38_hotness.csv \
      [--keep N] [--out out/reap_plan_qwen_N.json] [--size-per-expert-gb G]

Without --keep, prints the count-coverage curve (per keep-N: mean share of
routing selections retained, worst layer, and resulting size if
--size-per-expert-gb is given) so N can be chosen from data.

The hotness counter is the streaming cache's per-(layer,expert) selection
count (llama.cpp fork patch, LLAMA_MOE_STREAM_HOTNESS_OUT). It is count-based
(no router-gate weighting): on Kimi-K3 the count-based keep-640 recovered
90.9% of gate-weighted saliency vs 93.5% for saliency-based selection —
usable, and the gap must be disclosed wherever results are published.
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size-per-expert-gb", type=float, default=None,
                    help="expert bytes per (expert, all layers) in GB, for size column")
    ap.add_argument("--base-gb", type=float, default=0.0,
                    help="non-expert size in GB, added to the size column")
    args = ap.parse_args()

    counts = defaultdict(dict)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            counts[int(row["layer"])][int(row["expert"])] = int(row["count"])
    layers = sorted(counts)
    n_expert = max(max(c) for c in counts.values()) + 1
    mat = np.zeros((len(layers), n_expert), dtype=np.float64)
    for i, l in enumerate(layers):
        for e, c in counts[l].items():
            mat[i, e] = c
    print(f"layers: {len(layers)}  experts/layer: {n_expert}  "
          f"total selections: {mat.sum():.0f}")

    def coverage(n: int) -> tuple[float, float]:
        per = []
        for i in range(len(layers)):
            row = mat[i]
            top = np.sort(row)[::-1][:n]
            per.append(top.sum() / row.sum() if row.sum() else 1.0)
        return float(np.mean(per)), float(np.min(per))

    if args.keep is None:
        print(f"{'keep':>6} {'cov-mean':>9} {'cov-min':>8}", end="")
        if args.size_per_expert_gb:
            print(f" {'size-GB':>8}")
        else:
            print()
        for n in range(n_expert // 4, n_expert + 1, max(n_expert // 32, 1)):
            m, mn = coverage(n)
            line = f"{n:>6} {m*100:8.2f}% {mn*100:7.2f}%"
            if args.size_per_expert_gb:
                line += f" {args.base_gb + args.size_per_expert_gb*n:8.1f}"
            print(line)
        return

    m, mn = coverage(args.keep)
    print(f"keep-{args.keep}: coverage mean {m*100:.2f}%  min {mn*100:.2f}%")
    plan = {"mode": "uniform", "num_experts": n_expert, "layers": {}}
    for i, l in enumerate(layers):
        keep = np.argsort(mat[i])[::-1][: args.keep]
        plan["layers"][str(l)] = {"keep": sorted(int(x) for x in keep),
                                  "bits": {"all": "inherit"}}
    out = args.out or f"out/reap_plan_qwen_{args.keep}.json"
    json.dump(plan, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
