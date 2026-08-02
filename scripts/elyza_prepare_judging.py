#!/usr/bin/env python3
"""Assemble blinded judging batches from two builds' ELYZA generations.

  scripts/elyza_prepare_judging.py --a out/elyza_reap640ja.jsonl --b out/elyza_reap640.jsonl \
      --tasks out/elyza100.jsonl --out-dir out/judging --batches 10

Two artifacts per batch file:
- rubric items: {task_id, input, reference, eval_aspect, answer, key} for BOTH
  builds, keys anonymised ("m1"/"m2" per-item random swap) — absolute 1-5 scoring
  against the ELYZA rubric.
- pairwise items: same pair presented once, order randomised, for A/B preference.

The mapping key->build is stored separately (out/judging/mapping.json) so the
judge never sees build names; order swaps cancel position bias in aggregate.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="build A generations (jsonl)")
    ap.add_argument("--b", required=True, help="build B generations (jsonl)")
    ap.add_argument("--tag-a", default="reap640ja")
    ap.add_argument("--tag-b", default="reap640")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(args.tasks)}
    ga = {json.loads(l)["id"]: json.loads(l) for l in open(args.a)}
    gb = {json.loads(l)["id"]: json.loads(l) for l in open(args.b)}
    ids = sorted(set(ga) & set(gb))
    print(f"{len(ids)} tasks with both generations")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    items = []
    for i in ids:
        t = tasks[i]
        swap = rng.random() < 0.5
        first, second = (args.tag_a, args.tag_b) if not swap else (args.tag_b, args.tag_a)
        mapping[str(i)] = {"m1": first, "m2": second}
        g = {args.tag_a: ga[i], args.tag_b: gb[i]}
        items.append({
            "task_id": i,
            "input": t["input"],
            "reference": t["output"],
            "eval_aspect": t["eval_aspect"],
            "m1_answer": g[first]["answer"],
            "m2_answer": g[second]["answer"],
        })

    (out / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
    per = (len(items) + args.batches - 1) // args.batches
    for b in range(args.batches):
        chunk = items[b * per : (b + 1) * per]
        if not chunk:
            break
        (out / f"batch_{b:02d}.json").write_text(
            json.dumps(chunk, ensure_ascii=False, indent=1))
    print(f"wrote {args.batches} batches + mapping.json -> {out}")


if __name__ == "__main__":
    main()
