#!/usr/bin/env python3
"""Held-out text files for llama-perplexity, one per domain.

  scripts/make_ppl_corpora.py --out-dir out/ppl

Held-out matters: reap_calibrate consumes the *head* of each training stream,
so scoring perplexity there would grade a build on the very data its prune was
fitted to (kimi-k3-mlx's perplexity harness makes the same point with
--skip-tokens). Japanese and Chinese use the C4 validation splits; FineWeb and
codeparrot have no validation split here, so those skip deep past any prefix a
12 MB calibration corpus could plausibly have consumed.
"""

import argparse
import os

SPECS = [
    # (name, loader kwargs, text field, skip docs, min chars per doc)
    ("ja", dict(path="allenai/c4", name="ja", split="validation"), "text", 0, 400),
    ("zh", dict(path="allenai/c4", name="zh", split="validation"), "text", 0, 400),
    ("en", dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train"),
     "text", 100_000, 400),
    ("code", dict(path="codeparrot/codeparrot-clean-valid", split="train"),
     "content", 20_000, 400),
]

TARGET_CHARS = 500_000  # ~120-170k tokens per domain; plenty for stable ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    from datasets import load_dataset

    os.makedirs(a.out_dir, exist_ok=True)
    for name, kw, field, skip, min_chars in SPECS:
        ds = load_dataset(**kw, streaming=True)
        out, got, seen = [], 0, 0
        for row in ds:
            seen += 1
            if seen <= skip:
                continue
            t = row.get(field) or ""
            if not isinstance(t, str) or len(t) < min_chars:
                continue
            out.append(t[:20_000])
            got += len(out[-1])
            if got >= TARGET_CHARS:
                break
        path = os.path.join(a.out_dir, f"ppl_{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out))
        print(f"[ppl] {name}: {got:,} chars, {len(out)} docs -> {path}")


if __name__ == "__main__":
    main()
