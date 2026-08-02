#!/usr/bin/env python3
"""Turn captured agent request dumps into a per-task calibration corpus.

  scripts/transcripts_to_calib.py --dump-dir out/reqdump_rerun --out out/calib_traj.txt

For each task rerun, the proxy captured every turn's request; the LAST request
of a task holds the full conversation (system prompt, tool schemas rendered by
the CLI, every tool result, every assistant reply). That text is what the model
actually processed, so running it through reap_calibrate with one UNIQUE label
per task yields per-task expert demand from the full 896-expert router — no
code changes needed: the per-source tagging machinery treats each task as its
own "source".

Grouping: request files are assigned to tasks by detecting conversation resets
(a request whose message count is not larger than the previous one starts a new
task). Adjust with --task-boundaries if the heuristic misfires.

Known gap: thinking traces are not in the requests (the CLI does not feed
reasoning back), so demand from the think channel is not measured here.
"""

import argparse
import json
from pathlib import Path


def flatten(messages, tools) -> str:
    parts = []
    if tools:
        parts.append(json.dumps(tools, ensure_ascii=False))
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        if c:
            parts.append(str(c))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            parts.append(str(fn.get("name", "")) + " " + str(fn.get("arguments", "")))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None,
                    help="comma-separated task labels in run order (else task-01..)")
    a = ap.parse_args()

    files = sorted(Path(a.dump_dir).glob("req_*.json"))
    if not files:
        raise SystemExit(f"no req_*.json under {a.dump_dir}")

    # group into tasks by message-count resets
    groups, prev_len = [], 0
    for f in files:
        body = json.loads(f.read_text())
        n = len(body.get("messages") or [])
        if n <= prev_len or not groups:
            groups.append([])
        groups[-1].append(body)
        prev_len = n

    labels = (a.labels.split(",") if a.labels
              else [f"task-{i+1:02d}" for i in range(len(groups))])
    if len(labels) != len(groups):
        raise SystemExit(f"{len(groups)} tasks detected but {len(labels)} labels given")

    order, chars = [], []
    with open(a.out, "w", encoding="utf-8") as f:
        for label, reqs in zip(labels, groups):
            final = max(reqs, key=lambda b: len(json.dumps(b)))
            doc = flatten(final.get("messages") or [], final.get("tools"))
            f.write(doc)
            f.write("\n\n")
            order.append(label)
            chars.append(len(doc) + 2)
            print(f"  {label}: {len(reqs)} turns captured, final transcript {len(doc):,} chars")

    json.dump({"order": order, "chars": chars}, open(a.out + ".sources.json", "w"))
    print(f"wrote {a.out} (+.sources.json), {len(groups)} tasks")


if __name__ == "__main__":
    main()
