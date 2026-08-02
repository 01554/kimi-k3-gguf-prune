#!/usr/bin/env python3
"""Generate ELYZA-tasks-100 answers from a llama-server instance.

  scripts/elyza_gen.py --tag reap640ja --port 8090 --tasks out/elyza100.jsonl \
      --out out/elyza_reap640ja.jsonl [--limit 3] [--offset 0]

Writes one JSON line per task: {id, input, answer, reasoning_len, gen_tokens,
secs}. Appends, and skips ids already present in --out, so an interrupted run
resumes by re-running the same command.

K3 is thinking-only; `reasoning_effort: "low"` keeps the think channel short so
a 100-task run stays overnight-sized at ~3 tok/s decode. Both builds are run
with identical settings, so the comparison is fair even if "low" is not the
model's best setting in absolute terms.
"""

import argparse
import json
import time
import urllib.request


def ask(port, text, effort, max_tokens, retries=2):
    body = {
        "model": "k3",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        # the template's real knob; measured: thinking 93 chars vs 760 for the
        # top-level reasoning_effort field on the same prompt
        "chat_template_kwargs": {"thinking_effort": effort},
        "stream": False,
    }
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=7200) as r:
                return json.load(r)
        except Exception as e:  # transient server hiccups: retry once or twice
            last = e
            time.sleep(10)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=3000)
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks)]
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(a.out)}
    except FileNotFoundError:
        pass

    todo = [t for t in tasks[a.offset:] if t["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[{a.tag}] {len(todo)} tasks to run ({len(done)} already done)")

    with open(a.out, "a") as f:
        for t in todo:
            t0 = time.time()
            out = ask(a.port, t["input"], a.effort, a.max_tokens)
            secs = time.time() - t0
            ch = out["choices"][0]
            msg = ch["message"]
            rec = {
                "id": t["id"],
                "input": t["input"],
                "answer": msg.get("content") or "",
                "reasoning_len": len(msg.get("reasoning_content") or ""),
                "finish": ch.get("finish_reason"),
                "gen_tokens": out.get("usage", {}).get("completion_tokens", 0),
                "secs": round(secs, 1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  id={rec['id']:>3} {rec['secs']:>6.0f}s {rec['gen_tokens']:>5} tok "
                  f"finish={rec['finish']} ans={len(rec['answer'])} chars", flush=True)


if __name__ == "__main__":
    main()
