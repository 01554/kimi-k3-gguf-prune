# Replicating the SWE-Lancer results

Everything needed to re-run our numbers on your own hardware, one command per
task set. Please do — especially the `full896-stream` arm, whose result we
find hard to believe ourselves (see the model card's caveats).

## What you need

| piece | where | size / note |
|---|---|---|
| llama.cpp with K3 + MoE streaming | [`01554/llama.cpp`, branch `k3-stream`](https://github.com/01554/llama.cpp/tree/k3-stream) | Unsloth's K3 fork (PR 48) + upstream [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) pre-merged. Neither is in mainline yet |
| a model to test | [Kimi-K3-REAP-512GB-GGUF](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF) (REAP640 441 GB / REAP576 478 GB) or [unsloth UD-IQ2_XXS](https://huggingface.co/unsloth/Kimi-K3-GGUF) (711 GB, for the streamed arm) | download ONE, with `--include` |
| SWE-Lancer harness + our solver | [`01554/frontier-evals`](https://github.com/01554/frontier-evals), `project/swelancer` | our `KimiCliSolver` runs Moonshot's Kimi Code CLI inside the stock task container; grading untouched |
| Docker + the SWE-Lancer task image | per the harness README (`swelancer/swelancer_x86:releasev1`) | the heaviest setup step |
| Hardware | resident builds: machine that fits the model in RAM/VRAM; streamed arm: ~460 GiB RAM + fast NVMe | Mac Studio 512 GB is what we used |

## Steps

```bash
# 1. engine
git clone -b k3-stream https://github.com/01554/llama.cpp
cd llama.cpp && cmake -B build -DGGML_METAL=ON   # -DGGML_CUDA=ON on NVIDIA
cmake --build build --config Release -j --target llama-server

# 2. model (pick ONE)
hf download hellohazime/Kimi-K3-REAP-512GB-GGUF --include "REAP576-IQ2_XXS/*" --local-dir models
# or: --include "REAP640-IQ1_S/*"
# or: hf download unsloth/Kimi-K3-GGUF --include "UD-IQ2_XXS/*" --local-dir models

# 3. harness (then follow its README once: uv sync + build the task image)
git clone -b k3-replication https://github.com/01554/frontier-evals
cd frontier-evals/project/swelancer

# 4. run
LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server \
MODEL=/path/to/models/REAP576-IQ2_XXS/Kimi-K3-REAP576-IQ2_XXS.gguf \
  scripts/replicate_k3_reap.sh reap576 trio
```

`replicate_k3_reap.sh <build> <taskset>` handles the rest: correct server
flags per build, the exact rollout caps we used (10800 s resident /
18000 s streamed), one attempt per task, per-task CSVs under
`replication_results/`.

Task sets: `probe` (3), `differential` (5), `trio` (the 3 tasks at the center
of the streamed-arm mystery), `battle16` (16), `all24` — **or any explicit
SWE-Lancer IC-SWE Diamond task IDs**:

```bash
scripts/replicate_k3_reap.sh full896-stream 14294 15925
```

That makes divide-and-conquer easy: if a few people each take a slice of
[`results.csv`](results.csv) (or tasks we never ran — all 198 IC-SWE Diamond
IDs work), the table fills itself. Say which IDs you're taking in the thread
so work doesn't double up.

### Sample session — the cell we most want checked

Task **14294** ($4,000) is where our strangest result lives: the pruned
478 GB build solved it, the full 711 GB model it was carved from did not
(streamed from SSD; single attempts). It is also the smallest-input task in
our whole selection, so it's the fastest one to replicate.

```
# the pruned build — we measured PASS:
$ LLAMA_SERVER=~/llama.cpp/build/bin/llama-server \
  MODEL=models/REAP576-IQ2_XXS/Kimi-K3-REAP576-IQ2_XXS.gguf \
  scripts/replicate_k3_reap.sh reap576 14294
...
14294,0,True,0,0,0,4000.0

# the full model it was cut from — we measured FAIL:
$ LLAMA_SERVER=~/llama.cpp/build/bin/llama-server \
  MODEL=models/UD-IQ2_XXS/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf \
  scripts/replicate_k3_reap.sh full896-stream 14294
...
14294,0,False,0,0,0,0.0
```

Update: our own labeled second attempt on the full streamed model **passed**
14294 (and the other two trio tasks) — the first-run 0/3 did not replicate.
That is exactly why this kit exists: single agentic runs at temperature 1.0
carry real variance, and no cell in the table should be trusted until someone
else reproduces it. Full-VRAM (non-streamed) runs of any task remain the most
valuable contribution.

Reading the CSV line: `question_id, attempt_id, correct, input_tokens,
output_tokens, reasoning_tokens, earned`. `correct=True` + `earned` means the
stock SWE-Lancer grader paid out. The three token columns are always 0 (the
solver doesn't report usage) — ignore them.

## About the rollout cap (read before judging failures)

In our battle16 runs, **8 of 10 failures were cap-terminated** — the agent was
still working when the 10800 s wall hit. So a `fail` row often means "did not
finish in 3 h on a 3 tok/s machine", not "cannot solve".

Our own follow-up protocol, which you're welcome to copy: re-run **only
cap-terminated failures** (never natural exits or passes — those results stay
valid) at double the cap, and report those as separate results labeled with
the cap, e.g. `reap576_iq2xxs_21600s`. To do that with this kit:

```bash
ROLLOUT_CAP=21600 scripts/replicate_k3_reap.sh reap576 battle16
# delete only the cap-terminated .csv files from replication_results/ first,
# so the resume feature re-runs exactly those
```

Never mix caps inside one reported column — the cap goes in the label.

## Ground rules for comparable numbers

- One attempt per task; don't re-roll failures. The script's resume feature
  skips completed tasks, never repeats them.
- Report `correct` / `earned` from the CSVs; token columns are always zero
  (the solver doesn't report usage).
- If you change any condition (cap, sampling, context, cache size), please
  say so alongside your numbers — condition drift is how these comparisons
  die.
- Our reference numbers per task: [`results.csv`](results.csv). Conditions:
  [`README.md`](README.md).

Post findings to the Reddit thread, or open an issue here.
