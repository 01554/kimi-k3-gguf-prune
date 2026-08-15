# SWE-Lancer task selection and per-task results

Exact task lists behind every number published on the
[Kimi-K3-REAP-512GB-GGUF](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF)
model card, so anyone can re-run the same tasks. Machine-readable results:
[`results.csv`](results.csv).

Task **content is not mirrored here** — the tasks belong to OpenAI's
[SWE-Lancer benchmark](https://arxiv.org/abs/2502.12115) (IC SWE, Diamond
split) and can be looked up by `question_id` in the official release. We
publish IDs, prices, selection rules and outcomes only.

## Common conditions (all experiments)

- Agent: Moonshot's [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code)
  inside the stock SWE-Lancer task container, pointed at a local
  `llama-server` (Unsloth llama.cpp fork, K3 branch).
- Sampling: temperature 1.0, top-p 0.95; context 131,072; `--cache-reuse 0`.
- Rollout cap: 10,800 s per task (the full-896 SSD-streamed check used
  18,000 s to compensate for its slower decode).
- **One attempt per task.** Genuine failures were never re-rolled. Two tasks
  (24508_791, 15815_1) hit a harness config error before the model was ever
  invoked; those were re-scheduled once and the re-run counts as the first
  attempt.
- Grading: stock SWE-Lancer, unmodified. The solver does not report token
  counts (zeros in raw run CSVs are an artifact).

## Task sets

Probe and differential were both selected by one metric: the **`input_tokens`
column of the K2.7 baseline's full-run `results.csv`** — i.e., the input
volume each task actually consumed in our
[earlier 198-task K2.7 run](https://zenn.dev/hellohazime/articles/kimi_k27_code_swelancer_local),
agent trajectory included. Sorting ascending and taking the head reproduces
both selections exactly, order and all.

**probe (3 tasks)** — the 3 smallest-input tasks among the 93 the K2.7
baseline solved; used as a does-it-still-work gate for every new build.

**differential (5 tasks)** — the 5 smallest-input tasks among the 105 the
K2.7 baseline failed (recorded inputs 16k–104k tokens).

**battle16 (16 tasks)** — the 1.56 bpw × 640 vs 1.91 bpw × 576 head-to-head
extension, currently running. Selection is fully reproducible: from the same
105 K2.7-failed tasks, sort ascending by `len(title) + len(description)`
(from the benchmark's task table), drop the 5 already used by the
differential set, take the first 16. Prize prices were not consulted during
selection (they range $250–$32,000, Σ$42,000). Note: this metric differs
from the one used for probe/differential; both are fully specified here, so
every set is reproducible.

Final per-build scores quoted on the model card = probe + differential
(8 tasks); battle16 results will be added to `results.csv` per task as runs
complete, then rolled into 24-task totals.

## Also tested on the differential trio (14294 / 15815_1 / 15925)

- Full 896-expert UD-IQ2_XXS streamed from SSD (llama.cpp MoE-streaming
  patch, ~2/3 decode speed): 0/3 — the oddity discussed on the model card.
- 4-bit × 240-expert prune (F32 router): degenerated on the agentic prompt;
  not run to completion on these tasks.

Replication kit: see [REPLICATION.md](REPLICATION.md) — one command re-runs any task set on your hardware.

**qwen38_a95b_udiq1s_10800s** — a different *model* on the same tasks:
unpruned Qwen3.8-2.4T-A95B (unsloth UD-IQ1_S, 508 GB) served via llama.cpp
MoE streaming (cache 400 GiB, ~5-6 tok/s decode), driven by **Qwen Code CLI
0.21.10–0.21.11** (the maker's-own-agent principle: each model gets its
vendor's official CLI, as Kimi models get Kimi Code CLI). Same 10800 s cap,
one attempt per task. 4 of 8 tasks measured so far; `pending` cells are
deferred, not failures. Note: one earlier batch was discarded due to a solver
env bug (stream idle timeout) before these runs.

**qwen38_reap256_iq1s_10800s** — our expert+width-pruned build of the model
above: [Qwen3.8-2.4T-A95B-REAP-256GB-GGUF](https://huggingface.co/hellohazime/Qwen3.8-2.4T-A95B-REAP-256GB-GGUF)
(keep-304 of 512 experts × 6/8 width superblocks, 246 GB), served fully
resident (`-ngl 99 -c 131072 --jinja`, temp 1.0 / top-p 0.95 / top-k 20,
~10 tok/s decode), same Qwen Code CLI (pinned 0.21.11), same 10800 s cap, one
attempt per task. Complete: 3/8, $2,000 (probe 3/3, differential 0/5).
All fails so far ended well under the cap (natural fails, not
cap-terminated).
