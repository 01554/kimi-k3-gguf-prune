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

**probe (3 tasks)** — tasks the 341 GB 2-bit K2.7-Code baseline solved in our
[earlier full 198-task run](https://zenn.dev/hellohazime/articles/kimi_k27_code_swelancer_local);
used as a does-it-still-work gate for every new build.

**differential (5 tasks)** — from the 105 tasks that K2.7 baseline failed,
the five with the smallest task inputs. Honest note: the exact size metric
used for this pick was lost (it predates our records; we could not reproduce
it from title/description lengths or price). The five task IDs stand as
published.

**battle16 (16 tasks)** — the 1.56 bpw × 640 vs 1.91 bpw × 576 head-to-head
extension, currently running. Selection is fully reproducible: from the same
105 K2.7-failed tasks, sort ascending by `len(title) + len(description)`
(from the benchmark's task table), drop the 5 already used by the
differential set, take the first 16. Prize prices were not consulted during
selection (they range $250–$32,000, Σ$42,000).

Final per-build scores quoted on the model card = probe + differential
(8 tasks); battle16 results will be added to `results.csv` per task as runs
complete, then rolled into 24-task totals.

## Also tested on the differential trio (14294 / 15815_1 / 15925)

- Full 896-expert UD-IQ2_XXS streamed from SSD (llama.cpp MoE-streaming
  patch, ~2/3 decode speed): 0/3 — the oddity discussed on the model card.
- 4-bit × 240-expert prune (F32 router): degenerated on the agentic prompt;
  not run to completion on these tasks.
