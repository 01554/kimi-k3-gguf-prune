---
license: other
license_name: modified-mit
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
# direct base: this is an expert-prune of Unsloth's 1-bit quant, whose bytes it
# inherits verbatim; Moonshot's original is the grandparent via that quant
base_model: unsloth/Kimi-K3-GGUF
pipeline_tag: text-generation
language:
  - ja
tags:
  - kimi-k3
  - gguf
  - expert-pruning
  - reap
  - llama.cpp
  - mac-studio
  - japanese
---

# Kimi-K3-REAP640ja-IQ1_S-GGUF

**The Japanese-calibrated sibling of
[Kimi-K3-REAP640-IQ1_S-GGUF](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF):
Kimi-K3 for Japanese, on a single 512 GB Mac Studio.**

Same recipe as REAP640 — Unsloth's
[UD-IQ1_S dynamic 1-bit quant](https://huggingface.co/unsloth/Kimi-K3-GGUF)
(594 GB, all 896 experts) REAP-pruned to **640 experts / 441 GB** — but the
keep-list comes from a **Japanese + Chinese** calibration instead of English +
code. Same size, same expert count per layer; the two builds differ only in
*which* 640 experts survive (~473 of 640 per layer are shared, ~167 swapped).

On ELYZA-tasks-100 (Japanese instruction benchmark, blinded LLM judge) this
build scores **4.16/5** where the en+code sibling scores **1.81/5**, winning
the blinded pairwise comparison **83–6** with 11 ties. Japanese held-out
perplexity drops from 19.46 to **4.46**.

| | |
|---|---|
| experts | 640 of 896 per MoE layer (uniform), REAP saliency ranking |
| calibration | Japanese + Chinese subset of a tagged multi-domain corpus — **97.3%** ja+zh saliency mass retained (en+code retention drops to 66.8%) |
| quantization | untouched — surviving experts are byte-identical to UD-IQ1_S (slab copy along the expert axis, no requantization) |
| router / norms | F32, inherited intact from the Unsloth quant |
| size | 441.4 GB, **single file** (fits 512 GB unified memory with headroom) |
| measured | ELYZA-tasks-100 generation on Mac Studio M3 Ultra 512 GB: ~3.0 tok/s effective decode |

Chinese rides along (it was 10% of the calibration and Japanese borrows heavily
from Chinese experts — 42.8% top-expert overlap): zh perplexity is 4.10 vs the
sibling's 7.93.

**This is not a coding build.** Code perplexity roughly doubles (2.00 → 3.87)
and agentic use is untested here. For coding agents, use
[REAP640](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF).

Full write-up (Japanese):
[枝刈りKimi K3の日本語版を作って、ELYZA-tasks-100で比べた](https://zenn.dev/hellohazime/articles/kimi_k3_reap640ja_elyza)


**Provenance**: the exact pruning plan is published — [`plans/reap_plan_640ja.json`](https://github.com/01554/kimi-k3-gguf-prune/tree/main/plans) (sha256 `30b862a465790480…`, full hash in SHA256SUMS); plan + source quant + the MIT slicer reproduce this file's bytes. Split-half: calibrating on Japanese alone vs Chinese alone agrees on 83.8% of the keep-set (oracle retention 89–91%) — see plans/README.md.

## Verified vs. not verified

Honest scorecard: exactly what has been measured, and what has not.

**Verified:**

| claim | evidence |
|---|---|
| Loads and serves on one 512 GB M3 Ultra, full Metal offload | 100-task ELYZA generation run end-to-end (5.8 h, ~3.0 tok/s effective) |
| Japanese generation quality vs the en+code sibling | ELYZA-tasks-100: rubric mean **4.16/5** vs 1.81/5; blinded order-randomized pairwise **83–6–11** (judge: Claude Sonnet — numbers are **not** comparable to GPT-4-judged leaderboards, only between these two builds) |
| Held-out perplexity (48×2048-token chunks, C4-validation-based) | ja **4.46** / zh 4.10 / en 8.50 / code 3.87 (sibling: ja 19.46 / zh 7.93 / en 7.44 / code 2.00) |
| Pruning is lossless for surviving experts | identity-prune is byte-identical (pinned by tests); router/norms stay F32 |
| Generation settings disclosed | max_tokens 4096, thinking_effort low, temp 1.0, top-p 0.95, identical for both builds; 4/100 answers hit the 4096 cap. A 16k-budget recheck rescued none of them: 2 re-ran to clean completion *within* the original budget (stochastic thinking runaways at temp 1.0), 1 was still empty at 16k (114k chars of thinking), 1 hit 16k again in the answer body. The cap is not the bottleneck |

**Verified operating envelope** — what this card's numbers actually cover:

| measured at | detail |
|---|---|
| Single-turn Japanese generation | ELYZA-tasks-100, max_tokens 4096, thinking low, temp 1.0 / top-p 0.95 — the rubric/pairwise numbers above |
| Perplexity | 2048-token windows, 4 domains (ja / en / code / zh table above) |
| Serving | 131,072 context configured and stable; ELYZA prompts exercise only the short end of it |
| Modality | text tensors only (no mmproj shipped) |

Agentic tool-calling sessions have been run **only on the en+code sibling** —
its SWE-Lancer numbers belong to that build alone. The code-perplexity
doubling above is the honest predictor for coding work here, and the judge
flagged factual slips inside fluent Japanese: 1.6-bit experts are fluent
before they are precise.

## Download & run

This repo ships one 441 GB GGUF, no shards (the Hub allows files up to 500 GB;
`hf download` resumes interrupted transfers):

```bash
hf download hellohazime/Kimi-K3-REAP640ja-IQ1_S-GGUF \
    Kimi-K3-REAP640ja-IQ1_S.gguf --local-dir .
```

Kimi-K3 support is not in mainline llama.cpp yet. Build the
[Unsloth fork](https://github.com/unslothai/llama.cpp) at its K3 PR
(built on top of [llama.cpp PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185)):

```bash
git clone https://github.com/unslothai/llama.cpp
cd llama.cpp && git fetch origin pull/48/head:kimi-k3 && git checkout kimi-k3
cmake -B build -DGGML_METAL=ON        # Apple Silicon; use -DGGML_CUDA=ON on NVIDIA
cmake --build build --config Release -j --target llama-server

./build/bin/llama-server -m Kimi-K3-REAP640ja-IQ1_S.gguf \
    --port 8090 -ngl 99 -c 131072 --jinja --cache-reuse 0 \
    --temp 1.0 --top-p 0.95
```

- `--cache-reuse 0` is **required**: partial prefix-cache reuse corrupts the
  KDA recurrent state (known issue, see the PR discussion).
- K3 is thinking-only; reasoning arrives in `reasoning_content`. Control depth
  with `chat_template_kwargs: {"thinking_effort": "low" | "high" | "max"}`.
- Sampling per Moonshot: `temperature 1.0, top_p 0.95`.

## How it was made

Identical pipeline to the sibling build, documented in
[01554/kimi-k3-gguf-prune](https://github.com/01554/kimi-k3-gguf-prune). A
tagged multi-domain calibration corpus (ja 30% / en 25% / code 25% / zh 10% /
5-language tail 10%, token shares) was streamed once through the 1.56 TB MXFP4
source with per-source saliency recording; the keep-640 plan sums only the
`lang-ja` + `chinese` labels. Saliency measurement and planning use
pipenetwork's [kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx)
scripts (`reap_calibrate.py` / `reap_subset.py` / `reap_plan.py`); the GGUF
slicing (byte-slab copy along the expert axis, router rows and `exp_probs_b`
renumbered to keep order) is this project's only original code.

Credits: [Moonshot AI](https://huggingface.co/moonshotai) (Kimi-K3),
[Unsloth](https://huggingface.co/unsloth) (dynamic 1-bit quant whose protected
router/norms this build inherits),
[Cerebras REAP](https://github.com/CerebrasResearch/reap) (saliency criterion),
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) (calibration
machinery), [ELYZA](https://huggingface.co/datasets/elyza/ELYZA-tasks-100)
(ELYZA-tasks-100).

## 日本語の説明

Moonshot AIの2.8兆パラメータモデル
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)を、Mac Studio(512GB)
1台で動くサイズ(441GB)に枝刈りした日本語版です。

公開済みの[REAP640](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF)は
英語+コードで校正したため、日本語を入れると中国語混じりの出力に崩れます。
この版は校正を日本語+中国語に変えて、同じ640個構成で作り直したものです。
サイズも構成も同じで、残っているexpertの顔ぶれだけが違います。

ELYZA-tasks-100(盲検・提示順ランダムのClaude Sonnet判定)で4.16/5。
英語+コード版は同条件で1.81/5でした。判定者がリーダーボード(GPT-4系)と
違うため、数値を外部のスコアと直接比較はできません。

注意点は2つ。コーディング能力は劣化しています(code perplexity 2.00→3.87)。
エージェント用途の検証(Kimi Code CLI、SWE-Lancer)はこの版では行っていません。
その用途にはREAP640を使ってください。

作った経緯と実測の詳細:
[枝刈りKimi K3の日本語版を作って、ELYZA-tasks-100で比べた](https://zenn.dev/hellohazime/articles/kimi_k3_reap640ja_elyza)
