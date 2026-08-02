---
license: other
license_name: modified-mit
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
# direct base: this is an expert-prune of Unsloth's 1-bit quant, whose bytes it
# inherits verbatim; Moonshot's original is the grandparent via that quant
base_model: unsloth/Kimi-K3-GGUF
pipeline_tag: text-generation
tags:
  - kimi-k3
  - gguf
  - expert-pruning
  - reap
  - llama.cpp
  - mac-studio
---

# Kimi-K3-REAP640-IQ1_S-GGUF

**Kimi-K3 that fits and runs on a single 512 GB Mac Studio.**

Unsloth's [UD-IQ1_S dynamic 1-bit quant](https://huggingface.co/unsloth/Kimi-K3-GGUF)
(594 GB, all 896 experts), REAP-pruned to **640 experts / 441 GB** with an
English + code calibration corpus. Driven end-to-end by Moonshot's own
[Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) on real SWE-Lancer
tasks: **5/8 solved, $3,500 earned — including 2 tasks the 341 GB 2-bit
K2.7-Code quant failed.**

| | |
|---|---|
| experts | 640 of 896 per MoE layer (uniform), REAP saliency ranking |
| calibration | English web + code only — **93.53%** saliency mass retained |
| quantization | untouched — surviving experts are byte-identical to UD-IQ1_S (slab copy along the expert axis, no requantization) |
| router / norms | F32, inherited intact from the Unsloth quant |
| size | 441.4 GB (fits 512 GB unified memory with KV + compute headroom) |
| measured | Mac Studio M3 Ultra 512 GB: ~47 tok/s prefill, ~3.0 tok/s decode, full Metal offload |


Full write-up — how it was built, what failed along the way, and the verification: [English](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac#english-version) / [日本語](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac).

This is a coding-agent build, not a general-purpose one: Chinese, Japanese and
other languages were deliberately sacrificed by the calibration choice (the
pruned experts are the ones those languages used).

## Verified vs. not verified

Honest scorecard: exactly what has been measured, and what has not.

**Verified:**

| claim | evidence |
|---|---|
| Loads and runs on one 512 GB M3 Ultra, full Metal offload | measured: 441.4 GB, ~220 s load |
| Speed | measured: ~47 tok/s prefill, ~3.0 tok/s decode |
| Drives Moonshot's Kimi Code CLI end-to-end (24 tools, ~24k-token system prompt) | 8 SWE-Lancer IC-SWE Diamond tasks run end-to-end: 3/3 on K2.7-solved tasks ($2,000/$2,000) plus 2/5 on K2.7-failed tasks ($1,500); grading untouched |
| Not a strict subset of the 2-bit K2.7 baseline | **2/5 solved** on the five cheapest tasks that K2.7-Q2 failed |
| Survives the exact agentic request that deterministically breaks the 4-bit MLX REAP builds | replayed byte-identical request → clean on-task tool call |
| Pruning is lossless for surviving experts | identity-prune is byte-identical (pinned by tests); router/norms stay F32 |
| en+code saliency retention | 93.53% of routed saliency mass at keep-640 |
| Held-out perplexity (this build) | measured, 48×2048-token chunks: code **2.00** / en **7.44** / zh 7.93 / **ja 19.46** — the deliberate en+code trade, quantified |

**Not verified:**

| open question | status |
|---|---|
| Full 198-task SWE-Lancer performance | 8 tasks run so far; a full run takes weeks at 3 tok/s |
| MMLU / standard benchmarks | not measured (held-out perplexity is measured, see above) |
| Long-context quality beyond ~30k prompt tokens | context is set to 131k but only exercised to ~30k |
| Chinese, Japanese generation quality | perplexity is measured (ja ≈2.6× en); generation/agentic quality in those languages is not |
| Vision | **mmproj not included**; this prune touched text tensors only. Unsloth's mmproj may work but is untested here |
| Sustained multi-day agent sessions | longest observed run: ~68 min/task |

## Verification details

Driven by [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) (Moonshot's
own agent, 24 tools, ~24k-token system prompt) inside SWE-Lancer task
containers, via `llama-server`:

- **3/3 correct** on tasks the 2-bit K2.7 baseline also solved
  (28096_836, 18827_741, 29618_781 — $2,000/$2,000), grading untouched.
- **2/5 correct** on the five cheapest tasks that same K2.7 baseline *failed*
  (24508_791 $1,000, 27353_776 $500) — so this build is not a strict subset of
  K2.7's ability despite 1-bit experts and a 29% expert prune.
- The same agentic request **deterministically degenerates** on the 4-bit MLX
  REAP builds that keep only 242–326 experts (242@4bit → hard repetition
  loops; 326 mixed → worse). Keeping more experts at ~1.6 bpw beats keeping
  fewer at 4 bpw for agentic coherence.
- The unpruned 594 GB quant answers the identical request correctly (verified
  via disk-offload), so the prune's damage on this workload is not observable
  at this task scale.

## Build & run

Kimi-K3 support is not in mainline llama.cpp yet. Build the
[Unsloth fork](https://github.com/unslothai/llama.cpp) at its K3 PR
(built on top of [llama.cpp PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185)):

```bash
git clone https://github.com/unslothai/llama.cpp
cd llama.cpp && git fetch origin pull/48/head:kimi-k3 && git checkout kimi-k3
cmake -B build -DGGML_METAL=ON        # Apple Silicon; use -DGGML_CUDA=ON on NVIDIA
cmake --build build --config Release -j --target llama-server

./build/bin/llama-server -m Kimi-K3-REAP640-IQ1_S-00001-of-00010.gguf \
    --port 8090 -ngl 99 -c 131072 --jinja --cache-reuse 0 \
    --temp 1.0 --top-p 0.95
```

- `--cache-reuse 0` is **required**: partial prefix-cache reuse corrupts the
  KDA recurrent state (known issue, see the PR discussion).
- K3 is thinking-only; reasoning arrives in `reasoning_content`. Control depth
  with `reasoning_effort` (`low` / `high` / `max`).
- Sampling per Moonshot: `temperature 1.0, top_p 0.95` (agentic: `top_p 1.0`).

Point any OpenAI-compatible agent at it. Kimi Code CLI config:

```toml
default_model = "local-k3"
[providers.local-llamacpp]
type = "openai"
base_url = "http://127.0.0.1:8090/v1"
api_key = "local"
[models.local-k3]
provider = "local-llamacpp"
model = "k3"
max_context_size = 131072
```

## How it was made

Expert saliency and the keep-640 selection were produced with the calibration
scripts from pipenetwork's [kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx)
repo (`reap_calibrate.py` / `reap_plan.py` — REAP saliency
`gate·‖expert output‖` streamed layer-by-layer over the 1.56 TB MXFP4 source,
peak ~58 GB RAM), with the calibration mix swapped to English + code only. The
only new code is [a small script](https://github.com/01554/kimi-k3-gguf-prune)
that applies the plan to the GGUF: a byte-slab slice along the outermost expert
axis (expert slabs are contiguous and quantization blocks never cross them),
with the router rows and `exp_probs_b` renumbered to the keep order. An
identity prune reproduces the input byte-for-byte (pinned by tests).

Credits: [Moonshot AI](https://huggingface.co/moonshotai) (Kimi-K3, Kimi Code
CLI), [Unsloth](https://huggingface.co/unsloth) (dynamic 1-bit quant whose
protected router/norms this build inherits), [Cerebras
REAP](https://github.com/CerebrasResearch/reap) (saliency criterion),
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) (calibration
machinery and the measured warnings this build steers by).

## 日本語の説明

Moonshot AI の 2.8 兆パラメータモデル
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) を、Mac Studio(512 GB)
1台で動くサイズにしたものです。

K3 の最小の GGUF(Unsloth の 1-bit 版)でも 594 GB あり、512 GB の Mac には
載りません。このモデルは、英語とプログラミングにほぼ使われない専門家
(expert)を 896 個中 256 個削って 441 GB にしたものです。残った expert の
中身は Unsloth 版と 1 バイトも違いません(削っただけで、再圧縮していません)。

英語のコーディングエージェント専用です。Moonshot 純正の Kimi Code CLI を
繋いで SWE-Lancer の実タスク 8 本中 5 本を解きました($3,500)。うち 2 本は、
2bit の K2.7-Code では解けなかった問題です。その代わり中国語・日本語などは
意図的に犠牲にしています(削った expert がそれらを担っていたため)。

作った経緯と手法の詳細(日本語):
[Kimi K3を441GBに枝刈りして、Mac Studio 1台で動かした](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac)
