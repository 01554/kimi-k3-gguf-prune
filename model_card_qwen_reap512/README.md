---
license: other
license_name: qwen3.8-max
license_link: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/LICENSE
base_model: unsloth/Qwen3.8-2.4T-A95B-GGUF
pipeline_tag: text-generation
tags:
  - qwen3.8
  - gguf
  - expert-pruning
  - reap
  - llama.cpp
---

# Qwen3.8-2.4T-A95B, expert-pruned to fit in 512 GB of memory.

The unpruned Q2 quant of Qwen3.8
([UD-IQ2_XXS](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF), 656.6 GB)
**cannot run on a 512 GiB machine at all** — it exceeds total RAM. This build
prunes the experts an **English + code** deployment doesn't route to, keeping
the healthier Q2 quantization intact: **404 GB (376 GiB), fully resident on a
512 GiB Mac at ~9.5 tok/s.**

**No fork, no PR branch, no custom runtime.** This file loads and generates
on stock mainline llama.cpp — verified by running it there (greedy outputs
identical to our fork). Download and `llama-server`, nothing else.

| | |
|---|---|
| experts | 304 of 512 per MoE layer (uniform), selected by measured routing counts; **width untouched** (unlike our [256 GB sibling](https://huggingface.co/hellohazime/Qwen3.8-2.4T-A95B-REAP-256GB-GGUF)) |
| calibration | English web + code corpus (200k tokens) through the unpruned model; per-(layer, expert) router selection counts |
| selection coverage | 89.7% of routing selections retained (worst layer 83.2%) |
| quantization | untouched — byte-slab copy of UD-IQ2_XXS (layer-mixed ~1.9 bpw experts), no requantization |
| size | 404 GB = 376 GiB — full `-ngl 99` offload fits under macOS's Metal working-set ceiling with room for KV |
| KLD vs unpruned UD-IQ2_XXS | held-out en: mean 0.110, median 0.024, argmax agreement 87.6%, PPL 8.70 → 9.31 (×1.07) · held-out code: mean 0.196, median 0.007, **argmax agreement 90.0%**, PPL 1.75 → 2.03 (×1.16) |

Expert selection is **count-based** (router hit counts), not gate-weighted
REAP saliency — on Kimi-K3 this shortcut recovered 90.9% of saliency mass vs
93.5%. The counts come from the same measured workload as the 256 GB sibling;
the two builds share their keep-set.

Like every calibration-pruned build: **what the corpus leaves out is what
gets deleted.** Non-English languages and off-domain abilities are
deliberately sacrificed.

## Reading the quality numbers

Measured exactly as on the [256 GB sibling's card](https://huggingface.co/hellohazime/Qwen3.8-2.4T-A95B-REAP-256GB-GGUF)
(teacher-forced held-out text, ~128k tokens per domain, base logits from the
unpruned quant): the numbers answer *how far does this build drift from the
model it was cut from*. Median KLD 0.007 on code means half of all code
tokens are essentially untouched; the top-1 token survives 90% of the time.
Side-by-side with the 256 GB build (which also cuts expert width and starts
from a 1.56 bpw base), this build keeps roughly **twice** the fidelity on
every metric — en argmax 87.6% vs 79.2%, code 90.0% vs 86.6%, PPL overhead
×1.07-1.16 vs ×1.20-1.26. That gap is the measured price of the last 158 GB.

## Verified vs. not verified

**Verified:**

| claim | evidence |
|---|---|
| KLD / argmax agreement vs the unpruned quant | table above, held-out en and code |
| Slicing is lossless for surviving weights | identity prune byte-identical — pinned by tests in the [tooling repo](https://github.com/01554/kimi-k3-gguf-prune) |
| Loads and generates | **stock mainline llama.cpp** (9.3 tok/s) and our fork (9.5 tok/s), greedy outputs identical, full `-ngl 99` resident on an M3 Ultra 512 GB |
| Agentic sanity check | two labeled conditions, per our protocol: bare prompt **1/3** — two one-turn deaths obeying the benchmark's phantom ```python scaffold instruction (log-verified, same trap class as the parent's single bench fail) · with the standard counter-note (`promptv1m`) **3/3** — the trapped cells become normal grind-passes (20-min death → 74-min \$1,000 win). Full data: [swelancer-local-subset-evals](https://github.com/01554/swelancer-local-subset-evals) |

**What the sanity check actually demonstrates** — and what most quant/prune
releases never show: this build drove Qwen's own CLI agent through **three
real, paid SWE-Lancer tasks end-to-end** — multi-hour tool-calling sessions
(29 / 74 / 185 minutes) against a real React Native codebase — and **solved
all three** ($2,000 in original prize value) under the disclosed conditions.
Not a perplexity table: the model plans, edits files, runs shells, and
finishes. Per current protocol new builds get this gate plus the KLD oracle
instead of a full benchmark run; per-task cells live in
[swelancer-local-subset-evals](https://github.com/01554/swelancer-local-subset-evals).

**Not verified:** long context, multilingual (broken by design), anything
off-corpus.

## Run

```bash
llama-server -m Qwen3.8-2.4T-A95B-REAP-512GB-IQ2_XXS.gguf \
    --port 8090 -ngl 99 -c 131072 --jinja \
    --temp 1.0 --top-p 0.95 --top-k 20
```

Works on stock mainline llama.cpp (no fork needed; Qwen3.5-MoE support).
Sampling per the Unsloth card; hybrid thinking model.

## How it was made

Same pipeline as the [256 GB sibling](https://huggingface.co/hellohazime/Qwen3.8-2.4T-A95B-REAP-256GB-GGUF)
([tooling, MIT](https://github.com/01554/kimi-k3-gguf-prune)), minus the
width cut: count the router selections over the calibration corpus, keep the
top 304 experts per layer, slice the GGUF as raw byte slabs, renumber the
router. blk.92 (a stored, never-executed MTP block) is sliced blind to keep
the file loadable. No requantization anywhere.

Credits: [Qwen team](https://huggingface.co/Qwen),
[Unsloth](https://huggingface.co/unsloth) (UD-IQ2_XXS this inherits),
[Cerebras REAP](https://github.com/CerebrasResearch/reap),
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx).

## 日本語の説明

Qwen3.8のQ2量子化(UD-IQ2_XXS、656.6GB)は512GiB機のRAM総量を超えていて
そもそも載りません。これは英語+コード用途でほぼ呼ばれないexpertを実測
ルーティングで削って**404GB(376GiB)**にした版で、512GiB機に`-ngl 99`で
全載せ・約9.5 tok/sで動きます。幅は削っていないので、256GB版より忠実度が
全指標で約2倍良い(code argmax一致90.0%、KLD中央値0.007)。多言語は設計上
壊れています。素のmainline llama.cppで動作検証済み(fork不要)。単なる
perplexity表ではなく、Qwen公式CLIで実在の有償SWE-Lancerタスク3問を
数時間のツール往復の末に**3問とも解き切った**ことを検証済みです。
