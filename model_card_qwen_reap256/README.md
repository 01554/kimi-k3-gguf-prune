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
  - width-pruning
  - reap
  - llama.cpp
---

# Qwen3.8-2.4T-A95B, pruned to fit in 256 GB of memory.

Straight talk first: **if you have a 512 GiB machine, you don't need this** —
the unpruned [UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF)
(508 GB) runs fine there by streaming experts from SSD (llama.cpp MoE
streaming, 5.3–6.3 tok/s decode measured on an M3 Ultra). This build exists
for the machines below that: it cuts Qwen3.8 down to what an **English +
code** deployment actually uses, along two axes at once —

1. **expert pruning**: keep the 304 of 512 experts per MoE layer that a
   measured en+code workload actually routes to;
2. **width pruning**: inside every surviving expert, keep the 6 of 8
   256-channel superblocks that carry the most activation energy
   (intermediate width 2048 → 1536).

Result: **246 GB (229 GiB)**, quantization untouched — every surviving byte
is byte-identical to the UD-IQ1_S it came from (1.56 bpw experts, no
requantization anywhere).

| | |
|---|---|
| experts | 304 of 512 per MoE layer (uniform), selected by measured routing counts |
| expert width | 1536 of 2048 (6 of 8 superblocks per expert), selected by measured activation energy |
| calibration | English web + code corpus (200k tokens) through the unpruned model: router selection counts (expert axis) + llama-imatrix activation stats (width axis) |
| selection coverage | 89.6% of routing selections retained (worst layer 83.2%); kept width blocks carry 79.0% of activation energy (mean per expert) |
| quantization | untouched — byte-slab copy of UD-IQ1_S, no requantization |
| size | 246 GB = 229 GiB |
| KLD vs unpruned UD-IQ1_S | held-out en: mean 0.239, median 0.097, argmax agreement 79.2%, PPL 9.05 → 10.90 (×1.20) · held-out code: mean 0.288, median 0.020, argmax agreement 86.6%, PPL 1.89 → 2.38 (×1.26) |

Expert selection is **count-based** (how often the router picked each
expert), not gate-weighted REAP saliency: on Kimi-K3 the same shortcut
recovered 90.9% of saliency mass vs 93.5% for full saliency selection.
Disclosed here because it is a methodological difference from our
[K3 builds](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF).
Width selection uses llama-imatrix's per-expert stats on the
`ffn_down_exps` input — the squared activations of each intermediate
channel — summed per 256-channel superblock.

## Reading the quality numbers

**How they were measured.** ~128k tokens each of held-out English (FineWeb,
skipped far past anything the calibration corpus touched) and held-out code
(codeparrot validation) were teacher-forced through the **unpruned**
UD-IQ1_S, saving its full next-token distributions; the pruned model then
ran the identical text and `llama-perplexity --kl-divergence` compared the
two, token by token. So every number below answers one question: *how far
does this build drift from the exact model it was cut from?* (Not from
FP16 — quantization loss is inherited from the parent and identical by
construction.)

**KL divergence — mean vs median.** KLD is the per-token distance between
the two probability distributions; 0 = identical. The shape matters more
than the average: on code the **median is 0.020** — half of all tokens are
essentially untouched — while the **mean of 0.288** is dragged up by a
small heavy tail (99th percentile: 3.8) where the pruned model disagrees
badly. Pruning damage is *concentrated*, not spread evenly. English shows
the same shape but wider (median 0.097): the en+code calibration protects
code harder than prose, by design.

**Argmax agreement.** The fraction of tokens where the pruned model's #1
choice equals the parent's — i.e., how often greedy decoding would pick the
same token. **86.6% on code / 79.2% on English.** For scale: a published
K3 width-50 prune reported 73.7% against its parent; higher is better, and
100% would mean the prune changed nothing that greedy decoding can see.

**Perplexity ratio.** ×1.26 on code (1.89 → 2.38), ×1.20 on English.
Treat this as a sanity check, not a verdict — PPL averages away exactly
the tail structure that KLD exposes, and the tail is where agentic
failures live.

**Did the tail matter? Yes.** The benchmark below is the tail made
visible: routine tasks (head of the distribution) all pass; the five
hardest tasks (which live in the tail) all fail. If your workload is
routine coding assistance, these numbers say the prune barely touches
you; if it leans on rare, hard reasoning, they say run the unpruned
model.

Like every calibration-pruned build: **what the corpus leaves out is what
gets deleted.** Non-English languages and off-domain abilities are
deliberately sacrificed. Do not use this for multilingual work.

## Verified vs. not verified

**Verified:**

| claim | evidence |
|---|---|
| KLD / argmax agreement vs the unpruned quant | measured on held-out en and code text (table above) |
| Slicing is lossless for surviving weights | identity prune byte-identical; subset equals the naive numpy slice — pinned by tests in the [tooling repo](https://github.com/01554/kimi-k3-gguf-prune) |
| Loads and generates | yes — on **stock mainline llama.cpp** (no fork needed; verified on `4c1a0af`) and on our fork; greedy outputs identical across both, 9.6–10.0 tok/s decode resident on an M3 Ultra |
| Drives an agent CLI | one-shot smoke with Qwen Code 0.21.11 against `llama-server --jinja`: wrote a file via tool calls, ran it, reported correct output. A smoke test, not a benchmark |

**Measured — SWE-Lancer (8 tasks, one attempt each, 10800 s cap, Qwen Code CLI 0.21.11):**

**3 of 8 tasks solved ($2,000 of the $13,500 at stake).** The 8 tasks are
real paid freelance jobs from the SWE-Lancer benchmark, in two groups:

- **3 easy sanity checks** (tasks a 2-bit Kimi-K2.7 baseline could solve):
  **all 3 passed** — including one the unpruned parent model failed on a
  tool-calling format stumble.
- **5 hard tasks** (selected precisely because that baseline failed them,
  prizes $500–$4,000): **all 5 failed.** Every failure was the model
  finishing with a wrong answer well under the 3-hour limit — not running
  out of time. The unpruned parent solves most of these.

That is the KLD tail above, made concrete: routine agentic work survives
the prune; the hardest tasks do not. If you have the RAM for the unpruned
model, run that. Per-task results, exact conditions and every other build
we compare against:
[swelancer-local-subset-evals](https://github.com/01554/swelancer-local-subset-evals).

**Not verified (yet):**

| open question | status |
|---|---|
| Long context, multilingual (broken by design), anything off-corpus | not measured / not intended |

## Run

Works with llama.cpp's Qwen3.5-MoE support (mainline or the
[fork](https://github.com/01554/llama.cpp/tree/k3-stream) we build):

```bash
llama-server -m Qwen3.8-2.4T-A95B-REAP-256GB-IQ1_S.gguf \
    --port 8090 -ngl 99 -c 131072 --jinja \
    --temp 1.0 --top-p 0.95 --top-k 20
```

Sampling per the Unsloth card. The model is a hybrid thinking model
(reasoning arrives in `<think>`; effort via `reasoning_effort`).

**Memory reality check (macOS):** 229 GiB is ~90% of a 256 GiB machine, and
macOS caps Metal's working set well below that by default. Raise it
(`sudo sysctl iogpu.wired_limit_mb=<value>`) and keep context moderate; if
compute still hits the ceiling, lower `-ngl` a few layers — we hit exactly
this on our 512 GiB machine with the 508 GB unpruned model (full offload
OOMs at compute time; `-ngl 80` runs). Honestly: **1-bit quants stream well
— on a 256 GB machine, `--moe-stream` on the unpruned model is a perfectly
good alternative to this build.** This build is for when you want everything
resident.

## How it was made

Same pipeline family as our Kimi-K3 REAP builds
([tooling, MIT](https://github.com/01554/kimi-k3-gguf-prune)), extended to
two axes:

- **Expert axis**: stream the calibration corpus through the unpruned model,
  count per-(layer, expert) router selections (via a small
  [llama.cpp patch](https://github.com/01554/llama.cpp/tree/k3-stream) that
  dumps the MoE-streaming cache's hotness counters), keep the top 304 per
  layer.
- **Width axis**: run `llama-imatrix` (resident) on the same corpus; its
  per-expert `ffn_down_exps` input stats are each intermediate channel's
  activation energy. Keep each expert's top 6 of 8 256-channel superblocks
  (`scripts/make_plan_ew.py`).
- **The MTP layer**: blk.92 is a stored next-token-prediction (nextn) block
  that llama.cpp never executes — it has zero routing counts and no imatrix
  signal. It is sliced to the same uniform shape (required for the file to
  load) with an arbitrary expert selection; if some future runtime starts
  using MTP layers for speculative decoding, expect that path to be degraded
  in this build.
- **Slice** (`scripts/prune_gguf_ew.py`): one pass over the GGUF, copying
  byte slabs. 256-channel cuts align with quant superblocks, so nothing is
  requantized — gate/up lose whole row blocks, down loses the matching
  intra-row spans, the router rows are renumbered to keep order, and
  `expert_count` / `expert_feed_forward_length` are rewritten. Pinned by
  synthetic identity/subset tests.

Credits: [Qwen team](https://huggingface.co/Qwen) (Qwen3.8),
[Unsloth](https://huggingface.co/unsloth) (UD-IQ1_S quant this inherits),
[Cerebras REAP](https://github.com/CerebrasResearch/reap) (the expert-pruning
idea), [mmnga](https://huggingface.co/mmnga) (superblock-aligned width cuts),
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) (the calibration
methodology our pipeline grew from).

## 日本語の説明

Qwen3.8(2.4兆パラメータMoE)を、英語+コードの用途でほぼ使われない部分から
二段階で削った版です。(1) 各層512個のexpertのうち実測ルーティングで選ばれ
ない208個を削除、(2) 残ったexpertの中間層幅2048chのうち活性エネルギーの低い
512ch(256chブロック2個)を削除。量子化はいじらず全てバイトコピーなので、残っ
た重みは元のUD-IQ1_Sと完全一致です。品質数値の読み方: KLDは「切り出し元
とどれだけ分布がズレたか」(0=同一)。codeは中央値0.020=トークンの半分は
ほぼ無傷で、被害は上位1%のテールに集中(99%点3.8)。argmax一致86.6%は
「greedyなら87%のトークンで親と同じ字を打つ」の意。詳細は英語節参照。246GB(229GiB)になり、256GiB機に全載せ
できます(Metalのwired上限引き上げ推奨、詳細は上のRunセクション)。512GiB機
なら削っていない元をSSDストリーミングで動かす方が良い(実測5.3〜6.3 tok/s)。
日本語を含む多言語は設計上壊れています。
