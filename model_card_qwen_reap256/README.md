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

Reading the KLD row: on code — the target domain — most tokens are barely
touched (median KLD 0.020) and the top-1 prediction survives 86.6% of the
time; the damage concentrates in a heavy tail (99th percentile KLD 3.8). On
general English the spread is wider (median 0.097, argmax 79.2%). Judge the
build by the agentic benchmark once it lands, not by PPL alone.

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

**Not verified (yet):**

| open question | status |
|---|---|
| SWE-Lancer agentic performance | planned: same 8-task set our K3 builds ran; unpruned Qwen3.8 passed 3 of the 4 tasks measured so far (the run was paused for this build) |
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
た重みは元のUD-IQ1_Sと完全一致です。246GB(229GiB)になり、256GiB機に全載せ
できます(Metalのwired上限引き上げ推奨、詳細は上のRunセクション)。512GiB機
なら削っていない元をSSDストリーミングで動かす方が良い(実測5.3〜6.3 tok/s)。
日本語を含む多言語は設計上壊れています。
