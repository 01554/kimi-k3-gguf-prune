---
license: other
license_name: modified-mit
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
base_model: moonshotai/Kimi-K3
base_model_relation: quantized
tags:
  - kimi-k3
  - gguf
  - expert-pruning
  - reap
  - llama.cpp
  - mac-studio
---

# Kimi-K3-REAP640-IQ1_S-GGUF

## What is this? / これは何？

**Moonshot AI の 2.8 兆パラメータモデル [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) を、
Mac Studio（512 GB）1台で動くサイズにしたものです。**

K3 の最小の GGUF（Unsloth の 1-bit 版）でも 594 GB あり、512 GB の Mac には
載りません。このモデルは、**英語とプログラミングにほぼ使われない専門家
（expert）を 896 個中 256 個削って 441 GB にしたもの**です。残った expert の
中身は Unsloth 版と 1 バイトも違いません（削っただけで、再圧縮していません）。

英語のコーディングエージェント専用です。実際に Moonshot 純正の
[Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) を繋いで
SWE-Lancer の実タスクを解けることを確認済み。**その代わり中国語・日本語などは
意図的に犠牲にしています**（削った expert がそれらを担っていたため）。

---

**Kimi-K3 that fits and runs on a single 512 GB Mac Studio.**
Unsloth's [UD-IQ1_S dynamic 1-bit quant](https://huggingface.co/unsloth/Kimi-K3-GGUF)
(594 GB, all 896 experts), REAP-pruned to **640 experts / 441 GB** with an
English + code calibration corpus.

| | |
|---|---|
| experts | 640 of 896 per MoE layer (uniform), REAP saliency ranking |
| calibration | English web + code only — **93.53%** saliency mass retained |
| quantization | untouched — surviving experts are byte-identical to UD-IQ1_S (slab copy along the expert axis, no requantization) |
| router / norms | F32, inherited intact from the Unsloth quant |
| size | 441.4 GB (fits 512 GB unified memory with KV + compute headroom) |
| measured | Mac Studio M3 Ultra 512 GB: ~47 tok/s prefill, ~3.0 tok/s decode, full Metal offload |

## Verified vs. not verified / 確認済みと未確認

Honest scorecard. This model is three days old; here is exactly what has been
measured and what has not.

**Verified（確認済み）:**

| claim | evidence |
|---|---|
| Loads and runs on one 512 GB M3 Ultra, full Metal offload | measured: 441.4 GB, ~220 s load |
| Speed | measured: ~47 tok/s prefill, ~3.0 tok/s decode |
| Drives Moonshot's Kimi Code CLI end-to-end (24 tools, ~24k-token system prompt) | 3/3 SWE-Lancer IC-SWE Diamond tasks solved, $2,000/$2,000, containerized grading untouched |
| Survives the exact agentic request that deterministically breaks the 4-bit MLX REAP builds | replayed byte-identical request → clean on-task tool call |
| Pruning is lossless for surviving experts | identity-prune is byte-identical (pinned by tests); router/norms stay F32 |
| en+code saliency retention | 93.53% of routed saliency mass at keep-640 |

**Not verified（未確認）:**

| open question | status |
|---|---|
| Full 198-task SWE-Lancer performance | only 3 tasks run; a full run takes weeks at 3 tok/s |
| Tasks the original K2.7-Q2 run failed | 5-task probe **in progress** at time of writing; card will be updated |
| Perplexity / MMLU / standard benchmarks | not measured at all |
| Long-context quality beyond ~30k prompt tokens | context is set to 131k but only exercised to ~30k |
| Chinese, Japanese, and every other language | expected broken by design (en+code calibration); degree not measured |
| Vision | **mmproj not included**; this prune touched text tensors only. Unsloth's mmproj may work but is untested here |
| Sustained multi-day agent sessions | longest observed run: ~68 min/task |

## Verification details

Driven by [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) (Moonshot's
own agent, 24 tools, ~24k-token system prompt) inside SWE-Lancer task
containers, via `llama-server`:

- **3/3 correct** on SWE-Lancer IC-SWE (Diamond) smoke tasks
  (28096_836, 18827_741, 29618_781 — $2,000/$2,000 earned), grading untouched.
- The same agentic request **deterministically degenerates** on the 4-bit MLX
  REAP builds that keep only 242–326 experts (242@4bit → hard repetition
  loops; 326 mixed → worse). Keeping more experts at ~1.6 bpw beats keeping
  fewer at 4 bpw for agentic coherence.
- The unpruned 594 GB quant answers the identical request correctly (verified
  via disk-offload), so the prune's damage on this workload is not observable
  at this task scale.

## What was traded away — on purpose

Calibration decides what survives a prune. This build was calibrated on
**35% multi-language code / 20% Python / 45% English web, nothing else**.
Chinese, Japanese and other languages will be degraded, likely severely
(kimi-k3-mlx measured total collapse of Chinese on an en+code-calibrated
prune). This is a coding-agent build, not a general-purpose one.

## Run it

Needs Kimi-K3 support in llama.cpp — currently the
[Unsloth fork](https://github.com/unslothai/llama.cpp) branch
`kimi-k3-fullsize-vision` (built on [PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185)):

```bash
llama-server -m Kimi-K3-REAP640-IQ1_S-00001-of-00010.gguf \
    -ngl 99 -c 131072 --jinja --cache-reuse 0 \
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
[providers.local-mlx]
type = "openai"
base_url = "http://127.0.0.1:8090/v1"
api_key = "local"
[models.local-k3]
provider = "local-mlx"
model = "k3"
max_context_size = 131072
```

## How it was made

[kimi-k3-gguf-prune](https://github.com/hellohazime/kimi-k3-gguf-prune) —
REAP saliency (`gate·‖expert output‖` streamed layer-by-layer over the 1.56 TB
MXFP4 source, peak ~58 GB RAM), uniform keep-640 plan, then a pure byte-slab
slice of the GGUF along the outermost expert axis with the router rows and
`exp_probs_b` renumbered to the keep order. An identity prune is byte-identical
by test.

Credits: [Moonshot AI](https://huggingface.co/moonshotai) (Kimi-K3, Kimi Code
CLI), [Unsloth](https://huggingface.co/unsloth) (dynamic 1-bit quant whose
protected router/norms this build inherits), [Cerebras
REAP](https://github.com/CerebrasResearch/reap) (saliency criterion),
[kimi-k3-mlx](https://huggingface.co/pipenetwork) (calibration machinery and
the measured warnings this build steers by).
