---
license: other
license_name: modified-mit
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
# direct base: these are expert-prunes of Unsloth's dynamic quants, whose bytes
# they inherit verbatim; Moonshot's original is the grandparent via those quants
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

# Kimi-K3, expert-pruned to fit in 512 GB of memory.

One memory budget, two ways to spend it. These are REAP expert-pruned builds
of Unsloth's [dynamic quants](https://huggingface.co/unsloth/Kimi-K3-GGUF) of
Moonshot's [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (2.8T-param
MoE, 896 experts per layer), cut to run **fully resident** on one 512 GB
machine. Instead of only shrinking bits per weight, they drop the experts an
**English + code** deployment rarely routes to. Both builds share the same
calibration corpus and tooling, and differ only in how they spend the memory
budget: more experts at fewer bits, or fewer experts at more bits.

| build | experts kept | en+code saliency | avg expert bpw | size | verification |
|---|---|---|---|---|---|
| [`REAP640-IQ1_S/`](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF/tree/main/REAP640-IQ1_S) | 640/896 | 93.5% | ~1.6 | 441.4 GB, 10 shards | SWE-Lancer 5/8, $3,500 earned |
| [`REAP576-IQ2_XXS/`](https://huggingface.co/hellohazime/Kimi-K3-REAP-512GB-GGUF/tree/main/REAP576-IQ2_XXS) | 576/896 | 90.2% | ~1.9 | 478.5 GB, single file | SWE-Lancer **7/8, $13,000** earned — incl. 3 tasks none of our other setups had solved |

Both run at ~3.0 tok/s decode / ~48 tok/s prefill on a Mac Studio M3 Ultra
512 GB with full Metal offload. The 576 keep-set is a strict subset of the 640
keep-set (same saliency ranking), so the pair isolates the experts-vs-bits
trade cleanly.

## Download one build, not the repo

**A full-repo download fetches both builds (~920 GB). Pick one:**

```bash
# REAP640-IQ1_S (441 GB, 10 shards)
hf download hellohazime/Kimi-K3-REAP-512GB-GGUF --include "REAP640-IQ1_S/*" --local-dir .

# REAP576-IQ2_XXS (478 GB, single file)
hf download hellohazime/Kimi-K3-REAP-512GB-GGUF --include "REAP576-IQ2_XXS/*" --local-dir .
```

`hf download` resumes interrupted transfers.

## Which one

**REAP640-IQ1_S** is the proven build: driven end-to-end by Moonshot's
[Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) on real SWE-Lancer
IC-SWE Diamond tasks — 3/3 on tasks the 341 GB 2-bit K2.7-Code baseline solved,
plus 2/5 on tasks it failed ($3,500 total, grading untouched). Held-out
perplexity: code 2.00 / en 7.44 / zh 7.93 / ja 19.46.

**REAP576-IQ2_XXS** starts from the higher-fidelity quant (Unsloth's published
top-1 agreement with the unquantized model: 84.1% for UD-IQ2_XXS vs 78.9% for
UD-IQ1_S, measured before pruning) and pays for it with 64 fewer experts per
layer. Full 8-task result, one attempt per task, same protocol as REAP640:

| task | K2.7-Q2 (341 GB) | REAP640 | REAP576 |
|---|---|---|---|
| 28096_836 | pass | pass | pass $500 |
| 18827_741 | pass | pass | pass $1,000 |
| 29618_781 | pass | pass | pass $500 |
| 24508_791 | fail | pass $1,000 | pass $1,000 |
| 27353_776 | fail | pass $500 | **fail** |
| 14294 | fail | fail | **pass $4,000** |
| 15815_1 | fail | fail | **pass $4,000** |
| 15925 | fail | fail | **pass $2,000** |

**7/8, $13,000** (REAP640: 5/8, $3,500). The three bottom-row tasks had not
been solved by anything **we** had tested — not the 2-bit K2.7-Code baseline,
not REAP640, and not the full-896-expert UD-IQ2_XXS streamed from SSD. Other
people's pruned K3 builds exist and we have not run them on these tasks.
Grading is stock SWE-Lancer, untouched. Exact task IDs, selection rules and
per-task results for every experiment:
[the eval repo](https://github.com/01554/swelancer-local-subset-evals). Two of the five differential tasks hit a harness config error on
the first scheduling (the model was never invoked) and were re-run once; the
27353_776 failure was a genuine attempt and was **not** re-rolled.

Caveats, honestly: every cell is a single attempt at temperature 1.0. One
oddity got a follow-up. We could not fit the full 896-expert model into this
machine's memory, so to check it we force-ran it anyway, streaming experts
from SSD (llama.cpp's MoE-streaming patch, ~2/3 the decode speed) — and,
oddly, it failed all three bottom-row tasks that this pruned subset of the
very same weights then solved. We then re-ran those three on the full
streamed model as an explicitly-labeled second attempt: **it solved all
three.** The 0/3 did not replicate. Read it as run-to-run variance of
single-attempt agentic runs, not as pruning adding capability — both
attempts are recorded separately in
[the eval repo's per-task results](https://github.com/01554/swelancer-local-subset-evals) (old results.csv URL remains as a synced mirror).
The practical lesson stands: single-run rows in any such table (ours
included) carry real variance. Tool-call stability also wobbles: in 4
replays of a captured 24-tool agentic request, 1 leaked XTML markers into
the arguments (the full task runs completed regardless).

Neither build speaks Chinese or Japanese — the calibration choice deliberately
sacrifices them (the pruned experts are the ones those languages used). For
Japanese, use the Japanese-calibrated sibling
[Kimi-K3-REAP640ja-IQ1_S-GGUF](https://huggingface.co/hellohazime/Kimi-K3-REAP640ja-IQ1_S-GGUF)
(ELYZA-tasks-100 4.16/5 vs REAP640's 1.81/5).

## Build & run

Kimi-K3 support is not in mainline llama.cpp yet. Build the
[Unsloth fork](https://github.com/unslothai/llama.cpp) at its K3 PR
(built on top of [llama.cpp PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185)):

```bash
git clone https://github.com/unslothai/llama.cpp
cd llama.cpp && git fetch origin pull/48/head:kimi-k3 && git checkout kimi-k3
cmake -B build -DGGML_METAL=ON        # Apple Silicon; use -DGGML_CUDA=ON on NVIDIA
cmake --build build --config Release -j --target llama-server

# REAP640: point at the first shard; REAP576: point at the single file
./build/bin/llama-server -m REAP640-IQ1_S/Kimi-K3-REAP640-IQ1_S-00001-of-00010.gguf \
    --port 8090 -ngl 99 -c 131072 --jinja --cache-reuse 0 \
    --temp 1.0 --top-p 0.95
```

- `--cache-reuse 0` is **required**: partial prefix-cache reuse corrupts the
  KDA recurrent state (known issue, see the PR discussion).
- K3 is thinking-only; reasoning arrives in `reasoning_content`. Control depth
  with `chat_template_kwargs: {"thinking_effort": "low" | "high" | "max"}`.
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

## How they were made

Expert saliency and keep-list planning use pipenetwork's
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) scripts
(`reap_calibrate.py` / `reap_plan.py` — REAP saliency `gate·‖expert output‖`
streamed layer-by-layer over the 1.56 TB MXFP4 source), with the calibration
mix swapped to English + code. The GGUF surgery is
[a small script](https://github.com/01554/kimi-k3-gguf-prune): a byte-slab
slice along the outermost expert axis (quantization blocks never cross expert
boundaries ⇒ no requantization, zero added quant error), router rows and
`exp_probs_b` renumbered to keep order. Identity-prune is byte-identical,
pinned by tests. Surviving experts are byte-identical to the Unsloth quants
they came from.

Full write-up — how it was built, what failed along the way, verification:
[English](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac#english-version) /
[日本語](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac).

Credits: [Moonshot AI](https://huggingface.co/moonshotai) (Kimi-K3, Kimi Code
CLI), [Unsloth](https://huggingface.co/unsloth) (dynamic quants whose protected
router/norms these builds inherit), [Cerebras
REAP](https://github.com/CerebrasResearch/reap) (saliency criterion),
[kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) (calibration
machinery and the measured warnings these builds steer by).

## 日本語の説明

Moonshot AIの2.8兆パラメータモデル Kimi-K3 を、Mac Studio(512GB)1台で動く
サイズに枝刈りしたビルド集です。同じ512GBの予算を「expert多め×低bit」で使う
REAP640-IQ1_S(441GB、SWE-Lancer 8タスク検証済み)と、「expert少なめ×高bit」で
使うREAP576-IQ2_XXS(478GB、検証進行中)の2つが入っています。

リポジトリ丸ごとダウンロードすると両方(約920GB)落ちてくるので、上の
`--include` 付きコマンドでどちらか片方だけ取得してください。

英語+コード校正のため中国語・日本語は意図的に壊れています。日本語用途は
[日本語校正版](https://huggingface.co/hellohazime/Kimi-K3-REAP640ja-IQ1_S-GGUF)へ。

経緯と実測の詳細:
[Kimi K3を441GBに枝刈りして、Mac Studio 1台で動かした](https://zenn.dev/hellohazime/articles/kimi_k3_reap640_512gb_mac)
