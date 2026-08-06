# kimi-k3-gguf-prune

REAP-prune Unsloth's **Kimi-K3 UD-IQ1_S** (594 GB, 896 experts) down to a GGUF
that fits and runs on a 512 GB M3 Ultra Mac Studio — targeting **640/896
experts ≈ 437 GB**.

## Why this shape

- Every K3 build that fits in 512 GB today is an MLX REAP build keeping only
  242–326 of 896 experts at 4 bit. Those builds deterministically degenerate on
  long agentic prompts (Kimi Code CLI, 24 tools, ~30k tokens) — measured, not
  guessed.
- Unsloth's dynamic 1-bit goes the other way: keep **all** experts, cut bits,
  and protect what matters (router and layernorms at full precision, attention
  and shared experts at 4–8 bit). Their published ablation: uniform low-bit
  quantization produces exactly the "infinite repetitions" failure we saw;
  protecting the sensitive 1–2% fixes it.
- This repo combines the two: Unsloth's per-tensor protection **plus** a mild
  REAP prune (71% of experts kept vs 27% in REAP73) to close the remaining
  594→512 GB gap.

## Size arithmetic

UD-IQ1_S ≈ 550 GB routed experts + 44 GB everything else
(2722.7B expert params × ~1.6 bpw checks out). One expert slab ≈ 6.7 MB × 92
MoE layers.

| keep | size | note |
|---|---|---|
| 896 | 594 GB | doesn't fit |
| 672 | ~456 GB | same footprint as the MLX 451 GB builds |
| **640** | **~437 GB** | default — headroom for KV + compute |


## Prerequisites: PipeNetwork/kimi-k3-mlx (the brains of the selection)

The calibration and expert-selection stages are **not this repo's code** — they
run [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx),
included here as a git submodule:

```bash
git submodule update --init   # pulls kimi-k3-mlx into ./kimi-k3-mlx
```

What lives where:

| stage | code | repo |
|---|---|---|
| corpus assembly | [make_calib.py](https://github.com/PipeNetwork/kimi-k3-mlx/blob/main/scripts/make_calib.py) (this repo's `make_calib_*.py` only swap the MIX) | kimi-k3-mlx |
| expert saliency measurement | [reap_calibrate.py](https://github.com/PipeNetwork/kimi-k3-mlx/blob/main/scripts/reap_calibrate.py) | kimi-k3-mlx |
| keep-list planning | [reap_plan.py](https://github.com/PipeNetwork/kimi-k3-mlx/blob/main/scripts/reap_plan.py) | kimi-k3-mlx |
| GGUF slab slicing + router renumbering | `scripts/prune_gguf.py` | this repo |

The upstream code is referenced rather than vendored because kimi-k3-mlx does
not currently ship a license for its own sources; a submodule keeps it under
its authors' terms. The saliency stage additionally needs `mlx` (Apple Silicon)
and reads the 1.56 TB MXFP4 source model.

## Pipeline

1. **Baseline probe** — replay a captured, deterministically-failing Kimi CLI
   request against *unpruned* IQ1_S via the Unsloth llama.cpp fork
   (`--jinja --cache-reuse 0`, mmap + CPU). If unpruned 1-bit already
   degenerates, pruning cannot help and the plan stops here.
2. **Saliency** — REAP scores from the HF source (MXFP4), streamed layer by
   layer (`kimi-k3-mlx/scripts/reap_calibrate.py`, peak ~58 GB RAM), calibrated
   on an **English + code** corpus only (this deployment needs nothing else;
   the en+code subset build measured 68.4% saliency retention vs 59.1% mixed).
3. **Plan** — `reap_plan.py --mode uniform`, keep 640 per layer. Uniform, not
   global: GGUF carries a single `expert_count` metadata key.
4. **Prune** (`scripts/prune_gguf.py`) — slice expert tensors along the expert
   axis as raw byte slabs (axis 0 is outermost ⇒ each expert is a contiguous
   slab; quantization blocks never cross expert boundaries ⇒ **no requantize,
   zero added quant error**). Reorder router rows + `exp_probs_b` to the keep
   order, update `expert_count`.
5. **Verify** — identity-prune must be byte-identical (`tests/`); then short
   generation, then the captured request, then 3 SWE-Lancer tasks via Kimi Code
   CLI pointed straight at `llama-server` (the fork has native K3 chat template,
   tool-call parsing and reasoning separation — no shim layer needed).

## The one trap

Pruning renumbers experts. Router row *i* of the pruned model must be old row
`keep[i]`, and `exp_probs_b` likewise, or the model loads and generates fluent
text while routing every token to the wrong expert. `tests/test_prune.py` pins
this the same way kimi-k3-mlx does: identity prune ⇒ byte-identical, and a
deliberately rotated router must be detected.

## Runtime settings (from Unsloth docs / PR #26185)

- `--jinja --cache-reuse 0` (KDA recurrent state corrupts under partial
  prefix-cache reuse), `--temp 1.0 --top-p 0.95`
- K3 is thinking-only (`preserve_thinking` always on); effort via
  `reasoning_effort` = low/high/max
