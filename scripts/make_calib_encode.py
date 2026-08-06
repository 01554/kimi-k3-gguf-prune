#!/usr/bin/env python3
"""English + code calibration corpus for REAP saliency on Kimi-K3.

  KIMI_K3_MLX_SCRIPTS=/path/to/kimi-k3-mlx/scripts \
  scripts/make_calib_encode.py --out out/calib_encode.txt --mb 12 \
      --src /path/to/kimi-k3-src

This deployment serves coding agents in English only, so the mix drops every
other language on purpose. kimi-k3-mlx measured what that trade buys and costs:
the en+code subset build retained 68.4% of saliency mass vs 59.1% for the
11-source mixed corpus — and collapsed entirely on Chinese, which is the point:
whatever the corpus under-represents is what the prune deletes.

Implemented by monkeypatching MIX in kimi-k3-mlx's make_calib.py rather than
editing it: that file belongs to the other repo and its default mix is its own
deployment choice; this repo's choice lives here.

Mix (token shares, same 4:3 code:web ratio as the measured en+code subset):
    35%  code-multi    OpenCoder annealing corpus ("multi" is the upstream
                       label; measured content is ~62% detected-Python, ~0%
                       other named languages -- effectively algorithmic Python)
    20%  code-python   real Python files
    45%  web-en        FineWeb
"""

import os
import sys
from pathlib import Path

def _find_mlx_scripts():
    env = os.environ.get("KIMI_K3_MLX_SCRIPTS")
    here = Path(__file__).resolve()
    candidates = [Path(env)] if env else [
        here.parent.parent / "kimi-k3-mlx" / "scripts",         # git submodule
        here.parent.parent.parent / "kimi-k3-mlx" / "scripts",  # sibling clone
    ]
    for c in candidates:
        if (c / "make_calib.py").exists():
            return c
    raise SystemExit(
        "kimi-k3-mlx scripts not found. Run `git submodule update --init`, "
        "clone PipeNetwork/kimi-k3-mlx next to this repo, or set KIMI_K3_MLX_SCRIPTS."
    )

MLX_SCRIPTS = _find_mlx_scripts()

sys.path.insert(0, str(MLX_SCRIPTS))
import make_calib  # noqa: E402

make_calib.MIX = [
    ("code-multi", dict(path="OpenCoder-LLM/opc-annealing-corpus",
                        name="algorithmic_corpus", split="train"), "text", 0.35, None),
    ("code-python", dict(path="codeparrot/codeparrot-clean-valid",
                         split="train"), "content", 0.20, None),
    ("web-en", dict(path="HuggingFaceFW/fineweb",
                    name="sample-10BT", split="train"), "text", 0.45, None),
]

if __name__ == "__main__":
    make_calib.main()
