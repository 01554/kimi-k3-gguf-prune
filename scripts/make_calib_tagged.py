#!/usr/bin/env python3
"""Tagged multi-domain calibration corpus: one calibration, many builds.

  KIMI_K3_MLX_SCRIPTS=/path/to/kimi-k3-mlx/scripts \
  scripts/make_calib_tagged.py --out out/calib_tagged.txt --mb 12 --src /path/to/kimi-k3-src

reap_calibrate.py records saliency per source label (via the .sources.json
manifest make_calib writes), and reap_subset.py can then sum any subset of
labels into a targeted saliency file WITHOUT re-running calibration. So this
mix exists to cover every domain a future build might want, with fat shares for
the ones that need statistical weight:

    30%  lang-ja       C4 ja, kana-fraction filtered — the ja-specialized build
                       is the next planned variant, so ja gets the largest share
    25%  web-en        FineWeb
    25%  code          OpenCoder annealing (15%) + real Python (10%)
    10%  chinese       C4 zh — kept fat-ish because ja borrows zh experts
                       (42.8% top-expert overlap, 1.6x chance)
    10%  tail          ko/de/fr/es/ru at 2% each, so those buckets exist at all

Shares are token shares (make_calib's default when the tokenizer loads).
"""

import os
import sys
from pathlib import Path

MLX_SCRIPTS = Path(os.environ.get(
    "KIMI_K3_MLX_SCRIPTS",
    Path(__file__).resolve().parent.parent.parent / "kimi-k3-mlx" / "scripts",
))
if not (MLX_SCRIPTS / "make_calib.py").exists():
    raise SystemExit(
        f"kimi-k3-mlx scripts not found at {MLX_SCRIPTS}; "
        "set KIMI_K3_MLX_SCRIPTS to that repo's scripts/ directory"
    )

sys.path.insert(0, str(MLX_SCRIPTS))
import make_calib  # noqa: E402

make_calib.MIX = [
    ("lang-ja", dict(path="allenai/c4", name="ja", split="train"),
     "text", 0.30, make_calib._is_real_japanese),
    ("web-en", dict(path="HuggingFaceFW/fineweb",
                    name="sample-10BT", split="train"), "text", 0.25, None),
    ("code-multi", dict(path="OpenCoder-LLM/opc-annealing-corpus",
                        name="algorithmic_corpus", split="train"), "text", 0.15, None),
    ("code-python", dict(path="codeparrot/codeparrot-clean-valid",
                         split="train"), "content", 0.10, None),
    ("chinese", dict(path="allenai/c4", name="zh", split="train"),
     "text", 0.10, make_calib._is_real_chinese),
] + [
    (f"lang-{lg}", dict(path="allenai/c4", name=lg, split="train"),
     "text", 0.10 / 5, None)
    for lg in ("ko", "de", "fr", "es", "ru")
]

if __name__ == "__main__":
    make_calib.main()
