"""Pin the pruning behaviour on a tiny synthetic GGUF.

The failure mode worth pinning is silent: a mis-renumbered router loads and
runs while sending every token to the wrong expert. So (1) an identity prune
must be byte-identical for every tensor, and (2) a subset prune must equal the
straightforward numpy slice in *keep order* — including the router rows and
routing bias, where order is the whole point.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "prune_gguf.py"

N_EXPERT, N_FF, N_EMBD, LAYERS = 8, 6, 4, 2


def build_tiny(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), "kimi-k3")
    w.add_uint32("kimi-k3.expert_count", N_EXPERT)
    w.add_string("general.name", "tiny")
    tensors = {}
    for l in range(LAYERS):
        for proj in ("up", "gate", "down"):
            name = f"blk.{l}.ffn_{proj}_exps.weight"
            t = rng.standard_normal((N_EXPERT, N_FF, N_EMBD)).astype(np.float32)
            tensors[name] = t
            w.add_tensor(name, t)
        name = f"blk.{l}.ffn_gate_inp.weight"
        t = rng.standard_normal((N_EXPERT, N_EMBD)).astype(np.float32)
        tensors[name] = t
        w.add_tensor(name, t)
        name = f"blk.{l}.exp_probs_b.bias"
        t = rng.standard_normal((N_EXPERT,)).astype(np.float32)
        tensors[name] = t
        w.add_tensor(name, t)
    w.add_tensor("token_embd.weight", rng.standard_normal((10, N_EMBD)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return tensors


def run_prune(src: Path, plan: dict, out: Path, tmp: Path):
    plan_p = tmp / "plan.json"
    plan_p.write_text(json.dumps({"keep": {str(k): v for k, v in plan.items()}}))
    subprocess.run(
        [sys.executable, str(SCRIPT), "prune", "--src", str(src),
         "--plan", str(plan_p), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    return GGUFReader(str(out))


def test_identity_prune_is_byte_identical(tmp_path):
    src = tmp_path / "tiny.gguf"
    ref = build_tiny(src)
    plan = {l: list(range(N_EXPERT)) for l in range(LAYERS)}
    r = run_prune(src, plan, tmp_path / "out.gguf", tmp_path)
    assert int(r.fields["kimi-k3.expert_count"].contents()) == N_EXPERT
    for t in r.tensors:
        got = np.asarray(t.data).reshape(-1).view(np.uint8)
        want = ref.get(t.name)
        if want is None:
            continue
        assert got.tobytes() == want.astype(np.float32).reshape(-1).view(np.uint8).tobytes(), t.name


def test_subset_prune_reorders_router_in_keep_order(tmp_path):
    src = tmp_path / "tiny.gguf"
    ref = build_tiny(src)
    # deliberately non-monotonic keep order: renumbering must follow it exactly
    keep = [5, 1, 6]
    plan = {l: keep for l in range(LAYERS)}
    r = run_prune(src, plan, tmp_path / "out.gguf", tmp_path)
    assert int(r.fields["kimi-k3.expert_count"].contents()) == len(keep)
    by_name = {t.name: t for t in r.tensors}
    for l in range(LAYERS):
        for proj in ("up", "gate", "down"):
            name = f"blk.{l}.ffn_{proj}_exps.weight"
            got = np.asarray(by_name[name].data).reshape(len(keep), N_FF, N_EMBD)
            np.testing.assert_array_equal(got, ref[name][keep])
        got = np.asarray(by_name[f"blk.{l}.ffn_gate_inp.weight"].data).reshape(len(keep), N_EMBD)
        np.testing.assert_array_equal(got, ref[f"blk.{l}.ffn_gate_inp.weight"][keep])
        got = np.asarray(by_name[f"blk.{l}.exp_probs_b.bias"].data).reshape(len(keep))
        np.testing.assert_array_equal(got, ref[f"blk.{l}.exp_probs_b.bias"][keep])


def test_non_uniform_plan_rejected(tmp_path):
    src = tmp_path / "tiny.gguf"
    build_tiny(src)
    plan = {0: [0, 1, 2], 1: [0, 1]}
    plan_p = tmp_path / "plan.json"
    plan_p.write_text(json.dumps({"keep": {str(k): v for k, v in plan.items()}}))
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "prune", "--src", str(src),
         "--plan", str(plan_p), "--out", str(tmp_path / "out.gguf")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "non-uniform" in (res.stderr + res.stdout)
