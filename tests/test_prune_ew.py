"""Pin the combined expert+width slicer on a tiny synthetic GGUF.

Width cuts add a second silent failure mode on top of router renumbering:
gate/up and down must select the SAME width blocks per expert, and the down
gather works on intra-row spans while gate/up works on whole row runs. So
(1) identity (all experts, all blocks) must be byte-identical, (2) a subset
must equal the naive numpy slice per projection geometry, and (3) both KV
fields must be rewritten.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "prune_gguf_ew.py"

ARCH = "tinymoe"
N_EXPERT, N_FF, N_EMBD, LAYERS = 8, 8, 4, 2
BLOCK = 2
N_BLOCKS = N_FF // BLOCK


def build_tiny(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), ARCH)
    w.add_uint32(f"{ARCH}.expert_count", N_EXPERT)
    w.add_uint32(f"{ARCH}.expert_feed_forward_length", N_FF)
    w.add_string("general.name", "tiny")
    tensors = {}
    for l in range(LAYERS):
        for proj in ("up", "gate"):
            name = f"blk.{l}.ffn_{proj}_exps.weight"
            t = rng.standard_normal((N_EXPERT, N_FF, N_EMBD)).astype(np.float32)
            tensors[name] = t
            w.add_tensor(name, t)
        name = f"blk.{l}.ffn_down_exps.weight"
        t = rng.standard_normal((N_EXPERT, N_EMBD, N_FF)).astype(np.float32)
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


def run_prune(src: Path, layers_plan: dict, out: Path, tmp: Path):
    plan_p = tmp / "plan_ew.json"
    plan_p.write_text(json.dumps({
        "num_experts": N_EXPERT, "n_ff_exp": N_FF, "block": BLOCK,
        "layers": layers_plan,
    }))
    subprocess.run(
        [sys.executable, str(SCRIPT), "prune", "--src", str(src),
         "--plan", str(plan_p), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    return GGUFReader(str(out))


def test_identity_is_byte_identical(tmp_path):
    src = tmp_path / "tiny.gguf"
    ref = build_tiny(src)
    plan = {str(l): {"keep": list(range(N_EXPERT)),
                     "wblocks": {str(e): list(range(N_BLOCKS))
                                 for e in range(N_EXPERT)}}
            for l in range(LAYERS)}
    r = run_prune(src, plan, tmp_path / "out.gguf", tmp_path)
    assert int(r.fields[f"{ARCH}.expert_count"].contents()) == N_EXPERT
    assert int(r.fields[f"{ARCH}.expert_feed_forward_length"].contents()) == N_FF
    for t in r.tensors:
        want = ref.get(t.name)
        if want is None:
            continue
        got = np.asarray(t.data).reshape(-1).view(np.uint8)
        assert got.tobytes() == want.reshape(-1).view(np.uint8).tobytes(), t.name


def test_subset_matches_naive_numpy_slice(tmp_path):
    src = tmp_path / "tiny.gguf"
    ref = build_tiny(src)
    # non-monotonic keep order + different width blocks per expert
    keep = [5, 1, 6]
    wb = {5: [0, 2, 3], 1: [1, 2, 3], 6: [0, 1, 3]}
    n_wb = 3
    n_ff_new = n_wb * BLOCK
    plan = {str(l): {"keep": keep,
                     "wblocks": {str(e): wb[e] for e in keep}}
            for l in range(LAYERS)}
    r = run_prune(src, plan, tmp_path / "out.gguf", tmp_path)
    assert int(r.fields[f"{ARCH}.expert_count"].contents()) == len(keep)
    assert int(r.fields[f"{ARCH}.expert_feed_forward_length"].contents()) == n_ff_new
    by_name = {t.name: t for t in r.tensors}
    for l in range(LAYERS):
        for proj in ("up", "gate"):
            name = f"blk.{l}.ffn_{proj}_exps.weight"
            got = np.asarray(by_name[name].data).reshape(len(keep), n_ff_new, N_EMBD)
            want = np.stack([
                ref[name][e].reshape(N_BLOCKS, BLOCK, N_EMBD)[wb[e]]
                .reshape(n_ff_new, N_EMBD)
                for e in keep])
            np.testing.assert_array_equal(got, want, err_msg=name)
        name = f"blk.{l}.ffn_down_exps.weight"
        got = np.asarray(by_name[name].data).reshape(len(keep), N_EMBD, n_ff_new)
        want = np.stack([
            ref[name][e].reshape(N_EMBD, N_BLOCKS, BLOCK)[:, wb[e]]
            .reshape(N_EMBD, n_ff_new)
            for e in keep])
        np.testing.assert_array_equal(got, want, err_msg=name)
        got = np.asarray(by_name[f"blk.{l}.ffn_gate_inp.weight"].data) \
            .reshape(len(keep), N_EMBD)
        np.testing.assert_array_equal(got, ref[f"blk.{l}.ffn_gate_inp.weight"][keep])
        got = np.asarray(by_name[f"blk.{l}.exp_probs_b.bias"].data).reshape(len(keep))
        np.testing.assert_array_equal(got, ref[f"blk.{l}.exp_probs_b.bias"][keep])


def test_non_uniform_wblocks_rejected(tmp_path):
    src = tmp_path / "tiny.gguf"
    build_tiny(src)
    # uniform within each layer, different across layers -> must be refused
    plan = {"0": {"keep": [0, 1], "wblocks": {"0": [0, 1, 2], "1": [0, 1, 2]}},
            "1": {"keep": [0, 1], "wblocks": {"0": [0, 1], "1": [0, 1]}}}
    plan_p = tmp_path / "plan_ew.json"
    plan_p.write_text(json.dumps({
        "num_experts": N_EXPERT, "n_ff_exp": N_FF, "block": BLOCK,
        "layers": plan,
    }))
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "prune", "--src", str(src),
         "--plan", str(plan_p), "--out", str(tmp_path / "out.gguf")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "non-uniform" in (res.stderr + res.stdout)
