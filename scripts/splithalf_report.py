#!/usr/bin/env python3
"""Source-level split-half validation of expert selection (plans/README.md).

Splits the calibration by source domain, derives the keep-set from each
half, and reports keep overlap plus oracle retention (how much of half B's
saliency mass half A's selection captures, vs B's own optimal). Domain
splits are a distribution shift, i.e. deliberately harsher than the random
split-half some cards report — the result bounds domain sensitivity.

  scripts/splithalf_report.py            # the shipped K3 builds
"""

import json

import numpy as np


def topn(sal, n):
    return [set(np.argsort(sal[l])[::-1][:n]) for l in range(sal.shape[0])]


def retention(sal_b, keep_a, keep_b):
    r = []
    for l in range(sal_b.shape[0]):
        if sal_b[l].sum() <= 0:
            continue
        r.append(sal_b[l][list(keep_a[l])].sum()
                 / max(sal_b[l][list(keep_b[l])].sum(), 1e-12))
    return float(np.mean(r)), float(np.min(r))


def report(per_source, labels, n, partitions):
    for a_idx, b_idx in partitions:
        sal_a = per_source[list(a_idx)].sum(0)
        sal_b = per_source[list(b_idx)].sum(0)
        ka, kb = topn(sal_a, n), topn(sal_b, n)
        ov = float(np.mean([len(a & b) / n for a, b in zip(ka, kb)]))
        r_ab = retention(sal_b, ka, kb)
        r_ba = retention(sal_a, kb, ka)
        name = "+".join(labels[i] for i in a_idx) + " vs " + "+".join(labels[i] for i in b_idx)
        print(f"  {name:<38} overlap {ov*100:5.1f}%  "
              f"retention {r_ab[0]*100:5.1f}%/{r_ba[0]*100:5.1f}% "
              f"(min {min(r_ab[1], r_ba[1])*100:.1f}%)")


def verify_plan(plan_file, npz_file, n):
    plan = json.load(open(plan_file))
    sal = np.load(npz_file, allow_pickle=True)["saliency"]
    ok = all(set(plan["layers"][str(l)]["keep"]) == set(np.argsort(sal[l])[::-1][:n])
             for l in range(sal.shape[0]) if str(l) in plan["layers"])
    print(f"provenance {plan_file}: plan == top-{n} of {npz_file} -> {ok}")


def main():
    e = np.load("out/reap_saliency_encode.npz", allow_pickle=True)
    labels = [str(x) for x in e["source_labels"]]
    parts = [((i,), tuple(j for j in range(len(labels)) if j != i))
             for i in range(len(labels))]
    for n, build in ((640, "REAP640"), (576, "REAP576")):
        print(f"== {build} (keep-{n}) source-level split-half")
        report(e["per_source"], labels, n, parts)

    t = np.load("out/reap_saliency_tagged.npz", allow_pickle=True)
    tl = [str(x) for x in t["source_labels"]]
    pair = (tl.index("lang-ja"), tl.index("chinese"))
    print("== REAP640ja (keep-640) lang-ja vs chinese")
    report(t["per_source"], tl, 640, [((pair[0],), (pair[1],))])

    verify_plan("plans/reap_plan_640.json", "out/reap_saliency_encode.npz", 640)
    verify_plan("plans/reap_plan_576.json", "out/reap_saliency_encode.npz", 576)
    verify_plan("plans/reap_plan_640ja.json", "out/sal_lang-ja_chinese.npz", 640)


if __name__ == "__main__":
    main()
