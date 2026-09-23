"""Diagnose control 4's failure on HAR (E25 sec A, control 4).

The restated control reads acc(shuffled) = 0.286 on HAR against a chance of
0.1667 and a bar of 0.036. Class imbalance is ruled out by measurement: the
subject-disjoint test sets are near-balanced and the best constant predictor
scores 0.191.

FOUR VARIANTS, so the diagnosis is not a guess:

  A  label-vector shuffle     what the control does now (e25a_curve.py:254)
  B  feature-row shuffle      breaks the pairing from the other side
  C  class RELABEL map        the hypothesis: a consistent renaming leaves
                              fixed points, so the probe learns perfectly under
                              renamed classes and scores the fixed points' share
  D  no shuffle               the ceiling this cell's draw can reach

C is included because its arithmetic fits the observation suspiciously well (two
fixed points x 0.17 x 0.8 = 0.27 against 0.286 on HAR; one x 0.10 x 0.9 = 0.09
against 0.094 on MNIST). The IMPLEMENTATION already rules it out --
`rng.permutation(y_tr[sel])` permutes the label VECTOR and is not a consistent
renaming, verified -- but a hypothesis that predicts two numbers to within 0.02
is worth measuring rather than arguing away, and if C reproduces the observation
while A is what runs, the coincidence needs its own explanation.

PER-DRAW, not pooled. Fixed-point structure predicts a large spread across draws
-- near chance on some, well above on others -- and a pooled mean hides exactly
that (catch 26's shape).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,     # noqa: E402
                                    PROBE_SUBSET_SEED)
from scripts.cure_screen import BATCH                                              # noqa: E402
from scripts.e25a_curve import fit_logreg, score, balanced_draw, DRAW_SEED         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="s72_off", choices=["s72_off", "e18_pmd_mlp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--out", default="runs/e25/a/c4_diagnosis.json")
    a = ap.parse_args()
    dev = torch.device("cpu")

    if a.arm == "s72_off":
        from src.data.har_subject import HARSubjectBenchmark
        bench = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
        # CONSTRUCTION ASSERT, added 2026-09-21 after this script ran without one.
        # `har_subject`'s partition is x86-only: the laptop builds 5d047e4213d1
        # and the executed runs are 1104af185c87 (an exact window-count tie broken
        # differently on ARM). e25a_curve.py asserts this and refused; THIS script
        # did not, so its first run silently diagnosed a partition no checkpoint
        # was trained on. A diagnostic that does not verify its own construction
        # is the defect it was written to find, one level up.
        pfp = bench.partition_fingerprint()
        assert pfp == "1104af185c87", (
            f"HAR partition {pfp} is not the as-executed 1104af185c87; this is the "
            f"ARM/x86 split -- run on Modal, not the laptop")
        ck = f"runs/ckpt_e10off_ec_seed{a.seed}/mafc_seed{a.seed}_fp32"
        C, T = 6, 4
    else:
        from scripts.cure_screen import E18_FACTORY
        bench = E18_FACTORY["permuted"](a.seed)
        ck = f"runs/ckpt_e18_pmd_mlp_seed{a.seed}/mafc_seed{a.seed}_fp32"
        C, T = 10, 4
    data = load_task_data(bench, n_tasks=5, probe_subset_seed=PROBE_SUBSET_SEED)
    m4 = load(ck, T)
    from sklearn.preprocessing import StandardScaler

    rows = []
    print("=" * 100)
    print(f"C4 DIAGNOSIS  arm={a.arm} seed={a.seed} C={C} chance={1/C:.4f} draws={a.draws}")
    print("=" * 100)
    for k in range(4):
        xtr, ytr, xte, yte = data[k]
        f_tr, _ = features_and_logits(m4, xtr, ytr, k, False, dev)
        f_te, _ = features_and_logits(m4, xte, yte, k, False, dev)
        Ztr, Zte = f_tr.numpy().astype(np.float64), f_te.numpy().astype(np.float64)
        y_tr, y_te = ytr.numpy(), yte.numpy()
        sc = StandardScaler().fit(Ztr)
        Xtr, Xte = sc.transform(Ztr), sc.transform(Zte)
        maj = float(np.bincount(y_te, minlength=C).max() / len(y_te))
        per = {v: [] for v in "ABCD"}
        fixed = []
        for d in range(a.draws):
            rng = np.random.default_rng(DRAW_SEED + 1000 * a.seed + 100 * k + d)
            n = min(a.n_per_class, int(np.bincount(y_tr, minlength=C).min()))
            sel = balanced_draw(y_tr, n, rng, C)
            Xs, ys = Xtr[sel], y_tr[sel]
            # A: label-vector shuffle (what the control does)
            W, b, *_ = fit_logreg(Xs, rng.permutation(ys), C)
            per["A"].append(score(W, b, Xte, y_te))
            # B: feature-row shuffle
            W, b, *_ = fit_logreg(Xs[rng.permutation(len(ys))], ys, C)
            per["B"].append(score(W, b, Xte, y_te))
            # C: class relabel map
            pmap = rng.permutation(C)
            fixed.append(int((pmap == np.arange(C)).sum()))
            W, b, *_ = fit_logreg(Xs, pmap[ys], C)
            per["C"].append(score(W, b, Xte, y_te))
            # D: no shuffle
            W, b, *_ = fit_logreg(Xs, ys, C)
            per["D"].append(score(W, b, Xte, y_te))
        r = {"task": k, "n_test": len(y_te), "chance": 1 / C, "majority_class_rate": maj,
             "n_per_class": n, "fixed_points_mean": float(np.mean(fixed)),
             **{f"{v}_mean": float(np.mean(per[v])) for v in "ABCD"},
             **{f"{v}_spread": float(max(per[v]) - min(per[v])) for v in "ABCD"},
             **{f"{v}_draws": per[v] for v in "ABCD"}}
        rows.append(r)
        print(f"  task {k} (n_test {len(y_te)}, majority {maj:.3f}, chance {1/C:.3f}):")
        for v, name in (("A", "label-vector shuffle"), ("B", "feature-row shuffle"),
                        ("C", "class relabel map   "), ("D", "no shuffle          ")):
            print(f"    {v} {name} mean {np.mean(per[v]):.4f}  spread {max(per[v])-min(per[v]):.4f}"
                  f"  range [{min(per[v]):.3f}, {max(per[v]):.3f}]")
        print(f"      mean fixed points of the relabel map: {np.mean(fixed):.2f} of {C}")

    print("\n  POOLED over the four tasks:")
    for v, name in (("A", "label-vector shuffle (the control)"), ("B", "feature-row shuffle"),
                    ("C", "class relabel map"), ("D", "no shuffle")):
        vals = [x for r in rows for x in r[f"{v}_draws"]]
        print(f"    {v} {name:<36} {np.mean(vals):.4f}  [{min(vals):.3f}, {max(vals):.3f}]")
    print(f"    chance {1/C:.4f} | mean majority-class rate "
          f"{np.mean([r['majority_class_rate'] for r in rows]):.4f}")
    out = {"arm": a.arm, "seed": a.seed, "n_classes": C, "chance": 1 / C,
           "draws": a.draws, "rows": rows,
           "note": ("A is what control 4 runs. C is the hypothesis whose arithmetic fits the "
                    "observation; the implementation already rules it out (permutation of the "
                    "label VECTOR, not a consistent renaming), and this measures it anyway.")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
