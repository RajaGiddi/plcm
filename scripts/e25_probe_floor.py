"""The probe floor -- how far `refit_probe` stops short of its own optimum.

RULING (2026-09-21): `refit_probe` is NOT changed. Every refit number in the
paper comes from it, and re-fitting at a tighter tolerance three days before
submission would invalidate all of them. What is wrong is the WORDING -- the
docstring says "to optimality", sec 3.1 says "to convergence", and only Appendix
D's "to the library's default tolerance" is true. This script measures the gap
so the corrected sentence can carry a number.

Per cell it fits the SAME features twice:

    default   sklearn LogisticRegression(C=1, max_iter=5000)      <- what refit_probe does
    tight     the same with tol=1e-10                             <- its own optimum

and reports the accuracy difference on the two refits the decomposition uses,
the era ceiling and the final-encoder refit, plus the difference it induces in
F_enc = refit(theta_k) - refit(theta_T).

DIRECTION. A probe that stops short recovers slightly LESS, which inflates F_enc
and deflates the readout share -- the direction sec 3.1's lower-bound argument
already covers. Measured, not assumed: the sign is reported per cell and counted.

The objective is reported beside the accuracy because accuracy is discrete in
test samples and cannot resolve a small optimisation gap, while the objective
can. That is the second witness `probe_n_iter` could never be: an iteration
count compared against the CAP cannot see termination on TOLERANCE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, PROBE_SUBSET_SEED, LBFGS_MAX_ITER)
from scripts.cure_screen import E18_FACTORY, BATCH                                   # noqa: E402

SCRATCH = {
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",  "runs/e10off_ec_seed{s}",  "har_subject", 6,  5),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted",   10, 5),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted", 10, 5),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_rmd_mlp_seed{s}", "rotated",   10, 5),
}
PRETRAINED = {"e12_base": ("runs/ckpt_e12_base_seed{s}/mafc_seed{s}", 5, 20),
              "e14_base": ("runs/ckpt_e14_base_seed{s}/mafc_seed{s}", 5, 20)}
HAR_PARTITION = "1104af185c87"
TIGHT_TOL = 1e-10


def fit_both(Ztr, ytr, Zte, yte, n_classes):
    """(default, tight) fits of the SAME features. Default is refit_probe's call."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ztr)
    Xtr, Xte = sc.transform(Ztr), sc.transform(Zte)

    def obj(clf):
        Z = Xtr @ clf.coef_.T + clf.intercept_
        m = Z.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(Z - m).sum(1))
        return float(lse.sum() - Z[np.arange(len(ytr)), ytr].sum()
                     + 0.5 * float((clf.coef_ ** 2).sum()))

    out = {}
    for tag, kw in (("default", {}), ("tight", {"tol": TIGHT_TOL})):
        clf = LogisticRegression(max_iter=LBFGS_MAX_ITER, solver="lbfgs", C=1.0, **kw).fit(Xtr, ytr)
        out[tag] = {"acc": float((clf.predict(Xte) == yte).mean()),
                    "objective": obj(clf), "n_iter": int(clf.n_iter_[0])}
    out["acc_delta"] = out["default"]["acc"] - out["tight"]["acc"]
    out["objective_excess"] = out["default"]["objective"] - out["tight"]["objective"]
    return out


def main():
    ap = argparse.ArgumentParser(description="probe floor: default vs tight tolerance")
    ap.add_argument("--arm", required=True, choices=sorted(SCRATCH) + sorted(PRETRAINED))
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    root = args.ckpt_root.rstrip("/")
    pre = args.arm in PRETRAINED
    rows = []
    print("=" * 104 + f"\nPROBE FLOOR  arm={args.arm}  default tol 1e-4 (refit_probe) vs tight "
          f"{TIGHT_TOL:.0e}\n" + "=" * 104)

    for s in seeds:
        if pre:
            from src.data.split_cifar100 import SplitCIFAR100Benchmark
            from scripts.e12_decompose import load_era as load_pre, features_and_logits as fal_pre
            from torch.utils.data import DataLoader
            ck_t, n_classes, num_tasks = PRETRAINED[args.arm]
            ckpt = ck_t.replace("runs/", root + "/", 1).format(s=s)
            bench = SplitCIFAR100Benchmark(num_tasks=num_tasks, batch_size=BATCH,
                                           root=args.data_root, seed=s,
                                           remap_labels=True, download=False)
            T = num_tasks - 1
            m_final = load_pre(ckpt, T, device)
            tasks = range(T if args.max_tasks is None else min(T, args.max_tasks))
            for k in tasks:
                tr, te = bench.get_task_loaders(k)
                tr = DataLoader(tr.dataset, batch_size=BATCH, shuffle=False)
                m_era = load_pre(ckpt, k, device)
                fe_tr, _, y_tr = fal_pre(m_era, tr, k, False, device)
                fe_te, _, y_te = fal_pre(m_era, te, k, False, device)
                f4_tr, _, y_tr2 = fal_pre(m_final, tr, k, False, device)
                assert torch.equal(y_tr, y_tr2), "feature loader not deterministic"
                f4_te, _, _ = fal_pre(m_final, te, k, False, device)
                rows.append(cell_row(args.arm, s, k, fe_tr, fe_te, f4_tr, f4_te,
                                     y_tr, y_te, n_classes))
        else:
            ck_t, run_t, kind, n_classes, num_tasks = SCRATCH[args.arm]
            ck_t = ck_t.replace("runs/", root + "/", 1)
            run_t = run_t.replace("runs/", root + "/", 1)
            arm = json.load(open(f"{run_t.format(s=s)}/mafc_results.json"))["arm"]
            assert arm["era_checkpoints"] is True, arm
            if kind == "har_subject":
                from src.data.har_subject import HARSubjectBenchmark
                bench = HARSubjectBenchmark(num_tasks=num_tasks, root=".", batch_size=BATCH)
                assert bench.partition_fingerprint() == HAR_PARTITION
            else:
                bench = E18_FACTORY[kind](s)
                assert arm["content_fingerprint"] == bench.content_fingerprint()
            data = load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=args.probe_seed)
            d = ck_t.format(s=s)
            m4 = load(d, num_tasks - 1)
            tasks = range((num_tasks - 1) if args.max_tasks is None
                          else min(num_tasks - 1, args.max_tasks))
            for k in tasks:
                xtr, ytr, xte, yte = data[k]
                m_era = load(d, k)
                g = assert_path_identity(m4, xte, k, False, device, label=f"s{s}/T{k}", verbose=False)
                assert g["p_b"], (s, k)
                fe_tr, _ = features_and_logits(m_era, xtr, ytr, k, False, device)
                fe_te, _ = features_and_logits(m_era, xte, yte, k, False, device)
                f4_tr, _ = features_and_logits(m4, xtr, ytr, k, False, device)
                f4_te, _ = features_and_logits(m4, xte, yte, k, False, device)
                rows.append(cell_row(args.arm, s, k, fe_tr, fe_te, f4_tr, f4_te,
                                     ytr, yte, n_classes))

    summarize(args.arm, rows, args.out)
    return 0


def cell_row(arm, s, k, fe_tr, fe_te, f4_tr, f4_te, y_tr, y_te, n_classes):
    to = lambda t: t.numpy().astype(np.float64)
    yt, ye = y_tr.numpy(), y_te.numpy()
    ceil = fit_both(to(fe_tr), yt, to(fe_te), ye, n_classes)
    fin = fit_both(to(f4_tr), yt, to(f4_te), ye, n_classes)
    r = {"arm": arm, "seed": s, "task": k, "n_test": int(ye.shape[0]),
         "dim": int(f4_tr.shape[1]), "ceiling": ceil, "final": fin,
         "F_enc_default": ceil["default"]["acc"] - fin["default"]["acc"],
         "F_enc_tight": ceil["tight"]["acc"] - fin["tight"]["acc"]}
    r["F_enc_delta"] = r["F_enc_default"] - r["F_enc_tight"]
    print(f"  s{s} k{k}: ceiling {ceil['default']['acc']:.4f}/{ceil['tight']['acc']:.4f} "
          f"(iters {ceil['default']['n_iter']}/{ceil['tight']['n_iter']}, excess "
          f"{ceil['objective_excess']:.3f}) | final {fin['default']['acc']:.4f}/"
          f"{fin['tight']['acc']:.4f} | F_enc {r['F_enc_default']:+.4f}/"
          f"{r['F_enc_tight']:+.4f} (delta {r['F_enc_delta']:+.4f})")
    return r


def summarize(arm, rows, out):
    d = np.array([r["F_enc_delta"] for r in rows])
    ca = np.array([r["ceiling"]["acc_delta"] for r in rows])
    fa = np.array([r["final"]["acc_delta"] for r in rows])
    ex = np.array([r["ceiling"]["objective_excess"] for r in rows]
                  + [r["final"]["objective_excess"] for r in rows])
    it = [r["ceiling"]["default"]["n_iter"] for r in rows] + [r["final"]["default"]["n_iter"] for r in rows]
    itt = [r["ceiling"]["tight"]["n_iter"] for r in rows] + [r["final"]["tight"]["n_iter"] for r in rows]
    s = {"arm": arm, "n_cells": len(rows), "tight_tol": TIGHT_TOL,
         "max_abs_F_enc_delta": float(np.abs(d).max()), "mean_F_enc_delta": float(d.mean()),
         "max_abs_acc_delta": float(max(np.abs(ca).max(), np.abs(fa).max())),
         "objective_excess_max": float(ex.max()), "objective_excess_mean": float(ex.mean()),
         "n_iter_default_median": int(np.median(it)), "n_iter_tight_median": int(np.median(itt)),
         "F_enc_inflated_cells": int((d > 0).sum()), "F_enc_deflated_cells": int((d < 0).sum()),
         "F_enc_unchanged_cells": int((d == 0).sum()),
         "ruling": ("refit_probe is unchanged. This is the floor its default tolerance "
                    "imposes, so the paper's sentence can carry a number."),
         "rows": rows}
    print(f"\n  {arm}: {len(rows)} cells | max |F_enc delta| {s['max_abs_F_enc_delta']:.4f} "
          f"| max |acc delta| {s['max_abs_acc_delta']:.4f} | objective excess max "
          f"{s['objective_excess_max']:.3f} | iters {s['n_iter_default_median']} vs "
          f"{s['n_iter_tight_median']}")
    print(f"  direction: F_enc inflated in {s['F_enc_inflated_cells']}, deflated in "
          f"{s['F_enc_deflated_cells']}, unchanged in {s['F_enc_unchanged_cells']}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(s, open(out, "w"), indent=2, default=float)
    print(f"  wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
