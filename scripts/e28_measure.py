"""E28 sec 3 -- the measurement that actually tests the method.

Per arm, seed, task k, and held-out swap h (never seen in training):

  clean          deployed shared head on unswapped task-k input
  swapped        the same head on h-swapped input, NO REPAIR
  refit clean    linear probe fit and tested on unswapped features
  refit swapped  linear probe fit and tested on h-swapped features
  Delta_refit    refit clean - refit swapped        <- the primary quantity
  delta          E27's equivariance defect: affine A fit Z_swapped -> Z_clean,
                 residual on test, with ||A - I||_F / ||I||_F beside it.
                 A ~ I with small delta is INVARIANCE; A != I with small delta
                 is EQUIVARIANCE.
  swap-ID probe  can a fresh linear probe read WHICH held-out swap was applied,
                 from the deployed feature? 20-way, chance 0.05.

ON THE THIRD NUMBER. The contract asks for "the auxiliary head's accuracy on
held-out swaps". THAT HEAD IS NOT IN ANY CHECKPOINT: `_aux_perm_loss` builds it
on the TRAINER (trainer.py), and checkpoints save `model.state_dict()`, so the
trained head is gone. Recorded as a defect in the trainer change -- a head whose
accuracy the contract wants to read must be saved -- and fixed for future runs.

What is measured instead answers the question the contract gives for it,
"whether channel identity is really in the tensor", and answers it better: a
FRESH probe fit to the deployed feature. The trained head's accuracy conflates
what the tensor contains with how well that particular head was fit, and it
exists only for B2, so it cannot compare arms. A fresh probe applies to B0, B1
and B2 alike, which turns a single number into the comparison the reading needs.

PRIMARY SET is tasks 3 and 4, by construction: the tasks whose frame carries a
permutation. Tasks 0-2 are reported beside it against their own floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import BATCH                                               # noqa: E402
from scripts.d1_fit import Affine                                                    # noqa: E402
from scripts.e28_heldout import apply_perm, HAR_PARTITION                            # noqa: E402

N_CLASSES, NUM_TASKS = 6, 5
LAM = 1e-3
PRIMARY = (3, 4)

ARMS = {
    "b0_ep30":      "e28b_b0_ep30_seed{s}",
    "b1_ep30":      "e28b_b1_ep30_seed{s}",
    "b2_ep30":      "e28b_b2_ep30_seed{s}",
    "b2_p025_ep30": "e28b_b2_p025_ep30_seed{s}",
}
CKPT = {
    "b0_ep30":      "ckpt_e28b_b0_ep30_seed{s}/mafc_seed{s}_fp32",
    "b1_ep30":      "ckpt_e28b_b1_ep30_seed{s}/mafc_seed{s}_fp32",
    "b2_ep30":      "ckpt_e28b_b2_ep30_seed{s}/mafc_seed{s}_fp32",
    "b2_p025_ep30": "ckpt_e28b_b2_p025_ep30_seed{s}/mafc_seed{s}_fp32",
}


def swap_id_probe(F_by_h, y_n, rng):
    """20-way: which held-out swap produced this feature? Chance = 1/len(F_by_h)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n_h = len(F_by_h)
    per = min(400, F_by_h[0].shape[0])
    X = np.concatenate([F[:per] for F in F_by_h])
    y = np.concatenate([np.full(per, i) for i in range(n_h)])
    idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(sc.transform(X[tr]), y[tr])
    return float((clf.predict(sc.transform(X[te])) == y[te]).mean()), 1.0 / n_h


def main():
    ap = argparse.ArgumentParser(description="E28 sec 3 measurement")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--heldout", default="runs/e28/heldout.json")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    ho = json.load(open(args.heldout))
    H = [tuple(p) for p in ho["heldout"]]
    print("=" * 104 + f"\nE28 MEASURE  arm={args.arm} seed={args.seed}  heldout {ho['fingerprint']}\n" + "=" * 104)

    run = json.load(open(f"{root}/{ARMS[args.arm].format(s=args.seed)}/mafc_results.json"))
    arm = run["arm"]
    print(f"  arm identity: aug_family={arm.get('aug_family')} p={arm.get('aug_p')} "
          f"aux={arm.get('aux_perm_weight')} heldout={arm.get('aug_heldout_fingerprint')} "
          f"epochs={arm.get('epochs_per_task')}")
    assert arm.get("aug_heldout_fingerprint") in (None, ho["fingerprint"]), "held-out set mismatch"

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=BATCH)
    assert bench.partition_fingerprint() == HAR_PARTITION
    data = load_task_data(bench, n_tasks=NUM_TASKS, probe_subset_seed=args.probe_seed)
    T = NUM_TASKS - 1
    # THE ERA EPOCH INDEX IS NOT 9 HERE. `channel_decomp.load` defaults to epoch=9
    # because every arm before E28b trained 10 epochs per task; E28b trains 30, so
    # its era files are task{k}_epoch29.pt and the default raised FileNotFound on
    # all twelve jobs. Derive it from the run's OWN record rather than hardcoding a
    # second constant that the next budget change would break in the same way.
    ep = int(arm.get("epochs_per_task", 10)) - 1
    print(f"  era epoch index derived from the artifact: {ep} (epochs_per_task={arm.get('epochs_per_task')})")
    m4 = load(f"{root}/{CKPT[args.arm].format(s=args.seed)}", T, epoch=ep)
    rng = np.random.default_rng(20260922 + args.seed)

    rows, t0 = [], time.time()
    for k in range(NUM_TASKS):
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m4, xte, k, False, device, label=f"{args.arm}/s{args.seed}/T{k}", verbose=False)
        assert g["p_b"]
        f = lambda x, y: features_and_logits(m4, x, y, k, False, device)
        F_tr, _ = f(xtr, ytr); F_te, l_te = f(xte, yte)
        clean = float((l_te.argmax(1) == yte).float().mean())
        refit_clean = refit_probe(F_tr, ytr, F_te, yte, N_CLASSES)
        Zc_tr, Zc_te = F_tr.numpy().astype(np.float64), F_te.numpy().astype(np.float64)
        per_h, F_by_h = [], []
        for h in H:
            Fs_tr, _ = f(apply_perm(xtr, h), ytr)
            Fs_te, ls_te = f(apply_perm(xte, h), yte)
            Zs_tr, Zs_te = Fs_tr.numpy().astype(np.float64), Fs_te.numpy().astype(np.float64)
            mu, sd = Zs_tr.mean(0), Zs_tr.std(0) + 1e-8
            A = Affine.fit(Zs_tr, Zc_tr, mu, sd, LAM)
            d = float(np.linalg.norm(Zc_te - A.apply(Zs_te)) / max(np.linalg.norm(Zc_te), 1e-12))
            Ar = A.M_raw(); I = np.eye(Ar.shape[0])
            per_h.append({"h": list(h),
                          "swapped": float((ls_te.argmax(1) == yte).float().mean()),
                          "refit_swapped": refit_probe(Fs_tr, ytr, Fs_te, yte, N_CLASSES),
                          "delta": d,
                          "A_minus_I_rel": float(np.linalg.norm(Ar - I) / np.linalg.norm(I))})
            F_by_h.append(Zs_te)
        sid, chance = swap_id_probe(F_by_h, yte.numpy(), rng)
        m = lambda key: float(np.mean([p[key] for p in per_h]))
        dref = [refit_clean - p["refit_swapped"] for p in per_h]
        row = {"arm": args.arm, "seed": args.seed, "task": k, "primary": k in PRIMARY,
               "n_test": int(len(yte)), "clean": clean, "refit_clean": refit_clean,
               "swapped": m("swapped"), "swap_drop": clean - m("swapped"),
               "refit_swapped": m("refit_swapped"),
               "delta_refit": float(np.mean(dref)),
               "delta_refit_min": float(np.min(dref)), "delta_refit_max": float(np.max(dref)),
               "recovery": (float(np.mean(dref)) and
                            (m("refit_swapped") - m("swapped")) / max(refit_clean - m("swapped"), 1e-9)),
               "equivariance_delta": m("delta"), "A_minus_I_rel": m("A_minus_I_rel"),
               "swap_id_acc": sid, "swap_id_chance": chance,
               "res95": 1.96 * float(np.sqrt(0.25 / len(yte)))}
        rows.append(row)
        print(f"  k={k}{'*' if k in PRIMARY else ' '}: clean {clean:.3f} swapped {row['swapped']:.3f} "
              f"(drop {row['swap_drop']:+.3f}) | refit {refit_clean:.3f}->{row['refit_swapped']:.3f} "
              f"D_refit {row['delta_refit']:+.4f} | delta {row['equivariance_delta']:.3f} "
              f"||A-I|| {row['A_minus_I_rel']:.3f} | swap-ID {sid:.3f} (chance {chance:.3f})")

    prim = [r for r in rows if r["primary"]]
    out = {"arm": args.arm, "seed": args.seed, "heldout_fingerprint": ho["fingerprint"],
           "n_heldout": len(H), "epochs_per_task": arm.get("epochs_per_task"),
           "aug_p": arm.get("aug_p"), "aux_perm_weight": arm.get("aux_perm_weight"),
           "primary_tasks": list(PRIMARY),
           "primary": {k: float(np.mean([r[k] for r in prim]))
                       for k in ("clean", "swapped", "swap_drop", "refit_clean", "refit_swapped",
                                 "delta_refit", "equivariance_delta", "A_minus_I_rel", "swap_id_acc")},
           "aux_head_note": ("the trained auxiliary head is NOT in the checkpoint "
                             "(built on the trainer, not the model); swap_id_acc is a FRESH "
                             "probe on the deployed feature, which answers the same question "
                             "and applies to every arm"),
           "rows": rows, "seconds": round(time.time() - t0, 1)}
    p = out["primary"]
    print(f"\n  PRIMARY (tasks 3,4): clean {p['clean']:.3f} swapped {p['swapped']:.3f} | "
          f"refit {p['refit_clean']:.3f}->{p['refit_swapped']:.3f} | D_refit {p['delta_refit']:+.4f} | "
          f"delta {p['equivariance_delta']:.3f} ||A-I|| {p['A_minus_I_rel']:.3f} | swap-ID {p['swap_id_acc']:.3f}")
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"measure_{args.arm}_seed{args.seed}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"  wrote {fp}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
