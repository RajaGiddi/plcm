"""E20 positive control -- the MLP probe must pass where the linear probe must fail.

Contract: docs/E20_prereg.md sec 4 (catch 25: a gate that has only ever passed
is indistinguishable from one that cannot fail, so before the MLP probe reads a
live cell it is shown to recover a target the linear probe provably cannot).

Target, on a feature matrix F with labels y:

    r  = z - mu_y(train)                          within-class residual, class means from TRAIN
    e1, e2 = top-2 principal directions of the POOLED train residuals
    b_i = [ e_i . r  >  median_y( e_i . r | train ) ]   per-class median split, balanced in every class
    y' = b1 XOR b2

    The XOR checkerboard on the two DOMINANT WITHIN-CLASS directions. Class-free
    twice over: the bits are residual quantities and each is balanced inside
    every class, so y' is balanced in every class and carries no information
    about which cluster a point is in -- the mechanism that made v1, v5 and v6
    linearly decodable (any function of the cluster is linearly separable in
    256-d). Learnable: the structure lives in the directions of largest
    within-class variance, not a random needle (v4). TWO printed conditions,
    both required, both stated before the live run:
        (i)  acc_linear(y') <= 0.55 on the live features;
        (ii) acc_mlp(y') - acc_linear(y') >= 0.15 for EVERY probe seed.
    The synthetic run uses ANISOTROPIC within-class noise (a few dominant
    directions), because a synthetic shaped unlike the live features validates
    nothing about them.

DESIGN HISTORY -- four corrections, every failure on disk, because a positive
control that has only been designed is not yet a control:
  v1 (y mod 2) XOR [PC1 > 0]              synthetic FAIL: linear 0.9635 -- PC1 is a
      function of the class under separated clusters; XOR of two class functions
      is a dichotomy of 10 means, linearly separable in 256-d.
  v2 (y mod 2) XOR [w.(z - mu_y) > 0]      synthetic PASS (0.51 / 0.84); LIVE FAIL:
      linear 0.679 -- real within-class distributions are skewed, a class-mean
      split is not 50/50, parity plus a per-class constant decodes it linearly.
  v3 (y mod 2) XOR [w.z > per-class median] synthetic PASS; LIVE FAIL: linear
      0.6725 -- the class geometry and the projection still interact linearly.
  v4 XOR of two random whitened half-spaces  linear 0.503 as designed, but the
      MLP read 0.54: an XOR planted in two random directions of a 256-d space is
      a needle no 512-unit MLP finds from 4000 samples. A target BOTH instruments
      fail is not a control either.
  v5 band-vs-tails on PC1                synthetic FAIL: linear 0.89 -- the "0.75 by
      geometry" bound assumed the other 255 coordinates carry nothing about the
      PC1 band; on clustered data the tails of PC1 ARE particular classes, and a
      hyperplane in the full space separates extreme clusters from middle ones.
      Any target that is a function of ONE direction inherits class structure
      through the others.
  v6 XOR of median-split PC1, PC2         synthetic FAIL: linear 0.758 -- the
      dominant PCs assign whole clusters to quadrants; v1's mechanism again.
  v7 (this): the target varies WITHIN clusters along the dominant within-class
      directions, balanced inside every class; the linear-fails half is a
      PRINTED CONDITION on the live features rather than an argument.
The lesson, in one line: a positive control for a stronger instrument must be a
case the weaker instrument fails and the stronger one passes, BOTH halves
verified as printed numbers on the live features -- never on a toy, never by
an argument about geometry.

Two modes, both recorded:
  --synthetic   Gaussian features (n=4000/2000, d=256) with a planted class
                structure, run locally before any job launches: shows the
                recipe works at all, with no checkpoint in the loop.
  --live        task-0 features at theta_T of the primary arm, on Modal, the
                first thing that runs before the live decomposition. Extraction
                routes through channel_decomp (catch 32).

Usage:
    python scripts/e20_positive_control.py --synthetic
    modal run modal_runner.py::analysis --argv "scripts/e20_positive_control.py --live --arm mlp"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.channel_decomp as cd

BAR = 0.15
LINEAR_CAP = 0.55      # condition (i): the weaker instrument must fail, on the live features
N_TR, N_TE, D, K = 4000, 2000, 256, 10


def xor_target(Ftr, ytr, Fte, yte):
    """The nonlinear target. Standardization, class means and the direction w
    are all fixed on TRAIN only (the probes' own discipline)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    Ztr, Zte = sc.transform(Ftr), sc.transform(Fte)
    from sklearn.decomposition import PCA
    classes = np.unique(ytr)
    mu = {c: Ztr[ytr == c].mean(0) for c in classes}                  # TRAIN only
    R_tr = Ztr - np.stack([mu[c] for c in ytr])
    R_te = Zte - np.stack([mu[c] for c in yte])
    E = PCA(n_components=2).fit(R_tr).components_                      # pooled residual PCs, TRAIN only
    P_tr, P_te = R_tr @ E.T, R_te @ E.T
    med = {c: np.median(P_tr[ytr == c], axis=0) for c in classes}      # per-class, TRAIN only
    B_tr = (P_tr > np.stack([med[c] for c in ytr])).astype(int)
    B_te = (P_te > np.stack([med[c] for c in yte])).astype(int)
    return B_tr[:, 0] ^ B_tr[:, 1], B_te[:, 0] ^ B_te[:, 1]


def run_control(Ftr, ytr, Fte, yte, label):
    ytr2, yte2 = xor_target(Ftr, ytr, Fte, yte)
    t = lambda a: torch.from_numpy(np.asarray(a))
    info = {}
    acc_lin = cd.refit_probe(t(Ftr), t(ytr2), t(Fte), t(yte2), 2, info=info)
    minfo = {}
    acc_mlp = cd.refit_probe_mlp(t(Ftr), t(ytr2), t(Fte), t(yte2), 2, info=minfo)
    per_seed = minfo["mlp_acc_per_seed"]
    margins = [a - acc_lin for a in per_seed]
    print(f"  [{label}] train/test gaps: linear {info.get('probe_gap'):+.4f}  MLP per seed {[round(g, 4) for g in minfo['mlp_gap_per_seed']]}"
          f"  (train acc = test acc + gap; a low MLP train acc is underfit, a high one with low test is overfit)")
    ok = all(m >= BAR for m in margins) and minfo["mlp_converged"] and acc_lin <= LINEAR_CAP
    print(f"  [{label}] target balance {yte2.mean():.3f} | linear {acc_lin:.4f} (n_iter {info.get('probe_n_iter')})"
          f" | MLP per seed {[round(a, 4) for a in per_seed]} (n_iter {minfo['mlp_n_iter']}, converged {minfo['mlp_converged']})")
    print(f"  [{label}] (i) linear {acc_lin:.4f} <= {LINEAR_CAP} -> {'ok' if acc_lin <= LINEAR_CAP else 'FAIL'};"
          f"  (ii) margins {[round(m, 4) for m in margins]} >= {BAR} every seed -> {'ok' if all(m >= BAR for m in margins) else 'FAIL'}"
          f"  => {'PASS' if ok else 'FAIL'}")
    return {"label": label, "acc_linear": acc_lin, "acc_mlp_per_seed": per_seed,
            "margins": margins, "bar": BAR, "converged": minfo["mlp_converged"],
            "n_iter": minfo["mlp_n_iter"], "pass": bool(ok)}


def synthetic():
    rng = np.random.default_rng(0)
    # planted class structure so that (y mod 2) is linearly decodable but not
    # trivially: class means on a random 10-simplex, isotropic noise
    means = rng.normal(size=(K, D)) * 0.6
    ytr = rng.integers(0, K, N_TR); yte = rng.integers(0, K, N_TE)
    # anisotropic within-class noise: a handful of dominant directions, as real
    # features have, so the residual-PC construction is exercised as it will be
    # on the live data rather than on an isotropic ball where it is a needle
    scale = np.concatenate([np.array([6.0, 4.0, 2.5, 1.5]), np.ones(D - 4) * 0.5])
    Q = np.linalg.qr(rng.normal(size=(D, D)))[0]
    noise = lambda n: (rng.normal(size=(n, D)) * scale) @ Q.T
    Ftr = means[ytr] + noise(N_TR)
    Fte = means[yte] + noise(N_TE)
    return run_control(Ftr, ytr, Fte, yte, "synthetic")


def live(arm: str, ckpt_root: str, device):
    from scripts.e16_decompose import ARMS, resolve
    from src.data.permuted_mnist import PermutedMNISTBenchmark
    seed = 42
    ckpt_tmpl, _ = resolve(arm, "mnist")
    ckpt_tmpl = ckpt_tmpl.replace("runs/", ckpt_root.rstrip("/") + "/", 1)
    data = cd.load_task_data(PermutedMNISTBenchmark(num_tasks=5, batch_size=256, seed=seed))
    xtr, ytr, xte, yte = data[0]
    m_final = cd.load(ckpt_tmpl.format(s=seed), 4)
    cd.assert_path_identity(m_final, xte, 0, False, device, label="e20-live", verbose=False)
    f_tr, _ = cd.features_and_logits(m_final, xtr, ytr, 0, False, device)
    f_te, _ = cd.features_and_logits(m_final, xte, yte, 0, False, device)
    return run_control(f_tr.numpy(), ytr.numpy(), f_te.numpy(), yte.numpy(), f"live:{arm}/seed{seed}/task0@theta_T")


def main():
    ap = argparse.ArgumentParser(description="E20 positive control")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--arm", default="mlp")
    ap.add_argument("--ckpt-root", default="/runs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.synthetic or args.live):
        ap.error("choose --synthetic and/or --live")
    print("=" * 100 + "\nE20 POSITIVE CONTROL v7  y' = XOR of per-class-median splits of the top-2 within-class residual PCs; (i) linear <= 0.55 AND (ii) MLP - linear >= 0.15   MLP must beat linear by >= 0.15, every seed\n" + "=" * 100)
    results = []
    if args.synthetic:
        results.append(synthetic())
    if args.live:
        results.append(live(args.arm, args.ckpt_root, torch.device(args.device)))
    out = args.out or ("runs/e20/positive_control_live.json" if args.live else "runs/e20/positive_control_synthetic.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"results": results, "all_pass": all(r["pass"] for r in results),
               "sklearn_version": cd._SKLEARN_VERSION}, open(out, "w"), indent=2)
    print(f"\n  ALL {'PASS' if all(r['pass'] for r in results) else 'FAIL -- the MLP probe does not read nonlinear structure; nothing live is read'}")
    print(f"  wrote {out}")
    return 0 if all(r["pass"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
