"""E25 A -- the supervision curve (docs/E25_prereg.md sec A).

How many labeled old-task examples does a linear readout repair need to recover
most of the refit gap?

    A0            deployed                                   floor
    A_n           classifier on n per class, BALANCED         the curve
    A-ridge_n,g   same, with l2 pull toward h_k               does the stale head carry structure?
    A_inf         the paper's refit                           ceiling

THE SCALER IS FIT ON THE FULL UNLABELED TASK-k FEATURES, the classifier on the n
labeled ones. `refit_probe` fits its scaler on whatever it is handed, which at
n = 1 is 5-10 points in 256 or 2048 dimensions; a repair with few LABELS still
has the unlabeled old-task inputs every self-supervised arm in this program
assumed, so withholding them would handicap the arm for no reason the contract
states.

EQUIVALENCE RUNS ON THE FULL TRAINING TENSORS, not at n_max. Scaler and
classifier fit on the same set is literally `refit_probe`'s input and must
reproduce it to 1e-6. It cannot be run at n_max: on HAR and MNIST the balanced
set is a STRICT SUBSET of what `refit_probe` trains on, and two different
training sets cannot agree to 1e-6. (On Split-CIFAR every task holds exactly 500
per class, so there A_nmax IS A_inf and recovery reads 1.0 at the top of the
curve by construction. Stated per arm, because it is true of one family only.)

A-ridge's target is h_k = cure_screen.era_head_of re-expressed in the probe's
STANDARDIZED coordinates. Raw logits are W z + b and the probe sees
u = (z - mu)/sd, so z = sd*u + mu gives Wt = W diag(sd), bt = b + W mu. Getting
this wrong pulls toward the wrong point while still looking like a ridge, which
is why control 3 pins gamma -> inf to the head's own measured accuracy.

sklearn cannot penalise toward a nonzero target, so the solver is new (catch 32:
a new path either routes through the audited one or ships an equivalence test).
Its convention is FOUND by the equivalence test rather than asserted here:
sklearn's multinomial objective is nll_SUM + 0.5*(1/C)*||W||^2 with the intercept
unpenalised, so gamma = 0.5 at C = 1 should match, and the test is what says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import E18_FACTORY, era_head_of, BATCH                      # noqa: E402
from scripts.d1_fit import sha, se95                                                 # noqa: E402

SEEDS = [42, 1337, 2024]
N_GRID = [1, 2, 5, 10, 20, 50, 100]          # n_max appended per cell, measured
N_DRAWS = 3
DRAW_SEED = 20260921
GAMMAS = [0.1, 1.0, 10.0]
GAMMA_SKLEARN = 0.5                          # the equivalence test is what certifies this
GAMMA_INF = 1e6
C4_DRAWS = 20                                # control 4: n=1 is what broke it
GUARD = 0.05                                 # per-cell denominator guard, as R1

SCRATCH = {   # (ckpt template, run template, construction, n_classes, num_tasks)
    "s72_off":     ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",  "runs/e10off_ec_seed{s}",  "har_subject", 6,  5),
    "e18_pmd_mlp": ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted",   10, 5),
}
PRETRAINED = {  # (ckpt dir template, n_classes, num_tasks, epoch)
    "e12_base": ("runs/ckpt_e12_base_seed{s}/mafc_seed{s}", 5, 20, 4),
    "e14_base": ("runs/ckpt_e14_base_seed{s}/mafc_seed{s}", 5, 20, 4),
}
HAR_PARTITION = "1104af185c87"


# ------------------------------------------------------------------- solver --
def fit_logreg(X, y, n_classes, W0=None, b0=None, gamma=GAMMA_SKLEARN, gamma_b=None,
               maxiter=5000, return_obj=False):
    """Multinomial logistic regression with an l2 pull toward (W0, b0).

    minimise  -sum_i log p_{y_i} + gamma*||W - W0||_F^2 + gamma_b*||b - b0||^2

    THE INTERCEPT TERM IS NOT COSMETIC. sklearn leaves the intercept
    unpenalised, so `gamma_b` defaults to 0 and the sklearn equivalence holds.
    But A-ridge's target is a READOUT, (W_k, b_k) as a unit, and with the
    intercept free the gamma -> inf limit does NOT reproduce h_k: W is pinned
    while b keeps fitting the data. Measured on e18_pmd_mlp seed 42 task 0:
    0.6895 against h_k's 0.6185, a 7.1pp gap that reads as a broken must-fail
    and is really a half-pinned target. Callers targeting a head pass
    gamma_b = gamma; control 3 is what catches it if they do not.
    """
    from scipy.optimize import minimize
    n, d = X.shape
    W0 = np.zeros((n_classes, d)) if W0 is None else np.asarray(W0, float)
    b0 = np.zeros(n_classes) if b0 is None else np.asarray(b0, float)
    gamma_b = 0.0 if gamma_b is None else float(gamma_b)
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1.0
    nd = n_classes * d

    def obj(v):
        W, b = v[:nd].reshape(n_classes, d), v[nd:]
        Z = X @ W.T + b
        m = Z.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(Z - m).sum(1))
        nll = float(lse.sum() - Z[np.arange(n), y].sum())
        D, db = W - W0, b - b0
        P = np.exp(Z - lse[:, None])
        G = P - Y
        gW = G.T @ X + 2.0 * gamma * D
        gb = G.sum(0) + 2.0 * gamma_b * db
        f = nll + gamma * float((D * D).sum()) + gamma_b * float((db * db).sum())
        return f, np.concatenate([gW.ravel(), gb])

    r = minimize(obj, np.concatenate([W0.ravel(), b0]), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10})
    W, b = r.x[:nd].reshape(n_classes, d), r.x[nd:]
    if return_obj:
        return W, b, int(r.nit), bool(r.success), float(r.fun)
    return W, b, int(r.nit), bool(r.success)


def score(W, b, X, y) -> float:
    return float(((X @ W.T + b).argmax(1) == y).mean())


def balanced_draw(y, n, rng, n_classes):
    """n per class. Returns None if any class has fewer than n."""
    idx = []
    for c in range(n_classes):
        pool = np.flatnonzero(y == c)
        if pool.size < n:
            return None
        idx.append(rng.choice(pool, size=n, replace=False))
    return np.concatenate(idx)


# ------------------------------------------------------------------ one cell --
def run_cell(Z_tr, y_tr, Z_te, y_te, h, n_classes, acc_deployed, seed, k, out, verbose=True):
    from sklearn.preprocessing import StandardScaler
    n_tr = Z_tr.shape[0]
    counts = np.bincount(y_tr, minlength=n_classes)
    n_max = int(counts.min())

    sc_unlab = StandardScaler().fit(Z_tr)            # the FULL unlabeled task-k features
    Xtr_u, Xte_u = sc_unlab.transform(Z_tr), sc_unlab.transform(Z_te)
    mu, sd = sc_unlab.mean_, sc_unlab.scale_

    # h_k in the probe's standardized coordinates
    Wk = h.weight.detach().cpu().numpy().astype(np.float64)
    bk = h.bias.detach().cpu().numpy().astype(np.float64)
    Wt, bt = Wk * sd, bk + Wk @ mu

    acc_inf = refit_probe(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr),
                          torch.from_numpy(Z_te).float(), torch.from_numpy(y_te), n_classes)
    acc_hk = score(Wt, bt, Xte_u, y_te)

    cell = {"seed": seed, "task": k, "n_train": int(n_tr), "n_test": int(y_te.shape[0]),
            "dim": int(Z_tr.shape[1]), "class_counts": counts.tolist(), "n_max": n_max,
            "acc_deployed": acc_deployed, "acc_refit_full": acc_inf, "acc_h_k": acc_hk,
            "denom": acc_inf - acc_deployed, "valid": (acc_inf - acc_deployed) >= GUARD,
            "res95": se95(acc_inf, int(y_te.shape[0])), "curve": {}, "ridge": {}}

    grid = [n for n in N_GRID if n <= n_max] + ([n_max] if n_max not in N_GRID else [])
    for n in grid:
        accs, ridge_accs = [], {g: [] for g in GAMMAS}
        for dr in range(N_DRAWS):
            rng = np.random.default_rng(DRAW_SEED + 1000 * seed + 100 * k + 10 * n + dr)
            sel = balanced_draw(y_tr, n, rng, n_classes)
            if sel is None:
                continue
            W, b, *_ = fit_logreg(Xtr_u[sel], y_tr[sel], n_classes)
            accs.append(score(W, b, Xte_u, y_te))
            for g in GAMMAS:
                Wr, br, *_ = fit_logreg(Xtr_u[sel], y_tr[sel], n_classes, W0=Wt, b0=bt,
                                        gamma=g, gamma_b=g)   # the target is the whole readout
                ridge_accs[g].append(score(Wr, br, Xte_u, y_te))
        if not accs:
            continue
        rec = lambda v: {"mean": float(np.mean(v)), "spread": float(max(v) - min(v)), "draws": v}
        cell["curve"][str(n)] = rec(accs)
        cell["ridge"][str(n)] = {str(g): rec(v) for g, v in ridge_accs.items()}
    if verbose:
        pts = "  ".join(f"n{n}:{cell['curve'][str(n)]['mean']:.3f}" for n in grid if str(n) in cell["curve"])
        print(f"    s{seed} k{k}: A0 {acc_deployed:.4f} | h_k {acc_hk:.4f} | Ainf {acc_inf:.4f} "
              f"| n_max {n_max} | {pts}")
    out.append(cell)
    return cell, (Xtr_u, y_tr, Xte_u, y_te, Wt, bt, sc_unlab)


def controls(ctx, n_classes, Z_tr, y_tr, Z_te, y_te, acc_deployed, acc_hk, acc_inf):
    """Controls 1-4 and 7 on one cell. The equivalence runs on the FULL tensors."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xtr_u, _, Xte_u, _, Wt, bt, _ = ctx
    out = {}
    # 1. EQUIVALENCE. The bar is the OBJECTIVE, not the accuracy.
    #
    # `refit_probe` is sklearn at its DEFAULT tol=1e-4, and on these features that
    # stops ~45 iterations in, 1.13 objective units above its own optimum, while a
    # tight-tolerance fit needs ~490. Its docstring says "fit to OPTIMALITY" and its
    # recorded witness is `probe_n_iter < max_iter` -- which cannot see early
    # stopping on TOLERANCE, only on iteration count. So an accuracy bar of 1e-6
    # against refit_probe is unsatisfiable for a reason that is not about this
    # solver: two solvers cannot agree to 1e-6 when one is not at its optimum.
    #
    # The equivalence therefore compares what the claim is actually about -- the
    # objective -- against sklearn run to convergence, and prints refit_probe's
    # default-tol value beside it as the program's incumbent.
    sc = StandardScaler().fit(Z_tr)
    Xtr, Xte = sc.transform(Z_tr), sc.transform(Z_te)
    W, b, nit, ok, fval = fit_logreg(Xtr, y_tr, n_classes, return_obj=True)
    ref = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0, tol=1e-10).fit(Xtr, y_tr)
    Zs = Xtr @ ref.coef_.T + ref.intercept_
    m = Zs.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Zs - m).sum(1))
    f_ref = float(lse.sum() - Zs[np.arange(len(y_tr)), y_tr].sum()) \
        + GAMMA_SKLEARN * float((ref.coef_ ** 2).sum())
    a = score(W, b, Xte, y_te)
    a_ref = float((ref.predict(Xte) == y_te).mean())
    out["c1_equivalence"] = {
        "solver_objective": fval, "sklearn_tight_objective": f_ref,
        "objective_rel_delta": abs(fval - f_ref) / max(abs(f_ref), 1e-12),
        "solver_acc": a, "sklearn_tight_acc": a_ref,
        "refit_probe_acc_default_tol": acc_inf, "sklearn_tight_n_iter": int(ref.n_iter_[0]),
        "gamma": GAMMA_SKLEARN, "converged": ok, "n_iter": nit,
        "pass": abs(fval - f_ref) / max(abs(f_ref), 1e-12) <= 1e-6 and a == a_ref}
    # 3. must-fail: gamma -> inf pins the WHOLE readout and reproduces acc(h_k Z_T)
    Wr, br, *_ = fit_logreg(Xtr_u[:64], y_tr[:64], n_classes, W0=Wt, b0=bt,
                            gamma=GAMMA_INF, gamma_b=GAMMA_INF)
    a_inf = score(Wr, br, Xte_u, y_te)
    out["c3_ridge_mustfail"] = {"acc_at_gamma_1e6": a_inf, "acc_h_k": acc_hk,
                                "abs_delta": abs(a_inf - acc_hk),
                                "pass": abs(a_inf - acc_hk) <= se95(acc_hk, int(y_te.shape[0]))}
    # 4. PLUMBING, RESTATED 2026-09-21 before the pretrained headline was read.
    #
    # The bar was "shuffled-label A_n does not exceed A0 beyond binomial
    # resolution". That control discriminates NOWHERE. On permuted CIFAR the
    # deployed accuracy is itself at chance (0.22 against 0.20), so it compares
    # noise to noise and failed the ViT smoke for that reason. On the scratch
    # arms it could not fail either: shuffled labels read ~0.17 on HAR against a
    # deployed 0.47, so it passed trivially. Catch 25 exactly -- a gate that has
    # only ever passed, next to one that cannot discriminate, are the same defect
    # seen from two sides.
    #
    # Restated against CHANCE, which is what "the label path is not leaking"
    # actually means and does not depend on how good A0 happens to be:
    #     |acc(shuffled) - 1/C|  <=  1.96 * SE_binomial(1/C, n_test)
    # Chance is 0.20 on the CIFAR heads, 1/6 on HAR, 0.10 on MNIST. Applied to
    # every arm, scratch included, so the comparison is uniform.
    # The TARGET is chance and does not move. What was wrong was the ESTIMATOR:
    # one draw, against a bar built from binomial SE on the test set alone.
    # Diagnosed over 20 draws (scripts/e25a_c4_diag.py, runs/e25/a/c4_diagnosis.json):
    # on HAR the shuffled probe ranges [0.038, 0.367] across draws and POOLS TO
    # 0.1664 against a chance of 0.1667. The single draw the control happened to
    # take read 0.286 -- inside that range and nowhere near its mean. The
    # draw-to-draw variance is 4-8x the binomial term and the bar omitted it
    # entirely, so the control was measuring which 120 training points it drew.
    # Program rule, applied to a control's own measurement: nothing at n=1; and
    # R3's form, tolerance = max(the instrument's own spread, binomial SE).
    # Both estimators are reported. The single-draw value keeps the contracted
    # bar so the change is visible; the ruling on which is the control is the
    # user's.
    n_te = int(y_te.shape[0])
    chance = 1.0 / n_classes
    n_sh = min(20, int(np.bincount(y_tr, minlength=n_classes).min()))
    draws = []
    for d in range(C4_DRAWS):
        rng = np.random.default_rng(DRAW_SEED + 7919 * d)
        sel = balanced_draw(y_tr, n_sh, rng, n_classes)
        W, b, *_ = fit_logreg(Xtr_u[sel], rng.permutation(y_tr[sel]), n_classes)
        draws.append(score(W, b, Xte_u, y_te))
    a_sh = draws[0]                                   # the contracted single draw
    mean_sh = float(np.mean(draws))
    binom = se95(chance, n_te)
    spread = float(stats.t.ppf(0.975, len(draws) - 1) * np.std(draws, ddof=1) / np.sqrt(len(draws)))
    bar_multi = max(binom, spread)
    out["c4_plumbing"] = {
        "chance": chance, "n_test": n_te, "n_per_class": n_sh, "n_draws": C4_DRAWS,
        "acc_shuffled_mean": mean_sh, "acc_shuffled_draws": draws,
        "draw_min": float(min(draws)), "draw_max": float(max(draws)),
        "bar_binomial": binom, "bar_draw_t95": spread, "bar": bar_multi,
        "abs_delta_from_chance": abs(mean_sh - chance),
        "pass": abs(mean_sh - chance) <= bar_multi,
        "single_draw_as_contracted": {
            "acc_shuffled": a_sh, "bar_1.96SE_at_chance": binom,
            "abs_delta_from_chance": abs(a_sh - chance),
            "pass": abs(a_sh - chance) <= binom},
        "superseded_bar_vs_A0": {"acc_deployed": acc_deployed,
                                 "would_pass": a_sh <= acc_deployed + se95(acc_deployed, n_te)}}
    return out


def main():
    ap = argparse.ArgumentParser(description="E25 A: the supervision curve")
    ap.add_argument("--arm", required=True, choices=sorted(SCRATCH) + sorted(PRETRAINED))
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tasks", type=int, default=None, help="smoke: cap the old tasks scored")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    root = args.ckpt_root.rstrip("/")
    os.makedirs(args.out_dir, exist_ok=True)
    pre = args.arm in PRETRAINED
    print("=" * 104 + f"\nE25 A  arm={args.arm} ({'pretrained' if pre else 'scratch'}) "
          f"device={device} draws={N_DRAWS} grid={N_GRID}+n_max\n" + "=" * 104)

    for s in seeds:
        cells, ctrl = [], None
        if pre:
            from src.data.split_cifar100 import SplitCIFAR100Benchmark
            from scripts.e12_decompose import load_era as load_pre, features_and_logits as fal_pre
            from scripts.e12_p3 import assert_p3a, assert_p3b, positive_controls
            from torch.utils.data import DataLoader
            ck_t, n_classes, num_tasks, epoch = PRETRAINED[args.arm]
            ckpt = ck_t.replace("runs/", root + "/", 1).format(s=s)
            bench = SplitCIFAR100Benchmark(num_tasks=num_tasks, batch_size=BATCH,
                                           root=args.data_root, seed=s,
                                           remap_labels=True, download=False)
            T = num_tasks - 1
            m_final = load_pre(ckpt, T, device)
            _, te0 = bench.get_task_loaders(0)
            xprobe = next(iter(te0))[0][:16].to(device)
            pc = positive_controls(m_final, xprobe, 0, verbose=True)     # control 8, both halves
            assert pc["p3a_control"] and pc["p3b_control"], pc
            print(f"  control 8: P3a/P3b positive controls fired {pc}")
            tasks = range(T if args.max_tasks is None else min(T, args.max_tasks))
            for k in tasks:
                tr, te = bench.get_task_loaders(k)
                tr = DataLoader(tr.dataset, batch_size=BATCH, shuffle=False)
                m_era = load_pre(ckpt, k, device)
                assert_p3a(m_final, xprobe, k, label=f"s{s}/final/T{k}")
                assert_p3b(m_final, xprobe, k, label=f"s{s}/final/T{k}")
                f_tr, _, y_tr = fal_pre(m_final, tr, k, False, device)
                f_te, l_te, y_te = fal_pre(m_final, te, k, False, device)
                _, _, y_tr2 = fal_pre(m_final, tr, k, False, device)
                assert torch.equal(y_tr, y_tr2), "feature loader not deterministic"
                h = era_head_of(m_era, k)
                a0 = float((l_te.argmax(1) == y_te).float().mean())
                cell, ctx = run_cell(f_tr.numpy().astype(np.float64), y_tr.numpy(),
                                     f_te.numpy().astype(np.float64), y_te.numpy(),
                                     h, n_classes, a0, s, k, cells)
                if ctrl is None:
                    ctrl = controls(ctx, n_classes, f_tr.numpy().astype(np.float64), y_tr.numpy(),
                                    f_te.numpy().astype(np.float64), y_te.numpy(),
                                    a0, cell["acc_h_k"], cell["acc_refit_full"])
        else:
            ck_t, run_t, kind, n_classes, num_tasks = SCRATCH[args.arm]
            ck_t = ck_t.replace("runs/", root + "/", 1)
            run_t = run_t.replace("runs/", root + "/", 1)
            arm = json.load(open(f"{run_t.format(s=s)}/mafc_results.json"))["arm"]
            assert arm["use_task_heads"] is False and arm["era_checkpoints"] is True, arm
            T = num_tasks - 1
            if kind == "har_subject":
                from src.data.har_subject import HARSubjectBenchmark
                bench = HARSubjectBenchmark(num_tasks=num_tasks, root=".", batch_size=BATCH)
                assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
            else:
                bench = E18_FACTORY[kind](s)
                assert arm["content_fingerprint"] == bench.content_fingerprint()
            data = load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=args.probe_seed)
            d = ck_t.format(s=s)
            m4 = load(d, T)
            tasks = range(T if args.max_tasks is None else min(T, args.max_tasks))
            for k in tasks:
                xtr, ytr, xte, yte = data[k]
                g = assert_path_identity(m4, xte, k, False, device, label=f"s{s}/T{k}", verbose=False)
                assert g["p_b"], (s, k)
                f_tr, _ = features_and_logits(m4, xtr, ytr, k, False, device)
                f_te, l_te = features_and_logits(m4, xte, yte, k, False, device)
                h = era_head_of(load(d, k), k)
                a0 = float((l_te.argmax(1) == yte).float().mean())
                cell, ctx = run_cell(f_tr.numpy().astype(np.float64), ytr.numpy(),
                                     f_te.numpy().astype(np.float64), yte.numpy(),
                                     h, n_classes, a0, s, k, cells)
                cell["input_sha"] = sha(xte)
                if ctrl is None:
                    ctrl = controls(ctx, n_classes, f_tr.numpy().astype(np.float64), ytr.numpy(),
                                    f_te.numpy().astype(np.float64), yte.numpy(),
                                    a0, cell["acc_h_k"], cell["acc_refit_full"])
        for nm, c in (ctrl or {}).items():
            print(f"  {nm}: {'PASS' if c.get('pass') else 'FAIL'}  " +
                  " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in c.items() if k != "pass"))
        p = os.path.join(args.out_dir, f"curve_seed{s}.json")
        json.dump({"arm": args.arm, "pretrained": pre, "seed": s, "n_draws": N_DRAWS,
                   "draw_seed": DRAW_SEED, "gammas": GAMMAS, "guard": GUARD,
                   "probe_subset_seed": args.probe_seed, "controls": ctrl,
                   "scaler": "fit on the FULL unlabeled task-k train features; classifier on the n labeled",
                   "cells": cells}, open(p, "w"), indent=2, default=float)
        print(f"  wrote {p}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
