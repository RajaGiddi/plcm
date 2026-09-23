"""D1 sec 3-4 -- controls and the four diagnostics, from the per-cell scalars
d1_fit.py wrote (runs/d1/{arm}/cells_seed{s}.json) joined with the head-drift
split (runs/head_drift/{arm}.json). Never touches features. Every verdict is
derived from the arrays printed beside it (catch 22).

  C-ID      acc_orig / acc_ceiling reproduce the seeded decomposition, 1e-6, else nothing read
  C-INPUT   the input tensors' sha1, recorded per cell (both passes read the same object)
  C-PLUMB   S fit on (Z_T, Z_T): ||S - I||_F, printed, NOT scored (cannot fail by algebra)
  C-SHUF    own(k) - own_shuf(k) > 1.96 SE_binomial in 12/12 cells per arm (must-fail of the pairing)
  C-TARGET  own(k) targets the ERA accuracy: acc_ceiling - own(k) against 1.96 SE per cell
  C-lambda  per-arm verdicts identical across lambda in {1e-4, 1e-3, 1e-2}
  D1        transfer loss own(k) - cross(j->k): adjacent vs distant; deploy(k) vs own(k)
  D2        spectrum of T (re-laid) and T' (old frame); residual; F_enc against residual and log sigma_d; re-laid F_enc
  D3        F_frozen = acc_ceiling - acc(h_k Z_T') regressed on ||D_in||, ||D_out|| (h_k projector, old-frame T'); D_out W^T == 0 witnessed
  D4        label marginals
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ["s72_off", "e18_pmd_lstm", "e18_pmd_mlp", "e18_rmd_mlp"]
TOL = 1e-6


def ols(y, X):
    """OLS with intercept; returns coef, 95% t-intervals, r2. X: (n, p)."""
    n = len(y); Xa = np.column_stack([np.ones(n), X]); p = Xa.shape[1]
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta; dof = max(n - p, 1); s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(Xa.T @ Xa); se = np.sqrt(np.diag(cov)); q = stats.t.ppf(0.975, dof)
    r2 = 1 - (resid @ resid) / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return beta[1:], (beta[1:] - q * se[1:], beta[1:] + q * se[1:]), float(r2)


def partial_r(y, x, z):
    """correlation of y and x controlling for z."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_arm(arm):
    files = sorted(glob.glob(f"runs/d1/{arm}/cells_seed*.json"))
    if not files:
        return None
    seeds = [json.load(open(f)) for f in files]
    hd = {(r["seed"], r["task"]): r for r in json.load(open(f"runs/head_drift/{arm}.json"))["rows"]}
    cells = []
    for sd in seeds:
        for c in sd["cells"]:
            h = hd[(c["seed"], c["task"])]
            assert abs(h["acc_orig"] - c["acc_orig"]) < TOL and abs(h["acc_ceiling"] - c["acc_ceiling"]) < TOL, (arm, c["seed"], c["task"])
            c["acc_frozen"] = h["acc_frozen"]; c["F_frozen"] = c["acc_ceiling"] - h["acc_frozen"]; c["dH"] = h["dH"]
            cells.append(c)
    return {"cells": cells, "transfer": {sd["seed"]: sd["transfer"] for sd in seeds}, "lambdas": seeds[0]["lambdas"], "lambda_main": seeds[0]["lambda_main"]}


def arm_verdicts(A, lam):
    """Per-arm D1/D2 readings at one lambda -- used for C-lambda and the row."""
    L = str(lam); cells = A["cells"]
    own = np.array([c["by_lambda"][L]["own"] for c in cells]); dep = np.array([c["by_lambda"][L]["deploy"] for c in cells])
    res = np.array([c["res95"] for c in cells])
    # transfer matrices exist at lambda_main only; C-lambda compares own/deploy/spectra across lambdas
    spread_T = np.mean([c["by_lambda"][L]["spec_T"]["spread"] for c in cells])
    spread_Tp = np.mean([c["by_lambda"][L]["spec_Tp"]["spread"] for c in cells])
    return {"own": float(own.mean()), "deploy": float(dep.mean()), "deploy_below_own_cells": int(np.sum(dep < own - res)), "res95": float(res.mean()),
            "spread_T": float(spread_T), "spread_Tp": float(spread_Tp), "spread_T_lt_0.5": bool(spread_T < 0.5)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/d1/controls.json"); args = ap.parse_args()
    data = {a: load_arm(a) for a in ARMS}
    present = [a for a in ARMS if data[a]]
    print("=" * 110 + f"\nD1 DIAGNOSE   arms present {present}\n" + "=" * 110)
    if len(present) < len(ARMS):
        print("  incomplete -- the row is not read until every arm's artifacts exist")
    out = {"arms": {}}
    pooled_cells = []

    print("\nCONTROLS")
    for a in present:
        A = data[a]; cells = A["cells"]; L = str(A["lambda_main"])
        cid = max(c["C_ID_delta"] for c in cells)
        shas = sorted({c["input_sha"]["test"] for c in cells})
        plumb = [c["by_lambda"][L]["plumb_dev"] for c in cells]
        shuf_ok = sum(c["by_lambda"][L]["own"] - c["by_lambda"][L]["own_shuf"] > c["res95"] for c in cells)
        shuf_ok_old = sum(c["by_lambda"][L]["own_old"] - c["by_lambda"][L]["own_shuf_old"] > c["res95"] for c in cells)
        gap = np.array([c["acc_ceiling"] - c["by_lambda"][L]["own"] for c in cells]); res = np.array([c["res95"] for c in cells])
        target_over = int(np.sum(gap > res))
        verd = {lam: arm_verdicts(A, lam) for lam in A["lambdas"]}
        # C-lambda: a READING is a verdict. The first run flagged s72_off on an own() shift of 0.8pp under a 0.5pp
        # threshold -- finer than HAR's 3.3pp resolution (R3's error, in this check); the bar is the resolution.
        lam_flip = any(verd[l]["spread_T_lt_0.5"] != verd[A["lambda_main"]]["spread_T_lt_0.5"]
                       or abs(verd[l]["own"] - verd[A["lambda_main"]]["own"]) > verd[l]["res95"] for l in A["lambdas"])
        owns = ", ".join(f"{verd[l]['own']:.4f}" for l in A["lambdas"]); spreads = ", ".join(f"{verd[l]['spread_T']:.2f}" for l in A["lambdas"])
        print(f"  {a:<13} C-ID max|d| {cid:.1e} -> {'PASS' if cid < TOL else 'FAIL -- nothing read'} | C-INPUT {len(cells)} cells, {len(shas)} distinct test-input hashes recorded"
              f" | C-PLUMB ||S-I|| {min(plumb):.2e}..{max(plumb):.2e} (printed, not scored)"
              f"\n{'':15}C-SHUF re-laid {shuf_ok}/{len(cells)}, old-frame {shuf_ok_old}/{len(cells)} -> {'PASS' if shuf_ok == len(cells) and shuf_ok_old == len(cells) else 'FAIL'}"
              f" | C-TARGET gap ceiling-own pooled {gap.mean():+.4f}, cells over 1.96SE: {target_over}/{len(cells)}"
              f" | C-lambda own {owns}; spread_T {spreads} -> {'FLIPS' if lam_flip else 'stable'}")
        out["arms"][a] = {"C_ID": cid, "C_SHUF": [shuf_ok, shuf_ok_old, len(cells)], "C_TARGET_gap": float(gap.mean()), "C_TARGET_over": target_over, "C_lambda_stable": not lam_flip, "plumb": plumb}
        if cid >= TOL:
            continue
        for c in cells:
            c["arm"] = a
        pooled_cells += cells

    # ---- D1 ------------------------------------------------------------------
    print("\nD1  transfer loss own(k) - cross(j->k) [re-laid, h_k]; adjacent |j-k|=1 vs distant; and deploy(k) - own(k)")
    for a in present:
        A = data[a]; L = str(A["lambda_main"]); cells = {(c["seed"], c["task"]): c for c in A["cells"]}
        adj, dist, adj_o, dist_o = [], [], [], []
        over_res, over_5, n_pairs = 0, 0, 0
        for s, tr in A["transfer"].items():
            M = np.array(tr["relaid"]); Mo = np.array(tr["old"])
            for j in range(4):
                for k in range(4):
                    if j == k: continue
                    loss = M[k][k] - M[j][k]; loss_o = Mo[k][k] - Mo[j][k]; n_pairs += 1
                    (adj if abs(j - k) == 1 else dist).append(loss); (adj_o if abs(j - k) == 1 else dist_o).append(loss_o)
                    over_res += loss > cells[(int(s), k)]["res95"]; over_5 += loss > 0.05
        dep_own = [c["by_lambda"][L]["deploy"] - c["by_lambda"][L]["own"] for c in A["cells"]]
        dep_below = sum(c["by_lambda"][L]["deploy"] < c["by_lambda"][L]["own"] - c["res95"] for c in A["cells"])
        print(f"  {a:<13} adjacent {np.mean(adj):+.4f} [{min(adj):+.3f},{max(adj):+.3f}]  distant {np.mean(dist):+.4f} [{min(dist):+.3f},{max(dist):+.3f}]"
              f"  pairs > 1.96SE {over_res}/{n_pairs}, > 5pp {over_5}/{n_pairs} | old-frame adjacent {np.mean(adj_o):+.4f} distant {np.mean(dist_o):+.4f}"
              f" | deploy-own pooled {np.mean(dep_own):+.4f}, cells deploy < own-res: {dep_below}/{len(dep_own)}")
        out["arms"][a].update({"D1": {"adjacent": float(np.mean(adj)), "distant": float(np.mean(dist)), "over_res": over_res, "over_5pp": over_5, "n_pairs": n_pairs,
                                       "adjacent_old": float(np.mean(adj_o)), "distant_old": float(np.mean(dist_o)),
                                       "deploy_minus_own": float(np.mean(dep_own)), "deploy_below_own_cells": dep_below}})

    # ---- D2 ------------------------------------------------------------------
    print("\nD2  spectrum (standardized coords): T re-laid | T' old frame; residual of S on test; F_enc vs residual and log sigma_d; re-laid F_enc")
    for a in present:
        A = data[a]; L = str(A["lambda_main"]); cells = A["cells"]
        g = lambda key, sub=None: np.array([c["by_lambda"][L][key] if sub is None else c["by_lambda"][L][key][sub] for c in cells])
        fe = np.array([c["F_enc"] for c in cells]); fer = np.array([c["F_enc_relaid"] for c in cells]); res = np.array([c["res95"] for c in cells])
        r_sig = float(np.corrcoef(fe, np.log(np.maximum(g("spec_Tp", "sigma_d"), 1e-12)))[0, 1]); r_res = float(np.corrcoef(fe, g("residual_Sp_test"))[0, 1])
        print(f"  {a:<13} T: spread {g('spec_T','spread').mean():.2f} mass {g('spec_T','mass_09_11').mean():.2f} sig_d {g('spec_T','sigma_d').mean():.3f} n<0.5 {g('spec_T','n_below_0.5').mean():.0f}/256"
              f" | T': spread {g('spec_Tp','spread').mean():.2f} mass {g('spec_Tp','mass_09_11').mean():.2f} sig_d {g('spec_Tp','sigma_d').mean():.3f}"
              f" | resid S test {g('residual_S_test').mean():.3f}, S' {g('residual_Sp_test').mean():.3f}"
              f" | F_enc {fe.mean():+.4f} -> re-laid {fer.mean():+.4f} (res {res.mean():.3f}) | within-arm r(F_enc, log sig_d) {r_sig:+.2f}, r(F_enc, resid) {r_res:+.2f}")
        out["arms"][a].update({"D2": {"spread_T": float(g('spec_T','spread').mean()), "mass_T": float(g('spec_T','mass_09_11').mean()), "spread_Tp": float(g('spec_Tp','spread').mean()),
                                       "residual_S_test": float(g('residual_S_test').mean()), "residual_Sp_test": float(g('residual_Sp_test').mean()),
                                       "F_enc": float(fe.mean()), "F_enc_relaid": float(fer.mean()), "res95": float(res.mean())}})
    if len(pooled_cells) >= 24:
        L = str(data[present[0]]["lambda_main"])
        fe = np.array([c["F_enc"] for c in pooled_cells]); lsd = np.log(np.maximum([c["by_lambda"][L]["spec_Tp"]["sigma_d"] for c in pooled_cells], 1e-12)); rs = np.array([c["by_lambda"][L]["residual_Sp_test"] for c in pooled_cells])
        r_sig, r_res = float(np.corrcoef(fe, lsd)[0, 1]), float(np.corrcoef(fe, rs)[0, 1])
        pr_sig, pr_res = partial_r(fe, lsd, rs), partial_r(fe, rs, lsd)
        print(f"  pooled {len(pooled_cells)} cells: r(F_enc, log sigma_d) {r_sig:+.2f} (partial | resid {pr_sig:+.2f});  r(F_enc, resid) {r_res:+.2f} (partial | log sigma_d {pr_res:+.2f})")
        out["D2_pooled"] = {"n": len(pooled_cells), "r_sigma_d": r_sig, "r_resid": r_res, "partial_r_sigma_d": pr_sig, "partial_r_resid": pr_res}

    # ---- D3 ------------------------------------------------------------------
    print("\nD3  F_frozen = acc_ceiling - acc(h_k Z_T') on ||D_in||, ||D_out|| (h_k projector, old-frame T'); D_out W^T witnessed; dH beside")
    for a in present + (["pooled"] if len(pooled_cells) >= 24 else []):
        cells = pooled_cells if a == "pooled" else data[a]["cells"]; L = str(data[present[0]]["lambda_main"])
        y = np.array([c["F_frozen"] for c in cells]); X = np.array([[c["by_lambda"][L]["D_old_hk"]["in"], c["by_lambda"][L]["D_old_hk"]["out"]] for c in cells])
        wit = max(c["by_lambda"][L]["D_old_hk"]["Dout_WT_max"] for c in cells)
        coef, (lo, hi), r2 = ols(y, X)
        dh = np.array([c["dH"] for c in cells])
        print(f"  {a:<13} n={len(cells)} F_frozen {y.mean():.4f} | beta_in {coef[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]  beta_out {coef[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]  R2 {r2:.2f}"
              f" | ||D_in|| {X[:,0].mean():.2f} ||D_out|| {X[:,1].mean():.2f} | D_out W^T max {wit:.1e} | dH {dh.mean():+.4f}")
        (out["arms"].setdefault(a, {}) if a != "pooled" else out.setdefault("D3_pooled", {})).update(
            {"D3": {"beta_in": float(coef[0]), "ci_in": [float(lo[0]), float(hi[0])], "beta_out": float(coef[1]), "ci_out": [float(lo[1]), float(hi[1])], "r2": r2, "Dout_WT_max": wit, "F_frozen": float(y.mean()), "dH": float(dh.mean())}} if a != "pooled" else
            {"beta_in": float(coef[0]), "ci_in": [float(lo[0]), float(hi[0])], "beta_out": float(coef[1]), "ci_out": [float(lo[1]), float(hi[1])], "r2": r2})

    # ---- D4 ------------------------------------------------------------------
    print("\nD4  train label marginals (min / max class fraction over tasks)")
    for a in present:
        fr = [np.array(c["label_marginal_train"]) / sum(c["label_marginal_train"]) for c in data[a]["cells"]]
        print(f"  {a:<13} min class fraction {min(f.min() for f in fr):.3f}  max {max(f.max() for f in fr):.3f}  (uniform = {1/len(fr[0]):.3f})")
        out["arms"][a]["D4"] = {"min_frac": float(min(f.min() for f in fr)), "max_frac": float(max(f.max() for f in fr))}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    for a in present:
        json.dump(out["arms"][a], open(f"runs/d1/{a}/diag.json", "w"), indent=2, default=float)
    print(f"\n  wrote {args.out} and runs/d1/<arm>/diag.json")


if __name__ == "__main__":
    main()
