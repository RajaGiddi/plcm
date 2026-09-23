"""D1 sec 3 D3 on the pretrained arms -- controls, spectrum, and the lemma's
regression, from runs/d1/{e12_base,e14_base}/cells_seed{s}.json (d1_fit_pretrained.py).
Here the deployed head is the frozen per-task W_k, so F_frozen = F_total exactly
and D3 is the direct test. Scores the two pending rows of runs/d1_row.json.

  C-ID     acc_orig / acc_ceiling reproduce the E12/E14 decomposition, 1e-6
  C-SHUF   own - own_shuf > 1.96 SE per cell (must-fail of the pairing), 57/57 per arm
  C-PLUMB  printed, not scored
  D2       spectrum of T' (old frame), residual of S'
  D3       F_frozen on ||D_in||, ||D_out|| (W_k projector): lemma predicts beta_out's interval includes 0, beta_in > 0
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ARMS = ["e12_base", "e14_base"]
TOL = 1e-6


def ols(y, X):
    n = len(y); Xa = np.column_stack([np.ones(n), X]); p = Xa.shape[1]
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta; dof = max(n - p, 1); s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xa.T @ Xa))); q = stats.t.ppf(0.975, dof)
    r2 = 1 - (resid @ resid) / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return beta[1:], (beta[1:] - q * se[1:], beta[1:] + q * se[1:]), float(r2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/d1/pretrained_diag.json"); ap.add_argument("--row", default="runs/d1_row.json")
    args = ap.parse_args()
    out = {}; pooled = []
    print("=" * 100 + "\nD1 DIAGNOSE -- PRETRAINED ARMS (frozen per-task heads: F_frozen == F_total)\n" + "=" * 100)
    for a in ARMS:
        files = sorted(glob.glob(f"runs/d1/{a}/cells_seed*.json"))
        if len(files) < 3:
            print(f"  {a}: {len(files)}/3 seeds -- not read"); continue
        cells = [c for f in files for c in json.load(open(f))["cells"]]; L = str(json.load(open(files[0]))["lambda_main"])
        cid = max(c["C_ID_delta"] for c in cells)
        shuf = sum(c["by_lambda"][L]["own_old"] - c["by_lambda"][L]["own_shuf_old"] > c["res95"] for c in cells)
        plumb = [c["by_lambda"][L]["plumb_dev"] for c in cells]
        g = lambda key, sub=None: np.array([c["by_lambda"][L][key] if sub is None else c["by_lambda"][L][key][sub] for c in cells])
        y = np.array([c["F_frozen"] for c in cells]); X = np.column_stack([g("D_old_hk", "in"), g("D_old_hk", "out")]); wit = float(g("D_old_hk", "Dout_WT_max").max())
        coef, (lo, hi), r2 = ols(y, X)
        own = g("own_old"); ceil = np.array([c["acc_ceiling"] for c in cells]); dep = np.array([c["acc_orig"] for c in cells])
        print(f"  {a:<9} n={len(cells)} d={cells[0]['d']} | C-ID {cid:.1e} -> {'PASS' if cid < TOL else 'FAIL'} | C-SHUF {shuf}/{len(cells)} | C-PLUMB ||S-I|| {min(plumb):.2e}..{max(plumb):.2e}"
              f"\n{'':11}D2 T': spread {g('spec_Tp','spread').mean():.2f} mass {g('spec_Tp','mass_09_11').mean():.2f} sig_d {g('spec_Tp','sigma_d').mean():.4f} n<0.5 {g('spec_Tp','n_below_0.5').mean():.0f}/{cells[0]['d']} | resid S' test {g('residual_Sp_test').mean():.3f}"
              f" | repair: deployed {dep.mean():.4f} -> own {own.mean():.4f} vs ceiling {ceil.mean():.4f}"
              f"\n{'':11}D3 F_frozen {y.mean():.4f} | beta_in {coef[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]  beta_out {coef[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]  R2 {r2:.2f} | ||D_in|| {X[:,0].mean():.2f} ||D_out|| {X[:,1].mean():.2f} | D_out W^T max {wit:.1e}")
        out[a] = {"n": len(cells), "C_ID": cid, "C_SHUF": [shuf, len(cells)], "D2": {"spread_Tp": float(g('spec_Tp','spread').mean()), "mass": float(g('spec_Tp','mass_09_11').mean()), "residual_Sp_test": float(g('residual_Sp_test').mean())},
                  "repair": {"deployed": float(dep.mean()), "own": float(own.mean()), "ceiling": float(ceil.mean())},
                  "D3": {"beta_in": float(coef[0]), "ci_in": [float(lo[0]), float(hi[0])], "beta_out": float(coef[1]), "ci_out": [float(lo[1]), float(hi[1])], "r2": r2, "Dout_WT_max": wit, "F_frozen": float(y.mean())}}
        if cid < TOL:
            pooled += cells
    if len(pooled) >= 57 * 2:
        L = "0.001"; y = np.array([c["F_frozen"] for c in pooled]); X = np.column_stack([[c["by_lambda"][L]["D_old_hk"]["in"] for c in pooled], [c["by_lambda"][L]["D_old_hk"]["out"] for c in pooled]])
        coef, (lo, hi), r2 = ols(y, X)
        print(f"  pooled    n={len(pooled)} | beta_in {coef[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]  beta_out {coef[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]  R2 {r2:.2f}")
        out["pooled"] = {"beta_in": float(coef[0]), "ci_in": [float(lo[0]), float(hi[0])], "beta_out": float(coef[1]), "ci_out": [float(lo[1]), float(hi[1])], "r2": r2}
    if all(a in out for a in ARMS):
        p_in = all(out[a]["D3"]["beta_in"] > 0 and out[a]["D3"]["ci_in"][0] > 0 for a in ARMS)
        p_out = all(out[a]["D3"]["ci_out"][0] <= 0 <= out[a]["D3"]["ci_out"][1] for a in ARMS)
        print(f"\n  D3 predictions (both pretrained arms): beta_in > 0 excluding 0 -> {'fired' if p_in else 'MISS'} (60%);  beta_out interval includes 0 -> {'fired' if p_out else 'MISS'} (55%)")
        if os.path.exists(args.row):
            row = json.load(open(args.row))
            for r in row["predictions"]:
                if r["prediction"].startswith("D3: ||D_in||"): r["outcome"] = "fired" if p_in else "MISS"
                if r["prediction"].startswith("D3: ||D_out||"): r["outcome"] = "fired" if p_out else "MISS"
            json.dump(row, open(args.row, "w"), indent=2); print(f"  updated {args.row}")
        out["predictions"] = {"beta_in_positive_excl0": p_in, "beta_out_includes_0": p_out}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float); print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
