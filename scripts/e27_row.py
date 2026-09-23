"""E27 row -- does the equivariance defect predict readout-repair loss?
(docs/E27_prereg.md sec 4, signed v2.)

Primary: Spearman rho(delta_tilde, G) over the 48 primary cells, bootstrap over
seeds. G = C0deg - C3 in accuracy, which is (forgetting under bridging) minus
(forgetting under re-layout) since both subtract the same ceiling.

PER-CELL GUARD, and it is not cosmetic. delta > 1 means the affine fit is WORSE
than predicting zero: the fit failed, and the number is not a measurement of
equivariance. Two of 117 cells exceed it, and one of them (arm B, seed 2024,
k=0) reads 15.1 and single-handedly moves that arm's pooled delta from 0.597 to
1.806 and its delta_tilde from 1.24 to 3.58 -- enough to flip prediction 2's
comparison. Pooling hid it; the per-cell guard is what surfaces it (catch 26).
Excluded cells are counted and named, never dropped silently, and the
forced-inclusion figure is printed beside.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

# arm -> (cure artifact, key, decomposition artifact or None, family)
PRIMARY = {
    "s72_off":      ("runs/e10ec_seeded/cures_e10.json", "HAR/OFF",   "LSTM"),
    "e18_pmd_mlp":  ("runs/e18_pmd_mlp/cures.json",      "MNIST/OFF", "MLP"),
    "e18_pmd_lstm": ("runs/e18_pmd_lstm/cures.json",     "MNIST/OFF", "LSTM"),
    "e23_har_mlp":  ("runs/e23_mlp_screen/cures_e10.json", "HAR/OFF", "MLP"),
}
SEPARATE = {"e18_rmd_mlp": ("runs/e18_rmd_mlp/cures.json", "MNIST/OFF", "MLP")}  # lossy relay
SECONDARY_ONLY = ("e23b_t20",)                                                   # no cure screen
GUARD = 1.0
BOOT = 5000


def load_defect(arm):
    return [json.load(open(f)) for f in sorted(glob.glob(f"runs/e27/{arm}/defect_seed*.json"))]


def cells(arm):
    out = []
    for d in load_defect(arm):
        for r in d["rows"]:
            out.append(r)
    return out


def gaps(path, key):
    g = {}
    for r in json.load(open(path))[key]:
        g[(r["seed"], r["task"])] = {"G": r["C0deg"] - r["C3"], "C3": r["C3"], "C0deg": r["C0deg"]}
    return g


def boot_rho(x, y, seeds, rng, n=BOOT):
    """Bootstrap over SEEDS, not cells: cells within a seed share a checkpoint."""
    uniq = sorted(set(seeds))
    idx = {s: [i for i, ss in enumerate(seeds) if ss == s] for s in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for s in pick for i in idx[s]]
        if len(set(np.asarray(y)[sel])) < 2:
            continue
        out.append(stats.spearmanr(np.asarray(x)[sel], np.asarray(y)[sel]).statistic)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description="E27 row")
    ap.add_argument("--out", default="runs/e27_row.json")
    args = ap.parse_args()
    rng = np.random.default_rng(20260922)
    print("=" * 112 + "\nE27 ROW   does the equivariance defect predict readout-repair loss?\n" + "=" * 112)
    out = {"guard": GUARD, "arms": {}}

    # ---- controls, from any artifact (identical across jobs) -----------------------
    c = load_defect("s72_off")[0]["controls"]
    pc = c["positive_square_invertible"]
    print(f"\n  CONTROLS  positive (lam {pc['lam']}, {pc['draws']} draws): max delta "
          f"784 {pc['delta_max']['784']:.2e}, 1152 {pc['delta_max']['1152']:.2e} vs bar {pc['bar']} "
          f"-> {'PASS' if pc['pass'] else 'FAIL'}")
    print(f"            floor 784->256 {c['floor_random_784_256']['delta']:.4f} vs {c['floor_random_784_256']['target']} "
          f"-> {'PASS' if c['floor_random_784_256']['pass'] else 'FAIL'} | shuffled-shape "
          f"{c['must_fail_shape_synthetic']['shuffled']:.3f} > {c['must_fail_shape_synthetic']['unshuffled']:.3f} "
          f"-> {'PASS' if c['must_fail_shape_synthetic']['pass'] else 'FAIL'}")
    out["controls"] = c

    # ---- per arm --------------------------------------------------------------------
    print(f"\n  {'arm':<14}{'fam':<5}{'cells':>6}{'excl':>5}{'delta':>9}{'rand':>8}{'d~':>8}"
          f"{'d~ forced':>11}{'must-fail':>11}{'plumb':>9}")
    fam_valid = {}
    for arm in list(PRIMARY) + list(SEPARATE) + list(SECONDARY_ONLY):
        rs = cells(arm)
        if not rs:
            print(f"  {arm:<14} no artifacts"); continue
        ok = [r for r in rs if r["delta"] <= GUARD]
        ex = [r for r in rs if r["delta"] > GUARD]
        fam = rs[0]["family"]
        mf = sum(r["must_fail_ok"] for r in rs)
        plumb = max(r["plumb_max_abs"] for r in rs)
        dt = float(np.mean([r["delta_tilde"] for r in ok])) if ok else float("nan")
        fam_valid.setdefault(fam, []).append(dt)
        out["arms"][arm] = {
            "family": fam, "cells": len(rs), "excluded": len(ex),
            "excluded_cells": [{"seed": r["seed"], "task": r["task"], "delta": r["delta"]} for r in ex],
            "delta_valid": float(np.mean([r["delta"] for r in ok])) if ok else None,
            "delta_forced": float(np.mean([r["delta"] for r in rs])),
            "delta_rand": float(np.mean([r["delta_rand"] for r in ok])) if ok else None,
            "delta_tilde_valid": dt,
            "delta_tilde_forced": float(np.mean([r["delta_tilde"] for r in rs])),
            "must_fail": f"{mf}/{len(rs)}", "plumb_max": plumb,
            "relay_rel_err": (float(np.mean([r["relay_rel_err"] for r in rs]))
                              if rs[0]["relay_rel_err"] is not None else None)}
        a = out["arms"][arm]
        print(f"  {arm:<14}{fam:<5}{len(rs):>6}{len(ex):>5}{a['delta_valid']:>9.4f}{a['delta_rand']:>8.4f}"
              f"{dt:>8.4f}{a['delta_tilde_forced']:>11.4f}{mf:>7}/{len(rs):<4}{plumb:>9.1e}")
        for r in ex:
            print(f"  {'':>19}EXCLUDED seed {r['seed']} k={r['task']}: delta {r['delta']:.3f} "
                  f"(train {r['delta_train']:.3f}, val {r['delta_val']:.3f}) — the fit failed on test only")

    for f in sorted(fam_valid):
        print(f"\n  {f} arms, mean delta~ over valid cells: {np.mean(fam_valid[f]):.4f}  "
              f"(per arm {[round(v, 3) for v in fam_valid[f]]})")
    out["family_delta_tilde"] = {f: float(np.mean(v)) for f, v in fam_valid.items()}

    # ---- primary correlation ---------------------------------------------------------
    X, Y, S, rawX, lab = [], [], [], [], []
    for arm, (path, key, fam) in PRIMARY.items():
        if not os.path.exists(path):
            print(f"  {arm}: cure artifact missing {path}"); continue
        g = gaps(path, key)
        for r in cells(arm):
            k = (r["seed"], r["task"])
            if k not in g or r["delta"] > GUARD:
                continue
            X.append(r["delta_tilde"]); rawX.append(r["delta"]); Y.append(g[k]["G"])
            S.append(r["seed"]); lab.append((arm, *k))
    n = len(X)
    print(f"\n  PRIMARY  rho(delta~, G) over {n} cells from {len(PRIMARY)} arms "
          f"(guard removed {48 - n} of 48)")
    if n >= 8:
        rho = stats.spearmanr(X, Y).statistic
        lo, hi = boot_rho(X, Y, S, rng)
        rho_raw = stats.spearmanr(rawX, Y).statistic
        fires = rho >= 0.6 and lo > 0
        print(f"    rho(delta~, G) = {rho:+.3f}   bootstrap over seeds [{lo:+.3f}, {hi:+.3f}]")
        print(f"    rho(raw delta, G) = {rho_raw:+.3f}   (comparison, sec 4 item 3)")
        out["primary"] = {"n": n, "rho_tilde": float(rho), "ci": [lo, hi],
                          "rho_raw": float(rho_raw), "fires": bool(fires)}
        print("\n    within-arm (12 cells each, low power, reported not scored):")
        wa = {}
        for arm in PRIMARY:
            xi = [X[i] for i in range(n) if lab[i][0] == arm]
            yi = [Y[i] for i in range(n) if lab[i][0] == arm]
            if len(xi) >= 6:
                wa[arm] = float(stats.spearmanr(xi, yi).statistic)
                print(f"      {arm:<14} rho {wa[arm]:+.3f}  (n={len(xi)})")
        out["within_arm"] = wa
        rd = ("The equivariance defect PREDICTS readout-repair loss." if fires else
              ("Positive but not at the registered bar." if rho > 0 else
               "The defect does not predict the loss."))
        print(f"\n  READING (sec 4): {rd}")
        out["reading"] = rd

    # ---- predictions ------------------------------------------------------------------
    print("\n  PREDICTIONS (sec 5)")
    preds = []
    def add(stmt, odds, hit, note=""):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<62} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")
    p = out.get("primary")
    add("rho(delta~, G) >= 0.6 with interval excluding zero", 40,
        None if not p else p["fires"], "" if not p else f"rho {p['rho_tilde']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}]")
    L, M = out["family_delta_tilde"].get("LSTM"), out["family_delta_tilde"].get("MLP")
    add("mean delta~ on LSTM arms exceeds MLP arms", 70, None if L is None or M is None else L > M,
        f"LSTM {L:.3f} vs MLP {M:.3f}" if L and M else "")
    mlp = [v for a, v in out["arms"].items() if v["family"] == "MLP"]
    lstm = [v for a, v in out["arms"].items() if v["family"] == "LSTM"]
    add("delta~ < 1 on the MLP arms (training moved toward equivariance)", 55,
        None if not mlp else all(v["delta_tilde_valid"] < 1 for v in mlp),
        ", ".join(f"{v['delta_tilde_valid']:.2f}" for v in mlp))
    add("delta~ > 1 on the LSTM arms", 50, None if not lstm else all(v["delta_tilde_valid"] > 1 for v in lstm),
        ", ".join(f"{v['delta_tilde_valid']:.2f}" for v in lstm))
    add("positive control below 0.01", 95, pc["pass"],
        f"max {max(max(v) for v in pc['delta_per_draw'].values()):.1e}")
    allmf = all(r["must_fail_ok"] for a in out["arms"] for r in cells(a))
    add("must-fail passes on every cell", 90, allmf,
        f"{sum(r['must_fail_ok'] for a in out['arms'] for r in cells(a))}/"
        f"{sum(len(cells(a)) for a in out['arms'])}")
    out["predictions"] = preds
    sc = [q for q in preds if q["outcome"] != "pending"]
    print(f"\n    scored {len(sc)}: fired {sum(q['outcome']=='fired' for q in sc)}, "
          f"miss {sum(q['outcome']=='MISS' for q in sc)}, "
          f"did not fire {sum(q['outcome']=='did not fire' for q in sc)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
