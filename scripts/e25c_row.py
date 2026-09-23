"""E25 C row -- Mummadi's objective at its own weight (docs/E25_prereg.md sec C).

THE CONTRACTED VERDICT IS REPORTED FIRST. The addendum's two estimators are
reported BESIDE it, never in place of it, and both are computed on lambda = 1
and lambda = 40 by the same function, same cells, same order -- comparing two
lambda values under two estimators is not permitted (addendum, "both lambda or
neither").

    contracted   A4 - A3 against E21's floor: max(spread over realizations,
                 mean per-cell jitter spread)          e21_row.py:139-141
    estimator 1  paired over realizations, t95 interval, blocked on the wiring
    estimator 2  per-cell sign count out of 36

LEVEL 0 IS NOT READ. One realization at eps = 0 and a deterministic swap search
make both floor components zero BY CONSTRUCTION, so any non-zero delta clears a
floor that cannot be non-zero (catch 35(a)).

A3 comes from E21's artifacts: it is TENT, lambda does not enter it, and C ran
A4 alone. What proves the two harnesses are the same is C-ID on the arms that do
not depend on lambda -- A0, A1, A6 and SNAP must reproduce E21 EXACTLY. That is
non-tautological: a different harness, benchmark or checkpoint moves them.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

LAM40 = "runs/e25/c/lam40/level{li}_real{r}.json"
LAM1 = "runs/e21/swap/level{li}_real{r}.json"
SIGNFLIP = "runs/e25/c/signflip/level1_real{r}.json"
LEVELS = {0: 0, 1: 1, 2: 2, 3: 3}          # level_idx -> m
CID_ARMS = ["A0", "A1", "A6", "SNAP"]
TOL = 1e-6


def load(tmpl, li, r):
    p = tmpl.format(li=li, r=r)
    return json.load(open(p)) if os.path.exists(p) else None


def reals(li):
    return range(3) if li > 0 else [0]


def pooled(rows, arm):
    return float(np.mean([r[arm] for r in rows]))


def floor_of(per_real, jitters):
    """E21's floor, replicated: max(realization spread, mean per-cell jitter spread)."""
    return max(float(max(per_real) - min(per_real)), float(np.mean(jitters)))


def estimators(src, hi, lo, label):
    """BOTH addendum estimators, on whatever source is handed in. Same code for
    lambda = 1 and lambda = 40 -- that is the addendum's condition, enforced by
    there being one function."""
    out = {}
    for li, m in LEVELS.items():
        if li == 0:
            out[str(m)] = {"readable": False,
                           "why": "one realization at eps=0 and a deterministic search: "
                                  "both floor components are zero by construction (catch 35a)"}
            continue
        per_real, alld = [], []
        for r in reals(li):
            rows_hi, rows_lo = src(li, r)
            if rows_hi is None or rows_lo is None:
                break
            d = [a - b for a, b in zip([x[hi] for x in rows_hi], [x[lo] for x in rows_lo])]
            per_real.append(float(np.mean(d)))
            alld += d
        if not alld:
            out[str(m)] = {"readable": False, "why": "artifacts missing"}
            continue
        a = np.array(alld)
        t95 = float(stats.t.ppf(0.975, len(per_real) - 1) * np.std(per_real, ddof=1)
                    / np.sqrt(len(per_real))) if len(per_real) > 1 else float("nan")
        mu = float(np.mean(per_real))
        out[str(m)] = {"readable": True, "mean": mu, "per_real": per_real,
                       "paired_t95": t95, "paired_clears_zero": bool(mu - t95 > 0),
                       "cells_positive": int((a > 0).sum()), "n_cells": int(a.size),
                       "realizations_positive": int(sum(v > 0 for v in per_real)),
                       "n_realizations": len(per_real)}
    out["_label"] = label
    return out


def main():
    ap = argparse.ArgumentParser(description="E25 C row")
    ap.add_argument("--out", default="runs/e25c_row.json")
    args = ap.parse_args()
    print("=" * 104 + "\nE25 C ROW   Mummadi's objective at its own weight (lambda = 40)\n" + "=" * 104)
    res = {"contract": "docs/E25_prereg.md sec C + addendum"}

    # ---- C-ID: the lambda-independent arms must reproduce E21 exactly -----------
    print("\nC-ID  arms that do not depend on lambda must reproduce E21 EXACTLY")
    worst, missing = defaultdict(float), []
    for li in LEVELS:
        for r in reals(li):
            n40, n1 = load(LAM40, li, r), load(LAM1, li, r)
            if n40 is None or n1 is None:
                missing.append((li, r)); continue
            i1 = {(x["seed"], x["task"]): x for x in n1["rows"]}
            for x in n40["rows"]:
                y = i1[(x["seed"], x["task"])]
                for a in CID_ARMS:
                    worst[a] = max(worst[a], abs(x[a] - y[a]))
    if missing:
        print(f"  MISSING artifacts: {missing}")
    cid_ok = bool(worst) and max(worst.values()) <= TOL
    for a in CID_ARMS:
        print(f"    {a:5} max |delta| = {worst[a]:.3e}")
    print(f"  C-ID -> {'PASS' if cid_ok else 'FAIL -- nothing read'}")
    res["c_id"] = {"arms": dict(worst), "pass": cid_ok, "tol": TOL, "missing": missing}
    if not cid_ok:
        json.dump(res, open(args.out, "w"), indent=2, default=float)
        print(f"  wrote {args.out}"); return 1

    # ---- contracted verdict ------------------------------------------------------
    print("\nCONTRACTED VERDICT  A4(lambda=40) - A3, against E21's floor")
    contracted = {}
    for li, m in LEVELS.items():
        pr4, pr3, j4, j3 = [], [], [], []
        for r in reals(li):
            n40, n1 = load(LAM40, li, r), load(LAM1, li, r)
            pr4.append(pooled(n40["rows"], "A4")); pr3.append(pooled(n1["rows"], "A3"))
            j4.append(float(np.mean([x.get("A4_spread", 0.0) for x in n40["rows"]])))
            j3.append(float(np.mean([x.get("A3_spread", 0.0) for x in n1["rows"]])))
        if li == 0:
            contracted[str(m)] = {"A4": float(np.mean(pr4)), "A3": float(np.mean(pr3)),
                                  "delta": float(np.mean(pr4) - np.mean(pr3)),
                                  "readable": False, "why": "catch 35(a): no floor exists"}
            print(f"    m={m}  A4 {np.mean(pr4):.4f}  A3 {np.mean(pr3):.4f}  "
                  f"delta {np.mean(pr4)-np.mean(pr3):+.4f}  -> NOT READ (no floor)")
            continue
        fl = max(floor_of(pr4, j4), floor_of(pr3, j3))
        d = float(np.mean(pr4) - np.mean(pr3))
        contracted[str(m)] = {"A4": float(np.mean(pr4)), "A3": float(np.mean(pr3)),
                              "delta": d, "floor": fl, "beats": bool(d > fl), "readable": True}
        print(f"    m={m}  A4 {np.mean(pr4):.4f}  A3 {np.mean(pr3):.4f}  delta {d:+.4f} "
              f"[floor {fl:.4f}] -> {'BEATS' if d > fl else 'within floor' if abs(d) <= fl else 'BELOW'}")
    res["contracted"] = contracted

    # ---- addendum estimators, both lambdas, one function ------------------------
    src40 = lambda li, r: ((load(LAM40, li, r) or {}).get("rows"), (load(LAM1, li, r) or {}).get("rows"))
    src1 = lambda li, r: ((load(LAM1, li, r) or {}).get("rows"), (load(LAM1, li, r) or {}).get("rows"))
    e40 = estimators(src40, "A4", "A3", "lambda=40")
    e1 = estimators(src1, "A4", "A3", "lambda=1")
    res["addendum"] = {"lambda40": e40, "lambda1": e1}
    print("\nADDENDUM  paired over realizations, and the per-cell sign count")
    print(f"    {'m':>3} {'lam':>5} {'mean d':>9} {'t95':>9} {'paired':>14} {'reals +':>9} {'cells +':>9}")
    for lab, e in (("40", e40), ("1", e1)):
        for m in ("1", "2", "3"):
            v = e.get(m, {})
            if not v.get("readable"):
                continue
            print(f"    {m:>3} {lab:>5} {v['mean']:+9.4f} {v['paired_t95']:9.4f} "
                  f"{'clears zero' if v['paired_clears_zero'] else 'includes zero':>14} "
                  f"{v['realizations_positive']:>4}/{v['n_realizations']:<4} "
                  f"{v['cells_positive']:>4}/{v['n_cells']:<4}")
    print("    Realization-level unanimity is never reported as consistency without the")
    print("    cell count beside it (catch 35(b)).")

    # ---- sign-flip must-fail -----------------------------------------------------
    print("\nMUST-FAIL  sign-flipped diversity (reward collapse) must do WORSE than A3, 36/36")
    worse, tot, per = 0, 0, []
    for r in reals(1):
        sf, n1 = load(SIGNFLIP, 1, r), load(LAM1, 1, r)
        if sf is None:
            print(f"    realization {r}: artifact missing"); continue
        i1 = {(x["seed"], x["task"]): x for x in n1["rows"]}
        for x in sf["rows"]:
            a3 = i1[(x["seed"], x["task"])]["A3"]
            tot += 1; worse += int(x["A4"] < a3)
            per.append({"seed": x["seed"], "task": x["task"], "A4_signflip": x["A4"], "A3": a3})
    mf = {"worse_than_A3": worse, "cells": tot, "pass": tot > 0 and worse == tot, "per_cell": per}
    res["must_fail_signflip"] = mf
    print(f"    {worse}/{tot} cells worse than A3 -> {'PASS' if mf['pass'] else 'FAIL'}")

    # ---- flag regression ----------------------------------------------------------
    reg = None
    for p in ("runs/e25/c_dl/lam1_regression_raw.json", "runs/e25/c/lam1_regression_raw.json"):
        if os.path.exists(p):
            reg = json.load(open(p)); break
    if reg:
        i1 = {(x["seed"], x["task"]): x for x in load(LAM1, 1, 0)["rows"]}
        w = {}
        for x in reg["rows"]:
            y = i1[(x["seed"], x["task"])]
            for a in ("A0", "A1", "A2", "A3", "A4", "A5", "A5d", "A6", "A7", "SNAP"):
                if a in x and a in y:
                    w[a] = max(w.get(a, 0.0), abs(x[a] - y[a]))
        exact = [a for a, v in w.items() if v == 0.0]
        res["flag_regression"] = {"per_arm_max_abs": w, "bit_identical_arms": exact,
                                  "note": "A2 and A7 route through refit_probe; their deltas are "
                                          "whole test windows (1/344 and 2/409), which is E21's "
                                          "ruled A2/A7 window floor arriving independently"}
        print(f"\nFLAG REGRESSION  bit-identical on {len(exact)}/{len(w)} arms: {sorted(exact)}")
        for a in ("A2", "A7"):
            if a in w:
                print(f"    {a}: {w[a]:.3e}  (refit-based arm; E21's ruled window floor)")

    # ---- predictions ---------------------------------------------------------------
    print("\nPREDICTIONS (sec C)")
    preds = []

    def add(stmt, odds, hit):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v})
        print(f"  {stmt:<74} ~{odds:>2}%  {v}")

    c1 = contracted.get("1", {})
    add("A4 at lambda=40 beats A3 by more than floor at m=1", 40,
        None if not c1.get("readable") else c1["beats"])
    d40 = e40.get("1", {}).get("mean"); d1v = e1.get("1", {}).get("mean")
    add("lambda=40 beats lambda=1", 45, None if d40 is None or d1v is None else d40 > d1v)
    # "recovers the exact map": A4 within resolution of A6, the true-map relayout
    rec, tot_r = 0, 0
    for r in reals(1):
        n40 = load(LAM40, 1, r)
        for x in n40["rows"]:
            n = 409 if x["task"] == 0 else {1: 344, 2: 392, 3: 383}[x["task"]]
            tot_r += 1; rec += int(x["A4"] >= x["A6"] - 1.96 * np.sqrt(0.25 / n))
    add("A4 at lambda=40 recovers the exact map at m=1 on >= 27/36", 30,
        None if tot_r == 0 else rec >= 27)
    add("sign-flip must-fail passes 36/36", 85, None if tot == 0 else mf["pass"])
    res["predictions"] = preds
    res["exact_map_recovered"] = {"cells": rec, "of": tot_r}
    scored = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n  scored {len(scored)}: fired {sum(p['outcome']=='fired' for p in scored)}, "
          f"miss {sum(p['outcome']=='MISS' for p in scored)}, "
          f"did not fire {sum(p['outcome']=='did not fire' for p in scored)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
