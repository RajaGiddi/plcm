"""Per-cell rho recompute with denominator assignment and propagated CIs.

WHY THIS EXISTS. cure_screen.py scored rho as a RATIO OF POOLED MEANS: it
averaged accuracies over all (seed, task) cells and divided once. Under that
form the RHO_MIN_DENOM guard is checked against the pooled denominator, which
is healthy by construction even when individual cells are degenerate -- E10's
per-cell denominators span 0.0026 to 0.556, a 200x range, and 4 of 24 cells sit
at or below the guard without ever tripping it. Pooling is itself a way of
hiding a failed cell.

  RULE (general): any guard on a ratio's denominator applies PER-CELL, never
  post-pooling.

DENOMINATOR ASSIGNMENT (amendment, made with live values known -- disclosed).
The bias adjustment subtracts R -- the convex probe's advantage over the era
model's deployed head -- from the ceiling, charging the recipe advantage against
the cure. That is correct only for cures that do NOT share the ceiling's fitting
recipe:

  * era-head cures (C0, C1, C0C1, C2) read through the stored era head and get
    no recipe benefit -> graded against the BIAS-ADJUSTED denominator.
  * recipe-matched cures (C3) fit their head with the same convex procedure as
    the ceiling itself, so the recipe gap cancels by construction -> graded
    against the RAW denominator. Dividing C3 by the R-shrunk denominator
    double-credits it, which is the whole of why E10's C3 read rho = 1.026,
    above its own ceiling.

The rationale is RECIPE SYMMETRY alone. No mechanism story about why R is large
is needed, and none is available: the "era head underfits on smaller data"
hypothesis is falsified (E10's per-task train counts vary 2.8% while R varies
6x, and R vs n_train correlates the wrong way, +0.58).

CIs. Per-cell binomial standard errors on n_test windows, propagated through
rho by first-order expansion in (cure, floor, refit, R), then pooled over valid
cells. Terms are treated as independent although they share a test set; that
positive correlation makes this an OVERESTIMATE of the interval, which is the
safe direction. The across-cell spread is reported separately -- it is
heterogeneity, not measurement error, and the two should not be conflated.

Usage:
    python scripts/rho_percell.py --cures runs/e10/cures_e10.json --bench e10
    python scripts/rho_percell.py --cures runs/e8/cures.json      --bench e8
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

RHO_MIN_DENOM = 0.02
ERA_HEAD_CURES = ["C0", "C1", "C0C1", "C2"]     # graded vs bias-adjusted denom
RECIPE_MATCHED = ["C3"]                          # graded vs raw denom
ALL_CURES = ERA_HEAD_CURES + RECIPE_MATCHED

# held-out test-subject window counts, per task
E10_NTEST = {0: 409, 1: 344, 2: 392, 3: 383}
E8_NTEST = {k: 2000 for k in range(4)}           # N_TEST cap in channel_decomp


def binom_se(p: float, n: int) -> float:
    """Binomial standard error of an accuracy estimate."""  # [scalar]
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def rho_and_se(cure: float, floor: float, refit: float, R: float,
               n: int, adjusted: bool) -> tuple[float, float, float]:
    """rho for one cell, its propagated SE, and the denominator used.

    rho = (cure - floor) / D,  D = refit - R - floor (adjusted) or refit - floor (raw)
    """
    D = refit - R - floor if adjusted else refit - floor
    if D <= 0:
        return float("nan"), float("nan"), D
    rho = (cure - floor) / D
    se_c, se_f, se_r = binom_se(cure, n), binom_se(floor, n), binom_se(refit, n)
    # d(rho)/d(cure) = 1/D ; d/d(floor) = (rho-1)/D ; d/d(refit) = -rho/D
    var = (se_c / D) ** 2 + ((rho - 1.0) * se_f / D) ** 2 + (rho * se_r / D) ** 2
    if adjusted:                                  # R = acc_rc - acc_ceiling, both binomial
        se_R = math.sqrt(binom_se(R + floor, n) ** 2 + binom_se(floor, n) ** 2)
        var += (rho * se_R / D) ** 2
    return rho, math.sqrt(var), D


def score_arm(rows: list[dict[str, Any]], ntest: dict[int, int],
              cure: str) -> dict[str, Any]:
    """Mean-of-ratios over cells whose ASSIGNED denominator clears the guard."""
    adjusted = cure in ERA_HEAD_CURES
    kept, excluded = [], []
    for r in rows:
        n = ntest[r["task"]]
        rho, se, D = rho_and_se(r[cure], r["acc_orig"], r["acc_refit"], r["R"],
                                n, adjusted)
        rec = {"seed": r["seed"], "task": r["task"], "D": D, "rho": rho, "se": se}
        (excluded if (D <= RHO_MIN_DENOM or rho != rho) else kept).append(rec)
    if not kept:
        return {"cure": cure, "n_kept": 0, "n_excl": len(excluded),
                "excluded": excluded, "mean": float("nan")}
    m = sum(k["rho"] for k in kept) / len(kept)
    se_pool = math.sqrt(sum(k["se"] ** 2 for k in kept)) / len(kept)
    spread = math.sqrt(sum((k["rho"] - m) ** 2 for k in kept) / max(len(kept) - 1, 1))
    return {"cure": cure, "adjusted": adjusted, "n_kept": len(kept),
            "n_excl": len(excluded), "excluded": excluded, "kept": kept,
            "mean": m, "se": se_pool, "spread": spread}


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-cell rho with guard at cell level")
    ap.add_argument("--cures", required=True)
    ap.add_argument("--bench", choices=["e10", "e8"], required=True)
    args = ap.parse_args()
    ntest = E10_NTEST if args.bench == "e10" else E8_NTEST
    data = json.load(open(args.cures))

    print(f"per-cell rho  |  {args.cures}  |  guard RHO_MIN_DENOM = {RHO_MIN_DENOM}")
    print(f"n_test per task: {ntest}")

    for arm, rows in data.items():
        rows = [r for r in rows if r["task"] in ntest]
        print("\n" + "=" * 78)
        print(f"{arm}   ({len(rows)} cells)")
        print("=" * 78)
        print(f"  {'cure':<6}{'denom':>10}{'kept':>7}{'excl':>6}"
              f"{'mean rho':>11}{'95% CI':>20}{'cell sd':>10}")
        for cure in ALL_CURES:
            s = score_arm(rows, ntest, cure)
            if s["n_kept"] == 0:
                print(f"  {cure:<6}{'--':>10}{0:>7}{s['n_excl']:>6}"
                      f"{'ALL CELLS EXCLUDED':>31}")
                continue
            lo, hi = s["mean"] - 1.96 * s["se"], s["mean"] + 1.96 * s["se"]
            print(f"  {cure:<6}{'adj' if s['adjusted'] else 'RAW':>10}"
                  f"{s['n_kept']:>7}{s['n_excl']:>6}{s['mean']:>+11.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{s['spread']:>10.3f}")

        ex = score_arm(rows, ntest, "C0")["excluded"]        # adjusted-denom exclusions
        if ex:
            names = ", ".join(f"s{e['seed']}/t{e['task']}(D={e['D']:+.4f})" for e in ex)
            print(f"\n  EXCLUDED (adjusted denom <= {RHO_MIN_DENOM}): "
                  f"{len(ex)}/{len(rows)} cells -- {names}")
        exr = score_arm(rows, ntest, "C3")["excluded"]
        if exr:
            names = ", ".join(f"s{e['seed']}/t{e['task']}(D={e['D']:+.4f})" for e in exr)
            print(f"  EXCLUDED (raw denom <= {RHO_MIN_DENOM}):      "
                  f"{len(exr)}/{len(rows)} cells -- {names}")
        else:
            print(f"  EXCLUDED (raw denom <= {RHO_MIN_DENOM}):      0/{len(rows)} cells")


if __name__ == "__main__":
    main()
