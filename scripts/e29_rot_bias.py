"""E29 Part B — the rotation estimator's transfer bias.

§0 withdrew v1's identifiability argument and replaced the question. The
argument was that gravity fixes tilt and leaves a spin ambiguity; but
`har_subject._apply` standardizes BEFORE applying the map, so the reference
mean is zero (||mu_ref|| = 1.14e-4) and there are no mean directions to align.
Rotation turns out to be FULLY identifiable from covariance alone -- 0.00
degrees, residual 0.0000, on matched subjects -- because the three triples
share one R and the cross-triple blocks pin it.

WHAT IS LEFT TO MEASURE IS THE BIAS. Across disjoint subjects the error was
8.72 degrees for EVERY true rotation, and that constant is a property with a
proof, not a defect (catch 35 was checked and cleared): writing B = R.A and
using the orthogonal invariance of the Frobenius norm,

    ||blk(RA) S blk(RA)^T - blk(R) S_Q blk(R)^T||  =  ||blk(A) S blk(A)^T - S_Q||

so the objective does not depend on R at all. The estimator is exactly
equivariant and its error is a subject-transfer bias. This script measures the
DISTRIBUTION of that bias over pairings, over n, and over pool composition.

THE RESTART COUNT IS A REQUIRED ARGUMENT WITH NO DEFAULT, and its positive
control runs before any measurement. The objective has a strong spurious
minimum near 180 degrees; at 1-2 restarts the optimizer lands in it and the
equivariance check fails with a 163 degree spread, at 8 it passes at 0.0000.
A control that has only ever passed is indistinguishable from one that cannot
fail (catch 25), so the failing case is demonstrated here every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e29_common import (                                          # noqa: E402
    N_CHANNELS, NUM_TASKS, MIN_RESTARTS, blockdiag, geodesic_deg, fit_rotation,
    window_stats, pool_indices, synthetic_pool)

N_GRID = [16, 64, 256, 1024]
SPECS = ["two_dynamic", "static_plus_dynamic", "three_static", "all6", "all6_skew80"]


def three_rotations(seed=11):
    return [("identity", np.eye(3)),
            ("spec_30z", Rot.from_euler("z", 30, degrees=True).as_matrix()),
            ("random", Rot.random(random_state=seed).as_matrix())]


def estimate(S_ref, C_ref, A_query, R_true, n_start, seed):
    """Apply R_true to a query pool, estimate it back, report the error."""
    S_q, C_q = window_stats(A_query @ blockdiag(R_true).T)
    Rh, resid = fit_rotation(S_ref, C_ref, S_q, C_q, n_start=n_start, seed=seed)
    return geodesic_deg(R_true, Rh), resid


def main():
    ap = argparse.ArgumentParser(description="E29 Part B: rotation transfer bias")
    ap.add_argument("--n-start", type=int, required=True,
                    help=f"SO(3) restarts; required, no default, must be >= {MIN_RESTARTS}")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="runs/e29/rot_bias.json")
    ap.add_argument("--controls-out", default="runs/e29/controls_rot.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.n_start < MIN_RESTARTS:
        raise SystemExit(f"--n-start {args.n_start} < {MIN_RESTARTS}")
    global N_GRID
    if args.smoke:
        N_GRID = [64]
    t0 = time.time()
    print("=" * 106 + "\nE29 PART B — the rotation estimator's transfer bias\n" + "=" * 106)
    print(f"  n_start = {args.n_start} (required argument, minimum {MIN_RESTARTS})")

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128, no_shift=True)
    fp = bench.partition_fingerprint()
    fp_shifted = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128,
                                     no_shift=False).partition_fingerprint()
    print(f"  partition {fp} (no_shift); shifted benchmark hashes {fp_shifted}")
    pools = {g: [a.numpy() for a in bench.raw_windows(g, True)] for g in range(NUM_TASKS)}

    # ------------------- controls, before any measurement ----------------------
    print("\n  CONTROLS")
    ctl = {}
    rngc = np.random.default_rng(7)
    # (1) positive control for the EQUIVARIANCE CHECK: it must fail when starved.
    Sr, Cr = window_stats(pools[0][0])
    # Whether a STARVED optimizer lands in the spurious ~180 degree minimum is
    # seed-dependent: on one seed a single restart happens to start in the right
    # basin and the check passes, which would make this control look vacuous when
    # it is not. So the control SCANS seeds and asserts the failure EXISTS,
    # rather than hoping one arbitrary seed exhibits it.
    starved, healthy = {}, {}
    Xc = pools[3][0]
    for ns in (1, 2):                       # the refused regime, reached on purpose
        spreads = []
        for sd in range(8):
            e = [_starved(Sr, Cr, Xc, R, ns, args.seed + sd) for _, R in three_rotations()]
            spreads.append(float(max(e) - min(e)))
        starved[ns] = {"spreads_by_seed": spreads, "max_spread": float(max(spreads)),
                       "seeds_failing": int(sum(s > 1.0 for s in spreads)), "of": len(spreads)}
    e = [estimate(Sr, Cr, Xc, R, args.n_start, args.seed)[0] for _, R in three_rotations()]
    healthy[args.n_start] = {"errors": e, "spread": float(max(e) - min(e))}
    ok_starved = all(v["seeds_failing"] > 0 for v in starved.values())
    ok_healthy = all(v["spread"] < 1e-3 for v in healthy.values())
    ctl["equivariance_positive_control"] = {
        "starved": starved, "healthy": healthy,
        "can_fail": ok_starved, "passes_when_healthy": ok_healthy,
        "pass": bool(ok_starved and ok_healthy)}
    for ns, v in starved.items():
        print(f"    equivariance @ n_start={ns}: fails on {v['seeds_failing']}/{v['of']} seeds, "
              f"max spread {v['max_spread']:7.2f} deg -> "
              f"{'CAN FAIL (control is live)' if v['seeds_failing'] else 'NEVER FAILS — control is vacuous'}")
    for ns, v in healthy.items():
        print(f"    equivariance @ n_start={ns}: spread {v['spread']:8.4f} deg -> "
              f"{'PASSES' if v['spread'] < 1e-3 else 'FAIL'}")
    # (3) must-fail FIRST, because it supplies the yardstick the positive
    # control is scored against. Isotropic covariance identifies no rotation.
    iso_ref = synthetic_pool(4096, np.random.default_rng(9), anisotropic=False)
    Si, Ci = window_stats(iso_ref)
    e_iso = [estimate(Si, Ci,
                      synthetic_pool(2048, np.random.default_rng(90 + i), anisotropic=False),
                      Rot.random(random_state=100 + i).as_matrix(),
                      args.n_start, args.seed)[0] for i in range(6)]
    unif = np.degrees(np.linalg.norm(Rot.random(2000, random_state=3).as_rotvec(), axis=1))
    lo, hi = float(np.percentile(unif, 5)), float(np.percentile(unif, 95))

    # (2) positive synthetic, SCORED AGAINST THE MUST-FAIL'S OWN DISTRIBUTION.
    #
    # BAR CHANGE, RECORDED. The first version drew reference and query from the
    # same array and was scored at "< 1 degree", which suited a self-match where
    # the error is ~0. Fixing the control to draw INDEPENDENTLY (see
    # synthetic_pool) reintroduces genuine finite-sample error, so the old bar
    # was priced against a construction that no longer exists. It is repriced on
    # a principle rather than on the observed value, which is the only way to
    # reprice a bar after seeing the number: (a) the error must fall with n --
    # sampling error shrinks, a structural failure does not -- and (b) at the
    # largest n it must sit below the 5th percentile of the uniform-rotation
    # distribution, i.e. the estimator must do detectably better than no
    # information at all. Both criteria are scale-free and neither was chosen by
    # looking at 2.86 degrees.
    syn_curve = {}
    for n in (256, 1024, 4096):
        syn_ref = synthetic_pool(4096, np.random.default_rng(8), cov_seed=1)
        syn_qry = synthetic_pool(n, np.random.default_rng(80 + n), cov_seed=1)
        Ss, Cs = window_stats(syn_ref)
        syn_curve[n] = estimate(Ss, Cs, syn_qry, three_rotations()[2][1],
                                args.n_start, args.seed)[0]
    falls = syn_curve[4096] < syn_curve[256]
    beats_chance = syn_curve[4096] < lo
    ctl["positive_synthetic"] = {"error_by_n_deg": syn_curve, "falls_with_n": bool(falls),
                                 "below_uniform_5th_pct": bool(beats_chance),
                                 "uniform_5th_pct": lo, "pass": bool(falls and beats_chance)}
    print(f"    positive synthetic (independent draws): "
          + ", ".join(f"n={n} {v:.2f}°" for n, v in syn_curve.items())
          + f" | falls with n: {falls} | below uniform 5th pct ({lo:.0f}°): {beats_chance} -> "
          + ('PASS' if falls and beats_chance else 'FAIL'))

    inside = sum(lo <= e <= hi for e in e_iso)
    ctl["must_fail_isotropic"] = {"errors_deg": e_iso, "uniform_90pct_band": [lo, hi],
                                  "inside_band": inside, "of": len(e_iso),
                                  "pass": inside >= 4}
    print(f"    must-fail (isotropic): errors {[round(e,1) for e in e_iso]} deg; "
          f"uniform-rotation 90% band [{lo:.0f},{hi:.0f}] -> {inside}/6 inside -> "
          f"{'PASS' if inside >= 4 else 'FAIL'}")
    ctl["partition_fingerprint"], ctl["partition_fingerprint_shifted"] = fp, fp_shifted
    os.makedirs(os.path.dirname(args.controls_out) or ".", exist_ok=True)
    json.dump(ctl, open(args.controls_out, "w"), indent=2, default=float)
    if not all(v.get("pass", True) for v in ctl.values() if isinstance(v, dict)):
        print("\n  CONTROLS FAILED — not reporting measurements"); return 1

    # ------------------- 1. bias across subject pairings -----------------------
    out = {"n_start": args.n_start, "partition_fingerprint": fp,
           "partition_fingerprint_shifted": fp_shifted, "controls": ctl,
           "pairings": {}, "composition": {}}
    print("\n  1. BIAS ACROSS SUBJECT PAIRINGS (reference group -> query group, all ordered pairs)")
    print(f"  {'n':>6}{'pairings':>10}{'median':>10}{'IQR':>18}{'min':>8}{'max':>8}"
          f"{'equivariance spread':>22}")
    rng = np.random.default_rng(args.seed)
    for n in N_GRID:
        errs, spreads, cells = [], [], []
        for gi in range(NUM_TASKS):
            Sr, Cr = window_stats(pools[gi][0])
            for gj in range(NUM_TASKS):
                if gi == gj:
                    continue
                Xq, yq = pools[gj]
                idx = pool_indices(yq, "all6", n, rng)
                if idx is None:
                    continue
                per_R = [estimate(Sr, Cr, Xq[idx], R, args.n_start, args.seed)
                         for _, R in three_rotations()]
                e = [p[0] for p in per_R]
                errs.append(float(np.mean(e))); spreads.append(float(max(e) - min(e)))
                cells.append({"ref": gi, "query": gj, "errors_deg": e,
                              "equivariance_spread": float(max(e) - min(e)),
                              "residual": float(per_R[0][1])})
        q1, q3 = np.percentile(errs, [25, 75])
        out["pairings"][f"n{n}"] = {"median_deg": float(np.median(errs)), "iqr": [float(q1), float(q3)],
                                    "min": float(min(errs)), "max": float(max(errs)),
                                    "max_equivariance_spread": float(max(spreads)),
                                    "n_pairings": len(errs), "cells": cells}
        print(f"  {n:>6}{len(errs):>10}{np.median(errs):>9.2f}°"
              f"{f'[{q1:.2f}, {q3:.2f}]':>18}{min(errs):>7.2f}°{max(errs):>7.2f}°"
              f"{max(spreads):>21.6f}°")

    # ------------------- 2. bias against pool composition ----------------------
    print("\n  2. BIAS AGAINST POOL COMPOSITION (reference group 0 -> query group 3)")
    Sr, Cr = window_stats(pools[0][0])
    Xq, yq = pools[3]
    print(f"  {'pool':<24}" + "".join(f"{f'n={n}':>12}" for n in N_GRID))
    for spec in SPECS:
        row = []
        for n in N_GRID:
            idx = pool_indices(yq, spec, n, np.random.default_rng(args.seed + n))
            row.append(None if idx is None else
                       estimate(Sr, Cr, Xq[idx], three_rotations()[2][1], args.n_start, args.seed)[0])
        out["composition"][spec] = row
        print(f"  {spec:<24}" + "".join(f"{(f'{v:.2f}°' if v is not None else '--'):>12}" for v in row))

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}  ({out['seconds']}s)")
    return 0


def _starved(S_ref, C_ref, A, R_true, n_start, seed):
    """The starved fit, bypassing fit_rotation's guard ON PURPOSE and only here.

    fit_rotation refuses n_start < MIN_RESTARTS, which is the point of the
    guard. The positive control has to reach the refused regime to show the
    check can fail, so it reimplements the loop -- narrowly, in one place, for
    the control alone. Every measurement goes through fit_rotation.
    """
    from scipy.optimize import minimize
    S_q, C_q = window_stats(A @ blockdiag(R_true).T)
    nS, nC = (S_ref ** 2).sum(), (C_ref ** 2).sum()

    def obj(v):
        B = blockdiag(Rot.from_rotvec(v).as_matrix())
        return ((B @ S_ref @ B.T - S_q) ** 2).sum() / nS + ((B @ C_ref @ B.T - C_q) ** 2).sum() / nC

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        r = minimize(obj, Rot.random(random_state=int(rng.integers(1e6))).as_rotvec(),
                     method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    return geodesic_deg(R_true, Rot.from_rotvec(best.x).as_matrix())


if __name__ == "__main__":
    sys.exit(main())
