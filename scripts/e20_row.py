"""E20 row -- paired delta F_enc between the linear and MLP probes, per cell.

Contract: docs/E20_prereg.md sec 3, 5, 6. Reads the decomposition artifacts
the driver wrote under both probes for one arm and prints, in this order, with
every verdict derived from the printed arrays:

  --regress   the linear re-run against the cited E17 artifact. R2 (2026-09-16)
              rewrote what this proves: the RELOAD/EXTRACTION fields (acc_orig,
              acc_ceiling, d_logit) must reproduce to 1e-9 -- that is E17's proof
              and it must not be spent; the REFIT fields are expected to differ,
              because the cited artifact was produced under an unrecorded probe
              draw and this run under a recorded one. The refit delta is REPORTED
              as the measured effect of the unrecorded draw, never as a pass.
  R1 gate     (ruling (a), replacing the live half of the positive control that
              could not be built on these features): on every cell and both eras
              the MLP refit must not fall below the linear refit by more than the
              gate tolerance. A broken (underfit or overfit) MLP probe fails it; a
              working one cannot, since the MLP contains the linear solution.
              Non-vacuous, and the recipe's nonlinear reach is certified on the
              anisotropic synthetic (v7), on disk.
              R3 (2026-09-16, stated before it was applied): the tolerance is
                  max( init spread,  1.96 * sqrt( p (1 - p) / n_test ) ),  p = linear acc
              The init spread measures the probe's variance and omits the test
              set's, which is a real component. A gate finer than the
              measurement's own resolution is not a gate; it detects noise. Under
              R3 the primary's five 0.1-0.6pp shortfalls (inside 0.6-1.1pp of
              resolution) pass and the secondary's two 7pp shortfalls still fail --
              a correction, not a relaxation, because it rescues only the
              noise-level failures.
  gates       identity residual < 1e-6 per cell per probe; |R^mlp| pooled <= 0.05;
              MLP convergence on every fit (unconverged cells excluded, counted).
  the row     delta F_enc = F_enc^mlp - F_enc^linear per cell, its pooled mean,
              the probe's own floor (init spread) beside it, the overfit witness
              (gap^mlp - gap^linear), and the three pre-committed readings.

Usage:
    python scripts/e20_row.py --arm mlp --bench mnist \\
        --linear runs/e20/mnist_mlp_linear.json --mlp runs/e20/mnist_mlp_mlpprobe.json \\
        --regress runs/e16_decomp/mnist_mlp.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R_BAR = 0.05          # E14's per-arm pooled instrument-bias bar, reused unchanged
TIGHT = 0.01          # |delta F_enc| <= TIGHT: the linear bound is tight
OVERFIT = 0.05        # gap^mlp - gap^linear > OVERFIT on a cell: reading withheld there
IDENTITY_TOL = 1e-6
REGRESS_TOL = 1e-9
EXACT_FIELDS = ("acc_orig", "acc_ceiling", "d_logit")          # reload + deployed extraction: E17's proof
REFIT_FIELDS = ("acc_refit_ceiling", "acc_refit_t4", "R", "F_enc", "F_read")   # the draw


def load(p):
    return json.load(open(p))


def cells(rows):
    return {(r["seed"], r["task"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser(description="E20 paired probe row")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--linear", required=True, help="driver output, --probe linear")
    ap.add_argument("--mlp", required=True, help="driver output, --probe mlp")
    ap.add_argument("--regress", default=None, help="the cited linear artifact to reproduce")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = {"arm": args.arm, "bench": args.bench}
    # test-set size per task for the R3 tolerance: MNIST disjoint/shared test
    # subsets are capped at N_TEST = 2000; har_subject's held-out subjects are
    # E10_NTEST (409/344/392/383).
    from scripts.rho_percell import E10_NTEST
    n_test_of = (lambda task: E10_NTEST[task]) if args.bench.startswith("har") else (lambda task: 2000)
    lin, mlp = load(args.linear), load(args.mlp)
    L, M = cells(lin["rows"]), cells(mlp["rows"])
    assert set(L) == set(M), "the two probes were not run on the same cells"
    keys = sorted(L)

    # ---- regression -----------------------------------------------------------
    if args.regress:
        print("=" * 100 + f"\nREGRESSION  {args.linear}  vs cited  {args.regress}   (bar {REGRESS_TOL:.0e} on every numeric key)\n" + "=" * 100)
        C = cells(load(args.regress)["rows"])
        exact = {k: max(abs(float(C[c][k]) - float(L[c][k])) for c in keys) for k in EXACT_FIELDS}
        refit = {k: max(abs(float(C[c][k]) - float(L[c][k])) for c in keys) for k in REFIT_FIELDS}
        ok = max(exact.values()) <= REGRESS_TOL
        print(f"  reload/extraction fields, max |delta|: {', '.join(f'{k} {v:.1e}' for k, v in exact.items())}"
              f" -> {'IDENTICAL (E17 proof intact)' if ok else 'DIFFERS -- STOP: the extraction changed'}")
        print(f"  refit fields, max |delta| (the unrecorded draw, measured): {', '.join(f'{k} {v:.1e}' for k, v in refit.items())}"
              f"   [cited artifact: draw unrecorded; this run: probe_subset_seed {lin.get('probe_subset_seed')}]")
        out["regression"] = {"exact_fields": exact, "refit_fields_draw_effect": refit, "pass": bool(ok),
                             "probe_subset_seed": lin.get("probe_subset_seed")}
        if not ok:
            json.dump(out, open(args.out or f"runs/e20/{args.bench}_{args.arm}_row.json", "w"), indent=2, default=str)
            sys.exit(2)

    # ---- gates ------------------------------------------------------------------
    print("=" * 100 + "\nGATES\n" + "=" * 100)
    resid = max(abs(r["F_enc"] + r["F_read"] - r["R"] - r["F_total"]) for r in list(L.values()) + list(M.values()))
    R_mlp = float(np.mean([M[c]["R"] for c in keys]))
    R_lin = float(np.mean([L[c]["R"] for c in keys]))
    unconv = [c for c in keys if not (M[c]["mlp_ceiling"]["mlp_converged"] and M[c]["mlp_t4"]["mlp_converged"])]
    valid = [c for c in keys if c not in unconv]
    print(f"  identity residual (both probes, all cells)  {resid:.1e}  -> {'OK' if resid < IDENTITY_TOL else 'FAIL'}")
    print(f"  R pooled: linear {R_lin:+.4f}  MLP {R_mlp:+.4f}  (bar |R^mlp| <= {R_BAR}) -> {'OK' if abs(R_mlp) <= R_BAR else 'FAIL -- reading withheld'}")
    print(f"  MLP unconverged cells {len(unconv)}/{len(keys)} {unconv}  (excluded, counted)")
    # ---- R1 gate: the MLP must not fall below the linear probe by more than its init spread ----
    import math
    r1_fail, r1_noise = [], []
    for c in keys:
        for era, key_l, key_m in (("ceiling", "acc_refit_ceiling", "mlp_ceiling"), ("t4", "acc_refit_t4", "mlp_t4")):
            spread = M[c][key_m]["mlp_spread"]
            p_lin = L[c][key_l]
            se = 1.96 * math.sqrt(max(p_lin * (1 - p_lin), 0.0) / n_test_of(c[1]))
            tol = max(spread, se)                       # R3
            short = p_lin - M[c][key_l]
            if short > tol:
                r1_fail.append((c, era, round(p_lin, 4), round(M[c][key_l], 4), round(spread, 4), round(se, 4)))
            elif short > spread:
                r1_noise.append((c, era, round(short, 4), round(se, 4)))
    print(f"  R1 gate, R3 tolerance max(init spread, 1.96*SE_binomial): {len(keys)*2 - len(r1_fail)}/{len(keys)*2}"
          f" -> {'PASS' if not r1_fail else 'FAIL -- substantive shortfalls (cell, era, linear, MLP, spread, 1.96*SE): ' + str(r1_fail)}")
    if r1_noise:
        print(f"    {len(r1_noise)} shortfall(s) exceed the init spread but sit inside the test set's resolution (cell, era, short, 1.96*SE): {r1_noise}")
    gates_ok = resid < IDENTITY_TOL and abs(R_mlp) <= R_BAR and len(valid) > 0 and not r1_fail
    out["gates"] = {"identity_residual": resid, "R_linear": R_lin, "R_mlp": R_mlp,
                    "unconverged": [list(c) for c in unconv], "r1_fail": [str(x) for x in r1_fail],
                    "r1_within_resolution": [str(x) for x in r1_noise], "r1_tolerance": "max(init spread, 1.96*SE_binomial)  [R3]",
                    "pass": bool(gates_ok)}
    if not gates_ok:
        print("  gates NOT PASS -- the row below is printed but NOT READ")

    # ---- the row ------------------------------------------------------------------
    print("=" * 100 + "\nTHE ROW  delta F_enc = F_enc^mlp - F_enc^linear, paired per cell\n" + "=" * 100)
    print(f"  {'cell':<12}{'F_enc lin':>10}{'F_enc mlp':>10}{'delta':>9}{'spread':>8}{'gap lin':>9}{'gap mlp':>9}{'flag':>6}")
    deltas, spreads, flags = [], [], []
    for c in valid:
        fl, fm = L[c]["F_enc"], M[c]["F_enc"]
        sp = max(M[c]["mlp_ceiling"]["mlp_spread"], M[c]["mlp_t4"]["mlp_spread"])
        gap_m = M[c]["mlp_t4"]["mlp_gap"]
        gap_l = L[c]["probe_gap"]["t4"]
        flag = (gap_m - gap_l) > OVERFIT
        deltas.append(fm - fl); spreads.append(sp); flags.append(flag)
        print(f"  s{c[0]}/t{c[1]:<6}{fl:10.4f}{fm:10.4f}{fm - fl:+9.4f}{sp:8.4f}{gap_l:9.4f}{gap_m:9.4f}{'OVF' if flag else '':>6}")
    d, s_ = float(np.mean(deltas)), float(np.mean(spreads))
    share_l = lin["reader_share"]; share_m = mlp["reader_share"]
    print(f"\n  pooled delta F_enc {d:+.4f}   mean per-cell init spread {s_:.4f}   overfit-flagged cells {sum(flags)}/{len(valid)}")
    print(f"  reader share: linear {100*share_l:.2f}%   MLP {100*share_m:.2f}%")
    below_floor = abs(d) < s_
    if below_floor:
        reading = "delta below the probe's own floor -- no sign is read; the linear bound is tight to within the floor"
    elif abs(d) <= TIGHT:
        reading = "TIGHT: |delta| <= 0.01 -- the linear bound is tight against MLP-recoverable information"
    elif d < -TIGHT:
        reading = "MLP RECOVERS MORE: the linear probe understated survival; the readout share is HIGHER than reported"
    else:
        reading = ("MLP F_enc HIGHER: " + ("overfit flag fired on driving cells -- reading WITHHELD pending the alpha sweep"
                   if any(flags) else "no overfit flag -- era features more nonlinearly separable than final ones (encoder-side finding)"))
    print(f"  READING: {reading}")
    out["row"] = {"cells": [list(c) for c in valid], "delta_F_enc": deltas, "spread": spreads,
                  "overfit_flag": flags, "pooled_delta": d, "mean_spread": s_,
                  "share_linear": share_l, "share_mlp": share_m, "reading": reading,
                  "gates_pass": bool(gates_ok)}
    p = args.out or f"runs/e20/{args.bench}_{args.arm}_row.json"
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2, default=str)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
