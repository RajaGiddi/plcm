"""E21 row -- curves, crossings, controls, floors, predictions. Contract sec 5-7.

Reads runs/e21/{family}/level{l}_real{r}.json (12 cells x arms each) and
prints, in this order, every verdict derived from the printed arrays:

  C-ID     A0/A1/A2/A6 at level 0 reproduce the seeded S72 screen per cell (1e-6)
  C-EXACT  A6 constant across every level and realization (1e-6)
  C-WIT    every row's p_b True
  C-DT     A5-deranged < A5 in 12/12 cells at every level >= 1, every realization
  C-DRIFT  A3/A4/A5 - A1 at level 0, per cell (a measurement; two-sided reading)
  C-FLOOR  per level: spread over realizations of each arm's pooled forgetting, and
           the mean per-cell jitter spread of the optimized arms
  curves   forgetting_arm(l) = mean over cells and realizations of (R[j,j] - acc);
           diag form; intervals over the three model seeds
  crossing per family: the first level where A2 < A1 by more than the floor; and
           A7 vs A1-on-refined-map (matched map quality)
  headline 2: A5 vs A3/A4 by more than the floor, per family
  predictions (sec 7), scored

A4 is printed but NOT READ until docs/E21_mummadi.md exists (its objective is
the intent form). Usage:  python scripts/e21_row.py --out runs/e21_row.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

FAMILIES = {"gain": [0.0, 0.05, 0.10, 0.20, 0.30], "offset": [0.0, 0.05, 0.10, 0.20, 0.30], "swap": [0, 1, 2, 3]}
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A5d", "A6", "A7", "SNAP"]
SEEDS = [42, 1337, 2024]
MATRIX = "runs/e10off_ec_seed{s}/mafc_results.json"
SCREEN = "runs/e10ec_seeded/cures_e10.json"
REALISTIC = {"gain": 0.05, "offset": 0.10, "swap": 0}
TOL = 1e-6
# har_subject test windows per task, read from HARSubjectBenchmark.get_task_loaders (one held-out subject per group:
# subjects 25/29/26/30/28). Every A2 delta below is printed in units of 1/n_test_k, which checks these against the data.
N_TEST = {0: 409, 1: 344, 2: 392, 3: 383, 4: 382}


def load_all():
    data = {}
    for fam, levels in FAMILIES.items():
        for li in range(len(levels)):
            for r in (range(3) if li > 0 else [0]):
                p = f"runs/e21/{fam}/level{li}_real{r}.json"
                if Path(p).exists():
                    data[(fam, li, r)] = json.load(open(p))["rows"]
    return data


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/e21_row.json")
    ap.add_argument("--a2-windows", type=float, default=0.0,
                    help="C-ID tolerance for A2 ONLY (bridging's head refit), in TEST WINDOWS per cell: the per-cell bar is "
                         "N/n_test_k. Contract sec 5 says 1e-6 (N=0). The sweeps measured a cross-platform refit floor of "
                         "1-2 windows in 4/12 cells (2/409 in task 0, 1/392 and 1/383 in tasks 2-3; arm64 screen vs x86 jobs). "
                         "Any N>0 is a RULING, cited in the row; the same floor is carried onto every A2/A7 delta.")
    args = ap.parse_args()
    data = load_all()
    expected = sum(1 + 3 * (len(v) - 1) for v in FAMILIES.values())
    print("=" * 100 + f"\nE21 ROW   artifacts {len(data)}/{expected}\n" + "=" * 100)
    out = {"n_artifacts": len(data), "expected": expected}
    if len(data) < expected:
        print("  incomplete -- the row is not read until every artifact exists"); missing = True
    else:
        missing = False
    diag = {s: {k: float(np.array(json.load(open(MATRIX.format(s=s)))["accuracy_matrix"])[k, k]) for k in range(4)} for s in SEEDS}
    screen = {(r["seed"], r["task"]): r for r in json.load(open(SCREEN))["HAR/OFF"]}

    # ---- controls ------------------------------------------------------------
    print("\nCONTROLS")
    cid = {}
    ref = {"A0": "acc_orig", "A1": "C0deg", "A2": "C3", "A6": "C0deg"}
    a2_bar = lambda task: (args.a2_windows / N_TEST[task] + 1e-9) if args.a2_windows > 0 else TOL
    for fam in FAMILIES:
        rows = data.get((fam, 0, 0), [])
        if rows:
            per_arm = {a: max(abs(r[a] - screen[(r["seed"], r["task"])][col]) for r in rows) for a, col in ref.items()}
            n_off = {a: sum(abs(r[a] - screen[(r["seed"], r["task"])][col]) >= TOL for r in rows) for a, col in ref.items()}
            a2_windows = {f"s{r['seed']}t{r['task']}": (r["A2"] - screen[(r["seed"], r["task"])]["C3"]) * N_TEST[r["task"]] for r in rows}
            a2_off = {k: round(v, 3) for k, v in a2_windows.items() if abs(v) >= 0.5}
            ok = all(per_arm[a] < TOL for a in ("A0", "A1", "A6")) and all(abs(r["A2"] - screen[(r["seed"], r["task"])]["C3"]) < a2_bar(r["task"]) for r in rows)
            cid[fam] = {"max_abs_delta": per_arm, "cells_off": n_off, "a2_delta_windows": a2_off, "a2_windows_bar": args.a2_windows}
            print(f"  C-ID    {fam:<7} level 0 vs seeded S72 screen: " + "  ".join(f"{a} {v:.1e} ({n_off[a]}/12 off)" for a, v in per_arm.items())
                  + f"  A2 off by {a2_off or 'none'} windows -> {'PASS' if ok else 'FAIL -- nothing read'}   [A2 bar {args.a2_windows:g} window(s)/cell, others {TOL:.0e}]")
    a6 = {}
    for (fam, li, r), rows in data.items():
        for row in rows:
            a6.setdefault((fam, row["seed"], row["task"]), []).append(row["A6"])
    cex = max((max(v) - min(v)) for v in a6.values()) if a6 else float("nan")
    print(f"  C-EXACT A6 constant across levels/realizations: max spread {cex:.1e} -> {'PASS' if cex < TOL else 'FAIL'}")
    pb = all(row["p_b"] for rows in data.values() for row in rows)
    print(f"  C-WIT   path identity on every cell -> {'PASS' if pb else 'FAIL'}")
    def phimax(fit):
        p = np.array(fit["phi"], dtype=float)
        return float(np.abs(p).max()) if p.ndim == 1 else float(np.abs(p - np.eye(len(p))).max())
    cdt_fail = [(fam, li, r, row["seed"], row["task"]) for (fam, li, r), rows in data.items() if li > 0 for row in rows if not (row["A5d"] < row["A5"])]
    n_cdt = sum(len(rows) for (fam, li, r), rows in data.items() if li > 0)
    print(f"  C-DT    deranged teacher < A5: {n_cdt - len(cdt_fail)}/{n_cdt} cells -> {'PASS' if not cdt_fail else 'FAIL (bar 12/12 at every level >= 1; recorded with cause, not re-barred)'}")
    for fam, li, r, s, t in cdt_fail:
        row = [x for x in data[(fam, li, r)] if x["seed"] == s and x["task"] == t][0]
        print(f"          {fam} L{li} r{r} s{s} t{t}: A1 {row['A1']:.4f}  A5 {row['A5']:.4f} (A5-A1 {row['A5']-row['A1']:+.4f}, |phi|max {max(phimax(f) for f in row['A5_fits']):.2f})"
              f"  A5d {row['A5d']:.4f} (|phi|max {max(phimax(f) for f in row['A5d_fits']):.3f}, n_iter {[f['n_iter'] for f in row['A5d_fits']]})")
    n_unconv = {a: sum(not f["converged"] for rows in data.values() for row in rows for f in row[a + "_fits"]) for a in ("A3", "A4", "A5", "A5d")}
    print(f"  C-CONV  unconverged fits per arm (excluded and counted per sec 4): {n_unconv}")
    print("  C-DRIFT at level 0 (arm - A1 in accuracy, pooled over cells; a measurement, read against the level-0 jitter floor):")
    drift = {}
    for fam in FAMILIES:
        rows = data.get((fam, 0, 0), [])
        if rows:
            drift[fam] = {a: {"delta": float(np.mean([row[a] - row["A1"] for row in rows])),
                              "jitter_floor": float(np.mean([row[a + "_spread"] for row in rows])),
                              "phi_norm": float(np.mean([np.mean([np.linalg.norm(np.array(f["phi"], dtype=float) - (0 if np.array(f["phi"]).ndim == 1 else np.eye(9))) for f in row[a + "_fits"]]) for row in rows]))}
                          for a in ("A3", "A4", "A5")}
            print(f"          {fam:<7} " + "  ".join(f"{a} {v['delta']:+.4f} [jitter floor {v['jitter_floor']:.4f}, |phi*| {v['phi_norm']:.2f}]" for a, v in drift[fam].items()))
    out["controls"] = {"C_ID": cid, "C_EXACT": cex, "C_WIT": pb, "C_DT_fail": cdt_fail, "C_CONV": n_unconv, "C_DRIFT": drift}
    if missing:
        json.dump(out, open(args.out, "w"), indent=2); return

    # ---- curves -----------------------------------------------------------------
    print("\nCURVES  forgetting (diag form) pooled over 12 cells; mean over realizations;"
          "\n        [floor = max(spread over realizations, mean per-cell jitter spread, ruled refit window floor on A2/A7) -- sec 5 C-FLOOR; printed r/j]")
    curves = {}
    for fam, levels in FAMILIES.items():
        print(f"\n  {fam}  levels {levels}   realistic <= {REALISTIC[fam]}")
        print(f"  {'level':<7}" + "".join(f"{a:>20}" for a in ARMS))
        curves[fam] = {}
        for li, lv in enumerate(levels):
            per_arm = {}
            for a in ARMS:
                vals, jit = [], []
                for r in (range(3) if li > 0 else [0]):
                    rows = data[(fam, li, r)]
                    vals.append(float(np.mean([diag[row["seed"]][row["task"]] - row[a] for row in rows])))
                    jit.append(float(np.mean([row.get(a + "_spread", 0.0) for row in rows])))
                rf, jf = float(max(vals) - min(vals)), float(np.mean(jit))
                wf = float(np.mean([args.a2_windows / N_TEST[k] for k in range(4)])) if a in ("A2", "A7") else 0.0   # refit floor, ruled
                per_arm[a] = {"mean": float(np.mean(vals)), "floor": max(rf, jf, wf), "floor_real": rf, "floor_jitter": jf, "floor_window": wf, "per_real": vals}
            curves[fam][lv] = per_arm
            print(f"  {lv:<7}" + "".join(f"{per_arm[a]['mean']:8.4f}[{per_arm[a]['floor_real']:.3f}/{per_arm[a]['floor_jitter']:.3f}]" for a in ARMS))
    out["curves"] = {fam: {str(lv): v for lv, v in c.items()} for fam, c in curves.items()}

    # ---- crossings + headlines ---------------------------------------------------
    print("\nHEADLINE 1  A1 vs A2 per family (first level where A2 < A1 by more than both floors); and A7 vs A1-on-A5-map")
    verdicts = {}
    for fam, levels in FAMILIES.items():
        cross = None
        for lv in levels:
            c = curves[fam][lv]; fl = max(c["A1"]["floor"], c["A2"]["floor"])
            if c["A2"]["mean"] < c["A1"]["mean"] - fl:
                cross = lv; break
        both_deg = all(abs(curves[fam][lv]["A1"]["mean"] - curves[fam][lv]["A2"]["mean"]) <= max(curves[fam][lv]["A1"]["floor"], curves[fam][lv]["A2"]["floor"]) for lv in levels[1:])
        shape = ("CROSSING at " + str(cross) + (" (INSIDE the realistic range)" if cross <= REALISTIC[fam] else " (outside the realistic range)")) if cross is not None \
                else ("both within floor of each other at every level" if both_deg else "A1 below A2 at every level -- no regime")
        a7 = {lv: curves[fam][lv]["A7"]["mean"] - curves[fam][lv]["A5"]["mean"] for lv in levels}
        print(f"  {fam:<7} {shape};  A7 - A5 (bridging on refined map vs re-layout on refined map): " + ", ".join(f"{lv}: {v:+.4f}" for lv, v in a7.items()))
        verdicts[fam] = {"shape": shape, "crossing_level": cross, "a7_minus_a5": {str(k): v for k, v in a7.items()}}
    print("\nHEADLINE 2  A5 vs A3/A4 (positive = A5 better, in forgetting units; A4 NOT READ until the transcription lands)")
    for fam, levels in FAMILIES.items():
        wins = []
        for lv in levels[1:]:
            c = curves[fam][lv]; fl = max(c["A5"]["floor"], c["A3"]["floor"], c["A4"]["floor"])
            wins.append((lv, c["A3"]["mean"] - c["A5"]["mean"], c["A4"]["mean"] - c["A5"]["mean"], fl))
        beats = any(w[1] > w[3] and w[2] > w[3] for w in wins)
        print(f"  {fam:<7} " + "  ".join(f"{lv}: vsA3 {d3:+.4f} vsA4 {d4:+.4f} [fl {fl:.3f}]" for lv, d3, d4, fl in wins) + f"  -> A5 beats both by > floor: {'YES' if beats else 'no'}")
        # paired per-realization differences (same perturbation draw for both arms) -- a measurement beside the registered read
        paired = {lv: [(curves[fam][lv]["A3"]["per_real"][i] - curves[fam][lv]["A5"]["per_real"][i], curves[fam][lv]["A4"]["per_real"][i] - curves[fam][lv]["A5"]["per_real"][i]) for i in range(3)] for lv in levels[1:]}
        print(f"          paired per realization: " + "  ".join(f"{lv}: vsA3 [{', '.join(f'{p[0]:+.3f}' for p in ps)}] vsA4 [{', '.join(f'{p[1]:+.3f}' for p in ps)}]" for lv, ps in paired.items()))
        verdicts[fam]["a5_beats_both"] = beats
        verdicts[fam]["a5_paired"] = {str(k): v for k, v in paired.items()}
    out["verdicts"] = verdicts

    # ---- predictions --------------------------------------------------------------
    print("\nPREDICTIONS (sec 7)")
    g, o, sw = curves["gain"], curves["offset"], curves["swap"]
    flat = lambda c, lvs: all(abs(c[lv]["A1"]["mean"] - c[0.0]["A1"]["mean"]) <= max(c[lv]["A1"]["floor"], 1e-3) for lv in lvs)
    preds = {
        "A1 within floor of eps=0 across the realistic range (gain <=0.05, offset <=0.10)": flat(g, [0.05]) and flat(o, [0.05, 0.10]),
        "A1 loses >= 20pp at one transposition": (sw[1]["A1"]["mean"] - sw[0]["A1"]["mean"]) >= 0.20,
        "a crossing exists in at least one family": any(v["crossing_level"] is not None for v in verdicts.values()),
        "if it exists, it is in swap": (verdicts["swap"]["crossing_level"] is not None) if any(v["crossing_level"] is not None for v in verdicts.values()) else None,
        "A5 beats both A3 and A4 by > floor in at least one family": any(v["a5_beats_both"] for v in verdicts.values()),
        "C-DRIFT: A3 or A4 below A1 by > floor at eps=0 in at least one family": any(drift[f][a]["delta"] < -max(curves[f][FAMILIES[f][0]][a]["floor"], 1e-3) for f in FAMILIES for a in ("A3", "A4")),
        "A5's margin over A3 concentrates where the perturbation induces confident errors (mechanism)": None,  # no margin exists to localize unless the line above fires
    }
    for k, v in preds.items():
        print(f"  {k:<80} {'fired' if v else ('n/a' if v is None else 'MISS')}")
    out["predictions"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
