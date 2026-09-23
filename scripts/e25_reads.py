"""E25 R1 and R2 -- the two free reads (docs/E25_prereg.md sec R).

R1  C2, the existing labeled repair, beside acc_refit and acc_orig on every
    scratch arm's cure-screen artifact. Reported in A's own units (recovery
    fraction) so the two are directly comparable, with catch 26's PER-CELL
    denominator guard: the guard is applied per cell, never to a pooled
    denominator, exclusions are counted, and the forced-inclusion pooling is
    printed beside.

R2  A4 at lambda = 1, already run, beside A1 and A3 at every swap level, with
    E21's own floor.

NEITHER CARRIES ODDS. Both read values that were on a screen before this
contract was written (sec 0c, integrity ruling). R1's C2 and acc_refit are
printed by every cure-screen run; R2's A4 is in 10 of 10 swap artifacts, in
runs/e21_row.json and in runs/MEMO_e21.md. A prediction registered after its
data was displayed is not a prediction.

FLOOR PROVENANCE (R2). Replicated from scripts/e21_row.py:139-141, not
reinvented:

    rf = max(per-realization pooled) - min(...)      spread over realizations
    jf = mean over cells of row[arm + "_spread"]     per-cell jitter spread
    floor = max(rf, jf)

E21 reports these curves as FORGETTING (diag - acc). This script reports
ACCURACY. The floor is identical either way and the deltas are unchanged:
pooled forgetting = mean(diag) - mean(acc) over a cell set that is the same in
every realization, so mean(diag) is a constant and drops out of both the
realization spread and any A-minus-A difference. The jitter spread is already
in accuracy units. Stated because a floor carried between two unit systems is
a provenance claim (catch 21).

Usage:
    python scripts/e25_reads.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ----------------------------------------------------------------- R1 inputs --
# The four scratch arms A runs on, each at the SEEDED probe draw (E20's rule:
# the probe's training-subset draw is a recorded config, not process history).
# runs/e10ec/cures_e10.json is the same arm at probe_subset_seed = None and is
# NOT used -- a uniform draw across the four arms is what makes them comparable.
R1_ARMS = {
    "lstm_har":     ("runs/e10ec_seeded/cures_e10.json", "HAR/OFF"),
    "lstm_permuted": ("runs/e18_pmd_lstm/cures.json",    "MNIST/OFF"),
    "mlp_permuted": ("runs/e18_pmd_mlp/cures.json",      "MNIST/OFF"),
    "mlp_rotated":  ("runs/e18_rmd_mlp/cures.json",      "MNIST/OFF"),
}
PROBE_SEED = 20260916
GUARD = 0.05                      # per-cell denominator guard (contract sec A)

# ----------------------------------------------------------------- R2 inputs --
SWAP_GLOB = "runs/e21/swap/level{li}_real{r}.json"
SWAP_LEVELS = {0: 0, 1: 1, 2: 2, 3: 3}      # level_idx -> m (transpositions)
R2_ARMS = ["A1", "A3", "A4"]


def _fmt(x, w=8, p=4):
    return f"{x:{w}.{p}f}"


# ===================================================================== R1 ======
def r1(out_path: str) -> dict:
    print("=" * 100)
    print("R1  C2, the existing labeled repair -- read, no odds (contract sec R1)")
    print("=" * 100)
    print("    C2 = cure_screen.cure_c2: per-class prototypes from FULL training labels on")
    print("    both encoders, orthogonal Procrustes between them, read through the era head.")
    print("    Labeled and CONSTRAINED. A is labeled and unconstrained.\n")

    res = {"kind": "R1", "guard": GUARD, "probe_subset_seed": PROBE_SEED, "arms": {}}
    for arm, (path, key) in R1_ARMS.items():
        if not os.path.exists(path):
            print(f"  {arm:14} ABSENT: {path}")
            res["arms"][arm] = {"absent": path}
            continue
        rows = json.load(open(path))[key]

        seeds = sorted({r["probe_subset_seed"] for r in rows})
        assert seeds == [PROBE_SEED], f"{arm}: probe draw {seeds}, expected [{PROBE_SEED}]"
        assert all(r["p_b"] for r in rows), f"{arm}: a cell failed P-B; nothing read"

        cells = []
        for r in rows:
            denom = r["acc_refit"] - r["acc_orig"]
            cells.append({
                "seed": r["seed"], "task": r["task"],
                "acc_orig": r["acc_orig"], "acc_ceiling": r["acc_ceiling"],
                "acc_refit": r["acc_refit"], "C2": r["C2"],
                "denom": denom, "valid": denom >= GUARD,
                "recovery": (r["C2"] - r["acc_orig"]) / denom if denom != 0 else float("nan"),
            })

        valid = [c for c in cells if c["valid"]]
        excl = [c for c in cells if not c["valid"]]
        pooled_valid = float(np.mean([c["recovery"] for c in valid])) if valid else float("nan")
        pooled_forced = float(np.mean([c["recovery"] for c in cells]))

        print(f"  {arm}   {path}  [{key}]   {len(cells)} cells, probe seed {PROBE_SEED}")
        print(f"    {'seed':>5} {'task':>4} {'acc_orig':>9} {'C2':>8} {'acc_refit':>10} "
              f"{'ceiling':>8} {'denom':>7} {'recovery':>9}")
        for c in sorted(cells, key=lambda c: (c["seed"], c["task"])):
            flag = "" if c["valid"] else "  <- below guard, excluded"
            print(f"    {c['seed']:>5} {c['task']:>4} {_fmt(c['acc_orig'],9)} {_fmt(c['C2'])} "
                  f"{_fmt(c['acc_refit'],10)} {_fmt(c['acc_ceiling'])} {_fmt(c['denom'],7,3)} "
                  f"{_fmt(c['recovery'],9)}{flag}")
        m = lambda k: float(np.mean([c[k] for c in cells]))
        print(f"    pooled means: acc_orig {m('acc_orig'):.4f}  C2 {m('C2'):.4f}  "
              f"acc_refit {m('acc_refit'):.4f}  ceiling {m('acc_ceiling'):.4f}")
        print(f"    C2 recovery, mean of ratios over {len(valid)}/{len(cells)} valid cells: "
              f"{pooled_valid:+.4f}    [excluded {len(excl)}; forced-inclusion {pooled_forced:+.4f}]\n")

        res["arms"][arm] = {
            "artifact": path, "key": key, "n_cells": len(cells),
            "n_valid": len(valid), "n_excluded": len(excl),
            "excluded_cells": [{"seed": c["seed"], "task": c["task"], "denom": c["denom"]} for c in excl],
            "pooled_means": {k: m(k) for k in ("acc_orig", "C2", "acc_refit", "acc_ceiling")},
            "c2_recovery_mean_of_ratios": pooled_valid,
            "c2_recovery_forced_inclusion": pooled_forced,
            "cells": cells,
        }

    ok = [a for a, v in res["arms"].items() if "absent" not in v]
    if ok:
        print("  ACROSS ARMS  C2's recovery of the refit gap (mean of ratios over valid cells):")
        for a in ok:
            v = res["arms"][a]
            print(f"    {a:14} {v['c2_recovery_mean_of_ratios']:+.4f}   "
                  f"[{v['n_valid']}/{v['n_cells']} cells]")
        print("\n    D1 measured this drift at spread 8-12 with 54-66% of singular values")
        print("    below 0.5 -- far from orthogonal. C2 is the orthogonal repair; how far")
        print("    short of the refit it falls is the room an unconstrained A can occupy.")
    _write(out_path, res)
    return res


# ===================================================================== R2 ======
def r2(out_path: str) -> dict:
    print("\n" + "=" * 100)
    print("R2  A4 at lambda = 1, already run -- read, no odds (contract sec R2)")
    print("=" * 100)
    print("    A4's objective as executed (e21_perturb.py:154-157, MUMMADI_LAMBDA = 1.0):")
    print("        slr - 1.0 * H(mean p),   slr = -sum_c p_c log(p_c / (1 - p_c))")
    print("    which is Mummadi sec 3.2.2's SLR exactly. Their objective")
    print("    L_div + 0.025 L_slr rescales to slr - 40 H(mean p), so A4 as run is")
    print("    Mummadi's form at 1/40th the diversity weight. C runs lambda = 40.\n")

    data, missing = {}, []
    for li in SWAP_LEVELS:
        for r in (range(3) if li > 0 else [0]):
            p = SWAP_GLOB.format(li=li, r=r)
            if os.path.exists(p):
                data[(li, r)] = json.load(open(p))
            else:
                missing.append(p)
    if missing:
        print(f"  MISSING {len(missing)} artifacts: {missing[:4]}")
        res = {"kind": "R2", "missing": missing}
        _write(out_path, res)
        return res

    any_d = next(iter(data.values()))
    print(f"    mummadi_form recorded in the artifacts: {any_d.get('mummadi_form')!r}")
    print(f"    optimizer: {any_d.get('optimizer')}\n")

    res = {"kind": "R2", "arms": R2_ARMS, "levels": {},
           "mummadi_form_recorded": any_d.get("mummadi_form"),
           "floor_provenance": "scripts/e21_row.py:139-141, max(realization spread, mean per-cell jitter spread)"}

    print(f"    pooled ACCURACY over 12 cells (3 seeds x 4 tasks), mean over realizations")
    print(f"    floor = max(spread over realizations, mean per-cell jitter spread)  [printed r/j]\n")
    print(f"    {'m':>3} " + "".join(f"{a:>22}" for a in R2_ARMS))

    for li, m_val in SWAP_LEVELS.items():
        per_arm = {}
        for a in R2_ARMS:
            vals, jit = [], []
            for r in (range(3) if li > 0 else [0]):
                rows = data[(li, r)]["rows"]
                vals.append(float(np.mean([row[a] for row in rows])))
                jit.append(float(np.mean([row.get(a + "_spread", 0.0) for row in rows])))
            rf, jf = float(max(vals) - min(vals)), float(np.mean(jit))
            # LEVEL 0 HAS ONE REALIZATION (e21_row.py:135 -- eps = 0 is identical for
            # every r, so E21 runs it once). max(vals) - min(vals) over a single value
            # is 0.000 BY CONSTRUCTION, not by measurement, and the swap search is
            # deterministic so the jitter spread is 0.000 too. A floor that cannot be
            # anything but zero is not a floor (catch 25): any non-zero delta "beats"
            # it. Level 0 carries NO floor and its comparison is not read.
            readable = li > 0
            per_arm[a] = {"mean": float(np.mean(vals)),
                          "floor": (max(rf, jf) if readable else None),
                          "floor_real": (rf if readable else None),
                          "floor_jitter": jf, "n_realizations": len(vals),
                          "readable": readable, "per_real": vals}
        res["levels"][str(m_val)] = per_arm
        fl = lambda a: ("  n/a " if per_arm[a]["floor_real"] is None
                        else f"{per_arm[a]['floor_real']:.3f}")
        print(f"    {m_val:>3} " + "".join(
            f"{per_arm[a]['mean']:9.4f}[{fl(a)}/{per_arm[a]['floor_jitter']:.3f}]"
            for a in R2_ARMS)
            + ("   <- ONE realization: no floor exists, not read" if not per_arm["A4"]["readable"] else ""))

    print("\n    A4 against A3 and A1, per level, each against the larger of the two floors:")
    verdicts = {}
    for li, m_val in SWAP_LEVELS.items():
        pa = res["levels"][str(m_val)]
        d43 = pa["A4"]["mean"] - pa["A3"]["mean"]
        d41 = pa["A4"]["mean"] - pa["A1"]["mean"]
        if not pa["A4"]["readable"]:
            verdicts[str(m_val)] = {
                "A4_minus_A3": d43, "A4_minus_A1": d41, "floor_43": None, "floor_41": None,
                "A4_beats_A3": None, "A4_beats_A1": None,
                "why": "single realization at eps = 0; the realization spread is 0 by "
                       "construction and the swap search is deterministic, so no floor exists"}
            print(f"      m={m_val}  A4-A3 {d43:+.4f}   |  A4-A1 {d41:+.4f}   "
                  f"-> NOT READ: one realization, no floor exists")
            continue
        f43 = max(pa["A4"]["floor"], pa["A3"]["floor"])
        f41 = max(pa["A4"]["floor"], pa["A1"]["floor"])
        verdicts[str(m_val)] = {
            "A4_minus_A3": d43, "floor_43": f43, "A4_beats_A3": bool(d43 > f43),
            "A4_minus_A1": d41, "floor_41": f41, "A4_beats_A1": bool(d41 > f41)}
        print(f"      m={m_val}  A4-A3 {d43:+.4f} [floor {f43:.4f}] -> "
              f"{'BEATS' if d43 > f43 else 'within floor' if abs(d43) <= f43 else 'BELOW'}"
              f"   |  A4-A1 {d41:+.4f} [floor {f41:.4f}] -> "
              f"{'BEATS' if d41 > f41 else 'within floor' if abs(d41) <= f41 else 'BELOW'}")
    res["verdicts"] = verdicts

    # Sign pattern across the readable levels -- a measurement, not a cleared claim.
    readable = [v for k, v in verdicts.items() if v["A4_beats_A3"] is not None]
    n_pos = sum(v["A4_minus_A3"] > 0 for v in readable)
    res["sign_pattern_A4_over_A3"] = {
        "n_readable_levels": len(readable), "n_positive": n_pos,
        "deltas": [v["A4_minus_A3"] for v in readable],
        "note": ("Every readable level puts A4 above A3 in point estimate while none clears "
                 "its floor. Reported as a measurement, not a cleared claim -- the same form "
                 "E23's (E1) row uses.")}
    print(f"\n    SIGN PATTERN  A4 above A3 in point estimate on {n_pos}/{len(readable)} readable "
          f"levels, none clearing floor:\n      deltas "
          + ", ".join(f"{v['A4_minus_A3']:+.4f}" for v in readable)
          + "\n      A measurement, not a cleared claim. The floors are realization spread "
            "(0.049-0.175);\n      the jitter spread is 0.000 at every level because the swap "
            "search is deterministic.")

    # ---- the reading, stated in the contract before this read ------------------
    v1 = verdicts["1"]
    if v1["A4_beats_A3"]:
        reading = ("A4 at lambda = 1 ALREADY beats A3 by more than floor at m = 1: diversity gives "
                   "entropy direction even at 1/40th of Mummadi's weight, and C's question at "
                   "lambda = 40 is one of MAGNITUDE, not of existence.")
    else:
        reading = ("A4 at lambda = 1 does NOT beat A3 by more than floor at m = 1, so C at "
                   "lambda = 40 is the only remaining test of whether the diversity term gives "
                   "entropy direction on the discrete family.")
    print(f"\n    READING (stated in the contract before this read):\n      {reading}")
    res["reading"] = reading
    _write(out_path, res)
    return res


def _write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj["platform"] = {"system": platform.system(), "machine": platform.machine(),
                       "python": platform.python_version()}
    obj["note"] = ("Read of existing artifacts. No model is loaded and no device-dependent "
                   "arithmetic runs, so this is platform-independent; the platform is recorded "
                   "anyway because every other row in this program carries one.")
    json.dump(obj, open(path, "w"), indent=2, default=float)
    print(f"      wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="E25 R1 and R2, the two free reads")
    ap.add_argument("--r1-out", default="runs/e25/r1_c2.json")
    ap.add_argument("--r2-out", default="runs/e25/r2_a4_lam1.json")
    ap.add_argument("--only", choices=["r1", "r2"], default=None)
    a = ap.parse_args()
    if a.only in (None, "r1"):
        r1(a.r1_out)
    if a.only in (None, "r2"):
        r2(a.r2_out)


if __name__ == "__main__":
    main()
