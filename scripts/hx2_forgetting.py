"""
E11 — H-X2 recompute: end-to-end forgetting after the best storage-honest cure.

Registered form (docs/E10_prereg.md:204, catch-6 ruling): "cure-corrected final
row, then the standard forgetting formula." The cure-corrected version replaces
R[T-1,j] with the cure's accuracy on task j. TWO formulas are implemented and the
choice is RECORDED in the artifact, because they are different quantities:

    --formula diag   forgetting_cure = mean_j ( R[j,j] - acc_cure_j )
    --formula peak   forgetting_cure = mean_j max(0, max_l R[l,j] - acc_cure_j)

CORRECTION (2026-09-15, found by scripts/c0deg_controls.py). This docstring
previously stated that the diag form IS "the standard formula
(src/training/metrics.py)". It is not: `metrics.py:85-106` implements the
peak-clipped form (Chaudhry et al.'s peak-minus-final, clipped at 0), which is
what every recorded `forgetting` field and the ledger's P2 row use. The ledger's
H-X2 row (0.4812 -> 0.0886) was computed in the diag form under a docstring that
cited metrics.py for it -- a premise carried in a docstring, catch 28's shape.
The two forms differ by the positive backward transfer the peak form folds in
and the clipping; on E10/OFF they move C3 across the H-X2 bar (diag 0.0886 MET,
peak 0.1198 NOT MET). `--formula diag` stays the default so the flag-off path
reproduces runs/e11_hx2.json bit-for-bit; the artifact now records `formula`.

PROTOCOL RULING (E11): the headline is computed on the INTERSECTION CELL SET --
cells valid under the per-cell guard for EVERY cure entering the comparison --
with per-cure full-population numbers as a secondary column, and the exclusion
table printed beside both. Differencing cures scored over different cell
populations is the pooling error in a new costume (catch 26's family); the
intersection is the only population on which "cure A vs cure B" is a like-for-like
sentence. If the intersection shrinks until the CIs swallow the comparison, THAT
is the reported result -- "cure ranking unresolvable at this benchmark's cell
validity" -- which is true rather than tidy.

THE SNAPSHOT IS SCORED ALONGSIDE THE CURES. EV3 established that trivially
deploying the era snapshot dominates C3 at strictly less storage, so the honest
"best storage-honest method" row must include it or it reports the second-best
method as the best. C2 is EXCLUDED as oracle (tier-2, oracle theta_4 prototypes).

CONSISTENCY CHECK, non-tautological: the screen's `acc_orig` must equal the
accuracy matrix's final row R[4,k] for the same run. Two artifacts produced by
different scripts; if they disagree, they do not describe the same run and
nothing below is readable.

Usage:
    python scripts/hx2_forgetting.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rho_percell import RHO_MIN_DENOM, rho_and_se, E10_NTEST, ERA_HEAD_CURES

# Storage-honest methods only. C2 is oracle.
#
# C0deg -- old-task inputs re-laid-out into the CURRENT frame via M_4 . M_k^-1,
# scored by the current model -- was excluded here as "a labelled ceiling
# artifact". That label was written for E8's shared-window HAR, where relayout
# into task 4's frame reproduced task 4's test set exactly. E10 replaced that
# construction with subject-disjoint content (task k's test subject is held out
# from every training group), and the label was carried over without the
# re-verification catch 21 demands. On E10 C0deg is a legitimate storage-honest
# method: one encoder, no snapshot, no refit, and a strict SUBSET of C3's
# resources (the maps only). It enters behind `--include-c0deg` so that the
# flag-off path reproduces runs/e11_hx2.json bit-for-bit -- the artifact the
# ledger's H-X2 row already cites is not altered by adding a method beside it.
# Its controls (no-shift exactness, artifact consistency, wrong-map must-fail)
# are scripts/c0deg_controls.py -> runs/e10/c0deg_controls.json, and the row
# is not citable until that artifact reads all_pass.
HONEST = ["C0", "C1", "C0C1", "C3", "SNAP"]
HONEST_WITH_C0DEG = ["C0", "C0deg", "C1", "C0C1", "C3", "SNAP"]
BAR = 0.10                       # H-X2, unchanged
SEEDS = [42, 1337, 2024]
TASKS = [0, 1, 2, 3]
MATRIX = {"HAR/OFF": "runs/e10_off_seed{s}/mafc_results.json",
          "HAR/ON": "runs/e10_on_seed{s}/mafc_results.json"}
# S72 (runs/MEMO_c0deg.md sec 6): the bridging arm relaunched with era
# checkpoints + fp32 shadow, and its floor pair (replicates a/b of seed 42).
# Each set names its cures file, its matrices, and its "seeds"; `e10` stays the
# default so the flag-off path reproduces runs/e11_hx2.json byte for byte.
ARM_SETS = {
    "e10": {"cures": "runs/e11_e10/cures_e10.json", "seeds": [42, 1337, 2024],
            "matrix": MATRIX},
    "e10ec": {"cures": "runs/e10ec/cures_e10.json", "seeds": [42, 1337, 2024],
              "matrix": {"HAR/OFF": "runs/e10off_ec_seed{s}/mafc_results.json"}},
    "e10ecfloor": {"cures": "runs/e10ecfloor/cures_e10.json", "seeds": ["a", "b"],
                   "matrix": {"HAR/OFF": "runs/e10off_ec_floor4tec_{s}/mafc_results.json"}},
}
# R2 (runs/MEMO_e20.md): the S72 screen re-run with a RECORDED probe draw. These
# are the citable S72 values; `e10ec` (unseeded) stays as measured under an
# unrecorded draw. Same checkpoints, same matrices.
ARM_SETS["e10ec_seeded"] = {"cures": "runs/e10ec_seeded/cures_e10.json", "seeds": [42, 1337, 2024],
                            "matrix": {"HAR/OFF": "runs/e10off_ec_seed{s}/mafc_results.json"}}
ARM_SETS["e10ecfloor_seeded"] = {"cures": "runs/e10ecfloor_seeded/cures_e10.json", "seeds": ["a", "b"],
                                 "matrix": {"HAR/OFF": "runs/e10off_ec_floor4tec_{s}/mafc_results.json"}}
# E18 (docs/E18_prereg.md): the disjoint-content MNIST constructions. The screen
# writes the row key "MNIST/OFF"; each test subset is exactly 2000 images per
# task (10k / 5), which is also the N_TEST cap, so the binomial SE uses 2000.
E18_NTEST = {0: 2000, 1: 2000, 2: 2000, 3: 2000}
for _tag in ("e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp"):
    ARM_SETS[_tag] = {"cures": f"runs/{_tag}/cures.json", "seeds": [42, 1337, 2024],
                      "matrix": {"MNIST/OFF": f"runs/{_tag}_seed{{s}}/mafc_results.json"}, "ntest": E18_NTEST}
    ARM_SETS[_tag + "_floor"] = {"cures": f"runs/{_tag}_floor/cures.json", "seeds": ["a", "b"],
                                 "matrix": {"MNIST/OFF": f"runs/{_tag}_floor4tec_{{s}}/mafc_results.json"}, "ntest": E18_NTEST}


def cell_valid(r, cure, ntest):
    """Does this cell clear the per-cell denominator guard for this cure?"""
    adjusted = cure in ERA_HEAD_CURES
    _, _, D = rho_and_se(r[cure], r["acc_orig"], r["acc_refit"], r["R"],
                         ntest[r["task"]], adjusted)
    return D > RHO_MIN_DENOM, D


def main():
    global HONEST, SEEDS, MATRIX
    ap = argparse.ArgumentParser(description="E11 H-X2 recompute")
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="e10",
                    help="which run set: e10 (the ledger's), e10ec (S72 relaunch), e10ecfloor")
    ap.add_argument("--cures", default=None, help="default: the arm set's cures file")
    ap.add_argument("--out", default=None,
                    help="default runs/e11_hx2.json; runs/e11_hx2_c0deg.json with --include-c0deg")
    ap.add_argument("--include-c0deg", action="store_true",
                    help="score C0deg as a storage-honest method (see HONEST_WITH_C0DEG)")
    ap.add_argument("--formula", choices=["diag", "peak"], default="diag",
                    help="diag: R[j,j]-acc (the H-X2 row's form). peak: metrics.py's "
                         "max(0, max_l R[l,j]-acc), the recorded-forgetting form")
    args = ap.parse_args()
    if args.include_c0deg:
        HONEST = HONEST_WITH_C0DEG
    arm_set = ARM_SETS[args.arms]
    SEEDS, MATRIX = arm_set["seeds"], arm_set["matrix"]
    if args.cures is None:
        args.cures = arm_set["cures"]
    if args.out is None:
        # Any flag-on run writes BESIDE the cited artifact, never over it.
        tag = ("_c0deg" if args.include_c0deg else "") + ("_peak" if args.formula == "peak" else "")
        args.out = (f"runs/e11_hx2{tag}.json" if args.arms == "e10"
                    else f"runs/{args.arms}/hx2{tag}.json")
    cures = json.load(open(args.cures))
    ntest = arm_set.get("ntest", E10_NTEST)

    print("=" * 100)
    print("E11 — H-X2 RECOMPUTE   end-to-end forgetting after the best "
          f"storage-honest cure   (bar <= 0.10)   formula={args.formula}")
    print("=" * 100)

    out = {}
    for arm, rows in cures.items():
        for r in rows:
            r["SNAP"] = r["acc_ceiling"]           # trivial use of C3's ingredient

        # ---- consistency: the screen and the matrix must describe one run ----
        diag, peak, worst_gap = {}, {}, 0.0
        for s in SEEDS:
            m = np.array(json.load(open(MATRIX[arm].format(s=s)))["accuracy_matrix"],
                         dtype=float)
            diag[s] = {k: float(m[k, k]) for k in TASKS}
            peak.setdefault(s, {}).update({k: float(np.nanmax(m[:, k])) for k in TASKS})
            for r in [x for x in rows if x["seed"] == s]:
                worst_gap = max(worst_gap, abs(r["acc_orig"] - float(m[4, r["task"]])))
        # Tolerance is float32 epsilon (~1.2e-07), not a fudge: both artifacts
        # store means of float32 accuracies through JSON, so they agree to
        # representation and no further. Set below that, this gate fires on
        # round-trip noise (measured 2.3e-08) and reports a run mismatch that
        # does not exist -- a gate that cannot pass is as useless as one that
        # cannot fail. Set above ~1e-4 it would stop discriminating runs.
        assert worst_gap < 1e-6, (
            f"{arm}: screen acc_orig disagrees with the accuracy matrix final row "
            f"by {worst_gap:.2e} — the two artifacts do not describe the same run")

        print(f"\n{'='*100}\n{arm}   (screen acc_orig == matrix final row, "
              f"max |diff| {worst_gap:.1e})\n{'='*100}")

        # ---- exclusion table, printed BESIDE the comparison ------------------
        valid = {c: set() for c in HONEST}
        print(f"  per-cell validity (guard D > {RHO_MIN_DENOM}):")
        header = f"    {'cell':<12}" + "".join(f"{c:>10}" for c in HONEST)
        print(header)
        for r in sorted(rows, key=lambda r: (r["seed"], r["task"])):
            key = (r["seed"], r["task"])
            line = f"    s{r['seed']}/t{r['task']:<8}"
            for c in HONEST:
                ok, D = cell_valid(r, c, ntest)
                if ok:
                    valid[c].add(key)
                line += f"{('ok' if ok else 'EXCL'):>10}"
            print(line)
        inter = set.intersection(*(valid[c] for c in HONEST))
        print(f"\n  intersection cell set: {len(inter)}/{len(rows)} cells "
              f"{sorted(f's{a}/t{b}' for a, b in inter)}")
        for c in HONEST:
            missing = sorted(f"s{a}/t{b}" for a, b in (valid[c] - inter))
            if missing:
                print(f"    {c:<6} valid on {len(valid[c])} cells; drops from the "
                      f"intersection: {missing}")

        # ---- forgetting, intersection and full population --------------------
        def forgetting(cure, cells):
            """Per seed then across seeds; the per-cell term is set by --formula."""
            per_seed, ses = [], []
            for s in SEEDS:
                items = [r for r in rows if r["seed"] == s
                         and (r["seed"], r["task"]) in cells]
                if not items:
                    continue
                if args.formula == "diag":
                    vals = [diag[s][r["task"]] - r[cure] for r in items]
                else:
                    vals = [max(0.0, peak[s][r["task"]] - r[cure]) for r in items]
                per_seed.append(float(np.mean(vals)))
                # binomial SE of the cure term, propagated through the mean
                se = math.sqrt(sum(
                    max(r[cure] * (1 - r[cure]), 0.0) / ntest[r["task"]]
                    for r in items)) / len(items)
                ses.append(se)
            if not per_seed:
                return float("nan"), float("nan"), 0
            return (float(np.mean(per_seed)),
                    float(math.sqrt(sum(x * x for x in ses)) / len(ses)),
                    len(per_seed))

        allcells = {(r["seed"], r["task"]) for r in rows}
        print(f"\n  {'method':<8}{'forgetting (intersection)':>28}"
              f"{'forgetting (full pop)':>26}{'n cells':>9}")
        res = {}
        base_i, base_se, _ = forgetting("acc_orig", inter)
        base_f, _, _ = forgetting("acc_orig", allcells)
        print(f"  {'none':<8}{base_i:>20.4f} +-{1.96*base_se:<5.3f}"
              f"{base_f:>22.4f}{len(inter):>13}   <- uncured baseline")
        for c in HONEST:
            fi, se, _ = forgetting(c, inter)
            ff, _, _ = forgetting(c, valid[c])
            res[c] = {"intersection": fi, "se": se, "full_pop": ff,
                      "n_valid": len(valid[c])}
            print(f"  {c:<8}{fi:>20.4f} +-{1.96*se:<5.3f}{ff:>22.4f}"
                  f"{len(valid[c]):>13}")

        # ---- SNAP is an IDENTITY, not a measurement -------------------------
        # acc_ceiling is the era model evaluated on its own task, which IS the
        # accuracy matrix's diagonal R[k,k] (verified: max |diff| 2.9e-08 over
        # all 24 cells). So forgetting_SNAP = mean_j(R[j,j] - R[j,j]) = 0 BY
        # CONSTRUCTION. It cannot fail, so it cannot be evidence -- ranking it
        # as "best" would make H-X2 trivially MET while measuring nothing. It is
        # the "just keep every model" anchor, printed and labelled, never ranked.
        snap_gap = max(abs(res["SNAP"]["intersection"]), abs(res["SNAP"]["full_pop"]))
        if args.formula == "diag":
            assert snap_gap < 1e-6, (
                f"SNAP forgetting {snap_gap:.2e} != 0 — the identity "
                f"acc_ceiling == R[k,k] does not hold, so this reasoning is wrong")
            print(f"\n  NOTE: SNAP forgetting is 0.0000 BY CONSTRUCTION (acc_ceiling "
                  f"== R[k,k], verified to {snap_gap:.1e}). It is the degenerate\n"
                  f"  'store every era model' anchor — printed, never ranked. The "
                  f"H-X2 verdict is read off the cures.")
        else:
            # Under the peak form SNAP reads mean_j(peak_j - R[j,j]) >= 0: the
            # positive backward transfer the clipped-peak definition counts as
            # forgetting of the era snapshot. Not an identity; printed, still
            # never ranked.
            print(f"\n  NOTE: SNAP forgetting under --formula peak is "
                  f"{res['SNAP']['intersection']:.4f}: mean_j(peak_j - R[j,j]), the\n"
                  f"  backward transfer the peak form folds in. Printed, never ranked.")

        # ---- verdict, computed from the values printed above -----------------
        rankable = {c: v for c, v in res.items() if c != "SNAP"}
        best = min(rankable, key=lambda c: rankable[c]["intersection"])
        bf, bse = rankable[best]["intersection"], rankable[best]["se"]
        met = bf <= BAR
        lo, hi = bf - 1.96 * bse, bf + 1.96 * bse
        print(f"\n  best storage-honest CURE on the intersection: {best} "
              f"at forgetting {bf:.4f} [{lo:.4f}, {hi:.4f}]")
        # The bar is on the estimate; whether the interval also clears it is a
        # separate, reportable fact rather than a second verdict.
        print(f"  H-X2 (<= {BAR}): {'MET' if met else 'NOT MET'}"
              f"   (CI upper {hi:.4f} {'also clears' if hi <= BAR else 'EXCEEDS'} the bar)")

        # Is the ranking resolvable, or do the CIs swallow it?
        ranked = sorted(rankable.items(), key=lambda kv: kv[1]["intersection"])
        (c1, v1), (c2, v2) = ranked[0], ranked[1]
        sep = (v2["intersection"] - 1.96 * v2["se"]) > (v1["intersection"] + 1.96 * v1["se"])
        print(f"  ranking {c1} vs runner-up {c2}: "
              f"{'RESOLVED — intervals separate' if sep else 'UNRESOLVABLE at this cell validity — intervals overlap'}")
        out[arm] = {"best": best, "forgetting": bf, "met": met,
                    "resolved": bool(sep), "n_intersection": len(inter),
                    "per_method": res, "uncured": base_i}
        if args.formula != "diag" or args.include_c0deg or args.arms != "e10":
            # Only flag-on artifacts carry the extra keys, so the flag-off
            # output stays byte-identical to runs/e11_hx2.json. The cell list is
            # what lets another method (LwF, from its own matrices) be scored on
            # THIS population rather than a different one.
            out[arm]["formula"] = args.formula
            out[arm]["honest"] = list(HONEST)
            out[arm]["arms"] = args.arms
            out[arm]["intersection_cells"] = sorted([list(c) for c in inter], key=str)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
