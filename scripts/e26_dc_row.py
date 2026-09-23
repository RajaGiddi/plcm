"""E26 DC row -- drift-compensation ports (docs/E26_prereg.md sec DC).

Reads runs/e26/dc/{arm}/seed{s}.json, joins A_inf from the E23 decompositions
and A_10 from E25 A where read, scores the seven registered predictions.

Floors: 5pp bars are priced against refit_probe's floors (pooled 0.0003-0.0007,
per-cell up to 0.0140 on ResNet), both far below 5pp. Within-arm intervals are
t95 over run seeds; LDC also carries its projector-seed spread per cell.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

# A_10 JOIN REFUSED (2026-09-21). The contract's A_10 is "E25 A, where read". E25 A's
# pretrained arms are e12_base / e14_base, whose artifacts record benchmark
# `cifar100` (B1, unpermuted); DC ran on e23_vit / e23_rn, benchmark
# `cifar100_permuted` (B6). Different checkpoints, different benchmark. Joining
# them would compare a repair on one arm against a compensation on another --
# catch 30's family, arm identity read from the artifact, not the contract. The
# curve path is kept in the tuple as documentation and the join is disabled; the
# prediction reads "not comparable" until an A_10 exists on B6.
PRE = {"e23_rn": ("resnet", "runs/e23/e23_rn/decomp_seed{s}.json", None),
       "e23_vit": ("vit", "runs/e23/e23_vit/decomp_seed{s}.json", None)}
A10_REFUSED = ("E25 A pretrained ran on cifar100 (B1: e12_base/e14_base); DC ran on "
               "cifar100_permuted (B6: e23_vit/e23_rn). Not the same arm; join refused.")
SCRATCH = ["s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp", "e23b_t20"]
PP5 = 0.05


def t95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def load_arm(arm):
    fs = [f for f in sorted(glob.glob(f"runs/e26/dc/{arm}/seed*.json")) if "unnorm" not in f]
    return [json.load(open(f)) for f in fs]


def pooled(ds, key):
    per_seed = [float(np.mean([r[key] for r in d["rows"]])) for d in ds]
    return float(np.mean(per_seed)), t95(per_seed), per_seed


def main():
    ap = argparse.ArgumentParser(description="E26 DC row")
    ap.add_argument("--out", default="runs/e26_dc_row.json")
    args = ap.parse_args()
    print("=" * 112 + "\nE26 DC ROW   SDC and LDC ports, adapted to CE-trained features and task-IL NCM\n" + "=" * 112)
    out, preds = {"arms": {}}, []

    for arm in list(PRE) + SCRATCH:
        ds = load_arm(arm)
        if not ds:
            print(f"  {arm:<13} no artifacts"); continue
        keys = ["A0", "N_naive", "N_SDC_0.2", "N_SDC_0.3", "N_SDC_1.0", "N_LDC_mean", "N_LDC_ls", "N_oracle"]
        row = {k: pooled(ds, k) for k in keys}
        rows = [r for d in ds for r in d["rows"]]
        # A_inf and A_10 joins (pretrained only, where the artifacts exist)
        ainf = a10 = None
        if arm in PRE:
            _, ref_t, curve_t = PRE[arm]
            vals, v10 = [], []
            for d in ds:
                rp = ref_t.format(s=d["seed"]); cp = curve_t.format(s=d["seed"]) if curve_t else None
                if os.path.exists(rp):
                    vals.append(float(np.mean([r["acc_refit"] for r in json.load(open(rp))["rows"]])))
                if curve_t and os.path.exists(cp):
                    cells = json.load(open(cp))["cells"]
                    v10.append(float(np.mean([c["curve"]["10"]["mean"] for c in cells if "10" in c["curve"]])))
            if vals: ainf = (float(np.mean(vals)), t95(vals), vals)
            if v10: a10 = (float(np.mean(v10)), t95(v10), v10)
        # per-cell derived quantities
        gap_rec = []
        for r in rows:
            den = r["N_oracle"] - r["N_naive"]
            if den >= 0.05:
                gap_rec.append((r["N_LDC_mean"] - r["N_naive"]) / den)
        ldc_vs_boundaries = np.corrcoef([r["boundaries_crossed"] for r in rows], [r["N_LDC_mean"] for r in rows])[0, 1] if len(rows) > 2 else float("nan")
        ctrl = [d["controls"] for d in ds]
        arm_out = {
            "seeds": [d["seed"] for d in ds], "n_cells": len(rows),
            **{k: {"mean": v[0], "t95": v[1], "per_seed": v[2]} for k, v in row.items()},
            "A_inf": None if ainf is None else {"mean": ainf[0], "t95": ainf[1]},
            "A_10": None if a10 is None else {"mean": a10[0], "t95": a10[1]},
            "ldc_gap_recovery_mean_of_ratios": float(np.mean(gap_rec)) if gap_rec else float("nan"),
            "ldc_gap_recovery_cells": len(gap_rec),
            "ldc_beats_naive": row["N_LDC_mean"][0] - row["N_naive"][0] > max(row["N_LDC_mean"][1], row["N_naive"][1]),
            "ldc_beats_A0": row["N_LDC_mean"][0] - row["A0"][0] > max(row["N_LDC_mean"][1], row["A0"][1]),
            "sdc03_beats_naive": row["N_SDC_0.3"][0] - row["N_naive"][0] > max(row["N_SDC_0.3"][1], row["N_naive"][1]),
            "ldc_within_5pp_of_Ainf": None if ainf is None else abs(row["N_LDC_mean"][0] - ainf[0]) <= PP5,
            "ldc_ls_minus_adam": row["N_LDC_ls"][0] - row["N_LDC_mean"][0],
            "corr_ldc_vs_boundaries": float(ldc_vs_boundaries),
            "controls": {
                "ldc_identity_from_random_relfro": [c["ldc_identity"]["from_random_init_relfro"] for c in ctrl],
                "ldc_identity_tol": ctrl[0]["ldc_identity"]["tol"],
                "sdc_must_fail": [c["sdc_must_fail"]["cells_neg_below_naive"] for c in ctrl],
                "ldc_must_fail": [c["ldc_must_fail"]["cells_shuf_below_naive"] for c in ctrl],
                "of": [c["sdc_must_fail"]["of"] for c in ctrl],
                "oracle_identity_pass": all(c["oracle_identity"]["pass"] for c in ctrl),
                "c_plumb_pass": all(c["c_plumb"]["pass"] for c in ctrl)}}
        out["arms"][arm] = arm_out
        fmt = lambda k: f"{row[k][0]:.3f}±{row[k][1]:.3f}"
        print(f"\n  {arm:<13} seeds {arm_out['seeds']} cells {len(rows)}")
        print(f"    A0 {fmt('A0')} | naive {fmt('N_naive')} | SDC 0.2/0.3/1.0 {row['N_SDC_0.2'][0]:.3f}/{row['N_SDC_0.3'][0]:.3f}/{row['N_SDC_1.0'][0]:.3f} "
              f"| LDC {fmt('N_LDC_mean')} (LS {row['N_LDC_ls'][0]:.3f}) | oracle {fmt('N_oracle')}"
              + (f" | A_inf {ainf[0]:.3f}" if ainf else "") + (f" | A_10 {a10[0]:.3f}" if a10 else ""))
        print(f"    LDC beats naive {arm_out['ldc_beats_naive']} | beats A0 {arm_out['ldc_beats_A0']} | gap recovery "
              f"{arm_out['ldc_gap_recovery_mean_of_ratios']:.3f} over {len(gap_rec)} cells | r(LDC, boundaries) {ldc_vs_boundaries:+.3f}"
              + (f" | within 5pp of A_inf {arm_out['ldc_within_5pp_of_Ainf']}" if ainf else ""))
        print(f"    controls: LDC identity from random {np.mean(arm_out['controls']['ldc_identity_from_random_relfro']):.2f} vs tol "
              f"{arm_out['controls']['ldc_identity_tol']} | SDC must-fail {arm_out['controls']['sdc_must_fail']}/{arm_out['controls']['of']} "
              f"| LDC must-fail {arm_out['controls']['ldc_must_fail']}/{arm_out['controls']['of']} | oracle id {arm_out['controls']['oracle_identity_pass']} | plumb {arm_out['controls']['c_plumb_pass']}")

    # ---- predictions ---------------------------------------------------------------
    print("\n  PREDICTIONS (sec DC)")
    A = out["arms"]
    pre = [A[a] for a in PRE if a in A]

    def add(stmt, odds, hit, note=""):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<70} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")

    add("N-LDC beats N-naive, pretrained arms", 70, None if not pre else all(a["ldc_beats_naive"] for a in pre))
    add("N-LDC beats A0, pretrained", 55, None if not pre else all(a["ldc_beats_A0"] for a in pre))
    rn = A.get("e23_rn")
    add("N-LDC within 5pp of A_inf, ResNet", 20, None if not rn or rn["ldc_within_5pp_of_Ainf"] is None else rn["ldc_within_5pp_of_Ainf"])
    add("N-LDC recovers >= 50% of the naive->oracle gap, ResNet", 45,
        None if not rn or np.isnan(rn["ldc_gap_recovery_mean_of_ratios"]) else rn["ldc_gap_recovery_mean_of_ratios"] >= 0.5,
        "" if not rn else f"{rn['ldc_gap_recovery_mean_of_ratios']:.3f}")
    add("N-SDC_0.3 beats N-naive by more than floor, any arm", 35, None if not A else any(a["sdc03_beats_naive"] for a in A.values()),
        "arms: " + ", ".join(k for k, a in A.items() if a["sdc03_beats_naive"]) if A else "")
    both = [a for a in pre if a["A_10"] is not None]
    add("A_10 beats N-LDC where both are read", 65, None if not both else all(a["A_10"]["mean"] > a["N_LDC_mean"]["mean"] for a in both),
        "NOT COMPARABLE: " + A10_REFUSED if not both else "")
    out["a10_join"] = A10_REFUSED
    # The prediction names A3 AND the pretrained arms; it is not scored until all three
    # are in. Scoring it on whichever subset has landed is the n=1 habit in a new coat.
    long_arms = [A[a] for a in ("e23b_t20", "e23_rn", "e23_vit") if a in A]
    add("N-LDC falls with boundaries crossed (r < -0.3), A3 and pretrained", 75,
        None if len(long_arms) < 3 else all(a["corr_ldc_vs_boundaries"] < -0.3 for a in long_arms),
        "r = " + ", ".join(f"{a['corr_ldc_vs_boundaries']:+.2f}" for a in long_arms) if long_arms else "")
    scored = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n    scored {len(scored)}: fired {sum(p['outcome']=='fired' for p in scored)}, "
          f"miss {sum(p['outcome']=='MISS' for p in scored)}, did not fire {sum(p['outcome']=='did not fire' for p in scored)}")

    # ---- the reading that could change the paper ------------------------------------
    if pre:
        r = max(pre, key=lambda a: a["N_LDC_mean"]["mean"])
        if any(a["ldc_within_5pp_of_Ainf"] for a in pre if a["ldc_within_5pp_of_Ainf"] is not None):
            reading = "N-LDC within 5pp of A_inf on a pretrained arm: THE PAPER'S CLAIM IS NARROWED -- class means stored at task end are enough."
        elif all(a["ldc_beats_A0"] and a["ldc_beats_naive"] for a in pre):
            reading = "Drift compensation helps on the pretrained arms but sits well short of the refit: the gap is the price of not having labels at repair time."
        elif all(not a["ldc_beats_naive"] for a in pre):
            reading = "Linear drift compensation fails on these trunks, consistent with D1's non-linear drift."
        else:
            reading = "Mixed across backbones; stated per backbone."
        print(f"\n  READING (pre-committed table, sec DC): {reading}")
        out["reading"] = reading
    out["predictions"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
