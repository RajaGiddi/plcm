"""E26 FS row -- the frame sweep (docs/E26_prereg.md sec FS).

Reads runs/e26/fs/{arm}/deployed_seed{s}.json and scores the six registered
predictions against the four hypotheses:

    H-own   peak at j = k      H-last  peak at j = T
    H-flat  no peak within floor          H-base  peak at j = 0

ONE STRUCTURAL FACT THE ROW MUST CARRY. At k = 0 the own frame IS the base
frame (perms[0] is None), so H-own and H-base are the same event on that cell.
H-base is therefore counted on k >= 1 only, where the two frames differ, and
that is stated beside the count rather than folded into it.

"A majority of (k, seed) cells" is more than half of the cells the arm has.
Where the deployed head reads at chance in every frame, the argmax is noise and
H-flat is the reading; the refit P_k(j) is the informative quantity there, and
prediction 6 is about it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ARMS = {"e23_rn": "ResNet-50", "e23_vit": "ViT-B/16", "e23b_t20": "A3 scratch LSTM"}


def load(arm):
    return [json.load(open(f)) for f in sorted(glob.glob(f"runs/e26/fs/{arm}/deployed_seed*.json"))]


def main():
    ap = argparse.ArgumentParser(description="E26 FS row")
    ap.add_argument("--out", default="runs/e26_fs_row.json")
    args = ap.parse_args()
    print("=" * 108 + "\nE26 FS ROW   where does each old task's content peak across the twenty frames?\n" + "=" * 108)
    out = {"arms": {}}
    for arm, label in ARMS.items():
        ds = load(arm)
        if not ds:
            print(f"\n  {label:<16} no artifacts"); continue
        rows = [r for d in ds for r in d["rows"]]
        n = len(rows)
        k1 = [r for r in rows if r["task"] >= 1]                  # cells where own != base
        cid = all(d["controls"]["c_id_pass"] for d in ds)
        relay = all(d["controls"]["relay_exact"] for d in ds)
        mf = sum(d["controls"]["must_fail_unseen_below_seen_max"]["cells"] for d in ds)
        mf_of = sum(d["controls"]["must_fail_unseen_below_seen_max"]["of"] for d in ds)
        # P flatness per cell: range of the refit across its frames vs binomial resolution at the mean
        p_flat = []
        for r in rows:
            if r["P"]:
                v = np.array(list(r["P"].values())); pm = float(v.mean())
                p_flat.append(float(v.max() - v.min()) <= 1.96 * float(np.sqrt(pm * (1 - pm) / r["n_test"])))
        # deployed-at-chance cells: peak within resolution of chance (1/5)
        chance = 0.2
        at_chance = sum(r["peak"] - chance <= r["res95_at_peak"] for r in rows)
        c = {"cells": n, "seeds": [d["seed"] for d in ds],
             "peak_at_own": sum(r["peak_at_own"] for r in rows),
             "peak_at_last": sum(r["peak_at_last"] for r in rows),
             "peak_at_base_k_ge_1": sum(r["peak_at_base"] for r in k1), "cells_k_ge_1": len(k1),
             "own_within_2": sum(r["own_within_2"] for r in rows),
             "flat": sum(r["flat"] for r in rows),
             "deployed_peak_at_chance": at_chance,
             "mean_frames_within_res": float(np.mean([len(r["frames_within_res_of_peak"]) for r in rows])),
             "P_flat_cells": sum(p_flat), "P_cells": len(p_flat),
             "mean_D_own": float(np.mean([r["D_own"] for r in rows])),
             "mean_D_last": float(np.mean([r["D_last"] for r in rows])),
             "mean_D_base": float(np.mean([r["D_base"] for r in rows])),
             "mean_peak": float(np.mean([r["peak"] for r in rows])),
             "controls": {"c_id_pass": cid, "relay_exact": relay, "must_fail": f"{mf}/{mf_of}",
                          "c_id_jk_max": max(d["controls"]["c_id_jk_max_abs"] for d in ds),
                          "c_id_jT_max": max(d["controls"]["c_id_jT_max_abs"] for d in ds)}}
        c["majority_own"] = c["peak_at_own"] > n / 2
        c["majority_last"] = c["peak_at_last"] > n / 2
        c["majority_base"] = c["peak_at_base_k_ge_1"] > len(k1) / 2 if k1 else False
        c["majority_own_within_2"] = c["own_within_2"] > n / 2
        c["majority_flat"] = c["flat"] > n / 2
        c["P_flat_majority"] = c["P_flat_cells"] > c["P_cells"] / 2 if c["P_cells"] else None
        out["arms"][arm] = c
        print(f"\n  {label:<16} seeds {c['seeds']} cells {n} | C-ID {'PASS' if cid else 'FAIL'} "
              f"(j=k {c['controls']['c_id_jk_max']:.4f}, j=T {c['controls']['c_id_jT_max']:.4f}) | relay exact {relay} | must-fail {mf}/{mf_of}")
        print(f"    peaks: own {c['peak_at_own']}/{n}  last {c['peak_at_last']}/{n}  base {c['peak_at_base_k_ge_1']}/{len(k1)} (k>=1)  "
              f"own±2 {c['own_within_2']}/{n}  flat {c['flat']}/{n}  | deployed peak at chance {at_chance}/{n}  | frames within res of peak, mean {c['mean_frames_within_res']:.1f}/20")
        print(f"    mean D: own {c['mean_D_own']:.3f}  last {c['mean_D_last']:.3f}  base {c['mean_D_base']:.3f}  peak {c['mean_peak']:.3f}  | P flat {c['P_flat_cells']}/{c['P_cells']}")

    # ---- predictions ---------------------------------------------------------------
    print("\n  PREDICTIONS (sec FS)")
    A = out["arms"]; preds = []

    def add(stmt, odds, hit, note=""):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<70} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")

    vit, rn, a3 = A.get("e23_vit"), A.get("e23_rn"), A.get("e23b_t20")
    add("pretrained, argmax D_k(j) = k on a majority of cells, ViT", 55, None if not vit else vit["majority_own"],
        "" if not vit else f"{vit['peak_at_own']}/{vit['cells']}")
    add("same, ResNet", 50, None if not rn else rn["majority_own"], "" if not rn else f"{rn['peak_at_own']}/{rn['cells']}")
    add("pretrained peak within +-2 frames of k, either backbone", 60,
        None if not (vit or rn) else any(a["majority_own_within_2"] for a in (vit, rn) if a))
    add("H-base: pretrained peaks at j=0 on a majority of cells, either backbone", 25,
        None if not (vit or rn) else any(a["majority_base"] for a in (vit, rn) if a), "counted on k>=1, where own != base")
    add("scratch A3 peaks at j=T on a majority of cells", 90, None if not a3 else a3["majority_last"],
        "" if not a3 else f"{a3['peak_at_last']}/{a3['cells']}")
    add("P_k(j) flat within floor across the six frames on pretrained", 50,
        None if not (vit or rn) else all(a["P_flat_majority"] for a in (vit, rn) if a and a["P_flat_majority"] is not None))
    scored = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n    scored {len(scored)}: fired {sum(p['outcome']=='fired' for p in scored)}, "
          f"miss {sum(p['outcome']=='MISS' for p in scored)}, did not fire {sum(p['outcome']=='did not fire' for p in scored)}")

    # ---- the reading, from the contract's table -------------------------------------
    if vit and rn:
        own = [a["majority_own"] for a in (vit, rn)]
        if all(own):
            reading = "H-own on both backbones: \"keeps handling for each task's own format\" -- measured."
        elif any(own):
            reading = "H-own on one backbone: stated per backbone."
        elif all(a["majority_last"] for a in (vit, rn)):
            reading = "H-last on the pretrained arms: E23's reading needs a different explanation; sec 4.3 rewritten."
        else:
            reading = ("H-flat on the deployed head: \"does not converge to the latest format\" only; the stronger "
                       "clause dropped. The refit is the quantity that reads here.")
        print(f"\n  READING (contract table, sec FS): {reading}")
        out["reading"] = reading
    out["predictions"] = preds
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
