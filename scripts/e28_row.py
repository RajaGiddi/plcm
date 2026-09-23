"""E28 row -- does training for equivariance make readout repair sufficient?

Reads runs/e28/measure_{arm}_seed{s}.json for the four E28b arms and scores
sec 7's predictions on the PRIMARY SET (tasks 3 and 4, the tasks whose frame
carries a permutation, fixed by construction).

WHAT DISTINGUISHES THE TWO OUTCOMES, stated before the numbers are read:

  INVARIANT   swapped ~ clean with NO repair; A ~ I (small ||A - I||);
              swap-ID probe at chance -- channel identity is gone from the
              feature, which is why the swap costs nothing.
  EQUIVARIANT swapped < clean; refit recovers nearly all of it (small
              Delta_refit); A != I; swap-ID probe ABOVE chance -- the identity
              is still in the tensor, moved rather than discarded.

Both give a small Delta_refit, so Delta_refit alone cannot tell them apart.
That is why ||A - I|| and the swap-ID probe are read beside it.

COMPETENCE AGGREGATION. Sec 5 says "within 5pp of B0's, per task" without
naming how seeds combine. The contracted verdict here is PER TASK PER SEED, as
executed. The per-task MEAN over seeds is printed beside it, labelled, because
the ambiguity was found after the data -- not as a verdict. Every future
contract states the aggregation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ARMS = ["b0_ep30", "b1_ep30", "b2_ep30", "b2_p025_ep30"]
LABEL = {"b0_ep30": "B0 (no aug)", "b1_ep30": "B1 (aug only)",
         "b2_ep30": "B2 p=0.5", "b2_p025_ep30": "B2 p=0.25"}
RUN = {"b0_ep30": "e28b_b0_ep30_seed{s}", "b1_ep30": "e28b_b1_ep30_seed{s}",
       "b2_ep30": "e28b_b2_ep30_seed{s}", "b2_p025_ep30": "e28b_b2_p025_ep30_seed{s}"}
SEEDS = [42, 1337, 2024]


def t95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def load(arm):
    return [json.load(open(f)) for f in sorted(glob.glob(f"runs/e28/measure_{arm}_seed*.json"))]


def diag(run_dir):
    M = np.array(json.load(open(f"runs/{run_dir}/mafc_results.json"))["accuracy_matrix"])
    return [M[k, k] for k in range(5)]


def main():
    ap = argparse.ArgumentParser(description="E28 row")
    ap.add_argument("--out", default="runs/e28_row.json")
    args = ap.parse_args()
    print("=" * 116)
    print("E28 ROW   does training for equivariance make readout repair sufficient?  (primary set: tasks 3, 4)")
    print("=" * 116)
    out = {"primary_tasks": [3, 4], "arms": {}}

    print(f"\n  {'arm':<16}{'clean':>8}{'swapped':>9}{'drop':>8}{'refit cl':>10}{'refit sw':>10}"
          f"{'D_refit':>10}{'delta':>8}{'||A-I||':>9}{'swap-ID':>9}")
    for a in ARMS:
        ds = load(a)
        if not ds:
            print(f"  {LABEL[a]:<16} no artifacts"); continue
        P = {k: [d["primary"][k] for d in ds] for k in ds[0]["primary"]}
        rec = {k: {"mean": float(np.mean(v)), "t95": t95(v), "per_seed": v} for k, v in P.items()}
        rec["seeds"] = [d["seed"] for d in ds]
        rec["chance_swap_id"] = ds[0]["rows"][0]["swap_id_chance"]
        out["arms"][a] = rec
        print(f"  {LABEL[a]:<16}{rec['clean']['mean']:>8.3f}{rec['swapped']['mean']:>9.3f}"
              f"{rec['swap_drop']['mean']:>8.3f}{rec['refit_clean']['mean']:>10.3f}"
              f"{rec['refit_swapped']['mean']:>10.3f}{rec['delta_refit']['mean']:>+10.4f}"
              f"{rec['equivariance_delta']['mean']:>8.3f}{rec['A_minus_I_rel']['mean']:>9.3f}"
              f"{rec['swap_id_acc']['mean']:>9.3f}")
    ch = out["arms"][ARMS[0]]["chance_swap_id"] if ARMS[0] in out["arms"] else 0.05
    print(f"  {'':16}{'':>8}{'':>9}{'':>8}{'':>10}{'':>10}{'':>10}{'':>8}{'':>9}{'chance ' + f'{ch:.2f}':>9}")

    # ---- invariant or equivariant, per arm -------------------------------------------
    print("\n  WHICH PROPERTY (the distinction Delta_refit alone cannot make)")
    for a in ARMS:
        if a not in out["arms"]:
            continue
        r = out["arms"][a]
        drop, AmI, sid = r["swap_drop"]["mean"], r["A_minus_I_rel"]["mean"], r["swap_id_acc"]["mean"]
        keeps_id = sid > ch * 2
        survives = drop < 0.05
        verdict = ("INVARIANT (survives the swap unrepaired; identity discarded)" if survives and not keeps_id
                   else "EQUIVARIANT (swap costs, identity retained)" if not survives and keeps_id
                   else "invariant-ish but identity retained" if survives and keeps_id
                   else "neither: the swap costs and the identity is gone")
        print(f"    {LABEL[a]:<16} drop {drop:+.3f}  ||A-I|| {AmI:.3f}  swap-ID {sid:.3f} -> {verdict}")
        out["arms"][a]["property"] = verdict

    # ---- competence, both aggregations -------------------------------------------------
    print("\n  COMPETENCE (sec 5). Contracted: per task PER SEED. Mean-over-seeds printed beside it,")
    print("  as an aggregation ambiguity found AFTER the data — not a verdict.")
    B0 = {s: diag(RUN["b0_ep30"].format(s=s)) for s in SEEDS}
    comp = {}
    for a in ARMS:
        D = np.array([[diag(RUN[a].format(s=s))[k] - B0[s][k] for k in range(5)] for s in SEEDS])
        per_seed_worst = D.min(1)
        mean_per_task = D.mean(0)
        comp[a] = {"worst_cell": float(D.min()), "pass_per_seed": bool(D.min() >= -0.05),
                   "mean_per_task": mean_per_task.tolist(),
                   "worst_mean_task": float(mean_per_task.min()),
                   "pass_mean_over_seeds": bool(mean_per_task.min() >= -0.05),
                   "per_seed_worst": per_seed_worst.tolist()}
        print(f"    {LABEL[a]:<16} per-seed worst {D.min():+.3f} -> {'PASS' if comp[a]['pass_per_seed'] else 'FAIL'}"
              f"   |  worst per-task mean {mean_per_task.min():+.3f} -> "
              f"{'PASS' if comp[a]['pass_mean_over_seeds'] else 'FAIL'}")
    out["competence"] = comp

    # ---- predictions --------------------------------------------------------------------
    print("\n  PREDICTIONS (sec 7), on the primary set")
    preds = []
    def add(stmt, odds, hit, note=""):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<62} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")
    A = out["arms"]
    b0, b2 = A.get("b0_ep30"), A.get("b2_ep30")
    if b0 and b2:
        fl = max(b0["delta_refit"]["t95"], b2["delta_refit"]["t95"])
        d = b0["delta_refit"]["mean"] - b2["delta_refit"]["mean"]
        add("B2's Delta_refit below B0's by more than floor", 65, d > fl,
            f"B0 {b0['delta_refit']['mean']:+.4f} vs B2 {b2['delta_refit']['mean']:+.4f}, floor {fl:.4f}")
        add("B2's Delta_refit below 0.02", 20, b2["delta_refit"]["mean"] < 0.02,
            f"{b2['delta_refit']['mean']:+.4f}")
    b1 = A.get("b1_ep30")
    if b1:
        add("B1 swapped within 2pp of clean (invariance achieved)", 60,
            abs(b1["swap_drop"]["mean"]) <= 0.02, f"drop {b1['swap_drop']['mean']:+.4f}")
    if comp:
        add("B1 passes competence", 65, comp["b1_ep30"]["pass_per_seed"],
            f"worst {comp['b1_ep30']['worst_cell']:+.3f}")
        add("B2 passes competence at some mu", 75,
            comp["b2_ep30"]["pass_per_seed"] or comp["b2_p025_ep30"]["pass_per_seed"],
            f"worst {comp['b2_ep30']['worst_cell']:+.3f}")
    if b1 and b2:
        add("B2 shows ||A-I|| larger than B1's while delta is comparable", 50,
            b2["A_minus_I_rel"]["mean"] > b1["A_minus_I_rel"]["mean"]
            and abs(b2["equivariance_delta"]["mean"] - b1["equivariance_delta"]["mean"]) < 0.1,
            f"||A-I|| B1 {b1['A_minus_I_rel']['mean']:.3f} vs B2 {b2['A_minus_I_rel']['mean']:.3f}; "
            f"delta {b1['equivariance_delta']['mean']:.3f} vs {b2['equivariance_delta']['mean']:.3f}")
    if b0:
        add("B0 must-fail passes on the primary set", 90,
            b0["delta_refit"]["mean"] > 0.05, f"{b0['delta_refit']['mean']:+.4f}")
    out["predictions"] = preds
    sc = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n    scored {len(sc)}: fired {sum(p['outcome']=='fired' for p in sc)}, "
          f"miss {sum(p['outcome']=='MISS' for p in sc)}, "
          f"did not fire {sum(p['outcome']=='did not fire' for p in sc)}")

    # ---- the reading -----------------------------------------------------------------------
    if b0 and b1 and b2:
        b2_works = (b2["swap_drop"]["mean"] > 0.05 and
                    b2["delta_refit"]["mean"] < b0["delta_refit"]["mean"] - max(b0["delta_refit"]["t95"], b2["delta_refit"]["t95"]) and
                    b2["swap_id_acc"]["mean"] > ch * 2)
        b1_invariant = b1["swap_drop"]["mean"] < 0.05
        if b2_works:
            rd = ("THE METHOD WORKS on HAR permutations: B2 loses accuracy under an unseen swap, a "
                  "readout refit recovers it, and channel identity is still readable. Paper 2 has its result.")
        elif b1_invariant and b2["swap_drop"]["mean"] < 0.05:
            rd = ("INVARIANCE SUFFICES: both arms survive an unseen swap with no repair at all, so "
                  "permutations do not need the method. Paper 2 must show a family where invariance costs.")
        elif b2["delta_refit"]["mean"] >= b0["delta_refit"]["mean"]:
            rd = ("NEITHER ARM LOWERS Delta_refit: augmentation does not produce the property on this "
                  "encoder. The method fails here as specified.")
        else:
            rd = "Mixed; read per arm against the table above."
        print(f"\n  READING (sec 6): {rd}")
        print("  NOTE: every augmented arm fails the contracted competence bar, so these are "
              "DESCRIPTIVE under sec 5's rule.")
        out["reading"] = rd
        out["competence_caveat"] = ("all augmented arms fail the contracted per-seed competence bar; "
                                    "the method reading is descriptive, not a comparison")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
