"""E25 B row -- per-step drift (docs/E25_prereg.md sec B).

Reads runs/e25/b/{arm}/steps_seed{s}.json and scores the five registered
predictions.

ONE DEVIATION, NAMED RATHER THAN ABSORBED. The contract's third measurement is
"composed repair vs D1 single-shot vs REFIT", and prediction 5 is "A3's 20-step
composition within 5pp of refit". What `e25b_steps.py` recorded is
`acc_ceiling` = acc(h_k, Z_k), the ERA HEAD on era features -- the deployed
ceiling, not the convex refit. The two differ: the refit is fit to the era
features and is the higher number. Reported against the ceiling with the
substitution stated, and the prediction is marked accordingly rather than
scored against a quantity it does not name. Ruling is the user's.

Floors are per cell: 1.96*SE_binomial on that cell's test set, already carried in
the artifact as `res95`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ARMS = ["s72_off", "e18_pmd_mlp", "e18_pmd_lstm", "e18_rmd_mlp", "e23b_t20"]
FAMILY = {"s72_off": "LSTM", "e18_pmd_lstm": "LSTM", "e23b_t20": "LSTM",
          "e18_pmd_mlp": "MLP", "e18_rmd_mlp": "MLP"}
BAR = 0.15                # sec B: per-step residual bar, against D1's 41-45%
PP5 = 0.05


def t95(x):
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def load(arm):
    return [json.load(open(f)) for f in sorted(glob.glob(f"runs/e25/b/{arm}/steps_seed*.json"))]


def main():
    ap = argparse.ArgumentParser(description="E25 B row")
    ap.add_argument("--out", default="runs/e25b_row.json")
    args = ap.parse_args()
    print("=" * 112 + "\nE25 B ROW   per-step drift: is one boundary closer to linear, and do the "
          "steps compose?\n" + "=" * 112)
    res, out = {}, {}

    print(f"\n  {'arm':<14}{'fam':<5}{'seeds':<8}{'cells':>6}{'step resid':>12}{'single':>9}"
          f"{'deployed':>10}{'composed':>10}{'1-shot':>9}{'ceiling':>9}{'shuf':>8}{'plumb':>8}")
    for arm in ARMS:
        ds = load(arm)
        if not ds:
            print(f"  {arm:<14} no artifacts"); continue
        cells = [c for d in ds for c in d["cells"]]
        m = lambda k: float(np.mean([c[k] for c in cells]))
        per_seed_res = [float(np.mean([c["residual_step_mean"] for c in d["cells"]])) for d in ds]
        row = {"seeds": sorted(d["seed"] for d in ds), "n_cells": len(cells),
               "num_tasks": ds[0]["num_tasks"], "family": FAMILY[arm],
               "residual_step_mean": m("residual_step_mean"),
               "residual_step_t95": t95(per_seed_res),
               "residual_step_max": float(np.max([c["residual_step_max"] for c in cells])),
               "residual_single_shot": m("residual_single_shot"),
               "acc_deployed": m("acc_deployed"), "acc_composed": m("acc_repaired_composed"),
               "acc_single_shot": m("acc_repaired_single_shot"), "acc_ceiling": m("acc_ceiling"),
               "acc_shuffled": m("acc_shuffled"), "c_plumb_norm": m("c_plumb_norm"),
               "spread_step_mean": m("spread_step_mean"),
               "res95_mean": m("res95"),
               "composed_vs_chain_maxabs": float(np.max([c["composed_vs_chain_maxabs"] for c in cells]))}
        row["residual_under_bar"] = bool(row["residual_step_mean"] < BAR)
        row["composed_vs_single_pp"] = row["acc_composed"] - row["acc_single_shot"]
        row["composed_within_5pp_of_single"] = bool(abs(row["composed_vs_single_pp"]) <= PP5)
        row["composed_within_5pp_of_ceiling"] = bool(abs(row["acc_composed"] - row["acc_ceiling"]) <= PP5)
        # C-SHUF must-fail: shuffled pairing collapses the composed repair to the floor
        row["c_shuf_cells_at_or_below_deployed"] = int(sum(
            c["acc_shuffled"] <= c["acc_deployed"] + c["res95"] for c in cells))
        row["c_shuf_pass"] = row["c_shuf_cells_at_or_below_deployed"] == len(cells)
        share = [p for d in ds for p in d.get("sharing", [])]
        if share:
            row["sharing"] = {"n_pairs": len(share),
                              "loss_mean": float(np.mean([p["loss"] for p in share])),
                              "loss_min": float(np.min([p["loss"] for p in share])),
                              "loss_max": float(np.max([p["loss"] for p in share])),
                              "within_res": int(sum(abs(p["loss"]) <= p["res95"] for p in share))}
        out[arm] = row
        print(f"  {arm:<14}{FAMILY[arm]:<5}{str(row['seeds']):<8}{len(cells):>6}"
              f"{row['residual_step_mean']:>12.3f}{row['residual_single_shot']:>9.3f}"
              f"{row['acc_deployed']:>10.4f}{row['acc_composed']:>10.4f}"
              f"{row['acc_single_shot']:>9.4f}{row['acc_ceiling']:>9.4f}"
              f"{row['acc_shuffled']:>8.4f}{row['c_plumb_norm']:>8.2f}")

    print("\n  CONTROLS")
    for arm, r in out.items():
        sh = r.get("sharing")
        print(f"    {arm:<14} C-SHUF {r['c_shuf_cells_at_or_below_deployed']}/{r['n_cells']} "
              f"{'PASS' if r['c_shuf_pass'] else 'FAIL'} | composed==chain max "
              f"{r['composed_vs_chain_maxabs']:.1e} | C-PLUMB {r['c_plumb_norm']:.2f}"
              + (f" | sharing {sh['n_pairs']} pairs, own-cross {sh['loss_mean']:+.4f} "
                 f"[{sh['loss_min']:+.4f},{sh['loss_max']:+.4f}], within res "
                 f"{sh['within_res']}/{sh['n_pairs']}" if sh else ""))

    print("\n  COMPOSITION CONSISTENCY (a reading, reported either way)")
    for arm, r in out.items():
        print(f"    {arm:<14} composed {r['acc_composed']:.4f} vs single-shot "
              f"{r['acc_single_shot']:.4f} ({r['composed_vs_single_pp']:+.4f}) -> "
              f"{'within 5pp' if r['composed_within_5pp_of_single'] else 'OUTSIDE 5pp'}")

    # ---- predictions --------------------------------------------------------------
    print("\n  PREDICTIONS (sec B)")
    preds = []

    def add(stmt, odds, hit, note=""):
        v = "pending" if hit is None else ("fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<66} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")

    mlp = [r for a, r in out.items() if FAMILY[a] == "MLP"]
    lstm = [r for a, r in out.items() if FAMILY[a] == "LSTM"]
    add("per-step residual < 15% on the MLP arms", 50,
        None if not mlp else all(r["residual_under_bar"] for r in mlp),
        "" if not mlp else "residuals " + ", ".join(f"{r['residual_step_mean']:.3f}" for r in mlp))
    add("per-step residual < 15% on the LSTM arms", 35,
        None if not lstm else all(r["residual_under_bar"] for r in lstm),
        "" if not lstm else "residuals " + ", ".join(f"{r['residual_step_mean']:.3f}" for r in lstm))
    add("composed within 5pp of single-shot", 55,
        None if not out else all(r["composed_within_5pp_of_single"] for r in out.values()))
    shared = [r["sharing"] for r in out.values() if r.get("sharing")]
    add("per-step repair shared across tasks within floor", 30,
        None if not shared else all(s["within_res"] == s["n_pairs"] for s in shared),
        "" if not shared else "within res " + ", ".join(f"{s['within_res']}/{s['n_pairs']}" for s in shared))
    a3 = out.get("e23b_t20")
    add("A3's 20-step composition within 5pp of refit", 30,
        None if not a3 else a3["composed_within_5pp_of_ceiling"],
        "SCORED AGAINST THE CEILING, not the refit -- see the module docstring; "
        "the contract names the refit and the artifact records acc(h_k, Z_k)")
    scored = [p for p in preds if p["outcome"] != "pending"]
    print(f"\n    scored {len(scored)}: fired {sum(p['outcome']=='fired' for p in scored)}, "
          f"miss {sum(p['outcome']=='MISS' for p in scored)}, "
          f"did not fire {sum(p['outcome']=='did not fire' for p in scored)}")

    # ---- the pre-committed reading ---------------------------------------------------
    if out:
        any_under = any(r["residual_under_bar"] for r in out.values())
        comp_ok = all(r["composed_within_5pp_of_single"] for r in out.values())
        if not any_under:
            reading = ("PER-STEP DRIFT IS AS NON-LINEAR AS ENDPOINT DRIFT. No arm's per-step "
                       "residual clears the 15% bar, and each sits close to its own single-shot "
                       "residual, so the non-linearity is PER-STEP. Sec B's third branch: no "
                       "rolling method has a foundation here.")
        elif comp_ok:
            reading = ("LOCALLY LINEAR AND COMPOSABLE: a one-checkpoint rolling method has a "
                       "foundation.")
        else:
            reading = ("Steps linear but errors accumulate under composition: needs re-anchoring.")
        print(f"\n  READING (pre-committed, sec B): {reading}")
        res["reading"] = reading
    res.update({"arms": out, "predictions": preds, "bar": BAR,
                "deviation": ("prediction 5 scored against acc_ceiling = acc(h_k, Z_k); the "
                              "contract names the refit. Stated, not absorbed.")})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
