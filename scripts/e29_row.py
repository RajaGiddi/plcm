"""E29 row — scores the ten registered predictions and emits the readings.

AGGREGATION IS NOT RESTATED HERE. Part A's verdicts are read from
`runs/e29/pools.json`, which carries the definition it was measured under
(a target recovers at >= 9 of 10 draws; a pool succeeds at >= 20 of 22
targets). A row script that recomputed the verdict from raw rates would be a
parallel implementation of the bar, which is how a contract's stated
aggregation quietly becomes two different aggregations.

ONE DEFINITION THE CONTRACT LEFT OPEN, recorded rather than chosen silently:
prediction B2 says the error "plateaus by n = 256 (bias, not variance)" without
saying what plateau means. Scored here as the median at n = 1024 sitting within
20% of the median at n = 256 -- stated in this docstring, printed beside the
verdict, and flagged in the output so it is read as a definition supplied after
the fact rather than one registered with the prediction.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PLATEAU_TOL = 0.20


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(description="E29 row")
    ap.add_argument("--out", default="runs/e29_row.json")
    args = ap.parse_args()
    pools = load("runs/e29/pools.json")
    rot = load("runs/e29/rot_bias.json")
    reads = load("runs/e29/reads.json")
    tol = [json.load(open(f)) for f in sorted(glob.glob("runs/e29/tolerance_seed*.json"))]
    ds = [json.load(open(f)) for f in sorted(glob.glob("runs/e29/downstream_seed*.json"))]
    out = {"predictions": [], "parts": {}}
    print("=" * 112)
    print("E29 ROW   recovering an unknown format change from input statistics")
    print("=" * 112)

    if reads:
        print(f"\n  §0a REMEASURED on partition {reads['partition_fingerprint_shifted']} "
              f"(Modal: {reads['is_modal_partition']})")
        print(f"    {reads['branch']}")
        out["reads_branch"] = reads["branch"]

    # ---------------- Part A ------------------------------------------------------
    preds = []

    def add(stmt, odds, hit, note=""):
        v = ("unmeasurable" if (hit is None and "UNMEASURABLE" in note)
             else "pending" if hit is None
             else "fired" if hit else ("MISS" if odds >= 50 else "did not fire"))
        preds.append({"prediction": stmt, "odds": odds, "outcome": v, "note": note})
        print(f"    {stmt:<58} ~{odds:>2}%  {v}{('  [' + note + ']') if note else ''}")

    if pools:
        agg = pools["aggregation"]
        print(f"\n  PART A — pool composition (n=256; target recovers at "
              f">={agg['target_recovers_at']}/{agg['draws_per_target']}, pool succeeds at "
              f">={agg['pool_succeeds_at']}/{agg['n_targets']})")
        print(f"  {'pool':<26}{'succeeds':>10}{'targets rec':>13}{'pooled exact':>14}")
        tab = {}
        for key, v in pools["pools"].items():
            if v.get("unbuildable"):
                continue
            tab[key] = v
        for key in sorted(tab):
            if not key.endswith("_n256"):
                continue
            v = tab[key]
            print(f"  {key.replace('_n256',''):<26}{('YES' if v['succeeds'] else 'no'):>10}"
                  f"{v['targets_recovering']:>9}/{v['n_targets']}{v['pooled_exact_rate']:>14.3f}")
        out["parts"]["A"] = {k: {"succeeds": v["succeeds"],
                                 "targets_recovering": v["targets_recovering"],
                                 "pooled_exact_rate": v["pooled_exact_rate"]}
                             for k, v in tab.items()}
        print("\n  PREDICTIONS — Part A")

        def succ(key):
            v = tab.get(key)
            if v is None:
                # NOT a miss and not a did-not-fire: the cell has no measurement.
                # A5 (contiguous stride 2 at n=256) needs 512 consecutive source
                # windows and HAR subjects hold ~300, so the arm is unbuildable
                # at the contracted n for EVERY subject. The CLAUSE -> JOB ->
                # ARTIFACT table checked that the clause had a job; nothing
                # checked that the job's cell could exist. Recorded as
                # unmeasurable, with the reason, rather than scored.
                return None, "UNMEASURABLE at the contracted n (see note)"
            return v["succeeds"], (f"{v['targets_recovering']}/{v['n_targets']} targets, "
                                   f"pooled {v['pooled_exact_rate']:.2f}")
        for key, stmt, odds in (
                ("two_dynamic_n256", "A1 two dynamic activities, balanced, succeeds", 55),
                ("static_plus_dynamic_n256", "A2 one static + one dynamic, balanced, succeeds", 60),
                ("three_static_n256", "A3 three static activities only, succeeds", 25),
                ("all6_skew80_n256", "A4 all six skewed 80/20, succeeds", 50),
                ("contiguous_stride2_n256", "A5 contiguous session, stride 2, succeeds", 45),
                ("contiguous_stride1_n256", "A6 contiguous session, stride 1, succeeds", 35)):
            h, note = succ(key)
            add(stmt, odds, h, note)

    # ---------------- Part B ------------------------------------------------------
    if rot:
        print("\n  PART B — transfer bias across subject pairings")
        print(f"  {'n':>6}{'pairings':>10}{'median':>10}{'IQR':>18}{'max equivar spread':>21}")
        for k in sorted(rot["pairings"], key=lambda s: int(s[1:])):
            p = rot["pairings"][k]
            iqr = f"[{p['iqr'][0]:.2f}, {p['iqr'][1]:.2f}]"
            print(f"  {k[1:]:>6}{p['n_pairings']:>10}{p['median_deg']:>9.2f}°"
                  f"{iqr:>18}{p['max_equivariance_spread']:>20.2e}°")
        out["parts"]["B"] = rot["pairings"]

    print("\n  PREDICTIONS — Part B")
    if rot:
        p256 = rot["pairings"].get("n256")
        p1024 = rot["pairings"].get("n1024")
        if p256:
            add("B1 median error across pairings at n=256 below 10 deg", 55,
                p256["median_deg"] < 10.0, f"median {p256['median_deg']:.2f} deg")
        if p256 and p1024:
            rel = abs(p1024["median_deg"] - p256["median_deg"]) / max(p256["median_deg"], 1e-9)
            add("B2 error plateaus by n=256 (bias, not variance)", 65, rel < PLATEAU_TOL,
                f"n256 {p256['median_deg']:.2f} vs n1024 {p1024['median_deg']:.2f}, "
                f"rel {rel:.2f} < {PLATEAU_TOL} [DEFINITION SUPPLIED AFTER THE FACT]")
    if tol:
        d9 = [r["by_theta"]["9.0"]["drop_from_clean"] if "9.0" in r["by_theta"]
              else r["by_theta"][9.0]["drop_from_clean"] for t in tol for r in t["rows"]]
        add("B3 deployed accuracy at 9 deg error within 2pp of exact", 50,
            float(np.mean(d9)) <= 0.02, f"mean drop {np.mean(d9):+.4f} over {len(d9)} cells")
    if ds:
        e = [t["summary"]["rot"]["est_minus_exact"] for t in ds]
        ep = [t["summary"]["perm"]["est_minus_exact"] for t in ds]
        add("B4 estimated re-layout within 2pp of exact (rotations)", 45,
            abs(float(np.mean(e))) <= 0.02,
            f"rot {np.mean(e):+.4f}; permutations {np.mean(ep):+.4f} (reported beside)")

    out["predictions"] = preds
    sc = [p for p in preds if p["outcome"] not in ("pending", "unmeasurable")]
    print(f"\n  scored {len(sc)}: fired {sum(p['outcome']=='fired' for p in sc)}, "
          f"miss {sum(p['outcome']=='MISS' for p in sc)}, "
          f"did not fire {sum(p['outcome']=='did not fire' for p in sc)}, "
          f"unmeasurable {sum(p['outcome']=='unmeasurable' for p in preds)}, "
          f"pending {sum(p['outcome']=='pending' for p in preds)}")

    # ---------------- readings -------------------------------------------------------
    rd = []
    if pools:
        P = pools["pools"]
        def ok(k):
            return bool(P.get(k, {}).get("succeeds"))
        # THE CONTRACT'S READINGS TABLE HAD A GAP, found on reading the data and
        # recorded rather than papered over. It offered "a two-activity pool
        # suffices" OR "only the full mix suffices", treating "two-activity" as
        # one category. It is not: `static_plus_dynamic` succeeds 22/22 while
        # `two_dynamic` fails 0/22, both at n = 256 with two activities. What
        # separates them is CONTRAST, not count. A first version of this reading
        # keyed only on `two_dynamic` and therefore printed "only the full
        # activity mix suffices" while a two-activity pool sat in the same table
        # succeeding -- a verdict disagreeing with the numbers beside it, which
        # is catch 22's exact shape.
        two_ok = [k for k in ("two_dynamic_n256", "static_plus_dynamic_n256") if ok(k)]
        if two_ok and not ok("two_dynamic_n256"):
            rd.append("A: a two-activity pool of a few hundred windows suffices, BUT ONLY IF THE "
                      "TWO CONTRAST. static+dynamic recovers 22/22 at n=256; two dynamic "
                      "activities recover 0/22 and three static recover 0/22. The requirement P5 "
                      "must state is not 'how many activities' but 'at least one static and one "
                      "dynamic' — a calibration period can be short and passive, and must be "
                      "varied in kind rather than in count.")
        elif two_ok:
            rd.append("A: a two-activity pool of a few hundred windows suffices — label-free "
                      "re-layout is practical for channel swaps with a short, varied calibration "
                      "period, and P5 is built on it with that requirement stated.")
        elif ok("all6_n256"):
            rd.append("A: only the full activity mix suffices — P5 needs a deliberate calibration "
                      "protocol, not passive windows.")
        else:
            rd.append("A: no composition tested reaches the bar at n=256; the requirement is "
                      "larger than any pool here supplies.")
    if rot and tol:
        p256 = rot["pairings"].get("n256")
        # THE TOLERANCE IS READ OFF TASK 4 ONLY, and that is a correction, not a
        # convenience. B0's task-2 accuracy in the final row is 0.48 (recorded
        # matrix) and 0.3878 on the probe subset: a task the model has largely
        # lost. The downstream arm makes the consequence unmissable -- a RANDOM
        # rotation scores 0.4721 on task 2 against 0.3878 clean, i.e. destroying
        # the input BEATS it. A tolerance curve is a measure of how much error a
        # working readout absorbs, and task 2 has no working readout to absorb
        # it with, so its flat curve reports absence of signal as robustness.
        # Catch 24's shape: price the trivial case first -- at 0.39, how much
        # can a perturbation cost?
        t4 = [d.get("4") or d.get(4) for d in (x["largest_theta_within_2pp"] for x in tol)]
        t4 = [float(v) for v in t4 if v is not None]
        if p256 and t4:
            tol_deg = float(np.median(t4))
            med, q3, mx = p256["median_deg"], p256["iqr"][1], p256["max"]
            rd.append(f"B: the MEDIAN transfer bias ({med:.1f} deg) sits inside the model's "
                      f"tolerance ({tol_deg:.0f} deg on task 4, the task with a working readout), "
                      f"so rotations are recoverable well enough from covariance on a typical "
                      f"subject pair. But the spread is the story the median hides: the upper "
                      f"quartile is {q3:.1f} deg and the worst pairing {mx:.1f} deg, both OUTSIDE "
                      f"tolerance. P5 extends to rotations for most deployments and fails for a "
                      f"minority, and P6's equivariant encoder is what would have to absorb that "
                      f"tail.")
    for r in rd:
        print(f"\n  READING: {r}")
    out["readings"] = rd
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
