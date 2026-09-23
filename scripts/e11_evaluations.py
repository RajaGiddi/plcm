"""
E11 — the three evaluations that run BEFORE any ledger edit.

The recompute moved every number toward flattering the method (E10 C3 rho
0.615 -> 0.974, H-X1 NOT MET -> MET). That is the direction this program treats
as most suspect, so the ledger is held until these three run, in this order:

  EV1  EXCLUSION-SET RECONCILIATION (bookkeeping; gates the other two)
       The per-cell guard excluded a DIFFERENT set of cells before and after the
       correction. If the new rho is computed over a friendlier subset of cells,
       the improvement is partly a change of denominator population, not of
       method. Prints every cell's denominator old vs new, and which crossed.

  EV2  DEGENERATE-CELL AUDIT
       C0/C0C1 carry cell sd 8.6 on OFF. A mean-of-ratios over cells whose
       denominators approach zero is dominated by whichever cell is closest to
       degenerate. Prints per-cell rho ranked by |rho|, and the mean recomputed
       without the worst cell, so the reader can see what the headline rests on.

  EV3  CATCH 24 -- IS C3 DOMINATED BY ITS OWN INGREDIENT?  (the one with teeth)
       C3 requires the full era snapshot to label its pseudo-data. The TRIVIAL
       use of that same resource is to deploy the snapshot itself for task k.
       That baseline is already recorded in every screen row as `acc_ceiling`
       (the era model's own deployed logits on task k's test set), so it is
       computed here on the SAME instrument, cells and denominators as C3 -- no
       cross-instrument comparison, which is the defect that produced the 1.24
       figure's cousin in the first place.

       If rho_snapshot >= rho_C3 at <= C3's storage, then C3 is dominated by its
       own ingredient and the honest claim is MECHANISM-shaped ("the reader
       channel is repairable by generative refit, so the damage is reader-side")
       rather than DEPLOYMENT-shaped ("use C3").

Usage:
    python scripts/e11_evaluations.py --old runs/e10/cures_e10.json \
        --new runs/e11_e10/cures_e10.json --bench e10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rho_percell import (RHO_MIN_DENOM, rho_and_se, score_arm,
                                 E10_NTEST, E8_NTEST)

CURES = ["C0", "C1", "C0C1", "C2", "C3"]


def cellkey(r):
    return f"s{r['seed']}/t{r['task']}"


def excluded_set(scored):
    return {f"s{c['seed']}/t{c['task']}" for c in scored["excluded"]}


def evaluation1(old, new, ntest):
    print("=" * 96)
    print("EV1 — EXCLUSION-SET RECONCILIATION   (did the guard change which cells count?)")
    print("=" * 96)
    print(f"  guard: denominator > {RHO_MIN_DENOM}\n")
    stable = {}
    for arm in new:
        if arm not in old:
            continue
        print(f"  --- {arm} ---")
        o_by = {cellkey(r): r for r in old[arm]}
        for cure in CURES:
            so, sn = score_arm(old[arm], ntest, cure), score_arm(new[arm], ntest, cure)
            ex_o, ex_n = excluded_set(so), excluded_set(sn)
            only_o, only_n = sorted(ex_o - ex_n), sorted(ex_n - ex_o)
            stable[(arm, cure)] = not only_o and not only_n
            print(f"    {cure:<5} kept {so['n_kept']:>2} -> {sn['n_kept']:>2}   "
                  f"excluded old {sorted(ex_o) or '[]'}  new {sorted(ex_n) or '[]'}")
            if not stable[(arm, cure)]:
                print(f"    {'':<5} POPULATION CHANGED: only-old {only_o}  "
                      f"only-new {only_n}  (both: {sorted(ex_o & ex_n)})")
                # "The population moved" is not yet a finding: what matters is
                # which WAY the moved cells would have pushed the mean. An
                # exclusion that removes a rho of -30 is the guard working; one
                # that removes a rho below the mean is the headline being helped.
                for c in sn["excluded"]:
                    if f"s{c['seed']}/t{c['task']}" in only_n:
                        direction = ("ABOVE" if c["rho"] > sn["mean"] else "BELOW")
                        print(f"    {'':<5}   new-excluded s{c['seed']}/t{c['task']}: "
                              f"rho would be {c['rho']:+.3f} (D={c['D']:+.4f}), "
                              f"{direction} the kept mean {sn['mean']:+.3f} -> "
                              f"excluding it {'LOWERS' if c['rho'] > sn['mean'] else 'RAISES'} "
                              f"the headline")

        print(f"\n    per-cell C3 denominator (RAW), old -> new:")
        for r in sorted(new[arm], key=lambda r: (r["seed"], r["task"])):
            k = cellkey(r)
            ro = o_by.get(k)
            _, _, Dn = rho_and_se(r["C3"], r["acc_orig"], r["acc_refit"], r["R"],
                                  ntest[r["task"]], adjusted=False)
            if ro is None:
                print(f"      {k:<12}   (absent old)   new D {Dn:+.4f}")
                continue
            _, _, Do = rho_and_se(ro["C3"], ro["acc_orig"], ro["acc_refit"], ro["R"],
                                  ntest[ro["task"]], adjusted=False)
            crossed = (Do <= RHO_MIN_DENOM) != (Dn <= RHO_MIN_DENOM)
            print(f"      {k:<12}   old D {Do:+.4f}   new D {Dn:+.4f}"
                  f"{'   <- CROSSED THE GUARD' if crossed else ''}")
        print()
    # Sensitivity for the HEADLINE cure: what does C3 read if the guard-excluded
    # cells are forced back in? The guard is principled and pre-existing
    # (RHO_MIN_DENOM, catch 26), so exclusion is correct -- but when exclusion
    # RAISES the number, the reader is owed the value without it.
    print("  C3 SENSITIVITY — guard-excluded cells forced back in:")
    for arm in new:
        s = score_arm(new[arm], ntest, "C3")
        allc = s["kept"] + s["excluded"]
        forced = sum(c["rho"] for c in allc) / len(allc)
        print(f"    {arm:<10} guarded {s['mean']:+.3f} ({s['n_kept']} cells)   "
              f"forced-inclusion {forced:+.3f} ({len(allc)} cells)   "
              f"delta {forced - s['mean']:+.3f}")
    ok = all(stable.values())
    print(f"\n  EV1 VERDICT: {'same cell population throughout' if ok else 'POPULATION MOVED — the improvement is not measured on a fixed cell set; EV2/EV3 must be read with that in mind'}")
    return ok


def evaluation2(new, ntest):
    print("\n" + "=" * 96)
    print("EV2 — DEGENERATE-CELL AUDIT   (what does the mean-of-ratios rest on?)")
    print("=" * 96)
    for arm in new:
        print(f"\n  --- {arm} ---")
        for cure in CURES:
            s = score_arm(new[arm], ntest, cure)
            if not s.get("kept"):
                continue
            kept = sorted(s["kept"], key=lambda c: -abs(c["rho"]))
            worst, rest = kept[0], kept[1:]
            m_rest = sum(c["rho"] for c in rest) / len(rest) if rest else float("nan")
            print(f"    {cure:<5} mean {s['mean']:+.3f}  sd {s['spread']:.3f}  "
                  f"| worst cell s{worst['seed']}/t{worst['task']} "
                  f"rho {worst['rho']:+.3f} (D={worst['D']:+.4f})  "
                  f"| mean without it {m_rest:+.3f}")
            if abs(m_rest - s["mean"]) > 0.10:
                print(f"    {'':<5} ^ mean shifts {abs(m_rest - s['mean']):.3f} when "
                      f"the single worst cell is removed — headline rests on it")


def evaluation3(new, ntest):
    print("\n" + "=" * 96)
    print("EV3 — CATCH 24: is C3 dominated by the era snapshot it already requires?")
    print("=" * 96)
    print("  Trivial use of C3's assumed resource = DEPLOY THE SNAPSHOT for task k.")
    print("  That is `acc_ceiling` in every row: the era model's own deployed logits")
    print("  on task k's test set — same instrument, same cells, same denominators")
    print("  as C3. Storage: C3 needs the snapshot PLUS a generator; the trivial")
    print("  baseline needs the snapshot alone.\n")
    out = {}
    for arm in new:
        rows = [dict(r) for r in new[arm]]
        for r in rows:
            r["SNAP"] = r["acc_ceiling"]          # not in ERA_HEAD_CURES -> raw denom
        snap = score_arm(rows, ntest, "SNAP")
        c3 = score_arm(rows, ntest, "C3")
        print(f"  --- {arm} ---")
        print(f"    C3        mean rho {c3['mean']:+.3f}  "
              f"[{c3['mean']-1.96*c3['se']:+.3f}, {c3['mean']+1.96*c3['se']:+.3f}]  "
              f"kept {c3['n_kept']}  sd {c3['spread']:.3f}")
        print(f"    SNAPSHOT  mean rho {snap['mean']:+.3f}  "
              f"[{snap['mean']-1.96*snap['se']:+.3f}, {snap['mean']+1.96*snap['se']:+.3f}]  "
              f"kept {snap['n_kept']}  sd {snap['spread']:.3f}   (raw denominator, as C3)")
        gap = snap["mean"] - c3["mean"]
        dom = snap["mean"] >= c3["mean"]
        print(f"    -> snapshot {'DOMINATES' if dom else 'does NOT dominate'} C3 by "
              f"{gap:+.3f}, at strictly less storage")
        print(f"       => {'deployment-shaped claim (use C3) is not supportable; mechanism-shaped claim is' if dom else 'C3 beats the trivial use of its own ingredient; deployment-shaped claim survives'}")
        out[arm] = {"c3": c3["mean"], "snapshot": snap["mean"], "gap": gap}
    return out


def main():
    ap = argparse.ArgumentParser(description="E11 pre-ledger evaluations")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--bench", choices=["e10", "e8"], default="e10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ntest = E10_NTEST if args.bench == "e10" else E8_NTEST
    old, new = json.load(open(args.old)), json.load(open(args.new))

    ok = evaluation1(old, new, ntest)
    evaluation2(new, ntest)
    res = evaluation3(new, ntest)
    if args.out:
        json.dump({"population_stable": ok, "ev3": res}, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
