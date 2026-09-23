"""
E12 — H-V1 from the per-run decompositions. Pure arithmetic; no GPU, no model.

Contract: docs/E12_prereg.md sec 4.
  H-V1: task-IL, base arm — reader share F_read/(F_enc+F_read) >= 0.50 pooled,
  CI LOWER BOUND above 0.50.

PROTOCOL, inherited without amendment:
  * per-cell computation with the denominator guard applied PER CELL, never
    post-pooling (catch 26); exclusions counted and printed, since an exclusion
    rate is a reportable fact about the benchmark rather than a footnote
  * mean-of-ratios, not ratio-of-means
  * BOTH denominators printed. The identity F_enc + F_read - R = F_total gives
    two defensible bases: the raw channel sum (F_enc + F_read) and the deployed
    forgetting it must explain (F_total). They differ by R, the instrument's own
    bias, so quoting one without the other hides where that bias lands.
  * pooled |R| <= 0.05 per arm, the instrument gate — no channel claim is read
    from an arm that fails it
  * Welch beside the CIs wherever two arms are compared (sec 7)
  * pooled-with-CI is PRIMARY for the ViT section; per-seed values are shown but
    never leaned on (the task-0 drop's seed spread was 31.6pp)

Usage:
    python scripts/e12_hv1.py
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

HV1_BAR = 0.50
R_BAR = 0.05
DENOM_MIN = 0.02          # per-cell guard, same constant as the rho screens


def cell_shares(rows):
    """Per-cell reader share under both denominators, with the guard applied."""
    out = []
    for r in rows:
        raw = r["F_enc"] + r["F_read"]
        adj = r["F_total"]
        rec = {"seed": r["seed"], "task": r["task"], "F_read": r["F_read"],
               "d_raw": raw, "d_adj": adj,
               "share_raw": r["F_read"] / raw if abs(raw) > DENOM_MIN else None,
               "share_adj": r["F_read"] / adj if abs(adj) > DENOM_MIN else None,
               "n_test": r["n_test"], "acc_refit": r["acc_refit"],
               "acc_orig": r["acc_orig"], "acc_ceiling": r["acc_ceiling"]}
        out.append(rec)
    return out


def pooled(vals):
    """Mean, SE of the mean, 95% CI — across cells."""
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return float("nan"), float("nan"), (float("nan"), float("nan")), 0
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, se, (m - 1.96 * se, m + 1.96 * se), len(v)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


def main():
    ap = argparse.ArgumentParser(description="E12 H-V1")
    ap.add_argument("--dir", default="runs/e12_decomp")
    ap.add_argument("--out", default="runs/e12_hv1.json")
    args = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        name = Path(f).stem                      # base_42
        arm, seed = name.rsplit("_", 1)
        runs.setdefault(arm, {})[int(seed)] = json.load(open(f))

    print("=" * 100)
    print("E12 — H-V1: does the reader channel carry the majority on a ViT?")
    print("=" * 100)

    summary = {}
    for arm in ("base", "adapt"):
        if arm not in runs:
            continue
        allrows = [r for s in sorted(runs[arm]) for r in runs[arm][s]]
        cells = cell_shares(allrows)

        # ---- instrument gate first: no channel claim from a failing arm ------
        Rm, Rse, Rci, _ = pooled([r["R"] for r in allrows])
        r_ok = abs(Rm) <= R_BAR
        worst_R = max(allrows, key=lambda r: abs(r["R"]))
        print(f"\n{'-'*100}\n{arm.upper()} — {len(runs[arm])} seeds x "
              f"{len(runs[arm][sorted(runs[arm])[0]])} old tasks = {len(allrows)} cells")
        print(f"{'-'*100}")
        print(f"  instrument gate: pooled R {Rm:+.4f} [{Rci[0]:+.4f}, {Rci[1]:+.4f}] "
              f"vs {R_BAR} -> {'PASS' if r_ok else 'FAIL'}  "
              f"(worst cell {worst_R['R']:+.4f} s{worst_R['seed']}/T{worst_R['task']}, "
              f"{sum(abs(r['R'])>R_BAR for r in allrows)}/{len(allrows)} cells over)")

        idmax = max(r["identity_resid"] for r in allrows)
        p3max = max(r["p3a"] for r in allrows)
        fp = [r["fp16_delta"] for r in allrows if r.get("fp16_delta") is not None]
        print(f"  identity max resid {idmax:.1e} | P3a max |d| {p3max:.1e} | "
              f"fp16 reload max |d| {max(abs(x) for x in fp) if fp else float('nan'):.5f}")

        # ---- channels --------------------------------------------------------
        fe, _, feci, _ = pooled([r["F_enc"] for r in allrows])
        fr, _, frci, _ = pooled([r["F_read"] for r in allrows])
        ft, _, ftci, _ = pooled([r["F_total"] for r in allrows])
        print(f"  F_total {ft:.4f} [{ftci[0]:.4f}, {ftci[1]:.4f}]")
        print(f"  F_enc   {fe:.4f} [{feci[0]:.4f}, {feci[1]:.4f}]")
        print(f"  F_read  {fr:+.4f} [{frci[0]:+.4f}, {frci[1]:+.4f}]")

        # ---- reader share, both denominators, per-cell then pooled -----------
        for tag, key, dkey in (("raw (F_enc+F_read)", "share_raw", "d_raw"),
                               ("adj (F_total)", "share_adj", "d_adj")):
            m, se, ci, n = pooled([c[key] for c in cells])
            excl = sum(1 for c in cells if c[key] is None)
            print(f"  reader share, {tag:<20} {m*100:+7.2f}% "
                  f"[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]   "
                  f"kept {n}/{len(cells)}"
                  f"{f' (excluded {excl}, |denom| <= {DENOM_MIN})' if excl else ''}")

        share_m, share_se, share_ci, _ = pooled([c["share_raw"] for c in cells])
        met = (share_m >= HV1_BAR) and (share_ci[0] > HV1_BAR)
        print(f"  H-V1 ({arm}): share {share_m*100:.2f}% vs bar {HV1_BAR*100:.0f}%, "
              f"CI lower {share_ci[0]*100:+.2f}% -> {'MET' if met else 'NOT MET'}")

        # ---- the identity column --------------------------------------------
        t0 = [r for r in allrows if r["task"] == 0]
        print(f"  TASK 0 across seeds: ceiling "
              f"{np.mean([r['acc_ceiling'] for r in t0]):.4f} -> deployed "
              f"{np.mean([r['acc_orig'] for r in t0]):.4f} -> REFIT "
              f"{np.mean([r['acc_refit'] for r in t0]):.4f}   "
              f"(chance 0.2000)")
        refit_all = [r["acc_refit"] for r in allrows]
        print(f"  refit over ALL old tasks: {np.mean(refit_all):.4f} "
              f"[{np.min(refit_all):.4f}, {np.max(refit_all):.4f}] vs chance 0.2000")

        summary[arm] = {"R": Rm, "r_ok": bool(r_ok), "F_enc": fe, "F_read": fr,
                        "F_total": ft, "share_raw": share_m,
                        "share_ci": list(share_ci), "hv1_met": bool(met),
                        "n_cells": len(cells)}

    # ---- arm comparison ----------------------------------------------------
    if "base" in summary and "adapt" in summary:
        a = [r["F_total"] for s in sorted(runs["base"]) for r in runs["base"][s]]
        b = [r["F_total"] for s in sorted(runs["adapt"]) for r in runs["adapt"][s]]
        t, df = welch(a, b)
        print(f"\n{'-'*100}")
        print(f"  base vs adapt, F_total: {np.mean(a):.4f} vs {np.mean(b):.4f}  "
              f"Welch t={t:+.3f} df={df:.1f}")

    print("\n" + "=" * 100)
    print("VERDICT (computed from the values printed above)")
    print("=" * 100)
    base = summary.get("base", {})
    if not base.get("r_ok"):
        print("  instrument gate FAILED on the base arm — no channel claim is read")
    elif base.get("hv1_met"):
        print("  H-V1 MET — the reader channel carries the majority on a ViT")
    else:
        print(f"  H-V1 NOT MET — reader share {base.get('share_raw', float('nan'))*100:.2f}% "
              f"(CI upper {base.get('share_ci',[0,0])[1]*100:+.2f}%) is far below "
              f"the {HV1_BAR*100:.0f}% bar")
        print("  -> BRANCH (C): the decomposition is architecture- or scale-bound.")
        print("     Paper 1 states it as measured on small encoders, with this as")
        print("     the boundary. The ViT's forgetting is ENCODER-side.")
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
