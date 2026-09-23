"""
E12 — read P1 and P2 from the precondition probe. Gates everything downstream.

Contract: docs/E12_prereg.md sec 2. No decomposition number is read until both
report, and the verdicts here are computed from the arrays printed beside them
(catch 22), never written to mirror an expectation.

  P1  competence, SCALE-FREE. The frozen-probe arm is the reference:
      it must clear 0.85 (it did: min 0.8940, mean 0.9741), and the trainable
      trunk must land within 5pp of that mean per task. An absolute bar on a
      pretrained ViT is close to unfailable — catch 25's category — which is
      why the reference does the work.

  P2  forgetting exists AND is sequence-caused. Two halves:
      (a) task-0 deployed final-row accuracy falls >= 0.15 below its own ceiling
      (b) the repeat-task control (identical data, fresh heads) forgets <= 0.05
      (a) is read here; (b) needs its own runs and is reported when they land.
      Half (a) alone cannot separate forgetting from optimization drift, which
      is the whole reason (b) exists (E5b's lesson, ported).

Usage:
    python scripts/e12_preconditions.py
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

SEEDS = [42, 1337, 2024]
P1_MARGIN = 0.05        # trainable trunk within 5pp of the frozen reference
P2_DROP_BAR = 0.15      # task-0 deployed drop
P2_CTRL_BAR = 0.05      # repeat-task control


def load_matrix(path):
    return np.array(json.load(open(path))["accuracy_matrix"], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="E12 preconditions P1/P2")
    ap.add_argument("--probe", default="runs/e12p_base_seed{s}/mafc_results.json")
    ap.add_argument("--frozen", default="runs/e12_frozen_probe.json")
    ap.add_argument("--control", default="runs/e12rpt_seed{s}/mafc_results.json")
    ap.add_argument("--out", default="runs/e12_preconditions.json")
    args = ap.parse_args()

    print("=" * 96)
    print("E12 PRECONDITIONS — P1 (competence) and P2 (forgetting exists, "
          "sequence-caused)")
    print("=" * 96)

    # ---- reference ---------------------------------------------------------
    if not Path(args.frozen).exists():
        print(f"  frozen reference missing: {args.frozen} — P1 cannot be read")
        return 1
    fz = json.load(open(args.frozen))
    ref = fz["mean_diag"]
    print(f"\n  frozen-probe reference: mean per-task DIAG {ref:.4f} "
          f"(min {fz['min_diag']:.4f}, forgetting 0.0000 by construction)")

    # ---- P1 ----------------------------------------------------------------
    print("\n" + "-" * 96)
    print(f"P1 — trainable-trunk per-task DIAG within {P1_MARGIN:.2f} of {ref:.4f}")
    print("-" * 96)
    rows, diags_all = [], []
    for s in SEEDS:
        p = args.probe.format(s=s)
        if not Path(p).exists():
            print(f"  seed {s}: MISSING {p}")
            continue
        M = load_matrix(p)
        diag = np.array([M[k, k] for k in range(M.shape[0])])
        diags_all.append(diag)
        rows.append({"seed": s, "diag_mean": float(diag.mean()),
                     "diag_min": float(diag.min()), "diag": diag.tolist()})
        print(f"  seed {s}: DIAG mean {diag.mean():.4f}  min {diag.min():.4f}  "
              f"max {diag.max():.4f}")
    if not rows:
        return 1
    diag_mean = float(np.mean([r["diag_mean"] for r in rows]))
    gap = ref - diag_mean
    p1 = abs(gap) <= P1_MARGIN
    print(f"\n  trainable mean DIAG {diag_mean:.4f} vs frozen {ref:.4f} -> "
          f"gap {gap:+.4f}  ({'within' if p1 else 'OUTSIDE'} {P1_MARGIN:.2f})")
    print(f"  P1: {'PASS' if p1 else 'FAIL'}")
    # The margin is itself a result, not just a gate.
    print(f"\n  COST/BENEFIT OF TRAINABILITY, per task: fine-tuning "
          f"{'beats' if gap < 0 else 'trails'} the frozen probe by "
          f"{abs(gap):.4f} on DIAG.\n  That margin is the numerator of the "
          f"cost-of-trainability sentence the frozen arm exists to price.")

    # ---- P2(a) -------------------------------------------------------------
    print("\n" + "-" * 96)
    print(f"P2(a) — task-0 deployed drop >= {P2_DROP_BAR}")
    print("-" * 96)
    drops = []
    for r, s in zip(rows, [r["seed"] for r in rows]):
        M = load_matrix(args.probe.format(s=s))
        ceil0, final0 = float(M[0, 0]), float(M[-1, 0])
        drops.append(ceil0 - final0)
        r.update({"task0_ceiling": ceil0, "task0_final": final0,
                  "task0_drop": ceil0 - final0})
        print(f"  seed {s}: task-0 ceiling {ceil0:.4f} -> final {final0:.4f}  "
              f"drop {ceil0-final0:+.4f}")
    drop_mean = float(np.mean(drops))
    p2a = drop_mean >= P2_DROP_BAR
    print(f"\n  mean task-0 drop {drop_mean:.4f} vs bar {P2_DROP_BAR} -> "
          f"{'PASS' if p2a else 'FAIL'}")

    # ---- P2(b) -------------------------------------------------------------
    print("\n" + "-" * 96)
    print(f"P2(b) — repeat-task control forgetting <= {P2_CTRL_BAR}")
    print("-" * 96)
    ctrl = []
    for s in SEEDS:
        p = args.control.format(s=s)
        if Path(p).exists():
            f = float(json.load(open(p))["forgetting"])
            ctrl.append(f)
            print(f"  seed {s}: repeat-task forgetting {f:+.4f}")
    if ctrl:
        c_mean = float(np.mean(ctrl))
        p2b = c_mean <= P2_CTRL_BAR
        print(f"\n  mean {c_mean:.4f} vs bar {P2_CTRL_BAR} -> "
              f"{'PASS' if p2b else 'FAIL'}")
    else:
        p2b = None
        print("  NOT YET RUN — P2 is INCOMPLETE. Half (a) alone cannot separate")
        print("  task-sequence forgetting from optimization drift, which is the")
        print("  entire reason this control exists. No decomposition is read.")

    # ---- verdict, from the values above ------------------------------------
    print("\n" + "=" * 96)
    ok = p1 and p2a and (p2b is True)
    print(f"  P1 {'PASS' if p1 else 'FAIL'} · P2(a) {'PASS' if p2a else 'FAIL'} · "
          f"P2(b) {'PASS' if p2b else ('FAIL' if p2b is False else 'PENDING')}")
    print(f"  PRECONDITIONS: {'CLEARED — decomposition may be read' if ok else 'HELD'}")
    json.dump({"reference": ref, "p1": bool(p1), "p1_gap": gap,
               "p2a": bool(p2a), "p2a_drop": drop_mean,
               "p2b": p2b, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
