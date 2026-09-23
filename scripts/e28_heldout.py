"""E28 sec 1 + the sec 5 must-fail, on EXISTING checkpoints. No training.

Two jobs, in this order, because the second can void the pilot:

  1. Construct the held-out permutation set H (20 permutations, recorded seed,
     excluding the identity and every training map's permutation component) and
     fingerprint it.

  2. Measure B0's readout-repair loss under an unseen swap:

         Delta_refit = refit(clean features) - refit(h-swapped features)

     on `ckpt_e10off_ec` -- the S72 OFF checkpoints that are already B0. If a
     NORMALLY TRAINED encoder loses nothing when its channels are permuted,
     there is nothing for an equivariance-trained encoder to fix and the pilot
     is void. Running this before the trainer change is priced the same way
     catch 24 prices a trivial use: measure what the resource already gives you
     before building the method that assumes it does not.

THE TRAINING FAMILY'S PERMUTATION COMPONENT IS EXACTLY {identity, PERM}.
`SHIFT_SPEC` (har_shift.py:133-143) puts a "perm" key only on specs 3 and 4, and
both carry the same `PERM`. So "unseen" is exact set membership against two
elements, not a distance -- which is why the permutation family is the clean
first test and the continuous components are not.

SWAPS ACT IN STANDARDIZED SPACE. `har_subject._apply` (:126-129) standardizes
with task-0 statistics and THEN applies the task's channel map, so the tensor
this script receives is already standardized and mapped; permuting its channel
axis is a swap in standardized space, which is the space sec 1 specifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import BATCH                                               # noqa: E402
from src.data.har_shift import SHIFT_SPEC, N_CHANNELS, spec_fingerprint             # noqa: E402

HELDOUT_SEED = 20260922
N_HELDOUT = 20
HAR_PARTITION = "1104af185c87"
N_CLASSES, NUM_TASKS = 6, 5


def training_perm_components():
    """Every permutation the five training specs contain, as tuples."""
    out = {tuple(range(N_CHANNELS))}                      # identity: specs 0, 1, 2
    for k, spec in SHIFT_SPEC.items():
        if "perm" in spec:
            out.add(tuple(spec["perm"]))
    return out


def build_heldout(seed=HELDOUT_SEED, n=N_HELDOUT):
    excl = training_perm_components()
    rng = np.random.default_rng(seed)
    H, seen = [], set(excl)
    while len(H) < n:
        p = tuple(int(v) for v in rng.permutation(N_CHANNELS))
        if p in seen:
            continue
        seen.add(p); H.append(p)
    assert not (set(H) & excl), "held-out set intersects the training family"
    fp = hashlib.sha1(repr(sorted(H)).encode()).hexdigest()[:12]
    return H, fp, sorted(excl)


def apply_perm(x: torch.Tensor, p) -> torch.Tensor:
    """Channel permutation on [n, T, 9] in STANDARDIZED space. Output channel i
    holds input channel p[i], matching `channel_affine`'s P convention
    (P[i, j] = 1 means output i takes input j)."""
    return x[..., list(p)]


def main():
    ap = argparse.ArgumentParser(description="E28: held-out set and the B0 must-fail")
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--n-heldout", type=int, default=N_HELDOUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/e28/b0_mustfail.json")
    ap.add_argument("--heldout-out", default="runs/e28/heldout.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    seeds = [int(s) for s in args.seeds.split(",")]

    H, fp, excl = build_heldout(n=args.n_heldout)
    print("=" * 104 + "\nE28  held-out permutation set, and B0's must-fail on existing checkpoints\n" + "=" * 104)
    print(f"  training family's permutation components ({len(excl)}): " +
          ", ".join(str(list(e)) for e in excl))
    print(f"  held-out set: {len(H)} permutations, seed {HELDOUT_SEED}, fingerprint {fp}")
    print(f"  spec_fingerprint {spec_fingerprint()}  |  disjoint from training family: "
          f"{not (set(H) & set(excl))}")
    os.makedirs(os.path.dirname(args.heldout_out) or ".", exist_ok=True)
    json.dump({"seed": HELDOUT_SEED, "n": len(H), "fingerprint": fp,
               "heldout": [list(p) for p in H],
               "training_perm_components": [list(e) for e in excl],
               "spec_fingerprint": spec_fingerprint(),
               "space": "standardized (har_subject.py:126-129)",
               "convention": "output channel i holds input channel p[i], as channel_affine's P"},
              open(args.heldout_out, "w"), indent=2)
    print(f"  wrote {args.heldout_out}")

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=BATCH)
    assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
    data = load_task_data(bench, n_tasks=NUM_TASKS, probe_subset_seed=args.probe_seed)
    T = NUM_TASKS - 1

    rows, t0 = [], time.time()
    for s in seeds:
        d = f"{root}/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32"
        arm = json.load(open(f"{root}/e10off_ec_seed{s}/mafc_results.json"))["arm"]
        assert arm["benchmark"] == "har_subject" and arm["era_checkpoints"], arm
        m4 = load(d, T)
        # ALL FIVE TASKS, including the final one. Every prior experiment in this
        # program loops `range(T)` because it studies forgetting on OLD tasks, and
        # that convention was inherited here without checking that it fits. It does
        # not: E28's primary set is "tasks whose frame carries a permutation",
        # which is 3 AND 4, and task 4 is the task the deployed model is currently
        # doing -- the cell a post-deployment sensor swap actually lands in. There
        # is no forgetting on task 4 for theta_T, but Delta_refit is well defined
        # there and is the most practically relevant number in the pilot.
        for k in range(NUM_TASKS):
            xtr, ytr, xte, yte = data[k]
            g = assert_path_identity(m4, xte, k, False, device, label=f"s{s}/T{k}", verbose=False)
            assert g["p_b"], (s, k)
            f = lambda x, y: features_and_logits(m4, x, y, k, False, device)
            F_tr, _ = f(xtr, ytr); F_te, l_te = f(xte, yte)
            clean = float((l_te.argmax(1) == yte).float().mean())
            refit_clean = refit_probe(F_tr, ytr, F_te, yte, N_CLASSES)
            per_h = []
            for h in H:
                Fs_tr, _ = f(apply_perm(xtr, h), ytr)
                Fs_te, ls_te = f(apply_perm(xte, h), yte)
                # re-layout identity control: undo h, must reproduce clean exactly
                inv = np.argsort(np.asarray(h))
                _, lr_te = f(apply_perm(apply_perm(xte, h), tuple(int(v) for v in inv)), yte)
                per_h.append({
                    "h": list(h),
                    "swapped": float((ls_te.argmax(1) == yte).float().mean()),
                    "refit_swapped": refit_probe(Fs_tr, ytr, Fs_te, yte, N_CLASSES),
                    "relayout_identity_abs": abs(float((lr_te.argmax(1) == yte).float().mean()) - clean)})
            dref = [clean and (refit_clean - p["refit_swapped"]) for p in per_h]
            row = {"arm": "B0", "seed": s, "task": k, "n_test": int(len(yte)),
                   "clean": clean, "refit_clean": refit_clean,
                   "swapped_mean": float(np.mean([p["swapped"] for p in per_h])),
                   "refit_swapped_mean": float(np.mean([p["refit_swapped"] for p in per_h])),
                   "delta_refit_mean": float(np.mean(dref)),
                   "delta_refit_min": float(np.min(dref)), "delta_refit_max": float(np.max(dref)),
                   "relayout_identity_max": float(max(p["relayout_identity_abs"] for p in per_h)),
                   "res95": 1.96 * float(np.sqrt(0.25 / len(yte))), "per_h": per_h}
            rows.append(row)
            print(f"  s{s} k{k}: clean {clean:.4f} | swapped {row['swapped_mean']:.4f} "
                  f"(drop {clean - row['swapped_mean']:+.4f}) | refit {refit_clean:.4f} -> "
                  f"{row['refit_swapped_mean']:.4f} | DELTA_refit {row['delta_refit_mean']:+.4f} "
                  f"[{row['delta_refit_min']:+.4f},{row['delta_refit_max']:+.4f}] | relay-id "
                  f"{row['relayout_identity_max']:.1e}")

    dm = float(np.mean([r["delta_refit_mean"] for r in rows]))
    res = float(np.mean([r["res95"] for r in rows]))
    relay_ok = max(r["relayout_identity_max"] for r in rows) == 0.0
    out = {"heldout_fingerprint": fp, "n_heldout": len(H), "seeds": seeds,
           "cells": len(rows), "delta_refit_pooled": dm, "res95_mean": res,
           "must_fail_pass": bool(dm > res),
           "swap_drop_pooled": float(np.mean([r["clean"] - r["swapped_mean"] for r in rows])),
           "refit_clean_pooled": float(np.mean([r["refit_clean"] for r in rows])),
           "refit_swapped_pooled": float(np.mean([r["refit_swapped_mean"] for r in rows])),
           "relayout_identity_exact": relay_ok, "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    print(f"\n  POOLED  clean {np.mean([r['clean'] for r in rows]):.4f} -> swapped "
          f"{np.mean([r['swapped_mean'] for r in rows]):.4f} (no repair) | refit "
          f"{out['refit_clean_pooled']:.4f} -> {out['refit_swapped_pooled']:.4f}")
    print(f"  MUST-FAIL  B0 Delta_refit {dm:+.4f} vs resolution {res:.4f} -> "
          f"{'PASS -- there is something for the method to fix' if out['must_fail_pass'] else 'FAIL -- PILOT IS VOID'}")
    print(f"  re-layout identity exact on every cell: {relay_ok}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"  wrote {args.out}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
