"""
E12/B4 — the blast-radius regression: did either amendment spend E11's proof?

Two orthogonal amendments touch `PLCM`, the component every experiment shares:

  A1  `backbone="vit"`            — a native-readout ViT path
  A2  `head_routing_by_hint=True` — task_hint selects the head, not just the adapter

Both default OFF. This asserts that with them off, the deployed path is
UNCHANGED — and "unchanged" is not a code review, it is E11's proof re-run:
`PLCM.forward` reproduces each recorded accuracy matrix EXACTLY (E11 measured
0.0000 across 8/8 cells on both arms). If an amendment perturbs the shared path,
that reproduction breaks, and this script says which arm broke it.

TWO ARMS, SEPARATELY, BY RULING — E10 (HAR path) and E4 (MNIST path) — because a
combined check cannot say which amendment or which path is at fault.

ENVIRONMENT: the HAR arm must run on Modal. The E10 partition is
architecture-dependent (subjects 2/5 tie-break differently on ARM vs x86), which
changes the calibration constants and so every task's test tensors; only the
execution environment reproduces the data the checkpoints were trained on. The
MNIST arm runs anywhere — those checkpoints were trained on the laptop.

Usage:
    python scripts/e12_regression.py --arms mnist
    modal run modal_runner.py::analysis --argv "scripts/e12_regression.py --arms har"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import load
from scripts.three_column import build_bench, deployed_full

TASKS = [0, 1, 2, 3]
ARMS = {
    "mnist": [("E4/OFF", {s: f"checkpoints/e4_off_seed{s}/mafc_seed{s}" for s in (42,)},
               {42: "runs/e4_off_seed42/mafc_results.json"}, "mnist"),
              ("E4/ON", {42: "checkpoints/fullrank_ref/mafc_seed42"},
               {42: "runs/fullrank_ref/mafc_results.json"}, "mnist")],
    "har": [("E10/OFF", {42: "runs/ckpt_e10_off_seed42/mafc_seed42"},
             {42: "runs/e10_off_seed42/mafc_results.json"}, "har_subject"),
            ("E10/ON", {42: "runs/ckpt_e10_on_seed42/mafc_seed42"},
             {42: "runs/e10_on_seed42/mafc_results.json"}, "har_subject")],
}


def check(arm, ckpt, results, bench_key, device, seed=42):
    """Re-run E11's exactness proof on this arm with both flags at default."""
    d, rj = ckpt[seed], results[seed]
    if not (Path(d).exists() and Path(rj).exists()):
        return {"arm": arm, "status": "MISSING", "detail": d if not Path(d).exists() else rj}

    model = load(d, 4)
    # The flags must be OFF by default — assert the default, do not set it.
    assert model.backbone != "vit", "backbone default is not 'lstm'"
    assert getattr(model, "head_routing_by_hint", False) is False, (
        "head_routing_by_hint defaults to True — the amendment is not opt-in")

    mat = np.array(json.load(open(rj))["accuracy_matrix"], dtype=float)
    bench = build_bench(bench_key, seed, 128)
    worst, cells = 0.0, []
    for k in TASKS:
        _, te = bench.get_task_loaders(k)
        acc, _ = deployed_full(model, te, k, device)
        gap = abs(acc - float(mat[4, k]))
        worst = max(worst, gap)
        cells.append({"task": k, "matrix": float(mat[4, k]), "rerun": acc, "gap": gap})
    return {"arm": arm, "status": "EXACT" if worst == 0.0 else "PERTURBED",
            "max_gap": worst, "cells": cells}


def main():
    ap = argparse.ArgumentParser(description="E12 blast-radius regression")
    ap.add_argument("--arms", choices=["mnist", "har", "both"], default="mnist")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    keys = ["mnist", "har"] if args.arms == "both" else [args.arms]
    print("=" * 96)
    print("E12/B4 REGRESSION — with both amendments OFF, is the deployed path "
          "still E11-exact?")
    print("=" * 96)
    print("  amendments: A1 backbone='vit' (default off) · "
          "A2 head_routing_by_hint (default off)\n")

    rows = []
    for key in keys:
        for arm, ckpt, results, bench_key in ARMS[key]:
            r = check(arm, ckpt, results, bench_key, device)
            rows.append(r)
            if r["status"] == "MISSING":
                print(f"  {arm:<10} MISSING  {r['detail']}")
                continue
            print(f"  {arm:<10} {r['status']:<10} max |rerun - matrix| = {r['max_gap']:.1e}")
            for c in r["cells"]:
                print(f"    {'':<8} T{c['task']}  matrix {c['matrix']:.4f}  "
                      f"rerun {c['rerun']:.4f}  gap {c['gap']:.1e}")

    ok = [r for r in rows if r["status"] == "EXACT"]
    bad = [r for r in rows if r["status"] == "PERTURBED"]
    print()
    print(f"  {len(ok)}/{len(rows)} arms bit-identical to E11's recorded matrices")
    if bad:
        print(f"  REGRESSION: {[r['arm'] for r in bad]} — an amendment altered the "
              f"shared path. Do not proceed to B5.")
    else:
        print("  No amendment spent E11's proof. B4's regression condition is met.")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return 0 if (rows and not bad and not any(r["status"] == "MISSING" for r in rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
