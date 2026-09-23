"""E23 arm B, P1 reference: a linear readout on the FLATTENED 128 x 9 windows, per
task -- the trivial use of arm B's resource (catch 24), the frozen-reference
analogue of E12/E14's frozen-trunk probe for a scratch MLP that has no
pretrained trunk. Same benchmark construction and probe draw as the S72 arm
(`har_subject`, x86 partition gated, `load_task_data` with seed 20260916),
`refit_probe` fit to optimality (no knobs). P1 as contracted: reference per-task
DIAG >= 0.85, and the trainable arm-B trunk within 5pp of the reference mean.

Usage: python scripts/e23_har_linear_ref.py --out /runs/e23/frozen_mlp.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import load_task_data, refit_probe, PROBE_SUBSET_SEED   # noqa: E402
from src.data.har_subject import HARSubjectBenchmark                                # noqa: E402

HAR_PARTITION = "1104af185c87"
P1_BAR = 0.85


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    args = ap.parse_args()
    har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
    pfp = har.partition_fingerprint()
    print(f"E23 ARM-B LINEAR REFERENCE  har_subject partition {pfp} -> {'AS-EXECUTED' if pfp == HAR_PARTITION else 'NOT the executed partition -- STOP'}")
    if pfp != HAR_PARTITION:
        return 1
    data = load_task_data(har, n_tasks=5, probe_subset_seed=args.probe_seed)
    rows = []
    for k in range(5):
        xtr, ytr, xte, yte = data[k]
        acc = refit_probe(xtr.reshape(len(xtr), -1), ytr, xte.reshape(len(xte), -1), yte, har.num_classes)
        rows.append({"task": k, "diag": acc, "n_train": int(len(ytr)), "n_test": int(len(yte)), "dim": int(xtr[0].numel())})
        print(f"  task {k}: linear-on-windows DIAG {acc:.4f}  (n_train {len(ytr)}, n_test {len(yte)}, dim {xtr[0].numel()})")
    diag = [r["diag"] for r in rows]
    out = {"arm": "e23_har_mlp reference", "benchmark": "har_subject", "partition": pfp, "probe_subset_seed": args.probe_seed, "reference": "linear readout on flattened 128x9 windows, refit_probe",
           "rows": rows, "mean_diag": float(np.mean(diag)), "min_diag": float(np.min(diag)), "per_seed_mean": [float(np.mean(diag))], "p1_bar": P1_BAR, "clears": bool(np.min(diag) >= P1_BAR)}
    print(f"  RESULT mean DIAG {out['mean_diag']:.4f} min {out['min_diag']:.4f} vs bar {P1_BAR} -> {'CLEARS' if out['clears'] else 'FAILS (the pairing is void as contracted; reported)'}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2); print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
