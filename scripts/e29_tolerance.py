"""E29 Part B.3 — the tolerance curve: how much rotation error can the model absorb?

This is deliberately INDEPENDENT OF ANY ESTIMATOR. It perturbs the deployed
input by a rotation of known angle and reads accuracy, so it answers "how good
does an estimate have to be" without entangling that answer with how good this
particular estimator happens to be. Part B's bias distribution is then read
against this curve rather than against a bar invented for it.

CONSTRUCTION. Tasks 2 and 4 are the tasks whose frames contain the rotation.
For each theta in {0, 2, 5, 9, 15, 30} degrees, twenty random axes, the
blockwise rotation of that angle is applied to task-k's test input and the
deployed head is scored with NO repair of any kind.

theta = 0 IS THE PLUMBING CHECK and it must reproduce exact re-layout: at
theta = 0 the perturbation is the identity, so accuracy must equal the
unperturbed deployed accuracy to floating-point. A curve whose left endpoint
does not land on the clean number is measuring something other than what it
claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,    # noqa: E402
                                    assert_path_identity, PROBE_SUBSET_SEED)
from scripts.cure_screen import BATCH                                             # noqa: E402
from scripts.e29_common import NUM_TASKS, blockdiag, geodesic_deg                 # noqa: E402

THETAS = [0.0, 2.0, 5.0, 9.0, 15.0, 30.0]
N_AXES = 20
TASKS = (2, 4)                     # the frames that contain the rotation
HAR_PARTITION = "1104af185c87"


def rot_at_angle(theta_deg: float, axis: np.ndarray) -> np.ndarray:
    a = axis / max(np.linalg.norm(axis), 1e-12)
    return Rot.from_rotvec(np.deg2rad(theta_deg) * a).as_matrix()


def main():
    ap = argparse.ArgumentParser(description="E29 tolerance curve")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    t0 = time.time()
    print("=" * 104 + f"\nE29 TOLERANCE  B0 seed {args.seed}\n" + "=" * 104)

    run = json.load(open(f"{root}/e10off_ec_seed{args.seed}/mafc_results.json"))
    arm = run["arm"]
    assert arm["benchmark"] == "har_subject" and arm["era_checkpoints"], arm
    print(f"  arm: benchmark={arm['benchmark']} adapters={arm['use_input_adapters']} "
          f"heads={arm['use_task_heads']} epochs={arm.get('epochs_per_task')}")

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=BATCH)
    assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
    data = load_task_data(bench, n_tasks=NUM_TASKS, probe_subset_seed=args.probe_seed)
    ep = int(arm.get("epochs_per_task", 10)) - 1
    m = load(f"{root}/ckpt_e10off_ec_seed{args.seed}/mafc_seed{args.seed}_fp32",
             NUM_TASKS - 1, epoch=ep)

    rng = np.random.default_rng(20260922 + args.seed)
    rows = []
    for k in TASKS:
        _, _, xte, yte = data[k]
        g = assert_path_identity(m, xte, k, False, device, label=f"e29tol/s{args.seed}/T{k}",
                                 verbose=False)
        assert g["p_b"], (args.seed, k)
        _, l = features_and_logits(m, xte, yte, k, False, device)
        clean = float((l.argmax(1) == yte).float().mean())
        per_theta = {}
        for th in THETAS:
            accs = []
            for _ in range(N_AXES if th > 0 else 1):
                B = torch.from_numpy(
                    blockdiag(rot_at_angle(th, rng.normal(size=3))).astype(np.float32))
                _, lp = features_and_logits(m, xte @ B.T, yte, k, False, device)
                accs.append(float((lp.argmax(1) == yte).float().mean()))
            per_theta[th] = {"mean": float(np.mean(accs)), "min": float(np.min(accs)),
                             "max": float(np.max(accs)), "n_axes": len(accs),
                             "drop_from_clean": float(clean - np.mean(accs))}
        # PLUMBING: theta = 0 must reproduce the clean number exactly.
        z = per_theta[0.0]["mean"]
        ok0 = abs(z - clean) < 1e-6
        print(f"  task {k}: clean {clean:.4f} | theta=0 {z:.4f} -> "
              f"{'PLUMBING PASS' if ok0 else 'PLUMBING FAIL (the curve is not the deployed path)'}")
        if not ok0:
            return 1
        for th in THETAS[1:]:
            p = per_theta[th]
            print(f"      theta {th:>5.1f}deg: acc {p['mean']:.4f} "
                  f"(min {p['min']:.4f}, max {p['max']:.4f})  drop {p['drop_from_clean']:+.4f}")
        rows.append({"task": k, "clean": clean, "plumbing_pass": ok0,
                     "n_test": int(len(yte)), "by_theta": per_theta})

    # the practical summary: the largest theta whose mean drop stays within 2pp
    tol = {}
    for r in rows:
        within = [th for th in THETAS if r["by_theta"][th]["drop_from_clean"] <= 0.02]
        tol[r["task"]] = max(within) if within else None
    out = {"seed": args.seed, "tasks": list(TASKS), "thetas": THETAS, "n_axes": N_AXES,
           "partition_fingerprint": bench.partition_fingerprint(),
           "arm": arm, "rows": rows, "largest_theta_within_2pp": tol,
           "seconds": round(time.time() - t0, 1)}
    print(f"\n  largest theta whose mean drop stays within 2pp: {tol}")
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"tolerance_seed{args.seed}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"  wrote {fp}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
