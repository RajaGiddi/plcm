"""E29 downstream — does an ESTIMATED re-layout work as well as a known one?

Three deployed accuracies per (task, target), on B0's final checkpoint:

  no repair     the head reads h-transformed input directly
  exact         h^-1 applied, i.e. the map is known -- the ceiling this program
                has been quoting since E8
  estimated     R-hat (or P-hat) estimated from UNLABELLED query windows against
                the task's own stored statistics, then inverted and applied

Both families are run, because they are the two E29 measures and they fail
differently: permutations are recovered exactly or not at all, rotations carry
a continuous transfer bias.

THE REFERENCE IS THE TASK'S OWN TRAIN WINDOWS and the query is its TEST
windows, which are SUBJECT-DISJOINT from them by construction
(har_subject.py: five train subjects, one held-out test subject per group). So
the transfer bias Part B measures is inside this number rather than assumed
away -- this is the deployment case, where the stored statistics come from the
population the device was fitted on and the query comes from whoever is
wearing it now.
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
from scripts.e29_common import (NUM_TASKS, N_CHANNELS, MIN_RESTARTS, PermSearch,  # noqa: E402
                                window_stats, apply_perm, blockdiag, geodesic_deg,
                                fit_rotation)

TASKS_PERM = (3, 4)                # frames carrying a permutation
TASKS_ROT = (2, 4)                 # frames carrying a rotation
HAR_PARTITION = "1104af185c87"


def acc(m, x, y, k, device):
    _, l = features_and_logits(m, x, y, k, False, device)
    return float((l.argmax(1) == y).float().mean())


def main():
    ap = argparse.ArgumentParser(description="E29 downstream: estimated vs exact re-layout")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-start", type=int, required=True,
                    help=f"SO(3) restarts; required, no default, >= {MIN_RESTARTS}")
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--heldout", default="runs/e28/heldout.json")
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.n_start < MIN_RESTARTS:
        raise SystemExit(f"--n-start {args.n_start} < {MIN_RESTARTS}")
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    t0 = time.time()
    print("=" * 104 + f"\nE29 DOWNSTREAM  B0 seed {args.seed}  n_query={args.n_query}\n" + "=" * 104)

    run = json.load(open(f"{root}/e10off_ec_seed{args.seed}/mafc_results.json"))
    arm = run["arm"]
    assert arm["benchmark"] == "har_subject" and arm["era_checkpoints"], arm
    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=BATCH)
    assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
    data = load_task_data(bench, n_tasks=NUM_TASKS, probe_subset_seed=args.probe_seed)
    ep = int(arm.get("epochs_per_task", 10)) - 1
    m = load(f"{root}/ckpt_e10off_ec_seed{args.seed}/mafc_seed{args.seed}_fp32",
             NUM_TASKS - 1, epoch=ep)
    ho = json.load(open(args.heldout))
    H = [tuple(p) for p in ho["heldout"]][:6]      # six held-out swaps is enough per task
    rng = np.random.default_rng(20260922 + args.seed)

    rows = []
    # ---------------- permutations --------------------------------------------
    print("\n  PERMUTATIONS (reference = the task's own train windows)")
    for k in TASKS_PERM:
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m, xte, k, False, device, label=f"e29ds/s{args.seed}/T{k}",
                                 verbose=False)
        assert g["p_b"], (args.seed, k)
        clean = acc(m, xte, yte, k, device)
        search = PermSearch(*window_stats(xtr.numpy()))
        for h in H:
            xs = torch.from_numpy(apply_perm(xte.numpy(), h))
            qi = rng.choice(len(xs), min(args.n_query, len(xs)), replace=False)
            r = search.search(xs.numpy()[qi], h)
            inv = np.argsort(np.asarray(r["best"]))
            est = acc(m, torch.from_numpy(apply_perm(xs.numpy(), tuple(int(v) for v in inv))),
                      yte, k, device)
            exact = acc(m, torch.from_numpy(
                apply_perm(xs.numpy(), tuple(int(v) for v in np.argsort(np.asarray(h))))),
                yte, k, device)
            rows.append({"family": "perm", "task": k, "h": list(h), "clean": clean,
                         "no_repair": acc(m, xs, yte, k, device), "exact": exact,
                         "estimated": est, "recovered_exactly": r["exact"],
                         "true_rank": r["true_rank"], "est_minus_exact": est - exact})
        pr = [r for r in rows if r["family"] == "perm" and r["task"] == k]
        print(f"    task {k}: clean {clean:.4f} | no repair {np.mean([r['no_repair'] for r in pr]):.4f}"
              f" | exact {np.mean([r['exact'] for r in pr]):.4f}"
              f" | ESTIMATED {np.mean([r['estimated'] for r in pr]):.4f}"
              f" | recovered {sum(r['recovered_exactly'] for r in pr)}/{len(pr)}")

    # ---------------- rotations ------------------------------------------------
    print("\n  ROTATIONS")
    for k in TASKS_ROT:
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m, xte, k, False, device, label=f"e29ds/s{args.seed}/R{k}",
                                 verbose=False)
        assert g["p_b"], (args.seed, k)
        clean = acc(m, xte, yte, k, device)
        S_ref, C_ref = window_stats(xtr.numpy())
        for i in range(6):
            R = Rot.random(random_state=200 + i).as_matrix()
            B = blockdiag(R)
            xs = xte.numpy() @ B.T.astype(np.float32)
            qi = rng.choice(len(xs), min(args.n_query, len(xs)), replace=False)
            Rh, resid = fit_rotation(S_ref, C_ref, *window_stats(xs[qi]),
                                     n_start=args.n_start, seed=args.seed)
            est = acc(m, torch.from_numpy((xs @ blockdiag(Rh.T).T).astype(np.float32)),
                      yte, k, device)
            exact = acc(m, torch.from_numpy((xs @ blockdiag(R.T).T).astype(np.float32)),
                        yte, k, device)
            rows.append({"family": "rot", "task": k, "clean": clean,
                         "no_repair": acc(m, torch.from_numpy(xs.astype(np.float32)), yte, k, device),
                         "exact": exact, "estimated": est,
                         "angle_error_deg": geodesic_deg(R, Rh), "residual": resid,
                         "est_minus_exact": est - exact})
        rr = [r for r in rows if r["family"] == "rot" and r["task"] == k]
        print(f"    task {k}: clean {clean:.4f} | no repair {np.mean([r['no_repair'] for r in rr]):.4f}"
              f" | exact {np.mean([r['exact'] for r in rr]):.4f}"
              f" | ESTIMATED {np.mean([r['estimated'] for r in rr]):.4f}"
              f" | median angle err {np.median([r['angle_error_deg'] for r in rr]):.2f}deg")

    summ = {}
    for fam in ("perm", "rot"):
        f = [r for r in rows if r["family"] == fam]
        summ[fam] = {"clean": float(np.mean([r["clean"] for r in f])),
                     "no_repair": float(np.mean([r["no_repair"] for r in f])),
                     "exact": float(np.mean([r["exact"] for r in f])),
                     "estimated": float(np.mean([r["estimated"] for r in f])),
                     "est_minus_exact": float(np.mean([r["est_minus_exact"] for r in f])),
                     "within_2pp_of_exact": bool(abs(np.mean([r["est_minus_exact"] for r in f])) <= 0.02)}
        print(f"\n  {fam.upper()}: estimated - exact = {summ[fam]['est_minus_exact']:+.4f} -> "
              f"{'within 2pp' if summ[fam]['within_2pp_of_exact'] else 'NOT within 2pp'}")
    out = {"seed": args.seed, "n_query": args.n_query, "n_start": args.n_start,
           "heldout_fingerprint": ho["fingerprint"], "arm": arm,
           "partition_fingerprint": bench.partition_fingerprint(),
           "summary": summ, "rows": rows, "seconds": round(time.time() - t0, 1)}
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"downstream_seed{args.seed}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"\n  wrote {fp}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
