"""E29 §0a — the pre-signing reads, re-measured on Modal's partition.

The four observations in §0a were made on this laptop, which builds
`partition_fingerprint` 5d047e4213d1 while Modal builds 1104af185c87. E29 is
entirely a statement about subject-pool statistics, so a different partition
means different groups and different cross-subject drift. The reads carry NO
odds -- they were observed before signing -- but they must be reproduced where
the checkpoints live, and the contract wrote a branch for each way the two
could disagree.

The construction matches §0's exactly: `no_shift=True` leaves every group
standardized with task-0 constants and unmapped, which is what §0's `unmap`
produced for a reference at task 0 and a query at group 2 or 3. So this is a
re-measurement, not a re-definition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e29_common import (NUM_TASKS, N_CHANNELS, MIN_RESTARTS, PermSearch,   # noqa: E402
                                window_stats, apply_perm, pool_indices, blockdiag,
                                geodesic_deg, fit_rotation)

LAPTOP = {"mixed_exact_n64": 1.00, "mixed_exact_n16": 0.91,
          "single_activity_exact_n256": 0.00, "matched_rotation_deg": 0.00,
          "disjoint_rotation_deg": 8.72}


def main():
    ap = argparse.ArgumentParser(description="E29 0a reads, re-measured")
    ap.add_argument("--n-start", type=int, required=True)
    ap.add_argument("--heldout", default="runs/e28/heldout.json")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="runs/e29/reads.json")
    args = ap.parse_args()
    if args.n_start < MIN_RESTARTS:
        raise SystemExit(f"--n-start {args.n_start} < {MIN_RESTARTS}")
    t0 = time.time()
    print("=" * 104 + "\nE29 §0a — the pre-signing reads, re-measured on this partition\n" + "=" * 104)

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128, no_shift=True)
    fp = bench.partition_fingerprint()
    fp_shifted = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128,
                                     no_shift=False).partition_fingerprint()
    on_modal = fp_shifted == "1104af185c87"
    print(f"  partition {fp} (no_shift) / {fp_shifted} (shifted) | "
          f"{'MODAL partition' if on_modal else 'NOT the Modal partition'}")

    Xr, _ = bench.raw_windows(0, True)
    Xq, yq = bench.raw_windows(3, True)
    Xr, Xq, yq = Xr.numpy(), Xq.numpy(), yq.numpy()
    S_ref, C_ref = window_stats(Xr)
    search = PermSearch(S_ref, C_ref)
    H = [tuple(p) for p in json.load(open(args.heldout))["heldout"]]
    targets = ([tuple(range(N_CHANNELS)), (2, 0, 1, 5, 3, 4, 7, 8, 6)] + H)
    rng = np.random.default_rng(args.seed)
    out = {"partition_fingerprint": fp, "partition_fingerprint_shifted": fp_shifted,
           "is_modal_partition": on_modal, "laptop_values": LAPTOP, "reads": {}}

    print("\n  READ 1: mixed-activity pool, subject-disjoint, all 22 targets")
    for n in (16, 64, 256, 1024):
        ex = []
        for h in targets:
            for _ in range(3):
                idx = pool_indices(yq, "all6", n, rng)
                ex.append(search.search(apply_perm(Xq[idx], h), h)["exact"])
        out["reads"][f"mixed_exact_n{n}"] = float(np.mean(ex))
        print(f"    n={n:>5}: exact recovery {np.mean(ex)*100:>5.1f}%")

    print("\n  READ 2: single-activity pools, n=256")
    sa = {}
    for c in range(6):
        pool = np.where(yq == c)[0]
        if len(pool) < 32:
            continue
        ex, ch = [], []
        for h in targets:
            idx = rng.choice(pool, min(256, len(pool)), replace=False)
            r = search.search(apply_perm(Xq[idx], h), h)
            ex.append(r["exact"]); ch.append(r["channels_correct"])
        sa[c] = {"exact": float(np.mean(ex)), "mean_channels": float(np.mean(ch))}
        print(f"    activity {c}: exact {np.mean(ex)*100:>5.1f}%  channels {np.mean(ch):.2f}/9")
    out["reads"]["single_activity"] = sa
    out["reads"]["single_activity_exact_n256"] = float(np.mean([v["exact"] for v in sa.values()]))

    print("\n  READ 3/4: rotation, matched vs disjoint subjects")
    R = Rot.random(random_state=101).as_matrix()
    e_match = geodesic_deg(R, fit_rotation(S_ref, C_ref, *window_stats(Xr @ blockdiag(R).T),
                                           n_start=args.n_start, seed=args.seed)[0])
    e_disj = [geodesic_deg(Rq, fit_rotation(S_ref, C_ref, *window_stats(Xq @ blockdiag(Rq).T),
                                            n_start=args.n_start, seed=args.seed)[0])
              for _, Rq in (("a", np.eye(3)), ("b", R))]
    out["reads"]["matched_rotation_deg"] = e_match
    out["reads"]["disjoint_rotation_deg"] = float(np.mean(e_disj))
    out["reads"]["disjoint_equivariance_spread"] = float(max(e_disj) - min(e_disj))
    print(f"    matched subjects : {e_match:.4f}deg  (laptop 0.00)")
    print(f"    disjoint subjects: {np.mean(e_disj):.4f}deg, equivariance spread "
          f"{max(e_disj)-min(e_disj):.2e}deg  (laptop 8.72)")

    # ---------------- the contract's branches, fired from the numbers ----------
    mixed_ok = out["reads"]["mixed_exact_n64"] >= 0.9
    single_recovers = out["reads"]["single_activity_exact_n256"] > 0.1
    if single_recovers:
        branch = ("MODAL DISAGREES (single-activity pools recover): the laptop result was "
                  "partition-specific. Part A's predictions are scored as registered and the "
                  "composition framing is reported as mis-scoped for this partition — a "
                  "partition-dependence finding, not a re-scoping after the fact.")
    elif not mixed_ok:
        branch = ("MODAL DISAGREES (mixed pools fail): Part A's premise does not hold where the "
                  "checkpoints were trained. Part A is void; Part B stands, since the tolerance "
                  "curve does not depend on the estimator.")
    else:
        branch = ("MODAL AGREES with the laptop on both directions: mixed pools recover, "
                  "single-activity pools do not. Part A proceeds as written.")
    out["branch"] = branch
    out["rotation_note"] = ("a different disjoint-subject angle is EXPECTED: the laptop value was "
                            "one pairing on another partition; Part B's distribution is the number "
                            "reported")
    print(f"\n  BRANCH: {branch}")
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"  wrote {args.out}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
