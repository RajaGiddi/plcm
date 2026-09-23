"""E29 Part A — what the query pool has to contain.

§0 answered v1's question before signing: with a MIXED-activity pool, exact
recovery of all 22 targets is 100% at n >= 64 across disjoint subjects, and a
SINGLE-activity pool of n = 256 recovers nothing. So the open question is not
"does statistical recovery work" but what a pool must contain for it to, which
is the number a deployment needs: a device after a format change sees whatever
its user happens to be doing.

AGGREGATION, stated in the same place as the bar (E28's lesson, adopted as a
standing rule). A TARGET recovers if it is exact in at least 9 of its 10 draws.
A POOL succeeds if at least 20 of the 22 targets recover. The pooled exact rate
is printed BESIDE that verdict and never in place of it: a pooled rate can read
90% while two targets never recover at all (catch 35(b)).

THE FRAME. Every pool is built through `HARSubjectBenchmark(no_shift=True)`,
the benchmark's own control path, where every group is standardized with task-0
constants and NOT mapped. So group q's windows arrive already in the reference
frame and the ONLY unknown is the permutation h this script applies. That
avoids inverting `channel_affine` here, which would be a parallel
implementation of the map (catch 32) in the one place it is easiest to get
backwards.

THE TWO CONTIGUOUS ARMS EXIST BECAUSE CONSECUTIVE WINDOWS OVERLAP BY HALF.
Verified before signing: window k's second half is bitwise identical to window
k+1's first half, and the lag-1 correlation of window means is 0.584. So "the
first n windows" carries far fewer than n independent observations, and a
stride-1 arm compared against random draws would vary time-order and effective
sample size at once. Stride 2 shares no samples: stride2-vs-random isolates
time order, stride1-vs-stride2 isolates the overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e29_common import (                                          # noqa: E402
    N_CHANNELS, NUM_TASKS, DRAWS_PER_TARGET, TARGET_RECOVERS_AT, POOL_SUCCEEDS_AT,
    PermSearch, window_stats, apply_perm, pool_indices, contiguous_indices,
    assert_contiguous, synthetic_pool, fingerprint)

SPECS = ["two_dynamic", "static_plus_dynamic", "three_static", "all6", "all6_skew80"]
N_GRID = [16, 64, 256, 1024]


def effective_rank(S: np.ndarray) -> float:
    """Participation ratio of the eigenvalues: (sum l)^2 / sum l^2. A pool whose
    covariance is near rank-1 cannot pin nine channels, and this says so."""
    l = np.linalg.eigvalsh(S).clip(0)
    return float(l.sum() ** 2 / max((l ** 2).sum(), 1e-30))


def score_targets(search, X, y, targets, spec, n, rng, subject_rows=None, stride=None):
    """One pool spec at one n: every target, every draw. Returns per-target rows."""
    rows = []
    for name, h in targets:
        ex, ch, rk = [], [], []
        for d in range(DRAWS_PER_TARGET):
            if stride is not None:
                # contiguous: vary the START, not the membership
                starts = max(len(subject_rows) - n * stride, 0)
                idx = contiguous_indices(subject_rows, n, stride,
                                         start=int(rng.integers(0, starts + 1)))
                if idx is None:
                    return None
                assert assert_contiguous(idx, stride), (spec, stride)
            else:
                idx = pool_indices(y, spec, n, rng)
                if idx is None:
                    return None
            r = search.search(apply_perm(X[idx], h), h)
            ex.append(r["exact"]); ch.append(r["channels_correct"]); rk.append(r["true_rank"])
        rows.append({"target": name, "h": list(h), "n_exact": int(sum(ex)),
                     "recovers": bool(sum(ex) >= TARGET_RECOVERS_AT),
                     "mean_channels": float(np.mean(ch)),
                     "median_true_rank": float(np.median(rk)), "max_true_rank": int(max(rk))})
    return rows


def verdict(rows):
    rec = sum(r["recovers"] for r in rows)
    pooled = float(np.mean([r["n_exact"] for r in rows]) / DRAWS_PER_TARGET)
    return {"targets_recovering": rec, "n_targets": len(rows),
            "succeeds": bool(rec >= POOL_SUCCEEDS_AT), "pooled_exact_rate": pooled,
            "targets_never_recovering": [r["target"] for r in rows if r["n_exact"] == 0]}


def main():
    ap = argparse.ArgumentParser(description="E29 Part A: pool composition")
    ap.add_argument("--heldout", default="runs/e28/heldout.json")
    ap.add_argument("--ref-group", type=int, default=0)
    ap.add_argument("--query-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="runs/e29/pools.json")
    ap.add_argument("--controls-out", default="runs/e29/controls.json")
    ap.add_argument("--smoke", action="store_true",
                    help="exercise every branch on a reduced grid; not a measurement")
    args = ap.parse_args()
    global N_GRID, DRAWS_PER_TARGET, TARGET_RECOVERS_AT, POOL_SUCCEEDS_AT
    if args.smoke:
        # A smoke exercises every BRANCH; it is not a measurement and its
        # verdicts are not read. The bars are scaled with the draw count so the
        # aggregation code runs, rather than passing trivially or never firing.
        N_GRID = [16, 64]
        DRAWS_PER_TARGET, TARGET_RECOVERS_AT, POOL_SUCCEEDS_AT = 2, 2, 3
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print("=" * 108 + "\nE29 PART A — what the query pool has to contain\n" + "=" * 108)

    from src.data.har_subject import HARSubjectBenchmark
    bench = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128, no_shift=True)
    fp = bench.partition_fingerprint()
    # `no_shift` is inside the fingerprint blob, so the control path hashes
    # differently from the shifted benchmark the checkpoints were trained on.
    # Record BOTH, or this artifact cannot be tied to the 1104af185c87 the
    # downstream checkpoints carry (the E16 construction-drift catch, one level
    # down: a fingerprint that is right and names a different construction).
    fp_shifted = HARSubjectBenchmark(num_tasks=NUM_TASKS, root=".", batch_size=128,
                                     no_shift=False).partition_fingerprint()
    ref_subs, qry_subs = bench.train_subjects[args.ref_group], bench.train_subjects[args.query_group]
    assert not (set(ref_subs) & set(qry_subs)), "reference and query pools share a subject"
    print(f"  partition {fp} (no_shift control path); shifted benchmark hashes {fp_shifted} "
          f"(Modal records 1104af185c87)\n  reference subjects {ref_subs} | "
          f"query subjects {qry_subs} | DISJOINT")

    Xr, _ = bench.raw_windows(args.ref_group, True)
    Xq, yq = bench.raw_windows(args.query_group, True)
    Xr, Xq, yq = Xr.numpy(), Xq.numpy(), yq.numpy()
    S_ref, C_ref = window_stats(Xr)
    t1 = time.time(); search = PermSearch(S_ref, C_ref)
    print(f"  9! precompute {time.time()-t1:.1f}s  |  reference n={len(Xr)}  query n={len(Xq)}")

    ho = json.load(open(args.heldout))
    H = [tuple(p) for p in ho["heldout"]]
    targets = ([("identity", tuple(range(N_CHANNELS)))]
               + [("PERM", (2, 0, 1, 5, 3, 4, 7, 8, 6))]
               + [(f"h{i}", h) for i, h in enumerate(H)])
    if args.smoke:
        targets = targets[:4]
    print(f"  targets {len(targets)} (identity + PERM + {len(H)} held-out, fp {ho['fingerprint']})")
    print(f"  AGGREGATION: target recovers at >={TARGET_RECOVERS_AT}/{DRAWS_PER_TARGET} draws; "
          f"pool succeeds at >={POOL_SUCCEEDS_AT}/{len(targets)} targets\n")

    # ---------------- controls, first, before any measurement ------------------
    print("  CONTROLS")
    ctl = {}
    idn = tuple(range(N_CHANNELS))
    # (1) plumbing: identity recovered exactly at the largest n
    big = pool_indices(yq, "all6", 1024, np.random.default_rng(1))
    r = search.search(Xq[big], idn)
    ctl["plumbing_identity"] = {"exact": r["exact"], "true_rank": r["true_rank"], "pass": r["exact"]}
    print(f"    plumbing (h=identity, n=1024): exact={r['exact']} rank={r['true_rank']} "
          f"-> {'PASS' if r['exact'] else 'FAIL'}")
    # (2) positive synthetic: exact at every n >= 64
    ps = {}
    for n in (64, 256, 1024):
        A = synthetic_pool(n, np.random.default_rng(2))
        sy = PermSearch(*window_stats(synthetic_pool(4096, np.random.default_rng(2))))
        h = tuple(np.random.default_rng(3).permutation(N_CHANNELS).tolist())
        ps[n] = sy.search(apply_perm(A, h), h)["exact"]
    ctl["positive_synthetic"] = {"exact_by_n": ps, "pass": all(ps.values())}
    print(f"    positive synthetic: exact at n>=64 {ps} -> {'PASS' if all(ps.values()) else 'FAIL'}")
    # (3) must-fail: a noise reference must not recover anything
    noise_ref = PermSearch(*window_stats(synthetic_pool(4096, np.random.default_rng(4))))
    hits = 0
    for name, h in targets[:8]:
        idx = pool_indices(yq, "all6", 256, np.random.default_rng(5))
        hits += noise_ref.search(apply_perm(Xq[idx], h), h)["exact"]
    ctl["must_fail_noise_reference"] = {"exact_hits": int(hits), "of": 8,
                                        "chance_rate": 1 / 362880, "pass": hits == 0}
    print(f"    must-fail (noise reference): {hits}/8 exact, chance 1/362880 -> "
          f"{'PASS' if hits == 0 else 'FAIL'}")
    ctl["subject_disjoint"] = {"ref": ref_subs, "query": qry_subs, "pass": True}
    ctl["partition_fingerprint"] = fp
    ctl["partition_fingerprint_shifted"] = fp_shifted
    os.makedirs(os.path.dirname(args.controls_out) or ".", exist_ok=True)
    json.dump(ctl, open(args.controls_out, "w"), indent=2, default=float)
    if not all(v.get("pass", True) for v in ctl.values() if isinstance(v, dict)):
        print("\n  CONTROLS FAILED — not reporting measurements"); return 1

    # ---------------- per-activity information content --------------------------
    print("\n  PER-ACTIVITY EFFECTIVE RANK (participation ratio of the covariance)")
    er = {}
    for c in range(6):
        pool = np.where(yq == c)[0]
        er[c] = effective_rank(window_stats(Xq[pool])[0]) if len(pool) > 8 else None
        print(f"    activity {c}: n={len(pool):>5}  effective rank "
              f"{er[c]:.2f}" if er[c] else f"    activity {c}: too few")
    er["all6"] = effective_rank(S_ref)
    print(f"    reference (all six): {er['all6']:.2f} of {N_CHANNELS}")

    # ---------------- the grid ---------------------------------------------------
    out = {"partition_fingerprint": fp, "partition_fingerprint_shifted": fp_shifted,
           "heldout_fingerprint": ho["fingerprint"],
           "ref_group": args.ref_group, "query_group": args.query_group,
           "ref_subjects": ref_subs, "query_subjects": qry_subs,
           "aggregation": {"draws_per_target": DRAWS_PER_TARGET,
                           "target_recovers_at": TARGET_RECOVERS_AT,
                           "pool_succeeds_at": POOL_SUCCEEDS_AT, "n_targets": len(targets)},
           "effective_rank": er, "controls": ctl, "pools": {}}

    print(f"\n  {'pool':<22}{'n':>6}{'succeeds':>10}{'targets rec':>13}{'pooled exact':>14}"
          f"{'mean chans':>12}{'never rec':>11}")
    for spec in SPECS:
        for n in N_GRID:
            rows = score_targets(search, Xq, yq, targets, spec, n, rng)
            if rows is None:
                print(f"  {spec:<22}{n:>6}{'unbuildable at this n':>48}")
                out["pools"][f"{spec}_n{n}"] = {"unbuildable": True}; continue
            v = verdict(rows); v["rows"] = rows; v["spec"], v["n"] = spec, n
            out["pools"][f"{spec}_n{n}"] = v
            print(f"  {spec:<22}{n:>6}{('YES' if v['succeeds'] else 'no'):>10}"
                  f"{v['targets_recovering']:>9}/{len(targets)}{v['pooled_exact_rate']:>14.3f}"
                  f"{np.mean([r['mean_channels'] for r in rows]):>12.2f}"
                  f"{len(v['targets_never_recovering']):>11}")

    # contiguous arms, per subject, both strides
    subj = qry_subs[0]
    rows_of_subj = np.where(bench._s == subj)[0]
    print(f"\n  CONTIGUOUS ARMS (subject {subj}, {len(rows_of_subj)} windows in recording order)")
    # map absolute rows to positions inside the query pool
    qmask = np.isin(bench._s, qry_subs)
    qpos = {int(a): i for i, a in enumerate(np.where(qmask)[0])}
    srows = np.array([qpos[int(a)] for a in rows_of_subj])
    for stride in (1, 2):
        for n in N_GRID:
            rows = score_targets(search, Xq, yq, targets, f"contig_s{stride}", n, rng,
                                 subject_rows=srows, stride=stride)
            key = f"contiguous_stride{stride}_n{n}"
            if rows is None:
                print(f"    stride {stride}, n={n:<5} unbuildable (subject has too few windows)")
                out["pools"][key] = {"unbuildable": True}; continue
            v = verdict(rows); v["rows"] = rows; v["spec"], v["n"], v["stride"] = key, n, stride
            out["pools"][key] = v
            print(f"    stride {stride}, n={n:<5} succeeds={'YES' if v['succeeds'] else 'no':<4} "
                  f"targets {v['targets_recovering']}/{len(targets)}  "
                  f"pooled {v['pooled_exact_rate']:.3f}")

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
