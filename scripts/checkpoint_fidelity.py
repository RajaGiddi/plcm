"""
E11 Step 3 — the residual: is a RELOADED checkpoint the deployed model?

Contract: docs/E11_instrument_correction.md (step 3)

Step 2 found PLCM.forward re-run on a reloaded checkpoint does not reproduce the
accuracy matrix it was written from -- systematically HIGH on OFF arms. Catch 19's
lesson is that a plausible story is not a diagnosis, so this measures rather than
argues.

CANDIDATE (diagnosed by content, not filename): PLCM.task_stats is a plain dict,
not a registered buffer. It is therefore absent from state_dict(), so
load_from_checkpoint() returns a model with task_stats == {} -- and forward()'s
coordinate-alignment block is gated on `self.task_stats` being non-empty. A
reloaded model skips an alignment the deployed model applied. Same weights,
different function.

THE TEST IS A REPAIR, NOT AN ARGUMENT. task_stats[k] is, by construction, the
running (mean, var) at the task-k -> k+1 boundary, which is exactly what
task{k}_epoch9.pt's running_mean / running_var buffers hold. Restore them from
the checkpoint chain and re-run:

  * if the gap closes  -> diagnosed, and every reloaded-checkpoint number in this
                          program was computed on a model missing this path
  * if it does not     -> the candidate is dead; report the residual as open

Usage:
    python scripts/checkpoint_fidelity.py --benchmark mnist --seeds 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import load
from scripts.three_column import ARMS, fmt, build_bench, deployed_full

TASKS = [0, 1, 2, 3]


def restore_task_stats(model, ckpt_dir: str, upto: int = 4, epoch: int = 9) -> int:
    """task_stats[k] <- the running (mean, var) saved at the end of task k.

    set_task(k+1) snapshots running_mean/var BEFORE any task-(k+1) update, so the
    end-of-task-k checkpoint holds exactly the snapshotted values. This is a
    reconstruction from recorded state, not an estimate.
    """
    n = 0
    for k in range(upto):
        p = Path(ckpt_dir) / f"task{k}_epoch{epoch}.pt"
        if not p.exists():
            continue
        sd = torch.load(p, weights_only=True, map_location="cpu")["model_state"]
        model.task_stats[k] = (sd["running_mean"].clone(), sd["running_var"].clone())
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="E11 step 3: checkpoint fidelity")
    ap.add_argument("--benchmark", default="mnist",
                    choices=["mnist", "har_shift", "har_subject"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 100)
    print("E11 STEP 3 — IS A RELOADED CHECKPOINT THE DEPLOYED MODEL?")
    print("=" * 100)

    # ---- 0. Is task_stats in the serialized state at all? --------------------
    any_arm = next(iter(ARMS[args.benchmark].values()))
    probe_dir = fmt(any_arm[0], seeds[0])
    sd = torch.load(Path(probe_dir) / "task4_epoch9.pt",
                    weights_only=True, map_location="cpu")["model_state"]
    stat_keys = [k for k in sd if "task_stat" in k]
    print(f"\n  state_dict keys matching 'task_stat': {stat_keys or 'NONE'}")
    print(f"  running_mean present: {'running_mean' in sd}   "
          f"running_var present: {'running_var' in sd}   "
          f"stats_initialized: {bool(sd.get('stats_initialized', False))}")

    rows = []
    for arm, (ckpt_t, use_ad, res_t) in ARMS[args.benchmark].items():
        for seed in seeds:
            d, rj = fmt(ckpt_t, seed), fmt(res_t, seed)
            if not (Path(d).exists() and Path(rj).exists()):
                print(f"  MISSING {arm}/seed{seed}")
                continue
            mat = np.array(json.load(open(rj))["accuracy_matrix"], dtype=float)
            bench = build_bench(args.benchmark, seed)

            m_bare = load(d, 4)
            m_fix = load(d, 4)
            n_restored = restore_task_stats(m_fix, d)
            print(f"\n  {arm}/seed{seed}: reloaded task_stats={len(m_bare.task_stats)}, "
                  f"restored={n_restored}, coord_align={m_bare.use_coord_align}, "
                  f"bank={m_bare.memory_bank.size}")

            for k in TASKS:
                _, te = bench.get_task_loaders(k)
                a_bare, _ = deployed_full(m_bare, te, k, device)
                _, te = bench.get_task_loaders(k)
                a_fix, _ = deployed_full(m_fix, te, k, device)
                rows.append(dict(arm=arm, seed=seed, task=k, matrix=float(mat[4, k]),
                                 reloaded=a_bare, restored=a_fix))

    print("\n" + "=" * 100)
    print(f"  {'arm':<10}{'seed':>6}{'task':>5}{'matrix':>10}{'reloaded':>10}"
          f"{'restored':>10}{'rel-mat':>10}{'res-mat':>10}")
    for r in rows:
        print(f"  {r['arm']:<10}{r['seed']:>6}{r['task']:>5}{r['matrix']:>10.4f}"
              f"{r['reloaded']:>10.4f}{r['restored']:>10.4f}"
              f"{r['reloaded']-r['matrix']:>+10.4f}{r['restored']-r['matrix']:>+10.4f}")

    print("\n" + "=" * 100)
    print("VERDICT (computed from the columns above)")
    print("=" * 100)
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        d_rel = float(np.mean([abs(r["reloaded"] - r["matrix"]) for r in rs]))
        d_res = float(np.mean([abs(r["restored"] - r["matrix"]) for r in rs]))
        n_exact = sum(abs(r["restored"] - r["matrix"]) < 1e-9 for r in rs)
        if d_res < 1e-9:
            v = "DIAGNOSED — restoring task_stats reproduces the matrix exactly"
        elif d_res < d_rel:
            v = (f"PARTIAL — gap {d_rel:.4f} -> {d_res:.4f} "
                 f"({n_exact}/{len(rs)} cells exact); residual remains")
        else:
            v = f"CANDIDATE DEAD — restoring did not reduce the gap ({d_rel:.4f} -> {d_res:.4f})"
        print(f"  {arm:<10} mean|reloaded-matrix| {d_rel:.4f}   "
              f"mean|restored-matrix| {d_res:.4f}   -> {v}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
