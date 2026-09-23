"""E26 FS on A3 -- the scratch comparison (docs/E26_prereg.md sec FS).

The same sweep as e26_fs.py, on E23-B's A3: scratch LSTM, disjoint-content
Permuted MNIST, twenty tasks, shared head. The deployed head is W_T on every
cell (there are no per-task heads), which is what E23-B's C-ID reproduces.
Writes the same schema as the pretrained script into runs/e26/fs/e23b_t20/ so
e26_fs_row.py reads all three arms alike.

Rendering is `cure_screen.mnist_relayout(x, k, j, perms)` -- E23-B's verified
path (C-RELAY: identity at k -> k and exact round trip). Frames run 0..19 and
perms[0] is None; the relay handles it. C-ID against E23-B's decomposition:
D_k(k) vs `acc_orig`, D_k(T) vs `acc_relaid`, at 1e-6 (fp32 shadow, so exact).
Must-fail: another seed's permutation at index T, unseen in training.
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
from scripts.channel_decomp import (load, features_and_logits, load_task_data,   # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import mnist_relayout, BATCH                            # noqa: E402
from src.data.permuted_mnist import PermutedMNISTBenchmark                       # noqa: E402

NUM_TASKS, CHUNKS, N_CLASSES = 20, 20, 10
REFIT_DRAW_SEED = 20260921
UNSEEN_SEED = 7
CID_TOL = 1e-6


def acc(l, y):
    return float((l.argmax(1) == y).float().mean())


def main():
    ap = argparse.ArgumentParser(description="E26 FS on A3")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--ref", required=True, help="runs/e23b/e23b_t20/decomp_seed{s}.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    ck = f"{root}/ckpt_e23b_t20_seed{args.seed}/mafc_seed{args.seed}_fp32"
    ra = json.load(open(f"{root}/e23b_t20_seed{args.seed}/mafc_results.json"))["arm"]
    assert ra["num_tasks"] == NUM_TASKS and ra.get("content_chunks") == CHUNKS and ra["era_checkpoints"], ra
    bench = PermutedMNISTBenchmark(num_tasks=NUM_TASKS, batch_size=BATCH, seed=args.seed,
                                   disjoint_content=True, content_chunks=CHUNKS)
    assert ra["construction_fingerprint"] == bench.construction_fingerprint()
    P = bench.permutations
    T = NUM_TASKS - 1
    assert len(P) == NUM_TASKS
    P_unseen = PermutedMNISTBenchmark(num_tasks=NUM_TASKS, batch_size=BATCH, seed=UNSEEN_SEED,
                                      disjoint_content=True, content_chunks=CHUNKS).permutations[T]
    assert P_unseen is not None and (P[T] is None or not torch.equal(P_unseen, P[T]))
    print("=" * 100 + f"\nE26 FS  A3 (e23b_t20) seed {args.seed}  ckpt {ck}\n" + "=" * 100)

    data = load_task_data(bench, n_tasks=NUM_TASKS, probe_subset_seed=PROBE_SUBSET_SEED)
    m_final = load(ck, T)
    ref_rows = {r["task"]: r for r in json.load(open(args.ref))["rows"]}
    rng = np.random.default_rng(REFIT_DRAW_SEED + args.seed)
    rows, cid = [], {"j=k": [], "j=T": []}
    relay_max = 0.0
    tasks = list(range(T))[: args.max_tasks] if args.max_tasks else list(range(T))
    t_start = time.time()
    # the relay, as a function that also handles the unseen frame
    P_ext = list(P) + [P_unseen]
    for k in tasks:
        t0 = time.time()
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m_final, xte, k, False, device, label=f"s{args.seed}/T{k}", verbose=False)
        assert g["p_b"], (args.seed, k)
        # relay exactness: k -> k identity and round trip k -> j -> k, every j
        D = {}
        for j in range(NUM_TASKS):
            xj = mnist_relayout(xte, k, j, P)
            back = mnist_relayout(xj, j, k, P)
            relay_max = max(relay_max, float((back - xte).abs().max()))
            if j == k:
                relay_max = max(relay_max, float((xj - xte).abs().max()))
            _, l = features_and_logits(m_final, xj, yte, k, False, device)
            D[j] = acc(l, yte)
        assert relay_max == 0.0, relay_max
        xu = mnist_relayout(xte, k, NUM_TASKS, P_ext)
        _, lu = features_and_logits(m_final, xu, yte, k, False, device)
        D_unseen = acc(lu, yte)
        r = ref_rows[k]
        cid["j=k"].append(abs(D[k] - r["acc_orig"])); cid["j=T"].append(abs(D[T] - r["acc_relaid"]))
        Pk, extra = {}, []
        if not args.skip_refit:
            extra = [int(j) for j in rng.choice([j for j in range(NUM_TASKS) if j not in (k, T, 0)], 3, replace=False)]
            for j in sorted({k, T, 0, *extra}):
                f_tr, _ = features_and_logits(m_final, mnist_relayout(xtr, k, j, P), ytr, k, False, device)
                f_te, _ = features_and_logits(m_final, mnist_relayout(xte, k, j, P), yte, k, False, device)
                Pk[j] = refit_probe(f_tr, ytr, f_te, yte, N_CLASSES)
        vals = np.array([D[j] for j in range(NUM_TASKS)]); jmax = int(vals.argmax())
        n_te = int(len(yte))
        res95 = 1.96 * float(np.sqrt(vals[jmax] * (1 - vals[jmax]) / n_te))
        within = [int(j) for j in range(NUM_TASKS) if vals[jmax] - vals[j] <= res95]
        row = {"arm": "e23b_t20", "seed": args.seed, "task": k, "n_test": n_te,
               "D": {str(j): D[j] for j in range(NUM_TASKS)}, "D_unseen_frame": D_unseen,
               "seen_max": float(vals.max()), "unseen_below_seen_max": bool(D_unseen <= vals.max()),
               "argmax_j": jmax, "peak": float(vals[jmax]), "res95_at_peak": res95,
               "frames_within_res_of_peak": within,
               "peak_at_own": jmax == k, "peak_at_last": jmax == T, "peak_at_base": jmax == 0,
               "own_within_2": abs(jmax - k) <= 2, "flat": len(within) == NUM_TASKS,
               "D_own": D[k], "D_last": D[T], "D_base": D[0],
               "cid_jk_abs": abs(D[k] - r["acc_orig"]), "cid_jT_abs": abs(D[T] - r["acc_relaid"]),
               "P": {str(j): v for j, v in Pk.items()}, "P_extra_frames": extra,
               "relay_max_abs": relay_max,
               "input_sha": hashlib.sha1(yte.numpy().tobytes()).hexdigest()[:12],
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  k={k:>2}: peak j={jmax:>2} ({vals[jmax]:.3f}) | own {D[k]:.3f} last {D[T]:.3f} base {D[0]:.3f} unseen {D_unseen:.3f} "
              f"| within-res {len(within):>2}/20 | C-ID {row['cid_jk_abs']:.1e}/{row['cid_jT_abs']:.1e}"
              + (f" | P " + " ".join(f"{j}:{v:.3f}" for j, v in sorted(Pk.items())) if Pk else "") + f" | {row['seconds']}s")
    cid_ok = max(cid["j=k"]) <= CID_TOL and max(cid["j=T"]) <= CID_TOL
    summary = {"arm": "e23b_t20", "backbone": "lstm", "seed": args.seed, "n_frames": NUM_TASKS, "T": T,
               "construction_fingerprint": bench.construction_fingerprint(), "unseen_seed": UNSEEN_SEED,
               "controls": {"c_id_jk_max_abs": max(cid["j=k"]), "c_id_jT_max_abs": max(cid["j=T"]),
                            "c_id_floor": CID_TOL, "c_id_pass": cid_ok,
                            "relay_exactness_max_abs": relay_max, "relay_exact": relay_max == 0.0,
                            "must_fail_unseen_below_seen_max": {"cells": sum(r["unseen_below_seen_max"] for r in rows), "of": len(rows)},
                            "path_identity": "assert_path_identity per cell, P-B asserted"},
               "counts": {"peak_at_own": sum(r["peak_at_own"] for r in rows), "peak_at_last": sum(r["peak_at_last"] for r in rows),
                          "peak_at_base": sum(r["peak_at_base"] for r in rows), "own_within_2": sum(r["own_within_2"] for r in rows),
                          "flat": sum(r["flat"] for r in rows), "cells": len(rows)},
               "rows": rows, "seconds_total": round(time.time() - t_start, 1)}
    c = summary["counts"]
    print(f"\n  C-ID j=k {max(cid['j=k']):.1e} j=T {max(cid['j=T']):.1e} -> {'PASS' if cid_ok else 'FAIL'} | relay exact {relay_max == 0.0} "
          f"| must-fail {summary['controls']['must_fail_unseen_below_seen_max']}")
    print(f"  peaks: own {c['peak_at_own']}/{c['cells']} last {c['peak_at_last']}/{c['cells']} base {c['peak_at_base']}/{c['cells']} "
          f"own±2 {c['own_within_2']}/{c['cells']} flat {c['flat']}/{c['cells']}")
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"deployed_seed{args.seed}.json")
    json.dump(summary, open(p, "w"), indent=2, default=float)
    json.dump({"arm": "e23b_t20", "seed": args.seed, "T": T,
               "rows": [{"task": r["task"], "P": r["P"], "P_extra_frames": r["P_extra_frames"]} for r in rows]},
              open(os.path.join(args.out_dir, f"refit_seed{args.seed}.json"), "w"), indent=2, default=float)
    print(f"  wrote {p}  ({summary['seconds_total']}s)")
    return 0 if cid_ok else 1


if __name__ == "__main__":
    sys.exit(main())
