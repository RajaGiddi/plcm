"""E26 FS -- the frame sweep (docs/E26_prereg.md sec FS).

E23 found that re-laying old inputs into the LATEST frame costs accuracy on
pretrained trunks, and the paper explains that as "the trunk keeps handling for
every frame rather than converging to the last". That is inferred from one
comparison (frame k vs frame T). FS measures it: task k's content is evaluated
in EVERY frame j and the peak is located.

    D_k(j) = acc( h_k on f_theta_T( frame_j(x_k) ) )       all j in 0..19
    P_k(j) = refit on task-k TRAIN in frame j under theta_T, read on TEST in frame j
                                                             j in {k, T, 0} + 3 drawn once

Four hypotheses, registered: H-own (peak at j = k), H-last (j = T), H-flat (no
peak within floor), H-base (j = 0, the unpermuted layout the trunk was
pretrained on).

RENDERING IS THE DATASET'S OWN. A frame-j view of task k is produced by setting
the dataset's `perm` attribute (e23_decompose.py:94-97), exactly as E23 did. The
algebraic relay bapply(bapply(x, inv(P_k)), P_j) is NOT the path; it is the
exactness CONTROL, asserted at 0.0 on every j including j = 0 where P[0] is
None. Frames run 0..19 -- twenty of them, frame 0 the base layout
(split_cifar100.py:173-180).

Everything else is inherited from E23 rather than re-implemented (catch 32):
the shift-fingerprint gate, `dec.load_era`, `dec.features_and_logits` (sequential
loaders, label equality asserted), the P3 positive controls and per-cell
`assert_p3`. C-ID: D_k(k) against E23's `acc_orig` and D_k(T) against E23's
`acc_orig_relaid`, at the fp16 reload floor, or nothing is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import refit_probe                                     # noqa: E402
from scripts.e23_decompose import BACKBONES, bapply, acc                            # noqa: E402
from src.data.split_cifar100 import SplitCIFAR100Benchmark, invert_perm            # noqa: E402

REFIT_DRAW_SEED = 20260921          # the three extra refit frames, drawn once, recorded
UNSEEN_SEED = 7                     # a permutation set no B6 arm was trained on
CID_FLOOR = 0.008                   # E23 C-RELOAD fp16 reload floor, max over both backbones


def relay_to(x, P, src, dst):
    """Algebraic relay src -> dst; None means the identity layout (task 0)."""
    n = x[0].numel()
    inv = invert_perm(P[src]) if P[src] is not None else torch.arange(n)
    fwd = P[dst] if P[dst] is not None else torch.arange(n)
    return bapply(bapply(x, inv), fwd)


def main():
    ap = argparse.ArgumentParser(description="E26 FS: frame sweep, pretrained B6 arms")
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref", required=True, help="E23 decomposition artifact for this arm/seed (C-ID)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=None, help="smoke: cap the old tasks")
    ap.add_argument("--skip-refit", action="store_true", help="smoke: D_k(j) only")
    args = ap.parse_args()
    device = torch.device(args.device)
    modname, cfg_path, arm = BACKBONES[args.backbone]
    dec = importlib.import_module(modname)
    from scripts.e12_p3 import assert_p3, positive_controls
    root = args.ckpt_root.rstrip("/")
    ckpt_dir = f"{root}/ckpt_{arm}_seed{args.seed}/mafc_seed{args.seed}"
    run_arm = json.load(open(f"{root}/{arm}_seed{args.seed}/mafc_results.json"))["arm"]
    assert run_arm["benchmark"] == "cifar100_permuted" and run_arm["use_task_heads"] is True \
        and run_arm["era_checkpoints"] is True and run_arm["backbone"] == args.backbone, run_arm
    print("=" * 100 + f"\nE26 FS  {arm} seed {args.seed}  ckpt {ckpt_dir}\n" + "=" * 100)

    bench = SplitCIFAR100Benchmark(num_tasks=dec.NUM_TASKS, batch_size=dec.BATCH, root=args.data_root,
                                   seed=args.seed, remap_labels=True, shift_mode="patch", download=False)
    expected = (yaml.safe_load(open(cfg_path)).get("benchmark") or {}).get("expected_shift") or {}
    sh = bench.shift_fingerprint()
    assert expected.get(args.seed) == sh, f"shift fingerprint {sh} != registered {expected.get(args.seed)}"
    T = dec.FINAL
    P = bench.perms
    n_frames = len(P)
    assert n_frames == dec.NUM_TASKS and P[0] is None, (n_frames, P[0] is None)
    # the unseen frame for the must-fail: another seed's permutation set, index T
    P_unseen = SplitCIFAR100Benchmark(num_tasks=dec.NUM_TASKS, batch_size=dec.BATCH, root=args.data_root,
                                      seed=UNSEEN_SEED, remap_labels=True, shift_mode="patch",
                                      download=False).perms[T]
    assert not torch.equal(P_unseen, P[T]), "unseen frame equals a seen one"

    def loaders(k, perm):
        tr, te = bench._make(k, True), bench._make(k, False)
        tr.perm, te.perm = perm, perm
        return (DataLoader(tr, batch_size=dec.BATCH, shuffle=False),
                DataLoader(te, batch_size=dec.BATCH, shuffle=False))

    ref_rows = {r["task"]: r for r in json.load(open(args.ref))["rows"]}
    m_final = dec.load_era(ckpt_dir, T, device)
    _, te1 = loaders(1, P[1]); xprobe = next(iter(te1))[0][:16].to(device)
    ctrl = positive_controls(m_final, xprobe, task_k=1, verbose=False)
    assert ctrl["p3a_control"] and ctrl["p3b_control"], ctrl
    print("  P3 positive controls FIRED on the final checkpoint")

    rng = np.random.default_rng(REFIT_DRAW_SEED + args.seed)
    rows, relay_max, cid = [], 0.0, {"j=k": [], "j=T": []}
    tasks = list(dec.OLD_TASKS)[: args.max_tasks] if args.max_tasks else list(dec.OLD_TASKS)
    t_start = time.time()
    for k in tasks:
        t0 = time.time()
        # raw batch of task k in its own frame, for the relay-exactness control
        _, te_k = loaders(k, P[k]); xb_k = next(iter(te_k))[0]
        D, relay_err, n_te, y_ref = {}, {}, None, None
        for j in range(n_frames):
            _, te_j = loaders(k, P[j])
            xb_j = next(iter(te_j))[0]
            relay_err[j] = float((relay_to(xb_k, P, k, j) - xb_j).abs().max())   # exactness control, every j
            relay_max = max(relay_max, relay_err[j])
            if j in (k, T):
                assert_p3(m_final, xb_j[:16].to(device), k, label=f"final/T{k}/frame{j}")
            _, l_j, y_j = dec.features_and_logits(m_final, te_j, k, False, device)
            if y_ref is None:
                y_ref, n_te = y_j, int(len(y_j))
            assert torch.equal(y_j, y_ref), f"task {k} frame {j}: label order differs"
            D[j] = acc(l_j, y_j)
        assert relay_max == 0.0, f"relay exactness violated: {relay_max:.3e}"

        # must-fail: an unseen frame must not beat the seen-frame maximum
        te_u = DataLoader(bench._make(k, False), batch_size=dec.BATCH, shuffle=False)
        te_u.dataset.perm = P_unseen
        _, l_u, y_u = dec.features_and_logits(m_final, te_u, k, False, device)
        assert torch.equal(y_u, y_ref)
        D_unseen = acc(l_u, y_u)

        # C-ID against E23's artifact
        r = ref_rows[k]
        cid["j=k"].append(abs(D[k] - r["acc_orig"])); cid["j=T"].append(abs(D[T] - r["acc_orig_relaid"]))

        # refit subset
        Pk = {}
        if not args.skip_refit:
            extra = [int(j) for j in rng.choice([j for j in range(n_frames) if j not in (k, T, 0)], 3, replace=False)]
            for j in sorted({k, T, 0, *extra}):
                tr_j, te_j = loaders(k, P[j])
                f_tr, _, y_tr = dec.features_and_logits(m_final, tr_j, k, False, device)
                f_te, _, y_te = dec.features_and_logits(m_final, te_j, k, False, device)
                assert torch.equal(y_te, y_ref)
                n_cls = dec.CLASSES_PER_TASK if hasattr(dec, "CLASSES_PER_TASK") else int(y_te.max()) + 1
                Pk[j] = refit_probe(f_tr, y_tr, f_te, y_te, n_cls)
            Pk_extra = extra
        else:
            Pk_extra = []

        vals = np.array([D[j] for j in range(n_frames)])
        jmax = int(vals.argmax())
        res95 = 1.96 * float(np.sqrt(vals[jmax] * (1 - vals[jmax]) / n_te))
        within = [int(j) for j in range(n_frames) if vals[jmax] - vals[j] <= res95]
        row = {"arm": arm, "seed": args.seed, "task": k, "n_test": n_te,
               "D": {str(j): D[j] for j in range(n_frames)},
               "D_unseen_frame": D_unseen, "seen_max": float(vals.max()),
               "unseen_below_seen_max": bool(D_unseen <= vals.max()),
               "argmax_j": jmax, "peak": float(vals[jmax]), "res95_at_peak": res95,
               "frames_within_res_of_peak": within,
               "peak_at_own": jmax == k, "peak_at_last": jmax == T, "peak_at_base": jmax == 0,
               "own_within_2": abs(jmax - k) <= 2,
               "flat": len(within) == n_frames,
               "D_own": D[k], "D_last": D[T], "D_base": D[0],
               "cid_jk_abs": abs(D[k] - r["acc_orig"]), "cid_jT_abs": abs(D[T] - r["acc_orig_relaid"]),
               "P": {str(j): v for j, v in Pk.items()}, "P_extra_frames": Pk_extra,
               "relay_max_abs": max(relay_err.values()),
               "input_sha": hashlib.sha1(y_ref.numpy().tobytes()).hexdigest()[:12],
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  k={k:>2}: peak j={jmax:>2} ({vals[jmax]:.3f}) | own {D[k]:.3f} last {D[T]:.3f} base {D[0]:.3f} "
              f"unseen {D_unseen:.3f} | within-res frames {len(within):>2}/{n_frames} | C-ID {row['cid_jk_abs']:.4f}/{row['cid_jT_abs']:.4f}"
              + (f" | P: " + " ".join(f"{j}:{v:.3f}" for j, v in sorted(Pk.items())) if Pk else "")
              + f" | {row['seconds']}s")

    cid_ok = max(cid["j=k"]) <= CID_FLOOR and max(cid["j=T"]) <= CID_FLOOR
    summary = {
        "arm": arm, "backbone": args.backbone, "seed": args.seed, "n_frames": n_frames, "T": T,
        "shift_fingerprint": sh, "unseen_seed": UNSEEN_SEED, "refit_draw_seed": REFIT_DRAW_SEED + args.seed,
        "controls": {
            "c_id_jk_max_abs": max(cid["j=k"]), "c_id_jT_max_abs": max(cid["j=T"]),
            "c_id_floor": CID_FLOOR, "c_id_pass": cid_ok,
            "relay_exactness_max_abs": relay_max, "relay_exact": relay_max == 0.0,
            "must_fail_unseen_below_seen_max": {"cells": sum(r["unseen_below_seen_max"] for r in rows),
                                                "of": len(rows)},
            "p3_positive_controls_fired": True},
        "counts": {"peak_at_own": sum(r["peak_at_own"] for r in rows),
                   "peak_at_last": sum(r["peak_at_last"] for r in rows),
                   "peak_at_base": sum(r["peak_at_base"] for r in rows),
                   "own_within_2": sum(r["own_within_2"] for r in rows),
                   "flat": sum(r["flat"] for r in rows), "cells": len(rows)},
        "rows": rows, "seconds_total": round(time.time() - t_start, 1)}
    c = summary["counts"]
    print(f"\n  C-ID j=k max {max(cid['j=k']):.4f}, j=T max {max(cid['j=T']):.4f} vs floor {CID_FLOOR} -> "
          f"{'PASS' if cid_ok else 'FAIL -- nothing read'} | relay exact {relay_max == 0.0} | "
          f"must-fail {summary['controls']['must_fail_unseen_below_seen_max']}")
    print(f"  peaks: own {c['peak_at_own']}/{c['cells']}  last {c['peak_at_last']}/{c['cells']}  "
          f"base {c['peak_at_base']}/{c['cells']}  own±2 {c['own_within_2']}/{c['cells']}  flat {c['flat']}/{c['cells']}")
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"deployed_seed{args.seed}.json")
    json.dump(summary, open(p, "w"), indent=2, default=float)
    # the contract's clause table names a separate refit artifact; written from the
    # same run so both paths exist and the row can read either
    pr = os.path.join(args.out_dir, f"refit_seed{args.seed}.json")
    json.dump({"arm": arm, "seed": args.seed, "T": T, "refit_draw_seed": REFIT_DRAW_SEED + args.seed,
               "rows": [{"task": r["task"], "P": r["P"], "P_extra_frames": r["P_extra_frames"],
                         "D_at_P_frames": {j: r["D"][j] for j in r["P"]}} for r in rows]},
              open(pr, "w"), indent=2, default=float)
    print(f"  wrote {p} and {pr}  ({summary['seconds_total']}s)")
    return 0 if cid_ok else 1


if __name__ == "__main__":
    sys.exit(main())
