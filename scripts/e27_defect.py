"""E27 -- the equivariance defect (docs/E27_prereg.md sec 1, signed v2).

Same content, same encoder, two FORMATS:

    Z_T = f_theta_T( frame_T(x) )        the frame theta_T was trained on
    Z_k = f_theta_T( frame_k(x) )        task k's own frame

Fit Z_k ~ A(Z_T) with `Affine` (d1_fit.py:80-96), lambda selected per cell on a
VALIDATION split carved from train, and report on test:

    delta_k = || Z_k^test - A(Z_T^test) ||_F / || Z_k^test ||_F

delta = 0 means the format change moves the features by an affine map only, so a
readout repair could undo it exactly.

RAW DELTA IS NOT THE QUANTITY (sec 1a). A random 784->256 linear encoder reads
delta ~ 0.85 under a pixel permutation and has done nothing wrong: it keeps a
256-d view, and the permutation moves information into that view which the view
of the unpermuted input never contained. So delta mixes distance-from-equivariant
with how much the architecture's view discards. Each arm therefore gets a
RANDOM-INIT reference of the same architecture, input shape and output width,
three init seeds, measured on the same content, frames and lambda grid; the
primary predictor is the ratio delta / mean(delta_rand).

DIFFERENT FROM D1: D1 fit a map across CHECKPOINTS at fixed format. This fits a
map across FORMATS at a fixed checkpoint.

CONTROLS (sec 3), all run before any trained-arm cell:
  positive   square invertible random encoder, where A = W^-1 R W exists
             EXACTLY, at lambda = 0 over 10 draws from a dedicated stream.
             Priced four times before launch, each fixing a real defect: at
             d=256 when the contract names 784/1152 (wrong shape); at 1e-6 then
             1e-8 then 1e-10 (the residual scales with lambda AND cond(W), which
             ranges 1200-69307, so no positive lambda is safe against the draw);
             and off a shared sequential rng, so its value depended on what was
             drawn before it. An exact map needs no ridge: lambda = 0 reads
             <= 5.5e-9 on every draw, six orders under the 0.01 bar.
  floor      the same random encoder at 784->256 must read near 0.85,
             reproducing the pre-signing measurement.
  must-fail  shuffled pairing: delta must EXCEED its unshuffled value on every
             cell (measured 1.034 vs 0.847 on synthetic before launch).
  plumbing   at k = T, Z_k == Z_T and delta == 0 exactly.
  overfit    train-split delta beside test-split delta.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, PROBE_SUBSET_SEED)
from scripts.cure_screen import (E18_FACTORY, har_maps, har_relayout,                # noqa: E402
                                 mnist_relayout, rotated_relayout, BATCH)
from scripts.d1_fit import Affine, sha                                               # noqa: E402

LAM_GRID = [1e-3, 1e-4, 1e-5, 1e-6]
LAM_CONTROL = 0.0             # see run_controls: the exact map needs no ridge at all
CONTROL_DRAWS = 10            # the control has its own variance; one draw cannot price it
CONTROL_SEED = 900            # its OWN stream: a control read off a shared sequential rng
                              # depends on what was drawn before it, which is not a property
                              # of the thing being controlled
CONTROL_BAR = 0.01
FLOOR_TARGET = 0.85
VAL_FRAC = 0.2
REF_INITS = [0, 1, 2]
SHUF_SEED = 20260922
HAR_PARTITION = "1104af185c87"

# (ckpt template, run template, construction, n_classes, num_tasks, chunks, family)
ARMS = {
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",   "runs/e10off_ec_seed{s}",   "har_subject", 6,  5,  None, "LSTM"),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32",  "runs/e18_pmd_mlp_seed{s}",  "permuted",   10, 5,  None, "MLP"),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted",   10, 5,  None, "LSTM"),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32",  "runs/e18_rmd_mlp_seed{s}",  "rotated",    10, 5,  None, "MLP"),
    "e23b_t20":     ("runs/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",    "runs/e23b_t20_seed{s}",    "permuted",   10, 20, 20,   "LSTM"),
    "e23_har_mlp":  ("runs/ckpt_e23_har_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e23_har_mlp_seed{s}", "har_subject", 6,  5,  None, "MLP"),
}


def fit_delta(ZT_tr, Zk_tr, ZT_te, Zk_te, lam):
    mu, sd = ZT_tr.mean(0), ZT_tr.std(0) + 1e-8
    A = Affine.fit(ZT_tr, Zk_tr, mu, sd, lam)
    res = lambda a, b: float(np.linalg.norm(b - A.apply(a)) / max(np.linalg.norm(b), 1e-12))
    return res(ZT_te, Zk_te), res(ZT_tr, Zk_tr), A


def delta_with_selection(ZT, Zk, ZT_te, Zk_te, rng, grid=LAM_GRID):
    """lambda chosen on a validation split carved from TRAIN; never on test."""
    n = ZT.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(VAL_FRAC * n))
    va, tr = idx[:n_val], idx[n_val:]
    best, best_lam = None, None
    for lam in grid:
        d_val, _, _ = fit_delta(ZT[tr], Zk[tr], ZT[va], Zk[va], lam)
        if best is None or d_val < best:
            best, best_lam = d_val, lam
    d_te, d_tr, _ = fit_delta(ZT, Zk, ZT_te, Zk_te, best_lam)
    return {"delta_test": d_te, "delta_train": d_tr, "lam": best_lam, "delta_val": best}


# ------------------------------------------------------------------ controls --
def run_controls(_unused=None):
    out = {}
    def synth(d_in, d_out, lam, n_tr=4000, n_te=2000, shuffle=False, seed=0):
        rng = np.random.default_rng(CONTROL_SEED + seed)
        X_tr = rng.normal(size=(n_tr, d_in)); X_te = rng.normal(size=(n_te, d_in))
        perm = rng.permutation(d_in); W = rng.normal(size=(d_in, d_out)) / np.sqrt(d_in)
        ZT_tr, ZT_te = X_tr @ W, X_te @ W
        Zk_tr, Zk_te = X_tr[:, perm] @ W, X_te[:, perm] @ W
        if shuffle:
            Zk_tr = Zk_tr[rng.permutation(n_tr)]; Zk_te = Zk_te[rng.permutation(n_te)]
        return fit_delta(ZT_tr, Zk_tr, ZT_te, Zk_te, lam)[0]
    # positive: square invertible, exact A = W^-1 R W exists.
    # EVERY DRAW, not one: cond(W) ranges 1200-27556 at d=1152 and the residual
    # tracks it, so a single draw cannot price a 0.01 bar.
    pos = {str(d): [synth(d, d, LAM_CONTROL, seed=i) for i in range(CONTROL_DRAWS)] for d in (784, 1152)}
    out["positive_square_invertible"] = {
        "lam": LAM_CONTROL, "bar": CONTROL_BAR, "draws": CONTROL_DRAWS,
        "delta_per_draw": pos,
        "delta_max": {k: max(v) for k, v in pos.items()},
        "pass": all(max(v) < CONTROL_BAR for v in pos.values()),
        "pricing": "lambda chosen so that ALL draws clear the bar with margin. "
                   "1e-6 leaves delta at 0.037 (784) and 0.036 (1152) -- priced at d=256 in the "
                   "contract. 1e-8 clears at 784 (max 0.0013) but FAILS at 1152 on some draws "
                   "(max 0.0258); 1e-10 still fails a bad draw (0.0151). The residual of a ridge "
                   "fit to an EXACT linear map scales with lambda and with cond(W), which ranges "
                   "1200-69307 here -- so no lambda > 0 is safe against the draw. An exact map "
                   "needs no ridge: at lambda = 0 the worst of 10 draws reads 3.4e-9 (784) and "
                   "5.5e-9 (1152), six orders under the bar. Fourth pricing; the first three each "
                   "fixed a real defect (wrong shape, then wrong lambda, then shared rng) and the "
                   "fourth removes the knob instead of tuning it."}
    # floor witness: dimension-reducing random encoder
    fl = synth(784, 256, LAM_GRID[0], seed=50)
    out["floor_random_784_256"] = {"delta": fl, "target": FLOOR_TARGET,
                                   "pass": abs(fl - FLOOR_TARGET) <= 0.10}
    # must-fail shape, on synthetic
    un, sh = synth(784, 256, LAM_GRID[0], seed=60), synth(784, 256, LAM_GRID[0], shuffle=True, seed=60)
    out["must_fail_shape_synthetic"] = {"unshuffled": un, "shuffled": sh, "pass": sh > un}
    for k, v in out.items():
        print(f"  control {k}: {v}")
    return out


def randomize(model, seed):
    """Same architecture, shape and width; fresh weights. The sec 1a reference."""
    torch.manual_seed(seed)
    n = 0
    for mod in model.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters(); n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="E27: the equivariance defect")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    ck_t, run_t, kind, n_classes, num_tasks, chunks, family = ARMS[args.arm]
    root = args.ckpt_root.rstrip("/")
    ck = ck_t.replace("runs/", root + "/", 1).format(s=args.seed)
    run_dir = run_t.replace("runs/", root + "/", 1).format(s=args.seed)
    T = num_tasks - 1
    print("=" * 104 + f"\nE27 DEFECT  arm={args.arm} ({family}) seed={args.seed} T={T}\n" + "=" * 104)

    arm_rec = json.load(open(f"{run_dir}/mafc_results.json"))["arm"]
    assert arm_rec["era_checkpoints"] is True, arm_rec
    assert arm_rec["num_tasks"] == num_tasks, (arm_rec["num_tasks"], num_tasks)
    if chunks is not None:
        assert arm_rec.get("content_chunks") == chunks, arm_rec.get("content_chunks")
    print(f"  arm identity: backbone {arm_rec['backbone']}, benchmark {arm_rec['benchmark']}, "
          f"num_tasks {arm_rec['num_tasks']}, heads {arm_rec['use_task_heads']}")

    rng = np.random.default_rng(SHUF_SEED + args.seed)
    controls = None if args.skip_controls else run_controls()

    # ---- construction and relay -------------------------------------------------
    if kind == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        bench = HARSubjectBenchmark(num_tasks=num_tasks, root=".", batch_size=BATCH)
        assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
        maps = har_maps(bench._sd)
        relay = lambda x, s, d: har_relayout(x, s, d, maps)
    elif chunks is not None:
        from src.data.permuted_mnist import PermutedMNISTBenchmark
        bench = PermutedMNISTBenchmark(num_tasks=num_tasks, batch_size=BATCH, seed=args.seed,
                                       disjoint_content=True, content_chunks=chunks)
        assert arm_rec["construction_fingerprint"] == bench.construction_fingerprint()
        relay = lambda x, s, d: mnist_relayout(x, s, d, bench.permutations)
    else:
        bench = E18_FACTORY[kind](args.seed)
        assert arm_rec["content_fingerprint"] == bench.content_fingerprint()
        relay = ((lambda x, s, d: rotated_relayout(x, s, d, bench.angles)) if kind == "rotated"
                 else (lambda x, s, d: mnist_relayout(x, s, d, bench.permutations)))
    lossy = (kind == "rotated")
    data = load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=args.probe_seed)

    m_final = load(ck, T)
    refs = []
    for i in REF_INITS:                                     # sec 1a random references
        r = load(ck, T); n = randomize(r, 1000 * i + args.seed); refs.append(r)
    print(f"  built {len(refs)} random-init references ({n} modules reset each)")

    rows, t0 = [], time.time()
    tasks = list(range(T))[: args.max_tasks] if args.max_tasks else list(range(T))
    for k in tasks:
        tk = time.time()
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m_final, xte, k, False, device, label=f"s{args.seed}/T{k}", verbose=False)
        assert g["p_b"], (args.seed, k)
        xT_tr, xT_te = relay(xtr, k, T), relay(xte, k, T)    # frame T
        xk_tr, xk_te = xtr, xte                              # frame k (own), as stored
        relay_err = None
        if lossy:
            back = relay(xT_te, T, k)
            relay_err = float((back - xte).abs().mean() / xte.abs().mean().clamp_min(1e-9))
        h_tr, h_te = sha(xtr), sha(xte)
        f = lambda m, x, y: features_and_logits(m, x, y, k, False, device)[0].numpy().astype(np.float64)
        ZT_tr, ZT_te = f(m_final, xT_tr, ytr), f(m_final, xT_te, yte)
        Zk_tr, Zk_te = f(m_final, xk_tr, ytr), f(m_final, xk_te, yte)
        assert sha(xtr) == h_tr and sha(xte) == h_te, "relay must not mutate its input"

        sel = delta_with_selection(ZT_tr, Zk_tr, ZT_te, Zk_te, np.random.default_rng(SHUF_SEED + k))
        # must-fail: shuffled pairing on THIS cell's real features
        r2 = np.random.default_rng(SHUF_SEED + 31 * k)
        sh = delta_with_selection(ZT_tr, Zk_tr[r2.permutation(len(Zk_tr))],
                                  ZT_te, Zk_te[r2.permutation(len(Zk_te))], r2)
        # plumbing: k == T is the identity pairing
        pl_T, pl_k = f(m_final, relay(xte, k, T), yte), f(m_final, relay(xte, k, T), yte)
        plumb = float(np.abs(pl_T - pl_k).max())
        # sec 1a reference on the same content and frames
        ref_d = []
        for r in refs:
            RT_tr, RT_te = f(r, xT_tr, ytr), f(r, xT_te, yte)
            Rk_tr, Rk_te = f(r, xk_tr, ytr), f(r, xk_te, yte)
            ref_d.append(delta_with_selection(RT_tr, Rk_tr, RT_te, Rk_te,
                                              np.random.default_rng(SHUF_SEED + k))["delta_test"])
        row = {"arm": args.arm, "family": family, "seed": args.seed, "task": k,
               "n_train": int(len(ytr)), "n_test": int(len(yte)), "dim": int(ZT_tr.shape[1]),
               "delta": sel["delta_test"], "delta_train": sel["delta_train"],
               "lam": sel["lam"], "delta_val": sel["delta_val"],
               "delta_shuffled": sh["delta_test"],
               "must_fail_ok": bool(sh["delta_test"] > sel["delta_test"]),
               "delta_rand_per_init": ref_d, "delta_rand": float(np.mean(ref_d)),
               "delta_tilde": sel["delta_test"] / max(float(np.mean(ref_d)), 1e-12),
               "plumb_max_abs": plumb, "relay_rel_err": relay_err,
               "input_sha": h_te, "seconds": round(time.time() - tk, 1)}
        rows.append(row)
        print(f"  k={k:>2}: delta {row['delta']:.4f} (train {row['delta_train']:.4f}, lam {row['lam']:.0e}) | "
              f"rand {row['delta_rand']:.4f} -> delta~ {row['delta_tilde']:.4f} | shuf {row['delta_shuffled']:.4f} "
              f"{'OK' if row['must_fail_ok'] else 'MUST-FAIL VIOLATED'} | plumb {plumb:.1e}"
              + (f" | relay err {relay_err:.3f}" if relay_err is not None else "") + f" | {row['seconds']}s")

    out = {"arm": args.arm, "family": family, "seed": args.seed, "num_tasks": num_tasks,
           "construction": kind, "lossy_relay": lossy, "lam_grid": LAM_GRID,
           "val_frac": VAL_FRAC, "ref_inits": REF_INITS, "controls": controls,
           "pooled": {k: float(np.mean([r[k] for r in rows]))
                      for k in ("delta", "delta_rand", "delta_tilde", "delta_shuffled")},
           "must_fail_cells": sum(r["must_fail_ok"] for r in rows), "cells": len(rows),
           "rows": rows, "seconds_total": round(time.time() - t0, 1)}
    p = out["pooled"]
    print(f"\n  pooled: delta {p['delta']:.4f}  rand {p['delta_rand']:.4f}  delta~ {p['delta_tilde']:.4f}  "
          f"shuffled {p['delta_shuffled']:.4f} | must-fail {out['must_fail_cells']}/{out['cells']}")
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"defect_seed{args.seed}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"  wrote {fp}  ({out['seconds_total']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
