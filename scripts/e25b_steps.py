"""E25 B -- per-step drift (docs/E25_prereg.md sec B).

D1 fit ONE map from theta_k to theta_T and found it 41-45% non-linear. B asks
whether a SINGLE BOUNDARY is closer to linear, and whether the per-step maps
compose.

Per (arm, seed, old task k < T), with each encoder at home in its own frame:

    Z_t^(k) = f_{theta_t}( relay(x_k, k -> t) )        for t = k .. T
                                                        relay(x, k, k) is the identity

    drift   S_t : Z_{t-1} -> Z_t      repair  R_t : Z_t -> Z_{t-1}       both fit DIRECTLY,
                                                                          never inverted

    composed repair   Rhat^(k)(Z) = R_{k+1}.apply( ... R_T.apply(Z) )
    read              acc( h_k( Rhat^(k)(Z_T) ) )        h_k = cure_screen.era_head_of

COMPOSITION IS BY CHAINING `.apply`, NOT BY MULTIPLYING `M_raw`. `Affine.fit`
carries an intercept `c_std` and `M_raw()` returns the LINEAR PART ONLY
(d1_fit.py:91-92), so a product of `M_raw`s silently drops every step's offset.
`.apply` returns raw coordinates including the intercept, so the chain is exact.
The composed raw affine (A, c) is then recovered EXACTLY by pushing the identity
through the chain -- a composition of affine maps is affine, so
c = chain(0) and A[:, i] = chain(e_i) - c is not an approximation. That is what
makes the cross-task read and the composed spectrum cheap to store.

Controls (sec B): C-PLUMB fit on (Z, Z); C-SHUF must-fail (shuffled pairing at
every step collapses the composed repair to A0); composition consistency against
D1's single-shot fit, reported either way. On the ROTATED arm every relay is
lossy except at multiples of 90 degrees, so the per-step relay reconstruction
error is printed beside the residual and the 15% bar is not attributed to the
encoder where the two are comparable.

Path identity asserted per cell (catch 28); models through PLCM.load_era (catch
29); data through the audited loader with the recorded draw (catch 32).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,      # noqa: E402
                                    assert_path_identity, PROBE_SUBSET_SEED)
from scripts.cure_screen import (E18_FACTORY, era_head_of, har_maps, har_relayout,  # noqa: E402
                                 mnist_relayout, rotated_relayout, BATCH)
from scripts.d1_fit import Affine, spectrum, sha, acc, se95                          # noqa: E402

SEEDS = [42, 1337, 2024]
LAM = 1e-3                     # D1's LAM_MAIN
SHUF_SEED = 20260921
HAR_PARTITION = "1104af185c87"
RESIDUAL_BAR = 0.15            # sec B: "Bar 15% vs D1's 41-45%"

# (ckpt template, run template, construction, n_classes, num_tasks, content_chunks)
ARMS = {
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",  "runs/e10off_ec_seed{s}",  "har_subject", 6,  5,  None),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted",   10, 5,  None),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted", 10, 5,  None),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_rmd_mlp_seed{s}", "rotated",   10, 5,  None),
    "e23b_t20":     ("runs/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",   "runs/e23b_t20_seed{s}",   "permuted",   10, 20, 20),
}


def compose_affine(chain, d: int) -> tuple[np.ndarray, np.ndarray]:
    """EXACT raw affine of a chain of Affine maps applied in list order.

    Each `Affine.apply` is affine in raw coordinates, so the composition is too.
    Pushing [0; I] through the chain reads the offset off the first row and the
    linear part off the rest. Exact, not a fit: asserted against a live batch by
    the caller.
    """
    probe = np.vstack([np.zeros((1, d)), np.eye(d)])
    for f in chain:
        probe = f.apply(probe)
    c = probe[0]
    A = (probe[1:] - c)                      # rows are A[i, :] = (A^T)[:, i]
    return A, c


def apply_affine(Z: np.ndarray, Ac) -> np.ndarray:
    A, c = Ac
    return Z @ A + c


def head_acc(h, Z: np.ndarray, y: torch.Tensor) -> float:
    with torch.no_grad():
        dev = next(h.parameters()).device
        return acc(h(torch.from_numpy(Z).float().to(dev)).cpu(), y)


def main():
    ap = argparse.ArgumentParser(description="E25 B: per-step drift")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-cross", type=int, default=12,
                    help="cap on ordered (k, j) pairs scored for repair SHARING; the "
                         "20-task arm has 342 and the measurement saturates well before that")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    ckpt_t, run_t, kind, n_classes, num_tasks, chunks = ARMS[args.arm]
    T = num_tasks - 1
    root = args.ckpt_root.rstrip("/")
    ckpt_t = ckpt_t.replace("runs/", root + "/", 1)
    run_t = run_t.replace("runs/", root + "/", 1)
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 104)
    print(f"E25 B  arm={args.arm} kind={kind} tasks={num_tasks} probe seed {args.probe_seed} lam {LAM}")
    print("=" * 104)

    # ---- C-WIT: arm identity from each run's own artifact -----------------------
    for s in seeds:
        arm = json.load(open(f"{run_t.format(s=s)}/mafc_results.json"))["arm"]
        assert arm["use_task_heads"] is False and arm["era_checkpoints"] is True \
            and arm["use_input_adapters"] is False, arm
        assert arm["num_tasks"] == num_tasks, (arm["num_tasks"], num_tasks)
        if kind != "har_subject":
            assert arm.get("disjoint_content") is True, arm
        if chunks is not None:                       # E23-B's C-CONSTR
            assert arm.get("content_chunks") == chunks, (arm.get("content_chunks"), chunks)
    print(f"  C-WIT: shared head, era ON, adapters off, num_tasks={num_tasks}"
          + (f", content_chunks={chunks}" if chunks else "") + " -- from the artifacts")

    # ---- construction + relay ---------------------------------------------------
    if kind == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        har = HARSubjectBenchmark(num_tasks=num_tasks, root=".", batch_size=BATCH)
        pfp = har.partition_fingerprint()
        print(f"  har_subject partition {pfp} -> {'AS-EXECUTED' if pfp == HAR_PARTITION else 'STOP'}")
        if pfp != HAR_PARTITION:
            return 1
        maps = har_maps(har._sd)
        hdata = load_task_data(har, n_tasks=num_tasks, probe_subset_seed=args.probe_seed)
        data_for_seed = lambda _s: (hdata, maps)
        relay = har_relayout
    elif chunks is not None:
        from src.data.permuted_mnist import PermutedMNISTBenchmark

        def data_for_seed(seed):
            bench = PermutedMNISTBenchmark(num_tasks=num_tasks, batch_size=BATCH, seed=seed,
                                           disjoint_content=True, content_chunks=chunks)
            a = json.load(open(f"{run_t.format(s=seed)}/mafc_results.json"))["arm"]
            assert a["construction_fingerprint"] == bench.construction_fingerprint(), \
                (a["construction_fingerprint"], bench.construction_fingerprint())
            return (load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=args.probe_seed),
                    bench.permutations)
        relay = mnist_relayout
    else:
        def data_for_seed(seed):
            bench = E18_FACTORY[kind](seed)
            fp = json.load(open(f"{run_t.format(s=seed)}/mafc_results.json"))["arm"]["content_fingerprint"]
            assert fp == bench.content_fingerprint(), (fp, bench.content_fingerprint())
            return (load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=args.probe_seed),
                    bench.angles if kind == "rotated" else bench.permutations)
        relay = rotated_relayout if kind == "rotated" else mnist_relayout

    lossy = (kind == "rotated")
    for s in seeds:
        data, maps = data_for_seed(s)
        d = ckpt_t.format(s=s)
        models = {}                      # theta_t, cached per seed

        def theta(t):
            if t not in models:
                models[t] = load(d, t)
            return models[t]

        cells, composed = [], {}
        for k in range(T):
            xtr, ytr, xte, yte = data[k]
            h = era_head_of(theta(k), k)
            dim = None
            Z_tr, Z_te, relay_err = {}, {}, {}
            for t in range(k, T + 1):
                xr_tr = relay(xtr, k, t, maps) if t != k else xtr
                xr_te = relay(xte, k, t, maps) if t != k else xte
                if lossy:                                   # sec B: the relay is not exact off 90 deg
                    back = relay(xr_te, t, k, maps) if t != k else xr_te
                    relay_err[t] = float((back - xte).abs().mean() / xte.abs().mean().clamp_min(1e-9))
                m = theta(t)
                g = assert_path_identity(m, xr_te, k, False, device, label=f"s{s}/k{k}/t{t}", verbose=False)
                assert g["p_b"], (s, k, t)
                f_tr, _ = features_and_logits(m, xr_tr, ytr, k, False, device)
                f_te, _ = features_and_logits(m, xr_te, yte, k, False, device)
                Z_tr[t] = f_tr.numpy().astype(np.float64)
                Z_te[t] = f_te.numpy().astype(np.float64)
                dim = Z_tr[t].shape[1]

            # ---- per-step fits --------------------------------------------------
            steps, R_chain, S_chain = [], [], []
            for t in range(k + 1, T + 1):
                A_tr, B_tr = Z_tr[t - 1], Z_tr[t]
                mu, sd = A_tr.mean(0), A_tr.std(0) + 1e-8          # source stats, as D1 uses era stats
                S = Affine.fit(A_tr, B_tr, mu, sd, LAM)            # drift  Z_{t-1} -> Z_t
                mu2, sd2 = B_tr.mean(0), B_tr.std(0) + 1e-8
                R = Affine.fit(B_tr, A_tr, mu2, sd2, LAM)          # repair Z_t -> Z_{t-1}
                # BOTH coordinate systems, because the composed map can only be read
                # in RAW: each step standardizes by its own source, so there is no
                # single standardized frame for a chain. `spectrum_std` is the
                # D1-comparable number (its spread 8-12 is standardized);
                # `spectrum_raw` is the one that compares to `spread_composed`.
                rec = {"t": t,
                       "residual_drift": S.residual(Z_te[t - 1], Z_te[t]),
                       "residual_repair": R.residual(Z_te[t], Z_te[t - 1]),
                       "spectrum": spectrum(S.M_std),
                       "spectrum_raw": spectrum(S.M_raw())}
                if lossy:
                    rec["relay_rel_err_src"] = relay_err[t - 1]
                    rec["relay_rel_err_dst"] = relay_err[t]
                steps.append(rec)
                S_chain.append(S)
                R_chain.append(R)

            # composed repair: apply R_T first, then R_{T-1}, ... , then R_{k+1}
            Rc = compose_affine(list(reversed(R_chain)), dim)
            Sc = compose_affine(S_chain, dim)
            # the basis trick is exact -- prove it on the live batch rather than assert it in prose
            z = Z_te[T]
            for f in reversed(R_chain):
                z = f.apply(z)
            comp_err = float(np.abs(z - apply_affine(Z_te[T], Rc)).max())
            assert comp_err < 1e-6, f"composed affine != chained apply: {comp_err:.3e}"
            composed[k] = Rc

            acc_deployed = head_acc(h, Z_te[T], yte)      # h_k reading final features, no repair
            acc_repaired = head_acc(h, apply_affine(Z_te[T], Rc), yte)
            acc_ceiling = head_acc(h, Z_te[k], yte)

            # ---- C-SHUF must-fail: shuffled pairing at every step ----------------
            rng = np.random.default_rng(SHUF_SEED + 100 * s + k)
            Rsh = []
            for t in range(k + 1, T + 1):
                A_tr, B_tr = Z_tr[t], Z_tr[t - 1]
                perm = rng.permutation(B_tr.shape[0])
                mu2, sd2 = A_tr.mean(0), A_tr.std(0) + 1e-8
                Rsh.append(Affine.fit(A_tr, B_tr[perm], mu2, sd2, LAM))
            acc_shuf = head_acc(h, apply_affine(Z_te[T], compose_affine(list(reversed(Rsh)), dim)), yte)

            # ---- C-PLUMB (printed, not scored) -----------------------------------
            mu0, sd0 = Z_tr[T].mean(0), Z_tr[T].std(0) + 1e-8
            plumb = float(np.linalg.norm(Affine.fit(Z_tr[T], Z_tr[T], mu0, sd0, LAM).M_raw()
                                         - np.eye(dim)))

            # ---- D1's single-shot fit, for the composition-consistency reading ----
            muk, sdk = Z_tr[k].mean(0), Z_tr[k].std(0) + 1e-8
            single = Affine.fit(Z_tr[T], Z_tr[k], muk, sdk, LAM)
            acc_single = head_acc(h, single.apply(Z_te[T]), yte)

            res_all = [st["residual_repair"] for st in steps]
            cell = {"seed": s, "task": k, "n_steps": len(steps), "dim": dim,
                    "n_test": int(yte.shape[0]), "input_sha": sha(xte),
                    "acc_deployed": acc_deployed, "acc_repaired_composed": acc_repaired,
                    "acc_repaired_single_shot": acc_single, "acc_ceiling": acc_ceiling,
                    "acc_shuffled": acc_shuf, "res95": se95(acc_ceiling, int(yte.shape[0])),
                    "residual_step_mean": float(np.mean(res_all)),
                    "residual_step_max": float(np.max(res_all)),
                    "residual_single_shot": single.residual(Z_te[T], Z_te[k]),
                    "spread_step_mean": float(np.mean([st["spectrum"]["spread"] for st in steps])),
                    "spread_step_mean_raw": float(np.mean([st["spectrum_raw"]["spread"] for st in steps])),
                    "spread_composed_raw": spectrum(Sc[0])["spread"],
                    "c_plumb_norm": plumb, "composed_vs_chain_maxabs": comp_err,
                    "steps": steps}
            if lossy:
                cell["relay_rel_err_by_t"] = {str(t): v for t, v in relay_err.items()}
            cells.append(cell)
            print(f"  s{s} k{k}: steps {len(steps):>2} | deployed {acc_deployed:.4f} -> composed "
                  f"{acc_repaired:.4f} (single-shot {acc_single:.4f}, ceiling {acc_ceiling:.4f}) | "
                  f"shuf {acc_shuf:.4f} | step residual {np.mean(res_all):.3f} vs single "
                  f"{single.residual(Z_te[T], Z_te[k]):.3f} | plumb {plumb:.2f}")

            # keep the final-frame test features for the cross-task read
            cell["_Z_T"] = Z_te[T]
            cell["_y"] = yte
            cell["_h"] = h

        # ---- repair SHARING: task j's composed repair applied to task k ----------
        # Stride-sample rather than truncate. The 20-task arm has 342 ordered pairs
        # and taking the first 12 would take k = 0 twelve times, turning a question
        # about SHARING ACROSS TASKS into a question about task 0.
        all_pairs = [(k, j) for k in range(T) for j in range(T) if j != k]
        stride = max(1, len(all_pairs) // args.max_cross)
        pairs = all_pairs[::stride][:args.max_cross]
        share = []
        by_k = {c["task"]: c for c in cells}
        for k, j in pairs:
            ck = by_k[k]
            a = head_acc(ck["_h"], apply_affine(ck["_Z_T"], composed[j]), ck["_y"])
            share.append({"k": k, "j": j, "gap": abs(k - j), "acc_cross": a,
                          "acc_own": ck["acc_repaired_composed"],
                          "loss": ck["acc_repaired_composed"] - a, "res95": ck["res95"]})
        if share:
            losses = [p["loss"] for p in share]
            within = sum(abs(p["loss"]) <= p["res95"] for p in share)
            print(f"  s{s} SHARING: {len(share)} ordered pairs | own - cross mean "
                  f"{np.mean(losses):+.4f} [{min(losses):+.4f}, {max(losses):+.4f}] | "
                  f"within resolution {within}/{len(share)}")

        for c in cells:                                  # drop the tensors before serializing
            for kk in ("_Z_T", "_y", "_h"):
                c.pop(kk, None)
        out = {"arm": args.arm, "kind": kind, "seed": s, "num_tasks": num_tasks,
               "content_chunks": chunks, "lam": LAM, "probe_subset_seed": args.probe_seed,
               "residual_bar": RESIDUAL_BAR, "shuf_seed": SHUF_SEED,
               "composition": "chained Affine.apply, raw coordinates with intercept; "
                              "composed affine recovered exactly and asserted against the chain",
               "cells": cells, "sharing": share}
        p = os.path.join(args.out_dir, f"steps_seed{s}.json")
        json.dump(out, open(p, "w"), indent=2, default=float)
        print(f"  wrote {p}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
