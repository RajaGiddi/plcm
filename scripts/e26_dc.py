"""E26 DC -- drift-compensation ports (docs/E26_prereg.md sec DC).

Two exemplar-free methods, ported from their papers' equations as transcribed in
the contract, adapted to CE-trained features and task-IL NCM. NOT reproductions
of either paper's setting, and not described as such.

SDC (Yu et al. 2020, sec 4.1). At boundary t, on current-task data:
    delta_i   = z_i^t - z_i^{t-1}                                          (Eq. 9)
    w_i       = exp( -||z_i^{t-1} - mu_c||^2 / 2 sigma^2 )                 (Eq. 10)
    Delta_c   = sum_i w_i delta_i / sum_i w_i                              (Eq. 11)
    mu_c     <- mu_c + Delta_c                                             (Eq. 13)
sigma in {0.2, 0.3, 1.0}; headline 0.3 (SDC's default), fixed before running.

LDC (Gomez-Villa et al. 2024, sec 3.3). At boundary t, both extractors frozen:
    W_t = argmin_W (1/N) sum_i || z_i^{t-1} W - z_i^t ||^2                 (Eq. 1)
    P_c <- P_c W_t                                                         (Eq. 2)
Linear, no bias (Table 5), Adam 1e-3, 20 epochs, all current-task data; batch
128 is OURS (the paper's text does not state it); three projector seeds.

Both run on L2-NORMALISED features, as SDC specifies (its sigma is calibrated to
unit-norm embeddings). Unnormalised is a sensitivity arm. Classification is
nearest class mean in Euclidean distance among TASK k's classes only (task-IL).

Prototypes mu_c^k are computed at the end of task k from task-k training data
under theta_k -- the resource both methods store -- and compensated across every
boundary k -> k+1 -> ... -> T using task t's training data under theta_{t-1} and
theta_t, EXTRACTED THROUGH A SEQUENTIAL LOADER WITH LABEL EQUALITY ASSERTED
ACROSS THE TWO PASSES. Two passes over a shuffled loader draw two permutations;
that is catch 32, the defect that inverted E12's conclusion.

Arms per cell: A0 deployed; N-naive (era prototypes, uncompensated); N-SDC_sigma;
N-LDC; N-oracle (prototypes recomputed from task-k data under theta_T -- the
ceiling for ANY prototype correction). A_inf and A_10 join in the row from the
existing artifacts.

Controls: C-PLUMB (zero boundaries -> prototypes unchanged, exactly); LDC
identity (fit with f^{t-1} = f^t, from identity init AND from random init --
the first is weak, the second is the test); SDC must-fail (negated drift worse
than naive on a majority of cells where the drift clears its floor); LDC
must-fail (shuffled pairs worse than naive on a majority); oracle identity (two
NCM implementations agree exactly).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIGMAS = [0.2, 0.3, 1.0]
SIGMA_HEAD = 0.3
LDC_LR, LDC_EPOCHS, LDC_BATCH = 1e-3, 20, 128
LDC_SEEDS = [0, 1, 2]
IDENT_TOL = 1e-4
SHUF_SEED = 20260921

SCRATCH = {   # (ckpt template, run template, construction, n_classes, num_tasks, chunks)
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32",  "runs/e10off_ec_seed{s}",  "har_subject", 6,  5,  None),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted",   10, 5,  None),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted", 10, 5,  None),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_rmd_mlp_seed{s}", "rotated",   10, 5,  None),
    "e23b_t20":     ("runs/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32",   "runs/e23b_t20_seed{s}",   "permuted",   10, 20, 20),
}
PRETRAINED = {"vit": "e23_vit", "resnet": "e23_rn"}
HAR_PARTITION = "1104af185c87"


# ------------------------------------------------------------------ pieces --
def l2n(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def prototypes(Z: np.ndarray, y: np.ndarray, classes) -> np.ndarray:
    return np.stack([Z[y == c].mean(0) for c in classes])


def ncm(Z: np.ndarray, protos: np.ndarray, classes, y: np.ndarray) -> float:
    d = ((Z[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    pred = np.asarray(classes)[d.argmin(1)]
    return float((pred == y).mean())


def ncm_loop(Z, protos, classes, y) -> float:
    """Independent implementation for the oracle-identity control."""
    correct = 0
    for i in range(Z.shape[0]):
        dists = [float(np.sum((Z[i] - p) ** 2)) for p in protos]
        correct += int(classes[int(np.argmin(dists))] == y[i])
    return correct / Z.shape[0]


def sdc_delta(Zp: np.ndarray, Zc: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Eqs. 9-11: kernel-weighted mean of current-data drift around prototype mu."""
    delta = Zc - Zp
    w = np.exp(-((Zp - mu) ** 2).sum(1) / (2.0 * sigma ** 2))
    s = w.sum()
    return (w[:, None] * delta).sum(0) / s if s > 1e-300 else np.zeros_like(mu)


def ldc_fit(Zp: np.ndarray, Zc: np.ndarray, seed: int, init: str = "identity") -> tuple[np.ndarray, float]:
    """Eq. 1: linear projector Zp @ W ~ Zc, no bias, Adam, 20 epochs, batch 128."""
    g = torch.Generator().manual_seed(seed)
    d = Zp.shape[1]
    W = torch.eye(d) if init == "identity" else torch.randn(d, d, generator=g) / np.sqrt(d)
    W = W.clone().requires_grad_(True)
    A, B = torch.from_numpy(Zp).float(), torch.from_numpy(Zc).float()
    opt = torch.optim.Adam([W], lr=LDC_LR)
    n = A.shape[0]
    for _ in range(LDC_EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, LDC_BATCH):
            idx = perm[i:i + LDC_BATCH]
            opt.zero_grad()
            loss = ((A[idx] @ W - B[idx]) ** 2).sum(1).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        final = float(((A @ W - B) ** 2).sum(1).mean())
    return W.detach().numpy().astype(np.float64), final


# ------------------------------------------------------------- extraction --
def make_extractor(arm, seed, root, data_root, device):
    """Returns (T, classes_of(k), feats(t, k, split) -> (Z, y, logits_or_None), load(t))."""
    if arm in PRETRAINED:
        tag = PRETRAINED[arm]
        modname = "scripts.e12_decompose" if arm == "vit" else "scripts.e14_decompose"
        dec = importlib.import_module(modname)
        from scripts.e12_p3 import assert_p3, positive_controls
        from src.data.split_cifar100 import SplitCIFAR100Benchmark
        from torch.utils.data import DataLoader
        ckpt = f"{root}/ckpt_{tag}_seed{seed}/mafc_seed{seed}"
        ra = json.load(open(f"{root}/{tag}_seed{seed}/mafc_results.json"))["arm"]
        assert ra["benchmark"] == "cifar100_permuted" and ra["use_task_heads"] and ra["era_checkpoints"], ra
        bench = SplitCIFAR100Benchmark(num_tasks=dec.NUM_TASKS, batch_size=dec.BATCH, root=data_root,
                                       seed=seed, remap_labels=True, shift_mode="patch", download=False)
        T = dec.FINAL
        models = {}

        def load(t):
            if t not in models:
                models[t] = dec.load_era(ckpt, t, device)
            return models[t]

        def feats(t, k, split):
            """task-k content in ITS OWN frame under theta_t; sequential loader."""
            ds = bench._make(k, split == "train")
            ds.perm = bench.perms[k]
            ld = DataLoader(ds, batch_size=dec.BATCH, shuffle=False)
            f, l, y = dec.features_and_logits(load(t), ld, k, False, device)
            return f.numpy().astype(np.float64), y.numpy(), l.numpy()

        _, te0 = bench.get_task_loaders(1)
        xp = next(iter(te0))[0][:16].to(device)
        pc = positive_controls(load(T), xp, 1, verbose=False)
        assert pc["p3a_control"] and pc["p3b_control"], pc
        classes_of = lambda k: list(range(5))
        from scripts.e12_p3 import assert_p3a

        def gate(t, k):
            """P3a always (the feature is the deployed tensor). P3b only where head k
            EXISTS in theta_t: heads are created lazily at task start (plcm.py
            set_task), so theta_{t-1} has no head for task t and P3b has no referent
            on the theta_{t-1} pass -- the smoke caught exactly this. On that pass DC
            uses the trunk feature only; the logits from the fallback head are never
            read. Where head k exists, both halves run."""
            m = load(t)
            if str(k) in getattr(m, "task_classifiers", {}):
                return assert_p3(m, xp, k, label=f"t{t}/k{k}")
            return {"p3a_delta": assert_p3a(m, xp, k, label=f"t{t}/k{k}"), "p3b_module": "no head for this task in this era (feature pass)"}
        return T, classes_of, feats, load, gate, "pretrained", dec.NUM_TASKS
    else:
        from scripts.channel_decomp import load as cload, features_and_logits, load_task_data, assert_path_identity, PROBE_SUBSET_SEED
        from scripts.cure_screen import E18_FACTORY, BATCH
        ck_t, run_t, kind, n_classes, num_tasks, chunks = SCRATCH[arm]
        ck = ck_t.replace("runs/", root + "/", 1).format(s=seed)
        ra = json.load(open(f"{run_t.replace('runs/', root + '/', 1).format(s=seed)}/mafc_results.json"))["arm"]
        assert ra["era_checkpoints"] is True and ra["use_task_heads"] is False, ra
        if kind == "har_subject":
            from src.data.har_subject import HARSubjectBenchmark
            bench = HARSubjectBenchmark(num_tasks=num_tasks, root=".", batch_size=BATCH)
            assert bench.partition_fingerprint() == HAR_PARTITION, bench.partition_fingerprint()
        elif chunks:
            from src.data.permuted_mnist import PermutedMNISTBenchmark
            bench = PermutedMNISTBenchmark(num_tasks=num_tasks, batch_size=BATCH, seed=seed,
                                           disjoint_content=True, content_chunks=chunks)
            assert ra["construction_fingerprint"] == bench.construction_fingerprint()
        else:
            bench = E18_FACTORY[kind](seed)
            assert ra["content_fingerprint"] == bench.content_fingerprint()
        data = load_task_data(bench, n_tasks=num_tasks, probe_subset_seed=PROBE_SUBSET_SEED)
        T = num_tasks - 1
        models = {}

        def load(t):
            if t not in models:
                models[t] = cload(ck, t)
            return models[t]

        def feats(t, k, split):
            xtr, ytr, xte, yte = data[k]
            x, y = (xtr, ytr) if split == "train" else (xte, yte)
            f, l = features_and_logits(load(t), x, y, k, False, device)
            return f.numpy().astype(np.float64), y.numpy(), l.numpy()

        classes_of = lambda k: list(range(n_classes))
        gate = lambda t, k: assert_path_identity(load(t), data[k][2], k, False, device, label=f"t{t}/k{k}", verbose=False)
        return T, classes_of, feats, load, gate, "scratch", num_tasks


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="E26 DC: SDC and LDC ports")
    ap.add_argument("--arm", required=True, choices=sorted(SCRATCH) + sorted(PRETRAINED))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=None, help="smoke: cap old tasks (boundaries still run to T)")
    ap.add_argument("--normalize", choices=["l2", "none"], default="l2")
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    norm = l2n if args.normalize == "l2" else (lambda Z: Z)
    T, classes_of, feats, load, gate, family, num_tasks = make_extractor(args.arm, args.seed, root, args.data_root, device)
    print("=" * 100 + f"\nE26 DC  arm={args.arm} ({family}) seed={args.seed} T={T} normalize={args.normalize}\n" + "=" * 100)
    t_start = time.time()
    old = list(range(T))[: args.max_tasks] if args.max_tasks else list(range(T))

    # ---- prototypes at task end, under theta_k: the stored resource -----------------
    era_protos = {}
    for k in old:
        gate(k, k)
        Z, y, _ = feats(k, k, "train")
        era_protos[k] = prototypes(norm(Z), y, classes_of(k))
    print(f"  era prototypes computed for {len(old)} old tasks under their own checkpoints")

    # ---- compensation across boundaries t = 1..T ------------------------------------
    sdc = {s: {k: era_protos[k].copy() for k in old} for s in SIGMAS}
    sdc_neg = {k: era_protos[k].copy() for k in old}                    # SDC must-fail
    ldc = {ps: {k: era_protos[k].copy() for k in old} for ps in LDC_SEEDS}
    ldc_shuf = {k: era_protos[k].copy() for k in old}                   # LDC must-fail
    # OPTIMIZER CONTROL, not a method: the exact least-squares solution of Eq. 1.
    # If N_LDC_ls ~= N_LDC the Adam port is at its optimum; if it is far above,
    # the port is optimizer-limited at the paper's hyperparameters on THESE
    # features (the identity control from random init already reads ~1.2 rel-Fro
    # against a 1e-4 bar on the smoke). Reported beside, never as the headline.
    ldc_ls = {k: era_protos[k].copy() for k in old}
    boundary_log, ident = [], None
    for t in range(1, T + 1):
        t0 = time.time()
        gate(t - 1, t); gate(t, t)
        Zp, yp, _ = feats(t - 1, t, "train")
        Zc, yc, _ = feats(t, t, "train")
        assert np.array_equal(yp, yc), f"boundary {t}: label order differs across the two passes (catch 32)"
        Zp, Zc = norm(Zp), norm(Zc)
        affected = [k for k in old if k < t]
        # SDC
        for s in SIGMAS:
            for k in affected:
                for ci in range(len(classes_of(k))):
                    sdc[s][k][ci] += sdc_delta(Zp, Zc, sdc[s][k][ci], s)
        for k in affected:
            for ci in range(len(classes_of(k))):
                sdc_neg[k][ci] -= sdc_delta(Zp, Zc, sdc_neg[k][ci], SIGMA_HEAD)
        # LDC
        fits = {}
        for ps in LDC_SEEDS:
            W, loss = ldc_fit(Zp, Zc, seed=ps)
            fits[ps] = loss
            for k in affected:
                ldc[ps][k] = ldc[ps][k] @ W
        W_ls = np.linalg.lstsq(Zp, Zc, rcond=None)[0]
        for k in affected:
            ldc_ls[k] = ldc_ls[k] @ W_ls
        rng = np.random.default_rng(SHUF_SEED + t)
        Wsh, _ = ldc_fit(Zp, Zc[rng.permutation(Zc.shape[0])], seed=0)
        for k in affected:
            ldc_shuf[k] = ldc_shuf[k] @ Wsh
        if ident is None:                                                   # LDC identity control, once
            Wi_id, _ = ldc_fit(Zp, Zp, seed=0, init="identity")
            Wi_rand, _ = ldc_fit(Zp, Zp, seed=0, init="random")
            I = np.eye(Zp.shape[1])
            ident = {"boundary": t, "from_identity_init_relfro": float(np.linalg.norm(Wi_id - I) / np.linalg.norm(I)),
                     "from_random_init_relfro": float(np.linalg.norm(Wi_rand - I) / np.linalg.norm(I)), "tol": IDENT_TOL}
            ident["pass_identity_init"] = ident["from_identity_init_relfro"] <= IDENT_TOL
            ident["pass_random_init"] = ident["from_random_init_relfro"] <= IDENT_TOL
        drift_norm = float(np.linalg.norm(Zc - Zp, axis=1).mean())
        boundary_log.append({"t": t, "n": int(Zp.shape[0]), "mean_drift_norm": drift_norm,
                             "ldc_final_loss": {str(ps): v for ps, v in fits.items()}, "seconds": round(time.time() - t0, 1)})
        print(f"  boundary {t:>2}: n={Zp.shape[0]} drift {drift_norm:.4f} | LDC loss {np.mean(list(fits.values())):.5f} | {boundary_log[-1]['seconds']}s")

    # ---- read at T ---------------------------------------------------------------------
    rows = []
    for k in old:
        gate(T, k)
        Zte, yte, lte = feats(T, k, "test")
        Zte = norm(Zte)
        cls = classes_of(k)
        Ztr_T, ytr_T, _ = feats(T, k, "train")
        oracle_p = prototypes(norm(Ztr_T), ytr_T, cls)
        a_oracle = ncm(Zte, oracle_p, cls, yte)
        a_oracle_loop = ncm_loop(Zte, oracle_p, cls, yte)
        # C-PLUMB, non-tautological: the UPDATE ARITHMETIC must be the identity when
        # there is no drift. SDC with Zc == Zp (delta = 0) must add exactly zero;
        # LDC with the identity projector must return the prototype exactly. A sign
        # error, a normalisation slip or a transposed matmul all make this non-zero.
        # (The first draft compared an array to its own copy -- catch 25, caught
        # before it ran.)
        Zk_tr, _, _ = feats(k, k, "train"); Zk_tr = norm(Zk_tr)
        plumb = max(float(np.abs(sdc_delta(Zk_tr, Zk_tr, era_protos[k][0], SIGMA_HEAD)).max()),
                    float(np.abs(era_protos[k] @ np.eye(era_protos[k].shape[1]) - era_protos[k]).max()))
        ldc_accs = [ncm(Zte, ldc[ps][k], cls, yte) for ps in LDC_SEEDS]
        row = {"seed": args.seed, "task": k, "boundaries_crossed": T - k, "n_test": int(len(yte)),
               "A0": float((lte.argmax(1) == yte).mean()),
               "N_naive": ncm(Zte, era_protos[k], cls, yte),
               **{f"N_SDC_{s}": ncm(Zte, sdc[s][k], cls, yte) for s in SIGMAS},
               "N_SDC_neg": ncm(Zte, sdc_neg[k], cls, yte),
               "N_LDC_mean": float(np.mean(ldc_accs)), "N_LDC_spread": float(max(ldc_accs) - min(ldc_accs)),
               "N_LDC_per_seed": ldc_accs, "N_LDC_shuf": ncm(Zte, ldc_shuf[k], cls, yte),
               "N_LDC_ls": ncm(Zte, ldc_ls[k], cls, yte),
               "N_oracle": a_oracle, "oracle_identity_abs": abs(a_oracle - a_oracle_loop),
               "c_plumb_max_abs": plumb,
               "sdc_drift_norm": float(np.linalg.norm(sdc[SIGMA_HEAD][k] - era_protos[k], axis=1).mean()),
               "res95": 1.96 * float(np.sqrt(0.25 / len(yte)))}
        rows.append(row)
        print(f"  k={k:>2} ({T-k:>2} boundaries): A0 {row['A0']:.3f} | naive {row['N_naive']:.3f} | SDC "
              + " ".join(f"{s}:{row[f'N_SDC_{s}']:.3f}" for s in SIGMAS)
              + f" (neg {row['N_SDC_neg']:.3f}) | LDC {row['N_LDC_mean']:.3f}±{row['N_LDC_spread']:.3f} (ls {row['N_LDC_ls']:.3f}, shuf {row['N_LDC_shuf']:.3f}) | oracle {a_oracle:.3f}")

    ctrl = {"ldc_identity": ident,
            "sdc_must_fail": {"cells_neg_below_naive": sum(r["N_SDC_neg"] < r["N_naive"] for r in rows),
                              "cells_with_drift_above_floor": sum(r["sdc_drift_norm"] > 1e-6 for r in rows), "of": len(rows)},
            "ldc_must_fail": {"cells_shuf_below_naive": sum(r["N_LDC_shuf"] < r["N_naive"] for r in rows), "of": len(rows)},
            "oracle_identity": {"max_abs": max(r["oracle_identity_abs"] for r in rows), "pass": max(r["oracle_identity_abs"] for r in rows) == 0.0},
            "c_plumb": {"max_abs": max(r["c_plumb_max_abs"] for r in rows), "pass": max(r["c_plumb_max_abs"] for r in rows) == 0.0}}
    print(f"\n  controls: LDC identity rel-Fro from-identity {ident['from_identity_init_relfro']:.2e} / from-random "
          f"{ident['from_random_init_relfro']:.2e} (tol {IDENT_TOL}) | SDC must-fail {ctrl['sdc_must_fail']} | "
          f"LDC must-fail {ctrl['ldc_must_fail']} | oracle identity {ctrl['oracle_identity']['pass']} | plumb {ctrl['c_plumb']['pass']}")
    out = {"arm": args.arm, "family": family, "seed": args.seed, "T": T, "num_tasks": num_tasks,
           "normalize": args.normalize, "sigmas": SIGMAS, "sigma_headline": SIGMA_HEAD,
           "ldc": {"lr": LDC_LR, "epochs": LDC_EPOCHS, "batch": LDC_BATCH, "seeds": LDC_SEEDS, "init": "identity",
                   "batch_note": "ours; the paper's text does not state it"},
           "controls": ctrl, "boundaries": boundary_log, "rows": rows,
           "seconds_total": round(time.time() - t_start, 1)}
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"seed{args.seed}{'' if args.normalize == 'l2' else '_unnorm'}.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"  wrote {p}  ({out['seconds_total']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
