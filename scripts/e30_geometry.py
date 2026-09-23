"""E30 — does class geometry survive drift in ORDER? (docs contract v2)

The question: every label-free readout repair in this program failed because a
label-free objective cannot tell a correct readout from one with the classes
relabelled. A stored c x c matrix of distances between class means is side
information that is NOT a function of the current feature distribution, so it
could break that symmetry -- IF the RANKING of those distances survives drift
even where the distances themselves do not.

THREE STEPS, AND STEP 0 RUNS FIRST FOR A REASON.

  step 0  SELF-AMBIGUITY, from the era checkpoint alone, before any final
          feature is read. Split task k's train data in half, build D_A and
          D_B under theta_k, and count the relabellings sigma that align
          D_A to sigma(D_B) at least as well as the identity does. A cell is
          IDENTIFIABLE when that count is 1.
          This replaces v1's identity control, which compared D_era against
          ITSELF and therefore ranked the identity first BY CONSTRUCTION --
          a gate that can only fail where the matrix has exact automorphisms,
          which is precisely the case worth measuring (catch 25). Without
          step 0 a low exact rate is unattributable: "the geometry did not
          survive drift" and "the geometry was never identifiable" produce
          the same number.

  step 1  geometry WITH labels, the upper bound. Class means under theta_k
          and theta_T on the same task-k content, each standardized by its own
          checkpoint's task-k statistics so per-dimension scale change does
          not register as geometry. Spearman rho over the upper triangle, the
          best relabelling, and the true relabelling's RANK.

  step 2  the operational version, WITHOUT labels. k-means on theta_T features
          of unlabelled task-k TRAIN inputs; match cluster-mean distances to
          D_era; assign each cluster a class; score TEST inputs by nearest
          cluster mean. Purity printed beside, so a failure separates the
          clustering from the matching.

SIGN CONVENTION. `quadratic_assignment` MINIMISES tr(A^T P B P^T); matching two
matrices MAXIMISES it, so every call passes -B. Verified on the container: with
-B, `faq` recovers a planted permutation exactly at n = 5, 10, 20; with +B it
fails at every size, and `2opt` fails at 10 and 20 even with the right sign.

NO PARALLEL IMPLEMENTATIONS. The scratch arms come from `e27_defect.ARMS` and
load through `channel_decomp`; the pretrained arms load through
`e12_decompose` / `e14_decompose`, which are B1's own modules and the ones B6
borrows. Labels travel with features from the same pass in both paths, so
catch 32's misalignment cannot arise here by construction.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXHAUSTIVE_MAX_C = 6           # 720 permutations; 10! = 3.6M needs 2.9GB, so no
CONTROL_SEED = 20260922
KMEANS_N_INIT = 10

# Pretrained arms. (run dir, ckpt dir, decomposition module, n_tasks, n_classes)
PRE_ARMS = {
    "b1_vit": ("e12_base_seed{s}", "ckpt_e12_base_seed{s}/mafc_seed{s}", "scripts.e12_decompose", 20, 5),
    "b1_rn":  ("e14_base_seed{s}", "ckpt_e14_base_seed{s}/mafc_seed{s}", "scripts.e14_decompose", 20, 5),
    "b6_vit": ("e23_vit_seed{s}",  "ckpt_e23_vit_seed{s}/mafc_seed{s}",  "scripts.e12_decompose", 20, 5),
    "b6_rn":  ("e23_rn_seed{s}",   "ckpt_e23_rn_seed{s}/mafc_seed{s}",   "scripts.e14_decompose", 20, 5),
}


# ----------------------------------------------------------------- geometry --
def class_means(F: np.ndarray, y: np.ndarray, c: int):
    """[c, d] class means of STANDARDIZED features, and the per-class counts."""
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Z = (F - mu) / sd
    M = np.stack([Z[y == j].mean(0) if (y == j).sum() else np.full(Z.shape[1], np.nan)
                  for j in range(c)])
    return M, np.array([(y == j).sum() for j in range(c)])


def dmat(M: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=-1)
    return d


def score(D1: np.ndarray, D2: np.ndarray, sigma) -> float:
    """Alignment cost of relabelling D2 by sigma: lower is better."""
    s = np.asarray(sigma)
    return float(np.linalg.norm(D1 - D2[np.ix_(s, s)]))


def exhaustive(D1, D2, truth=None):
    """Every relabelling, scored. Returns best, the truth's rank, and the number
    scoring at least as well as the truth (the ambiguity, BY RANK not by bar)."""
    c = D1.shape[0]
    perms = list(itertools.permutations(range(c)))
    sc = np.array([score(D1, D2, p) for p in perms])
    order = np.argsort(sc, kind="stable")
    best = perms[order[0]]
    out = {"best": list(best), "score_best": float(sc[order[0]]),
           "n_perms": len(perms)}
    if truth is not None:
        t = tuple(truth)
        ti = perms.index(t)
        out["true_rank"] = int(np.where(order == ti)[0][0])
        out["ambiguity"] = int((sc <= sc[ti] + 1e-12).sum())
        out["exact"] = bool(best == t)
        out["score_truth"] = float(sc[ti])
    return out


def faq_assign(D1, D2, truth=None):
    """c = 10: one answer from `faq` (with -B), plus a one-transposition
    neighbourhood as a LOWER BOUND on ambiguity. A rank among 10! = 3,628,800
    would need ~2.9 GB of conjugated matrices, so it is not computed and no
    number pretends to be it."""
    from scipy.optimize import quadratic_assignment
    c = D1.shape[0]
    r = quadratic_assignment(D1, -D2, method="faq")
    best = tuple(int(v) for v in r.col_ind)
    s_best = score(D1, D2, best)
    nb = 0
    for i, j in itertools.combinations(range(c), 2):
        p = list(best); p[i], p[j] = p[j], p[i]
        if score(D1, D2, p) <= s_best + 1e-12:
            nb += 1
    out = {"best": list(best), "score_best": float(s_best), "n_perms": None,
           "neighbourhood_ties": nb, "neighbourhood_size": c * (c - 1) // 2,
           "true_rank": None, "ambiguity": None}
    if truth is not None:
        out["exact"] = bool(best == tuple(truth))
        out["score_truth"] = score(D1, D2, truth)
    return out


def match(D1, D2, c, truth=None):
    return exhaustive(D1, D2, truth) if c <= EXHAUSTIVE_MAX_C else faq_assign(D1, D2, truth)


def self_ambiguity(D_A, D_B, c):
    """Step 0. How many relabellings align the two HALVES of the era data at
    least as well as the identity does? 1 means identifiable."""
    ident = tuple(range(c))
    s_id = score(D_A, D_B, ident)
    if c <= EXHAUSTIVE_MAX_C:
        perms = list(itertools.permutations(range(c)))
        sc = np.array([score(D_A, D_B, p) for p in perms])
        n = int((sc <= s_id + 1e-12).sum())
        return {"self_ambiguity": n, "identifiable": bool(n == 1),
                "identity_score": float(s_id), "best_score": float(sc.min()),
                "exhaustive": True}
    nb = 0
    for i, j in itertools.combinations(range(c), 2):
        p = list(ident); p[i], p[j] = p[j], p[i]
        if score(D_A, D_B, p) <= s_id + 1e-12:
            nb += 1
    return {"self_ambiguity": None, "identifiable": bool(nb == 0),
            "neighbourhood_ties": nb, "identity_score": float(s_id),
            "exhaustive": False}


# ----------------------------------------------------------------- controls --
def solver_control(out):
    """Runs before any cell. The SAME call with +B must fail, or the sign
    convention is not being exercised and the control proves nothing."""
    from scipy.optimize import quadratic_assignment
    rng = np.random.default_rng(CONTROL_SEED)
    rec = {}
    for n in (5, 10):
        A = rng.normal(size=(n, n)); A = A @ A.T
        P = rng.permutation(n); B = A[np.ix_(P, P)]
        good = quadratic_assignment(A, -B, method="faq")
        bad = quadratic_assignment(A, B, method="faq")
        eg = float(np.abs(A - B[np.ix_(good.col_ind, good.col_ind)]).max())
        eb = float(np.abs(A - B[np.ix_(bad.col_ind, bad.col_ind)]).max())
        rec[n] = {"minus_B_err": eg, "plus_B_err": eb,
                  "minus_B_exact": bool(eg < 1e-8), "plus_B_fails": bool(eb > 1e-8)}
        print(f"    solver n={n}: -B err {eg:.2e} ({'EXACT' if eg < 1e-8 else 'FAIL'}) | "
              f"+B err {eb:.2e} ({'fails as it must' if eb > 1e-8 else 'DID NOT FAIL'})")
    rec["pass"] = all(v["minus_B_exact"] and v["plus_B_fails"] for v in rec.values()
                      if isinstance(v, dict))
    out["solver_control"] = rec
    return rec["pass"]


def positive_synthetic(out, c=5):
    """A planted configuration under a random MONOTONE distortion of distances
    plus noise must be recovered exactly; under an ORDER-DESTROYING distortion
    it must not. Tests that the matcher responds to order and only to order."""
    rng = np.random.default_rng(CONTROL_SEED + 1)
    M = rng.normal(size=(c, 12))
    D = dmat(M)
    iu = np.triu_indices(c, 1)
    mono = D.copy()
    v = D[iu]
    mono[iu] = v ** 1.7 + 0.01 * rng.normal(size=v.shape)      # monotone + noise
    mono = np.triu(mono, 1); mono = mono + mono.T
    destroy = D.copy()
    destroy[iu] = rng.permutation(v)                            # order destroyed
    destroy = np.triu(destroy, 1); destroy = destroy + destroy.T
    truth = tuple(range(c))
    r_mono = match(D, mono, c, truth)
    r_dest = match(D, destroy, c, truth)
    rec = {"monotone_exact": bool(r_mono.get("exact")),
           "order_destroyed_exact": bool(r_dest.get("exact")),
           "monotone_rank": r_mono.get("true_rank"),
           "destroyed_rank": r_dest.get("true_rank"),
           "pass": bool(r_mono.get("exact") and not r_dest.get("exact"))}
    print(f"    positive synthetic: monotone distortion exact={rec['monotone_exact']} | "
          f"order-destroyed exact={rec['order_destroyed_exact']} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}")
    out["positive_synthetic"] = rec
    return rec["pass"]


# --------------------------------------------------------------- extraction --
def pretrained_cells(arm, seed, root, device, data_root):
    """Yield (k, F_era, y_era, F_fin, y_fin, F_era_te, y_era_te, F_fin_te, y_fin_te)."""
    rundir, ckpt, modname, n_tasks, c = PRE_ARMS[arm]
    a = json.load(open(f"{root}/{rundir.format(s=seed)}/mafc_results.json"))["arm"]
    # E30's OWN arm gate: it accepts `cifar100` (B1) as well as the permuted
    # variant. `e23_decompose`'s assert is NOT widened -- that gate protects B6's
    # own runs and is not ours to relax.
    assert a["benchmark"] in ("cifar100", "cifar100_permuted"), a["benchmark"]
    assert a["use_task_heads"] and a["era_checkpoints"], a
    dec = importlib.import_module(modname)
    from src.data.split_cifar100 import SplitCIFAR100Benchmark
    bench = SplitCIFAR100Benchmark(
        num_tasks=n_tasks, batch_size=dec.BATCH, root=data_root, seed=seed,
        remap_labels=bool(a.get("remap_labels", True)),
        shift_mode="patch" if a["benchmark"] == "cifar100_permuted" else None,
        download=False)
    d = f"{root}/{ckpt.format(s=seed)}"
    T = n_tasks - 1
    m_fin = dec.load_era(d, T, device)
    meta = {"arm_record": a, "class_order_fingerprint": bench.class_order_fingerprint(),
            "shift_fingerprint": bench.shift_fingerprint(), "n_classes": c,
            "n_tasks": n_tasks, "module": modname}
    for k in range(T):
        tr, te = bench.get_task_loaders(k)
        m_era = dec.load_era(d, k, device)
        Fe, Le, Ye = dec.features_and_logits(m_era, tr, k, False, device)
        Fe_te, Le_te, Ye_te = dec.features_and_logits(m_era, te, k, False, device)
        Ff, Lf, Yf = dec.features_and_logits(m_fin, tr, k, False, device)
        Ff_te, Lf_te, Yf_te = dec.features_and_logits(m_fin, te, k, False, device)
        del m_era
        assert torch.equal(Ye, Yf) and torch.equal(Ye_te, Yf_te), (
            "labels differ between the two checkpoints' passes over the same loader")
        yield k, (Fe.numpy(), Ye.numpy(), Ff.numpy(), Fe_te.numpy(), Ye_te.numpy(),
                  Ff_te.numpy(), float((Lf_te.argmax(1) == Yf_te).float().mean()))
    del m_fin
    yield "meta", meta


def scratch_cells(arm, seed, root, device, probe_seed):
    from scripts.e27_defect import ARMS as E27_ARMS
    from scripts.channel_decomp import (load, features_and_logits, load_task_data,
                                        assert_path_identity)
    from scripts.cure_screen import BATCH
    ck_t, run_t, kind, c, n_tasks, chunks, _bk = E27_ARMS[arm]
    a = json.load(open(f"{root}/{run_t.format(s=seed)}/mafc_results.json"))["arm"]
    if kind == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        bench = HARSubjectBenchmark(num_tasks=n_tasks, root=".", batch_size=BATCH)
        assert bench.partition_fingerprint() == "1104af185c87", bench.partition_fingerprint()
    elif chunks is not None:
        from src.data.permuted_mnist import PermutedMNISTBenchmark
        bench = PermutedMNISTBenchmark(num_tasks=n_tasks, batch_size=BATCH, seed=seed,
                                       disjoint_content=True, content_chunks=chunks)
    else:
        from scripts.e27_defect import E18_FACTORY
        bench = E18_FACTORY[kind](seed)
    data = load_task_data(bench, n_tasks=n_tasks, probe_subset_seed=probe_seed)
    # E27's templates are rooted at "runs/"; re-root them at --ckpt-root so the
    # same registry serves both the laptop and the container's /runs mount.
    ck = ck_t.format(s=seed)
    ck = f"{root}/{ck[5:]}" if ck.startswith("runs/") else f"{root}/{ck}"
    T = n_tasks - 1
    ep = int(a.get("epochs_per_task", 10)) - 1
    m_fin = load(ck, T, epoch=ep)
    meta = {"arm_record": a, "n_classes": c, "n_tasks": n_tasks, "kind": kind}
    for k in range(T):
        xtr, ytr, xte, yte = data[k]
        g = assert_path_identity(m_fin, xte, k, False, device,
                                 label=f"e30/{arm}/s{seed}/T{k}", verbose=False)
        assert g["p_b"], (arm, seed, k)
        m_era = load(ck, k, epoch=ep)
        Fe, _ = features_and_logits(m_era, xtr, ytr, k, False, device)
        Fe_te, _ = features_and_logits(m_era, xte, yte, k, False, device)
        Ff, _ = features_and_logits(m_fin, xtr, ytr, k, False, device)
        Ff_te, Lf_te = features_and_logits(m_fin, xte, yte, k, False, device)
        del m_era
        yield k, (Fe.numpy(), ytr.numpy(), Ff.numpy(), Fe_te.numpy(), yte.numpy(),
                  Ff_te.numpy(), float((Lf_te.argmax(1) == yte).float().mean()))
    del m_fin
    yield "meta", meta


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="E30 class geometry")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--probe-seed", type=int, default=20260916)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=None, help="smoke only")
    args = ap.parse_args()
    device = torch.device(args.device)
    root = args.ckpt_root.rstrip("/")
    t0 = time.time()
    print("=" * 112 + f"\nE30 GEOMETRY  arm={args.arm} seed={args.seed} device={device}\n" + "=" * 112)

    ctl = {}
    print("  CONTROLS (before any cell)")
    if not solver_control(ctl):
        print("  SOLVER CONTROL FAILED — no cell is read"); return 1
    if not positive_synthetic(ctl):
        print("  POSITIVE SYNTHETIC FAILED — no cell is read"); return 1

    gen = (pretrained_cells(args.arm, args.seed, root, device, args.data_root)
           if args.arm in PRE_ARMS else
           scratch_cells(args.arm, args.seed, root, device, args.probe_seed))
    # THE CLASS COUNT COMES FROM THE ARM REGISTRY, resolved once before any cell.
    # A first draft derived it inside the loop with `6 if "har" in arm else 10`,
    # which is a substring test standing in for a recorded field -- the same
    # cheap-signal substitution catch 30 is about, one level down. `e23_har_mlp`
    # would have passed it by luck and any future arm named otherwise would not.
    if args.arm in PRE_ARMS:
        c = PRE_ARMS[args.arm][4]
    else:
        from scripts.e27_defect import ARMS as _E27
        c = _E27[args.arm][3]
    print(f"  classes c = {c} (from the arm registry)")
    rng = np.random.default_rng(CONTROL_SEED + args.seed)
    rows, meta = [], None
    from sklearn.cluster import KMeans
    for k, payload in gen:
        if k == "meta":
            meta = payload; break
        if args.max_tasks is not None and k >= args.max_tasks:
            continue
        Fe, y, Ff, Fe_te, y_te, Ff_te, deployed = payload

        # ---- step 0: self-ambiguity, BEFORE any final-checkpoint geometry ----
        idx = rng.permutation(len(y)); half = len(y) // 2
        MA, _ = class_means(Fe[idx[:half]], y[idx[:half]], c)
        MB, _ = class_means(Fe[idx[half:]], y[idx[half:]], c)
        DA, DB = dmat(MA), dmat(MB)
        sa = self_ambiguity(DA, DB, c)
        iu = np.triu_indices(c, 1)
        sa["split_half_rho"] = float(spearmanr(DA[iu], DB[iu]).statistic)

        # ---- step 1: geometry with labels ------------------------------------
        Me, _ = class_means(Fe, y, c)
        Mf, _ = class_means(Ff, y, c)
        De, Df = dmat(Me), dmat(Mf)
        truth = tuple(range(c))
        m1 = match(De, Df, c, truth)
        rho = float(spearmanr(De[iu], Df[iu]).statistic)

        # ---- must-fail: era means from SHUFFLED labels -----------------------
        y_sh = rng.permutation(y)
        Ms, _ = class_means(Fe, y_sh, c)
        m_mf = match(dmat(Ms), Df, c, truth)

        # ---- step 2: unlabelled, k-means on final TRAIN, scored on TEST ------
        muf, sdf = Ff.mean(0), Ff.std(0) + 1e-8
        Zf, Zf_te = (Ff - muf) / sdf, (Ff_te - muf) / sdf
        km = KMeans(n_clusters=c, n_init=KMEANS_N_INIT, random_state=args.seed).fit(Zf)
        Dc = dmat(km.cluster_centers_)
        m2 = match(De, Dc, c, None)
        assign = np.asarray(m2["best"])          # cluster j -> class assign[j]
        lab_te = km.predict(Zf_te)
        pseudo = float((assign[lab_te] == y_te).mean())
        # purity: the best any assignment of THESE clusters could reach (labels used)
        best_map = np.array([np.bincount(y_te[lab_te == j], minlength=c).argmax()
                             if (lab_te == j).sum() else 0 for j in range(c)])
        purity = float((best_map[lab_te] == y_te).mean())

        rows.append({"task": k, "c": c, "n_train": int(len(y)), "n_test": int(len(y_te)),
                     "deployed_acc": deployed, "step0": sa, "spearman_rho": rho,
                     "step1": m1, "must_fail": m_mf,
                     "step2": {"pseudo_label_acc_test": pseudo, "purity_test": purity,
                               "ratio": pseudo / max(purity, 1e-9),
                               "assignment": [int(v) for v in assign]}})
        print(f"  k={k:>2}: ident={sa['identifiable']} (self-amb "
              f"{sa.get('self_ambiguity') or sa.get('neighbourhood_ties')}) "
              f"split-half rho {sa['split_half_rho']:+.3f} | rho {rho:+.3f} "
              f"exact={m1.get('exact')} rank={m1.get('true_rank')} | "
              f"pseudo {pseudo:.3f} purity {purity:.3f}")

    ident = [r for r in rows if r["step0"]["identifiable"]]
    ex_id = [r for r in ident if r["step1"].get("exact")]
    ex_all = [r for r in rows if r["step1"].get("exact")]
    summ = {"n_cells": len(rows), "n_identifiable": len(ident),
            "exact_over_identifiable": (len(ex_id) / len(ident)) if ident else None,
            "exact_over_all": (len(ex_all) / len(rows)) if rows else None,
            "median_rho": float(np.median([r["spearman_rho"] for r in rows])) if rows else None,
            "median_split_half_rho": float(np.median([r["step0"]["split_half_rho"]
                                                      for r in rows])) if rows else None,
            "must_fail_exact": int(sum(bool(r["must_fail"].get("exact")) for r in rows)),
            "mean_pseudo_over_purity": float(np.mean([r["step2"]["ratio"] for r in rows]))
            if rows else None,
            "min_pseudo_over_purity": float(np.min([r["step2"]["ratio"] for r in rows]))
            if rows else None}
    print(f"\n  SUMMARY  identifiable {summ['n_identifiable']}/{summ['n_cells']} | "
          f"exact over identifiable {summ['exact_over_identifiable']} | "
          f"exact over all {summ['exact_over_all']} | median rho {summ['median_rho']} | "
          f"must-fail exact {summ['must_fail_exact']}/{summ['n_cells']}")
    out = {"arm": args.arm, "seed": args.seed, "meta": meta, "controls": ctl,
           "summary": summ, "rows": rows, "seconds": round(time.time() - t0, 1)}
    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"seed{args.seed}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"  wrote {fp}  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
