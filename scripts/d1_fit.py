"""D1 sec 2 -- paired extraction and the linear drift fits (docs/D1_prereg.md, signed v2).

Per (arm, seed, old task k < T), on the audited loader's task-k train draw:

    Z_k   = f_{theta_k}(x_i)              x_i in FRAME k   (era encoder at home; the features h_k was fit on)
    Z_T   = f_{theta_T}(relay_T(x_i))     same samples in FRAME T (final encoder at home)      -- D1 v1 fed relay(x) to theta_k: wrong
    Z_T'  = f_{theta_T}(x_i)              x_i in frame k read by theta_T (deployed input; presentation + encoder drift)

    S  : Z_T  -> Z_k   REPAIR direction, fit directly (no inversion)      T  : Z_k -> Z_T   DRIFT direction (spectrum, projector algebra)
    S' : Z_T' -> Z_k                                                       T' : Z_k -> Z_T'

Fits are affine ridge regressions on features standardized with the ERA pass's
train statistics (mu, sd of Z_k), ridge lambda*n*I on the linear part, lambda
swept over {1e-4, 1e-3, 1e-2}; the linear part is stored in both standardized
and raw coordinates (M_raw = diag(1/sd) M~ diag(sd)). Everything that needs the
raw features is computed here and stored as scalars per cell: the transfer
matrices own/cross/deploy (sec 3 D1), spectra (D2), ||D_in||/||D_out|| under
h_k and W_T (D3), the re-laid F_enc (E23's (E1) on the scratch arms), and the
controls C-INPUT, C-PLUMB, C-SHUF, C-TARGET (sec 4). d1_diagnose.py reads the
scalars; it never touches features.

Path identity is asserted on every cell (catch 28); models come through
PLCM.load_era (catch 29); data through load_task_data with the recorded draw
(catch 32 / R2); acc_orig and acc_ceiling are C-ID'd against the seeded
decomposition (nothing read on failure).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,     # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import (E18_FACTORY, era_head_of, har_maps, har_relayout,  # noqa: E402
                                 mnist_relayout, rotated_relayout, BATCH)

SEEDS = [42, 1337, 2024]
CUR = 4
TOL = 1e-6
LAMS = [1e-4, 1e-3, 1e-2]
LAM_MAIN = 1e-3
SHUF_SEED = 20260919
HAR_PARTITION = "1104af185c87"
ARMS = {
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32", "runs/e10off_ec_seed{s}", "har_subject", 6),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted", 10),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted", 10),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_rmd_mlp_seed{s}", "rotated", 10),
}


def sha(t: torch.Tensor) -> str:
    return hashlib.sha1(t.contiguous().numpy().tobytes()).hexdigest()[:12]


def acc(logits, y) -> float:
    return float((logits.argmax(1) == y).float().mean())


def se95(p: float, n: int) -> float:
    return 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / n))


class Affine:
    """B ~ A M + c, fit in coordinates standardized by (mu, sd) -- the era pass's
    train statistics -- with ridge lam*n on the linear part. apply() works in
    raw coordinates so a map fit on task j can be applied to task k's features."""

    def __init__(self, mu, sd, M_std, c_std):
        self.mu, self.sd, self.M_std, self.c_std = mu, sd, M_std, c_std

    @classmethod
    def fit(cls, A, B, mu, sd, lam):
        As, Bs = (A - mu) / sd, (B - mu) / sd
        n, d = As.shape
        Ac, Bc = As - As.mean(0), Bs - Bs.mean(0)
        M = np.linalg.solve(Ac.T @ Ac + lam * n * np.eye(d), Ac.T @ Bc)
        c = Bs.mean(0) - As.mean(0) @ M
        return cls(mu, sd, M, c)

    def apply(self, A):
        return (((A - self.mu) / self.sd) @ self.M_std + self.c_std) * self.sd + self.mu

    def M_raw(self):
        return (self.M_std.T / self.sd).T * self.sd          # diag(1/sd) M diag(sd)

    def residual(self, A, B) -> float:
        return float(np.linalg.norm(B - self.apply(A)) / np.linalg.norm(B))


def projector_norms(M_raw, W):
    """||(T-I) P_W||_F and ||(T-I)(I-P_W)||_F, row convention: D_out W^T = 0 identically."""
    d = M_raw.shape[0]
    P = W.T @ np.linalg.solve(W @ W.T, W)
    D = M_raw - np.eye(d)
    D_in, D_out = D @ P, D @ (np.eye(d) - P)
    return float(np.linalg.norm(D_in)), float(np.linalg.norm(D_out)), float(np.abs(D_out @ W.T).max())


def spectrum(M_std):
    s = np.linalg.svd(M_std, compute_uv=False)
    return {"sigma_1": float(s[0]), "sigma_d": float(s[-1]), "spread": float(np.log(s[0] / max(s[-1], 1e-12))),
            "mass_09_11": float(np.mean((s >= 0.9) & (s <= 1.1))), "n_below_0.5": int(np.sum(s < 0.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--ref", required=True, help="seeded decomposition artifact (C-ID, F_enc, acc_refit_ceiling)")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    ckpt_t, run_t, kind, n_classes = ARMS[args.arm]
    root = args.ckpt_root.rstrip("/")
    ckpt_t, run_t = ckpt_t.replace("runs/", root + "/", 1), run_t.replace("runs/", root + "/", 1)
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 100 + f"\nD1 FIT  arm={args.arm} kind={kind} probe seed {args.probe_seed} lambdas {LAMS} (main {LAM_MAIN})\n" + "=" * 100)

    for s in seeds:                                                    # C-WIT: arm identity from the runs' own artifacts
        arm = json.load(open(f"{run_t.format(s=s)}/mafc_results.json"))["arm"]
        assert arm["use_task_heads"] is False and arm["era_checkpoints"] is True and arm["use_input_adapters"] is False, arm
        if kind != "har_subject":
            assert arm.get("disjoint_content") is True, arm
    rd = json.load(open(args.ref)); ref = {(r["seed"], r["task"]): r for r in (rd if isinstance(rd, list) else rd["rows"])}

    if kind == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=BATCH)
        pfp = har.partition_fingerprint()
        print(f"  har_subject partition {pfp} -> {'AS-EXECUTED' if pfp == HAR_PARTITION else 'NOT the executed partition -- STOP'}")
        if pfp != HAR_PARTITION:
            return 1
        maps = har_maps(har._sd)
        hdata = load_task_data(har, probe_subset_seed=args.probe_seed)
        data_for_seed = lambda _s: (hdata, maps)
        relay = lambda x, src, dst, m: har_relayout(x, src, dst, m)
    else:
        def data_for_seed(seed):
            bench = E18_FACTORY[kind](seed)
            fp = json.load(open(f"{run_t.format(s=seed)}/mafc_results.json"))["arm"]["content_fingerprint"]
            assert fp == bench.content_fingerprint(), (fp, bench.content_fingerprint())
            return load_task_data(bench, n_tasks=4, probe_subset_seed=args.probe_seed), (bench.angles if kind == "rotated" else bench.permutations)
        relay = (lambda x, src, dst, m: rotated_relayout(x, src, dst, m)) if kind == "rotated" else (lambda x, src, dst, m: mnist_relayout(x, src, dst, m))

    for s in seeds:
        data, maps = data_for_seed(s)
        d = ckpt_t.format(s=s)
        m4 = load(d, CUR)
        W_T = m4.classifier.weight.detach().cpu().numpy().astype(np.float64)
        b_T = m4.classifier.bias.detach().cpu().numpy().astype(np.float64)
        cells, keep, mats = [], {}, {}
        for k in range(4):
            xtr, ytr, xte, yte = data[k]
            m_era = load(d, k)
            h = era_head_of(m_era, k)
            W_k, b_k = h.weight.detach().cpu().numpy().astype(np.float64), h.bias.detach().cpu().numpy().astype(np.float64)
            g4 = assert_path_identity(m4, xte, k, False, device, label=f"s{s}/t{k}/final", verbose=False)
            ge = assert_path_identity(m_era, xte, k, False, device, label=f"s{s}/t{k}/era", verbose=False)
            assert g4["p_b"] and ge["p_b"], (g4["p_b"], ge["p_b"])
            h_tr, h_te = sha(xtr), sha(xte)                                 # C-INPUT: the same tensors feed every pass below
            xtr_T, xte_T = relay(xtr, k, CUR, maps), relay(xte, k, CUR, maps)
            assert sha(xtr) == h_tr and sha(xte) == h_te, "relay must not mutate its input"
            f = lambda m, x, y: [t.numpy().astype(np.float64) if i == 0 else t for i, t in enumerate(features_and_logits(m, x, y, k, False, device))]
            Zk_tr, _ = f(m_era, xtr, ytr);  Zk_te, le_te = f(m_era, xte, yte)
            ZT_tr, _ = f(m4, xtr_T, ytr);   ZT_te, _ = f(m4, xte_T, yte)
            Zp_tr, _ = f(m4, xtr, ytr);     Zp_te, l4_te = f(m4, xte, yte)
            acc_ceiling, acc_orig = acc(le_te, yte), acc(l4_te, yte)
            r = ref[(s, k)]
            cid = max(abs(acc_orig - r["acc_orig"]), abs(acc_ceiling - r["acc_ceiling"]))
            n_te = int(len(yte)); res95 = se95(acc_ceiling, n_te)
            mu, sd = Zk_tr.mean(0), Zk_tr.std(0) + 1e-8                   # ERA train statistics
            y_te = yte.numpy()
            head = lambda W, b, Z: (Z @ W.T + b).argmax(1)
            cell = {"seed": s, "task": k, "n_train": int(len(ytr)), "n_test": n_te, "input_sha": {"train": h_tr, "test": h_te},
                    "acc_ceiling": acc_ceiling, "acc_orig": acc_orig, "C_ID_delta": cid, "res95": res95,
                    "F_enc": r["F_enc"], "F_read": r["F_read"], "acc_refit_ceiling": r["acc_refit_ceiling"], "acc_refit_t4": r["acc_refit_t4"],
                    "label_marginal_train": np.bincount(ytr.numpy(), minlength=n_classes).tolist(), "by_lambda": {}}
            rng = np.random.default_rng(SHUF_SEED + 100 * s + k)
            perm = rng.permutation(len(ytr))
            for lam in LAMS:
                S = Affine.fit(ZT_tr, Zk_tr, mu, sd, lam);  T = Affine.fit(Zk_tr, ZT_tr, mu, sd, lam)
                Sp = Affine.fit(Zp_tr, Zk_tr, mu, sd, lam); Tp = Affine.fit(Zk_tr, Zp_tr, mu, sd, lam)
                S_shuf, Sp_shuf = Affine.fit(ZT_tr[perm], Zk_tr, mu, sd, lam), Affine.fit(Zp_tr[perm], Zk_tr, mu, sd, lam)
                plumb = Affine.fit(ZT_tr, ZT_tr, mu, sd, lam)
                own = float(np.mean(head(W_k, b_k, S.apply(ZT_te)) == y_te));   own_p = float(np.mean(head(W_k, b_k, Sp.apply(Zp_te)) == y_te))
                dep = float(np.mean(head(W_T, b_T, S.apply(ZT_te)) == y_te));   dep_p = float(np.mean(head(W_T, b_T, Sp.apply(Zp_te)) == y_te))
                shuf = float(np.mean(head(W_k, b_k, S_shuf.apply(ZT_te)) == y_te)); shuf_p = float(np.mean(head(W_k, b_k, Sp_shuf.apply(Zp_te)) == y_te))
                din_h, dout_h, zero_h = projector_norms(Tp.M_raw(), W_k);   din_T, dout_T, _ = projector_norms(Tp.M_raw(), W_T)
                din_hr, dout_hr, _ = projector_norms(T.M_raw(), W_k)
                cell["by_lambda"][str(lam)] = {
                    "residual_S_train": S.residual(ZT_tr, Zk_tr), "residual_S_test": S.residual(ZT_te, Zk_te),
                    "residual_Sp_train": Sp.residual(Zp_tr, Zk_tr), "residual_Sp_test": Sp.residual(Zp_te, Zk_te),
                    "own": own, "deploy": dep, "own_shuf": shuf, "own_old": own_p, "deploy_old": dep_p, "own_shuf_old": shuf_p,
                    "plumb_dev": float(np.linalg.norm(plumb.M_std - np.eye(len(mu)))),
                    "spec_T": spectrum(T.M_std), "spec_Tp": spectrum(Tp.M_std), "spec_S": spectrum(S.M_std),
                    "D_old_hk": {"in": din_h, "out": dout_h, "Dout_WT_max": zero_h}, "D_old_WT": {"in": din_T, "out": dout_T},
                    "D_relaid_hk": {"in": din_hr, "out": dout_hr}}
                if lam == LAM_MAIN:
                    mats[f"t{k}"] = {"mu": mu, "sd": sd, "S": S.M_std, "S_c": S.c_std, "T": T.M_std, "T_c": T.c_std,
                                     "Sp": Sp.M_std, "Sp_c": Sp.c_std, "Tp": Tp.M_std, "Tp_c": Tp.c_std, "W_k": W_k, "b_k": b_k}
                    keep[k] = (S, Sp, ZT_te, Zp_te, y_te, W_k, b_k)
            # re-laid F_enc: era refit (frame k, from the decomposition) minus refit on theta_T's re-laid features -- E23's (E1) on this arm
            refit_relaid = refit_probe(torch.from_numpy(ZT_tr).float(), ytr, torch.from_numpy(ZT_te).float(), yte, n_classes)
            refit_old = refit_probe(torch.from_numpy(Zp_tr).float(), ytr, torch.from_numpy(Zp_te).float(), yte, n_classes)
            cell.update({"acc_refit_T_relaid": refit_relaid, "F_enc_relaid": r["acc_refit_ceiling"] - refit_relaid,
                         "acc_refit_T_old_recomputed": refit_old, "refit_old_delta_vs_ref": refit_old - r["acc_refit_t4"]})
            m = cell["by_lambda"][str(LAM_MAIN)]
            print(f"  s{s}/t{k}: C-ID {cid:.1e} | ceiling {acc_ceiling:.4f} deployed {acc_orig:.4f} | own {m['own']:.4f} deploy {m['deploy']:.4f} shuf {m['own_shuf']:.4f}"
                  f" | old-frame own {m['own_old']:.4f} deploy {m['deploy_old']:.4f} | resid S {m['residual_S_test']:.3f} S' {m['residual_Sp_test']:.3f}"
                  f" | T' spread {m['spec_Tp']['spread']:.2f} mass {m['spec_Tp']['mass_09_11']:.2f} sig_d {m['spec_Tp']['sigma_d']:.3f}"
                  f" | D_in {m['D_old_hk']['in']:.2f} D_out {m['D_old_hk']['out']:.2f} (D_out W^T max {m['D_old_hk']['Dout_WT_max']:.1e})"
                  f" | F_enc {r['F_enc']:+.4f} relaid {cell['F_enc_relaid']:+.4f} | refit-old vs ref {cell['refit_old_delta_vs_ref']:+.4f}")
            cells.append(cell)
        # ---- D1 transfer within the seed: cross(j->k) = h_k reading task k's features repaired with task j's map ----------------
        tr = {"relaid": [[None] * 4 for _ in range(4)], "old": [[None] * 4 for _ in range(4)], "deploy_relaid": [[None] * 4 for _ in range(4)]}
        for j in range(4):
            S_j, Sp_j = keep[j][0], keep[j][1]
            for k in range(4):
                _, _, ZT_te, Zp_te, y_te, W_k, b_k = keep[k]
                tr["relaid"][j][k] = float(np.mean(((S_j.apply(ZT_te)) @ W_k.T + b_k).argmax(1) == y_te))
                tr["old"][j][k] = float(np.mean(((Sp_j.apply(Zp_te)) @ W_k.T + b_k).argmax(1) == y_te))
                tr["deploy_relaid"][j][k] = float(np.mean(((S_j.apply(ZT_te)) @ W_T.T + b_T).argmax(1) == y_te))
        np.savez_compressed(f"{args.out_dir}/fits_seed{s}.npz", **{f"{t}_{n}": v for t, dd in mats.items() for n, v in dd.items()}, W_T=W_T, b_T=b_T)
        json.dump({"arm": args.arm, "kind": kind, "seed": s, "probe_subset_seed": args.probe_seed, "ref": args.ref,
                   "lambdas": LAMS, "lambda_main": LAM_MAIN, "ridge": "lambda * n * I on the linear part, features standardized by the era pass's train mean/sd",
                   "shuffle_seed": SHUF_SEED, "cells": cells, "transfer": tr},
                  open(f"{args.out_dir}/cells_seed{s}.json", "w"), indent=2, default=float)
        print(f"  transfer (rows j = map, cols k = task; re-laid, h_k):\n" + "\n".join("     " + "  ".join(f"{v:.3f}" for v in row) for row in tr["relaid"]))
        print(f"  wrote {args.out_dir}/fits_seed{s}.npz, cells_seed{s}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
