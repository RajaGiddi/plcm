"""D1 sec 3 D3 on the PRETRAINED arms (docs/D1_prereg.md, signed v2) -- the direct
test of the subspace lemma: here the deployed readout for task k IS the frozen
per-task head W_k (bit-identical across era checkpoints, verified 2026-09-19),
so F_frozen = acc(W_k Z_k) - acc(W_k Z_T') = acc_ceiling - acc_orig = F_total.

No known map on Split-CIFAR-100, so only the old-frame fits exist:

    Z_k  = f_{theta_k}(x_i)     Z_T' = f_{theta_T}(x_i)        x_i = task-k train set, sequential loader (the E12/E14 path)
    S' : Z_T' -> Z_k  (repair)  T' : Z_k -> Z_T'  (drift)      ridge lambda*n*I on era-standardized features, lambda swept

Per cell: spectra of T', residual of S', ||D_in||/||D_out|| under the deployed
W_k with the witness D_out W_k^T == 0, own(k) = acc(W_k S'(Z_T'^test)) (== deploy
here), C-SHUF, C-PLUMB, and C-ID against the E12/E14 decomposition artifact
(acc_orig, acc_ceiling to 1e-6). Extraction and path identity are the E12/E14
decomposition's own (`features_and_logits`, `assert_p3`, `load_era`), imported
from the backbone's script so no parallel path exists (catch 32).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_fit import Affine, projector_norms, spectrum, se95, LAMS, LAM_MAIN, SHUF_SEED   # noqa: E402
from src.data.split_cifar100 import SplitCIFAR100Benchmark                                     # noqa: E402

TOL = 1e-6
ARMS = {"e12_base": ("scripts.e12_decompose", "ckpt_e12_base_seed{s}", "runs/e12_decomp_v2/base_{s}.json"),
        "e14_base": ("scripts.e14_decompose", "ckpt_e14_base_seed{s}", "runs/e14_decomp/base_{s}.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    modname, ckpt_t, ref_t = ARMS[args.arm]
    import importlib
    dec = importlib.import_module(modname)                     # the backbone's own decomposition path
    from scripts.e12_p3 import assert_p3
    ref_path = ref_t.format(s=args.seed).replace("runs/", args.ckpt_root.rstrip("/") + "/", 1)
    ref = {r["task"]: r for r in json.load(open(ref_path))}
    ckpt_dir = os.path.join(args.ckpt_root, ckpt_t.format(s=args.seed), f"mafc_seed{args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 100 + f"\nD1 PRETRAINED  arm={args.arm} seed={args.seed} ckpt {ckpt_dir} ref {ref_path}\n" + "=" * 100)

    bench = SplitCIFAR100Benchmark(num_tasks=dec.NUM_TASKS, seed=args.seed, root=args.data_root, batch_size=dec.BATCH, remap_labels=True)
    m_final = dec.load_era(ckpt_dir, dec.FINAL, device)
    arm = json.load(open(os.path.join(args.ckpt_root, f"{args.arm.replace('_base', '')}_base_seed{args.seed}", "mafc_results.json")))["arm"]
    assert arm["use_task_heads"] is True and arm["era_checkpoints"] is True and arm["use_input_adapters"] is False and arm["benchmark"] == "cifar100", arm
    print(f"  arm identity: per-task heads, adapters off, era checkpoints, benchmark {arm['benchmark']} (artifact)")

    cells = []
    for k in dec.OLD_TASKS:
        tr, te = bench.get_task_loaders(k)
        tr = DataLoader(tr.dataset, batch_size=dec.BATCH, shuffle=False)   # sequential: pairs stay together (catch 32)
        te = DataLoader(te.dataset, batch_size=dec.BATCH, shuffle=False)
        xprobe = next(iter(te))[0][:16].to(device)
        m_era = dec.load_era(ckpt_dir, k, device)
        assert_p3(m_era, xprobe, k, label=f"era{k}"); assert_p3(m_final, xprobe, k, label=f"final/T{k}")
        # (features, logits, labels) -- labels come from the SAME pass as the features (catch 32);
        # the era and final passes must agree on them, which is the C-INPUT witness here.
        Zk_tr, _, ytr = dec.features_and_logits(m_era, tr, k, False, device); Zk_te, le_te, yte = dec.features_and_logits(m_era, te, k, False, device)
        Zp_tr, _, ytr_f = dec.features_and_logits(m_final, tr, k, False, device); Zp_te, l4_te, yte_f = dec.features_and_logits(m_final, te, k, False, device)
        assert torch.equal(ytr, ytr_f) and torch.equal(yte, yte_f), f"task {k}: era and final passes read different label orders"
        Zk_tr, Zk_te, Zp_tr, Zp_te = (z.numpy().astype(np.float64) for z in (Zk_tr, Zk_te, Zp_tr, Zp_te))
        y_te = yte.numpy()
        acc_ceiling = float((le_te.argmax(1) == yte).float().mean()); acc_orig = float((l4_te.argmax(1) == yte).float().mean())
        r = ref[k]; cid = max(abs(acc_orig - r["acc_orig"]), abs(acc_ceiling - r["acc_ceiling"]))
        # the deployed head for task k -- frozen; the final model's copy is bit-identical to the era's (verified)
        h_fin, h_era = m_final.task_classifiers[str(k)], m_era.task_classifiers[str(k)]
        assert torch.equal(h_fin.weight.detach().cpu(), h_era.weight.detach().cpu()) and torch.equal(h_fin.bias.detach().cpu(), h_era.bias.detach().cpu()), f"task {k}: head not frozen"
        W_k = h_fin.weight.detach().cpu().numpy().astype(np.float64); b_k = h_fin.bias.detach().cpu().numpy().astype(np.float64)
        head = lambda Z: (Z @ W_k.T + b_k).argmax(1)
        n_te = len(y_te); mu, sd = Zk_tr.mean(0), Zk_tr.std(0) + 1e-8
        rng = np.random.default_rng(SHUF_SEED + 100 * args.seed + k); perm = rng.permutation(len(ytr))
        cell = {"seed": args.seed, "task": k, "n_train": int(len(ytr)), "n_test": int(n_te), "d": int(Zk_tr.shape[1]),
                "input_sha": {"train": hashlib.sha1(ytr.numpy().tobytes()).hexdigest()[:12]},
                "acc_ceiling": acc_ceiling, "acc_orig": acc_orig, "C_ID_delta": cid, "res95": se95(acc_ceiling, n_te),
                "F_enc": r["F_enc"], "F_read": r["F_read"], "F_total": acc_ceiling - acc_orig, "F_frozen": acc_ceiling - acc_orig,
                "acc_refit_ceiling": r["acc_refit_ceiling"], "acc_refit_t4": r["acc_refit"],
                "label_marginal_train": np.bincount(ytr.numpy(), minlength=dec.CLASSES_PER_TASK if hasattr(dec, "CLASSES_PER_TASK") else 5).tolist(), "by_lambda": {}}
        for lam in LAMS:
            Sp = Affine.fit(Zp_tr, Zk_tr, mu, sd, lam); Tp = Affine.fit(Zk_tr, Zp_tr, mu, sd, lam)
            Sp_shuf = Affine.fit(Zp_tr[perm], Zk_tr, mu, sd, lam); plumb = Affine.fit(Zp_tr, Zp_tr, mu, sd, lam)
            own = float(np.mean(head(Sp.apply(Zp_te)) == y_te)); shuf = float(np.mean(head(Sp_shuf.apply(Zp_te)) == y_te))
            din, dout, wit = projector_norms(Tp.M_raw(), W_k)
            cell["by_lambda"][str(lam)] = {"residual_Sp_train": Sp.residual(Zp_tr, Zk_tr), "residual_Sp_test": Sp.residual(Zp_te, Zk_te),
                                           "own_old": own, "deploy_old": own, "own_shuf_old": shuf,
                                           "plumb_dev": float(np.linalg.norm(plumb.M_std - np.eye(len(mu)))),
                                           "spec_Tp": spectrum(Tp.M_std), "spec_Sp": spectrum(Sp.M_std),
                                           "D_old_hk": {"in": din, "out": dout, "Dout_WT_max": wit}}
        m = cell["by_lambda"][str(LAM_MAIN)]
        print(f"  t{k:>2}: C-ID {cid:.1e} | ceiling {acc_ceiling:.4f} deployed {acc_orig:.4f} F_frozen {cell['F_frozen']:+.4f} | own {m['own_old']:.4f} shuf {m['own_shuf_old']:.4f}"
              f" | resid S' {m['residual_Sp_test']:.3f} | T' spread {m['spec_Tp']['spread']:.2f} mass {m['spec_Tp']['mass_09_11']:.2f} n<0.5 {m['spec_Tp']['n_below_0.5']}/{cell['d']}"
              f" | D_in {m['D_old_hk']['in']:.2f} D_out {m['D_old_hk']['out']:.2f} (D_out W^T max {m['D_old_hk']['Dout_WT_max']:.1e})")
        cells.append(cell)
    json.dump({"arm": args.arm, "kind": "cifar100_no_map", "seed": args.seed, "ref": ref_path, "lambdas": LAMS, "lambda_main": LAM_MAIN,
               "ridge": "lambda * n * I on the linear part, features standardized by the era pass's train mean/sd", "shuffle_seed": SHUF_SEED, "cells": cells},
              open(f"{args.out_dir}/cells_seed{args.seed}.json", "w"), indent=2, default=float)
    print(f"  wrote {args.out_dir}/cells_seed{args.seed}.json  (C-ID max {max(c['C_ID_delta'] for c in cells):.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
