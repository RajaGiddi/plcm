"""E23-B sec 3 -- raw vs re-laid on the scratch LSTM at two sequence lengths
(docs/E23B_seqlen_prereg.md, registered before launch 2026-09-20).

The quantities are E23's, on MNIST: for each old task k < T,

    raw      acc_orig    = deployed theta_T, h_T on x_k in FRAME k
    re-laid  acc_relaid  = deployed theta_T, h_T on mnist_relayout(x_k, k -> T)      (C0deg's input path)
    ceiling  acc_ceiling = era theta_k, h_k on x_k                                    (frame k, NEVER re-laid)
    F_enc        = refit(era features)   - refit(theta_T features, frame k)
    F_enc_relaid = refit(era features)   - refit(theta_T features, RE-LAID)

plus the wrong-source relay (task k+1's permutation, or k-1 for the last old
task) as the must-fail. Everything routes through the audited loader
(`load_task_data`, recorded draw), `PLCM.load_era` (catch 29),
`features_and_logits` + `assert_path_identity` (catch 28) and `refit_probe`;
the relay is `cure_screen.mnist_relayout`, the same function C0deg uses.

C-CONSTR: `num_tasks`, `content_chunks`, `content_fingerprint` and (when the
artifact has it) `construction_fingerprint` are asserted against the rebuilt
benchmark -- the content hash alone cannot separate T=5 from T=20 (see
src/data/permuted_mnist.py).

Usage:
  python scripts/e23_seqlen.py --arm e23b_t20 --out runs/e23b/e23b_t20/decomp_seed42.json --seed 42
  python scripts/e23_seqlen.py --arm e18_pmd_lstm --ref runs/e18_pmd_lstm/decomp.json ...   (C-ID)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,    # noqa: E402
                                    assert_path_identity, refit_probe, PROBE_SUBSET_SEED)
from scripts.cure_screen import mnist_relayout, BATCH                              # noqa: E402
from src.data.permuted_mnist import PermutedMNISTBenchmark                          # noqa: E402

N_CLASSES = 10
TOL = 1e-6
ARMS = {  # arm: (num_tasks, content_chunks, checkpoint template, run template)
    "e18_pmd_lstm": (5, None, "runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}"),
    "e23b_t5c20": (5, 20, "runs/ckpt_e23b_t5c20_seed{s}/mafc_seed{s}_fp32", "runs/e23b_t5c20_seed{s}"),
    "e23b_t20": (20, 20, "runs/ckpt_e23b_t20_seed{s}/mafc_seed{s}_fp32", "runs/e23b_t20_seed{s}"),
    "e23b_t5c20_floor": (5, 20, "runs/ckpt_e23b_t5c20_floor4tec_a/mafc_seed42_fp32", "runs/e23b_t5c20_floor4tec_a"),
    "e23b_t20_floor": (20, 20, "runs/ckpt_e23b_t20_floor4tec_a/mafc_seed42_fp32", "runs/e23b_t20_floor4tec_a"),
}


def acc(l, y):
    return float((l.argmax(1) == y).float().mean())


def se95(p, n):
    return 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--ref", default=None, help="C-ID: a seeded decomposition to reproduce (acc_orig, acc_ceiling)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    T, chunks, ckpt_t, run_t = ARMS[args.arm]
    root = args.ckpt_root.rstrip("/")
    ckpt_dir = ckpt_t.replace("runs/", root + "/", 1).format(s=args.seed)
    run_dir = run_t.replace("runs/", root + "/", 1).format(s=args.seed)
    print("=" * 100 + f"\nE23-B  arm={args.arm} T={T} chunks={chunks} seed={args.seed}\n" + "=" * 100)

    arm = json.load(open(f"{run_dir}/mafc_results.json"))["arm"]
    bench = PermutedMNISTBenchmark(num_tasks=T, batch_size=BATCH, seed=args.seed, disjoint_content=True, content_chunks=chunks)
    # C-WIT + C-CONSTR: the content hash is blind to the chunk boundaries, so the shape is asserted too.
    assert arm["use_task_heads"] is False and arm["era_checkpoints"] is True and arm["use_input_adapters"] is False, arm
    assert arm.get("disjoint_content") is True and arm.get("content_fingerprint") == bench.content_fingerprint(), (arm.get("content_fingerprint"), bench.content_fingerprint())
    assert arm.get("num_tasks") == T and arm.get("content_chunks") == chunks, (arm.get("num_tasks"), arm.get("content_chunks"), T, chunks)
    if arm.get("construction_fingerprint") is not None:
        assert arm["construction_fingerprint"] == bench.construction_fingerprint(), (arm["construction_fingerprint"], bench.construction_fingerprint())
    print(f"  C-WIT/C-CONSTR: shared head, adapters off, era ckpts; num_tasks {T}, chunks {chunks}, content {bench.content_fingerprint()}, construction {bench.construction_fingerprint()}")

    data = load_task_data(bench, n_tasks=T, probe_subset_seed=args.probe_seed)
    perms = bench.permutations
    # C-RELAY: identity and round-trip, on task 1's test batch
    xb = data[1][2][:64]
    r_id = float((mnist_relayout(xb, 1, 1, perms) - xb).abs().max())
    r_rt = float((mnist_relayout(mnist_relayout(xb, 1, T - 1, perms), T - 1, 1, perms) - xb).abs().max())
    assert r_id == 0.0 and r_rt == 0.0, (r_id, r_rt)
    print(f"  C-RELAY: k->k identity {r_id:.1e}; round trip k->T->k {r_rt:.1e} -> exact")

    ref = None
    if args.ref:
        rd = json.load(open(args.ref)); ref = {(r["seed"], r["task"]): r for r in (rd if isinstance(rd, list) else rd["rows"])}

    m4 = load(ckpt_dir, T - 1)
    rows, cid_fail, wrong_fail = [], [], []
    for k in range(T - 1):
        xtr, ytr, xte, yte = data[k]
        k_wrong = k + 1 if k + 1 < T - 1 else k - 1
        m_era = load(ckpt_dir, k)
        ge = assert_path_identity(m_era, xte, k, False, device, label=f"era{k}", verbose=False)
        g4 = assert_path_identity(m4, xte, k, False, device, label=f"final/T{k}", verbose=False)
        assert ge["p_b"] and g4["p_b"], (ge["p_b"], g4["p_b"])
        xtr_r, xte_r = mnist_relayout(xtr, k, T - 1, perms), mnist_relayout(xte, k, T - 1, perms)
        xte_w = mnist_relayout(xte, k_wrong, T - 1, perms)          # wrong SOURCE frame: the must-fail
        f_e_tr, _ = features_and_logits(m_era, xtr, ytr, k, False, device)
        f_e_te, l_e_te = features_and_logits(m_era, xte, yte, k, False, device)
        f_4_tr, _ = features_and_logits(m4, xtr, ytr, k, False, device)
        f_4_te, l_4_te = features_and_logits(m4, xte, yte, k, False, device)
        f_r_tr, _ = features_and_logits(m4, xtr_r, ytr, k, False, device)
        f_r_te, l_r_te = features_and_logits(m4, xte_r, yte, k, False, device)
        _, l_w_te = features_and_logits(m4, xte_w, yte, k, False, device)
        acc_ceiling, acc_orig, acc_relaid, acc_wrong = acc(l_e_te, yte), acc(l_4_te, yte), acc(l_r_te, yte), acc(l_w_te, yte)
        acc_rc = refit_probe(f_e_tr, ytr, f_e_te, yte, N_CLASSES)
        acc_r4 = refit_probe(f_4_tr, ytr, f_4_te, yte, N_CLASSES)
        acc_rr = refit_probe(f_r_tr, ytr, f_r_te, yte, N_CLASSES)
        row = {"arm": args.arm, "seed": args.seed, "task": k, "T": T, "content_chunks": chunks,
               "n_train": int(len(ytr)), "n_test": int(len(yte)), "res95": se95(acc_ceiling, len(yte)),
               "acc_ceiling": acc_ceiling, "acc_refit_ceiling": acc_rc, "R": acc_rc - acc_ceiling,
               "acc_orig": acc_orig, "acc_refit": acc_r4, "F_enc": acc_rc - acc_r4, "F_read": acc_r4 - acc_orig, "F_total": acc_ceiling - acc_orig,
               "acc_relaid": acc_relaid, "acc_refit_relaid": acc_rr, "F_enc_relaid": acc_rc - acc_rr, "F_read_relaid": acc_rr - acc_relaid,
               "F_total_relaid": acc_ceiling - acc_relaid, "acc_wrong_perm": acc_wrong, "wrong_source_task": k_wrong, "p_b": True}
        row["identity_raw"] = abs(row["F_enc"] + row["F_read"] - row["R"] - row["F_total"])
        row["identity_relaid"] = abs(row["F_enc_relaid"] + row["F_read_relaid"] - row["R"] - row["F_total_relaid"])
        if ref is not None and (args.seed, k) in ref:
            r = ref[(args.seed, k)]
            d = max(abs(acc_orig - r["acc_orig"]), abs(acc_ceiling - r["acc_ceiling"]))
            row["C_ID_delta"] = d
            if d >= TOL:
                cid_fail.append((k, d))
        if not acc_wrong < acc_relaid:
            wrong_fail.append(k)
        rows.append(row)
        print(f"  t{k:>2}: ceiling {acc_ceiling:.4f} | raw deployed {acc_orig:.4f} refit {acc_r4:.4f} F_enc {row['F_enc']:+.4f}"
              f" | re-laid deployed {acc_relaid:.4f} refit {acc_rr:.4f} F_enc {row['F_enc_relaid']:+.4f}"
              f" | wrong-perm {acc_wrong:.4f} | res {row['res95']:.3f}" + (f" | C-ID {row['C_ID_delta']:.1e}" if "C_ID_delta" in row else ""))
    m = lambda key: float(np.mean([r[key] for r in rows]))
    out = {"arm": args.arm, "seed": args.seed, "T": T, "content_chunks": chunks, "probe_subset_seed": args.probe_seed,
           "content_fingerprint": bench.content_fingerprint(), "construction_fingerprint": bench.construction_fingerprint(),
           "relay_checks": {"identity": r_id, "round_trip": r_rt},
           "C_ID": {"ref": args.ref, "pass": (ref is not None and not cid_fail), "fail": cid_fail} if ref is not None else None,
           "wrong_perm_must_fail": {"pass": not wrong_fail, "fail_tasks": wrong_fail, "n": len(rows)},
           "identity_max": max(max(r["identity_raw"], r["identity_relaid"]) for r in rows),
           "pooled": {k: m(k) for k in ("F_enc", "F_read", "R", "F_total", "F_enc_relaid", "F_read_relaid", "F_total_relaid",
                                        "acc_orig", "acc_relaid", "acc_ceiling", "acc_wrong_perm", "res95")}, "rows": rows}
    print(f"\n  pooled ({len(rows)} cells): F_enc raw {out['pooled']['F_enc']:+.4f} -> re-laid {out['pooled']['F_enc_relaid']:+.4f} (res {out['pooled']['res95']:.3f})"
          f" | deployed raw {out['pooled']['acc_orig']:.4f} -> re-laid {out['pooled']['acc_relaid']:.4f} (ceiling {out['pooled']['acc_ceiling']:.4f})"
          f" | wrong-perm {out['pooled']['acc_wrong_perm']:.4f}, must-fail {'PASS' if not wrong_fail else 'FAIL ' + str(wrong_fail)}"
          + (f" | C-ID {'PASS' if not cid_fail else 'FAIL ' + str(cid_fail)}" if ref is not None else "") + f" | identity {out['identity_max']:.1e}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
