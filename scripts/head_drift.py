"""Head drift -- what F_read contains on a SHARED-head arm (paper sec 3.1, 2026-09-19).

sec 3.1 as drafted says "one readout per task, frozen at task end", so that any
change between the two decomposition terms arises from the encoder alone. That
is true on the per-task-head arms (E12/E14: `set_task` freezes the outgoing
head, src/models/plcm.py:322-324; verified bitwise on the checkpoints). Every
scratch-trained arm in Table 1 records `use_task_heads: False`: one shared head
W_T that trains through every task; the era head h_k exists only as a snapshot.
There, the deployed readout for task k is W_T, and

    F_read = acc_refit_T - acc(W_T Z_T)
           = [acc_refit_T - acc(h_k Z_T)]  +  [acc(h_k Z_T) - acc(W_T Z_T)]
           =        F_read^frozen           +            dH

F_read^frozen is the quantity a per-task-head arm's F_read IS (the stored head
reading drifted features); dH is head drift, which no feature-drift model can
explain and which can carry either sign. This script measures acc(h_k Z_T) per
cell on the same population as the seeded decomposition (C-ID: acc_orig and
acc_ceiling must reproduce that artifact to 1e-6, else nothing is read).

Deployed features come from `features_and_logits` through the deployed path
with `assert_path_identity` on every cell (catch 28); models via
`channel_decomp.load` (PLCM.load_era, catch 29); data via the audited
`load_task_data` with the recorded probe draw (catch 32 / R2).

Usage:
  python scripts/head_drift.py --arm e18_pmd_mlp --ref runs/e18_pmd_mlp/decomp.json --out runs/head_drift/e18_pmd_mlp.json
  python scripts/head_drift.py --arm s72_off --ckpt-root /runs --ref /runs/e20/seeded/har_s72_off_linear.json ...  (x86: har_subject partition)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import (load, features_and_logits, load_task_data,   # noqa: E402
                                    assert_path_identity, PROBE_SUBSET_SEED)
from scripts.cure_screen import E18_FACTORY, era_head_of                          # noqa: E402

SEEDS = [42, 1337, 2024]
CUR = 4
TOL = 1e-6
ARMS = {
    # arm: (checkpoint template, run-artifact template, kind, n_classes)
    "s72_off":      ("runs/ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32", "runs/e10off_ec_seed{s}", "har_subject", 6),
    "e18_pmd_mlp":  ("runs/ckpt_e18_pmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_mlp_seed{s}", "permuted", 10),
    "e18_pmd_lstm": ("runs/ckpt_e18_pmd_lstm_seed{s}/mafc_seed{s}_fp32", "runs/e18_pmd_lstm_seed{s}", "permuted", 10),
    "e18_rmd_mlp":  ("runs/ckpt_e18_rmd_mlp_seed{s}/mafc_seed{s}_fp32", "runs/e18_rmd_mlp_seed{s}", "rotated", 10),
}
HAR_PARTITION = "1104af185c87"     # as-executed (x86); the laptop partition differs -- gate, never assume


def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--ref", default=None, help="seeded decomposition artifact to join against (C-ID + F_read split)")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SUBSET_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    ckpt_t, run_t, kind, n_classes = ARMS[args.arm]
    root = args.ckpt_root.rstrip("/")
    ckpt_t = ckpt_t.replace("runs/", root + "/", 1)
    run_t = run_t.replace("runs/", root + "/", 1)
    print("=" * 96 + f"\nHEAD DRIFT  arm={args.arm}  kind={kind}  probe seed {args.probe_seed}\n" + "=" * 96)

    # ---- arm identity from the runs' own artifacts (catch 30) -------------------
    for s in SEEDS:
        arm = json.load(open(f"{run_t.format(s=s)}/mafc_results.json"))["arm"]
        assert arm["use_task_heads"] is False, f"{args.arm} seed {s}: not a shared-head arm ({arm['use_task_heads']})"
        assert arm["era_checkpoints"] is True and arm["use_input_adapters"] is False, arm
        if kind != "har_subject":
            assert arm.get("disjoint_content") is True, arm
    print("  arm identity: shared head, adapters off, era checkpoints -- 3/3 seeds")

    # ---- data through the audited loader ---------------------------------------
    if kind == "har_subject":
        from src.data.har_subject import HARSubjectBenchmark
        har = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
        pfp = har.partition_fingerprint()
        print(f"  har_subject partition {pfp} -> {'AS-EXECUTED' if pfp == HAR_PARTITION else 'NOT the executed partition -- STOP'}")
        if pfp != HAR_PARTITION:
            return 1
        hdata = load_task_data(har, probe_subset_seed=args.probe_seed)
        data_for_seed = lambda _s: hdata
    else:
        def data_for_seed(seed):
            bench = E18_FACTORY[kind](seed)
            fp = json.load(open(f"{run_t.format(s=seed)}/mafc_results.json"))["arm"]["content_fingerprint"]
            assert fp == bench.content_fingerprint(), f"seed {seed}: artifact construction {fp} != rebuilt {bench.content_fingerprint()}"
            return load_task_data(bench, n_tasks=4, probe_subset_seed=args.probe_seed)

    ref = None
    if args.ref:
        rd = json.load(open(args.ref))
        ref = {(r["seed"], r["task"]): r for r in (rd if isinstance(rd, list) else rd["rows"])}
        print(f"  joining against {args.ref} ({len(ref)} cells)")

    rows, cid_fail = [], []
    for s in SEEDS:
        data = data_for_seed(s)
        d = ckpt_t.format(s=s)
        m4 = load(d, CUR)
        for k in range(4):
            _, _, xte, yte = data[k]
            m_era = load(d, k)
            g4 = assert_path_identity(m4, xte, k, False, device, label=f"s{s}/t{k}/final", verbose=False)
            ge = assert_path_identity(m_era, xte, k, False, device, label=f"s{s}/t{k}/era", verbose=False)
            assert g4["p_b"] and ge["p_b"], f"s{s}/t{k}: single-head readout must hold on a shared-head arm ({g4['p_b']}, {ge['p_b']})"
            f4, l4 = features_and_logits(m4, xte, yte, k, False, device)
            fe, le = features_and_logits(m_era, xte, yte, k, False, device)
            h_k = era_head_of(m_era, k)
            with torch.no_grad():
                l_frozen = h_k(f4.to(next(h_k.parameters()).device))       # era head reading FINAL features
                l_WT_era = m4.classifier(fe.to(next(m4.parameters()).device))  # final head reading ERA features (symmetric record)
            row = {"seed": s, "task": k, "n_test": int(len(yte)),
                   "acc_orig": acc(l4, yte), "acc_ceiling": acc(le, yte),
                   "acc_frozen": acc(l_frozen.cpu(), yte), "acc_WT_on_era": acc(l_WT_era.cpu(), yte),
                   "p_b": bool(g4["p_b"] and ge["p_b"])}
            if ref is not None:
                r = ref[(s, k)]
                d_orig, d_ceil = abs(row["acc_orig"] - r["acc_orig"]), abs(row["acc_ceiling"] - r["acc_ceiling"])
                if d_orig >= TOL or d_ceil >= TOL:
                    cid_fail.append((s, k, d_orig, d_ceil))
                row.update({"acc_refit_t4": r["acc_refit_t4"], "acc_refit_ceiling": r["acc_refit_ceiling"],
                            "F_enc": r["F_enc"], "R": r["R"],
                            "F_read": r["acc_refit_t4"] - row["acc_orig"],
                            "F_read_frozen": r["acc_refit_t4"] - row["acc_frozen"],
                            "dH": row["acc_frozen"] - row["acc_orig"],
                            "F_total": row["acc_ceiling"] - row["acc_orig"],
                            "F_total_frozen": row["acc_ceiling"] - row["acc_frozen"]})
            rows.append(row)
            print(f"  s{s}/t{k}: ceiling {row['acc_ceiling']:.4f}  W_T·Z_T {row['acc_orig']:.4f}  h_k·Z_T {row['acc_frozen']:.4f}  W_T·Z_k {row['acc_WT_on_era']:.4f}"
                  + (f"  | F_read {row['F_read']:+.4f} = frozen {row['F_read_frozen']:+.4f} + dH {row['dH']:+.4f}" if ref is not None else ""))

    out = {"arm": args.arm, "kind": kind, "probe_subset_seed": args.probe_seed, "ref": args.ref, "rows": rows}
    if ref is not None:
        print(f"\n  C-ID vs {os.path.basename(args.ref)}: acc_orig / acc_ceiling reproduce to 1e-6 in {len(rows) - len(cid_fail)}/{len(rows)} cells -> "
              + ("PASS" if not cid_fail else f"FAIL {cid_fail[:4]} -- nothing read"))
        out["C_ID"] = {"pass": not cid_fail, "fail_cells": cid_fail}
        if not cid_fail:
            m = lambda key: float(np.mean([r[key] for r in rows]))
            ident = max(abs(r["F_read"] - r["F_read_frozen"] - r["dH"]) for r in rows)
            pooled = {key: m(key) for key in ("F_enc", "R", "F_read", "F_read_frozen", "dH", "F_total", "F_total_frozen")}
            pooled["share"] = pooled["F_read"] / (pooled["F_enc"] + pooled["F_read"])
            pooled["share_frozen"] = pooled["F_read_frozen"] / (pooled["F_enc"] + pooled["F_read_frozen"])
            per_seed = {}
            for s in SEEDS:
                rs = [r for r in rows if r["seed"] == s]; ms = lambda key: float(np.mean([r[key] for r in rs]))
                per_seed[s] = {"F_read": ms("F_read"), "F_read_frozen": ms("F_read_frozen"), "dH": ms("dH"),
                               "share": ms("F_read") / (ms("F_enc") + ms("F_read")),
                               "share_frozen": ms("F_read_frozen") / (ms("F_enc") + ms("F_read_frozen"))}
            out.update({"pooled": pooled, "per_seed": per_seed, "identity_residual": ident})
            print(f"  identity F_read = F_read_frozen + dH: max residual {ident:.1e}")
            print(f"  pooled: F_enc {pooled['F_enc']:.4f}  F_read {pooled['F_read']:.4f} = frozen {pooled['F_read_frozen']:.4f} + dH {pooled['dH']:+.4f}"
                  f"   share {100*pooled['share']:.1f} -> frozen-head share {100*pooled['share_frozen']:.1f}"
                  f"   F_total {pooled['F_total']:.4f} -> frozen-head F_total {pooled['F_total_frozen']:.4f}")
            print("  dH per seed: " + "  ".join(f"s{s} {v['dH']:+.4f}" for s, v in per_seed.items()))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
