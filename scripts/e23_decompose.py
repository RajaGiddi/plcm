"""E23 sec 4 -- the decomposition on cifar100_permuted (B6), raw AND re-laid
(docs/E23_prereg.md, signed v2; option (b): era checkpoints, no fp32 shadow).

Per old task k on the trained A-ViT / A-RN checkpoints, through the backbone's
own decomposition path (`e12_decompose` / `e14_decompose`: load_era,
features_and_logits with labels from the same pass, the P3 gates and their
positive controls -- no parallel implementation, catch 32):

    era terms   acc_ceiling, acc_refit_ceiling         on x_k in its OWN frame P_k, never re-laid
    raw         acc_orig, acc_refit_T                    theta_T on x_k in P_k        (the decomposition as run everywhere)
    re-laid     acc_orig_relaid, acc_refit_T_relaid      theta_T on x_k rendered under P_T  (C0deg's input path)

    F_enc       = acc_refit_ceiling - acc_refit_T          F_enc_relaid = acc_refit_ceiling - acc_refit_T_relaid
    F_read      = acc_refit_T - acc_orig                   F_read_relaid = acc_refit_T_relaid - acc_orig_relaid
    R           = acc_refit_ceiling - acc_ceiling          identity F_enc + F_read - R = F_total, both frames

The re-laid input is the dataset's OWN rendering of task-k content under P_T
(`SplitCIFAR100Task` with `perm = perms[T]`), and the algebraic relay
`apply_perm(apply_perm(x, invert_perm(P_k)), P_T)` on the P_k rendering is
asserted bitwise equal to it on a batch (C-ID), as is the identity relay
k -> k. Must-fail: re-laying with the WRONG source permutation (P_{k+1};
P_{k-1} for the last old task) must read below the right one in 19/19 tasks.
C-RELOAD: the checkpoint's stored boundary accuracy vs the fp16 reload's own
(the identity gate's resolution under option (b)).

Usage: python scripts/e23_decompose.py --backbone vit --seed 42 --ckpt-root /runs --out /runs/e23/e23_vit/decomp_seed42.json
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import refit_probe                                     # noqa: E402
from src.data.split_cifar100 import SplitCIFAR100Benchmark, apply_perm, invert_perm  # noqa: E402

BACKBONES = {"vit": ("scripts.e12_decompose", "configs/e12_vit.yaml", "e23_vit"),
             "resnet": ("scripts.e14_decompose", "configs/e14_resnet.yaml", "e23_rn")}


def acc(l, y):
    return float((l.argmax(1) == y).float().mean())


def bapply(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Batched `apply_perm`: the dataset's helper is per-image ([3, S, S]); this applies the same flat
    permutation to every image of a [B, 3, S, S] batch and is asserted equal to it on the first image."""
    out = x.reshape(x.shape[0], -1)[:, perm].reshape(x.shape)
    assert torch.equal(out[0], apply_perm(x[0], perm)), "bapply != apply_perm on image 0"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None, help="override: e.g. /runs/ckpt_e23_vit_floor42/mafc_seed42 (the C-FLOOR relaunch)")
    ap.add_argument("--run-dir", default=None, help="override: e.g. /runs/e23_vit_floor_seed42 (its results artifact, for arm identity)")
    args = ap.parse_args()
    device = torch.device(args.device)
    modname, cfg_path, arm = BACKBONES[args.backbone]
    dec = importlib.import_module(modname)
    from scripts.e12_p3 import assert_p3, positive_controls
    root = args.ckpt_root.rstrip("/")
    ckpt_dir = args.ckpt_dir or f"{root}/ckpt_{arm}_seed{args.seed}/mafc_seed{args.seed}"
    run_arm = json.load(open(f"{args.run_dir or f'{root}/{arm}_seed{args.seed}'}/mafc_results.json"))["arm"]
    print("=" * 100 + f"\nE23 DECOMPOSE  {arm} seed {args.seed}  ckpt {ckpt_dir}\n" + "=" * 100)
    assert run_arm["benchmark"] == "cifar100_permuted" and run_arm["use_task_heads"] is True and run_arm["era_checkpoints"] is True \
        and run_arm["use_input_adapters"] is False and run_arm["backbone"] == args.backbone, run_arm
    print(f"  arm identity (artifact): {args.backbone}, cifar100_permuted, per-task heads, adapters off, era checkpoints, fp16 {run_arm.get('checkpoint_fp16')}")

    bench = SplitCIFAR100Benchmark(num_tasks=dec.NUM_TASKS, batch_size=dec.BATCH, root=args.data_root, seed=args.seed,
                                   remap_labels=True, shift_mode="patch", download=False)
    expected = (yaml.safe_load(open(cfg_path)).get("benchmark") or {}).get("expected_shift") or {}
    sh, co = bench.shift_fingerprint(), bench.class_order_fingerprint()
    assert expected.get(args.seed) == sh, f"shift fingerprint {sh} != registered {expected.get(args.seed)}"
    print(f"  shift fingerprint {sh} GATED -> MATCH; class order {co}")
    T = dec.FINAL
    P = bench.perms

    def loaders(k, perm):
        tr, te = bench._make(k, True), bench._make(k, False)
        tr.perm, te.perm = perm, perm
        return (DataLoader(tr, batch_size=dec.BATCH, shuffle=False), DataLoader(te, batch_size=dec.BATCH, shuffle=False))

    # ---- C-ID on the relay itself, on one batch of task 1 (has a non-identity P_k) ---------------------------------
    _, te1 = loaders(1, P[1]); xb = next(iter(te1))[0]
    x_id = bapply(bapply(xb, invert_perm(P[1])), P[1])
    _, te1T = loaders(1, P[T]); xbT = next(iter(te1T))[0]
    x_alg = bapply(bapply(xb, invert_perm(P[1])), P[T])
    relay_checks = {"identity_max_abs": float((x_id - xb).abs().max()), "algebraic_vs_dataset_max_abs": float((x_alg - xbT).abs().max())}
    assert relay_checks["identity_max_abs"] == 0.0 and relay_checks["algebraic_vs_dataset_max_abs"] == 0.0, relay_checks
    print(f"  relay C-ID: k->k identity {relay_checks['identity_max_abs']:.1e}; algebraic relay vs the dataset's P_T rendering {relay_checks['algebraic_vs_dataset_max_abs']:.1e} -> exact")

    m_final = dec.load_era(ckpt_dir, T, device)
    xprobe = xb[:16].to(device)
    ctrl = positive_controls(m_final, xprobe, task_k=1, verbose=False)
    assert ctrl["p3a_control"] and ctrl["p3b_control"], f"positive controls did not both fire: {ctrl}"
    print("  P3 positive controls FIRED on the final checkpoint (raw frame)")

    rows, wrong_fail = [], []
    for k in dec.OLD_TASKS:
        tr_raw, te_raw = loaders(k, P[k]); tr_rel, te_rel = loaders(k, P[T])
        k_wrong = k + 1 if k + 1 < T else k - 1
        m_era = dec.load_era(ckpt_dir, k, device)
        ck = torch.load(Path(ckpt_dir) / f"task{k}_epoch{dec.EPOCH_IDX}.pt", weights_only=True, map_location="cpu")
        acc_boundary = ck.get("acc_boundary")
        xp = next(iter(te_raw))[0][:16].to(device); xpr = next(iter(te_rel))[0][:16].to(device)
        assert_p3(m_era, xp, k, label=f"era{k}"); assert_p3(m_final, xp, k, label=f"final/T{k}/raw"); assert_p3(m_final, xpr, k, label=f"final/T{k}/relaid")
        f_e_tr, _, y_tr = dec.features_and_logits(m_era, tr_raw, k, False, device)
        f_e_te, l_e_te, y_te = dec.features_and_logits(m_era, te_raw, k, False, device)
        f_4_tr, _, y_tr4 = dec.features_and_logits(m_final, tr_raw, k, False, device)
        f_4_te, l_4_te, y_te4 = dec.features_and_logits(m_final, te_raw, k, False, device)
        f_r_tr, _, y_trr = dec.features_and_logits(m_final, tr_rel, k, False, device)
        f_r_te, l_r_te, y_ter = dec.features_and_logits(m_final, te_rel, k, False, device)
        assert torch.equal(y_tr, y_tr4) and torch.equal(y_tr, y_trr) and torch.equal(y_te, y_te4) and torch.equal(y_te, y_ter), f"task {k}: label order differs across passes"
        # wrong-source relay on the test set only (deployed accuracy) -- the must-fail
        te_w = DataLoader(bench._make(k, False), batch_size=dec.BATCH, shuffle=False)
        te_w.dataset.perm = P[k]
        lw = []
        with torch.no_grad():
            for x, _ in te_w:
                xw = bapply(bapply(x, invert_perm(P[k_wrong]) if P[k_wrong] is not None else torch.arange(x[0].numel())), P[T])
                out = m_final(xw.to(device), store_memories=False, task_hint=k, apply_adapter=False)
                lw.append(out["logits"].float().cpu())
        acc_wrong = acc(torch.cat(lw), y_te)
        n = dec.CLASSES_PER_TASK if hasattr(dec, "CLASSES_PER_TASK") else int(y_te.max()) + 1
        acc_ceiling, acc_orig, acc_orig_rel = acc(l_e_te, y_te), acc(l_4_te, y_te), acc(l_r_te, y_te)
        acc_rc = refit_probe(f_e_tr, y_tr, f_e_te, y_te, n)
        acc_r4 = refit_probe(f_4_tr, y_tr, f_4_te, y_te, n)
        acc_rr = refit_probe(f_r_tr, y_tr, f_r_te, y_te, n)
        row = dict(arm=arm, seed=args.seed, task=k, n_train=int(len(y_tr)), n_test=int(len(y_te)),
                   acc_ceiling=acc_ceiling, acc_refit_ceiling=acc_rc, R=acc_rc - acc_ceiling,
                   acc_orig=acc_orig, acc_refit=acc_r4, F_enc=acc_rc - acc_r4, F_read=acc_r4 - acc_orig, F_total=acc_ceiling - acc_orig,
                   acc_orig_relaid=acc_orig_rel, acc_refit_relaid=acc_rr, F_enc_relaid=acc_rc - acc_rr, F_read_relaid=acc_rr - acc_orig_rel, F_total_relaid=acc_ceiling - acc_orig_rel,
                   acc_orig_wrong_perm=acc_wrong, wrong_source_task=k_wrong,
                   acc_boundary=acc_boundary, fp16_delta=(None if acc_boundary is None else float(acc_ceiling - acc_boundary)),
                   input_sha=hashlib.sha1(y_te.numpy().tobytes()).hexdigest()[:12])
        row["identity_raw"] = abs(row["F_enc"] + row["F_read"] - row["R"] - row["F_total"]); row["identity_relaid"] = abs(row["F_enc_relaid"] + row["F_read_relaid"] - row["R"] - row["F_total_relaid"])
        if not acc_wrong < acc_orig_rel:
            wrong_fail.append(k)
        rows.append(row)
        fp16 = "n/a" if row["fp16_delta"] is None else f"{row['fp16_delta']:+.4f}"
        print(f"  t{k:>2}: ceiling {acc_ceiling:.3f} | raw: deployed {acc_orig:.3f} refit {acc_r4:.3f} F_enc {row['F_enc']:+.4f} | re-laid: deployed {acc_orig_rel:.3f} refit {acc_rr:.3f} F_enc {row['F_enc_relaid']:+.4f}"
              f" | wrong-perm deployed {acc_wrong:.3f} | fp16 delta {fp16}")
    m = lambda key: float(np.mean([r[key] for r in rows]))
    out = {"arm": arm, "backbone": args.backbone, "seed": args.seed, "benchmark": "cifar100_permuted", "shift_fingerprint": sh, "class_order_fingerprint": co,
           "option": "b: era checkpoints, no fp32 shadow; identity gate at the measured fp16 reload floor", "relay_checks": relay_checks,
           "wrong_perm_must_fail": {"pass": not wrong_fail, "fail_tasks": wrong_fail, "n": len(rows)},
           "identity_max": max(max(r["identity_raw"], r["identity_relaid"]) for r in rows),
           "fp16_reload_floor": max(abs(r["fp16_delta"]) for r in rows if r["fp16_delta"] is not None) if any(r["fp16_delta"] is not None for r in rows) else None,
           "pooled": {k: m(k) for k in ("F_enc", "F_read", "R", "F_total", "F_enc_relaid", "F_read_relaid", "F_total_relaid", "acc_orig", "acc_orig_relaid", "acc_ceiling")}, "rows": rows}
    print(f"\n  pooled: F_enc raw {out['pooled']['F_enc']:+.4f} -> re-laid {out['pooled']['F_enc_relaid']:+.4f} | deployed raw {out['pooled']['acc_orig']:.4f} -> re-laid {out['pooled']['acc_orig_relaid']:.4f} (ceiling {out['pooled']['acc_ceiling']:.4f})"
          f" | wrong-perm must-fail {'PASS' if not wrong_fail else 'FAIL ' + str(wrong_fail)} {len(rows)-len(wrong_fail)}/{len(rows)} | identity {out['identity_max']:.1e} | fp16 reload floor {out['fp16_reload_floor']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
