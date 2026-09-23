"""
E15 — ViT tier-0, prototypes-only. One seed per invocation.

Contract: docs/E15_vit_tier0_prereg.md (signed 2026-08-10).

THE QUESTION. E12 measured F_enc = 0.06 on the ViT — the features barely moved.
Do stored class prototypes still span the right subspace, so projecting the
deployed feature onto that span recovers a real fraction of the reader gap?

ONE MEASURED CURE. `proj` — project the deployed theta_T feature onto the stored
era-prototype span, read with the deployed (frozen, task-k) head. Stored per
task: prototypes only, 5 x 768 floats (~15KB fp32). No raw data, no era head, no
generative step.

WHY NO ERA-HEAD ROW (contract sec 0). On the ViT, head k is created at task k,
frozen at its boundary, and P3b verified by module identity that it is what
already scores task k. "Apply the stored era head" therefore reproduces
`acc_orig` EXACTLY — a comparison whose two sides are the same object. It is
printed as acc_orig with this note and never scored. The collapse is itself a
result: the era-head component was deployed all along and forgetting is 0.72
anyway.

RANK CONTROL (sec 2a), run and printed BEFORE the cure is scored. Task k has 5
classes, so the span is rank 5 in 768 dims — HAR projected 6-of-256, MNIST
10-of-256. A null could mean features moved (theory) OR the projection is too
lossy to read through (geometry). So the identical projection is applied AT THE
ERA CHECKPOINT, where features definitionally match the prototypes built from
them. Within 0.05 of the era ceiling -> information-preserving, the theta_T
verdict is readable. Otherwise -> "unreadable at this rank", branch (C), and
explicitly NOT evidence against the theory.

REUSE, NOT RECOMPUTE. `acc_refit` and `R` come from E12's v2 decomposition (same
instrument, same cells) rather than being re-fit; `acc_orig` is recomputed here
and asserted EQUAL to v2's, which is a non-tautological check that both runs
describe the same cells. `span_of` and the audited loader are imported.

Usage:
    modal run --detach modal_runner.py::spawn_analysis --experiment e15
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import load, load_task_data
from scripts.cure_screen import era_head_of
from scripts.e12_decompose import features_and_logits, NUM_TASKS, FINAL, OLD_TASKS, EPOCH_IDX, BATCH
from scripts.e12_p3 import assert_p3, positive_controls
from scripts.transport_estimate import span_of
from src.data.split_cifar100 import SplitCIFAR100Benchmark

GUARD = 0.02
RANK_CTRL_BAR = 0.05      # instrument-bias convention; this is an instrument question
HP1_BAR = 0.30


def main():
    ap = argparse.ArgumentParser(description="E15 ViT tier-0 (prototypes-only)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--decomp", default="runs/e12_decomp_v2/base_{s}.json")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    s = args.seed
    ckpt_dir = os.path.join(args.ckpt_root, f"ckpt_e12_base_seed{s}", f"mafc_seed{s}")
    out_path = args.out or f"runs/e15_tier0/seed_{s}.json"

    print("=" * 104)
    print(f"E15 — ViT TIER-0, PROTOTYPES-ONLY   seed={s}  device={device}")
    print("=" * 104)

    prior = {(r["seed"], r["task"]): r for r in json.load(open(args.decomp.format(s=s)))}
    bench = SplitCIFAR100Benchmark(num_tasks=NUM_TASKS, batch_size=BATCH,
                                   root=args.root, seed=s, remap_labels=True,
                                   download=False)
    m4 = load(ckpt_dir, FINAL, epoch=EPOCH_IDX)
    m4.to(device).eval()

    # ---- PRECONDITION BLOCK, printed before any cure number ------------------
    print("\n" + "-" * 104)
    print("PRECONDITIONS")
    print("-" * 104)
    _, te0 = bench.get_task_loaders(0)
    xprobe = next(iter(te0))[0][:16].to(device)
    ctrl = positive_controls(m4, xprobe, task_k=0, verbose=True)
    assert ctrl["p3a_control"] and ctrl["p3b_control"], f"controls did not fire: {ctrl}"
    print("  P3a/P3b positive controls: BOTH FIRED")

    ds0 = te0.dataset
    seq = DataLoader(ds0, batch_size=64, shuffle=False)
    xl, yl = next(iter(seq))
    xd = torch.stack([ds0[i][0] for i in range(xl.shape[0])])
    yd = torch.tensor([ds0[i][1] for i in range(xl.shape[0])])
    aligned = torch.equal(xl, xd) and torch.equal(yl, yd)
    g = torch.Generator().manual_seed(0)
    sx, _ = next(iter(DataLoader(ds0, batch_size=64, shuffle=True, generator=g)))
    fired = not torch.equal(sx, xd)
    assert aligned and fired, "catch-32 alignment gate failed"
    print(f"  catch-32 alignment: order == dataset indexing PASS | "
          f"shuffled control FIRED")
    print(f"  E12 reproduction floor quoted: aggregate ~0.4pp, PER-CELL ~15pp "
          f"-> pooled-with-CI is primary")

    rows = []
    for k in OLD_TASKS:
        tr, te = bench.get_task_loaders(k)
        tr = DataLoader(tr.dataset, batch_size=BATCH, shuffle=False)   # catch 32
        m_era = load(ckpt_dir, k, epoch=EPOCH_IDX)
        m_era.to(device).eval()
        # assert_p3, not assert_path_identity: the latter is the LSTM-FAMILY gate
        # (its catch-28 sizing block calls model.lstm expecting (out,(h,c))) and
        # a ViTEncoder returns a single tensor. Same gate name, different
        # architecture contract — the E12 preamble's point, met in the wiring.
        xe = next(iter(te))[0][:16].to(device)
        assert_p3(m_era, xe, k, label=f"era{k}")
        assert_p3(m4, xprobe, k, label=f"final/T{k}")

        head = era_head_of(m_era, k)          # == the deployed head for task k
        f_e_tr, _, y_tr = features_and_logits(m_era, tr, k, False, device)
        f_e_te, l_e_te, y_te = features_and_logits(m_era, te, k, False, device)
        f_4_te, l_4_te, _ = features_and_logits(m4, te, k, False, device)

        acc_ceiling = float((l_e_te.argmax(1) == y_te).float().mean())
        acc_orig = float((l_4_te.argmax(1) == y_te).float().mean())

        # Consistency with E12's v2 cells — non-tautological: two separate runs
        # of the deployed path must agree on the floor or they are not the same
        # cells and the reused refit/R do not belong to them.
        p = prior[(s, k)]
        assert abs(acc_orig - p["acc_orig"]) < 1e-9, (
            f"T{k}: acc_orig {acc_orig} != E12 v2 {p['acc_orig']} — different cells")

        # STORED PROTOTYPES -> span. span_of imported, not re-derived.
        cls = sorted(set(int(v) for v in y_tr))
        B = torch.stack([f_e_tr[y_tr == c].mean(0) for c in cls]).numpy().astype(np.float64)
        Pb = span_of(B)
        P = torch.from_numpy((Pb @ Pb.T).astype(np.float32))

        # features_and_logits returns CPU tensors; the head lives on `device`.
        with torch.no_grad():
            Pd = P.to(device)
            def read(f):
                return head(f.to(device) @ Pd).argmax(1).cpu()
            acc_proj = float((read(f_4_te) == y_te).float().mean())
            acc_proj_era = float((read(f_e_te) == y_te).float().mean())

        D = p["acc_refit"] - p["R"] - acc_orig
        rho = (acc_proj - acc_orig) / D if abs(D) > 1e-12 else float("nan")
        se = math.sqrt(max(acc_proj * (1 - acc_proj), 0) / len(y_te)) / max(abs(D), 1e-9)
        rows.append(dict(seed=s, task=k, n_test=int(len(y_te)), rank=int(Pb.shape[1]),
                         acc_orig=acc_orig, acc_ceiling=acc_ceiling,
                         acc_refit=p["acc_refit"], R=p["R"], acc_proj=acc_proj,
                         acc_proj_era=acc_proj_era,
                         rank_ctrl_gap=acc_ceiling - acc_proj_era,
                         D=D, rho=rho, se=se))
        print(f"  T{k:<2} rank {Pb.shape[1]}/768 | ceiling {acc_ceiling:.4f} "
              f"proj@era {acc_proj_era:.4f} (gap {acc_ceiling-acc_proj_era:+.4f}) | "
              f"orig {acc_orig:.4f} proj {acc_proj:.4f} D {D:+.4f} rho {rho:+.3f}",
              flush=True)
        del m_era

    # ---- rank control verdict, BEFORE the cure is read ----------------------
    gaps = [r["rank_ctrl_gap"] for r in rows]
    ctrl_ok = float(np.mean(gaps)) <= RANK_CTRL_BAR
    print("\n" + "=" * 104)
    print(f"RANK CONTROL (sec 2a): mean era ceiling - proj@era = {np.mean(gaps):+.4f} "
          f"(max {max(gaps):+.4f}) vs bar {RANK_CTRL_BAR}")
    verdict = ("PASS — projection is information-preserving; the theta_T "
               "verdict is readable" if ctrl_ok else
               "FAIL — UNREADABLE AT THIS RANK (branch C); NOT evidence "
               "against the theory")
    print(f"  -> {verdict}")

    print(f"\n  era-head component: printed as acc_orig "
          f"{np.mean([r['acc_orig'] for r in rows]):.4f} — DEGENERATE by construction "
          f"(head k is the deployed head for task k, P3b-verified). Never scored.")

    keep = [r for r in rows if r["D"] > GUARD and r["rho"] == r["rho"]]
    def pool(rs):
        if not rs: return float("nan"), float("nan")
        m = sum(r["rho"] for r in rs) / len(rs)
        return m, math.sqrt(sum(r["se"] ** 2 for r in rs)) / len(rs)
    gm, gse = pool(keep); fm, _ = pool([r for r in rows if r["rho"] == r["rho"]])
    print(f"\n  proj rho  guarded {gm:+.3f} [{gm-1.96*gse:+.3f}, {gm+1.96*gse:+.3f}] "
          f"({len(keep)}/{len(rows)} cells)   forced {fm:+.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump({"seed": s, "rows": rows, "rank_control_ok": bool(ctrl_ok),
               "rank_ctrl_mean_gap": float(np.mean(gaps)),
               "rho_guarded": gm, "rho_forced": fm, "n_kept": len(keep)},
              open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
