"""E26 FS, frozen-trunk arm -- what §4.3 turns on.

FS found that on a fine-tuned pretrained trunk (theta_T), a linear refit on
task-k content reads BEST in frame 0, the unpermuted layout the trunk was
pretrained in: 45/54 cells on ResNet, 28/36 on ViT, ~4 pp above the own frame,
the last frame and three random permuted frames, which all read alike.

That has two readings and the deployed numbers cannot separate them:

  (a) TRAINING TAUGHT THE SCRAMBLED FORMATS.   The frozen trunk would read far
      worse on permuted frames than theta_T does, and fine-tuning closed that
      gap.  Evidence: frozen(permuted) << theta_T(permuted).
  (b) TRAINING COST SKILL ON THE ORIGINAL LAYOUT.  The frozen trunk would read
      far better at frame 0 than theta_T does, and fine-tuning spent that.
      Evidence: frozen(base) >> theta_T(base).

They are not exclusive; the question is the SIZE of each. This script measures
the same six sampled frames on the FROZEN trunk -- ImageNet weights, never
fine-tuned, timm parity asserted -- so frozen and theta_T are compared frame by
frame under one protocol.

PROTOCOL IDENTITY IS THE POINT. E23's frozen probe (`e23_frozen_probe.py`)
already measured the frozen trunk, but only at each task's OWN frame, and
E12/E14's frozen probes measured unpermuted CIFAR under a different extraction.
Quoting 0.97 from one and 0.78 from another would be a cross-protocol
comparison of exactly the kind catch 34 is about. Here the frames, the loaders,
the probe (`refit_probe`), the class count and the frame subset are FS's own:
the frozen run reproduces `e23_frozen_probe`'s own-frame number as its C-ID,
and differs from FS only in which encoder produces the features.

There is no deployed arm: a frozen trunk has no trained head, so D_k(j) has no
referent. The refit is the whole measurement, and the refit is where FS's
finding lives.
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
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.channel_decomp import refit_probe                                        # noqa: E402
from scripts.e23_decompose import bapply                                              # noqa: E402
from scripts.e26_fs import REFIT_DRAW_SEED, relay_to                                  # noqa: E402
from src.data.split_cifar100 import (SplitCIFAR100Benchmark, IMAGE_SIZE,              # noqa: E402
                                     CLASSES_PER_TASK)

NUM_TASKS, BATCH = 20, 128
CFG = {"vit": "configs/e12_vit.yaml", "resnet": "configs/e14_resnet.yaml"}


@torch.no_grad()
def feats(encoder, loader, device):
    F, Y = [], []
    for x, y in loader:
        F.append(encoder(x.to(device)).float().cpu()); Y.append(y)
    return torch.cat(F), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser(description="E26 FS on the frozen trunk")
    ap.add_argument("--backbone", required=True, choices=["vit", "resnet"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref-frozen", default=None,
                    help="e23_frozen_probe artifact for the own-frame C-ID")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.backbone == "vit":
        from scripts.e12_frozen_probe import ViTEncoder as Enc, VIT_MODEL as MODEL
    else:
        from scripts.e14_frozen_probe import ResNetEncoder as Enc, RESNET_MODEL as MODEL
    expected = (yaml.safe_load(open(CFG[args.backbone])).get("benchmark") or {}).get("expected_shift") or {}
    print("=" * 100 + f"\nE26 FS FROZEN  {args.backbone} ({MODEL}) seed {args.seed}\n" + "=" * 100)

    encoder = Enc(model_name=MODEL, pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    d = encoder.assert_matches_timm(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    print(f"  timm parity max|delta| = {d:.1e}")

    bench = SplitCIFAR100Benchmark(num_tasks=NUM_TASKS, seed=args.seed, root=args.data_root,
                                   batch_size=BATCH, remap_labels=True, shift_mode="patch",
                                   download=False)
    fp = bench.shift_fingerprint()
    assert expected.get(args.seed) == fp, f"shift fingerprint {fp} != registered {expected.get(args.seed)}"
    P, T = bench.perms, NUM_TASKS - 1
    assert P[0] is None
    print(f"  shift fingerprint {fp} GATED -> MATCH")

    def loaders(k, perm):
        tr, te = bench._make(k, True), bench._make(k, False)
        tr.perm, te.perm = perm, perm
        return (DataLoader(tr, batch_size=BATCH, shuffle=False),
                DataLoader(te, batch_size=BATCH, shuffle=False))

    ref = None
    if args.ref_frozen and os.path.exists(args.ref_frozen):
        ref = {r["task"]: r["diag"] for r in json.load(open(args.ref_frozen))["rows"]
               if r["seed"] == args.seed}

    rng = np.random.default_rng(REFIT_DRAW_SEED + args.seed)   # FS's draw, same stream
    rows, relay_max, cid = [], 0.0, []
    tasks = list(range(T))[: args.max_tasks] if args.max_tasks else list(range(T))
    t0 = time.time()
    for k in tasks:
        tk0 = time.time()
        _, te_k = loaders(k, P[k]); xb_k = next(iter(te_k))[0]
        extra = [int(j) for j in rng.choice([j for j in range(NUM_TASKS) if j not in (k, T, 0)], 3, replace=False)]
        frames = sorted({k, T, 0, *extra})
        Pk, y_ref = {}, None
        for j in frames:
            tr_j, te_j = loaders(k, P[j])
            xb_j = next(iter(te_j))[0]
            relay_max = max(relay_max, float((relay_to(xb_k, P, k, j) - xb_j).abs().max()))
            Ftr, ytr = feats(encoder, tr_j, device)
            Fte, yte = feats(encoder, te_j, device)
            if y_ref is None:
                y_ref = yte
            assert torch.equal(yte, y_ref), f"task {k} frame {j}: label order differs"
            Pk[j] = refit_probe(Ftr, ytr, Fte, yte, CLASSES_PER_TASK)
        assert relay_max == 0.0, relay_max
        if ref is not None and k in ref:
            cid.append(abs(Pk[k] - ref[k]))
        best = max(Pk, key=Pk.get)
        rows.append({"backbone": args.backbone, "seed": args.seed, "task": k,
                     "n_test": int(len(y_ref)), "P": {str(j): v for j, v in Pk.items()},
                     "P_extra_frames": extra, "frames": frames,
                     "P_own": Pk[k], "P_last": Pk[T], "P_base": Pk[0],
                     "P_other3": float(np.mean([Pk[j] for j in extra])),
                     "best_frame": int(best), "best_is_base": int(best) == 0,
                     "best_is_own": int(best) == k, "best_is_last": int(best) == T,
                     "cid_own_abs": None if ref is None or k not in ref else abs(Pk[k] - ref[k]),
                     "seconds": round(time.time() - tk0, 1)})
        print(f"  k={k:>2}: best j={best:>2} | own {Pk[k]:.3f} last {Pk[T]:.3f} base {Pk[0]:.3f} "
              f"other3 {rows[-1]['P_other3']:.3f}" +
              (f" | C-ID {cid[-1]:.4f}" if cid else "") + f" | {rows[-1]['seconds']}s")

    k1 = [r for r in rows if r["task"] >= 1]
    summary = {"arm": f"frozen_{args.backbone}", "backbone": args.backbone, "model": MODEL,
               "seed": args.seed, "frozen": True, "n_frames_sampled": 6,
               "shift_fingerprint": fp, "timm_parity_max_abs": float(d),
               "refit_draw_seed": REFIT_DRAW_SEED + args.seed,
               "controls": {"relay_exactness_max_abs": relay_max, "relay_exact": relay_max == 0.0,
                            "c_id_own_vs_e23_frozen_max_abs": (max(cid) if cid else None),
                            "c_id_cells": len(cid)},
               "counts": {"cells": len(rows), "best_base_k_ge_1": sum(r["best_is_base"] for r in k1),
                          "best_own": sum(r["best_is_own"] for r in rows),
                          "best_last": sum(r["best_is_last"] for r in rows),
                          "cells_k_ge_1": len(k1)},
               "pooled": {kk: float(np.mean([r[kk] for r in rows]))
                          for kk in ("P_own", "P_last", "P_base", "P_other3")},
               "rows": rows, "seconds_total": round(time.time() - t0, 1)}
    c, p = summary["counts"], summary["pooled"]
    print(f"\n  pooled refit: own {p['P_own']:.3f}  last {p['P_last']:.3f}  base {p['P_base']:.3f}  other3 {p['P_other3']:.3f}")
    print(f"  best frame: base {c['best_base_k_ge_1']}/{c['cells_k_ge_1']} (k>=1)  own {c['best_own']}/{c['cells']}  last {c['best_last']}/{c['cells']}")
    if cid:
        print(f"  C-ID vs e23_frozen_probe own-frame: max |delta| {max(cid):.4f} over {len(cid)} cells")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"frozen_seed{args.seed}.json")
    json.dump(summary, open(out, "w"), indent=2, default=float)
    print(f"  wrote {out}  ({summary['seconds_total']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
