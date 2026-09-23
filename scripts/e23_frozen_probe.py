"""E23 P1 -- the frozen-trunk reference on cifar100_permuted (docs/E23_prereg.md sec 3).

E12/E14's frozen-probe scripts extract features from UNPERMUTED CIFAR-100 once
(`extract(encoder, train, root, device)`: "No adapter, no shift"), which is the
right reference for B1 and the wrong one for B6, where each task's images carry
that task's patch-consistent permutation. Here the features are extracted PER
TASK through `SplitCIFAR100Benchmark(shift_mode="patch")`'s own loaders, so the
permutation is applied by the same `__getitem__` the training runs used
(src/data/split_cifar100.py), with the shift fingerprint asserted against the
config's registered value. Trunk frozen (ImageNet weights, timm parity asserted
as in the E12/E14 scripts), one linear readout per task fit to optimality
(`refit_probe`). P1: frozen DIAG >= 0.85 per task, and the trainable trunk
(the E23 run) within 5pp of this reference.

Usage: python scripts/e23_frozen_probe.py --backbone resnet --out /runs/e23/frozen_rn.json
"""

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
from scripts.channel_decomp import refit_probe                                          # noqa: E402
from src.data.split_cifar100 import SplitCIFAR100Benchmark, IMAGE_SIZE, CLASSES_PER_TASK  # noqa: E402

SEEDS = [42, 1337, 2024]
P1_BAR = 0.85
NUM_TASKS = 20
BATCH = 128


@torch.no_grad()
def feats(encoder, loader, device):
    F, Y = [], []
    for x, y in loader:
        F.append(encoder(x.to(device)).float().cpu()); Y.append(y)
    return torch.cat(F), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["vit", "resnet"])
    ap.add_argument("--root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.backbone == "vit":
        from scripts.e12_frozen_probe import ViTEncoder as Enc, VIT_MODEL as MODEL
        cfg = "configs/e12_vit.yaml"
    else:
        from scripts.e14_frozen_probe import ResNetEncoder as Enc, RESNET_MODEL as MODEL
        cfg = "configs/e14_resnet.yaml"
    expected = (yaml.safe_load(open(cfg)).get("benchmark") or {}).get("expected_shift") or {}
    print("=" * 96 + f"\nE23 FROZEN-PROBE REFERENCE on cifar100_permuted -- {args.backbone} ({MODEL}), per-task permuted extraction\n" + "=" * 96)
    encoder = Enc(model_name=MODEL, pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    d = encoder.assert_matches_timm(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    print(f"  timm parity max|delta| = {d:.1e}")
    rows = []
    for seed in SEEDS:
        bench = SplitCIFAR100Benchmark(num_tasks=NUM_TASKS, seed=seed, root=args.root, batch_size=BATCH, remap_labels=True, shift_mode="patch", download=False)
        fp = bench.shift_fingerprint()
        assert expected.get(seed) == fp, f"seed {seed}: shift fingerprint {fp} != registered {expected.get(seed)}"
        print(f"  seed {seed}: shift fingerprint {fp} GATED -> MATCH; class order {bench.class_order_fingerprint()}")
        t0 = time.time()
        for k in range(NUM_TASKS):
            tr, te = bench.get_task_loaders(k)
            tr = DataLoader(tr.dataset, batch_size=BATCH, shuffle=False); te = DataLoader(te.dataset, batch_size=BATCH, shuffle=False)
            Ftr, ytr = feats(encoder, tr, device); Fte, yte = feats(encoder, te, device)
            acc = refit_probe(Ftr, ytr, Fte, yte, CLASSES_PER_TASK)
            rows.append({"seed": seed, "task": k, "classes": bench.task_classes[k], "perm": "none" if bench.perms[k] is None else "patch", "diag": acc, "n_train": int(len(ytr)), "n_test": int(len(yte))})
        accs = [r["diag"] for r in rows if r["seed"] == seed]
        print(f"  seed {seed}: mean DIAG {np.mean(accs):.4f}  min {np.min(accs):.4f}  max {np.max(accs):.4f}  ({time.time()-t0:.0f} s)")
    diag = [r["diag"] for r in rows]
    out = {"backbone": args.backbone, "model": MODEL, "benchmark": "cifar100_permuted", "shift_mode": "patch", "rows": rows,
           "mean_diag": float(np.mean(diag)), "min_diag": float(np.min(diag)),
           "per_seed_mean": [float(np.mean([r["diag"] for r in rows if r["seed"] == s])) for s in SEEDS],
           "p1_bar": P1_BAR, "clears": bool(np.min(diag) >= P1_BAR)}
    print(f"\n  RESULT  mean DIAG {out['mean_diag']:.4f}  min {out['min_diag']:.4f}  bar {P1_BAR} -> {'CLEARS' if out['clears'] else 'FAILS'} (n={len(rows)})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
