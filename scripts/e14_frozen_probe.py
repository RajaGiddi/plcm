"""
E14 — the frozen-probe arm: the trivial use of the assumed resource, and P1's
reference.

Contract: docs/E14_resnet_prereg.md sec 1 (arms), sec 2 (P1).

WHY THIS ARM IS FIRST AND FIRST-CLASS. E12 assumes ImageNet pretraining. Catch 24
says: price the TRIVIAL use of an assumed resource before measuring anything
built on it. The trivial use here is to not fine-tune at all — freeze the trunk,
fit a per-task linear probe. **Forgetting is then zero by construction**: the
trunk never moves and each task's probe is never touched again. That is not a
result, it is the definition of the baseline, and it is exactly why the arm must
be reported rather than assumed away.

What it buys:
  * the honest framing — "forgetting in the trainable-trunk regime, measured
    against the zero-forgetting frozen alternative";
  * P1's reference, which makes that gate SCALE-FREE (an absolute 0.85 bar on a
    pretrained ViT is close to unfailable — catch 25's category).

EFFICIENCY, AND WHY IT CHANGES NOTHING. Frozen features do not depend on the
class order, so all 60k CLS vectors are extracted ONCE and the 20x5 tasks are
formed by row selection per seed. Three seeds differ only in which classes group
together, which is the whole of what the seed controls in this arm.

Probe recipe is the program's convex lock, unchanged: standardize on TRAIN
statistics, multinomial logistic regression, lbfgs, fit to convergence.

Usage:
    modal run modal_runner.py::analysis_gpu --argv "scripts/e14_frozen_probe.py"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.channel_decomp import refit_probe
from src.data.split_cifar100 import (IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
                                     N_CLASSES_TOTAL, NUM_TASKS, CLASSES_PER_TASK)
from src.models.resnet_base import ResNetEncoder, RESNET_MODEL

SEEDS = [42, 1337, 2024]
P1_BAR = 0.85          # the frozen reference must itself clear this, or the pairing is void


@torch.no_grad()
def extract(encoder, train: bool, root: str, device, batch_size: int = 256):
    """Post-norm CLS features for a whole CIFAR-100 split. No adapter, no shift."""
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.CIFAR100(root=root, train=train, download=False, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, ys = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(dl):
        f = encoder(x.to(device))                      # [B, 768] post-norm CLS
        feats.append(f.float().cpu())
        ys.append(y)
        if i % 40 == 0:
            print(f"    {'train' if train else 'test'} batch {i}/{len(dl)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return torch.cat(feats), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser(description="E14 frozen-probe arm")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", default="runs/e14_frozen_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 96)
    print("E14 FROZEN-PROBE ARM — the trivial use of ImageNet pretraining (catch 24)")
    print("=" * 96)
    print(f"  device {device} | model {RESNET_MODEL}")

    encoder = ResNetEncoder(model_name=RESNET_MODEL, pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Sanity that the frozen path is the registered one, before any number.
    d = encoder.assert_matches_timm(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    print(f"  timm parity max|delta| = {d:.1e}\n")

    print("  extracting CLS features (once; frozen features do not depend on "
          "class order)")
    F_tr, y_tr = extract(encoder, True, args.root, device)
    F_te, y_te = extract(encoder, False, args.root, device)
    print(f"  train {tuple(F_tr.shape)}  test {tuple(F_te.shape)}\n")

    rows = []
    for seed in SEEDS:
        order = [int(c) for c in np.random.RandomState(seed).permutation(N_CLASSES_TOTAL)]
        for k in range(NUM_TASKS):
            cls = order[k * CLASSES_PER_TASK:(k + 1) * CLASSES_PER_TASK]
            remap = {c: i for i, c in enumerate(cls)}
            m_tr = torch.isin(y_tr, torch.tensor(cls))
            m_te = torch.isin(y_te, torch.tensor(cls))
            ytr_k = torch.tensor([remap[int(v)] for v in y_tr[m_tr]])
            yte_k = torch.tensor([remap[int(v)] for v in y_te[m_te]])
            acc = refit_probe(F_tr[m_tr], ytr_k, F_te[m_te], yte_k, CLASSES_PER_TASK)
            rows.append({"seed": seed, "task": k, "classes": cls, "diag": acc,
                         "n_train": int(m_tr.sum()), "n_test": int(m_te.sum())})
        accs = [r["diag"] for r in rows if r["seed"] == seed]
        print(f"  seed {seed}: mean DIAG {np.mean(accs):.4f}  "
              f"min {np.min(accs):.4f}  max {np.max(accs):.4f}")

    print("\n" + "=" * 96)
    print("RESULT")
    print("=" * 96)
    allacc = [r["diag"] for r in rows]
    mean, mn = float(np.mean(allacc)), float(np.min(allacc))
    per_seed = [float(np.mean([r["diag"] for r in rows if r["seed"] == s])) for s in SEEDS]
    print(f"  per-task DIAG over {len(rows)} cells (20 tasks x {len(SEEDS)} seeds)")
    print(f"    mean {mean:.4f}   min {mn:.4f}   per-seed means "
          f"{['%.4f' % v for v in per_seed]}")
    # Verdict computed from the values printed above (catch 22).
    clears = mn >= P1_BAR
    print(f"\n  P1 reference clause: min per-task DIAG {mn:.4f} vs bar {P1_BAR} -> "
          f"{'CLEARS' if clears else 'FAILS — benchmark/backbone pairing is VOID'}")
    print(f"  FORGETTING: 0.0000 BY CONSTRUCTION — the trunk never moves and each "
          f"task's probe is never revisited.\n  This is the definition of the "
          f"baseline, not a measurement of it.")
    print(f"\n  This arm is now P1's reference: the trainable trunk must land "
          f"within 5pp of {mean:.4f} per task.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"model": RESNET_MODEL, "rows": rows, "mean_diag": mean, "min_diag": mn,
               "per_seed_mean": per_seed, "p1_bar": P1_BAR, "clears": bool(clears),
               "forgetting": 0.0}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")
    return 0 if clears else 1


if __name__ == "__main__":
    sys.exit(main())
