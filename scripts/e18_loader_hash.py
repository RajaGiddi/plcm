"""E18 P0 -- loader-level regression for the `disjoint_content` amendment.

Hashes the first N batches of every MNIST loader the flag-off path produces
(Permuted and Rotated, train with shuffle under a fixed torch seed, test
sequential), for three benchmark seeds. Run ONCE on the pre-amendment code to
write the reference, then after the amendment with --check: every hash must be
identical, or the flag-off construction changed. A syntax check cannot see
this; the hash can.

Usage:
    python scripts/e18_loader_hash.py --write runs/e18/loader_hash_reference.json
    python scripts/e18_loader_hash.py --check runs/e18/loader_hash_reference.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.permuted_mnist import PermutedMNISTBenchmark
from src.data.rotated_mnist import RotatedMNISTBenchmark

SEEDS = [42, 1337, 2024]
N_BATCHES = 50


def hash_loader(loader, n=N_BATCHES):
    h = hashlib.sha1()
    for i, (x, y) in enumerate(loader):
        h.update(x.numpy().tobytes()); h.update(y.numpy().tobytes())
        if i + 1 >= n:
            break
    return h.hexdigest()[:16]


def hashes(**extra):
    out = {}
    for name, cls in (("permuted", PermutedMNISTBenchmark), ("rotated", RotatedMNISTBenchmark)):
        for seed in SEEDS:
            b = cls(num_tasks=5, batch_size=128, seed=seed, **extra)
            for k in range(5):
                torch.manual_seed(1000 + k)          # the shuffle draw is part of the path
                tr, te = b.get_task_loaders(k)
                out[f"{name}/seed{seed}/task{k}/train"] = hash_loader(tr)
                out[f"{name}/seed{seed}/task{k}/test"] = hash_loader(te)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write"); ap.add_argument("--check")
    args = ap.parse_args()
    h = hashes()
    if args.write:
        json.dump(h, open(args.write, "w"), indent=2)
        print(f"wrote {len(h)} hashes to {args.write}")
    if args.check:
        ref = json.load(open(args.check))
        diff = {k: (ref.get(k), h.get(k)) for k in set(ref) | set(h) if ref.get(k) != h.get(k)}
        print(f"flag-off loaders: {len(h)} hashes, {len(diff)} differ -> {'IDENTICAL' if not diff else 'CHANGED -- STOP'}")
        for k, v in sorted(diff.items())[:10]:
            print("  ", k, v)
        return 0 if not diff else 2


if __name__ == "__main__":
    sys.exit(main() or 0)
