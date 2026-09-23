"""
E12 — compute class-order and shift fingerprints for candidate seeds.

Contract: docs/E12_prereg.md sec 7 ("fingerprints computed in the execution
environment") and sec 1 (5 seeds on base/adapt).

WHY A SCRIPT RATHER THAN A ONE-LINER. E10's registered partition fingerprint was
computed on the laptop while every run executed on x86, and the two disagreed
about the data while agreeing about the label. `_e12_seeds()` refuses to launch
an unregistered seed for exactly that reason, so the way to add seeds is to
compute their fingerprints WHERE THE RUNS EXECUTE and paste those values — not to
compute them locally and hope, and not to relax the refusal.

Run it in BOTH environments and compare. B1/B6 established that this benchmark's
fingerprints are invariant across arm64/Darwin and x86_64/Linux (unlike E10's,
which had a greedy-balancer tie); confirming that again for new seeds is cheap
and is the only thing that licenses pasting a locally-computed value.

Usage:
    python scripts/e12_register_seeds.py --seeds 7,1234
    modal run modal_runner.py::analysis --argv "scripts/e12_register_seeds.py --seeds 7,1234"
"""

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.split_cifar100 import SplitCIFAR100Benchmark


def main():
    ap = argparse.ArgumentParser(description="E12 seed registration")
    ap.add_argument("--seeds", default="7,1234")
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 88)
    print("E12 SEED REGISTRATION — fingerprints computed where they are printed")
    print("=" * 88)
    print(f"  {platform.python_version()} on {platform.machine()} / {platform.system()}")
    print()
    print("  paste into configs/e12_vit.yaml:")
    print()
    print("  expected_class_order:")
    for s in seeds:
        b = SplitCIFAR100Benchmark(root=args.root, seed=s, download=False)
        print(f"    {s}: {b.class_order_fingerprint()}")
    print("  expected_shift:")
    for s in seeds:
        b = SplitCIFAR100Benchmark(root=args.root, seed=s, download=False,
                                   shift_mode="patch")
        print(f"    {s}: {b.shift_fingerprint()}")

    print()
    print("  sanity (must hold for every seed): 20 tasks x 5 classes, disjoint, "
          "all 100 covered")
    for s in seeds:
        b = SplitCIFAR100Benchmark(root=args.root, seed=s, download=False)
        seen = set()
        ok = True
        for cs in b.task_classes:
            if len(cs) != 5 or (set(cs) & seen):
                ok = False
            seen |= set(cs)
        print(f"    seed {s}: tasks {len(b.task_classes)}  classes {len(seen)}  "
              f"disjoint+sized {ok}  first task {b.task_classes[0]}")


if __name__ == "__main__":
    main()
